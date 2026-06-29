import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from natten import NeighborhoodAttention1D   # 原本使用的邻域注意力库，已注释掉
# from timm.models.layers import DropPath      # 原本使用的随机深度库，已注释掉

# ============================================================
# 文件说明：
#   本文件定义了多种"序列/点云编码器"，用于将时序轨迹、地图点云
#   等原始数据编码为固定维度的向量表示。
#
# ⚠️ 注意：本文件的类并未被 DiffusionPlanningHead 直接调用！
#   - DiffusionPlanningHead 使用的时间步编码在 common_layers.py
#   - 本文件是 pl_diffusion_models 库的底层工具，供地图/轨迹编码器使用
#   - 学习重点：理解各编码器的输入输出 shape 和设计思路
# ============================================================


# ============================================================
# 一、NATSequenceEncoder（序列特征提取器）
# ------------------------------------------------------------
# 作用：将时间序列数据（如历史轨迹）编码为一个全局特征向量
# 结构：类似视觉领域的 FPN（特征金字塔网络），但用于 1D 序列
#
# 输入：[B, C, T]  B=batch, C=通道数(如坐标维度), T=时间步数
# 输出：[B, n]     n = embed_dim * 2^(num_levels-1)，最深层的特征维度
#
# 整体流程：
#   Raw序列 → ConvTokenizer → 多级NATBlock（逐级下采样）→ FPN融合 → 取最后时刻特征
# ============================================================
class NATSequenceEncoder(nn.Module):
    def __init__(
        self,
        in_chans=3,          # 输入通道数，如轨迹坐标 (x, y, heading) = 3
        embed_dim=32,        # 初始 embedding 维度，逐级翻倍：32 → 64 → 128
        mlp_ratio=3,         # MLP 隐层维度倍数（相对于 embed_dim）
        kernel_size=[3, 3, 5],   # 每级 NAT 注意力的卷积核大小
        depths=[2, 2, 2],        # 每级 NATBlock 包含的 NATLayer 层数
        num_heads=[2, 4, 8],     # 每级多头注意力的头数（随深度增加）
        out_indices=[0, 1, 2],   # 哪些层的输出参与 FPN 融合
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()

        # ① 第一步：将原始序列用 1D 卷积映射到 embed_dim 维
        self.embed = ConvTokenizer(in_chans, embed_dim)

        self.num_levels = len(depths)   # 共 3 级（level 0, 1, 2）
        # 每级的特征维度：[32, 64, 128]（逐级翻倍）
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.out_indices = out_indices

        # ② 生成随机深度（drop path）的概率列表，从 0 线性增长到 drop_path_rate
        # 目的：深层网络中越靠后的层越容易被随机跳过，起到正则化效果
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ③ 构建多级 NATBlock（类似 Swin Transformer 的 Stage 设计）
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),           # 本级特征维度
                depth=depths[i],                      # 本级 NATLayer 层数
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
                dilations=None,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),  # 最后一级不下采样
            )
            self.levels.append(level)

        # ④ 每个输出层后接一个 LayerNorm（命名为 norm0, norm1, norm2）
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)  # 动态注册为模块属性

        # ⑤ FPN 侧边连接：将各层特征统一映射到最深层维度 n
        # 例如：level0(32) → n(128)，level1(64) → n(128)，level2(128) → n(128)
        n = self.num_features[-1]   # 最深层维度，如 128
        self.lateral_convs = nn.ModuleList()
        for i_layer in self.out_indices:
            self.lateral_convs.append(
                nn.Conv1d(self.num_features[i_layer], n, 3, padding=1)
            )

        # ⑥ FPN 输出层：融合后再过一个 1D 卷积
        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)

    def forward(self, x):
        """
        输入：x: [B, C, T]  (B=batch, C=输入通道数, T=序列长度)
        输出：   [B, n]     (每个样本的全局特征向量，取序列最后时刻)

        数据流：
          [B,C,T] → ConvTokenizer → [B,T,embed_dim]
                  → Level0(NATBlock) → [B,T/2,64]  (下采样)
                  → Level1(NATBlock) → [B,T/4,128] (下采样)
                  → Level2(NATBlock) → [B,T/4,128] (不下采样)
                  → FPN融合 → [B,128,T/2]
                  → 取最后时刻 out[:,:,-1] → [B,128]
        """
        # Step1: 用 1D 卷积嵌入，输出 [B, T, embed_dim]
        x = self.embed(x)

        # Step2: 逐级通过 NATBlock，收集各层输出
        out = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)  # x: 下采样后的特征; xo: 下采样前的特征（用于 FPN 侧连）
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)               # LayerNorm
                out.append(x_out.permute(0, 2, 1).contiguous())  # [B,T,C] → [B,C,T]

        # Step3: FPN 侧边连接，将各层特征映射到相同维度 n
        # laterals[i]: [B, n, T_i]，其中 T_i 随层数递减
        laterals = [
            lateral_conv(out[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Step4: 自顶向下融合（从最深层向浅层加权融合）
        # 将深层特征上采样后加到浅层特征上
        for i in range(len(out) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                scale_factor=(laterals[i - 1].shape[-1] / laterals[i].shape[-1]),
                mode="linear",
                align_corners=False,
            )

        # Step5: 最浅层特征经过最终卷积，得到融合后的特征图
        out = self.fpn_conv(laterals[0])   # [B, n, T_0]

        # Step6: 只取序列最后一个时刻的特征作为全局表示
        # 直觉：最后时刻"见过"了整个历史序列
        return out[:, :, -1]   # [B, n]


# ============================================================
# 二、ConvTokenizer（序列的初始 Embedding）
# ------------------------------------------------------------
# 作用：将原始序列用 1D 卷积映射到高维 embedding 空间
#       类似 ViT 中用 2D 卷积做图像 Patch Embedding 的角色
#
# 输入：[B, in_chans, L]  （B=batch, in_chans=原始维度, L=序列长度）
# 输出：[B, L, embed_dim] （转置后符合 Transformer 输入格式）
# ============================================================
class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        # 1D 卷积：核大小3，stride=1，padding=1 → 长度不变，只改变通道数
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x: [B, in_chans, L]
        # proj 后：[B, embed_dim, L]
        # permute 后：[B, L, embed_dim] ← Transformer 标准格式：(batch, seq_len, dim)
        x = self.proj(x).permute(0, 2, 1)  # B, C, L -> B, L, C
        if self.norm is not None:
            x = self.norm(x)
        return x


# ============================================================
# 三、ConvDownsampler（序列下采样）
# ------------------------------------------------------------
# 作用：将序列长度减半，同时特征维度翻倍
#       类似图像中的池化层，但用 stride=2 的 1D 卷积实现
#
# 输入：[B, L, dim]
# 输出：[B, L/2, 2*dim]
# ============================================================
class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # stride=2：序列长度减半；输出通道 2*dim：维度翻倍（参数量不变）
        self.reduction = nn.Conv1d(
            dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        # x: [B, L, dim]
        # permute → [B, dim, L]，卷积后 → [B, 2*dim, L/2]，再 permute → [B, L/2, 2*dim]
        x = self.reduction(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        return x


# ============================================================
# 四、Mlp（标准两层 MLP，Transformer FFN 的通用实现）
# ------------------------------------------------------------
# 作用：Transformer 中 Attention 后的前馈网络
#       in → hidden → out，用 GELU 激活
#
# 输入：[..., in_features]  （任意 shape，最后一维是特征维）
# 输出：[..., out_features]
# ============================================================
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,   # 默认等于 in_features
        out_features=None,      # 默认等于 in_features
        act_layer=nn.GELU,      # GELU 激活（Transformer 标准）
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 两层全连接：in → hidden → out
        x = self.fc1(x)
        x = self.act(x)    # GELU 激活
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================
# 五、NATLayer（单层 NAT 注意力 + FFN）
# ------------------------------------------------------------
# NAT = Neighborhood Attention Transformer（邻域注意力）
# 原始 NAT 只关注局部邻域内的 token（降低计算量），
# 但此处已改为标准 MultiheadAttention（注释掉了 NAT，换成全局注意力）
#
# 结构（标准 Pre-Norm Transformer 层）：
#   x → LayerNorm → Attention → 残差 → LayerNorm → MLP → 残差
#
# 输入/输出：[B, L, dim]（形状不变）
# ============================================================
class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,     # NAT 邻域大小（现在未使用，已换为全局 MHA）
        dilation=None,     # NAT 空洞率（现在未使用）
        mlp_ratio=4.0,     # MLP 隐层维度 = dim * mlp_ratio
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,     # DropPath 概率（随机跳过整层，正则化）
        dropout=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)   # Attention 前的 LayerNorm

        # ⚠️ 注意：原本是 NeighborhoodAttention1D（邻域注意力）
        # 为避免依赖 natten 库，改成标准 MultiheadAttention（全局注意力）
        # 效果上：局部→全局，计算量增大但无需额外安装库
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)

        # ⚠️ drop_path 也被注释掉（依赖 timm 库），改成普通 Dropout
        # drop_path 的作用：训练时以一定概率跳过整个残差分支（比 dropout 更强的正则化）
        self.dropout = nn.Dropout(dropout)
        self.norm2 = norm_layer(dim)   # MLP 前的 LayerNorm
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        # Pre-Norm 结构（先 Norm 再 Attention，与原始 Transformer 的 Post-Norm 相反）
        shortcut = x                      # 残差连接的起点
        x = self.norm1(x)
        x = self.attn(x)                  # ⚠️ 此处 MultiheadAttention 需要 (query, key, value) 三个参数
                                          # 这里只传了一个 x，实际上是 bug（self-attention 应传三次 x）
                                          # 正确写法：x, _ = self.attn(x, x, x)
        x = shortcut + self.drop_path(x)  # ⚠️ drop_path 未定义（应为 self.dropout），也是遗留 bug
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================
# 六、NATBlock（多层 NATLayer + 可选下采样）
# ------------------------------------------------------------
# 作用：将多个 NATLayer 堆叠成一个 Stage（类比 Swin Transformer 的 Stage）
#       每个 Stage 结束时可选择下采样（序列长度减半，维度翻倍）
#
# 输入：[B, L, dim]
# 输出：
#   - 若有下采样：(下采样后特征 [B,L/2,2*dim], 下采样前特征 [B,L,dim])
#   - 若无下采样：(原始特征, 原始特征)（两个相同）
# ============================================================
class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,        # 该 Block 包含的 NATLayer 层数
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,   # 是否在 Block 末尾做下采样
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # 堆叠 depth 个 NATLayer
        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]   # 每层的 drop_path 概率不同（线性递增）
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        # 最后一级 Block 不做下采样（已到达最深层）
        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        # 逐层通过 NATLayer
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x       # 无下采样：两个输出相同
        return self.downsample(x), x   # 有下采样：(下采样后, 下采样前)
        # FPN 需要下采样前的特征（xo）做侧边连接，所以两个都返回


# ============================================================
# 七、PointsEncoder（点云/多边形点的全局特征提取）
# ------------------------------------------------------------
# 作用：将一组不定数量的点（如地图多边形的顶点）编码为固定维度向量
#       使用 PointNet 思想：逐点 MLP → MaxPool 全局特征 → 再次 MLP
#
# 使用场景：地图 encoder 中，对每条车道线的点集提取特征
#
# 输入：
#   x:    [B, M, feat_channel]  B=batch×多边形数, M=每个多边形的点数, C=点特征维度
#   mask: [B, M]                有效点的掩码（True=有效点）
# 输出：
#   [B, encoder_channel]        每个多边形的全局特征向量
# ============================================================
class PointsEncoder(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        # 第一阶段 MLP：逐点特征提取
        # feat_channel → 128 → 256
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        # 第二阶段 MLP：融合局部特征和全局特征
        # 输入 512 = 256(逐点) + 256(全局MaxPool)
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoder_channel),
        )

    @torch.amp.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x, mask=None):
        """
        PointNet 风格的点云编码，支持 mask（跳过无效填充点）

        x:    [B, M, feat_channel]
        mask: [B, M]  True=有效点
        返回：[B, encoder_channel]
        """
        bs, n, c = x.shape
        device = x.device

        if os.getenv("DEPLOY") != 'True':
            # ---- 训练/评估模式：用 mask 索引有效点，节省计算 ----
            mask = mask.bool()

            # Step1: 只对有效点做第一阶段 MLP
            x_valid = self.first_mlp(x[mask])       # [N_valid, 256]
            x_features = torch.zeros(bs, n, 256, device=device)
            x_features[mask] = x_valid              # 将有效点特征填回原位置

            # Step2: MaxPool 全局特征（PointNet 核心操作）
            # 对每个多边形的所有点取最大值，得到全局描述符
            pooled_feature = x_features.max(dim=1)[0]   # [B, 256]

            # Step3: 拼接逐点特征和广播的全局特征
            # 让每个点"知道"整体的全局信息
            x_features = torch.cat(
                [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
            )   # [B, M, 512]

            # Step4: 第二阶段 MLP，只处理有效点
            x_features_valid = self.second_mlp(x_features[mask])
            res = torch.zeros(bs, n, self.encoder_channel, device=device)
            res[mask] = x_features_valid

        else:
            # ---- 部署模式（ONNX/TensorRT）：不用 bool 索引，改为 mask 乘法 ----
            # 原因：TensorRT 不支持动态索引，用乘法代替
            x = x.view(bs * n, -1)
            x_features = self.first_mlp(x).view(bs, n, -1)
            mask_value = mask.unsqueeze(-1).repeat(1, 1, x_features.shape[-1])
            x_features = x_features * mask_value             # 无效点清零
            pooled_feature = x_features.max(dim=1, keepdim=True)[0]
            x_features = torch.cat(
                [x_features, pooled_feature.repeat(1, n, 1)], dim=-1
            )
            x_features = x_features.view(bs * n, -1)
            res = self.second_mlp(x_features).view(bs, n, -1)
            mask_value = mask.unsqueeze(-1).repeat(1, 1, res.shape[-1])
            res = res * mask_value

        # 最终 MaxPool：将变长点集压缩为固定长度的全局向量
        res = res.max(dim=1)[0]   # [B, encoder_channel]
        return res