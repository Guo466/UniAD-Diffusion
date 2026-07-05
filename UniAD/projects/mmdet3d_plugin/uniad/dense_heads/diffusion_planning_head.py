# ================================================================================
# DiffusionPlanningHead —— 融合 UniAD + pl_diffusion_models 的扩散规划头
#
# 【整体架构一句话描述】
#   把 UniAD 上游输出的 BEV 特征图 + Agent 跟踪特征 + 自车状态，
#   喂给一个基于 Flow Matching 的 DiT，让它学会从高斯噪声中"去噪"出合理的行驶轨迹。
#
# 【方案 A 融合思路（Quick Validation）】
#   UniAD Pipeline 输出（来自 MotionHead）：
#     - bev_embed       : (H*W, B, D=256)  BEV 场景特征（BEVFormer 生成的鸟瞰图）
#     - track_query     : (B, N, D=256)    所有 agent 的跟踪特征（TrackHead 维护）
#     - sdc_traj_query  : (num_layers, B, P, D)  自车候选轨迹特征（MotionHead 生成）
#     - sdc_track_query : (B, D)           自车跟踪特征（TrackHead 维护的自车状态）
#
#   通过 BEV 桥接层 + Ego 桥接层，将上述特征转为 DiT 所需的：
#     - context     : (B, N_ctx, D)  场景上下文（agent token + 压缩BEV token 拼接）
#     - ego_context : (B, 1, D)      自车状态特征（运动轨迹 + 跟踪状态融合）
#     - ego_routing : (B, 1, D)      自车意图特征（自车状态 + 驾驶命令融合）
#
#   DiT 扩散解码器（Rectified Flow / Flow Matching）输出：
#     - 自车未来 planning_steps 步轨迹（训练：FM MSE loss；推理：Euler ODE 5步）
#
# 【接口兼容性】
#   接口与 PlanningHeadSingleMode 保持完全兼容，只需在 config 中修改 type 即可替换。
# ================================================================================

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
# grad_checkpoint: 梯度检查点，训练时不保存中间激活值，前向传播时按需重算，以时间换显存
from mmdet.models.builder import HEADS, build_loss
# HEADS: mmdetection 的模型注册表，用 @HEADS.register_module() 注册后，
#        可以在 config 中用 type='DiffusionPlanningHead' 字符串实例化模型


# ============================================================
# 【第一部分】Tensor 变形工具函数
# 这三个函数解决一个核心问题：
#   BEVFormer 输出的 BEV 特征是 (H*W, B, D) 的展平序列格式，
#   但 Conv2d 需要 (B, D, H, W) 的空间格式。
#   这三个函数负责在两种格式之间自由转换。
# ============================================================

def _bev_seq_to_2d(x, h, w):
    """
    (h*w, B, D) → (B, D, h, w)
    用途：将 BEV 序列特征恢复为空间特征图，以便做 Conv2d 压缩
    原理：
      reshape(h, w, b, d) → 把 h*w 维度拆开
      permute(2,3,0,1)    → 调整维度顺序为 (B, D, h, w)
    """
    seq, b, d = x.shape
    return x.reshape(h, w, b, d).permute(2, 3, 0, 1).contiguous()


def _bev_2d_to_seq(x):
    """
    (B, D, h, w) → (h*w, B, D)
    用途：Conv 处理完毕后，将特征图转回 BEVFormer 原始格式（BEV Adapter 用）
    """
    b, d, h, w = x.shape
    return x.permute(2, 3, 0, 1).reshape(h * w, b, d).contiguous()


def _bev_2d_to_flat(x):
    """
    (B, D, h, w) → (B, h*w, D)
    用途：将压缩后的 BEV 特征图展平为 Transformer 格式（batch-first，序列在第1维）
    注意：与 _bev_2d_to_seq 的区别在于 batch 维度位置不同：
      _bev_2d_to_seq  → (h*w, B, D)，BEVFormer 原始格式（seq-first）
      _bev_2d_to_flat → (B, h*w, D)，Transformer batch-first 格式
    """
    b, d, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, d).contiguous()


# ============================================================
# 【第二部分】基础工具类（全部自包含，无外部依赖）
# 学习顺序建议：RMSNorm → MLP2 → TimestepEmbedder → modulate → DiTBlock → FinalLayer → SinusoidalPE
# ============================================================

class RMSNorm(nn.Module):
    """
    【RMS Layer Normalization】
    与 LayerNorm 的区别：
      - LayerNorm:  x_norm = (x - mean) / std * gamma + beta  （有均值中心化）
      - RMSNorm:    x_norm = x / rms(x) * gamma               （只做缩放，不减均值）

    优点：计算更简单，训练更稳定，LLaMA / Gemma 等大模型都在用。
    这里没有可学习的 gamma（elementwise_affine=False），只做纯归一化。

    数学公式：
      RMS(x) = sqrt( mean(x²) + eps )
      output  = x / RMS(x)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 防止除以零的小常数

    def forward(self, x):
        # x.pow(2).mean(-1, keepdim=True): 计算最后一维的均方值 → (B, ..., 1)
        # torch.rsqrt: 倒数开方，即 1/sqrt(x)，比 1/sqrt(x) 更数值稳定
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MLP2(nn.Module):
    """
    【两层 MLP + GELU + RMSNorm】
    这是整个文件中最基础的构件，几乎所有模块都用它做特征投影。

    结构：fc1 → GELU → RMSNorm → fc2
    （注意：norm 在激活之后、fc2 之前，起到稳定中间特征分布的作用）

    为什么用 GELU 而不是 ReLU？
      GELU(x) = x * Φ(x)，是概率性软门控，梯度比 ReLU 更平滑，
      Transformer 系列模型的标准选择（BERT/GPT/DiT 都用 GELU）。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, use_norm=True):
        super().__init__()
        out_features     = out_features or in_features      # 默认输出维度 = 输入维度
        hidden_features  = hidden_features or in_features   # 默认隐层维度 = 输入维度
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.norm = RMSNorm(hidden_features) if use_norm else nn.Identity()
        self.fc2  = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # 数据流：x → fc1 → GELU → RMSNorm → fc2 → output
        return self.fc2(self.norm(self.act(self.fc1(x))))


class TimestepEmbedder(nn.Module):
    """
    【时间步嵌入：Flow Matching 的核心组件之一】

    作用：将扩散过程中的时间步 t（一个标量）转化为向量，
          让 DiT 知道"当前处于噪声去除的第几步"。

    为什么需要这个？
      Flow Matching 训练时，每个样本都有一个随机时间步 t ∈ [0, 1]，
      模型需要根据 t 的大小来决定预测的"方向"（t=0 时轨迹接近纯噪声，
      t=1 时接近真实轨迹）。单个标量 t 信息量太少，需要先变成高维向量。

    编码方式（两步）：
      Step1: 正弦编码（sinusoidal_embedding）
        - 类比 Transformer 里的位置编码，但这里编码的是时间步而非位置
        - 用不同频率的 cos/sin 函数表示 t，频率从高到低，每个分量感知不同"尺度"的 t
        - 公式：freqs[i] = exp(-log(10000) * i / half)
                cos(t * freqs[i]) 和 sin(t * freqs[i]) 拼接

      Step2: MLP 投影（Linear → SiLU → Linear）
        - 将正弦编码的固定维度 freq_dim=256 投影到模型的 hidden_size=256
        - SiLU(x) = x * sigmoid(x)，比 ReLU 更平滑，DiT 原文用的激活函数

    输入：t: (B,)  每个样本的时间步（已缩放到 [0, 1000]）
    输出：(B, hidden_size)
    """
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),  # 正弦编码 → 模型维度
            nn.SiLU(),                          # Sigmoid Linear Unit 激活
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(t, dim, max_period=10000):
        """
        正弦时间步编码（参考 DDPM / OpenAI Glide 实现）

        参数：
            t          : (B,) 时间步标量（此处已乘以 1000，即 t ∈ [0, 1000]）
            dim        : 编码维度（此处 freq_dim=256）
            max_period : 控制最低频率（越大→最低频率越低→对大 t 更敏感）

        计算步骤：
          1. 生成 half=128 个指数间隔的频率：freqs[i] = exp(-log(10000) * i / 128)
             → freqs[0]=1.0（最高频），freqs[127]≈0.0001（最低频）
          2. 外积：args[b, i] = t[b] * freqs[i]  → (B, 128)
          3. 拼接 cos 和 sin：emb = [cos(args), sin(args)] → (B, 256)

        直觉：用不同频率的三角函数感知时间步 t 的不同"尺度"，
              高频分量区分相邻的小 t 差异，低频分量区分较大的 t 变化。
        """
        half = dim // 2
        # 生成 half 个频率（对数间隔，从1到约0.0001）
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        # args[b, i] = t[b] * freqs[i]，外积，形状 (B, half)
        args = t[:, None].float() * freqs[None]
        # 拼接 cos 和 sin → (B, dim)
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果 dim 是奇数，补一列 0（一般不会触发，dim=256 是偶数）
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        # t: (B,) → 正弦编码 → (B, freq_dim=256) → MLP → (B, hidden_size=256)
        return self.mlp(self.sinusoidal_embedding(t, self.freq_dim))


def modulate(x, shift, scale):
    """
    【AdaLN（Adaptive Layer Normalization）的调制操作】

    这是 DiT（Diffusion Transformer）论文的核心创新点：
    用时间步 t 动态地调整每个 Transformer 层的归一化参数。

    传统 LayerNorm：x_norm = (x - mean) / std * γ + β （γ, β 是固定参数）
    AdaLN-Zero：   x_mod  = x_norm * (1 + scale) + shift  （scale, shift 由 y=t_emb 动态生成）

    为什么有效？
      - t 决定了当前处于扩散过程的哪个阶段
      - 不同阶段，网络应该有不同的"侧重点"（t 近 1 关注轨迹细节，t 近 0 关注整体方向）
      - 通过 scale/shift 动态调整特征分布，让网络自适应地调整各层行为

    参数：
      x     : (..., D)   LayerNorm 归一化后的特征
      shift : (B, D)     偏移量（动态生成的 β 替代）
      scale : (B, D)     缩放量（动态生成的 γ 替代，注意是乘以 1+scale 而非 scale）

    注意 unsqueeze(1)：
      scale 是 (B, D)，x 是 (B, T, D)，需要在 T 维度广播
      → scale.unsqueeze(1) 变成 (B, 1, D)，自动广播到 (B, T, D)
    """
    if shift is None:
        shift = torch.zeros_like(scale)  # 如果没有 shift，用 0（保持中心不变）
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    【核心：单个 DiT Block（Diffusion Transformer Block）】

    DiT Block 是本项目的核心结构，学习时请重点理解。

    【结构（每个 Block 内部的数据流）】：
      输入 x (B,T,D) ← ego 轨迹 token（T=6个时间步）
      输入 context (B,N,D) ← 场景上下文（N个agent + 64个BEV token）
      输入 y (B,D) ← 时间步嵌入（告知当前扩散步 t）

      Step1: y → adaLN_modulation → 6组参数 (s1,c1, s2,c2, s3,c3)
             这6组参数分别用于调制 norm1/norm2/norm3 的 shift/scale

      Step2: Self-Attention（轨迹 token 内部交互）
             x_mod = modulate(norm1(x), s1, c1)
             x = x + self_attn(x_mod, x_mod, x_mod)

      Step3: Cross-Attention（轨迹 token 查询场景上下文）★ 关键
             x_mod = modulate(norm2(x), s2, c2)
             x = x + cross_attn(query=x_mod, key=context, value=context)
             ← 这一步让"规划轨迹"去"看"周围的 agent 和地图信息

      Step4: FFN（前馈网络）
             x = x + ffn(modulate(norm3(x), s3, c3))

    【与标准 Transformer Block 的区别】：
      标准: LN(x) → Attention → Add   （LN 参数固定）
      DiT:  AdaLN(x, y) → Attention → Add  （LN 参数由时间步 y 动态生成）

    【elementwise_affine=False 的含义】：
      nn.LayerNorm 默认有可学习的 γ 和 β 参数（elementwise_affine=True）
      这里设为 False，因为 shift/scale 已经由 adaLN_modulation 动态提供，
      不需要额外的固定参数，参数量更少，也避免冗余。
    """
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        # 三个归一化层，关闭固定参数（由 adaLN 动态生成）
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)  # Self-Attn 前
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)  # Cross-Attn 前
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)  # FFN 前
        self.num_heads = heads

        # Self-Attention：ego 轨迹 token 之间相互交流（T=6 个时间步）
        # batch_first=True：输入格式 (B, T, D)，更符合直觉
        self.self_attn  = nn.MultiheadAttention(dim, heads, batch_first=True)

        # Cross-Attention：轨迹 token "看"场景上下文
        # query: 轨迹 token (B, T, D)
        # key/value: 场景上下文 (B, N, D)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # FFN（前馈网络）：两层线性变换 + GELU
        ffn_dim = int(dim * mlp_ratio)  # 隐层维度 = dim * mlp_ratio（默认扩展2倍）
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim),
        )

        # 【AdaLN 调制网络】：时间步嵌入 y → 6组 shift/scale 参数
        # SiLU 激活 + Linear(dim → 6*dim)
        # 6*dim = 6组参数：(s1, c1) for norm1, (s2, c2) for norm2, (s3, c3) for norm3
        # 其中 s=shift（偏移），c=scale（缩放）
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, context, y, self_attn_mask=None, cross_attn_mask=None):
        """
        Args:
            x              : (B, T, D)   ego 轨迹 token，T=planning_steps=6
            context        : (B, N, D)   场景上下文（agent + BEV token）
            y              : (B, D)      时间步 conditioning（由 TimestepEmbedder 生成）
            self_attn_mask : (B, T, T)   轨迹 token 的 attention mask（可选）
            cross_attn_mask: (B, T, N)   跨注意力 mask（可选，当前代码未传入）
        """
        # 【Step1】由时间步嵌入 y 生成 6 组 AdaLN 参数
        # y: (B, D) → Linear → (B, 6*D) → chunk(6) → 6个 (B, D)
        s1, c1, s2, c2, s3, c3 = self.adaLN_modulation(y).chunk(6, dim=-1)

        # 【Step2】Self-Attention（轨迹 token 内部交互）
        # 目的：让6个时间步的轨迹 token 相互"感知"，建立时序依赖
        x_mod = modulate(self.norm1(x), s1, c1)  # 动态归一化
        if self_attn_mask is not None:
            # 将布尔 mask 转为 float attention mask（True位置 → -inf，被遮蔽）
            # 需要扩展到多头格式：(B, T, T) → (B*heads, T, T)
            B, T = x.shape[0], x.shape[1]
            sa = (self_attn_mask.unsqueeze(1)
                  .expand(-1, self.num_heads, -1, -1)
                  .reshape(B * self.num_heads, T, T)
                  .float()
                  .masked_fill(self_attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                               .reshape(B * self.num_heads, T, T).bool(), float('-inf')))
            attn_out, _ = self.self_attn(x_mod, x_mod, x_mod, attn_mask=sa)
        else:
            # 无 mask：全局自注意力（T=6 的序列，开销极小）
            attn_out, _ = self.self_attn(x_mod, x_mod, x_mod)
        x = x + attn_out  # 残差连接

        # 【Step3】Cross-Attention（轨迹 token 查询场景上下文）★ 最关键的一步
        # 目的：让规划轨迹"感知"周围的 agent 和地图信息，做出合理的避让/跟随决策
        # query = 轨迹 token（我想去哪）
        # key/value = 场景上下文（周围有什么）
        x_mod = modulate(self.norm2(x), s2, c2)
        attn_out, _ = self.cross_attn(x_mod, context, context)
        # cross_attn(query, key, value) → 轨迹 token 用自己的 query 去 attend 场景中的信息
        x = x + attn_out  # 残差连接

        # 【Step4】FFN（前馈网络，逐位置非线性变换）
        x = x + self.ffn(modulate(self.norm3(x), s3, c3))  # 残差连接
        return x


class FinalLayer(nn.Module):
    """
    【DiT 输出头：AdaLN → Linear → 轨迹坐标】

    这是 DiT 的最后一层，将 Transformer 的 hidden 特征投影到实际输出空间。

    与普通线性层的区别：在 Linear 之前，用时间步嵌入 y 做一次 AdaLN 调制，
    让输出的尺度和偏移也随扩散时间步动态变化。

    输入：
      x : (B, T, D)   经过所有 DiTBlock 处理后的轨迹 token
      y : (B, D)      时间步嵌入
    输出：
      (B, T, output_size)  预测的轨迹 delta（归一化后的逐步位移，output_size=2 即 x,y）
    """
    def __init__(self, hidden_size, output_size=2):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size)  # 投影到输出维度（2=x,y坐标）
        # 只需 2 组参数：shift 和 scale
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, y):
        # y → (shift, scale) → 动态归一化 x → Linear 投影
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)  # 各 (B, D)
        return self.linear(modulate(self.norm(x), shift, scale))  # (B, T, output_size)


class SinusoidalPE(nn.Module):
    """
    【正弦位置编码（Sinusoidal Positional Encoding）】

    用途：给 ego 轨迹 token 注入时间维度的位置信息，
         让 DiT 知道"第 1 步轨迹"和"第 6 步轨迹"是不同位置的 token。

    与 TimestepEmbedder 的区别：
      TimestepEmbedder: 编码扩散时间步 t（0~1000，控制噪声程度）
      SinusoidalPE:     编码 token 的位置序号（0~5，对应规划轨迹的 6 个时刻）

    实现：经典 Transformer 位置编码（Vaswani et al. 2017），参数不可学习，
         注册为 buffer（跟随模型 .to(device) 移动，但不参与梯度计算）。

    公式：
      PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
      PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=80):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 分母：10000^(2i/d_model)，取对数再 exp 更稳定
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # 偶数维：sin
        pe[:, 1::2] = torch.cos(pos * div)  # 奇数维：cos
        self.register_buffer('pe', pe)      # 不参与梯度，但跟随 .to(device)

    def forward(self, x):
        # x: (B, T, D)，pe[:T]: (T, D) → unsqueeze(0) → (1, T, D) → 广播到 (B, T, D)
        return x + self.pe[:x.size(1)].unsqueeze(0)


# ============================================================
# 【第三部分】桥接模块：UniAD 特征格式 → DiT 输入格式
# 这是"接口层"，负责把上游模块产出的各种特征统一转换为 DiT 能理解的格式。
# ============================================================

class BEVToContextBridge(nn.Module):
    """
    【场景上下文桥接层】
    将 UniAD 的 BEV 特征图（200×200）和 track_query（N个agent）
    融合压缩为 DiT 所需的 context token 序列。

    【为什么要压缩 BEV？】
      原始 BEV: 200×200=40000 个 token，直接做 Cross-Attention 计算量爆炸
      压缩后：  8×8=64 个 token，Cross-Attention 从 O(T×40000) 降至 O(T×(N+64))

    【压缩方式】：两次 stride=5 的 Conv2d（200→40→8），信息有损但保留全局结构
    
    输入：
        bev_embed   : (H*W, B, D)   展平的 BEV 特征（seq-first 格式）
        track_query : (B, N, D)     所有 agent 的跟踪特征
    输出：
        context  : (B, N+64, D)      合并的场景上下文 token
        mask_ctx : (B, N+64, N+64)   全 True 的 mask（所有 token 相互可见）
    """
    def __init__(self, bev_in_dim=256, track_in_dim=256, out_dim=256,
                 n_bev_tokens=64, bev_h=200, bev_w=200, max_agents=50):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        # 【BEV 压缩卷积】：两次 5×5 stride=5 卷积，尺寸 200→40→8
        # stride=5 保证整除（200/5=40，40/5=8），无需 padding
        self.bev_compress = nn.Sequential(
            nn.Conv2d(bev_in_dim, out_dim, kernel_size=5, stride=5, padding=0),  # 200×200 → 40×40
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=5, stride=5, padding=0),     # 40×40 → 8×8
            nn.ReLU(),
        )
        compressed_h = bev_h // 5 // 5   # 200//5//5 = 8
        compressed_w = bev_w // 5 // 5   # 8
        self.actual_bev_tokens = compressed_h * compressed_w  # 8*8 = 64

        # 【Track 特征投影】：Linear + LayerNorm，将 track_query 投影到统一的 out_dim 维度
        self.track_proj = nn.Sequential(nn.Linear(track_in_dim, out_dim), nn.LayerNorm(out_dim))

        # 【BEV token 可学习位置编码】：64 个位置各有一个可学习向量
        # 告知 DiT 每个 BEV token 的"空间位置"（类比 ViT 的 patch position embedding）
        self.bev_pos_embed = nn.Parameter(torch.randn(1, self.actual_bev_tokens, out_dim) * 0.02)
        # 初始化为小值（0.02 * randn），避免初始时位置编码过强压制内容特征

        # 【Agent 内部 Transformer】：让 agent 之间相互感知（如：别车的行驶方向会影响自车决策）
        # 显存优化配置：nhead=4（原8），FFN dim=out_dim（原2*out_dim），层数=1（原2）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=out_dim,
            dropout=0.1, batch_first=True
        )
        self.agent_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, bev_embed, track_query, track_mask=None):
        B = track_query.shape[0]
        H, W = self.bev_h, self.bev_w

        # ① BEV 特征压缩
        bev_2d     = _bev_seq_to_2d(bev_embed, H, W)          # (B, D, 200, 200)
        bev_tokens = self.bev_compress(bev_2d)                 # (B, D, 8, 8)
        bev_tokens = _bev_2d_to_flat(bev_tokens)               # (B, 64, D)
        bev_tokens = bev_tokens + self.bev_pos_embed[:, :bev_tokens.shape[1]]
        # 加位置编码：bev_pos_embed (1, 64, D) 广播到 (B, 64, D)

        # ② 投影 track 特征
        track_feat = self.track_proj(track_query)              # (B, N, D)

        # ③ Agent 内部交互
        # 条件：N > 0（无 agent 时跳过，避免 Transformer 处理空序列出错）
        if track_feat.shape[1] > 0:
            # kpm（key_padding_mask）：标记哪些 agent 是 padding（无效的）
            # ~track_mask：True 表示"此 agent 应被忽略"（PyTorch 约定）
            kpm = (~track_mask) if track_mask is not None else None
            if self.training:
                # 训练时用梯度检查点：不保存 agent_encoder 的中间激活，反向传播时重算
                track_feat = grad_checkpoint(
                    self.agent_encoder, track_feat,
                    None, kpm, use_reentrant=False
                )
            else:
                track_feat = self.agent_encoder(track_feat, src_key_padding_mask=kpm)

        # ④ 拼接 agent 特征和 BEV token，构成完整 context
        # 顺序：[track_feat | bev_tokens]，DiT 通过 Cross-Attention 同时看到所有
        context = torch.cat([track_feat, bev_tokens], dim=1)   # (B, N+64, D)
        N_ctx   = context.shape[1]
        # 全 True mask：所有 context token 相互可见（当前不做稀疏 attention）
        mask_ctx = torch.ones(B, N_ctx, N_ctx, device=context.device, dtype=torch.bool)
        return context, mask_ctx


class EgoContextBridge(nn.Module):
    """
    【自车上下文桥接层】
    从 UniAD MotionHead 输出中提取自车状态特征，生成 DiT 所需的两路条件信号：
      - ego_context（自车状态）：告知 DiT "自车当前的运动状态和周围感知"
      - ego_routing（导航意图）：告知 DiT "自车想要往哪个方向走"

    【两路输出的设计逻辑】：
      ego_context：来自 sdc_traj_query（运动预测）+ sdc_track_query（跟踪状态）融合
                  → 编码"当前场景下自车的运动状态"
      ego_routing：在 ego_context 基础上，再融合 command（右转/直行/左转）的嵌入
                  → 编码"自车的驾驶意图"，用于调制 DiT 的时间步嵌入 y

    在 _dit_forward 中，这两路信号的使用方式：
      x (轨迹 token) += vector_in(ego_context)  ← 直接加到轨迹 token 上（特征层面）
      y (时间步嵌入) += routing_in(ego_routing)  ← 加到时间步嵌入 y 上（条件层面）

    输入：
        sdc_traj_query  : (num_layers, B, P, D)  MotionHead 各层输出的 SDC 轨迹 query
        sdc_track_query : (B, D)                 TrackHead 维护的 SDC 跟踪状态
        command         : (B,)                   驾驶命令（0=右转, 1=直行, 2=左转）
    输出：
        ego_context : (B, 1, D)
        ego_routing : (B, 1, D)
    """
    def __init__(self, in_dim=256, out_dim=256, n_commands=3):
        super().__init__()
        # ego_fuser：融合 sdc_traj 特征和 sdc_track 特征
        # 输入：dim*2（拼接两路 256 维特征）→ 输出：dim
        self.ego_fuser = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )
        # command_embed：将离散的驾驶命令（0/1/2）嵌入为连续向量
        # Embedding(3, out_dim)：3类命令，每类一个 256 维可学习向量
        self.command_embed = nn.Embedding(n_commands, out_dim)

        # routing_fuser：将 ego_feat 和 command 嵌入融合为 routing 特征
        self.routing_fuser = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )

    def forward(self, sdc_traj_query, sdc_track_query, command):
        # ① 取 MotionHead 最后一层的 SDC 轨迹 query，沿 P 维取最大值
        # sdc_traj_query[-1]: (B, P, D)，P=6 条候选轨迹
        # .max(dim=1)[0]: 取 6 条候选中特征值最大的，相当于"选最置信的那条"
        sdc_traj  = sdc_traj_query[-1].max(dim=1)[0]          # (B, D)

        # ② 分离梯度（detach），避免规划损失反向传播到跟踪头
        # 跟踪头有自己的损失，两个任务的梯度不应互相污染
        sdc_track = sdc_track_query.detach()                   # (B, D)

        # ③ 融合自车轨迹特征和跟踪特征
        ego_feat  = self.ego_fuser(torch.cat([sdc_traj, sdc_track], dim=-1))  # (B, D)

        # ④ 将驾驶命令映射为嵌入向量
        # command: (B,) LongTensor → Embedding → (B, D)
        cmd_embed = self.command_embed(command)                 # (B, D)

        # ⑤ 融合 ego 特征和 command 嵌入，生成 routing（导航意图）特征
        routing   = self.routing_fuser(torch.cat([ego_feat, cmd_embed], dim=-1))  # (B, D)

        # unsqueeze(1)：添加"序列"维度，(B, D) → (B, 1, D)
        # 这样与 context (B, N, D) 格式统一，便于后续操作
        return ego_feat.unsqueeze(1), routing.unsqueeze(1)     # (B,1,D), (B,1,D)


# ============================================================
# 【第四部分】主模块：DiffusionPlanningHead
# 这是所有组件的"组装者"和"调度者"
# ============================================================

@HEADS.register_module()  # 注册到 mmdet HEADS 注册表
class DiffusionPlanningHead(nn.Module):
    """
    【主模块：基于 Flow Matching（Rectified Flow）的自车规划头】

    替换 PlanningHeadSingleMode，接口完全兼容：
        forward_train(bev_embed, outs_motion, sdc_planning, sdc_planning_mask, command, gt_future_boxes)
        forward_test (bev_embed, outs_motion, outs_occflow, command)

    【Flow Matching（Rectified Flow）基本原理】
      传统方法：直接回归轨迹坐标
      本方法：学习一个"速度场"（velocity field），从高斯噪声"流动"到真实轨迹

      训练：
        1. 随机采样时间步 t ∈ [0,1]（Sigmoid 变换使 t 集中在 0.4~0.6 附近）
        2. 插值：z_t = t * x_data + (1-t) * noise  （在数据和噪声之间线性插值）
        3. DiT 预测"方向"：pred = DiT(z_t, t, context) ≈ x_data - noise
        4. 损失：MSE(pred, x_data - noise)

      推理（Euler ODE 积分）：
        1. 从高斯噪声 x_0 出发
        2. 重复 n_steps 次：x_{i+1} = x_i + DiT(x_i, t_i, context) * (1/n_steps)
        3. x_final → 反归一化 → 累积求和 → 绝对坐标轨迹

    【核心流程】：
        1. BEV(200×200) → Conv 压缩 → 64 BEV token
        2. track_query(N个agent) → Agent Encoder → agent 特征
        3. 拼接 → DiT context (N+64, D)
        4. sdc_traj/track_query + command → ego_context(1,D) + ego_routing(1,D)
        5. DiT Flow Matching（训练/推理不同路径）
    """

    def __init__(
        self,
        embed_dims=256,                 # Transformer 特征维度，与 UniAD 对齐
        planning_steps=6,               # 规划步数（6步 × 0.5s = 3秒）
        n_bev_tokens=64,                # BEV 压缩后的 token 数（8×8=64）
        dit_depth=4,                    # DiTBlock 层数（配置文件中设为 2 节省显存）
        dit_heads=8,                    # Multi-head Attention 头数
        sample_steps=5,                 # 推理时 Euler ODE 积分步数
        bev_h=200,                      # BEV 特征图高度（nuScenes 默认）
        bev_w=200,                      # BEV 特征图宽度
        max_agents=50,                  # 最多处理的 agent 数量
        output_dim=2,                   # 输出维度（2=x,y；3=x,y,heading）
        flow_matching_loss_weight=1.0,  # FM 损失的权重系数
        ade_loss_weight=0.5,            # ADE 辅助损失权重（0=关闭，>0=开启）
                                        # 改进：在归一化空间计算，量纲与 FM loss 完全一致（均约 0~2）
                                        # 因此 0.5 的权重合理：FM(1.0) 主导速度场，ADE(0.5) 辅助加速收敛
        loss_planning=None,             # 兼容接口，不使用（被 FM loss 替代）
        loss_collision=None,            # 碰撞损失（可选，显存紧张时禁用）
        loss_kinematic=None,            # 兼容接口，不使用
        planning_eval=False,            # 训练时是否实时评估（耗显存，一般 False）
        use_col_optim=False,            # 推理时是否用 CasADi 碰撞避免优化
        col_optim_args=dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0),
        with_adapter=False,             # 是否启用 BEV Adapter（显存优化时禁用）
        n_commands=3,                   # 驾驶命令类别数（右转/直行/左转）
    ):
        super().__init__()
        # 保存需要在 forward 中使用的参数
        self.planning_steps            = planning_steps
        self.sample_steps              = sample_steps
        self.output_dim                = output_dim
        self.flow_matching_loss_weight = flow_matching_loss_weight
        self.ade_loss_weight           = ade_loss_weight  # ADE 辅助损失权重
        self.planning_eval             = planning_eval
        # 训练迭代计数器（用于早期诊断，只在前 N 个 iter 打印详细信息）
        self._iter_count               = 0
        self.use_col_optim             = use_col_optim
        self.embed_dims                = embed_dims

        # ---- 桥接层（UniAD格式 → DiT格式）----
        self.bev_track_bridge = BEVToContextBridge(
            bev_in_dim=embed_dims, track_in_dim=embed_dims, out_dim=embed_dims,
            n_bev_tokens=n_bev_tokens, bev_h=bev_h, bev_w=bev_w, max_agents=max_agents,
        )
        self.ego_bridge = EgoContextBridge(in_dim=embed_dims, out_dim=embed_dims, n_commands=n_commands)

        # ---- DiT 扩散解码器组件 ----
        # t_embedder:  时间步嵌入，将标量 t 映射到 D 维向量
        self.t_embedder   = TimestepEmbedder(embed_dims)
        # ego_time_pe: 轨迹 token 的位置编码，区分 6 个时间步
        self.ego_time_pe  = SinusoidalPE(embed_dims, max_len=planning_steps + 10)
        # preproj:     将 (x,y) 噪声轨迹从 2 维投影到 D=256 维（"升维"）
        self.preproj      = MLP2(output_dim, embed_dims, embed_dims)
        # vector_in:   将 ego_context 特征融合到轨迹 token（特征层面的条件注入）
        self.vector_in    = MLP2(embed_dims, embed_dims, embed_dims)
        # routing_in:  将 ego_routing 特征融合到时间步嵌入 y（条件层面的驾驶意图注入）
        self.routing_in   = MLP2(embed_dims, embed_dims, embed_dims)
        # context_in:  对 context token 做额外投影（统一表示空间）
        self.context_in   = MLP2(embed_dims, embed_dims, embed_dims)
        # dit_blocks:  核心 DiT Block 堆叠（dit_depth 个，每个包含 Self/Cross-Attn + FFN）
        self.dit_blocks   = nn.ModuleList([
            DiTBlock(dim=embed_dims, heads=dit_heads, mlp_ratio=2.0)
            for _ in range(dit_depth)
        ])
        # final_layer: 将 D 维 token 投影回 (x,y) 轨迹坐标空间
        self.final_layer  = FinalLayer(hidden_size=embed_dims, output_size=output_dim)

        # ---- 轨迹归一化统计量（register_buffer）----
        # register_buffer：将 tensor 注册为模型的 buffer，不参与梯度计算，
        #                   但会随模型保存/加载，并跟随 .to(device) 移动
        # 这些统计量来自 nuScenes 数据集的轨迹分布估计：
        #   delta_x（前向位移）：均值≈0.4m/step，标准差≈0.3m/step（城区匀速行驶）
        #   delta_y（侧向位移）：均值≈0（对称），标准差≈0.15m/step
        self.register_buffer(
            'traj_mean', torch.tensor([0.4, 0.0] if output_dim == 2 else [0.4, 0.0, 0.0])
        )
        self.register_buffer(
            'traj_std',  torch.tensor([0.3, 0.15] if output_dim == 2 else [0.3, 0.15, 0.1])
        )

        # ---- 碰撞损失（兼容 PlanningHeadSingleMode 接口）----
        self.loss_collision = nn.ModuleList()
        if loss_collision is not None:
            for cfg in loss_collision:
                self.loss_collision.append(build_loss(cfg))
        # 当前配置（base_e2e_diffusion.py）中 loss_collision=None，此列表为空

        # ---- BEV Adapter（可选，显存优化版）----
        # 在 BEV 特征上叠加一个轻量卷积残差，微调 BEV 特征分布以适配规划任务
        # 显存优化：从原版 3 层（中间通道 128）→ 1 层（中间通道 64），减少 3/4 参数
        self.with_adapter = with_adapter
        if with_adapter:
            self.bev_adapter = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 4, 3, padding=1),  # 降维：256→64
                nn.ReLU(),
                nn.Conv2d(embed_dims // 4, embed_dims, 1),             # 升维：64→256
            )

        # ----------------------------------------------------------------
        # 【关键初始化】DiT 全链路零初始化（AdaLN-Zero 标准方案）
        # ----------------------------------------------------------------
        # 问题根源（之前的版本只初始化了 adaLN_modulation，但不够）：
        #   - adaLN_modulation 零初始化 → shift=0, scale=0
        #   - modulate(x, 0, 0) = x（不是0！而是 LayerNorm 后的 x）
        #   - self_attn(x, x, x) / cross_attn / ffn 仍然产生非零输出
        #   - 这些非零输出通过残差路径累积，最终 pred ≠ 0，FM loss 爆炸
        #
        # 正确方案（参考 DiT 原文 Sec 3.1 AdaLN-Zero）：
        #   同时将 attention/FFN 的最后一个投影层零初始化
        #   → 每个 Block 的残差贡献 = 0
        #   → Block 退化为恒等映射（输出 = 输入）
        #   → final_layer.linear 零初始化后，整个 DiT 输出 pred = 0
        #   → FM loss = ||target||² = ||gt_norm - noise||² ≈ 2（正态分布差的方差）
        #
        # 初始化列表：
        #   1. final_layer.linear           → 零初始化（直接截断最终输出）
        #   2. final_layer.adaLN_modulation 最后一层 → 零初始化
        #   3. 每个 DiTBlock：
        #      a. adaLN_modulation 最后一层 → 零初始化（调制参数=0，恒等调制）
        #      b. self_attn.out_proj        → 零初始化（自注意力残差=0）
        #      c. cross_attn.out_proj       → 零初始化（交叉注意力残差=0）
        #      d. ffn 最后一层（Linear）    → 零初始化（FFN残差=0）

        # final_layer 输出投影
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        # final_layer AdaLN 调制层
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        # 每个 DiTBlock：出口投影全部零初始化
        for block in self.dit_blocks:
            # AdaLN 调制输出层（shift/scale=0 → 恒等调制，但注意 modulate 仍传递 x）
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
            # Self-Attention 输出投影（零初始化 → 自注意力的残差贡献=0）
            nn.init.zeros_(block.self_attn.out_proj.weight)
            nn.init.zeros_(block.self_attn.out_proj.bias)
            # Cross-Attention 输出投影（零初始化 → 交叉注意力的残差贡献=0）
            nn.init.zeros_(block.cross_attn.out_proj.weight)
            nn.init.zeros_(block.cross_attn.out_proj.bias)
            # FFN 最后一层（零初始化 → FFN的残差贡献=0）
            nn.init.zeros_(block.ffn[-1].weight)
            nn.init.zeros_(block.ffn[-1].bias)

    # ------------------------------------------------------------------
    # 【辅助方法】轨迹归一化 / 反归一化
    # ------------------------------------------------------------------

    def normalize_traj(self, delta):
        """
        将轨迹增量（位移）归一化到接近标准正态分布。
        delta: (B, T, output_dim)  逐步位移，单位：米

        为什么要归一化？
          Flow Matching 假设学习的是从 N(0,1) 到数据分布的"流"，
          如果数据的量纲与噪声差异很大（如 delta_x 均值 0.4m 但噪声均值 0），
          训练会很难收敛。归一化后，数据和噪声在相同量级，Flow Matching 更稳定。
        """
        return (delta - self.traj_mean) / (self.traj_std + 1e-6)

    def denormalize_traj(self, delta_norm):
        """
        将归一化后的增量反归一化回实际坐标（米）。
        推理时使用：DiT 输出的是归一化空间的预测，需转回实际坐标。
        """
        return delta_norm * self.traj_std + self.traj_mean

    # ------------------------------------------------------------------
    # 【核心方法①】DiT 单步前向（预测 drift/速度场）
    # ------------------------------------------------------------------

    def _dit_forward(self, z_t, context, t_emb, ego_context, ego_routing):
        """
        DiT 单次前向传播，预测当前时刻的"drift"（流动方向）。

        【数据流详解】：
          z_t (B,T,2)  → preproj  → x (B,T,D)   ← 原始噪声轨迹升维
          ego_time_pe(x)           → x (B,T,D)   ← 注入时间步位置编码
          vector_in(ego_context)   → x (B,T,D)   ← 注入自车状态（特征层面）
          context_in(context)      → ctx (B,N,D)  ← 统一投影场景特征
          t_emb + routing_in(ego_routing) → y (B,D) ← 融合时间步和驾驶意图
          for each DiTBlock:
            x = block(x, ctx, y)                 ← 轨迹感知场景，并被 t/意图调制
          final_layer(x, y)        → pred (B,T,2) ← 降维输出轨迹 drift

        Args:
            z_t         : (B, T, output_dim)  当前时刻的噪声轨迹（Flow中的 z(t)）
            context     : (B, N_ctx, D)       场景上下文（agent + BEV token）
            t_emb       : (B, D)              时间步嵌入（由 t_embedder 生成）
            ego_context : (B, 1, D)           自车状态特征
            ego_routing : (B, 1, D)           自车驾驶意图特征
        Returns:
            pred : (B, T, output_dim)  预测的 drift（即速度场 v(z_t, t)）
        """
        B, T, _ = z_t.shape

        # ① 原始噪声轨迹升维（2D坐标→256D特征）
        x   = self.preproj(z_t)                                     # (B, T, D)

        # ② 注入时间步位置编码（区分规划的 T=6 个不同时刻）
        x   = self.ego_time_pe(x)                                   # (B, T, D)

        # ③ 注入自车状态（将 ego_context 扩展到每个时间步并相加）
        # ego_context: (B, 1, D) → vector_in → (B, 1, D) → expand → (B, T, D)
        x   = x + self.vector_in(ego_context).expand(-1, T, -1)     # (B, T, D)

        # ④ 投影场景上下文（DiTBlock 的 Cross-Attn 将用它作为 key/value）
        ctx = self.context_in(context)                               # (B, N_ctx, D)

        # ⑤ 融合时间步嵌入和驾驶意图（y 是 DiTBlock 的 AdaLN 条件信号）
        # routing_in(ego_routing): (B, 1, D) → .squeeze(1) → (B, D)
        y   = t_emb + self.routing_in(ego_routing).squeeze(1)       # (B, D)

        # ⑥ 串行通过所有 DiT Block
        for block in self.dit_blocks:
            if self.training:
                # 训练时：梯度检查点（不保存 block 内部激活值，反向时重算）
                # 作用：将每个 block 的显存占用从 O(B*T*D*layers) 降至 O(B*T*D)
                x = grad_checkpoint(block, x, ctx, y, use_reentrant=False)
            else:
                # 推理时：正常前向（无需节省显存）
                x = block(x, ctx, y)

        # ⑦ 最终投影回轨迹坐标空间（256D→2D）
        return self.final_layer(x, y)                                # (B, T, output_dim)

    # ------------------------------------------------------------------
    # 【核心方法②】Euler ODE 采样（推理时使用）
    # ------------------------------------------------------------------

    @torch.no_grad()  # 推理时不需要梯度
    def _sample_traj(self, context, ego_context, ego_routing, B, device, n_steps=5):
        """
        【Euler ODE 积分采样：从噪声到轨迹】

        Flow Matching 的推理过程（Rectified Flow ODE）：
          dz/dt = v(z, t)    ← v 是 DiT 预测的速度场
          积分：z(1) = z(0) + ∫v(z(t), t)dt

        用 Euler 方法近似积分（n_steps 步）：
          x_0 ~ N(0, I)                          ← 从高斯噪声出发
          for i in 0..n_steps-1:
            t_i = i / n_steps                   ← 当前时刻（0→1）
            drift_i = DiT(x_i, t_i, context)    ← 预测速度场
            x_{i+1} = x_i + drift_i * (1/n_steps)  ← Euler 更新步
          x_final ≈ x_data（归一化空间的轨迹增量）

        最后反归一化 + 累积求和 → 绝对坐标轨迹

        Args:
            context, ego_context, ego_routing: 来自 BEVToContextBridge 和 EgoContextBridge
            B      : batch size
            device : 计算设备
            n_steps: Euler 积分步数（越多越精确，但越慢；默认 5 步）
        Returns:
            traj : (B, T, output_dim)  绝对坐标轨迹（单位：米，ego坐标系）
        """
        T = self.planning_steps
        # 初始化：纯高斯噪声（Flow 的起点 z(0) ~ N(0,I)）
        x = torch.randn(B, T, self.output_dim, device=device)

        # Euler ODE 积分（n_steps 步）
        for i in range(n_steps):
            t_val = float(i) / n_steps              # t 从 0 递增到 (n-1)/n
            t     = torch.full((B,), t_val * 1000.0, device=device)  # 缩放到 [0,1000]
            t_emb = self.t_embedder(t)              # (B, D)
            # 预测当前位置 x 处的速度场方向（drift）
            drift = self._dit_forward(x, context, t_emb, ego_context, ego_routing)
            # Euler 更新：沿 drift 方向前进一小步
            x     = x + drift * (1.0 / n_steps)

        # 反归一化：从"标准化空间"转回实际坐标（米）
        delta   = self.denormalize_traj(x)          # (B, T, output_dim)，逐步增量

        # 累积求和：将逐步增量转为绝对坐标
        # delta: [d0, d1, d2, d3, d4, d5]
        # traj:  [d0, d0+d1, d0+d1+d2, ...]  ← 相对于自车当前位置的累积距离
        traj    = torch.cumsum(delta, dim=1)        # (B, T, output_dim)
        return traj

    # ------------------------------------------------------------------
    # 【核心方法③】训练前向传播
    # ------------------------------------------------------------------

    def forward_train(self, bev_embed, outs_motion, sdc_planning, sdc_planning_mask,
                      command, gt_future_boxes=None):
        """
        【训练入口】接口与 PlanningHeadSingleMode.forward_train 完全一致。

        【训练的完整数据流】：
          ① 准备输入特征（fp16→fp32, BEV Adapter, track_query 检查）
          ② BEVToContextBridge → context (B, N+64, D)
             EgoContextBridge  → ego_ctx (B,1,D), eg_rout (B,1,D)
          ③ 准备 GT 轨迹（绝对坐标→逐步增量→归一化）
          ④ Flow Matching 前向：
             - 采样时间步 t ~ sigmoid(randn)
             - 插值：z_t = t * gt_norm + (1-t) * noise
             - 预测：pred = DiT(z_t, t, context)
             - 目标：target = gt_norm - noise
             - 损失：MSE(pred, target)
          ⑤（可选）碰撞损失
          ⑥ 返回损失字典 + 预测轨迹

        Args:
            bev_embed        : (H*W, B, D)   BEVFormer 输出的 BEV 特征
            outs_motion      : dict           MotionHead 的输出，包含：
                               sdc_traj_query  (num_layers, B, P, D)
                               sdc_track_query (B, D)
                               bev_pos         (B, D, H, W)  —— 此处不使用
                               track_query     (B, N, D)     —— 所有 agent 特征
            sdc_planning     : (B, 1, T, 3)  GT 轨迹，最后维度 [x, y, heading]
            sdc_planning_mask: (B, 1, T, 1)  有效掩码
            command          : (B,)           驾驶命令
            gt_future_boxes  : list           碰撞损失用（None 则不计算碰撞损失）
        Returns:
            dict: {'losses': {'loss_flow_matching': ...}, 'outs_motion': {...}}
        """
        # ================================================================
        # ① 数据类型保障（fp16 → fp32）
        # ================================================================
        # 问题背景：UniAD 训练时开启了 auto_fp16（混合精度），上游模块输出 fp16。
        # 但 MLP/LayerNorm 的权重是 fp32（mmdet 的默认行为），
        # 如果不转换，fp16 输入 × fp32 权重 会导致精度损失或 NaN。
        bev_embed       = bev_embed.float()
        sdc_traj_query  = outs_motion['sdc_traj_query'].float()
        sdc_track_query = outs_motion['sdc_track_query'].float()
        raw_track       = outs_motion.get('track_query', None)
        track_query     = raw_track.float() if raw_track is not None else None
        sdc_planning     = sdc_planning.float()
        sdc_planning_mask = sdc_planning_mask.float()
        B               = sdc_track_query.shape[0]

        # ================================================================
        # ② BEV Adapter（可选，当 with_adapter=True 时激活）
        # ================================================================
        # 作用：在 BEV 特征上做轻量卷积残差，微调 BEV 特征以适配规划任务
        # 当前配置（base_e2e_diffusion.py）with_adapter=False，此块跳过
        if self.with_adapter:
            bh, bw = self.bev_track_bridge.bev_h, self.bev_track_bridge.bev_w
            bev_2d   = _bev_seq_to_2d(bev_embed, bh, bw)    # seq→2D
            bev_2d   = bev_2d + self.bev_adapter(bev_2d)     # 残差加
            bev_embed = _bev_2d_to_seq(bev_2d)               # 2D→seq

        # track_query 缺失时用零向量占位（保证后续拼接不出错）
        if track_query is None:
            track_query = torch.zeros(B, 1, self.embed_dims, device=bev_embed.device)

        # command 可能是 Python list（DataLoader 传来的 meta 数据），
        # 需转为 LongTensor 才能送入 nn.Embedding
        if not isinstance(command, torch.Tensor):
            command = torch.tensor(command, dtype=torch.long, device=bev_embed.device)

        # ================================================================
        # ③ 特征提取（通过两个桥接层）
        # ================================================================
        # context: (B, N+64, D)，场景上下文 token
        # ego_ctx: (B, 1, D)，自车状态特征
        # eg_rout: (B, 1, D)，自车驾驶意图特征
        context, _       = self.bev_track_bridge(bev_embed, track_query)
        ego_ctx, eg_rout = self.ego_bridge(sdc_traj_query, sdc_track_query, command)

        # ================================================================
        # ④ 准备 GT 轨迹（绝对坐标 → 逐步增量 → 归一化）
        # ================================================================
        # sdc_planning: (B, 1, T, 3)，取第 0 个 "mode"（规划头只有 1 个 mode）
        gt_traj    = sdc_planning[:, 0, :, :self.output_dim]   # (B, T, 2)，取 x,y

        # traj_mask: (B, T)，标记哪些时间步有效（True=有效）
        # sdc_planning_mask[:, 0, :, 0]: 取第0个 mode，去掉最后的 singleton 维
        traj_mask  = sdc_planning_mask[:, 0, :, 0].bool()     # (B, T)

        # 【绝对坐标 → 逐步增量（delta）】
        # gt_traj 是累积坐标（每步相对于自车起点的距离）
        # 需转为逐步差分（每步相对于上一步的位移），便于归一化和 Flow Matching
        gt_delta   = torch.zeros_like(gt_traj)
        gt_delta[:, 0, :]  = gt_traj[:, 0, :]                 # 第0步：直接等于绝对坐标
        gt_delta[:, 1:, :] = gt_traj[:, 1:, :] - gt_traj[:, :-1, :]  # 后续步：差分

        # 归一化到接近标准正态分布，并用 mask 清零无效步
        gt_norm    = self.normalize_traj(gt_delta) * traj_mask.unsqueeze(-1).float()
        # gt_norm: (B, T, output_dim)，这是 Flow Matching 的"目标分布" x_data

        # ================================================================
        # ⑤ Flow Matching 核心训练逻辑
        # ================================================================
        # 【Rectified Flow 训练目标推导】
        # 我们想训练 v_θ(z_t, t) 使得：
        #   z(0) = noise ~ N(0,I)
        #   z(1) = x_data（GT 归一化轨迹）
        #   z(t) = t*x_data + (1-t)*noise  （直线插值路径）
        # 速度场的真值：dz/dt = x_data - noise = target
        # 损失：E[||v_θ(z_t, t) - target||²]

        noise   = torch.randn_like(gt_norm)   # 高斯噪声，与 gt_norm 同形状

        # 时间步采样：t = sigmoid(Z)，Z~N(0,1)
        # sigmoid 变换使 t 集中在 0.4~0.6 附近（中间时刻），
        # 比均匀采样更有利于训练（中间时刻的 z_t 信息量最丰富）
        t       = torch.sigmoid(torch.randn(B, device=gt_norm.device))  # (B,)，t ∈ (0,1)

        # 插值：z_t = t * x_data + (1-t) * noise
        # t.view(B,1,1)：将 t 从 (B,) 扩展为 (B,1,1)，以广播到 (B,T,output_dim)
        z_t     = t.view(B, 1, 1) * gt_norm + (1 - t.view(B, 1, 1)) * noise  # (B, T, 2)

        # 速度场真值：从噪声到数据的方向
        target  = gt_norm - noise              # (B, T, 2)

        # 时间步嵌入（将 t 从 [0,1] 缩放到 [0,1000] 后嵌入）
        t_emb   = self.t_embedder(t * 1000.0)  # (B, D)

        # DiT 预测速度场
        pred    = self._dit_forward(z_t, context, t_emb, ego_ctx, eg_rout)  # (B, T, 2)

        # 数值安全保护：DiT 初期随机权重可能产生极大的输出（数百甚至更大），
        # clamp 到 [-100, 100] 防止 (pred-target)^2 产生 NaN/Inf，
        # 100 远大于归一化空间的正常值范围（约 -5 ~ 5），不影响正常收敛。
        pred    = torch.clamp(pred, -100.0, 100.0)

        # ================================================================
        # 【早期诊断】前 100 个 iter 打印关键指标；第 1 个 iter 若异常直接终止
        # ================================================================
        self._iter_count += 1
        _DIAG_ITERS = 100   # 诊断窗口：只在前 100 个 iter 打印
        _DIAG_FREQ  = 5     # 每 5 个 iter 打印一次（与 log_config.interval 对齐）
        if self._iter_count <= _DIAG_ITERS and self._iter_count % _DIAG_FREQ == 0:
            with torch.no_grad():
                # ----- 关键中间量 -----
                pred_abs_max  = pred.abs().max().item()
                pred_abs_mean = pred.abs().mean().item()
                target_rms    = target.pow(2).mean().sqrt().item()
                gt_norm_rms   = gt_norm.pow(2).mean().sqrt().item()
                gt_delta_max  = torch.cat([
                    gt_traj[:, :1, :], gt_traj[:, 1:, :] - gt_traj[:, :-1, :]
                ], dim=1).abs().max().item()
                t_val_mean    = t.mean().item()
                has_nan_p     = torch.isnan(pred).any().item()
                has_nan_t     = torch.isnan(target).any().item()
                has_nan_gn    = torch.isnan(gt_norm).any().item()
                theory_fm     = target_rms ** 2

                # ----- 打印诊断信息 -----
                sep = "=" * 70
                print(f"\n{sep}")
                print(f"[DiT 早期诊断] iter={self._iter_count}  t_mean={t_val_mean:.3f}")
                print(f"  【GT 轨迹原始】 gt_delta_max = {gt_delta_max:.4f} m  （正常：0.2~2m）")
                print(f"  【GT 归一化后】 gt_norm_rms  = {gt_norm_rms:.4f}  "
                      f"{'✅ 正常' if 0.3 < gt_norm_rms < 5.0 else '❌ 异常！traj_mean/std 设置有误'}")
                print(f"  【target(v*)】  target_rms   = {target_rms:.4f}  （理论值 ≈ 1.41）")
                print(f"  【DiT 输出】    pred_abs_max = {pred_abs_max:.6f}  "
                      f"{'✅ 正常（零初始化生效）' if pred_abs_max < 1.0 else ('⚠️  偏大但可接受' if pred_abs_max < 10.0 else '❌ 异常！零初始化未生效')}")
                print(f"  【DiT 输出】    pred_abs_mean= {pred_abs_mean:.6f}")
                print(f"  【理论 FM loss】≈ {theory_fm:.4f}  （实际见 planning.loss_flow_matching）")
                print(f"  【NaN 检测】    pred={has_nan_p}  target={has_nan_t}  gt_norm={has_nan_gn}  "
                      f"{'✅ 无 NaN' if not (has_nan_p or has_nan_t or has_nan_gn) else '❌ 存在 NaN！'}")
                risk = "🟢 低" if pred_abs_max < 1.0 else ("🟡 中" if pred_abs_max < 10.0 else "🔴 高")
                print(f"  【梯度爆炸风险】{risk}  pred_abs_max={pred_abs_max:.4f}")
                print(sep)

                # ----- 第 1 次诊断：异常则立即终止，避免浪费时间 -----
                # 只在 iter=5（第一次诊断）做强检查，之后只打印不终止
                if self._iter_count == _DIAG_FREQ:
                    errors = []
                    if pred_abs_max > 10.0:
                        errors.append(
                            f"pred_abs_max={pred_abs_max:.4f} > 10.0：\n"
                            f"    → 零初始化未生效！检查 out_proj/ffn 是否已零初始化。\n"
                            f"    → 执行：grep -n 'out_proj' projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py"
                        )
                    if not (0.3 < gt_norm_rms < 5.0):
                        errors.append(
                            f"gt_norm_rms={gt_norm_rms:.4f} 超出 [0.3, 5.0]：\n"
                            f"    → traj_mean/traj_std 设置不合理，GT 轨迹未被正确归一化。\n"
                            f"    → 当前 traj_mean={self.traj_mean.tolist()}, traj_std={self.traj_std.tolist()}"
                        )
                    if has_nan_p or has_nan_t or has_nan_gn:
                        errors.append(
                            f"检测到 NaN：pred={has_nan_p} target={has_nan_t} gt_norm={has_nan_gn}\n"
                            f"    → 输入数据或权重已损坏，检查 load_from 权重文件。"
                        )
                    if theory_fm > 50.0:
                        errors.append(
                            f"理论初始 FM loss={theory_fm:.2f} > 50：\n"
                            f"    → gt_norm 量级过大，归一化参数严重不匹配。"
                        )
                    if errors:
                        err_msg = "\n\n".join(errors)
                        raise RuntimeError(
                            f"\n{'!'*70}\n"
                            f"[DiT 早期诊断] 第 {self._iter_count} 个 iter 检测到异常，训练终止！\n"
                            f"{'!'*70}\n"
                            f"{err_msg}\n"
                            f"{'!'*70}\n"
                            f"修复后重新运行：\n"
                            f"  rm -rf projects/work_dirs/stage2_e2e/base_e2e_diffusion/\n"
                            f"  bash tools/run_train_single.sh projects/configs/stage2_e2e/base_e2e_diffusion.py"
                        )
                    else:
                        print(f"[DiT 早期诊断] ✅ iter={self._iter_count} 所有指标正常，训练继续！\n")

        # ================================================================
        # ⑥ 计算损失
        # ================================================================
        # Flow Matching MSE Loss
        # mask_e: (B, T, 1)，只在有效时间步计算损失
        mask_e  = traj_mask.unsqueeze(-1).float()
        # 分母：有效步数 × output_dim，避免不同序列长度导致损失不均衡
        fm_loss = ((pred - target) ** 2 * mask_e).sum() / (mask_e.sum() * self.output_dim + 1e-6)
        fm_loss = fm_loss * self.flow_matching_loss_weight
        # 最终保险：防止极端情况下 loss 仍为 NaN/Inf（如 mask 全零）
        fm_loss = torch.nan_to_num(fm_loss, nan=0.0, posinf=100.0)
        losses  = {'loss_flow_matching': fm_loss}

        # ================================================================
        # ADE 辅助损失（加速收敛的关键）
        # ================================================================
        # 原理：FM 的单步预测 pred 是对 (gt_norm - noise) 的估计，
        #       即 pred ≈ x_data - noise，则 pred + noise ≈ x_data（GT 归一化轨迹）
        # 直接从 pred 推导预测轨迹，与 GT 轨迹做 L2 对比，形成直接监督。
        # 这条梯度路径比 FM loss 更短，收敛更快（缺点是略微破坏"速度场一致性"）。
        if self.ade_loss_weight > 0:
            # ----------------------------------------------------------------
            # 【归一化空间 ADE 辅助损失】
            # 关键设计：在归一化空间计算，量纲与 FM loss 一致（均约 0~2），
            #           因此权重可以直接与 flow_matching_loss_weight 对齐（如 0.5）。
            #
            # 推导：
            #   z_t = t*x_data + (1-t)*noise
            #   x_data_est = z_t + pred*(1-t)
            #              ≈ [t*x_data + (1-t)*noise] + (x_data-noise)*(1-t)
            #              = t*x_data + (1-t)*noise + (1-t)*x_data - (1-t)*noise
            #              = x_data   ← 理想情况下恰好等于 GT 归一化轨迹 ✓
            #
            # x_data_est 是在归一化空间对 gt_norm（GT 归一化增量序列）的估计，
            # 与 gt_norm 做 L1/L2 对比，量纲完全一致（无量纲，约 0~2）。
            # ----------------------------------------------------------------
            t_view     = t.view(B, 1, 1)
            # 归一化空间中对 x_data（GT 归一化轨迹增量）的估计
            x_data_est = z_t.detach() + pred * (1.0 - t_view)      # (B, T, output_dim)

            # 在归一化空间直接计算逐步 L2（量纲与 FM loss 一致）
            # gt_norm: (B, T, output_dim)，GT 归一化增量序列（即 x_data）
            ade_norm_loss = ((x_data_est - gt_norm) ** 2 * mask_e).sum(-1)  # (B, T)
            ade_norm_loss = ade_norm_loss.sqrt().mean()  # 对 B×T 求均值
            losses['loss_ade_aux'] = ade_norm_loss * self.ade_loss_weight

        # ================================================================
        # ⑦ 碰撞损失（当前配置下跳过，loss_collision 为空列表）
        # ================================================================
        # 如果启用碰撞损失，需要先采样一条预测轨迹（3步快速ODE），
        # 再计算预测轨迹与 GT 未来障碍物框的碰撞代价。
        # 额外 ODE 采样消耗显存，8GB 环境下默认禁用。
        if len(self.loss_collision) > 0 and gt_future_boxes is not None:
            with torch.no_grad():
                # 快速采样（3步 ODE，比推理时的 5 步更快）
                s_traj = self._sample_traj(context, ego_ctx, eg_rout, B, gt_norm.device, n_steps=3)
            # CollisionLoss 需要 (1, T, 3) 格式的 GT 轨迹（含 heading）
            gt_heading = sdc_planning[:, 0, :, 2:3]            # (B, T, 1)，真实 heading
            gt_traj_3d = torch.cat([gt_traj, gt_heading], dim=-1)  # (B, T, 3)
            gt_traj_3d_b0 = gt_traj_3d[:1]    # (1, T, 3)，取 batch 第 0 个样本
            s_traj_b0     = s_traj[:1]          # (1, T, 2)
            traj_mask_b0  = torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1)
            # gt_future_boxes[0]: 解包 batch 维度
            # [1:planning_steps+1]: 跳过第 0 帧（当前帧），取未来 T 帧
            future_boxes = gt_future_boxes[0][1:self.planning_steps + 1]
            for i, col_fn in enumerate(self.loss_collision):
                losses[f'loss_collision_{i}'] = col_fn(
                    s_traj_b0,     # (1, T, 2)：预测轨迹（绝对坐标）
                    gt_traj_3d_b0, # (1, T, 3)：GT 轨迹（含 heading）
                    traj_mask_b0,  # (1, T)：有效掩码
                    future_boxes   # list[LiDARInstance3DBoxes]
                )

        # ================================================================
        # ⑧ 构造返回值（与 PlanningHeadSingleMode 接口兼容）
        # ================================================================
        # 训练时直接用 FM 的单步预测（已经是对 x_data 的估计），
        # 反归一化 + 累积求和 = 预测轨迹（用于训练期间的评估，detach 不参与梯度）
        with torch.no_grad():
            pred_delta = self.denormalize_traj(pred.detach())   # (B, T, 2)，反归一化
            pred_traj  = torch.cumsum(pred_delta, dim=1)        # (B, T, 2)，绝对坐标
        outs_planning = {
            'sdc_traj':     pred_traj,   # (B, T, 2)
            'sdc_traj_all': pred_traj,   # 保留两个 key，与接口兼容
        }
        return dict(losses=losses, outs_motion=outs_planning)

    # ------------------------------------------------------------------
    # 【核心方法④】推理前向传播
    # ------------------------------------------------------------------

    def forward_test(self, bev_embed, outs_motion, outs_occflow=None, command=None):
        """
        【推理入口】接口与 PlanningHeadSingleMode.forward_test 完全一致。

        【推理与训练的区别】：
          训练：用 FM 单步预测（一次 DiT 前向）+ MSE loss
          推理：用 Euler ODE 积分（n_steps=5 次 DiT 前向），质量更高

        outs_occflow：OccFlow 头的输出（占用预测），
                      在原版 PlanningHeadSingleMode 推理时用于碰撞优化，
                      DiffusionPlanningHead 不使用（接口兼容性保留）

        Returns:
            dict: {'sdc_traj': (1, T, 2), 'sdc_traj_all': (1, T, 2)}
        """
        # ① 数据类型保障（与 forward_train 相同）
        bev_embed       = bev_embed.float()
        sdc_traj_query  = outs_motion['sdc_traj_query'].float()
        sdc_track_query = outs_motion['sdc_track_query'].float()
        raw_track       = outs_motion.get('track_query', None)
        track_query     = raw_track.float() if raw_track is not None else None
        B               = sdc_track_query.shape[0]
        device          = sdc_track_query.device

        # ② BEV Adapter（可选，与 forward_train 一致）
        if self.with_adapter:
            bh, bw = self.bev_track_bridge.bev_h, self.bev_track_bridge.bev_w
            bev_2d   = _bev_seq_to_2d(bev_embed, bh, bw)
            bev_2d   = bev_2d + self.bev_adapter(bev_2d)
            bev_embed = _bev_2d_to_seq(bev_2d)

        if track_query is None:
            track_query = torch.zeros(B, 1, self.embed_dims, device=device)

        if not isinstance(command, torch.Tensor):
            command = torch.tensor(command, dtype=torch.long, device=device)

        # ③ 特征提取（与训练完全相同）
        context, _ = self.bev_track_bridge(bev_embed, track_query)
        ego_ctx, eg_rout = self.ego_bridge(sdc_traj_query, sdc_track_query, command)

        # ④ Euler ODE 采样（推理专属，训练时不调用）
        # _sample_traj 内部用 @torch.no_grad() 装饰，推理时自动关闭梯度
        pred_traj = self._sample_traj(context, ego_ctx, eg_rout, B, device,
                                       n_steps=self.sample_steps)
        # pred_traj: (B, T, 2)，已完成反归一化 + 累积求和，单位：米

        return {
            'sdc_traj':     pred_traj,   # (B, T, 2)，最终规划轨迹
            'sdc_traj_all': pred_traj,   # 保留两个 key，与接口兼容
        }