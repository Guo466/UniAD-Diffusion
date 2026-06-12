# ================================================================================
# 【文件说明】bevformer_head.py —— BEVFormer 检测头
#
# 这个文件是 BEVFormer 的"大脑"，负责把图像特征转换成最终的 3D 检测结果。
# 它处于整个流水线的中间层：
#
#   图像特征 (backbone+neck输出)
#        ↓
#   BEVFormerHead（本文件）
#    ├─ BEV Embedding（可学习的 BEV 查询网格）
#    ├─ Object Query Embedding（可学习的检测查询）
#    ├─ PerceptionTransformer（Encoder生成BEV + Decoder解码检测框）
#    ├─ 分类头 cls_branches（预测目标类别）
#    └─ 回归头 reg_branches（预测目标位置/尺寸/朝向/速度）
#        ↓
#   3D 检测结果（类别分数 + 边界框坐标）
#
# 文件包含两个类：
#   - BEVFormerHead：标准检测头
#   - BEVFormerHead_GroupDETR：分组 DETR 变体（训练时用多组 Query 增强效果）
#
# 【注意】这个文件是独立版 BEVFormer (bevformer.py) 配套的检测头，
# UniAD 的 track_head.py (BEVFormerTrackHead) 是在此基础上扩展的跟踪版本。
# ================================================================================

import copy
import torch
import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
# Linear: mmcv 封装的全连接层（等价于 nn.Linear，但有更好的权重初始化）
# bias_init_with_prob: 根据目标概率计算 bias 初始值，用于分类头初始化

from mmcv.utils import TORCH_VERSION, digit_version
# 用于版本比较，兼容不同 PyTorch 版本的 API

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
# multi_apply: 对列表中的每个元素并行调用同一函数，相当于 map()，但支持多返回值
# reduce_mean: 在多 GPU 分布式训练时，对一个标量值求所有 GPU 的平均（同步操作）

from mmdet.models.utils.transformer import inverse_sigmoid
# inverse_sigmoid: sigmoid 的逆函数，即 log(x / (1-x))
# 作用：把归一化的 [0,1] 坐标转回"未经 sigmoid 激活"的 logit 空间，方便做残差加法

from mmdet.models import HEADS
# HEADS: mmdet 的检测头注册表

from mmdet.models.dense_heads import DETRHead
# DETRHead: mmdet 的 DETR 检测头基类，提供匈牙利匹配、损失计算等通用功能
# BEVFormerHead 继承它，复用这些功能，只需重写 forward 和部分初始化逻辑

from mmdet3d.core.bbox.coders import build_bbox_coder
# build_bbox_coder: 根据配置构建边界框编解码器
# 编码：把原始 GT 格式转为模型训练用的归一化格式
# 解码：把模型预测的归一化值转回真实的 3D 坐标（用于推理）

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
# normalize_bbox: 将真实的 3D 边界框坐标归一化到 [0,1] 范围（用于损失计算）

from mmcv.runner import force_fp32, auto_fp16
# force_fp32: 强制用 float32 精度（防止损失计算时 fp16 精度不足）
# auto_fp16: 自动将指定输入转为 fp16 加速前向传播


@HEADS.register_module()
class BEVFormerHead(DETRHead):
    """BEVFormer 检测头（独立版，用于纯 3D 检测任务）。

    【在整体架构中的角色】
    本类是 BEVFormer 检测器（bevformer.py）的 pts_bbox_head，
    它接收多尺度图像特征，通过 PerceptionTransformer 生成 BEV 特征图，
    再用 Object Query 解码出 3D 检测框。

    【继承关系】
    BEVFormerHead → DETRHead → BaseHead
    继承自 DETRHead 的核心能力：
      - 匈牙利匹配（assigner）：将预测框与 GT 框一一对应
      - 采样器（sampler）：标记正样本（匹配成功）和负样本（背景）
      - 通用损失计算框架

    Args:
        with_box_refine (bool): 是否在 Decoder 每一层都迭代精细化参考点坐标。
                                True = 每层都有独立的回归头（更准确但参数更多）
                                False = 所有层共享同一个回归头
        as_two_stage (bool): 是否使用两阶段检测（Encoder 先生成候选框，Decoder 再精细化）。
                             UniAD 中一般为 False（单阶段）。
        transformer: PerceptionTransformer 的配置字典（包含 Encoder + Decoder）
        bbox_coder: 边界框编解码器配置（定义坐标格式和归一化方式）
        num_cls_fcs (int): 分类头的全连接层数量
        code_weights (list): 各维度回归损失的权重，共 10 维
                             [x, y, z, w, l, h, sin(θ), cos(θ), vx, vy]
                             默认速度维度权重 0.2（速度比位置精度要求低）
        bev_h, bev_w (int): BEV 特征图的高和宽（格子数量）
                            如 200x200 表示把场景划分为 200x200 个 BEV 格子
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,      # 是否逐层精细化框坐标
                 as_two_stage=False,          # 是否两阶段检测
                 transformer=None,            # PerceptionTransformer 配置
                 bbox_coder=None,             # 3D 边界框编解码器配置
                 num_cls_fcs=2,               # 分类头全连接层数
                 code_weights=None,           # 各维度损失权重（10维）
                 bev_h=30,                    # BEV 特征图高度（格子数）
                 bev_w=30,                    # BEV 特征图宽度（格子数）
                 **kwargs):

        # ── 保存 BEV 网格尺寸（必须在 super().__init__ 之前设置，因为父类初始化会用到）
        self.bev_h = bev_h   # 例如 200（BEV 特征图有 200 行格子）
        self.bev_w = bev_w   # 例如 200（BEV 特征图有 200 列格子）
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            # 两阶段模式：把标志传入 transformer 配置，让 Encoder 也输出候选框
            transformer['as_two_stage'] = self.as_two_stage

        # ── 边界框编码维度（code_size）────────────────────────────────
        # 3D 检测框通常编码为 10 维：
        # [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
        #   cx/cy/cz: 中心点坐标   w/l/h: 宽/长/高   yaw: 朝向角   vx/vy: 速度
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10  # 默认 10 维

        # ── 各维度的损失权重 ─────────────────────────────────────────
        # 不同维度的预测难度不同，需要给予不同的权重：
        # 位置(x,y,z)、尺寸(w,l,h)、朝向(sin,cos) 权重=1.0（精确预测）
        # 速度(vx,vy) 权重=0.2（速度预测难度大，权重降低）
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,      # x, y, z（位置）
                                 1.0, 1.0, 1.0,      # w, l, h（尺寸）
                                 1.0, 1.0,            # sin(θ), cos(θ)（朝向）
                                 0.2, 0.2]            # vx, vy（速度，权重降低）

        # ── 构建边界框编解码器 ────────────────────────────────────────
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        # pc_range: 点云感知范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        # nuScenes 中通常为 [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]（单位：米）
        self.real_w = self.pc_range[3] - self.pc_range[0]  # 场景真实宽度（米），如 102.4m
        self.real_h = self.pc_range[4] - self.pc_range[1]  # 场景真实高度（米），如 102.4m

        self.num_cls_fcs = num_cls_fcs - 1  # 实际使用的中间层数（不含最后输出层）

        # ── 调用父类 DETRHead 的初始化（构建匹配器、损失函数等）────────
        super(BEVFormerHead, self).__init__(
            *args, transformer=transformer, **kwargs)

        # 将 code_weights 转为不可训练的 nn.Parameter（这样可以被保存到模型权重，但不参与梯度更新）
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self):
        """初始化检测头的网络层（分类头 + 回归头 + Embedding）。

        BEVFormer 的检测头有两种网络：
        1. cls_branches（分类头）：Query 向量 → 各类别的概率分数
        2. reg_branches（回归头）：Query 向量 → 3D 框的坐标/尺寸/速度

        由于 Transformer Decoder 有多层（如 6 层），每层都会输出一次预测，
        需要为每层都准备一套分类头和回归头（辅助损失）。
        """
        # ── 构建单个分类头：Linear → LayerNorm → ReLU（重复若干次）→ Linear(输出类别数)
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            # LayerNorm：对每个样本在特征维度上做归一化，稳定训练
            cls_branch.append(nn.ReLU(inplace=True))
            # ReLU：非线性激活，inplace=True 表示直接修改输入 Tensor，节省内存
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        # 最后一层输出类别数（如 10 类 + 1 背景类）
        fc_cls = nn.Sequential(*cls_branch)  # 把列表打包成顺序模型

        # ── 构建单个回归头：Linear → ReLU（重复若干次）→ Linear(输出 code_size 维)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        # 最后输出 10 维（cx, cy, cz, w, l, h, sin_θ, cos_θ, vx, vy）
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            """深拷贝一个模块 N 次，得到 N 个参数独立的副本。"""
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # ── 决定需要几套头（对应 Decoder 的层数）────────────────────────
        # Decoder 通常有 6 层，每层都会输出预测结果用于辅助损失（深监督）
        # 两阶段模式还需要额外一套头给 Encoder 输出用
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            # 逐层精细化模式：每层有独立参数的头（参数更多，但效果更好）
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            # 共享参数模式：所有层使用同一套参数（节省参数量）
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        # ── 创建两种可学习的 Embedding（单阶段模式专用）────────────────
        if not self.as_two_stage:
            # 1. BEV Embedding：BEV 特征图上每个格子的初始内容特征
            #    形状：[bev_h * bev_w, embed_dims]，如 [40000, 256]
            #    可以理解为：在 200x200 的地面网格上，每个格子有一个初始"问题向量"
            #    Encoder 会用这些向量作为 Query，向 6 个相机图像"提问"，填充 BEV 内容
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

            # 2. Object Query Embedding：检测框的查询向量
            #    形状：[num_query, embed_dims * 2]，如 [900, 512]
            #    维度 *2 是因为：前 256 维是内容向量（content query），
            #                    后 256 维是位置向量（position query / reference point）
            #    每个 Query 代表"我想找一个目标"，Decoder 会让它关注 BEV 特征来确定目标位置
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """初始化网络权重。

        分类头的最后一层 bias 用特殊方法初始化：
        用 bias_init_with_prob(0.01) 设置初始 bias，使得初始预测概率接近 0.01，
        避免训练初期所有 Query 都预测为正样本，导致训练不稳定。
        （这是 Focal Loss 论文中提出的技巧）
        """
        self.transformer.init_weights()  # 初始化 Transformer 的权重
        if self.loss_cls.use_sigmoid:
            # sigmoid 分类器需要特殊 bias 初始化
            bias_init = bias_init_with_prob(0.01)  # 对应 sigmoid(bias) ≈ 0.01
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """前向传播：图像特征 → BEV 特征 → 3D 检测结果。

        【整体流程】
        1. 准备 BEV Query（bev_queries）和 Object Query（object_query_embeds）
        2. 生成位置编码（bev_pos）
        3. 调用 PerceptionTransformer：
           a. Encoder：BEV Query 通过 SCA+TSA → BEV 特征图（bev_embed）
           b. Decoder：Object Query 通过交叉注意力关注 BEV → 每层输出预测（hs）
        4. 逐层解码：hs → 分类分数 + 3D 框坐标（带参考点残差）
        5. 把归一化坐标转换回真实世界坐标（米）

        Args:
            mlvl_feats (list[Tensor]): 多尺度图像特征，每个 shape = [B, N_cam, C, H, W]
                                       mlvl = multi-level（多尺度/多层级）
            img_metas (list[dict]): 图像元信息（相机内外参等）
            prev_bev (Tensor, optional): 上一帧 BEV 特征，用于时序自注意力
            only_bev (bool): True 时只生成 BEV 特征不做检测解码（用于历史帧计算）

        Returns:
            dict: 包含以下键：
                - 'bev_embed': BEV 特征图，shape [B, bev_h*bev_w, C]
                - 'all_cls_scores': 所有 Decoder 层的分类分数，shape [num_dec, B, num_query, num_cls]
                - 'all_bbox_preds': 所有 Decoder 层的框预测，shape [num_dec, B, num_query, 10]
                - 'enc_cls_scores': Encoder 分类分数（两阶段时有值，否则 None）
                - 'enc_bbox_preds': Encoder 框预测（两阶段时有值，否则 None）
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape  # 获取 batch size 和摄像头数量
        dtype = mlvl_feats[0].dtype  # 获取数据类型（fp16 或 fp32）

        # ── Step 1: 准备 Object Query 和 BEV Query ───────────────────────
        # Object Query：900 个可学习向量，每个代表"我想找一个 3D 目标"
        # shape = [num_query, embed_dims*2] = [900, 512]
        # 前 256 维：内容 query；后 256 维：位置 query（用来生成初始参考点）
        object_query_embeds = self.query_embedding.weight.to(dtype)

        # BEV Query：200*200=40000 个可学习向量，对应 BEV 地面网格的每个格子
        # shape = [bev_h*bev_w, embed_dims] = [40000, 256]
        bev_queries = self.bev_embedding.weight.to(dtype)

        # ── Step 2: 生成 BEV 位置编码 ────────────────────────────────────
        # bev_mask 是全零的掩码（这里不做任何 mask，只是为了调用 positional_encoding 的接口）
        # positional_encoding 根据位置生成正弦/余弦位置编码，让模型知道每个格子在哪里
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        # bev_pos shape = [B, embed_dims, bev_h, bev_w]，即每个格子的位置编码

        # ── Step 3: 调用 Transformer（根据 only_bev 决定走哪条路）────────
        if only_bev:
            # 只需要 BEV 特征（如历史帧处理），不需要检测解码
            # 只走 Encoder，跳过 Decoder，节省计算量
            return self.transformer.get_bev_features(
                mlvl_feats,       # 多尺度图像特征（K=Value）
                bev_queries,      # BEV Query（Q）
                self.bev_h,
                self.bev_w,
                self.real_h,      # 场景真实高度（米），用于计算格子间距
                self.real_w,      # 场景真实宽度（米）
                grid_length=(self.real_h / self.bev_h,   # 每个格子代表多少米
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,  # BEV 位置编码
                img_metas=img_metas,
                prev_bev=prev_bev,  # 上一帧 BEV（用于时序自注意力）
            )
        else:
            # 完整前向：Encoder 生成 BEV + Decoder 解码检测框
            outputs = self.transformer(
                mlvl_feats,             # 图像特征（Encoder 的 Key/Value 来源）
                bev_queries,            # BEV Query（Encoder 的 Query）
                object_query_embeds,    # Object Query（Decoder 的 Query）
                self.bev_h,
                self.bev_w,
                self.real_h,
                self.real_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                # 逐层精细化时传入各层的回归头（用于在 Decoder 内部实时更新参考点）
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

        # ── Step 4: 解包 Transformer 输出 ────────────────────────────────
        bev_embed, hs, init_reference, inter_references = outputs
        # bev_embed: Encoder 输出的 BEV 特征图，shape = [B, bev_h*bev_w, C]
        # hs: Decoder 每层的输出（Object Query 经过注意力后的向量），
        #     shape = [num_dec_layers, num_query, B, embed_dims]
        # init_reference: Decoder 输入时 Object Query 的初始参考点（3D位置先验），
        #                 shape = [B, num_query, 3]，归一化坐标 [0,1]
        # inter_references: 每个 Decoder 层输出的更新后参考点，用于下一层的注意力

        hs = hs.permute(0, 2, 1, 3)
        # 调整维度顺序：[num_dec_layers, num_query, B, C] → [num_dec_layers, B, num_query, C]
        # 方便后续按 batch 维度索引

        # ── Step 5: 逐 Decoder 层解码检测结果 ────────────────────────────
        # BEVFormer 采用"深监督"（Deep Supervision）：
        # Decoder 每一层都输出一次预测，每层都计算损失（辅助损失），
        # 最终只用最后一层的预测，但中间层的损失帮助训练更快收敛
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):  # 遍历每个 Decoder 层
            # 取当前层的参考点（用于残差预测）
            if lvl == 0:
                reference = init_reference   # 第 0 层用初始参考点
            else:
                reference = inter_references[lvl - 1]  # 后续层用上一层更新的参考点
            # 注：参考点是归一化坐标 [0,1]，需要先转回 logit 空间才能做残差加法
            reference = inverse_sigmoid(reference)
            # inverse_sigmoid(x) = log(x / (1-x))，即 sigmoid 的逆运算

            # 分类预测：Query 向量 → 各类别的 logit 分数
            outputs_class = self.cls_branches[lvl](hs[lvl])
            # shape = [B, num_query, num_classes]

            # 回归预测：Query 向量 → 框参数的残差（相对于参考点的偏移量）
            tmp = self.reg_branches[lvl](hs[lvl])
            # shape = [B, num_query, code_size=10]
            # tmp 的含义：[Δcx, Δcy, Δw, Δl, Δcz, Δh, Δsinθ, Δcosθ, Δvx, Δvy]
            # 即相对于参考点的"偏移量"，而非绝对坐标

            assert reference.shape[-1] == 3  # 参考点是 3D：(x, y, z)

            # ── 残差解码：预测值 = 参考点 + 偏移量（在 logit 空间相加，再 sigmoid）──
            # X/Y 方向（地面平面坐标）：
            tmp[..., 0:2] += reference[..., 0:2]   # logit 空间相加（残差）
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid() # sigmoid 映射到 [0,1] 的归一化坐标

            # Z 方向（高度）：
            tmp[..., 4:5] += reference[..., 2:3]    # 注意：Z 在 reference 的第2维，在 tmp 的第4维
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            # ── 反归一化：从 [0,1] 映射回真实世界坐标（单位：米）────────
            # 公式：真实坐标 = 归一化值 × 场景范围 + 场景最小值
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            # X：[-51.2m, 51.2m]  →  归一化值 0.5 对应 0m（中心）
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            # Y：[-51.2m, 51.2m]
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])
            # Z：[-5.0m, 3.0m]

            outputs_coord = tmp  # 保存当前层解码后的坐标
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # 把所有层的结果堆叠成张量
        # outputs_classes shape: [num_dec_layers, B, num_query, num_classes]
        # outputs_coords  shape: [num_dec_layers, B, num_query, code_size]
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        # ── Step 6: 打包输出字典 ────────────────────────────────────────
        outs = {
            'bev_embed': bev_embed,              # BEV 特征图（供时序融合和下游任务用）
            'all_cls_scores': outputs_classes,   # 所有层的分类分数（训练时每层都参与损失）
            'all_bbox_preds': outputs_coords,    # 所有层的框坐标预测
            'enc_cls_scores': None,              # Encoder 分类分数（单阶段为 None）
            'enc_bbox_preds': None,              # Encoder 框预测（单阶段为 None）
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """为单张图像计算分类和回归的监督目标（匈牙利匹配的核心步骤）。

        【问题背景】
        DETR 系列模型输出固定数量的 Query（如 900 个），但 GT 框的数量是不固定的。
        需要把"预测的 900 个框"与"GT 的 K 个框"做最优一一匹配，才能计算损失。
        这就是"匈牙利匹配"（二分图最优匹配算法）。

        【匹配结果的含义】
        - pos_inds（正样本）：被成功匹配到 GT 的 Query 索引（共 K 个）
        - neg_inds（负样本）：没有匹配到任何 GT 的 Query 索引（共 900-K 个，即背景）

        Args:
            cls_score (Tensor): 单张图像的分类分数，shape [num_query, num_classes]
            bbox_pred (Tensor): 单张图像的框预测，shape [num_query, code_size]
            gt_labels (Tensor): GT 类别标签，shape [num_gt]
            gt_bboxes (Tensor): GT 3D 框，shape [num_gt, code_size]
            gt_bboxes_ignore: 忽略框（None）

        Returns:
            tuple: (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)
                - labels: 每个 Query 的目标类别，正样本=GT类别，负样本=num_classes（背景）
                - label_weights: 每个 Query 的分类损失权重（全为 1.0）
                - bbox_targets: 每个 Query 的目标框坐标（只有正样本有效）
                - bbox_weights: 每个 Query 的回归损失权重（正样本=1.0，负样本=0.0）
                - pos_inds: 正样本 Query 的索引
                - neg_inds: 负样本 Query 的索引
        """
        num_bboxes = bbox_pred.size(0)  # Query 总数（如 900）
        gt_c = gt_bboxes.shape[-1]      # GT 框的编码维度

        # ── 匈牙利匹配：找到预测框与 GT 框的最优一一对应关系 ─────────────
        # assigner 计算每个预测框与每个 GT 框的匹配代价（分类代价 + 回归代价），
        # 再用匈牙利算法找全局最优匹配
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        # sampler 根据匹配结果分配正负样本标签
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds  # 正样本 Query 的索引（已匹配到 GT）
        neg_inds = sampling_result.neg_inds  # 负样本 Query 的索引（背景）

        # ── 生成分类目标 ─────────────────────────────────────────────────
        # 初始化所有 Query 的标签为 num_classes（背景类）
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        # 正样本位置填入对应 GT 的类别
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)  # 所有 Query 的分类权重都是 1

        # ── 生成回归目标 ─────────────────────────────────────────────────
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]  # 初始化为全零
        bbox_weights = torch.zeros_like(bbox_pred)              # 权重初始化为全零
        bbox_weights[pos_inds] = 1.0  # 只有正样本参与回归损失计算

        # 正样本的回归目标 = 与之匹配的 GT 框坐标
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """为整个 batch 计算监督目标（对每张图像调用 _get_target_single）。

        【multi_apply 的作用】
        multi_apply(func, list1, list2, ...) 相当于：
            [func(list1[i], list2[i], ...) for i in range(len(list1))]
        并把每个返回值的各元素分别收集成列表。

        Args:
            cls_scores_list: batch 中每张图像的分类分数列表
            bbox_preds_list: batch 中每张图像的框预测列表
            gt_bboxes_list:  batch 中每张图像的 GT 框列表
            gt_labels_list:  batch 中每张图像的 GT 标签列表

        Returns:
            tuple: 包含 labels_list, label_weights_list, bbox_targets_list,
                   bbox_weights_list, num_total_pos, num_total_neg
                   - num_total_pos: batch 中正样本总数（用于损失归一化）
                   - num_total_neg: batch 中负样本总数
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)  # batch size
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # 对 batch 中每张图像调用 _get_target_single，并行处理
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)

        # numel()：返回 Tensor 中元素的总数量
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))  # 正样本总数
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))  # 负样本总数

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """计算单个 Decoder 层的分类损失 + 回归损失。

        【深监督机制】
        Decoder 有多层（如 6 层），每层都调用这个函数计算损失。
        虽然最终只使用最后一层的预测结果，但每层的损失都参与反向传播，
        让梯度能有效传递到 Decoder 的底层，加速训练收敛。

        Args:
            cls_scores (Tensor): 单层所有图像的分类分数，shape [B, num_query, num_classes]
            bbox_preds (Tensor): 单层所有图像的框预测，shape [B, num_query, code_size]
            gt_bboxes_list: GT 框列表（每张图像一个 Tensor）
            gt_labels_list: GT 标签列表
            gt_bboxes_ignore_list: 忽略框（None）

        Returns:
            tuple: (loss_cls, loss_bbox)
                - loss_cls: 分类损失（Focal Loss 或交叉熵）
                - loss_bbox: 回归损失（L1 Loss）
        """
        num_imgs = cls_scores.size(0)  # batch size

        # 把 batch 维拆开，得到每张图像的预测列表（方便 get_targets 逐图处理）
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        # 计算匈牙利匹配目标（正样本分配）
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # 把 batch 中所有图像的目标拼接成一个大 Tensor（方便统一计算损失）
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ── 分类损失（Focal Loss）────────────────────────────────────────
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # reshape：[B, num_query, num_cls] → [B*num_query, num_cls]，方便计算

        # cls_avg_factor：用于归一化损失的分母
        # = 正样本数 + 负样本数 × bg_cls_weight（背景权重，一般 < 1，降低背景的影响）
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            # 多 GPU 训练时：对所有 GPU 上的 avg_factor 求平均，保证损失的一致性
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)  # 防止除以 0
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # ── 回归损失（L1 Loss）────────────────────────────────────────────
        # 多 GPU 同步正样本数量（用于回归损失归一化）
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        # clamp(min=1)：防止正样本数为 0 时除以 0

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # reshape：[B, num_query, code_size] → [B*num_query, code_size]

        # 将 GT 框坐标归一化（转为 [0,1] 范围），与模型预测的归一化输出对齐
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)

        # 过滤掉 NaN/Inf（异常值，防止损失爆炸）
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        # 将各维度权重乘到 bbox_weights 上（速度维度权重为 0.2）
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # 处理极端情况下可能产生的 NaN 值（PyTorch >= 1.8 才有 nan_to_num）
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)   # NaN → 0
            loss_bbox = torch.nan_to_num(loss_bbox) # NaN → 0

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """计算总损失（汇总所有 Decoder 层的损失）。

        【深监督的完整流程】
        Decoder 有 N 层（如 6 层），每层都有预测结果：
          - 第 0 层预测 → loss_single → d0.loss_cls, d0.loss_bbox
          - 第 1 层预测 → loss_single → d1.loss_cls, d1.loss_bbox
          - ...
          - 第 5 层预测 → loss_single → loss_cls, loss_bbox（最终层，名字不带前缀）
        
        所有层的损失求和后，通过反向传播同时更新所有层的参数。

        Args:
            gt_bboxes_list (list): GT 3D 边界框（LiDARInstance3DBoxes 对象）
            gt_labels_list (list[Tensor]): GT 类别标签
            preds_dicts (dict): forward() 的输出，包含 all_cls_scores 等
            gt_bboxes_ignore: 忽略框（None）
            img_metas: 图像元信息

        Returns:
            dict: 损失字典，如：
                {'loss_cls': 0.3, 'loss_bbox': 0.8,
                 'd0.loss_cls': 0.5, 'd0.loss_bbox': 1.2, ...}
        """
        all_cls_scores = preds_dicts['all_cls_scores']  # [num_dec, B, num_query, num_cls]
        all_bbox_preds = preds_dicts['all_bbox_preds']  # [num_dec, B, num_query, code_size]
        enc_cls_scores = preds_dicts['enc_cls_scores']  # Encoder 分类（单阶段为 None）
        enc_bbox_preds = preds_dicts['enc_bbox_preds']  # Encoder 回归（单阶段为 None）

        num_dec_layers = len(all_cls_scores)  # Decoder 层数
        device = gt_labels_list[0].device

        # ── 将 GT 框从 LiDAR 格式转为统一的 Tensor 格式 ─────────────────
        # LiDARInstance3DBoxes 是 mmdet3d 的 3D 框数据结构
        # gravity_center：重力中心点（x, y, z），即框底面中心的高度 + 半高
        # tensor[:, 3:]：框的其他属性（w, l, h, yaw, vx, vy）
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # 最终格式：[num_gt, 9] = [cx, cy, cz, w, l, h, yaw, vx, vy]

        # 将相同的 GT 复制 num_dec_layers 份（每层都用同一批 GT 做匹配）
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # ── 对所有 Decoder 层并行计算损失 ────────────────────────────────
        # multi_apply 会把 all_cls_scores 的第一维（num_dec_layers）展开，
        # 对每一层分别调用 loss_single
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # ── (可选) Encoder 的损失（两阶段模式） ──────────────────────────
        if enc_cls_scores is not None:
            # 两阶段：Encoder 输出的候选框也需要监督
            # 此时使用二值标签（只区分前景/背景，不区分具体类别）
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # ── 最后一层 Decoder 的损失（最终预测，权重最重）─────────────────
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # ── 前几层 Decoder 的辅助损失（深监督）─────────────────────────
        # 名字格式：d0.loss_cls, d1.loss_cls, ...（d = decoder layer）
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """推理时：将模型输出解码为最终的 3D 检测框。

        【流程】
        模型输出（归一化坐标 + logit 分数）
            → bbox_coder.decode（阈值过滤 + 坐标反归一化）
            → 转为 LiDAR3DBox 对象
            → 返回 [bboxes, scores, labels]

        Args:
            preds_dicts (dict): forward() 的输出字典
            img_metas (list[dict]): 图像元信息（含 box_type_3d）
            rescale (bool): 是否缩放回原图尺寸（3D 检测一般不需要）

        Returns:
            list: 每个 batch 样本的检测结果 [bboxes, scores, labels]
        """
        # bbox_coder.decode：
        # 1. 取最后一层 Decoder 的预测（all_cls_scores[-1], all_bbox_preds[-1]）
        # 2. 对分类分数做 sigmoid，得到 [0,1] 的概率
        # 3. 按置信度阈值过滤低置信度框
        # 4. 将归一化坐标还原为真实世界坐标
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)  # batch size
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']  # shape [num_det, code_size]，实际坐标（米）

            # 高度坐标修正：
            # nuScenes 中 cz 是框底面中心（不是重心）
            # 这里把 z 坐标从"框中心" → "框底面"：z_bottom = z_center - h/2
            # bboxes[:, 2] 是 cz，bboxes[:, 5] 是 h
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            # 将 Tensor 包装为 LiDAR3DBox 对象（mmdet3d 的数据结构，支持旋转、IoU 计算等）
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']   # 置信度分数
            labels = preds['labels']   # 类别标签

            ret_list.append([bboxes, scores, labels])

        return ret_list


# ================================================================================
# BEVFormerHead_GroupDETR：分组 DETR 变体
#
# 【核心思想】
# 普通 DETR 训练时，N 个 Query 与 K 个 GT 做匈牙利匹配（N >> K），
# 大量 Query 被分配为负样本，正负样本极度不均衡。
#
# Group DETR 的解法：
# 训练时，把 N 个 Query 分成 G 组，每组独立与 GT 做匈牙利匹配。
# 这样每组都能有 K 个正样本，总正样本数变为 G×K，
# 大幅增加正样本比例，让梯度信号更充足，训练更稳定。
#
# 推理时，只使用 1 组 Query（N/G 个），回到标准 DETR 的推理方式。
# ================================================================================
@HEADS.register_module()
class BEVFormerHead_GroupDETR(BEVFormerHead):
    """分组 DETR 变体的 BEVFormer 检测头。

    训练时：使用 group_detr × num_query 个 Query（分组匹配，增加正样本）
    推理时：只使用 num_query 个 Query（第一组）

    Args:
        group_detr (int): 分组数量 G。如 G=5，则训练用 5×900=4500 个 Query，
                          推理只用 900 个。
    """
    def __init__(self,
                 *args,
                 group_detr=1,  # 分组数量（默认 1 即不分组，退化为普通 BEVFormerHead）
                 **kwargs):
        self.group_detr = group_detr
        assert 'num_query' in kwargs
        # 训练时扩大 Query 数量为 G 倍（在父类初始化时生效）
        kwargs['num_query'] = group_detr * kwargs['num_query']
        super().__init__(*args, **kwargs)

    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """前向传播（与父类基本相同，推理时只取第一组 Query）。

        【与父类 BEVFormerHead.forward 的唯一区别】
        推理（not self.training）时，object_query_embeds 只取前 num_query/group_detr 个，
        即只使用第一组 Query，与训练时保持推理行为一致。
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)

        if not self.training:
            # ★ 关键区别：推理时只用第一组 Query（1/group_detr 的数量）
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats, bev_queries,
                self.bev_h, self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos, img_metas=img_metas, prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats, bev_queries, object_query_embeds,
                self.bev_h, self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas, prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        # 解码逻辑与父类完全相同（参考父类 forward 的注释）
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """分组 DETR 的损失计算（核心区别：对每个 Query 组独立做匈牙利匹配）。

        【与父类 loss 的区别】
        父类：所有 N 个 Query 一起与 GT 做匹配（N >> K，大量负样本）
        本类：将 N 个 Query 分成 G 组，每组独立与 GT 匹配（每组 K 个正样本）
             总正样本数 = G × K，梯度信号更充足

        【计算流程】
        1. 对每个 Query 组分别切片得到 group_cls_scores / group_bbox_preds
        2. 对每组调用 loss_single（内部做独立匈牙利匹配）
        3. 把 G 组的损失除以 G 后累加（等效于取平均）

        Returns:
            dict: 损失字典（格式与父类相同）
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']  # [num_dec, B, G*num_query, num_cls]
        all_bbox_preds = preds_dicts['all_bbox_preds']  # [num_dec, B, G*num_query, code_size]
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        assert enc_cls_scores is None and enc_bbox_preds is None  # 分组DETR不支持两阶段

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # GT 框转换为统一 Tensor 格式（同父类）
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # ── 初始化损失字典（先置为 0，后面逐组累加）────────────────────
        loss_dict = dict()
        loss_dict['loss_cls'] = 0
        loss_dict['loss_bbox'] = 0
        for num_dec_layer in range(all_cls_scores.shape[0] - 1):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = 0
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = 0

        # ── 每个 Query 组独立做匈牙利匹配并计算损失 ──────────────────────
        num_query_per_group = self.num_query // self.group_detr  # 每组的 Query 数（如 900）
        for group_index in range(self.group_detr):
            # 按 Query 维度切片，取第 group_index 组的预测
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index + 1) * num_query_per_group
            # shape: [num_dec, B, num_query_per_group, num_cls]
            group_cls_scores = all_cls_scores[:, :, group_query_start:group_query_end, :]
            group_bbox_preds = all_bbox_preds[:, :, group_query_start:group_query_end, :]

            # 对当前组做多层损失计算（深监督）
            losses_cls, losses_bbox = multi_apply(
                self.loss_single, group_cls_scores, group_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list)

            # 损失除以组数后累加（等效于 G 组损失的平均值）
            loss_dict['loss_cls'] += losses_cls[-1] / self.group_detr
            loss_dict['loss_bbox'] += losses_bbox[-1] / self.group_detr

            # 辅助层损失（深监督）
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls'] += loss_cls_i / self.group_detr
                loss_dict[f'd{num_dec_layer}.loss_bbox'] += loss_bbox_i / self.group_detr
                num_dec_layer += 1

        return loss_dict