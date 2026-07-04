#----------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)   #
# Source code: https://github.com/OpenDriveLab/UniAD                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                 #
# Modified from panoptic_segformer (https://github.com/zhiqi-li/Panoptic-SegFormer)#
#--------------------------------------------------------------------------------- #

# ══════════════════════════════════════════════════════════════════════════════════
# 【文件说明】panseg_head.py —— UniAD 全景语义分割头（Seg Head）
# ══════════════════════════════════════════════════════════════════════════════════
#
# 【功能定位】
#   本模块是 UniAD 端到端系统的第二个感知模块 —— "地图语义分割头"（Seg Head）。
#   它从 Track Head 输出的 BEV 特征图出发，预测 BEV 视角下的全景分割结果，
#   具体包括：
#     1. Things（可数实例）：车道线（divider / crossing / contour 3类）
#     2. Stuff（不可数背景）：可行驶区域（drivable area）
#
#   输出的分割结果将传递给 Motion Head，为后续运动预测提供语义背景信息。
#
# 【全景分割 Panoptic Segmentation 是什么？】
#   全景分割 = 实例分割（Instance Seg）+ 语义分割（Semantic Seg）
#     - 实例分割（Things）：识别"有多少个物体"以及"每个像素属于哪个实例"，如行人、车辆、车道线
#     - 语义分割（Stuff）： 识别"整块区域属于哪类"，如道路、天空、草地
#   在自动驾驶中，车道线属于 Things（有形状界限），可行驶区域属于 Stuff（面状背景）。
#
# 【两阶段 Decoder 设计（核心架构）】
#   本模块使用"两阶段"解码策略，参考 Panoptic SegFormer 论文：
#   ┌─────────────────────────────────────────────────────────────────┐
#   │  阶段1：位置解码器（Location Decoder）                           │
#   │  输入：BEV 特征图 + query_embedding                              │
#   │  职责：预测边界框位置 (cx, cy, w, h)，判断哪些 query 对应真实目标 │
#   │  输出：all_cls_scores, all_bbox_preds, reference, query, memory  │
#   ├─────────────────────────────────────────────────────────────────┤
#   │  阶段2：掩码解码器（Mask Decoder）                               │
#   │  输入：阶段1 筛选出的高质量 query（通过 filter_query 过滤）       │
#   │  职责：预测每个实例的像素级掩码（mask）                          │
#   │  输出：mask_things（车道线掩码），mask_stuff（可行驶区域掩码）     │
#   └─────────────────────────────────────────────────────────────────┘
#
# 【Things / Stuff 分离设计】
#   - things_mask_head：专门处理 Things 的掩码解码（TransformerHead 架构）
#   - stuff_mask_head：专门处理 Stuff 的掩码解码（TransformerHead 架构）
#   两者分开是因为 Things 需要实例感知（每个实例独立），Stuff 需要语义感知（整体预测）。
#
# 【与 track_head.py 的关系】
#   - track_head.py 输出 BEV 特征图 → panseg_head.py 消费 BEV 特征图
#   - 两者共享 BEV 特征，但预测目标完全不同（3D 目标 vs 2D 地图元素）
#
# ══════════════════════════════════════════════════════════════════════════════════

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from .seg_head_plugin import SegDETRHead, IOU  # 父类 SegDETRHead 和 IoU 计算工具


@HEADS.register_module()
class PansegformerHead(SegDETRHead):
    """
    全景分割头（Panoptic SegFormer Head）

    继承自 SegDETRHead，在 BEV 特征空间内完成 Things + Stuff 的全景分割。

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                       PansegformerHead 数据流                            │
    │                                                                          │
    │  BEV 特征图 (bev_embed)                                                  │
    │       │                                                                  │
    │       ▼                                                                  │
    │  [位置解码器 transformer]  ──→ cls_scores, bbox_preds, query, memory     │
    │       │                                                                  │
    │       ▼                                                                  │
    │  [filter_query]  ──→ 筛选高质量 query（具有高置信度/低代价的 Things 查询）│
    │       │                                                                  │
    │       ├──→ [things_mask_head]  ──→ Things 掩码（车道线）                 │
    │       └──→ [stuff_mask_head]   ──→ Stuff 掩码（可行驶区域）              │
    │                                                                          │
    │  输出：lane_mask, drivable_mask（传递给 Motion Head）                    │
    └──────────────────────────────────────────────────────────────────────────┘

    Args:
        bev_h (int): BEV 特征图的高度（网格数量），例如 200
        bev_w (int): BEV 特征图的宽度（网格数量），例如 200
        canvas_size (tuple): 输出掩码的像素大小，例如 (200, 400)
        pc_range (list): 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        with_box_refine (bool): 是否在位置解码器中逐层精化边界框参考点
        as_two_stage (bool): 是否采用 Encoder 输出生成 proposal（两阶段检测）
        transformer (ConfigDict): 位置解码器 Transformer 配置
        quality_threshold_things (float): Things 掩码的置信度阈值（推理时用）
        quality_threshold_stuff (float): Stuff 掩码的置信度阈值
        overlap_threshold_things (float): Things 掩码的重叠面积阈值（避免重叠）
        overlap_threshold_stuff (float): Stuff 掩码的重叠面积阈值
        thing_transformer_head (dict): Things 掩码解码器配置
        stuff_transformer_head (dict): Stuff 掩码解码器配置
        loss_mask (dict): 掩码损失函数配置（默认 DiceLoss）
    """

    def __init__(
            self,
            *args,
            bev_h,                        # BEV 特征图高度（像素级网格数）
            bev_w,                        # BEV 特征图宽度
            canvas_size,                  # 输出分割图的像素大小
            pc_range,                     # 点云空间范围 [xmin, ymin, zmin, xmax, ymax, zmax]
            with_box_refine=False,        # 是否使用边界框逐层精化
            as_two_stage=False,           # 是否使用两阶段模式
            transformer=None,             # 位置解码器配置
            quality_threshold_things=0.25,  # Things 推理质量阈值
            quality_threshold_stuff=0.25,   # Stuff 推理质量阈值
            overlap_threshold_things=0.4,   # Things 掩码重叠阈值（用于全景合并）
            overlap_threshold_stuff=0.2,    # Stuff 掩码重叠阈值
            thing_transformer_head=dict(
                type='TransformerHead',   # Things 掩码解码器（Transformer 结构）
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            stuff_transformer_head=dict(
                type='TransformerHead',   # Stuff 掩码解码器（Transformer 结构）
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            loss_mask=dict(type='DiceLoss', weight=2.0),  # 掩码损失（Dice Loss）
            train_cfg=dict(
                assigner=dict(type='HungarianAssigner',
                              cls_cost=dict(type='ClassificationCost',
                                            weight=1.),
                              reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                              iou_cost=dict(type='IoUCost',
                                            iou_mode='giou',
                                            weight=2.0)),
                sampler=dict(type='PseudoSampler'),
            ),
            **kwargs):
        # ── BEV 空间基本参数 ────────────────────────────────────────────────
        self.bev_h = bev_h          # BEV 网格高度
        self.bev_w = bev_w          # BEV 网格宽度
        self.canvas_size = canvas_size  # 输出分割图大小
        self.pc_range = pc_range    # 点云空间范围
        self.real_w = self.pc_range[3] - self.pc_range[0]  # 真实世界宽度（米）
        self.real_h = self.pc_range[4] - self.pc_range[1]  # 真实世界高度（米）

        # ── 网络结构参数 ─────────────────────────────────────────────────────
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.quality_threshold_things = 0.1   # 注意：硬编码覆盖了传入的参数值
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        self.fp16_enabled = False  # 是否开启半精度（FP16）模式

        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage  # 向 transformer 传递两阶段标记

        # 掩码解码器的 Transformer 层数
        self.num_dec_things = thing_transformer_head['num_decoder_layers']  # Things 掩码解码层数
        self.num_dec_stuff = stuff_transformer_head['num_decoder_layers']   # Stuff 掩码解码层数

        # ── 调用父类初始化（包含 transformer、assigner、sampler 等基础组件）──
        super(PansegformerHead, self).__init__(*args,
                                            transformer=transformer,
                                            train_cfg=train_cfg,
                                            **kwargs)

        # ── 训练配置：带掩码的 assigner 和 sampler ──────────────────────────
        if train_cfg:
            # 带掩码的采样器（用于计算 mask loss 时的正负样本分配）
            sampler_cfg = train_cfg['sampler_with_mask']
            self.sampler_with_mask = build_sampler(sampler_cfg, context=self)

            # 带掩码的匈牙利匹配器（同时考虑 cls + bbox + mask 三个代价项）
            assigner_cfg = train_cfg['assigner_with_mask']
            self.assigner_with_mask = build_assigner(assigner_cfg)

            # 过滤低质量 query 的匈牙利匹配器（max_pos 限制每个 GT 最多被匹配几次）
            # max_pos=3 表示每个 GT 最多分配给 3 个 query，平衡显存消耗
            self.assigner_filter = build_assigner(
                dict(
                    type='HungarianAssigner_filter',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost',
                                  weight=5.0,
                                  box_format='xywh'),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                    max_pos=3  # 显存受限时可设为 1，1080Ti 也可训练
                ), )

        # ── 构建掩码损失函数和掩码解码器 ────────────────────────────────────
        self.loss_mask = build_loss(loss_mask)                         # DiceLoss
        self.things_mask_head = build_transformer(thing_transformer_head)  # Things 掩码解码器
        self.stuff_mask_head = build_transformer(stuff_transformer_head)   # Stuff 掩码解码器
        self.count = 0  # 调试计数器（目前未使用）

    def _init_layers(self):
        """初始化各分支的网络层（分类分支 + 回归分支 + 掩码分支）"""

        # ── BEV 位置编码（单阶段模式才需要）───────────────────────────────
        if not self.as_two_stage:
            # bev_embedding: BEV 网格上每个位置的可学习位置编码
            # shape: (bev_h * bev_w, embed_dims)
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        # ── 分类分支 ────────────────────────────────────────────────────────
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)  # Things 分类头（多类）
        fc_cls_stuff = Linear(self.embed_dims, 1)                 # Stuff 分类头（二分类：有/无）

        # ── 回归分支（MLP: embed_dims → embed_dims → ... → 4）──────────────
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))  # 输出 (cx, cy, w, h)
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            """复制 N 份相同结构但参数独立的模块（用于深监督）"""
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # ── 决定需要多少个预测头（深监督用）──────────────────────────────
        # 两阶段时需要额外一个头处理 encoder 输出的 proposal
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        # ── 位置解码器的多层预测分支（深监督：每个 decoder 层都有独立的分类/回归头）──
        if self.with_box_refine:
            # with_box_refine=True：每层参数独立（逐层精化参考点）
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            # with_box_refine=False：所有层共享同一套参数
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        # ── Query Embedding（单阶段模式）──────────────────────────────────
        if not self.as_two_stage:
            # query_embedding 前 embed_dims 维 = Query 内容，后 embed_dims 维 = Query 位置
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

        # ── Stuff Query（每个 stuff 类别一个可学习的 Query）──────────────
        # stuff_query shape: (num_stuff_classes, embed_dims * 2)
        # 前 embed_dims: content query；后 embed_dims: position query
        self.stuff_query = nn.Embedding(self.num_stuff_classes, self.embed_dims * 2)

        # ── 掩码解码器的多层预测分支（深监督：每个 mask decoder 层独立）──
        self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things)     # Things 掩码解码器的回归头
        self.cls_thing_branches = _get_clones(fc_cls, self.num_dec_things)    # Things 掩码解码器的分类头
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)  # Stuff 掩码解码器的分类头

    def init_weights(self):
        """初始化权重，使用 Focal Loss 偏置初始化和回归分支零初始化"""
        self.transformer.init_weights()

        # ── 分类头：使用先验概率初始化偏置（避免初始训练时正样本被淹没）──
        # bias = log(p/(1-p))，p=0.01 → bias ≈ -4.6
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_thing_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)

        # ── 回归头：最后一层权重/偏置全部初始化为 0（预测偏移量初始为 0）──
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.reg_branches2:
            constant_init(m[-1], 0, bias=0)

        # 第一层回归头的 w/h 维（索引 2:4）偏置初始化为 -2.0
        # 这是因为 sigmoid(-2) ≈ 0.12，初始预测尺寸较小，符合目标尺寸的先验分布
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    @force_fp32(apply_to=('bev_embed', ))
    def forward(self, bev_embed):
        """
        前向传播：阶段1 位置解码。

        将 BEV 特征图输入 Transformer，得到分类分数、边界框预测和 query 特征。
        这些输出是后续掩码解码的输入。

        Args:
            bev_embed (Tensor): BEV 特征图，shape (bev_h*bev_w, bs, embed_dims)
                注意：BEV 特征以 "序列形式" 传入（不是 4D 图像格式）

        Returns:
            dict: 包含以下字段
                - 'bev_embed': 原始 BEV 特征（传递给掩码解码器用）
                - 'outputs_classes': 各层分类分数 [nb_dec, bs, num_query, num_cls]
                - 'outputs_coords': 各层边界框预测 [nb_dec, bs, num_query, 4]
                - 'enc_outputs_class': 两阶段时 encoder 输出的分类分数（否则 None）
                - 'enc_outputs_coord': 两阶段时 encoder 输出的坐标（否则 None）
                - 'args_tuple': (memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl)
                    这些参数将传递给掩码解码器
                - 'reference': 最后一层的参考点（用于掩码解码器的残差坐标解码）
        """
        # bev_embed shape: (bev_h*bev_w, bs, embed_dims)
        _, bs, _ = bev_embed.shape

        # ── 将 BEV 特征从序列形式转换为图像形式 ──────────────────────────
        # 转换：(seq_len, bs, C) → (bs, bev_h, bev_w, C) → (bs, C, bev_h, bev_w)
        mlvl_feats = [torch.reshape(bev_embed, (bs, self.bev_h, self.bev_w, -1)).permute(0, 3, 1, 2)]
        # 初始化 mask（全 0，表示所有位置均有效，不被 padding mask 屏蔽）
        img_masks = mlvl_feats[0].new_zeros((bs, self.bev_h, self.bev_w))

        # ── 记录各尺度特征图的 H×W，供掩码解码器使用 ───────────────────
        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]

        # ── 为每个尺度生成 mask 和位置编码 ─────────────────────────────
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            # 将 mask 插值到当前特征图大小，再转换为 bool 类型
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            # 根据 mask 生成正弦位置编码（告诉模型"我在哪里"）
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        # ── 准备 query embedding ─────────────────────────────────────
        query_embeds = None
        if not self.as_two_stage:
            # 单阶段：使用可学习的 query_embedding，shape: (num_query, embed_dims*2)
            query_embeds = self.query_embedding.weight

        # ── 调用位置解码器 Transformer ───────────────────────────────
        # memory: encoder 输出的 BEV 特征（供 cross-attention 使用），shape: (seq_len, bs, C)
        # query_pos: query 位置编码，shape: (num_query, bs, C)
        # hs: 所有 decoder 层的 query 输出，shape: (nb_dec, num_query, bs, C)
        # init_reference: 初始参考点（第 0 层），shape: (bs, num_query, 2 or 4)
        # inter_references: 中间层参考点，shape: (nb_dec-1, bs, num_query, 2 or 4)
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None
        )

        # ── 调整维度顺序，方便后续处理 ──────────────────────────────
        memory = memory.permute(1, 0, 2)      # (seq_len, bs, C) → (bs, seq_len, C)
        query = hs[-1].permute(1, 0, 2)       # 取最后一层 decoder 输出的 query
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        # ── 打包传递给掩码解码器的参数 ──────────────────────────────
        # args_tuple 包含掩码解码所需的完整上下文信息
        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

        # ── 逐层解码位置（深监督）──────────────────────────────────
        hs = hs.permute(0, 2, 1, 3)  # (nb_dec, num_query, bs, C) → (nb_dec, bs, num_query, C)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            # 获取当前层的参考点
            if lvl == 0:
                reference = init_reference  # 第 0 层使用初始参考点
            else:
                reference = inter_references[lvl - 1]  # 后续层使用上一层更新后的参考点

            # 将参考点从 [0,1] 映射回 logit 空间（做残差加法）
            reference = inverse_sigmoid(reference)

            # 分类预测
            outputs_class = self.cls_branches[lvl](hs[lvl])

            # 回归预测（输出 logit 偏移量）
            tmp = self.reg_branches[lvl](hs[lvl])

            # 残差坐标解码：预测值 = 参考点 + 偏移量（在 logit 空间做加法）
            if reference.shape[-1] == 4:
                # 参考点包含 (cx, cy, w, h)，全部加偏移
                tmp += reference
            else:
                # 参考点只有 (cx, cy)，只加前两维
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # sigmoid 映射到 [0,1] 归一化坐标
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # 将列表 stack 为 Tensor，方便后续批量计算损失
        outputs_classes = torch.stack(outputs_classes)  # (nb_dec, bs, num_query, num_cls)
        outputs_coords = torch.stack(outputs_coords)    # (nb_dec, bs, num_query, 4)

        # ── 组装输出字典 ─────────────────────────────────────────────
        outs = {
            'bev_embed': None if self.as_two_stage else bev_embed,
            'outputs_classes': outputs_classes,   # 位置解码器各层分类结果
            'outputs_coords': outputs_coords,     # 位置解码器各层坐标结果
            'enc_outputs_class': enc_outputs_class if self.as_two_stage else None,
            'enc_outputs_coord': enc_outputs_coord.sigmoid() if self.as_two_stage else None,
            'args_tuple': args_tuple,  # 传给掩码解码器的上下文
            'reference': reference,   # 最后一层参考点（掩码解码器做残差用）
        }

        return outs

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple', 'reference'))
    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        gt_labels_list,
        gt_bboxes_list,
        gt_masks_list,
        img_metas=None,
        gt_bboxes_ignore=None,
    ):
        """
        总损失函数：汇总所有 decoder 层的损失（深监督）。

        策略：
          - 位置解码器前 L-1 层：只计算 cls + bbox + iou 损失（浅层引导）
          - 位置解码器最后一层（第 L 层）：同时计算位置损失 + 掩码损失（完整监督）
          - 掩码解码器各层（things/stuff）：单独计算掩码损失（深监督）
          - Things/Stuff 损失权重动态调整（按 GT 数量比例）

        Args:
            all_cls_scores (Tensor): 位置解码器各层分类分数 [nb_dec, bs, num_query, cls_out_channels]
            all_bbox_preds (Tensor): 位置解码器各层边界框预测 [nb_dec, bs, num_query, 4]
            enc_cls_scores (Tensor): 两阶段时 encoder 分类分数（否则 None）
            enc_bbox_preds (Tensor): 两阶段时 encoder 坐标预测（否则 None）
            args_tuple (tuple): 传给掩码解码器的上下文元组
            reference (Tensor): 最后一层参考点
            gt_labels_list (list[Tensor]): 每张图的 GT 类别标签
            gt_bboxes_list (list[Tensor]): 每张图的 GT 边界框（xyxy 格式）
            gt_masks_list (list[Tensor]): 每张图的 GT 实例掩码
            img_metas (list[dict]): 图像元信息
            gt_bboxes_ignore (None): 忽略的边界框（本模型不支持，必须为 None）

        Returns:
            dict[str, Tensor]: 各损失项字典，例如：
                - 'loss_cls': 最后层分类损失（Things）
                - 'loss_bbox': 最后层边界框回归损失
                - 'loss_iou': 最后层 IoU 损失
                - 'loss_mask_things': Things 掩码损失
                - 'loss_mask_stuff': Stuff 掩码损失
                - 'd{i}.loss_xxx': 其他层的深监督损失
        """
        # 使用 canvas_size 作为图像尺寸（BEV 坐标系）
        img_metas[0]['img_shape'] = (self.canvas_size[0], self.canvas_size[1], 3)

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # ── 按类别 ID 将 GT 分为 Things 和 Stuff ────────────────────────
        # Things：类别 ID < num_things_classes（车道线等）
        # Stuff： 类别 ID >= num_things_classes（可行驶区域等）
        gt_things_lables_list = []
        gt_things_bboxes_list = []
        gt_things_masks_list = []
        gt_stuff_labels_list = []
        gt_stuff_masks_list = []
        for i, each in enumerate(gt_labels_list):
            things_selected = each < self.num_things_classes  # 布尔掩码，选出 things
            stuff_selected = things_selected == False          # 布尔掩码，选出 stuff

            gt_things_lables_list.append(gt_labels_list[i][things_selected])
            gt_things_bboxes_list.append(gt_bboxes_list[i][things_selected])
            gt_things_masks_list.append(gt_masks_list[i][things_selected])

            gt_stuff_labels_list.append(gt_labels_list[i][stuff_selected])
            gt_stuff_masks_list.append(gt_masks_list[i][stuff_selected])

        # ── 位置解码器前 L-1 层：只计算 cls + bbox + iou ─────────────────
        # 用 multi_apply 对每一层调用 loss_single（父类方法）
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_things_bboxes_list for _ in range(num_dec_layers - 1)]
        all_gt_labels_list = [gt_things_lables_list for _ in range(num_dec_layers - 1)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers - 1)]
        img_metas_list = [img_metas for _ in range(num_dec_layers - 1)]

        # 前 L-1 层只有位置监督，不涉及掩码
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores[:-1], all_bbox_preds[:-1],
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        # ── 最后一层（第 L 层）：位置 + 掩码联合监督 ─────────────────────
        # loss_single_panoptic 会先做 query filter，再进行掩码解码和损失计算
        losses_cls_f, losses_bbox_f, losses_iou_f, \
        losses_masks_things_f, losses_masks_stuff_f, \
        loss_mask_things_list_f, loss_mask_stuff_list_f, \
        loss_iou_list_f, loss_bbox_list_f, loss_cls_list_f, loss_cls_stuff_list_f, \
        things_ratio, stuff_ratio = self.loss_single_panoptic(
            all_cls_scores[-1], all_bbox_preds[-1], args_tuple, reference,
            gt_things_bboxes_list, gt_things_lables_list, gt_things_masks_list,
            (gt_stuff_labels_list, gt_stuff_masks_list), img_metas,
            gt_bboxes_ignore)

        # ── 组装损失字典 ─────────────────────────────────────────────
        loss_dict = dict()

        # Encoder 输出的 proposal 损失（仅两阶段模式）
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_things_lables_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_things_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls * things_ratio
            loss_dict['enc_loss_bbox'] = enc_losses_bbox * things_ratio
            loss_dict['enc_loss_iou'] = enc_losses_iou * things_ratio

        # 最后一层（第 L 层）的位置损失（乘以动态权重 things_ratio）
        loss_dict['loss_cls'] = losses_cls_f * things_ratio
        loss_dict['loss_bbox'] = losses_bbox_f * things_ratio
        loss_dict['loss_iou'] = losses_iou_f * things_ratio
        # 最后一层的掩码损失
        loss_dict['loss_mask_things'] = losses_masks_things_f * things_ratio
        loss_dict['loss_mask_stuff'] = losses_masks_stuff_f * stuff_ratio

        # 掩码解码器各层的深监督损失（d{i}.xxx 格式）
        num_dec_layer = 0
        for i in range(len(loss_mask_things_list_f)):
            loss_dict[f'd{i}.loss_mask_things_f'] = loss_mask_things_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_iou_f'] = loss_iou_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_bbox_f'] = loss_bbox_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_cls_f'] = loss_cls_list_f[i] * things_ratio
        for i in range(len(loss_mask_stuff_list_f)):
            loss_dict[f'd{i}.loss_mask_stuff_f'] = loss_mask_stuff_list_f[i] * stuff_ratio
            loss_dict[f'd{i}.loss_cls_stuff_f'] = loss_cls_stuff_list_f[i] * stuff_ratio

        # 位置解码器前 L-1 层的损失
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls, losses_bbox, losses_iou):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * things_ratio
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i * things_ratio
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i * things_ratio
            num_dec_layer += 1

        return loss_dict

    def filter_query(self,
                     cls_scores_list,
                     bbox_preds_list,
                     gt_bboxes_list,
                     gt_labels_list,
                     img_metas,
                     gt_bboxes_ignore_list=None):
        """
        Query 过滤函数：利用位置解码器的代价（cost）筛选高质量的 Things Query。

        背景：
          - 位置解码器输出 num_query 个 query，其中大多数是背景（负样本）
          - 只有被分配到 GT 的 query（正样本）才有意义传递给掩码解码器
          - 过滤掉负样本可以显著减少掩码解码器的计算量

        策略：
          - 调用 assigner_filter（带 max_pos 限制的匈牙利匹配器）
          - 返回每张图的正/负样本掩码（pos_inds_mask_list, neg_inds_mask_list）

        Args:
            cls_scores_list (list[Tensor]): 每张图的分类分数 [num_query, cls_out_channels]
            bbox_preds_list (list[Tensor]): 每张图的边界框预测 [num_query, 4]
            gt_bboxes_list (list[Tensor]): 每张图的 GT 框
            gt_labels_list (list[Tensor]): 每张图的 GT 标签
            img_metas (list[dict]): 图像元信息

        Returns:
            tuple: (pos_inds_mask_list, neg_inds_mask_list, labels_list, ...)
                - pos_inds_mask_list: 每张图中被选为正样本的 query 索引列表
                - neg_inds_mask_list: 每张图中被选为负样本的 query 索引列表
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        # 逐图进行 query 过滤（multi_apply 并行化处理 batch 中的每张图）
        (pos_inds_mask_list, neg_inds_mask_list, labels_list,
         label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._filter_query_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return pos_inds_mask_list, neg_inds_mask_list, labels_list, label_weights_list, bbox_targets_list, \
               bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list, neg_inds_list

    def _filter_query_single(self,
                             cls_score,
                             bbox_pred,
                             gt_bboxes,
                             gt_labels,
                             img_meta,
                             gt_bboxes_ignore=None):
        """
        对单张图像进行 query 过滤（_filter_query 的单图实现）。

        步骤：
          1. 用 assigner_filter 做匈牙利匹配（限制 max_pos=3）
          2. 返回正/负样本索引和对应的目标标签

        Args:
            cls_score (Tensor): 单张图的分类分数 [num_query, cls_out_channels]
            bbox_pred (Tensor): 单张图的边界框预测 [num_query, 4]
            gt_bboxes (Tensor): GT 框 [num_gts, 4]（xyxy 格式）
            gt_labels (Tensor): GT 标签 [num_gts]
            img_meta (dict): 图像元信息

        Returns:
            tuple: (pos_ind_mask, neg_ind_mask, labels, label_weights,
                    bbox_targets, bbox_weights, pos_inds, neg_inds)
        """
        num_bboxes = bbox_pred.size(0)

        # 数值安全保护：梯度爆炸可能导致 bbox_pred/cls_score 含 NaN/Inf，
        # 在传入 assigner_filter 之前截断，防止 cost matrix 无效引发崩溃。
        bbox_pred = torch.nan_to_num(bbox_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        cls_score = torch.nan_to_num(cls_score, nan=0.0, posinf=1e4, neginf=-1e4)

        # 匈牙利匹配（返回正样本掩码、负样本掩码、匹配结果）
        pos_ind_mask, neg_ind_mask, assign_result = self.assigner_filter.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore)

        # 采样（PseudoSampler 直接返回所有正负样本，不做随机采样）
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds  # 正样本 query 的索引
        neg_inds = sampling_result.neg_inds  # 负样本 query 的索引

        # ── 构建分类目标标签 ───────────────────────────────────────────
        # 初始化所有 query 为背景类（num_things_classes）
        labels = gt_bboxes.new_full((num_bboxes, ), self.num_things_classes, dtype=torch.long)
        # 正样本 query 分配对应 GT 类别
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # ── 构建边界框回归目标 ────────────────────────────────────────
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0  # 只对正样本计算回归损失

        img_h, img_w, _ = img_meta['img_shape']
        # 将 GT 框归一化到 [0,1]，并从 xyxy 转换为 cxcywh 格式
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        return (pos_ind_mask, neg_ind_mask, labels, label_weights,
                bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets_with_mask(self,
                              cls_scores_list,
                              bbox_preds_list,
                              masks_preds_list_thing,
                              gt_bboxes_list,
                              gt_labels_list,
                              gt_masks_list,
                              img_metas,
                              gt_bboxes_ignore_list=None):
        """
        为整个 batch 计算带掩码的监督目标（_get_target_single_with_mask 的 batch 版本）。

        在掩码解码器中，匹配算法同时考虑：
          - 分类代价（cls cost）
          - 边界框代价（bbox cost）
          - 掩码 IoU 代价（mask iou cost）

        Args:
            cls_scores_list (list[Tensor]): 过滤后的 query 分类分数
            bbox_preds_list (list[Tensor]): 过滤后的 query 边界框预测
            masks_preds_list_thing (list[Tensor]): 过滤后的 query 对应的掩码预测
            gt_bboxes_list, gt_labels_list, gt_masks_list: GT 标注

        Returns:
            tuple: 包含各类目标标签、权重以及正负样本索引
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        # 逐图计算目标
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single_with_mask,
                                      cls_scores_list, bbox_preds_list,
                                      masks_preds_list_thing, gt_bboxes_list,
                                      gt_labels_list, gt_masks_list, img_metas,
                                      gt_bboxes_ignore_list)

        num_total_pos_thing = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg_thing = sum((inds.numel() for inds in neg_inds_list))

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, mask_weights_list,
                num_total_pos_thing, num_total_neg_thing, pos_inds_list)

    def _get_target_single_with_mask(self,
                                     cls_score,
                                     bbox_pred,
                                     masks_preds_things,
                                     gt_bboxes,
                                     gt_labels,
                                     gt_masks,
                                     img_meta,
                                     gt_bboxes_ignore=None):
        """
        对单张图像计算带掩码的监督目标（get_targets_with_mask 的单图实现）。

        与 filter_query 不同的是：
          - filter_query 只用位置代价（用于筛选 query）
          - 此函数使用位置 + 掩码代价（用于最终损失计算）

        Args:
            cls_score (Tensor): 过滤后 query 的分类分数
            bbox_pred (Tensor): 过滤后 query 的边界框预测
            masks_preds_things (Tensor): 过滤后 query 的掩码预测
            gt_bboxes (Tensor): GT 框
            gt_labels (Tensor): GT 类别
            gt_masks (Tensor): GT 像素级掩码
            img_meta (dict): 图像元信息

        Returns:
            tuple: (labels, label_weights, bbox_targets, bbox_weights,
                    mask_target, mask_weights, pos_inds, neg_inds)
        """
        num_bboxes = bbox_pred.size(0)
        gt_masks = gt_masks.float()

        # 数值安全保护：梯度爆炸可能导致 bbox_pred/cls_score 含 NaN/Inf
        bbox_pred = torch.nan_to_num(bbox_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        cls_score = torch.nan_to_num(cls_score, nan=0.0, posinf=1e4, neginf=-1e4)

        # 带掩码的匈牙利匹配（assigner_with_mask 同时考虑掩码代价）
        assign_result = self.assigner_with_mask.assign(
            bbox_pred, cls_score, masks_preds_things,
            gt_bboxes, gt_labels, gt_masks, img_meta, gt_bboxes_ignore)

        # 带掩码的采样
        sampling_result = self.sampler_with_mask.sample(
            assign_result, bbox_pred, gt_bboxes, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # ── 分类目标 ─────────────────────────────────────────────────
        labels = gt_bboxes.new_full((num_bboxes, ), self.num_things_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # ── 边界框回归目标 ───────────────────────────────────────────
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # ── 掩码目标 ─────────────────────────────────────────────────
        # mask_weights: 正样本权重为 1，负样本为 0（只对正样本计算 Dice Loss）
        mask_weights = masks_preds_things.new_zeros(num_bboxes)
        mask_weights[pos_inds] = 1.0

        pos_gt_masks = sampling_result.pos_gt_masks  # 正样本对应的 GT 掩码
        _, w, h = pos_gt_masks.shape
        # 初始化掩码目标为全 0，再将正样本位置填入对应 GT 掩码
        mask_target = masks_preds_things.new_zeros([num_bboxes, w, h])
        mask_target[pos_inds] = pos_gt_masks

        return (labels, label_weights, bbox_targets, bbox_weights,
                mask_target, mask_weights, pos_inds, neg_inds)

    def get_filter_results_and_loss(self, cls_scores, bbox_preds,
                                    cls_scores_list, bbox_preds_list,
                                    gt_bboxes_list, gt_labels_list, img_metas,
                                    gt_bboxes_ignore_list):
        """
        先对 query 进行过滤，再计算位置解码器的分类/IoU/bbox 损失。

        这是 loss_single_panoptic 的第一步：
          1. filter_query：用 assigner_filter 筛选高质量 query（pos_inds_mask_list）
          2. 计算位置解码器损失（cls + iou + bbox）
          3. 返回 pos_inds_mask_list 供后续掩码解码使用

        Args:
            cls_scores (Tensor): 最后层分类分数 [bs, num_query, num_cls]
            bbox_preds (Tensor): 最后层边界框预测 [bs, num_query, 4]
            cls_scores_list (list[Tensor]): 按图拆分的分类分数
            bbox_preds_list (list[Tensor]): 按图拆分的边界框预测
            gt_bboxes_list, gt_labels_list: GT 标注

        Returns:
            tuple: (loss_cls, loss_iou, loss_bbox, pos_inds_mask_list, num_total_pos_thing)
        """
        # ── Step 1: query 过滤，获取正样本掩码 ──────────────────────
        pos_inds_mask_list, neg_inds_mask_list, labels_list, label_weights_list, bbox_targets_list, \
        bbox_weights_list, num_total_pos_thing, num_total_neg_thing, pos_inds_list, neg_inds_list = self.filter_query(
            cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list,
            img_metas, gt_bboxes_ignore_list)

        # 将 batch 维度展开，合并所有图的监督信息
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ── Step 2: 计算分类损失（Focal Loss）──────────────────────
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # avg_factor = 正样本数 + 负样本数×bg_cls_weight（调节背景损失权重）
        cls_avg_factor = num_total_pos_thing * 1.0 + \
                         num_total_neg_thing * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            # 多 GPU 同步平均
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # 跨 GPU 同步正样本数量（用于归一化 bbox 损失）
        num_total_pos_thing = loss_cls.new_tensor([num_total_pos_thing])
        num_total_pos_thing = torch.clamp(reduce_mean(num_total_pos_thing), min=1).item()

        # ── Step 3: 计算 IoU 损失和 bbox 回归损失 ──────────────────
        # 构建每张图像的 rescale 因子（用于将归一化坐标转为真实像素坐标）
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # 将归一化的 cxcywh 坐标转换为真实像素 xyxy 坐标（计算 IoU 需要）
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # IoU 损失（默认 GIoU Loss，对形状和位置双重约束）
        loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos_thing)

        # L1 回归损失（在归一化坐标空间计算）
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos_thing)

        return loss_cls, loss_iou, loss_bbox, pos_inds_mask_list, num_total_pos_thing

    def loss_single_panoptic(self,
                             cls_scores,
                             bbox_preds,
                             args_tuple,
                             reference,
                             gt_bboxes_list,
                             gt_labels_list,
                             gt_masks_list,
                             gt_panoptic_list,
                             img_metas,
                             gt_bboxes_ignore_list=None):
        """
        全景分割联合损失函数（位置解码器最后一层 + 掩码解码器所有层）。

        这是整个 PansegformerHead 最核心的函数，包含：
          1. 调用 get_filter_results_and_loss 完成位置损失 + query 过滤
          2. 将过滤后的 Things Query 送入 things_mask_head 预测 Things 掩码
          3. 将 Stuff Query 送入 stuff_mask_head 预测 Stuff 掩码
          4. 对各层掩码解码结果进行深监督（Dice Loss）
          5. 动态调整 Things / Stuff 损失权重

        Args:
            cls_scores (Tensor): 位置解码器最后一层分类分数 [bs, num_query, num_cls]
            bbox_preds (Tensor): 位置解码器最后一层边界框预测 [bs, num_query, 4]
            args_tuple (tuple): 掩码解码所需上下文（memory, memory_mask, ...）
            reference (Tensor): 最后层参考点（用于残差坐标解码）
            gt_bboxes_list (list[Tensor]): Things 的 GT 框
            gt_labels_list (list[Tensor]): Things 的 GT 标签
            gt_masks_list (list[Tensor]): Things 的 GT 掩码
            gt_panoptic_list (tuple): (gt_stuff_labels_list, gt_stuff_masks_list)
            img_metas (list[dict]): 图像元信息
            gt_bboxes_ignore_list: 忽略的框（必须为 None）

        Returns:
            tuple: 包含 Things/Stuff 各类损失、各层深监督损失、动态权重 things_ratio/stuff_ratio
        """
        num_imgs = cls_scores.size(0)
        gt_stuff_labels_list, gt_stuff_masks_list = gt_panoptic_list
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        # ── Step 1: 计算位置损失 + 过滤高质量 Query ─────────────────────
        # pos_inds_mask_list：每张图中被分配到 GT 的 query 索引（用于掩码解码）
        loss_cls, loss_iou, loss_bbox, pos_inds_mask_list, num_total_pos_thing = self.get_filter_results_and_loss(
            cls_scores, bbox_preds, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)

        # 解包掩码解码器需要的上下文信息
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        BS, _, dim_query = query.shape[0], query.shape[1], query.shape[-1]

        # ── Step 2: 构建 Things Query（只取位置解码器中被筛选出的正样本 query）──
        # len_query = batch 中最多正样本数（需要 padding 对齐）
        len_query = max([len(pos_ind) for pos_ind in pos_inds_mask_list])
        thing_query = torch.zeros([BS, len_query, dim_query], device=query.device)

        # ── Step 3: 构建 Stuff Query（每个 stuff 类别有固定的可学习 query）──
        # stuff_query.weight shape: (num_stuff_classes, embed_dims * 2)
        # 前半段 = content query，后半段 = position query
        stuff_query, stuff_query_pos = torch.split(self.stuff_query.weight, self.embed_dims, dim=1)
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(BS, -1, -1)  # (BS, num_stuff, embed_dims)
        stuff_query = stuff_query.unsqueeze(0).expand(BS, -1, -1)          # (BS, num_stuff, embed_dims)

        # 将过滤后的正样本 query 特征填入 thing_query
        for i in range(BS):
            thing_query[i, :len(pos_inds_mask_list[i])] = query[i, pos_inds_mask_list[i]]

        # ── 初始化各层深监督收集列表 ─────────────────────────────────────
        mask_preds_things = []          # 最终层 Things 掩码
        mask_preds_stuff = []           # 最终层 Stuff 掩码
        mask_preds_inter_things = [[] for _ in range(self.num_dec_things)]  # 中间层 Things 掩码
        mask_preds_inter_stuff = [[] for _ in range(self.num_dec_stuff)]    # 中间层 Stuff 掩码
        cls_thing_preds = [[] for _ in range(self.num_dec_things)]
        cls_stuff_preds = [[] for _ in range(self.num_dec_stuff)]
        BS, NQ, L = bbox_preds.shape
        new_bbox_preds = [
            torch.zeros([BS, len_query, L]).to(bbox_preds.device)
            for _ in range(self.num_dec_things)
        ]

        # ── Step 4: 掩码解码 ────────────────────────────────────────────
        # things_mask_head：以 thing_query 为 Q，memory 为 KV，预测每个 query 的掩码
        # 输出 mask_things shape: (BS, len_query, H*W, 1)（注意力权重图就是掩码）
        mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
            memory, memory_mask, None, thing_query, None, None, hw_lvl=hw_lvl)

        # stuff_mask_head：以 stuff_query 为 Q，预测每个 stuff 类别的掩码
        mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
            memory, memory_mask, None, stuff_query, None, stuff_query_pos, hw_lvl=hw_lvl)

        # 去除最后的维度 1
        mask_things = mask_things.squeeze(-1)                          # (BS, len_query, H*W)
        mask_inter_things = torch.stack(mask_inter_things, 0).squeeze(-1)  # (num_dec, BS, len_query, H*W)

        mask_stuff = mask_stuff.squeeze(-1)
        mask_inter_stuff = torch.stack(mask_inter_stuff, 0).squeeze(-1)

        # ── 逐图整理 Things/Stuff 的掩码预测 ─────────────────────────────
        for i in range(BS):
            # 取出该图中有效的 Things 掩码（只保留正样本数量）
            tmp_i = mask_things[i][:len(pos_inds_mask_list[i])].reshape(-1, *hw_lvl[0])
            mask_preds_things.append(tmp_i)

            pos_ind = pos_inds_mask_list[i]
            reference_i = reference[i:i + 1, pos_ind, :]  # 对应正样本的参考点

            # 遍历 Things 掩码解码器的每一层（深监督）
            for j in range(self.num_dec_things):
                tmp_i_j = mask_inter_things[j][i][:len(pos_inds_mask_list[i])].reshape(-1, *hw_lvl[0])
                mask_preds_inter_things[j].append(tmp_i_j)

                # 同时做边界框预测（残差解码，类似 bevformer_head 的做法）
                query_things = query_inter_things[j]
                t1, t2, t3 = query_things.shape
                tmp = self.reg_branches2[j](query_things.reshape(t1 * t2, t3)).reshape(t1, t2, 4)
                if len(pos_ind) == 0:
                    # 处理空正样本的 PyTorch broadcast bug
                    tmp = tmp.sum() + reference_i
                elif reference_i.shape[-1] == 4:
                    tmp += reference_i
                else:
                    assert reference_i.shape[-1] == 2
                    tmp[..., :2] += reference_i

                outputs_coord = tmp.sigmoid()
                new_bbox_preds[j][i][:len(pos_inds_mask_list[i])] = outputs_coord
                cls_thing_preds[j].append(self.cls_thing_branches[j](query_things.reshape(t1 * t2, t3)))

            # Stuff 掩码（每个 Stuff 类别一个掩码，reshape 为 H×W 格式）
            tmp_i = mask_stuff[i].reshape(-1, *hw_lvl[0])
            mask_preds_stuff.append(tmp_i)

            for j in range(self.num_dec_stuff):
                tmp_i_j = mask_inter_stuff[j][i].reshape(-1, *hw_lvl[0])
                mask_preds_inter_stuff[j].append(tmp_i_j)

                query_stuff = query_inter_stuff[j]
                s1, s2, s3 = query_stuff.shape
                cls_stuff_preds[j].append(self.cls_stuff_branches[j](query_stuff.reshape(s1 * s2, s3)))

        # ── 拼接所有图的掩码为 Tensor ──────────────────────────────────
        masks_preds_list_thing = [mask_preds_things[i] for i in range(num_imgs)]  # 保留 list 格式（给 get_targets_with_mask 用）
        mask_preds_things = torch.cat(mask_preds_things, 0)
        mask_preds_inter_things = [torch.cat(each, 0) for each in mask_preds_inter_things]
        cls_thing_preds = [torch.cat(each, 0) for each in cls_thing_preds]
        cls_stuff_preds = [torch.cat(each, 0) for each in cls_stuff_preds]
        mask_preds_stuff = torch.cat(mask_preds_stuff, 0)
        mask_preds_inter_stuff = [torch.cat(each, 0) for each in mask_preds_inter_stuff]

        # 只保留过滤后正样本对应的分类分数和边界框预测（给带掩码的 assigner 使用）
        cls_scores_list = [cls_scores_list[i][pos_inds_mask_list[i]] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds_list[i][pos_inds_mask_list[i]] for i in range(num_imgs)]

        # ── Step 5: 计算带掩码的匈牙利匹配目标（用于最终 Dice Loss）──────
        gt_targets = self.get_targets_with_mask(
            cls_scores_list, bbox_preds_list, masks_preds_list_thing,
            gt_bboxes_list, gt_labels_list, gt_masks_list, img_metas, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, _, _, pos_inds_list) = gt_targets

        thing_labels = torch.cat(labels_list, 0)
        things_weights = torch.cat(label_weights_list, 0)

        bboxes_taget = torch.cat(bbox_targets_list)
        bboxes_weights = torch.cat(bbox_weights_list)

        # 构建 rescale 因子（归一化坐标 → 真实像素坐标，计算 IoU 用）
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds_list):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bboxes_gt = bbox_cxcywh_to_xyxy(bboxes_taget) * factors

        # ── 准备 Things 掩码 GT ─────────────────────────────────────────
        mask_things_gt = torch.cat(mask_targets_list, 0).to(torch.float)
        mask_weight_things = torch.cat(mask_weights_list, 0).to(thing_labels.device)

        # ── 准备 Stuff 掩码 GT ─────────────────────────────────────────
        # Stuff 的目标：为每个 stuff 类别构建一张二值掩码
        mask_stuff_gt = []
        mask_weight_stuff = []
        stuff_labels = []
        num_total_pos_stuff = 0
        for i in range(BS):
            num_total_pos_stuff += len(gt_stuff_labels_list[i])  # 统计 stuff GT 数量

            # 将 stuff 标签从全局类 ID 转换为 stuff 内部类 ID（减去 num_things_classes）
            select_stuff_index = gt_stuff_labels_list[i] - self.num_things_classes

            # mask_weight_i_stuff：哪些 stuff 类别有 GT（有 GT=1，无 GT=0）
            mask_weight_i_stuff = torch.zeros([self.num_stuff_classes])
            mask_weight_i_stuff[select_stuff_index] = 1

            # 初始化当前图的 stuff 掩码（全 False），再填入有 GT 的类别
            stuff_masks = torch.zeros(
                (self.num_stuff_classes, *mask_targets_list[i].shape[-2:]),
                device=mask_targets_list[i].device).to(torch.bool)
            stuff_masks[select_stuff_index] = gt_stuff_masks_list[i].to(torch.bool)
            mask_stuff_gt.append(stuff_masks)

            select_stuff_index = torch.cat([
                select_stuff_index,
                torch.tensor([self.num_stuff_classes], device=select_stuff_index.device)
            ])

            # stuff_labels：有 GT 的类别标签为 0（正样本），无 GT 为 1（负样本）
            stuff_labels.append(1 - mask_weight_i_stuff)
            mask_weight_stuff.append(mask_weight_i_stuff)

        mask_weight_stuff = torch.cat(mask_weight_stuff, 0).to(thing_labels.device)
        stuff_labels = torch.cat(stuff_labels, 0).to(thing_labels.device)
        mask_stuff_gt = torch.cat(mask_stuff_gt, 0).to(torch.float)

        # 多 GPU 同步 stuff 正样本数（用于归一化）
        num_total_pos_stuff = loss_cls.new_tensor([num_total_pos_stuff])
        num_total_pos_stuff = torch.clamp(reduce_mean(num_total_pos_stuff), min=1).item()

        # ── Step 6: 计算最终层 Things/Stuff 掩码损失（Dice Loss）────────
        # Things 掩码损失（上采样 2x 后与 GT 对比）
        if mask_preds_things.shape[0] == 0:
            # 边缘情况：无正样本时，返回零损失
            loss_mask_things = (0 * mask_preds_things).sum()
        else:
            # 上采样 2x（分辨率从 BEV 网格恢复为更细粒度）
            mask_preds = F.interpolate(mask_preds_things.unsqueeze(0), scale_factor=2.0, mode='bilinear').squeeze(0)
            # GT 同步调整到预测掩码的分辨率
            mask_targets_things = F.interpolate(mask_things_gt.unsqueeze(0), size=mask_preds.shape[-2:], mode='bilinear').squeeze(0)
            # Dice Loss：衡量预测掩码与 GT 掩码的重叠度（对掩码分割任务效果好）
            loss_mask_things = self.loss_mask(mask_preds, mask_targets_things, mask_weight_things, avg_factor=num_total_pos_thing)

        # Stuff 掩码损失
        if mask_preds_stuff.shape[0] == 0:
            loss_mask_stuff = (0 * mask_preds_stuff).sum()
        else:
            mask_preds = F.interpolate(mask_preds_stuff.unsqueeze(0), scale_factor=2.0, mode='bilinear').squeeze(0)
            mask_targets_stuff = F.interpolate(mask_stuff_gt.unsqueeze(0), size=mask_preds.shape[-2:], mode='bilinear').squeeze(0)
            loss_mask_stuff = self.loss_mask(mask_preds, mask_targets_stuff, mask_weight_stuff, avg_factor=num_total_pos_stuff)

        # ── Step 7: 计算掩码解码器各中间层的深监督损失 ──────────────────
        loss_mask_things_list = []
        loss_mask_stuff_list = []
        loss_iou_list = []
        loss_bbox_list = []

        # Things 掩码解码器各层（深监督）
        for j in range(len(mask_preds_inter_things)):
            mask_preds_this_level = mask_preds_inter_things[j]
            if mask_preds_this_level.shape[0] == 0:
                loss_mask_j = (0 * mask_preds_this_level).sum()
            else:
                mask_preds_this_level = F.interpolate(
                    mask_preds_this_level.unsqueeze(0), scale_factor=2.0, mode='bilinear').squeeze(0)
                loss_mask_j = self.loss_mask(mask_preds_this_level, mask_targets_things,
                                             mask_weight_things, avg_factor=num_total_pos_thing)
            loss_mask_things_list.append(loss_mask_j)

            # 边界框损失（注意：此处乘以 *0，即在掩码解码器中 bbox 损失被强制清零）
            # 原因：掩码解码器不专门预测 bbox，强行监督无意义
            bbox_preds_this_level = new_bbox_preds[j].reshape(-1, 4)
            bboxes_this_level = bbox_cxcywh_to_xyxy(bbox_preds_this_level) * factors
            loss_iou_j = self.loss_iou(bboxes_this_level, bboxes_gt, bboxes_weights,
                                       avg_factor=num_total_pos_thing) * 0  # 清零
            if bboxes_taget.shape[0] != 0:
                loss_bbox_j = self.loss_bbox(bbox_preds_this_level, bboxes_taget, bboxes_weights,
                                             avg_factor=num_total_pos_thing) * 0  # 清零
            else:
                loss_bbox_j = bbox_preds_this_level.sum() * 0
            loss_iou_list.append(loss_iou_j)
            loss_bbox_list.append(loss_bbox_j)

        # Stuff 掩码解码器各层（深监督）
        for j in range(len(mask_preds_inter_stuff)):
            mask_preds_this_level = mask_preds_inter_stuff[j]
            if mask_preds_this_level.shape[0] == 0:
                loss_mask_j = (0 * mask_preds_this_level).sum()
            else:
                mask_preds_this_level = F.interpolate(
                    mask_preds_this_level.unsqueeze(0), scale_factor=2.0, mode='bilinear').squeeze(0)
                loss_mask_j = self.loss_mask(mask_preds_this_level, mask_targets_stuff,
                                             mask_weight_stuff, avg_factor=num_total_pos_stuff)
            loss_mask_stuff_list.append(loss_mask_j)

        # ── Step 8: 计算掩码解码器各层的分类损失 ───────────────────────
        loss_cls_thing_list = []
        loss_cls_stuff_list = []
        thing_labels = thing_labels.reshape(-1)

        for j in range(len(mask_preds_inter_things)):
            # 注意：此处 Things 分类损失也被乘以 *0（清零）
            # 原因：只有部分 query 被送入掩码解码器（query filter 导致不平衡），监督会造成偏差
            cls_scores_j = cls_thing_preds[j]
            if cls_scores_j.shape[0] == 0:
                loss_cls_thing_j = cls_scores_j.sum() * 0
            else:
                loss_cls_thing_j = self.loss_cls(
                    cls_scores_j, thing_labels, things_weights,
                    avg_factor=num_total_pos_thing) * 2 * 0  # 清零
            loss_cls_thing_list.append(loss_cls_thing_j)

        for j in range(len(mask_preds_inter_stuff)):
            cls_scores_j = cls_stuff_preds[j]
            if cls_scores_j.shape[0] == 0:
                loss_cls_stuff_j = cls_stuff_preds[j].sum() * 0
            else:
                # Stuff 分类损失保留（有效监督）
                loss_cls_stuff_j = self.loss_cls(
                    cls_stuff_preds[j], stuff_labels.to(torch.long),
                    avg_factor=num_total_pos_stuff) * 2
            loss_cls_stuff_list.append(loss_cls_stuff_j)

        # ── Step 9: 动态计算 Things/Stuff 损失权重 ──────────────────────
        # 根据 GT 数量比例自动调整权重，避免 Things/Stuff 样本不平衡时某类损失主导
        # things_ratio = n_things / (n_things + n_stuff)
        # stuff_ratio  = n_stuff  / (n_things + n_stuff)
        things_ratio, stuff_ratio = \
            num_total_pos_thing / (num_total_pos_stuff + num_total_pos_thing), \
            num_total_pos_stuff / (num_total_pos_stuff + num_total_pos_thing)

        return (loss_cls, loss_bbox, loss_iou,
                loss_mask_things, loss_mask_stuff,
                loss_mask_things_list, loss_mask_stuff_list,
                loss_iou_list, loss_bbox_list,
                loss_cls_thing_list, loss_cls_stuff_list,
                things_ratio, stuff_ratio)

    def forward_test(self,
                     pts_feats=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     img_metas=None,
                     rescale=False):
        """
        测试（推理）阶段的前向传播。

        步骤：
          1. 调用 self()（即 forward）得到位置解码结果
          2. 调用 get_bboxes 完成掩码解码和全景后处理
          3. 计算各类 IoU 指标（drivable / lanes / divider / crossing / contour）

        Args:
            pts_feats (Tensor): BEV 特征图，shape (bev_h*bev_w, bs, embed_dims)
            gt_lane_labels (list): GT 车道线类别标签（用于计算评估指标）
            gt_lane_masks (list): GT 车道线掩码（用于计算评估指标）
            img_metas (list[dict]): 图像元信息
            rescale (bool): 是否将检测框缩放回原始图像尺寸

        Returns:
            list[dict]: 每张图像的预测结果字典，包含：
                - 'pts_bbox': 预测的边界框和分割结果
                - 'ret_iou': 各类别 IoU 指标
                - 'args_tuple': 掩码解码上下文（传递给下游 Motion Head）
        """
        bbox_list = [dict() for i in range(len(img_metas))]

        # 前向推理（位置解码器）
        pred_seg_dict = self(pts_feats)

        # 掩码解码 + 全景后处理（NMS-free，基于置信度阈值和重叠阈值过滤）
        results = self.get_bboxes(
            pred_seg_dict['outputs_classes'],
            pred_seg_dict['outputs_coords'],
            pred_seg_dict['enc_outputs_class'],
            pred_seg_dict['enc_outputs_coord'],
            pred_seg_dict['args_tuple'],
            pred_seg_dict['reference'],
            img_metas,
            rescale=rescale)

        # 计算评估指标（不参与梯度，使用 no_grad 节省显存）
        with torch.no_grad():
            # 可行驶区域 IoU
            drivable_pred = results[0]['drivable']
            drivable_gt = gt_lane_masks[0][0, -1]  # GT 的最后一通道是 drivable
            drivable_iou, drivable_intersection, drivable_union = IOU(
                drivable_pred.view(1, -1), drivable_gt.view(1, -1))

            # 车道线整体 IoU（所有类别合并）
            lane_pred = results[0]['lane']
            lanes_pred = (results[0]['lane'].sum(0) > 0).int()
            lanes_gt = (gt_lane_masks[0][0][:-1].sum(0) > 0).int()
            lanes_iou, lanes_intersection, lanes_union = IOU(
                lanes_pred.view(1, -1), lanes_gt.view(1, -1))

            # 三类车道线（divider/crossing/contour）各自的 IoU
            divider_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 0].sum(0) > 0).int()
            crossing_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 1].sum(0) > 0).int()
            contour_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 2].sum(0) > 0).int()
            divider_iou, divider_intersection, divider_union = IOU(lane_pred[0].view(1, -1), divider_gt.view(1, -1))
            crossing_iou, crossing_intersection, crossing_union = IOU(lane_pred[1].view(1, -1), crossing_gt.view(1, -1))
            contour_iou, contour_intersection, contour_union = IOU(lane_pred[2].view(1, -1), contour_gt.view(1, -1))

            ret_iou = {
                'drivable_intersection': drivable_intersection,
                'drivable_union': drivable_union,
                'lanes_intersection': lanes_intersection,
                'lanes_union': lanes_union,
                'divider_intersection': divider_intersection,
                'divider_union': divider_union,
                'crossing_intersection': crossing_intersection,
                'crossing_union': crossing_union,
                'contour_intersection': contour_intersection,
                'contour_union': contour_union,
                'drivable_iou': drivable_iou,
                'lanes_iou': lanes_iou,
                'divider_iou': divider_iou,
                'crossing_iou': crossing_iou,
                'contour_iou': contour_iou,
            }

        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou'] = ret_iou
            result_dict['args_tuple'] = pred_seg_dict['args_tuple']  # 传递给下游 Motion Head

        return bbox_list

    @auto_fp16(apply_to=("bev_feat", "prev_bev"))
    def forward_train(self,
                      bev_feat=None,
                      img_metas=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      ):
        """
        训练阶段的前向传播。

        步骤：
          1. 调用 self()（即 forward）执行位置解码
          2. 组装 loss 所需的输入
          3. 调用 self.loss() 计算全部损失

        Args:
            bev_feat (Tensor): BEV 特征图，shape (bev_h*bev_w, bs, embed_dims)
            img_metas (list[dict]): 图像元信息
            gt_lane_labels (list[Tensor]): GT 车道线类别标签
            gt_lane_bboxes (list[Tensor]): GT 车道线边界框
            gt_lane_masks (list[Tensor]): GT 车道线像素级掩码

        Returns:
            tuple:
                - losses_seg (dict): 所有损失项（cls/bbox/iou/mask）
                - pred_seg_dict (dict): 前向推理输出字典（供下游模块使用）
        """
        # 数值安全保护：其他子头（motion/planning）的梯度爆炸会污染共享 BEV 特征，
        # 导致 seg_head 的 bbox_pred/cls_score 含有 NaN，进而使匈牙利匹配崩溃。
        # 在 BEV 特征入口处截断 NaN，保证本子头的 forward 不受其他子头梯度影响。
        if bev_feat is not None:
            bev_feat = torch.nan_to_num(bev_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        # 位置解码前向传播
        pred_seg_dict = self(bev_feat)

        # 组装损失计算所需的所有输入
        loss_inputs = [
            pred_seg_dict['outputs_classes'],   # 各层分类分数
            pred_seg_dict['outputs_coords'],    # 各层边界框预测
            pred_seg_dict['enc_outputs_class'], # encoder 分类分数（两阶段时）
            pred_seg_dict['enc_outputs_coord'], # encoder 坐标预测（两阶段时）
            pred_seg_dict['args_tuple'],        # 掩码解码上下文
            pred_seg_dict['reference'],         # 最后层参考点
            gt_lane_labels,
            gt_lane_bboxes,
            gt_lane_masks,
        ]
        losses_seg = self.loss(*loss_inputs, img_metas=img_metas)

        return losses_seg, pred_seg_dict

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """
        对单张图像进行目标检测后处理（推理时使用）。

        步骤：
          1. 对分类分数取 sigmoid，选取 top-k 得分最高的 query
          2. 将边界框从 cxcywh 格式转换为 xyxy 格式，并反归一化到像素坐标
          3. 将置信度分数拼接到边界框最后一维

        Args:
            cls_score (Tensor): 单张图的分类分数 [num_query, num_cls]
            bbox_pred (Tensor): 单张图的归一化边界框 [num_query, 4]（cxcywh）
            img_shape (tuple): 图像尺寸 (H, W, 3)
            scale_factor: 缩放因子（用于 rescale）
            rescale (bool): 是否缩放回原图尺寸

        Returns:
            tuple: (bbox_index, det_bboxes, det_labels)
                - bbox_index: 被选中的 query 索引（传递给掩码解码器）
                - det_bboxes: 检测框 [max_per_img, 5]（xyxy + score）
                - det_labels: 类别标签 [max_per_img]
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)  # 最大保留目标数

        # ── 基于 sigmoid 的 top-k 筛选（Focal Loss 模式）──────────────
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            # 展开为一维，选 top-k 分数
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            # 类别索引 = 全局索引 % 类别数
            det_labels = indexes % self.num_things_classes
            # query 索引 = 全局索引 // 类别数
            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            # softmax 模式（非 Focal Loss）
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        # ── 坐标反归一化（cxcywh → xyxy → 像素坐标）──────────────────
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]  # x 坐标 × 图像宽
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]  # y 坐标 × 图像高
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])        # 截断到图像边界内
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)

        # 将 score 拼接到 bbox 最后一维 → [x1, y1, x2, y2, score]
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return bbox_index, det_bboxes, det_labels

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list', 'args_tuple'))
    def get_bboxes(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        img_metas,
        rescale=False,
    ):
        """
        推理阶段的全景分割后处理（掩码解码 + 全景合并）。

        这是推理阶段与 loss_single_panoptic 对应的函数，核心逻辑：
          1. 用 _get_bboxes_single 获取 top-k 的 Things Query 和检测框
          2. 将选中的 query 送入 things_mask_head 和 stuff_mask_head 得到掩码
          3. 基于置信度阈值和重叠阈值完成全景合并（NMS-free）
          4. 提取可行驶区域（drivable）和各类型车道线（lane）分割图

        Args:
            all_cls_scores (Tensor): 各层分类分数 [nb_dec, bs, num_query, num_cls]
            all_bbox_preds (Tensor): 各层边界框预测 [nb_dec, bs, num_query, 4]
            enc_cls_scores, enc_bbox_preds: 两阶段输出（None if 单阶段）
            args_tuple: 掩码解码上下文
            reference: 参考点
            img_metas: 图像元信息
            rescale (bool): 是否缩放回原图

        Returns:
            list[dict]: 每张图的结果字典，包含：
                - 'bbox': Things 检测框 [N, 5]（xyxy + score）
                - 'segm': Things 实例掩码 [N, H, W]（bool）
                - 'labels': Things 类别标签 [N]
                - 'panoptic': 全景分割结果（numpy array），格式 (H, W, 2)
                    - channel 0：类别 ID
                    - channel 1：实例 ID（Things 有，Stuff 无）
                - 'drivable': 可行驶区域掩码（bool，H×W）
                - 'lane': 各类型车道线掩码 [num_things_classes, H, W]（int）
                - 'lane_score': 各类型车道线置信度图 [num_things_classes, H, W]
                - 'score_list': 所有类别的掩码得分图
                - 'stuff_score_list': Stuff 类别得分
        """
        # 取位置解码器最后一层的输出
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        # ── 初始化结果收集列表 ──────────────────────────────────────────
        seg_list = []           # Things 实例掩码
        stuff_score_list = []   # Stuff 类别得分
        panoptic_list = []      # 全景分割图（H, W, 2）
        bbox_list = []          # Things 检测框
        labels_list = []        # Things 类别标签
        drivable_list = []      # 可行驶区域掩码
        lane_list = []          # 各类型车道线掩码
        lane_score_list = []    # 各类型车道线置信度
        score_list = []         # 所有掩码的得分图

        # ── 逐图处理 ────────────────────────────────────────────────────
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]

            # 使用 canvas_size 作为固定分辨率（BEV 坐标系）
            img_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            ori_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            scale_factor = 1

            # ── Step 1: 获取 top-k Things Query ──────────────────────
            # index: 被选中的 query 索引，bbox: 检测框，labels: 类别
            index, bbox, labels = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale)

            i = img_id
            # 提取被选中的 Things query 特征和位置编码
            thing_query = query[i:i + 1, index, :]           # (1, top_k, embed_dims)
            thing_query_pos = query_pos[i:i + 1, index, :]

            # 将 Things query 和 Stuff query 拼接（联合解码）
            # joint_query 前半段 = Things query，后半段 = Stuff content query
            joint_query = torch.cat([
                thing_query,
                self.stuff_query.weight[None, :, :self.embed_dims]  # Stuff content
            ], 1)
            stuff_query_pos = self.stuff_query.weight[None, :, self.embed_dims:]  # Stuff position

            # ── Step 2: 掩码解码 ──────────────────────────────────────
            # Things 掩码解码（joint_query 前半段 = Things query）
            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[i:i + 1], memory_mask[i:i + 1], None,
                joint_query[:, :-self.num_stuff_classes], None, None, hw_lvl=hw_lvl)

            # Stuff 掩码解码（joint_query 后半段 = Stuff query）
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[i:i + 1], memory_mask[i:i + 1], None,
                joint_query[:, -self.num_stuff_classes:], None, stuff_query_pos, hw_lvl=hw_lvl)

            # 将 Things 和 Stuff 掩码在 query 维度拼接
            attn_map = torch.cat([mask_things, mask_stuff], 1)
            attn_map = attn_map.squeeze(-1)  # (1, total_queries, H*W)

            # Stuff 最后层 query 特征 → 分类分数（有/无该 stuff 类别）
            stuff_query_last = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1](stuff_query_last).sigmoid().reshape(-1)

            # reshape 掩码到 (total_queries, H, W) 格式
            mask_pred = attn_map.reshape(-1, *hw_lvl[0])

            # 上采样到输出分辨率（canvas_size）
            mask_pred = F.interpolate(mask_pred.unsqueeze(0), size=ori_shape[:2], mode='bilinear').squeeze(0)

            # ── Step 3: 全景后处理 ────────────────────────────────────
            masks_all = mask_pred
            score_list.append(masks_all)  # 保存所有类别掩码得分（传给下游）

            # 最后一个掩码通道是可行驶区域（drivable area）
            drivable_list.append(masks_all[-1] > 0.5)  # 二值化

            # 去掉最后 num_stuff_classes 个掩码（即 Stuff），只保留 Things 掩码
            masks_all = masks_all[:-self.num_stuff_classes]

            # 二值化掩码
            seg_all = masks_all > 0.5
            sum_seg_all = seg_all.sum((1, 2)).float() + 1  # 每个 mask 的像素数（+1 防止除零）

            # 使用检测框 score 作为基础分数
            scores_all = bbox[:, -1]   # shape: (top_k,)
            bboxes_all = bbox
            labels_all = labels

            # ── 掩码加权：score *= mask_score^2 ─────────────────────
            # mask_score = 正样本像素平均得分（衡量掩码质量）
            # 乘以平方是对低质量掩码的惩罚
            seg_scores = (masks_all * seg_all.float()).sum((1, 2)) / sum_seg_all
            scores_all *= (seg_scores ** 2)

            # 按最终分数降序排列（高分优先）
            scores_all, index = torch.sort(scores_all, descending=True)
            masks_all = masks_all[index]
            labels_all = labels_all[index]
            bboxes_all = bboxes_all[index]
            seg_all = seg_all[index]
            bboxes_all[:, -1] = scores_all  # 更新排序后的分数

            # ── 分离 Things 和 Stuff（只保留 Things 用于实例分割）────
            things_selected = labels_all < self.num_things_classes
            stuff_selected = labels_all >= self.num_things_classes
            bbox_th = bboxes_all[things_selected][:100]   # 最多保留 100 个 Things 实例
            labels_th = labels_all[things_selected][:100]
            seg_th = seg_all[things_selected][:100]
            labels_st = labels_all[stuff_selected]
            scores_st = scores_all[stuff_selected]
            masks_st = masks_all[stuff_selected]

            stuff_score_list.append(scores_st)

            # ── 全景合并：按分数由高到低依次"填入"全景图 ─────────────
            # results[0]: 类别 ID 图（0=背景）
            # results[1]: 实例 ID 图（0=背景，Things 有唯一 ID）
            results = torch.zeros((2, *mask_pred.shape[-2:]), device=mask_pred.device).to(torch.long)
            id_unique = 1  # 实例 ID 从 1 开始递增

            # lane 和 lane_score：各类型车道线的二值掩码和置信度图
            lane = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]),
                               device=mask_pred.device).to(torch.long)
            lane_score = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]),
                                     device=mask_pred.device).to(mask_pred.dtype)

            for i, scores in enumerate(scores_all):
                # ── 置信度阈值过滤 ────────────────────────────────────
                if labels_all[i] < self.num_things_classes and scores < self.quality_threshold_things:
                    continue
                elif labels_all[i] >= self.num_things_classes and scores < self.quality_threshold_stuff:
                    continue

                _mask = masks_all[i] > 0.5   # 当前实例/类别的二值掩码
                mask_area = _mask.sum().item()
                # 与已填入像素的重叠面积
                intersect = _mask & (results[0] > 0)
                intersect_area = intersect.sum().item()

                # ── 重叠阈值过滤（遮挡过多则跳过）───────────────────
                if labels_all[i] < self.num_things_classes:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area) > self.overlap_threshold_things:
                        continue
                else:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area) > self.overlap_threshold_stuff:
                        continue

                # 若有重叠，只填入尚未被占用的像素
                if intersect_area > 0:
                    _mask = _mask & (results[0] == 0)

                # 填入类别 ID
                results[0, _mask] = labels_all[i]

                # Things 类别：填入实例 ID 和车道线掩码
                if labels_all[i] < self.num_things_classes:
                    lane[labels_all[i], _mask] = 1
                    lane_score[labels_all[i], _mask] = masks_all[i][_mask]
                    results[1, _mask] = id_unique  # 每个 Things 实例有唯一 ID
                    id_unique += 1

            # 整理全景图（转换为 numpy，格式 (H, W, 2)）
            file_name = img_metas[img_id]['pts_filename'].split('/')[-1].split('.')[0]
            panoptic_list.append(
                (results.permute(1, 2, 0).cpu().numpy(), file_name, ori_shape))

            bbox_list.append(bbox_th)
            labels_list.append(labels_th)
            seg_list.append(seg_th)
            lane_list.append(lane)
            lane_score_list.append(lane_score)

        # ── 组装最终结果列表 ──────────────────────────────────────────
        results = []
        for i in range(len(img_metas)):
            results.append({
                'bbox': bbox_list[i],               # Things 检测框 [N, 5]
                'segm': seg_list[i],                # Things 实例掩码 [N, H, W]
                'labels': labels_list[i],           # Things 类别标签 [N]
                'panoptic': panoptic_list[i],        # 全景分割图 (H, W, 2)
                'drivable': drivable_list[i],        # 可行驶区域掩码 (H, W)
                'score_list': score_list[i],         # 所有类别掩码得分
                'lane': lane_list[i],                # 车道线掩码 [num_things_cls, H, W]
                'lane_score': lane_score_list[i],    # 车道线置信度图
                'stuff_score_list': stuff_score_list[i],  # Stuff 得分
            })
        return results