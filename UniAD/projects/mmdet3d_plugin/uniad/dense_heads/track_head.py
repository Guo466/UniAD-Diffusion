#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# Modified from bevformer (https://github.com/fundamentalvision/BEVFormer)        #
#---------------------------------------------------------------------------------#

# ================================================================================
# 【文件说明】track_head.py —— BEVFormer 跟踪检测头
#
# 【在 UniAD 中的地位】
# 本文件是 UniAD Stage1（感知层）的核心组件之一，负责：
#   1. 从多摄像头图像特征生成 BEV（鸟瞰图）特征图
#   2. 基于 BEV 特征，用 Query 解码出 3D 检测框 + 历史/未来轨迹
#   3. 计算训练损失（匈牙利匹配 + Focal Loss + L1 Loss）
#   4. 推理时解码预测框，供 uniad_track.py 的 QIM 跟踪模块使用
#
# 【与 BEVFormerHead（bevformer_head.py）的区别】
# ┌─────────────────┬──────────────────────────────┬──────────────────────────────────┐
# │                 │ BEVFormerHead（独立检测）      │ BEVFormerTrackHead（跟踪感知）     │
# ├─────────────────┼──────────────────────────────┼──────────────────────────────────┤
# │ 使用场景         │ bevformer.py（独立检测器）      │ uniad_track.py（端到端系统）       │
# │ BEV生成          │ forward() 内部调用            │ get_bev_features() 独立接口       │
# │ 检测解码          │ forward() 内部调用            │ get_detections() 独立接口         │
# │ 额外输出         │ 无                            │ 轨迹预测 + Query特征 + 参考点       │
# │ 调用方式         │ 外部直接调用 forward()         │ uniad_track.py 分开调用两个接口     │
# └─────────────────┴──────────────────────────────┴──────────────────────────────────┘
#
# 【被调用的上层文件】
# uniad_track.py → track_head.get_bev_features() + track_head.get_detections()
#
# 【关键概念：code_size = 10】
# 每个 3D 检测框用 10 个数描述：
#   [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
#    中心x 中心y 中心z 宽  长   高   朝向sin  朝向cos 速度x 速度y
# ================================================================================

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
# Linear：带初始化的全连接层（等同于 nn.Linear，但初始化方式不同）
# bias_init_with_prob：根据期望正样本概率计算偏置初始值（用于 Focal Loss 的初始化技巧）

from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
# multi_apply：对 batch 中的每个样本并行调用同一函数（类似 map）
# reduce_mean：多 GPU 训练时，跨 GPU 同步并求均值（分布式训练用）

from mmdet.models.utils.transformer import inverse_sigmoid
# inverse_sigmoid：sigmoid 的反函数，即 log(x/(1-x))
# 用途：在 logit（对数概率）空间中做残差相加，再 sigmoid 映射回 [0,1]

from mmdet.models import HEADS
# HEADS：mmdet 的检测头注册表，用于从配置文件自动创建检测头实例

from mmdet.models.dense_heads import DETRHead
# DETRHead：DETR 检测头基类
# 提供了：assigner（匈牙利匹配器）、sampler（正负样本采样器）
#         loss_cls（分类损失）、loss_bbox（回归损失）等通用组件
# BEVFormerTrackHead 继承它，复用这些通用能力

from mmdet3d.core.bbox.coders import build_bbox_coder
# build_bbox_coder：根据配置字典构建 bbox 编解码器
# 编码：将真实世界坐标（米）→ 归一化坐标 [0,1]（训练用）
# 解码：将归一化坐标 → 真实世界坐标（推理用）

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
# normalize_bbox：将 3D 框坐标归一化到 pc_range 范围内的 [0,1]
# 例如：x=10m, pc_range_x=[-51.2, 51.2] → x_norm = (10-(-51.2))/(51.2-(-51.2)) ≈ 0.598

from mmcv.runner import force_fp32, auto_fp16
# force_fp32：强制用 float32 精度（防止 fp16 精度不够）
# auto_fp16：自动将指定参数转为 float16


@HEADS.register_module()
# 注册到检测头注册表，配置文件中 type='BEVFormerTrackHead' 时自动找到此类
class BEVFormerTrackHead(DETRHead):
    """UniAD 的 BEVFormer 跟踪检测头。

    【核心职责】
    与独立 BEVFormerHead 的区别在于：将前向传播拆成两个独立接口：
      1. get_bev_features()：只生成 BEV 特征图（供 uniad_track.py 共享使用）
      2. get_detections()：基于 BEV 特征，解码 3D 检测框 + 轨迹预测

    这种拆分的好处：uniad_track.py 可以先生成 BEV 特征，
    再对活跃跟踪对象（Active Instances）和新生对象（Birth Instances）
    分别调用 get_detections()，实现灵活的跟踪框架。

    【额外能力（相比 BEVFormerHead）】
    新增 past_traj_reg_branch：预测每个 Query 对应目标的历史+未来轨迹
    输出 last_ref_points：解码后目标的归一化参考点，供下一帧跟踪用
    输出 query_feats：Decoder 各层的 Query 特征，供 QIM 交互模块使用

    Args:
        with_box_refine (bool): 是否使用迭代框优化（每层 Decoder 都更新参考点）
        as_two_stage (bool): 是否两阶段模式（UniAD 不使用，固定为 False）
        transformer: Transformer 配置，包含 Encoder 和 Decoder
        bbox_coder: 3D 框编解码器配置
        num_cls_fcs (int): 分类分支的全连接层数
        code_weights (list): 各维度回归损失的权重（长度=code_size=10）
        bev_h, bev_w (int): BEV 特征图的高度和宽度（格子数）
        past_steps (int): 历史轨迹步数（默认4步）
        fut_steps (int): 未来轨迹步数（默认4步）
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,     # 是否迭代细化参考点（每层 Decoder 更新）
                 as_two_stage=False,         # 是否两阶段（UniAD 不用）
                 transformer=None,           # Transformer 配置字典
                 bbox_coder=None,            # 3D 框编解码器配置
                 num_cls_fcs=2,              # 分类分支 FC 层数
                 code_weights=None,          # 各维度的损失权重
                 bev_h=30,                   # BEV 特征图高度（行数/格子数）
                 bev_w=30,                   # BEV 特征图宽度（列数/格子数）
                 past_steps=4,               # 轨迹预测的历史步数（过去几帧）
                 fut_steps=4,                # 轨迹预测的未来步数（预测未来几帧）
                 **kwargs):

        # ── 在调用父类初始化前，先设置必须的属性 ──────────────────────────
        # 原因：父类 __init__ 中会调用 _init_layers()，而 _init_layers 需要这些属性
        self.bev_h = bev_h   # BEV 网格行数，例如 200（表示200×200的鸟瞰格子）
        self.bev_w = bev_w   # BEV 网格列数
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        # with_box_refine=True：每层 Decoder 都有独立的 reg_branch（独立参数，不共享）
        # with_box_refine=False：所有 Decoder 层共享同一套 reg_branch 参数

        assert as_two_stage is False, 'as_two_stage is not supported yet.'
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        # code_size：3D 检测框的参数维度 = 10
        # 含义：[cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10

        # code_weights：各维度回归损失的权重
        # 默认：x,y,z,w,l,h,sin,cos 权重为 1.0，速度 vx,vy 权重为 0.2
        # 速度权重小是因为速度估计难度更大、噪声更多，避免速度损失主导训练
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            #                    cx   cy   cz   w    l    h   sinθ cosθ  vx   vy

        # 构建 3D 框编解码器（根据配置字典找到对应类并实例化）
        self.bbox_coder = build_bbox_coder(bbox_coder)

        # pc_range：点云感知范围，格式 [x_min, y_min, z_min, x_max, y_max, z_max]（米）
        # 例如 [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]  # BEV 实际宽度（米），如 102.4m
        self.real_h = self.pc_range[4] - self.pc_range[1]  # BEV 实际高度（米），如 102.4m

        self.num_cls_fcs = num_cls_fcs - 1  # 分类分支中间层数（不含最后输出层）
        self.past_steps = past_steps  # 历史轨迹步数
        self.fut_steps = fut_steps    # 未来轨迹步数

        # 调用父类初始化（DETRHead → BaseDetHead）
        # 父类会创建：assigner（匈牙利匹配器）、sampler、loss_cls、loss_bbox 等
        # 同时调用 _init_layers() 创建神经网络层
        super(BEVFormerTrackHead, self).__init__(
            *args, transformer=transformer, **kwargs)

        # 将 code_weights 存为不可训练的参数（nn.Parameter 保证它跟随模型保存/加载）
        # requires_grad=False：不更新这些权重，它们是固定的超参数
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self):
        """初始化检测头的所有神经网络层。

        创建三套分支，每套都有 num_pred 个副本（对应 Decoder 的每一层）：
          1. cls_branches：分类分支（Embed → 类别 logit）
          2. reg_branches：回归分支（Embed → 10维框参数）
          3. past_traj_reg_branches：轨迹预测分支（Embed → (past+fut)×2 轨迹点）

        【深监督机制】
        Decoder 有 num_pred 层，每层都有独立的预测头，每层都计算损失。
        这样即使是浅层的 Decoder，也能得到梯度反传，加速收敛。
        """
        # ── 分类分支：MLP，输出每个 Query 对应的类别 logit ──────────────
        # 结构：Linear → LayerNorm → ReLU → ... → Linear（输出层）
        # LayerNorm：对每个样本的特征维度做归一化，比 BatchNorm 更适合 Transformer
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        # cls_out_channels = num_classes（分类数，nuScenes 为 10 类）
        fc_cls = nn.Sequential(*cls_branch)

        # ── 回归分支：MLP，输出 code_size=10 维的框参数 ──────────────────
        # 结构：Linear → ReLU → ... → Linear（输出 10 维）
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # ── 轨迹预测分支：MLP，预测历史+未来轨迹 ────────────────────────
        # 输出维度：(past_steps + fut_steps) * 2
        # 含义：每个时间步预测 (x偏移, y偏移)，共 (4+4)*2=16 维
        # 注意：这里是相对于当前位置的偏移量（不是绝对坐标）
        past_traj_reg_branch = []
        for _ in range(self.num_reg_fcs):
            past_traj_reg_branch.append(
                Linear(self.embed_dims, self.embed_dims))
            past_traj_reg_branch.append(nn.ReLU())
        past_traj_reg_branch.append(
            Linear(self.embed_dims, (self.past_steps + self.fut_steps)*2))
        past_traj_reg_branch = nn.Sequential(*past_traj_reg_branch)

        def _get_clones(module, N):
            """创建 N 个模块的深拷贝，参数独立（用于 with_box_refine=True 模式）"""
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # ── 确定 Decoder 层数，决定需要几套预测分支 ──────────────────────
        # 单阶段模式：num_pred = Decoder 层数（通常 6 层）
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            # 迭代细化模式：每层 Decoder 用独立参数（完全独立，互不影响）
            # 这样每层可以专注优化自己的预测，层与层之间的改进是真正的"细化"
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.past_traj_reg_branches = _get_clones(
                past_traj_reg_branch, num_pred)
        else:
            # 共享参数模式：所有层共享同一套参数
            # 优点：参数量少，不易过拟合；缺点：每层学到的特征类似
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.past_traj_reg_branches = nn.ModuleList(
                [past_traj_reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            # bev_embedding：BEV 查询的可学习 Embedding
            # 共 bev_h * bev_w 个位置（如 200×200=40000 个 BEV 网格）
            # 每个位置有 embed_dims 维的特征向量
            # 作用：作为 BEV Encoder 的输入 Query，"提问"相机图像获取对应特征
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """初始化模型权重。

        【Focal Loss 偏置初始化技巧】
        分类分支输出层的偏置用 bias_init_with_prob(0.01) 初始化，
        对应每个 Query 初始时预测为正样本的概率约为 1%。

        这个技巧来自 Focal Loss 论文，目的是：
        训练初期，大多数 Query 预测为背景（概率接近 0），
        前景类别的梯度不会被大量背景的梯度"淹没"，
        避免训练初期损失爆炸，加速收敛。
        """
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            # bias_init_with_prob(p)：计算 log(p/(1-p))，即 sigmoid^{-1}(p)
            # p=0.01 → bias ≈ -4.6，使初始分类概率约为 1%
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
        """【接口1】从多尺度图像特征生成 BEV 特征图。

        这是 BEVFormerTrackHead 的第一个核心接口，由 uniad_track.py 调用。
        负责将多相机图像的 2D 特征"升维"到 BEV 的 2D 顶视图特征。

        【工作原理（空间交叉注意力）】
        1. 准备 BEV Query：bev_h×bev_w 个可学习向量，每个对应一个俯视格子
        2. 为每个 BEV 格子生成参考点（在 BEV 空间中的 3D 坐标）
        3. 把参考点投影到每个相机的图像平面（用相机内外参矩阵）
        4. 在投影点周围做变形卷积注意力（Deformable Attention），提取图像特征
        5. 融合所有相机的特征 → 得到该 BEV 格子的特征向量
        6. (可选) 用时序注意力融合上一帧的 BEV 特征（prev_bev）

        Args:
            mlvl_feats (list[Tensor]): 多尺度图像特征（FPN 输出）
                每个 Tensor 的 shape: [B, N_cam, C, H_i, W_i]
                N_cam=6（nuScenes 有6个摄像头：前、前左、前右、后、后左、后右）
            img_metas (list[dict]): 图像元信息，包含相机内外参矩阵
                关键字段：'lidar2img'（从激光雷达坐标系到图像坐标系的变换矩阵）
            prev_bev (Tensor, optional): 上一帧的 BEV 特征
                shape: [B, bev_h*bev_w, C]，用于时序融合

        Returns:
            tuple:
                - bev_embed (Tensor): BEV 特征图，shape [bev_h*bev_w, B, C]
                - bev_pos (Tensor): BEV 位置编码，shape [B, C, bev_h, bev_w]
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        # bev_queries：BEV 网格的可学习 Embedding
        # shape: [bev_h*bev_w, embed_dims]
        # 这 bev_h*bev_w 个向量就是 Transformer Encoder 的 Query（Q）
        # 它们向相机图像特征（作为 K 和 V）"提问"
        bev_queries = self.bev_embedding.weight.to(dtype)

        # bev_mask：全零掩码，用于生成位置编码（sin/cos 位置编码）
        # shape: [B, bev_h, bev_w]
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)

        # bev_pos：BEV 空间的正弦位置编码
        # shape: [B, C, bev_h, bev_w]
        # 作用：让每个 BEV 格子"知道"自己在 BEV 空间中的位置
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # ★ 调用 Transformer Encoder，执行空间交叉注意力
        # 具体实现在 modules/transformer.py 的 get_bev_features()
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,                              # 图像特征（K, V 来源）
            bev_queries,                             # BEV Query（Q 来源）
            self.bev_h,
            self.bev_w,
            self.real_h,                             # BEV 覆盖的真实高度（米）
            self.real_w,                             # BEV 覆盖的真实宽度（米）
            grid_length=(self.real_h / self.bev_h,   # 每个 BEV 格子对应的真实尺寸（米）
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,                         # BEV 位置编码
            prev_bev=prev_bev,                       # 上一帧 BEV（时序注意力用）
            img_metas=img_metas,                     # 相机参数（投影用）
        )
        return bev_embed, bev_pos

    def get_detections(
        self,
        bev_embed,                    # BEV 特征图（来自 get_bev_features 的输出）
        object_query_embeds=None,     # 目标检测的 Object Query（含坐标和内容两部分）
        ref_points=None,              # 参考点：每个 Query 对应的初始 3D 位置（logit 空间）
        img_metas=None,               # 图像元信息
    ):
        """【接口2】基于 BEV 特征，解码检测结果（3D 框 + 轨迹预测）。

        这是 BEVFormerTrackHead 的第二个核心接口，由 uniad_track.py 调用。
        运行 Transformer Decoder，将 Object Query 的初始状态
        通过多层交叉注意力细化，最终解码为 3D 检测框和轨迹预测。

        【工作原理（Decoder 迭代细化）】
        for lvl in range(num_decoder_layers):
            Query 做自注意力（Query 之间互相交流）
            Query 做交叉注意力（向 BEV 特征图"提问"）
            cls_branches[lvl]：分类头，输出类别 logit
            reg_branches[lvl]：回归头，输出框参数偏移量
            (若 with_box_refine) 更新参考点：新参考点 = 解码结果

        【残差解码】
        网络不直接预测绝对坐标，而是预测相对于参考点的偏移量：
          预测坐标 = sigmoid(inverse_sigmoid(参考点) + 偏移量)
        这样网络只需学"微调"，比直接预测绝对位置更容易学习。

        Args:
            bev_embed (Tensor): BEV 特征图，shape [bev_h*bev_w, B, C]
            object_query_embeds (Tensor): Query 的 Embedding（位置部分+内容部分）
                shape: [num_query, embed_dims*2]
            ref_points (Tensor): 每个 Query 的初始参考点（logit 空间的 3D 坐标）
                shape: [B, num_query, 3]，值域为 logit（无界）
            img_metas (list[dict]): 图像元信息

        Returns:
            dict: 包含以下键：
                'all_cls_scores':      [num_dec, B, num_query, num_cls]  各层分类分数
                'all_bbox_preds':      [num_dec, B, num_query, 10]       各层框预测（真实坐标）
                'all_past_traj_preds': [num_dec, B, num_query, T, 2]     各层轨迹预测
                'enc_cls_scores':      None（单阶段不使用）
                'enc_bbox_preds':      None（单阶段不使用）
                'last_ref_points':     [B, num_query, 3]  最终层的参考点（logit 空间，供下帧跟踪用）
                'query_feats':         [num_dec, B, num_query, C]  各层 Query 特征
        """
        # 确保 BEV 特征图尺寸匹配
        assert bev_embed.shape[0] == self.bev_h * self.bev_w

        # ★ 调用 Transformer Decoder
        # 返回：
        #   hs: [num_dec, num_query, B, C] 各层 Query 的隐状态
        #   init_reference: [B, num_query, 3] 初始参考点（sigmoid 空间）
        #   inter_references: [num_dec, B, num_query, 3] 各层更新后的参考点
        hs, init_reference, inter_references = self.transformer.get_states_and_refs(
            bev_embed,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            reference_points=ref_points,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )

        # hs 原始 shape: [num_dec, num_query, B, C]
        # permute → [num_dec, B, num_query, C]（更符合直觉：batch 在第二维）
        hs = hs.permute(0, 2, 1, 3)

        outputs_classes = []  # 收集各层分类结果
        outputs_coords = []   # 收集各层框坐标结果
        outputs_trajs = []    # 收集各层轨迹预测结果

        # ── 对每一层 Decoder 的输出做解码 ────────────────────────────────
        for lvl in range(hs.shape[0]):
            # ── Step 1: 获取本层的参考点（logit 空间）──────────────────
            if lvl == 0:
                # 第一层：使用初始参考点
                # ref_points 是 logit 空间，sigmoid 转为 [0,1] 归一化坐标
                reference = ref_points.sigmoid()
            else:
                # 后续层：使用上一层更新后的参考点（已在 [0,1] 空间）
                reference = inter_references[lvl - 1]

            # 转回 logit 空间，方便后续做残差相加
            reference = inverse_sigmoid(reference)
            # reference shape: [B, num_query, 3]，值域无界（logit 空间）

            # ── Step 2: 分类预测 ────────────────────────────────────────
            # cls_branches[lvl]: [B, num_query, embed_dims] → [B, num_query, num_cls]
            outputs_class = self.cls_branches[lvl](hs[lvl])
            # 输出是 logit（未经 sigmoid），后续 loss_cls 内部会做 sigmoid

            # ── Step 3: 回归预测（残差解码）────────────────────────────
            # reg_branches[lvl]: [B, num_query, embed_dims] → [B, num_query, 10]
            tmp = self.reg_branches[lvl](hs[lvl])  # 10维偏移量

            # ── Step 4: 轨迹预测 ────────────────────────────────────────
            # past_traj_reg_branches[lvl]: → [B, num_query, (past+fut)*2]
            # reshape 为 [B, num_query, past_steps+fut_steps, 2]
            # 最后一维 2 = (x偏移, y偏移)（相对于当前位置的轨迹点）
            outputs_past_traj = self.past_traj_reg_branches[lvl](hs[lvl]).view(
                tmp.shape[0], -1, self.past_steps + self.fut_steps, 2)

            assert reference.shape[-1] == 3  # 参考点必须是 3D [x, y, z]

            # ── Step 5: 残差相加 + sigmoid（xy 和 z 分别处理）─────────
            # 注意：tmp 是偏移量，reference 是参考点（都在 logit 空间）
            # logit 空间相加 → 等价于"在归一化坐标空间做乘法缩放"
            #
            # xy（水平位置）：
            tmp[..., 0:2] += reference[..., 0:2]   # logit 空间：偏移量 + 参考点
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid() # sigmoid → [0,1] 归一化坐标

            # z（高度）：注意 z 在 tmp 中是第5维（index=4），即 tmp[..., 4:5]
            # 这是因为 tmp 格式：[cx,cy,w,l,cz,h,sinθ,cosθ,vx,vy]
            #                     0  1 2 3  4  5  6    7   8   9
            tmp[..., 4:5] += reference[..., 2:3]   # logit 空间：z 偏移 + z 参考
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid() # sigmoid → [0,1]

            # ── Step 6: 保存归一化参考点（供下帧跟踪初始化用）──────────
            # last_ref_points：归一化 [0,1] 空间的 [x, y, z]
            # 下一帧可以用它作为这些目标的初始参考点，实现跨帧位置传递
            last_ref_points = torch.cat(
                [tmp[..., 0:2], tmp[..., 4:5]], dim=-1,
            )
            # shape: [B, num_query, 3]

            # ── Step 7: 反归一化到真实世界坐标（米）────────────────────
            # 归一化坐标 [0,1] → 真实世界坐标（米）
            # 公式：x_real = x_norm * (x_max - x_min) + x_min
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])   # x: [-51.2, 51.2]
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])   # y: [-51.2, 51.2]
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])   # z: [-5.0, 3.0]

            outputs_coord = tmp   # 最终 10 维框参数（真实世界坐标 + 速度）
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_trajs.append(outputs_past_traj)

        # ── 收集所有层的预测结果 ─────────────────────────────────────────
        outputs_classes = torch.stack(outputs_classes)  # [num_dec, B, num_query, num_cls]
        outputs_coords = torch.stack(outputs_coords)    # [num_dec, B, num_query, 10]
        outputs_trajs = torch.stack(outputs_trajs)      # [num_dec, B, num_query, T, 2]

        # last_ref_points 转回 logit 空间（供 uniad_track.py 的跟踪模块使用）
        # 原因：跟踪模块用这些参考点初始化下一帧的 Query，需要 logit 空间的值
        last_ref_points = inverse_sigmoid(last_ref_points)

        outs = {
            'all_cls_scores': outputs_classes,       # 各层分类分数（logit）
            'all_bbox_preds': outputs_coords,        # 各层框预测（真实坐标/米）
            'all_past_traj_preds': outputs_trajs,    # 各层轨迹预测（偏移量）
            'enc_cls_scores': None,                  # 单阶段模式不使用
            'enc_bbox_preds': None,                  # 单阶段模式不使用
            'last_ref_points': last_ref_points,      # 最终参考点（logit，跟踪用）
            'query_feats': hs,                       # 各层 Query 特征（QIM 交互用）
        }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """为单张图像计算匈牙利匹配的分配结果，得到每个 Query 对应的学习目标。

        【匈牙利匹配的作用】
        DETR-style 检测器没有 Anchor 和 NMS，Query 和 GT 之间的对应关系是动态的。
        训练时，我们需要确定"哪个 Query 负责预测哪个 GT 目标"——这就是匈牙利匹配。

        匹配标准（二分图最小代价匹配）：
          代价 = α × 分类 Focal Loss + β × L1 距离
        最优匹配：使总代价最小的一对一分配方案

        匹配结果：
          - 正样本 Query（pos_inds）：与某个 GT 匹配，负责学习检测那个目标
          - 负样本 Query（neg_inds）：未匹配，学习预测"背景"类

        Args:
            cls_score (Tensor): 单张图的分类分数 [num_query, num_cls]（logit）
            bbox_pred (Tensor): 单张图的框预测 [num_query, 10]（真实坐标）
            gt_labels (Tensor): GT 类别标签 [num_gts]
            gt_bboxes (Tensor): GT 3D 框 [num_gts, 10]
            gt_bboxes_ignore: 忽略的 GT 框（通常为 None）

        Returns:
            tuple:
                labels (Tensor): 每个 Query 的目标类别 [num_query]
                    正样本→对应GT类别，负样本→num_classes（背景类别索引）
                label_weights (Tensor): 分类损失权重 [num_query]，全为 1.0
                bbox_targets (Tensor): 每个正样本 Query 的目标框 [num_query, 10]
                bbox_weights (Tensor): 回归损失权重 [num_query, 10]
                    正样本→1.0，负样本→0.0（不计算回归损失）
                pos_inds (Tensor): 正样本 Query 的索引
                neg_inds (Tensor): 负样本 Query 的索引
        """
        num_bboxes = bbox_pred.size(0)  # num_query（通常 900）
        gt_c = gt_bboxes.shape[-1]       # GT 框的维度数

        # ── 匈牙利匹配：计算预测框与 GT 框的最优分配 ─────────────────────
        # self.assigner 是 HungarianAssigner3D（从配置文件构建）
        # 内部计算代价矩阵（分类代价 + 距离代价），用 scipy.linear_sum_assignment 求最优解
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        # ── 采样：将匹配结果转为正/负样本索引 ────────────────────────────
        # self.sampler 通常是 PseudoSampler（DETR 不做随机采样，所有匹配结果直接使用）
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds  # 正样本 Query 的索引 Tensor
        neg_inds = sampling_result.neg_inds  # 负样本 Query 的索引 Tensor

        # ── 构建分类目标（labels）──────────────────────────────────────────
        # 初始化：所有 Query 标记为背景（num_classes 是背景的特殊标签）
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,  # 背景类别索引
                                    dtype=torch.long)
        # 正样本：覆盖为对应 GT 的真实类别
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # 分类权重：所有 Query（包括背景）都参与分类损失，权重均为 1
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # ── 构建回归目标（bbox_targets）──────────────────────────────────
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        # 回归权重：只有正样本才计算回归损失
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0  # 正样本权重=1，负样本=0（背景不学习位置）

        # 正样本的目标框 = 对应 GT 框
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """为整个 batch 的图像并行计算匈牙利匹配目标。

        对 batch 中每张图像调用 _get_target_single，收集所有结果。

        Args:
            cls_scores_list (list[Tensor]): batch 中每张图的分类分数
            bbox_preds_list (list[Tensor]): batch 中每张图的框预测
            gt_bboxes_list (list[Tensor]): batch 中每张图的 GT 框
            gt_labels_list (list[Tensor]): batch 中每张图的 GT 标签
            gt_bboxes_ignore_list: 通常为 None

        Returns:
            tuple: 包含所有图像的匹配目标，以及正负样本总数
                - labels_list, label_weights_list
                - bbox_targets_list, bbox_weights_list
                - num_total_pos: 整个 batch 的正样本总数（用于归一化损失）
                - num_total_neg: 整个 batch 的负样本总数
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # multi_apply：对 batch 中每张图像并行调用 _get_target_single
        # 等同于：[_get_target_single(scores[i], preds[i], ...) for i in range(bs)]
        # 但内部做了并行化处理
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)

        # 统计整个 batch 的正/负样本总数（用于计算归一化因子）
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """计算单个 Decoder 层的损失（深监督机制的基本单元）。

        【深监督（Deep Supervision）】
        Transformer Decoder 有 N 层，每层都有预测输出。
        对每一层都计算损失并反向传播，好处：
          - 浅层 Decoder 也能得到直接的梯度信号，不用等梯度从深层"传回来"
          - 相当于多个检测器协同训练，最终只用最深层的输出做推理
          - 加速收敛，提升最终精度

        【损失计算流程】
        1. get_targets()  → 匈牙利匹配 → 正负样本索引
        2. loss_cls()     → Focal Loss（分类，所有 Query 参与）
        3. loss_bbox()    → L1 Loss（回归，只有正样本参与）

        Args:
            cls_scores (Tensor): 本层分类分数 [B, num_query, num_cls]
            bbox_preds (Tensor): 本层框预测  [B, num_query, 10]（真实坐标）
            gt_bboxes_list (list): GT 框列表（每张图一个 Tensor）
            gt_labels_list (list): GT 标签列表
            gt_bboxes_ignore_list: 忽略的 GT（通常为 None）

        Returns:
            tuple:
                loss_cls (Tensor): 分类损失（标量）
                loss_bbox (Tensor): 回归损失（标量）
        """
        num_imgs = cls_scores.size(0)  # batch size
        # 将 batch 维度拆开，变成 list（供 get_targets 的 multi_apply 使用）
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        # ── Step 1: 匈牙利匹配，获取每个 Query 的学习目标 ──────────────
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # 将 batch 中各图的目标拼接（方便后续向量化计算）
        labels = torch.cat(labels_list, 0)               # [B*num_query]
        label_weights = torch.cat(label_weights_list, 0) # [B*num_query]
        bbox_targets = torch.cat(bbox_targets_list, 0)   # [B*num_query, 10]
        bbox_weights = torch.cat(bbox_weights_list, 0)   # [B*num_query, 10]

        # ── Step 2: 分类损失（Focal Loss）──────────────────────────────
        # reshape：[B, num_query, num_cls] → [B*num_query, num_cls]
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)

        # 计算加权平均因子（分类损失的归一化系数）
        # = 正样本数 × 1.0 + 负样本数 × bg_cls_weight
        # bg_cls_weight 通常为 0（背景样本对损失的贡献降低）
        # 归一化的目的：让损失不随 batch 大小或目标数量剧烈变化
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            # 分布式训练：跨 GPU 同步并求均值（确保所有 GPU 用相同的归一化因子）
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)  # 防止分母为0
        loss_cls = self.loss_cls(
            cls_scores,       # 预测：[B*num_query, num_cls] logit
            labels,           # 目标：[B*num_query] 类别索引（正样本=类别，负样本=背景）
            label_weights,    # 权重：[B*num_query] 全为 1.0
            avg_factor=cls_avg_factor)

        # ── Step 3: 回归损失（L1 Loss）──────────────────────────────────
        # 多 GPU 时同步正样本总数（用于归一化 L1 Loss）
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # reshape：[B, num_query, 10] → [B*num_query, 10]
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))

        # 将 GT 框的真实坐标（米）归一化到 [0,1]（匹配预测框的空间）
        # normalize_bbox：x_real → (x_real - x_min) / (x_max - x_min)
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)

        # 过滤掉含有 NaN 或 Inf 的 GT 框（数据噪声防护）
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        # 应用 code_weights：对不同维度的 L1 Loss 加权
        # 速度维度权重 0.2（相对位置维度），防止速度估计误差主导训练
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],              # 预测框（已归一化坐标）
            normalized_bbox_targets[isnotnan, :10], # GT 框（已归一化坐标）
            bbox_weights[isnotnan, :10],            # 权重（正样本=code_weights，负样本=0）
            avg_factor=num_total_pos)

        # nan_to_num：将 NaN/Inf 替换为 0（极端情况的保护，防止训练崩溃）
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """汇总所有 Decoder 层的损失（深监督总入口）。

        【调用链】
        uniad_track.py
          → track_head.loss(gt_bboxes_list, gt_labels_list, preds_dicts)
              → [对每个 Decoder 层] loss_single()
                  → get_targets()（匈牙利匹配）
                  → Focal Loss + L1 Loss

        【GT 框格式转换】
        输入的 gt_bboxes 是 mmdet3d 的 LiDARInstance3DBoxes 对象，
        内部存储格式为 [x, y, z, w, l, h, yaw]（7维）。
        但损失计算需要 [cx, cy, cz, w, l, h, sinθ, cosθ, vx, vy]（10维）。
        转换方式：使用 gravity_center 取中心点 + tensor[:,3:] 取其余维度。

        Args:
            gt_bboxes_list (list): 每张图的 GT 3D 框（LiDARInstance3DBoxes 对象）
            gt_labels_list (list[Tensor]): 每张图的 GT 类别标签
            preds_dicts (dict): get_detections() 的返回值
                包含 all_cls_scores / all_bbox_preds 等

        Returns:
            dict: 损失字典，例如：
                {
                  'loss_cls': 0.3,           # 最后一层 Decoder 的分类损失（主损失）
                  'loss_bbox': 0.7,          # 最后一层 Decoder 的回归损失（主损失）
                  'd0.loss_cls': 0.5,        # 第0层 Decoder 的辅助分类损失
                  'd0.loss_bbox': 1.1,       # 第0层 Decoder 的辅助回归损失
                  'd1.loss_cls': 0.4, ...    # 以此类推
                }
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # 从预测字典中取出各层的预测结果
        all_cls_scores = preds_dicts['all_cls_scores']  # [num_dec, B, Q, cls]
        all_bbox_preds = preds_dicts['all_bbox_preds']  # [num_dec, B, Q, 10]
        enc_cls_scores = preds_dicts['enc_cls_scores']  # None（单阶段）
        enc_bbox_preds = preds_dicts['enc_bbox_preds']  # None（单阶段）

        num_dec_layers = len(all_cls_scores)  # Decoder 层数
        device = gt_labels_list[0].device

        # ── GT 框格式转换：mmdet3d 格式 → 标准 Tensor 格式 ───────────────
        # gt_bboxes.gravity_center：3D 框的重心坐标 [num_gts, 3]（cx, cy, cz）
        # gt_bboxes.tensor[:, 3:]：其余维度 [num_gts, 7]（w, l, h, sinθ, cosθ, vx, vy）
        # 拼接后：[num_gts, 10]
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        # 为每个 Decoder 层复制一份 GT（损失计算时每层独立匹配）
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # ── 对所有 Decoder 层并行计算损失 ────────────────────────────────
        # multi_apply：等同于
        # losses_cls = [], losses_bbox = []
        # for i in range(num_dec_layers):
        #     lc, lb = loss_single(all_cls_scores[i], all_bbox_preds[i], ...)
        #     losses_cls.append(lc), losses_bbox.append(lb)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # 两阶段模式的 Encoder 损失（单阶段不走此分支，enc_cls_scores=None）
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # ── 汇总损失：最后一层为主损失，其余层为辅助损失 ─────────────────
        # 最后一层预测质量最好（经过最多细化），作为主损失
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # 前 N-1 层作为辅助损失（深监督），用 d0, d1, ... 前缀区分
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """推理阶段：将预测结果解码为标准 3D 检测框。

        【与 BEVFormerHead.get_bboxes 的区别】
        返回列表中多了 bbox_index 和 mask 两个字段，
        用于 uniad_track.py 中的跟踪关联（哪些 Query 是有效检测）。

        【解码流程】
        bbox_coder.decode：
          1. 取最后一层 Decoder 的输出（质量最好）
          2. 对分类分数做 sigmoid（logit → 概率）
          3. 按置信度阈值过滤，保留高置信度的预测
          4. 归一化坐标 [0,1] → 真实世界坐标（米）
          5. 可选：按分数排序，截断到 max_num 个检测框

        【z 坐标修正】
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        解释：模型预测的 cz 是框的中心高度，但 mmdet3d 的坐标约定是框底面高度。
        转换：底面高度 = 中心高度 - 框高度/2

        Args:
            preds_dicts (dict): get_detections() 的返回值
            img_metas (list[dict]): 图像元信息
            rescale (bool): 3D 检测通常不需要缩放，默认 False

        Returns:
            list: 每个元素对应 batch 中一个样本的检测结果：
                [bboxes, scores, labels, bbox_index, mask]
                - bboxes: LiDARInstance3DBoxes 对象（3D 边界框）
                - scores: Tensor，每个框的置信度分数
                - labels: Tensor，每个框的类别索引
                - bbox_index: Tensor，有效框在所有 Query 中的原始索引（跟踪用）
                - mask: BoolTensor，标记哪些 Query 被判定为有效检测（跟踪用）
        """
        # ★ bbox_coder.decode：核心解码步骤（归一化坐标 → 真实坐标，阈值过滤等）
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)  # batch size
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']  # [num_det, 9]（或更多维度）

            # z 坐标修正：中心高度 → 底面高度
            # mmdet3d 约定 z = 底面高度（不是中心），需要减去半个高度
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            # 将原始 Tensor 包装为 mmdet3d 的 3D 框对象（LiDARInstance3DBoxes）
            # img_metas[i]['box_type_3d'] 指定框的类型（通常是 LiDARInstance3DBoxes）
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)

            scores = preds['scores']           # 置信度分数 [num_det]
            labels = preds['labels']           # 类别索引 [num_det]
            bbox_index = preds['bbox_index']   # 原始 Query 索引（跟踪用）
            mask = preds['mask']               # 有效框掩码（跟踪用）

            ret_list.append([bboxes, scores, labels, bbox_index, mask])

        return ret_list
    