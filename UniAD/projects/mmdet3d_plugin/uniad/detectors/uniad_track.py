#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# ================================================================================
# 文件概述：UniADTrack —— Stage1 检测 + 跟踪模块
#
# 这个文件实现了 UniAD 的第一阶段：BEV 特征提取 + 3D 目标检测 + 多目标跟踪（MOT）
#
# 核心功能：
#   1. BEVFormer：将 6 个相机图像融合为统一的 BEV（鸟瞰图）特征
#   2. TrackHead（BEVFormerTrackHead）：在 BEV 特征上做检测 + 跟踪
#
# 跟踪机制（DETR-Track 范式）：
#   - 每个目标用一个"Track Query"（向量）表示
#   - Track Query 跨帧持续存在，携带目标的历史信息
#   - 检测：对每帧初始化 N=900 个候选 Query，预测检测框和类别
#   - 跟踪：通过匈牙利匹配将 Query 与 GT 对应，连续匹配 = 成功跟踪
#
# 关键组件：
#   - query_embedding：N+1 个可学习 Query（900个目标Query + 1个自车Query）
#   - reference_points：每个 Query 对应的 3D 参考点（目标位置先验）
#   - memory_bank：保存历史 K 帧的 Query 特征，提供长期记忆
#   - query_interact（QIM）：Query 交互模块，更新 Query 状态（新增/删除/更新）
#   - track_base：跟踪基础模块，管理活跃目标（基于置信度阈值过滤）
#
# 特殊设计：
#   - Query 索引 900（第901个）固定用于表示自车（SDC/Ego Vehicle）
#   - velo_update：利用速度预测在帧间更新目标参考点（预测目标下一帧的位置）
#
# 与 UniAD（子类）的关系：
#   UniADTrack 负责"看见什么、谁在哪里"（检测+跟踪）
#   UniAD 在此基础上额外回答"会去哪里、怎么开"（预测+规划）
# ================================================================================

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector  # mmdet3d 通用双阶段检测基类
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask      # 网格遮挡数据增强
import copy
import math
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox  # 将检测框归一化到 [0,1]
from mmdet.models import build_loss
from einops import rearrange
from mmdet.models.utils.transformer import inverse_sigmoid  # sigmoid 的反函数，用于坐标空间转换
# 跟踪相关子模块：
from ..dense_heads.track_head_plugin import (
    MemoryBank,              # 记忆库：保存历史帧的 Query 特征
    QueryInteractionModule,  # QIM：Query 交互模块，管理 Query 的生命周期
    Instances,               # 通用实例容器（类似字典，但支持索引和切片）
    RuntimeTrackerBase,      # 运行时跟踪器：基于阈值管理活跃目标
)


@DETECTORS.register_module()
class UniADTrack(MVXTwoStageDetector):
    """UniAD 的检测和跟踪模块（Stage 1）。
    
    继承 MVXTwoStageDetector，在其基础上添加了：
      - 时序跟踪机制（Track Query 跨帧传递）
      - 速度感知的参考点更新（velo_update）
      - 记忆库（MemoryBank）
      - Query 交互模块（QIM）
    
    核心思想：DETR-Track
      传统检测器每帧独立预测，不知道目标的跨帧对应关系。
      DETR-Track 让每个目标对应一个持久的 Query，
      这个 Query 从第一帧开始就跟随目标，不断被更新，
      实现"我一直在跟踪这辆车"的效果。
    
    Args:
        use_grid_mask (bool): 是否使用 GridMask 数据增强（随机遮挡图像区域，防止过拟合）
        img_backbone: 图像特征提取网络（如 ResNet）
        img_neck: 特征金字塔网络（如 FPN，融合多尺度特征）
        pts_bbox_head: BEVFormerTrackHead（BEV特征生成 + DETR检测头）
        video_test_mode (bool): 是否使用视频时序推理模式（必须为 True）
        loss_cfg: 跟踪损失函数配置（匈牙利匹配 + 分类/回归损失）
        qim_args: Query 交互模块参数
            fp_ratio: 假阳性（误检）Query 的保留比例
            random_drop: 随机丢弃 Query 的概率（防止过拟合）
        mem_args: 记忆库参数
            memory_bank_len: 保存历史帧数（默认4帧）
        bbox_coder: 检测框编解码器配置
        pc_range: 点云范围（激光雷达坐标系，±51.2m）
        embed_dims (int): Query 特征维度（256）
        num_query (int): 候选 Query 数量（900，不含自车 Query）
        num_classes (int): 目标类别数（10类）
        score_thresh (float): 跟踪置信度阈值（低于此值的目标停止跟踪）
        filter_score_thresh (float): 过滤置信度阈值（低于此值的目标不参与后续任务）
        miss_tolerance (int): 连续消失帧数容忍度（超过此值则删除该目标）
        gt_iou_threshold (float): 训练时筛选有效目标的 IoU 阈值
        freeze_img_backbone (bool): 是否冻结图像骨干网络（Stage2 微调时用）
        freeze_bev_encoder (bool): 是否冻结 BEV 编码器
        queue_length (int): 时序队列长度（使用多少帧历史，默认3帧）
    """
    def __init__(
        self, 
        use_grid_mask=False,       # 是否使用 GridMask 增强
        img_backbone=None,         # 图像骨干网络（ResNet 等）
        img_neck=None,             # 特征金字塔（FPN 等）
        pts_bbox_head=None,        # BEVFormerTrackHead（核心模块）
        train_cfg=None,
        test_cfg=None,
        pretrained=None,           # 预训练权重路径
        video_test_mode=False,     # 是否使用时序推理模式
        loss_cfg=None,             # 跟踪损失配置
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,          # Dropout 比例
            update_query_pos=False,    # 是否更新 Query 位置编码
            fp_ratio=0.3,              # 假阳性 Query 保留比例（30%的误检保留，增加难负样本）
            random_drop=0.1,           # 随机丢弃 Query 的概率
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,  # 进入记忆库的置信度阈值
            memory_bank_len=4,             # 记忆库长度（保存4帧历史特征）
        ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 后处理范围
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],              # 点云范围
            max_num=300,         # 最大输出目标数
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,      # 不使用 NMS（跟踪器自行管理）
            iou_thres=0.3,
        ),
        pc_range=None,           # 点云感知范围（x,y,z 的 min/max）
        embed_dims=256,          # Query 特征维度
        num_query=900,           # 候选 Query 数量
        num_classes=10,          # 目标类别数
        vehicle_id_list=None,    # 需要跟踪的类别 ID 列表（仅跟踪车辆类）
        score_thresh=0.2,        # 跟踪置信度阈值（低于此值 → 停止跟踪）
        filter_score_thresh=0.1, # 过滤阈值（低于此值 → 不输出给下游任务）
        miss_tolerance=5,        # 消失容忍帧数（连续5帧未检测到则删除）
        gt_iou_threshold=0.0,    # 训练时筛选有效跟踪目标的 IoU 最低值
        freeze_img_backbone=False,  # 是否冻结图像骨干（用于 Stage2 微调）
        freeze_img_neck=False,      # 是否冻结特征金字塔
        freeze_bn=False,            # 是否冻结 BatchNorm
        freeze_bev_encoder=False,   # 是否冻结 BEV 编码器
        queue_length=3,             # 时序队列长度（使用过去 3 帧）
    ):
        # 调用父类 MVXTwoStageDetector 初始化（构建 img_backbone、img_neck、pts_bbox_head）
        super(UniADTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        # ---- GridMask 数据增强 ----
        # 随机在图像上挖去方格区域，强迫模型不依赖局部纹理特征
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False       # 混合精度标志（由 auto_fp16 装饰器控制）
        self.embed_dims = embed_dims    # 256
        self.num_query = num_query      # 900
        self.num_classes = num_classes  # 10
        self.vehicle_id_list = vehicle_id_list  # 需要跟踪的类别 ID
        self.pc_range = pc_range        # 感知范围
        self.queue_length = queue_length  # 历史帧数

        # ---- 可选：冻结部分网络参数（Stage2 微调时只训练新增头）----
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()  # 冻结 BN 统计量
            for param in self.img_backbone.parameters():
                param.requires_grad = False  # 不计算梯度
        
        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # ---- 时序模式 ----
        self.video_test_mode = video_test_mode
        assert self.video_test_mode  # UniAD 必须使用时序模式

        # prev_frame_info：存储上一帧的信息（用于时序 BEV 对齐）
        self.prev_frame_info = {
            "prev_bev": None,      # 上一帧的 BEV 特征（用于时序融合）
            "scene_token": None,   # 上一帧的场景标识（检测是否切换场景）
            "prev_pos": 0,         # 上一帧自车位置
            "prev_angle": 0,       # 上一帧自车朝向角
        }

        # ---- Query 嵌入（可学习参数）----
        # num_query+1 = 901 个 Query：前900个是目标候选 Query，最后1个是自车（SDC）Query
        # 维度 embed_dims*2 = 512：前256维是特征（content），后256维是位置（position）
        self.query_embedding = nn.Embedding(self.num_query + 1, self.embed_dims * 2)

        # reference_points：将 Query 的前256维映射到 3D 参考点坐标（归一化 [0,1]）
        # 用作 Transformer cross-attention 的位置先验
        self.reference_points = nn.Linear(self.embed_dims, 3)

        # ---- 跟踪核心组件 ----
        self.mem_bank_len = mem_args["memory_bank_len"]  # 记忆库长度

        # track_base：运行时跟踪器，管理每个 Query 的"活跃/睡眠/删除"状态
        # - score_thresh: 高于此值 → 激活目标
        # - filter_score_thresh: 低于此值 → 不输出给下游
        # - miss_tolerance: 连续消失超过此帧数 → 删除目标
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )

        # query_interact（QIM）：Query 交互模块
        # 每帧检测结束后，决定：
        #   - 哪些 Query 继续跟踪（更新状态）
        #   - 哪些 Query 从新候选中激活（检测到新目标）
        #   - 哪些 Query 停止跟踪（目标消失）
        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        # bbox_coder：检测框编解码器
        # 将 Transformer 输出的归一化坐标解码为实际的 3D 检测框
        self.bbox_coder = build_bbox_coder(bbox_coder)

        # memory_bank：记忆库
        # 保存每个目标过去 K 帧的 Query 特征，提供长期历史上下文
        # 让模型在目标被遮挡时仍能"记住"它的特征
        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )

        # criterion：损失函数（匈牙利匹配 + 分类/回归损失）
        self.criterion = build_loss(loss_cfg)

        # 推理时的状态变量（跨帧持续）
        self.test_track_instances = None  # 当前场景的跟踪实例
        self.l2g_r_mat = None             # 上一帧 Local→Global 旋转矩阵
        self.l2g_t = None                 # 上一帧 Local→Global 平移向量
        self.gt_iou_threshold = gt_iou_threshold  # 筛选有效目标的 IoU 阈值

        # BEV 尺寸（从 pts_bbox_head 中读取，通常 200×200）
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

    def extract_img_feat(self, img, len_queue=None):
        """从多相机图像中提取多尺度特征。
        
        流程：
          1. 将 (B, N_cam, C, H, W) 的图像重塑为 (B*N_cam, C, H, W)
          2. 可选：应用 GridMask 数据增强
          3. img_backbone：提取多尺度特征（如 ResNet 的 C3/C4/C5 层）
          4. img_neck（FPN）：融合多尺度特征，统一通道数
          5. 将特征重塑回 (B, N_cam, C, h, w) 格式
        
        Args:
            img (Tensor): 多相机图像，shape (B*len_queue, N_cam, C, H, W)
                          或 (B, N_cam, C, H, W)
            len_queue (int, optional): 时序帧数，提供时自动按帧分组
        
        Returns:
            list[Tensor]: 多尺度特征列表，每个元素 shape (B, len_queue, N_cam, c, h, w)
                          或 (B, N_cam, c, h, w)
        """
        if img is None:
            return None
        assert img.dim() == 5  # 必须是 5 维张量
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)  # 合并 batch 和 camera 维度，统一处理
        
        if self.use_grid_mask:
            img = self.grid_mask(img)  # 应用 GridMask 数据增强
        
        img_feats = self.img_backbone(img)  # 主干网络提取特征
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)  # FPN 融合多尺度特征

        # 将特征重塑回带有 batch 和 camera 维度的格式
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                # 训练时：按时序帧分组 (B//len_queue, len_queue, N_cam, c, h, w)
                img_feat_reshaped = img_feat.view(B // len_queue, len_queue, N, c, h, w)
            else:
                # 推理时：直接分组 (B, N_cam, c, h, w)
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        """初始化所有 Track Query 为空状态（场景开始时调用）。
        
        为 num_query+1=901 个 Query 创建初始状态：
          - query：可学习的 512 维向量（256内容 + 256位置）
          - ref_pts：3D 参考点（初始由 query 的位置部分映射得到）
          - obj_idxes：目标 ID，初始为 -1（未激活）
          - 其他属性（scores、pred_boxes、mem_bank 等）初始化为全零
        
        Returns:
            Instances: 包含所有初始化 Query 的实例容器
        """
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (901, 512)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight  # (901, 512) 可学习 Query

        # 将每个 Query 的位置部分（前256维）映射为 3D 参考点坐标
        # reference_points：256 → 3（x, y, z，归一化到 [0,1]）
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        # 初始化预测框为全零（10维：xy, wl, z, h, sin, cos, vx, vy, vz）
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query = query  # 存储完整 512 维 Query

        # output_embedding：TrackHead 输出的特征嵌入（用于后续 MotionHead）
        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device  # dim>>1 = 256（半精度）
        )

        # obj_idxes：每个 Query 对应的目标 ID
        # -1：未激活（候选 Query）
        # >=0：已激活的跟踪目标（全局唯一 ID）
        # -2：自车（SDC）专用 ID
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )

        # matched_gt_idxes：与 GT 的匹配索引（匈牙利匹配结果）
        # -1：未匹配到 GT
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )

        # disappear_time：目标连续消失的帧数（超过 miss_tolerance 则删除）
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = pred_boxes_init  # 10维预测框
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        # mem_bank：记忆库，存储每个 Query 过去 K 帧的特征
        # shape: (num_queries, mem_bank_len, embed_dims)
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        # mem_padding_mask：记忆库的有效性掩码（True=无效/空位）
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """利用速度预测，将目标参考点从上一帧坐标系更新到当前帧坐标系。
        
        核心思想：
          如果上一帧检测到一辆车在位置 P1，速度为 v，
          那么 time_delta 秒后，这辆车应该在 P2 ≈ P1 + v × time_delta。
          通过速度预测下一帧的位置，让 Query 的参考点"提前到达"目标可能的位置，
          从而提高 Transformer cross-attention 的效率（不用从零搜索）。
        
        坐标变换流程：
          ref_pts（逆sigmoid空间，上一帧）
            ↓ sigmoid → 归一化坐标 [0,1]
            ↓ 反归一化 → 真实米制坐标（激光雷达坐标系 l1）
            ↓ + velocity × time_delta → 速度更新后的位置
            ↓ @ l2g_r1 + l2g_t1 → 全局坐标系（Global）
            ↓ - l2g_t2, @ inv(l2g_r2) → 当前帧激光雷达坐标系（l2）
            ↓ 归一化 → [0,1]
            ↓ inverse_sigmoid → 逆sigmoid空间
          ref_pts（逆sigmoid空间，当前帧）
        
        Args:
            ref_pts (Tensor): 上一帧的参考点，shape (num_query, 3)，逆sigmoid空间
            velocity (Tensor): 目标速度，shape (num_query, 2)，单位 m/s，激光雷达坐标系
            l2g_r1 (Tensor): 上一帧 Local→Global 旋转矩阵，(3, 3)
            l2g_t1 (Tensor): 上一帧 Local→Global 平移向量，(1, 3)
            l2g_r2 (Tensor): 当前帧 Local→Global 旋转矩阵，(3, 3)
            l2g_t2 (Tensor): 当前帧 Local→Global 平移向量，(1, 3)
            time_delta (float): 两帧之间的时间差，单位秒
        
        Returns:
            ref_pts (Tensor): 更新后的参考点，shape (num_query, 3)，逆sigmoid空间
        """
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)

        # 速度补零：vx, vy → (vx, vy, 0)，z 方向速度设为 0
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)  # (num_query, 3)

        # Step1：逆sigmoid → sigmoid → 归一化坐标 [0,1]
        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range  # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        # Step2：反归一化 → 真实米制坐标（激光雷达坐标系）
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )  # x: [0,1] → [-51.2, 51.2]
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )  # y: [0,1] → [-51.2, 51.2]
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )  # z: [0,1] → [-5.0, 3.0]

        # Step3：速度更新（线性运动假设）
        reference_points = reference_points + velo_pad * time_delta

        # Step4：Local(l1) → Global 坐标变换
        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2

        # Step5：Global → Local(l2) 坐标变换（当前帧的激光雷达坐标系）
        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)  # Global→Local 旋转矩阵
        ref_pts = ref_pts @ g2l_r

        # Step6：重新归一化到 [0,1]
        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        # Step7：归一化坐标 → 逆sigmoid空间（与 DETR 的参考点格式对齐）
        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def _copy_tracks_for_loss(self, tgt_instances):
        """复制 Track Instances 的关键属性用于多层损失计算。
        
        TrackHead 是多层 Decoder（如 6 层），每层都需要计算损失（辅助损失）。
        但预测值（scores、boxes、logits）需要每层独立，不能共享。
        因此对前 nb_dec-1 层，需要复制一个"空白"的 instances，只保留跟踪状态。
        
        Args:
            tgt_instances: 原始 Track Instances
        
        Returns:
            Instances: 复制了跟踪状态但清空了预测值的新实例
        """
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        # 保留跟踪状态相关属性（跨层共享）
        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        # 清空预测相关属性（每层独立预测，不共享）
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)

    def get_history_bev(self, imgs_queue, img_metas_list):
        """迭代获取历史帧的 BEV 特征（不计算梯度，节省显存）。
        
        用于训练时：当前帧需要利用历史帧的 BEV 来做时序融合，
        但历史帧的 BEV 特征不需要参与梯度计算（只作为条件输入）。
        
        Args:
            imgs_queue (Tensor): 历史帧图像，shape (B, len_queue, N_cam, C, H, W)
            img_metas_list (list): 历史帧的元信息列表
        
        Returns:
            prev_bev (Tensor): 最后一帧的 BEV 特征，作为当前帧的历史条件
        """
        self.eval()  # 切换到推理模式（冻结 BN 统计量）
        with torch.no_grad():  # 不计算梯度
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                # 逐帧更新 BEV（时序滚动）
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats,
                    img_metas=img_metas,
                    prev_bev=prev_bev)  # 将上一帧 BEV 作为历史输入
        self.train()  # 恢复训练模式
        return prev_bev  # 返回最后一帧的 BEV 特征

    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        """生成当前帧的 BEV 特征。
        
        两种调用方式：
          1. 训练时：传入 prev_img，先计算历史 BEV，再用于当前帧
          2. 推理时：直接传入 prev_bev（上一帧已经计算好的 BEV）
        
        如果配置了 freeze_bev_encoder，则使用 no_grad 不更新 BEV 编码器参数。
        
        Args:
            imgs (Tensor): 当前帧图像，shape (B, N_cam, C, H, W)
            img_metas (list): 当前帧元信息
            prev_img (Tensor, optional): 历史帧图像（训练时使用）
            prev_img_metas (list, optional): 历史帧元信息
            prev_bev (Tensor, optional): 历史 BEV 特征（推理时使用）
        
        Returns:
            bev_embed (Tensor): BEV 特征，shape (H*W, B, D) = (40000, 1, 256)
            bev_pos (Tensor): BEV 位置编码，shape (1, 256, 200, 200)
        """
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None  # 两种历史输入方式不能同时使用
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            # 冻结 BEV 编码器：Stage2 中只训练各任务头，不更新 BEVFormer
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)

        # 确保输出格式为 (H*W, B, D)（某些实现输出 (B, H*W, D)，这里统一转置）
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)

        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    @auto_fp16(apply_to=("img", "prev_bev"))
    def _forward_single_frame_train(
        self,
        img,              # 当前帧图像，(B, N_cam, C, H, W)
        img_metas,        # 当前帧元信息
        track_instances,  # 当前帧的跟踪实例（来自上一帧，或初始化的空实例）
        prev_img,         # 历史帧图像（用于构建历史 BEV）
        prev_img_metas,   # 历史帧元信息
        l2g_r1=None,      # 当前帧的 Local→Global 旋转矩阵
        l2g_t1=None,      # 当前帧的 Local→Global 平移向量
        l2g_r2=None,      # 下一帧的 Local→Global 旋转矩阵（用于 velo_update）
        l2g_t2=None,      # 下一帧的 Local→Global 平移向量
        time_delta=None,  # 当前帧到下一帧的时间差
        all_query_embeddings=None,    # 收集所有 Decoder 层的 Query 特征
        all_matched_indices=None,     # 收集所有 Decoder 层的匹配结果
        all_instances_pred_logits=None,  # 收集所有层的分类预测
        all_instances_pred_boxes=None,   # 收集所有层的框预测
    ):
        """对单帧执行检测和跟踪的前向传播（训练时调用）。
        
        注意：只支持 Batch Size = 1
        
        流程：
          1. get_bevs()：生成当前帧 BEV 特征
          2. pts_bbox_head.get_detections()：在 BEV 上检测目标（多层 Decoder）
          3. velo_update()：用速度预测目标在下一帧的参考点位置
          4. 对每层 Decoder 的输出调用 criterion.match_for_single_frame()：
             匈牙利匹配，为每个 Query 分配 GT 目标（或背景）
          5. select_active_track_query()：筛选出有效的跟踪 Query
          6. select_sdc_track_query()：提取自车（SDC）Query 的状态
          7. memory_bank()：更新记忆库
          8. query_interact（QIM）：更新 Query 状态（激活新目标/删除消失目标）
        
        Args:
            l2g_r2=None 表示这是训练序列的最后一帧，不需要更新参考点
        
        Returns:
            dict: 包含 bev_embed、track_query、sdc 相关信息等
        """
        # ---- Step 1: 生成 BEV 特征 ----
        # 注意：这里传入 prev_img 让 get_bevs 计算历史 BEV（训练模式）
        bev_embed, bev_pos = self.get_bevs(
            img, img_metas,
            prev_img=prev_img, prev_img_metas=prev_img_metas,
        )
        # bev_embed: (H*W, B, D) = (40000, 1, 256)

        # ---- Step 2: TrackHead 检测（多层 Decoder）----
        # get_detections：
        #   - 输入：BEV 特征 + 每个 Query 的向量 + 参考点
        #   - 输出：每层 Decoder 的分类 logit、框坐标、历史轨迹预测
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,   # (901, 512) 当前帧的 Query
            ref_points=track_instances.ref_pts,           # (901, 3) 参考点
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]   # (nb_dec, B, 901, num_classes) 各层分类得分
        output_coords = det_output["all_bbox_preds"]    # (nb_dec, B, 901, 10) 各层框预测
        output_past_trajs = det_output["all_past_traj_preds"]  # 历史轨迹预测
        last_ref_pts = det_output["last_ref_points"]    # (B, 901, 3) 最后一层的参考点
        query_feats = det_output["query_feats"]         # (nb_dec, B, 901, 256) 各层 Query 特征

        # 整理当前帧的输出（取最后一层 Decoder 的预测作为主预测）
        out = {
            "pred_logits": output_classes[-1],   # 最后一层分类预测
            "pred_boxes": output_coords[-1],     # 最后一层框预测
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }

        # 计算跟踪置信度分数（取各类别中的最大概率）
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # track_scores: (901,) 每个 Query 的综合置信度

        # ---- Step 3: 速度更新参考点（为下一帧预测位置）----
        nb_dec = output_classes.size(0)  # Decoder 层数（如 6）

        # 为前 nb_dec-1 层创建独立的 track_instances 副本（用于辅助损失）
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # 保存最后层的 Query 特征
        velo = output_coords[-1, 0, :, -2:]  # 预测速度：(901, 2)，最后两维是 vx, vy

        if l2g_r2 is not None:
            # 非最后帧：用速度更新参考点，预测目标下一帧的位置
            ref_pts = self.velo_update(
                last_ref_pts[0],  # 当前帧参考点
                velo,             # 预测速度
                l2g_r1, l2g_t1,  # 当前帧坐标变换
                l2g_r2, l2g_t2,  # 下一帧坐标变换
                time_delta=time_delta,  # 帧间时间差
            )
        else:
            # 最后帧：无需更新参考点
            ref_pts = last_ref_pts[0]

        # 更新 track_instances 的参考点
        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim // 2])
        track_instances.ref_pts[..., :2] = ref_pts[..., :2]  # 只更新 xy，保留 z

        track_instances_list.append(track_instances)  # 最后一层用原始实例

        # ---- Step 4: 每层 Decoder 执行匈牙利匹配 ----
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            # 将当前层的预测结果写入实例
            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]    # (901, num_cls)
            track_instances.pred_boxes = output_coords[i, 0]      # (901, 10)
            track_instances.pred_past_trajs = output_past_trajs[i, 0]

            out["track_instances"] = track_instances
            # match_for_single_frame：匈牙利匹配，为每个 Query 分配 GT 目标（或背景）
            # if_step=True 表示最后一层，需要真正更新跟踪状态
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])

        # ---- Step 5: 筛选有效跟踪 Query ----
        # active_index：满足以下条件的 Query 才算有效跟踪目标：
        #   - obj_idxes >= 0：已被分配全局 ID（已激活）
        #   - iou >= gt_iou_threshold：与 GT 的 IoU 足够大
        #   - matched_gt_idxes >= 0：成功匹配到 GT
        active_index = (
            (track_instances.obj_idxes >= 0) &
            (track_instances.iou >= self.gt_iou_threshold) &
            (track_instances.matched_gt_idxes >= 0)
        )
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        # 提取最后一个固定的自车 Query（动态索引，兼容任意 num_query）
        _sdc_idx = len(track_instances) - 1
        out.update(self.select_sdc_track_query(track_instances[_sdc_idx], img_metas))

        # ---- Step 6: 更新记忆库 ----
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        # ---- Step 7: QIM 更新 Query 状态 ----
        # QIM 决定哪些 Query 继续跟踪、哪些新目标被激活、哪些消失目标被删除
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()  # 新候选 Query
        tmp["track_instances"] = track_instances  # 当前已更新的 Query
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances  # 更新后传入下一帧
        return out

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        """筛选并整理有效跟踪目标的 Query 特征和检测结果。
        
        从所有 901 个 Query 中，筛选出确实跟踪到目标的 Query，
        并将其特征和检测框整理为后续任务（MotionHead 等）需要的格式。
        
        Args:
            track_instances: 所有 Query 的实例
            active_index (Tensor): 布尔掩码，True = 有效跟踪目标
            img_metas: 图像元信息（用于坐标解码）
            with_mask (bool): 是否使用置信度掩码过滤低置信度目标
        
        Returns:
            dict: 包含 track_query_embeddings（Query 特征）和
                  track_query_matched_idxes（GT 匹配索引）等
        """
        result_dict = self._track_instances2results(
            track_instances[active_index], img_metas, with_mask=with_mask)
        # 额外添加 Query 特征嵌入（供 MotionHead 使用）
        result_dict["track_query_embeddings"] = (
            track_instances.output_embedding[active_index]
            [result_dict['bbox_index']]
            [result_dict['mask']]
        )
        # 额外添加 GT 匹配索引（供 OccHead 中的占用掩码对齐使用）
        result_dict["track_query_matched_idxes"] = (
            track_instances.matched_gt_idxes[active_index]
            [result_dict['bbox_index']]
            [result_dict['mask']]
        )
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        """提取自车（SDC/Ego Vehicle）Query 的状态。
        
        第 900 个 Query（索引900）固定用于表示自车，
        其特征（sdc_embedding）会传给 MotionHead 和 PlanningHead。
        
        Args:
            sdc_instance: 自车 Query 的实例（单个）
            img_metas: 图像元信息
        
        Returns:
            dict: 包含自车的检测框、置信度、嵌入特征等
        """
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']             # 自车预测框
        out["sdc_scores_3d"] = result_dict['scores_3d']           # 自车检测置信度
        out["sdc_track_scores"] = result_dict['track_scores']     # 自车跟踪置信度
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]   # 自车 Query 特征（256维）
        return out

    @auto_fp16(apply_to=("img", "points"))
    def forward_track_train(self,
                            img,             # (B, len_queue, N_cam, C, H, W) 时序图像
                            gt_bboxes_3d,    # GT 3D 检测框
                            gt_labels_3d,    # GT 类别标签
                            gt_past_traj,    # GT 历史轨迹
                            gt_past_traj_mask,  # GT 历史轨迹有效性掩码
                            gt_inds,         # GT 实例 ID（跨帧跟踪 ID）
                            gt_sdc_bbox,     # GT 自车检测框
                            gt_sdc_label,    # GT 自车类别
                            l2g_t,           # 各帧 Local→Global 平移向量
                            l2g_r_mat,       # 各帧 Local→Global 旋转矩阵
                            img_metas,       # 各帧元信息
                            timestamp):      # 各帧时间戳
        """跟踪任务的训练前向传播：对 len_queue 帧逐帧处理。
        
        训练时与推理时的关键区别：
          - 训练：将 len_queue=3 帧作为一个 clip 一起处理（类似短视频）
            帧0 → 帧1 → 帧2，跨帧传递 track_instances
          - 推理：每次只处理当前帧（逐帧在线推理）
        
        流程：
          1. 初始化所有 Query 为空状态（_generate_empty_tracks）
          2. 准备每帧的 GT 实例（normalize_bbox 归一化坐标）
          3. 调用 criterion.initialize_for_single_clip 初始化损失计算
          4. 逐帧调用 _forward_single_frame_train 处理每帧
          5. 只取最后一帧的输出（传给 MotionHead 等后续模块）
        
        Returns:
            tuple: (losses, out)
                losses: 跟踪损失字典（分类损失 + 框回归损失 + 轨迹损失）
                out: 最后一帧的跟踪结果（bev_embed、track_query 等）
        """
        track_instances = self._generate_empty_tracks()  # 初始化空 Query
        num_frame = img.size(1)  # 时序帧数（通常 len_queue=3）

        # ---- 准备每帧的 GT 实例 ----
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
            boxes = normalize_bbox(boxes, self.pc_range)  # 归一化到 [0,1] 范围
            sd_boxes = gt_sdc_bbox[0][i].tensor.to(img.device)
            sd_boxes = normalize_bbox(sd_boxes, self.pc_range)

            gt_instances.boxes = boxes               # GT 检测框
            gt_instances.labels = gt_labels_3d[0][i]  # GT 类别
            gt_instances.obj_ids = gt_inds[0][i]     # GT 跟踪 ID（跨帧唯一标识）
            gt_instances.past_traj = gt_past_traj[0][i].float()        # GT 历史轨迹
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()

            # 将自车框广播到与目标数量相同（方便后续联合处理）
            gt_instances.sdc_boxes = torch.cat(
                [sd_boxes for _ in range(boxes.shape[0])], dim=0)
            gt_instances.sdc_labels = torch.cat(
                [gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
            gt_instances_list.append(gt_instances)

        # 初始化损失计算器（注册 GT，为后续匈牙利匹配做准备）
        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()

        # ---- 逐帧处理 ----
        for i in range(num_frame):
            # 历史帧：用于构建历史 BEV（第0帧时只有1帧历史）
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)

            img_single = torch.stack([img_[i] for img_ in img], dim=0)  # 当前帧图像
            img_metas_single = [copy.deepcopy(img_metas[0][i])]

            # 最后一帧没有"下一帧"，不需要速度更新
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]  # 帧间时间差（秒）

            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []

            frame_res = self._forward_single_frame_train(
                img_single,
                img_metas_single,
                track_instances,     # 上一帧的跟踪状态（帧间传递）
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],     # 当前帧坐标变换
                l2g_t[0][i],
                l2g_r2,              # 下一帧坐标变换（最后帧为 None）
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
            )
            # 更新 track_instances：传递给下一帧
            track_instances = frame_res["track_instances"]

        # ---- 只取最后一帧的输出（传给 UniAD 的后续任务）----
        get_keys = [
            "bev_embed", "bev_pos",
            "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
            "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores",
            "sdc_track_bbox_results", "sdc_embedding"
        ]
        out.update({k: frame_res[k] for k in get_keys})

        losses = self.criterion.losses_dict  # 收集所有帧的跟踪损失
        return losses, out

    def upsample_bev_if_tiny(self, outs_track):
        """如果是 tiny 版本模型，将 BEV 特征从 100×100 上采样到 200×200。
        
        UniAD 有两个版本：
          - 标准版：BEV 分辨率 200×200
          - Tiny 版：BEV 分辨率 100×100（节省显存），推理前需上采样到 200×200
        
        检测条件：bev_embed 的第0维 == 100*100 = 10000。
        
        同时上采样：bev_embed、prev_bev、bev_pos 三个张量。
        训练时和推理时的维度格式略有不同，分别处理。
        
        Args:
            outs_track (dict): 跟踪输出字典
        
        Returns:
            dict: 上采样后的跟踪输出字典
        """
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # ---- BEV 特征上采样：(10000, B, 256) → (40000, B, 256) ----
            bev_embed = outs_track["bev_embed"]  # [10000, 1, 256]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(bev_embed, '(h w) b c -> b c h w', h=h, w=w)  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)                   # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, 'b c h w -> (h w) b c')
            outs_track["bev_embed"] = bev_embed

            # ---- prev_bev 上采样（历史 BEV）----
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    # 训练时 prev_bev 格式：[B, H*W, C]
                    prev_bev = rearrange(prev_bev, 'b (h w) c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)
                    prev_bev = rearrange(prev_bev, 'b c h w -> b (h w) c')
                    outs_track["prev_bev"] = prev_bev
                else:
                    # 推理时 prev_bev 格式：[H*W, B, C]
                    prev_bev = rearrange(prev_bev, '(h w) b c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)
                    prev_bev = rearrange(prev_bev, 'b c h w -> (h w) b c')
                    outs_track["prev_bev"] = prev_bev

            # ---- bev_pos 上采样（BEV 位置编码）----
            bev_pos = outs_track["bev_pos"]        # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track

    def _forward_single_frame_inference(
        self,
        img,             # 当前帧图像，(B, N_cam, C, H, W)
        img_metas,       # 当前帧元信息
        track_instances, # 上一帧传递的跟踪实例
        prev_bev=None,   # 上一帧的 BEV 特征（推理时直接传入，避免重算）
        l2g_r1=None,     # 上一帧 Local→Global 旋转矩阵
        l2g_t1=None,     # 上一帧 Local→Global 平移向量
        l2g_r2=None,     # 当前帧 Local→Global 旋转矩阵（用于更新参考点到下一帧）
        l2g_t2=None,     # 当前帧 Local→Global 平移向量
        time_delta=None, # 帧间时间差
    ):
        """对单帧执行检测和跟踪的前向传播（推理时调用）。
        
        与训练版本（_forward_single_frame_train）的主要区别：
          1. 不使用 prev_img，直接接收 prev_bev（上一帧已经计算好的 BEV）
          2. 先执行速度更新（velo_update），再做 BEV 生成和检测
          3. 调用 track_base.update() 更新跟踪状态（分配全局 ID）
          4. 不计算损失，只更新跟踪状态
        
        推理流程：
          1. velo_update：先用上一帧预测速度，更新活跃目标的参考点位置
          2. get_bevs()：生成当前帧 BEV 特征
          3. pts_bbox_head.get_detections()：执行检测（只用最后层 Decoder 输出）
          4. 更新 track_instances 的预测结果
          5. track_base.update()：分配跟踪 ID（新目标激活、消失目标计时）
          6. select_active_track_query()：筛选有效目标
          7. memory_bank()：更新记忆库
          8. query_interact（QIM）：更新 Query 状态
        
        Args:
            l2g_r1/l2g_t1: 上一帧的坐标变换（速度更新用）
            l2g_r2/l2g_t2: 当前帧的坐标变换（更新参考点到"下下帧"用）
        
        Returns:
            dict: 当前帧的跟踪结果
        """
        # =====================================================================
        # Step 1: 速度更新（在 BEV 特征提取之前，先预测目标当前位置）
        # =====================================================================
        # 分离活跃目标（obj_idxes >= 0）和非活跃候选（obj_idxes < 0）
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            # 对活跃目标执行速度更新：将上一帧参考点更新到当前帧坐标系
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]  # 上一帧预测的速度
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim // 2])
            active_inst.ref_pts[..., :2] = ref_pts[..., :2]  # 更新 xy，保留 z

        # 重新合并（先非活跃，再活跃，保持索引顺序）
        track_instances = Instances.cat([other_inst, active_inst])

        # =====================================================================
        # Step 2: BEV 特征提取 + 检测
        # =====================================================================
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"]  # (nb_dec, B, 901, num_classes)
        output_coords = det_output["all_bbox_preds"]   # (nb_dec, B, 901, 10)
        last_ref_pts = det_output["last_ref_points"]   # (B, 901, 3)
        query_feats = det_output["query_feats"]        # (nb_dec, B, 901, 256)

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        # =====================================================================
        # Step 3: 更新跟踪实例的预测结果
        # =====================================================================
        # 只用最后一层 Decoder 的输出（推理时不需要辅助损失）
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values  # (901,)
        track_instances.scores = track_scores
        track_instances.pred_logits = output_classes[-1, 0]    # (901, num_cls)
        track_instances.pred_boxes = output_coords[-1, 0]      # (901, 10)
        track_instances.output_embedding = query_feats[-1][0]  # (901, 256)
        track_instances.ref_pts = last_ref_pts[0]              # 更新参考点

        # 动态获取最后一个 Query（自车 Query），ID=-2，兼容任意 num_query
        track_instances.obj_idxes[len(track_instances) - 1] = -2

        # =====================================================================
        # Step 4: 跟踪状态更新（分配/维护全局跟踪 ID）
        # =====================================================================
        # track_base.update 做三件事：
        #   - 高置信度的新 Query → 分配新的全局 ID（新目标出现）
        #   - 连续消失超过 miss_tolerance 的目标 → 删除（disappear_time 超限）
        #   - 其他活跃目标 → 维持原 ID（正常跟踪）
        self.track_base.update(track_instances, None)

        # 筛选输出给下游任务的目标：活跃 + 置信度 >= filter_score_thresh
        active_index = (
            (track_instances.obj_idxes >= 0) &
            (track_instances.scores >= self.track_base.filter_score_thresh)
        )
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(
            track_instances[track_instances.obj_idxes == -2], img_metas))

        # =====================================================================
        # Step 5: 记忆库更新 + QIM 更新
        # =====================================================================
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)

        out["track_instances_fordet"] = track_instances   # 用于最终检测结果输出
        out["track_instances"] = out_track_instances      # 传给下一帧
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
    ):
        """推理时的跟踪入口函数（只支持 bs=1 和时序输入）。
        
        管理跨帧的跟踪状态：
          - 新场景第一帧：初始化所有 Query，清空历史 BEV
          - 后续帧：使用上一帧的 track_instances 和 prev_bev 继续跟踪
        
        Args:
            img (Tensor): 当前帧图像，(1, N_cam, C, H, W)
            l2g_t (Tensor): 当前帧 Local→Global 平移
            l2g_r_mat (Tensor): 当前帧 Local→Global 旋转矩阵
            img_metas (list): 当前帧元信息（含 scene_token）
            timestamp (float): 当前帧时间戳
        
        Returns:
            list[dict]: 包含当前帧的跟踪结果（检测框、跟踪 ID、BEV 特征等）
        """
        bs = img.size(0)

        # ---- 检测是否是新场景（切换场景时重置跟踪状态）----
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            # 新场景第一帧：重置所有跟踪状态
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None  # 没有历史 BEV
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
        else:
            # 后续帧：使用上一帧的跟踪实例继续跟踪
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp  # 帧间时间差
            l2g_r1 = self.l2g_r_mat                  # 上一帧坐标变换
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat                       # 当前帧坐标变换
            l2g_t2 = l2g_t

        # ---- 更新帧信息（供下一帧使用）----
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        # ---- 执行单帧推理 ----
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img, img_metas, track_instances,
            prev_bev, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta,
        )

        # ---- 保存当前帧状态供下一帧使用 ----
        self.prev_bev = frame_res["bev_embed"]              # 当前 BEV 作为下一帧的历史
        track_instances = frame_res["track_instances"]      # 更新后的 Query
        track_instances_fordet = frame_res["track_instances_fordet"]  # 用于输出检测结果

        self.test_track_instances = track_instances  # 保存跟踪状态

        # ---- 整理输出 ----
        results = [dict()]
        get_keys = [
            "bev_embed", "bev_pos",
            "track_query_embeddings", "track_bbox_results",
            "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"
        ]
        if self.with_motion_head:
            # 如果有 MotionHead，还需要传递自车相关信息
            get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores",
                         "sdc_track_bbox_results", "sdc_embedding"]
        results[0].update({k: frame_res[k] for k in get_keys})
        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        return results

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        """将 Track Instances 的预测结果解码为标准的检测结果格式。
        
        流程：
          1. 将 Query 的预测（归一化坐标 + 类别 logit）整理为字典
          2. 调用 bbox_coder.decode() 解码：归一化坐标 → 实际 3D 检测框（米）
          3. 整理为统一的结果字典格式
        
        Args:
            track_instances: 跟踪实例（含 pred_logits、pred_boxes、scores 等）
            img_metas: 元信息（含坐标类型、传感器外参等）
            with_mask (bool): 是否使用置信度掩码过滤低分目标
        
        Returns:
            dict: 标准检测结果字典，包含：
                boxes_3d：3D 检测框（LidarInstance3DBoxes）
                scores_3d：检测置信度
                labels_3d：类别 ID
                track_scores：跟踪置信度
                track_ids：跟踪 ID（全局唯一）
                bbox_index：在 BBox Coder 中的框索引
                mask：有效框的布尔掩码
                track_bbox_results：打包格式（用于评估器）
        """
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,   # (N, num_classes) 分类 logit
            bbox_preds=track_instances.pred_boxes,    # (N, 10) 归一化框坐标
            track_scores=track_instances.scores,      # (N,) 跟踪置信度
            obj_idxes=track_instances.obj_idxes,      # (N,) 跟踪 ID
        )
        # bbox_coder.decode：归一化坐标 → 真实米制 3D 检测框
        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)  # 包装为 LidarInstance3DBoxes 对象
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]
        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]

        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[
                bboxes.to("cpu"), scores.cpu(), labels.cpu(),
                bbox_index.cpu(), bboxes_dict["mask"].cpu()
            ]]
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """将检测实例（所有 Query）的预测结果解码为标准检测格式。
        
        与 _track_instances2results 的区别：
          - _track_instances2results：只处理通过筛选的活跃跟踪目标
          - _det_instances2results：处理所有 Query（包括低置信度的候选）
            用于输出完整的检测结果（供评测 3D 检测指标使用）
        
        Args:
            instances: 所有检测实例
            results (list[dict]): 要更新的结果列表
            img_metas: 元信息
        
        Returns:
            list[dict]: 更新了检测结果（boxes_3d_det、scores_3d_det 等）的结果列表
        """
        # 过滤空预测（无目标时直接返回）
        if instances.pred_logits.numel() == 0:
            return [None]

        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),    # 检测框（用于 det 指标评估）
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]
    