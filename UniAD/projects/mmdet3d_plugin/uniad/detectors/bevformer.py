# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

# ================================================================================
# 【文件说明】bevformer.py —— 独立版 BEVFormer 检测器
#
# 【职责】
# 这个文件是一个"组装器"，把三个子模块串联成完整的检测流水线：
#   1. img_backbone（ResNet 等）：从图像提取特征
#   2. img_neck（FPN）：融合多尺度特征
#   3. pts_bbox_head（BEVFormerHead）：生成 BEV + 解码检测框  ← 重点在这里
#
# 【与 BEVFormerHead 的关系】
# bevformer.py（本文件）是"管理者"，只负责：
#   - 准备图像特征（extract_img_feat）
#   - 在正确的时机调用 pts_bbox_head（BEVFormerHead）
#   - 管理训练/推理分支、历史帧缓存
#
# 真正的核心计算（BEV 生成、匈牙利匹配、损失计算）全在 BEVFormerHead 里。
# 本文件通过 self.pts_bbox_head 来调用它：
#   self.pts_bbox_head(...)         → 调用 BEVFormerHead.forward()  生成检测结果
#   self.pts_bbox_head.loss(...)    → 调用 BEVFormerHead.loss()     计算损失
#   self.pts_bbox_head.get_bboxes(...)→ 调用 BEVFormerHead.get_bboxes() 解码预测框
#
# 【使用场景】
# 独立训练 3D 目标检测（不含跟踪/规划），对应配置文件：
#   configs/bevformer/base_bevformer.py
# ================================================================================

import torch
from mmcv.runner import force_fp32, auto_fp16
# force_fp32：强制某函数用 float32 精度运行，防止 fp16 下溢或精度不足
# auto_fp16：自动将指定输入 Tensor 转为 float16，加速 GPU 计算、节省显存

from mmdet.models import DETECTORS
# DETECTORS：mmdet 的全局检测器注册表
# 配置文件里写 type='BEVFormer' 时，框架通过这个注册表找到本类并实例化

from mmdet3d.core import bbox3d2result
# bbox3d2result：将 (bboxes, scores, labels) 三元组打包成标准输出字典
# 输出格式：{'boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...}

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
# MVXTwoStageDetector：mmdet3d 的多模态两阶段检测基类
# 它已经帮我们搭好了 backbone/neck/head 的初始化框架和调用接口
# BEVFormer 继承它，只需实现自己特有的前向逻辑，其余复用基类

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
# GridMask：数据增强技术
# 训练时在图像上随机"遮掉"一些矩形格子（像给图像打上棋盘格状马赛克）
# 目的：让模型不能依赖图像某个固定区域，增强泛化能力

import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
# run_time：一个计时装饰器，用于测量某函数执行耗时（调试/性能分析时用）


@DETECTORS.register_module()
# 把 BEVFormer 类注册到检测器注册表
# 效果：配置文件里写 type='BEVFormer'，框架就能自动找到并创建这个类的实例
class BEVFormer(MVXTwoStageDetector):
    """独立版 BEVFormer 3D 检测器。

    【整体流程】
    输入：多摄像头图像序列（含历史帧）
      ↓
    extract_img_feat：backbone + neck 提取多尺度图像特征
      ↓
    pts_bbox_head（BEVFormerHead）：
      ├─ Encoder：BEV Query 通过空间交叉注意力向相机图像"提问"→ BEV 特征图
      ├─ (时序) Temporal SA：融合上一帧 BEV 特征
      └─ Decoder：Object Query 关注 BEV → 解码 3D 检测框
      ↓
    输出：3D 检测框 + BEV 特征（可传给下一帧）

    Args:
        video_test_mode (bool): 推理时是否使用时序 BEV（True=视频逐帧模式）
    """

    def __init__(self,
                 use_grid_mask=False,       # 是否启用 GridMask 数据增强
                 pts_voxel_layer=None,      # 以下点云相关参数 BEVFormer 不使用，均为 None
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,         # 图像骨干网络（如 ResNet-101），提取 CNN 特征
                 pts_backbone=None,         # 点云骨干（不用）
                 img_neck=None,             # 图像颈部网络（如 FPN），融合多尺度特征
                 pts_neck=None,             # 点云颈部（不用）
                 pts_bbox_head=None,        # ★ BEVFormerHead：整个框架最核心的模块
                                            #   负责 BEV 特征生成 + 3D 检测框解码
                 img_roi_head=None,         # 图像 ROI 头（一般不用）
                 img_rpn_head=None,         # RPN 候选框头（一般不用）
                 train_cfg=None,            # 训练配置（匹配器参数、损失权重等）
                 test_cfg=None,             # 测试配置（置信度阈值、NMS 参数等）
                 pretrained=None,           # 预训练权重路径
                 video_test_mode=False      # 推理时是否使用历史帧的 BEV
                 ):

        # 调用父类初始化，自动构建 backbone、neck、head 等子模块
        # 父类会根据传入的配置字典，用注册表找到对应的类并实例化
        # 例如：pts_bbox_head={'type': 'BEVFormerHead', ...} → 创建 BEVFormerHead 对象
        #       并赋值给 self.pts_bbox_head
        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        # GridMask 参数说明：
        # use_h=True：遮挡横向格子  use_w=True：遮挡纵向格子
        # rotate=1：格子轻微随机旋转  offset=False：不随机偏移起始位置
        # ratio=0.5：被遮挡的格子占 50%  mode=1：遮挡为黑色  prob=0.7：70% 概率应用
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False  # 关闭父类的 fp16 标志（由 @auto_fp16 装饰器控制）

        # ── 推理时的时序状态缓存 ──────────────────────────────────────────
        self.video_test_mode = video_test_mode

        # prev_frame_info：推理时维护的"上一帧状态"
        # 每处理完一帧，都把当前帧的信息存入这里，供下一帧做时序融合用
        self.prev_frame_info = {
            'prev_bev': None,       # 上一帧的 BEV 特征图 Tensor
                                    # BEVFormerHead 的时序自注意力会用它对齐历史信息
            'scene_token': None,    # 上一帧所属场景的唯一 ID
                                    # nuScenes 数据集中每段行驶场景有一个 token
                                    # 若 token 变了，说明场景切换，必须清空 prev_bev
            'prev_pos': 0,          # 上一帧自车的全局位置 [x, y, z]（单位：米）
                                    # 来自 CAN bus（车辆内部传感器总线）
            'prev_angle': 0,        # 上一帧自车的偏航角（车头朝向，单位：弧度）
                                    # 与 prev_pos 一起用于计算两帧间的位姿变化量
        }


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """从多摄像头图像中提取多尺度特征。

        【处理流程】
        原始图像 [B, N_cam, C, H, W]
          → reshape 成 [B*N_cam, C, H, W]（把摄像头维合并进 batch 维，批量处理）
          → (可选) GridMask 数据增强
          → img_backbone（ResNet）提取特征
          → img_neck（FPN）融合多尺度特征
          → reshape 回 [B, N_cam, C', H', W'] 或 [B, len_queue, N_cam, C', H', W']

        Args:
            img (Tensor): 输入图像
                - 单帧：shape [B, N_cam, C, H, W]
                - 多帧：shape [B*len_queue, N_cam, C, H, W]（历史帧已合并进 B 维）
            img_metas (list[dict]): 每张图像的元信息（相机内外参矩阵等）
            len_queue (int, optional): 历史帧队列长度。
                不为 None 时，输出特征会额外保留时间维度。

        Returns:
            list[Tensor]: 多尺度特征列表（对应 FPN 的不同层级），每个 Tensor：
                - 无 len_queue: shape [B, N_cam, C', H', W']
                - 有 len_queue: shape [B, len_queue, N_cam, C', H', W']
        """
        B = img.size(0)  # 注意：此时 B 可能是 batch_size * len_queue
        if img is not None:
            # ── 将 5D 输入（含摄像头维）压平为 4D（供 backbone 处理）──────
            if img.dim() == 5 and img.size(0) == 1:
                # batch=1 时直接 squeeze 去掉多余维度
                # squeeze_() 是"原地操作"，不新建 Tensor，节省显存
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                # batch>1 时：[B, N_cam, C, H, W] → [B*N_cam, C, H, W]
                # 把"batch×摄像头"当成一个大 batch，让 backbone 统一处理
                # 比循环处理每个摄像头快得多（GPU 并行）
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            # ── (可选) GridMask 数据增强 ────────────────────────────────
            if self.use_grid_mask:
                # 随机在图像上遮挡网格区域
                # 注意：只在训练时应用（GridMask 内部会判断 training 模式）
                img = self.grid_mask(img)

            # ── backbone 提取特征 ────────────────────────────────────────
            # img_backbone 通常是 ResNet，输出多个尺度的特征图：
            #   stride=8  的 C3 层：分辨率较高，感受野较小
            #   stride=16 的 C4 层：分辨率中等
            #   stride=32 的 C5 层：分辨率较低，语义较强
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                # 某些 backbone 返回 dict 格式，统一转为 list
                img_feats = list(img_feats.values())
        else:
            return None

        # ── neck（FPN）融合多尺度特征 ─────────────────────────────────────
        if self.with_img_neck:
            # FPN（Feature Pyramid Network，特征金字塔网络）：
            # 将 C3/C4/C5 的特征做"自顶向下"的融合：
            #   高层特征（语义强）通过上采样传递给低层特征（细节丰富）
            # 融合后每个尺度都同时拥有"看得远"和"看得细"的能力
            img_feats = self.img_neck(img_feats)

        # ── reshape 回含有 batch/摄像头/时间 维度的格式 ──────────────────
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()  # BN = B * N_cam
            if len_queue is not None:
                # 处理历史帧：reshape 为 [B/len_queue, len_queue, N_cam, C', H', W']
                # 恢复出 batch 维（真实 batch 大小）和时间队列维
                img_feats_reshaped.append(
                    img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                # 处理单帧：reshape 为 [B, N_cam, C', H', W']
                # 恢复出 batch 维和摄像头维
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    # 自动将 img 转为 float16 精度传入（减少显存，加速计算）
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """提取图像特征的统一对外接口（封装 extract_img_feat）。

        父类 MVXTwoStageDetector 定义了 extract_feat 这个接口，
        子类必须实现它。这里直接委托给 extract_img_feat。
        """
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """训练阶段的 3D 检测分支前向传播。

        【这里是调用 BEVFormerHead 的第一个入口（训练）】

        调用链：
          本函数
            → self.pts_bbox_head(...)         # 调用 BEVFormerHead.forward()
                  → Encoder：图像特征 → BEV 特征图
                  → Decoder：Object Query → 检测预测
                  → 返回 outs 字典
            → self.pts_bbox_head.loss(...)    # 调用 BEVFormerHead.loss()
                  → 匈牙利匹配：预测框 vs GT 框
                  → Focal Loss（分类）+ L1 Loss（定位）
                  → 返回损失字典

        Args:
            pts_feats (list[Tensor]): 图像特征（虽然叫 pts，实为图像特征，历史命名问题）
            gt_bboxes_3d (list): GT 3D 边界框列表（每个样本一个 LiDARInstance3DBoxes）
            gt_labels_3d (list[Tensor]): GT 类别标签列表
            img_metas (list[dict]): 图像元信息（内外参矩阵等）
            gt_bboxes_ignore: 需要忽略的 GT 框（遮挡严重等），通常为 None
            prev_bev (Tensor, optional): 上一帧的 BEV 特征，用于时序自注意力

        Returns:
            dict: 损失字典，例如：
                {'loss_cls': 0.3, 'loss_bbox': 0.7,
                 'd0.loss_cls': 0.5, 'd0.loss_bbox': 1.1, ...}
        """
        # ★ 第一次调用 BEVFormerHead：执行完整前向传播
        # self.pts_bbox_head 就是 BEVFormerHead 实例（由父类初始化时创建）
        # 调用它等同于调用 BEVFormerHead.__call__() → BEVFormerHead.forward()
        outs = self.pts_bbox_head(
            pts_feats,   # 多尺度图像特征（BEV Encoder 的输入）
            img_metas,   # 相机内外参，用于将 BEV 参考点投影到图像上
            prev_bev     # 上一帧 BEV 特征（时序融合用，可为 None）
        )
        # outs 是 BEVFormerHead.forward() 返回的字典，包含：
        # {
        #   'bev_embed':      BEV 特征图 [B, bev_h*bev_w, C]
        #   'all_cls_scores': 所有 Decoder 层的分类分数 [num_dec, B, 900, num_cls]
        #   'all_bbox_preds': 所有 Decoder 层的框预测   [num_dec, B, 900, 10]
        #   'enc_cls_scores': None（单阶段模式）
        #   'enc_bbox_preds': None（单阶段模式）
        # }

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]

        # ★ 第二次调用 BEVFormerHead：计算损失
        # BEVFormerHead.loss() 会：
        #   1. 对每个 Decoder 层调用 loss_single()
        #   2. loss_single() 内部做匈牙利匹配（预测框 ↔ GT 框）
        #   3. 计算 Focal Loss（分类）+ L1 Loss（位置回归）
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        """虚拟前向传播，仅用于验证模型结构是否正确（不涉及真实数据）。
        传入 dummy_metas=None 绕过元信息处理，常用于调试或测量推理速度。
        """
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """统一的前向传播入口，根据 return_loss 自动分发到训练或推理分支。

        【mmdet 框架约定】
        - 训练时（return_loss=True）：  调用 forward_train，返回损失字典
        - 推理时（return_loss=False）：  调用 forward_test，返回检测结果

        注：训练和推理的输入数据格式不同，由框架在调用 forward() 前自动准备好。
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """（训练专用）迭代计算历史帧的 BEV 特征，不保存梯度。

        【为什么需要这个函数？】
        BEVFormer 的时序注意力需要"上一帧的 BEV 特征"作为参考。
        但若把所有历史帧都放进计算图，梯度链条会很长，占用大量显存。

        解决方案：用 torch.no_grad() 提前把历史帧的 BEV 特征算出来，
        只保留最后一帧的 BEV 结果，作为当前帧训练时的 prev_bev 输入。

        【这里是调用 BEVFormerHead 的第二个入口（仅生成 BEV）】
        用 only_bev=True 参数告诉 BEVFormerHead：
          "只需要运行 Encoder 生成 BEV 特征图，不需要 Decoder 解码检测框"
        这样节省了 Decoder 的计算量。

        【迭代逻辑示例】假设 len_queue=3（历史队列3帧）：
          第0帧（prev_bev=None） → 调用 BEVFormerHead(only_bev=True) → BEV_0
          第1帧（prev_bev=BEV_0）→ 调用 BEVFormerHead(only_bev=True) → BEV_1
          第2帧（prev_bev=BEV_1）→ 调用 BEVFormerHead(only_bev=True) → BEV_2
          返回 BEV_2，作为当前帧（第3帧）训练时的 prev_bev

        Args:
            imgs_queue (Tensor): 历史帧图像，shape [B, len_queue, N_cam, C, H, W]
            img_metas_list (list[list[dict]]): 历史帧的元信息

        Returns:
            Tensor: 最后一历史帧的 BEV 特征，shape [B, bev_h*bev_w, C_bev]
        """
        # 切换到 eval 模式：关闭 Dropout，BatchNorm 使用统计均值
        # 这样历史帧 BEV 的计算更稳定（不受训练时随机性影响）
        self.eval()

        with torch.no_grad():   # 不计算梯度，节省显存和计算量
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape

            # 把 [B, len_queue, N_cam, C, H, W] 压平为 [B*len_queue, N_cam, C, H, W]
            # 目的：一次性批量提取所有历史帧的图像特征（效率更高）
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            # img_feats_list 中每个 Tensor shape = [B, len_queue, N_cam, C', H', W']

            # 按时间顺序逐帧处理，每帧都用前一帧的 BEV 作为时序参考
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]

                if not img_metas[0]['prev_bev_exists']:
                    # 若当前帧没有历史帧（场景起始），清空 prev_bev
                    prev_bev = None

                # 取第 i 帧在所有尺度上的特征：
                # each_scale[:, i] 从 [B, len_queue, N_cam, C', H', W'] 取第 i 个时间步
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]

                # ★ 调用 BEVFormerHead（only_bev 模式）
                # only_bev=True → BEVFormerHead.forward() 只运行 Encoder，返回 BEV 特征
                # 不运行 Decoder（节省历史帧的计算量）
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)

            # 恢复训练模式（重新启用 Dropout、BatchNorm 训练行为）
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,            # 点云数据（BEVFormer 不用，为 None）
                      img_metas=None,         # 元信息，每个元素对应 batch 中一个样本的多帧信息
                      gt_bboxes_3d=None,      # GT 3D 边界框（当前帧）
                      gt_labels_3d=None,      # GT 类别标签（当前帧）
                      gt_labels=None,         # GT 2D 标签（可选）
                      gt_bboxes=None,         # GT 2D 框（可选）
                      img=None,               # 图像序列 [B, T, N_cam, C, H, W]
                                              # T = len_queue（队列长度，含当前帧）
                      proposals=None,         # RPN 候选框（不用）
                      gt_bboxes_ignore=None,  # 忽略的 GT 框
                      img_depth=None,         # 深度图（可选）
                      img_mask=None,          # 图像掩码（可选）
                      ):
        """训练主流程。

        【流程概述】
        Step 1. 分离 img：历史帧 img[:,:-1] 和 当前帧 img[:,-1]
        Step 2. obtain_history_bev(历史帧) → prev_bev（无梯度）
        Step 3. extract_feat(当前帧) → 当前帧图像特征（有梯度）
        Step 4. forward_pts_train(图像特征, GT, prev_bev) → 损失

        【梯度策略】
        - 历史帧：用 no_grad 计算 BEV，只当"背景信息"用，不参与反向传播
        - 当前帧：正常参与反向传播，通过梯度更新模型权重

        Returns:
            dict: 总损失字典，如 {'loss_cls': 0.3, 'loss_bbox': 0.7, ...}
        """
        # ── Step 1: 分离历史帧和当前帧 ──────────────────────────────────
        len_queue = img.size(1)           # 时间队列总长度（含当前帧）
        prev_img = img[:, :-1, ...]       # 历史帧：取除最后一帧外的所有帧
        img = img[:, -1, ...]             # 当前帧：取最后一帧，shape=[B, N_cam, C, H, W]

        # ── Step 2: 计算历史帧的 BEV 特征（无梯度）──────────────────────
        prev_img_metas = copy.deepcopy(img_metas)
        # deepcopy：防止 obtain_history_bev 内部修改 img_metas 影响后续使用
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        # ── Step 3: 提取当前帧的图像特征（有梯度）──────────────────────
        img_metas = [each[len_queue-1] for each in img_metas]
        # 取每个样本最后一帧（当前帧）的元信息
        if not img_metas[0]['prev_bev_exists']:
            # 若场景刚开始（前面没有历史帧），不使用 prev_bev
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # ── Step 4: 前向传播 + 计算损失 ─────────────────────────────────
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d,
            img_metas, gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        """推理主流程（逐帧在线处理，维护跨帧状态）。

        【与训练的本质区别】
        训练：每次调用处理完整的时间窗口（当前帧 + 若干历史帧）
        推理：只处理当前帧，通过 self.prev_frame_info 缓存上一帧信息

        【BEV 坐标对齐的必要性】
        上一帧的 BEV 特征是以"上一帧自车位置"为中心建立的坐标系。
        当前帧自车已经移动了，两帧的坐标系不重合。
        需要把"移动了多少（Δpos）、转向了多少（Δangle）"告诉 BEVFormerHead，
        让它在时序注意力中自动做坐标对齐。

        Args:
            img_metas (list[list[dict]]): 推理时为双层嵌套格式
            img: 当前帧图像
        """
        # 输入格式验证
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # ── Step 1: 检测场景切换，若切换则清空历史 BEV ──────────────────
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # scene_token 是 nuScenes 数据集中每段驾驶场景的唯一标识符
            # 新场景的第一帧不能使用上一场景的 BEV 特征
            self.prev_frame_info['prev_bev'] = None
        # 更新缓存的 scene_token 为当前帧
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # ── Step 2: 根据模式决定是否使用时序 BEV ────────────────────────
        if not self.video_test_mode:
            # 单帧模式：不使用历史 BEV，每帧独立推理
            self.prev_frame_info['prev_bev'] = None

        # ── Step 3: 计算自车位姿增量（Δpos, Δangle）────────────────────
        # can_bus 是 CAN 总线数据（车辆传感器实时数据流）：
        #   can_bus[:3] = 自车的全局 3D 位置 [x, y, z]（单位：米）
        #   can_bus[-1] = 自车的偏航角（yaw angle，车头朝北为0，顺时针增大）
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])    # 保存当前帧绝对位置
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])  # 保存当前帧绝对朝向角

        if self.prev_frame_info['prev_bev'] is not None:
            # 有上一帧：把 can_bus 改成"相对增量"（当前 - 上一帧）
            # Δpos   = 当前位置 - 上一帧位置（自车移动了多少米）
            # Δangle = 当前朝向 - 上一帧朝向（自车转向了多少弧度）
            # BEVFormerHead 的时序自注意力模块用这两个增量做 BEV 坐标对齐
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            # 没有上一帧（场景第一帧或单帧模式）：增量为 0，不做坐标偏移
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        # ── Step 4: 执行实际推理 ─────────────────────────────────────────
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0],
            prev_bev=self.prev_frame_info['prev_bev'], **kwargs)

        # ── Step 5: 更新缓存，供下一帧使用 ─────────────────────────────
        self.prev_frame_info['prev_pos'] = tmp_pos       # 保存本帧绝对位置
        self.prev_frame_info['prev_angle'] = tmp_angle   # 保存本帧绝对朝向角
        self.prev_frame_info['prev_bev'] = new_prev_bev  # 保存本帧 BEV 特征
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """推理阶段的 3D 检测分支前向传播。

        【这里是调用 BEVFormerHead 的第三个入口（推理）】

        调用链：
          本函数
            → self.pts_bbox_head(...)           # 调用 BEVFormerHead.forward()
                  → Encoder + Decoder → outs 字典
            → self.pts_bbox_head.get_bboxes(...)# 调用 BEVFormerHead.get_bboxes()
                  → bbox_coder.decode：归一化坐标 → 真实坐标（米）
                  → 阈值过滤低置信度框
                  → 返回 [bboxes, scores, labels] 列表

        Args:
            x (list[Tensor]): 多尺度图像特征
            img_metas: 图像元信息
            prev_bev: 上一帧 BEV 特征（可为 None）
            rescale (bool): 是否将框缩放回原图尺寸（3D 检测一般不需要）

        Returns:
            tuple:
                - new_prev_bev (Tensor): 本帧生成的 BEV 特征（供下一帧用）
                - bbox_results (list[dict]): 格式化检测结果列表
        """
        # ★ 调用 BEVFormerHead.forward()：生成 BEV + 预测检测结果
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        # outs['bev_embed']：本帧 BEV 特征，下一帧时序融合用
        # outs['all_cls_scores']、outs['all_bbox_preds']：各层分类/回归预测

        # ★ 调用 BEVFormerHead.get_bboxes()：将预测解码为真实 3D 框
        # 内部步骤：
        #   1. 取最后一层 Decoder 的输出（最精准的预测）
        #   2. sigmoid 得到分类概率
        #   3. 按阈值过滤（保留高置信度的框）
        #   4. 坐标从 [0,1] 归一化空间 → 真实世界坐标（米）
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        # bbox3d2result：将 (bboxes, scores, labels) 三元组打包为标准字典
        # 输出：[{'boxes_3d': LiDARBox, 'scores_3d': Tensor, 'labels_3d': Tensor}, ...]
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        # 同时返回 BEV 特征（供下一帧时序融合）和检测结果
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """单帧完整推理流程（无数据增强）。

        【流程】
        原始图像 → extract_feat → 图像特征 → simple_test_pts → BEV特征 + 检测结果

        Args:
            img_metas: 图像元信息
            img (Tensor, optional): 原始图像，shape [B, N_cam, C, H, W]
            prev_bev: 上一帧 BEV 特征
            rescale: 是否缩放框坐标

        Returns:
            tuple:
                - new_prev_bev (Tensor): 本帧 BEV 特征（给下一帧用）
                - bbox_list (list[dict]): 检测结果，每个 dict 含 'pts_bbox' 键
        """
        # 提取图像特征（backbone + neck）
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # 初始化结果容器：每个 batch 样本对应一个空字典
        bbox_list = [dict() for i in range(len(img_metas))]

        # 3D 检测推理：返回 BEV 特征和检测结果
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)

        # 将检测结果填入字典的 'pts_bbox' 键
        # 命名沿用了点云检测惯例（pts = points），实为图像感知的结果
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return new_prev_bev, bbox_list
    