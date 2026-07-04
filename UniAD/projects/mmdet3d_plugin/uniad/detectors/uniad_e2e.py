#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# ================================================================================
# 文件概述：UniAD —— 端到端自动驾驶总模型（Stage 2 完整版）
#
# 这个文件是整个 UniAD 系统的"总指挥"，把所有子任务模块串联在一起：
#
#   多相机图像输入
#       │
#       ▼
#   UniADTrack（父类）
#   ├── BEVFormer：将多相机图像融合为统一的 BEV（鸟瞰图）特征
#   └── TrackHead：目标检测 + 多目标跟踪
#       │  输出：bev_embed（BEV特征）、track_query（目标查询）
#       ▼
#   SegHead（地图分割）
#       │  输出：车道线、道路边界等地图要素
#       ▼
#   MotionHead（运动预测）
#       │  输出：每个目标的多模态轨迹预测、traj_query
#       ▼
#   OccHead（占用预测）
#       │  输出：未来各时间步的 BEV 占用掩码
#       ▼
#   PlanningHead（轨迹规划）
#       │  输出：自车未来轨迹（x, y 坐标序列）
#       ▼
#   最终输出：各任务损失（训练）/ 各任务预测结果（推理）
#
# 设计理念："Planning-oriented"（规划导向）
#   不是独立训练各个子任务，而是以"如何让自车开得更好"为最终目标，
#   让前序任务（检测、跟踪、预测）的结果直接服务于最终的规划任务。
#
# 类继承关系：
#   MVXTwoStageDetector（mmdet3d基类）
#       └── UniADTrack（Stage1：检测+跟踪）
#               └── UniAD（Stage2：完整端到端，本文件）
# ================================================================================

import torch
from mmcv.runner import auto_fp16   # 自动混合精度装饰器（加速训练）
from mmdet.models import DETECTORS  # 检测器注册表
import copy
import os
from ..dense_heads.seg_head_plugin import IOU     # 交并比（Intersection over Union）评估指标
from .uniad_track import UniADTrack               # 父类：包含 BEVFormer + 检测 + 跟踪
from mmdet.models.builder import build_head       # 根据配置字典构建 Head 模块


@DETECTORS.register_module()
class UniAD(UniADTrack):
    """UniAD 端到端自动驾驶模型（完整版）。
    
    继承 UniADTrack（已包含 BEVFormer + 检测 + 跟踪），
    在其基础上添加：地图分割、运动预测、占用预测、轨迹规划四个子任务头。
    
    训练时：返回所有子任务的加权损失之和
    推理时：返回所有子任务的预测结果
    
    Args:
        seg_head (dict): 地图分割头配置（SegHead，预测车道线等地图要素）
        motion_head (dict): 运动预测头配置（MotionHead，预测其他目标的未来轨迹）
        occ_head (dict): 占用预测头配置（OccHead，预测 BEV 占用掩码）
        planning_head (dict): 规划头配置（PlanningHead，预测自车轨迹）
        task_loss_weight (dict): 各子任务损失的权重
            默认所有任务权重均为 1.0
    """
    def __init__(
        self,
        seg_head=None,       # 地图分割头（可选）
        motion_head=None,    # 运动预测头（可选）
        occ_head=None,       # 占用预测头（可选）
        planning_head=None,  # 规划头（可选）
        task_loss_weight=dict(
            track=1.0,       # 跟踪任务损失权重
            map=1.0,         # 地图分割任务损失权重
            motion=1.0,      # 运动预测任务损失权重
            occ=1.0,         # 占用预测任务损失权重
            planning=1.0     # 规划任务损失权重
        ),
        **kwargs,  # 传给父类 UniADTrack 的其他参数（BEVFormer、TrackHead 等配置）
    ):
        # 调用父类 UniADTrack 的初始化（构建 BEVFormer + TrackHead）
        super(UniAD, self).__init__(**kwargs)
        
        # 根据配置字典构建各子任务头（如果配置不为 None 则构建）
        if seg_head:
            self.seg_head = build_head(seg_head)        # 构建地图分割头
        if occ_head:
            self.occ_head = build_head(occ_head)        # 构建占用预测头
        if motion_head:
            self.motion_head = build_head(motion_head)  # 构建运动预测头
        if planning_head:
            self.planning_head = build_head(planning_head)  # 构建规划头
        
        self.task_loss_weight = task_loss_weight
        # 验证权重字典包含且仅包含这5个任务
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}

    # =========================================================================
    # 属性检查：判断各子任务头是否存在
    # 用法：if self.with_motion_head: ...（避免未配置某个头时报错）
    # =========================================================================

    @property
    def with_planning_head(self):
        """是否配置了规划头。"""
        return hasattr(self, 'planning_head') and self.planning_head is not None
    
    @property
    def with_occ_head(self):
        """是否配置了占用预测头。"""
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        """是否配置了运动预测头。"""
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        """是否配置了地图分割头。"""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_dummy(self, img):
        """用于模型结构分析的虚拟前向传播（不计算实际结果）。"""
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """统一入口：根据 return_loss 决定调用训练还是测试前向传播。
        
        训练时：return_loss=True  → 调用 forward_train()，返回损失字典
        推理时：return_loss=False → 调用 forward_test()，返回预测结果
        
        注意输入格式的区别：
          训练时：img 和 img_metas 是单层嵌套（Tensor 和 list[dict]）
          测试时：img 和 img_metas 是双层嵌套（list[Tensor] 和 list[list[dict]]）
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))  # 对 img 和 points 使用 FP16 半精度（节省显存、加速）
    def forward_train(self,
                      # ---- 基础输入 ----
                      img=None,            # 多相机图像，shape (B, len_queue, N_cam, C, H, W)
                                           # len_queue：时序帧数，N_cam：相机数（通常6个）
                      img_metas=None,      # 图像元信息（相机内外参、时间戳等）
                      
                      # ---- 跟踪任务 GT ----
                      gt_bboxes_3d=None,   # GT 3D 检测框列表
                      gt_labels_3d=None,   # GT 3D 检测框类别标签
                      gt_inds=None,        # GT 目标的实例 ID（用于跟踪匹配）
                      l2g_t=None,          # Local→Global 坐标变换的平移向量
                      l2g_r_mat=None,      # Local→Global 坐标变换的旋转矩阵
                      timestamp=None,      # 每帧的时间戳（用于速度计算）
                      
                      # ---- 地图分割任务 GT ----
                      gt_lane_labels=None,  # GT 车道线类别标签
                      gt_lane_bboxes=None,  # GT 车道线边界框
                      gt_lane_masks=None,   # GT 车道线掩码
                      
                      # ---- 运动预测任务 GT ----
                      gt_fut_traj=None,          # GT 其他目标的未来轨迹（相对坐标）
                      gt_fut_traj_mask=None,      # GT 未来轨迹的有效性掩码
                      gt_past_traj=None,          # GT 其他目标的历史轨迹
                      gt_past_traj_mask=None,     # GT 历史轨迹的有效性掩码
                      gt_sdc_bbox=None,           # GT 自车检测框
                      gt_sdc_label=None,          # GT 自车类别标签
                      gt_sdc_fut_traj=None,       # GT 自车未来轨迹
                      gt_sdc_fut_traj_mask=None,  # GT 自车未来轨迹有效性掩码
                      
                      # ---- 占用预测任务 GT ----
                      gt_segmentation=None,       # GT 语义分割图（有无目标占用）
                      gt_instance=None,           # GT 实例分割图（各目标的占用区域）
                      gt_occ_img_is_valid=None,   # GT 各帧是否有效（序列完整性标记）
                      
                      # ---- 规划任务 GT ----
                      sdc_planning=None,       # GT 自车规划轨迹（用于监督）
                      sdc_planning_mask=None,  # GT 自车规划轨迹的有效性掩码
                      command=None,            # 高层导航指令（直行/左转/右转）
                      gt_future_boxes=None,    # GT 未来时刻的目标框（用于规划碰撞检测）
                      **kwargs,
                      ):
        """训练前向传播：按顺序执行5个子任务，汇总所有损失后返回。
        
        执行顺序（严格串行，后者依赖前者输出）：
          1. TrackHead  → bev_embed + track_query（BEV特征 + 跟踪查询）
          2. SegHead    → 地图分割损失（不影响后续输入）
          3. MotionHead → 轨迹预测损失 + traj_query（轨迹查询供后续使用）
          4. OccHead    → 占用预测损失（依赖 traj_query）
          5. PlanningHead → 规划损失（依赖 traj_query + occ_mask）
        
        Returns:
            dict: 损失字典，键名格式为 "{任务前缀}.{损失名称}"
                例如：{'track.loss_cls': 0.3, 'motion.loss_traj': 0.1, ...}
                最终合并为一个标量供优化器反向传播
        """
        losses = dict()  # 汇总所有任务损失的字典
        
        # img.size(1)：时序帧数（len_queue），通常为3（使用过去3帧）
        len_queue = img.size(1)

        # =====================================================================
        # Step 1: 跟踪任务（父类 UniADTrack 实现）
        # =====================================================================
        # forward_track_train 完成两件事：
        #   a. BEVFormer：多相机图像 → BEV 特征图（200×200×256）
        #   b. TrackHead：BEV 特征 → 目标检测框 + 跟踪 Query
        # 输出：
        #   losses_track：跟踪任务的损失
        #   outs_track：包含 bev_embed、track_query、track_query_pos 等
        losses_track, outs_track = self.forward_track_train(
            img, gt_bboxes_3d, gt_labels_3d,
            gt_past_traj, gt_past_traj_mask,
            gt_inds, gt_sdc_bbox, gt_sdc_label,
            l2g_t, l2g_r_mat, img_metas, timestamp
        )
        # 给跟踪损失加上 "track." 前缀，并乘以权重
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)  # 合并到总损失字典
        
        # 如果是 tiny 版本模型，对 BEV 特征进行上采样（恢复分辨率）
        outs_track = self.upsample_bev_if_tiny(outs_track)

        # 提取后续任务需要的关键输出
        bev_embed = outs_track["bev_embed"]  # BEV 特征图，(H*W, B, 256)，用于所有后续任务
        bev_pos   = outs_track["bev_pos"]    # BEV 位置编码，配合 Transformer 使用

        # 数值安全保护：上一个 iter 的梯度爆炸可能使模型参数含有 NaN/Inf，
        # 导致本 iter 的 BEV 特征输出就带有 NaN，进而传染到所有子头。
        # 在共享 BEV 特征流入各子头之前统一截断，保证各子头 forward 不受污染。
        bev_embed = torch.nan_to_num(bev_embed, nan=0.0, posinf=1e4, neginf=-1e4)

        # 只取最后一帧的 img_metas（前几帧仅用于构建历史 BEV，不参与后续任务）
        img_metas = [each[len_queue-1] for each in img_metas]

        # =====================================================================
        # Step 2: 地图分割任务（SegHead）
        # =====================================================================
        # SegHead 预测 BEV 中的地图要素：车道线、道路边界、人行横道等
        outs_seg = dict()
        if self.with_seg_head:          
            losses_seg, outs_seg = self.seg_head.forward_train(
                bev_embed,      # BEV 特征（共享输入）
                img_metas,
                gt_lane_labels, gt_lane_bboxes, gt_lane_masks  # 地图 GT
            )
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)

        # =====================================================================
        # Step 3: 运动预测任务（MotionHead）
        # =====================================================================
        # MotionHead 预测场景中所有目标（含自车）的多模态未来轨迹
        # 关键输出 outs_motion 包含：
        #   - traj_query：轨迹查询（传给 OccHead）
        #   - track_query、track_query_pos：更新后的目标查询
        #   - sdc_traj_query：自车轨迹查询（传给 PlanningHead）
        outs_motion = dict()
        if self.with_motion_head:
            ret_dict_motion = self.motion_head.forward_train(
                bev_embed,
                gt_bboxes_3d, gt_labels_3d,
                gt_fut_traj, gt_fut_traj_mask,          # 其他目标的轨迹 GT
                gt_sdc_fut_traj, gt_sdc_fut_traj_mask,  # 自车轨迹 GT
                outs_track=outs_track,   # 来自 TrackHead 的跟踪 Query
                outs_seg=outs_seg        # 来自 SegHead 的地图特征（用于运动-地图交互）
            )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos  # 注入位置编码供后续使用
            losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            losses.update(losses_motion)

        # =====================================================================
        # Step 4: 占用预测任务（OccHead）
        # =====================================================================
        # OccHead 预测未来各时间步的 BEV 占用掩码（每个目标在哪些格子）
        # 依赖 outs_motion 中的 traj_query、track_query 等
        if self.with_occ_head:
            # 特殊情况处理：场景中没有被跟踪的目标时（track_query 为空）
            # 用零向量填充，避免后续运算报错
            if outs_motion['track_query'].shape[1] == 0:
                # TODO: rm hard code（未来应该从配置中读取这些维度）
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]  # -1 表示没有匹配的 GT
            
            losses_occ = self.occ_head.forward_train(
                bev_embed,
                outs_motion,                      # 包含 traj_query 等运动预测输出
                gt_inds_list=gt_inds,             # GT 实例 ID（用于匈牙利匹配对齐）
                gt_segmentation=gt_segmentation,  # GT 语义分割（有无占用）
                gt_instance=gt_instance,          # GT 实例分割（谁占了哪里）
                gt_img_is_valid=gt_occ_img_is_valid,  # 帧有效性标记
            )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)

        # =====================================================================
        # Step 5: 规划任务（PlanningHead）
        # =====================================================================
        # PlanningHead 根据 BEV 特征、运动预测输出、高层指令，
        # 预测自车未来轨迹（6步 × 0.5s = 3秒）
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(
                bev_embed,
                outs_motion,          # 包含 sdc_traj_query（自车轨迹查询）和 occ 相关信息
                sdc_planning,         # GT 自车轨迹（监督信号）
                sdc_planning_mask,    # GT 自车轨迹有效性
                command,              # 高层指令（直行/左转/右转）
                gt_future_boxes       # GT 未来目标框（碰撞检测用）
            )
            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)
        
        # =====================================================================
        # 后处理：将 NaN 替换为 0，防止某个任务的 loss 为 NaN 导致整体训练崩溃
        # =====================================================================
        for k, v in losses.items():
            losses[k] = torch.nan_to_num(v)
        
        return losses  # 返回所有任务的加权损失字典（用于反向传播）
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        """对损失字典中的每个损失乘以对应任务权重，并添加任务前缀。
        
        例如：
          输入: {'loss_cls': 0.5, 'loss_bbox': 0.3}, prefix='track'
          输出: {'track.loss_cls': 0.5 × weight, 'track.loss_bbox': 0.3 × weight}
        
        任务权重来自 task_loss_weight 配置，允许调整各任务在总损失中的占比。
        例如，如果规划任务更重要，可以把 planning 的权重设为 2.0。
        
        Args:
            loss_dict (dict): 某个任务的损失字典
            prefix (str): 任务前缀名称（'track'/'map'/'motion'/'occ'/'planning'）
        
        Returns:
            dict: 添加前缀和权重缩放后的损失字典
        """
        loss_factor = self.task_loss_weight[prefix]  # 获取该任务的权重
        # 字典推导式：给每个 key 加前缀，给每个 value 乘权重
        loss_dict = {f"{prefix}.{k}": v * loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     # ---- 基础输入 ----
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     
                     # ---- 地图分割 GT（仅用于评估，不参与推理）----
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     
                     # ---- 规划 GT（仅用于评估）----
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,  # 高层指令（推理时作为条件输入）
 
                     # ---- 占用预测 GT（仅用于评估）----
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     **kwargs
                    ):
        """推理/测试前向传播：按顺序执行各子任务，返回预测结果字典。
        
        与 forward_train 的主要区别：
          1. 不计算损失，直接返回预测结果
          2. 处理时序状态：维护 prev_frame_info（上一帧的位置、角度、BEV特征）
          3. 支持在线推理：逐帧处理，利用历史帧信息改善当前帧预测
        
        时序状态处理：
          - 每个场景（scene_token 改变）的第一帧，清空历史信息
          - 后续帧利用上一帧的 BEV 特征（prev_bev）做时序融合
          - can_bus 记录自车运动信息（用于 BEV 坐标对齐）
        
        Returns:
            list[dict]: 每个样本的预测结果，包含：
                'token': 当前帧的唯一标识
                跟踪结果、地图分割结果、运动预测结果
                'occ': 占用预测结果（seg_out、ins_seg_out 等）
                'planning': 规划结果（planning_gt、result_planning）
        """
        # 输入格式验证
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # =====================================================================
        # 时序状态管理：处理场景切换和自车运动补偿
        # =====================================================================
        # 检查是否切换到新场景（scene_token 改变 = 数据集中的新段）
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # 新场景第一帧：清空历史 BEV（不能用上一个场景的特征）
            self.prev_frame_info['prev_bev'] = None
        # 更新当前场景 token
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # 如果不使用时序模式（video_test_mode=False），每帧都当作独立处理
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # ---- 自车运动补偿（can_bus）----
        # can_bus 是 CAN 总线数据，记录自车的绝对位置和朝向
        # BEVFormer 需要的是相对运动（两帧之间的位移和旋转角变化）
        # 所以这里把绝对坐标转换为相对于上一帧的增量
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])    # 当前帧绝对位置
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])  # 当前帧绝对角度
        
        if self.prev_frame_info['scene_token'] is None:
            # 第一帧：没有上一帧，位移和角度增量设为0
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        else:
            # 后续帧：用当前值减去上一帧的值，得到增量
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        
        # 保存当前帧的绝对位置/角度，供下一帧使用
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        # 解包（测试时输入是双层嵌套的，这里取第一层）
        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        # =====================================================================
        # Step 1: 跟踪任务
        # =====================================================================
        result = [dict() for i in range(len(img_metas))]  # 初始化结果列表
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)
        # result_track[0] 包含：bev_embed、track_query、检测框等

        # tiny 模型的 BEV 上采样
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]  # BEV 特征，供所有后续任务使用

        # =====================================================================
        # Step 2: 地图分割任务
        # =====================================================================
        if self.with_seg_head:
            result_seg = self.seg_head.forward_test(
                bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale
            )
            # result_seg[0]：地图分割预测结果（车道线等）

        # =====================================================================
        # Step 3: 运动预测任务
        # =====================================================================
        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(
                bev_embed,
                outs_track=result_track[0],  # 跟踪 Query
                outs_seg=result_seg[0]        # 地图特征
            )
            outs_motion['bev_pos'] = result_track[0]['bev_pos']  # 注入位置编码

        # =====================================================================
        # Step 4: 占用预测任务
        # =====================================================================
        outs_occ = dict()
        if self.with_occ_head:
            # 检查是否有有效的跟踪目标
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            
            outs_occ = self.occ_head.forward_test(
                bev_embed,
                outs_motion,
                no_query=occ_no_query,          # 没有目标时直接返回全零预测
                gt_segmentation=gt_segmentation,  # GT（仅用于评估对比）
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ  # 保存占用预测结果
        
        # =====================================================================
        # Step 5: 规划任务
        # =====================================================================
        if self.with_planning_head:
            # 打包规划任务的 GT（用于评估，不影响推理）
            planning_gt = dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
            result_planning = self.planning_head.forward_test(
                bev_embed,
                outs_motion,   # 包含自车轨迹 Query（sdc_traj_query）
                outs_occ,      # 占用掩码（用于碰撞避免优化）
                command        # 高层指令（直行/左转/右转）
            )
            result[0]['planning'] = dict(
                planning_gt=planning_gt,          # 保存 GT 用于评估
                result_planning=result_planning,  # 保存规划预测结果
            )

        # =====================================================================
        # 后处理：清理不需要返回的中间变量（节省内存，避免传输大张量）
        # =====================================================================
        # 跟踪结果：删除大型中间张量（prev_bev、各种 query embedding）
        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        # 地图分割：删除不需要的中间结果
        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        
        # 运动预测：删除所有 query（不需要返回 query 向量）
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        
        # 占用预测：删除大型中间张量（只保留最终的 seg_out 和 ins_seg_out）
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(
                result[0]['occ'],
                pop_list=[
                    'seg_out_mask', 'flow_out', 'future_states_occ',
                    'pred_ins_masks', 'pred_raw_occ',
                    'pred_ins_logits',    # 原始 logit（体积大，不需要返回）
                    'pred_ins_sigmoid'    # sigmoid 概率图（体积大，不需要返回）
                ]
            )
        
        # =====================================================================
        # 汇总各任务结果到最终字典
        # =====================================================================
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']  # 保存帧标识（用于评估时对齐）
            res.update(result_track[i])    # 合并跟踪结果
            if self.with_motion_head:
                res.update(result_motion[i])  # 合并运动预测结果
            if self.with_seg_head:
                res.update(result_seg[i])     # 合并地图分割结果
            # occ 和 planning 已经直接写入 result[i]

        return result  # 返回预测结果列表，每个元素对应一个样本


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    """从任务结果字典中删除不需要返回的中间变量。
    
    删除规则：
      1. 自动删除：key 以 'query'、'query_pos' 或 'embedding' 结尾的项
         （这些是 Transformer 的中间 Query 向量，体积大且评估时不需要）
      2. 手动删除：pop_list 中指定的 key
    
    Args:
        task_result (dict): 某个任务的输出字典
        pop_list (list, optional): 需要额外删除的 key 列表
    
    Returns:
        dict: 删除后的字典（原地修改）
    """
    all_keys = list(task_result.keys())
    for k in all_keys:
        # 自动删除以这些后缀结尾的 key（都是大型 Query 张量）
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)  # None：key 不存在时不报错
    return task_result