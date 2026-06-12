#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# ================================================================================
# 文件概述：MotionHead —— 运动预测头
#
# 功能：
#   预测场景中所有被跟踪智能体（车辆、行人等）的未来轨迹，
#   同时也预测 SDC（自动驾驶车辆本身）的未来轨迹。
#
# 核心思想：
#   - 每个智能体有多条"候选轨迹模式"（anchor），对应不同的可能行为（直行/左转/右转等）
#   - 通过 Transformer（MotionFormer）融合：
#       ① BEV 特征（场景环境信息）
#       ② Track Query（每个智能体的历史跟踪特征）
#       ③ Lane Query（车道线/地图信息）
#   - 输出每条候选轨迹的概率分数 + 具体坐标（用高斯分布参数表示不确定性）
#
# 输入输出关系：
#   输入：BEV 特征图 + 跟踪结果（track queries + bboxes） + 地图分割结果（lane queries）
#   输出：每个智能体的多模态轨迹预测（num_modes 条轨迹 + 对应概率）
#
# 在 UniAD 流水线中的位置：
#   TrackHead → SegHead → MotionHead → OccHead → PlanningHead
#   MotionHead 接收 TrackHead 和 SegHead 的输出，为 PlanningHead 提供轨迹查询特征
# ================================================================================

import torch
import copy
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.functional import (
    bivariate_gaussian_activation,  # 将网络输出解码为双变量高斯分布参数
    norm_points,                     # 将点坐标归一化到 [0,1]
    pos2posemb2d,                    # 将2D位置坐标转换为位置嵌入向量
    anchor_coordinate_transform      # 将 anchor 轨迹从智能体坐标系变换到场景坐标系
)
from .motion_head_plugin.motion_utils import nonlinear_smoother  # 非线性优化平滑GT轨迹（训练用）
from .motion_head_plugin.base_motion_head import BaseMotionHead  # 基类，包含网络层的构建逻辑


@HEADS.register_module()
class MotionHead(BaseMotionHead):
    """
    MotionHead：运动预测头，预测场景中所有智能体（含 SDC）的未来轨迹。

    核心机制：多模态轨迹预测（Multi-modal Trajectory Prediction）
      - 对每个智能体，不预测单一轨迹，而是预测 num_anchor 条候选轨迹
      - 每条候选轨迹对应一种"行为模式"（如直行、左转、右转）
      - 同时预测每条轨迹的概率分数，取最高分的作为最终预测结果
      - 这样能捕捉未来的不确定性，比单条轨迹预测更鲁棒

    Anchor（先验轨迹）机制：
      - 用 K-Means 聚类离线生成一组代表性轨迹模板（存储在文件中）
      - 按类别分组（vehicle/pedestrian/cyclist 等行为模式不同）
      - 推理时以 anchor 作为初始参考，网络预测相对于 anchor 的偏移量

    Args:
        predict_steps (int): 预测未来轨迹的时间步数，默认12步（对应12×0.5s=6秒）
        transformerlayers (dict): MotionFormer Transformer 层的配置字典
        bbox_coder: 边界框编解码器
        num_cls_fcs (int): 分类/回归分支的全连接层数量
        bev_h (int): BEV 特征图的高度（格子数）
        bev_w (int): BEV 特征图的宽度（格子数）
        embed_dims (int): Transformer 的嵌入维度（Query/Key/Value 的向量长度）
        num_anchor (int): 每个类别组的 anchor 轨迹数量（候选模式数）
        det_layer_num (int): 检测 Transformer 的层数（用于融合多层 track query）
        group_id_list (list): 类别分组列表，将类别合并为几大组（如车辆组、行人组）
        pc_range: 点云/BEV 的空间范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        use_nonlinear_optimizer (bool): 训练时是否用非线性优化器平滑 GT 轨迹
        anchor_info_path (str): 预先聚类好的 anchor 轨迹文件路径（.pkl 格式）
        vehicle_id_list (list[int]): 车辆类别 ID 列表，用于筛选出车辆智能体
    """
    def __init__(self,
                 *args,
                 predict_steps=12,
                 transformerlayers=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 bev_h=30,
                 bev_w=30,
                 embed_dims=256,
                 num_anchor=6,
                 det_layer_num=6,
                 group_id_list=[],
                 pc_range=None,
                 use_nonlinear_optimizer=False,
                 anchor_info_path=None,
                 loss_traj=dict(),
                 num_classes=0,
                 vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
                 **kwargs):
        super(MotionHead, self).__init__()
        
        # ---- BEV 特征图尺寸 ----
        self.bev_h = bev_h   # BEV 高度（行数）
        self.bev_w = bev_w   # BEV 宽度（列数）
        
        # ---- 网络结构参数 ----
        # num_cls_fcs-1：分类分支的隐藏层数（最后一层单独加，所以减1）
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_cls_fcs - 1  # 回归分支的隐藏层数
        self.embed_dims = embed_dims         # 嵌入向量维度（256）
        
        # ---- anchor 相关参数 ----
        self.num_anchor = num_anchor                      # 每组的 anchor 数量（候选轨迹数）
        self.num_anchor_group = len(group_id_list)        # 类别组数（如车辆组、行人组等）
        
        # ---- 类别 → 组 的映射表 ----
        # 将细粒度类别 ID 映射到粗粒度组 ID
        # 例如：car(0)、truck(1)、bus(2) 都属于 vehicle 组（组0）
        # 这样不同类别的车辆可以共享同一套 anchor 轨迹
        self.cls2group = [0 for i in range(num_classes)]  # 初始化所有类别默认为组0
        for i, grouped_ids in enumerate(group_id_list):
            for gid in grouped_ids:
                self.cls2group[gid] = i   # 将类别 gid 分配到组 i
        self.cls2group = torch.tensor(self.cls2group)  # 转为 tensor 便于 GPU 索引
        
        # ---- 其他参数 ----
        self.pc_range = pc_range                              # BEV 空间范围
        self.predict_steps = predict_steps                    # 预测步数（12）
        self.vehicle_id_list = vehicle_id_list                # 车辆类别 ID 列表
        self.use_nonlinear_optimizer = use_nonlinear_optimizer  # 是否用非线性优化平滑 GT
        
        # ---- 初始化各子模块（继承自 BaseMotionHead）----
        self._load_anchors(anchor_info_path)   # 从文件加载 K-Means anchor 轨迹
        self._build_loss(loss_traj)            # 构建轨迹损失函数
        self._build_layers(transformerlayers, det_layer_num)  # 构建 Transformer 等网络层
        self._init_layers()                    # 初始化分类/回归分支（每个 decoder layer 一套）

    def forward_train(self,
                      bev_embed,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_sdc_fut_traj=None, 
                      gt_sdc_fut_traj_mask=None, 
                      outs_track={},
                      outs_seg={}
                  ):
        """训练模式的前向传播。

        流程：
          1. 从 outs_track 中提取 track query 和 SDC query，拼接在一起统一处理
          2. 调用 forward()（即 __call__）执行 MotionFormer 前向传播，得到轨迹预测
          3. 计算轨迹预测损失
          4. 将 SDC 的轨迹 query 单独提取出来，供后续 PlanningHead 使用
          5. 过滤出车辆类别的 query，用于后续 OccHead

        Args:
            bev_embed (Tensor): BEV 特征图，shape (H*W, B, C)
            gt_bboxes_3d (list): 每张图的 3D GT 边界框
            gt_labels_3d (list): 每张图的类别标签
            gt_fut_traj (list[Tensor]): 其他智能体的 GT 未来轨迹，shape (N_obj, 12, 2)
            gt_fut_traj_mask (list[Tensor]): GT 轨迹的有效性掩码，shape (N_obj, 12, 2)
            gt_sdc_fut_traj (list[Tensor]): SDC 的 GT 未来轨迹，shape (1, 12, 2)
            gt_sdc_fut_traj_mask (list[Tensor]): SDC GT 轨迹掩码
            outs_track (dict): TrackHead 的输出（包含 track_query_embeddings, sdc_embedding 等）
            outs_seg (dict): SegHead 的输出（包含 lane_query 等地图特征）

        Returns:
            dict: 包含以下键：
                - 'losses': 轨迹损失字典
                - 'outs_motion': 运动预测结果（含 sdc_traj_query 等，供后续使用）
                - 'track_boxes': 跟踪边界框结果
        """
        # ============================================================
        # Step 1: 准备 Track Query —— 将普通智能体 query 和 SDC query 拼接
        # ============================================================
        
        # track_query: 所有被跟踪智能体的特征嵌入
        # [None, None, ...] 在前面增加两个维度 → shape: (1, 1, A_track, D)
        # 其中 1 对应 num_dec=1（这里只取最后一层 decoder 的输出），
        # A_track 是被跟踪智能体数量，D=embed_dims=256
        track_query = outs_track['track_query_embeddings'][None, None, ...]  # (1, 1, A_track, D)
        
        # all_matched_idxes: GT 匹配索引（每个 track query 对应哪个 GT 目标）
        # 用列表包一层，方便后续批量处理
        all_matched_idxes = [outs_track['track_query_matched_idxes']]  # [shape: (A_track,)]
        
        # track_boxes: 跟踪到的边界框信息 (bboxes, scores, labels, bbox_index, mask)
        track_boxes = outs_track['track_bbox_results']
        
        # ---- 将 SDC 的信息追加到末尾，统一处理 ----
        # SDC 的 GT 匹配索引设为 gt_fut_traj[0].shape[0]（指向 GT 列表的最后一个位置）
        # gt_fut_traj[0].shape[0] = N_obj，即 SDC 的 GT 轨迹追加在所有目标 GT 的最后
        sdc_match_index = torch.zeros((1,), dtype=all_matched_idxes[0].dtype, device=all_matched_idxes[0].device)
        sdc_match_index[0] = gt_fut_traj[0].shape[0]   # SDC 指向 GT 的最后一个位置
        
        # 将 SDC 的 match index 追加到 all_matched_idxes 末尾
        all_matched_idxes = [torch.cat([all_matched_idxes[0], sdc_match_index], dim=0)]
        
        # 将 SDC 的 GT 未来轨迹追加到其他智能体 GT 的末尾
        gt_fut_traj[0] = torch.cat([gt_fut_traj[0], gt_sdc_fut_traj[0]], dim=0)
        gt_fut_traj_mask[0] = torch.cat([gt_fut_traj_mask[0], gt_sdc_fut_traj_mask[0]], dim=0)
        
        # 将 SDC embedding 追加到 track_query 末尾
        # sdc_embedding: (D,) → [None,None,None,:] → (1,1,1,D)，拼接后 A_track+1
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        # track_query: shape (1, 1, A_track+1, D)，最后一个是 SDC
        
        # 同样将 SDC 的 bbox 信息追加到 track_boxes 末尾
        sdc_track_boxes = outs_track['sdc_track_bbox_results']
        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)
        
        # ============================================================
        # Step 2: 从 SegHead 的输出中提取车道/地图信息
        # ============================================================
        # outs_seg['args_tuple'] 包含了 SegHead 计算中间结果：
        # memory: 编码后的多尺度特征
        # memory_mask: 特征掩码
        # memory_pos: 特征位置编码
        # lane_query: 车道线/地图元素的 Query 特征，shape (B, M, D)
        # lane_query_pos: 车道线 Query 的位置编码
        # hw_lvl: 各尺度特征图的高宽
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
        
        # ============================================================
        # Step 3: 执行 MotionFormer 前向传播（调用 forward() 函数）
        # ============================================================
        # forward() 是核心函数，内部运行 MotionFormer Transformer
        # 输入：BEV特征 + track query + lane query + anchor 位置编码
        # 输出：outs_motion 字典，包含 all_traj_scores, all_traj_preds, traj_query 等
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        
        # ============================================================
        # Step 4: 计算轨迹损失
        # ============================================================
        loss_inputs = [gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, outs_motion, all_matched_idxes, track_boxes]
        losses = self.loss(*loss_inputs)

        # ============================================================
        # Step 5: 分离 SDC query 和其他智能体 query
        # ============================================================
        # 内部辅助函数：根据类别标签过滤出车辆类别的 query
        def filter_vehicle_query(outs_motion, all_matched_idxes, gt_labels_3d, vehicle_id_list):
            """只保留车辆类别的智能体 query，过滤掉行人、骑手等。

            原因：OccHead 主要关注车辆的占用预测，行人等小目标影响不大，
                  过滤后可以减少计算量，提高车辆预测的准确性。
            """
            # 获取每个 track query 对应的 GT 类别标签
            # all_matched_idxes[0]: 每个 query 的 GT 索引
            # gt_labels_3d[0][-1]: 最后一帧的所有 GT 类别标签
            query_label = gt_labels_3d[0][-1][all_matched_idxes[0]]
            
            # 构建车辆掩码：遍历所有车辆类别 ID，命中任一即为车辆
            vehicle_mask = torch.zeros_like(query_label)
            for veh_id in vehicle_id_list:
                vehicle_mask |= query_label == veh_id   # 按位或，只要是车辆类别就标记
            
            # 只保留车辆 query
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, vehicle_mask>0]
            outs_motion['track_query'] = outs_motion['track_query'][:, vehicle_mask>0]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, vehicle_mask>0]
            all_matched_idxes[0] = all_matched_idxes[0][vehicle_mask>0]
            return outs_motion, all_matched_idxes

        # 先将最后一个（SDC）的 match index 移除（因为 SDC 不参与车辆过滤）
        all_matched_idxes[0] = all_matched_idxes[0][:-1]
        
        # 从 outs_motion 中分离 SDC 的轨迹 query（最后一个）
        # sdc_traj_query: SDC 的轨迹 query，供 PlanningHead 使用
        # shape: [n_dec, b, n_mode, d] = [3, 1, 6, 256]
        # 其中 n_mode=num_anchor=6（6条候选轨迹）
        outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]         # [3, 1, 6, 256]
        
        # sdc_track_query: SDC 的 track 特征，供 PlanningHead 使用
        # shape: [b, d] = [1, 256]
        outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]          # [1, 256]
        outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]  # [1, 256]
        
        # 从 traj_query 中去掉最后一个（SDC），剩余的是其他智能体
        # shape: [n_dec, b, nq, n_mode, d] = [3, 1, N_obj, 6, 256]
        outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]            # [3, 1, 3, 6, 256]
        outs_motion['track_query'] = outs_motion['track_query'][:, :-1]             # [1, N_obj, 256]
        outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]     # [1, N_obj, 256]

        # 过滤出车辆类别的 query（OccHead 输入只需要车辆）
        outs_motion, all_matched_idxes = filter_vehicle_query(outs_motion, all_matched_idxes, gt_labels_3d, self.vehicle_id_list)
        outs_motion['all_matched_idxes'] = all_matched_idxes

        ret_dict = dict(losses=losses, outs_motion=outs_motion, track_boxes=track_boxes)
        return ret_dict

    def forward_test(self, bev_embed, outs_track={}, outs_seg={}):
        """测试/推理模式的前向传播。

        与 forward_train 的主要区别：
          - 没有 GT 信息，不计算损失
          - 调用 get_trajs() 生成最终轨迹预测结果（用于评估指标）
          - 同样分离 SDC query 供 PlanningHead 使用

        Args:
            bev_embed (Tensor): BEV 特征图，shape (H*W, B, C)
            outs_track (dict): TrackHead 的输出
            outs_seg (dict): SegHead 的输出

        Returns:
            tuple:
                - traj_results (list[dict]): 每张图的轨迹预测结果（含坐标和分数）
                - outs_motion (dict): 运动特征字典（供后续 OccHead/PlanningHead 使用）
        """
        # ---- 准备 track query，将 SDC 追加到末尾（同 forward_train）----
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        track_boxes = outs_track['track_bbox_results']
        
        # 拼接 SDC embedding（追加到最后一位）
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']

        # 拼接 SDC 的边界框信息
        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)      
        
        # 提取 SegHead 的 lane query
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
        
        # ---- 执行 MotionFormer 前向传播 ----
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        
        # ---- 生成可评估的轨迹预测结果（含坐标和分数） ----
        traj_results = self.get_trajs(outs_motion, track_boxes)
        
        # ---- 整理 track_boxes 信息，设置 SDC 标签为0（车辆类） ----
        bboxes, scores, labels, bbox_index, mask = track_boxes[0]
        outs_motion['track_scores'] = scores[None, :]   # 跟踪置信度分数，shape (1, A_total)
        labels[-1] = 0   # SDC 强制设为类别0（车辆），保证后续车辆过滤时 SDC 不被过滤掉

        # 内部辅助函数：过滤出车辆类别的 query（测试时版本）
        def filter_vehicle_query(outs_motion, labels, vehicle_id_list):
            """测试时过滤非车辆 query。

            注意：与训练版本不同，这里直接用预测的 labels 而非 GT labels 来过滤。
            """
            if len(labels) < 1:  # 如果没有任何智能体（除 SDC 外无目标），直接返回 None
                return None

            # 构建车辆掩码
            vehicle_mask = torch.zeros_like(labels)
            for veh_id in vehicle_id_list:
                vehicle_mask |= labels == veh_id
            
            # 过滤各 query（只保留车辆）
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, vehicle_mask>0]
            outs_motion['track_query'] = outs_motion['track_query'][:, vehicle_mask>0]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, vehicle_mask>0]
            outs_motion['track_scores'] = outs_motion['track_scores'][:, vehicle_mask>0]
            return outs_motion
        
        outs_motion = filter_vehicle_query(outs_motion, labels, self.vehicle_id_list)
        
        # ---- 分离 SDC 的轨迹 query（与 forward_train 相同逻辑） ----
        outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]          # SDC 轨迹 query
        outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]           # SDC track 特征
        outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]   # SDC 位置编码
        
        # 从结果中移除 SDC（最后一个），剩余为其他车辆
        outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
        outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
        outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]
        outs_motion['track_scores'] = outs_motion['track_scores'][:, :-1]

        return traj_results, outs_motion

    @auto_fp16(apply_to=('bev_embed', 'track_query', 'lane_query', 'lane_query_pos', 'lane_query_embed', 'prev_bev'))
    def forward(self, 
                bev_embed, 
                track_query, 
                lane_query, 
                lane_query_pos, 
                track_bbox_results):
        """核心前向传播函数（由 forward_train/forward_test 调用）。

        这是 MotionHead 的核心计算逻辑，执行以下步骤：
          1. 提取每个智能体的中心点位置，生成 track query 的位置编码
          2. 构建多层次 anchor 位置嵌入：
             - agent level：以智能体自身坐标系表示的 anchor（类别相关的运动模式）
             - scene level ego：转换到全局/场景坐标系（含平移变换）
             - scene level offset：转换到场景坐标系（仅旋转，不含平移，即相对偏移）
          3. 用 MotionFormer（多层 Transformer decoder）融合所有信息
          4. 对每层 decoder 的输出分别预测轨迹坐标和分数

        Args:
            bev_embed (Tensor): BEV 特征图，shape (H*W, B, D)
                由 BEVFormer 生成，包含整个场景的鸟瞰视角语义信息
            track_query (Tensor): 跟踪查询特征，shape (1, 1, A_total, D)
                A_total = A_track + 1（普通智能体 + SDC）
                包含了被跟踪目标的历史特征信息
            lane_query (Tensor): 车道线/地图元素的查询特征，shape (B, M, D)
                由 SegHead 生成，包含地图语义信息
            lane_query_pos (Tensor): 车道线 query 的位置编码，shape (B, M, D)
            track_bbox_results (list): 跟踪结果，每个元素为 (bboxes, scores, labels, bbox_index, mask)

        Returns:
            dict: 包含以下键：
                'all_traj_scores': shape (num_layers, B, A_total, num_anchor)
                    每个 decoder 层输出的每条候选轨迹的概率（log-softmax 后）
                'all_traj_preds': shape (num_layers, B, A_total, num_anchor, predict_steps, 5)
                    每条候选轨迹每个时间步的高斯分布参数 (μx, μy, σx, σy, ρ)
                    注：经过 cumsum 后变为绝对坐标
                'valid_traj_masks': shape (B, A_total)，全 True，表示所有 query 有效
                'traj_query': shape (num_layers, B, A_total, num_anchor, D)
                    每层 decoder 输出的轨迹查询特征（供后续 OccHead/PlanningHead 使用）
                'track_query': shape (B, A_total, D)，输入的 track query（原始）
                'track_query_pos': shape (B, A_total, D)，track query 的位置编码
        """
        dtype = track_query.dtype
        device = track_query.device
        num_groups = self.kmeans_anchors.shape[0]  # 类别组数

        # ============================================================
        # Step 1: 取 track_query 最后一层 decoder 的输出
        # ============================================================
        # track_query 原始 shape: (1, 1, A_total, D)（前两维是 batch 相关的）
        # 取最后一帧（-1）后 shape: (1, A_total, D) = (B, A_total, D)
        track_query = track_query[:, -1]
        
        # ============================================================
        # Step 2: 提取每个智能体的中心点位置，生成位置编码
        # ============================================================
        # 从 track_bbox_results 中提取每个 bbox 的中心点 (x, y)，并归一化到 [0,1]
        # reference_points_track: shape (B, A_total, 2)
        reference_points_track = self._extract_tracking_centers(
            track_bbox_results, self.pc_range)
        
        # pos2posemb2d: 将2D坐标转换为高维位置嵌入向量（类似 sinusoidal 位置编码）
        # boxes_query_embedding_layer: MLP，将位置嵌入映射到 embed_dims 维度
        # track_query_pos: shape (B, A_total, D)，每个智能体的位置编码
        track_query_pos = self.boxes_query_embedding_layer(pos2posemb2d(reference_points_track.to(device)))
        
        # ============================================================
        # Step 3: 构建 Anchor 位置嵌入
        # ============================================================
        # learnable_motion_query_embedding: 可学习的运动查询嵌入
        # shape: (num_anchor * num_groups, D) → 按 num_anchor 分组 → (num_groups, num_anchor, D)
        learnable_query_pos = self.learnable_motion_query_embedding.weight.to(dtype)
        learnable_query_pos = torch.stack(torch.split(learnable_query_pos, self.num_anchor, dim=0))
        # 现在 learnable_query_pos: (num_groups, num_anchor, D)

        # ---- 加载 K-Means anchor 轨迹 ----
        # kmeans_anchors: shape (num_groups, num_anchor, predict_steps, 2)
        # 每组有 num_anchor 条预先聚类好的代表性轨迹（智能体自身坐标系）
        agent_level_anchors = self.kmeans_anchors.to(dtype).to(device).view(
            num_groups, self.num_anchor, self.predict_steps, 2).detach()
        # 注意：.detach() 表示 anchor 本身不参与梯度计算（只作为先验知识）

        # ---- 坐标系变换：将 anchor 转换到场景坐标系 ----
        # scene_level_ego_anchors: 全局坐标（含平移），shape (B, A_total, num_groups, num_anchor, predict_steps, 2)
        # 即把每个智能体的 anchor 轨迹转换到以 ego 为原点的全局坐标系
        scene_level_ego_anchors = anchor_coordinate_transform(
            agent_level_anchors, track_bbox_results, with_translation_transform=True)
        
        # scene_level_offset_anchors: 相对偏移（仅旋转，不含平移），同 shape
        # 即只旋转到与 ego 朝向一致，但保持以智能体当前位置为原点
        scene_level_offset_anchors = anchor_coordinate_transform(
            agent_level_anchors, track_bbox_results, with_translation_transform=False)

        # ---- 坐标归一化到 [0, 1] ----
        agent_level_norm = norm_points(agent_level_anchors, self.pc_range)              # 智能体坐标系归一化
        scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)      # 全局坐标系归一化
        scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)  # 偏移坐标系归一化

        # ---- 生成三种层次的 anchor 位置嵌入 ----
        # 只取每条 anchor 轨迹的最后一个时间步（终点位置）作为代表点
        # [..., -1, :] 表示取最后一个时间步的 (x, y)
        
        # agent_level_embedding: 智能体自身坐标系的位置嵌入，shape (num_groups, num_anchor, D)
        # 编码了"这种类别的智能体通常有什么样的运动模式"
        agent_level_embedding = self.agent_level_embedding_layer(
            pos2posemb2d(agent_level_norm[..., -1, :]))
        
        # scene_level_ego_embedding: 全局坐标系的位置嵌入，shape (B, A_total, num_groups, num_anchor, D)
        # 编码了"这个 anchor 轨迹在场景中的绝对位置"
        scene_level_ego_embedding = self.scene_level_ego_embedding_layer(
            pos2posemb2d(scene_level_ego_norm[..., -1, :]))
        
        # scene_level_offset_embedding: 相对偏移的位置嵌入，同 shape
        # 编码了"这个 anchor 轨迹相对于当前智能体位置的偏移"
        scene_level_offset_embedding = self.scene_level_offset_embedding_layer(
            pos2posemb2d(scene_level_offset_norm[..., -1, :]))

        # ---- 将 agent_level 和 learnable 嵌入扩展到 batch 维度 ----
        batch_size, num_agents = scene_level_ego_embedding.shape[:2]
        
        # agent_level_embedding: (num_groups, num_anchor, D) → (B, A_total, num_groups, num_anchor, D)
        agent_level_embedding = agent_level_embedding[None, None, ...].expand(batch_size, num_agents, -1, -1, -1)
        
        # learnable_embed: (num_groups, num_anchor, D) → (B, A_total, num_groups, num_anchor, D)
        learnable_embed = learnable_query_pos[None, None, ...].expand(batch_size, num_agents, -1, -1, -1)

        # ============================================================
        # Step 4: 按类别分组选取对应的 anchor 嵌入
        # ============================================================
        # group_mode_query_pos: 根据每个智能体的类别，
        # 从 (B, A, G, P, D) 中选取对应组 G 的那条 anchor，输出 (B, A, P, D)
        # 即：车辆用车辆组的 anchor，行人用行人组的 anchor
        
        # 保存 scene_level_offset_anchors 用于后续 MotionFormer 的参考轨迹初始化
        # shape: (B, A_total, num_groups, num_anchor, predict_steps, 2) → (B, A_total, num_anchor, predict_steps, 2)
        scene_level_offset_anchors = self.group_mode_query_pos(track_bbox_results, scene_level_offset_anchors)
        
        # 各嵌入按类别分组选取，从 (B, A, G, P, D) → (B, A, P, D)
        agent_level_embedding = self.group_mode_query_pos(track_bbox_results, agent_level_embedding)
        scene_level_ego_embedding = self.group_mode_query_pos(track_bbox_results, scene_level_ego_embedding)
        scene_level_offset_embedding = self.group_mode_query_pos(track_bbox_results, scene_level_offset_embedding)
        learnable_embed = self.group_mode_query_pos(track_bbox_results, learnable_embed)

        # init_reference: 初始参考轨迹（用于 MotionFormer 内部的迭代细化）
        # .detach() 表示参考轨迹本身不直接传递梯度（通过预测偏移来间接优化）
        init_reference = scene_level_offset_anchors.detach()

        # ============================================================
        # Step 5: MotionFormer 多层 Transformer Decoder 前向传播
        # ============================================================
        # 收集每一层 decoder 的输出
        outputs_traj_scores = []  # 各层的轨迹概率分数
        outputs_trajs = []        # 各层的轨迹坐标预测

        # motionformer: 多层交叉注意力 Transformer
        # inter_states: 各层 decoder 输出的智能体特征，shape (num_layers, B, A_total, num_anchor, D)
        # inter_references: 各层的迭代细化后的参考轨迹（每层对轨迹进行微调）
        inter_states, inter_references = self.motionformer(
            track_query,            # (B, A_total, D)：每个智能体的身份特征
            lane_query,             # (B, M, D)：车道线/地图特征（作为 cross-attention 的 key/value）
            track_query_pos=track_query_pos,      # track query 位置编码
            lane_query_pos=lane_query_pos,        # lane query 位置编码
            track_bbox_results=track_bbox_results,  # 用于内部计算
            bev_embed=bev_embed,                  # BEV 场景特征（作为全局上下文）
            reference_trajs=init_reference,       # anchor 轨迹（初始参考）
            traj_reg_branches=self.traj_reg_branches,   # 各层的回归分支（传入供内部迭代使用）
            traj_cls_branches=self.traj_cls_branches,   # 各层的分类分支
            # 三种层次的 anchor 位置嵌入（对应三种坐标系视角）
            agent_level_embedding=agent_level_embedding,
            scene_level_ego_embedding=scene_level_ego_embedding,
            scene_level_offset_embedding=scene_level_offset_embedding,
            learnable_embed=learnable_embed,
            # 各嵌入层（MLP），传入供 MotionFormer 内部动态更新位置编码
            agent_level_embedding_layer=self.agent_level_embedding_layer,
            scene_level_ego_embedding_layer=self.scene_level_ego_embedding_layer,
            scene_level_offset_embedding_layer=self.scene_level_offset_embedding_layer,
            # BEV 特征图的空间形状，用于 deformable attention
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=device),
            level_start_index=torch.tensor([0], device=device))

        # ============================================================
        # Step 6: 对每层 decoder 的输出分别预测轨迹
        # ============================================================
        for lvl in range(inter_states.shape[0]):
            # ---- 分类分支：预测每条候选轨迹的概率分数 ----
            # traj_cls_branches[lvl]: 当前层的分类头（输出 logit）
            # inter_states[lvl]: shape (B, A_total, num_anchor, D)
            # outputs_class: shape (B, A_total, num_anchor, 1) → squeeze → (B, A_total, num_anchor)
            outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])
            
            # ---- 回归分支：预测每条候选轨迹的坐标偏移 ----
            # traj_reg_branches[lvl]: 当前层的回归头
            # tmp: shape (B, A_total, num_anchor, predict_steps * 5)
            tmp = self.traj_reg_branches[lvl](inter_states[lvl])
            
            # unflatten_traj: 将最后一维从 (predict_steps * 5) 变形为 (predict_steps, 5)
            # tmp: shape (B, A_total, num_anchor, predict_steps, 5)
            # 5个值表示高斯分布参数：(delta_x, delta_y, log_σx, log_σy, ρ_logit)
            tmp = self.unflatten_traj(tmp)
            
            # ---- 累加求和得到绝对坐标（cumsum trick） ----
            # 网络输出的是相邻时间步之间的位移（delta_x, delta_y）
            # 通过 cumsum 将相对位移转换为从原点出发的绝对坐标
            # 例如：[Δx₁, Δx₂, Δx₃] → [Δx₁, Δx₁+Δx₂, Δx₁+Δx₂+Δx₃]
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
            # dim=3 对应 predict_steps 这一维

            # ---- 对分类 logit 做 log-softmax，得到对数概率 ----
            # squeeze(3) 去掉维度3（原来的 1 维），shape (B, A_total, num_anchor)
            outputs_class = self.log_softmax(outputs_class.squeeze(3))
            outputs_traj_scores.append(outputs_class)

            # ---- 对每个样本应用双变量高斯激活 ----
            # bivariate_gaussian_activation: 将网络的原始输出（无限制的实数）
            # 转换为合法的高斯分布参数（σ > 0，|ρ| < 1 等约束）
            for bs in range(tmp.shape[0]):
                tmp[bs] = bivariate_gaussian_activation(tmp[bs])
            outputs_trajs.append(tmp)
        
        # 将列表堆叠成张量
        # outputs_traj_scores: (num_layers, B, A_total, num_anchor)
        outputs_traj_scores = torch.stack(outputs_traj_scores)
        # outputs_trajs: (num_layers, B, A_total, num_anchor, predict_steps, 5)
        outputs_trajs = torch.stack(outputs_trajs)

        # ---- 构建有效掩码（全True，所有 query 都有效） ----
        B, A_track, D = track_query.shape
        valid_traj_masks = track_query.new_ones((B, A_track)) > 0  # shape (B, A_total)，全 True

        # ---- 打包输出 ----
        outs = {
            'all_traj_scores': outputs_traj_scores,  # 各层轨迹概率
            'all_traj_preds': outputs_trajs,          # 各层轨迹坐标（高斯参数）
            'valid_traj_masks': valid_traj_masks,     # 有效掩码（全True）
            'traj_query': inter_states,               # 各层 decoder 输出特征（供后续模块使用）
            'track_query': track_query,               # 原始 track query
            'track_query_pos': track_query_pos,       # track query 位置编码
        }

        return outs

    def group_mode_query_pos(self, bbox_results, mode_query_pos):
        """根据每个智能体的类别，从多组 anchor 中选取对应组的 anchor 嵌入。

        背景：不同类别的智能体有不同的运动模式，
              车辆组的 anchor 描述车辆的典型运动（直行、左转、右转等），
              行人组的 anchor 描述行人的典型运动（随机游走、停留等）。
              这个函数确保每个智能体使用正确类别组的 anchor。

        原理：
            输入 mode_query_pos 有多组（G组）anchor 嵌入，
            根据 cls2group 查表，找到每个智能体 j 的类别组索引，
            然后取 mode_query_pos[i, j, group_idx] 作为该智能体的 anchor 嵌入。

        Args:
            bbox_results (list): 边界框结果，每个元素为 (bboxes, scores, labels, bbox_index, mask)
            mode_query_pos (Tensor): 各组 anchor 嵌入，shape (B, A, G, P, D)
                B=batch_size, A=num_agents, G=num_groups, P=num_anchor_per_group, D=embed_dims

        Returns:
            Tensor: 按类别选取后的 anchor 嵌入，shape (B, A, P, D)
                每个智能体只保留其对应类别组的 P 个 anchor
        """
        batch_size = len(bbox_results)
        agent_num = mode_query_pos.shape[1]  # 智能体数量 A
        batched_mode_query_pos = []
        self.cls2group = self.cls2group.to(mode_query_pos.device)  # 确保在同一设备
        
        # TODO: vectorize this（当前是嵌套循环，后续可向量化加速）
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            label = labels.to(mode_query_pos.device)
            
            # cls2group[label]: 将类别 ID 映射到组 ID
            # 例如：car(0)→组0, truck(1)→组0, pedestrian(7)→组1
            grouped_label = self.cls2group[label]   # shape (A,)
            
            grouped_mode_query_pos = []
            for j in range(agent_num):
                # 取第 j 个智能体对应组的 anchor 嵌入
                # grouped_label[j]: 第 j 个智能体的组索引
                # mode_query_pos[i, j, group_idx]: shape (P, D)
                grouped_mode_query_pos.append(mode_query_pos[i, j, grouped_label[j]])
            
            # 堆叠后 shape: (A, P, D)
            batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
        
        # 堆叠 batch 维度，shape: (B, A, P, D)
        return torch.stack(batched_mode_query_pos)

    @force_fp32(apply_to=('preds_dicts_motion'))
    def loss(self,
             gt_bboxes_3d,
             gt_fut_traj,
             gt_fut_traj_mask,
             preds_dicts_motion,
             all_matched_idxes,
             track_bbox_results):
        """计算运动预测损失。

        损失计算策略（与深度检测头类似的辅助损失机制）：
          - 对每一层 decoder 的输出都计算一次损失（辅助损失）
          - 最终损失以最后一层的结果为主，前几层作为辅助（前缀 d0, d1, ...）
          - 这样可以让中间层也学到有用的表示，加速训练收敛

        损失组成（由 compute_loss_traj 计算）：
          - loss_traj: 总轨迹损失（分类 + 回归的加权和）
          - l_class: 分类损失（预测哪条模式的概率，用 NLL Loss）
          - l_reg: 回归损失（轨迹坐标的精确度，用高斯负对数似然）
          - min_ade: 最小平均位移误差（取最优模式计算，评估指标）
          - min_fde: 最小终点位移误差（取最优模式，评估指标）
          - mr: 未命中率（Miss Rate，终点误差 > 2m 的比例）

        Args:
            gt_bboxes_3d (list): GT 3D 边界框
            gt_fut_traj (list[Tensor]): GT 未来轨迹，已包含 SDC（最后一行）
            gt_fut_traj_mask (list[Tensor]): GT 轨迹有效性掩码
            preds_dicts_motion (dict): forward() 的输出字典
            all_matched_idxes (list[Tensor]): GT 匹配索引
            track_bbox_results (list): 跟踪边界框结果

        Returns:
            dict: 损失字典，包括最后一层的损失和各中间层的带前缀损失
        """
        all_traj_scores = preds_dicts_motion['all_traj_scores']  # (num_layers, B, A, num_anchor)
        all_traj_preds = preds_dicts_motion['all_traj_preds']    # (num_layers, B, A, num_anchor, steps, 5)

        num_dec_layers = len(all_traj_scores)  # decoder 层数（通常为3）

        # 将 GT 轨迹复制 num_dec_layers 份（每层损失都需要对比 GT）
        all_gt_fut_traj = [gt_fut_traj for _ in range(num_dec_layers)]
        all_gt_fut_traj_mask = [gt_fut_traj_mask for _ in range(num_dec_layers)]

        losses_traj = []
        
        # 先计算 GT 匹配（只需计算一次，所有层共用相同的 GT 匹配结果）
        # compute_matched_gt_traj: 根据匹配索引从 GT 中提取对应目标的轨迹
        # 如果开启 use_nonlinear_optimizer，还会对 GT 轨迹做物理约束平滑
        gt_fut_traj_all, gt_fut_traj_mask_all = self.compute_matched_gt_traj(
            all_gt_fut_traj[0], all_gt_fut_traj_mask[0], all_matched_idxes, track_bbox_results, gt_bboxes_3d)
        
        # 对每一层 decoder 分别计算损失
        for i in range(num_dec_layers):
            loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr = self.compute_loss_traj(
                all_traj_scores[i], all_traj_preds[i],
                gt_fut_traj_all, gt_fut_traj_mask_all, all_matched_idxes)
            losses_traj.append((loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr))

        # ---- 构建损失字典 ----
        loss_dict = dict()
        
        # 最后一层的损失作为主损失
        loss_dict['loss_traj'] = losses_traj[-1][0]
        loss_dict['l_class'] = losses_traj[-1][1]
        loss_dict['l_reg'] = losses_traj[-1][2]
        loss_dict['min_ade'] = losses_traj[-1][3]
        loss_dict['min_fde'] = losses_traj[-1][4]
        loss_dict['mr'] = losses_traj[-1][5]

        # 中间层的损失作为辅助损失（加 'd{n}.' 前缀区分）
        num_dec_layer = 0
        for loss_traj_i in losses_traj[:-1]:   # 排除最后一层（已记录为主损失）
            loss_dict[f'd{num_dec_layer}.loss_traj'] = loss_traj_i[0]
            loss_dict[f'd{num_dec_layer}.l_class'] = loss_traj_i[1]
            loss_dict[f'd{num_dec_layer}.l_reg'] = loss_traj_i[2]
            loss_dict[f'd{num_dec_layer}.min_ade'] = loss_traj_i[3]
            loss_dict[f'd{num_dec_layer}.min_fde'] = loss_traj_i[4]
            loss_dict[f'd{num_dec_layer}.mr'] = loss_traj_i[5]
            num_dec_layer += 1

        return loss_dict

    def compute_matched_gt_traj(self,
                                gt_fut_traj,
                                gt_fut_traj_mask,
                                all_matched_idxes,
                                track_bbox_results,
                                gt_bboxes_3d):
        """根据匹配索引从 GT 中提取每个 track query 对应的 GT 轨迹。

        问题背景：
            TrackHead 输出的 track_query 与 GT 目标之间有对应关系（通过匈牙利匹配）。
            all_matched_idxes[i][j] = k 表示第 i 张图的第 j 个 track query 对应第 k 个 GT 目标。
            这个函数就是按照这个匹配关系，提取出每个 track query 对应的 GT 轨迹。

        额外功能（use_nonlinear_optimizer=True）：
            对提取出的 GT 轨迹进行物理约束平滑（nonlinear_smoother）。
            原始 GT 轨迹可能有噪声（位置标注误差），物理平滑后的轨迹更加自然，
            有助于网络学到更符合物理规律的运动预测。
            注意：SDC 的 GT 轨迹不参与平滑处理（注释中的 TODO）。

        Args:
            gt_fut_traj (list[Tensor]): GT 未来轨迹，shape (N_obj+1, predict_steps, 2)
                最后一行是 SDC 的 GT 轨迹
            gt_fut_traj_mask (list[Tensor]): GT 轨迹有效性掩码，shape (N_obj+1, predict_steps, 2)
            all_matched_idxes (list[Tensor]): 匹配索引，shape (A_total,)
                值 ≥ 0 表示有效匹配，= -1 表示该 track query 没有对应 GT
            track_bbox_results (list): 跟踪边界框（用于非线性平滑时的初始状态）
            gt_bboxes_3d (list): GT 3D 边界框（含朝向角，用于非线性平滑的运动学约束）

        Returns:
            tuple:
                gt_fut_traj_all (Tensor): 匹配后的 GT 轨迹，shape (N_valid, predict_steps, 2)
                gt_fut_traj_mask_all (Tensor): 匹配后的掩码，shape (N_valid, predict_steps)
                    注意：mask 在这里被合并为每步单一值（all()），方便损失计算
        """
        num_imgs = len(all_matched_idxes)
        gt_fut_traj_all = []
        gt_fut_traj_mask_all = []
        
        for i in range(num_imgs):
            matched_gt_idx = all_matched_idxes[i]   # 第 i 张图的匹配索引，shape (A_total,)
            
            # valid_traj_masks: 有效 track query 的布尔掩码（匹配到 GT 的才有效）
            # matched_gt_idx >= 0 表示有对应的 GT 目标（=-1 表示新生成的目标，尚无 GT 对应）
            valid_traj_masks = matched_gt_idx >= 0
            
            # 按匹配索引提取 GT 轨迹，然后只保留有效的（valid_traj_masks）
            # matched_gt_fut_traj: shape (N_valid, predict_steps, 2)
            matched_gt_fut_traj = gt_fut_traj[i][matched_gt_idx][valid_traj_masks]
            matched_gt_fut_traj_mask = gt_fut_traj_mask[i][matched_gt_idx][valid_traj_masks]
            
            if self.use_nonlinear_optimizer:
                # ---- 对 GT 轨迹进行物理约束平滑（仅训练时可选） ----
                # TODO: sdc query 暂不支持非线性优化器平滑
                
                # 提取有效的 track 边界框（用于提供初始位置和朝向）
                bboxes = track_bbox_results[i][0].tensor[:len(valid_traj_masks)].to(valid_traj_masks.device)[valid_traj_masks]
                
                # 提取 GT 3D 边界框（用于运动学约束，提供速度、加速度等约束）
                matched_tensor = gt_bboxes_3d[i][-1].tensor
                matched_indices = matched_gt_idx[:-1]     # 去掉最后一个（SDC）
                valid_masks = valid_traj_masks[:-1]       # 去掉最后一个（SDC）
                matched_gt_bboxes_3d = matched_tensor.to(valid_masks.device)[
                    matched_indices.to(valid_masks.device)][valid_masks]
                
                # 临时分离 SDC 的 GT 轨迹（不参与平滑）
                sdc_gt_fut_traj = matched_gt_fut_traj[-1:]           # SDC 的 GT 轨迹
                sdc_gt_fut_traj_mask = matched_gt_fut_traj_mask[-1:] # SDC 的掩码
                matched_gt_fut_traj = matched_gt_fut_traj[:-1]       # 去掉 SDC，只平滑其他目标
                matched_gt_fut_traj_mask = matched_gt_fut_traj_mask[:-1]
                bboxes = bboxes[:-1]
                
                # 非线性平滑：对 GT 轨迹施加运动学约束（如最大加速度、最大转向角）
                # 使得 GT 轨迹更符合真实车辆/行人的物理运动规律
                matched_gt_fut_traj, matched_gt_fut_traj_mask = nonlinear_smoother(
                    matched_gt_bboxes_3d, matched_gt_fut_traj, matched_gt_fut_traj_mask, bboxes)
                
                # 平滑完成后，将 SDC 的 GT 轨迹重新拼接回末尾
                matched_gt_fut_traj = torch.cat([matched_gt_fut_traj, sdc_gt_fut_traj], dim=0)
                matched_gt_fut_traj_mask = torch.cat([matched_gt_fut_traj_mask, sdc_gt_fut_traj_mask], dim=0)
            
            # 将 mask 从 (N_valid, predict_steps, 2) 合并为 (N_valid, predict_steps)
            # torch.all(..., dim=-1)：只有 (x, y) 两个维度都有效时，该时间步才算有效
            matched_gt_fut_traj_mask = torch.all(matched_gt_fut_traj_mask > 0, dim=-1)
            
            gt_fut_traj_all.append(matched_gt_fut_traj)
            gt_fut_traj_mask_all.append(matched_gt_fut_traj_mask)
        
        # 将所有图的结果在第0维（目标数量）拼接
        # gt_fut_traj_all: (N_valid_total, predict_steps, 2)
        # gt_fut_traj_mask_all: (N_valid_total, predict_steps)
        gt_fut_traj_all = torch.cat(gt_fut_traj_all, dim=0)
        gt_fut_traj_mask_all = torch.cat(gt_fut_traj_mask_all, dim=0)
        return gt_fut_traj_all, gt_fut_traj_mask_all

    def compute_loss_traj(self,
                          traj_scores,
                          traj_preds,
                          gt_fut_traj_all,
                          gt_fut_traj_mask_all,
                          all_matched_idxes):
        """计算单层 decoder 输出的轨迹损失。

        这个函数处理 batch 内的拼接，然后调用 self.loss_traj 计算实际损失。

        损失计算逻辑（由 loss_traj 实现，通常是 MR Loss 或 minADE Loss）：
          1. 对每个智能体的 num_anchor 条候选轨迹，与 GT 轨迹逐一比较
          2. 找到最接近 GT 的那条候选轨迹（Winner-take-all 策略）
          3. 对"赢家"轨迹计算回归损失（高斯负对数似然）
          4. 对分类分数计算交叉熵损失（让"赢家"的概率最高）

        Args:
            traj_scores (Tensor): 当前层的轨迹概率（log-softmax），shape (B, A_total, num_anchor)
            traj_preds (Tensor): 当前层的轨迹坐标，shape (B, A_total, num_anchor, predict_steps, 5)
            gt_fut_traj_all (Tensor): 匹配后的 GT 轨迹，shape (N_valid_total, predict_steps, 2)
            gt_fut_traj_mask_all (Tensor): 匹配后的 GT 掩码，shape (N_valid_total, predict_steps)
            all_matched_idxes (list[Tensor]): 匹配索引（每张图一个）

        Returns:
            tuple: (总损失, 分类损失, 回归损失, minADE, minFDE, 未命中率)
        """
        num_imgs = traj_scores.size(0)  # batch 大小
        traj_prob_all = []
        traj_preds_all = []
        
        for i in range(num_imgs):
            matched_gt_idx = all_matched_idxes[i]
            valid_traj_masks = matched_gt_idx >= 0  # 只保留有效匹配的 query
            
            # 取第 i 张图中有效 track query 的预测结果
            # batch_traj_prob: shape (N_valid_i, num_anchor)
            batch_traj_prob = traj_scores[i, valid_traj_masks, :]
            # batch_traj_preds: shape (N_valid_i, num_anchor, predict_steps, 5)
            batch_traj_preds = traj_preds[i, valid_traj_masks, ...]
            
            traj_prob_all.append(batch_traj_prob)
            traj_preds_all.append(batch_traj_preds)
        
        # 拼接 batch 内所有图的结果
        traj_prob_all = torch.cat(traj_prob_all, dim=0)    # (N_valid_total, num_anchor)
        traj_preds_all = torch.cat(traj_preds_all, dim=0)  # (N_valid_total, num_anchor, steps, 5)
        
        # 调用损失函数计算具体数值
        traj_loss, l_class, l_reg, l_minade, l_minfde, l_mr = self.loss_traj(
            traj_prob_all, traj_preds_all, gt_fut_traj_all, gt_fut_traj_mask_all)
        return traj_loss, l_class, l_reg, l_minade, l_minfde, l_mr

    @force_fp32(apply_to=('preds_dicts'))
    def get_trajs(self, preds_dicts, bbox_results):
        """将 forward() 的输出整理成可评估的轨迹预测结果（仅测试时调用）。

        在测试阶段，只需要最终的预测结果（可视化/计算评估指标），
        不需要所有中间层的结果，但这里保留了所有层的输出（加 '_0', '_1', ... 后缀）。

        Args:
            preds_dicts (dict): forward() 的输出字典（含 all_traj_preds, all_traj_scores）
            bbox_results (list): 边界框结果（用于确定 batch_size）

        Returns:
            list[dict]: 每张图一个字典，包含：
                'traj': 最后一层的轨迹预测，shape (A_total, num_anchor, predict_steps, 5)
                'traj_scores': 最后一层的轨迹概率，shape (A_total, num_anchor)
                'traj_0', 'traj_scores_0', ...: 各中间层的轨迹预测（带层号后缀）
        """
        num_samples = len(bbox_results)  # batch 大小
        num_layers = preds_dicts['all_traj_preds'].shape[0]  # decoder 层数
        ret_list = []
        
        for i in range(num_samples):
            preds = dict()
            for j in range(num_layers):
                # 最后一层不加后缀，中间层加 '_0', '_1', ... 后缀
                subfix = '_' + str(j) if j < (num_layers - 1) else ''
                
                # 取第 j 层、第 i 张图的轨迹预测
                traj = preds_dicts['all_traj_preds'][j, i]          # (A_total, num_anchor, steps, 5)
                traj_scores = preds_dicts['all_traj_scores'][j, i]  # (A_total, num_anchor)
                
                # 转移到 CPU（评估时不需要 GPU 计算）
                traj_scores, traj = traj_scores.cpu(), traj.cpu()
                preds['traj' + subfix] = traj
                preds['traj_scores' + subfix] = traj_scores
            ret_list.append(preds)
        return ret_list
    