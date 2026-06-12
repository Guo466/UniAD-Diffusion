#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmdet.models import LOSSES


@LOSSES.register_module()
class PlanningLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        return torch.sum(err * mask)/(torch.sum(mask) + 1e-5)


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        super(CollisionLoss, self).__init__()
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        self.weight = weight
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(1, )
        dump_sdc = []
        for i in range(n_futures):
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev 
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])  
                dump_sdc.append(sdc_bev_box.cpu().detach().numpy())
                for j in range(future_gt_bbox_corners.shape[0]):
                    inter_sum += self.inter_bbox(sdc_bev_box, future_gt_bbox_corners[j].to(sdc_traj_all.device))  
        return inter_sum * self.weight
        
    def inter_bbox(self, corners_a, corners_b):
        xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        xa2, ya2 = torch.min(corners_a[:, 0]), torch.min(corners_a[:, 1])
        xb1, yb1 = torch.max(corners_b[:, 0]), torch.max(corners_b[:, 1])
        xb2, yb2 = torch.min(corners_b[:, 0]), torch.min(corners_b[:, 1])
        
        xi1, yi1 = min(xa1, xb1), min(ya1, yb1)
        xi2, yi2 = max(xa2, xb2), max(ya2, yb2)
        intersect = max((xi1 - xi2), xi1.new_zeros(1, ).to(xi1.device)) * max((yi1 - yi2), xi1.new_zeros(1,).to(xi1.device))
        return intersect

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        corners = torch.tensor([
            [w/2, -l/2], [w/2, l/2], [-w/2, l/2], [-w/2,-l/2]  
        ]).to(x.device) # 4,2
        rot_mat = torch.tensor(
            [[torch.cos(theta), torch.sin(theta)],
             [-torch.sin(theta), torch.cos(theta)]]
        ).to(x.device)
        new_corners = rot_mat @ corners.T + torch.tensor(bbox[:2])[:, None].to(x.device)
        return new_corners.T


@LOSSES.register_module()
class KinematicFeasibilityLoss(nn.Module):
    """运动学可行性损失 —— 约束规划轨迹满足车辆物理限制。

    背景与动机：
        UniAD 原始的 PlanningLoss 只计算预测轨迹与 GT 轨迹的 ADE（平均位移误差），
        完全不考虑轨迹的物理可执行性。这导致模型在弯道、紧急制动等场景中
        可能预测出"数学上接近 GT 但物理上无法执行"的轨迹：
          - 相邻两步之间的速度突变（违反加速度限制）
          - 速度变化率突变（违反 jerk/加加速度限制，造成不舒适）
          - 终点偏差过大（ADE 好但 FDE 差，模型不关注最终到达位置）

        作为控制专业背景，我们知道车辆运动学约束（自行车模型）对轨迹可执行性
        至关重要。本损失从训练阶段就引入这些约束，让模型学会生成"能开的轨迹"。

    三个子损失：

    1. loss_accel（加速度平滑损失）：
       约束相邻时间步之间的速度变化量（即加速度）不能过大。
       速度 v_t = (pos_{t} - pos_{t-1}) / dt
       加速度 a_t = (v_t - v_{t-1}) / dt = (pos_{t} - 2*pos_{t-1} + pos_{t-2}) / dt²
       损失 = mean(||a_t||²)，惩罚加速度平方和。

    2. loss_jerk（加加速度/jerk 平滑损失）：
       约束加速度变化率（jerk）不能过大，保证乘坐舒适性。
       jerk_t = (a_t - a_{t-1}) / dt
       损失 = mean(||jerk_t||²)，权重比加速度项小（次要约束）。

    3. loss_fde（终点位移误差损失）：
       专门监督最后一个时间步的预测精度（Final Displacement Error）。
       原始 PlanningLoss 对所有时间步求平均，模型可能牺牲终点精度换取
       中间步的精度。FDE 损失强制模型关注"最终到哪里"。
       损失 = ||pred_T - gt_T||₂（最后一步的欧氏距离）。

    Args:
        dt (float): 相邻时间步的时间间隔，单位秒。UniAD 中为 0.5s/步。
        weight_accel (float): 加速度平滑损失的权重。
            建议范围 0.1~1.0，过大会过度约束轨迹形状。
        weight_jerk (float): jerk 平滑损失的权重。
            建议比 weight_accel 小一个数量级（jerk 是高阶约束）。
        weight_fde (float): 终点误差损失的权重。
            建议与 PlanningLoss（ADE）同量级，如 1.0。

    输入（forward 参数）：
        sdc_traj (Tensor): 预测轨迹，shape (B, T, 2)，(x, y) 坐标，单位米。
            B=batch_size（通常1），T=planning_steps（6步），2=(x,y)。
        gt_sdc_fut_traj (Tensor): GT 轨迹，shape (B, T, 2)，格式同上。
        mask (Tensor): 有效帧掩码，shape (B, T)，True=有效，False=忽略该步。

    返回：
        Tensor: 标量损失值 = weight_accel*loss_accel
                            + weight_jerk*loss_jerk
                            + weight_fde*loss_fde
    """

    def __init__(self,
                 dt=0.5,
                 weight_accel=0.5,
                 weight_jerk=0.1,
                 weight_fde=1.0):
        super(KinematicFeasibilityLoss, self).__init__()
        self.dt = dt                    # 时间步长（秒），UniAD 中每步 0.5s
        self.weight_accel = weight_accel  # 加速度平滑项权重
        self.weight_jerk = weight_jerk    # jerk 平滑项权重
        self.weight_fde = weight_fde      # 终点误差项权重

    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        """计算运动学可行性损失。

        Args:
            sdc_traj (Tensor): 预测轨迹 (B, T, 2)，累积坐标（非位移），单位米
            gt_sdc_fut_traj (Tensor): GT 轨迹 (B, T, 2)，格式同上
            mask (Tensor): 有效掩码 (B, T)，True=有效时间步

        Returns:
            Tensor: 标量，三个子损失的加权和
        """
        # 只取 (x, y) 两维（GT 可能含 heading 第三维，这里不用）
        pred = sdc_traj[..., :2]        # (B, T, 2)
        gt   = gt_sdc_fut_traj[..., :2] # (B, T, 2)

        # ── 子损失1：加速度平滑损失 ──────────────────────────────────────────
        # 原理：轨迹坐标的二阶差分 ≈ 加速度（乘以 1/dt² 得到真实加速度）
        # 这里直接用坐标差分而不除以 dt²，因为 dt 是常数，不影响梯度方向
        # pos:   [p0, p1, p2, p3, p4, p5]  shape (B, T, 2)
        # vel:   [p1-p0, p2-p1, ...]       shape (B, T-1, 2)  一阶差分 ≈ 速度×dt
        # accel: [v1-v0, v2-v1, ...]       shape (B, T-2, 2)  二阶差分 ≈ 加速度×dt²
        accel = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]  # (B, T-2, 2)

        # 对应的掩码：只在两端都有效的时间步上计算（需要 t-1, t, t+1 三步都有效）
        # mask[:, 1:-1]: 去掉首尾，对应二阶差分的有效范围
        accel_mask = mask[:, 1:-1]  # (B, T-2)

        # 加速度的 L2 范数平方：||a_t||² = ax_t² + ay_t²
        accel_sq = torch.sum(accel ** 2, dim=-1)  # (B, T-2)

        # 有效步的均值（除以有效步数，避免 mask 导致量级变化）
        loss_accel = torch.sum(accel_sq * accel_mask) / (torch.sum(accel_mask) + 1e-5)

        # ── 子损失2：jerk（加加速度）平滑损失 ───────────────────────────────
        # 原理：加速度的一阶差分 ≈ jerk（加加速度，衡量舒适性的关键指标）
        # accel: shape (B, T-2, 2)
        # jerk:  shape (B, T-3, 2)  三阶差分
        if pred.shape[1] >= 4:  # 至少 4 步才能计算 jerk（T-3 >= 1）
            jerk = accel[:, 1:, :] - accel[:, :-1, :]  # (B, T-3, 2)

            # 掩码：需要 t-2~t+1 四步都有效
            jerk_mask = mask[:, 2:-1]  # (B, T-3)

            jerk_sq = torch.sum(jerk ** 2, dim=-1)  # (B, T-3)
            loss_jerk = torch.sum(jerk_sq * jerk_mask) / (torch.sum(jerk_mask) + 1e-5)
        else:
            # 规划步数太少时跳过 jerk 损失（避免空张量报错）
            loss_jerk = pred.new_zeros(1,).squeeze()

        # ── 子损失3：FDE（终点位移误差）损失 ────────────────────────────────
        # 原理：专门监督最后一个时间步的预测精度
        # ADE 对所有步求平均，可能让模型忽视终点；FDE 强制关注"最终到哪里"
        # pred[:, -1, :]: 最后一步的预测坐标 (B, 2)
        # gt[:, -1, :]:   最后一步的 GT 坐标   (B, 2)
        final_pred = pred[:, -1, :]  # (B, 2)
        final_gt   = gt[:, -1, :]    # (B, 2)

        # 最后一步的有效性掩码：(B,)
        final_mask = mask[:, -1]  # (B,)

        # 欧氏距离 ||pred_T - gt_T||₂
        fde = torch.sqrt(
            torch.sum((final_pred - final_gt) ** 2, dim=-1) + 1e-6
        )  # (B,)，加 1e-6 防止梯度爆炸（sqrt(0) 的梯度为 inf）

        loss_fde = torch.sum(fde * final_mask) / (torch.sum(final_mask) + 1e-5)

        # ── 加权求和 ────────────────────────────────────────────────────────
        total_loss = (
            self.weight_accel * loss_accel
            + self.weight_jerk  * loss_jerk
            + self.weight_fde   * loss_fde
        )

        return total_loss