import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils.loss_utils import repeated_indices_minkowski_signed_distance_obb
from utils.loss_utils import kinematics_from_xy, heading_and_yaw_rate_from_vel, decompose_accele_to_long_lat

class RepeatedIndicesMinkowskiCollisionLoss(torch.nn.Module):
    def __init__(self, margin: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, 
        ego_center, # [B, T, 2]
        ego_theta, # [B, T]
        ego_size, # [B, 2] or [B, T, 2]
        oth_center, # [BA, T, 2]
        oth_theta, # [BA, T]
        oth_size, # [BA, 2] or [BA, T, 2]
        others_num, # [B]
        repeated_indices, # [BA]
        invalid_mask=None,
        corner_lw_margin: Optional[Tuple[float, float]] = (0.2, 0.1) # [2]
    ):
        signed_distance = repeated_indices_minkowski_signed_distance_obb(
            ego_center, ego_theta, ego_size, 
            oth_center, oth_theta, oth_size, 
            repeated_indices,
            invalid_mask=invalid_mask, 
            corner_lw_margin=corner_lw_margin
        ) # [BA, T]
        B, T = ego_center.shape[0], ego_center.shape[1]
        loss_list, violation_list = [], []
        others_num_cumsum_with_zero = torch.cat((torch.tensor([0], device=ego_center.device), torch.cumsum(others_num, dim=0)), dim=0) # [B+1]
        for i in range(B):
            start = others_num_cumsum_with_zero[i]
            end = others_num_cumsum_with_zero[i+1]
            if end <= start:
                min_signed_distance = torch.full((T,), float('inf'), device=ego_center.device)  # no other agents
            else:
                min_signed_distance = signed_distance[start:end].min(dim=0).values # [T]
            # min_signed_distance = signed_distance[others_num_cumsum_with_zero[i]:others_num_cumsum_with_zero[i+1]].min(dim=0).values # [A,T], [T] 
            violation = (self.margin - min_signed_distance).clamp(min=0.0) # [T]
            violation_list.append((violation>0).float().sum())
            loss = violation.mean() if self.reduction == 'mean' else violation.sum()
            loss_list.append(loss)
        loss = sum(loss_list) / B
        return loss, sum(violation_list) / B

class SoftplusHingeComfortLoss(torch.nn.Module):
    """
    Diffferentiable comfort loss combining soft constraints on:
        - longitudinal acceleration |long_accel| <= max_long_accel
        - lateral acceleration      |lat_accel| <= max_lat_accel
        - yaw rate                  |yaw_rate| <= max_yaw_rate
        - jerk magnitude            ||jerk|| <= jerk_max
    
    Penalty uses a smooth softplus hinge to avoid gradient spikes.
    """
    def __init__(
        self,
        dt=0.1,
        max_long_accel=2.4,
        max_brake_accel=3.5,
        max_lat_accel=2.0,
        max_yaw_rate=0.95,
        max_jerk=6.0,
        softness=0.2,
        weights=(5.0, 1.0, 1.0, 1.0, 5.0),
        reduction='mean'
    ):
        super().__init__()
        self.dt = dt
        self.max_long_accel = max_long_accel
        self.max_brake_accel = max_brake_accel
        self.max_lat_accel = max_lat_accel
        self.max_yaw_rate = max_yaw_rate
        self.max_jerk = max_jerk
        self.softness = softness
        self.w_long_accel, self.w_long_brake, self.w_lat_accel, self.w_yaw_rate, self.w_jerk = weights
        self.reduction = reduction

    @staticmethod
    def _soft_hinge(x, limit, softness):
        """
        Smooth penalty for |x| > limit:
            softplus((|x| - limit)/softness)
        """
        return F.softplus((x.abs() - limit)/softness)

    def forward(
        self,
        traj_xy, # [B, T, 2]
    ):
        vel, speed, accel, jerk = kinematics_from_xy(traj_xy, dt=self.dt)
        heading, yaw_rate = heading_and_yaw_rate_from_vel(vel, dt=self.dt)
        long_accel, lat_accel = decompose_accele_to_long_lat(vel, speed, accel)
        jerk_mag = torch.linalg.norm(jerk, dim=-1)

        long_accel_mask = long_accel>=0.0
        long_brake_mask = long_accel<0.0


        # Compute softplus-hinge penalties
        long_accel_penalty = self._soft_hinge(long_accel, self.max_long_accel, self.softness) * long_accel_mask # [B, T-1]
        long_brake_penalty = self._soft_hinge(long_accel, self.max_brake_accel, self.softness) * long_brake_mask # [B, T-1]
        lat_accel_penalty = self._soft_hinge(lat_accel, self.max_lat_accel, self.softness) # [B, T-1]
        yaw_rate_penalty = self._soft_hinge(yaw_rate, self.max_yaw_rate, self.softness * 0.5) # [B, T]
        jerk_penalty = self._soft_hinge(jerk_mag, self.max_jerk, self.softness) # [B, T-2]

        per_batch_penalty = (
            self.w_long_accel * long_accel_penalty.mean(dim=-1) +
            self.w_long_brake * long_brake_penalty.mean(dim=-1) +
            self.w_lat_accel * lat_accel_penalty.mean(dim=-1) +
            self.w_yaw_rate * yaw_rate_penalty.mean(dim=-1) +
            self.w_jerk * jerk_penalty.mean(dim=-1)
        ) # [B,]

        if self.reduction == 'mean':
            loss = per_batch_penalty.mean()
        elif self.reduction == 'sum':
            loss = per_batch_penalty.sum()
        else:
            loss = per_batch_penalty

        # Optional metrics for logging
        with torch.no_grad():
            stats = dict(
                long_accel_rms = float((torch.sqrt(long_accel[long_accel_mask]**2).mean()).item()),
                long_brake_rms = float((torch.sqrt(long_accel[long_brake_mask]**2).mean()).item()),
                lat_accel_rms = float((torch.sqrt(lat_accel**2).mean()).item()),
                yaw_rate_rms = float((torch.sqrt(yaw_rate**2).mean()).item()),
                jerk_rms = float((torch.sqrt(jerk_mag**2).mean()).item()),
                frac_long_accel_exceed = float(((long_accel[long_accel_mask].abs() > self.max_long_accel).float().mean()).item()),
                frac_long_brake_exceed = float(((long_accel[long_brake_mask].abs() > self.max_brake_accel).float().mean()).item()),
                frac_lat_accel_exceed = float(((lat_accel.abs() > self.max_lat_accel).float().mean()).item()),
                frac_yaw_rate_exceed = float(((yaw_rate.abs() > self.max_yaw_rate).float().mean()).item()),
                frac_jerk_exceed = float(((jerk_mag > self.max_jerk).float().mean()).item()),
            )
        return loss, stats

# *******************
# ***Progress Loss***
# *******************

class AdaptiveProgressLoss(nn.Module):
    """
    Differentiable progress loss

    Progress is measured as effective displacement towards intended
    motion direction, penalizing stagnation and reward consistent advancement.
    """
    def __init__(
        self,
        dt=0.1,
        expected_speed_range=(3.0, 20.0), # 10.8 km/h - 72 km/h
        stagnation_threshold=1.0, # 3.6 km/h
        displacement_efficiency_weight=0.2,
        direction_consistency_weight=0.2,
        speed_adequacy_weight=0.6,
        temporal_decay=0.96
    ):
        super().__init__()
        self.dt = dt
        self.min_expected_speed, self.max_expected_speed = expected_speed_range
        self.stagnation_threshold = stagnation_threshold
        self.displacement_efficiency_weight = displacement_efficiency_weight
        self.direction_consistency_weight = direction_consistency_weight
        self.speed_adequacy_weight = speed_adequacy_weight
        self.temporal_decay = temporal_decay

    def forward(
        self,
        ego_planned_traj, # [B, T, 2]
        ego_gt_traj, # [B, T, 2]
        ego_gt_valid, # [B, T]
        importance_weights = None, # [B, T]
    ):
        """
        Compute adaptive progress loss encouraging meaningful advancement.

        Args:
            ego_planned_traj: planning trajectory [B, T, 2]
            reference_direction: direction of intended motion [B, 2]
            importance_weights: relative importance of each timestep [B, T]

        Returns:
            loss: progress loss
            stats: dictionary of progress metrics
        """
        B, T, _ = ego_planned_traj.shape

        # Generate temporal importance if not provided
        if importance_weights is None:
            importance_weights = self._generate_temporal_weights(ego_planned_traj) # [B, T-1]

        # Compute core progress components
        displacement_efficiency = self._compute_displacement_efficiency(ego_planned_traj, ego_gt_traj, ego_gt_valid) # [B, T-1]

        movement_adequacy = self._compute_movement_adequacy(ego_planned_traj, ego_gt_traj, ego_gt_valid) # [B, T-1]

        directional_consistency = self._compute_directional_consistency(ego_planned_traj, ego_gt_traj, ego_gt_valid) # [B, T-1]

        # Combine components with adaptive weights
        progress_scores = (
            self.displacement_efficiency_weight * displacement_efficiency +
            self.direction_consistency_weight * directional_consistency +
            self.speed_adequacy_weight * movement_adequacy
        )

        # Combine with temporal importance
        progress_scores = progress_scores * importance_weights # [B, T-1]

        # Compute loss
        # loss = 1.0 - progress_scores.mean(dim=-1) # [B]
        valid_mask = ego_gt_valid[..., :-1] & ego_gt_valid[..., 1:]
        loss = 1.0 - (progress_scores*valid_mask).sum(dim=-1) / (importance_weights*valid_mask).sum(dim=-1).clamp(min=1e-6) # [B], range [0, 1]
        smooth_loss = F.softplus(loss, beta=10.0) # [B] range score:1,loss:0, smooth:0.0693  score0,loss:1, smooth:1

        # Optional metrics for logging
        with torch.no_grad():
            displacements = torch.diff(ego_planned_traj, dim=-2)
            actual_speeds = torch.linalg.norm(displacements, dim=-1) / self.dt
            stats = {
                'progress_scores': progress_scores.mean().item(),
                'displacement_efficiency': displacement_efficiency.mean().item(),
                'directional_consistency': directional_consistency.mean().item(),
                'movement_adequacy': movement_adequacy.mean().item(),
                'mean_speed_kmph': actual_speeds.mean().item() * 3.6,
                'min_speed_kmph': actual_speeds.min().item() * 3.6,
                'max_speed_kmph': actual_speeds.max().item() * 3.6,
            }

        return smooth_loss.mean(), stats

    def _generate_temporal_weights(self, ego_planned_traj):
        B, T, _ = ego_planned_traj.shape
        # Exponentially increasing weights (later timesteps more important)
        time_indices = torch.arange(T-1, dtype=torch.float32, device=ego_planned_traj.device)
        exp_weights = torch.exp(time_indices * (1.0 - self.temporal_decay))
        exp_weights = exp_weights / exp_weights.max()
        return exp_weights.unsqueeze(0).expand(B, -1) # [B, T-1]
    
    def _compute_displacement_efficiency(self, ego_planned_traj, ego_gt_traj, ego_gt_valid):
        """Measure how efficiently the planned trajectory is moving in the intended direction."""
        # Planned displacement
        planned_displacements = torch.diff(ego_planned_traj, dim=-2) # [B, T-1, 2]
        planned_magnitudes = torch.linalg.norm(planned_displacements, dim=-1) # [B, T-1]

        # GT displacement
        gt_displacements = torch.diff(ego_gt_traj, dim=-2) # [B, T-1, 2]
        gt_magnitudes = torch.linalg.norm(gt_displacements, dim=-1) # [B, T-1]
        gt_displacements_valid_mask = ego_gt_valid[..., :-1] & ego_gt_valid[..., 1:] # [B, T-1]
        gt_directions = gt_displacements / (gt_magnitudes.unsqueeze(-1) + 1e-6) # [B, T-1, 2]
        
        # Extract valid values by masking
        planned_displacements = planned_displacements * gt_displacements_valid_mask.unsqueeze(-1)
        planned_magnitudes = planned_magnitudes * gt_displacements_valid_mask
        gt_directions = gt_directions * gt_displacements_valid_mask.unsqueeze(-1) # [B, T-1, 2]
        gt_magnitudes = gt_magnitudes * gt_displacements_valid_mask

        # Compute efficiency ratio
        directional_projection = (planned_displacements * gt_directions).sum(dim=-1) # [B, T-1]
        efficiency_ratio = ((directional_projection + 1e-6) / (planned_magnitudes + 1e-6)) # [B, T-1]

        # Apply sigmoid to convert to 0-1 range
        efficiency_scores = torch.sigmoid(efficiency_ratio * 10.0) # [B, T-1] # scaled by 2.0 to make it more sensitive

        return efficiency_scores # [B, T-1]

    def _compute_movement_adequacy(self, ego_planned_traj, ego_gt_traj, ego_gt_valid):
        """Evaluate if movement speeds are adequate (not too slow or too fast)"""
        velocities = torch.diff(ego_planned_traj, dim=-2) # [B, T-1, 2]
        speeds = torch.linalg.norm(velocities, dim=-1) / self.dt # [B, T-1]

        gt_velocities = torch.diff(ego_gt_traj, dim=-2) # [B, T-1, 2]
        gt_speeds = torch.linalg.norm(gt_velocities, dim=-1) / self.dt # [B, T-1]

        stop_mask = gt_speeds < self.stagnation_threshold # [B, T-1]
        stop_mask = stop_mask & ego_gt_valid[..., :-1] & ego_gt_valid[..., 1:] # [B, T-1]
        nonstop_mask = torch.logical_not(stop_mask) & ego_gt_valid[..., :-1] & ego_gt_valid[..., 1:] # [B, T-1]

        # Compute adequacy scores using smooth functions
        # Greatly penalize speeds below minimum stagnation threshold when ego is moving
        # slow_penalty = F.softplus(self.min_expected_speed - speeds, beta=2.0) * nonstop_mask # [B, T-1]

        # Gently penalize speeds above maximum expected speed (to avoid over-aggressive planning)
        # fast_penalty = F.softplus(speeds - self.max_expected_speed, beta=0.5)

        # Penalize speeds when ego is not moving
        nonstop_penalty = F.softplus(speeds, beta=0.5) * stop_mask # [B, T-1]

        # Combine penalties
        speed_penalties = nonstop_penalty #+ slow_penalty + fast_penalty 
        adequacy_scores = torch.exp(-speed_penalties) # [B, T-1]
        # adequacy_scores = torch.tanh(1.0 / (1.0 + speed_penalties))

        return adequacy_scores # [B, T-1]
    
    def _compute_directional_consistency(self, ego_planned_traj, ego_gt_traj, ego_gt_valid):
        """Compute directional consistency between planned and GT trajectories."""
        # Planned displacement
        planned_displacements = torch.diff(ego_planned_traj, dim=-2) # [B, T-1, 2]
        planned_magnitudes = torch.linalg.norm(planned_displacements, dim=-1) # [B, T-1]
        planned_directions = planned_displacements / (planned_magnitudes.unsqueeze(-1) + 1e-6) # [B, T-1, 2]

        # GT displacement
        gt_displacements = torch.diff(ego_gt_traj, dim=-2) # [B, T-1, 2]
        gt_magnitudes = torch.linalg.norm(gt_displacements, dim=-1) # [B, T-1]
        gt_directions = gt_displacements / (gt_magnitudes.unsqueeze(-1) + 1e-6) # [B, T-1, 2]
        gt_directions_valid_mask = ego_gt_valid[..., :-1] & ego_gt_valid[..., 1:] # [B, T-1]

        # Extract valid values by masking
        planned_directions = planned_directions * gt_directions_valid_mask.unsqueeze(-1)
        planned_magnitudes = planned_magnitudes * gt_directions_valid_mask
        gt_directions = gt_directions * gt_directions_valid_mask.unsqueeze(-1) # [B, T-1, 2]
        gt_magnitudes = gt_magnitudes * gt_directions_valid_mask

        # Compute directional consistency
        direction_alignment = (planned_directions * gt_directions).sum(dim=-1) # [B, T-1]
        consistency_scores = (direction_alignment + 1.0) / 2.0 # [B, T-1] convert from [-1, 1] to [0, 1]

        # Weight by movement magnitude (stronger movements should be more consistent)
        # normalized_magnitudes = planned_magnitudes / (self.max_expected_speed * self.dt)
        # normalized_magnitudes = planned_magnitudes / (self.max_expected_speed * self.dt)
        # magnitude_weights = torch.tanh(normalized_magnitudes) # [B, T-1]
        # magnitude_weights = torch.sigmoid(normalized_magnitudes - 0.5)
        # magnitude_weights = torch.sigmoid(consistency_scores)
        # weighted_consistency = consistency_scores * magnitude_weights

        return consistency_scores # [B, T-1]