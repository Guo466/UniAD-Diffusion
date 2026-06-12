import torch
from torch import Tensor
from typing import Optional, Tuple

def batch_label_ADE(pred:Tensor,label:Tensor,target_agents_num:Tensor,valid_mask = None, mode = 'marginal'):
    """
    calculate labels with min ADE.

    Args:
    - pred:  predicted trajectories, size->[ B*A , 1, M , T, 2 ]
    - label:  gt prediction labels, size->[ B*A ,1, T, 2 ]
    - valid_mask: pos valid mask->[ B*A, 1, T ]
    
    Returns :
    Index of certain mode trajectory which has the minimum FDE(Default: one in six).
    size->[B, ]  
    """
    dis = torch.sum((pred - label.unsqueeze(2)) ** 2, dim=-1)
    dis = dis.masked_fill(torch.isnan(dis), 1000000)
    B = target_agents_num.shape[0]
    M,T = dis.shape[-2:]
    if valid_mask is None:
        valid_mask = torch.ones(target_agents_num.sum(),1,T) # all 
    dis = dis * valid_mask.unsqueeze(-2) # B*A,1,M,T
    if mode == 'marginal':
        dis_mean = dis.sum(dim = -1 )/torch.clip(valid_mask.sum(dim = -1,keepdim = True),min=1.)
        return torch.min(dis_mean,dim = -1)
    else:
        # scene level label ADE
        # recover B,1,M
        batch_dis = torch.zeros(B,1,M).to(device=dis.device)
        cur = 0
        for i,a_n in enumerate(target_agents_num):
            cur_dis,cur_mask = dis[cur:cur+a_n,:],valid_mask[cur:cur+a_n,:]
            cur_dis_mean = cur_dis.sum(dim = -1 )/torch.clip(cur_mask.sum(dim = -1,keepdim = True),min=1.)
            batch_dis[i] = cur_dis_mean.sum(dim=0)/a_n
            cur += a_n
        return torch.min(batch_dis,dim = -1)

# **********************************************
# ***SAT-based minkowski collision loss utils***
# **********************************************

EPS = 1e-6
def rotation_matrix(theta):
    """
    Vectorized 2D rotation matrix.

    Args:
        theta (torch.Tensor): [..., ]

    Returns:
        torch.Tensor: [..., 2, 2]
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return torch.stack(
        (
            torch.stack((cos_theta, -sin_theta), dim=-1),
            torch.stack((sin_theta, cos_theta), dim=-1)
        ),
        dim=-2
    )

def obb_corners(center, half_extents, theta):
    """
    Vectorized computation of oriented bounding box corners. (Counter-clockwise ordered corners)

    Args:
        center (torch.Tensor): [..., 2]
        half_extents (torch.Tensor): [..., 2]
        theta (torch.Tensor): [..., 1]

    Returns:
        torch.Tensor: [..., 4, 2]
    """
    corner_base = torch.tensor([
        [-1., -1.],
        [1., -1.],
        [1., 1.],
        [-1., 1.],
    ], device=center.device, dtype=center.dtype) # [4, 2]
    # TODO: reshape corner_base to [..., 4, 2]
    real_corners = corner_base * half_extents.unsqueeze(-2) # [..., 4, 2]
    R = rotation_matrix(theta) # [..., 2, 2]
    corners = real_corners @ R.transpose(-2, -1) # [..., 4, 2]
    return corners + center.unsqueeze(-2) # [..., 4, 2]

# SAT required
def polygon_edge_normals(corners):
    """
    Vectorized computation of polygon edge normals.

    Args:
        corners (torch.Tensor): [..., 4, 2] (Counter-clockwise ordered corners)

    Returns:
        torch.Tensor: [..., 4, 2] (Normal vectors of edges)
    """
    edges = torch.roll(corners, shifts=-1, dims=-2) - corners # [..., 4, 2]
    normals = torch.stack(
        (
            -edges[..., 1],
            edges[..., 0]
        ), dim=-1
    )
    norms = torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=EPS)
    return normals / norms

def project_polygon_on_axis(axes, corners):
    """
    Project a polygon onto an axis.

    Args:
        axes (torch.Tensor): [..., axis_num, 2]
        corners (torch.Tensor): [..., 4, 2]

    Returns:
        min_projections (torch.Tensor): [..., axis_num]
        max_projections (torch.Tensor): [..., axis_num]
    """
    proj = (axes.unsqueeze(-2) * corners.unsqueeze(-3)).sum(dim=-1) # [..., axis_num, 4]
    min_proj = torch.min(proj, dim=-1) # [..., axis_num]
    max_proj = torch.max(proj, dim=-1) # [..., axis_num]
    return min_proj.values, max_proj.values

def sat_overlap_depth(axes_all, corners_a, corners_b):
    """
    Compute overlap depth between two polygons using Separating Axis Theorem (SAT).

    Args:
        axes_all (torch.Tensor): [..., axis_num, 2]
        corners_a (torch.Tensor): [..., 4, 2]
        corners_b (torch.Tensor): [..., 4, 2]

    Returns:
        overlap_per_axis (torch.Tensor): [..., axis_num]
        min_overlap_depth (torch.Tensor): [..., ]
    """
    a_min, a_max = project_polygon_on_axis(axes_all, corners_a) # [..., axis_num]
    b_min, b_max = project_polygon_on_axis(axes_all, corners_b) # [..., axis_num]
    overlap = torch.minimum(a_max, b_max) - torch.maximum(a_min, b_min) # [..., axis_num]
    return overlap, overlap.min(dim=-1).values # [..., axis_num], [...,]

def point_to_segment_distance(point, segment_start, segment_end):
    """
    Compute distance from a point to a segment.

    Args:
        point (torch.Tensor): [..., Np, 2]
        segment_start (torch.Tensor): [..., Ns, 2]
        segment_end (torch.Tensor): [..., Ns, 2]
    
    Returns:
        distance (torch.Tensor): [..., Np, Ns]
    """
    segment_vector = segment_end - segment_start # [..., Ns, 2]
    segment_vector_start_to_point = point.unsqueeze(-2) - segment_start.unsqueeze(-3) # [..., Np, Ns, 2]
    # Squared segment length (avoid divide-by-zero)
    segment_length_squared = (segment_vector*segment_vector).sum(dim=-1).clamp(min=EPS) # [..., Ns, 1]
    # Project point onto segment vector
    ## |A|*|B|*cos(theta) / |B|^2 = |A|*cos(theta) / |B|
    projection_factor = (segment_vector_start_to_point * segment_vector.unsqueeze(-3)).sum(dim=-1) / segment_length_squared.unsqueeze(-2) # [..., Np, Ns]
    # Clamp projection factor to [0, 1]
    projection_factor = torch.clamp(projection_factor, min=0.0, max=1.0) # [..., Np, Ns]
    # Closest point on each segment to each point
    ## B_start + |A|*cos(theta) * |B| / |B| = B_start + |A|*cos(theta)
    closest_point_on_segment = segment_start.unsqueeze(-3) + projection_factor.unsqueeze(-1) * segment_vector.unsqueeze(-3) # [..., Np, Ns, 2]
    # Calculate distance
    distance = torch.linalg.norm(point.unsqueeze(-2) - closest_point_on_segment, dim=-1) # [..., Np, Ns]
    return distance

def polygon_distance(corners_a, corners_b):
    """
    Euclidean distance between two convex polygons via vertex-edge distance.
    
    Args:
        corners_a (torch.Tensor): [..., 4, 2]
        corners_b (torch.Tensor): [..., 4, 2]

    Returns:
        distance (torch.Tensor): [..., ]
    """
    edges_a_start = corners_a
    edges_a_end = torch.roll(corners_a, shifts=-1, dims=-2)
    edges_b_start = corners_b
    edges_b_end = torch.roll(corners_b, shifts=-1, dims=-2)
    # Calculate minimum distance from A verteices to B edges
    dist_a_to_b = point_to_segment_distance(corners_a, edges_b_start, edges_b_end).min(dim=-1).values.min(dim=-1).values # [..., ]
    # Calculate minimum distance from B verteices to A edges
    dist_b_to_a = point_to_segment_distance(corners_b, edges_a_start, edges_a_end).min(dim=-1).values.min(dim=-1).values # [..., ]
    return torch.minimum(dist_a_to_b, dist_b_to_a) # [..., ]

def repeated_indices_minkowski_signed_distance_obb(
    ego_center, # [B, T, 2]
    ego_theta, # [B, T]
    ego_size, # [B,2] or [B, T, 2]
    oth_center, # [BA, T, 2]
    oth_theta, # [BA, T]
    oth_size, # [BA, 2] or [BA, T, 2],
    repeated_indices, # [BA]
    invalid_mask: Optional[torch.Tensor] = None, # [BA, T] True to ignore
    corner_lw_margin: Optional[Tuple[float, float]] = (0.2, 0.1) # [2]
):

    """
    Signed distance between ego OBB and others' OBBs using SAT-based Minkowski distance.
    Positive value means ego OBB is away from others' OBB, negative value means ego OBB is colliding with others' OBB.

    Returns:
        signed_distance (torch.Tensor): [B, A, T]
    """
    B = ego_center.shape[0]
    BA, T = oth_center.shape[0], oth_center.shape[1]
    # Broadcast ego sizes to [B, T, 2]
    if ego_size.dim() == 2:
        ego_size = ego_size.unsqueeze(1).expand(B, T, 2)
    # Broadcast other sizes to [BA, T, 2]
    if oth_size.dim() == 2:
        oth_size = oth_size.unsqueeze(-2).expand(BA, T, 2)

    if corner_lw_margin is not None:
        corner_lw_margin = torch.tensor(corner_lw_margin, device=ego_center.device) # [2]
    else:
        corner_lw_margin = torch.zeros(2, device=ego_center.device) # [2]
    
    # Build OBB corners
    ego_corners = obb_corners(ego_center, 0.5*ego_size + corner_lw_margin.view(*[1]*len(ego_center.shape[:-1]), 2), ego_theta) # [B, T, 4, 2]
    oth_corners = obb_corners(oth_center, 0.5*oth_size + corner_lw_margin.view(*[1]*len(oth_center.shape[:-1]), 2), oth_theta) # [BA, T, 4, 2]
    ## expand ego_corners to [BA, T, 4, 2]
    ego_corners = ego_corners[repeated_indices] # [BA, T, 4, 2]
    
    # Collect axes
    ego_axes = polygon_edge_normals(ego_corners) # [BA, T, 4, 2]
    oth_axes = polygon_edge_normals(oth_corners) # [BA, T, 4, 2]
    all_axes = torch.cat((ego_axes, oth_axes), dim=-2) # [BA, T, 8, 2]

    # SAT overlaps and min overlap across axes
    overlap_per_axis, min_overlap_depth = sat_overlap_depth(all_axes, ego_corners, oth_corners) # [BA, T, 8], [BA, T]
    overlapping_mask = (overlap_per_axis > 0).all(dim=-1) # [BA, T]

    # Collect penetration depth when overlapping
    penetration_depth = torch.where(overlapping_mask, min_overlap_depth, torch.zeros_like(min_overlap_depth)) # [BA, T]

    # Collect separation distance when disjoint
    disjoint_distance = polygon_distance(ego_corners, oth_corners) # [BA, T]

    # Compute signed distance
    signed_distance = torch.where(overlapping_mask, -penetration_depth, disjoint_distance) # [BA, T]

    # Apply invalid mask
    if invalid_mask is not None:
        signed_distance = signed_distance.masked_fill(invalid_mask, float('inf')) # [BA, T]

    return signed_distance

def ego_planning_loss_preprocess(pred_trajs, pred_scores, target_agents_gt, target_agents_gt_valid,target_current_info, target_lw, target_agents_num):
    """
    Args:
        pred_trajs: [B, A, M, T, C]
        pred_scores: [B, A, M]
        target_agents_gt: [B*A, 3]
        target_gt_valid: [B*A]
        target_current_info: [B*A, 3]
        target_lw: [B*A, 2]
        target_agents_num: [B]

    Returns:
        ego_center: [B, T, 2]
        ego_theta: [B, T]
        ego_size: [B, 2]
        other_center: [B*(A-1), T, 2]
        other_theta: [B*(A-1), T]
        other_size: [B*(A-1), 2]
        others_num: [B]
        repeated_indices: [B*(A-1)]
        other_invalid_mask: [B*(A-1), T]
        ego_gt_traj: [B, T, 2]
        ego_gt_valid: [B, T]
    """
    # get best mode preds
    B,A,M,T,C = pred_trajs.shape
    best_mode_idxs = torch.argmax(pred_scores, dim=-1).unsqueeze(-1) # [B, 1]
    batch_idxs = torch.arange(B, device=pred_trajs.device).unsqueeze(-1).expand(-1, pred_trajs.shape[1]) # [B, A]
    agent_idxs = torch.arange(pred_trajs.shape[1], device=pred_trajs.device).unsqueeze(0).expand(B, -1) # [B, A]
    best_pred_trajs = pred_trajs[batch_idxs, agent_idxs, best_mode_idxs]# [B, A, fut_ts, 6]
    ego_pred_xy = best_pred_trajs[:, 0, :, :2] # [B, fut_ts, 2]
    ego_pred_vel = torch.diff(ego_pred_xy, dim=-2, prepend=ego_pred_xy[:, :1]/0.1) # [B, fut_ts, 2]
    ego_pred_heading = torch.atan2(ego_pred_vel[..., 1], torch.clamp(ego_pred_vel[..., 0], min=EPS)) # [B, fut_ts]
    scene_centric_other_gt_xy, scene_centric_other_heading, other_gt_valid, other_lw, ego_lw, others_num = [], [], [], [], [], []
    ego_gt_traj, ego_gt_valid = [], []
    target_agents_num_cumsum_with_zero = torch.cat((torch.tensor([0], device=pred_trajs.device), torch.cumsum(target_agents_num, dim=0)), dim=0) # [B+1]
    for i in range(B):
        per_batch_other_gt = target_agents_gt[target_agents_num_cumsum_with_zero[i]+1:target_agents_num_cumsum_with_zero[i+1]] # ignore ego, [A-1, fut_ts, 5]
        per_batch_other_gt_valid = target_agents_gt_valid[target_agents_num_cumsum_with_zero[i]+1:target_agents_num_cumsum_with_zero[i+1]] # ignore ego, [A-1, fut_ts]
        per_batch_other_current_info = target_current_info[target_agents_num_cumsum_with_zero[i]+1:target_agents_num_cumsum_with_zero[i+1]] # [A-1, 3]
        per_batch_other_lw = target_lw[target_agents_num_cumsum_with_zero[i]+1:target_agents_num_cumsum_with_zero[i+1]] # [A-1, 2]
        per_batch_ego_lw = target_lw[target_agents_num_cumsum_with_zero[i]] # [2]
        yaw_rotation_matrix_2d = torch.stack(
            (torch.stack((torch.cos(per_batch_other_current_info[..., 2]), -torch.sin(per_batch_other_current_info[..., 2])), dim=-1), 
            torch.stack((torch.sin(per_batch_other_current_info[..., 2]), torch.cos(per_batch_other_current_info[..., 2])), dim=-1)),
            dim=-2
        ) # [A-1, 2, 2]
        # Collecting scene-centric agent gt xy, heading, valid, lw, ego lw
        scene_centric_other_gt_xy.append(per_batch_other_gt[..., :2]  @ yaw_rotation_matrix_2d.transpose(-2, -1) + per_batch_other_current_info[..., :2].unsqueeze(-2)) # [A-1, fut_ts, 2]
        scene_centric_other_heading.append(per_batch_other_gt[..., -1] + per_batch_other_current_info[..., 2].unsqueeze(-1)) # [A-1, fut_ts]
        other_gt_valid.append(per_batch_other_gt_valid)
        other_lw.append(per_batch_other_lw)
        ego_lw.append(per_batch_ego_lw)
        others_num.append(target_agents_num[i]-1)
        ego_gt_traj.append(target_agents_gt[target_agents_num_cumsum_with_zero[i]][..., :2]) # [fut_ts, 2]
        ego_gt_valid.append(target_agents_gt_valid[target_agents_num_cumsum_with_zero[i]]) # [fut_ts]
    ego_center = ego_pred_xy # [B, fut_ts, 2]
    ego_theta = ego_pred_heading # [B, fut_ts]
    ego_size = torch.stack(ego_lw, dim=0) # [B, 2]
    other_center = torch.cat(scene_centric_other_gt_xy, dim=0) # [B*(A-1), fut_ts, 2]
    other_theta = torch.cat(scene_centric_other_heading, dim=0) # [B*(A-1), fut_ts]
    other_size = torch.cat(other_lw, dim=0) # [B*(A-1), 2]
    
    others_num = torch.stack(others_num, dim=0) # [B]
    repeated_indices = torch.arange(B, device=pred_trajs.device).repeat_interleave(others_num) # [B*(A-1)]
    other_valid_mask = torch.cat(other_gt_valid, dim=0) # [B*(A-1), fut_ts]
    other_invalid_mask = torch.logical_not(other_valid_mask) # [B*(A-1), fut_ts]

    ego_gt_traj = torch.stack(ego_gt_traj, dim=0) # [B, fut_ts, 2]
    ego_gt_valid = torch.stack(ego_gt_valid, dim=0) # [B, fut_ts]
    return ego_center, ego_theta, ego_size, \
            other_center, other_theta, other_size, \
            others_num, repeated_indices, other_invalid_mask, \
            ego_gt_traj, ego_gt_valid

# ***************************************
# ***Softplus-hinge Comfort Loss utils***
# ***************************************

# def finite_diff(x, dim, dt):
#     """First order finite difference"""
#     dx = torch.roll(x, shifts=-1, dims=dim) - x
#     return dx / dt

def angle_diff(a2, a1):
    """Angle difference between two angles"""
    return torch.atan2(torch.sin(a2 - a1), torch.cos(a2 - a1))

def kinematics_from_xy(
    traj_xy, # [B, T, 2]
    dt=0.1,
    eps=1e-6
):
    """
    Returns:
        vel: [B, T, 2]
        speed: [B, T]
        accel: [B, T, 2]
        jerk: [B, T, 2]
    """
    T = traj_xy.shape[-2]
    traj_xy_for_diff = torch.cat((torch.zeros_like(traj_xy[:, :1]), traj_xy), dim=-2) # [B, T+1, 2]
    vel = torch.diff(traj_xy_for_diff, dim=-2)/dt # [B, T, 2]
    speed = torch.linalg.norm(vel, dim=-1).clamp(min=eps) # [B, T]
    accel = torch.diff(vel, dim=-2)/dt # [B, T-1, 2]
    jerk = torch.diff(accel, dim=-2)/dt # [B, T-2, 2]
    return vel, speed, accel, jerk

def heading_and_yaw_rate_from_vel(
    vel, # [B, T, 2]
    dt=0.1,
    eps=1e-6
):
    """
    Returns:
        heading: [B, T] (rad)
        yaw_rate: [B, T] (rad/s)
    """
    heading = torch.atan2(vel[..., 1], vel[..., 0]) # [B, T]
    heading_for_diff = torch.cat((torch.zeros_like(heading[:, :1]), heading), dim=-1) # [B, T+1]
    delta_heading = angle_diff(heading_for_diff[...,1:], heading_for_diff[...,:-1]) # [B, T]
    yaw_rate = delta_heading/ dt
    return heading, yaw_rate

def decompose_accele_to_long_lat(
    vel, # [B, T, 2]
    speed, # [B, T]
    accel, # [B, T, 2]
):
    """
    Returns:
        long_accel: [B, T-1] (along velocity direction)
        lat_accel: [B, T-1]  (to the left of velocity direction)
    """
    valid_T = accel.shape[-2]
    # Unit tangent vector and left normal vector
    tangent = vel / speed.unsqueeze(-1) # [B, T, 2]
    left_normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1) # [B, T-1, 2]
    # Decompose acceleration into long and lateral components
    long_accel = (accel * tangent[:, -valid_T:]).sum(dim=-1) # [B, T-1], 1:T
    lat_accel = (accel * left_normal[:, -valid_T:]).sum(dim=-1) # [B, T-1], 1:T
    return long_accel, lat_accel

