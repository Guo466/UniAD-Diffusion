import numpy as np
import tqdm
from utils.metrics_utils import get_obb_corners_vectorized, get_axes_vectorized, project_vectorized
from utils.metrics_utils import compute_kinematics
from utils.metrics_utils import compute_displacement_efficiency, compute_movement_adequacy, compute_directional_consistency

# ************************
# ***Collision metrics***
# ************************

def repeated_indices_numpy_collision_metrics(
    ego_xy, # [B, T, 2]
    ego_heading, # [B, T]
    ego_size, # [B, T, 2]
    other_xy, # [BA, T, 2]
    other_heading, # [BA, T]
    other_size, # [BA, T, 2]
    other_num, # [B]
    repeated_indices, # []
    other_valid_mask=None
):
    """
    Vectorized numpy version of SAT-based collision detection.

    Args:
        ego_xy: [B, T, 2] ego vehicle positions
        ego_heading: [B, T] ego vehicle headings
        ego_size: [B, T, 2] ego vehicle sizes
        other_xy: [BA, T, 2] other vehicle positions
        other_heading: [BA, T] other vehicle headings
        other_size: [BA, T, 2] other vehicle sizes
        repeated_indices: [BA] repeated indices
        other_valid_mask: [BA, T] other vehicle valid masks

    Returns:
        collision_masks: [B, T] collision status per scenario and timestep
    """
    B, T = ego_xy.shape[:2]
    BA = other_xy.shape[0]

    # Get ego corners: [B, T, 4, 2]
    ego_corners = get_obb_corners_vectorized(ego_xy, ego_size, ego_heading)
    
    # Get other corners: [BA, T, 4, 2]
    other_corners = get_obb_corners_vectorized(other_xy, other_size, other_heading)
    
    # Prepare for broadcasting: ego [B, 1, T, 4, 2], other [BA, T, 4, 2]
    ego_corners = ego_corners[repeated_indices]  # [BA, T, 4, 2]
    
    # Get axes for both sets of corners
    axes1 = get_axes_vectorized(ego_corners)  # [BA, T, 4, 2]
    axes2 = get_axes_vectorized(other_corners)  # [BA, T, 4, 2]
    all_axes = np.concatenate([axes1, axes2], axis=-2)  # [BA, T, 8, 2]
    
    # Check separations along all axes
    # For axes from corners1
    ego_min, ego_max = project_vectorized(ego_corners, all_axes)  # [BA, T, 8]
    other_min, other_max = project_vectorized(other_corners, all_axes)  # [BA, T, 8]

    overlap_per_axis = np.minimum(ego_max, other_max) - np.maximum(ego_min, other_min)  # [BA, T, 8]
    overlap_mask = np.all(overlap_per_axis > 0, axis=-1)  # [BA, T]

    # other valid mask
    overlap_mask = overlap_mask & other_valid_mask # [BA, T]

    collision_cnt = 0
    other_num_cumsum_with_zero = np.concatenate((np.array([0]), np.cumsum(other_num))) # [B+1]
    for i in tqdm.trange(B, desc='Collision metrics: '):
        start = other_num_cumsum_with_zero[i]
        end = other_num_cumsum_with_zero[i+1]
        if end <= start:
            collision_cnt += 0  # no other agents
        else:
            collision_at_t = np.any(overlap_mask[start:end], axis=0) # [T]
            collision_cnt += np.any(collision_at_t)
    collision_rate = collision_cnt / B

    return {'collision_rate': collision_rate}

# *********************
# ***Comfort metrics***
# *********************

def numpy_comfort_metrics(ego_xy, dt=0.1, max_long_accel=2.40, max_brake_accel=3.5, max_lat_accel=2.0, max_yaw_rate=0.95, max_jerk=6.0):
    """
    Vectorized numpy version of comfort metrics.
    """
    B, T = ego_xy.shape[:2]
    vel, speed, accel, jerk, \
        heading, yaw_rate, \
            long_accel, lat_accel, jerk_magnitude = compute_kinematics(ego_xy, dt)

    # compute violation mask
    long_accel_violation_mask = long_accel[long_accel>=0.0] > max_long_accel # [N, ]
    long_brake_violation_mask = long_accel[long_accel<0.0] < -max_brake_accel # [N, ]
    long_volation_mask = np.logical_or(long_accel > max_long_accel, long_accel < -max_brake_accel) # [B, T]
    lat_accel_violation_mask = np.abs(lat_accel) > max_lat_accel # [B, T]
    yaw_rate_violation_mask = np.abs(yaw_rate) > max_yaw_rate # [B, T]
    jerk_violation_mask = jerk_magnitude > max_jerk # [B, T]

    # compute violation mask per scenario
    # long_accel_violation_mask_per_scenario = np.any(long_accel_violation_mask, axis=-1) # [B]
    # lat_accel_violation_mask_per_scenario = np.any(lat_accel_violation_mask, axis=-1) # [B]
    # yaw_rate_violation_mask_per_scenario = np.any(yaw_rate_violation_mask, axis=-1) # [B]
    # jerk_violation_mask_per_scenario = np.any(jerk_violation_mask, axis=-1) # [B]

    # compute violation rate
    long_accel_violation_rate = np.mean(long_accel_violation_mask)
    long_brake_violation_rate = np.mean(long_brake_violation_mask)
    lat_accel_violation_rate = np.mean(lat_accel_violation_mask)
    yaw_rate_violation_rate = np.mean(yaw_rate_violation_mask)
    jerk_violation_rate = np.mean(jerk_violation_mask)

    # compute overall violation rate
    valid_T  = jerk_violation_mask.shape[-1]
    any_violation_rate = np.mean(long_volation_mask[:,:valid_T]| lat_accel_violation_mask[:,:valid_T] | yaw_rate_violation_mask[:,:valid_T] | jerk_violation_mask[:,:valid_T])
    return {
        'long_accel_violation_rate': long_accel_violation_rate,
        'long_brake_violation_rate': long_brake_violation_rate,
        'lat_accel_violation_rate': lat_accel_violation_rate,
        'yaw_rate_violation_rate': yaw_rate_violation_rate,
        'jerk_violation_rate': jerk_violation_rate,
        'any_violation_rate': any_violation_rate,
    }

# ************************
# ***Efficiency metrics***
# ************************

def numpy_efficiency_metrics(ego_planned_xy, ego_gt_xy, ego_gt_valid, 
                             dt=0.1, expected_speed_range=(4.0, 30.0), stagnation_threshold=1.0):
    """
    Vectorized numpy version of efficiency metrics.
    """
    displacement_efficiency_score = compute_displacement_efficiency(ego_planned_xy, ego_gt_xy, ego_gt_valid, dt)
    violation_rate, stagnation_rate, excess_speed_rate, nonstopping_rate = compute_movement_adequacy(ego_planned_xy, ego_gt_xy, ego_gt_valid, dt, stagnation_threshold, expected_speed_range[1])
    directional_consistency_score = compute_directional_consistency(ego_planned_xy, ego_gt_xy, ego_gt_valid, dt)
    return {
        'displacement_efficiency_score': displacement_efficiency_score,
        'speed_violation_rate': violation_rate,
        'stagnation_rate': stagnation_rate,
        'excess_speed_rate': excess_speed_rate,
        'nonstopping_rate': nonstopping_rate,
        'directional_consistency_score': directional_consistency_score,
    }
