import numpy as np
import math
import tqdm
from copy import deepcopy 
from utils.data_utils import get_yaw_rotation_matrix, transform_back_component_coordinates, wrap_angle

METRIC_INPUT_DICT= {'scenario_ids':[],'object_ids':{},'object_types':{},'forecasted_confidences':{},
                    'forecasted_trajectories':{}, 'gt_trajectories':{},'gt_is_valid':{}}

AV_OFFSET = 1

def accumulate_metrics_input_joint(model_output, mapping):
    pred_trajs,pred_scores = model_output['pred_trajs'],pred_scores['pred_scores']
    batch_size = len(pred_trajs)
    eval_dict = deepcopy(METRIC_INPUT_DICT)
    # for joint prediction metrics computation
    for i in range(batch_size):
        batch_data = mapping[i]
        # collect scenario_id
        scenario_id = mapping[i]['scenario_id']
        eval_dict['scenario_ids'].append(scenario_id)
        # initiate each scenario info
        for key in eval_dict.keys():
            if key != 'scenario_id':
                eval_dict[key][scenario_id] = []
        # collect obs_ids and obs_types [num_scenarios, num_groups,c num_agents]
        obs_ids = batch_data.valid_agents_ids[AV_OFFSET:batch_data.target_agents_num] # [num_agents, ]
        obs_types = batch_data.valid_agents_types[AV_OFFSET:batch_data.target_agents_num] # [num_agents, ]
        assert len(obs_ids) == len(obs_types) == 2, f'obs_of_interest num: {len(obs_ids)}, {len(obs_types)}'
        eval_dict['object_ids'][scenario_id].append(obs_ids)
        eval_dict['object_types'][scenario_id].append(obs_types)
        # collect prediction score
        single_batch_pred_scores = pred_scores[i][AV_OFFSET:] # [num_agents, num_modes]
        # for test
        sorted_score_idx = np.argsort(single_batch_pred_scores, axis=-1)[:, ::-1] # decending order
        single_batch_pred_scores = np.take_along_axis(single_batch_pred_scores, sorted_score_idx, axis=-1)
        assert len(single_batch_pred_scores) == 2
        single_batch_joint_pred_scores = single_batch_pred_scores.sum(axis=0) / single_batch_pred_scores.sum() # normalized to [num_modes, ]
        eval_dict['forecasted_confidences'][scenario_id].append(single_batch_joint_pred_scores)
        # collect prediction traj
        single_batch_pred_trajs = pred_trajs[i].cpu().numpy().astype(dtype=np.float32)[AV_OFFSET:,:,4::5,:] # [num_agents, num_modes, num_pred_steps, 2]
        # for test
        single_batch_pred_trajs = np.take_along_axis(single_batch_pred_trajs, sorted_score_idx[:,:,None,None], axis=1) # decending order trajs
        assert len(single_batch_pred_trajs) == 2
        ## convert pred from per-agent coordinate system to ego coordinate system
        batch_target_current_info = batch_data.target_current_info[AV_OFFSET:] # [num_agents, 3]
        yaw_rotation_matrix_2d = get_yaw_rotation_matrix(batch_target_current_info[:, -1])[:, :2, :2] # [num_agents, 2, 2]
        single_batch_pred_trajs = transform_back_component_coordinates(single_batch_pred_trajs,
                                                                       yaw_rotation_matrix_2d,
                                                                       batch_target_current_info[:, :2]) # [num_agents, num_modes, num_pred_steps, 2]
        eval_dict['forecasted_trajectories'][scenario_id].append(single_batch_pred_trajs.transpose(1,0,2,3)) # [num_modes, num_agents, num_pred_steps, 2]
        # collect gt states and valid flags
        eval_dict['gt_trajectories'][scenario_id].append(batch_data.scene_centric_target_gt_for_eval[AV_OFFSET:])
        eval_dict['gt_is_valid'][scenario_id].append(batch_data.target_gt_valid[AV_OFFSET:])
    return eval_dict
       
def accumulate_metrics_input_marginal(model_output, mapping):
    """
    for marginal prediction metrics computation, num_groups = num_agents, num_agents = 1

    """
    pred_trajs,pred_scores = model_output['pred_trajs'],model_output['pred_scores']
    batch_size = len(pred_trajs)
    eval_dict = deepcopy(METRIC_INPUT_DICT)
    for i in range(batch_size):
        batch_data = mapping[i]
        # collect scenario_id
        scenario_id = batch_data.scenario_id
        eval_dict['scenario_ids'].append(scenario_id)
        # collect obs_ids and obs_types [num_scenarios, num_groups, num_agents], where num_agents = 1
        # obs_ids = batch_data.valid_agents_ids[AV_OFFSET:batch_data.target_agents_num][:, None] # [num_groups, num_agents=1]
        obs_ids = [batch_data.valid_agents_ids[i] for i in range(batch_data.target_agents_num)]
        obs_types = batch_data.valid_agents_types[AV_OFFSET:batch_data.target_agents_num][:, None] # [num_groups, num_agents=1]
        print(obs_ids, obs_types)
        eval_dict['object_ids'][scenario_id] = obs_ids
        eval_dict['object_types'][scenario_id] = obs_types.astype(np.int32)
        # collect prediction score
        single_batch_pred_scores = pred_scores[i][AV_OFFSET:] # [num_groups, num_modes]
        # collect prediction traj
        single_batch_pred_trajs = pred_trajs[i][AV_OFFSET:,:,4::5,:] # [num_groups, num_modes, num_pred_steps, 2]
        ## convert pred from per-agent coordinate system to ego coordinate system
        batch_target_current_info = batch_data.target_current_info[AV_OFFSET:] # [num_agents, 3]
        yaw_rotation_matrix_2d = get_yaw_rotation_matrix(batch_target_current_info[:, -1])[:, :2, :2] # [num_groups, 2, 2]
        single_batch_pred_trajs = transform_back_component_coordinates(single_batch_pred_trajs,
                                                                       yaw_rotation_matrix_2d,
                                                                       batch_target_current_info[:, :2]) # [num_groups, num_modes, num_pred_steps, 2]
        single_batch_pred_trajs = single_batch_pred_trajs[:,:,None,:,:] # [num_groups, num_modes, num_agents=1, num_pred_steps, 2]
        eval_dict['forecasted_confidences'][scenario_id] = single_batch_pred_scores
        eval_dict['forecasted_trajectories'][scenario_id] = single_batch_pred_trajs
        # collect gt states and valid flags
        assert batch_data.scene_centric_target_gt_for_eval.shape[-1] == 7
        eval_dict['gt_trajectories'][scenario_id] = batch_data.scene_centric_target_gt_for_eval[AV_OFFSET:,None,:,:]  # [num_groups, num_agents=1, num_gt_steps, 7]
        eval_dict['gt_is_valid'][scenario_id] = batch_data.target_gt_valid[AV_OFFSET:,None,:]  # [num_groups, num_agents=1, num_gt_steps]
    return eval_dict

def ego_planning_metrics_preprocess(
    ego_planning_trajectories,
    ego_planning_confidences,
    ego_size,
    other_xy,
    other_heading,
    other_size,
    other_num,
    other_valid_mask,
    ego_gt_xy,
    ego_valid_mask
):
    B = len(ego_planning_trajectories)
    repeated_indices = np.repeat(np.arange(B), other_num)

    ego_xy_batch = np.stack(ego_planning_trajectories, axis=0) # [B, N_modes, T, 2]
    ego_conf_batch = np.stack(ego_planning_confidences, axis=0) # [B, N_modes]
    ego_size_batch = np.stack(ego_size, axis=0) # [B, 2]
    
    other_xy_batch = np.concatenate(other_xy, axis=0) # [BA, T, 2]
    BA, T = other_xy_batch.shape[:2]
    other_heading_batch = np.concatenate(other_heading, axis=0) # [BA, T]
    other_size_batch = np.concatenate(other_size, axis=0) # [BA, 2]
    other_valid_mask_batch = np.concatenate(other_valid_mask, axis=0) # [BA, T]

    if ego_size_batch.ndim == 2:
        ego_size_batch = ego_size_batch[..., np.newaxis, :].repeat(T, axis=-2)
    if other_size_batch.ndim == 2:
        other_size_batch = other_size_batch[..., np.newaxis, :].repeat(T, axis=-2)

    best_mode_idxs = np.argmax(ego_conf_batch, axis=-1) # [B]
    ego_xy_batch = ego_xy_batch[np.arange(B), best_mode_idxs] # [B, T, 2]
    ego_vel_batch = np.diff(ego_xy_batch, axis=-2, prepend=ego_xy_batch[:, :1]/0.1) # [B, T, 2]
    ego_heading_batch = np.arctan2(ego_vel_batch[:, :, 1], np.clip(ego_vel_batch[:, :, 0], a_min=1e-6, a_max=None)) # [B, T]

    ego_gt_xy_batch = np.stack(ego_gt_xy, axis=0)
    ego_valid_mask_batch = np.stack(ego_valid_mask, axis=0) # [B, T]
    return ego_xy_batch, ego_heading_batch, ego_size_batch, \
        other_xy_batch, other_heading_batch, other_size_batch, \
        other_num, repeated_indices, other_valid_mask_batch.astype(bool), \
        ego_gt_xy_batch, ego_valid_mask_batch.astype(bool)

# ***********************************
# ***SAT-based collision detection***
# ***********************************

def get_obb_corners_vectorized(centers, sizes, headings, margin=0.0):
    """
    Vectorized version to get the four corners of oriented bounding boxes.
    
    Args:
        centers: [..., 2] center positions (x, y)
        sizes: [..., 2] sizes (length, width)  
        headings: [...] orientations in radians
        margin: float, additional safety margin
        
    Returns:
        corners: [..., 4, 2] four corner points for each box
    """
    # Add margin to sizes
    sizes_with_margin = sizes + 2 * margin
    half_sizes = sizes_with_margin / 2
    
    # Corner offsets in local frame: [..., 4, 2]
    corners_local = np.stack([
        np.stack([-half_sizes[..., 0], -half_sizes[..., 1]], axis=-1),
        np.stack([half_sizes[..., 0], -half_sizes[..., 1]], axis=-1),
        np.stack([half_sizes[..., 0], half_sizes[..., 1]], axis=-1),
        np.stack([-half_sizes[..., 0], half_sizes[..., 1]], axis=-1)
    ], axis=-2)
    
    # Rotation matrices: [..., 2, 2] 
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    rot_matrices = np.stack([
        np.stack([cos_h, -sin_h], axis=-1),
        np.stack([sin_h, cos_h], axis=-1)
    ], axis=-2)
    
    # Transform to global frame: [..., 4, 2]
    corners_global = np.matmul(corners_local, rot_matrices.swapaxes(-2, -1)) + centers[..., np.newaxis, :]
    return corners_global

def get_axes_vectorized(corners):
        """Get perpendicular axes to edges."""
        # corners: [..., 4, 2]
        edges = np.roll(corners, shift=-1, axis=-2) - corners  # [..., 4, 2]
        axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)  # [..., 4, 2]
        
        # Normalize axes
        norms = np.linalg.norm(axes, axis=-1, keepdims=True)  # [..., 4, 1]
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return axes / norms  # [..., 4, 2]
    
def project_vectorized(corners, axes):
    """Project corners onto axes."""
    # corners: [..., 4, 2], axes: [..., 4, 2]
    projections = np.sum(corners[..., :, np.newaxis, :] * axes[..., np.newaxis, :, :], axis=-1)  # [..., 4, 4]
    return np.min(projections, axis=-2), np.max(projections, axis=-2)  # [..., 4]

def check_obb_overlap_vectorized(ego_xy, ego_heading, ego_size, other_xy, other_heading, other_size, other_valid):
    """
    Vectorized SAT-based collision detection for multiple scenarios, agents, and timesteps.
    
    Args:
        ego_xy: [B, T, 2] ego vehicle positions
        ego_heading: [B, T] ego vehicle headings  
        ego_size: [B, 2] ego vehicle sizes
        other_xy: [B, A, T, 2] other vehicle positions
        other_heading: [B, A, T] other vehicle headings
        other_size: [B, A, 2] other vehicle sizes
        other_valid: [B, A, T] other vehicle valid masks
        
    Returns:
        collision_masks: [B, T] collision status per scenario and timestep
    """
    B, T = ego_xy.shape[:2]
    A = other_xy.shape[1]
    
    # Get ego corners: [B, T, 4, 2]
    ego_corners = get_obb_corners_vectorized(ego_xy, ego_size[:, np.newaxis, :], ego_heading)
    
    # Get other corners: [B, A, T, 4, 2]  
    other_corners = get_obb_corners_vectorized(other_xy, other_size[:, :, np.newaxis, :], other_heading)
    
    # Prepare for broadcasting: ego [B, 1, T, 4, 2], other [B, A, T, 4, 2]
    ego_corners_bc = ego_corners[:, np.newaxis, :, :, :]  # [B, 1, T, 4, 2]
    
    # Check collisions using vectorized SAT
    overlap_masks = sat_overlap_vectorized(ego_corners_bc, other_corners)  # [B, A, T]
    
    # Apply valid mask - only consider valid other vehicles
    overlap_masks = overlap_masks & other_valid  # [B, A, T]
    
    # Collision at timestep t if ANY other vehicle collides with ego
    collision_masks = np.any(overlap_masks, axis=1)  # [B, T]
    
    return collision_masks

def sat_overlap_vectorized(corners1, corners2):
    """
    Vectorized Separating Axis Theorem overlap detection.
    
    Args:
        corners1: [..., 4, 2] first set of OBB corners
        corners2: [..., 4, 2] second set of OBB corners
        
    Returns:
        overlap: [...] boolean array indicating overlap
    """
    def get_axes_vectorized(corners):
        """Get perpendicular axes to edges."""
        # corners: [..., 4, 2]
        edges = np.roll(corners, shift=-1, axis=-2) - corners  # [..., 4, 2]
        axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)  # [..., 4, 2]
        
        # Normalize axes
        norms = np.linalg.norm(axes, axis=-1, keepdims=True)  # [..., 4, 1]
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return axes / norms  # [..., 4, 2]
    
    def project_vectorized(corners, axes):
        """Project corners onto axes."""
        # corners: [..., 4, 2], axes: [..., 4, 2]
        projections = np.sum(corners[..., :, np.newaxis, :] * axes[..., np.newaxis, :, :], axis=-1)  # [..., 4, 4]
        return np.min(projections, axis=-2), np.max(projections, axis=-2)  # [..., 4]
    
    # Get axes for both sets of corners
    axes1 = get_axes_vectorized(corners1)  # [..., 4, 2]
    axes2 = get_axes_vectorized(corners2)  # [..., 4, 2]
    
    # Check separations along all axes
    # For axes from corners1
    min1_on_axes1, max1_on_axes1 = project_vectorized(corners1, axes1)  # [..., 4]
    min2_on_axes1, max2_on_axes1 = project_vectorized(corners2, axes1)  # [..., 4]
    
    separated1 = (max1_on_axes1 < min2_on_axes1) | (max2_on_axes1 < min1_on_axes1)  # [..., 4]
    
    # For axes from corners2  
    min1_on_axes2, max1_on_axes2 = project_vectorized(corners1, axes2)  # [..., 4]
    min2_on_axes2, max2_on_axes2 = project_vectorized(corners2, axes2)  # [..., 4]
    
    separated2 = (max1_on_axes2 < min2_on_axes2) | (max2_on_axes2 < min1_on_axes2)  # [..., 4]
    
    # If separated along ANY axis, no overlap
    any_separated = np.any(separated1, axis=-1) | np.any(separated2, axis=-1)  # [...]
    
    return ~any_separated  # [...] overlap = not separated

# *********************
# ***Comfort metrics***
# *********************

def compute_kinematics(ego_xy, dt=0.1):
    """
    Compute kinematics from ego_xy.
    """
    B, T = ego_xy.shape[:2]
    ego_xy_for_diff = np.concatenate((np.zeros_like(ego_xy[:, :1]), ego_xy), axis=-2) # [B, T+1, 2]
    vel = np.diff(ego_xy_for_diff, axis=-2)/dt # [B, T, 2]
    speed = np.linalg.norm(vel, axis=-1).clip(min=1e-6) # [B, T]
    accel = np.diff(vel, axis=-2)/dt # [B, T-1, 2]
    jerk = np.diff(accel, axis=-2)/dt # [B, T-2, 2]
    heading = np.arctan2(vel[:, :, 1], np.clip(vel[:, :, 0], a_min=1e-6, a_max=None)) # [B, T]
    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))
    heading_for_diff = np.concatenate((np.zeros_like(heading[:, :1]), heading), axis=-1) # [B, T+1]
    delta_heading = angle_diff(heading_for_diff[...,1:], heading_for_diff[...,:-1]) # [B, T]
    yaw_rate = delta_heading/ dt

    # decompose accel to long and lat
    tangent = vel / speed[:, :, np.newaxis] # [B, T, 2]
    left_normal = np.stack([-tangent[:, :, 1], tangent[:, :, 0]], axis=-1) # [B, T, 2]
    valid_T  = accel.shape[-2]
    long_accel = (accel * tangent[:, :valid_T]).sum(axis=-1) # [B, T-1]
    lat_accel = (accel * left_normal[:, :valid_T]).sum(axis=-1) # [B, T-1]

    # jerk magnitude
    jerk_magnitude = np.linalg.norm(jerk, axis=-1) # [B, T]

    return vel, speed, accel, jerk, heading, yaw_rate, long_accel, lat_accel, jerk_magnitude
    
# ************************
# ***Efficiency metrics***
# ************************
    
def compute_displacement_efficiency(ego_planned_xy, ego_gt_xy, ego_gt_valid, dt=0.1):
    """
    Compute displacement efficiency.
    """
    planned_displacements = np.diff(ego_planned_xy, axis=-2) # [B, T-1, 2]
    planned_magnitudes = np.linalg.norm(planned_displacements, axis=-1) # [B, T-1]
    gt_displacements = np.diff(ego_gt_xy, axis=-2) # [B, T-1, 2]
    gt_magnitudes = np.linalg.norm(gt_displacements, axis=-1) # [B, T-1]
    gt_displacements_valid_mask = ego_gt_valid[..., :-1] & ego_gt_valid[..., 1:] # [B, T-1]
    gt_directions = gt_displacements / (gt_magnitudes[..., np.newaxis] + 1e-6) # [B, T-1, 2]
    
    # Extract valid values by masking
    planned_displacements = planned_displacements * gt_displacements_valid_mask[..., np.newaxis] # [B, T-1, 2]
    planned_magnitudes = planned_magnitudes * gt_displacements_valid_mask
    gt_directions = gt_directions * gt_displacements_valid_mask[..., np.newaxis] # [B, T-1, 2]
    gt_magnitudes = gt_magnitudes * gt_displacements_valid_mask

    # Compute efficiency ratio
    directional_projection = (planned_displacements * gt_directions).sum(axis=-1) # [B, T-1]
    efficiency_ratio = ((directional_projection + 1e-6) / (planned_magnitudes + 1e-6)) # [B, T-1]

    efficiency_ratio = np.clip(efficiency_ratio, a_min=0.0, a_max=1.0)
    efficiency_ratio_at_t = np.mean(efficiency_ratio, axis=-1)
    efficiency_ratio_per_scenario = np.mean(efficiency_ratio_at_t)
    return efficiency_ratio_per_scenario # [B, T-1]

def compute_movement_adequacy(ego_planned_xy, ego_gt_xy, ego_gt_valid, dt=0.1, stagnation_threshold=1.0, excess_speed_threshold=30.0):
    planned_velocities = np.diff(ego_planned_xy, axis=-2) / dt # [B, T-1, 2]
    planned_speeds = np.linalg.norm(planned_velocities, axis=-1) # [B, T-1]
    gt_velocities = np.diff(ego_gt_xy, axis=-2) / dt # [B, T-1, 2]
    gt_speeds = np.linalg.norm(gt_velocities, axis=-1) # [B, T-1]
    
    gt_stop_mask = gt_speeds < stagnation_threshold # [B, T-1]
    gt_stop_mask = gt_stop_mask & ego_gt_valid[..., :-1] & ego_gt_valid[..., 1:] # [B, T-1]
    gt_nonstop_mask = np.logical_not(gt_stop_mask) # [B, T-1]

    gt_stop_scenario_num = np.sum(gt_stop_mask)
    gt_nonstop_scenario_num = np.sum(gt_nonstop_mask)

    stagnation_mask =((stagnation_threshold - planned_speeds) > 0.0) & gt_nonstop_mask # [B, T-1]
    excess_speed_mask = (planned_speeds - excess_speed_threshold) > 0.0 # [B, T-1]
    nonstopping_mask = (planned_speeds > stagnation_threshold) & gt_stop_mask # [B, T-1]

    total_violation_mask = stagnation_mask | excess_speed_mask | nonstopping_mask # [B, T-1]
    violation_rate = np.mean(total_violation_mask)
    stagnation_rate = np.sum(stagnation_mask) / max(gt_nonstop_scenario_num, 1)
    excess_speed_rate = np.sum(excess_speed_mask) / max(gt_nonstop_scenario_num, 1)
    nonstopping_rate = np.sum(nonstopping_mask) / max(gt_stop_scenario_num, 1)
    return violation_rate, stagnation_rate, excess_speed_rate, nonstopping_rate

def compute_directional_consistency(ego_planned_xy, ego_gt_xy, ego_gt_valid, dt=0.1):
    # Planned direction
    planned_displacements = np.diff(ego_planned_xy, axis=-2) # [B, T-1, 2]
    planned_magnitudes = np.linalg.norm(planned_displacements, axis=-1) # [B, T-1]
    planned_directions = planned_displacements / (planned_magnitudes[..., np.newaxis] + 1e-6) # [B, T-1, 2]
    
    # GT direction
    gt_displacements = np.diff(ego_gt_xy, axis=-2) # [B, T-1, 2]
    gt_magnitudes = np.linalg.norm(gt_displacements, axis=-1) # [B, T-1]
    gt_directions = gt_displacements / (gt_magnitudes[..., np.newaxis] + 1e-6) # [B, T-1, 2]
    gt_directions_valid_mask = ego_gt_valid[..., :-1] & ego_gt_valid[..., 1:] # [B, T-1]
    
    # Extract valid values by masking
    planned_directions = planned_directions * gt_directions_valid_mask[..., np.newaxis] # [B, T-1, 2]
    planned_magnitudes = planned_magnitudes * gt_directions_valid_mask
    gt_directions = gt_directions * gt_directions_valid_mask[..., np.newaxis] # [B, T-1, 2]
    gt_magnitudes = gt_magnitudes * gt_directions_valid_mask

    # Compute directional consistency
    direction_alignment = (planned_directions * gt_directions).sum(axis=-1) # [B, T-1]
    consistency_scores = (direction_alignment + 1.0) / 2.0 # [B, T-1] convert from [-1, 1] to [0, 1]

    consistency_scores_at_t = np.mean(consistency_scores, axis=-1)
    consistency_scores_per_scenario = np.mean(consistency_scores_at_t)
    return consistency_scores_per_scenario
    
    