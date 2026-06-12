import numpy as np
import torch
import torch.distributed as dist
from torchmetrics import Metric
from typing import Dict, Optional
from metrics.joint_motion_metrics import compute_metrics
from metrics.ego_planning_metrics import repeated_indices_numpy_collision_metrics, numpy_comfort_metrics, numpy_efficiency_metrics
from metrics.ego_simple_metrics import _compute_ade_simple, _compute_fde_simple
from utils.metrics_utils import transform_back_component_coordinates, get_yaw_rotation_matrix
from utils.metrics_utils import ego_planning_metrics_preprocess

METRICS_CONFIG = {
    'track_steps_per_second': 10,
    'prediction_steps_per_second': 2,
    'track_history_samples': 0,
    'track_future_samples': 80,
    'speed_lower_bound': 1.4,
    'speed_upper_bound': 11.0,
    'speed_scale_lower': 0.5,
    'speed_scale_upper': 1.0,
    'step_configs': {5: [1.0,2.0], 9: [1.8, 3.6], 15: [3.0, 6.0]},
    'max_predictions': 6
}

class MotionForecastMetric(Metric):
    def __init__(self, is_waymo_dataset, **kwargs):
        super().__init__(**kwargs)
        # States that needed to be synced across devices
        self.add_state('gt_trajectories', default=[], dist_reduce_fx=None)
        self.add_state('gt_is_valid', default=[], dist_reduce_fx=None)
        self.add_state('forecasted_trajectories', default=[], dist_reduce_fx=None)
        self.add_state('forecasted_confidences', default=[], dist_reduce_fx=None)
        self.add_state('object_types', default=[], dist_reduce_fx=None)
        self.add_state('object_ids', default=[], dist_reduce_fx=None)
        self.add_state('scenario_ids', default=[], dist_reduce_fx=None)
        self.metrics_config = METRICS_CONFIG
        self.is_waymo_dataset = is_waymo_dataset
        self.return_AP_details = False
        self.AV_OFFSET = 1

    def update(self, model_output: dict, metrics_input: dict):
        batch_size = len(metrics_input['scenario_ids'])
        for i in range(batch_size):
            target_agents_num = len(metrics_input['gt_trajectories'][i])
            self.gt_trajectories.append(metrics_input['gt_trajectories'][i][self.AV_OFFSET:target_agents_num, None, :, :]) # [num_agents, num_groups=1, num_gt_steps, 7]
            self.gt_is_valid.append(metrics_input['gt_is_valid'][i][self.AV_OFFSET:target_agents_num, None, :]) # [num_agents, num_groups=1, num_gt_steps]
            self.object_types.append(metrics_input['object_types'][i][self.AV_OFFSET:target_agents_num][:, None]) # [num_agents, num_groups=1]
            self.object_ids.append(metrics_input['object_ids'][i][self.AV_OFFSET:target_agents_num])
            self.scenario_ids.append(metrics_input['scenario_ids'][i])
            single_batch_pred_scores = model_output['pred_scores'][i][self.AV_OFFSET:] # [num_agents, num_modes]
            single_batch_pred_trajs  = model_output['pred_trajs'][i][self.AV_OFFSET:,:,4::5,:] # [num_agents, num_modes, num_pred_steps, 2]
            batch_target_current_info = metrics_input['target_current_info'][i][self.AV_OFFSET:] 
            yaw_rotation_matrix_2d = get_yaw_rotation_matrix(batch_target_current_info[:, -1])[:, :2, :2] 
            single_batch_pred_trajs = transform_back_component_coordinates(single_batch_pred_trajs,
                                                                           yaw_rotation_matrix_2d,
                                                                           batch_target_current_info[:, :2]) # [num_agents, num_modes, num_pred_steps, 2]
            single_batch_pred_trajs = single_batch_pred_trajs[:,:,None,:,:] # [num_agents, num_modes, num_groups=1, num_pred_steps, 2]
            self.forecasted_trajectories.append(single_batch_pred_trajs) # [num_agents, num_modes, num_groups=1, num_pred_steps, 2]
            self.forecasted_confidences.append(single_batch_pred_scores) # [num_agents, num_modes]

    def compute(self):
        return compute_metrics(
            self.gt_trajectories, 
            self.gt_is_valid, 
            self.forecasted_trajectories, 
            self.forecasted_confidences, 
            self.object_types, 
            self.object_ids, 
            self.scenario_ids, 
            self.metrics_config,
            self.is_waymo_dataset,
            self.return_AP_details
        )
    

class JointMotionForecastMetric(Metric):
    def __init__(self, is_waymo_dataset, **kwargs):
        super().__init__(**kwargs)
        # States that needed to be synced across devices
        self.add_state('gt_trajectories', default=[], dist_reduce_fx=None)
        self.add_state('gt_is_valid', default=[], dist_reduce_fx=None)
        self.add_state('forecasted_trajectories', default=[], dist_reduce_fx=None)
        self.add_state('forecasted_confidences', default=[], dist_reduce_fx=None)
        self.add_state('object_types', default=[], dist_reduce_fx=None)
        self.add_state('object_ids', default=[], dist_reduce_fx=None)
        self.add_state('scenario_ids', default=[], dist_reduce_fx=None)
        self.metrics_config = METRICS_CONFIG
        self.is_waymo_dataset = is_waymo_dataset
        self.return_AP_details = False
        self.AV_OFFSET = 1 # 0 for waymo challenge  or with plan, 1 for dev

    def update(self, model_output: dict, metrics_input: dict):
        batch_size = len(metrics_input['scenario_ids'])
        for i in range(batch_size):
            target_agents_num = len(metrics_input['gt_trajectories'][i])
            self.gt_trajectories.append(metrics_input['gt_trajectories'][i][self.AV_OFFSET:target_agents_num, :, :][None]) # [num_groups=1, num_agents, num_gt_steps, 7]
            self.gt_is_valid.append(metrics_input['gt_is_valid'][i][self.AV_OFFSET:target_agents_num, :][None]) # [num_groups=1, num_agents, num_gt_steps]
            self.object_types.append(metrics_input['object_types'][i][self.AV_OFFSET:target_agents_num][None]) # [num_groups=1, num_agents, ]
            self.scenario_ids.append(metrics_input['scenario_ids'][i])
            self.object_ids.append(metrics_input['object_ids'][i][self.AV_OFFSET:target_agents_num])
            single_batch_pred_scores = model_output['pred_scores'][i] # [num_groups=1, num_modes]
            single_batch_pred_trajs  = model_output['pred_trajs'][i][self.AV_OFFSET:,:,4::5,:] # [num_agents, num_modes, num_pred_steps, 2]
            batch_target_current_info = metrics_input['target_current_info'][i][self.AV_OFFSET:]
            yaw_rotation_matrix_2d = get_yaw_rotation_matrix(batch_target_current_info[:, -1])[:, :2, :2]
            single_batch_pred_trajs = transform_back_component_coordinates(single_batch_pred_trajs,
                                                                        yaw_rotation_matrix_2d,
                                                                        batch_target_current_info[:, :2]) # [num_agents, num_modes, num_pred_steps, 2]
            single_batch_pred_trajs = single_batch_pred_trajs.transpose(1,0,2,3) # [num_modes, num_agents, num_pred_steps, 2]
            self.forecasted_trajectories.append(single_batch_pred_trajs[None]) # [num_groups=1, num_modes, num_agents, num_pred_steps, 2]
            self.forecasted_confidences.append(single_batch_pred_scores[None]) # [num_groups=1, num_modes]

    def compute(self):
        return compute_metrics(
            self.gt_trajectories, 
            self.gt_is_valid, 
            self.forecasted_trajectories, 
            self.forecasted_confidences, 
            self.object_types, 
            self.object_ids, 
            self.scenario_ids, 
            self.metrics_config,
            self.is_waymo_dataset,
            self.return_AP_details
        )
    
    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        super()._sync_dist(dist_sync_fn, process_group)
        self.gt_trajectories = self._gather_ndarray_list(self.gt_trajectories, process_group)
        self.gt_is_valid = self._gather_ndarray_list(self.gt_is_valid, process_group)
        self.forecasted_trajectories = self._gather_ndarray_list(self.forecasted_trajectories, process_group)
        self.forecasted_confidences = self._gather_ndarray_list(self.forecasted_confidences, process_group)
        self.object_types = self._gather_ndarray_list(self.object_types, process_group)
        self.object_ids = self._gather_ndarray_list(self.object_ids, process_group)
        self.scenario_ids = self._gather_ndarray_list(self.scenario_ids, process_group)

    def _gather_ndarray_list(self, list_of_ndarray, process_group=None):
        world_size = dist.get_world_size(group=process_group)
        list_gathered = [None] * world_size
        dist.all_gather_object(list_gathered, list_of_ndarray, group=process_group)
        for rank in range(1, world_size):
            assert (
                len(list_gathered[rank]) == len(list_gathered[0])
            ), f"Length mismatch between rank 0 and rank {rank}: {len(list_gathered[rank])} != {len(list_gathered[0])}"
        list_merged = []
        for rank in range(world_size):
            list_merged.extend(list_gathered[rank])
        return list_merged

class EgoMotionPlanningMetric(Metric):
    def __init__(self, is_waymo_dataset, **kwargs):
        super().__init__(**kwargs)
        # States that needed to be synced across devices
        self.add_state('ego_gt_trajectory', default=[], dist_reduce_fx=None)
        self.add_state('ego_gt_is_valid', default=[], dist_reduce_fx=None)
        self.add_state('ego_forecasted_trajectories', default=[], dist_reduce_fx=None)
        self.add_state('ego_forecasted_confidences', default=[], dist_reduce_fx=None)
        self.add_state('ego_object_types', default=[], dist_reduce_fx=None)
        self.add_state('ego_object_ids', default=[], dist_reduce_fx=None)
        self.add_state('scenario_ids', default=[], dist_reduce_fx=None)

        self.add_state('ego_planning_trajectories', default=[], dist_reduce_fx=None)
        self.add_state('ego_planning_confidences', default=[], dist_reduce_fx=None)
        self.add_state('ego_lw', default=[], dist_reduce_fx=None)
        self.add_state('other_gt_xy', default=[], dist_reduce_fx=None)
        self.add_state('other_gt_heading', default=[], dist_reduce_fx=None)
        self.add_state('other_gt_is_valid', default=[], dist_reduce_fx=None)
        self.add_state('other_lw', default=[], dist_reduce_fx=None)
        self.add_state('agents_num', default=[], dist_reduce_fx=None)
        self.add_state('ego_gt_xy', default=[], dist_reduce_fx=None)
        self.add_state('ego_gt_xy_is_valid', default=[], dist_reduce_fx=None)
        self.metrics_config = METRICS_CONFIG
        self.is_waymo_dataset = is_waymo_dataset
        self.return_AP_details = False
        self.AV_OFFSET = 1

    def update(self, model_output: dict, metrics_input: dict):
        batch_size = len(metrics_input['scenario_ids'])
        for i in range(batch_size):
            self.ego_gt_trajectory.append(metrics_input['gt_trajectories'][i][[0], ...][None]) # [num_groups=1, num_agents, num_gt_steps, 7]
            self.ego_gt_is_valid.append(metrics_input['gt_is_valid'][i][[0], ...][None]) # [num_groups=1, num_agents, num_gt_steps]
            self.ego_object_types.append(metrics_input['object_types'][i][[0], ...][None]) # [num_groups=1, num_agents, ]
            self.scenario_ids.append(metrics_input['scenario_ids'][i])
            self.ego_object_ids.append(metrics_input['object_ids'][i][0])
            single_batch_pred_trajs = model_output['pred_trajs'][i][[0], :, 4::5, :].transpose(1, 0, 2, 3) # [num_modes, num_agents, num_pred_steps, 2]
            signle_batch_pred_scores = model_output['pred_scores'][i] # [num_modes, ]
            self.ego_forecasted_trajectories.append(single_batch_pred_trajs[None]) # [num_groups=1, num_modes, num_agents, num_pred_steps, 2]
            self.ego_forecasted_confidences.append(signle_batch_pred_scores[None]) # [num_groups=1, num_modes]
            self.ego_planning_trajectories.append(model_output['pred_trajs'][i][0]) # [num_modes, num_pred_steps, 2]
            self.ego_planning_confidences.append(model_output['pred_scores'][i]) # [num_modes, ]
            self.ego_lw.append(metrics_input['target_lw'][i][0, ...]) # [2, ]
            self.other_gt_xy.append(metrics_input['gt_trajectories'][i][self.AV_OFFSET:, :, :2]) # [num_agents, num_gt_steps, 2]
            self.other_gt_heading.append(metrics_input['gt_trajectories'][i][self.AV_OFFSET:, :, 4]) # [num_agents, num_gt_steps]
            self.other_gt_is_valid.append(metrics_input['gt_is_valid'][i][self.AV_OFFSET:, ...]) # [num_agents, num_gt_steps]
            self.other_lw.append(metrics_input['target_lw'][i][self.AV_OFFSET:, ...]) # [num_agents, 2]
            self.agents_num.append(metrics_input['agents_num'][i]-1)
            self.ego_gt_xy.append(metrics_input['gt_trajectories'][i][0, :, :2])
            self.ego_gt_xy_is_valid.append(metrics_input['gt_is_valid'][i][0, ...]) # [num_groups=1, num_agents, num_gt_steps]
    def compute(self):
        metric_results = {}
        ego_motion_metrics = compute_metrics(
            self.ego_gt_trajectory, 
            self.ego_gt_is_valid, 
            self.ego_forecasted_trajectories, 
            self.ego_forecasted_confidences, 
            self.ego_object_types, 
            self.ego_object_ids, 
            self.scenario_ids, 
            self.metrics_config,
            self.is_waymo_dataset,
            self.return_AP_details
        )
        metric_results['EGO'] = ego_motion_metrics.pop('VEHICLE')
        ego_xy, ego_heading, ego_size, \
            other_xy, other_heading, other_size, \
            other_num, repeated_indices, other_valid_mask_batch, \
                ego_gt_xy_batch, ego_valid_mask_batch = ego_planning_metrics_preprocess(  
            self.ego_planning_trajectories,
            self.ego_planning_confidences,
            self.ego_lw,
            self.other_gt_xy,
            self.other_gt_heading,
            self.other_lw,
            self.agents_num,
            self.other_gt_is_valid,
            self.ego_gt_xy,
            self.ego_gt_xy_is_valid
        )
        metric_results['EGO']['collision_rate'] = repeated_indices_numpy_collision_metrics(
            ego_xy, ego_heading, ego_size, \
            other_xy, other_heading, other_size, \
            other_num, repeated_indices, other_valid_mask_batch)['collision_rate']
        metric_results['EGO']['comfort_metrics'] = numpy_comfort_metrics(ego_xy)
        metric_results['EGO']['efficiency_metrics'] = numpy_efficiency_metrics(ego_xy, ego_gt_xy_batch, ego_valid_mask_batch)

        return metric_results

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        super()._sync_dist(dist_sync_fn, process_group)
        self.ego_gt_trajectory = self._gather_ndarray_list(self.ego_gt_trajectory, process_group)
        self.ego_gt_is_valid = self._gather_ndarray_list(self.ego_gt_is_valid, process_group)
        self.ego_forecasted_trajectories = self._gather_ndarray_list(self.ego_forecasted_trajectories, process_group)
        self.ego_forecasted_confidences = self._gather_ndarray_list(self.ego_forecasted_confidences, process_group)
        self.ego_object_types = self._gather_ndarray_list(self.ego_object_types, process_group)
        self.ego_object_ids = self._gather_ndarray_list(self.ego_object_ids, process_group)
        self.scenario_ids = self._gather_ndarray_list(self.scenario_ids, process_group)
        self.ego_planning_trajectories = self._gather_ndarray_list(self.ego_planning_trajectories, process_group)
        self.ego_planning_confidences = self._gather_ndarray_list(self.ego_planning_confidences, process_group)
        self.ego_lw = self._gather_ndarray_list(self.ego_lw, process_group)
        self.other_gt_xy = self._gather_ndarray_list(self.other_gt_xy, process_group)
        self.other_gt_heading = self._gather_ndarray_list(self.other_gt_heading, process_group)
        self.other_gt_is_valid = self._gather_ndarray_list(self.other_gt_is_valid, process_group)
        self.other_lw = self._gather_ndarray_list(self.other_lw, process_group)
        self.agents_num = self._gather_ndarray_list(self.agents_num, process_group)
        self.ego_gt_xy = self._gather_ndarray_list(self.ego_gt_xy, process_group)
        self.ego_gt_xy_is_valid = self._gather_ndarray_list(self.ego_gt_xy_is_valid, process_group)

    def _gather_ndarray_list(self, list_of_ndarray, process_group=None):
        world_size = dist.get_world_size(group=process_group)
        list_gathered = [None] * world_size
        dist.all_gather_object(list_gathered, list_of_ndarray, group=process_group)
        for rank in range(1, world_size):
            assert (
                len(list_gathered[rank]) == len(list_gathered[0])
            ), f"Length mismatch between rank 0 and rank {rank}: {len(list_gathered[rank])} != {len(list_gathered[0])}"
        list_merged = []
        for rank in range(world_size):
            list_merged.extend(list_gathered[rank])
        return list_merged


class EgoSimpleMetric(Metric):
    """
    简化版自车指标统计类
    计算自车的3秒、5秒、8秒的ADE和FDE指标
    
    输入维度:
        - pred_trajs: [B, 1, 80, 3]
        - pred_fix_distace_path: [B, 1, 80, 3]
        - gt_trajectory: [B, 80, 5]
        - gt_mask: [B, 80]
        - gt_fix_distance_path: [B, 80, 5]
        - gt_mask_fix_distance_path: [B, 80]
    """
    
    def __init__(self, pred_interval: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.pred_interval = pred_interval
        
        self.add_state('pred_trajs_list', default=[], dist_reduce_fx=None)
        self.add_state('gt_trajs_list', default=[], dist_reduce_fx=None)
        self.add_state('gt_mask_list', default=[], dist_reduce_fx=None)
        
        self.add_state('pred_path_list', default=[], dist_reduce_fx=None)
        self.add_state('gt_path_list', default=[], dist_reduce_fx=None)
        self.add_state('gt_mask_path_list', default=[], dist_reduce_fx=None)
        self.add_state('multi_traj_mean_cosline_list', default=[], dist_reduce_fx=None)

    def update(self, model_output_dict: Dict, ego_future_dict: Dict):

        pred_trajs = model_output_dict['pred_trajs']  # [B, num_samples, 1, 80, 3]
        pred_fix_distance_path = model_output_dict['pred_fix_distace_path']  # [B, num_samples, 1, 80, 3]

        gt_trajectory = ego_future_dict['ego_future_status']  # [B, 80, 5]
        gt_mask = ego_future_dict['ego_future_mask']  # [B, 80]
        gt_fix_distance_path = ego_future_dict['ego_future_status_fixed']  # [B, 80, 5]
        gt_mask_fix_distance_path = ego_future_dict['ego_future_mask_fixed']  # [B, 80]
        
        B = pred_trajs.shape[0]
        num_samples = pred_trajs.shape[1]
        
        for b in range(B):
            pred_trajs_b = pred_trajs[b]  # [num_samples, 1, 80, 3]
            pred_fix_distance_path_b = pred_fix_distance_path[b]  # [num_samples, 1, 80, 3]
            gt_trajectory_b = gt_trajectory[b:b+1]  # [1, 80, 5]
            gt_mask_b = gt_mask[b:b+1]  # [1, 80]
            gt_fix_distance_path_b = gt_fix_distance_path[b:b+1]  # [1, 80, 5]
            gt_mask_fix_distance_path_b = gt_mask_fix_distance_path[b:b+1]  # [1, 80]
            
            # 计算多模态相似度并选择最佳轨迹
            multi_traj_mean_cosline = None
            if num_samples > 1:
                multi_traj_mean_cosline = self._compute_nulti_traj_cosline(pred_trajs_b)
                pred_trajs_b, pred_fix_distance_path_b = self._select_best_trajectory(
                    pred_trajs_b, pred_fix_distance_path_b, gt_trajectory_b, gt_mask_b,
                    gt_fix_distance_path_b, gt_mask_fix_distance_path_b
                )
            
            if isinstance(pred_trajs_b, torch.Tensor):
                pred_trajs_b = pred_trajs_b.detach().cpu().numpy()
            if isinstance(pred_fix_distance_path_b, torch.Tensor):
                pred_fix_distance_path_b = pred_fix_distance_path_b.detach().cpu().numpy()
            if isinstance(gt_trajectory_b, torch.Tensor):
                gt_trajectory_b = gt_trajectory_b.detach().cpu().numpy()
            if isinstance(gt_mask_b, torch.Tensor):
                gt_mask_b = gt_mask_b.detach().cpu().numpy()
            if isinstance(gt_fix_distance_path_b, torch.Tensor):
                gt_fix_distance_path_b = gt_fix_distance_path_b.detach().cpu().numpy()
            if isinstance(gt_mask_fix_distance_path_b, torch.Tensor):
                gt_mask_fix_distance_path_b = gt_mask_fix_distance_path_b.detach().cpu().numpy()
            
            pred_xy = pred_trajs_b[0, 0, :, :2]  # [80, 2]
            pred_path_xy = pred_fix_distance_path_b[0, 0, :, :2]  # [80, 2]
            gt_xy = gt_trajectory_b[0, :, :2]  # [80, 2]
            gt_path_xy = gt_fix_distance_path_b[0, :, :2]  # [80, 2]
            
            self.pred_trajs_list.append(pred_xy)
            self.gt_trajs_list.append(gt_xy)
            self.gt_mask_list.append(gt_mask_b[0])
            self.pred_path_list.append(pred_path_xy)
            self.gt_path_list.append(gt_path_xy)
            self.gt_mask_path_list.append(gt_mask_fix_distance_path_b[0])
            
            if multi_traj_mean_cosline is not None:
                self.multi_traj_mean_cosline_list.append(multi_traj_mean_cosline)
            
    def _select_best_trajectory(self, pred_trajs, pred_fix_distance_path, 
                                 gt_trajectory, gt_mask,
                                 gt_fix_distance_path, gt_mask_fix_distance_path):

        pred_xy = pred_trajs[:, 0, :, :2]  # [num_samples, 80, 2]
        gt_xy = gt_trajectory[0, :, :2]    # [80, 2]
        mask = gt_mask[0]                  # [80]
        
        errors = torch.norm(pred_xy - gt_xy.unsqueeze(0), dim=-1)  # [num_samples, 80]
        mask_expanded = mask.unsqueeze(0).float()  # [1, 80]
        masked_errors = errors * mask_expanded  # [num_samples, 80]
        valid_count = mask_expanded.sum().clamp(min=1)
        ade_per_sample = masked_errors.sum(dim=-1) / valid_count  # [num_samples]
        best_idx = ade_per_sample.argmin()
        best_pred_trajs = pred_trajs[best_idx:best_idx+1]  # [1, 1, 80, 3]
        
        pred_path_xy = pred_fix_distance_path[:, 0, :, :2]  # [num_samples, 80, 2]
        gt_path_xy = gt_fix_distance_path[0, :, :2]  # [80, 2]
        mask_path = gt_mask_fix_distance_path[0]  # [80]
        
        errors_path = torch.norm(pred_path_xy - gt_path_xy.unsqueeze(0), dim=-1)
        mask_path_expanded = mask_path.unsqueeze(0).float()
        masked_errors_path = errors_path * mask_path_expanded
        valid_count_path = mask_path_expanded.sum().clamp(min=1)
        ade_per_sample_path = masked_errors_path.sum(dim=-1) / valid_count_path
        best_idx_path = ade_per_sample_path.argmin()
        best_pred_fix_distance_path = pred_fix_distance_path[best_idx_path:best_idx_path+1]  # [1, 1, 80, 3]
        
        return best_pred_trajs, best_pred_fix_distance_path

    def _compute_nulti_traj_cosline(self, trajs: torch.Tensor, eps: float = 1e-8):

        B, K, T, D = trajs.shape  
        trajs_xy = trajs[..., :2]                      # [B, K, T, 2]
        x = trajs_xy.reshape(B, T * 2)                 # [B, T*2]
        mu = x.mean(dim=0, keepdim=True)               # [1, T*2]

        # cosine sim: (x·mu)/(|x||mu|)
        x_n = x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))
        mu_n = mu / (mu.norm(dim=-1, keepdim=True).clamp_min(eps))
        cos = (x_n * mu_n).sum(dim=-1)                 # [B]

        return cos.mean()                                       

    def compute(self) -> Dict[str, float]:
        num_samples = len(self.pred_trajs_list)
        if num_samples == 0:
            return {}
        
        num_steps = self.pred_trajs_list[0].shape[0]
        
        # 计算各个时间点的步数
        step_3s = min(int(3.0 / self.pred_interval), num_steps)
        step_5s = min(int(5.0 / self.pred_interval), num_steps)
        step_8s = min(int(8.0 / self.pred_interval), num_steps)
        
        # 初始化结果收集器
        metrics_collectors = {
            'ade_3s': [], 'fde_3s': [],
            'ade_5s': [], 'fde_5s': [],
            'ade_8s': [], 'fde_8s': [],
            'ade_3s_path': [], 'fde_3s_path': [],
            'ade_5s_path': [], 'fde_5s_path': [],
            'ade_8s_path': [], 'fde_8s_path': [],
        }
        
        # 对每个样本计算指标
        for i in range(num_samples):
            pred_xy = self.pred_trajs_list[i]  # [80, 2]
            gt_xy = self.gt_trajs_list[i]  # [80, 2]
            gt_mask = self.gt_mask_list[i]  # [80]
            
            pred_path = self.pred_path_list[i]  # [80, 2]
            gt_path = self.gt_path_list[i]  # [80, 2]
            gt_mask_path = self.gt_mask_path_list[i]  # [80]
            
            # 计算轨迹指标
            for step, suffix in [(step_3s, '3s'), (step_5s, '5s'), (step_8s, '8s')]:
                # pred_trajs 指标
                ade = _compute_ade_simple(pred_xy[:step], gt_xy[:step], gt_mask[:step])
                fde = _compute_fde_simple(pred_xy[:step], gt_xy[:step], gt_mask[:step])
                if ade is not None:
                    metrics_collectors[f'ade_{suffix}'].append(ade)
                if fde is not None:
                    metrics_collectors[f'fde_{suffix}'].append(fde)
                
                # pred_fix_distance_path 指标
                ade_path = _compute_ade_simple(pred_path[:step], gt_path[:step], gt_mask_path[:step])
                fde_path = _compute_fde_simple(pred_path[:step], gt_path[:step], gt_mask_path[:step])
                if ade_path is not None:
                    metrics_collectors[f'ade_{suffix}_path'].append(ade_path)
                if fde_path is not None:
                    metrics_collectors[f'fde_{suffix}_path'].append(fde_path)
        
        # 计算平均值
        results = {}
        for key, values in metrics_collectors.items():
            if len(values) > 0:
                results[key] = float(np.mean(values))
            else:
                results[key] = float('nan')
        
        # 计算多模态轨迹余弦相似度指标
        if len(self.multi_traj_mean_cosline_list) > 0:
            cosline_values = [v.item() if hasattr(v, 'item') else float(v) 
                              for v in self.multi_traj_mean_cosline_list]
            results['multi_traj_mean_cosline'] = float(np.mean(cosline_values))
        else:
            results['multi_traj_mean_cosline'] = float('nan')
        
        results['num_samples'] = num_samples
        
        return results

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        super()._sync_dist(dist_sync_fn, process_group)
        self.pred_trajs_list = self._gather_ndarray_list(self.pred_trajs_list, process_group)
        self.gt_trajs_list = self._gather_ndarray_list(self.gt_trajs_list, process_group)
        self.gt_mask_list = self._gather_ndarray_list(self.gt_mask_list, process_group)
        self.pred_path_list = self._gather_ndarray_list(self.pred_path_list, process_group)
        self.gt_path_list = self._gather_ndarray_list(self.gt_path_list, process_group)
        self.gt_mask_path_list = self._gather_ndarray_list(self.gt_mask_path_list, process_group)
        self.multi_traj_mean_cosline_list = self._gather_ndarray_list(self.multi_traj_mean_cosline_list, process_group)

    def _gather_ndarray_list(self, list_of_ndarray, process_group=None):
        world_size = dist.get_world_size(group=process_group)
        list_gathered = [None] * world_size
        dist.all_gather_object(list_gathered, list_of_ndarray, group=process_group)
        for rank in range(1, world_size):
            assert (
                len(list_gathered[rank]) == len(list_gathered[0])
            ), f"Length mismatch between rank 0 and rank {rank}: {len(list_gathered[rank])} != {len(list_gathered[0])}"
        list_merged = []
        for rank in range(world_size):
            list_merged.extend(list_gathered[rank])
        return list_merged