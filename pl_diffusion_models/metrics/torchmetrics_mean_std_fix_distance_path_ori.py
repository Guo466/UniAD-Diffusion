import numpy as np
import torch.distributed as dist
from torchmetrics import Metric
from metrics.joint_motion_metrics import compute_metrics
from metrics.ego_planning_metrics import repeated_indices_numpy_collision_metrics, numpy_comfort_metrics, numpy_efficiency_metrics
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

        self.add_state('ego_gt_xy', default=[], dist_reduce_fx=None)
        self.add_state('ego_gt_xy_is_valid', default=[], dist_reduce_fx=None)
        self.add_state('path_gt_for_loss', default=[], dist_reduce_fx=None)
        self.add_state('path_gt_valid', default=[], dist_reduce_fx=None)
        self.metrics_config = METRICS_CONFIG
        self.is_waymo_dataset = is_waymo_dataset
        self.return_AP_details = False
        self.AV_OFFSET = 1

    def update(self, model_output: dict, metrics_input: dict):
        batch_size = len(metrics_input['scenario_ids'])
        for i in range(batch_size):

            self.ego_gt_xy.append(metrics_input['gt_trajectories'][i][0, :, :2])
            self.ego_gt_xy_is_valid.append(metrics_input['gt_is_valid'][i][0, ...]) # [num_groups=1, num_agents, num_gt_steps]
            self.path_gt_for_loss.append(metrics_input['path_gt_for_loss'][i][0, :, :2])
            self.path_gt_valid.append(metrics_input['path_gt_valid'][i][0, :, 0])
    def compute(self):
        metric_results = {}


        ego_gt_xy = np.array(self.ego_gt_xy)
        ego_gt_xy_is_valid = np.array(self.ego_gt_xy_is_valid)
        path_gt_for_loss = np.array(self.path_gt_for_loss)
        path_gt_valid = np.array(self.path_gt_valid)
        ego_gt_xy = ego_gt_xy * ego_gt_xy_is_valid[:, :, np.newaxis]
        ego_dx_dy_is_valid = ego_gt_xy_is_valid[:, 1:] * ego_gt_xy_is_valid[:, :-1]
        ego_dx_dy = ego_gt_xy[:, 1:, :] - ego_gt_xy[:, :-1, :] # [B, T_f, 2]
        ego_dx_dy = ego_dx_dy * ego_dx_dy_is_valid[:, :, np.newaxis]
        ego_dx_dy = np.concatenate([ego_gt_xy[:, :1, :], ego_dx_dy], axis=1)

        path_gt_for_loss = path_gt_for_loss * path_gt_valid[:, :, np.newaxis]
        path_dx_dy_is_valid = path_gt_valid[:, 1:] * path_gt_valid[:, :-1]
        path_dx_dy = path_gt_for_loss[:, 1:, :] - path_gt_for_loss[:, :-1, :] # [B, T_f, 2]
        path_dx_dy = path_dx_dy * path_dx_dy_is_valid[:, :, np.newaxis]
        path_dx_dy = np.concatenate([path_gt_for_loss[:, :1, :], path_dx_dy], axis=1)
        metric_results['dx_mean'] = np.mean(ego_dx_dy[:, :, 0])
        metric_results['dx_std'] = np.std(ego_dx_dy[:, :, 0])
        metric_results['dy_mean'] = np.mean(ego_dx_dy[:, :, 1])
        metric_results['dy_std'] = np.std(ego_dx_dy[:, :, 1])
        metric_results['dx_mean_abs'] = np.mean(np.abs(ego_dx_dy[:, :, 0]))
        metric_results['dx_std_abs'] = np.std(np.abs(ego_dx_dy[:, :, 0]))
        metric_results['dy_mean_abs'] = np.mean(np.abs(ego_dx_dy[:, :, 1]))
        metric_results['dy_std_abs'] = np.std(np.abs(ego_dx_dy[:, :, 1]))
        metric_results['dx_max'] = np.max(ego_dx_dy[:, :, 0])
        metric_results['dx_min'] = np.min(ego_dx_dy[:, :, 0])
        metric_results['dy_max'] = np.max(ego_dx_dy[:, :, 1])
        metric_results['dy_min'] = np.min(ego_dx_dy[:, :, 1])
        metric_results['dx_max_abs'] = np.max(np.abs(ego_dx_dy[:, :, 0]))
        metric_results['dx_min_abs'] = np.min(np.abs(ego_dx_dy[:, :, 0]))
        metric_results['dy_max_abs'] = np.max(np.abs(ego_dx_dy[:, :, 1]))
        metric_results['dy_min_abs'] = np.min(np.abs(ego_dx_dy[:, :, 1]))
        metric_results['path_dx_mean'] = np.mean(path_dx_dy[:, :, 0])
        metric_results['path_dx_std'] = np.std(path_dx_dy[:, :, 0])
        metric_results['path_dy_mean'] = np.mean(path_dx_dy[:, :, 1])
        metric_results['path_dy_std'] = np.std(path_dx_dy[:, :, 1])
        metric_results['path_dx_mean_abs'] = np.mean(np.abs(path_dx_dy[:, :, 0]))
        metric_results['path_dx_std_abs'] = np.std(np.abs(path_dx_dy[:, :, 0]))
        metric_results['path_dy_mean_abs'] = np.mean(np.abs(path_dx_dy[:, :, 1]))
        metric_results['path_dy_std_abs'] = np.std(np.abs(path_dx_dy[:, :, 1]))
        return metric_results

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        super()._sync_dist(dist_sync_fn, process_group)

        self.ego_gt_xy = self._gather_ndarray_list(self.ego_gt_xy, process_group)
        self.ego_gt_xy_is_valid = self._gather_ndarray_list(self.ego_gt_xy_is_valid, process_group)
        self.path_gt_for_loss = self._gather_ndarray_list(self.path_gt_for_loss, process_group)
        self.path_gt_valid = self._gather_ndarray_list(self.path_gt_valid, process_group)

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

class EgoMotionPlanningMetric_mean_std_fix_distance_path(Metric):
    def __init__(self, is_waymo_dataset, **kwargs):
        super().__init__(**kwargs)
        # States that needed to be synced across devices

        self.add_state('ego_gt_xy', default=[], dist_reduce_fx=None)
        self.add_state('ego_gt_xy_is_valid', default=[], dist_reduce_fx=None)
        self.add_state('path_gt_for_loss', default=[], dist_reduce_fx=None)
        self.add_state('path_gt_valid', default=[], dist_reduce_fx=None)
        self.metrics_config = METRICS_CONFIG
        self.is_waymo_dataset = is_waymo_dataset
        self.return_AP_details = False
        self.AV_OFFSET = 1

    def update(self, model_output: dict, metrics_input: dict):
        batch_size = len(metrics_input['scenario_ids'])
        for i in range(batch_size):

            self.ego_gt_xy.append(metrics_input['gt_trajectories'][i][0, :, :2])
            self.ego_gt_xy_is_valid.append(metrics_input['gt_is_valid'][i][0, ...]) # [num_groups=1, num_agents, num_gt_steps]
            self.path_gt_for_loss.append(metrics_input['path_gt_for_loss'][i][0, :, :2])
            self.path_gt_valid.append(metrics_input['path_gt_valid'][i][0, :, 0])
    def compute(self):
        metric_results = {}


        ego_gt_xy = np.array(self.ego_gt_xy)
        ego_gt_xy_is_valid = np.array(self.ego_gt_xy_is_valid)
        path_gt_for_loss = np.array(self.path_gt_for_loss)
        path_gt_valid = np.array(self.path_gt_valid)
        ego_gt_xy = ego_gt_xy * ego_gt_xy_is_valid[:, :, np.newaxis]
        ego_dx_dy_is_valid = ego_gt_xy_is_valid[:, 1:] * ego_gt_xy_is_valid[:, :-1]
        ego_dx_dy = ego_gt_xy[:, 1:, :] - ego_gt_xy[:, :-1, :] # [B, T_f, 2]
        ego_dx_dy = ego_dx_dy * ego_dx_dy_is_valid[:, :, np.newaxis]
        ego_dx_dy = np.concatenate([ego_gt_xy[:, :1, :], ego_dx_dy], axis=1)

        path_gt_for_loss = path_gt_for_loss * path_gt_valid[:, :, np.newaxis]
        path_dx_dy_is_valid = path_gt_valid[:, 1:] * path_gt_valid[:, :-1]
        path_dx_dy = path_gt_for_loss[:, 1:, :] - path_gt_for_loss[:, :-1, :] # [B, T_f, 2]
        path_dx_dy = path_dx_dy * path_dx_dy_is_valid[:, :, np.newaxis]
        path_dx_dy = np.concatenate([path_gt_for_loss[:, :1, :], path_dx_dy], axis=1)
        metric_results['dx_mean'] = np.mean(ego_dx_dy[:, :, 0])
        metric_results['dx_std'] = np.std(ego_dx_dy[:, :, 0])
        metric_results['dy_mean'] = np.mean(ego_dx_dy[:, :, 1])
        metric_results['dy_std'] = np.std(ego_dx_dy[:, :, 1])
        metric_results['dx_mean_abs'] = np.mean(np.abs(ego_dx_dy[:, :, 0]))
        metric_results['dx_std_abs'] = np.std(np.abs(ego_dx_dy[:, :, 0]))
        metric_results['dy_mean_abs'] = np.mean(np.abs(ego_dx_dy[:, :, 1]))
        metric_results['dy_std_abs'] = np.std(np.abs(ego_dx_dy[:, :, 1]))
        metric_results['dx_max'] = np.max(ego_dx_dy[:, :, 0])
        metric_results['dx_min'] = np.min(ego_dx_dy[:, :, 0])
        metric_results['dy_max'] = np.max(ego_dx_dy[:, :, 1])
        metric_results['dy_min'] = np.min(ego_dx_dy[:, :, 1])
        metric_results['dx_max_abs'] = np.max(np.abs(ego_dx_dy[:, :, 0]))
        metric_results['dx_min_abs'] = np.min(np.abs(ego_dx_dy[:, :, 0]))
        metric_results['dy_max_abs'] = np.max(np.abs(ego_dx_dy[:, :, 1]))
        metric_results['dy_min_abs'] = np.min(np.abs(ego_dx_dy[:, :, 1]))
        metric_results['path_dx_mean'] = np.mean(path_dx_dy[:, :, 0])
        metric_results['path_dx_std'] = np.std(path_dx_dy[:, :, 0])
        metric_results['path_dy_mean'] = np.mean(path_dx_dy[:, :, 1])
        metric_results['path_dy_std'] = np.std(path_dx_dy[:, :, 1])
        metric_results['path_dx_mean_abs'] = np.mean(np.abs(path_dx_dy[:, :, 0]))
        metric_results['path_dx_std_abs'] = np.std(np.abs(path_dx_dy[:, :, 0]))
        metric_results['path_dy_mean_abs'] = np.mean(np.abs(path_dx_dy[:, :, 1]))
        metric_results['path_dy_std_abs'] = np.std(np.abs(path_dx_dy[:, :, 1]))
        return metric_results

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        super()._sync_dist(dist_sync_fn, process_group)

        self.ego_gt_xy = self._gather_ndarray_list(self.ego_gt_xy, process_group)
        self.ego_gt_xy_is_valid = self._gather_ndarray_list(self.ego_gt_xy_is_valid, process_group)
        self.path_gt_for_loss = self._gather_ndarray_list(self.path_gt_for_loss, process_group)
        self.path_gt_valid = self._gather_ndarray_list(self.path_gt_valid, process_group)

    def _gather_ndarray_list(self, list_of_ndarray, process_group=None):
        world_size = dist.get_world_size(group=process_group)
        list_gathered = [None] * world_size
        dist.all_gather(list_gathered, list_of_ndarray, group=process_group)
        for rank in range(1, world_size):
            assert (
                len(list_gathered[rank]) == len(list_gathered[0])
            ), f"Length mismatch between rank 0 and rank {rank}: {len(list_gathered[rank])} != {len(list_gathered[0])}"
        list_merged = []
        for rank in range(world_size):
            list_merged.extend(list_gathered[rank])
        return list_merged