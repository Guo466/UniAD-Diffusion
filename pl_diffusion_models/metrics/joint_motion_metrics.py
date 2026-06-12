import math
from typing import Dict,Union,List

import numpy as np
import torch
import torch.distributed as dist
from functools import cmp_to_key,partial

class Point:
    def __init__(self, *args):
        if len(args)==2: ## args: x_array/x_value, y_array/y_value
            self.x = args[0]
            self.y = args[1]
        elif len(args)==1:  ## args: points array
            self.x = args[0][:,0]
            self.y = args[0][:,1]
    def __add__(self, other):
        new_x = self.x + other.x
        new_y = self.y + other.y
        return Point(new_x, new_y)
    def __sub__(self, other):
        new_x = self.x - other.x
        new_y = self.y - other.y
        return Point(new_x, new_y)
    def __mul__(self, ratio):
        new_x = self.x * ratio
        new_y = self.y * ratio
        return Point(new_x, new_y)
    # external function define
    def rotate(self, theta):
        'rotate theta rad counterclockwise'
        rx = self.x * np.cos(theta) + self.y * np.sin(theta)
        ry = self.y * np.cos(theta) - self.x * np.sin(theta)
        return Point(rx, ry)
    def tolist(self):
        'return [x,y] in list'
        return [self.x, self.y]
    def toarray(self):
        'return N,2 np.array'
        return np.concatenate([np.array(self.x)[:,None],np.array(self.y)[:,None]],-1)
    def norm(self):
        'get Euclidean norm'
        return np.sqrt(self.x ** 2 + self.y ** 2)

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05
# define priority
trajectory_type_priority = {
    'STATIONARY': 0,
    'STRAIGHT': 1,
    'STRAIGHT_RIGHT': 2,
    'STRAIGHT_LEFT': 3,
    'RIGHT_TURN': 4,
    'LEFT_TURN': 5,
    'LEFT_U_TURN': 6,
    'RIGHT_U_TURN': 7
}
def torch_sum_all_reduce(M:Union[np.array,List],device):
    M_ = torch.sum(torch.tensor(M).to(device))
    dist.all_reduce(M_,op=dist.ReduceOp.SUM)
    return M_.item()
def torch_all_reduce(M:Union[np.array,List],device):
    M_ = torch.tensor(M).to(device)
    dist.all_reduce(M_,op=dist.ReduceOp.SUM)
    return M_.item()

def torch_sum_all_gather(M:Union[np.array,List],device):
    if isinstance(M,float) or isinstance(M,int):
        M = [M]
    M_ = torch.tensor(M,device=device)
    output_tensors = [M_.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, M_)
    concat = torch.cat(output_tensors, dim=0)
    return torch.sum(concat).item()

def torch_all_gather(M:Union[np.array,List],device):
    if isinstance(M,float) or isinstance(M,int):
        M = [M]
    M_ = torch.tensor(M,device=device)
    output_tensors = [M_.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, M_)
    concat = torch.cat(output_tensors, dim=0)
    return torch.sum(concat).item() 


def PredictionToTrackStep(metrics_config, prediction_step):
    ratio = int(metrics_config['track_steps_per_second'] / metrics_config['prediction_steps_per_second'])
    return (prediction_step + 1) * ratio + metrics_config['track_history_samples'] - 1

def compute_metrics(
    gt_trajectories,
    gt_is_valid,
    forecasted_trajectories,
    forecasted_confidences,
    object_types,
    object_ids,
    scenario_ids,
    metrics_config,
    is_waymo_dataset = False,
    return_AP_details = False,
) -> Dict[str, float]:
    '''
    Args:
        gt_trajectories: [num_scenarios, num_groups, num_agents, num_gt_steps, 7]
        gt_is_valid: [num_scenarios, num_groups, num_agents, num_gt_steps]
        forecasted_trajectories: [num_scenarios, num_groups, num_modes, num_agents, num_pred_steps, 2]
        forecasted_score: [num_scenarios, num_groups, num_modes]
        object_type: [num_scenarios, num_groups, num_agents]
        object_id: [num_scenarios, num_groups, num_agents]
        scenario_id: [num_scenarios]
        metrics_config: Dict(
            track_steps_per_second: 10
            prediction_steps_per_second: 2
            track_history_samples: 10
            track_future_samples: 80
            speed_lower_bound: 1.4
            speed_upper_bound: 11.0
            speed_scale_lower: 0.5
            speed_scale_upper: 1.0
            step_configs: {5: [1.0,2.0], 9: [1.8,3.6], 15: [3.0,6.0]}
            max_predictions: 6
        )
        is_waymo_dataset: Whether test on waymo dataset
        use_all_reduce_dist: Use all reduce on multi-gpus
    '''
    metric_results = {}
    min_ade, prob_min_ade, brier_min_ade = {}, {}, {}
    min_fde, prob_min_fde, brier_min_fde = {}, {}, {}
    miss_rate, prob_miss_rate = {}, {}
    average_precision, soft_average_precision = {}, {}
    mAP_pr_buckets, soft_mAP_pr_buckets = {}, {}
    min_ade1, min_fde1 = {}, {}

    # step_config
    step_configs = metrics_config['step_configs']

    # obs_types_registration = {0: 'OTHER', 1: 'VEHICLE', 2: 'CYCLIST', 3: 'PEDESTRIAN', 4: 'TRUCK_BUS'} # holo obs types
    if is_waymo_dataset:
        obs_types_registration = {0: 'UNSET', 1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'CYCLIST', 4: 'OTHER'} # waymo obs types
        object_type_priority = {
            'UNSET': 0,
            'OTHER': 1,
            'VEHICLE': 2,
            'PEDESTRIAN': 3,
            'CYCLIST': 4
        }
        target_obs_types = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'OTHER']
    else:
        obs_types_registration = {0: 'OTHER', 1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'CYCLIST', 4: 'TRUCK_BUS'}
        object_type_priority = {
            'OTHER': 0,
            'VEHICLE': 1,
            'TRUCK_BUS': 2,
            'PEDESTRIAN': 3,
            'CYCLIST': 4
        }
        target_obs_types = ['VEHICLE', 'TRUCK_BUS', 'PEDESTRIAN', 'CYCLIST']
    # Initiate metrics containers
    for predictable_obs_types in target_obs_types:
        metric_results[predictable_obs_types] = {}
        min_ade[predictable_obs_types] = {}
        min_fde[predictable_obs_types] = {}
        prob_min_ade[predictable_obs_types] = {}
        prob_min_fde[predictable_obs_types] = {}
        brier_min_ade[predictable_obs_types] = {}
        brier_min_fde[predictable_obs_types] = {}
        miss_rate[predictable_obs_types] = {}
        prob_miss_rate[predictable_obs_types] = {}
        min_ade1[predictable_obs_types] = {}
        min_fde1[predictable_obs_types] = {}
        mAP_pr_buckets[predictable_obs_types] = {}
        soft_mAP_pr_buckets[predictable_obs_types] = {}
        average_precision[predictable_obs_types] = {}
        soft_average_precision[predictable_obs_types] = {}
        for last_time_step in step_configs.keys():
            metric_results[predictable_obs_types][last_time_step] = {}
            min_ade[predictable_obs_types].update({last_time_step: []})
            min_fde[predictable_obs_types].update({last_time_step: []})
            prob_min_ade[predictable_obs_types].update({last_time_step: []})
            prob_min_fde[predictable_obs_types].update({last_time_step: []})
            brier_min_ade[predictable_obs_types].update({last_time_step: []})
            brier_min_fde[predictable_obs_types].update({last_time_step: []})
            miss_rate[predictable_obs_types].update({last_time_step: []})
            prob_miss_rate[predictable_obs_types].update({last_time_step: []})
            min_ade1[predictable_obs_types].update({last_time_step: []})
            min_fde1[predictable_obs_types].update({last_time_step: []})
            mAP_pr_buckets[predictable_obs_types][last_time_step] = {
                "STATIONARY":{"samples":[], "num_GT": 0}, 
                "STRAIGHT":{"samples":[], "num_GT": 0}, 
                "STRAIGHT_RIGHT":{"samples":[], "num_GT": 0}, 
                "STRAIGHT_LEFT":{"samples":[], "num_GT": 0}, 
                "RIGHT_TURN":{"samples":[], "num_GT": 0}, 
                "LEFT_TURN":{"samples":[], "num_GT": 0}, 
                "LEFT_U_TURN":{"samples":[], "num_GT": 0}, 
                "RIGHT_U_TURN":{"samples":[], "num_GT": 0}
            }
            soft_mAP_pr_buckets[predictable_obs_types][last_time_step] = {
                "STATIONARY":{"samples":[], "num_GT": 0}, 
                "STRAIGHT":{"samples":[], "num_GT": 0}, 
                "STRAIGHT_RIGHT":{"samples":[], "num_GT": 0}, 
                "STRAIGHT_LEFT":{"samples":[], "num_GT": 0}, 
                "RIGHT_TURN":{"samples":[], "num_GT": 0}, 
                "LEFT_TURN":{"samples":[], "num_GT": 0}, 
                "LEFT_U_TURN":{"samples":[], "num_GT": 0}, 
                "RIGHT_U_TURN":{"samples":[], "num_GT": 0}
            }
            average_precision[predictable_obs_types][last_time_step] = {
                "STATIONARY": 0, 
                "STRAIGHT": 0, 
                "STRAIGHT_RIGHT": 0, 
                "STRAIGHT_LEFT": 0, 
                "RIGHT_TURN": 0, 
                "LEFT_TURN": 0, 
                "LEFT_U_TURN": 0, 
                "RIGHT_U_TURN": 0
            }
            soft_average_precision[predictable_obs_types][last_time_step] = {
                "STATIONARY": 0, 
                "STRAIGHT": 0, 
                "STRAIGHT_RIGHT": 0, 
                "STRAIGHT_LEFT": 0, 
                "RIGHT_TURN": 0, 
                "LEFT_TURN": 0, 
                "LEFT_U_TURN": 0, 
                "RIGHT_U_TURN": 0
            }

    # iterate through all scenarios
    from tqdm import tqdm
    for scenario_idx, scenario_id in tqdm(enumerate(scenario_ids),total=len(scenario_ids),desc="Motion metrics: "):
        multi_model_predictions = forecasted_trajectories[scenario_idx] # [num_groups, num_modes, num_agents, num_pred_stesp, 2]
        # iterate through interactive groups for joint predictions or all agents for marginal predictions
        for group_idx, one_group_multi_model_prediction in enumerate(multi_model_predictions):
            # multi_model_prediction: [num_modes, num_agents, num_pred_steps, 2]
            # determine obs type for joint predictions or marginal predictions
            current_pred_obs_type = obs_types_registration[0]
            each_type_obs_idx = {
                'VEHICLE': [],
                'PEDESTRIAN': [],
                'CYCLIST': [],
                'OTHER': []
            }
            for obs_idx, obs_type_enum in enumerate(object_types[scenario_idx][group_idx]):
                if object_type_priority[obs_types_registration[obs_type_enum]] > object_type_priority[current_pred_obs_type]:
                    current_pred_obs_type = obs_types_registration[obs_type_enum]
                each_type_obs_idx[obs_types_registration[obs_type_enum]].append(obs_idx)
            if current_pred_obs_type == obs_types_registration[0]:
                print("Caution! All obs_types are {} at scenario {}, group {}. Skipped!".format(obs_types_registration[0], scenario_id, group_idx))
                continue
            for obs_type_str in each_type_obs_idx.keys():
                if len(each_type_obs_idx[obs_type_str]) == 0:
                    continue
                current_pred_obs_type = obs_type_str
                multi_model_prediction = one_group_multi_model_prediction[:, each_type_obs_idx[obs_type_str]] # [num_modes, num_agents_per_type, num_pred_steps, 2]
                # get gt trajectories
                object_gt, object_gt_valid = gt_trajectories[scenario_idx][group_idx][each_type_obs_idx[obs_type_str]], gt_is_valid[scenario_idx][group_idx][each_type_obs_idx[obs_type_str]] # [num_agents, num_gt_steps, 7]
                object_gt_traj = object_gt[:, :, :2] # [num_agents, num_gt_steps, 2]
                # interate through step config last-time steps
                for last_time_step in step_configs.keys():
                    curr_min_ade = float("inf")
                    curr_min_fde = float("inf")
                    min_ade_idx = 0
                    min_fde_idx = 0
                    max_num_traj = min(metrics_config['max_predictions'], len(multi_model_prediction))

                    # sort predictions by confidences
                    sorted_idx = np.argsort([-x for x in forecasted_confidences[scenario_idx][group_idx]], kind = "stable")
                    top_k_confidences = [forecasted_confidences[scenario_idx][group_idx][t] for t in sorted_idx[:max_num_traj]]
                    confidence_sum = sum(top_k_confidences)
                    normalized_k_confidences = [c / confidence_sum for c in top_k_confidences]
                    top_k_multi_model_prediction = [multi_model_prediction[t] for t in sorted_idx[:max_num_traj]] # [num_modes, num_agents, num_pred_steps, 2]

                    


                    # ****Compute minADE1, minFDE1****
                    curr_min_ade1 = AverageDisplacement(metrics_config, top_k_multi_model_prediction[0], (object_gt_traj, object_gt_valid), last_time_step, is_fde = False)
                    curr_min_fde1 = AverageDisplacement(metrics_config, top_k_multi_model_prediction[0], (object_gt_traj, object_gt_valid), last_time_step, is_fde = True)
                    if curr_min_ade1 is not None:
                        min_ade1[current_pred_obs_type][last_time_step].append(curr_min_ade1)
                    if curr_min_fde1 is not None:
                        min_fde1[current_pred_obs_type][last_time_step].append(curr_min_fde1)

                    # ****Compute minADE, minFDE****
                    curr_min_ade, min_ade_idx = compute_min_average_displacement(
                        metrics_config, top_k_multi_model_prediction, (object_gt_traj, object_gt_valid), last_time_step, is_fde = False
                    )
                    curr_min_fde, min_fde_idx = compute_min_average_displacement(
                        metrics_config, top_k_multi_model_prediction, (object_gt_traj, object_gt_valid), last_time_step, is_fde = True
                    )
                    if curr_min_ade is not None:
                        min_ade[current_pred_obs_type][last_time_step].append(curr_min_ade)
                        prob_min_ade[current_pred_obs_type][last_time_step].append(float(
                                min(
                                    -np.log(normalized_k_confidences[min_ade_idx]),
                                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                                )
                                + curr_min_ade))
                        brier_min_ade[current_pred_obs_type][last_time_step].append(float((1-normalized_k_confidences[min_ade_idx])**2 + curr_min_ade))
                    if curr_min_fde is not None:
                        min_fde[current_pred_obs_type][last_time_step].append(curr_min_fde)
                        prob_min_fde[current_pred_obs_type][last_time_step].append(float(
                            min(
                                -np.log(normalized_k_confidences[min_fde_idx]),
                                -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                            )
                            + curr_min_fde))
                        brier_min_fde[current_pred_obs_type][last_time_step].append(float((1-normalized_k_confidences[min_fde_idx])**2 + curr_min_fde))

                    # TODO: ****Compute Overlap Rate
                    
                    # ****Compute Miss Rate****
                    curr_miss_rate = compute_miss_rate(metrics_config, top_k_multi_model_prediction, (object_gt, object_gt_valid), step_configs, last_time_step)
                    if curr_miss_rate is not None:
                        miss_rate[current_pred_obs_type][last_time_step].append(curr_miss_rate)
                        prob_miss_rate[current_pred_obs_type][last_time_step].append(float(
                            1.0 if curr_miss_rate else (1.0 - normalized_k_confidences[min_fde_idx])))
                    # ****Compute mAP and soft mAP
                    bucket, trajectory_type = compute_mean_average_precision(
                        metrics_config, (object_gt, object_gt_valid), top_k_multi_model_prediction, step_configs, last_time_step, top_k_confidences, use_softmAP = False)
                    if bucket is not None and trajectory_type is not None:
                        mAP_pr_buckets[current_pred_obs_type][last_time_step][trajectory_type]["samples"].extend(bucket["samples"])
                        mAP_pr_buckets[current_pred_obs_type][last_time_step][trajectory_type]["num_GT"] += bucket["num_GT"]

                    soft_bucket, trajectory_type = compute_mean_average_precision(
                        metrics_config, (object_gt, object_gt_valid), top_k_multi_model_prediction, step_configs, last_time_step, top_k_confidences, use_softmAP = True)
                    if soft_bucket is not None and trajectory_type is not None:
                        soft_mAP_pr_buckets[current_pred_obs_type][last_time_step][trajectory_type]["samples"].extend(soft_bucket["samples"])
                        soft_mAP_pr_buckets[current_pred_obs_type][last_time_step][trajectory_type]["num_GT"] += soft_bucket["num_GT"]

    # Compute metrics results from metrics statistics
    for obs_type in target_obs_types:
        for last_time_step in step_configs.keys():
            metric_results[obs_type][last_time_step]['minADE1'] = sum(min_ade1[obs_type][last_time_step]) / max(1,len(min_ade1[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]['minFDE1'] = sum(min_fde1[obs_type][last_time_step]) / max(1,len(min_fde1[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["minADE"] = sum(min_ade[obs_type][last_time_step]) / max(1,len(min_ade[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["minFDE"] = sum(min_fde[obs_type][last_time_step]) / max(1,len(min_fde[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["p-minADE"] = sum(prob_min_ade[obs_type][last_time_step]) / max(1,len(prob_min_ade[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["p-minFDE"] = sum(prob_min_fde[obs_type][last_time_step]) / max(1,len(prob_min_fde[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["brier-minADE"] = sum(brier_min_ade[obs_type][last_time_step]) / max(1,len(brier_min_ade[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["brier-minFDE"] = sum(brier_min_fde[obs_type][last_time_step]) / max(1,len(brier_min_fde[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["miss_rate"] = sum(miss_rate[obs_type][last_time_step]) / max(1,len(miss_rate[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["p-miss_rate"] = sum(prob_miss_rate[obs_type][last_time_step]) / max(1,len(prob_miss_rate[obs_type][last_time_step]))
            metric_results[obs_type][last_time_step]["mAP"] = compute_mAP_metric(
               mAP_pr_buckets[obs_type][last_time_step], average_precision[obs_type][last_time_step])
            metric_results[obs_type][last_time_step]["soft_mAP"] = compute_mAP_metric(
               soft_mAP_pr_buckets[obs_type][last_time_step], soft_average_precision[obs_type][last_time_step])
        for last_time_step in mAP_pr_buckets[obs_type].keys():
            for trajectory_type, values in mAP_pr_buckets[obs_type][last_time_step].items():
                del values['samples']
        for last_time_step in soft_mAP_pr_buckets[obs_type].keys():
            for trajectory_type, values in soft_mAP_pr_buckets[obs_type][last_time_step].items():
                del values['samples']
    if return_AP_details:
        AP_details = {"AP": average_precision, "soft_AP": soft_average_precision, 
                  "mAP_buckets": mAP_pr_buckets, "soft_mAP_buckets": soft_mAP_pr_buckets}
        metric_results.update(AP_details)
    return metric_results

def PredictionTrajectoryToTrackStep(gt_trajectory, prediction_trajectory):
    align_prediction_trajectory = prediction_trajectory.copy()
    # get gt and prediction length
    gt_len = gt_trajectory.shape[0]
    pred_len = prediction_trajectory.shape[0]
    
    # Generate interpolation time points
    t_pred = np.linspace(0, 1, pred_len)
    t_gt = np.linspace(0, 1, gt_len)
    
    # Interpolate x and y coordinates separately
    # from scipy.interpolate import interp1d
    # x_interp = interp1d(t_pred, prediction_trajectory[:,0])(t_gt)
    # y_interp = interp1d(t_pred, prediction_trajectory[:,1])(t_gt)
    x_interp = np.interp(t_gt, t_pred, prediction_trajectory[:,0])
    y_interp = np.interp(t_gt, t_pred, prediction_trajectory[:,1])
    
    # Combine interpolated trajectory
    align_prediction_trajectory = np.stack([x_interp, y_interp], axis=1)
    return align_prediction_trajectory

def compute_min_average_displacement(metrics_config, top_k_multi_model_prediction, object_gt_info, last_time_step, is_fde = False):
    min_ade = float("inf")
    min_idx = 0
    for i, joint_prediction in enumerate(top_k_multi_model_prediction):
        ade = AverageDisplacement(metrics_config, joint_prediction, object_gt_info, last_time_step, is_fde)
        if ade is None:
            return None, None
        if ade < min_ade: 
            min_ade = ade
            min_idx = i
    return min_ade, min_idx

def AverageDisplacement(metrics_config, joint_prediction, object_gt_info, last_time_step, is_fde):
    total_joint_ade = 0
    ade_count = 0
    object_gt, object_gt_valid = object_gt_info
    for i, each_agent_traj in enumerate(joint_prediction):
        if is_fde:
            ade = compute_fde(metrics_config, each_agent_traj, (object_gt[i], object_gt_valid[i]), last_time_step)
        else:
            ade = compute_ade(metrics_config, each_agent_traj, (object_gt[i], object_gt_valid[i]), last_time_step)
        if ade is None:
            continue
        total_joint_ade += ade
        ade_count += 1
    # if all agent trajectories in the joint prediction were successfully evaluated,
    # return the mean ADE value
    # if ade_count != len(joint_prediction):
    if ade_count == 0:
        result = None
    else:
        result = total_joint_ade / ade_count
    return result

def compute_fde(metrics_config, forecasted_trajectory, object_gt_info, last_time_step):
    '''
    forecasted_trajectory: [num_pred_steps, 2]
    object_gt_info: (object_gt: [num_gt_steps,2], object_gt_valid: [num_gt_steps])
    '''
    object_gt, object_gt_valid = object_gt_info
    gt_step_idx = PredictionToTrackStep(metrics_config, last_time_step)
    object_gt_xy = object_gt[gt_step_idx]
    if object_gt_valid[gt_step_idx] == False:
        return None
    dx = forecasted_trajectory[last_time_step][0] - object_gt_xy[0]
    dy = forecasted_trajectory[last_time_step][1] - object_gt_xy[1]
    return math.hypot(dx, dy)

def compute_ade(metrics_config, forecasted_trajectory, object_gt_info, last_time_step):
    '''
    forecasted_trajectory: [num_pred_steps, 2]
    object_gt_info: (object_gt: [num_gt_steps,2], object_gt_valid: [num_gt_steps])
    '''
    object_gt, object_gt_valid = object_gt_info
    error = 0.0
    count = 0
    for i in range(last_time_step + 1):
        gt_step_idx = PredictionToTrackStep(metrics_config, i)
        object_gt_xy = object_gt[gt_step_idx]
        if object_gt_valid[gt_step_idx]:
            dx = forecasted_trajectory[i][0] - object_gt_xy[0]
            dy = forecasted_trajectory[i][1] - object_gt_xy[1]
            error += math.hypot(dx, dy)
            count += 1
    if count == 0:
        return None
    return error / count

def compute_miss_rate(metrics_config, top_k_multi_model_prediction, object_gt_info, step_configs, last_time_step):
    '''
    object_gt_info: (object_gt: [num_agents, num_gt_steps,7], object_gt_valid: [num_agents, num_gt_steps])
    '''
    has_valid_measurements = False
    for joint_prediction in top_k_multi_model_prediction:
        true_positive = False
        true_positive = IsMatch(metrics_config, joint_prediction, object_gt_info, step_configs, last_time_step)
        if true_positive is not None:
            # If a true postive is found, return 0
            if true_positive: 
                return 0.0
            has_valid_measurements = True
    if not has_valid_measurements:
        return None
    # If all are misses, return 1
    return 1.0

def IsMatch(metrics_config, joint_prediction, object_gt_info, step_configs, last_time_step):
    '''
    object_gt_info: (object_gt: [num_agents, num_gt_steps,7], object_gt_valid: [num_agents, num_gt_steps])
    '''
    xy_pos = [0, 1]
    heading_pos = 4
    # The inner-most dimensions of object_gt are [x, y, length, width, heading, velocity_x,velocity_y]
    object_gt, object_gt_valid = object_gt_info # [num_agents, num_gt_steps, 7]
    num_trajectories = len(joint_prediction) # agent_num
    displacements = [None] * num_trajectories
    measurement_count = 0
    for i in range(num_trajectories):
        trajectory = joint_prediction[i]
        object_gt_states = object_gt[i] # [num_gt_steps, 7]
        gt_step_idx = PredictionToTrackStep(metrics_config, last_time_step)
        # If one trajectory has no valid states to compute the displacement, bypass the rest of the trajectories in the joint prediction.
        if object_gt_valid[i][gt_step_idx] == False:
            continue
        object_gt_xy = object_gt_states[gt_step_idx][xy_pos] # [2,]
        object_gt_heading = object_gt_states[gt_step_idx][heading_pos] # [1,]
        
        # Compute displacement
        dx = trajectory[last_time_step][0] - object_gt_xy[0]
        dy = trajectory[last_time_step][1] - object_gt_xy[1]
        longitudinal_displacement, lateral_displacement = Point(dx, dy).rotate(object_gt_heading).tolist()

        # Compute Speed Scale Factor
        scale = SpeedScaleFactor(metrics_config, object_gt_states)
        # Transform the displacement
        lateral_displacement /= scale
        longitudinal_displacement /= scale
        displacements[i] = (longitudinal_displacement, lateral_displacement)
        measurement_count += 1
    # for waymo, joint
    # if measurement_count != num_trajectories:
    if measurement_count == 0:
        return None
    # if None in displacements:      
    #     return None
    miss = False
    for disp in displacements:
        if disp is None:
            continue
        longitudinal_disp, lateral_disp = disp
        if abs(lateral_disp) > step_configs[last_time_step][0] or \
           abs(longitudinal_disp) > step_configs[last_time_step][1]:
            miss = True
    
    return not miss

def SpeedScaleFactor(metrics_config, object_gt_states):
    v_xy_pos = [-2,-1]
    if len(object_gt_states) < metrics_config['track_history_samples']:
        raise ValueError('Internal Error: Track length is invalid')
    if metrics_config['speed_lower_bound'] >= metrics_config['speed_upper_bound']:
        raise ValueError('Speed upper bound must be greater than the speed lower bound.')
    pred_start_state = object_gt_states[metrics_config['track_history_samples']] # [7, ]
    vx = pred_start_state[v_xy_pos][0]
    vy = pred_start_state[v_xy_pos][1]
    speed = math.hypot(vx, vy)
    if speed < metrics_config['speed_lower_bound']: 
        scale = metrics_config['speed_scale_lower']
        return scale
    if speed > metrics_config['speed_upper_bound']: 
        scale = metrics_config['speed_scale_upper']
        return scale
    # Linearly interpolate in between the bounds
    fraction = (speed - metrics_config['speed_lower_bound']) / (metrics_config['speed_upper_bound'] - metrics_config['speed_lower_bound'])
    scale = metrics_config['speed_scale_lower'] + fraction * (metrics_config['speed_scale_upper'] - metrics_config['speed_scale_lower'])
    return scale
                
def compute_mAP_metric(pr_bucket, ap_buckets):
    total, count = 0, 0
    for trajectory_type, bucket in pr_bucket.items():
        if len(bucket["samples"]) != 0:
            AP = compute_AP(bucket)
            ap_buckets[trajectory_type] = AP
            total += AP
            count += 1
    # total_ = total 
    # count_ = max(count,1)
    # return total_/count_
    count = max(count,1)
    return total/count

def compute_AP(bucket):
    if len(bucket["samples"]) == 0: return 0.0
    pr_curve = compute_pr_curve(bucket["samples"], bucket["num_GT"])
    num_samples = len(pr_curve)
    highest = pr_curve[num_samples - 1]
    total_area = 0.0
    for i in reversed(range(num_samples)):
        # Find ascending local maximum of precision
        if pr_curve[i][0] > highest[0]:
            total_area += highest[0] * (highest[1] - pr_curve[i][1]) # precision * (r_n - r_n-1)
            highest = pr_curve[i]
    # Add the first area
    total_area += highest[0] * highest[1]
    return total_area

def compute_pr_curve(samples, num_GT):
    samples.sort(key = cmp_to_key(pr_curve_sort_rule), reverse = True)
    num_samples = len(samples)
    result = []
    cumul_num_true_positives = 0
    for i in range(num_samples):
        # If true positive
        if samples[i][1]: cumul_num_true_positives += 1
        positives = float(cumul_num_true_positives)
        precision = positives / (i + 1)
        recall = positives / num_GT
        result.append((precision, recall))
    return result

def pr_curve_sort_rule(sample1, sample2):
    # If sample1.confidence != sample2.confidence
    if sample1[0] != sample2[0]:
        # Sort according to confidences
        if sample1[0] > sample2[0]: return 1
        else: return -1
    # If sample1.confidence == sample2.confidence
    else:
        if sample1[1] == sample2[1]: return 0
        # If sample1 is FP, sample2 is TP, then sample1 > sample2
        if sample1[1] == False and sample2[1] == True: return 1
        # If sample1 is TP, sample2 is FP, then sample1 < sample2
        if sample1[1] == True and sample2[1] == False: return -1

def compute_mean_average_precision(
    metrics_config, object_gt_info, top_k_multi_model_prediction, step_configs, last_time_step, top_k_confidences, use_softmAP = False):
    '''
    object_gt_info: (object_gt: [num_agents, num_gt_steps,7], object_gt_valid: [num_agents, num_gt_steps])
    '''
    start_time_step = metrics_config['track_history_samples']
    objects_gt, objects_gt_valid = object_gt_info
    current_trajectory_type = 'STATIONARY'
    valid_type_found = False
    for i, trajectory in enumerate(top_k_multi_model_prediction[0]):
        object_gt_states = objects_gt[i] # [num_gt_steps, 7]
        object_gt_valid = objects_gt_valid[i] # [num_gt_steps]
        trajectory_type = ClassifyTrack((object_gt_states, object_gt_valid), start_time_step)
        if trajectory_type is not None:
            if trajectory_type_priority[trajectory_type] > trajectory_type_priority[current_trajectory_type]:
                current_trajectory_type = trajectory_type
            valid_type_found = True
    if valid_type_found == False:
        return None, None
    
    if current_trajectory_type == 'RIGHT_U_TURN': current_trajectory_type = 'RIGHT_TURN'

    bucket = {"samples": [], "num_GT": 0}
    already_found_positive = False
    measurement_taken = False
    for idx, joint_prediction in enumerate(top_k_multi_model_prediction):
        is_match = IsMatch(metrics_config, joint_prediction, object_gt_info, step_configs, last_time_step)
        if is_match is None:
            continue
        if not use_softmAP:
            is_TP = False if already_found_positive else is_match
            bucket["samples"].append((top_k_confidences[idx], is_TP))
        else:
            skip_result = is_match and already_found_positive
            if not skip_result:
                bucket["samples"].append((top_k_confidences[idx], is_match))
        if is_match: already_found_positive = True
        measurement_taken = True
    if measurement_taken == True:
        bucket["num_GT"] += 1

    return bucket, current_trajectory_type

def ClassifyTrack(object_gt_info, start_time_step):
    x_pos, y_pos = 0, 1
    heading_pos = 4
    vx_pos, vy_pos = -2, -1
    # Define parameters
    kMaxSpeedForStationary = 2.0                    # (m/s)
    kMaxDisplacementForStationary = 5.0             # (m)
    kMaxLateralDisplacementForStraight = 5.0        # (m)
    kMinLongitudinalDisplacementForUTurn = -5.0     # (m)
    kMaxAbsHeadingDiffForStraight = math.pi / 6.0   # (rad)

    object_gt_states, object_gt_valid = object_gt_info
    num_time_steps = len(object_gt_states)
    assert len(object_gt_states) == len(object_gt_valid)
    # Check gt trajectory point validity and find the last valid point
    last_valid_index = -1
    for i in range(num_time_steps-1, start_time_step, -1):
        if object_gt_valid[i] == True:
            last_valid_index = i
            break
    if last_valid_index == -1:
        return None
    # Compute the distance from first position to the last position,
    # heading difference, and x and y differences

    # Compute displacement
    start_state = object_gt_states[start_time_step]
    if object_gt_valid[start_time_step] == False:
        return None
    end_state = object_gt_states[last_valid_index]
    x_delta = end_state[x_pos] - start_state[x_pos]
    y_delta = end_state[y_pos] - start_state[y_pos]
    final_displacement = math.hypot(x_delta, y_delta)
    start_state_heading = start_state[heading_pos]
    end_state_heading = end_state[heading_pos]
    heading_diff = end_state_heading - start_state_heading
    heading_diff = math.remainder(heading_diff,2*math.pi)
    normalized_x_delta, normalized_y_delta = Point(x_delta, y_delta).rotate(start_state_heading).tolist()
    start_speed = math.hypot(start_state[vx_pos], start_state[vy_pos])
    end_speed = math.hypot(end_state[vx_pos], end_state[vy_pos])
    max_speed = max(start_speed, end_speed)

    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return "STATIONARY"
    if abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if abs(normalized_y_delta) < kMaxLateralDisplacementForStraight:
            return "STRAIGHT"
        return "STRAIGHT_RIGHT" if normalized_y_delta < 0 else "STRAIGHT_LEFT"
    if heading_diff < -kMaxAbsHeadingDiffForStraight and normalized_y_delta < 0:
        return "RIGHT_U_TURN" if normalized_x_delta < kMinLongitudinalDisplacementForUTurn else "RIGHT_TURN"
    if normalized_x_delta < kMinLongitudinalDisplacementForUTurn:
        return "LEFT_U_TURN"
    return "LEFT_TURN"