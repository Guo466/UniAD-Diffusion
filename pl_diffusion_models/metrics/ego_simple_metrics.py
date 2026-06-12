
import numpy as np
import torch
import math
from typing import Dict, Union, List, Tuple, Optional


def compute_ego_simple_metrics(
    pred_trajectory_dict: Dict[str, torch.Tensor],
    gt_trajectory: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_fix_distance_path: torch.Tensor,
    gt_mask_fix_distance_path: torch.Tensor,
    pred_interval: float = 0.1,
    return_details: bool = False,
) -> Dict[str, float]:
    # 从字典中提取预测轨迹
    pred_trajs = pred_trajectory_dict['pred_trajs']  # [B, 1, 80, 3]
    pred_fix_distance_path = pred_trajectory_dict['pred_fix_distace_path']  # [B, 1, 80, 3]

    # 转换为numpy数组
    if isinstance(pred_trajs, torch.Tensor):
        pred_trajs = pred_trajs.detach().cpu().numpy()
    if isinstance(pred_fix_distance_path, torch.Tensor):
        pred_fix_distance_path = pred_fix_distance_path.detach().cpu().numpy()
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(gt_fix_distance_path, torch.Tensor):
        gt_fix_distance_path = gt_fix_distance_path.detach().cpu().numpy()
    if isinstance(gt_mask_fix_distance_path, torch.Tensor):
        gt_mask_fix_distance_path = gt_mask_fix_distance_path.detach().cpu().numpy()
    
    # 处理预测轨迹维度: [B, 1, 80, 3] -> [B, 80, 2]
    # 去掉模态维度(squeeze dim=1)，只取x,y坐标(前2列)
    pred_trajs = pred_trajs[:, 0, :, :2]  # [B, 80, 2]
    pred_fix_distance_path = pred_fix_distance_path[:, 0, :, :2]  # [B, 80, 2]
    
    # 处理GT轨迹维度: [B, 80, 5] -> [B, 80, 2]，只取x,y坐标
    gt_xy = gt_trajectory[:, :, :2]  # [B, 80, 2]
    gt_fix_distance_xy = gt_fix_distance_path[:, :, :2]  # [B, 80, 2]
    
    batch_size = pred_trajs.shape[0]
    num_steps = pred_trajs.shape[1]
    
    # 计算各个时间点的步数
    step_3s = min(int(3.0 / pred_interval), num_steps)
    step_5s = min(int(5.0 / pred_interval), num_steps)
    step_8s = min(int(8.0 / pred_interval), num_steps)
    
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
    for b in range(batch_size):
        pred_xy_b = pred_trajs[b]  # [80, 2]
        gt_xy_b = gt_xy[b]  # [80, 2]
        gt_mask_b = gt_mask[b]  # [80]
        
        pred_path_b = pred_fix_distance_path[b]  # [80, 2]
        gt_path_b = gt_fix_distance_xy[b]  # [80, 2]
        gt_mask_path_b = gt_mask_fix_distance_path[b]  # [80]
        
        # 计算轨迹指标
        for step, suffix in [(step_3s, '3s'), (step_5s, '5s'), (step_8s, '8s')]:
            # pred_trajs 指标
            ade = _compute_ade_simple(pred_xy_b[:step], gt_xy_b[:step], gt_mask_b[:step])
            fde = _compute_fde_simple(pred_xy_b[:step], gt_xy_b[:step], gt_mask_b[:step])
            if ade is not None:
                metrics_collectors[f'ade_{suffix}'].append(ade)
            if fde is not None:
                metrics_collectors[f'fde_{suffix}'].append(fde)
            
            # pred_fix_distance_path 指标
            ade_path = _compute_ade_simple(pred_path_b[:step], gt_path_b[:step], gt_mask_path_b[:step])
            fde_path = _compute_fde_simple(pred_path_b[:step], gt_path_b[:step], gt_mask_path_b[:step])
            if ade_path is not None:
                metrics_collectors[f'ade_{suffix}_path'].append(ade_path)
            if fde_path is not None:
                metrics_collectors[f'fde_{suffix}_path'].append(fde_path)
    
    # 计算批量平均值
    results = {}
    for key, values in metrics_collectors.items():
        if len(values) > 0:
            results[key] = float(np.mean(values))
        else:
            results[key] = float('nan')
    
    if return_details:
        results['details'] = {
            'step_3s': step_3s,
            'step_5s': step_5s,
            'step_8s': step_8s,
            'batch_size': batch_size,
            'num_steps': num_steps,
        }
    
    return results


def _compute_ade_simple(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    gt_mask: np.ndarray,
) -> Optional[float]:

    if pred_xy.shape[0] != gt_xy.shape[0] or pred_xy.shape[0] != gt_mask.shape[0]:
        min_len = min(pred_xy.shape[0], gt_xy.shape[0], gt_mask.shape[0])
        pred_xy = pred_xy[:min_len]
        gt_xy = gt_xy[:min_len]
        gt_mask = gt_mask[:min_len]
    
    # 只计算有效时间步的误差
    valid_mask = gt_mask.astype(bool)
    if not valid_mask.any():
        return None
    
    # 计算每个时间步的欧氏距离
    errors = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=1))
    
    # 只对有效时间步求平均
    valid_errors = errors[valid_mask]
    if len(valid_errors) == 0:
        return None
    
    return float(np.mean(valid_errors))


def _compute_fde_simple(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    gt_mask: np.ndarray,
) -> Optional[float]:

    if pred_xy.shape[0] != gt_xy.shape[0] or pred_xy.shape[0] != gt_mask.shape[0]:
        min_len = min(pred_xy.shape[0], gt_xy.shape[0], gt_mask.shape[0])
        pred_xy = pred_xy[:min_len]
        gt_xy = gt_xy[:min_len]
        gt_mask = gt_mask[:min_len]
    
    # 找到最后一个有效的时间步
    valid_indices = np.where(gt_mask.astype(bool))[0]
    if len(valid_indices) == 0:
        return None
    
    last_valid_idx = valid_indices[-1]
    
    # 计算最后一个有效时间步的误差
    error = np.sqrt(np.sum((pred_xy[last_valid_idx] - gt_xy[last_valid_idx]) ** 2))
    
    return float(error)


def accumulate_ego_metrics(
    batch_results: List[Dict[str, float]]
) -> Dict[str, float]:

    if len(batch_results) == 0:
        return {}
    
    # 收集所有指标
    all_metrics = {}
    for result in batch_results:
        for key, value in result.items():
            if key != 'details':
                if key not in all_metrics:
                    all_metrics[key] = []
                if not np.isnan(value):
                    all_metrics[key].append(value)
    
    # 计算平均值
    avg_metrics = {}
    for key, values in all_metrics.items():
        if len(values) > 0:
            avg_metrics[key] = float(np.mean(values))
        else:
            avg_metrics[key] = float('nan')
    
    return avg_metrics


def compute_ego_simple_metrics_from_dataset_output(
    ego_future_dict: Dict,
    pred_trajectory_dict: Dict[str, torch.Tensor],
    pred_interval: float = 0.1
) -> Dict[str, float]:

    gt_trajectory = ego_future_dict['ego_future_status']
    gt_mask = ego_future_dict['ego_future_mask']
    gt_fix_distance_path = ego_future_dict['ego_future_status_fixed']
    gt_mask_fix_distance_path = ego_future_dict['ego_future_mask_fixed']
    
    return compute_ego_simple_metrics(
        pred_trajectory_dict=pred_trajectory_dict,
        gt_trajectory=gt_trajectory,
        gt_mask=gt_mask,
        gt_fix_distance_path=gt_fix_distance_path,
        gt_mask_fix_distance_path=gt_mask_fix_distance_path,
        pred_interval=pred_interval,
    )

