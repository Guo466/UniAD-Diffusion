"""
Diffusion Model Visualization Pipeline
重构自 Jupyter Notebook: Diffusion_Model_output_viz_v0.3.3_refactored_8_mode_with_ego_navi.ipynb
"""

import os
import sys
import pickle
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
import plotly.graph_objects as go

import os
import sys
current_dir = os.getcwd()
print(current_dir)
root_dir = current_dir  #+ '/../../pl_prediction_models_local/'
sys.path.append(root_dir)
os.chdir(root_dir)
torch.manual_seed(42)   # 添加随机种子

# 导入项目相关模块
try:
    from models import LITMODEL
    from datasets import LITDATASET
    from datasets.diffusion_dataset_ego_navi_fix_distance_path_oss import DiffusionDataset
    from utils.data_utils import load_config, get_yaw_rotation_matrix, transform_back_component_coordinates, wrap_angle
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目路径设置正确")


class DiffusionModelVisualizer:
    """扩散模型可视化器"""
    
    # 常量定义
    PAST_HORIZON = 11
    FUTURE_HORIZON = 80
    PLOT_STEP = 1
    CURRENT_TS = 11
    
    def __init__(self, config_path: str, model_checkpoint: str, device: str = "cpu"):
        """
        初始化可视化器
        
        Args:
            config_path: 数据配置文件路径
            model_checkpoint: 模型检查点路径
            device: 运行设备 (cpu/cuda/mps)
        """
        self.config_path = config_path
        self.model_checkpoint = model_checkpoint
        self.device = device
        
        # 计算派生常量
        self.PLOT_HORIZON = (self.PAST_HORIZON + self.FUTURE_HORIZON) // self.PLOT_STEP + 1 \
            if self.PLOT_STEP != 1 else self.PAST_HORIZON + self.FUTURE_HORIZON
        self.PLOT_FUTURE_START = self.PAST_HORIZON // self.PLOT_STEP + 1 \
            if self.PLOT_STEP != 1 else self.PAST_HORIZON // self.PLOT_STEP
        
        # 初始化组件
        self.model = None
        self.data_config = None
        
        # 可视化配置
        # Define static map elements color map, line style map and single/double line map
        self.static_map_color_map = {
            'unknown':          [[0], 'black'],
            'lane':             [[1,2,3], 'lightgreen'],
            'bike_lane':        [[3], 'lightpink'],
            'road_line_yellow': [[6,7,8], 'yellow'],
            'road_line_white':  [[9,10,11,12,13], 'white'],
            'boundary':         [[15,16], 'black'],
            'crosswalk':        [[18], 'lightblue'],
            'stopline':         [[14], 'red']
        }
        self.static_map_line_style_map = {
            'solid': [[7,8,11,12,14,18], 'solid'],
            'dash':  [[6,9,10,13], 'dash']
        }
        self.static_map_line_width_map = {
            'double': [[8,10,12,13], 4],
            'single': [[6,7,9,11], 2]
        }
        self.road_line_types = [6,7,8,9,10,11,12,13]
        
        # Define object type color map
        self.agent_type_color_map = {
            "vehicle": "blue", # blue
            "pedestrian": "purple", # purple
            "cyclist": "orange", # orange
            'ego': 'black', # black
            "other":'black', # black
            "unset": 'black' # black
        }
        self.agent_type_color_map_future = {
            "vehicle": "seagreen", # seagreen
            "pedestrian": "pink", # pink
            "cyclist": "saddlebrown", # saddlebrown
            'ego': 'red', # red
            "other":'black', # black
            "unset": 'black' # black
        }
        self.agent_type_map = {
            0: 'unset',
            1: 'vehicle',
            2: 'pedestrian',
            3: 'cyclist',
            4: 'other',
        }

        self.routing_color_map = {
            'unset': 'silver',
            'ego_navi': 'darkviolet',
            'ego_routing': 'pink',
            'navi': 'plum',
        }
    
    def load_model(self) -> None:
        """加载模型"""
        print(f"加载模型: {self.model_checkpoint}")
        self.model = LITMODEL.module_dict['LitMMDiTDiffusionModel'].load_from_checkpoint(
            self.model_checkpoint, _instantiator=None
        )
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        print(f"运行设备: {self.model.device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型加载完成, 参数数量: {total_params:,}")
    
    def load_data_config(self) -> None:
        """加载数据配置"""
        print(f"加载数据配置: {self.config_path}")
        self.data_config = load_config(self.config_path)
        self.data_config['overwrite_cache'] = False
        
        # 初始化数据集
        dataset = DiffusionDataset(
            data_config=self.data_config, 
            is_train=False, 
            cache_data_to_disk=True, 
            num_workers=1
        )
        print("数据配置加载完成")
    
    @staticmethod
    def tensor_to_ndarray(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
        """张量转numpy数组"""
        return tensor.detach().cpu().numpy().astype(dtype)
    
    def load_test_data(self, test_data_dir: str, start_idx: int = 150, horizon: int = 10) -> List[Any]:
        """
        加载测试数据
        
        Args:
            test_data_dir: 测试数据目录
            start_idx: 起始索引
            horizon: 时间步数
            
        Returns:
            测试数据列表
        """
        print(f"加载测试数据从: {test_data_dir}")
        
        # 排序数据文件
        sorted_test_data_file_names = sorted(
            os.listdir(test_data_dir), 
            key=lambda x: int(x.split('.')[0])
        )
        
        # 加载数据
        test_data_list = []
        for test_data_file_name in sorted_test_data_file_names:
            test_data_file_path = os.path.join(test_data_dir, test_data_file_name)
            with open(test_data_file_path, 'rb') as f:
                test_data = pickle.load(f)
            test_data_list.append(test_data)
        
        # 选择数据子集
        total_horizon = len(test_data_list) # len(test_data_list)
        if horizon == 0:
            horizon = len(test_data_list)    # 注意：这里覆盖了参数horizon！后期需要改。
        selected_data = test_data_list[start_idx:start_idx + horizon]
        
        print(f"数据加载完成: 总共 {total_horizon} 帧, 选择 {len(selected_data)} 帧 (索引 {start_idx}-{start_idx + horizon - 1})")
        return selected_data

    def process_single_frame(self, test_data: Any, frame_idx: int, 
                           print_time_debug: bool = False, print_traj_debug: bool = False) -> Dict[str, Any]:
        """
        处理单帧数据
        
        Args:
            test_data: 测试数据
            frame_idx: 帧索引
            print_time_debug: 是否打印时间调试信息
            print_traj_debug: 是否打印轨迹调试信息
            
        Returns:
            处理后的帧信息
        """
        if print_time_debug:
            start_time = time.time()
        
        # 静态地图信息提取
        # Static map
        static_map_features = test_data[0]['static_map_features']
        static_map_polyline_xy = static_map_features[..., :2]
        static_map_polyline_type = static_map_features[..., 0, -2]
        static_map_polyline_valid = static_map_features[..., -1]
        
        curr_static_map_info_dict = {
            'xy': static_map_polyline_xy,
            'type': static_map_polyline_type,
            'valid': static_map_polyline_valid
        }
        
        # 交通灯信息提取
        # Traffic light
        traffic_light_features = test_data[0]['dynamic_map_features']
        traffic_light_xy = traffic_light_features[..., :2] # x,y
        traffic_light_status = np.argmax(traffic_light_features[..., 0, -5:-1], axis=-1) # is_red, is_yellow, is_green, is_unknown
        traffic_light_color = np.array(['red', 'yellow', 'green', 'grey'])[traffic_light_status]# parse [is_red, is_yellow, is_green, is_unknown] to 'red', 'yellow', 'green', 'grey'
        traffic_light_valid = traffic_light_features[..., -1] # is_valid
        
        curr_traffic_light_info_dict = {
            'xy': traffic_light_xy,
            'color': traffic_light_color,
            'valid': traffic_light_valid
        }
        
        #  ego路由信息提取
        # Ego routing
        ego_routing_polylines = test_data[0]['ego_routing_features']
        ego_routing_polyline_xy = ego_routing_polylines[..., :2] # x,y
        ego_routing_polyline_valid = ego_routing_polylines[..., -1] # is_valid
        
        curr_ego_routing_info_dict = {
            'xy': ego_routing_polyline_xy,
            'valid': ego_routing_polyline_valid
        }
        
        # Ego navi
        ego_navi_features = test_data[0]['ego_navi_features']
        ego_navi_xy = ego_navi_features[..., :2] # x,y
        ego_navi_valid = ego_navi_features[..., -1] # is_valid

        curr_ego_navi_info_dict = {
            'xy': ego_navi_xy,
            'valid': ego_navi_valid
        }

        # Navi features
        navi_features = test_data[0]['navi_features']
        navi_xy = navi_features[..., :2] # x,y
        navi_valid = navi_features[..., -1] # is_valid

        curr_navi_info_dict = {
            'xy': navi_xy,
            'valid': navi_valid
        }

        # ego转向灯信息
        # Ego turn light and trafficlight
        # Ego turn light
        ego_turn_light_features = test_data[0]['ego_turn_light_features']
        ego_turn_light_features = np.argmax(ego_turn_light_features, axis=-1)
        curr_ego_turn_light_info = np.array(['left', 'right', 'no_turn'])[ego_turn_light_features] # parse [is_left, is_right, is_unknown] to 'left', 'right', 'no_turn'
        
        # 自车看到的交通灯信息
        # Ego traffic light
        ego_traffic_light_features = test_data[0]['ego_traffic_light_features']
        ego_traffic_light_xy = ego_traffic_light_features[..., :2]
        ego_traffic_light_status = np.argmax(ego_traffic_light_features[..., -5:-1], axis=-1)
        ego_traffic_light_color = np.array(['red', 'yellow', 'green', 'gray'])[ego_traffic_light_status] # parse [is_red, is_yellow, is_green, is_unknown] to 'red', 'yellow', 'green', 'gray'
        ego_traffic_light_valid = ego_traffic_light_features[..., -1]
        
        curr_ego_traffic_light_info_dict = {
            'xy': ego_traffic_light_xy,
            'color': ego_traffic_light_color,
            'valid': ego_traffic_light_valid
        }
        
        # 模型推理
        # Target agents past and future
        if print_time_debug:
            collate_start_time = time.time()
        
        data_tensor_dict = DiffusionDataset.collate_fn([test_data])
        
        if print_time_debug:
            collate_end_time = time.time()
            print(f"Time taken for collate: {collate_end_time - collate_start_time} seconds")
        
        model_input_dict, gt_info_dict, metrics_input = \
            data_tensor_dict['model_input'], data_tensor_dict['gt_info'], data_tensor_dict['motion_metrics_input']
        
        model_input_start_time = time.time()
        model_input = model_input_dict
        
        if print_time_debug:
            model_input_end_time = time.time()
            print(f"Time taken for model input: {model_input_end_time - model_input_start_time} seconds")

        # 基本信息提取
        # Basic info
        if print_time_debug:
            basic_info_start_time = time.time()
        valid_agents_ids = metrics_input['object_ids'][0]
        valid_agents_types = metrics_input['object_types'][0]
        valid_agents_num = len(valid_agents_ids)
        target_agents_num = gt_info_dict['target_agents_num'][0]
        valid_agents_past_features = self.tensor_to_ndarray(
            model_input_dict['valid_agents_features'][0][:valid_agents_num]
        )  # [A_valid,11,33]

        # Get best pred trajs
        if print_time_debug:
            model_start_time = time.time()
        # 模型推理
        # pred_trajs_tensor, _ = self.model.model.sample(model_input, gt_info_dict)

        def read_noise_txt(txt_path):
            if not os.path.exists(txt_path):
                return np.random.randn(8, 1, 80, 3)
            with open(txt_path, 'r') as file:
                lines = file.readlines()
            return np.array([[float(x) for x in line.split(',')] for line in lines])
        noise_txt_path = './visualization/noise_to_save.txt'
        input_noise = read_noise_txt(noise_txt_path)
        input_noise = torch.from_numpy(input_noise).to(self.device).to(torch.float32).reshape(8, 1, 80, 3)
        

        pred_trajs_tensor, pred_fix_distance_path_tensor = self.model.model.sample(model_input,input_noise,sample_steps=5)

        ego_multi_model_pred_trajs = pred_trajs_tensor[1:,:,:,:2] # [7, 1, 80, 2]

        # print("intent_mask.shape:", intent_mask.shape) # B,A,M,I,T
        # print("ego intent on first step:", intent_mask[0,0,0,0])
        if print_time_debug:
            model_end_time = time.time()
            print(f"Time taken for model: {model_end_time - model_start_time} seconds")
        # B,A,M,_ ,_= pred_trajs_tensor.shape# [B,A,M,80,5], where B is 1
        # pred_trajs_tensor = pred_trajs_tensor.reshape(B,A,M,80,-1) # [B,A,M,80,5], where B is 1

        # 处理预测结果
        pred_trajs = self.tensor_to_ndarray(pred_trajs_tensor[0][..., :2]) # [1, 80, 2]
        pred_headings = self.tensor_to_ndarray(pred_trajs_tensor[0][..., 2]) # [1, 80]
        
        # 选择最佳预测轨迹
        ## NOTE: use highest_prob or chosen mode
        #best_pred_trajs = pred_trajs[:, 2]
        best_pred_trajs = np.concatenate([pred_trajs, np.zeros([target_agents_num - 1, 80, 2])], axis=0)
        
        # 从轨迹差分计算航向角
        ## NOTE: derive headings from trajs
        #best_pred_headings = pred_headings[np.arange(pred_trajs.shape[0]), best_pred_scores_index] # [A_target, 80]
        best_pred_trajs_diff = np.diff(best_pred_trajs, axis=1) # [A_target, 79, 2]
        best_pred_headings = np.arctan2(
            best_pred_trajs_diff[..., 1] + 1e-6, 
            best_pred_trajs_diff[..., 0] + 1e-6
        )
        best_pred_headings = wrap_angle(np.append(best_pred_headings, best_pred_headings[..., [-1]], axis=-1))
        
        ego_centric_best_pred_trajs = best_pred_trajs # [A_target, 80, 2]
        ego_centric_best_pred_headings = best_pred_headings # [A_target, 80]

        fix_distance_path_best_pred_trajs = pred_fix_distance_path_tensor[0][..., :2] # [1, 80, 2]
        fix_distance_multi_model_pred_trajs = pred_fix_distance_path_tensor[1:,:,:,:2] # [7, 1, 80, 2]
        fix_distance_path_gt = test_data[0]['path_gt_for_loss'][0][...,:2] # [1, 80, 2]
        
        # 处理历史轨迹
        # Concatenate object full trajs
        valid_agents_past_trajs = valid_agents_past_features[:, :, -5:-3] # [A_valid, 11, 2]
        valid_agents_past_valid = valid_agents_past_features[:, :, -1] # [A_valid, 11]
        valid_agents_past_headings = valid_agents_past_features[:, :, -2] # [A_valid, 11]
        valid_agents_lw = valid_agents_past_features[:, :, -8:-6] # [A_valid, 11, 2]
        
        valid_agents_last_lw = np.zeros([valid_agents_num, 2], dtype=np.float32)
        for t in range(valid_agents_past_valid.shape[1]):
            curr_valid_agents_mask = valid_agents_past_valid[:, t] > 0
            valid_agents_last_lw[curr_valid_agents_mask] = valid_agents_lw[curr_valid_agents_mask, t]
        
        # 组合完整轨迹
        target_agents_full_trajs = np.concatenate([
            valid_agents_past_trajs[:target_agents_num], 
            ego_centric_best_pred_trajs
        ], axis=1)  # [A_target, 91, 2]
        
        target_agents_full_valid = np.concatenate([
            valid_agents_past_valid[:target_agents_num],
            np.concatenate([np.ones([1, 80]), np.ones([target_agents_num - 1, 80])], axis=0)
        ], axis=1)
        
        target_agents_full_headings = np.concatenate([
            valid_agents_past_headings[:target_agents_num],
            ego_centric_best_pred_headings
        ], axis=1)
        
        # 处理真实轨迹
        ## NOTE: get gt trajs
        gt_trajs = gt_info_dict['per_agent_centric_target_gt_for_loss'][..., :2] # [A_target, 80, 2]
        ## NOTE: derive headings from trajs
        gt_trajs_diff = np.diff(gt_trajs, axis=1) # [A_target, 79, 2]
        gt_headings = np.arctan2(gt_trajs_diff[..., 1], gt_trajs_diff[..., 0])
        gt_headings = wrap_angle(np.append(gt_headings, gt_headings[..., [-1]], axis=-1))
        
        if print_traj_debug:
            print(f"current ego state:", valid_agents_past_features[0,-1,:6])
            print(f"best_pred_trajs[0,:10]:", best_pred_trajs[0,:1])
            print(f"gt_trajs[0,:10]:", gt_trajs[0,:1])
            print('diff', np.linalg.norm(best_pred_trajs[0,:1]-self.tensor_to_ndarry(gt_trajs[0,:1]), axis=-1))
        
        # 转换真实轨迹到ego坐标系
        ## transform pred trajs from per-agent-centric to ego-centric
        target_current_info= metrics_input['target_current_info'][0] # [A_target, 3]
        target_current_xy = target_current_info[:, :2] # [A_target, 2]
        target_current_heading_gt = target_current_info[:, 2] # [A_target, ]
        rotation_matrix_2d_for_gt = get_yaw_rotation_matrix(target_current_heading_gt)[:,:2,:2] # [A_target, 2, 2]
        rotated_gt_trajs = np.einsum('...ij, ...kj->...ki', rotation_matrix_2d_for_gt, gt_trajs)
        ego_centric_gt_trajs = rotated_gt_trajs + target_current_xy[:, np.newaxis] # [A_target, 80, 2]
        # print("ego_centric_gt_trajs[1,:10]:", ego_centric_gt_trajs[0,:10])
        ego_centric_gt_headings = gt_headings + target_current_heading_gt[:, np.newaxis] # [A_target, 80]
        ego_centric_gt_headings = ego_centric_gt_headings[:target_agents_num]
        ego_centric_gt_trajs = ego_centric_gt_trajs[:target_agents_num]
        # print("ego_centric_gt_trajs shape:", ego_centric_gt_trajs.shape)
        # print("valid_agents_past_trajs shape:", valid_agents_past_trajs[:target_agents_num].shape)
        
        target_agents_full_gt_trajs = np.concatenate([
            valid_agents_past_trajs[:target_agents_num], 
            ego_centric_gt_trajs
        ], axis=1) # [A_target, 91, 2]
        
        target_agents_full_gt_valid = np.concatenate([
            valid_agents_past_valid[:target_agents_num],
            gt_info_dict['target_gt_valid'][:target_agents_num]
        ], axis=1)
        
        target_agents_full_gt_headings = np.concatenate([
            valid_agents_past_headings[:target_agents_num],
            ego_centric_gt_headings
        ], axis=1)
        
        target_agents_lw = valid_agents_last_lw[:target_agents_num] # [A_target, 2]
        target_agents_full_lw = target_agents_lw[:, np.newaxis].repeat(91, axis=1)
        target_agents_types = valid_agents_types[:target_agents_num] # [A_target, ]
        
        # 构建场景对象信息字典
        curr_scenario_object_info_dict = {}
        sdc_track_index = 0  # 假设第一个是自车
        
        for obj_idx, obj_id in enumerate(valid_agents_ids[:target_agents_num]):
            curr_scenario_object_info_dict[obj_id] = {}
            curr_scenario_object_info_dict[obj_id]['type'] = target_agents_types[obj_idx]
            curr_scenario_object_info_dict[obj_id]['full_trajs'] = target_agents_full_trajs[obj_idx][
                self.CURRENT_TS - self.PAST_HORIZON:self.CURRENT_TS + self.FUTURE_HORIZON:self.PLOT_STEP
            ]
            
            # 计算轨迹距离以确定是否使用固定航向
            trajs_distance = curr_scenario_object_info_dict[obj_id]['full_trajs'][-1] - \
                           curr_scenario_object_info_dict[obj_id]['full_trajs'][self.PLOT_FUTURE_START - 1]
            
            if np.hypot(trajs_distance[0], trajs_distance[1]) < 2.0:
                # 静止或移动很小，使用固定航向
                curr_scenario_object_info_dict[obj_id]['full_headings'] = np.ones(self.PLOT_HORIZON) * \
                    target_agents_full_headings[obj_idx][self.PLOT_FUTURE_START - 1]
            else:
                # 正常移动，使用计算出的航向
                curr_scenario_object_info_dict[obj_id]['full_headings'] = target_agents_full_headings[obj_idx][
                    self.CURRENT_TS - self.PAST_HORIZON:self.CURRENT_TS + self.FUTURE_HORIZON:self.PLOT_STEP
                ]
            
            curr_scenario_object_info_dict[obj_id]['full_lw'] = target_agents_full_lw[obj_idx][
                self.CURRENT_TS - self.PAST_HORIZON:self.CURRENT_TS + self.FUTURE_HORIZON:self.PLOT_STEP
            ]
            curr_scenario_object_info_dict[obj_id]['full_valid'] = target_agents_full_valid[obj_idx][
                self.CURRENT_TS - self.PAST_HORIZON:self.CURRENT_TS + self.FUTURE_HORIZON:self.PLOT_STEP
            ]
            
            # 真实轨迹信息
            # for gt
            curr_scenario_object_info_dict[obj_id]['full_gt_trajs'] = target_agents_full_gt_trajs[obj_idx][
                self.CURRENT_TS - self.PAST_HORIZON:self.CURRENT_TS + self.FUTURE_HORIZON:self.PLOT_STEP
            ]
            curr_scenario_object_info_dict[obj_id]['full_gt_valid'] = target_agents_full_gt_valid[obj_idx][
                self.CURRENT_TS - self.PAST_HORIZON:self.CURRENT_TS + self.FUTURE_HORIZON:self.PLOT_STEP
            ]
            
            gt_trajs_distance = curr_scenario_object_info_dict[obj_id]['full_gt_trajs'][-1] - \
                              curr_scenario_object_info_dict[obj_id]['full_gt_trajs'][self.PLOT_FUTURE_START - 1]
            
            if np.hypot(gt_trajs_distance[0], gt_trajs_distance[1]) < 2.0:
                curr_scenario_object_info_dict[obj_id]['full_gt_headings'] = np.ones(self.PLOT_HORIZON) * \
                    target_agents_full_gt_headings[obj_idx][self.PLOT_FUTURE_START - 1]
            else:
                curr_scenario_object_info_dict[obj_id]['full_gt_headings'] = target_agents_full_gt_headings[obj_idx][
                    self.CURRENT_TS - self.PAST_HORIZON:self.CURRENT_TS + self.FUTURE_HORIZON:self.PLOT_STEP
                ]
            if obj_id == 'ego':
                curr_scenario_object_info_dict[obj_id]['ego_multi_model_trajs'] = ego_multi_model_pred_trajs[:,0,:self.FUTURE_HORIZON:self.PLOT_STEP,:]
                curr_scenario_object_info_dict[obj_id]['ego_multi_model_trajs_input_noise'] = input_noise
                curr_scenario_object_info_dict[obj_id]['ego_fix_distance_path_gt'] = fix_distance_path_gt
                curr_scenario_object_info_dict[obj_id]['ego_best_fix_distance_path_trajs'] = fix_distance_path_best_pred_trajs[0]
                curr_scenario_object_info_dict[obj_id]['ego_multi_model_fix_distance_path_trajs'] = fix_distance_multi_model_pred_trajs[:,0]
        
        if print_time_debug:
            end_time = time.time()
            print(f"场景 {test_data[0]['scenario_id']} 处理时间: {end_time - start_time:.4f}秒")
            print(f"Time taken for scenario {test_data[0]['scenario_id']}: {end_time - start_time} seconds")
        
        # 返回所有处理后的信息
        return {
            'static_map_info': curr_static_map_info_dict,
            'traffic_light_info': curr_traffic_light_info_dict,
            'ego_routing_info': curr_ego_routing_info_dict,
            'ego_navi_features': curr_ego_navi_info_dict,
            'navi_features': curr_navi_info_dict,
            'ego_turn_light_info': curr_ego_turn_light_info,
            'ego_traffic_light_info': curr_ego_traffic_light_info_dict,
            'target_agents_info': curr_scenario_object_info_dict,
            'frame_idx': frame_idx
        }
    
    def process_static_map_info(self, static_map_info_dict: Dict) -> Dict:
        """
        处理静态地图信息
        
        Args:
            static_map_info_dict: 静态地图信息字典
            
        Returns:
            处理后的静态地图多边形字典
        """
        static_polylines_dict = {}
        
        for map_seg_id, map_seg_xy in enumerate(static_map_info_dict['xy']):
            map_seg_type_int = int(static_map_info_dict['type'][map_seg_id])
            map_seg_valid = static_map_info_dict['valid'][map_seg_id].astype(bool)
            map_seg_xy = map_seg_xy[map_seg_valid]
            
            # 根据类型分配绘图信息
            # extract unknown
            if map_seg_type_int in self.static_map_color_map['unknown'][0]:
                line_type = 'unknown'
                line_color = self.static_map_color_map[line_type][1]
                static_polylines_dict[map_seg_id] = self._assign_polyline_plot_info(map_seg_xy, map_seg_type_int, line_type, line_color)
            # extract lane
            if map_seg_type_int in self.static_map_color_map['lane'][0]:
                if map_seg_type_int in self.static_map_color_map['bike_lane'][0]:
                    line_type = 'bike_lane'
                else:
                    line_type = 'lane'
                line_color = self.static_map_color_map[line_type][1]
                static_polylines_dict[map_seg_id] = self._assign_polyline_plot_info(map_seg_xy, map_seg_type_int, line_type, line_color)
            # extract road line
            if map_seg_type_int in self.road_line_types:
                if map_seg_type_int in self.static_map_color_map['road_line_yellow'][0]:
                    line_type = 'road_line_yellow'
                else:
                    line_type = 'road_line_white'
                line_color = self.static_map_color_map[line_type][1]
                if map_seg_type_int in self.static_map_line_style_map['solid'][0]:
                    line_style = self.static_map_line_style_map['solid'][1]
                else:
                    line_style = self.static_map_line_style_map['dash'][1]
                if map_seg_type_int in self.static_map_line_width_map['double'][0]:
                    line_width = self.static_map_line_width_map['double'][1]
                else:
                    line_width = self.static_map_line_width_map['single'][1]
                static_polylines_dict[map_seg_id] = self._assign_polyline_plot_info(map_seg_xy, map_seg_type_int, line_type, 
                                                                            line_color, line_style, line_width)
            # extract road edge
            if map_seg_type_int in self.static_map_color_map['boundary'][0]:
                line_type = 'boundary'
                line_color = self.static_map_color_map[line_type][1]
                static_polylines_dict[map_seg_id] = self._assign_polyline_plot_info(map_seg_xy, map_seg_type_int, line_type, line_color)
            # extract crosswalk
            if map_seg_type_int in self.static_map_color_map['crosswalk'][0]:
                line_type = 'crosswalk'
                line_color = self.static_map_color_map[line_type][1]
                static_polylines_dict[map_seg_id] = self._assign_polyline_plot_info(map_seg_xy, map_seg_type_int, line_type, line_color)
            # extract stop line
            if map_seg_type_int in self.static_map_color_map['stopline'][0]:
                line_type = 'stopline'
                line_color = self.static_map_color_map[line_type][1]
                static_polylines_dict[map_seg_id] = self._assign_polyline_plot_info(map_seg_xy, map_seg_type_int, line_type, line_color)
        
        return static_polylines_dict
    
    def _assign_polyline_plot_info(self, map_seg_xy: np.ndarray, map_seg_type_int: int, 
                                 polyline_type: str, line_color: str, 
                                 line_style: str = 'solid', line_width: int = 2) -> Dict:
        """
        分配多边形绘图信息
        
        Args:
            map_seg_xy: 地图段坐标
            map_seg_type_int: 地图段类型整数
            polyline_type: 多边形类型
            line_color: 线条颜色
            line_style: 线条样式
            line_width: 线条宽度
            
        Returns:
            多边形信息字典
        """
        return {
            'vsd_type': str(map_seg_type_int),
            'type': polyline_type,
            'points_x': map_seg_xy[:, 0],
            'points_y': map_seg_xy[:, 1],
            'line_color': line_color,
            'line_style': line_style,
            'line_width': line_width
        }
    
    def process_target_agents_info(self, target_agents_info: Dict, max_agent_num: int) -> Dict:
        """
        处理目标智能体信息
        
        Args:
            target_agents_info: 目标智能体信息
            max_agent_num: 最大智能体数量
            
        Returns:
            处理后的智能体字典
        """
        target_agents_dict = {}
        sdc_track_index = 0  # 第一个是自车
        
        for agent_idx, agent_id in enumerate(target_agents_info.keys()):
            if agent_idx == sdc_track_index:
                agent_type = 'ego'
            else:
                agent_type = self.agent_type_map[int(target_agents_info[agent_id]['type'])]
            
            agents_trajs = target_agents_info[agent_id]['full_trajs']
            agents_headings = target_agents_info[agent_id]['full_headings']
            agents_lw = target_agents_info[agent_id]['full_lw']
            agents_valid = target_agents_info[agent_id]['full_valid']
            agents_gt_trajs = target_agents_info[agent_id]['full_gt_trajs']
            agents_gt_headings = target_agents_info[agent_id]['full_gt_headings']
            agents_gt_lw = target_agents_info[agent_id]['full_lw']
            agents_gt_valid = target_agents_info[agent_id]['full_gt_valid']
            
            # 当前位置和航向
            ## translation
            current_position_x = agents_trajs[self.CURRENT_TS - 1, 0]
            current_position_y = agents_trajs[self.CURRENT_TS - 1, 1]
            ## rotation(counterclock-wise given heading)
            current_heading = agents_headings[self.CURRENT_TS - 1]
            
            # 旋转矩阵
            #current_heading = 0
            rotation_matrix_2d = np.array([
                [np.cos(current_heading), -np.sin(current_heading)],
                [np.sin(current_heading), np.cos(current_heading)]
            ])
            
            # 边界框
            ## bounding box
            current_length = agents_lw[self.CURRENT_TS - 1, 0]
            current_width = agents_lw[self.CURRENT_TS - 1, 1]
            default_obj_polygon = np.array([
                [-current_length / 2, -current_width / 2],
                [-current_length / 2, current_width / 2],
                [current_length / 2, current_width / 2],
                [current_length / 2, -current_width / 2],
                [-current_length / 2, -current_width / 2]
            ])
            
            # 旋转边界框
            ## rotate the default bbox to current state
            rotated_obj_polygon = default_obj_polygon @ rotation_matrix_2d.T
            
            target_agents_dict[agent_id] = {
                'type': agent_type,
                'trajs': np.where(agents_valid[:, None], agents_trajs, np.nan),
                'polygon_x': rotated_obj_polygon[:, 0] + current_position_x,
                'polygon_y': rotated_obj_polygon[:, 1] + current_position_y,
                'headings': np.where(agents_valid, agents_headings, np.nan),
                'lw': np.where(agents_valid[:, None], agents_lw, np.nan),
                'valid': agents_valid,
                'gt_trajs': np.where(agents_gt_valid[:, None], agents_gt_trajs, np.nan),
                'gt_headings': np.where(agents_gt_valid, agents_gt_headings, np.nan),
                'gt_lw': np.where(agents_gt_valid[:, None], agents_gt_lw, np.nan),
                'gt_valid': agents_gt_valid,
                'future_color': self.agent_type_color_map_future[agent_type],
                'past_color': self.agent_type_color_map[agent_type],
                'gt_color': 'saddlebrown',
                'visible': True,
                'original_type': target_agents_info[agent_id]['type'],
            }
            if agent_idx == sdc_track_index:
                target_agents_dict[agent_id]['ego_multi_model_trajs'] = target_agents_info[agent_id]['ego_multi_model_trajs'] # [7, 1, 80, 2]
                target_agents_dict[agent_id]['ego_multi_model_trajs_color'] = ['orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown']
                target_agents_dict[agent_id]['ego_multi_model_input_noise'] = target_agents_info[agent_id]['ego_multi_model_trajs_input_noise']
                target_agents_dict[agent_id]['ego_best_fix_distance_path_trajs'] = target_agents_info[agent_id]['ego_best_fix_distance_path_trajs']
                target_agents_dict[agent_id]['ego_multi_model_fix_distance_path_trajs'] = target_agents_info[agent_id]['ego_multi_model_fix_distance_path_trajs']
                target_agents_dict[agent_id]['ego_fix_distance_path_gt'] = target_agents_info[agent_id]['ego_fix_distance_path_gt']
        
        # 填充到最大智能体数量
        obj_padding_num = max_agent_num - len(target_agents_info)
        for padded_agent_idx in range(obj_padding_num):
            padded_agent_type = 'unset'
            padded_agent_id = 'pad_' + str(padded_agent_idx)
            target_agents_dict[padded_agent_id] = {
                'type': padded_agent_type,
                'trajs': np.zeros((self.CURRENT_TS, 2)),
                'polygon_x': np.zeros((5,)),
                'polygon_y': np.zeros((5,)),
                'headings': np.zeros((self.CURRENT_TS,)),
                'lw': np.zeros((self.CURRENT_TS, 2)),
                'valid': np.zeros((self.CURRENT_TS,)),
                'gt_trajs': np.zeros((self.CURRENT_TS, 2)),
                'gt_headings': np.zeros((self.CURRENT_TS,)),
                'gt_lw': np.zeros((self.CURRENT_TS, 2)),
                'gt_valid': np.zeros((self.CURRENT_TS,)),
                'future_color': 'black',
                'past_color': 'black',
                'gt_color': 'black',
                'visible': True,
                'original_type': 'unset',
            }
        
        return target_agents_dict
    
    @staticmethod
    def hex_to_rgb(hexa: str) -> Tuple[int, int, int]:
        """十六进制颜色转RGB"""
        hexa = hexa.lstrip('#')
        return tuple(int(hexa[i:i + 2], 16) for i in (0, 2, 4))
    
    def create_visualization_frames(self, processed_frames: List[Dict], 
                                  max_polyline_num: int = 512,
                                  max_ego_navi_polyline_num:int = 16,
                                  max_ego_routing_polyline_num:int = 64,
                                  max_navi_polyline_num:int = 256) -> List[go.Frame]:
        """
        创建可视化帧
        
        Args:
            processed_frames: 处理后的帧数据列表
            max_polyline_num: 最大多边形数量
            max_ego_navi_ployline_num: 最大自车导航线数量
            max_ego_routing_polyline: 最大自车路由线数量
            max_navi_ployline_num: 最大普通导航线数量
            
        Returns:
            Plotly帧列表
        """
        print("创建可视化帧...")
        plot_frames = []
        
        # 透明度设置
        past_opacity = np.linspace(0.4, 0.8, self.PAST_HORIZON - 1)
        future_opacity = np.linspace(0.1, 0.8, self.FUTURE_HORIZON)[::-1]
        line_opacity = np.concatenate((past_opacity, [1.0], future_opacity))
        fill_color_opacity = np.concatenate((past_opacity / 2, [0.8], future_opacity / 2))
        
        traffic_light_default_size = 1.0
        
        for frame_idx, frame_data in enumerate(processed_frames):
            tmp_fig = go.Figure()
            
            # 绘制静态地图
            # Plot static map
            static_map = frame_data['static_map_polylines']
            for polyline_id, polyline_info in static_map.items():
                tmp_fig.add_trace(
                    go.Scatter(
                        y=polyline_info['points_x'],
                        x=polyline_info['points_y'],
                        customdata=list(range(len(polyline_info['points_x']))),
                        mode='lines',
                        line=dict(
                            color=polyline_info['line_color'],
                            dash=polyline_info['line_style'],
                            width=polyline_info['line_width']
                        ),
                        hovertemplate='id: ' + str(polyline_id) +
                                    '<br>type: ' + polyline_info['type'] +
                                    '<br>vsd_type: ' + polyline_info['vsd_type'] +
                                    '<br>idx in segment: %{customdata}<br>x:%{y}<br>y:%{x}' +
                                    '<br>segment length: ' + str(len(polyline_info['points_x'])) + '<extra></extra>'
                    )
                )
            
            # 地图填充
            map_padding_num = max_polyline_num - len(list(static_map.keys()))
            for _ in range(map_padding_num):
                tmp_fig.add_trace(
                    go.Scatter(
                        y=[0], x=[0], mode="lines", line=dict(color="silver"), visible=True
                    )
                )
            
            # Plot ego routing
            ego_routing = frame_data['routing_ployline_dict']
            for polyline_id, polyline_info in ego_routing.items():
                tmp_fig.add_trace(
                    go.Scatter(
                        y = polyline_info['points_x'],
                        x = polyline_info['points_y'],
                        customdata=list(range(len(polyline_info['points_x']))),
                        mode = 'lines',
                        line = dict(color=polyline_info['line_color'], width=polyline_info['line_width']),
                        hovertemplate='id: '+ str(polyline_id)+\
                                    '<br>type: '+polyline_info['type']+\
                                    '<br>idx in segment: %{customdata}<br>x:%{y}<br>y:%{x}'+\
                                    '<br>segment length: '+str(len(polyline_info['points_x']))+'<extra></extra>'
                    )
                )
            ego_routing_padding_num = max_ego_routing_polyline_num + max_ego_navi_polyline_num + max_navi_polyline_num - len(list(ego_routing.keys()))
            for _ in range(ego_routing_padding_num):
                tmp_fig.add_trace(
                    go.Scatter(
                        y = [0],
                        x = [0],
                        mode = "lines",
                        line = dict(color="silver"),
                    )
                )


            # # 绘制ego路由
            # ## plot ego routing
            # ego_routing_points = frame_data['ego_routing_info']
            # # (A_ego_routing, 11, 2) -> (A_ego_routing * 11, 2)
            # ego_routing_points_xy = ego_routing_points['xy'].reshape(-1, 2)
            # ego_routing_points_valid_mask = ego_routing_points['valid'].astype(bool).reshape(-1)
            # ego_routing_points_xy_valid = ego_routing_points_xy[ego_routing_points_valid_mask]
            
            # if len(ego_routing_points_xy_valid) > 0:
            #     tmp_fig.add_trace(
            #         go.Scatter(
            #             y=ego_routing_points_xy_valid[:, 0],
            #             x=ego_routing_points_xy_valid[:, 1],
            #             customdata=list(range(len(ego_routing_points_xy_valid))),
            #             mode='lines',
            #             line=dict(color='pink', width=2),
            #             hovertemplate='type: ego routing' +
            #                         '<br>idx in segment: %{customdata}<br>x:%{y}<br>y:%{x}<extra></extra>'
            #         )
            #     )
            # else:
            #     tmp_fig.add_trace(
            #         go.Scatter(y=[0], x=[0], mode="lines", line=dict(color="silver"), visible=True)
            #     )
            
            # 绘制ego交通灯
            ## Plot ego traffic light
            ego_traffic_light_info = frame_data['ego_traffic_light_info']
            ego_traffic_light_stop_point_x = ego_traffic_light_info['xy'][..., 0]
            ego_traffic_light_stop_point_y = ego_traffic_light_info['xy'][..., 1]
            ego_traffic_light_color = ego_traffic_light_info['color']
            ego_traffic_light_valid_mask = ego_traffic_light_info['valid'].astype(bool)
            
            ego_traffic_light_stop_point_x_valid = ego_traffic_light_stop_point_x[ego_traffic_light_valid_mask]
            ego_traffic_light_stop_point_y_valid = ego_traffic_light_stop_point_y[ego_traffic_light_valid_mask]
            ego_traffic_light_color_valid = ego_traffic_light_color[ego_traffic_light_valid_mask]
            default_ego_traffic_light_polygon = np.array([[ego_traffic_light_stop_point_x_valid - traffic_light_default_size, ego_traffic_light_stop_point_y_valid - traffic_light_default_size],
                                                [ego_traffic_light_stop_point_x_valid - traffic_light_default_size, ego_traffic_light_stop_point_y_valid + traffic_light_default_size],
                                                [ego_traffic_light_stop_point_x_valid + traffic_light_default_size, ego_traffic_light_stop_point_y_valid + traffic_light_default_size],
                                                [ego_traffic_light_stop_point_x_valid + traffic_light_default_size, ego_traffic_light_stop_point_y_valid - traffic_light_default_size],
                                                [ego_traffic_light_stop_point_x_valid - traffic_light_default_size, ego_traffic_light_stop_point_y_valid - traffic_light_default_size]])


            if len(ego_traffic_light_stop_point_x_valid) > 0:
                ego_traffic_light_color = ego_traffic_light_color_valid[0]
                default_ego_traffic_light_polygon = default_ego_traffic_light_polygon[...,0]
                
                tmp_fig.add_trace(
                    go.Scatter(
                        y=default_ego_traffic_light_polygon[:, 0],
                        x=default_ego_traffic_light_polygon[:, 1],
                        mode='lines',
                        fill='toself',
                        line_color=ego_traffic_light_color,
                        hovertext=' ego traffic light: ' + \
                                '<br>state: ' + ego_traffic_light_color,
                        hoverinfo='text',
                        hoveron='points'
                    )
                )
            else:
                tmp_fig.add_trace(
                    go.Scatter(
                        y=[0, 0, 0, 0, 0],
                        x=[0, 0, 0, 0, 0],
                        mode='lines',
                        fill='toself',
                        line_color='silver',
                        hovertext=' padded ego traffic light: ' + \
                                '<br>state: ' + 'invalid',
                        hoverinfo='text',
                        hoveron='points'
                    )
                )
            
            # 绘制智能体
            # Plot objects
            target_agents_dict = frame_data['target_agents_dict']
            
            for object_id, object_info in target_agents_dict.items():
                # 绘制真实轨迹（仅ego）
                # plot gt
                if object_id == 'ego':
                    tmp_fig.add_trace(
                        go.Scatter(
                            y=object_info['gt_trajs'][self.CURRENT_TS:, 0],
                            x=object_info['gt_trajs'][self.CURRENT_TS:, 1],
                            # customdata=list(range(1, len(object_info['gt_trajs'][CURRENT_TS:])+1)),
                            customdata=np.array([
                                    np.array(range(1, len(object_info['gt_trajs'][self.CURRENT_TS:])+1)),
                                    np.diff(object_info['gt_trajs'][self.CURRENT_TS:,0],1,prepend=0)/0.1,
                                    np.diff(object_info['gt_trajs'][self.CURRENT_TS:,1],1,prepend=0)/0.1,
                                    np.hypot(
                                        np.diff(object_info['gt_trajs'][self.CURRENT_TS:,0],1,prepend=0)/0.1, 
                                        np.diff(object_info['gt_trajs'][self.CURRENT_TS:,1],1,prepend=0)/0.1
                                        )]).T,
                            mode='markers+lines',
                            marker=dict(color=object_info['gt_color'], size=5),
                            line=dict(color=object_info['gt_color'], width=1),
                            hovertemplate='obs_id: ' + object_id + \
                                '<br> type: ' + object_info['type'] + \
                                '<br> timestep: %{customdata[0]}' + \
                                '<br> v_x: %{customdata[1]}' + \
                                '<br> v_y: %{customdata[2]}' + \
                                '<br> v: %{customdata[3]}' + \
                                '<br> gt_x: %{y}' + \
                                '<br> gt_y: %{x}' + \
                                '<br> original_type: ' + str(object_info['type']) + '<extra></extra>',
                            hoveron='points',
                            name="gt_" + object_id,
                            showlegend=False,
                            connectgaps=False
                        )
                    )
                    ego_fix_distance_path_gt = object_info['ego_fix_distance_path_gt']
                    tmp_fig.add_trace(
                        go.Scatter(
                            y=ego_fix_distance_path_gt[:,0], 
                            x=ego_fix_distance_path_gt[:,1], 
                            # customdata=list(range(1, len(object_info['gt_trajs'][CURRENT_TS:])+1)),
                            customdata = np.array([
                                    np.array(range(1, len(ego_fix_distance_path_gt)+1)),
                                    np.diff(ego_fix_distance_path_gt[:,0],1,prepend=0)/0.1,
                                    np.diff(ego_fix_distance_path_gt[:,1],1,prepend=0)/0.1,
                                    np.hypot(
                                        np.diff(ego_fix_distance_path_gt[:,0],1,prepend=0)/0.1, 
                                        np.diff(ego_fix_distance_path_gt[:,1],1,prepend=0)/0.1
                                        )]).T,
                            mode='markers+lines',
                            marker=dict(color=object_info['gt_color'], size=5, symbol='triangle-up'),
                            line=dict(color=object_info['gt_color'], width=1),
                            hovertemplate='obs_id: ' + object_id + \
                                '<br> type: ' + object_info['type'] + \
                                '<br> point_num: %{customdata[0]}' + \
                                '<br> gt_x: %{y}' + \
                                '<br> gt_y: %{x}' + \
                                '<br> v_x: %{customdata[1]}' + \
                                '<br> v_y: %{customdata[2]}' + \
                                '<br> v: %{customdata[3]}' + \
                                '<br> original_type: ' + str(object_info['type']) + '<extra></extra>',
                            hoveron='points',
                            name="gt_" + object_id,
                            showlegend=False,
                            connectgaps=False
                        )
                    )
                
                # 绘制未来轨迹
                # Plot future first
                tmp_fig.add_trace(
                    go.Scatter(
                        y=object_info['trajs'][self.CURRENT_TS:, 0],
                        x=object_info['trajs'][self.CURRENT_TS:, 1],
                        # customdata=list(range(1, len(object_info['trajs'][self.CURRENT_TS:]) + 1)),
                        # customdata=list(range(1, len(object_info['trajs'][CURRENT_TS:])+1)),
                        customdata = np.array([
                                    np.array(range(1, len(object_info['trajs'][self.CURRENT_TS:])+1)),
                                    np.diff(object_info['trajs'][self.CURRENT_TS:,0],1,prepend=0)/0.1,
                                    np.diff(object_info['trajs'][self.CURRENT_TS:,1],1,prepend=0)/0.1,
                                    np.hypot(np.diff(object_info['trajs'][self.CURRENT_TS:,0],1,prepend=0)/0.1, np.diff(object_info['trajs'][self.CURRENT_TS:,1],1,prepend=0)/0.1)]).T,
                        mode='markers+lines',
                        marker=dict(color=object_info['future_color'], size=5),
                        line=dict(color=object_info['future_color'], width=1),
                        hovertemplate='obs_id: ' + object_id + \
                            '<br> type: ' + object_info['type'] + \
                            '<br> timestep: %{customdata[0]}' + \
                            
                            '<br> pred_x: %{y}' + \
                            '<br> pred_y: %{x}' + \
                            '<br> v_x: %{customdata[1]}' + \
                            '<br> v_y: %{customdata[2]}' + \
                            '<br> v: %{customdata[3]}' + \
                            '<br> original_type: ' + str(object_info['type']) + '<extra></extra>',
                        hoveron='points',
                        name="pred_mode_0" + object_id,
                        showlegend=False,
                        connectgaps=False
                    )
                )
                if object_id == 'ego':
                    ego_multi_model_trajs = object_info['ego_multi_model_trajs']
                    ego_multi_model_color = object_info['ego_multi_model_trajs_color']
                    ego_multi_model_input_noise = object_info['ego_multi_model_input_noise']
                    ego_multi_model_fix_distance_path_trajs = object_info['ego_multi_model_fix_distance_path_trajs']
                    ego_best_fix_distance_path_trajs = object_info['ego_best_fix_distance_path_trajs']
                    tmp_fig.add_trace(
                        go.Scatter(
                            y=ego_best_fix_distance_path_trajs[:, 0],
                            x=ego_best_fix_distance_path_trajs[:, 1],
                            # customdata=list(range(1, len(object_info['trajs'][CURRENT_TS:])+1)),
                            
                            customdata = np.array([
                                        np.array(range(1, len(ego_best_fix_distance_path_trajs)+1)),
                                        np.diff(ego_best_fix_distance_path_trajs[:,0],1,prepend=0)/0.1,
                                        np.diff(ego_best_fix_distance_path_trajs[:,1],1,prepend=0)/0.1,
                                        np.hypot(np.diff(ego_best_fix_distance_path_trajs[:,0],1,prepend=0)/0.1, np.diff(ego_best_fix_distance_path_trajs[:,1],1,prepend=0)/0.1)]).T,
                            

                            mode='markers+lines',
                            marker=dict(color=object_info['future_color'], size=5, symbol='triangle-up'),
                            line=dict(color=object_info['future_color'], width=1),
                            hovertemplate='obs_id: ' + object_id + \
                                '<br> type: ' + object_info['type'] + \
                                '<br> point_num: %{customdata[0]}' + \
                                
                                '<br> pred_x: %{y}' + \
                                '<br> pred_y: %{x}' + \
                                '<br> v_x: %{customdata[1]}' + \
                                '<br> v_y: %{customdata[2]}' + \
                                '<br> v: %{customdata[3]}' + \
                                '<br> original_type: ' + str(object_info['type']) + '<extra></extra>',
                            hoveron='points',
                            name="pred_mode_0_fix_distance_path" + object_id,
                            showlegend=False,
                            connectgaps=False
                        )
                    )
                    for i in range(ego_multi_model_fix_distance_path_trajs.shape[0]):
                        tmp_fig.add_trace(
                            go.Scatter(
                                y=ego_multi_model_fix_distance_path_trajs[i,:, 0],
                                x=ego_multi_model_fix_distance_path_trajs[i,:, 1],
                                customdata = np.array([
                                    np.array(range(1, len(ego_multi_model_fix_distance_path_trajs[i])+1)),
                                    np.diff(ego_multi_model_fix_distance_path_trajs[i,:,0],1,prepend=0)/0.1,
                                    np.diff(ego_multi_model_fix_distance_path_trajs[i,:,1],1,prepend=0)/0.1,
                                    np.hypot(
                                        np.diff(ego_multi_model_fix_distance_path_trajs[i,:,0],1,prepend=0)/0.1, 
                                        np.diff(ego_multi_model_fix_distance_path_trajs[i,:,1],1,prepend=0)/0.1
                                        )]).T,
                                mode='markers+lines',
                                marker=dict(color=ego_multi_model_color[i], size=5, symbol='triangle-up'),
                                line=dict(color=ego_multi_model_color[i], width=1),
                                hovertemplate='obs_id: ' + object_id + f' mode:{i}'\
                                    '<br> type: ' + object_info['type'] + \
                                    '<br> point_num: %{customdata[0]}' + \
                                    '<br> pred_x: %{y}' + \
                                    '<br> pred_y: %{x}' + \
                                    '<br> v_x: %{customdata[1]}' + \
                                    '<br> v_y: %{customdata[2]}' + \
                                    '<br> v: %{customdata[3]}' + \
                                    '<br> original_type: ' + str(object_info['type']) + \
                                    '<extra></extra>',
                                hoveron='points',
                                name=f"pred_mode_{i+1}_fix_distance_path" + object_id,
                                showlegend=False,
                                connectgaps=False
                            )
                        )
                    for i in range(ego_multi_model_trajs.shape[0]):
                        tmp_fig.add_trace(
                            go.Scatter(
                                y=ego_multi_model_trajs[i,:, 0],
                                x=ego_multi_model_trajs[i,:, 1],
                                # customdata=list(range(1, len(ego_multi_model_trajs[i])+1)),
                                customdata = np.array([
                                    np.array(range(1, len(ego_multi_model_trajs[i])+1)),
                                    np.diff(ego_multi_model_trajs[i,:,0],1,prepend=0)/0.1,
                                    np.diff(ego_multi_model_trajs[i,:,1],1,prepend=0)/0.1,
                                    np.hypot(
                                        np.diff(ego_multi_model_trajs[i,:,0],1,prepend=0)/0.1, 
                                        np.diff(ego_multi_model_trajs[i,:,1],1,prepend=0)/0.1
                                        )]).T,
                                mode='markers+lines',
                                marker=dict(color=ego_multi_model_color[i], size=5),
                                line=dict(color=ego_multi_model_color[i], width=1),
                                hovertemplate='obs_id: ' + object_id + f' mode:{i}'\
                                    '<br> type: ' + object_info['type'] + \
                                    '<br> timestep: %{customdata[0]}' + \
                                    
                                    '<br> pred_x: %{y}' + \
                                    '<br> pred_y: %{x}' + \
                                    '<br> v_x: %{customdata[1]}' + \
                                    '<br> v_y: %{customdata[2]}' + \
                                    '<br> v: %{customdata[3]}' + \
                                    '<br> original_type: ' + str(object_info['type']) + \
                                        '<br> input_noise_std: ' + f'{torch.std(ego_multi_model_input_noise[i]).item()}' +  '<extra></extra>',
                                hoveron='points',
                                name=f"pred_mode_{i+1}" + object_id,
                                showlegend=False,
                                connectgaps=False
                            )
                        )
                    
                
                # 绘制当前边界框
                # TODO: Plot current bbox
                tmp_fig.add_trace(
                    go.Scatter(
                        y=object_info['polygon_x'],
                        x=object_info['polygon_y'],
                        mode='lines',
                        line_color=object_info['past_color'],
                        hovertemplate='obs_id: ' + object_id + \
                            '<br> type: ' + object_info['type'] + \
                            '<br> timestep: ' + str(0) + \
                            '<br> x: %{y}' + \
                            '<br> y: %{x}' + \
                            '<br> original_type: ' + str(object_info['original_type']) + '<extra></extra>',
                        hoveron='points',
                        name="current_" + object_id,
                        showlegend=False
                    )
                )
                
            
            # 更新布局
            # update the layout
            tmp_fig.update_layout(
                height=1200,
                width=800,
                xaxis=dict(range=[80, -80], autorange=False, zeroline=False, showgrid=False),
                yaxis=dict(range=[-80, 160], autorange=False, zeroline=False, showgrid=False),
                xaxis_title='y',
                yaxis_title='x',
                plot_bgcolor='silver',
                legend_x=1.2,
                showlegend=False
            )
            tmp_fig.update_xaxes(showgrid=False, zeroline=False)
            tmp_fig.update_yaxes(showgrid=False, zeroline=False)
            
            plot_frames.append(go.Frame(data=tmp_fig.data, layout=tmp_fig.layout, name=str(frame_idx)))
        
        print(f"创建了 {len(plot_frames)} 个可视化帧")
        return plot_frames
    
    def create_final_animation(self, plot_frames: List[go.Frame], title: str = 'mcap_viz') -> go.Figure:
        """
        创建最终动画
        
        Args:
            plot_frames: 绘图帧列表
            title: 动画标题
            
        Returns:
            最终Plotly图形
        """
        print("创建最终动画...")
        final_fig = go.Figure(frames=plot_frames)
        first_frame = plot_frames[0]
        final_fig.add_traces(first_frame.data)
        #final_fig.add_traces([go.Scatter(x=[],y=[],showlegend=False) for _ in range(len(object_ids_set)+max_polyline_num-len(first_frame.data))])
        final_fig.layout = first_frame.layout
        
        def slider_args(duration):
            return {
                "frame": {"duration": duration, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            }
        
        method = "animate"
        sliders = [
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(font={"size": 20}, prefix="Step:", visible=True, xanchor="right"),
                transition=dict(duration=0, easing="linear"),
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[f.name], slider_args(50)],
                        label=f'{k}',
                        method=method,
                    ) for k, f in enumerate(final_fig.frames)
                ],
            )
        ]
        
        final_fig = final_fig.update_layout(
            title=title,
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[None, dict(frame={"duration": 100, "redraw": False}, 
                                          fromcurrent=True, transition={"duration": 0, "easing": "linear"})],
                            label="Play",
                            method=method
                        ),
                        dict(
                            args=[[None], dict(frame={"duration": 0, "redraw": False}, 
                                            mode="immediate", transition={"duration": 0})],
                            label="Pause",
                            method=method
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    type="buttons",
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                ),
                dict(
                    buttons=[
                        dict(
                            args=[{"visible": True}],
                            label="show all",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [not ("gt_" in (trace.name or "")) or 
                                             ("pred_" in (trace.name or "")) for trace in final_fig.data]}],
                            label="show pred",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [not ("pred_" in (trace.name or "")) or 
                                             ("gt_" in (trace.name or "")) for trace in final_fig.data]}],
                            label="show gt",
                            method="restyle"
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 32},
                    showactive=True,
                    type="buttons",
                    x=0.3,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                ),
                dict(
                    buttons=[
                        dict(
                            args=[{"visible": [not (("pred_" in (trace.name or ""))) or ("pred_mode_0" in (trace.name or "")) for trace in final_fig.data]}],
                            label="mode_0",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [not (("pred_" in (trace.name or ""))) or ("pred_mode_1" in (trace.name or "")) for trace in final_fig.data]}],
                            label="mode_1",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [not (("pred_" in (trace.name or ""))) or ("pred_mode_2" in (trace.name or "")) for trace in final_fig.data]}],
                            label="mode_2",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [not (("pred_" in (trace.name or ""))) or ("pred_mode_3" in (trace.name or "")) for trace in final_fig.data]}],
                            label="mode_3",
                            method="restyle"
                        ),
                    ],
                    direction="left",
                    pad={"r": 10, "t": 32},
                    showactive=True,
                    type="buttons",
                    x=0.0,
                    xanchor="left",
                    y=1.0,
                    yanchor="top"
                ),
                dict(
                    buttons=[
                        dict(
                            args=[{"visible": [not (("pred_" in (trace.name or ""))) or ("pred_mode_4" in (trace.name or "")) for trace in final_fig.data]}],
                            label="mode_4",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [not (("pred_" in (trace.name or ""))) or ("pred_mode_5" in (trace.name or "")) for trace in final_fig.data]}],
                            label="mode_5",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [not (("pred_" in (trace.name or ""))) or ("pred_mode_6" in (trace.name or "")) for trace in final_fig.data]}],
                            label="mode_6",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [not (("pred_" in (trace.name or ""))) or ("pred_mode_7" in (trace.name or "")) for trace in final_fig.data]}],
                            label="mode_7",
                            method="restyle"
                        ),
                    ],
                    direction="left",
                    pad={"r": 10, "t": 32},
                    showactive=True,
                    type="buttons",
                    x=0.0,
                    xanchor="left",
                    y=0.95,
                    yanchor="top"
                )
            ],
            sliders=sliders
        )
        
        return final_fig
    

    def assign_routing_polyline_plot_info(self, map_seg_xy, polyline_type, 
                              line_color, line_width=2):
        polyline_dict               = dict()
        polyline_dict['type']       = polyline_type
        polyline_dict['points_x']   = map_seg_xy[:, 0]
        polyline_dict['points_y']   = map_seg_xy[:, 1]
        polyline_dict['line_color'] = line_color
        polyline_dict['line_width'] = line_width
        return polyline_dict

    def run_pipeline(self, test_data_dir: str, start_idx: int = 150, horizon: int = 10,
                    show_plot: bool = False) -> go.Figure:
        """
        运行完整可视化流程
        
        Args:
            test_data_dir: 测试数据目录
            start_idx: 起始索引
            horizon: 时间步数
            show_plot: 是否显示图形
            
        Returns:
            最终Plotly图形
        """
        print("开始扩散模型可视化流程...")
        
        # 1. 加载模型和配置
        self.load_model()
        # self.load_data_config() # Do not use this if data are already cached
        
        # 2. 加载测试数据
        test_data_list = self.load_test_data(test_data_dir, start_idx, horizon)
        
        # 3. 处理所有帧
        processed_frames = []
        max_agent_num = 0
        
        for idx, test_data in enumerate(test_data_list):
            print(f"处理帧 {idx + 1}/{len(test_data_list)}...")
            
            # 处理单帧
            frame_result = self.process_single_frame(test_data, idx)
            
            # 处理静态地图信息
            static_map_polylines = self.process_static_map_info(frame_result['static_map_info'])
            
            # 计算最大智能体数量
            current_agent_num = len(list(frame_result['target_agents_info'].keys()))
            if current_agent_num > max_agent_num:
                max_agent_num = current_agent_num
            
            # 处理目标智能体信息
            target_agents_dict = self.process_target_agents_info(
                frame_result['target_agents_info'], max_agent_num
            )


            routing_polyline_dict = {}
            ego_routing_info_dict = frame_result['ego_routing_info']

            navi_info_dict = frame_result['navi_features']
            for navi_id, navi_xy in enumerate(navi_info_dict['xy']):
                navi_xy_valid = navi_info_dict['valid'][navi_id].astype(bool)
                navi_xy = navi_xy[navi_xy_valid]
                if len(navi_xy) > 0:
                    routing_polyline_dict['navi_'+str(navi_id)] = self.assign_routing_polyline_plot_info(
                                                                    navi_xy, 'navi', self.routing_color_map['navi'])
            
            for ego_routing_id, ego_routing_xy in enumerate(ego_routing_info_dict['xy']):
                ego_routing_xy_valid = ego_routing_info_dict['valid'][ego_routing_id].astype(bool)
                ego_routing_xy = ego_routing_xy[ego_routing_xy_valid]
                if len(ego_routing_xy) > 0:
                    routing_polyline_dict['ego_routing_'+str(ego_routing_id)] = self.assign_routing_polyline_plot_info(
                                                                    ego_routing_xy, 'ego_routing', self.routing_color_map['ego_routing'], line_width=4)
            # routing_polyline_list.append(routing_polyline_dict)

            ego_navi_info_dict = frame_result['ego_navi_features']
            for ego_navi_id, ego_navi_xy in enumerate(ego_navi_info_dict['xy']):
                ego_navi_xy_valid = ego_navi_info_dict['valid'][ego_navi_id].astype(bool)
                ego_navi_xy = ego_navi_xy[ego_navi_xy_valid]
                if len(ego_navi_xy) > 0:
                    routing_polyline_dict['ego_navi_'+str(ego_navi_id)] = self.assign_routing_polyline_plot_info(
                                                                    ego_navi_xy, 'ego_navi', self.routing_color_map['ego_navi'], line_width=4)

            
            # 收集所有处理后的信息
            processed_frame = {
                'static_map_polylines': static_map_polylines,
                'traffic_light_info': frame_result['traffic_light_info'],
                'ego_routing_info': frame_result['ego_routing_info'],
                'ego_turn_light_info': frame_result['ego_turn_light_info'],
                'ego_traffic_light_info': frame_result['ego_traffic_light_info'],
                'target_agents_dict': target_agents_dict,
                'frame_idx': frame_result['frame_idx'],
                'routing_ployline_dict': routing_polyline_dict
            }
            
            processed_frames.append(processed_frame)
        
        print(f"最大智能体数量: {max_agent_num}")
        
        # 4. 创建可视化
        plot_frames = self.create_visualization_frames(processed_frames)
        final_fig = self.create_final_animation(plot_frames)
        
        # 5. 输出结果
        # 从model_checkpoint路径中提取模型名称
        model_dir = os.path.dirname(self.model_checkpoint)
        model_name = os.path.basename(model_dir) 
        viz_save_dir = f"./viz_results/{model_name}"
        os.makedirs(viz_save_dir, exist_ok=True)

        # 从test_data_dir路径中提取数据信息
        # test_data_dir = "/iag_ad_01/ad/yanghang2/ld/ld_v6_val_vsd/cache_ld_v6_val_vsd/routing_multi_left_lane_change/2025_07_04_10_50_31"
        path_parts = test_data_dir.split('/')
        # 获取倒数第二和最后一部分
        if len(path_parts) >= 2:
            data_scenario = path_parts[-2]  # "routing_multi_left_lane_change"
            data_timestamp = path_parts[-1]  # "2025_07_04_10_50_31"
            test_data_log = f"{data_scenario}_{data_timestamp}"
        else:
            # 如果路径不符合预期，使用备用方案
            test_data_log = os.path.basename(test_data_dir)

        start_frame = start_idx
        actual_horizon = len(test_data_list)
        final_fig.write_html(f"{viz_save_dir}/{test_data_log}_{start_frame}-{start_frame+actual_horizon}.html", auto_play=False)
        
        if show_plot:
            print("显示可视化结果...")
            final_fig.show()
        
        print("可视化流程完成!")
        return final_fig


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='扩散模型可视化工具')
    parser.add_argument('--config', type=str, required=True,
                       help='数据配置文件路径')
    parser.add_argument('--model', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='测试数据目录')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='起始数据索引')
    parser.add_argument('--horizon', type=int, default=0,
                       help='时间步数')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='运行设备')
    
    args = parser.parse_args()
    
    # 创建可视化器并运行流程
    visualizer = DiffusionModelVisualizer(
        config_path=args.config,
        model_checkpoint=args.model,
        device=args.device
    )
    
    final_fig = visualizer.run_pipeline(
        test_data_dir=args.data_dir,
        start_idx=args.start_idx,
        horizon=args.horizon,
        show_plot=False
    )
    
    # test sample 


    # visualizer = DiffusionModelVisualizer(
    #     config_path = os.path.join(root_dir, 'config', 'dataset', 'LitDiffusionDataset_viz.yaml'),
    #     model_checkpoint = "/iag_ad_01/ad/mayicheng/fix_distance_path_ego_navi_lr0.001_bs192/epoch=77-val_loss=0.0148.ckpt",
    #     device = "cpu"
    # )
    
    # final_fig = visualizer.run_pipeline(
    #     test_data_dir = "/iag_ad_01/ad/yanghang2/ld/1113_vsd/v0_fix/cache_v0_fix/turn_left/2025_11_08_10_23_28",
    #     start_idx = 0,
    #     horizon = 10,
    #     show_plot = False
    # )


if __name__ == "__main__":
    main()

'''
# 命令行使用
python diffusion_visualizer.py \
    --config path/to/config.yaml \
    --model path/to/model.ckpt \
    --data_dir path/to/test_data \
    --start_idx 150 \
    --horizon 10 \
    --device cpu
'''