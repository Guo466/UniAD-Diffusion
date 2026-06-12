import os
import pickle
import shutil
import numpy as np
import random
import torch
import lightning as L

from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass, asdict
from typing import Optional
from utils.data_utils import get_yaw_rotation_matrix, transform_component_coordinates, wrap_angle
from datasets.builder import LITDATASET
from utils.data_utils import load_config

VALID_AGENTS_MAX_NUM = 128
STATIC_MAP_MAX_NUM = 512
NAVI_MAX_NUM = 256
EGO_NAVI_MAX_NUM = 16
DYNAMIC_MAP_MAX_NUM = 64
EGO_ROUTING_MAX_NUM = 64

object_type_mapping = {
    'UNSET': 0,
    'VEHICLE': 1,
    'PEDESTRIAN': 2,
    'CYCLIST': 3,
    'OTHER': 4
}

static_map_type_mapping = {
    # unknown type
    "UNKNOWN": 0,
    "LANE_UNKNOWN": 0,
    "UNKNOWN_LINE": 0,
    # lane types
    "LANE_FREEWAY": 1,
    "LANE_SURFACE_STREET": 2,
    "LANE_SURFACE_UNSTRUCTURE" : 2,
    "LANE_BIKE_LANE": 3,
    # road line types
    "ROAD_LINE_BROKEN_SINGLE_WHITE": 6,
    "ROAD_LINE_SOLID_SINGLE_WHITE": 7,
    "ROAD_EDGE_SIDEWALK": 7,
    "ROAD_LINE_SOLID_DOUBLE_WHITE": 8,
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": 9,
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 10,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": 11,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": 12,
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": 13,
    # road edge types
    "ROAD_EDGE_BOUNDARY": 15,
    "ROAD_EDGE_MEDIAN": 16,
    # crosswalk types
    "CROSSWALK": 18,
    # stopline
    "STOP_LINE": 14
}

@dataclass
class MotionPredictionOutputData:
    """
    Attributes
    ----------
    components_num:                         valid agents num and map polylines num of shape [2, ]

    valid_agents_features:                  valid agents features for model training of shape [valid_agents_num, past_horizon, 32]
    valid_agents_ids:                       valid agents ids for model evaluation of shape [valid_agents_num, ]
    valid_agents_types:                     valid agents types for model evaluation of shape [valid_agents_num, ]

    static_map_features:                    map polylines features for model training of shape [map_polylines_num, map_polyline_max_points_num, 8]
    dynamic_map_features:                   traffic light(tl) polylines features for model training of shape [tl_polylines_num, tl_polyline_max_points_num, 9]

    ego_routing_features:                   ego routing features for model training of shape [num_ego_routing_features, max_points_num, 7]
    ego_light_features:                     ego light features for model training of shape [3, ]

    navi_features:                          navi polylines features for model training of shape [navi_polylines_num, map_polyline_max_points_num, 7]
    ego_navi_features:                      ego navi polylines features for model training of shape [ego_navi_polylines_num, map_polyline_max_points_num, 7]

    target_agents_num:                      passed from 'MotionPredictionInputData' for loss computation
    target_current_info:                    target agents scene-centric [x,y,heading] for model evaluation of shape [target_agents_num, 3], 3->[x,y,heading]
    target_lw:
    target_gt_valid:                        target agents groundtruth valid flag for model evaluation of shape [target_agents_num, future_horizon]
    per_agent_centric_target_gt_for_loss:   target agents groundtruth for loss computation of shape [target_agents_num, future_horizon, 5], 5->[x,y,vx,vy,heading]
    scene_centric_target_gt_for_eval:       target agents groundtruth for model evaluation of shape [target_agents_num, future_horizon, 7], 7->[x,y,l,w,heading,vx,vy]

    valid_agents_pos:                       valid agents last valid pos(ego_centric_x/y/heading) of shape [valid_agents_num, 3]
    map_polylines_pos:                      map polylines first pos(x/y) of shape [map_polylines_pos, 2]

    scenario_id:                            pickle file name
    record_id:                              pickle file directory name

    tracks_to_predict_ids:                  tracks to predict for model joint evaluation [0 or 2, ]
    dr_pos:                                 drone position for model joint evaluation of shape [3, ]
    dr_heading:                             drone heading for model joint evaluation of shape [1, ] 
    objects_of_interest:                    objects of interest ids for model joint evaluation [0 or 2, ]
    """
    # Valid Agents num and Map Polylines num for masking
    components_num: np.ndarray
    # Valid Agents features
    valid_agents_features: np.ndarray
    valid_agents_ids: np.ndarray
    valid_agents_types: np.ndarray
    # Map features
    static_map_features: np.ndarray
    dynamic_map_features: np.ndarray
    # Ego routing features
    ego_routing_features: np.ndarray
    # Navi  features
    navi_features: np.ndarray
    # Ego navi  features
    ego_navi_features: np.ndarray
    # Ego cmd features
    ego_turn_light_features: np.ndarray
    ego_traffic_light_features: np.ndarray
    # Target Agents groundtruth related
    target_agents_num: int
    target_current_info: np.ndarray
    target_lw: np.ndarray
    target_gt_valid: np.ndarray
    per_agent_centric_target_gt_for_loss: np.ndarray
    scene_centric_target_gt_for_eval: np.ndarray
    # Positional Encoding related
    valid_agents_pos: Optional[np.ndarray] = None
    map_polylines_pos: Optional[np.ndarray] = None
    # Log info
    scenario_id: Optional[str] = None
    record_id: Optional[str] = None
    # Object to predict
    tracks_to_predict_ids: Optional[np.ndarray] = None
    valid_agents_num: Optional[int] = None
    dr_pos: Optional[np.ndarray] = None
    dr_heading: Optional[np.ndarray] = None
    # Object for joint evaluation
    objects_of_interest: Optional[np.ndarray] = None

factor_base = 1
factor_tagged = 1
# scene_extract = {
#     'left_nudge': 1 * factor_base,                      # 左绕障
#     'left_change_lane': 0.5 * factor_base,             # 左变道
#     'left_change_lane_efficient': 0.5 * factor_base,   # 左效率变道
#     'right_nudge': 1 * factor_base,                     # 右绕障
#     'right_change_lane': 0.8 * factor_base,            # 右变道
#     'right_change_lane_efficient': 0.1 * factor_base,  # 右效率变道
#     'routing_multi_left_lane_change': 1 * factor_base, 
#     'routing_multi_right_lane_change': 1 * factor_base, 
#     'turn_left': 1.5 * factor_base,                       # 左转
#     'turn_left_static': 8 * factor_base,                # 左转静止
#     'interactive_turn_left': 0.4 * factor_base,           # 左转交互
#     'interactive_turn_left_static': 0.2 * factor_base,    # 左转交互静止
#     'turn_right': 1 * factor_base,                    # 右转
#     'turn_right_static': 0.4 * factor_base,               # 右转静止
#     'interactive_turn_right': 0.4 * factor_base,        # 右转交互
#     'interactive_turn_right_static': 0.01 * factor_base,   # 右转交互静止
#     'static': 30 * factor_base,                         # 静止
#     'interactive_agent_cross': 0.04 * factor_base,         # 横穿agent交互
#     'interactive_agent_large_dec': 0.5 * factor_base,     # 急减速agent交互
#     'deviation_correction': 0.8 * factor_base,         # 偏离回正
#     'large_curvature_lane_keeping': 1 * factor_base,    # 大曲率车道保持
#     'brake': 0.4 * factor_base,                           # 刹车
#     'brake2stop': 2 * factor_base,                      # 刹停
#     'acc': 2 * factor_base,                             # 加速
#     'safe_acc': 0.02 * factor_base,                      # 提速
#     'cross_lane_keeping': 5 * factor_base,              # 路口车道保持
#     'lane_keeping': 8 * factor_base,                    # 车道保持
#     'near_intersection': 0.2 * factor_base,               # 接近路口
#     'split': 0.06 * factor_base,                         # 分流
#     'merge': 0.04 * factor_base,                           # 合流
#     'roundabout': 1 * factor_base,                      # 环岛
#     'normal': 8 * factor_base,                          # 平庸

#     'tagged_left_nudge': 1 * factor_base,                      # 左绕障
#     'tagged_left_change_lane': 0.25 * factor_base,             # 左变道
#     'tagged_left_change_lane_efficient': 0.25 * factor_base,   # 左效率变道
#     'tagged_right_nudge': 1 * factor_base,                     # 右绕障
#     'tagged_right_change_lane': 0.25 * factor_base,            # 右变道
#     'tagged_right_change_lane_efficient': 0.25 * factor_base,  # 右效率变道
#     'tagged_routing_multi_left_lane_change': 20 * factor_base, 
#     'tagged_routing_multi_right_lane_change': 20 * factor_base, 
#     'tagged_turn_left': 0.5 * factor_base,                       # 左转
#     'tagged_turn_left_static': 8 * factor_base,                # 左转静止
#     'tagged_interactive_turn_left': 1 * factor_base,           # 左转交互
#     'tagged_interactive_turn_left_static': 1 * factor_base,    # 左转交互静止
#     'tagged_turn_right': 0.5 * factor_base,                    # 右转
#     'tagged_turn_right_static': 8 * factor_base,               # 右转静止
#     'tagged_interactive_turn_right': 0.5 * factor_base,        # 右转交互
#     'tagged_interactive_turn_right_static': 1 * factor_base,   # 右转交互静止
#     'tagged_static': 64 * factor_base,                         # 静止
#     'tagged_interactive_agent_cross': 0.05 * factor_base,         # 横穿agent交互
#     'tagged_interactive_agent_large_dec': 1 * factor_base,     # 急减速agent交互
#     'tagged_deviation_correction': 1 * factor_base,            # 偏离回正
#     'tagged_large_curvature_lane_keeping': 1 * factor_base,    # 大曲率车道保持
#     'tagged_brake': 1 * factor_base,                           # 刹车
#     'tagged_brake2stop': 8 * factor_base,                      # 刹停
#     'tagged_acc': 4 * factor_base,                             # 加速
#     'tagged_safe_acc': 4 * factor_base,                        # 提速
#     'tagged_cross_lane_keeping': 0.5 * factor_base,              # 路口车道保持
#     'tagged_lane_keeping': 8 * factor_base,                    # 车道保持
#     'tagged_near_intersection': 1 * factor_base,               # 接近路口
#     'tagged_split': 0.5 * factor_base,                         # 分流
#     'tagged_merge': 1 * factor_base,                           # 合流
#     'tagged_roundabout': 1 * factor_base,                      # 环岛
#     'tagged_normal': 8 * factor_base,                          # 平庸
# }

scene_extract = {
    'left_nudge': 0.5 * factor_base,                      # 左绕障
    'left_change_lane': 0.3 * factor_base,             # 左变道
    'left_change_lane_efficient': 0.2 * factor_base,   # 左效率变道
    'right_nudge': 1 * factor_base,                     # 右绕障
    'right_change_lane': 0.4 * factor_base,            # 右变道
    'right_change_lane_efficient': 0.1 * factor_base,  # 右效率变道
    'routing_multi_left_lane_change': 1 * factor_base, 
    'routing_multi_right_lane_change': 1 * factor_base, 
    'turn_left': 1.5 * factor_base,                       # 左转
    'turn_left_static': 8 * factor_base,                # 左转静止
    'interactive_turn_left': 0.4 * factor_base,           # 左转交互
    'interactive_turn_left_static': 0.2 * factor_base,    # 左转交互静止
    'turn_right': 1 * factor_base,                    # 右转
    'turn_right_static': 0.4 * factor_base,               # 右转静止
    'interactive_turn_right': 0.4 * factor_base,        # 右转交互
    'interactive_turn_right_static': 0.01 * factor_base,   # 右转交互静止
    'static': 30 * factor_base,                         # 静止
    'interactive_agent_cross': 0.06 * factor_base,         # 横穿agent交互
    'interactive_agent_large_dec': 0.8 * factor_base,     # 急减速agent交互
    'deviation_correction': 0.3 * factor_base,         # 偏离回正
    'large_curvature_lane_keeping': 0.5 * factor_base,    # 大曲率车道保持
    'brake': 0.4 * factor_base,                           # 刹车
    'brake2stop': 2 * factor_base,                      # 刹停
    'acc': 3 * factor_base,                             # 加速
    'safe_acc': 0.02 * factor_base,                      # 提速
    'cross_lane_keeping': 1.2 * factor_base,              # 路口车道保持
    'lane_keeping': 5 * factor_base,                    # 车道保持
    'near_intersection': 0.2 * factor_base,               # 接近路口
    'split': 0.01 * factor_base,                         # 分流
    'merge': 0.01 * factor_base,                           # 合流
    'roundabout': 0.01 * factor_base,                      # 环岛
    'normal': 3 * factor_base,                          # 平庸

    'tagged_left_nudge': 1 * factor_base,                      # 左绕障
    'tagged_left_change_lane': 0.25 * factor_base,             # 左变道
    'tagged_left_change_lane_efficient': 0.25 * factor_base,   # 左效率变道
    'tagged_right_nudge': 1 * factor_base,                     # 右绕障
    'tagged_right_change_lane': 0.25 * factor_base,            # 右变道
    'tagged_right_change_lane_efficient': 0.25 * factor_base,  # 右效率变道
    'tagged_routing_multi_left_lane_change': 20 * factor_base, 
    'tagged_routing_multi_right_lane_change': 20 * factor_base, 
    'tagged_turn_left': 0.5 * factor_base,                       # 左转
    'tagged_turn_left_static': 8 * factor_base,                # 左转静止
    'tagged_interactive_turn_left': 1 * factor_base,           # 左转交互
    'tagged_interactive_turn_left_static': 1 * factor_base,    # 左转交互静止
    'tagged_turn_right': 0.5 * factor_base,                    # 右转
    'tagged_turn_right_static': 8 * factor_base,               # 右转静止
    'tagged_interactive_turn_right': 0.5 * factor_base,        # 右转交互
    'tagged_interactive_turn_right_static': 1 * factor_base,   # 右转交互静止
    'tagged_static': 64 * factor_base,                         # 静止
    'tagged_interactive_agent_cross': 0.05 * factor_base,         # 横穿agent交互
    'tagged_interactive_agent_large_dec': 1 * factor_base,     # 急减速agent交互
    'tagged_deviation_correction': 1 * factor_base,            # 偏离回正
    'tagged_large_curvature_lane_keeping': 1 * factor_base,    # 大曲率车道保持
    'tagged_brake': 1 * factor_base,                           # 刹车
    'tagged_brake2stop': 8 * factor_base,                      # 刹停
    'tagged_acc': 4 * factor_base,                             # 加速
    'tagged_safe_acc': 4 * factor_base,                        # 提速
    'tagged_cross_lane_keeping': 0.5 * factor_base,              # 路口车道保持
    'tagged_lane_keeping': 8 * factor_base,                    # 车道保持
    'tagged_near_intersection': 1 * factor_base,               # 接近路口
    'tagged_split': 0.5 * factor_base,                         # 分流
    'tagged_merge': 1 * factor_base,                           # 合流
    'tagged_roundabout': 1 * factor_base,                      # 环岛
    'tagged_normal': 8 * factor_base,                          # 平庸
}


def point_to_line_distance(point, line_points):
    point = np.array(point)
    line_points = np.array(line_points)
    
    if len(line_points) < 2:
        min_distance = np.linalg.norm(line_points[0] - point)
        closest_point = line_points[0]
        segment_index = 0


    
    min_distance = float('inf')
    closest_point = None
    segment_index = -1
    
    for i in range(len(line_points) - 1):
        p1 = line_points[i]
        p2 = line_points[i + 1]
        
        dist, closest = point_to_segment_distance(point, p1, p2)
        
        if dist < min_distance:
            min_distance = dist
            closest_point = closest
            segment_index = i
    
    return min_distance, closest_point, segment_index

def point_to_segment_distance(point, seg_start, seg_end):
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    
    line_vec = seg_end - seg_start
    point_vec = point - seg_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq == 0:
        return np.linalg.norm(point_vec), seg_start
    
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = max(0, min(1, t))  
    
    projection = seg_start + t * line_vec
    
    dist = np.linalg.norm(point - projection)
    
    return dist, projection


@LITDATASET.register_module()
class LitDiffusionDataset(L.LightningDataModule):
    def __init__(self, config=None, batch_size=4, num_workers=4):
        super().__init__()
        if config is None:
            config_path = os.path.join(os.getcwd(), 'config', 'dataset', self.__class__.__name__+'.yaml')
            config = load_config(config_path)
        elif isinstance(config, str):
            config_path = os.path.join(os.getcwd(), 'config', 'dataset', config)
            config = load_config(config_path)
        self.data_config = config
        self.dataset = DiffusionDataset
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers == -1 else num_workers

    def prepare_data(self):
        self.dataset(self.data_config, is_train=True, cache_data_to_disk=True, num_workers=self.num_workers)
        self.dataset(self.data_config, is_train=False, cache_data_to_disk=True, num_workers=self.num_workers)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(self.data_config, is_train=True, cache_data_to_disk=False, num_workers=self.num_workers)
            self.val_dataset = self.dataset(self.data_config, is_train=False, cache_data_to_disk=False, num_workers=self.num_workers)
        if stage == "test":
            self.test_dataset = self.dataset(self.data_config, is_train=False, cache_data_to_disk=False, num_workers=self.num_workers)
        if stage == "predict":
            self.predict_dataset = self.dataset(self.data_config, is_train=False, cache_data_to_disk=False, num_workers=self.num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.train_dataset.collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=int(os.environ.get("PREFETCH_FACTOR", "4")))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.val_dataset.collate_fn, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.test_dataset.collate_fn, pin_memory=True, persistent_workers=True)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.predict_dataset.collate_fn, pin_memory=True, persistent_workers=True)

class DiffusionDataset(Dataset):
    def __init__(self, data_config, is_train=True, cache_data_to_disk=False, num_workers=4):
        if is_train:
            self.data_dir = data_config["train_dir"]
        else:
            self.data_dir = data_config["validation_dir"]
        self.data_config = data_config
        self.is_train = is_train
        self.local_rank = os.environ.get("LOCAL_RANK", 0)
        self.processed_data_loaded_in_memory = []
        self.data_chunk_size = self.data_config["data_chunk_size"]
        self.processed_data_paths = []
        self.num_workers = num_workers
        if cache_data_to_disk:
            self.cache_data_to_disk()
        else:
            self.load_data()

    def __len__(self):
        return len(self.processed_data_paths)
    
    def __getitem__(self, index):
        if self.data_config["store_data_in_memory"]:
            return self.processed_data_loaded_in_memory[index]
        else:
            data_path = self.processed_data_paths[index]
            if not os.path.exists(self.processed_data_paths[index]):
                # data_path="data_root/cached_file_path"
                data_path = "/".join(self.data_dir[0].split("/")[:-1]+self.processed_data_paths[index].split("/")[-3:]) 
            with open(data_path, 'rb') as f:
                return pickle.load(f)
            
    def cache_data_to_disk(self):
        # if self.is_train:
        #     print(f"Loading raw training data from {self.data_dir}...")
        # else:
        #     print(f"Loading raw validation data from {self.data_dir}...")
    
        for cnt, dataset_dir in enumerate(self.data_dir):
            dataset_name = dataset_dir.split('/')[-1]
            data_used_num = self.data_config["data_used_num"][cnt] // self.data_chunk_size if self.data_config["data_used_num"] is not None and self.data_config["data_used_num"][cnt] is not None else None
            self.cache_data_dir = os.path.join(dataset_dir, f'cache_{dataset_name}')
            self.cached_data_path = os.path.join(self.cache_data_dir, 'cached_data_paths.pkl')
            if os.path.exists(self.cache_data_dir) and os.path.exists(self.cached_data_path) and not self.data_config["overwrite_cache"]:
                print(f"{'Training' if self.is_train else 'Validation'} data: Cache data directory '{self.cache_data_dir}' already exists and overwrite_cache is False, skip caching")
            else:
                print(f"Start caching {'training' if self.is_train else 'validation'} data from {self.data_dir}")
                # read data summary and data mapping
                summary_path = os.path.join(dataset_dir, 'dataset_summary.pkl')
                mapping_path = os.path.join(dataset_dir, 'dataset_mapping.pkl')
                if os.path.exists(mapping_path):
                    # with open(summary_path, 'rb') as f:
                    #     summary = pickle.load(f)
                    with open(mapping_path, 'rb') as f:
                        mapping = pickle.load(f)
                else:
                    raise FileNotFoundError(f"Dataset summary or mapping file not found in {dataset_dir}")
                # scenario_file_names_list = list(summary.keys())
                scenario_file_names_list = list(mapping.keys())
                print(f"Find {len(scenario_file_names_list)} files in '{dataset_name}' dataset")
                # caching preparation
                if os.path.exists(self.cache_data_dir):
                    shutil.rmtree(self.cache_data_dir)
                os.makedirs(self.cache_data_dir, exist_ok=True)
                process_num = self.num_workers
                print(f"Using {process_num} processes to load and process diffusion data from '{dataset_name}' dataset...")
                scenario_file_names_list_chunks = np.array_split(scenario_file_names_list, process_num)
                scenario_file_splits_info = [(dataset_dir, mapping, list(scenario_file_names_list_chunk), dataset_name) for scenario_file_names_list_chunk in scenario_file_names_list_chunks]
                # save the scenario file splits in a tmp file
                os.makedirs('tmp', exist_ok=True)
                for i in range(process_num):
                    tmp_scenario_file_splits_path = os.path.join('tmp', f'scenario_file_splits_{i}.pkl')
                    with open(tmp_scenario_file_splits_path, 'wb') as f:
                        pickle.dump(scenario_file_splits_info[i], f)
                # multiprocessing data processing
                with Pool(processes=process_num) as p:
                    #results = p.map(self.process_data_chunk, list(range(process_num)))
                    #results = list(tqdm(p.imap(self.process_data_chunk, list(range(process_num))), total=process_num, desc="Computing feature"))
                    results = list(p.imap(self.process_data_chunk, list(range(process_num))))
                # collect results
                processed_data_paths_dict = {}
                for result in results:
                    processed_data_paths_dict.update(result)

                # save the cached data paths
                with open(os.path.join(self.cache_data_dir, 'cached_data_paths.pkl'), 'wb') as f:
                    pickle.dump(processed_data_paths_dict, f)
                print(f"Finished caching {len(processed_data_paths_dict)} chunked processed data from '{dataset_name}' {'training' if self.is_train else 'validation'} dataset")
                
                # clear the tmp files
                shutil.rmtree('tmp', ignore_errors=True)

    def load_data(self):
        self.processed_data_paths_dict = {}
        for cnt, dataset_dir in enumerate(self.data_dir):
            dataset_name = dataset_dir.split('/')[-1]
            data_used_num = self.data_config["data_used_num"][cnt] // self.data_chunk_size if self.data_config["data_used_num"] is not None and self.data_config["data_used_num"][cnt] is not None else None
            self.cache_data_dir = os.path.join(dataset_dir, f'cache_{dataset_name}')
            if not os.path.exists(self.cache_data_dir):
                raise FileNotFoundError(f"Local rank {self.local_rank}: Cache data directory {self.cache_data_dir} does not exist")
            else:
                processed_data_paths_dict = self.get_data_paths_from_cache(data_used_num)
                print(f"Local rank {self.local_rank}: Loaded {len(processed_data_paths_dict)} chunked cached {'training' if self.is_train else 'validation'} data from '{dataset_name}' dataset")
                # print the number of loaded data
            loaded_data_num = 0
            for file_info in processed_data_paths_dict.values():
                loaded_data_num += file_info['sample_num']
            print(f"Local rank {self.local_rank}: Loaded {loaded_data_num} processed {'training' if self.is_train else 'validation'} data from '{dataset_name}' dataset")
            self.processed_data_paths_dict.update(processed_data_paths_dict)

            # store data in memory
            if self.data_config["store_data_in_memory"]:
                print(f"Local rank {self.local_rank}: Storing data in memory...")
                for processed_data_path in processed_data_paths_dict.keys():
                    with open(processed_data_path, 'rb') as f:
                        processed_data = pickle.load(f)
                    self.processed_data_loaded_in_memory.append(processed_data)
                print(f"Stored {len(self.processed_data_loaded_in_memory)} chunked processed {'training' if self.is_train else 'validation'} data in memory")


        self.processed_data_paths = list(self.processed_data_paths_dict.keys())
        
        print(f"Local rank {self.local_rank}: Total {len(self.processed_data_paths)} chunked processed {'training' if self.is_train else 'validation'} data loaded")
        # Clear the processed data paths dict to save memory
        self.processed_data_paths_dict.clear()
        if self.is_train:
            self.processed_data_paths = self.sample_data(self.processed_data_paths, scene_extract)
            print(f"Local rank {self.local_rank}: Sampled {len(self.processed_data_paths)} chunked processed {'training' if self.is_train else 'validation'} data loaded")
        else:
            self.processed_data_paths = self.downsample_data_by_ratio(self.processed_data_paths, 0.25)
            print(f"Local rank {self.local_rank}: Downsampled {len(self.processed_data_paths)} chunked processed {'training' if self.is_train else 'validation'} data loaded")
        # self.processed_data_paths = self.sample_data(self.processed_data_paths, scene_extract)
        # print(f"Local rank {self.local_rank}: Sampled {len(self.processed_data_paths)} chunked processed {'training' if self.is_train else 'validation'} data loaded")
        

    @staticmethod
    def downsample_data_by_ratio(processed_data_paths, downsample_ratio):
        sampled_processed_data_paths = []
        for processed_data_path in processed_data_paths:
            if np.random.rand() < downsample_ratio:
                sampled_processed_data_paths.append(processed_data_path)
        return sampled_processed_data_paths
    
    @staticmethod
    def sample_data(processed_data_paths, scene_extract_factor_dict):
        sampled_processed_data_paths = []
        for processed_data_path in processed_data_paths:
            scene_tag = processed_data_path.split('/')[-3]
            if scene_tag in scene_extract_factor_dict:
                scene_extract_factor = scene_extract_factor_dict[scene_tag]
                if scene_extract_factor >= 1:
                    scene_downsample_ratio = 1 / scene_extract_factor
                    if np.random.rand() < scene_downsample_ratio:
                        sampled_processed_data_paths.append(processed_data_path)
                else:
                    scene_upsample_ratio = int(1 / scene_extract_factor)
                    sampled_processed_data_paths.extend([processed_data_path] * scene_upsample_ratio)
            else:
                sampled_processed_data_paths.append(processed_data_path)
        return sampled_processed_data_paths

    @staticmethod
    def collate_fn(batch):
        batch_samples = [entry[0] for entry in batch]
        def _prepare_array(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            return arr

        def _pad_first_dim(array_list, max_len, pad_value=0.0):
            if not array_list:
                return None
            first = _prepare_array(array_list[0])
            out_shape = (len(array_list), max_len, *first.shape[1:])
            output = np.full(out_shape, pad_value, dtype=first.dtype)
            for idx, arr in enumerate(array_list):
                arr = _prepare_array(arr)
                length = min(arr.shape[0], max_len)
                output[idx, :length] = arr[:length]
            return torch.from_numpy(output)

        def _stack_no_pad(array_list):
            return torch.from_numpy(np.stack([_prepare_array(arr) for arr in array_list]))

        def _concat_to_tensor(array_list):
            arrays = [_prepare_array(arr) for arr in array_list]
            return torch.from_numpy(np.concatenate(arrays, axis=0))

        model_input = {
            "valid_agents_features": _pad_first_dim(
                [sample["valid_agents_features"] for sample in batch_samples], VALID_AGENTS_MAX_NUM
            ),
            "valid_agents_pos": _pad_first_dim(
                [sample["valid_agents_pos"] for sample in batch_samples], VALID_AGENTS_MAX_NUM
            ),
            "static_map_features": _pad_first_dim(
                [sample["static_map_features"] for sample in batch_samples], STATIC_MAP_MAX_NUM
            ),
            "dynamic_map_features": _pad_first_dim(
                [sample["dynamic_map_features"] for sample in batch_samples], DYNAMIC_MAP_MAX_NUM
            ),
            "map_polylines_pos": _pad_first_dim(
                [sample["map_polylines_pos"] for sample in batch_samples], STATIC_MAP_MAX_NUM
            ),
            "ego_routing_features": _pad_first_dim(
                [sample["ego_routing_features"] for sample in batch_samples], EGO_ROUTING_MAX_NUM
            ),
            "ego_navi_features": _pad_first_dim(
                [sample["ego_navi_features"] for sample in batch_samples], EGO_NAVI_MAX_NUM
            ),
            "ego_turn_light_features": _stack_no_pad(
                [sample["ego_turn_light_features"] for sample in batch_samples]
            ),
            "ego_traffic_light_features": _stack_no_pad(
                [sample["ego_traffic_light_features"] for sample in batch_samples]
            ),
            "components_num": _stack_no_pad(
                [sample["components_num"] for sample in batch_samples]
            ),
        }

        valid_agents_future_gt = []
        for sample in batch_samples:
            arr = sample["per_agent_centric_target_gt_for_loss"]
            arr = np.concatenate([arr[..., :2], arr[..., 4:5]], axis=-1)
            valid_agents_future_gt.append(arr)
        model_input["valid_agents_future_gt"] = _pad_first_dim(
            valid_agents_future_gt, VALID_AGENTS_MAX_NUM
        )
        model_input["valid_agents_future_gt_valid"] = _pad_first_dim(
            [sample["target_gt_valid"] for sample in batch_samples], VALID_AGENTS_MAX_NUM
        )

        gt_info = {
            "per_agent_centric_target_gt_for_loss": _concat_to_tensor(
                [sample["per_agent_centric_target_gt_for_loss"] for sample in batch_samples]
            ),
            "target_current_info": _concat_to_tensor(
                [sample["target_current_info"] for sample in batch_samples]
            ),
            "target_lw": _concat_to_tensor(
                [sample["target_lw"] for sample in batch_samples]
            ),
            "target_gt_valid": _concat_to_tensor(
                [sample["target_gt_valid"] for sample in batch_samples]
            ),
            "target_agents_num": torch.stack(
                [torch.tensor(sample["target_agents_num"], dtype=torch.int32) for sample in batch_samples]
            ),
        }

        motion_metrics_input = {
            "gt_trajectories": [sample["scene_centric_target_gt_for_eval"] for sample in batch_samples],
            "gt_is_valid": [sample["target_gt_valid"] for sample in batch_samples],
            "object_types": [sample["valid_agents_types"] for sample in batch_samples],
            "object_ids": [sample["valid_agents_ids"] for sample in batch_samples],
            "scenario_ids": [sample["scenario_id"] for sample in batch_samples],
            "target_current_info": [sample["target_current_info"] for sample in batch_samples],
        }

        ego_planning_metrics_input = {
            "gt_trajectories": [sample["scene_centric_target_gt_for_eval"] for sample in batch_samples],
            "gt_is_valid": [sample["target_gt_valid"] for sample in batch_samples],
            "object_types": [sample["valid_agents_types"] for sample in batch_samples],
            "object_ids": [sample["valid_agents_ids"] for sample in batch_samples],
            "scenario_ids": [sample["scenario_id"] for sample in batch_samples],
            "target_lw": [sample["target_lw"] for sample in batch_samples],
            "agents_num": [sample["target_agents_num"] for sample in batch_samples],
        }

        submission_input = {
            "dr_pos": [sample["dr_pos"] for sample in batch_samples],
            "dr_heading": [sample["dr_heading"] for sample in batch_samples],
            "objects_of_interest": [sample["objects_of_interest"] for sample in batch_samples],
        }

        return {
            "model_input": model_input,
            "gt_info": gt_info,
            "motion_metrics_input": motion_metrics_input,
            "ego_planning_metrics_input": ego_planning_metrics_input,
            "submission_input": submission_input
        }
           
  

    def process_data_chunk(self, worker_index):
        with open(os.path.join('tmp', f'scenario_file_splits_{worker_index}.pkl'), 'rb') as f:
            scenario_file_splits_info = pickle.load(f)
        processed_data_paths_dict = {}
        dataset_dir, mapping, scenario_file_names_list, dataset_name = scenario_file_splits_info
        output_buffer = []
        save_cnt = 0
        for cnt, scenario_file_name in enumerate(tqdm(scenario_file_names_list, leave=False, position=0, 
                                             desc=f"Worker {worker_index} Number of raw files: {len(scenario_file_names_list)}")):
            # load scenario data
            scenario_file_path = os.path.join(dataset_dir, mapping[scenario_file_name], scenario_file_name)
            with open(scenario_file_path, 'rb') as f:
                scenario = pickle.load(f)
            # TODO: add data processing here
            # try:
            output = self.preprocess(scenario)
            output = self.process(output)
            output = self.postprocess(output)
            # except Exception as e:
            #     print(f"Error processing scenario {scenario_file_name} from '{dataset_name}' dataset: {e}")
            #     continue
            # output = [{'scenario_id': scenario['id']}]
            output_buffer += output
            while len(output_buffer) >= self.data_chunk_size:
                # save_path = os.path.join(self.cache_data_dir, f'{dataset_name}_{worker_index}_{save_cnt}.pkl')
                # print(scenario_file_name)
                save_dir = os.path.join(self.cache_data_dir, *mapping[scenario_file_name].split('/')[-2:])
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{scenario_file_name.split('|')[-1]}")
                to_save = output_buffer[:self.data_chunk_size]
                output_buffer = output_buffer[self.data_chunk_size:]
                with open(save_path, 'wb') as f:
                    pickle.dump(to_save, f)
                save_cnt += 1
                file_info = {}
                file_info['sample_num'] = len(to_save)
                processed_data_paths_dict[save_path] = file_info
        # save the last chunk
        # save_path = os.path.join(self.cache_data_dir, f'{dataset_name}_{worker_index}_{save_cnt}.pkl')
        save_dir = os.path.join(self.cache_data_dir, *mapping[scenario_file_name].split('/')[-2:])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{scenario_file_name.split('|')[-1]}")
        print(f"Worker {worker_index} save last chunk: {save_dir}")
        if isinstance(output_buffer, dict):
            output_buffer = [output_buffer]
        if len(output_buffer) > 0:
            with open(save_path, 'wb') as f:
                pickle.dump(output_buffer, f)
            save_cnt += 1
            file_info = {}
            file_info['sample_num'] = len(output_buffer)
            processed_data_paths_dict[save_path] = file_info
        print(f"worker {worker_index} save last chunk done: {save_dir}")

        return processed_data_paths_dict
    
    def preprocess(self, scenario, if_include_lwh=True):
        '''
        Preprocess the scenario data.
        Currently, only tracks and static map are used.
        Return:
            ret:
                scenario_id: str, scenario id
                sdc_id: str, ego vehicle id
                track_feature_dict: dict, contains track features with shape [all_objects_num, total_frames_num, 10]
                static_map_feature_dict: dict, contains static map features with shape [all_polylines_num, max_points_num, 8]
                sdc_track_index: int, ego vehicle track index
                timestamp_second: np.ndarray, timestamp in seconds with shape [total_frames_num]
                tracks_to_predict: list, track ids to predict
        '''
        # load essential information
        static_map_features = scenario['static_map_features']
        dynamic_map_features = scenario['dynamic_map_features']
        ego_routing_features = scenario['ego_routing_features'] if 'ego_routing_features' in scenario else None
        navi_features = scenario['navi_features'] if 'navi_features' in scenario else None # contains navi features and ego navi features
        dr_pos = scenario['metadata'].get('old_origin_in_current_coordinate', np.zeros(3))
        dr_heading = scenario['metadata'].get('old_heading_in_current_coordinate', np.zeros(1))
        objects_of_interest = scenario['metadata'].get('objects_of_interest', scenario['metadata']['tracks_to_predict'].keys())
        tracks = scenario['tracks']
        # useful parameters
        current_frame_index = self.data_config["current_frame_index"]
        past_frames_num = self.data_config["past_frames_num"]
        future_frames_num = self.data_config["future_frames_num"]
        total_frames_num = past_frames_num + future_frames_num
        start_frame_index = current_frame_index + 1 - past_frames_num
        end_frame_index = current_frame_index + 1 + future_frames_num

        # collect tracks info
        tracks_info_dict = {
            'object_ids': [],
            'object_types': [],
            'object_features': [],
        }
        for obj_id, obj_feature in tracks.items():
            object_states = obj_feature['states']
            for state_name, state_value in object_states.items():
                # expand shape=1 states to shape=2
                if isinstance(state_value, np.ndarray) and len(state_value.shape) == 1:
                    object_states[state_name] = np.expand_dims(state_value, axis=-1)
            if if_include_lwh:
                object_others = obj_feature['others']
                for other_name, other_value in object_others.items():
                    if isinstance(other_value, np.ndarray) and len(other_value.shape) == 1:
                        object_others[other_name] = np.expand_dims(other_value, axis=-1)
                # required object info list
                # [x,y,z,vx,vy,heading,l,w,h,valid]
                required_object_features = [object_states['position'], object_states['velocity'], object_states['heading'],
                                            object_others['length'], object_others['width'], object_others['height'], object_states['valid']]
            else:
                required_object_features = [object_states['position'], object_states['velocity'], object_states['heading'], object_states['valid']]
            required_object_features = np.concatenate(required_object_features, axis=-1) # numpy concatenate will automatically convert bool type to float
            required_object_features = required_object_features[start_frame_index:end_frame_index]
            # Pad required object features to the total frames number
            if required_object_features.shape[0] < total_frames_num:
                required_object_features = np.pad(required_object_features, ((0, total_frames_num - required_object_features.shape[0]), (0, 0)), mode='constant', constant_values=0)
            assert required_object_features.shape[0] == total_frames_num, f'Required object features shape {required_object_features.shape[0]} does not match total frames number {total_frames_num}'
            tracks_info_dict['object_features'].append(required_object_features)
            tracks_info_dict['object_ids'].append(obj_id)
            tracks_info_dict['object_types'].append(object_type_mapping[obj_feature['type']]) # convert object type to int
        # stack all object states
        tracks_info_dict['object_features'] = np.stack(tracks_info_dict['object_features'], axis=0)
        tracks_info_dict['object_types'] = np.array(tracks_info_dict['object_types'], dtype=np.float32)

        # collect static map info
        static_map_info_dict = {
            'lane': [],
            'road_line': [],
            'road_edge': [],
            'crosswalk': [],
            'stopline': []
        }

        # collect dynamic map info
        dynamic_map_info_dict = {
            'traffic_light': [],
        }

        # collect ego info, reserve for lane change signals.
        ego_info_dict = {
            'ego_routing_polyline': [],
            'ego_turn_light': [],
            'ego_traffic_light': [],
        }

        # navi feature info
        navi_info_dict = {
            'navi_polyline': [],
            'ego_navi_polyline': []
        }


        for map_seg_id, map_seg_feature in static_map_features.items():
            map_seg_type = map_seg_feature['type']
            # v30
            # if map_seg_type in ["SPEED_BUMP", "STOP_SIGN", "STOP_LINE", "DRIVEWAY"]:
            if map_seg_type in ["SPEED_BUMP", "STOP_SIGN", "DRIVEWAY"]:
                continue
            map_seg_type_int = static_map_type_mapping[map_seg_type]
            # ignore unknown type
            if map_seg_type_int == 0:
                continue
            # collect map seg info according to the type
            # extract polyline or polygon
            if map_seg_feature['polyline'] is not None:
                polyline = map_seg_feature['polyline']
                polyline = self.interpolate_polyline(polyline, self.data_config["map_points_interp_step"])
                if len(polyline) % self.data_config["map_points_sample_interval"] == 1:
                    downsampled_polyline = polyline[::self.data_config["map_points_sample_interval"]] # 2m interval
                else:
                    downsampled_polyline = np.append(polyline[::self.data_config["map_points_sample_interval"]], polyline[-1:], axis=0) # 2m interval
                resized_polylines_list = self.resize_polyline_to_fixed_size_features(downsampled_polyline, map_seg_type_int, self.data_config["max_points_num"]) # 20m polyline
            elif map_seg_feature['polygon'] is not None:
                polygon_corners = map_seg_feature['polygon'] # [4, 2]
                polygon_edges = np.stack((polygon_corners, np.roll(polygon_corners, -1, axis=0)), axis=1) # [4,2,2]
                downsampled_polylines = []
                for edge in polygon_edges:
                    edge = self.interpolate_polyline(edge, self.data_config["map_points_interp_step"])
                    if len(edge) % self.data_config["map_points_sample_interval"] == 1:
                        downsampled_edge = edge[::self.data_config["map_points_sample_interval"]] # 2m interval
                    else:
                        downsampled_edge = np.append(edge[::self.data_config["map_points_sample_interval"]], edge[-1:], axis=0) # 2m interval
                    downsampled_polylines.append(downsampled_edge)
            else:
                continue
            
            ## lane
            if map_seg_type_int in [1, 2, 3]:
                static_map_info_dict['lane'].extend(resized_polylines_list)
            ## road line
            elif map_seg_type_int in [6, 7, 8, 9, 10, 11, 12, 13]:   
                static_map_info_dict['road_line'].extend(resized_polylines_list)
            ## road edge
            elif map_seg_type_int in [15, 16]:
                static_map_info_dict['road_edge'].extend(resized_polylines_list)
            ## crosswalk
            elif map_seg_type_int == 18:
                for polyline in downsampled_polylines:
                    # polygon
                    resized_polylines_list = self.resize_polyline_to_fixed_size_features(polyline, map_seg_type_int, self.data_config["max_points_num"]) # 20m polyline
                    static_map_info_dict['crosswalk'].extend(resized_polylines_list)
            ## stopline
            elif map_seg_type_int == 14:
                static_map_info_dict['stopline'].extend(resized_polylines_list)
            ## ignore other types(stop_sign, speedbump, etc.) currently
            else:
                continue

        # collect dynamic map info
        # TODO: parse all traffic light
        # parse ego traffic light
        if 'egolane_traffic_light_0' in dynamic_map_features.keys():
            stop_point = dynamic_map_features['egolane_traffic_light_0']['stop_point']
            state = dynamic_map_features['egolane_traffic_light_0']['states']['object_state'][0] # only one time step.TODO: tracking all time steps
            is_valid   = 1.0
            is_red     = 1.0 if state == 'LANE_STATE_STOP' else 0.0
            is_yellow  = 1.0 if state == 'LANE_STATE_CAUTION' else 0.0 
            is_green   = 1.0 if state == 'LANE_STATE_GO' else 0.0
            is_unknown = 1.0 if state == 'LANE_STATE_UNKNOWN' or state == 'LANE_STATE_OFF' else 0.0
            # TODO: feature reserved ---> control lane id
            ego_info_dict['ego_traffic_light'] = np.array([*stop_point, 0.0, 0.0, is_red, is_yellow, is_green, is_unknown, is_valid])[np.newaxis, :] # reserved for time steps
            
        # parse all traffic lights
        for object_id, object_info in dynamic_map_features.items():
            stop_point = object_info['stop_point']
            state = object_info['states']['object_state'][0] # only one time step.TODO: tracking all time steps
            is_valid   = 1.0
            is_red     = 1.0 if state == 'LANE_STATE_STOP' else 0.0
            is_yellow  = 1.0 if state == 'LANE_STATE_CAUTION' else 0.0 
            is_green   = 1.0 if state == 'LANE_STATE_GO' else 0.0
            is_unknown = 1.0 if state == 'LANE_STATE_UNKNOWN' or state == 'LANE_STATE_OFF' else 0.0
            # TODO: feature reserved ---> control lane id
            traffic_light_feature = [*stop_point, 0.0, 0.0, is_red, is_yellow, is_green, is_unknown, is_valid]
            dynamic_map_info_dict['traffic_light'].append(np.array(traffic_light_feature)[np.newaxis, :])# reserved for time steps

        # parse navi features
        navi_to_ego_gt_last_pt_dist_list = []
        navi_resized_polylines_list = []
        for object_id, object_info in navi_features.items():
            raw_polyline = object_info['polyline']
            type = int(object_info['others']['type']) # 0: not ego navi 1: ego navi 2: u turn navi
            # all navi lanes
            polyline = self.interpolate_polyline(raw_polyline, self.data_config["map_points_interp_step"]) # all navi lanes
            if len(polyline) % self.data_config["map_points_sample_interval"] == 1:
                downsampled_polyline = polyline[::self.data_config["map_points_sample_interval"]] # 2m interval
            else:
                downsampled_polyline = np.append(polyline[::self.data_config["map_points_sample_interval"]], polyline[-1:], axis=0) # 2m interval
            resized_polylines_list = self.resize_polyline_to_fixed_size_features(downsampled_polyline, None, self.data_config["max_points_num"]) # 20m polyline
            navi_info_dict['navi_polyline'].extend(resized_polylines_list)
            # if type == 1: # ego navi lanes

            # choose the closest navi linese as the ego navi line
            sdc_last_xy = tracks_info_dict['object_features'][0][-1][:2]
            dist,_,_ = point_to_line_distance(sdc_last_xy,raw_polyline)
            navi_to_ego_gt_last_pt_dist_list.append(dist)
            navi_resized_polylines_list.append(resized_polylines_list)
        # only record 1 nearest line as ego navi
        if len(navi_to_ego_gt_last_pt_dist_list) > 0:
            min_index = np.argmin(navi_to_ego_gt_last_pt_dist_list)
            min_navi_poly_list = navi_resized_polylines_list[min_index]
            navi_info_dict['ego_navi_polyline'].extend(min_navi_poly_list)


        if ego_routing_features['polyline'] is not None and len(ego_routing_features['polyline']) > 0:
            # only for st dataset
            #ego_routing_polyline = self.interpolate_polyline(ego_routing_polyline, self.data_config["map_points_interp_step"])
            #downsampled_polyline = ego_routing_polyline[::self.data_config["map_points_sample_interval"]] # 2m interval
            ego_routing_polyline = ego_routing_features['polyline']
            if len(ego_routing_polyline) % self.data_config["map_points_sample_interval"] == 1:
                downsampled_polyline = ego_routing_polyline[::self.data_config["map_points_sample_interval"]] # 2m interval
            else:
                # append last point
                downsampled_polyline = np.append(ego_routing_polyline[::self.data_config["map_points_sample_interval"]], ego_routing_polyline[-1:], axis=0) # 2m interval
            resized_polylines_list = self.resize_polyline_to_fixed_size_features(downsampled_polyline, None, self.data_config["max_points_num"]) # type None for ego routing
            ego_info_dict['ego_routing_polyline'].extend(resized_polylines_list) # num_polylines x num_points x 7
        
        if len(ego_routing_features['others'].keys()) > 0:
            # parse ego turn light info
            # 0: off | 1: left | 2: right 
            is_left = 1.0 if ego_routing_features['others']['ego_light'] == 1 else 0.0 
            is_right = 1.0 if ego_routing_features['others']['ego_light'] == 2 else 0.0
            is_unknown = 1.0 if ego_routing_features['others']['ego_light'] == 3 else 0.0
            ego_info_dict['ego_turn_light'] = np.array([is_left, is_right, is_unknown])
        
        # stack all polyline features
        for map_seg_type, map_seg_features in static_map_info_dict.items():
            if len(map_seg_features) > 0:
                static_map_info_dict[map_seg_type] = np.stack(map_seg_features, axis=0) # num_polylines x num_points x 8
                assert len(static_map_info_dict[map_seg_type].shape) == 3
            else:
                static_map_info_dict[map_seg_type] = np.zeros((0, 11, 8), dtype=np.float32)

        for navi_type, navi_seg_features in navi_info_dict.items():
            if len(navi_seg_features) > 0:
                navi_info_dict[navi_type] = np.stack(navi_seg_features, axis=0) # num_polylines x num_points x 7
                assert len(navi_info_dict[navi_type].shape) == 3
            else:
                navi_info_dict[navi_type] = np.zeros((0, 11, 7), dtype=np.float32)
       
        # stack all traffic light features
        if len(dynamic_map_info_dict['traffic_light']) > 0:
            dynamic_map_info_dict['traffic_light'] = np.stack(dynamic_map_info_dict['traffic_light'], axis=0) # num_lights x time_steps x 10
            # print(dynamic_map_info_dict['traffic_light'].shape)
            assert len(dynamic_map_info_dict['traffic_light'].shape) == 3
        else:
            # with padding later
            dynamic_map_info_dict['traffic_light'] = np.zeros((0, 1, 10), dtype=np.float32)

        # protect ego info features
        if len(ego_info_dict['ego_routing_polyline']) > 0:
            ego_info_dict['ego_routing_polyline'] = np.stack(ego_info_dict['ego_routing_polyline'], axis=0) # num_polylines x num_points x 7
            # print(ego_info_dict['ego_routing_polyline'].shape)
            assert len(ego_info_dict['ego_routing_polyline'].shape) == 3
        else: 
            # with padding later
            ego_info_dict['ego_routing_polyline'] = np.zeros((0, 11, 7), dtype=np.float32)

        # without padding
        if len(ego_info_dict['ego_turn_light']) > 0:
            # print(ego_info_dict['ego_turn_light'].shape)
            assert len(ego_info_dict['ego_turn_light'].shape) == 1
        else:
            ego_info_dict['ego_turn_light'] = np.zeros((3), dtype=np.float32)

        # without padding
        if len(ego_info_dict['ego_traffic_light']) > 0:
            ego_info_dict['ego_traffic_light'] = np.stack(ego_info_dict['ego_traffic_light'], axis=0) # 1 x time_steps x 10
            # print(ego_info_dict['ego_traffic_light'].shape)
            assert len(ego_info_dict['ego_traffic_light'].shape) == 2
        else:
            ego_info_dict['ego_traffic_light'] = np.zeros((1, 10), dtype=np.float32)

        # padding objects_of_interest to VALID_AGENTS_MAX_NUM
        # objects_of_interest = np.pad(objects_of_interest, (0, VALID_AGENTS_MAX_NUM - max(objects_of_interest.shape[0],VALID_AGENTS_MAX_NUM)), mode='constant', constant_values=-1)
        # collect all info
        ret = {
            'scenario_id': scenario['id'],
            'sdc_id': scenario['sdc_id'],
            'track_feature_dict': tracks_info_dict,
            'static_map_feature_dict': static_map_info_dict,
            'dynamic_map_feature_dict': dynamic_map_info_dict,
            'navi_info_dict':navi_info_dict,
            'ego_info_dict': ego_info_dict,
            'dr_pos': dr_pos,
            'dr_heading': dr_heading,
            'objects_of_interest': objects_of_interest
        }
        ret['sdc_track_index'] = tracks_info_dict['object_ids'].index(ret['sdc_id'])
        ret['timestamp_second'] = np.arange(total_frames_num) * 0.1
        ret['current_frame_index'] = past_frames_num - 1
        ret['past_frames_num'] = past_frames_num
        ret['future_frames_num'] = future_frames_num
        objects_of_interest_set = set(objects_of_interest)
        tracks_to_predict_set = set(scenario['metadata']['tracks_to_predict'].keys())
        tracks_to_predict_set.update(objects_of_interest_set)
        ret['tracks_to_predict_ids'] = list(tracks_to_predict_set)
        ret['tracks_to_predict_index'] = [tracks_info_dict['object_ids'].index(track_id) for track_id in ret['tracks_to_predict_ids']]
        return ret

    def process(self, output, x_range=[-100.0, 200.0], y_range=[-100.0, 100.0]): 
        valid_flag_index = -1
        current_frame_index = output['current_frame_index']
        past_frames_num = output['past_frames_num']
        future_frames_num = output['future_frames_num']
        # 1. Agents feature processing
        # rearrange tracks order, av at the first place, then tracks to predict, then valid agents
        ## valid agents are agents that at least have one valid frame in past_frames and in x_range and y_range
        agent_features = output['track_feature_dict']['object_features']
        assert len(agent_features.shape) == 3, f"Agent features shape {agent_features.shape} is not 3D"
        valid_agents_mask = np.sum(agent_features[:,:past_frames_num,valid_flag_index], axis=-1) > 0
        agents_current_xy = agent_features[:,current_frame_index,:2]
        agents_in_range_mask = np.logical_and(agents_current_xy[:,0] > x_range[0], agents_current_xy[:,0] < x_range[1]) & \
                                np.logical_and(agents_current_xy[:,1] > y_range[0], agents_current_xy[:,1] < y_range[1])
        valid_agents_mask = np.logical_and(valid_agents_mask, agents_in_range_mask)
        valid_agents_index = np.where(valid_agents_mask)[0]
        ## extract av and tracks to predict
        sdc_track_index = output['sdc_track_index']
        tracks_to_predict_index = np.array(output['tracks_to_predict_index'])
        if sdc_track_index not in tracks_to_predict_index:
            tracks_to_predict_index = np.insert(tracks_to_predict_index, 0, sdc_track_index, axis=0)
        else:
            av_mask = (tracks_to_predict_index == sdc_track_index)
            tracks_to_predict_index = np.concatenate([tracks_to_predict_index[av_mask], tracks_to_predict_index[~av_mask]])
        tracks_to_predict_valid_mask = [valid_agents_mask[index] for index in tracks_to_predict_index]
        valid_target_agents_index = tracks_to_predict_index[tracks_to_predict_valid_mask]
        valid_target_agents_ids = [output['track_feature_dict']['object_ids'][index] for index in valid_target_agents_index]
        valid_target_agents_num = len(valid_target_agents_ids)
        assert output['sdc_id'] in valid_target_agents_ids, f"Ego vehicle {output['sdc_id']} is not in valid target agents"
        valid_target_agents_types = [output['track_feature_dict']['object_types'][index] for index in valid_target_agents_index]
        valid_target_agents_features = agent_features[valid_target_agents_index] # [num_valid_target_agents, total_frames_num, 10]
        ## extract valid agents other than av and tracks to predict
        valid_other_agents_index = np.setdiff1d(valid_agents_index, tracks_to_predict_index)
        valid_other_agents_ids = [output['track_feature_dict']['object_ids'][index] for index in valid_other_agents_index]
        valid_other_agents_num = len(valid_other_agents_ids)
        valid_other_agents_types = [output['track_feature_dict']['object_types'][index] for index in valid_other_agents_index]
        valid_other_agents_features = agent_features[valid_other_agents_index] # [num_valid_other_agents, total_frames_num, 10]
        ## concat av, tracks to predict, and valid other agents
        new_sdc_track_index = 0
        all_valid_agents_ids = valid_target_agents_ids + valid_other_agents_ids
        all_valid_agents_num = valid_target_agents_num + valid_other_agents_num
        all_valid_agents_features = np.concatenate([valid_target_agents_features, valid_other_agents_features], axis=0)
        all_valid_agents_types = np.concatenate([valid_target_agents_types, valid_other_agents_types], axis=0).astype(np.int32)
        
        # 2. Compose all valid agents features and target agents ground truth
        valid_agents_num = min(all_valid_agents_num, VALID_AGENTS_MAX_NUM)
        assert valid_agents_num > 0, f"No valid agents found"
        valid_agents_past_info = all_valid_agents_features[:valid_agents_num, :past_frames_num]
        valid_agents_types = all_valid_agents_types[:valid_agents_num]
        valid_agents_ids = all_valid_agents_ids[:valid_agents_num]
        # Unpack valid agents past features
        valid_agents_past_xyz, valid_agents_past_vxy, valid_agents_past_heading, valid_agents_past_lwh, valid_agents_past_valid = np.split(
            valid_agents_past_info, indices_or_sections=[3, 5, 6, 9], axis=-1
        )
        # Agent features formation
        ## Agent types in one-hot
        valid_agents_one_hot_types = np.zeros((valid_agents_num, past_frames_num, 6), dtype=np.float32)
        valid_agents_one_hot_types[valid_agents_types==object_type_mapping['VEHICLE'], :, 0] = 1                        # 1: is VEHICLE
        valid_agents_one_hot_types[valid_agents_types==object_type_mapping['PEDESTRIAN'], :, 1] = 1                     # 2: is PEDESTRIAN
        valid_agents_one_hot_types[valid_agents_types==object_type_mapping['CYCLIST'], :, 2] = 1                        # 3: is CYCLIST
        valid_agents_one_hot_types[:, :, 3] = valid_agents_types[:, np.newaxis]                                         # 4: object type
        valid_agents_one_hot_types[:, :, 4] = 1                                                                         # 5: is Valid Agent
        valid_agents_one_hot_types[new_sdc_track_index, :, 5] = 1                                                       # 6: is Ego Agent
        ## Timesteps in one-hot
        past_timesteps = output['timestamp_second'][:past_frames_num]
        past_one_hot_timesteps = np.zeros((valid_agents_num, past_frames_num, past_frames_num+1), dtype=np.float32)
        past_one_hot_timesteps[:, np.arange(past_frames_num), np.arange(past_frames_num)] = 1                           # 1~11: is 1~11 timesteps
        past_one_hot_timesteps[:, np.arange(past_frames_num), -1] = past_timesteps                                      # 12: past timesteps values from 0.1 to 1.1
        ## per-agent-centric xyz, vxy, heading transformation
        valid_current_xyz, valid_current_heading ,valid_current_lwh = valid_agents_past_xyz[:, current_frame_index], valid_agents_past_heading[:, current_frame_index, 0],valid_agents_past_lwh[:, current_frame_index]
        valid_agents_yaw_rotation_matrix_3d = get_yaw_rotation_matrix(-valid_current_heading)                      # shape = [valid_agents_num, 3, 3]
        valid_agents_yaw_rotation_matrix_2d = valid_agents_yaw_rotation_matrix_3d[:, :2, :2]                            # shape = [valid_agents_num, 2, 2]

        per_agent_centric_valid_agents_past_xyz = transform_component_coordinates(valid_agents_past_xyz, valid_agents_yaw_rotation_matrix_3d, translation=-valid_current_xyz)
        per_agent_centric_valid_agents_past_vxy = transform_component_coordinates(valid_agents_past_vxy, valid_agents_yaw_rotation_matrix_2d)
        per_agent_centric_valid_agents_past_heading = wrap_angle(valid_agents_past_heading - valid_current_heading[:, np.newaxis, np.newaxis])
        per_agent_centric_valid_agents_past_heading_sin = np.sin(per_agent_centric_valid_agents_past_heading)
        per_agent_centric_valid_agents_past_heading_cos = np.cos(per_agent_centric_valid_agents_past_heading)
        ## concat agent features
        valid_agents_past_features = np.concatenate(
            [per_agent_centric_valid_agents_past_xyz, 
             per_agent_centric_valid_agents_past_vxy, 
             per_agent_centric_valid_agents_past_heading_sin, 
             per_agent_centric_valid_agents_past_heading_cos,
             valid_agents_one_hot_types,
             past_one_hot_timesteps,
             valid_agents_past_lwh,
             valid_agents_past_xyz,
             valid_agents_past_heading,
             valid_agents_past_valid],
            axis=-1
        ) * valid_agents_past_valid  # shape = [valid_agents_num, past_frames_num, 32]
        ## valid agents last pos for positional encoding
        valid_agents_last_valid_indices = np.zeros((valid_agents_num,), dtype=np.int32)
        valid_agents_last_valid_pos_xy = np.zeros((valid_agents_num, 2), dtype=np.float32)
        valid_agents_last_valid_pos_heading =  np.zeros((valid_agents_num,), dtype=np.float32)
        for t in range(past_frames_num):
            curr_valid_agents_mask = valid_agents_past_valid[:, t, 0] > 0
            valid_agents_last_valid_indices[curr_valid_agents_mask] = t
            valid_agents_last_valid_pos_xy[curr_valid_agents_mask] = valid_agents_past_xyz[curr_valid_agents_mask, t, :2]
            valid_agents_last_valid_pos_heading[curr_valid_agents_mask] = valid_agents_past_heading[curr_valid_agents_mask, t, 0]
        
        # 3.1 Static map feature processing
        # rearrange static map order according to the distance to ego vehicle and x,y range
        ## concat all static map features
        static_map_feature_dict = output['static_map_feature_dict']
        static_map_features = []
        for map_seg_type, map_seg_features in static_map_feature_dict.items():
            static_map_features.append(map_seg_features)
        static_map_features = np.concatenate(static_map_features, axis=0) # shape = [num_static_map_features, max_points_num, 8]
        assert len(static_map_features.shape) == 3, f"Static map features shape {static_map_features.shape} is not 3D"
        static_map_in_range_mask = np.logical_and(static_map_features[:, 0, 0] > x_range[0], static_map_features[:, 0, 0] < x_range[1]) & \
                                    np.logical_and(static_map_features[:, 0, 1] > y_range[0], static_map_features[:, 0, 1] < y_range[1])
        valid_static_map_features = static_map_features[static_map_in_range_mask]
        assert len(valid_static_map_features.shape) == 3, f"valid map features shape {valid_static_map_features.shape} is not 3D"
        ## sort static map features by distance to ego vehicle
        # ego vehicle position
        sdc_current_xyz = valid_current_xyz[new_sdc_track_index]
        # compute static map polyline first point xy
        static_map_polyline_first_point_xy = valid_static_map_features[:, 0, :2]
        # compute distance and sort
        static_map_dist_to_sdc = np.linalg.norm(static_map_polyline_first_point_xy - sdc_current_xyz[:2], axis=-1)
        sorted_static_map_indices = np.argsort(static_map_dist_to_sdc)
        sorted_static_map_features = valid_static_map_features[sorted_static_map_indices]
        sorted_static_map_first_point_xy = static_map_polyline_first_point_xy[sorted_static_map_indices]
        # select top k static map features
        selected_static_map_features = sorted_static_map_features[:STATIC_MAP_MAX_NUM]
        sorted_static_map_first_point_xy = sorted_static_map_first_point_xy[:STATIC_MAP_MAX_NUM]
        assert selected_static_map_features.shape[-1] == 8, f"Selected static map features shape {selected_static_map_features.shape} is not 8D"
        static_map_features_num = selected_static_map_features.shape[0]
        # input_components_num = np.array([valid_agents_num, static_map_features_num])

        # 3.2 Dynamic map feature processing
        # rearrange dynamic map order according to the distance to ego vehicle and x,y range
        ## concat all dynamic map features
        dynamic_map_feature_dict = output['dynamic_map_feature_dict']
        dynamic_map_feature = []
        for map_seg_type, map_seg_features in dynamic_map_feature_dict.items():
            dynamic_map_feature.append(map_seg_features)
        dynamic_map_feature = np.concatenate(dynamic_map_feature, axis=0) # shape = [num_static_map_features, max_points_num, 10] # TODO: 3d with time steps
        assert len(dynamic_map_feature.shape) == 3, f"Dynamic map features shape {dynamic_map_feature.shape} is not 3D"
        dynamic_map_in_range_mask = np.logical_and(dynamic_map_feature[:, 0, 0] > x_range[0], dynamic_map_feature[:, 0, 0] < x_range[1]) & \
                                    np.logical_and(dynamic_map_feature[:, 0, 1] > y_range[0], dynamic_map_feature[:, 0, 1] < y_range[1])
        valid_dynamic_map_feature= dynamic_map_feature[dynamic_map_in_range_mask]
        assert len(valid_dynamic_map_feature.shape) == 3, f"valid map features shape {valid_dynamic_map_feature.shape} is not 3D"
        ## sort dynamic map features by distance to ego vehicle
        # compute dynamic map polyline first point xy
        dynamic_map_stop_point_xy = valid_dynamic_map_feature[:, 0, :2]
        # compute distance and sort dynamic_map_stop_point_xy
        dynamic_map_dist_to_sdc = np.linalg.norm(dynamic_map_stop_point_xy - sdc_current_xyz[:2], axis=-1)
        sorted_dynamic_map_indices = np.argsort(dynamic_map_dist_to_sdc)
        sorted_dynamic_map_features = valid_dynamic_map_feature[sorted_dynamic_map_indices]
        sorted_dynamic_map_stop_point_xy = dynamic_map_stop_point_xy[sorted_dynamic_map_indices]
        # select past frames and top k static map features.TODO: 3d with time steps
        selected_dynamic_map_features = sorted_dynamic_map_features[:DYNAMIC_MAP_MAX_NUM]
        sorted_dynamic_map_stop_point_xy = sorted_dynamic_map_stop_point_xy[:DYNAMIC_MAP_MAX_NUM]
        assert selected_dynamic_map_features.shape[-1] == 10 , f'selected dynamic map features shape {selected_dynamic_map_features.shape}is not 10D'
        dynamic_map_features_num = selected_dynamic_map_features.shape[0]

        # 3.3 Ego routing feature processing
        # rearrange ego routing polyline order according to the distance to ego vehicle and x,y range
        ego_routing_features= output['ego_info_dict']['ego_routing_polyline']  # shape = [num_ego_routing_features, max_points_num, 7]
        assert len(ego_routing_features.shape) == 3, f"Ego routing features shape {ego_routing_features.shape} is not 3D"
        # TODO: if routing use different x,y range
        ego_routing_in_range_mask = np.logical_and(ego_routing_features[:, 0, 0] > x_range[0], ego_routing_features[:, 0, 0] < x_range[1]) & \
                                    np.logical_and(ego_routing_features[:, 0, 1] > y_range[0], ego_routing_features[:, 0, 1] < y_range[1])
        valid_ego_routing_features = ego_routing_features[ego_routing_in_range_mask]
        assert len(valid_ego_routing_features.shape) == 3, f"valid ego routing features shape {valid_ego_routing_features.shape} is not 3D"
        # routing do not need sort
        # select top k ego routing features
        selected_ego_routing_features = valid_ego_routing_features[:EGO_ROUTING_MAX_NUM]
        assert selected_ego_routing_features.shape[-1] == 7, f"Selected ego routing features shape {selected_ego_routing_features.shape} is not 7D"
        ego_routing_features_num = selected_ego_routing_features.shape[0]

        # 3.4 Ego light feature processing in one-hot shape = [3, ], 0: off | 1: left | 2: right TODO: check both
        ego_turn_light_features = output['ego_info_dict']['ego_turn_light']
        ego_traffic_light_features = output['ego_info_dict']['ego_traffic_light']

        # 3.6 navi feature processing
        # rearrange static map order according to the distance to ego vehicle and x,y range
        ## concat all static map features
        navi_feature_dict = output['navi_info_dict']
        navi_features = []
        ego_navi_features = []
        for navi_type, navi_seg_features in navi_feature_dict.items():
            if navi_type == 'ego_navi_polyline':
                ego_navi_features.append(navi_seg_features)
            navi_features.append(navi_seg_features)
        navi_features = np.concatenate(navi_features, axis=0) # shape = [num_static_map_features, max_points_num, 7]
        ego_navi_features = np.concatenate(ego_navi_features, axis=0) # shape = [num_static_map_features, max_points_num, 7]
        assert len(navi_features.shape) == 3, f"navi features shape {navi_features.shape} is not 3D"
        assert len(ego_navi_features.shape) == 3, f"navi features shape {ego_navi_features.shape} is not 3D"

        navi_features_in_range_mask = np.logical_and(navi_features[:, 0, 0] > x_range[0], navi_features[:, 0, 0] < x_range[1]) & \
                                    np.logical_and(navi_features[:, 0, 1] > y_range[0], navi_features[:, 0, 1] < y_range[1])

        ego_navi_features_in_range_mask = np.logical_and(ego_navi_features[:, 0, 0] > x_range[0], ego_navi_features[:, 0, 0] < x_range[1]) & \
                                    np.logical_and(ego_navi_features[:, 0, 1] > y_range[0], ego_navi_features[:, 0, 1] < y_range[1])                         
        valid_navi_features = navi_features[navi_features_in_range_mask]
        valid_ego_navi_features = ego_navi_features[ego_navi_features_in_range_mask]
        assert len(valid_navi_features.shape) == 3, f"valid navi features shape {valid_navi_features.shape} is not 3D"
        assert len(valid_ego_navi_features.shape) == 3, f"valid navi features shape {valid_ego_navi_features.shape} is not 3D"
        ## sort valid navi features by distance to ego vehicle
        # ego vehicle position
        sdc_current_xyz = valid_current_xyz[new_sdc_track_index]
        # compute static map polyline first point xy
        navi_polyline_first_point_xy = valid_navi_features[:, 0, :2]
        ego_navi_polyline_first_point_xy = valid_ego_navi_features[:, 0, :2]
        # compute distance and sort
        navi_dist_to_sdc = np.linalg.norm(navi_polyline_first_point_xy - sdc_current_xyz[:2], axis=-1)
        ego_navi_dist_to_sdc = np.linalg.norm(ego_navi_polyline_first_point_xy - sdc_current_xyz[:2], axis=-1)
        sorted_navi_indices = np.argsort(navi_dist_to_sdc)
        sorted_ego_navi_indices = np.argsort(ego_navi_dist_to_sdc)
        sorted_navi_features = valid_navi_features[sorted_navi_indices]
        sorted_ego_navi_features = valid_ego_navi_features[sorted_ego_navi_indices]
        sorted_navi_polyline_first_point_xy = navi_polyline_first_point_xy[sorted_navi_indices]
        sorted_ego_navi_polyline_first_point_xy = ego_navi_polyline_first_point_xy[sorted_ego_navi_indices]

        # select top k static map features
        selected_navi_features = sorted_navi_features[:NAVI_MAX_NUM]
        selected_ego_navi_features = sorted_ego_navi_features[:EGO_NAVI_MAX_NUM]
        sorted_navi_first_point_xy = sorted_navi_polyline_first_point_xy[:NAVI_MAX_NUM]
        sorted_ego_navi_first_point_xy = sorted_ego_navi_polyline_first_point_xy[:EGO_NAVI_MAX_NUM]

        assert selected_navi_features.shape[-1] == 7, f"Selected navi features shape {selected_navi_features.shape} is not 7D"
        assert selected_ego_navi_features.shape[-1] == 7, f"Selected ego navi features shape {selected_ego_navi_features.shape} is not 7D"

        navi_features_num = selected_navi_features.shape[0]
        ego_navi_features_num = selected_ego_navi_features.shape[0]
        # input_components_num = np.array([valid_agents_num, static_map_features_num])

        # 3.5 record input components num
        input_components_num = np.array([valid_agents_num, static_map_features_num, dynamic_map_features_num, 
                                        ego_routing_features_num,ego_navi_features_num,navi_features_num])

        # 4. Valid agents future ground truth
        valid_agents_future_info = all_valid_agents_features[:valid_agents_num, past_frames_num:past_frames_num+future_frames_num]
        ## Unpack valid agents future features
        valid_agents_future_xyz, valid_agents_future_vxy, valid_agents_future_heading, valid_agents_future_lwh, valid_agents_future_valid = np.split(
            valid_agents_future_info, indices_or_sections=[3, 5, 6, 9], axis=-1
        )
        ## per-agent-centric xyz, vxy, heading transformation
        per_agent_centric_valid_agents_xy_gt = transform_component_coordinates(valid_agents_future_xyz[..., :2], 
                                                                                    valid_agents_yaw_rotation_matrix_2d, 
                                                                                    translation=-valid_current_xyz[..., :2])
        per_agent_centric_valid_agents_vxy_gt = transform_component_coordinates(valid_agents_future_vxy, valid_agents_yaw_rotation_matrix_2d)
        per_agent_centric_valid_agents_heading_gt = wrap_angle(valid_agents_future_heading - valid_current_heading[:, np.newaxis, np.newaxis])

        per_agent_centric_valid_agents_gt_for_loss = np.concatenate([per_agent_centric_valid_agents_xy_gt, 
                                                                     per_agent_centric_valid_agents_vxy_gt, 
                                                                     per_agent_centric_valid_agents_heading_gt], axis=-1) # shape = [valid_agents_num, future_frames_num, 5]
        
        scene_centric_valid_agents_gt_for_eval = np.concatenate([valid_agents_future_xyz[..., :2], 
                                                                 valid_agents_future_lwh[..., :2],
                                                                 valid_agents_future_heading,
                                                                 valid_agents_future_vxy], axis=-1) # shape = [valid_agents_num, future_frames_num, 7]
        
        # 5. Gather output
        output = MotionPredictionOutputData(
            components_num=input_components_num,
            valid_agents_features=valid_agents_past_features,
            valid_agents_ids=np.array(valid_agents_ids),
            valid_agents_types=valid_agents_types,
            static_map_features=selected_static_map_features,
            dynamic_map_features=selected_dynamic_map_features,
            ego_routing_features=selected_ego_routing_features,
            ego_turn_light_features=ego_turn_light_features,
            ego_traffic_light_features=ego_traffic_light_features,
            navi_features=selected_navi_features,
            ego_navi_features=selected_ego_navi_features,
            target_agents_num=valid_target_agents_num,
            target_current_info=np.stack([
                valid_current_xyz[:valid_target_agents_num, ..., 0],
                valid_current_xyz[:valid_target_agents_num, ..., 1],
                valid_current_heading[:valid_target_agents_num]
            ], axis=-1),
            target_lw=np.stack([
                valid_current_lwh[:valid_target_agents_num, ..., 0],
                valid_current_lwh[:valid_target_agents_num, ..., 1],
            ], axis=-1),
            target_gt_valid=valid_agents_future_valid[:valid_target_agents_num,...,0],
            per_agent_centric_target_gt_for_loss=per_agent_centric_valid_agents_gt_for_loss[:valid_target_agents_num],
            scene_centric_target_gt_for_eval=scene_centric_valid_agents_gt_for_eval[:valid_target_agents_num],
            valid_agents_pos=np.concatenate([valid_agents_last_valid_pos_xy,valid_agents_last_valid_pos_heading[...,None]], axis=-1),
            map_polylines_pos=sorted_static_map_first_point_xy,
            tracks_to_predict_ids=np.array(output['tracks_to_predict_ids']),
            valid_agents_num=valid_agents_num,
            scenario_id=np.array(output['scenario_id']),
            dr_pos=np.array(output['dr_pos']),
            dr_heading=np.array(output['dr_heading']),
            objects_of_interest=np.array(output['objects_of_interest'])
        )
        return output

    def postprocess(self, output):
        return [asdict(output)]

    @classmethod
    def resize_polyline_to_fixed_size_features(cls, polyline, map_seg_type_int, max_num_points = 11):
        # align all poylines dim to 3
        if polyline.shape[-1] == 2:
            polyline = np.concatenate([polyline, np.zeros((polyline.shape[0], 1))], axis=-1)
        resized_polyline_features = []
        # split original polyline into fixed size polylines
        while polyline.shape[0] >= max_num_points:
            curr_polyline = polyline[:max_num_points]
            try:
                curr_polyline_basis_vector = cls.get_polyline_basis_vector(curr_polyline)
                curr_polyline_valid_array = np.ones((curr_polyline.shape[0], 1))
                if map_seg_type_int is not None: # for static map
                    curr_polyline_type_array = np.array([map_seg_type_int] * curr_polyline.shape[0])
                    curr_polyline_type_array = np.expand_dims(curr_polyline_type_array, axis=-1)
                    curr_polyline_features = np.concatenate([curr_polyline, curr_polyline_basis_vector, curr_polyline_type_array, curr_polyline_valid_array], axis=-1) # dim = 8
                else: # for ego routing
                    curr_polyline_features = np.concatenate([curr_polyline, curr_polyline_basis_vector, 
                    curr_polyline_valid_array], axis=-1) # dim = 7
            except:
                if map_seg_type_int is not None: # for static map
                    curr_polyline_features = np.zeros((0, 8), dtype=np.float32)
                else: # for ego routing
                    curr_polyline_features = np.zeros((0, 7), dtype=np.float32)
            resized_polyline_features.append(curr_polyline_features)
            polyline = polyline[max_num_points-1:]
        # deal with the last polyline part
        if len(polyline) > 0:
            last_polyline = polyline
            last_polyline_points_num = last_polyline.shape[0]
            try:
                last_polyline_basis_vector = cls.get_polyline_basis_vector(last_polyline)
                last_polyline_valid_array = np.ones((last_polyline.shape[0], 1))
                if map_seg_type_int is not None:
                    last_polyline_type_array = np.array([map_seg_type_int] * last_polyline.shape[0])
                    last_polyline_type_array = np.expand_dims(last_polyline_type_array, axis=-1)
                    last_polyline_features = np.concatenate([last_polyline, last_polyline_basis_vector, last_polyline_type_array, last_polyline_valid_array], axis=-1) # dim = 8
                else:   
                    last_polyline_features = np.concatenate([last_polyline, last_polyline_basis_vector, 
                    last_polyline_valid_array], axis=-1) # dim = 7
                last_polyline_features = np.pad(last_polyline_features, ((0, max_num_points - last_polyline_points_num), (0, 0)))
            except:
                last_polyline_features = np.zeros((0, 8), dtype=np.float32)
            resized_polyline_features.append(last_polyline_features)
        return resized_polyline_features

    @classmethod
    def get_polyline_basis_vector(cls, polyline):
        polyline_pre = np.roll(polyline, shift=1, axis=0)
        polyline_pre[0] = polyline[0]
        diff = polyline - polyline_pre
        polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
        return polyline_dir

    @classmethod
    def interpolate_polyline(cls, polyline, step=0.5):
        if polyline.shape[0] < 2:
            return polyline
        # Only interpolate x and y
        polyline = polyline[:, :2]
        # Compute the cumulative distance along the polyline
        cum_dist = np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))
        cum_dist = np.insert(cum_dist, 0, 0)
        # Interpolate the polyline at a fixed step size
        interp_dist = np.arange(0, cum_dist[-1], step)
        interpolated_polyline = []
        # since np.interp only support 1d interpolate, interpolate each dim respectively
        for dim in range(polyline.shape[1]):
            # linear interpolation
            interpolated_polyline.append(np.interp(interp_dist, cum_dist, polyline[:, dim]))
        interpolated_polyline = np.stack(interpolated_polyline, axis=-1)
        # # OR, use scipy.interpolate.interp1d
        # interp_func = interp1d(cum_dist, polyline, kind='linear', axis=0)
        # interpolated_polyline = interp_func(interp_dist)
        # reserve the last point
        interpolated_polyline = np.concatenate([interpolated_polyline, polyline[-1:]], axis=0)
        # append z dim with zeros
        interpolated_polyline = np.concatenate([interpolated_polyline, np.zeros((interpolated_polyline.shape[0], 1))], axis=-1)
        return interpolated_polyline
                
    def get_data_paths_from_cache(self, data_used_num):
        '''
        Get data from cache
        Return:
            processed_data_loaded_dict: dict, {processed_data_file_path: metainfo}
        '''
        cached_data_paths_path = os.path.join(self.cache_data_dir, 'cached_data_paths.pkl')
        if os.path.exists(cached_data_paths_path):
            with open(cached_data_paths_path, 'rb') as f:
                processed_data_paths_dict = pickle.load(f)
        else:
            raise FileNotFoundError(f"File path list not found in {self.cache_data_dir}")
        
        data_paths_list = list(processed_data_paths_dict.items())
        # np.random.shuffle(data_paths_list)
        
        if data_used_num is not None:
            processed_data_paths_dict = dict(data_paths_list[:data_used_num])
        else:
            processed_data_paths_dict = dict(data_paths_list)
        return processed_data_paths_dict