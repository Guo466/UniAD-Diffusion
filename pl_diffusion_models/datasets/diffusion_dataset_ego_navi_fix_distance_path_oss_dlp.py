import os
import os.path as osp
import pickle
import cv2
import bisect
import numpy as np
import torch
import lightning as L
import math
import copy
import random
import json
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

from datasets.cache_utils import (
    get_scene_cache_file,
    init_shm_cache_dir,
    load_processed_cache,
    save_processed_cache,
)

import json
import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import Optional, List, Tuple
from utils.misc_batch import plantf_collate_fn, is_main_process
from datasets.builder import LITDATASET
from utils.data_utils import load_config
from utils.dataset_utils.ego_heading_aug import EgoHeadingAugmentation
from utils.dataset_utils.navi_aug import naviAugmentation, retain_single_navitopo_recommend_lane
from utils.dataset_utils.bad_sample_retry_filter import check_bad_sample_for_retry
from .remote_client import create_remote_client
from .secondary_index_loader_v2 import build_shared_secondary_index_dict, load_secondary_index_entry, load_scene_map_shards, load_date_split_shards
from .mining_loader import load_mining_overfit_shared
from .navi_link_encode_v1 import (
    encode_navi_link_action_v1,
    compute_navi_link_ded_nearest_numpy,
    compute_navi_link_ms_nearest_numpy,
)
from scipy import ndimage
from .frame_label_loader import build_frame_label_dict, load_frame_label, load_date_mining
from scipy.interpolate import interp1d
import random
from utils.utils import img2real, ndarray2json
from visualization.visualization import visualization_dataset

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

map_agent_cls = {
    1 : 1,  # pedestrian
    2 : 2,  # vehicle
    3 : 3,  # bike
    14 : 3,  # bike
    6: 2 # truck
}   # 0 reserve for ego

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

@LITDATASET.register_module()
class LitDlpDataset(L.LightningDataModule):
    def __init__(self, config=None, batch_size=4, num_workers=4):
        super().__init__()
        if config is None:
            config_path = os.path.join(os.getcwd(), 'config', 'dataset', self.__class__.__name__+'.yaml')
            config = load_config(config_path)
        elif isinstance(config, str):
            config_path = os.path.join(os.getcwd(), 'config', 'dataset', config)
            config = load_config(config_path)
        self.data_config = config
        self.dataset = DlpDataset
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers == -1 else num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # 预取二级索引到 /dev/shm，供当前 rank 的所有 worker 进程共享
        _config = copy.deepcopy(self.data_config)
        
        if stage in ("fit", None):
            # 获取 train 的索引
            enabled = self.data_config.get("load_secondary_index_shards", True)
            shm_dir_train = build_shared_secondary_index_dict(
                data_config=self.data_config,
                is_train=True,
                enabled=enabled,
                scene_extract_override = {
                    k: v * self.data_config.get("factor_base", 1)
                    for k, v in self.data_config.get("scene_extract").items()
                },
            )
            _config["prefetched_shm_dir_train"] = shm_dir_train

            prefetched_framelabel_shm_dir = build_frame_label_dict(
                data_config=self.data_config,
                is_train=True,
            )
            _config["prefetched_framelabel_shm_dir"] = prefetched_framelabel_shm_dir
            self.train_dataset = self.dataset(_config, is_train=True)
            self.val_dataset = self.dataset(_config, is_train=True)
            
            # 同时获取 test/eval 的索引，以便后续测试时可以直接使用（避免在 test stage 时重新下载）
            shm_dir_eval = build_shared_secondary_index_dict(
                data_config=self.data_config,
                is_train=False,
                enabled=enabled,
                scene_extract_override=self.data_config.get("scene_extract_test"),
            )
            _config["prefetched_shm_dir_eval"] = shm_dir_eval
        
        if stage == "test":
            # 获取 test/eval 的索引
            enabled = self.data_config.get("load_secondary_index_shards", True)
            shm_dir_eval = build_shared_secondary_index_dict(
                data_config=self.data_config,
                is_train=False,
                enabled=enabled,
                scene_extract_override=self.data_config.get("scene_extract_test"),
            )
            _config["prefetched_shm_dir_eval"] = shm_dir_eval

            prefetched_framelabel_shm_dir = build_frame_label_dict(
                data_config=self.data_config,
                is_train=False,
            )
            _config["prefetched_framelabel_shm_dir"] = prefetched_framelabel_shm_dir
            self.test_dataset = self.dataset(_config, is_train=False)
        
        if stage == "predict":
            # 获取 test/eval 的索引
            enabled = self.data_config.get("load_secondary_index_shards", True)
            shm_dir_eval = build_shared_secondary_index_dict(
                data_config=self.data_config,
                is_train=False,
                enabled=enabled,
                scene_extract_override=self.data_config.get("scene_extract_test"),
            )
            _config["prefetched_shm_dir_eval"] = shm_dir_eval

            prefetched_framelabel_shm_dir = build_frame_label_dict(
                data_config=self.data_config,
                is_train=False,
            )
            _config["prefetched_framelabel_shm_dir"] = prefetched_framelabel_shm_dir
            self.predict_dataset = self.dataset(_config, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=plantf_collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=int(os.environ.get("PREFETCH_FACTOR", "4")))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=plantf_collate_fn, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=plantf_collate_fn, pin_memory=True, persistent_workers=True)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=plantf_collate_fn, pin_memory=True, persistent_workers=True)

class DlpDataset(Dataset):
    def __init__(self, data_config, is_train=True, cache_data_to_disk=False, num_workers=4):
        if is_train:
            self.remote_tmp_data_dir = data_config["tmp_data_dir"]
            self.from_ceph = data_config["from_ceph"]
            self.use_datapro_pb = data_config.get("use_datapro_pb", False)
            self.data_client = create_remote_client(data_config)
        else:
            self.remote_tmp_data_dir = data_config['tmp_data_dir_test']
            self.from_ceph = data_config['from_ceph_test']
            self.use_datapro_pb = data_config.get("use_datapro_pb", False)
            self.data_client = create_remote_client(data_config)
        
        self.prefetched_framelabel_shm_dir = data_config.get("prefetched_framelabel_shm_dir", "")
        split_file = data_config['date_split_file'] if is_train else data_config['date_split_file_test']
        self.date_split = load_date_split_shards(self.remote_tmp_data_dir, self.data_client, split_file, is_train)
        self.date_mining = load_date_mining(self.prefetched_framelabel_shm_dir, self.remote_tmp_data_dir, self.data_client, is_train)
        
        # 使用 /dev/shm 预取目录（在 LitDlpDataset.setup 中创建并传递）
        if is_train:
            self.prefetched_shm_dir = data_config.get("prefetched_shm_dir_train", "")
        else:
            self.prefetched_shm_dir = data_config.get("prefetched_shm_dir_eval", "")
        
        # 索引查找顺序：优先使用 shm，缺失则回退原始远端目录
        self.index_search_dirs = self.remote_tmp_data_dir if isinstance(self.remote_tmp_data_dir, list) else [self.remote_tmp_data_dir] 
        self.work_dir = data_config["work_dir"]
        self.flag_train = is_train
        #self.flag_eval = is_train
        self.flag_eval = False
        self.data_augment = data_config["data_augment"]
        self.cache_target_scenes = set(data_config.get("cache_target_scenes") or [])
        if not self.cache_target_scenes:
            # 空配置表示关闭缓存功能
            self.shm_json_cache_dir = None
        else:
            # /dev/shm 缓存目录，用于重复样本直接复用 load_json 的结果
            self.shm_json_cache_dir = init_shm_cache_dir()
            print(f"[DlpDataset] shm cache enabled. dir={self.shm_json_cache_dir}, target_scenes={sorted(self.cache_target_scenes)}")
        self.max_n_agent = data_config["max_n_agent"]
        self.max_n_laneline = data_config["max_n_laneline"]
        self.laneline_fixed_pts = data_config["laneline_fixed_pts"]
        self.n_feature_laneline_attr = data_config["n_feature_laneline_attr"]
        self.n_feature_agent_attr = data_config["n_feature_agent_attr"]
        self.n_feature_agent_status = data_config["n_feature_agent_status"]
        self.use_guidance = data_config["use_guidance"]
        self.costmap_config = data_config["costmap_config"]
        self.occmap_config = data_config["occmap_config"]
        self.use_tos_data = data_config.get("use_tos_data", True)

        if self.flag_train:
            scene_extract_total = {}
            for k, v in data_config['scene_extract'].items():
                scene_extract_total[k] = v*data_config['factor_base']
        else:
            scene_extract_total = data_config['scene_extract_test']
        self.scene_extract = scene_extract_total
        self.agent_hist_step_threshold = data_config['agent_hist_step_threshold']
        self.ego_hist_steps = data_config['ego_hist_steps']
        self.agent_hist_steps = data_config['agent_hist_steps']
        self.future_steps = data_config['future_steps']
        self.future_steps_fixed = data_config['future_steps_fixed']
        self.planning_interval = data_config['planning_interval']
        self.planning_interval_fixed = data_config['planning_interval_fixed']
        self.route_gt_steps = data_config['route_gt_steps']
        self.use_new_laneline = data_config['use_new_laneline']
        self.use_centerline = data_config['use_centerline']
        self.use_route = data_config['use_route']
        self.use_sdmap = data_config['use_sdmap']
        self.max_navi_link_actions = int(data_config.get("max_navi_link_actions", 32))
        self.navi_link_actions_max_dist_m = float(data_config.get("navi_link_actions_max_dist_m", 200.0))
        self.use_multiNavi = data_config['use_multiNavi']
        self.route_cost_thre = data_config['route_cost_thre']
        # 是否使用 Amap laneInfo 信息（foreground/background lane）作为路径输入的附加特征
        self.use_route_laneinfo = bool(data_config.get("use_route_laneinfo", True))
        self.mean_std_file = data_config['mean_std_file']
        self.infer_conut = 0
        if os.path.exists(self.mean_std_file):
            with open(self.mean_std_file, 'rb') as f:
                self.mean_std_dict = pickle.load(f)
        else:
            print(f'Error, mean_std_file {self.mean_std_file} not found !')
        self.n_query_lateral = data_config['n_query_lateral']
        self.n_query_longi = data_config['n_query_longi']
        self.ego_care_normal_laneline = data_config['ego_care_normal_laneline']
        self.use_new_laneline = data_config['use_new_laneline']
        self.open_eval_mode = data_config['open_eval_mode']
        self.trainStage = data_config['trainStage']

        self.backuped_data = None
        self.scenes, self.n_frames, self.scene_map = self.preprocess(self.scene_extract, data_config)
        if self.n_frames:
            self.normal_item_num = min(self.n_frames[-1], int(float(data_config['max_frame'])))
        else:
            self.normal_item_num = 0
            if not (data_config.get('mining_file') or data_config.get('mining_overfit_file')):
                print("[DlpDataset][WARNING] n_frames is empty (no scene in scene_map matches scene_extract). "
                      "Set mining_file/mining_overfit_file for pure jsonl mode, or check tmp_data_dir / scene_extract.")
        self.item_num = self.normal_item_num

        # mining/jsonl 配置：支持仅使用 jsonl 指定数据训练或测试
        if self.flag_train:
            self.mining_files = data_config.get('mining_file') or data_config.get('mining_overfit_file')
            self.mining_mix_ratio = data_config.get('mining_mix_ratio')
        else:
            self.mining_files = data_config.get('mining_file_test') or data_config.get('mining_overfit_file_test')
            self.mining_mix_ratio = None  # 测试集仅支持纯 mining
        self.mining_overfit_dir = None
        self.mining_len = 0
        self.init_mining_shared()

        if self.flag_train:
            print('\nTrain dataset is built!')
        else:
            print('\nTest dataset is built!')

        # 是否启用 occ 逻辑（默认关闭）。关闭时 get_data 会直接走占位 occ_map 分支。
        self.enable_occ = bool(data_config.get("enable_occ", False))
        self.max_n_occ_polygon = data_config['max_n_occ_polygon']
        self.max_n_occ_polygon_point = data_config['max_n_occ_polygon_point']
        self.max_n_occ_polygon_attr = data_config['max_n_occ_polygon_attr']

        # 初始化某些增强
        self.ego_heading_aug = EgoHeadingAugmentation(data_config)
        self.navi_aug = naviAugmentation()
        # Hack: 模型输入仅保留一组 navitopo+推荐车道（用于验证单组输入性能）
        self.retain_single_navi_recommend = bool(data_config.get("retain_single_navi_recommend", False))
        self.retain_single_navi_tol = float(data_config.get("retain_single_navi_tol", 0.5))
        self.retain_single_navi_prob = float(data_config.get("retain_single_navi_recommend_prob", 1.0))
        # bad sample retry mode:
        #   - off:      关闭检测
        #   - dry_run:  仅统计命中，不影响 trainFlag_route
        #   - enforce:  命中即置 trainFlag_route=False，触发 __getitem__ 重采样
        # 兼容旧配置：
        #   enable_bad_sample_retry_filter / bad_sample_retry_filter_dry_run
        mode_raw = data_config.get("bad_sample_retry_mode", "enforce")
        mode = str(mode_raw).strip().lower()
        if mode not in {"off", "dry_run", "enforce"}:
            print(f"[DlpDataset] WARNING: invalid bad_sample_retry_mode={mode_raw}, fallback to enforce")
            mode = "enforce"
        self.bad_sample_retry_mode = mode
        self.bad_sample_route_dist_threshold_m = float(data_config.get("bad_sample_route_dist_threshold_m", 30.0))
        self.bad_sample_route_truncate_first_dist_thr_m = float(
            data_config.get("bad_sample_route_truncate_first_dist_thr_m", 50.0)
        )
        self.bad_sample_route_behind_m = float(data_config.get("bad_sample_route_behind_m", 5.0))
        self.bad_sample_route_ahead_m = float(data_config.get("bad_sample_route_ahead_m", 195.0))
        self.bad_sample_retry_log_interval = int(data_config.get("bad_sample_retry_log_interval", 1000))
        self.bad_sample_retry_hit_count = 0
        self.bad_sample_retry_rule_hit_count = {
            "rule1_navitopo_far": 0,
            "rule2_ego_off_route": 0,
            "rule3_future_far": 0,
        }
        # 初始化阶段随机抽检10条做OccChecker校验，默认关闭
        if data_config.get("enable_occ_checker", False):
            self._occ_init_check_ok = self._init_occ_check()
        else:
            self._occ_init_check_ok = True

        print('total frames: ', self.item_num)
        print('video duration: %.2f h' %(self.item_num / 10. / 3600.))

    def preprocess(self, scene_extract, data_config):
        n_frame = 0
        scenes = []
        n_frames = []
        scene_map = load_scene_map_shards(self.index_search_dirs, self.data_client, data_config, scene_extract, self.flag_train)
        if scene_map is not None:
            for scene, cum_path_list in scene_map.items():
                if scene not in scene_extract:
                    continue
                scenes.append(scene)
                extract = scene_extract[scene]
                n_frame += int(cum_path_list[-1][0] / extract)
                n_frames.append(n_frame)
        else:
            raise FileNotFoundError(f"[DlpDataset][ERROR] load scene map failed")
        return scenes, n_frames, scene_map
        
    def __len__(self):
        return self.item_num
                
    def __getitem__(self, idx_raw):
        """
        训练阶段：若当前样本的 trainFlag_route 为 False，则随机重新采样其它 idx，
        尝试多次，尽量避免将 trainFlag_route=False 的样本送入训练 batch。
        支持 mining 模式：当配置 mining_file 时，可从 jsonl 指定数据采样。
        """
        max_retry = 20 if self.flag_train else 1
        attempt = 0

        while True:
            # 第一次使用 DataLoader 传入的 idx_raw，后续重试随机采样新的 idx_raw
            if attempt == 0:
                cur_idx_raw = idx_raw
            else:
                cur_idx_raw = random.randint(0, self.item_num - 1)

            use_mining_sample = False
            real_idx = 0

            # 检查是否使用 mining/jsonl 数据
            if self.mining_len > 0 and self.mining_overfit_dir is not None:
                if self.mining_mix_ratio is not None and self.mining_mix_ratio > 0:
                    # 混合模式：前 mining_len*mining_mix_ratio 为 normal，后 mining_len 为 mining
                    normal_data_end = int(self.mining_len * self.mining_mix_ratio)
                    if cur_idx_raw > normal_data_end:
                        use_mining_sample = True
                        real_idx = cur_idx_raw - normal_data_end - 1
                    else:
                        if self.normal_item_num > 0:
                            cur_idx_raw = random.randint(0, int(self.normal_item_num) - 1)
                else:
                    # 纯 mining 模式
                    use_mining_sample = True
                    real_idx = cur_idx_raw

            if use_mining_sample:
                # 从 jsonl 缓存加载
                real_idx = real_idx % self.mining_len
                item_path = osp.join(self.mining_overfit_dir, f"{real_idx}.json")
                if not os.path.exists(item_path):
                    raise FileNotFoundError(f"[Mining] item file missing: {item_path}")
                with open(item_path, "r") as f:
                    item = json.load(f)
                date = item.get("date") or item.get("Date")
                idx = item.get("idx") or item.get("frame_id")
                scene = item.get("scene", "unknown")
                if "file_path" in item and item["file_path"]:
                    data_label_path = item["file_path"]
                else:
                    data_label_path = self._build_data_label_path(
                        date, idx, use_tos=self.use_tos_data, use_datapro_pb=self.use_datapro_pb
                    )
                if "frame_label" in item and item["frame_label"]:
                    fl = item["frame_label"]
                    multi_scenes_label = fl if isinstance(fl, list) else [fl]
                else:
                    multi_scenes_label = item.get("multi_scenes_label", [scene])
                    if not isinstance(multi_scenes_label, list):
                        multi_scenes_label = [multi_scenes_label] if multi_scenes_label else [scene]
                frame_id = idx
                try:
                    res, data_feaNotEmb, route_cost = self.get_data(
                        data_label_path, date, idx, scene, multi_scenes_label, None
                    )
                except Exception as e:
                    print(f"[Mining] Exception in get_data({date}, {idx}, {scene}): {e}")
                    if self.flag_train and self.backuped_data is not None:
                        return copy.deepcopy(self.backuped_data)
                    raise e
                trainFlag_route = res.pop("trainFlag_route")
                cost_route = res.pop("cost_route")
            else:
                # 从 normal 二级索引加载
                index = bisect.bisect_right(self.n_frames, cur_idx_raw)
                scene = self.scenes[index]      # static, brake, etc
                extract_rate = self.scene_extract[scene]
                if index != 0:
                    idx = cur_idx_raw - self.n_frames[index - 1]
                else:
                    idx = cur_idx_raw

                if extract_rate >= 1:
                    idx = idx * extract_rate - np.random.randint(extract_rate)
                else:
                    idx = idx * extract_rate
                idx = int(max(0, idx))   # 避免出现负数

                try:
                    ret = load_secondary_index_entry(
                        scene=scene,
                        idx=idx,
                        prefetched_shm_dir=self.prefetched_shm_dir,
                        data_client=self.data_client,
                        scene_map=self.scene_map,
                    )
                    date, frame_id, *_ = ret
                    multi_scenes_label = load_frame_label(
                        self.prefetched_framelabel_shm_dir,
                        date,
                        frame_id,
                        self.flag_train,
                        self.data_client,
                        self.use_tos_data,
                    )
                    if not multi_scenes_label:
                        multi_scenes_label = [scene]
                    data_label_path = self._build_data_label_path(
                        date, frame_id, use_tos=self.use_tos_data, use_datapro_pb=self.use_datapro_pb
                    )

                    res, data_feaNotEmb, route_cost = self.get_data(
                        data_label_path, date, idx, scene, multi_scenes_label, None
                    )
                    trainFlag_route = res.pop("trainFlag_route")
                    cost_route = res.pop("cost_route")
                except Exception as e:
                    print(f"[RemoteClient] Exception in get_data({idx}, {scene}, {frame_id if 'frame_id' in locals() else 'None'}, {'tos' if self.use_tos_data else 's3'}): {e}")
                    if self.flag_train and self.backuped_data is not None:
                        return copy.deepcopy(self.backuped_data)
                    else:
                        raise e

            # 推理 / 测试阶段不过滤，直接使用当前样本
            if not self.flag_train:
                break

            # 训练阶段仅当 trainFlag_route 为 True 时才接受该样本
            try:
                flag_route_bool = bool(trainFlag_route.item() if hasattr(trainFlag_route, "item") else trainFlag_route)
            except Exception:
                flag_route_bool = True

            if flag_route_bool:
                break

            attempt += 1
            if attempt >= max_retry:
                # 多次尝试仍未采到 trainFlag_route=True 的样本，避免死循环，发出告警并使用当前样本
                print(f"[DlpDataset] WARNING: exceed max_retry={max_retry} when filtering trainFlag_route for idx_raw={idx_raw}, "
                      f"using last sample with trainFlag_route={trainFlag_route}")
                break

        if self.flag_train:
            res.update({'scene': scene})
            # 历史帧小于 self.agent_hist_step_threshold 的 agent，不做预测
            n_agent_valid_hist_steps = res['agent_time_mask'][:,:self.agent_hist_steps].sum(dim=1)
            agent_mask_tmp = n_agent_valid_hist_steps < self.agent_hist_step_threshold
            res['agent_time_mask'][agent_mask_tmp][self.agent_hist_steps+1:] = False           
            # 把所有的角度转到-pi ~ pi之间
            res["agent_status"][:,:,4] = self.wrap_to_pi(res["agent_status"][:,:,4])
            res["ego_future_status"][:,4] = self.wrap_to_pi(res["ego_future_status"][:,4])
            res["ego_future_status_fixed"][:,4] = self.wrap_to_pi(res["ego_future_status_fixed"][:,4])
            res, _ = self.ego_heading_aug._apply_aug(res, scene)
            res["navitopo_pts_ori"] = res["navitopo_pts"].clone()
            res["navitopo_mask_ori"] = res["navitopo_mask"].clone()
            # res = self.navi_aug.dropNavi(res)
            if self.retain_single_navi_recommend:
                res = retain_single_navitopo_recommend_lane(
                    res,
                    tol=self.retain_single_navi_tol,
                    apply_probability=self.retain_single_navi_prob,
                )
        else:
            res["navitopo_pts_ori"] = res["navitopo_pts"].clone()
            res["navitopo_mask_ori"] = res["navitopo_mask"].clone()
            # if torch.sum(res['navitopo_pts']) != 0:
            #     res['route_pts'] = torch.zeros_like(res['route_pts'])

        if self.use_guidance:
            # obs_costmap = self.build_path_costmap(res, res["ego_future_status_fixed"])
            navitopo_rs, navitopo_rs_mask = self.build_path_piecewise_navitopo(res, res["ego_future_status_fixed"][:self.future_steps_fixed], 
                                                                               res["ego_future_mask_fixed"][:self.future_steps_fixed],
                                                                               switch_confirm_len=10,# 增加切换连续多少个点都匹配，减少频繁切换
                                                                               deal_num = 5,  #三阶贝塞尔处理跳变的点数
                                                                               enable_full_projection=True)  # 多候选全量投影
            # navi_costmap = self.build_path_rewardmap(res, res["ego_future_status_fixed"])
            # search_costmap = self.build_path_search_costmap(res, res["ego_future_status_fixed"])
            res.update({
                # 'obs_costmap' : obs_costmap,
                'navitopo_rs': navitopo_rs,
                'navitopo_rs_mask': navitopo_rs_mask
                # 'navi_costmap' : navi_costmap,
                # 'search_costmap' : search_costmap,
            })

        # 当执行了单组增强，或原始推荐车道数为1时，使用 stitched navitopo 作为 path 训练GT。
        # 保留原始 ego_future_status_fixed 不变，新增 *_train_gt 字段给模型优先使用。
        rec_cnt = int(res.get("recommend_lane_count_before_aug", torch.tensor(0)).item())
        aug_applied = bool(res.get("flag_single_navi_aug_applied", torch.tensor(False)).item())
        use_stitched_as_gt = bool(aug_applied or rec_cnt == 1)
        res["use_stitched_path_gt"] = torch.tensor(use_stitched_as_gt)
        res["ego_future_status_fixed_train_gt"] = res["ego_future_status_fixed"].clone()
        res["ego_future_mask_fixed_train_gt"] = res["ego_future_mask_fixed"].clone()
        if use_stitched_as_gt and ("navitopo_rs" in res) and ("navitopo_rs_mask" in res):
            # navitopo_rs 现在是 (1+N, T, 2)，只用第 0 路（分段匹配）作为 stitched GT
            navitopo_rs_best = res["navitopo_rs"][0]          # (T, 2)
            navitopo_rs_mask_best = res["navitopo_rs_mask"][0]  # (T,)
            valid_len = min(
                res["ego_future_status_fixed_train_gt"].shape[0],
                navitopo_rs_best.shape[0],
                res["ego_future_mask_fixed_train_gt"].shape[0],
                navitopo_rs_mask_best.shape[0],
            )
            if valid_len > 0:
                stitched_mask = navitopo_rs_mask_best[:valid_len].bool()
                res["ego_future_status_fixed_train_gt"][:valid_len, :2][stitched_mask] = navitopo_rs_best[:valid_len][stitched_mask]
                res["ego_future_mask_fixed_train_gt"][:valid_len] = stitched_mask
                # 近端平滑：使用贝塞尔过渡，并按弧长严格 2m 等距重采样（平滑段 20m）。
                smooth_xy_np, smooth_mask_np = self.smooth_stitched_prefix_bezier(
                    res["ego_future_status_fixed_train_gt"][:valid_len, :2].detach().cpu().numpy(),
                    res["ego_future_mask_fixed_train_gt"][:valid_len].detach().cpu().numpy().astype(bool),
                    step=2.0,
                    blend_len_m=20.0,
                )
                # smooth_xy_np = res["ego_future_status_fixed_train_gt"][:valid_len, :2].detach().cpu().numpy()
                # smooth_mask_np = res["ego_future_mask_fixed_train_gt"][:valid_len].detach().cpu().numpy().astype(bool)
                res["ego_future_status_fixed_train_gt"][:valid_len, :2] = torch.from_numpy(smooth_xy_np).to(
                    device=res["ego_future_status_fixed_train_gt"].device,
                    dtype=res["ego_future_status_fixed_train_gt"].dtype,
                )
                res["ego_future_mask_fixed_train_gt"][:valid_len] = torch.from_numpy(smooth_mask_np).to(
                    device=res["ego_future_mask_fixed_train_gt"].device,
                    dtype=res["ego_future_mask_fixed_train_gt"].dtype,
                )

        # 预计算逐点车道线权重（基于最终的 train_gt，stitch + 平滑之后）
        train_gt_xy_np = res["ego_future_status_fixed_train_gt"][:, :2].detach().cpu().numpy()
        ll_pts_np = res["laneline_pts"].detach().cpu().numpy() if isinstance(res["laneline_pts"], torch.Tensor) else res["laneline_pts"]
        ll_attrs_np = res["laneline_attrs"].detach().cpu().numpy() if isinstance(res["laneline_attrs"], torch.Tensor) else res["laneline_attrs"]
        ll_mask_np = res["laneline_mask"].detach().cpu().numpy() if isinstance(res["laneline_mask"], torch.Tensor) else res["laneline_mask"]
        res["laneline_point_weight"] = torch.from_numpy(
            self.precompute_laneline_point_weight(train_gt_xy_np, ll_pts_np, ll_attrs_np, ll_mask_np)
        )

        # 网络异常时返回备份数据（必须与正常 return 的键一致，否则多 worker / 多 batch 拼 batch 时 plantf_collate_fn 会 KeyError）
        if self.backuped_data is None and bool(res['trainFlag']) == True:
            self.backuped_data = copy.deepcopy(
                {'model_input': res, 'data_feaNotEmb': data_feaNotEmb, 'date': date}
            )
            self.backuped_data['model_input'].update({'trainFlag': torch.tensor(False)})
     
        if self.flag_eval:
            # eval 模式，需要记录 date 和 idx
            return {'model_input' : res, 'date' : date, 'idx' : idx}
        else:
            return {'model_input' : res, 'data_feaNotEmb' : data_feaNotEmb, 'date': date}

    def get_data(self,
        data_label_path, 
        date, 
        idx,
        scene,
        multi_scenes_label,
        route_cost
    ):

        # 文件命中缓存，直接返回get_data结果
        cache_path = get_scene_cache_file(self.shm_json_cache_dir, scene, data_label_path, self.cache_target_scenes)
        cached = load_processed_cache(cache_path)
        if cached:
            cached_data, cached_fea, cached_route_cost = cached
            return cached_data, cached_fea, cached_route_cost

        res = self.data_client.load_proto(data_label_path)
        if route_cost is None:
            route_cost = res.route.route_cost[-1]
        # 1) 先以 numpy 形式读取/组装，减少重复的 torch 张量创建
        agent_attrs_np, agent_status_np, agent_time_mask_np, agent_id_list_np = self.read_data_agent(res.agents, date, idx)
        laneline_pts_np, laneline_attrs_np, laneline_mask_np = self.read_data_laneline(
            res.lanelines, res.cross, res.route, date, idx, route_cost
        )
        ego_curr_status_np = self.read_data_ego_curr_status(res.ego)
        ego_status_np = self.read_data_ego_status(res.ego_history_time, res.ego)
        (
            ego_future_status_np,
            ego_future_mask_np,
            ego_future_status_fixed_np,
            ego_future_mask_fixed_np,
            ego_future_status_fixed_full_np,
            ego_future_mask_fixed_full_np,
            trainFlag_ego,
        ) = self.read_label_ego_future_status(res.ego.future_traj, res.ego.future_fixed_dis)
        risky_lines_mask_np = self.extract_risky_lines_mask(laneline_pts_np, laneline_attrs_np, laneline_mask_np)
        ego_curr_status_np, egolight_np = self.data_correct(ego_curr_status_np, date, idx)

        # 2) 统一 numpy -> torch
        agent_attrs = torch.from_numpy(agent_attrs_np)
        agent_status = torch.from_numpy(agent_status_np)
        agent_time_mask = torch.from_numpy(agent_time_mask_np)
        agent_id_list = torch.from_numpy(agent_id_list_np)
        laneline_pts = torch.from_numpy(laneline_pts_np)
        laneline_attrs = torch.from_numpy(laneline_attrs_np)
        laneline_mask = torch.from_numpy(laneline_mask_np)
        risky_lines_mask = torch.from_numpy(risky_lines_mask_np)
        ego_curr_status = torch.from_numpy(ego_curr_status_np)
        # (7, T) -> (T, 7)，与 EgoEncoder_path / ego_heading_aug 的 [...,:2] 索引一致
        ego_status = torch.from_numpy(ego_status_np).transpose(0, 1).contiguous()
        ego_future_status = torch.from_numpy(ego_future_status_np)
        ego_future_mask = torch.from_numpy(ego_future_mask_np)
        ego_future_status_fixed = torch.from_numpy(ego_future_status_fixed_np)
        ego_future_mask_fixed = torch.from_numpy(ego_future_mask_fixed_np)
        ego_future_status_fixed_full = torch.from_numpy(ego_future_status_fixed_full_np)
        ego_future_mask_fixed_full = torch.from_numpy(ego_future_mask_fixed_full_np)

        data = {
            'agent_attrs': agent_attrs,   # (n_max_agent, 3)
            'agent_status': agent_status,  # (n_max_agent, T, 6)
            'agent_time_mask': agent_time_mask,
            'laneline_pts': laneline_pts,
            'laneline_attrs': laneline_attrs,
            'laneline_mask': laneline_mask,
            'ego_curr_status': ego_curr_status,
            'ego_status': ego_status,
            'agent_id': agent_id_list,
            'map_type': torch.tensor(res.ego.map_type),
            'risky_lines_mask': risky_lines_mask,
        }

        ego_timestamp = torch.tensor(res.ego_world_pos.timestamp_ns)
        data.update({'timestamp': ego_timestamp})
        data.update({
            'ego_future_status': ego_future_status,
            'ego_future_mask': ego_future_mask,
            'ego_future_status_fixed': ego_future_status_fixed,
            'ego_future_mask_fixed': ego_future_mask_fixed,
            'ego_future_status_fixed_full': ego_future_status_fixed_full,
            'ego_future_mask_fixed_full': ego_future_mask_fixed_full,
        })

        if self.open_eval_mode:
            timestamp_hi, timestamp_mi, timestamp_lo = self.get_timestamp(res.timestamp_ns)
            data.update({
                'timestamp_hi': timestamp_hi,
                'timestamp_mi': timestamp_mi,
                'timestamp_lo': timestamp_lo,
                'agent_id_list': agent_id_list,
            })

        self.infer_conut += 1
        self.debug_info = {
            "infer_conut": self.infer_conut,
            "timestamp" : ego_timestamp,
            "curr_data_path" : data_label_path,
            "scene" : scene,
            "multi_scenes_label" : multi_scenes_label
        }
        data.update({'infer_conut': torch.tensor(self.infer_conut, dtype=torch.float32)})

        # data_correct 已将 ego_curr_status 修正为关灯状态，真实灯光另存
        data_feaNotEmb = {
            'egolight_ori': torch.as_tensor(egolight_np),
        }
        navitopo_pts, navitopo_attrs, navitopo_mask, trainFlag_navi = self.read_data_navitopo(
            res.data_laneline_navi_topo, data, data_feaNotEmb, multi_scenes_label, date, idx
        )
        if data['map_type'] != 8:
            trainFlag_navi = False
        route_pts, trainFlag_route, cost_route, route_lane_attrs, route_lane_recommend_mask = self.read_data_route(
            res.route, date, idx, route_cost, ego_future_status_fixed, ego_future_mask_fixed
        )

        run_bad_sample_check = self.flag_train and (self.bad_sample_retry_mode != "off")
        if run_bad_sample_check:
            bad_hit, bad_reasons, bad_detail = check_bad_sample_for_retry(
                sample=res,
                ego_future_status_fixed=ego_future_status_fixed,
                ego_future_mask_fixed=ego_future_mask_fixed,
                threshold_m=self.bad_sample_route_dist_threshold_m,
                route_truncate_first_dist_thr_m=self.bad_sample_route_truncate_first_dist_thr_m,
                route_behind_m=self.bad_sample_route_behind_m,
                route_ahead_m=self.bad_sample_route_ahead_m,
            )
            if bad_hit:
                self.bad_sample_retry_hit_count += 1
                for rule_name, hit in bad_detail.items():
                    if hit:
                        self.bad_sample_retry_rule_hit_count[rule_name] += 1
                if self.bad_sample_retry_log_interval > 0 and (
                    self.bad_sample_retry_hit_count % self.bad_sample_retry_log_interval == 0
                ):
                    print(
                        f"[BadSampleRetry][{self.bad_sample_retry_mode}] "
                        f"hits={self.bad_sample_retry_hit_count} "
                        f"rule1={self.bad_sample_retry_rule_hit_count['rule1_navitopo_far']} "
                        f"rule2={self.bad_sample_retry_rule_hit_count['rule2_ego_off_route']} "
                        f"rule3={self.bad_sample_retry_rule_hit_count['rule3_future_far']} "
                        f"last={';'.join(bad_reasons[:2]) if bad_reasons else 'unknown'}"
                    )
                if self.bad_sample_retry_mode == "enforce":
                    trainFlag_route = False

        trainFlag = (trainFlag_navi or trainFlag_route) and trainFlag_ego
        # enable_occ=False: 只用占位 occ_map，不强制 trainFlag=False（否则 loss mask 全为 0，分母为 0 会产生 NaN）
        occ_polygons_pts = torch.zeros(self.max_n_occ_polygon, self.max_n_occ_polygon_point, 2)
        occ_polygons_attrs = torch.zeros(self.max_n_occ_polygon, self.max_n_occ_polygon_attr)
        occ_polygons_mask = torch.ones(self.max_n_occ_polygon, dtype=torch.float32)

        is_infer_need_occ = not self.flag_train
        is_train_need_occ = self._occ_init_check_ok and trainFlag
        if self.enable_occ and (is_infer_need_occ or is_train_need_occ):
            occ_map, occ_invalid_mask, occ_polygons_pts, occ_polygons_attrs, occ_polygons_mask = (
                self.process_occ(res.occ_info, ego_future_status)
            )
        else:
            # if self.enable_occ:
            #     print(f"Disabled OCC. is_infer_need_occ: {is_infer_need_occ}; is_train_need_occ: {is_train_need_occ}, "
            #           f"trainFlag: {trainFlag}, occ_init_check_ok: {self._occ_init_check_ok}")
            H, W = 120, 275  # occ 栅格大小
            occ_map = torch.ones(H, W)         # 占位的空地图
            occ_invalid_mask = True
            # 这里去掉occ对trainFlag的操作。具体改动等庆宏的数据简化
            # trainFlag = False

        # 读取mean和std
        data = self.get_mean_std(data, self.mean_std_dict)

        (
            navi_la_dist,
            navi_la_main,
            navi_la_assist,
            navi_la_valid,
            navi_la_enc_focus,
            navi_la_enc_ms_kind,
            navi_la_enc_ded_kind,
        ) = self.read_navi_link_actions_from_proto(res)

        navi_ms_valid, navi_ms_dist, navi_ms_kind = compute_navi_link_ms_nearest_numpy(
            navi_la_enc_focus, navi_la_enc_ms_kind, navi_la_dist, navi_la_valid
        )
        navi_ded_valid, navi_ded_dist, navi_ded_kind = compute_navi_link_ded_nearest_numpy(
            navi_la_enc_focus, navi_la_enc_ded_kind, navi_la_dist, navi_la_valid
        )

        data.update({
            'navitopo_pts' : navitopo_pts,
            'navitopo_attrs' : navitopo_attrs,
            'navitopo_mask' : navitopo_mask,
            'route_pts' : route_pts,
            'route_lane_attrs': route_lane_attrs,
            'route_lane_recommend_mask': route_lane_recommend_mask,
            'trainFlag' : torch.tensor(trainFlag),
            'trainFlag_route' : torch.tensor(trainFlag_route),
            'cost_route' : cost_route,
            'occ_map': occ_map,
            'occ_polygons_pts': occ_polygons_pts,
            'occ_polygons_attrs': occ_polygons_attrs,
            'occ_polygons_mask': occ_polygons_mask,
            'navi_link_actions_dist': torch.from_numpy(navi_la_dist),
            'navi_link_actions_main': torch.from_numpy(navi_la_main),
            'navi_link_actions_assist': torch.from_numpy(navi_la_assist),
            'navi_link_actions_mask': torch.from_numpy(navi_la_valid),
            # v1 离散标签（datasets/navi_link_encode_v1.py），暂不接入模型
            'navi_link_encode_focus': torch.from_numpy(navi_la_enc_focus),
            'navi_link_encode_ms_kind': torch.from_numpy(navi_la_enc_ms_kind),
            'navi_link_encode_ded_kind': torch.from_numpy(navi_la_enc_ded_kind),
            # 200m 内最近一条 MAIN_SIDE（ms_kind 左/右）供 path encoder 使用
            'navi_link_ms_valid': torch.from_numpy(navi_ms_valid),
            'navi_link_ms_dist': torch.from_numpy(navi_ms_dist),
            'navi_link_ms_kind': torch.from_numpy(navi_ms_kind),
            # 200m 内最近一条 DEDICATED（ded_kind 左/右转专用道）独立单槽，与 MAIN_SIDE 对称供对比
            'navi_link_ded_valid': torch.from_numpy(navi_ded_valid),
            'navi_link_ded_dist': torch.from_numpy(navi_ded_dist),
            'navi_link_ded_kind': torch.from_numpy(navi_ded_kind),
        })

        # 缓存target scene的数据
        save_processed_cache(cache_path, data, data_feaNotEmb, route_cost)

        return data, data_feaNotEmb, route_cost

    def read_navi_link_actions_from_proto(self, raw_res):
        """
        从 DLPRawData.data_sdmap.navi_link_actions 读取后续 link 的累计距离与主/辅动作（见 proto LinkAction）。
        仅保留前方 navi_link_actions_max_dist_m（默认 200m）内的 link；返回定长 numpy，mask 为 True 表示该槽位有效。
        同时写入 v1 离散编码（navi_link_encode_*），规则见 datasets/navi_link_encode_v1.py。
        """
        max_k = self.max_navi_link_actions
        max_d = self.navi_link_actions_max_dist_m
        dist = np.zeros(max_k, dtype=np.float32)
        main = np.zeros(max_k, dtype=np.int32)
        assist = np.zeros(max_k, dtype=np.int32)
        valid = np.zeros(max_k, dtype=np.bool_)
        enc_focus = np.zeros(max_k, dtype=np.int32)
        enc_ms_kind = np.zeros(max_k, dtype=np.int32)
        enc_ded_kind = np.zeros(max_k, dtype=np.int32)
        try:
            acts = raw_res.data_sdmap.navi_link_actions.link_actions
        except Exception:
            return dist, main, assist, valid, enc_focus, enc_ms_kind, enc_ded_kind
        slot = 0
        for la in acts:
            d = float(la.accumulated_dist)
            if d < 0.0 or d > max_d:
                continue
            if slot >= max_k:
                break
            ma = int(la.main_action_type)
            asst = int(la.assist_action_type)
            dist[slot] = d
            main[slot] = ma
            assist[slot] = asst
            valid[slot] = True
            fo, ms, de = encode_navi_link_action_v1(asst, ma)
            enc_focus[slot] = fo
            enc_ms_kind[slot] = ms
            enc_ded_kind[slot] = de
            slot += 1
        return dist, main, assist, valid, enc_focus, enc_ms_kind, enc_ded_kind

    def data_correct(self, ego_curr_status_np, date, idx):
        # ego_curr_status = [
        #     input_dict['v'],
        #     input_dict['yaw_rate'],
        #     traffic_light_curr_lane,     # 0: unknown, 1: invalid, 2: off, 3: green, 4: yellow, 5: red   
        #     ego_light,    # 1: off, 2: left, 3: right; 0 reserved for agent light
        # ]
        val = ego_curr_status_np[3]
        egolight_np = np.array(val, dtype=np.float32)
        ego_curr_status_np[3] = np.float32(1)
        return ego_curr_status_np, egolight_np

    def read_data_agent(self, data_agent, date, idx):
        start = 100 - self.agent_hist_steps
        end = 100 + 1
        T = self.agent_hist_steps + 1
        F = self.n_feature_agent_status
        delta_time = 0.1

        out_attr = np.zeros((self.max_n_agent, self.n_feature_agent_attr), dtype=np.float32)
        out_status = np.zeros((self.max_n_agent, T, F), dtype=np.float32)
        out_mask = np.zeros((self.max_n_agent, T), dtype=np.bool_)
        out_id = np.zeros((self.max_n_agent, 1), dtype=np.float32)

        n = 0
        for in_dict in data_agent:
            valid_mask = list(in_dict.valid_mask[start:end])
            if in_dict.id == 0 or not valid_mask[self.agent_hist_steps]:
                continue

            raw_cls = in_dict.cls
            mapped_cls = map_agent_cls[raw_cls]

            points = in_dict.pos.points[start:end]
            num_points = len(points)
            if num_points == 0:
                continue
            pos = np.fromiter(
                (v for p in points for v in (p.x, p.y)),
                dtype=np.float32,
                count=num_points * 2,
            ).reshape(num_points, 2)
            vm = np.fromiter(valid_mask, dtype=np.bool_, count=len(valid_mask))

            v = np.zeros((T, 2), dtype=np.float32)
            H = self.agent_hist_steps
            pos_past = pos[: H + 1]
            vm_past = vm[: H + 1]
            T_past = pos_past.shape[0]
            for i in range(T_past):
                if i == 0:
                    if vm_past[0] and vm_past[1]:
                        v[i] = (pos_past[1] - pos_past[0]) / delta_time
                elif i == T_past - 1:
                    if vm_past[-2] and vm_past[-1]:
                        v[i] = (pos_past[-1] - pos_past[-2]) / delta_time
                else:
                    if vm_past[i - 1] and vm_past[i] and vm_past[i + 1]:
                        v[i] = (pos_past[i + 1] - pos_past[i - 1]) / (2 * delta_time)

            yaw = np.fromiter(in_dict.heading[start:end], dtype=np.float32, count=len(in_dict.heading[start:end]))
            score = np.fromiter(in_dict.scores[start:end], dtype=np.float32, count=len(in_dict.scores[start:end]))
            status = np.concatenate(
                [pos, v, yaw.reshape(-1, 1), score.reshape(-1, 1)], axis=1
            ).astype(np.float32, copy=False)

            if not valid_mask[self.agent_hist_steps - 1]:
                valid_mask[self.agent_hist_steps - 1] = True
                status[self.agent_hist_steps - 1] = status[self.agent_hist_steps]
                status[self.agent_hist_steps - 1, 0] = status[self.agent_hist_steps, 0] - status[self.agent_hist_steps, 2] * 0.1
                status[self.agent_hist_steps - 1, 1] = status[self.agent_hist_steps, 1] - status[self.agent_hist_steps, 3] * 0.1

            _id_np = np.asarray(in_dict.id, dtype=np.float32).reshape(-1)

            out_attr[n, :] = np.asarray([in_dict.size_y, in_dict.size_x, mapped_cls], dtype=np.float32)
            out_status[n, : status.shape[0], :] = status
            out_mask[n, : len(valid_mask)] = np.asarray(valid_mask, dtype=np.bool_)
            out_id[n, 0] = float(_id_np[0])

            n += 1
            if n >= self.max_n_agent:
                break

        if n == 0:
            out_attr[0, :] = np.asarray([4.0, 2.0, 2.0], dtype=np.float32)
            tmp_status = np.zeros((T, F), dtype=np.float32)
            for i in range(T):
                tmp_status[i] = np.asarray([
                    -100.0 - 0.0 * (self.agent_hist_steps - i) * self.planning_interval,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.2,
                ], dtype=np.float32)
            tmp_mask = np.ones((T,), dtype=np.bool_)
            n_hist_invalid = self.agent_hist_steps - 1
            if n_hist_invalid > 0:
                tmp_mask[:n_hist_invalid] = False
            out_status[0, :, :] = tmp_status
            out_mask[0, :] = tmp_mask
            out_id[0, 0] = 0.0

        return out_attr, out_status, out_mask, out_id
    
    def check_road_quality(self, points):
        """
        检查路边点集的角度变化，过滤掉角度过大的路边
        参数:
            points: 点集，每个点是(x,y)坐标对，可以是元组、列表或具有x,y属性的对象
        返回:
            bool: True-角度正常，False-角度过大需要过滤
        """
        pts_np = None
        try:
            if isinstance(points, np.ndarray):
                pts_np = points
            elif torch.is_tensor(points):
                pts_np = points.detach().cpu().numpy()
            else:
                pts_np = np.asarray(points, dtype=np.float32)
        except Exception:
            pts_np = None

        if pts_np is None:
            return True

        pts_np = np.asarray(pts_np, dtype=np.float32)
        if pts_np.ndim == 2:
            pts_np = pts_np.reshape(1, -1, 2)
            squeeze_out = True
        elif pts_np.ndim == 3:
            squeeze_out = False
        else:
            return True

        N, P, C = pts_np.shape
        if C != 2:
            pts_np = pts_np.reshape(N, -1, 2)
            N, P, _ = pts_np.shape

        if P < 3:
            out = np.ones((N,), dtype=np.bool_)
            return bool(out[0]) if squeeze_out else out

        prev = pts_np[:, :-2, :]
        curr = pts_np[:, 1:-1, :]
        nxt = pts_np[:, 2:, :]

        cx = curr[:, :, 0]
        cy = curr[:, :, 1]
        pos_ok = (cx >= 0.0) & (cx <= 50.0) & (np.abs(cy) <= 10.0)

        v1 = curr - prev
        v2 = nxt - curr
        v1_len = np.linalg.norm(v1, axis=2)
        v2_len = np.linalg.norm(v2, axis=2)
        len_ok = (v1_len >= 1e-3) & (v2_len >= 1e-3)

        ok = pos_ok & len_ok
        dot = (v1[:, :, 0] * v2[:, :, 0]) + (v1[:, :, 1] * v2[:, :, 1])
        denom = v1_len * v2_len
        cos_theta = np.ones_like(denom, dtype=np.float32)
        cos_theta[ok] = (dot[ok] / denom[ok]).astype(np.float32, copy=False)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        cos_150 = np.float32(-0.8660254)
        bad = ok & (cos_theta < cos_150)
        out = ~np.any(bad, axis=1)

        return bool(out[0]) if squeeze_out else out

    def read_data_navitopo(self, data_navitopo, res, data_feaNotEmb, multi_scenes_label, date, idx):
        navitopo_pts, navitopo_attrs = [], []
        trainFlag = True
        for navitopo in data_navitopo:
            pts = torch.tensor([[p.x, p.y] for p in navitopo.pts_fixed_num.points])
            attr = torch.tensor(list(navitopo.navi_topo))
            # 如果最晚换道点距离给的值是0，可以默认999
            if attr[2]==0.0:
                attr[2] = 999

            disToCur_norm = torch.norm(pts, dim=1)  # [Npt]
            min_distances, min_distances_idx = torch.min(disToCur_norm, dim=0)#Npt
            if min_distances > 20:#太远了不要
                continue
            if pts.shape[0] < 80:#太短了不要
                continue
            idx = min_distances_idx
            if idx >= pts.shape[0] - 10:    #删掉超过minidx只有10个点的
               continue
            if self.laneline_fixed_pts - (pts.shape[0] - idx) > 0:
                pts = torch.cat([
                    pts[idx:idx + 200], 
                    pts[-1:].repeat(max(0, 200 - (pts.shape[0] - idx)), 1)])[:200]
            else:
                pts = pts[idx : idx + 200, :]
            navitopo_pts.append(pts)
            navitopo_attrs.append(attr)
        if len(navitopo_attrs)==0:
            print(f'error. There is empty navitopo_pts in {date}, {idx}')
            navitopo_pts.append(torch.zeros((200, 2), dtype=torch.float32))
            navitopo_attrs.append(torch.zeros((3), dtype=torch.float32))
            trainFlag = False

        navitopo_pts = torch.stack(navitopo_pts, 0)
        navitopo_attrs = torch.stack(navitopo_attrs, 0)

        navitopo_pts = [navitopo_pts[i] for i in range(navitopo_pts.shape[0])]
        navitopo_attrs = navitopo_attrs.tolist()
        #需要讨论，先写死了attr 3
        navitopo_attrs = torch.tensor(navitopo_attrs, dtype=torch.long).reshape(-1, 3)
        navitopo_pts = torch.stack(navitopo_pts).reshape(-1, self.laneline_fixed_pts, 2)
        #212212 排序
        #先选长的。最长的那个的-5m范围内的，按照离自车的距离排序
        #剩下的按照和自车距离排序
        #理论上只用一个navi的不用排序，但是为了简化逻辑，还是都排序把
        N = navitopo_pts.shape[0]

        closest_idx = torch.argmin(torch.norm(navitopo_pts, dim=2), dim=1)
        # 计算实际路径长度（逐点累积距离）
        diffs = navitopo_pts[:, 1:] - navitopo_pts[:, :-1]
        navi_lengths = torch.sum(torch.norm(diffs, dim=2), dim=1)
        closest_points = navitopo_pts[torch.arange(N), closest_idx]
        distances = torch.norm(closest_points, dim=1)
        
        max_length = torch.max(navi_lengths)
        group1_mask = navi_lengths >= (max_length - 5.0)
        
        sort_weights = torch.where(group1_mask, distances, distances + 1e6)
        
        sorted_indices = torch.argsort(sort_weights)
        navitopo_pts = navitopo_pts[sorted_indices]
        navitopo_attrs = navitopo_attrs[sorted_indices]
        #删navi,有可能全删了,全删了就给个0进去。全删了是跳过还是继续？
        allmask = self.del_navi_light(navitopo_pts, navitopo_attrs, res, data_feaNotEmb, multi_scenes_label)
        data_feaNotEmb.update({
            'navitopo_pts_ori' : navitopo_pts.clone(),
            'del_accLight_mask' : allmask.copy()
        })
        if not any(allmask):
            trainFlag = False
            navitopo_pts = torch.zeros(1, navitopo_pts.shape[1], navitopo_pts.shape[2], dtype = navitopo_pts.dtype)
            navitopo_attrs = torch.zeros(1, navitopo_attrs.shape[1], dtype = navitopo_attrs.dtype)
        else:
            navitopo_pts = navitopo_pts[allmask]
            navitopo_attrs = navitopo_attrs[allmask]
        
        #navi太远的不要
        #找到最近的
        path = res['ego_future_status_fixed'][:self.future_steps_fixed, :2]
        path_valid_mask = res['ego_future_mask_fixed'][:self.future_steps_fixed]
        path = path[path_valid_mask]  # 取有效的path

        # 若没有任何有效 future 点，直接退化处理，避免越界并标记为无效样本
        if path.numel() == 0:
            trainFlag = False
            navitopo_pts = torch.zeros(1, navitopo_pts.shape[1], navitopo_pts.shape[2], dtype=navitopo_pts.dtype)
            navitopo_attrs = torch.zeros(1, navitopo_attrs.shape[1], dtype=navitopo_attrs.dtype)
            navitopo_pts, navitopo_pts_mask = padding(navitopo_pts, 5)
            navitopo_attrs, navitopo_attrs_mask = padding(navitopo_attrs, 5)
            all_zero_mask = torch.all(navitopo_pts == 0.0, dim=(1, 2))
            navitopo_mask = navitopo_attrs_mask.clone()
            navitopo_mask[all_zero_mask] = 1.0
            return navitopo_pts, navitopo_attrs, navitopo_mask, trainFlag

        path_Last = path[-1, :].unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        disToLast_norm = torch.norm(navitopo_pts - path_Last, dim=2)  # [N, M]
        min_distances_last, _ = torch.min(disToLast_norm, dim=1)
        _, closest_navi_last = torch.min(min_distances_last, dim=0)
        row_idx = closest_navi_last
        last_k = min(10, path.shape[0]) # 取有效path的最后10个点，不足10个点，有多少点用多少点
        path_Last_10 = path[-last_k:]
        path_Last_10 = path_Last_10.unsqueeze(1)  # [num_points, 1, 2]
        #####如果用多个，就光算个距离就行。如果用单个，就直接把navi赋值成单个的
        if self.use_multiNavi:
            navitopo_pts_expanded = navitopo_pts[row_idx:row_idx+1, :, :]
        else:
            navitopo_pts = navitopo_pts[row_idx:row_idx+1, :, :]
            navitopo_attrs = navitopo_attrs[row_idx:row_idx+1, :]
            navitopo_pts_expanded = navitopo_pts      # [1, 200, 2]
        # 过滤数据帧：如果path终点到navi的投影距离大于3m，则不使用该数据帧
        if self.use_multiNavi and self.flag_train:
            # 计算path终点到每条有效navitop线的投影距离
            path_Last_point = path[-1]  # [2]
            
            # 计算所有线段：seg_start和seg_end
            seg_start = navitopo_pts[:, :-1, :]  # [N_valid, M-1, 2]
            seg_end = navitopo_pts[:, 1:, :]     # [N_valid, M-1, 2]
            
            # 计算线段向量
            seg_vec = seg_end - seg_start  # [N_valid, M-1, 2]
            seg_len_sq = torch.sum(seg_vec ** 2, dim=2)  # [N_valid, M-1]
            
            # 计算点到线段起点的向量（使用广播）
            point_vec = path_Last_point.unsqueeze(0).unsqueeze(0) - seg_start  # [N_valid, M-1, 2]
            
            # 计算投影参数t: t = dot(point_vec, seg_vec) / seg_len_sq
            t = torch.sum(point_vec * seg_vec, dim=2) / (seg_len_sq + 1e-8)  # [N_valid, M-1]
            t = torch.clamp(t, 0.0, 1.0)  # 限制在[0, 1]范围内
            
            # 计算投影点
            projection_points = seg_start + t.unsqueeze(2) * seg_vec  # [N_valid, M-1, 2]
            
            # 计算投影距离
            proj_distances = torch.norm(path_Last_point.unsqueeze(0).unsqueeze(0) - projection_points, dim=2)  # [N_valid, M-1]
            
            # 对每条线取最小投影距离
            projection_distances, _ = torch.min(proj_distances, dim=1)  # [N_valid]
            if min(projection_distances) > 3:
                trainFlag = False
                navitopo_pts = torch.zeros(1, navitopo_pts.shape[1], navitopo_pts.shape[2], dtype = navitopo_pts.dtype)
                navitopo_attrs = torch.zeros(1, navitopo_attrs.shape[1], dtype = navitopo_attrs.dtype)
        
        if self.flag_train:
            distances = torch.sqrt(torch.sum((path_Last_10 - navitopo_pts_expanded) ** 2, dim=2))  # [num_points, 200]
            min_distances, _ = torch.min(distances, dim=1)  # [num_points]
            avg_distance = torch.mean(min_distances)
            if avg_distance >= 3:
                trainFlag = False
                navitopo_pts = torch.zeros(1, navitopo_pts.shape[1], navitopo_pts.shape[2], dtype = navitopo_pts.dtype)
                navitopo_attrs = torch.zeros(1, navitopo_attrs.shape[1], dtype = navitopo_attrs.dtype)
                #print(f'error. navi so far in {date}, {idx}')

        #需要讨论。最大navitopo是5根
        navitopo_pts, navitopo_pts_mask = padding(navitopo_pts, 5)
        navitopo_attrs, navitopo_attrs_mask = padding(navitopo_attrs, 5)
        all_zero_mask = torch.all(navitopo_pts == 0.0, dim=(1, 2))
        navitopo_mask = navitopo_attrs_mask.clone()
        navitopo_mask[all_zero_mask] = 1.0
        # print("navitopo_pts", navitopo_pts.sum(dim=(1,2)), "all_zero_mask", all_zero_mask, "navitopo_attrs_mask", navitopo_attrs_mask, "navitopo_mask", navitopo_mask)
        return navitopo_pts, navitopo_attrs, navitopo_mask, trainFlag

    def read_data_route(self, route, date, idx, route_cost, ego_future_status_fixed, ego_future_mask_fixed):
        """
        读取导航路径及其 Amap laneInfo 相关特征。

        返回：
            route_pts: Tensor, 形状 (1, route_len, 2)
                - 经过插值与截断/补齐后的自车坐标系下导航路径点序列。
            trainFlag_route: bool
                - 当前样本是否使用 route 作为训练监督的标记。
            cost_route: float
                - route 与 ego 未来轨迹之间的最大距离，用于筛掉过远的导航路径。
            route_lane_attrs: Tensor, 形状 (1, max_lanes=8, 4)
                - 每根车道 4 维属性：[valid, recommended, lane_type_main, is_bus_lane]，
                  其中：
                    valid           : 是否存在该车道（background_lane != LANE_TYPE_NULL），0/1；
                    recommended     : 是否为推荐/高亮车道（foreground_lane != LANE_TYPE_NULL），0/1；
                    lane_type_main  : 车道主类型枚举值（直接使用 background_lane 的原始枚举值，包含 0=直行、255=NULL 等，是否有效由 valid 单独表示）；
                    is_bus_lane     : 是否公交专用车道（background_lane == LANE_TYPE_BUS），0/1。
            route_lane_recommend_mask: Tensor, 形状 (1, max_lanes=8)
                - 按车道序号（左→右）编码的推荐车道 mask，位置 i 为 1 表示第 i 根车道被推荐，否则为 0。
        """
        route_len = 200
        # 先构造 laneInfo 相关特征，保证即便 route 几何无效也有稳定输出
        route_lane_attrs_np, route_lane_recommend_mask_np = self.build_route_lane_features(route)

        if len(route.ego_routing.points) <= 0:
            route_pts = torch.zeros((1, route_len, 2))
            route_lane_attrs = torch.from_numpy(route_lane_attrs_np).reshape(1, 8, 4)
            route_lane_recommend_mask = torch.from_numpy(route_lane_recommend_mask_np).reshape(1, 8)
            return route_pts, False, 0, route_lane_attrs, route_lane_recommend_mask

        route_pts = [[route.ego_routing.points[0].x, route.ego_routing.points[0].y]]
        for item in route.ego_routing.points:
            if [item.x, item.y] != route_pts[-1]:
                route_pts.append([item.x, item.y])

        # 若第一点距离 (0, 0) 超过 50m，则基于投影进行截断：
        # 在自车到 route 最近的投影点身后保留 5m，身前保留 195m
        route_pts_np = np.array(route_pts, dtype=np.float32)
        if route_pts_np.shape[0] > 0:
            first_dist = np.linalg.norm(route_pts_np[0, :2])
            if first_dist > 50.0 and route_pts_np.shape[0] > 1:
                points = route_pts_np[:, :2]  # [N, 2]
                # 计算每段长度和累计距离
                diffs = np.diff(points, axis=0)               # [N-1, 2]
                seg_len = np.sqrt(np.sum(diffs ** 2, axis=1)) # [N-1]
                cumulative_dist = np.concatenate(
                    [np.array([0.0], dtype=np.float32), np.cumsum(seg_len)]
                )                                             # [N]
                total_length = float(cumulative_dist[-1])

                if total_length > 0:
                    # 寻找自车 (0,0) 到折线的最近投影点及其弧长位置 s_proj
                    origin = np.array([0.0, 0.0], dtype=np.float32)
                    best_dist2 = None
                    s_proj = 0.0
                    for i in range(points.shape[0] - 1):
                        p0 = points[i]
                        p1 = points[i + 1]
                        v = p1 - p0
                        denom = float(np.dot(v, v))
                        if denom <= 1e-6:
                            t = 0.0
                            proj = p0
                        else:
                            # 自车在原点，点到线段的投影参数 t
                            t = -float(np.dot(p0 - origin, v)) / denom
                            t = float(np.clip(t, 0.0, 1.0))
                            proj = p0 + t * v
                        dist2 = float(np.dot(proj - origin, proj - origin))
                        if best_dist2 is None or dist2 < best_dist2:
                            best_dist2 = dist2
                            s_proj = float(cumulative_dist[i] + t * seg_len[i])

                    # 以 s_proj 为中心，向后保留 5m，向前保留 195m
                    keep_behind = 5.0
                    keep_ahead = 195.0
                    s_start = max(0.0, s_proj - keep_behind)
                    s_end = min(total_length, s_proj + keep_ahead)

                    if s_end > s_start:
                        # 在弧长 [s_start, s_end] 上截断 polyline
                        interp_x = interp1d(
                            cumulative_dist, points[:, 0], kind="linear",
                            bounds_error=False, fill_value="extrapolate",
                        )
                        interp_y = interp1d(
                            cumulative_dist, points[:, 1], kind="linear",
                            bounds_error=False, fill_value="extrapolate",
                        )
                        start_pt = np.array(
                            [float(interp_x(s_start)), float(interp_y(s_start))],
                            dtype=np.float32,
                        )
                        end_pt = np.array(
                            [float(interp_x(s_end)), float(interp_y(s_end))],
                            dtype=np.float32,
                        )

                        truncated = [start_pt]
                        # 插入中间原始点（弧长在 (s_start, s_end) 之间）
                        for i in range(1, points.shape[0] - 1):
                            if s_start < float(cumulative_dist[i]) < s_end:
                                truncated.append(points[i])
                        truncated.append(end_pt)

                        truncated = np.stack(truncated, axis=0)  # [M, 2]

                        # 去掉相邻重复点
                        dedup = [truncated[0]]
                        for i in range(1, truncated.shape[0]):
                            if not np.allclose(truncated[i], dedup[-1]):
                                dedup.append(truncated[i])

                        if len(dedup) > 1:
                            dedup_arr = np.stack(dedup, axis=0)
                            route_pts = dedup_arr.tolist()

        pts_fixed_num = self.interpolate_route_points(route_pts, target_step=1.0)
        if pts_fixed_num is None:
            route_pts = torch.zeros((1, route_len, 2))
            route_lane_attrs = torch.from_numpy(route_lane_attrs_np).reshape(1, 8, 4)
            route_lane_recommend_mask = torch.from_numpy(route_lane_recommend_mask_np).reshape(1, 8)
            return route_pts, False, 0, route_lane_attrs, route_lane_recommend_mask
        pts_fixed_num = torch.tensor(pts_fixed_num, dtype=torch.float32)
        point_vector = pts_fixed_num[1:,:] - pts_fixed_num[:-1,:] 
        if ego_future_mask_fixed is None or torch.sum(ego_future_mask_fixed) == 0:
            route_pts = torch.zeros((1, route_len, 2))
            route_lane_attrs = torch.from_numpy(route_lane_attrs_np).reshape(1, 8, 4)
            route_lane_recommend_mask = torch.from_numpy(route_lane_recommend_mask_np).reshape(1, 8)
            return route_pts, False, 0, route_lane_attrs, route_lane_recommend_mask
        valid_gt_points = ego_future_status_fixed[ego_future_mask_fixed]  # (M, 2)
        distances = []
        for gt_pt in valid_gt_points[:, :2]:
            diff = pts_fixed_num - gt_pt               # (L, 2)
            dist = torch.norm(diff, dim=1)              # (L,)
            min_dist = torch.min(dist)
            distances.append(min_dist)
        max_dist = torch.max(torch.tensor(distances))
        if max_dist > self.route_cost_thre:
            # print(f'Route too far from gt trajectory in {date}, {idx}, max_dist={max_dist:.2f}')
            route_pts = torch.zeros((1, route_len, 2))
            route_lane_attrs = torch.from_numpy(route_lane_attrs_np).reshape(1, 8, 4)
            route_lane_recommend_mask = torch.from_numpy(route_lane_recommend_mask_np).reshape(1, 8)
            return route_pts, False, max_dist, route_lane_attrs, route_lane_recommend_mask
        point_vector_norm = torch.norm(point_vector, dim=-1)
        if (point_vector_norm == 0).any():
            print(f'Error. There are same pts in interpolated route in {date}, {idx}', )
            route_pts = torch.zeros((1, route_len, 2))
            route_lane_attrs = torch.from_numpy(route_lane_attrs_np).reshape(1, 8, 4)
            route_lane_recommend_mask = torch.from_numpy(route_lane_recommend_mask_np).reshape(1, 8)
            return route_pts, False, max_dist, route_lane_attrs, route_lane_recommend_mask
        elif torch.equal(pts_fixed_num[0], pts_fixed_num[-1]):
            print(f'Error. The start route pt equals end route pt in {date}, {idx}', )
            route_pts = torch.zeros((1, route_len, 2))
            route_lane_attrs = torch.from_numpy(route_lane_attrs_np).reshape(1, 8, 4)
            route_lane_recommend_mask = torch.from_numpy(route_lane_recommend_mask_np).reshape(1, 8)
            return route_pts, False, max_dist, route_lane_attrs, route_lane_recommend_mask
        else:
            if pts_fixed_num.shape[0] < route_len:
                pts_fixed_num = torch.cat([
                    pts_fixed_num, 
                    pts_fixed_num[-1:].repeat(max(0, route_len - (pts_fixed_num.shape[0])), 1)])[:route_len]
            else:
                pts_fixed_num = pts_fixed_num[:route_len, :]
            route_pts = pts_fixed_num[:route_len, :].unsqueeze(0)
            route_lane_attrs = torch.from_numpy(route_lane_attrs_np).reshape(1, 8, 4)
            route_lane_recommend_mask = torch.from_numpy(route_lane_recommend_mask_np).reshape(1, 8)
            return route_pts, True, max_dist, route_lane_attrs, route_lane_recommend_mask

    def interpolate_route_points(self, route_pts, target_step=1.0):
        """
        将路由点列表插值为1米1个点
        route_pts: 列表，包含[x, y]坐标点
        target_step: 目标步长（米），默认为1米
        """
        # 将列表转换为numpy数组
        points = np.array(route_pts)
        
        # 计算累积距离
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_dist[-1]
        
        # 生成1米间隔的目标距离点
        target_distances = np.arange(0, total_length, target_step)
        
        # 线性插值
        interp_x = interp1d(cumulative_dist, points[:, 0], kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(cumulative_dist, points[:, 1], kind='linear',
                        bounds_error=False, fill_value='extrapolate')
        
        interp_x_vals = interp_x(target_distances)
        interp_y_vals = interp_y(target_distances)
        
        # 组合结果
        result = np.column_stack([interp_x_vals, interp_y_vals])
        
        # 确保包含终点（如果步长不能正好到达终点）
        if len(result) == 0 or not np.allclose(result[-1], points[-1], atol=0.1):
            result = np.vstack([result, points[-1]])
        
        return result

    def get_mean_std(self, data, mean_std_dict):
        '''
        output: traj.shape[6, 25] # 5s 5hz
                path.shape[6,40] # 80m 2m 40step
        含义:ego_future_status_x_mean,
            ego_future_status_x_std,
            ego_future_status_y_mean,
            ego_future_status_y_std,
            ego_delta_yaw_mean,
            ego_delta_yaw_std

            ego_future_status_fixed_x_mean,
            ego_future_status_fixed_x_std,
            ego_future_status_fixed_y_mean,
            ego_future_status_fixed_y_std,
            ego_fixed_delta_yaw_mean,
            ego_fixed_delta_yaw_std,
        '''
        collected_mean_std = []
        collected_mean_std_fixed = []
        key_list = [
            'ego_future_status_x',
            'ego_future_status_y',
            'ego_future_status_fixed_x',
            'ego_future_status_fixed_y',
            'ego_delta_yaw',
            'ego_fixed_delta_yaw',
        ]
        for key, value in mean_std_dict.items():
            for pattern in key_list:
                if pattern in key and 'fixed' not in key:
                    collected_mean_std.append(np.asarray(value['mean'], dtype=np.float32))
                    collected_mean_std.append(np.asarray(value['std'], dtype=np.float32))
                    break
                elif pattern in key and 'fixed' in key:
                    collected_mean_std_fixed.append(np.asarray(value['mean'], dtype=np.float32))
                    collected_mean_std_fixed.append(np.asarray(value['std'], dtype=np.float32))
                    break
        if collected_mean_std:
            mean_std_np = np.stack(collected_mean_std, axis=0)
            data['mean_std'] = torch.from_numpy(mean_std_np) if isinstance(mean_std_np, np.ndarray) else mean_std_np
        if collected_mean_std_fixed:
            mean_std_fixed_np = np.stack(collected_mean_std_fixed, axis=0)
            data['mean_std_fixed'] = torch.from_numpy(mean_std_fixed_np) if isinstance(mean_std_fixed_np, np.ndarray) else mean_std_fixed_np
        return data

    def read_data_laneline(self, data_laneline, roaditem, route, date, idx, route_cost):
        P = int(self.laneline_fixed_pts)
        max_n = int(self.max_n_laneline)
        n_attr = int(self.n_feature_laneline_attr)

        stop_n = len(roaditem.stoplines)
        max_candidates = max(int(len(data_laneline)) + int(stop_n), 1)
        pts_buf = np.zeros((max_candidates, P, 2), dtype=np.float32)
        attrs_buf = np.zeros((max_candidates, n_attr), dtype=np.int64)
        n = 0

        def _fill_pts_to_fixed(dst: np.ndarray, pts_any) -> None:
            pts_np = np.asarray(pts_any, dtype=np.float32)
            if pts_np.ndim != 2 or pts_np.shape[1] != 2:
                pts_np = pts_np.reshape(-1, 2).astype(np.float32, copy=False)
            m = int(pts_np.shape[0])
            if m <= 0:
                dst[:, :] = 0.0
                return
            if m >= P:
                dst[:, :] = pts_np[:P, :]
                return
            dst[:m, :] = pts_np
            dst[m:, :] = pts_np[m - 1:m, :]

        for laneline in data_laneline:
            if n >= max_candidates:
                break
            typ = laneline.type
            if typ == "centerline":
                if not self.use_centerline:
                    continue
                points = laneline.sampled_points.points
                num_points = len(points)
                if num_points == 0:
                    continue
                pts = np.fromiter(
                    (v for p in points for v in (p.x, p.y)),
                    dtype=np.float32,
                    count=num_points * 2,
                ).reshape(num_points, 2)
                if np.max(pts[:, 0]) < -30:
                    continue
                _fill_pts_to_fixed(pts_buf[n, :, :], pts)
                attrs_buf[n, 0] = 3
                attrs_buf[n, 1] = 3
                attrs_buf[n, 2] = 5
                n += 1

            elif typ == "laneline":
                if self.use_new_laneline:
                    points = laneline.sampled_points.points
                    num_points = len(points)
                    if num_points == 0:
                        continue
                    pts = np.fromiter(
                        (v for p in points for v in (p.x, p.y)),
                        dtype=np.float32,
                        count=num_points * 2,
                    ).reshape(num_points, 2)
                else:
                    start_x, end_x = laneline.start_x, laneline.end_x
                    c0, c1, c2, c3 = laneline.c0, laneline.c1, laneline.c2, laneline.c3
                    xs = np.linspace(start_x, end_x, P, dtype=np.float32)
                    ys = (c0 + c1 * xs + c2 * (xs ** 2) + c3 * (xs ** 3)).astype(np.float32)
                    pts = np.stack([xs, ys], axis=1).astype(np.float32, copy=False)

                edge_type = laneline.edge_type
                lane_type_raw = laneline.laneline_type
                if (1 <= edge_type <= 5 or 10 <= lane_type_raw <= 11) and self.check_road_quality(pts) is False:
                    continue

                _fill_pts_to_fixed(pts_buf[n, :, :], pts)

                color = laneline.lane_color
                if color == 2:
                    laneline_color = 1
                elif color == 3:
                    laneline_color = 2
                else:
                    laneline_color = 3

                if 1 <= edge_type <= 5 or 10 <= lane_type_raw <= 11:
                    laneline_type = 1
                else:
                    laneline_type = 2

                if lane_type_raw in [2, 5, 7, 9]:
                    laneline_style = 1
                elif lane_type_raw in [1, 6, 8]:
                    laneline_style = 2
                elif lane_type_raw in [3]:
                    laneline_style = 3
                elif lane_type_raw in [4]:
                    laneline_style = 4
                else:
                    laneline_style = 5

                attrs_buf[n, 0] = int(laneline_color)
                attrs_buf[n, 1] = int(laneline_type)
                attrs_buf[n, 2] = int(laneline_style)
                n += 1

        if len(roaditem.stoplines) > 0:
            for stopline in roaditem.stoplines:
                if n >= max_candidates:
                    break
                pt0 = stopline.points[0]
                pt1 = stopline.points[-1]   # 0429 在排查部署问题时发现stopline取点有问题
                x0, y0 = pt0.x, pt0.y
                x1, y1 = pt1.x, pt1.y
                start_dis = float(np.linalg.norm(np.asarray([x0, y0], dtype=np.float32)))
                end_dis = float(np.linalg.norm(np.asarray([x1, y1], dtype=np.float32)))
                if min(x0, x1) < -30 or start_dis > 150 or end_dis > 150:
                    continue
                if (x0 - x1) ** 2 + (y0 - y1) ** 2 > 50 * 50:
                    print(f'error. There is an abnormally long stop line in {date}, {idx}')
                    continue
                pts_fixed_num = self.interpolate_straight_line_simple(np.array([x0, x1]), np.array([y0, y1]), target_step=1.0)
                pts = np.asarray(pts_fixed_num, dtype=np.float32).reshape(-1, 2)
                _fill_pts_to_fixed(pts_buf[n, :, :], pts)
                attrs_buf[n, 0] = 1
                attrs_buf[n, 1] = 4
                attrs_buf[n, 2] = 5
                n += 1

        if n == 0:
            print(f'error. There is empty laneline_pts in {date}, {idx}')
            pts_buf[0, :, :] = 0.0
            attrs_buf[0, 0] = 0
            attrs_buf[0, 1] = 0
            attrs_buf[0, 2] = 0
            n = 1

        pts_all = pts_buf[:n, :, :]
        attrs_all = attrs_buf[:n, :]

        distances = np.linalg.norm(pts_all, axis=2)
        min_distances = distances.min(axis=1)
        order = np.argsort(min_distances)
        pts_all = pts_all[order]
        attrs_all = attrs_all[order]

        polygon_vec = pts_all[:, -1, :] - pts_all[:, 0, :]
        polygon_norm = np.linalg.norm(polygon_vec, axis=-1)
        # if (polygon_norm == 0).any():
        #     print(f'error. There is polygon that end pt is same as start pt in {date}, {idx}')

        nn = int(pts_all.shape[0])
        mask = np.zeros((max_n,), dtype=np.float32)
        out_pts = np.zeros((max_n, P, 2), dtype=np.float32)
        out_attrs = np.zeros((max_n, n_attr), dtype=np.int64)
        m = min(nn, max_n)
        out_pts[:m] = pts_all[:m]
        out_attrs[:m] = attrs_all[:m]
        if nn < max_n:
            mask[nn:] = 1.0

        return out_pts, out_attrs, mask

    def build_route_lane_features(self, route):
        """
        根据 RouteData.amap_lane_info 构造逐车道的导航高亮特征。
        输出:
            route_lane_attrs_np: (max_lanes, 4) -> [valid, recommended, lane_type_main, is_bus_lane]
            route_lane_recommend_mask_np: (max_lanes,) -> 是否推荐车道的mask

        AmapLaneType 枚举对应关系（高德导航车道类型）：
            0   LANE_TYPE_AHEAD
            1   LANE_TYPE_LEFT
            2   LANE_TYPE_AHEAD_LEFT
            3   LANE_TYPE_RIGHT
            4   LANE_TYPE_AHEAD_RIGHT
            5   LANE_TYPE_LU_TURN
            6   LANE_TYPE_LEFT_RIGHT
            7   LANE_TYPE_AHEAD_LEFT_RIGHT
            8   LANE_TYPE_RU_TURN
            9   LANE_TYPE_AHEAD_LU_TURN
            10  LANE_TYPE_AHEAD_RU_TURN
            11  LANE_TYPE_LEFT_LU_TURN
            12  LANE_TYPE_RIGHT_RU_TURN
            13  LANE_TYPE_LEFT_IN_AHEAD
            14  LANE_TYPE_LEFT_IN_LEFT_LU_TURN
            15  LANE_TYPE_RESERVED
            16  LANE_TYPE_AHEAD_LEFT_LU_TURN
            17  LANE_TYPE_RIGHT_RU_TURN_EX
            18  LANE_TYPE_LEFT_RU_TURN
            19  LANE_TYPE_AHEAD_RIGHT_RU_TURN
            20  LANE_TYPE_LEFT_LU_TURN_EX
            21  LANE_TYPE_BUS
            22  LANE_TYPE_EMPTY
            23  LANE_TYPE_VARIABLE
            255 LANE_TYPE_NULL（无效 / 未定义）
        """
        max_lanes = 8
        attr_dim = 4
        route_lane_attrs_np = np.zeros((max_lanes, attr_dim), dtype=np.float32)
        # invalid lane 的 mask 统一置为 -1，仅 valid lane 使用 0/1 表示是否推荐
        route_lane_recommend_mask_np = np.full((max_lanes,), -1.0, dtype=np.float32)

        # 允许通过配置整体关闭 laneInfo 特征
        if not getattr(self, "use_route_laneinfo", True):
            return route_lane_attrs_np, route_lane_recommend_mask_np

        if route is None or not hasattr(route, "amap_lane_info"):
            return route_lane_attrs_np, route_lane_recommend_mask_np

        lane_info = route.amap_lane_info
        # protobuf 中可能是默认实例，此时 repeated 字段长度为 0
        if lane_info is None:
            return route_lane_attrs_np, route_lane_recommend_mask_np

        foreground = list(lane_info.foreground_lane)
        background = list(lane_info.background_lane)

        LANE_TYPE_NULL = 255
        LANE_TYPE_BUS = 21

        for i in range(max_lanes):
            bg_type = background[i] if i < len(background) else LANE_TYPE_NULL
            fg_type = foreground[i] if i < len(foreground) else LANE_TYPE_NULL

            valid = float(bg_type != LANE_TYPE_NULL)
            recommended = float(fg_type != LANE_TYPE_NULL)
            # lane_type_main 使用原始枚举值（包括 0=LANE_TYPE_AHEAD，255=LANE_TYPE_NULL），
            # 是否有效由 valid 这一维单独表示，避免将 “直行” 与 “无效车道” 混淆到同一个数值 0。
            lane_type_main = float(bg_type)
            is_bus_lane = float(bg_type == LANE_TYPE_BUS) if valid > 0.0 else 0.0

            route_lane_attrs_np[i, 0] = valid
            route_lane_attrs_np[i, 1] = recommended
            route_lane_attrs_np[i, 2] = lane_type_main
            route_lane_attrs_np[i, 3] = is_bus_lane

            route_lane_recommend_mask_np[i] = recommended if valid > 0.0 else -1.0

        return route_lane_attrs_np, route_lane_recommend_mask_np

    @staticmethod
    def precompute_laneline_point_weight(traj_xy_np, laneline_pts_np, laneline_attrs_np,
                                         laneline_mask_np, dist_threshold=3.0, stride=5):
        """
        在 CPU/numpy 端预计算逐点车道线权重。

        规则（取距离 < dist_threshold 的最近 2 条车道线/curb）：
          - 没有车道线 → 0.5
          - 都是虚线   → 1.0
          - 1条实线/curb → 1.5
          - 2条实线/curb → 2.0

        Args:
            traj_xy_np:        (T, 2)   float32  轨迹 xy
            laneline_pts_np:   (N, P, 2) float32  车道线采样点
            laneline_attrs_np: (N, 3)   int       [color, laneline_type, laneline_style]
            laneline_mask_np:  (N,)     float32   0=valid, 1=invalid
            dist_threshold:    float    距离阈值 (m)
            stride:            int      对 P 维降采样步长（200→20），加速距离计算
        Returns:
            point_weight:      (T,) float32
        """
        T = traj_xy_np.shape[0]
        N = laneline_pts_np.shape[0]

        ltype  = laneline_attrs_np[:, 1]   # (N,)
        lstyle = laneline_attrs_np[:, 2]   # (N,)

        is_boundary      = (ltype == 1) | (ltype == 2)
        is_solid_or_curb = (ltype == 1) | ((ltype == 2) & (lstyle == 2))
        valid_lane       = (laneline_mask_np < 0.5) & is_boundary

        valid_idx = np.where(valid_lane)[0]
        if valid_idx.size == 0:
            return np.zeros(T, dtype=np.float32)

        lane_sub = laneline_pts_np[valid_idx, ::stride, :]          # (V, P', 2)
        is_sc_sub = is_solid_or_curb[valid_idx]                      # (V,)

        # (T, 1, 1, 2) - (1, V, P', 2) -> 平方距离
        dist2_threshold = dist_threshold * dist_threshold
        diff = traj_xy_np[:, None, None, :] - lane_sub[None, :, :, :]   # (T, V, P', 2)
        dist2 = (diff * diff).sum(axis=-1)                              # (T, V, P')
        min_dist2 = dist2.min(axis=-1)                                  # (T, V)

        V = valid_idx.size
        point_weight = np.full(T, 0.5, dtype=np.float32)

        for t in range(T):
            dists = min_dist2[t]                   # (V,)
            within_mask = dists < dist2_threshold
            if not within_mask.any():
                continue
            order = np.argsort(dists)
            count = 0
            n_solid = 0
            for idx in order:
                if count >= 2:
                    break
                if not within_mask[idx]:
                    break
                count += 1
                if is_sc_sub[idx]:
                    n_solid += 1
            if count == 0:
                continue
            if n_solid == 0:
                point_weight[t] = 1.0
            elif n_solid == 1:
                point_weight[t] = 1.5
            else:
                point_weight[t] = 2.0

        return point_weight

    def read_data_curb(self, laneline_pts, laneline_attrs, laneline_mask):
        """
        从 laneline_pts 中提取路沿点
        返回 shape: (N, P, 2)，不处理 padding
        """
        laneline_pts_np = laneline_pts.cpu().numpy() if isinstance(laneline_pts, torch.Tensor) else laneline_pts
        laneline_attrs_np = laneline_attrs.cpu().numpy() if isinstance(laneline_attrs, torch.Tensor) else laneline_attrs

        curb_pts = []
        num_lines = laneline_pts_np.shape[0]

        for i in range(num_lines):
            if laneline_mask[i]:
                continue
            pts = laneline_pts_np[i]          # (P, 2)
            color, laneline_type, laneline_style = laneline_attrs_np[i]
            if laneline_type == 1:
                curb_pts.append(pts)
        if len(curb_pts) == 0:
            return np.zeros((0, laneline_pts_np.shape[1], 2), dtype=laneline_pts_np.dtype)
        return np.stack(curb_pts, axis=0)     # (N, P, 2)


    def merge_centerline(self, centerlines):
        # centerlines is a non-empty list, each element is a list containing 20 (x, y)
        n = len(centerlines)
        start_pts = {}
        for i in range(n):
            start_pts[centerlines[i][0]] = i
        next_idxes = [None] * n
        for i in range(n):
            end_pt = centerlines[i][-1]
            if end_pt in start_pts:
                next_idxes[i] = start_pts[end_pt]
        merged_centerlines = []
        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
            tmp = centerlines[i]
            next_idx = next_idxes[i]
            while next_idx:
                tmp += centerlines[next_idx][1:]
                visited[next_idx] = True
                next_idx = next_idxes[next_idx]
            merged_centerlines.append(tmp)
        return merged_centerlines

    def interpolate_straight_line_simple(self, xs, ys, target_step=1.0):
        """
        简洁版本的直线插值
        """
        start_point = np.array([xs[0], ys[0]])
        end_point = np.array([xs[1], ys[1]])
        
        # 计算直线总长度和方向
        line_vector = end_point - start_point
        total_length = np.linalg.norm(line_vector)
        
        if total_length == 0:
            result = np.array([start_point])
        else:
            direction = line_vector / total_length
            
            # 计算插值点
            distances = np.arange(0, total_length, target_step)
            points = [start_point + direction * d for d in distances]
            
            # 添加终点
            if not np.array_equal(points[-1], end_point):
                points.append(end_point)
                
            result = np.array(points)
        
        return result

    def read_data_ego_status(self, ego_hist_status, ego_curr_status):
        # (n, 7)
        data = [[p.pos_x, p.pos_y, p.v_x, p.v_y, p.yaw, p.yaw_rate, bool(p.mask)] for p in ego_hist_status]
        # 添加当前 ego 状态
        data.append([0.0, 0.0, ego_curr_status.speed, 0.0, 0.0, ego_curr_status.angular_velocity, True])
        # (7, n)
        x = np.asarray(data, dtype=np.float32).T
        # (7, self.ego_hist_steps + 1)
        return x[:, -(self.ego_hist_steps + 1):]

    def read_data_ego_curr_status(self, ego_data):
        """
        ego_curr_status = [
            v,                      # 车速
            yaw_rate,               # 这里保留占位，当前填 0.0
            traffic_light_curr_lane,# 0: unknown, 1: invalid, 2: off, 3: green, 4: yellow, 5: red
            ego_light,              # 1: off, 2: left, 3: right; 0 reserved for agent light
        ]
        """
        if len(ego_data.egolane_traffic_lights) == 0:
            traffic_light_curr_lane = 1
        else:
            traffic_light_curr_lane = ego_data.egolane_traffic_lights[0].status
            mp_traffic_light = {
                6: 3,
                7: 4,
                8: 5,
                9: 1,
            }
            if traffic_light_curr_lane in mp_traffic_light:
                traffic_light_curr_lane = mp_traffic_light[traffic_light_curr_lane]
        ego_light = ego_data.ego_light + 1 if ego_data.ego_light >= 0 else 1     # 1: off, 2: left, 3: right; 0 reserved for agent light
        x = np.asarray([
            ego_data.speed,
            0.0,
            traffic_light_curr_lane,
            ego_light,
        ], dtype=np.float32)
        return x

    def visualize_polygons_real(self, ego_gt_pos, polygons, polygons_attrs, out_dir, mode, xlim=(-10, 100), ylim=(-30, 30)):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if isinstance(polygons, torch.Tensor):
            polygons = polygons.detach().cpu().numpy()

        plt.figure(figsize=(8, 6))
        for idx, (polygon, attr) in enumerate(zip(polygons, polygons_attrs)):
            cy, cx, tp, area, angle = attr
            color = (random.random(), random.random(), random.random())
            plt.scatter(cx, cy, color=color, s=25, marker="x", zorder=5,)
            plt.text(polygon[0, 0], polygon[0, 1], str(idx), fontsize=8, color=color, zorder=6) # start point
            plt.text(cx, cy, f"type={tp}\narea={area:.2f}", fontsize=8, color=color, verticalalignment="bottom", horizontalalignment="left", zorder=6,)
            plt.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=1)

            arrow_len = 2.0  # 根据你的坐标尺度自行调
            dx = arrow_len * np.cos(angle)
            dy = arrow_len * np.sin(angle)
            plt.arrow(cx, cy, dx, dy, color=color, width=0.2, head_width=1.0, head_length=1.5, zorder=6)
            if "occ_polygons_resampled" in mode:
                plt.scatter(polygon[:, 0], polygon[:, 1], marker="x", c="black", s=20, linewidths=1, zorder=7,)

        if ego_gt_pos is not None:
            plt.scatter(ego_gt_pos[:, 0], ego_gt_pos[:, 1], c='red',s=12, marker='o', label='trajectory')

        timestamp = self.format_ns_timestamp(self.debug_info['timestamp'])
        plt.axis('equal')
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{timestamp}_{self.infer_conut}_{mode}.png"), dpi=200)
        plt.close()

    def visualize_map(self, ego_gt_pos, occ_map, out_dir, mode, resolution):
        img = occ_map.detach().cpu().numpy() if isinstance(occ_map, torch.Tensor) else occ_map
        height, width = img.shape
        xs = ego_gt_pos.cpu().detach().numpy()[:,0] / resolution / width      
        ys = ego_gt_pos.cpu().detach().numpy()[:,1] / resolution / height + 0.5
        xs_px = np.round(xs * width).astype(np.int32)
        ys_px = np.round(ys * height).astype(np.int32)

        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        # for x, y in zip(xs_px, ys_px):
        #     cv2.circle(img_bgr, (x, y), radius=2, color=(0,0,255), thickness=-1)
        # img_bgr = cv2.flip(img_bgr, 0)
        timestamp = self.format_ns_timestamp(self.debug_info['timestamp'])
        cv2.imwrite(os.path.join(out_dir, f"{timestamp}_{self.infer_conut}_{mode}.png"), img_bgr)

    def visualize_polygons(self, ego_gt_pos, polygons, polygons_attrs, out_dir, mode, height=120, width=275, resolution=0.4):
        if isinstance(polygons, torch.Tensor):
            polygons = polygons.detach().cpu().numpy()

        img = np.ones((height, width), dtype=np.uint8) * 255
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 绘制ego轨迹点
        xs = ego_gt_pos.cpu().detach().numpy()[:,0] / resolution / width      
        ys = ego_gt_pos.cpu().detach().numpy()[:,1] / resolution / height + 0.5
        xs_px = np.round(xs * width).astype(np.int32)
        ys_px = np.round(ys * height).astype(np.int32)
        # for x, y in zip(xs_px, ys_px):
        #     cv2.circle(img_bgr, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

        for idx, (poly, attr) in enumerate(zip(polygons, polygons_attrs)):
            if len(poly) == 0: 
                continue 

            pts = poly.copy()
            pts = pts.astype(np.int32)
            # clamp to image bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1) 
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1) 
            pts_cv = pts.reshape(-1, 1, 2) 

            # 随机颜色绘制 polygon
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.polylines(img_bgr, [pts_cv], isClosed=True, color=color, thickness=2)

            start_pt, end_pt = tuple(pts_cv[0, 0]), tuple(pts_cv[-1, 0])
            cv2.putText(img_bgr, str(idx), start_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # 绘制 resampled 点
            if "occ_polygons_resampled" in mode:
                for x, y in pts:
                    cv2.line(img_bgr, (x-2, y-2), (x+2, y+2), color=(0,0,0), thickness=1)
                    cv2.line(img_bgr, (x-2, y+2), (x+2, y-2), color=(0,0,0), thickness=1)

            # ===== 绘制中心点并显示 tp 和 area =====
            cy, cx, tp, area, angle = attr
            cx = int(np.clip(cx, 0, width-1))
            cy = int(np.clip(cy, 0, height-1))
            cv2.circle(img_bgr, (cx, cy), radius=3, color=color, thickness=-1)
            text = f"{tp:.1f}, {area:.1f}"
            cv2.putText(img_bgr, text, (cx+2, cy-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            timestamp = self.format_ns_timestamp(self.debug_info['timestamp'])
        cv2.imwrite(os.path.join(out_dir, f"{timestamp}_{self.infer_conut}_{mode}.png"), img_bgr)

    def compute_contours(
        self, bin_occ_map: np.ndarray, origin_occ_map: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, float, float]]]:
        """
        Compute contours from a binary occupancy map.

        Args:
            bin_occ_map (np.ndarray): Binary occupancy map, shape (H, W).
            origin_occ_map (np.ndarray): Original occupancy map, shape (H, W).

        Returns:
            Tuple:
                - raw_polygons (List[np.ndarray]):
                    List of contour polygons, each with shape (N, 2).
                - raw_polygons_attr (List[Tuple[int, int, float, float]]):
                    Each tuple is (cy, cx, tp, area, angle).
        """
        binary = ((bin_occ_map < 0.5) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        raw_polygons = []
        raw_polygons_attr = []
        H, W = bin_occ_map.shape
        W_m1, H_m1 = W - 1, H - 1
        RAD_CONST = np.pi / 180.0
        HALF_PI = np.pi / 2.0
        for cnt in contours:
            pts = cnt[:, 0].astype(np.float32)

            M = cv2.moments(cnt)
            m00 = M["m00"]
            if m00 != 0:
                cx = int(M["m10"] / m00)
                cy = int(M["m01"] / m00)
            else:
                # 使用快速 mean 替代
                mean_pts = pts.mean(axis=0)
                cx, cy = int(mean_pts[0]), int(mean_pts[1])

            if cx < 0: cx = 0
            elif cx > W_m1: cx = W_m1
            if cy < 0: cy = 0
            elif cy > H_m1: cy = H_m1
            tp = origin_occ_map[cy, cx]
            area = cv2.contourArea(cnt)

            (rcx, rcy), (w, h), angle = cv2.minAreaRect(cnt)
            theta_pix = angle * RAD_CONST
            if w < h: theta_pix += HALF_PI

            raw_polygons.append(pts)
            raw_polygons_attr.append([cy, cx, tp, area, theta_pix])
        return raw_polygons, raw_polygons_attr

    def convert_polygon_to_real(self, polygon_img, attrs_img, resolution, i0, j0):
        polygon_real = img2real(polygon_img, resolution=resolution, i0=i0, j0=j0)
        attrs_real = []
        for cy, cx, tp, area_pixel, theta_pix in attrs_img:
            x = (cx - j0) * resolution
            y = (i0 - cy) * resolution
            area_real = area_pixel * (resolution ** 2)
            theta_real = -theta_pix
            if np.cos(theta_real) < 0: theta_real += np.pi  # 与自车前进的方向保持90°内
            attrs_real.append([y, x, tp, area_real, theta_real])
        return polygon_real, attrs_real

    def vis_costmap(self, costmap, costmap_config, save_path, other_pts_list=None, mode=""):
        import matplotlib.pyplot as plt
        """
        costmap: torch.Tensor or np.ndarray, shape (H, W)
        save_path: str, e.g. 'costmap_vehicle_frame.png'
        """
        # ---------- to numpy ----------
        costmap = costmap[0] if costmap.ndim == 3 else costmap  # (T, H, W) -> (H, W)
        if isinstance(costmap, torch.Tensor):
            costmap_np = costmap.detach().cpu().numpy()
        else:
            costmap_np = np.asarray(costmap)
        H, W = costmap_np.shape

        # ---------- build pixel grid ----------
        i = np.arange(H)
        j = np.arange(W)
        ii, jj = np.meshgrid(i, j, indexing="ij")      # (H, W)
        # ---------- pixel coords -> real coords (via img2real) ----------
        # img2real expects (..., 2) with (j, i)
        img_pts = np.stack([jj, ii], axis=-1)          # (H, W, 2)
        real_pts = img2real(img_pts, resolution=costmap_config["resolution"], 
                            i0=costmap_config["ego_i0"], j0=costmap_config["ego_j0"]) # (H, W, 2)

        x = real_pts[..., 0]                            # (H, W)
        y = real_pts[..., 1]                            # (H, W)
        # ---------- plot ----------
        plt.figure(figsize=(12, 4))

        # pcolormesh 保证真实坐标轴
        pcm = plt.pcolormesh(x,y,costmap_np,shading="auto",cmap="coolwarm_r")
        plt.colorbar(pcm, label="SDF (m)")
        if other_pts_list is not None:
            cmap = plt.get_cmap('tab20')
            for i, other_pts in enumerate(other_pts_list):
                if other_pts.ndim == 3: other_pts = other_pts.reshape(-1, 2)
                if isinstance(other_pts, torch.Tensor): other_pts = other_pts.detach().cpu().numpy()
                plt.scatter(other_pts[:, 0], other_pts[:, 1], c=[cmap(i % cmap.N)], s=1, marker='o')

        plt.title("Costmap SDF in Vehicle Frame")
        plt.axis("equal")

        # ---------- save & close ----------
        save_path = os.path.join(save_path, f"{mode}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def world_to_img(self, pt, resolution, costmap_width, costmap_length):
        """
        pt: [x, y] in meters (ego coord)
        return: [col, row] in image
        """
        x, y = pt
        col = int(x / resolution)
        row = int(costmap_width / 2 - y / resolution)

        if 0 <= col < costmap_length and 0 <= row < costmap_width:
            return [col, row]
        else:
            return None

    def select_closest_navitopo(self, navitopo_pts: torch.Tensor, navitopo_mask: torch.Tensor):
        """
        navitopo_pts:  (N, P, 2)
        navitopo_mask: (N,)  True=invalid, False=valid

        return:
            closest_idx: 原始 N 维中的 index
        """
        valid_mask = ~navitopo_mask.bool()
        if valid_mask.sum() == 0:
            return None
        device = navitopo_pts.device

        # 1. 取出有效的 navitopo 线
        valid_pts = navitopo_pts[valid_mask]                        # (M, P, 2)
        if valid_pts.numel() == 0:
            raise ValueError("No valid navitopo lines")

        # 2. 计算每条线到 (0,0) 的最小距离
        origin = torch.zeros(2, device=device)                      # (2,)
        dist2 = torch.sum((valid_pts - origin) ** 2, dim=-1)        # (M, P)
        min_dist2_per_line = dist2.min(dim=1).values                # (M,)

        # 3. 找最近的一条
        best_local_idx = min_dist2_per_line.argmin()                # 标量

        # 4. 映射回原始 index
        original_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        closest_idx = original_indices[best_local_idx]
        return closest_idx

    def _fill_polygon(self, mask, corners, w, h, res):
        img_poly = []
        for p in corners:
            pix = self.world_to_img(p, res, w, h)
            if pix is not None:
                img_poly.append(pix)

        if len(img_poly) >= 3:
            cv2.fillConvexPoly(mask, np.array(img_poly, dtype=np.int32), 0)

    def _draw_curb_mask_double_side(self, drivable_area_mask, curb_pts_np, costmap_width, costmap_length, costmap_resolution, curb_half_width=0.5):
        num_lines = curb_pts_np.shape[0]

        for i in range(num_lines):
            pts = curb_pts_np[i]
            num_pts = pts.shape[0]
            if num_pts < 2: continue
            for j in range(1, num_pts):
                pt1, pt2 = pts[j - 1], pts[j]
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                length = np.hypot(dx, dy)
                if length < 1e-6: continue
                cos_theta = dx / length
                sin_theta = dy / length
                nx, ny = sin_theta, -cos_theta
                corners = np.array([
                    [pt1[0] + curb_half_width * nx, pt1[1] + curb_half_width * ny],
                    [pt2[0] + curb_half_width * nx, pt2[1] + curb_half_width * ny],
                    [pt2[0] - curb_half_width * nx, pt2[1] - curb_half_width * ny],
                    [pt1[0] - curb_half_width * nx, pt1[1] - curb_half_width * ny],
                ])
                self._fill_polygon(drivable_area_mask, corners, costmap_width, costmap_length, costmap_resolution)

    def _draw_curb_mask_single_side(self, drivable_area_mask, curb_pts_np, nav_pts, costmap_width, costmap_length, costmap_resolution, curb_half_width=0.5):
        num_lines = curb_pts_np.shape[0]

        for i in range(num_lines):
            pts = curb_pts_np[i]
            num_pts = pts.shape[0]

            if num_pts < 2: continue

            # ---------- 用最近点确定方向 ----------
            diff = nav_pts[:, None, :] - pts[None, :, :]
            distances = np.linalg.norm(diff, axis=-1)

            tmp = np.argmin(distances)
            row = tmp // distances.shape[1]
            col = tmp % distances.shape[1]
            if col == num_pts - 1: col -= 1

            pt1, pt2 = pts[col], pts[col + 1]
            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            length = np.hypot(dx, dy)

            if length < 1e-6: continue

            cos_theta = dx / length
            sin_theta = dy / length
            nx, ny = sin_theta, -cos_theta

            nav_pt = nav_pts[row]
            test_pos = np.array([pt1[0] + curb_half_width * nx, pt1[1] + curb_half_width * ny])
            test_neg = np.array([pt1[0] - curb_half_width * nx, pt1[1] - curb_half_width * ny])
            factor = 1 if np.sum((test_pos - nav_pt) ** 2) > np.sum((test_neg - nav_pt) ** 2) else -1

            # ---------- 沿 curb 画 mask ----------
            for j in range(1, num_pts):
                pt1, pt2 = pts[j - 1], pts[j]
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                length = np.hypot(dx, dy)
                if length < 1e-6: continue

                cos_theta, sin_theta = dx / length, dy / length
                corners = np.array([
                    [pt1[0], pt1[1]],
                    [pt2[0], pt2[1]],
                    [pt2[0] + factor * curb_half_width * sin_theta,
                    pt2[1] - factor * curb_half_width * cos_theta],
                    [pt1[0] + factor * curb_half_width * sin_theta,
                    pt1[1] - factor * curb_half_width * cos_theta],
                ])
                self._fill_polygon(drivable_area_mask, corners, costmap_width, costmap_length, costmap_resolution)

    def build_curb_drivable_area_mask(self, curb_pts, navitopo_pts, navitopo_mask):
        costmap_width, costmap_length, costmap_resolution = self.costmap_config['width'], self.costmap_config['length'], self.costmap_config['resolution']
        drivable_area_mask = np.ones((costmap_width, costmap_length), dtype=np.uint8)

        curb_pts_np = curb_pts.cpu().numpy() if isinstance(curb_pts, torch.Tensor) else curb_pts
        # closest_idx = self.select_closest_navitopo(navitopo_pts, navitopo_mask)
        valid_idx = (~navitopo_mask.bool()).nonzero(as_tuple=True)[0]
        closest_idx = valid_idx[0].item() if valid_idx.numel() > 0 else None    # only use first valid navitopo

        nav_pts = torch.zeros_like(navitopo_pts[0])
        if closest_idx is None:
            self._draw_curb_mask_double_side(
                drivable_area_mask,
                curb_pts_np,
                costmap_width,
                costmap_length,
                costmap_resolution,
                curb_half_width=0.5,
            )
        else:
            nav_pts = navitopo_pts[closest_idx]
            nav_pts = nav_pts.cpu().numpy() if isinstance(nav_pts, torch.Tensor) else nav_pts
            self._draw_curb_mask_single_side(
                drivable_area_mask,
                curb_pts_np,
                nav_pts,
                costmap_width,
                costmap_length,
                costmap_resolution,
                curb_half_width=0.5,
            )

        debug = False
        if debug:
            info = f"{self.format_ns_timestamp(self.debug_info['timestamp'])}_{self.infer_conut}"
            save_path = "/mnt/afs/liuzhaoyang1/diffusion_codes/dif_17_guidance/tmp/costmap"
            self.vis_costmap(drivable_area_mask, self.costmap_config, save_path, other_pts_list=[curb_pts_np, nav_pts], mode=info+"_curb_drivable_area_mask")
            # ndarray2json(curb_pts_debug, save_path, mode=info+"_curb_pts")
            import pdb; pdb.set_trace()
        return drivable_area_mask

    def build_occ_map_drivable_area_mask(self, occ_map):
        if isinstance(occ_map, torch.Tensor):
            occ_map = occ_map.detach().cpu().numpy()

        self.costmap_config['resolution'] = self.occmap_config['resolution']
        if self.occmap_config['resolution'] == 0.4:                         # (120, 275)
            costmap_region = occ_map[10:110, 25:275]                        # (100, 250)
            self.costmap_config['width'] = costmap_region.shape[0]
            self.costmap_config['length'] = costmap_region.shape[1]
            self.costmap_config['ego_i0'] = self.costmap_config['width'] // 2
            self.costmap_config['ego_j0'] = 0
        elif self.occmap_config['resolution'] == 0.2:                       # (240, 550)
            occ_map = occ_map.reshape(self.occmap_config['height']//2, 2, self.occmap_config['width']//2, 2)
            # 规则: 只有全是1才是安全，否则都是障碍. 下采样: 
            occ_map = np.where((occ_map == 1).all(axis=(1, 3)), 1, 0)       # (120, 275)
            self.costmap_config['resolution'] = self.costmap_config['resolution'] * 2
            costmap_region = occ_map[10:110, 25:275]                        # (100, 250)
            self.costmap_config['width'] = costmap_region.shape[0]
            self.costmap_config['length'] = costmap_region.shape[1]
            self.costmap_config['ego_i0'] = self.costmap_config['width'] // 2
            self.costmap_config['ego_j0'] = 0

        costmap_width, costmap_length, costmap_resolution = self.costmap_config['width'], self.costmap_config['length'], self.costmap_config['resolution']
        drivable_area_mask = np.ones((costmap_width, costmap_length), dtype=np.uint8)

        non_drivable = costmap_region != 1
        construction_area = costmap_region == 3

        dilation_radius = 2                                 # 2 piex * 0.4 m/piex = 0.8m
        structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
        construction_dilated = ndimage.binary_dilation(construction_area, structure=structure)
        final_non_drivable = non_drivable | construction_dilated
        drivable_area_mask[final_non_drivable] = 0

        debug = False
        if debug:
            # --- before dilated ---
            info = f"{self.format_ns_timestamp(self.debug_info['timestamp'])}_{self.infer_conut}"
            save_path = "/mnt/afs/liuzhaoyang1/diffusion_codes/dif_82/tmp/costmap"
            self.vis_costmap(drivable_area_mask, self.costmap_config, save_path, mode=info+"_occ_drivable_area_mask_before_dilated")

            # --- after dilated ---
            drivable_area_mask_origin = np.ones((costmap_width, costmap_length), dtype=np.uint8)
            drivable_area_mask_origin[non_drivable] = 0
            self.vis_costmap(drivable_area_mask_origin, self.costmap_config, save_path, mode=info+"_occ_drivable_area_mask_after_dilated")
            # ndarray2json(occ_region, save_path, mode=info+"_occ_region")
            import pdb; pdb.set_trace()

        return drivable_area_mask

    def build_static_agents_drivable_area_mask(self, data):
        costmap_width, costmap_length, costmap_resolution = self.costmap_config['width'], self.costmap_config['length'], self.costmap_config['resolution']
        drivable_area_mask = np.ones((costmap_width, costmap_length), dtype=np.uint8)
        
        agent_attrs = data['agent_attrs']  # (n_max_agent, 3): [size_y, size_x, cls]
        agent_status = data['agent_status']  # (n_max_agent, T, 6): [x, y, vx, vy, yaw, score]
        agent_time_mask = data['agent_time_mask']  # (n_max_agent, T): True表示有效
        
        if isinstance(agent_attrs, torch.Tensor):
            agent_attrs = agent_attrs.detach().cpu().numpy()
        if isinstance(agent_status, torch.Tensor):
            agent_status = agent_status.detach().cpu().numpy()
        if isinstance(agent_time_mask, torch.Tensor):
            agent_time_mask = agent_time_mask.detach().cpu().numpy()
        
        curr_time_idx = self.agent_hist_steps
        
        n_agents = agent_attrs.shape[0]
        for i in range(n_agents):
            # 过滤1：检查当前时刻是否有效
            if not agent_time_mask[i, curr_time_idx]:
                continue
            
            pos = agent_status[i, curr_time_idx, :2]
            yaw = agent_status[i, curr_time_idx, 4]
            size_y = agent_attrs[i, 0]
            size_x = agent_attrs[i, 1]
            
            # 过滤2：检查位置是否在合理范围内（避免处理过远或过近的agents）
            if np.linalg.norm(pos) > 100 or np.linalg.norm(pos) < 1.0:
                continue
            
            # 过滤3：过滤动态agent，只保留静态agent
            valid_mask = agent_time_mask[i, :]
            start_idx = curr_time_idx
            while start_idx >= 0 and valid_mask[start_idx]:
                start_idx -= 1
            start_idx += 1
            end_idx = curr_time_idx + 1
            window_indices = np.arange(start_idx, end_idx)
            
            if len(window_indices) > 0:
                valid_velocities = agent_status[i, window_indices, 2:4]  # [vx, vy]
                mean_speed = np.mean(np.linalg.norm(valid_velocities, axis=1))
                if mean_speed > 0.5:
                    continue
                
                # if len(window_indices) > 1:
                #     first_pos = agent_status[i, window_indices[0], :2]
                #     last_pos = agent_status[i, window_indices[-1], :2]
                #     pos_diff = np.linalg.norm(last_pos - first_pos)
                #     if pos_diff > 0.5:
                #         continue
            
            half_length = size_y / 2.0  # 半长（前进方向）
            half_width = size_x / 2.0   # 半宽（横向）
            if (agent_attrs[i, 2] == 1.0) or (agent_attrs[i, 2] == 3.0):  # VRU体积膨胀
                half_length += 1.4
                half_width += 1.4

            local_corners = np.array([
                [-half_length, -half_width],
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width],
            ], dtype=np.float32)
            
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ], dtype=np.float32)
            
            world_corners = (rotation_matrix @ local_corners.T).T + pos
            
            img_poly = []
            for corner in world_corners:
                pix = self.world_to_img(corner, costmap_resolution, costmap_width, costmap_length)
                if pix is not None:
                    img_poly.append(pix)
            
            # 至少需要3个点才能构成多边形
            if len(img_poly) < 3:
                continue
            
            # 填充多边形为不可行驶区域（设为0）
            img_poly = np.array(img_poly, dtype=np.int32)
            cv2.fillConvexPoly(drivable_area_mask, img_poly, 0)
        
        debug = False
        if debug and np.any(drivable_area_mask == 0):
            import matplotlib.pyplot as plt
            H, W = drivable_area_mask.shape
            ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            real_pts = img2real(np.stack([jj, ii], axis=-1), resolution=costmap_resolution, i0=50, j0=0)  # (H, W, 2)
            x_map, y_map = real_pts[..., 0], real_pts[..., 1]
            pcm = plt.pcolormesh(x_map, y_map, drivable_area_mask, shading="auto", cmap="coolwarm_r")
            
            # valid_agents_pos = []
            # for i in range(n_agents):
            #     if agent_time_mask[i, curr_time_idx]:
            #         pos = agent_status[i, curr_time_idx, :2]
            #         if np.linalg.norm(pos) <= 150:
            #             valid_agents_pos.append(pos)
            # if len(valid_agents_pos) > 0:
            #     valid_agents_pos = np.array(valid_agents_pos)
            #     plt.scatter(valid_agents_pos[:, 0], valid_agents_pos[:, 1], c='r', s=5, marker='s', label='agents')
            
            plt.colorbar(pcm, label="Drivable Area")
            plt.axis("equal")
            timestamp = self.format_ns_timestamp(self.debug_info['timestamp'])
            save_path = "/iag_ad_vepfs_volc/iag_ad_vepfs_volc/linchengqi/pl_diffusion_models/work_dirs/static_only"
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"{timestamp}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()
        
        return drivable_area_mask

    def build_path_costmap(self, data, ego_future_status_fixed):
        future_steps = self.future_steps_fixed // self.planning_interval_fixed
        costmap_width, costmap_length, resolution = self.costmap_config['width'], self.costmap_config['length'], self.costmap_config['resolution']
        curb_pts_np = self.read_data_curb(data['laneline_pts'], data['laneline_attrs'], data['laneline_mask'])
        drivable_area_mask = np.ones((costmap_width, costmap_length), dtype=np.uint8)
        curb_drivable_area_mask = self.build_curb_drivable_area_mask(curb_pts_np, data['navitopo_pts'], data['navitopo_mask'])
        occ_map_drivable_area_mask = self.build_occ_map_drivable_area_mask(data['occ_map'])
        # static_agents_drivable_area_mask = self.build_static_agents_drivable_area_mask(data)
        drivable_area_mask = (curb_drivable_area_mask & occ_map_drivable_area_mask).astype(np.uint8)
        # drivable_area_mask = (occ_map_drivable_area_mask).astype(np.uint8)    # only occ
        # drivable_area_mask = (curb_drivable_area_mask).astype(np.uint8)       # only curb
        # drivable_area_mask = static_agents_drivable_area_mask.astype(np.uint8)  # only static agents

        dist_out = ndimage.distance_transform_edt(drivable_area_mask == 1)                  # (H, W)
        dist_in = ndimage.distance_transform_edt(drivable_area_mask == 0)                   # (H, W)
        drivable_area_sdf = (dist_out - dist_in) * resolution                               # (H, W): more value, more safe

        debug = False
        if debug:
            ego_gt_pos = ego_future_status_fixed[..., :2]
            ego_gt_pos = None
            info = f"{self.format_ns_timestamp(self.debug_info['timestamp'])}_{self.infer_conut}"
            save_path = "/mnt/afs/liuzhaoyang1/diffusion_codes/dif_17_guidance/tmp/costmap"
            os.makedirs(save_path, exist_ok=True)
            self.vis_costmap(curb_drivable_area_mask, self.costmap_config, save_path, ego_gt_pos, mode=info+"_curb_drivable_area_mask")
            self.vis_costmap(occ_map_drivable_area_mask, self.costmap_config, save_path, ego_gt_pos, mode=info+"_occ_map_drivable_area_mask")
            self.vis_costmap(drivable_area_mask, self.costmap_config, save_path, ego_gt_pos, mode=info+"_drivable_area_mask")
            self.vis_costmap(drivable_area_sdf, self.costmap_config, save_path, ego_gt_pos, mode=info+"_drivable_area_sdf")
            # ndarray2json(curb_drivable_area_mask, save_path, mode=info+"_curb_drivable_area_mask")
            # ndarray2json(occ_map_drivable_area_mask, save_path, mode=info+"_occ_map_drivable_area_mask")
            # ndarray2json(drivable_area_mask, save_path, mode=info+"_drivable_area_mask")
            # ndarray2json(drivable_area_sdf, save_path, mode=info+"_drivable_area_sdf")
            import pdb; pdb.set_trace()
        drivable_area_sdf = np.repeat(drivable_area_sdf[None, :, :], future_steps, axis=0)    # (T, 100, 250)
        return torch.tensor(drivable_area_sdf, dtype=torch.float32)

    def build_navitopo_drivable_area_mask(self, navitopo_pts, navitopo_mask, half_width = 0.3):
        if isinstance(navitopo_pts, torch.Tensor):
            navitopo_pts = navitopo_pts.detach().cpu().numpy()

        costmap_width, costmap_length, costmap_resolution = self.costmap_config['width'], self.costmap_config['length'], self.costmap_config['resolution']
        drivable_area_mask = np.zeros((costmap_width, costmap_length), dtype=np.uint8)

        valid_idx = (~navitopo_mask.bool()).nonzero(as_tuple=True)[0]
        closest_idx = valid_idx[0].item() if valid_idx.numel() > 0 else None    # only use first valid navitopo
        if closest_idx is None:
            return drivable_area_mask

        pts = navitopo_pts[closest_idx]     # (P, 2)
        navitopo_pts_debug = []
        if pts.shape[0] < 2: 
            return drivable_area_mask
        for j in range(1, pts.shape[0]):
            navitopo_pts_debug.append(pts[j])
            pt1, pt2 = pts[j - 1], pts[j]
            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            seg_len = np.hypot(dx, dy)
            if seg_len < 1e-6: 
                continue
            cos_theta, sin_theta = dx / seg_len, dy / seg_len
            nx, ny = sin_theta, -cos_theta
            corners = np.array([
                [pt1[0] + half_width * nx, pt1[1] + half_width * ny],
                [pt2[0] + half_width * nx, pt2[1] + half_width * ny],
                [pt2[0] - half_width * nx, pt2[1] - half_width * ny],
                [pt1[0] - half_width * nx, pt1[1] - half_width * ny],
            ])
            img_poly = []
            for p in corners:
                pix = self.world_to_img(p, costmap_resolution, costmap_width, costmap_length)
                if pix is not None:
                    img_poly.append(pix)
            if len(img_poly) < 3:
                continue
            img_poly = np.array(img_poly, dtype=np.int32)
            cv2.fillConvexPoly(drivable_area_mask, img_poly, 1)

        if len(navitopo_pts_debug) > 0:
            navitopo_pts_debug = np.vstack(navitopo_pts_debug)

        debug = False
        if debug and len(navitopo_pts_debug) > 0:
            info = f"{self.format_ns_timestamp(self.debug_info['timestamp'])}_{self.infer_conut}"
            save_path = "/mnt/afs/liuzhaoyang1/diffusion_codes/dif_guidance_0106/tmp/rewardmap"
            self.vis_costmap(drivable_area_mask, self.costmap_config, save_path, mode=info+"_navi_drivable_area_mask")
            import pdb; pdb.set_trace()
        return drivable_area_mask

    def build_path_rewardmap(self, data, ego_future_status_fixed):
        future_steps = self.future_steps_fixed // self.planning_interval_fixed
        width, length, resolution = self.costmap_config['width'], self.costmap_config['length'], self.costmap_config['resolution']
        drivable_area_mask = np.zeros((width, length), dtype=np.uint8)

        navitopo_drivable_area_mask = self.build_navitopo_drivable_area_mask(data['navitopo_pts'], data['navitopo_mask'])
        drivable_area_mask = (navitopo_drivable_area_mask).astype(np.uint8)    # only navitopo

        dist_out = ndimage.distance_transform_edt(drivable_area_mask == 1)                  # (H, W)
        dist_in = ndimage.distance_transform_edt(drivable_area_mask == 0)                   # (H, W)
        drivable_area_sdf = (dist_out - dist_in) * resolution                               # (H, W): more value, more safe

        debug = False
        if debug:
            info = f"{self.format_ns_timestamp(self.debug_info['timestamp'])}_{self.infer_conut}"
            ego_gt_pos = ego_future_status_fixed[..., :2]
            save_path = "/mnt/afs/liuzhaoyang1/diffusion_codes/dif_guidance_0106/tmp/rewardmap"
            self.vis_costmap(navitopo_drivable_area_mask, self.costmap_config, save_path, other_pts_list=[ego_gt_pos], mode=info+"_navi_drivable_area_mask")
            self.vis_costmap(drivable_area_mask, self.costmap_config, save_path, other_pts_list=[ego_gt_pos], mode=info+"_drivable_area_mask")
            self.vis_costmap(drivable_area_sdf, self.costmap_config, save_path, other_pts_list=[ego_gt_pos], mode=info+"_drivable_area_sdf")
            import pdb; pdb.set_trace()
        drivable_area_sdf = np.repeat(drivable_area_sdf[None, :, :], future_steps, axis=0)    # (T, 100, 250)
        return torch.tensor(drivable_area_sdf, dtype=torch.float32)


    def assign_navi_along_path(
        self,
        path,  # (T, 2)
        navitopo_pts,  # (N, P, 2)
        navitopo_mask,  # (N,)
        switch_confirm_len=3,
        min_stay_len=5,  # 切换后至少保持的长度
    ):
        T = path.shape[0]
        N = navitopo_pts.shape[0]

        best_idx = np.full(T, -1, dtype=np.int32)

        def point_to_polyline_proj_dist(points, polyline):
            """
            points:   (K, 2)
            polyline: (P, 2)
            return:   (K,)
            """
            seg_start = polyline[:-1]
            seg_end = polyline[1:]
            seg_vec = seg_end - seg_start
            seg_len_sq = np.sum(seg_vec**2, axis=1)

            p_vec = points[:, None, :] - seg_start[None, :, :]
            t = np.sum(p_vec * seg_vec[None, :, :], axis=2) / (seg_len_sq[None, :] + 1e-8)
            t = np.clip(t, 0.0, 1.0)

            proj = seg_start[None, :, :] + t[..., None] * seg_vec[None, :, :]
            dist = np.linalg.norm(points[:, None, :] - proj, axis=2)

            return dist.min(axis=1)

        last_idx = -1
        switch_cnt = 0
        stay_cnt = 0  # 切换后保持的计数

        for t in range(T):
            pt = path[t]

            dists = np.full(N, np.inf)
            for i in range(N):
                if navitopo_mask[i] != 0:
                    continue
                dists[i] = point_to_polyline_proj_dist(pt.reshape(1, 2), navitopo_pts[i])[0]

            cur_idx = int(np.argmin(dists))

            if last_idx == -1:
                # 初始赋值
                best_idx[t] = cur_idx
                last_idx = cur_idx
                switch_cnt = 0
                stay_cnt = 1
            elif cur_idx == last_idx:
                # 保持当前段
                best_idx[t] = cur_idx
                switch_cnt = 0
                stay_cnt += 1
            else:
                # 考虑切换到新段
                if stay_cnt >= min_stay_len:  # 确保在当前段至少保持了足够的长度
                    # 只根据连续匹配点数量判断是否切换
                    switch_cnt += 1
                    if switch_cnt >= switch_confirm_len:
                        # 确认切换到新段
                        last_idx = cur_idx
                        switch_cnt = 0
                        stay_cnt = 1
                # 否则继续保持当前段
                best_idx[t] = last_idx

        return best_idx


    def build_segments_from_assignment(self, best_idx):
        """
        best_idx: (T,)
        return: [(start, end, navi_idx), ...]
        """
        segments = []

        start = 0
        cur = best_idx[0]

        for i in range(1, len(best_idx)):
            if best_idx[i] != cur:
                segments.append((start, i - 1, cur))
                start = i
                cur = best_idx[i]

        segments.append((start, len(best_idx) - 1, cur))
        return segments
        
    def merge_frequent_switching_segments(self, segments, min_segment_len=3):
        """
        合并频繁切换的短段，避免折线问题
        segments: [(start, end, navi_idx), ...]
        min_segment_len: 最小段长度阈值
        return: 合并后的段列表
        """
        if not segments:
            return segments
            
        merged_segments = []
        current_group = []
        current_group_length = 0
        
        def merge_group(group):
            """合并一组短段"""
            if not group:
                return
                
            if len(group) == 1:
                merged_segments.append(group[0])
                return
                
            # 计算每个navitopo索引在组中的使用长度
            navi_counts = {}
            for s, e, navi_idx in group:
                length = e - s + 1
                navi_counts[navi_idx] = navi_counts.get(navi_idx, 0) + length
            
            # 选择使用最久的navitopo索引
            dominant_navi = max(navi_counts, key=navi_counts.get)
            
            # 合并整个组为一个段，使用主导navitopo
            group_start = group[0][0]
            group_end = group[-1][1]
            merged_segments.append((group_start, group_end, dominant_navi))
        
        for seg in segments:
            start, end, navi_idx = seg
            seg_length = end - start + 1
            
            if seg_length < min_segment_len:
                # 短段，加入当前组
                current_group.append(seg)
                current_group_length += seg_length
            else:
                # 长段，先处理之前的短段组
                if current_group:
                    merge_group(current_group)
                    current_group = []
                    current_group_length = 0
                # 添加当前长段
                merged_segments.append(seg)
        
        # 处理最后一个短段组
        if current_group:
            merge_group(current_group)
        
        return merged_segments

    def _unit_vec(self, vec, eps=1e-6):
        norm = np.linalg.norm(vec)
        if norm < eps:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def _bezier_curve(self, p0, p1, p2, p3, n_dense=400):
        ts = np.linspace(0.0, 1.0, n_dense, dtype=np.float32)[:, None]
        omt = 1.0 - ts
        curve = (omt ** 3) * p0 + 3.0 * (omt ** 2) * ts * p1 + 3.0 * omt * (ts ** 2) * p2 + (ts ** 3) * p3
        return curve.astype(np.float32)

    def _resample_polyline_by_arclength(self, polyline, s_query, eps=1e-6):
        if polyline.shape[0] < 2:
            return None
        seg = polyline[1:] - polyline[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        s = np.zeros(polyline.shape[0], dtype=np.float32)
        s[1:] = np.cumsum(seg_len)
        total_len = s[-1]
        if total_len < float(s_query[-1]) - 1e-4:
            return None
        s_query = np.minimum(s_query, total_len)
        idx = np.searchsorted(s, s_query, side="right") - 1
        idx = np.clip(idx, 0, polyline.shape[0] - 2)
        s0 = s[idx]
        s1 = s[idx + 1]
        p0 = polyline[idx]
        p1 = polyline[idx + 1]
        w = (s_query - s0) / (s1 - s0 + eps)
        return (p0 + w[:, None] * (p1 - p0)).astype(np.float32)

    def smooth_stitched_prefix_bezier(self, stitched_xy, stitched_mask, step=2.0, blend_len_m=20.0):
        """
        对 stitched 近端做贝塞尔平滑，并按弧长严格 2m 等距重采样。
        仅替换前缀 valid 点，不改后续段。
        """
        valid_idx = np.where(stitched_mask.astype(bool))[0]
        if valid_idx.size < 3:
            return stitched_xy, stitched_mask
        n_blend = max(2, int(round(blend_len_m / step)))
        n_use = min(n_blend, valid_idx.size)
        if n_use < 2:
            return stitched_xy, stitched_mask

        end_idx = int(valid_idx[n_use - 1])
        if end_idx <= 0:
            return stitched_xy, stitched_mask

        p0 = np.array([0.0, 0.0], dtype=np.float32)
        p3 = stitched_xy[end_idx].astype(np.float32)
        t0 = np.array([1.0, 0.0], dtype=np.float32)
        t1 = self._unit_vec(stitched_xy[end_idx] - stitched_xy[end_idx - 1])

        span = float(np.linalg.norm(p3 - p0))
        d0 = float(np.clip(0.35 * span, 4.0, 10.0))
        d1 = float(np.clip(0.35 * span, 4.0, 10.0))
        p1 = p0 + d0 * t0
        p2 = p3 - d1 * t1

        bezier_dense = self._bezier_curve(p0, p1, p2, p3, n_dense=400)
        s_query = np.arange(1, n_use + 1, dtype=np.float32) * float(step)
        prefix_pts = self._resample_polyline_by_arclength(bezier_dense, s_query)
        if prefix_pts is None:
            return stitched_xy, stitched_mask

        out_xy = stitched_xy.copy()
        out_mask = stitched_mask.copy()
        out_xy[valid_idx[:n_use]] = prefix_pts
        out_mask[valid_idx[:n_use]] = True
        return out_xy, out_mask


    def resample_navi_segment(
        self, navitopo_pts, path_seg, step=2.0, eps=1e-6, max_output_len=None  # (P, 2)  # (K, 2)
    ):
        K = path_seg.shape[0]
        P = navitopo_pts.shape[0]

        path_start = path_seg[0]

        best_dist = np.inf
        best_i = 0
        best_t = 0.0

        for i in range(P - 1):
            p0 = navitopo_pts[i]
            p1 = navitopo_pts[i + 1]
            v = p1 - p0
            vv = np.dot(v, v)
            if vv < eps:
                continue

            t = np.dot(path_start - p0, v) / vv
            t = np.clip(t, 0.0, 1.0)
            proj = p0 + t * v

            dist = np.linalg.norm(path_start - proj)
            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_t = t

        start_pt = navitopo_pts[best_i] + best_t * (
            navitopo_pts[best_i + 1] - navitopo_pts[best_i]
        )

        nav = np.concatenate([start_pt[None], navitopo_pts[best_i + 1 :]], axis=0)

        if nav.shape[0] < 2:
            return np.zeros((K, 2), dtype=np.float32), 0

        delta = nav[1:] - nav[:-1]
        seg_len = np.linalg.norm(delta, axis=1)

        s = np.zeros(nav.shape[0], dtype=np.float32)
        s[1:] = np.cumsum(seg_len)
        total_len = s[-1]

        max_points = int(np.floor(total_len / step)) + 1
        out_cap = K if max_output_len is None else int(max(1, max_output_len))
        valid_len = min(out_cap, max_points)

        if valid_len <= 0:
            return np.zeros((K, 2), dtype=np.float32), 0

        s_query = np.arange(valid_len, dtype=np.float32) * step
        s_query = np.minimum(s_query, total_len)

        idx = np.searchsorted(s, s_query, side="right") - 1
        idx = np.clip(idx, 0, nav.shape[0] - 2)

        s0 = s[idx]
        s1 = s[idx + 1]
        p0 = nav[idx]
        p1 = nav[idx + 1]

        w = (s_query - s0) / (s1 - s0 + eps)
        pts = p0 + w[:, None] * (p1 - p0)

        return pts.astype(np.float32), valid_len

    def build_path_piecewise_navitopo(
        self,
        data,
        ego_future_status_fixed,        # (T, 2)
        ego_future_status_fixed_mask,   # (T)
        step=2.0,
        switch_confirm_len=3,
        min_stay_len=5,  # 切换后至少保持的长度
        deal_num=10,     # 过渡段处理的点数量
        enable_full_projection=False,  # 是否生成每条 navi 的全量投影候选
    ):
        navitopo_pts = data['navitopo_pts'].detach().cpu().numpy()
        navitopo_mask = data['navitopo_mask'].detach().cpu().numpy()
        path = ego_future_status_fixed[:self.future_steps_fixed, :2].detach().cpu().numpy()
        T = path.shape[0]
        N = navitopo_pts.shape[0]

        # 返回维度：(1+N, T, 2) 和 (1+N, T)
        # index 0: 分段匹配的 Best Navi
        # index 1~N: 每条 navi 的全量投影候选
        target_all_all = np.zeros((1 + N, T, 2), dtype=np.float32)
        mask_all_all = np.zeros((1 + N, T), dtype=bool)

        if (navitopo_mask == 0).sum() == 0:
            return torch.from_numpy(target_all_all), torch.from_numpy(mask_all_all)

        valid_indices = np.where(ego_future_status_fixed_mask.detach().cpu().numpy().astype(bool))[0]
        if valid_indices.size == 0:
            return torch.from_numpy(target_all_all), torch.from_numpy(mask_all_all)
        path_valid = path[valid_indices]

        # 第 0 路使用的数组引用
        target_all = target_all_all[0]
        mask_all = mask_all_all[0]

        best_navi = self.assign_navi_along_path(
            path_valid, navitopo_pts, navitopo_mask, switch_confirm_len, min_stay_len
        )

        segments = self.build_segments_from_assignment(best_navi)
        
        # 检测并合并频繁切换的区域
        segments = self.merge_frequent_switching_segments(segments, min_segment_len=3)

        write_ptr = 0
        prev_seg_end = None
        prev_navi_idx = None

        for s, e, navi_idx in segments:
            if write_ptr >= T:
                break

            path_seg = path_valid[s : e + 1]
            seg_len = len(path_seg)

            seg_pts, valid_len = self.resample_navi_segment(
                navitopo_pts[navi_idx], path_seg, step
            )

            write_len = min(seg_len, valid_len)

            if write_len > 0:
                # 确保第一个点始终是自车初始位置
                if write_ptr == 0 and len(seg_pts) > 0:
                    # 保存自车初始位置
                    ego_initial_pos = path_valid[0] if len(path_valid) > 0 else path[0]
                    # 设置seg_pts的第一个点为自车初始位置
                    seg_pts[0] = ego_initial_pos
                # 如果不是第一段，且导航拓扑索引发生变化，添加平滑过渡桥梁
                if write_ptr > 0 and prev_navi_idx is not None and prev_navi_idx != navi_idx:
                    # 当第一段太短时跳过桥梁平滑，避免覆写起点
                    if write_ptr < deal_num:
                        pass  # 第一段太短，不做桥梁替换
                    else:
                        try:
                            # 获取Navitopo A的倒数第deal_num个点和Navitopo B的第deal_num个点作为过渡点
                            # 这样可以大幅增加过渡距离，使贝塞尔曲线更加自然平滑
                            navi_a_end = target_all[write_ptr - deal_num] if write_ptr >= deal_num else target_all[write_ptr - 1]  # Navitopo A的倒数第deal_num个点（或最后一个点）
                            navi_b_start = seg_pts[deal_num - 1] if len(seg_pts) >= deal_num else seg_pts[0]  # Navitopo B的第deal_num个点（或第一个点）
                            
                            # 计算两点之间的距离
                            distance = np.linalg.norm(navi_b_start - navi_a_end)
                            
                            # 如果距离很小，不需要插入桥梁
                            if distance < 0.5:
                                pass  # 距离太近，直接连接
                            else:
                                # 在Navitopo A的终点和Navitopo B的起点之间插入贝塞尔曲线桥梁
                                # 使用二次贝塞尔曲线，计算控制点
                                
                                # 计算切线方向用于三次贝塞尔曲线
                                # 计算Navitopo A的切线方向（从倒数第(deal_num+2)个点到倒数第(deal_num-2)个点）
                                tangent_offset = 2  # 切线计算的偏移量，保持与原逻辑一致
                                if write_ptr >= deal_num + tangent_offset:
                                    prev_tangent_start = target_all[write_ptr - (deal_num + tangent_offset)]
                                    prev_tangent_end = target_all[write_ptr - (deal_num - tangent_offset)]
                                else:
                                    prev_tangent_start = target_all[0]
                                    prev_tangent_end = navi_a_end
                                prev_tangent = prev_tangent_end - prev_tangent_start
                                prev_tangent_unit = prev_tangent / (np.linalg.norm(prev_tangent) + 1e-8)
                                
                                # 计算Navitopo B的切线方向（从第(deal_num-2)个点到第(deal_num+2)个点）
                                if len(seg_pts) >= deal_num + tangent_offset:
                                    curr_tangent_start = seg_pts[deal_num - tangent_offset]
                                    curr_tangent_end = seg_pts[deal_num + tangent_offset]
                                else:
                                    curr_tangent_start = navi_b_start
                                    curr_tangent_end = seg_pts[-1] if len(seg_pts) > 1 else seg_pts[0]
                                curr_tangent = curr_tangent_end - curr_tangent_start
                                curr_tangent_unit = curr_tangent / (np.linalg.norm(curr_tangent) + 1e-8)
                                
                                # 设置控制点距离（基于点间距，假设约2米）
                                control_distance = 8.0  # 控制点距离起点/终点的距离
                                
                                # 计算三次贝塞尔曲线的四个控制点
                                P0 = navi_a_end  # 起点：Navitopo A的倒数第10个点
                                P1 = navi_a_end + prev_tangent_unit * control_distance  # 起点控制
                                P2 = navi_b_start - curr_tangent_unit * control_distance  # 终点控制
                                P3 = navi_b_start  # 终点：Navitopo B的第10个点
                                
                                # 生成贝塞尔曲线上的点作为桥梁
                                # 由于现在过渡距离更长，需要更多的过渡点
                                n_bridge_points = deal_num  # 生成deal_num个桥梁点覆盖更长过渡距离
                                bridge_points = []
                                
                                # 三次贝塞尔曲线公式：B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
                                def cubic_bezier(t, p0, p1, p2, p3):
                                    return ((1-t)**3 * p0 + 
                                            3*(1-t)**2 * t * p1 + 
                                            3*(1-t) * t**2 * p2 + 
                                            t**3 * p3)
                                
                                # 生成桥梁点（包括起点和终点之间的所有过渡点）
                                for i in range(n_bridge_points):
                                    t = i / (n_bridge_points - 1.0)  # t从0到1
                                    bridge_point = cubic_bezier(t, P0, P1, P2, P3)
                                    bridge_points.append(bridge_point)
                                
                                # 替换原有的过渡区域
                                # 1. 替换Navitopo A的最后(deal_num-1)个点（从write_ptr-(deal_num-1)到write_ptr）
                                # 确保replace_start > 0，永远不覆写路径起点
                                replace_len = deal_num - 1
                                replace_start = max(write_ptr - replace_len, 1)  # 至少保留target_all[0]
                                replace_end = write_ptr
                                actual_replace_len = replace_end - replace_start
                                
                                # 2. 更新target_all中Navitopo A的部分
                                target_all[replace_start:replace_end] = np.array(bridge_points[:actual_replace_len])
                                mask_all[replace_start:replace_end] = True
                                
                                # 3. 替换seg_pts中Navitopo B的前(deal_num-1)个点
                                seg_pts_replace_len = min(replace_len, len(seg_pts))
                                seg_pts[:seg_pts_replace_len] = np.array(bridge_points[-seg_pts_replace_len:])
                        except Exception as e:
                        # 异常处理：如果过渡失败，不影响整体流程
                            pass
                        
                # 写入当前段的点
                # 确保有足够的空间
                if write_ptr + write_len > target_all.shape[0]:
                    # 扩展数组
                    new_target_all = np.zeros((write_ptr + write_len, 2), dtype=np.float32)
                    new_mask_all = np.zeros((write_ptr + write_len,), dtype=bool)
                    new_target_all[:target_all.shape[0]] = target_all
                    new_mask_all[:mask_all.shape[0]] = mask_all
                    target_all = new_target_all
                    mask_all = new_mask_all
                
                # 写入当前段的点
                target_all[write_ptr : write_ptr + write_len] = seg_pts[:write_len]
                mask_all[write_ptr : write_ptr + write_len] = True
                prev_seg_end = seg_pts[write_len - 1]
                prev_navi_idx = navi_idx
                write_ptr += write_len

            if write_len < seg_len:
                break

        # 末段继续沿当前 navitopo 外推，尽量填满原始长度 T
        if write_ptr < T and len(segments) > 0:
            last_navi_idx = segments[-1][2]
            if write_ptr > 0:
                anchor = target_all[write_ptr - 1 : write_ptr]
                skip_n = 1  # 避免与已写入最后一点重复
            else:
                anchor = path_valid[-1:]
                skip_n = 0
            
            # 计算剩余需要填充的长度
            remaining_len = T - write_ptr
            seg_pts_ext, valid_len_ext = self.resample_navi_segment(
                navitopo_pts[last_navi_idx],
                anchor,
                step,
                max_output_len=(remaining_len + skip_n),
            )
            
            if valid_len_ext > skip_n:
                pts_ext = seg_pts_ext[skip_n:valid_len_ext]
                write_len_ext = min(remaining_len, pts_ext.shape[0])
                if write_len_ext > 0:
                    # 确保有足够的空间
                    if write_ptr + write_len_ext > target_all.shape[0]:
                        # 扩展数组
                        new_target_all = np.zeros((T, 2), dtype=np.float32)
                        new_mask_all = np.zeros((T,), dtype=bool)
                        new_target_all[:target_all.shape[0]] = target_all
                        new_mask_all[:mask_all.shape[0]] = mask_all
                        target_all = new_target_all
                        mask_all = new_mask_all
                    
                    target_all[write_ptr : write_ptr + write_len_ext] = pts_ext[:write_len_ext]
                    mask_all[write_ptr : write_ptr + write_len_ext] = True
                    write_ptr += write_len_ext

        # 确保第 0 路结果长度为 T
        if target_all.shape[0] > T:
            target_all_all[0] = target_all[:T]
            mask_all_all[0] = mask_all[:T]
        elif target_all.shape[0] < T:
            target_all_all[0, :target_all.shape[0]] = target_all
            mask_all_all[0, :mask_all.shape[0]] = mask_all
        # else: 已经通过引用写入了 target_all_all[0]

        # 对第 0 路做 20m 贝塞尔平滑过渡
        if mask_all_all[0].any():
            smoothed_xy_0, smoothed_mask_0 = self.smooth_stitched_prefix_bezier(
                target_all_all[0].copy(), mask_all_all[0].copy(),
                step=step, blend_len_m=20.0,
            )
            target_all_all[0] = smoothed_xy_0
            mask_all_all[0] = smoothed_mask_0

        # ==========================================
        # 第 1~N 路：每条 navi 的全量投影（不分段）
        # ==========================================
        if enable_full_projection:
            for navi_idx in range(N):
                if navitopo_mask[navi_idx] != 0:
                    continue

                current_target = target_all_all[1 + navi_idx]
                current_mask = mask_all_all[1 + navi_idx]

                seg_pts, valid_len = self.resample_navi_segment(
                    navitopo_pts[navi_idx], path_valid, step
                )

                write_len = min(T, valid_len)
                if write_len > 0:
                    current_target[:write_len] = seg_pts[:write_len]
                    current_mask[:write_len] = True

                # 20m 贝塞尔平滑过渡：从 (0,0) 平滑接入 navi 投影线
                if current_mask.any():
                    smoothed_xy, smoothed_mask = self.smooth_stitched_prefix_bezier(
                        current_target.copy(), current_mask.copy(),
                        step=step, blend_len_m=20.0,
                    )
                    target_all_all[1 + navi_idx] = smoothed_xy
                    mask_all_all[1 + navi_idx] = smoothed_mask

        target_all_all = torch.from_numpy(target_all_all)
        mask_all_all = torch.from_numpy(mask_all_all)

        # ================== DEBUG ==================
        debug = False
        if debug:
            import matplotlib.pyplot as plt
            import time

            save_path = (
                "/mnt/afs/wanghaibo1/dlp_v2_codes/dif_25_0124/tmp/concat_navi/"
                + str(time.time()) + ".png"
            )

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

            # ---------- 左图：所有 navitopo（线） ----------
            for i in range(N):
                if navitopo_mask[i] != 0:
                    continue
                pts = navitopo_pts[i]
                ax0.plot(
                    pts[:, 0],
                    pts[:, 1],
                    linestyle="--",
                    color="red",
                    alpha=0.4,
                    linewidth=1.0
                )

            ax0.set_title("All navitopo")
            ax0.axis("equal")

            # ---------- 右图 ----------
            ax1.plot(
                path[:, 0],
                path[:, 1],
                "k--",
                linewidth=2.0,
                label="path"
            )

            for i in range(N):
                if navitopo_mask[i] != 0:
                    continue
                pts = navitopo_pts[i]
                ax1.plot(
                    pts[:, 0],
                    pts[:, 1],
                    linestyle="--",
                    color="red",
                    alpha=0.3,
                    linewidth=1.0
                )

            # 第 0 路（分段匹配）
            target_best_np = target_all_all[0].numpy()
            mask_best_np = mask_all_all[0].numpy().astype(bool)
            valid_idx_0 = np.where(mask_best_np)[0]
            if len(valid_idx_0) > 0:
                sel = target_best_np[valid_idx_0]
                ax1.scatter(
                    sel[:, 0], sel[:, 1],
                    c="red", s=5, label="best navi (piecewise)", zorder=5
                )

            # 第 1~N 路（全量投影）
            colors = ['blue', 'green', 'orange', 'purple', 'cyan']
            for navi_idx in range(N):
                if navitopo_mask[navi_idx] != 0:
                    continue
                fp_target = target_all_all[1 + navi_idx].numpy()
                fp_mask = mask_all_all[1 + navi_idx].numpy().astype(bool)
                valid_idx_k = np.where(fp_mask)[0]
                if len(valid_idx_k) > 0:
                    sel_k = fp_target[valid_idx_k]
                    c = colors[navi_idx % len(colors)]
                    ax1.scatter(
                        sel_k[:, 0], sel_k[:, 1],
                        c=c, s=3, label=f"navi_{navi_idx} (full proj)", zorder=4, alpha=0.6
                    )

            ax1.set_title("Path vs navitopo (multi-candidate)")
            ax1.axis("equal")
            ax1.legend(fontsize='small')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return target_all_all, mask_all_all
    
    def process_occ(self, data_occ, ego_future_status, debug=False):
        if data_occ.grid_width == 240 and data_occ.grid_length == 550:
            self.occmap_config["resolution"] = 0.2
            self.occmap_config["height"] = 240
            self.occmap_config["width"] = 550
            self.occmap_config["ego_i0"] = self.occmap_config["height"] // 2
            self.occmap_config["ego_j0"] = math.floor(10 / self.occmap_config["resolution"] + 0.5)
        elif data_occ.grid_width == 120 and data_occ.grid_length == 275:
            self.occmap_config["resolution"] = 0.4
            self.occmap_config["height"] = 120
            self.occmap_config["width"] = 275
            self.occmap_config["ego_i0"] = self.occmap_config["height"] // 2
            self.occmap_config["ego_j0"] = math.floor(10 / self.occmap_config["resolution"] + 0.5)

        # dim_0: (0~119) dim_1: (0~274) 
        height, width = self.occmap_config["height"], self.occmap_config["width"]
        # 0：unknown  1：地面  2：占位  3：施工地区
        occ_map = np.ones((height, width), dtype=np.float32)
        #import ipdb;ipdb.set_trace()
        # 向量化优化：批量处理所有数据点，避免循环
        if len(data_occ.cells) > 0:
            # 改回朴素循环，这里是OCC的绝对耗时卡点，由于data_occ是list，转换成numpy非常耗时，需要pb链路直接跳过list的数据形态后，耗时能基本都干掉
            for item in data_occ.cells:
                # item 结构通常为 [h, w, type, ...]；col/row 可能越界，裁剪到 [0, height-1] / [0, width-1]
                r = max(0, min(int(item.row), width - 1))
                c = max(0, min(int(item.col), height - 1))
                occ_map[height - 1 - c, r] = float(item.occ_type)
        # obtain occ feat
        binary_occ_map = occ_map.copy()
        # 1 is drivable area, and 0 is undrivable area
        binary_occ_map[binary_occ_map == 1] = 1
        binary_occ_map[binary_occ_map > 1] = 0
        if (binary_occ_map == 1).all() or (binary_occ_map == 0).all():
            occ_polygons_padding_pts = torch.zeros(
                self.max_n_occ_polygon, self.max_n_occ_polygon_point, 2
            )
            occ_polygons_padding_attrs = torch.zeros(
                self.max_n_occ_polygon, self.max_n_occ_polygon_attr
            )
            occ_polygons_padding_mask = torch.ones(
                self.max_n_occ_polygon, dtype=torch.float32
            )
        else:
            occ_polygons_pts_img, occ_polygons_attrs_img = self.compute_contours(
                binary_occ_map, occ_map
            )
            occ_polygons_pts, occ_polygons_attrs = self.convert_polygon_to_real(
                occ_polygons_pts_img, occ_polygons_attrs_img, 
                resolution=self.occmap_config["resolution"], i0=self.occmap_config["ego_i0"], j0=self.occmap_config["ego_j0"]
            )
            occ_polygons_pts_resampled = self.interpolate_polygons(
                occ_polygons_pts, num_points=self.max_n_occ_polygon_point
            )
            max_polys = self.max_n_occ_polygon
            num_polys = occ_polygons_pts_resampled.shape[0]
            n_copy = min(num_polys, max_polys)

            pts_np = np.zeros((max_polys, self.max_n_occ_polygon_point, 2), dtype=np.float32)
            mask_np = np.ones(max_polys, dtype=np.float32) # 默认全填充(1.0)
            if n_copy > 0:
                pts_np[:n_copy] = occ_polygons_pts_resampled[:n_copy]
                mask_np[:n_copy] = 0.0 # 有效区域设为 0.0

            attrs_np = np.zeros((max_polys, self.max_n_occ_polygon_attr), dtype=np.float32)
            if n_copy > 0:
                # 一次性将 list 转换为 numpy 数组
                attrs_np[:n_copy] = np.array(occ_polygons_attrs[:n_copy], dtype=np.float32)

            occ_polygons_padding_pts = torch.from_numpy(pts_np)
            occ_polygons_padding_mask = torch.from_numpy(mask_np)
            occ_polygons_padding_attrs = torch.from_numpy(attrs_np)

        ego_gt_pos = ego_future_status[..., :2]
        occ_invalid_mask = False
        
        debug = False
        if debug and occ_polygons_padding_mask.any():
            self.infer_conut += 1
            data_occ_copy = np.ones((height, width), dtype=np.float32)
            print("="*10 + " drawing occ map " + "="*10)
            if len(data_occ.cells) > 0:
                # 向量化优化 debug 模式下的循环
                occ_src = [[cell.row, cell.col, cell.occ_type] for cell in data_occ.cells]
                data_occ_np = np.array(occ_src, dtype=np.int64)
                row_debug = data_occ_np[:, 1]  # col in original code (对应原循环中的 col)
                col_debug = data_occ_np[:, 0]  # row in original code (对应原循环中的 row)
                type_debug = data_occ_np[:, 2].astype(np.float32)  # tp
                data_occ_copy[row_debug, col_debug] = type_debug
            data_occ_copy[data_occ_copy == 1] = 1   # 白
            data_occ_copy[data_occ_copy > 1] = 0     # 黑
            out_dir = "/mnt/afs/liuzhaoyang1/diffusion_codes/dif_82/tmp/occ"
            self.visualize_map(ego_gt_pos, data_occ_copy, out_dir, mode="data_occ", resolution=self.occmap_config["resolution"])
            self.visualize_map(ego_gt_pos, binary_occ_map, out_dir, mode="binary_occ_map", resolution=self.occmap_config["resolution"])
            self.visualize_polygons(ego_gt_pos, occ_polygons_pts_img, occ_polygons_attrs_img, out_dir, mode="occ_polygons_img",
                                    height=self.occmap_config["height"], width=self.occmap_config["width"], resolution=self.occmap_config["resolution"])
            self.visualize_polygons_real(ego_gt_pos, occ_polygons_pts, occ_polygons_attrs, out_dir, mode="occ_polygons")
            self.visualize_polygons_real(ego_gt_pos, occ_polygons_pts_resampled, occ_polygons_attrs, out_dir, mode="occ_polygons_resampled")
            self.visualize_polygons_real(ego_gt_pos, occ_polygons_padding_pts, occ_polygons_padding_attrs, out_dir, mode="occ_polygons_padding")
            # self.dump_occ_polygons_json(out_dir=out_dir,
            #                             data_occ=data_occ,
            #                             occ_map=occ_map,
            #                             ego_gt_pos=ego_gt_pos,
            #                             occ_polygons_pts=occ_polygons_pts,
            #                             occ_polygons_attrs=occ_polygons_attrs,
            #                             occ_polygons_pts_resampled=occ_polygons_pts_resampled,
            #                             occ_polygons_padding_pts=occ_polygons_padding_pts,
            #                             occ_polygons_padding_attrs=occ_polygons_padding_attrs,
            #                             occ_polygons_padding_mask=occ_polygons_padding_mask,
            #                         )
            import pdb; pdb.set_trace()
        # 将 occ_map 转换回 torch.Tensor 以适配后续流程
        occ_map_torch = torch.from_numpy(occ_map)
        return occ_map_torch, occ_invalid_mask, occ_polygons_padding_pts, occ_polygons_padding_attrs, occ_polygons_padding_mask
    
    def interpolate_polygons(
        self,
        raw_polygons: List[np.ndarray],
        num_points: int = 50,
    ) -> np.ndarray:
        """
        Batch version of interpolate_polygons using Offset Trick to eliminate Python loops.
        """
        if not raw_polygons:
            return np.zeros((0, num_points, 2), dtype=np.float32)
        # 1. Pre-filter invalid polygons
        valid_indices = [i for i, p in enumerate(raw_polygons) if p is not None and len(p) > 0]
        if not valid_indices:
            return np.zeros((0, num_points, 2), dtype=np.float32)
        valid_polys = [raw_polygons[i] for i in valid_indices]
        n_polys = len(valid_polys)
        lengths = np.array([len(p) for p in valid_polys])
        results = np.empty((n_polys, num_points, 2), dtype=np.float32)
        # 2. Handle single-point polygons
        is_single = (lengths == 1)
        if is_single.any():
            single_indices = np.where(is_single)[0]
            # [N, 2] -> [N, 1, 2] -> broadcast to [N, num_points, 2]
            results[single_indices] = np.stack([valid_polys[i][0] for i in single_indices])[:, None, :]
        # 3. Handle multi-point polygons using Offset Trick
        is_multi = (lengths >= 2)
        if is_multi.any():
            multi_indices = np.where(is_multi)[0]
            multi_polys = [valid_polys[i] for i in multi_indices]
            multi_lengths = lengths[multi_indices]
            # A. Concatenate points and compute distances (Use float64 for intermediate steps)
            all_pts = np.concatenate(multi_polys, axis=0).astype(np.float64)
            dx_dy = np.diff(all_pts, axis=0)
            dists = np.sqrt(np.sum(dx_dy**2, axis=1))
            # Reset distances at boundaries between polygons
            boundary_indices = np.cumsum(multi_lengths)[:-1] - 1
            dists[boundary_indices] = 0
            # B. Compute cumulative distances with offsets
            cum_dists = np.concatenate(([0], np.cumsum(dists)))
            poly_start_indices = np.concatenate(([0], np.cumsum(multi_lengths)[:-1]))
            # Relative cumulative distance within each polygon
            rel_cum_dists = cum_dists - np.repeat(cum_dists[poly_start_indices], multi_lengths)
            # C. Create global strictly increasing XP using offsets
            poly_total_lengths = rel_cum_dists[np.cumsum(multi_lengths) - 1]
            # Add a small buffer to ensure XP is strictly increasing
            offsets = np.concatenate(([0], np.cumsum(poly_total_lengths[:-1] + 1.0)))
            xp = rel_cum_dists + np.repeat(offsets, multi_lengths)
            # D. Create global query points
            query_rel = np.linspace(0, 1, num_points)
            query_pts = (poly_total_lengths[:, None] * query_rel[None, :]).ravel()
            query_xp = query_pts + np.repeat(offsets, num_points)
            # E. Batch interpolation (np.interp will work in float64 if xp/fp are float64)
            results[multi_indices, :, 0] = np.interp(query_xp, xp, all_pts[:, 0]).reshape(-1, num_points)
            results[multi_indices, :, 1] = np.interp(query_xp, xp, all_pts[:, 1]).reshape(-1, num_points)
        return results

    def to_python(self, obj):
        """Recursively convert numpy/tensor objects to python builtins."""
        import numpy as np
        import torch

        if isinstance(obj, dict):
            return {k: self.to_python(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self.to_python(v) for v in obj]

        elif isinstance(obj, tuple):
            return tuple(self.to_python(v) for v in obj)

        # numpy 数组 -> python list
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # numpy 数值 -> python 数值
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)

        # torch Tensor -> python list 或 python 数字
        elif isinstance(obj, torch.Tensor):
            if obj.dim() == 0:
                return obj.item()
            else:
                return obj.detach().cpu().tolist()
        return obj

    def dump_occ_polygons_json(
        self,
        out_dir,
        data_occ,
        occ_map,
        ego_gt_pos,
        occ_polygons_pts,
        occ_polygons_attrs,
        occ_polygons_pts_resampled,
        occ_polygons_padding_pts,
        occ_polygons_padding_attrs,
        occ_polygons_padding_mask,
    ):
        if isinstance(occ_map, torch.Tensor):
            occ_map_list = occ_map.detach().cpu().tolist()
        else:
            occ_map_list = occ_map

        if isinstance(ego_gt_pos, torch.Tensor):
            ego_gt_pos_list = ego_gt_pos.detach().cpu().tolist()
        else:
            ego_gt_pos_list = ego_gt_pos

        # 转为 python list
        data = {
            "ego_gt_pos": ego_gt_pos_list,  # n_future × 2
            "data_occ": data_occ,
            "occ_map": occ_map_list,        # H × W
            "polygons": [
                {
                    "raw": poly.tolist(),
                    "attr": attr, 
                    "resampled": resampled.tolist(),
                }
                for poly, attr, resampled in zip(
                    occ_polygons_pts, 
                    occ_polygons_attrs, 
                    occ_polygons_pts_resampled
                )
            ],
            "padding_pts": occ_polygons_padding_pts.tolist(),
            "padding_attrs": occ_polygons_padding_attrs.tolist(),
            "padding_mask": occ_polygons_padding_mask.tolist(),
        }

        data = self.to_python(data)
        timestamp = self.format_ns_timestamp(self.debug_info['timestamp'])
        out_path = os.path.join(out_dir, f"{timestamp}_{self.infer_conut}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved OCC polygon debug json to: {out_path}")

    def read_label_ego_future_status(self, future_traj, future_fixed_dis):
        points = future_traj.points
        n = len(points)
        pos = np.fromiter(
            (v for p in points for v in (p.pos_x, p.pos_y)),
            dtype=np.float32,
            count=n * 2,
        ).reshape(n, 2)
        v = np.fromiter(
            (v for p in points for v in (p.v_x, p.v_y)),
            dtype=np.float32,
            count=n * 2,
        ).reshape(n, 2)
        yaw = np.fromiter((p.yaw for p in points), dtype=np.float32, count=n).reshape(-1, 1)
        mask = np.fromiter((bool(p.mask) for p in points), dtype=np.bool_, count=n)

        points_f = future_fixed_dis.points
        n_f = len(points_f)
        pos_f = np.fromiter(
            (v for p in points_f for v in (p.pos_x, p.pos_y)),
            dtype=np.float32,
            count=n_f * 2,
        ).reshape(n_f, 2)
        v_f = np.fromiter(
            (v for p in points_f for v in (p.v_x, p.v_y)),
            dtype=np.float32,
            count=n_f * 2,
        ).reshape(n_f, 2)
        yaw_f = np.fromiter((p.yaw for p in points_f), dtype=np.float32, count=n_f).reshape(-1, 1)
        mask_f = np.fromiter((bool(p.mask) for p in points_f), dtype=np.bool_, count=n_f)
        if n_f < self.future_steps_fixed:
            pad_n = self.future_steps_fixed - n_f
            pos_f = np.pad(pos_f, ((0, pad_n), (0, 0)), mode="constant", constant_values=0.0)
            v_f = np.pad(v_f, ((0, pad_n), (0, 0)), mode="constant", constant_values=0.0)
            yaw_f = np.pad(yaw_f, ((0, pad_n), (0, 0)), mode="constant", constant_values=0.0)
            mask_f = np.pad(mask_f, (0, pad_n), mode="constant", constant_values=False)
            n_f = self.future_steps_fixed

        split = int(self.planning_interval / 0.1)
        ego_future_status = np.concatenate([pos, v, yaw], axis=1)[split - 1:self.future_steps:split]
        ego_future_mask = mask[split - 1:self.future_steps:split]
        trainFlag_traj = ~np.any(ego_future_status[:, :2] > 200)

        split_fixed = int(self.planning_interval_fixed / 1.0)
        ego_future_status_fixed = np.concatenate([pos_f, v_f, yaw_f], axis=1)[split_fixed - 1:self.future_steps_fixed:split_fixed]
        ego_future_mask_fixed = mask_f[split_fixed - 1:self.future_steps_fixed:split_fixed]
        trainFlag_path = ~np.any(ego_future_status_fixed[:, :2] > 200)
        # 使用path过滤异常帧
        is_valid_frame = self.filter_ego_future_status_fixed(
            ego_future_status_fixed,
            ego_future_mask_fixed,
        )

        ego_future_status_fixed_full = np.concatenate([pos_f, v_f, yaw_f], axis=1)[:self.route_gt_steps]
        ego_future_mask_fixed_full = mask_f[:self.route_gt_steps]
        
        if self.trainStage == 'stage_1':
            trainFlag_ego = trainFlag_path and is_valid_frame
        elif self.trainStage == 'stage_2':
            trainFlag_ego = trainFlag_traj and is_valid_frame  
        else:
            trainFlag_ego = (trainFlag_traj and trainFlag_path and is_valid_frame)

        return (
            ego_future_status,
            ego_future_mask,
            ego_future_status_fixed,
            ego_future_mask_fixed,
            ego_future_status_fixed_full,
            ego_future_mask_fixed_full,
            trainFlag_ego,
        )

    def filter_ego_future_status_fixed(self, ego_future_status_fixed, ego_future_mask_fixed):
        """Filter fixed future status by first point and turn angle constraints.

        Returns False if the first valid point has x < 0, or if any adjacent
        segment pair forms an angle > 90 degrees (dot product < 0).
        """
        valid_mask = ego_future_mask_fixed.astype(bool)  # only use valid future points
        if valid_mask.sum() == 0:
            return False

        # first valid point must be in front (x >= 0)
        first_idx = int(np.argmax(valid_mask))
        if ego_future_status_fixed[first_idx, 0] < 0:
            return False

        pts = ego_future_status_fixed[valid_mask][:, :2]
        if pts.shape[0] < 3:
            return False

        # compute adjacent direction vectors for turn angle check
        vecs = pts[1:] - pts[:-1]
        norms = np.linalg.norm(vecs, axis=1)
        if norms.size < 2:
            return False

        # angle > 90° if dot product < 0
        dot = np.sum(vecs[:-1] * vecs[1:], axis=1)
        valid_vec_mask = (norms[:-1] > 1e-6) & (norms[1:] > 1e-6)
        if np.any(dot[valid_vec_mask] < 0):
            return False

        return True

    def extract_risky_lines_mask(self, laneline_pts_np, laneline_attrs_np, laneline_mask_np):
        valid_mask = ~laneline_mask_np.astype(bool)
        color = laneline_attrs_np[:, 0]  # white, yellow, other
        laneline_type = laneline_attrs_np[:, 1]  # edge, laneline, centerline, stopline, route, other
        laneline_style = laneline_attrs_np[:, 2]  # solid, dash, left_dash_right_solid, left_solid_right_dash, other

        cond_color = (color == 2)
        cond_edge = (laneline_type == 1)
        cond_lane = (laneline_style == 1)
        front_mask = (laneline_pts_np[:, :, 0].max(axis=1) > 0)
        risky_lines_mask = (cond_edge | cond_lane | cond_color) & front_mask & valid_mask
        return risky_lines_mask.astype(np.bool_)

    def del_navi_light(self, navitopo_pts, navitopo_attrs, res, data_feaNotEmb, multi_scenes_label):#删navi
        #删的时候还没用上
        navi_pts = navitopo_pts
        disToCur_norm = torch.norm(navi_pts, dim=2)  # [N, M]
        min_distances, min_distances_Cur_idx = torch.min(disToCur_norm, dim=1)
        _, min_dist_navitopo_idx = torch.min(min_distances, dim=0)
        curdis_Y = navi_pts[torch.arange(navi_pts.shape[0]), min_distances_Cur_idx, 1]
        
        path = res['ego_future_status_fixed'][:self.future_steps_fixed, :2]          # [T, 2]
        path_valid_mask = res['ego_future_mask_fixed'][:self.future_steps_fixed]     # [T]
        path = path[path_valid_mask]                                                 # 仅保留有效 gt path 点
        if self.flag_train:
            if 'left_lane_change' in multi_scenes_label:
                keep_mask = (curdis_Y >= 1)
            elif 'right_lane_change' in multi_scenes_label:
                keep_mask = (curdis_Y <= -1)
            else:
                keep_mask = torch.ones(len(curdis_Y)).bool()
            # 多navi多一个根据属性删的逻辑
            if self.use_multiNavi:
                del_mask = (navitopo_attrs[:, 0] != 1) & (navitopo_attrs[:, 2] < 200)
                keep_mask = keep_mask & (~del_mask)
            # 进一步过滤：若某根 navi 与 gt path 最后一个有效点的最小距离超过 20m，则过滤掉该 navi
            if path.numel() > 0 and navi_pts.numel() > 0:
                last_gt_pt = path[-1]                                                   # [2]
                # 计算每根 navi 上各点到最后一个 gt 点的距离 [N_navi, M]
                dists = torch.norm(navi_pts - last_gt_pt.view(1, 1, 2), dim=2)
                min_dists, _ = torch.min(dists, dim=1)                                  # 每根 navi 的最小距离 [N_navi]
                near_mask = (min_dists <= 20.0)
                # keep_mask 当前是 bool tensor，按距离条件再收紧一次
                keep_mask = (keep_mask & near_mask).tolist()
            else:
                keep_mask = keep_mask.tolist()    
        else:
            # del_mask = (navitopo_attrs[:, 0] != 1) & (navitopo_attrs[:, 2] < 200)
            del_mask = (navitopo_attrs[:, 0] != 1)
            keep_mask = (~del_mask).tolist()
            # 非导航车道 如果被全部删除，保留一根离自车当前最近的
            if not any(keep_mask):
                keep_mask[min_dist_navitopo_idx] = True
            # 推理默认注释掉：
            # 进一步过滤：若某根 navi 与 gt path 最后一个有效点的最小距离超过 20m，则过滤掉该 navi                                    
            # if path.numel() > 0 and navi_pts.numel() > 0:
            #     last_gt_pt = path[-1]                                                   # [2]
            #     # 计算每根 navi 上各点到最后一个 gt 点的距离 [N_navi, M]
            #     dists = torch.norm(navi_pts - last_gt_pt.view(1, 1, 2), dim=2)
            #     min_dists, _ = torch.min(dists, dim=1)                                  # 每根 navi 的最小距离 [N_navi]
            #     near_mask = (min_dists <= 20.0)
            #     # keep_mask 当前是 bool tensor，按距离条件再收紧一次
            #     keep_mask = (keep_mask & near_mask).tolist()
            # else:
            #     keep_mask = keep_mask.tolist()

        if res['ego_future_mask'].sum() != self.future_steps//int(self.planning_interval/0.1):
            print('Error. ego_future_mask.sum() != self.future_steps//int(self.planning_interval/0.1)')

        return keep_mask
    
    def wrap_to_pi(self, tensor):
        return (tensor + math.pi) % (2 * math.pi) - math.pi

    def format_ns_timestamp(self, ts_ns) -> str:
        """将纳秒级时间戳转换为格式化时间，全部用 - 分隔"""
        dt = datetime.datetime.fromtimestamp(int(ts_ns) / 1e9)
        return dt.strftime("%Y-%m-%d-%H-%M-%S") + f"-{dt.microsecond}"

    def get_timestamp(self, timestamp_ns=None):
        if timestamp_ns is None:
            raise ValueError("Error. timestamp_ns is None, open eval cannot work")

        # 高 9 位、中间 6 位、低 6 位
        high_part = timestamp_ns // 10**12
        mid_part = (timestamp_ns % 10**12) // 10**6
        low_part = timestamp_ns % 10**6
        timestamp_hi = torch.tensor(high_part, dtype=torch.float32)
        timestamp_mi = torch.tensor(mid_part, dtype=torch.float32)
        timestamp_lo = torch.tensor(low_part, dtype=torch.float32)

        return timestamp_hi, timestamp_mi, timestamp_lo

    def init_mining_shared(self):
        """
        加载 mining jsonl 到共享内存。当配置了 mining_file 时调用。
        支持纯 mining 模式（仅 jsonl 数据）或混合模式（jsonl + normal）。
        """
        if self.mining_files is None:
            return
        mining_files = self.mining_files
        if isinstance(mining_files, str):
            mining_files = [mining_files]

        self.mining_overfit_dir, self.mining_len = load_mining_overfit_shared(mining_files)

        if self.mining_len > 0:
            if self.mining_mix_ratio is not None and self.mining_mix_ratio > 0 and self.normal_item_num > 0:
                # 混合模式: 1 mining : N normal
                self.item_num = int(self.mining_len * (self.mining_mix_ratio + 1))
                mode_str = f"mixed (1 mining : {self.mining_mix_ratio} normal)"
            else:
                if self.mining_mix_ratio and self.normal_item_num <= 0:
                    print(f"[Mining] Warning: normal_item_num=0, fallback to pure mining mode")
                # 纯 mining 模式：仅使用 jsonl 数据
                self.mining_mix_ratio = None
                self.item_num = self.mining_len
                mode_str = "pure mining (jsonl only)"
            print(f"[Mining] Loaded {self.mining_len} samples from jsonl, mode={mode_str}, total={self.item_num}")
        else:
            self.mining_overfit_dir = None
            if self.mining_files:
                print(f"[Mining] Warning: mining files provided but no data loaded, using normal data")

    def _build_data_label_path(self, date, frame_id, use_tos = False, use_gz = True, use_datapro_pb = False) -> str:
        """
        将 (date, frame_id) 转换成 one_frame_all_data 的 json 路径。
        该逻辑在 __getitem__ / _init_occ_check 中复用。
        """
        if date not in self.date_split.keys():
            raise ValueError(f'Error, date {date} not in date_split file, please check')

        if use_tos:
            return f"tos://{self.date_split[date]}/one_frame_all_data/dlp_raw_data/{frame_id}{'.gz' if use_gz else ''}"
        if use_datapro_pb:
            return f"s3://{self.date_split[date]}/{date}/dlp_data/one_frame_all_data/{frame_id}{'.gz' if use_gz else ''}"
        return f"s3://{self.date_split[date]}/one_frame_all_data/{frame_id}{'.gz' if use_gz else ''}"

    def _init_occ_check(self) -> bool:
        """
        初始化阶段随机抽检若干条（当前固定为 10 条，避免引入参数化）做 OccChecker 校验。
        - 每个 rank 都会执行
        - 随机种子与 Lightning 保持一致：复用 torch.initial_seed()
        - 抽样来自全量二级索引（物理样本池），不参考 __getitem__ 的 idx 映射逻辑
        """
        try:
            rank = int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0
        except Exception:
            rank = 0

        seed = int(torch.initial_seed())
        rng = random.Random(seed + rank)

        payloads = []
        weights = []
        if self.prefetched_shm_dir:
            # shm 模式：遍历目录下的 json 文件
            for fname in os.listdir(self.prefetched_shm_dir):
                if fname == "scene_num.json" or not fname.endswith(".json"):
                    continue
                fpath = osp.join(self.prefetched_shm_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        v = json.load(f)
                        if isinstance(v, list) and len(v) > 0:
                            payloads.append(v)
                            weights.append(len(v))
                except Exception:
                    continue

        if len(payloads) == 0:
            print(f"[OccInitCheck][rank={rank}] skip: no secondary-index shards in memory", flush=True)
            return True

        ok = True
        for _ in range(10):
            try:
                payload = rng.choices(payloads, weights=weights, k=1)[0]
                entry = rng.choice(payload)  # [date, frame_id, route_cost]
                date, frame_id = str(entry[0]), int(entry[1])

                data_label_path = self._build_data_label_path(
                    date, frame_id, use_tos=self.use_tos_data, use_datapro_pb=self.use_datapro_pb
                )

                frame = self.data_client.load_json(data_label_path)
                if not OccChecker(frame).check():
                    ok = False
            except Exception:
                ok = False

        print(f"[OccInitCheck][rank={rank}] ok={bool(ok)}, seed={seed}", flush=True)
        return bool(ok)

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def padding(x, max_n):
    mask = torch.zeros(max_n, dtype=torch.float)
    data_type = x.dtype
    if x.shape[0] < max_n:
        mask[x.shape[0]:] = 1.0
        assert x.ndim in [2, 3], 'unsupported x.ndim'
        if x.ndim == 2:
            padding_tensor = torch.zeros((max_n - x.shape[0], x.shape[1]), dtype=data_type)
        elif x.ndim == 3:
            padding_tensor = torch.zeros((max_n - x.shape[0], x.shape[1], x.shape[2]), dtype=data_type)
        x = torch.cat([x, padding_tensor], dim=0)
    else:
        x = x[:max_n]
    return x, mask

def padding2d(x, mask, max_n):
    if x.shape[0] < max_n:
        data_type = x.dtype
        padding_mask = torch.zeros((max_n - x.shape[0], x.shape[1]), dtype=torch.bool)
        padding_tensor = torch.zeros((max_n - x.shape[0], x.shape[1], x.shape[2]), dtype=data_type)
        x = torch.cat([x, padding_tensor], dim=0)
        mask = torch.cat([mask, padding_mask], dim=0)
    else:
        x = x[:max_n]
        mask = mask[:max_n]
    return x, mask

def custom_2d_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Interpolate 2D points uniformly along the arc length.

    Args:
        x (np.ndarray):
            X coordinates of the polyline, shape (N,).
        y (np.ndarray):
            Y coordinates of the polyline, shape (N,).
        num_points (int):
            Number of interpolated points.

    Returns:
        np.ndarray:
            Interpolated 2D points, shape (num_points, 2),
            dtype float64 (can be cast to float32 by caller).
    """

    # 计算累计距离
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    distances = np.concatenate(([0], distances))  # 在起点处插入0
    cumulative_distances = np.cumsum(distances)

    # 创建插值函数
    interp_func_x = interp1d(cumulative_distances, x, kind="linear")
    interp_func_y = interp1d(cumulative_distances, y, kind="linear")

    # 在累积距离的范围内生成均匀分布的插值点
    uniform_distances = np.linspace(
        cumulative_distances[0], cumulative_distances[-1], num_points
    )

    # 在均匀分布的距离点上评估插值函数
    x_interp = interp_func_x(uniform_distances)
    y_interp = interp_func_y(uniform_distances)

    return np.column_stack((x_interp, y_interp))



class OccChecker:
    def __init__(self, frame_data):
        self.data = frame_data
        if isinstance(self.data['data_occ'], dict):
            self.occ = self.data['data_occ']['occ']
        else:
            self.occ = self.data['data_occ']
        self.occ_size = [275, 120]
        self.occupation = np.zeros((self.occ_size[0], self.occ_size[1]), dtype=bool)

    def check(self):
        for coord in self.occ:  # [row,col,type,heig,low]
            if len(coord) != 5:
                print(f"coord size error: {len(coord)}")
                return False
            row, col = coord[0], coord[1]
            if not (isinstance(row, int) and isinstance(col, int)):
                print(f"occ row or col data type error")
                return False
            if not (0 <= row < self.occ_size[0] and 0 <= col < self.occ_size[1]):
                print(f"occ map index error")
                return False
            if self.occupation[row][col]:
                print(f"occupation error: {self.occupation[row][col]}")
                return False
            self.occupation[row][col] = True
        return True
