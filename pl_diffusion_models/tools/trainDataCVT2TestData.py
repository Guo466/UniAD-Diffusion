# -*- coding: utf-8 -*- 

import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd()+'/configs')

import time
import os.path as osp
import json
import argparse
from tqdm import tqdm
import importlib
import bisect
import random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
factor_base = 1

scene_extract = {
    'left_nudge': 1 * factor_base,                      # 左绕障
    'left_change_lane': 0.5 * factor_base,             # 左变道
    'left_change_lane_efficient': 0.5 * factor_base,   # 左效率变道
    'right_nudge': 1 * factor_base,                     # 右绕障
    'right_change_lane': 0.8 * factor_base,            # 右变道
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
    'interactive_agent_cross': 0.04 * factor_base,         # 横穿agent交互
    'interactive_agent_large_dec': 0.5 * factor_base,     # 急减速agent交互
    'deviation_correction': 0.8 * factor_base,         # 偏离回正
    'large_curvature_lane_keeping': 1 * factor_base,    # 大曲率车道保持
    'brake': 0.4 * factor_base,                           # 刹车
    'brake2stop': 2 * factor_base,                      # 刹停
    'acc': 2 * factor_base,                             # 加速
    'safe_acc': 0.02 * factor_base,                      # 提速
    'cross_lane_keeping': 5 * factor_base,              # 路口车道保持
    'lane_keeping': 8 * factor_base,                    # 车道保持
    'near_intersection': 0.2 * factor_base,               # 接近路口
    'split': 0.06 * factor_base,                         # 分流
    'merge': 0.04 * factor_base,                           # 合流
    'roundabout': 1 * factor_base,                      # 环岛
    'normal': 8 * factor_base,                          # 平庸

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
n_frame = 0
scenes = []
n_frames = []
file_dir = '/mnt/afs/dlp_data/navitopo_v3/merge/data_merge'
with open(os.path.join(file_dir, 'scene_data/scene_num.json'), 'r') as f:
    scene_num = json.load(f)

for scene, extract in scene_extract.items():
    if scene not in scene_num:
        print(f'Error, scene {scene} not in this dataset, please check')
        continue
    scenes.append(scene)
    n_frame += int(scene_num[scene] / extract)
    n_frames.append(n_frame)


filename = '/mnt/afs/dlp_data/navitopo_v3/merge/date_split.json'
with open(filename, 'r') as f:
    split = json.load(f)
for scene in scenes:
    os.makedirs(osp.join('./work_dirs/tmp', scene))
    maxVisFrame = 100
    scene_num = {
        'test': maxVisFrame
    }
    curVisFrame = 0
    test_0 = []
    data_split = {}
    filename = osp.join(file_dir, 'scene_data', f'{scene}_{0}.json')
    with open(filename, 'r') as f:
        tmp = json.load(f)
    print(filename, len(tmp))
    randidx = [random.randint(0, len(tmp)-1) for _ in range(maxVisFrame)]
    for i in randidx:
        test_0.append([tmp[i][0], tmp[i][1], tmp[i][2]])
        data_split[tmp[i][0]] = split[tmp[i][0]]
    
    filename = osp.join('./work_dirs/tmp', scene, 'test_0.json')
    with open(filename, 'w') as f:
        json.dump(test_0, f)
    
    filename = osp.join('./work_dirs/tmp', scene, 'scene_num.json')
    with open(filename, 'w') as f:
        json.dump(scene_num, f)
    
    filename = osp.join('./work_dirs/tmp', scene, 'date_split.json')
    with open(filename, 'w') as f:
        json.dump(data_split, f)