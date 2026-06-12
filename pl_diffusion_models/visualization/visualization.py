import os.path as osp
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import math
import random
from datetime import datetime, timezone
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.base import CAP_STYLE
import yaml
from visualization.navi_link_action_labels import format_main_action, format_assist_action, truncate_label

color_cv2 = {'red' : (0, 0, 205), 'orange' : (0, 140, 255), 'yellow' : (0, 215, 255), 'green' : (170,  205, 102), 'blue' : (255, 0, 0 ), 'purple' : (226, 43, 138), 'pink' : (180, 110, 255), 'grey' : (190, 190, 190), 'dark_grey' : (105, 105, 105), 'violet' : (238, 130, 238), 'dark_violet' : (211, 0, 148), 'black' : (0, 0, 0), 'white' : (255, 255, 255), 'brown' : (63, 133, 205), 'light_blue' : (255, 245, 0), 'grey_blue' : (200, 180, 160), 'olive' : (35, 142, 107), 'gold' :(0, 215, 255)}
color_traj_multimode = [
    color_cv2["red"],
    color_cv2["purple"],
    color_cv2["yellow"],
    color_cv2["dark_grey"],
    color_cv2["pink"],
    color_cv2["light_blue"],
    color_cv2["violet"],
    color_cv2["gold"],
]
traj_end_maker = [
    cv2.MARKER_CROSS,
    cv2.MARKER_TILTED_CROSS,
    cv2.MARKER_STAR,
    cv2.MARKER_SQUARE,
    cv2.MARKER_DIAMOND,
    cv2.MARKER_TRIANGLE_UP,
    cv2.MARKER_TRIANGLE_DOWN,
    None  # use circle
]
color_agent_multimode = [color_cv2['red'], color_cv2['purple'], color_cv2['yellow'], color_cv2['dark_grey'], color_cv2['pink'], color_cv2['light_blue']]
idx_intention_map = {
    0: 'left_turn_light_signal',
    1: 'without_turn_light_signal',
    2: 'right_turn_light_signal',
    3: 'left_lane_change_efficient',
    4: 'right_lane_change_efficient',
}
# for dataset

def visualization_dataset(res, data_feaNotEmb, rand_idx, img_dir, status='raw', write_img=True, draw_pred=False, fix_dist=False, length=1500, width=1200, agent_highlight_mask = None, laneline_highlight_mask=None, custom_note = None):
    # initialize_plot
    '''
    definition of cv2 canvas +x and +y
        ____________     ->x
    |            |
    |____________|
    |
    y
    
    definition of GAC +x and +y
    y
    |____________    
    |            |
    |____________| ->x
    '''
    # max_idx = torch.argmax(res['pred_prob_fixed'])
    # traj =  res['pred_traj_fixed'][max_idx]
    # min_dist_to_edge = 100
    # for i in range(50):
    #     if res['laneline_mask'][i]:
    #         continue
    #     if res['laneline_attrs'][i][-1] != 0:
    #         continue
    #     if res['laneline_attrs'][i][2] != 10:
    #         continue
    #     dist_matrix = torch.cdist(res['laneline_pts'][i], traj)
    #     min_dist_to_edge = min(min_dist_to_edge, torch.min(dist_matrix))
    #     # if torch.min(dist_matrix) < 5:
    #     #     print(res['laneline_attrs'][i])
    #     #     print(res['laneline_pts'][i])
    #     #     if rand_idx == 81 and res['laneline_attrs'][i][2] == 10:
    #     #         import pdb; pdb.set_trace()
    # print(rand_idx)
    # print(min_dist_to_edge)
    # print('-' * 10)
    with open('./config/dataset/LitDiffusionDataset_ego_navi_fix_distance_path_oss_dlp.yaml', 'r', encoding='utf-8') as file:
        conf = yaml.safe_load(file)
    planning_interval = conf['planning_interval']
    planning_interval_fixed = conf['planning_interval_fixed']
    if 'pred_traj' in res:
        multi_mode = (res['pred_traj'].shape[0] > 1)
    else:
        multi_mode = None
    
    if not multi_mode:
        pixel_per_meter = 6     # 1m -> 6 pixel
    else:
        pixel_per_meter = 10
    
    img = np.full((width, length, 3), 255, dtype="uint8")
    initialize_plot(img, width, length, rand_idx)
    if custom_note is not None:
        draw_label(img, (30, 180), custom_note, font_scale=0.5)
    # draw ego car, 
    #draw_ego(pixel_per_meter, img, res['ego_curr_status'], res['timestamp'], length, width)
    #res['egolight_ori'], navitopo_pts_ori, del_accLight_mask
    draw_ego(pixel_per_meter, img, res['ego_curr_status'], res['timestamp'], data_feaNotEmb['egolight_ori'], length, width)
    if 'agents_importance' in res:
        agents_importance = res['agents_importance']
    else:
        agents_importance = None
    if 'laneline_importance' in res:
        laneline_importance = res['laneline_importance']
    elif laneline_highlight_mask is not None:
        laneline_importance = torch.from_numpy(laneline_highlight_mask.astype(float)[..., np.newaxis])
    else:
        laneline_importance = None

    if 'reconstructed_laneline' in res:
        reconstructed_laneline = None#res['reconstructed_laneline']
    else:
        reconstructed_laneline = None
    # draw agents
    draw_rotate_obj(pixel_per_meter, img, res['agent_status'], res['agent_attrs'], res['agent_time_mask'], length, width, agents_importance = agents_importance, agent_highlight_mask = agent_highlight_mask, draw_traj=False)

    # draw lanelines
    draw_laneline(pixel_per_meter, img, res['laneline_pts'], res['laneline_attrs'], res['laneline_mask'], length, width, laneline_importance = laneline_importance, reconstructed_laneline = reconstructed_laneline)
    draw_navitopo(pixel_per_meter, img, res, data_feaNotEmb, length, width)
    draw_route(pixel_per_meter, img, res['route_pts'], length, width)
    # draw ego traj
    if "sdk_pred_traj" not in res.keys():
        if fix_dist:
            draw_trajectory(pixel_per_meter, img, res['ego_future_status_fixed'], res['ego_future_mask_fixed'], length, width, color_cv2['yellow'])
        else:
            # draw_trajectory(pixel_per_meter, img, res['ego_future_status'], res['ego_future_mask'], length, width, color_cv2['blue'], draw_pred=True, start_y_pixel=600)
            draw_trajectory(pixel_per_meter, img, res['ego_future_status'], res['ego_future_mask'], length, width, color_cv2['blue'])
            draw_trajectory(pixel_per_meter, img, res['ego_future_status_fixed'], res['ego_future_mask_fixed'], length, width, color_cv2['yellow'])
        if draw_pred:
            if fix_dist:
                draw_trajectory_multimode_fixed(pixel_per_meter, img, res['pred_traj_fixed'], res['pred_prob_fixed'], res['pred_prob'], res['pred_v'], np.ones(res['pred_traj_fixed'].shape[1]), length, width, color_cv2['purple'], planning_interval, draw_pred=True, start_y_pixel=150, topk=1, ego_gt=res['ego_future_status_fixed'])
            else:
                if 'pred_prob' not in res:
                    draw_trajectory(pixel_per_meter, img, res['pred_traj'], np.ones(res['pred_traj'].shape[0]), length, width, color_cv2['red'], draw_pred=True, start_y_pixel=150)
                else:
                    if not multi_mode:
                        draw_trajectory_multimode(pixel_per_meter, img, res['pred_traj'], res['pred_prob'], res['pred_prob'], res['pred_v'], res['occ_map'], res['occ_polygons_pts'], res['occ_polygons_attrs'], res['occ_polygons_mask'], np.ones(res['pred_traj'].shape[1]), length, width, [color_cv2['red']], planning_interval, draw_pred=True, start_y_pixel=150, topk=1, ego_gt=res['ego_future_status'])
                        draw_trajectory_multimode_fixed(pixel_per_meter, img, res['pred_traj_fixed'], res['pred_prob_fixed'], res['pred_prob'], res['pred_v'], np.ones(res['pred_traj_fixed'].shape[1]), length, width, color_cv2['purple'], planning_interval_fixed, draw_pred=True, start_y_pixel=150, topk=1, ego_gt=res['ego_future_status_fixed'], detail=False)
                    else:
                        draw_trajectory_multimode(pixel_per_meter, img, res['pred_traj'], res['pred_prob'], res['pred_prob'], res['pred_v'], res['occ_map'], res['occ_polygons_pts'], res['occ_polygons_attrs'], res['occ_polygons_mask'], np.ones(res['pred_traj'].shape[1]), length, width, color_traj_multimode, planning_interval, draw_pred=True, start_y_pixel=150, topk=1, traj_point_sample=2, ego_gt=res['ego_future_status'])
                # if 'agent_prediction' in res:
                #     draw_agent_trajctory(pixel_per_meter, img, res['agent_prediction'], res['agent_time_mask'], length, width, color_cv2['red'])
                #agent先关了
                #if 'agent_prediction' in res:
                #    draw_agent_trajctory(pixel_per_meter, img, res['agent_prediction'], res['agent_time_mask'], length, width, color_cv2['red'], img_dir, rand_idx)
                #elif 'agent_prediction_multimode_cls' in res:
                #    draw_agent_trajctory_multimode(pixel_per_meter, img, res['agent_prediction_multimode_cls'], res['agent_prediction_multimode_reg'], res['agent_time_mask'], length, width, color_agent_multimode, img_dir, rand_idx, thickness=-1, topK=1)
        
        # if 'description' in res:
        #     print(1)
        #     draw_label(img, (20, width - 100), res['description'], (0,0,0), 0.5)
    else:
        img_sdk = img.copy()
        res['occ_map'] = None
        if not multi_mode:
            draw_trajectory_multimode(pixel_per_meter, img, res['pred_traj'], res['pred_prob'], res['pred_prob'], res['pred_v'], res['occ_map'], np.ones(res['pred_traj'].shape[1]), length, width, [color_cv2['red']], planning_interval, draw_pred=True, start_y_pixel=150, topk=1)
            draw_trajectory_multimode(pixel_per_meter, img_sdk, res['sdk_pred_traj'], res['sdk_pred_prob'], res['sdk_pred_prob'], res['sdk_pred_v'], res['occ_map'], np.ones(res['sdk_pred_traj'].shape[1]), length, width, [color_cv2['blue']], planning_interval, draw_pred=True, start_y_pixel=150, topk=1)
        else:
            draw_trajectory_multimode(pixel_per_meter, img, res['pred_traj'], res['pred_prob'], res['pred_prob'], res['pred_v'], res['occ_map'], np.ones(res['pred_traj'].shape[1]), length, width, color_traj_multimode, planning_interval, draw_pred=True, start_y_pixel=150, topk=1, traj_point_sample=2)
            draw_trajectory_multimode(pixel_per_meter, img_sdk, res['sdk_pred_traj'], res['sdk_pred_prob'], res['sdk_pred_prob'], res['sdk_pred_v'], res['occ_map'], np.ones(res['sdk_pred_traj'].shape[1]), length, width, color_traj_multimode, planning_interval, draw_pred=True, start_y_pixel=150, topk=1, traj_point_sample=2)
        img = np.hstack((img, img_sdk))

    if write_img:
        map_type = res['map_type']
        if not osp.exists(img_dir):
            os.makedirs(img_dir)
        if status == "infer_multimode":
            draw_fixtime_trajs(res['pred_traj'], res['pred_prob'], osp.join(img_dir, f'{rand_idx}_{status}_fixtime.png'))
        else:
            if fix_dist:
                img_name = osp.join(img_dir, f'{rand_idx}_{status}_{map_type}_fixdist.png')
            else:
                img_name = osp.join(img_dir, f'{rand_idx}_{status}_{map_type}_fixtime.png')
            cv2.imwrite(img_name, img)
    return img


def visualization_compare(res, rand_idx, traj_fixed=False):
    # initialize_plot
    '''
    definition of cv2 canvas +x and +y
        ____________     ->x
    |            |
    |____________|
    |
    y
    
    definition of GAC +x and +y
    y
    |____________    
    |            |
    |____________| ->x
    '''
    length = 1500
    width = 1200 
    pixel_per_meter = 6     # 1m -> 6 pixel
    
    img = np.full((width, length, 3), 255, dtype="uint8")
    initialize_compare(img, width, length, rand_idx)

    # draw ego car, 
    draw_ego(pixel_per_meter, img, res['ego_curr_status'], data_feaNotEmb['egolight_ori'], length, width)

    # draw agents
    draw_rotate_obj(pixel_per_meter, img, res['agent_status'], res['agent_attrs'], res['agent_time_mask'], length, width, draw_traj=False)

    # draw lanelines
    draw_laneline(pixel_per_meter, img, res['laneline_pts'], res['laneline_attrs'], res['laneline_mask'], length, width)

    # draw ego traj
    if traj_fixed:
        draw_trajectory_multimode(pixel_per_meter, img, res['pred_traj_fixed'], res['pred_prob_fixed'], res['pred_prob'], res['pred_v'], np.ones(res['pred_traj_fixed'].shape[1]), length, width, color_cv2['red'], draw_pred=True, start_y_pixel=150, topk=1)
        draw_trajectory_multimode(pixel_per_meter, img, res['sdk_pred_traj_fixed'], res['sdk_pred_prob_fixed'], res['sdk_pred_prob'], res['sdk_pred_v'], np.ones(res['sdk_pred_traj_fixed'].shape[1]), length, width, color_cv2['blue'], draw_pred=True, start_y_pixel=150, edge_x_pixel=600, topk=1)
        draw_trajectory(pixel_per_meter, img, res['sdk_post_traj'], np.ones(res['sdk_post_traj'].shape[0]), length, width, color_cv2['green'], draw_pred=False, edge_x_pixel=700)
        # draw_agent_trajctory(pixel_per_meter, img, res['agent_prediction'], res['agent_time_mask'], length, width, color_cv2['red'])
    else:
        draw_trajectory_multimode(pixel_per_meter, img, res['pred_traj'], res['pred_prob'], res['pred_prob'], res['pred_v'], np.ones(res['pred_traj'].shape[1]), length, width, color_cv2['red'], draw_pred=True, start_y_pixel=150, topk=1)
        draw_trajectory_multimode(pixel_per_meter, img, res['sdk_pred_traj'], res['sdk_pred_prob'], res['sdk_pred_prob'], res['sdk_pred_v'], np.ones(res['sdk_pred_traj'].shape[1]), length, width, color_cv2['blue'], draw_pred=True, start_y_pixel=150, edge_x_pixel=600, topk=1)
        draw_trajectory(pixel_per_meter, img, res['sdk_post_traj'], np.ones(res['sdk_post_traj'].shape[0]), length, width, color_cv2['green'], draw_pred=False, edge_x_pixel=700)
        # draw_agent_trajctory(pixel_per_meter, img, res['agent_prediction'], res['agent_time_mask'], length, width, color_cv2['red'])
    return img


def initialize_plot(img, width, length, rand_idx):
    draw_label(img, (30, 30), 'frame: ' + str(rand_idx), font_scale=0.5)
    
    for i in range(3):
        cv2.circle(img, (length - 100 + i*5, 30), 2, color_cv2['dark_violet'], -1)
    draw_label(img, (length - 80, 35), 'hist traj', font_scale=0.5)

    for i in range(3):
        cv2.circle(img, (length - 100 + i*5, 60), 2, color_cv2['blue'], -1)
    draw_label(img, (length - 80, 65), 'gt traj', font_scale=0.5)

    for i in range(3):
        cv2.circle(img, (length - 100 + i*5, 90), 2, color_cv2['red'], -1)
    draw_label(img, (length - 80, 95), 'pred traj', font_scale=0.5)

    for i in range(3):
        cv2.circle(img, (length - 100 + i*5, 120), 2, color_cv2['yellow'], -1)
    draw_label(img, (length - 80, 125), 'gt_fixed', font_scale=0.5)

    for i in range(3):
        cv2.circle(img, (length - 100 + i*5, 150), 2, color_cv2['purple'], -1)
    draw_label(img, (length - 80, 155), 'fixed traj', font_scale=0.5)

    return


def initialize_compare(img, width, length, rand_idx):
    draw_label(img, (30, 30), 'frame: ' + str(rand_idx), font_scale=0.5)
    
    # for i in range(3):
    #     cv2.circle(img, (length - 100 + i*5, 30), 2, color_cv2['dark_violet'], -1)
    # draw_label(img, (length - 80, 35), 'hist traj', font_scale=0.5)

    for i in range(3):
        cv2.circle(img, (length - 100 + i*5, 60), 2, color_cv2['blue'], -1)
    draw_label(img, (length - 80, 65), 'sdk traj', font_scale=0.5)

    for i in range(3):
        cv2.circle(img, (length - 100 + i*5, 90), 2, color_cv2['red'], -1)
    draw_label(img, (length - 80, 95), 'torch traj', font_scale=0.5)

    for i in range(3):
        cv2.circle(img, (length - 100 + i*5, 120), 2, color_cv2['green'], -1)
    draw_label(img, (length - 80, 125), 'post traj', font_scale=0.5)

    return


def draw_ego(pixel_per_meter, img, ego_curr_status, timestamp, egolight_ori, length, width, text_offset_y=0, view_range=None):
    # assuming length=4m and width=2m
    # text_offset_y: 左上角 v / yaw_rate 的垂直偏移，避免与 path_compare 的 path L2 等文字重叠
    # view_range: (x_min, x_max, y_min, y_max) 时自车 (0,0) 按该视野映射到像素
    if view_range is None:
        cx, cy = length / 2, width / 2
    else:
        x_min, x_max, y_min, y_max = view_range
        cx = (0 - x_min) * pixel_per_meter
        cy = (y_max - 0) * pixel_per_meter
    hw, hh = 4 * pixel_per_meter / 2, 2 * pixel_per_meter / 2
    cv2.rectangle(img, (int(cx - hw), int(cy + hh)), (int(cx + hw), int(cy - hh)), (0, 0, 0), 2)
    ego_v = round(ego_curr_status[0].item(), 4)
    ego_yaw_rate = round(ego_curr_status[1].item(), 4)
    traffic_light = ego_curr_status[2].item()   # 0: unknown, 1: invalid, 2: off, 3: green, 4: yellow, 5: red
    ego_light = egolight_ori.item()       # 1: off, 2: left, 3: right; 0 reserved for agent light
    ts = np.array(timestamp) / 1e9
    utc_time = datetime.fromtimestamp(ts, tz=timezone.utc)
    draw_label(img, (30, 60 + text_offset_y), 'v: ' + str(ego_v), font_scale=0.5)
    draw_label(img, (30, 90 + text_offset_y), 'yaw_rate: ' + str(ego_yaw_rate), font_scale=0.5)
    draw_label(img, (30, 210), 'timestamp: ' + str(utc_time), font_scale=0.5)
    color_left = color_right = color_cv2['grey']
    if ego_light == 2:
        # turn left
        color_left = color_cv2['red']
    elif ego_light == 3:
        # turn right
        color_right = color_cv2['red']
    cv2.polylines(img, [np.array([[100,120], [100,170], [50,145]])], isClosed=True, color=color_left, thickness=3)
    cv2.polylines(img, [np.array([[120,120], [120,170], [170,145]])], isClosed=True, color=color_right, thickness=3)

    # draw_label(img, (30, 200), 'traffic light ', font_scale=0.5)
    if traffic_light == 3:
        # 绿灯
        # import pdb; pdb.set_trace()
        cv2.circle(img, (100, 200), 10, color_cv2['green'], -1)
    elif traffic_light == 5:
        # 红灯
        # import pdb; pdb.set_trace()
        cv2.circle(img, (100, 300), 10, color_cv2['red'], -1)
    # if ego_light == 2:


def draw_rotate_obj(pixel_per_meter, img, agents_status, agent_attrs, agents_mask, length, width, agents_importance = None, agent_highlight_mask = None, draw_traj=True, mode=0, idx=-1):
    # mode is for agent debug, 0, 1, 2 # 0: draw all, 1: draw only the agent with idx, 2: draw all agent pos but only the agent pos with idx
    max_n_agent = agents_mask.shape[0]
    if agents_importance is not None:
        # agent_highlight_mask = agents_importance.squeeze(-1)>0.4
        draw_label(img, (30, 120), 'max_agent_importance: ' + str(max(agents_importance.squeeze(-1)).item()), font_scale=0.5)
        max_idx = torch.argmax(agents_importance.squeeze(-1)).item()
        agents_risk_score = agents_importance.squeeze(-1)

    for i in range(max_n_agent):
        if not agents_mask[i, 20]:
            return
        if mode == 1 and i != idx:
            continue
        x, y, vx, vy, yaw, score =  agents_status[i, 20, ]
        size_y, size_x, cls = agent_attrs[i]
        corners = box_center_to_corners(x, y, size_y, size_x, yaw)
        for j in range(4):
            corners[j][0] = int(corners[j][0] * pixel_per_meter + length/2)
            corners[j][1] = int(width/2 - corners[j][1] * pixel_per_meter)
        x = int(x * pixel_per_meter + length/2)
        y = int(width/2 - y * pixel_per_meter)
        if (agents_importance is not None and max_idx == i) or (agent_highlight_mask is not None and agent_highlight_mask[i]):
            draw_label(img, (x, y+30), 'agent_v: ' + str(round(math.sqrt(vx**2+vy**2),3)), font_scale=0.5)
        vx = int(x + vx * 2)
        vy = int(y - vy * 2)
        if agents_importance is not None:
            #color = agents_risk_score[i].item() * color_cv2['red'] +  (1 - agents_risk_score[i]) * color_cv2['green']
            risk_color_portion = 1 - (agents_risk_score[i].item() - 1) ** 2
            color_red = tuple(int(risk_color_portion * c) for c in color_cv2['red'])
            color_green = tuple(int((1-risk_color_portion) * c) for c in color_cv2['green'])
            color = tuple(a + b for a, b in zip(color_red, color_green))

            cv2.fillPoly(img, np.int32([corners]), color=color)
        elif agent_highlight_mask is not None and agent_highlight_mask[i]:
            color = color_cv2['red']
        else:
            color = color_cv2['green']
        cv2.polylines(img, np.int32([corners]), isClosed=True, color=color, thickness=2)
        cv2.arrowedLine(img, (x, y), (vx, vy), color, 2)

        if mode == 2 and i != idx:
            continue

        if draw_traj:
            for j in range(agents_mask.shape[1]):
                if agents_mask[i][j]:
                    x, y = agents_status[i, j, :2]
                    x = int(x * pixel_per_meter + length/2)
                    y = int(width/2 - y * pixel_per_meter)
                    if j <= 20:
                        # agent history
                        color = color_cv2['dark_violet']
                    else:
                        # agent future
                        color = color_cv2['blue']
                    cv2.circle(img, (x, y), 1, color, -1)
        if mode == 1 and i == idx:
            break
def _world_to_pixel_path_compare(x, y, length, width, pixel_per_meter, view_range):
    """世界坐标 (x,y) 转像素。view_range=(x_min, x_max, y_min, y_max) 时按视野范围映射，否则按原逻辑（原点在画布中心）。"""
    if view_range is None:
        px = x * pixel_per_meter + length / 2
        py = width / 2 - y * pixel_per_meter
    else:
        x_min, x_max, y_min, y_max = view_range
        px = (x - x_min) * pixel_per_meter
        py = (y_max - y) * pixel_per_meter
    return int(px), int(py)


def draw_navitopo_path_compare(pixel_per_meter, img, res, data_feaNotEmb, length, width, view_range=None):
    """
    在 path_compare 中作为底层绘制 navitopo：浅蓝色细线，不遮挡 path/route。
    使用 navitopo_pts_ori（若有）以在 infer 时仍显示原始 navitopo。
    """
    navitopo_mask = res.get('navitopo_mask')
    if navitopo_mask is None:
        return
    # 优先用原始 navitopo（infer 时 model_input 可能被置零，res 或 data_feaNotEmb 中可能存了 ori）
    navitopo_pts = res.get('navitopo_pts_ori')
    if navitopo_pts is None:
        navitopo_pts = data_feaNotEmb.get('navitopo_pts_ori')
    if navitopo_pts is None:
        navitopo_pts = res.get('navitopo_pts')
    if navitopo_pts is None:
        return
    navitopo_pts = navitopo_pts.cpu().numpy() if torch.is_tensor(navitopo_pts) else np.asarray(navitopo_pts)
    navitopo_mask = navitopo_mask.cpu().numpy() if torch.is_tensor(navitopo_mask) else np.asarray(navitopo_mask)
    # mask 0 表示有效；只遍历 pts 与 mask 共有的数量，避免越界
    n_poly = min(navitopo_pts.shape[0], navitopo_mask.shape[0])
    n_pt = navitopo_pts.shape[1]
    color = color_cv2['grey_blue']
    thickness = 1
    for i in range(n_poly):
        if navitopo_mask[i]:
            continue
        if np.all(navitopo_pts[i] == 0):
            continue
        xs = navitopo_pts[i, :, 0]
        ys = navitopo_pts[i, :, 1]
        for j in range(n_pt - 1):
            x1, y1 = xs[j], ys[j]
            x2, y2 = xs[j + 1], ys[j + 1]
            x1_i, y1_i = _world_to_pixel_path_compare(x1, y1, length, width, pixel_per_meter, view_range)
            x2_i, y2_i = _world_to_pixel_path_compare(x2, y2, length, width, pixel_per_meter, view_range)
            cv2.line(img, (x1_i, y1_i), (x2_i, y2_i), color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_route(pixel_per_meter, img, res, length, width, view_range=None):
    route_pts = res
    route_fixed_pts = route_pts.shape[1]
    for i in range(route_pts.shape[0]):
        xs = route_pts[i, :, 0]
        ys = route_pts[i, :, 1]
        color = color_cv2['light_blue']

        for j in range(0, route_fixed_pts - 1, 2):
            x1, y1 = xs[j], ys[j]
            x2, y2 = xs[j+1], ys[j+1]
            x1, y1 = _world_to_pixel_path_compare(x1, y1, length, width, pixel_per_meter, view_range)
            x2, y2 = _world_to_pixel_path_compare(x2, y2, length, width, pixel_per_meter, view_range)
            cv2.line(img, (x1 + i*2, y1), (x2 + i*2, y2), color, thickness=2, lineType=cv2.LINE_AA)


def draw_path_only(pixel_per_meter, img, path_pts, path_mask, length, width, color=None, thickness=2, view_range=None):
    """
    在画布上仅绘制 fix_distance path 折线（path GT 或 path pred）。
    path_pts: (N, 2) 自车坐标系下 x,y，tensor 或 numpy
    path_mask: (N,) 有效为 True/1，无效为 False/0
    view_range: (x_min, x_max, y_min, y_max) 时按该视野范围映射世界坐标到像素
    """
    if color is None:
        color = color_cv2['green']
    path_pts = path_pts.cpu().numpy() if torch.is_tensor(path_pts) else np.asarray(path_pts)
    path_mask = path_mask.cpu().numpy() if torch.is_tensor(path_mask) else np.asarray(path_mask)
    if path_mask.dtype != bool:
        path_mask = path_mask.astype(bool)
    n = path_pts.shape[0]
    for j in range(n - 1):
        if not (path_mask[j] and path_mask[j + 1]):
            continue
        x1, y1 = path_pts[j, 0], path_pts[j, 1]
        x2, y2 = path_pts[j + 1, 0], path_pts[j + 1, 1]
        x1_i, y1_i = _world_to_pixel_path_compare(x1, y1, length, width, pixel_per_meter, view_range)
        x2_i, y2_i = _world_to_pixel_path_compare(x2, y2, length, width, pixel_per_meter, view_range)
        cv2.line(img, (x1_i, y1_i), (x2_i, y2_i), color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_navi_link_actions_path_compare(img, res, img_w_px, img_h_px, line_h=24, font_scale=0.5, max_show=18, max_dist_m=200.0):
    """
    绘制 data_sdmap.navi_link_actions：累计距离 accumulated_dist、主/辅动作（proto LinkAction）。
    仅展示前方 max_dist_m（默认 200m）内的条目；与 dataset 侧过滤一致，旧 pickle 亦在此二次裁剪。
    """
    if 'navi_link_actions_dist' not in res or 'navi_link_actions_mask' not in res:
        return
    dist = res['navi_link_actions_dist']
    main = res['navi_link_actions_main']
    assist = res['navi_link_actions_assist']
    mask = res['navi_link_actions_mask']
    dist = dist.detach().cpu().numpy().flatten() if torch.is_tensor(dist) else np.asarray(dist).flatten()
    main = main.detach().cpu().numpy().flatten() if torch.is_tensor(main) else np.asarray(main).flatten()
    assist = assist.detach().cpu().numpy().flatten() if torch.is_tensor(assist) else np.asarray(assist).flatten()
    mask = mask.detach().cpu().numpy().flatten() if torch.is_tensor(mask) else np.asarray(mask).flatten()
    mask = mask.astype(bool)
    raw_idx = np.flatnonzero(mask)
    idxs = np.array(
        [int(j) for j in raw_idx.tolist() if 0.0 <= float(dist[j]) <= max_dist_m],
        dtype=np.int64,
    )
    if idxs.size == 0:
        n_lines = 2
    else:
        n_show = min(idxs.size, max_show)
        n_lines = 2 + n_show + (1 if idxs.size > max_show else 0)
    y_start = int(img_h_px) - 12 - n_lines * line_h
    y_start = max(10, y_start)
    x0 = 10
    y = y_start
    draw_label(img, (x0, y), f'NaviLinkActions (<={max_dist_m:.0f}m, dist / main / assist):', font_scale=font_scale)
    y += line_h
    if idxs.size == 0:
        draw_label(img, (x0, y), '  (none within range)', font_scale=font_scale)
        return
    shown = 0
    for j in idxs.tolist():
        if shown >= max_show:
            break
        d, m, a = float(dist[j]), int(main[j]), int(assist[j])
        sm = format_main_action(m)
        sa = format_assist_action(a)
        line = f'  [{j}] dist={d:.1f}m {sm} / {sa}'
        draw_label(img, (x0, y), truncate_label(line, 90), font_scale=font_scale)
        y += line_h
        shown += 1
    if idxs.size > max_show:
        draw_label(img, (x0, y), f'  ... ({idxs.size - max_show} more)', font_scale=font_scale)


def compute_path_loss(res):
    """
    计算 path 预测与 GT 在有效步上的平均 L2 距离。
    返回 float 标量；无可用数据则返回 None。
    """
    if 'pred_traj_fixed' not in res or 'ego_future_status_fixed' not in res or 'ego_future_mask_fixed' not in res:
        return None
    gt = res['ego_future_status_fixed'][:, :2]
    mask = res['ego_future_mask_fixed']
    pred = res['pred_traj_fixed']
    if pred.ndim == 3:
        pred = pred[0]
    gt = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
    mask = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    pred = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    valid = mask.astype(bool)
    if not np.any(valid):
        return None
    T = min(gt.shape[0], pred.shape[0])
    gt = gt[:T]
    pred = pred[:T]
    valid = valid[:T]
    diff = pred - gt
    l2_per_point = np.sqrt((diff ** 2).sum(axis=-1))
    l2_per_point[~valid] = 0
    cnt = valid.sum()
    if cnt == 0:
        return None
    return float((l2_per_point.sum() / cnt))


def visualization_path_compare(res, data_feaNotEmb, rand_idx, img_dir, length=1500, width=1200, pixel_per_meter=6, write_img=True, path_loss=None):
    """
    Path GT vs Path Pred 画在一张图上对比。
    视野范围：x (-30, 200) m，y (-100, 100) m。
    颜色与当前一致：path GT = yellow (gt_fixed)，path pred = purple (fixed traj)。
    保存到 img_dir：{rand_idx}_path_compare.png
    """
    x_min, x_max, y_min, y_max = -30.0, 200.0, -100.0, 100.0
    length = int((x_max - x_min) * pixel_per_meter)
    width = int((y_max - y_min) * pixel_per_meter)
    view_range = (x_min, x_max, y_min, y_max)

    img = np.full((width, length, 3), 255, dtype="uint8")
    draw_label(img, (30, 30), f'frame: {rand_idx}', font_scale=0.5)

    if path_loss is None:
        path_loss = compute_path_loss(res)
    if path_loss is not None:
        draw_label(img, (30, 60), f'path L2: {path_loss:.4f}', font_scale=0.5)

    # 在 path_compare 图中展示 route_lane_attrs 信息，便于对齐高德 laneInfo 车道属性
    if 'route_lane_attrs' in res:
        route_lane_attrs = res['route_lane_attrs']  # 形状通常为 [1, max_lanes, 4]
        if isinstance(route_lane_attrs, torch.Tensor):
            route_lane_attrs_np = route_lane_attrs.detach().cpu().numpy()
        else:
            route_lane_attrs_np = np.asarray(route_lane_attrs)
        # 兼容 [1, N, 4] 或 [N, 4]
        if route_lane_attrs_np.ndim == 3:
            lane_attrs = route_lane_attrs_np[0]
        else:
            lane_attrs = route_lane_attrs_np
        # 放到画面中上部区域，避免与左上角 frame/path L2 文本重叠
        base_x = int(length * 0.35)
        base_y = 30
        draw_label(img, (base_x, base_y), 'route_lane_attrs: [valid, rec, type, bus]', font_scale=0.5)
        for lane_idx in range(lane_attrs.shape[0]):
            valid, rec, lane_type, is_bus = lane_attrs[lane_idx]
            if valid <= 0.0:
                continue
            text = f'lane {lane_idx}: v={int(valid)}, r={int(rec)}, t={int(lane_type)}, bus={int(is_bus)}'
            draw_label(img, (base_x, base_y + 20 * (lane_idx + 1)), text, font_scale=0.4)

    # SDMap NaviLinkActions：累计距离 + 主/辅导航动作（proto LinkAction）
    draw_navi_link_actions_path_compare(img, res, length, width)

    # text_offset_y=30 使 v / yaw_rate 显示在 path L2 下方，避免与 (30,60) 的 path L2 重叠
    draw_ego(pixel_per_meter, img, res['ego_curr_status'], res['timestamp'], data_feaNotEmb['egolight_ori'], length, width, text_offset_y=30, view_range=view_range)
    # navitopo 浅蓝色细线画在底层，避免遮挡 path
    draw_navitopo_path_compare(pixel_per_meter, img, res, data_feaNotEmb, length, width, view_range=view_range)
    draw_laneline(pixel_per_meter, img, res['laneline_pts'], res['laneline_attrs'], res['laneline_mask'], length, width, view_range=view_range)
    if 'route_pts' in res and res['route_pts'] is not None:
        draw_route(pixel_per_meter, img, res['route_pts'], length, width, view_range=view_range)

    # 动态障碍物（车辆/行人/自行车）
    if 'agent_status' in res and 'agent_attrs' in res and 'agent_time_mask' in res:
        _agent_status = res['agent_status']
        _agent_attrs = res['agent_attrs']
        _agent_mask = res['agent_time_mask']
        if torch.is_tensor(_agent_status):
            _agent_status = _agent_status.detach().cpu().numpy()
        if torch.is_tensor(_agent_attrs):
            _agent_attrs = _agent_attrs.detach().cpu().numpy()
        if torch.is_tensor(_agent_mask):
            _agent_mask = _agent_mask.detach().cpu().numpy()
        _cls_color = {1: (180, 130, 70), 2: (0, 180, 0), 3: (0, 165, 255)}  # 行人=蓝灰, 车辆=绿, 自行车=橙
        _t_curr = _agent_mask.shape[1] - 1  # 当前帧 index（最后一个历史步）
        for i in range(_agent_status.shape[0]):
            if not _agent_mask[i, _t_curr]:
                break
            x, y, vx, vy, yaw, score = _agent_status[i, _t_curr]
            size_y, size_x, cls = _agent_attrs[i]
            corners = box_center_to_corners(x, y, size_y, size_x, yaw)
            for j in range(4):
                corners[j][0] = int((corners[j][0] - x_min) * pixel_per_meter)
                corners[j][1] = int((y_max - corners[j][1]) * pixel_per_meter)
            color = _cls_color.get(int(cls), (0, 180, 0))
            cv2.polylines(img, np.int32([corners]), isClosed=True, color=color, thickness=2)
            cx_px = int((x - x_min) * pixel_per_meter)
            cy_px = int((y_max - y) * pixel_per_meter)
            vx_px = int(cx_px + vx * 2)
            vy_px = int(cy_px - vy * 2)
            cv2.arrowedLine(img, (cx_px, cy_px), (vx_px, vy_px), color, 1)

    path_gt = res['ego_future_status_fixed'][:, :2]
    path_gt_mask = res['ego_future_mask_fixed']
    draw_path_only(
        pixel_per_meter,
        img,
        path_gt,
        path_gt_mask,
        length,
        width,
        color=color_cv2['yellow'],
        thickness=2,
        view_range=view_range,
    )
    # 仅当该帧实际启用了 stitched GT 时才单独画出橙色线，避免与 raw gt 重叠造成误判。
    use_stitched_path_gt = bool(res.get('use_stitched_path_gt', False))
    stitched_drawn = False
    if use_stitched_path_gt and ('ego_future_status_fixed_train_gt' in res) and ('ego_future_mask_fixed_train_gt' in res):
        stitched_gt = res['ego_future_status_fixed_train_gt'][:, :2]
        stitched_gt_mask = res['ego_future_mask_fixed_train_gt']
    elif use_stitched_path_gt and ('navitopo_rs' in res) and ('navitopo_rs_mask' in res):
        _rs = res['navitopo_rs']
        _rs_mask = res['navitopo_rs_mask']
        # navitopo_rs 可能是 (1+N, T, 2) 多候选格式，只取第 0 路用于可视化
        stitched_gt = _rs[0] if _rs.dim() == 3 else _rs
        stitched_gt_mask = _rs_mask[0] if _rs_mask.dim() == 2 else _rs_mask
    else:
        stitched_gt = None
        stitched_gt_mask = None
    stitched_has_valid = False
    if stitched_gt_mask is not None:
        if torch.is_tensor(stitched_gt_mask):
            stitched_has_valid = bool(stitched_gt_mask.bool().any().item())
        else:
            stitched_has_valid = bool(np.asarray(stitched_gt_mask).astype(bool).any())
    if use_stitched_path_gt and stitched_gt is not None and stitched_gt_mask is not None and stitched_has_valid:
        draw_path_only(
            pixel_per_meter,
            img,
            stitched_gt,
            stitched_gt_mask,
            length,
            width,
            color=color_cv2['orange'],
            thickness=2,
            view_range=view_range,
        )
        stitched_drawn = True

    # 多条预测路径可视化：
    # - 第 0 条：深紫色、稍粗线条
    # - 第 1~N 条：浅紫色、细线条
    if 'pred_traj_fixed' in res and res['pred_traj_fixed'] is not None:
        path_pred = res['pred_traj_fixed']

        # 旧逻辑只画一条：如果是 (N, T, 2)，会直接取 path_pred[0]
        # 现在改为循环画所有 sample_num 条路径
        if path_pred.ndim == 2:
            path_pred = path_pred[None, ...]  # -> (1, T, 2)

        num_paths = path_pred.shape[0]
        for k in range(num_paths):
            curr_path = path_pred[k]
            curr_mask = np.ones(curr_path.shape[0], dtype=bool)

            # 0 号为深紫，其余为浅紫细线
            is_primary = (k == 0)
            color = color_cv2['dark_violet'] if is_primary else color_cv2['violet']
            thickness = 3 if is_primary else 1

            draw_path_only(
                pixel_per_meter,
                img,
                curr_path,
                curr_mask,
                length,
                width,
                color=color,
                thickness=thickness,
                view_range=view_range,
            )

    for i in range(3):
        cv2.circle(img, (length - 100 + i * 5, 90), 2, color_cv2['yellow'], -1)
    draw_label(img, (length - 80, 95), 'path gt', font_scale=0.5)
    for i in range(3):
        cv2.circle(img, (length - 100 + i * 5, 120), 2, color_cv2['purple'], -1)
    draw_label(img, (length - 80, 125), 'path pred', font_scale=0.5)
    if stitched_drawn:
        for i in range(3):
            cv2.circle(img, (length - 100 + i * 5, 150), 2, color_cv2['orange'], -1)
        draw_label(img, (length - 80, 155), 'path gt stitched', font_scale=0.5)

    if write_img and img_dir:
        # 将 path_compare 的 PNG 统一保存在 infer 目录下的子文件夹中，避免和其它文件混在一起
        save_dir = osp.join(img_dir, "path_compare")
        os.makedirs(save_dir, exist_ok=True)
        out_path = osp.join(save_dir, f'{rand_idx}_path_compare.png')
        cv2.imwrite(out_path, img)
    return img


def draw_navitopo(pixel_per_meter, img, res, data_feaNotEmb, length, width,):
    navitopo_mask_all = res['navitopo_mask']
    navitopo_mask_light = data_feaNotEmb['del_accLight_mask']
    draw_label(img, (30, 200), 'navitopo_mask_light: ' + str(navitopo_mask_light), font_scale=0.5)
    draw_label(img, (30, 250), 'navitopo_mask_all: ' + str(navitopo_mask_all), font_scale=0.5)
    #cls没了，所以navi就暂时置0了，0是离自车最近的
    #ego_future_cls = res['ego_future_cls']
    #closest_navi_last = torch.nonzero(ego_future_cls == 1)[0][0]
    closest_navi_last = 0
    draw_label(img, (30, 300), 'closest_navi_last: ' + str(closest_navi_last), font_scale=0.5)
    ego_light = data_feaNotEmb['egolight_ori'].item()
    draw_label(img, (30, 350), 'ego_light: ' + str(ego_light), font_scale=0.5)
    trainFlag = res.get('trainFlag', False)
    if trainFlag:
        trainFlag = res['trainFlag']
        draw_label(img, (30, 400), 'trainFlag: ' + str(trainFlag), font_scale=0.5)
    navitopo_pts_ori = data_feaNotEmb['navitopo_pts_ori']
    navitopo_pts = res['navitopo_pts']
    draw_label(img, (30, 450),  'len NaviPts Ori: ' + str(navitopo_pts_ori.shape), font_scale=0.5)
    
    n_lanes = min(10, navitopo_pts_ori.shape[0])
    for i in range(n_lanes):
        points = navitopo_pts_ori[i, :5, :]
        
        coord_strings = [f'({x:.2f},{y:.2f})' for x, y in points.tolist()]
        label = f'Lane {i}: {", ".join(coord_strings)}'
        
        draw_label(img, (400, 60 + i * 20), label, font_scale=0.3)

    bool_mask = (navitopo_mask_all == 0)
    navi_pts = navitopo_pts[bool_mask]

    max_n_navitopo = navitopo_mask_all.shape[0]
    laneline_fixed_pts = navi_pts.shape[1]
    for i in range(max_n_navitopo):
        if navitopo_mask_all[i]:
            return
        xs = navitopo_pts[i, :, 0]
        ys = navitopo_pts[i, :, 1]
        if i == closest_navi_last:
            color = color_cv2['green']
        else:
            color = color_cv2['grey']

        for j in range(0, laneline_fixed_pts - 1, 2):
            x1, y1 = xs[j], ys[j]
            x2, y2 = xs[j+1], ys[j+1]
            x1 = int(x1 * pixel_per_meter + length/2)
            y1 = int(width/2 - y1 * pixel_per_meter)
            x2 = int(x2 * pixel_per_meter + length/2)
            y2 = int(width/2 - y2 * pixel_per_meter)
            cv2.circle(img, (x1+i*2, y1), 4, color, 1)
            if j == 10:
                draw_label(img, (x1, y1 + 5), str(i), font_scale=1.5)
            if j < 20:
                label = f"{i}:, x{j}: {navitopo_pts[i, j, 0].item():.2f}, y{j}: {navitopo_pts[i, j, 1].item():.2f}"
                draw_label(img, (30 + i * 300, 700 + j * 10),  label, font_scale=0.5)
    # for i in range(navitopo_pts_ori.shape[0]):
    #     xs = navitopo_pts_ori[i, :, 0]
    #     ys = navitopo_pts_ori[i, :, 1]
    #     for j in range(0, laneline_fixed_pts - 1, 2):
    #         x1, y1 = xs[j], ys[j]
    #         x2, y2 = xs[j+1], ys[j+1]
    #         x1 = int(x1 * pixel_per_meter + length/2)
    #         y1 = int(width/2 - y1 * pixel_per_meter)
    #         x2 = int(x2 * pixel_per_meter + length/2)
    #         y2 = int(width/2 - y2 * pixel_per_meter)
    #         cv2.line(img, (x1, y1), (x2, y2), color_cv2['grey'], 1)


def draw_laneline(pixel_per_meter, img, laneline_pts, laneline_attrs, laneline_mask, length, width, laneline_fixed_pts=200, laneline_importance=None, reconstructed_laneline=None, viz_reconstructed_laneline_threshold=0.4, view_range=None):
    max_n_laneline = laneline_mask.shape[0]
    if laneline_importance is not None:
        draw_label(img, (30, 150), 'max_laneline_importance: ' + str(max(laneline_importance.squeeze(-1)).item()), font_scale=0.5)
        max_idx = torch.argmax(laneline_importance.squeeze(-1)).item()
        laneline_importance_score = laneline_importance.squeeze(-1)
    for i in range(max_n_laneline):
        if laneline_mask[i]:
            return
        xs = laneline_pts[i, :, 0]
        ys = laneline_pts[i, :, 1]
        valid_reconstruct_lane = False
        if reconstructed_laneline is not None and laneline_importance is not None:
            if laneline_importance_score[i].item() > viz_reconstructed_laneline_threshold:
                valid_reconstruct_lane = True
                xr = reconstructed_laneline[i, :, 0]
                yr = reconstructed_laneline[i, :, 1]
                color_r = color_cv2['dark_violet']
        # if laneline_importance is not None and laneline_importance_score[i].item() > 0.3:
        #     #color = agents_risk_score[i].item() * color_cv2['red'] +  (1 - agents_risk_score[i]) * color_cv2['green']
        #     importance_color_portion = 1 - (laneline_importance_score[i].item() - 1) ** 2
        #     color_red = tuple(int(importance_color_portion * c) for c in color_cv2['red'])
        #     color_green = tuple(int((1-importance_color_portion) * c) for c in color_cv2['green'])
        #     color = tuple(a + b for a, b in zip(color_red, color_green))
        if reconstructed_laneline is None and laneline_importance is not None and laneline_importance[i]>0.8:
            color = color_cv2['green']
        elif laneline_attrs[i][1] == 3:
            # centerline
            color = color_cv2['grey']
        elif laneline_attrs[i][1] == 4:
            # stop line
            color = color_cv2['red']
        elif laneline_attrs[i][1] == 5:
            # route
            color = color_cv2['light_blue']
        else:
            # laneline & road edge
            laneline_color, laneline_type, laneline_style = laneline_attrs[i]
            if (laneline_type == 1):
                color = color_cv2['olive']
            elif laneline_color == 2:
                color = color_cv2['yellow']
            else:
                color = color_cv2['black']
        flag_dash = False
        _, _, laneline_style = laneline_attrs[i]
        if laneline_style in [2, 3, 4]:
            flag_dash = True

        for j in range(laneline_fixed_pts - 1):
            x1, y1 = xs[j], ys[j]
            x2, y2 = xs[j+1], ys[j+1]
            x1, y1 = _world_to_pixel_path_compare(x1, y1, length, width, pixel_per_meter, view_range)
            x2, y2 = _world_to_pixel_path_compare(x2, y2, length, width, pixel_per_meter, view_range)
            if j == 0:
                #draw_label(img, (x2, y2), str(round(laneline_importance[i].item(),3)), font_scale=0.5)
                pass
            if flag_dash:
                if j % 2 == 0:
                    cv2.line(img, (x1, y1), (x2, y2), color, 1)
            else:
                cv2.line(img, (x1, y1), (x2, y2), color, 1)
            if j == 0:
                cv2.circle(img, (x1, y1), 2, color, 1)
            elif j == laneline_fixed_pts-2:
                cv2.circle(img, (x2, y2), 2, color, 1)
            if valid_reconstruct_lane:
                xr1, yr1 = xr[j], yr[j]
                xr2, yr2 = xr[j+1], yr[j+1]
                xr1, yr1 = _world_to_pixel_path_compare(xr1, yr1, length, width, pixel_per_meter, view_range)
                xr2, yr2 = _world_to_pixel_path_compare(xr2, yr2, length, width, pixel_per_meter, view_range)

                cv2.line(img, (xr1, yr1), (xr2, yr2), color_r, 1)
                if j == 0:
                    cv2.circle(img, (xr1, yr1), 2, color_r, 1)
                elif j == laneline_fixed_pts-2:
                    cv2.circle(img, (xr2, yr2), 2, color_r, 1)

def plot_trajectory_with_heading_cv2(i, img, trajectory_point, pixel_per_meter, length, width, heading, speed):
    """
    使用OpenCV绘制轨迹点及方向箭头（箭头长度与速度成正比）。

    参数:
        trajectory_points (np.ndarray): 轨迹点坐标，形状 (N, 2)，坐标范围建议归一化到 [0, 1]。
        headings (np.ndarray): 方向角度（弧度），形状 (N,)。
        speeds (np.ndarray): 速度大小，形状 (N,)。
        img_size (tuple): 输出图像大小 (width, height)。
    """

    # # 将轨迹点坐标缩放到图像尺寸
    # pts = (trajectory_points * np.array([img_size[0], img_size[1]])).astype(int)
    x, y = trajectory_point
    x1 = int(x * pixel_per_meter + length/2)
    y1 = int(width/2 - y * pixel_per_meter)

    # 绘制方向箭头
    # 计算箭头终点（根据速度和方向）
    dx = int(torch.cos(heading) * speed)
    dy = int(torch.sin(heading) * speed)
    end_pt = (x1 + dx, y1 + dy)

    # 绘制箭头（蓝色线段+箭头头部）
    cv2.arrowedLine(img, (x1, y1), end_pt, (255, 0, 0), 2, tipLength=0.3)

def draw_trajectory(pixel_per_meter, img, ego_future_status, ego_future_mask, length, width, color, img_dir=None, draw_pred=False, agent_idx=None, rand_idx=None, start_y_pixel=0, edge_x_pixel=250, img_save=False, ego_prob=None, thickness=None):
    # ego_future_status: (n_steps, 2)
    # ego_future_mask: (n_steps)
    circle_size = 2 if draw_pred else 1
    n_future_step = ego_future_mask.shape[0]
    if draw_pred:
        if start_y_pixel < width / 3:
            draw_label(img, (length - 300, start_y_pixel), 'pred:', color=(0,0,0), font_scale=0.5)
    final_idc = 0
    if thickness is None:
        thickness = -1
    for i in range(n_future_step):
        if draw_pred or ego_future_mask[i]:
            final_idc = i
            x, y = ego_future_status[i,:2]
            x1 = int(x * pixel_per_meter + length/2)
            y1 = int(width/2 - y * pixel_per_meter)
            cv2.circle(img, (x1, y1), circle_size, color, thickness=thickness)
            if draw_pred and agent_idx is None and i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39, 49]:
                idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39, 49].index(i)
                pos = (length - edge_x_pixel, start_y_pixel + idx*30)
                label_str = f'{i+1}th: x {round(float(x), 3)}, y {round(float(y), 3)}'
                draw_label(img, pos, label_str, color=(0,0,0), font_scale=0.5)
                # draw pred v , heading
                plot_trajectory_with_heading_cv2(i, img, ego_future_status[i, :2], pixel_per_meter, length, width, ego_future_status[i, 3], ego_future_status[i, 2])
        else:
            break
    if img_save and agent_idx is not None and rand_idx is not None:
        # 6. 保存图片（支持 PNG/JPG 等格式）
        output_path = osp.join(img_dir, f"{rand_idx}_{agent_idx}_trajectory_with_arrow.png")
        cv2.imwrite(output_path, img)  # 保存到当前目录
        print(f"图片已保存至: {output_path}")
    if agent_idx is not None and ego_prob is not None and final_idc>0:
        ego_prob = round(ego_prob.item(), 2)
        label_str = f'{ego_prob}'
        x = ego_future_status[final_idc, 0]
        y = ego_future_status[final_idc, 1]
        x1 = int(x * pixel_per_meter + length/2)
        y1 = int(width/2 - y * pixel_per_meter)
        draw_label(img, (x1, y1), label_str, color=color, font_scale=0.5)
    # for i in range(n_future_step-1):
    #     if ego_future_mask[i] and ego_future_mask[i+1]:
    #         x1, y1 = ego_future_status[i,:2]
    #         x2, y2 = ego_future_status[i+1,:2]
    #         x1 = int(x1 * pixel_per_meter + length/2)
    #         y1 = int(width/2 - y1 * pixel_per_meter)
    #         x2 = int(x2 * pixel_per_meter + length/2)
    #         y2 = int(width/2 - y2 * pixel_per_meter)
    #         cv2.line(img, (x1, y1), (x2, y2), color, 2)
    #         # cv2.circle(img, (x1, y1), 4, color, 2)
    #     else:
    #         break
    return

def draw_fixtime_trajs(traces, probs, savepth): 
    mode_num, future_steps = traces.shape[:2]
    traces = traces.detach().numpy()
    prob_max = probs.max()
    color_list = ['cyan', 'blue', 'black', 'green', 'brown'] # 左灯；无灯；右灯；左效率变道；右效率变道
    plt.figure()
    start, end, step = 0, future_steps, 1
    for j in range(mode_num):
        if j>=30:
            continue
        color = color_list[j//10]
        if probs[j]==prob_max:
            color = "red"
        traj = traces[j, start:end:step]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.2, linewidth=0.5, color=color)
        label_str = f"{j}-{round(float(probs[j]), 2)}"
        plt.text(traj[-1, 0]+random.uniform(-0.5, 0.5), traj[-1, 1]+random.uniform(-0.1, 0.1), label_str, fontsize=2, color=color)
    
    plt.xlim(-10, 80)
    plt.ylim(-15, 15)
    plt.savefig(savepth, dpi=300)

def draw_trajectory_multimode(pixel_per_meter, img, ego_future_status_fixed, ego_future_prob_fixed, ego_future_prob, ego_future_v, occ_map, occ_polygons_pts, occ_polygons_attrs, occ_polygons_mask, ego_future_mask, length, width, color_used, planning_interval, draw_pred=False, start_y_pixel=0, edge_x_pixel=300, topk=1, traj_point_sample=1, ego_gt=None):
    # ego_future_status_fixed: (n_steps, 2)
    # ego_future_mask: (n_steps)
    n_future_step = ego_future_mask.shape[0]
    n_mode = int(ego_future_status_fixed.shape[0])
    circle_size = 1
    if draw_pred:
        if n_mode > 1:
            circle_size = 3
        else:
            circle_size = 2
    # draw anchor_v
    #v是差分的，先注释了
    #anchor_v = torch.norm(ego_future_status_fixed[max_idx, 5 ,:2] - ego_future_status_fixed[max_idx, 3 , :2], dim=-1) / 0.2
    #draw_label(img, (30, 180), 'anchor_v: ' + str(round(anchor_v.item(),2)), font_scale=0.5)

    #转化绝对坐标和速度
    #net的sample里有转化的，这里已经是绝对坐标了
    ego_pred_xy = ego_future_status_fixed

    xyStart = torch.zeros((n_mode, 1, 2), device=ego_pred_xy.device, dtype=ego_pred_xy.dtype)
    xyForCalV = torch.cat([xyStart, ego_pred_xy], dim=1)
    ego_pred_v = torch.norm((xyForCalV[:, 1:, :] - xyForCalV[:, :-1, :]), dim=2) / 0.1 / (planning_interval / 0.1)

    if occ_map.shape[0] == 120 and occ_map.shape[1] == 275:
        occ_map_res = 0.4
        occ_map_ego_i0 = occ_map.shape[0] // 2
        occ_map_ego_j0 = math.floor(10 / occ_map_res + 0.5)
    elif occ_map.shape[0] == 240 and occ_map.shape[1] == 550:
        occ_map_res = 0.2
        occ_map_ego_i0 = occ_map.shape[0] // 2
        occ_map_ego_j0 = math.floor(10 / occ_map_res + 0.5)  
    # occ vis
    if occ_map is not None:
        occ_map[occ_map == 1] = 1 # 白
        occ_map[occ_map != 1] = 0 # 黑

        for i in range(occ_map.shape[0]):
            for j in range(occ_map.shape[1]):
                if occ_map[i,j] == 0:
                    x = (j - occ_map_ego_j0) * occ_map_res
                    y = (occ_map_ego_i0 - i) * occ_map_res
                    x1 = int(x * pixel_per_meter + length/2)
                    y1 = int(width/2 - y * pixel_per_meter)
                    cv2.circle(img, (x1, y1), 0, (48,48,48), -1)

    # occ polygon vis
    occ_polygons_pts = occ_polygons_pts.detach().cpu().numpy()
    for idx, (poly, attr) in enumerate(zip(occ_polygons_pts, occ_polygons_attrs)):
        if occ_polygons_mask[idx]:
            continue    # skip padding
        x, y = poly[:, 0], poly[:, 1]
        x_img = x * pixel_per_meter + length / 2
        y_img = width / 2 - y * pixel_per_meter

        pts = np.stack([x_img, y_img], axis=-1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=1)

        # draw attrs
        # start_pt, end_pt = tuple(pts[0, 0]), tuple(pts[-1, 0])
        # cv2.putText(img, str(idx), start_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,cv2.LINE_AA)

        # cy, cx, tp, area, angle = attr
        # cx_i, cy_i = int(cx * pixel_per_meter + length / 2), int(width / 2 - cy * pixel_per_meter)
        # cv2.drawMarker(img, (cx_i, cy_i), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1, line_type=cv2.LINE_AA,)
        # text_org = (cx_i + 3, cy_i - 3)   # 右上角偏移，避免压住中心点
        # cv2.putText( img, f"t={tp}, a={area:.1f}", text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA)

        # angle_i = -angle
        # arrow_len = 30  # 像素长度
        # dx, dy = arrow_len * np.cos(angle_i), arrow_len * np.sin(angle_i)
        # pt2 = (int(cx_i + dx), int(cy_i + dy))
        # cv2.arrowedLine(img, (cx_i, cy_i), pt2, color=(0, 255, 0), thickness=1, tipLength=0.3, line_type=cv2.LINE_AA,)

    # information output
    for j in range(n_mode):
        for i in range(n_future_step):
            if draw_pred or ego_future_mask[i]:
                x, y = ego_pred_xy[j,i,:2]
                v = ego_pred_v[j,i]
                if j == 0 and draw_pred and i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39, 49]:
                    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39, 49].index(i)
                    pos = (length - edge_x_pixel, start_y_pixel + idx*30)
                    # v = ego_future_v[tb_max_idx, i]
                    label_str = f'{i+1}th: x {round(float(x), 3)}, y {round(float(y), 3)}, v {round(float(v), 2)}'
                    draw_label(img, pos, label_str, color=(0,0,0), font_scale=0.5)
                    if ego_gt is not None:
                        ego_gt_v = (ego_gt[i,2] ** 2 + ego_gt[i,3] ** 2) ** 0.5
                        label_str = f'{i+1}th: x {round(float(ego_gt[i,0]), 3)}, y {round(float(ego_gt[i,1]), 3)}, v {round(float(ego_gt_v), 2)}'
                        pos = (length - 300, width // 2 + idx*30)
                        draw_label(img, pos, label_str, color=(0,0,0), font_scale=0.5)
            else:
                break
    # draw traj
    for j in range(n_mode):
        color = color_used[j]
        for i in range(0, n_future_step, traj_point_sample):
            if draw_pred or ego_future_mask[i]:
                x, y = ego_pred_xy[j,i,:2]
                x1 = int(x * pixel_per_meter + length/2)
                y1 = int(width/2 - y * pixel_per_meter)
                if n_mode > 1:
                    cv2.drawMarker(
                        img,
                        (x1, y1),
                        color,
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=circle_size,
                        thickness=1
                    )
                else:
                    cv2.circle(img, (x1, y1), circle_size, color, -1)
            else:
                break
        if draw_pred and n_mode > 1:
            x, y = ego_pred_xy[j, -1, :2]
            x1 = int(x * pixel_per_meter + length/2)
            y1 = int(width/2 - y * pixel_per_meter)
            marker = traj_end_maker[j]
            if marker is None:
                cv2.circle(img, (x1, y1), 4, color, -1)
            else:
                cv2.drawMarker(
                    img,
                    (x1, y1),
                    color,
                    markerType=marker,
                    markerSize=10,
                    thickness=1
                )
    return

def draw_trajectory_multimode_fixed(pixel_per_meter, img, ego_future_status_fixed, ego_future_prob_fixed, ego_future_prob, ego_future_v, ego_future_mask, length, width, color_used, planning_interval_fixed, draw_pred=False, start_y_pixel=0, edge_x_pixel=300, topk=1, ego_gt=None, detail=True):
    # ego_future_status_fixed: (n_steps, 2)
    # ego_future_mask: (n_steps)
    circle_size = 2 if draw_pred else 1
    n_future_step = ego_future_mask.shape[0]
    n_mode = ego_future_status_fixed.shape[0]
    #转化绝对坐标和速度
    #net的sample里有转化的，这里已经是绝对坐标了
    ego_pred_xy = ego_future_status_fixed

    xyStart = torch.zeros((n_mode, 1, 2), device=ego_pred_xy.device, dtype=ego_pred_xy.dtype)
    xyForCalV = torch.cat([xyStart, ego_pred_xy], dim=1)
    ego_pred_v = torch.norm((xyForCalV[:, 1:, :] - xyForCalV[:, :-1, :]), dim=2) / 0.1 / (planning_interval_fixed / 1)
    
    for i in range(n_future_step):
        if draw_pred or ego_future_mask[i]:
            for j in range(n_mode):
                x, y = ego_pred_xy[j,i,:2]
                v = ego_pred_v[j,i]
                x1 = int(x * pixel_per_meter + length/2)
                y1 = int(width/2 - y * pixel_per_meter)
                color = color_used
                cv2.circle(img, (x1, y1), circle_size, color, -1)
                if draw_pred and (j == 0) and i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39, 49]:
                    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39, 49].index(i)
                    if detail:
                        pos = (length - edge_x_pixel, start_y_pixel + idx*30)
                    else:
                        pos = (30, width // 2 + idx*30)
                    label_str = f'{i+1}th: x {round(float(x), 3)}, y {round(float(y), 3)}, v {round(float(v), 2)}'
                    draw_label(img, pos, label_str, color=(0,0,0), font_scale=0.5)
                    if detail and  ego_gt is not None and (j == 0):
                        ego_gt_v = (ego_gt[i,2] ** 2 + ego_gt[i,3] ** 2) ** 0.5
                        label_str = f'{i+1}th: x {round(float(ego_gt[i,0]), 3)}, y {round(float(ego_gt[i,1]), 3)}, v {round(float(ego_gt_v), 2)}'
                        if detail:
                            pos = (length - 300, width // 2 + idx*30)
                        else:
                            pos = (30, width // 2 + idx*30)
                        draw_label(img, pos, label_str, color=(0,0,0), font_scale=0.5)
        else:
            break
    return


def draw_agent_trajctory(pixel_per_meter, img, agents_status, agents_mask, length, width, color, img_dir=None, rand_idx=None, mode=0, idx=-1):
    # agents_status: (n_max_agent, 71, 6): x, y, vx, vy, yaw, score
    max_n_agent = agents_mask.shape[0]  # (max_n_agent, steps)
    for i in range(max_n_agent):
        if not agents_mask[i,20]:
            break
        if (mode == 1 or mode == 2) and i != idx:
            continue
        draw_trajectory(pixel_per_meter, img, agents_status[i,:], agents_mask[i,21:], length, width, color, img_dir, draw_pred=True, agent_idx=i, rand_idx=rand_idx)
        if mode == 1 and i == idx:
            break
    return

def draw_agent_trajctory_multimode(pixel_per_meter, img, agents_status_cls, agents_status_reg, agents_mask, length, width, color, img_dir, rand_idx=None, thickness=None, topK=None):
    # agents_status: (n_max_agent, 71, 6): x, y, vx, vy, yaw, score
    max_n_agent = agents_mask.shape[0]  # (max_n_agent, steps)
    for i in range(max_n_agent):
        if not agents_mask[i,20]:
            break
        sorted_indices = np.argsort(agents_status_cls[i].detach().numpy())[::-1]
        if topK is None or (topK>agents_status_reg.shape[1]):
            topK_indices = range(agents_status_reg.shape[1])
        else:
            topK_indices = sorted_indices[:topK]
        for mode_idc in topK_indices:
            draw_trajectory(pixel_per_meter, img, agents_status_reg[i,mode_idc], agents_mask[i,21:], length, width, color[mode_idc], img_dir, draw_pred=True, agent_idx=i, rand_idx=rand_idx, ego_prob=agents_status_cls[i,mode_idc], thickness=thickness)
    return

# def draw_route_pts(pixel_per_meter, img, route_pts, length, width, color):
#     n_pt = len(route_pts)
#     for i in range(n_pt - 1):
#         if route_pts[i] == route_pts[i+1]:
#             print('duplicate route points: ', route_pts[i])
#             continue
#         x1 = int(route_pts[i][0] * pixel_per_meter + length/2)
#         y1 = int(width/2 - route_pts[i][1] * pixel_per_meter)
#         x2 = int(route_pts[i+1][0] * pixel_per_meter + length/2)
#         y2 = int(width/2 - route_pts[i+1][1] * pixel_per_meter)
#         cv2.circle(img, (x1, y1), 4, color, -1)
#         cv2.circle(img, (x2, y2), 4, color, -1)
#         cv2.line(img, (x1, y1), (x2, y2), color, 1)


def box_center_to_corners(x_c, y_c, w, h, theta):
    """
    Converts (x_c, y_c, w, h, theta) to 4 corner points representation [(x0, y0) , ..., (x3, y3)].

    """
    center = np.array([x_c, y_c])
    center = np.tile(center, 4)
    
    dx = 0.5 * w
    dy = 0.5 * h
    cos = np.cos(theta)
    sin = np.sin(theta)

    dxcos = dx * cos
    dxsin = dx * sin
    dycos = dy * cos
    dysin = dy * sin

    dxy = np.array([- dxcos + dysin, - dxsin - dycos,
                      dxcos + dysin,   dxsin - dycos,
                      dxcos - dysin,   dxsin + dycos,
                    - dxcos - dysin, - dxsin + dycos])
    corners = center + dxy
    return   [[corners[0], corners[1]], 
              [corners[2], corners[3]], 
              [corners[4], corners[5]], 
              [corners[6], corners[7]]] # [4, 2]    


# for train

def visualize_pred_traj(data, trajectory, best_mode, probability, work_dir, pixel_per_meter=6):
    # data['rand_idx']      # torch.Size([bs])
    # trajectory.shape      # torch.Size([bs, n_mode, 50, 6])
    # best_mode.shape       # torch.Size([bs])
    # probability.shape     # torch.Size([32, 6])
    rand_idx = data['rand_idx']
    bs, n_mode, n_pt, n_channel = trajectory.shape
    ego_future_mask = data['ego_future_mask']   # torch.Size([bs, 50])
    for i in range(bs):
        if rand_idx[i] <= 0:
            continue
        img_name = osp.join(work_dir, 'visualization', f'{rand_idx[i].item()}_raw.png')
        img = cv2.imread(img_name)
        width, length, _ = img.shape
        for j in range(n_mode):
            if j == best_mode[i]:
                # best traj
                color = color_cv2['green']
            else:
                # color = color_cv2['olive']
                continue
            for k in range(n_pt-1):
                if ego_future_mask[i][k] and ego_future_mask[i][k+1]:
                    x1, y1 = trajectory[i,j,k,:2]
                    x2, y2 = trajectory[i,j,k+1,:2]
                    x1 = int(x1 * pixel_per_meter + length/2)
                    y1 = int(width/2 - y1 * pixel_per_meter)
                    x2 = int(x2 * pixel_per_meter + length/2)
                    y2 = int(width/2 - y2 * pixel_per_meter)
                    cv2.line(img, (x1, y1), (x2, y2), color, 1)
                else:
                    break
            score = probability[i][j].detach().item()
            draw_label(img, (100, 20), 'pred_score: ' + str(round(score, 2)), font_scale=0.35)
        cv2.imwrite(img_name, img)


def draw_label(img, pos, label_str, color=(0,0,0), font_scale=0.35):
    """draw label_str on the img"""
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(label_str, font, font_scale, 1)
    # Show text.
    txt_top_left = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, label_str, txt_top_left, font, font_scale,  color, lineType=cv2.LINE_AA)
    return img


# for loss

def draw_cover_circle(offset=1.7, rc=1.3, ego_length=4.886, ego_width=1.885):
    pixel_per_meter = 100     # 1m -> 6 pixel
    length, width = 1000, 500
    img = np.full((width, length, 3), 255, dtype="uint8")
    ego_pt1 = [1, 1]
    ego_pt2 = [1 + ego_length, 1 + ego_width]
    
    cv2.rectangle(img, (int(ego_pt1[0] * pixel_per_meter), int(ego_pt1[1] * pixel_per_meter)), \
    (int(ego_pt2[0] * pixel_per_meter), int(ego_pt2[1] * pixel_per_meter)), (0,0,0), 2)
    
    center1 = (1 + 0.935, 1 + ego_width / 2)
    
    center2 = (center1[0] + offset, center1[1])
    
    center3 = (center1[0] + offset * 2, center1[1])
    
    cv2.circle(img, (int(center1[0] * pixel_per_meter), int(center1[1] * pixel_per_meter)), int(rc * pixel_per_meter), (0,0,0), 1)
    cv2.circle(img, (int(center2[0] * pixel_per_meter), int(center2[1] * pixel_per_meter)), int(rc * pixel_per_meter), (0,0,0), 1)
    cv2.circle(img, (int(center3[0] * pixel_per_meter), int(center3[1] * pixel_per_meter)), int(rc * pixel_per_meter), (0,0,0), 1)
    
    cv2.imwrite('tmp.png', img)


def draw_heading(save_dir, i, heading):

    x = range(heading.shape[0])
    heading = heading.detach().numpy()
    plt.scatter(x, heading, color='blue')

    plt.xlabel('planning steps')
    plt.ylabel('yaw')

    # 显示图形
    filename = osp.join(save_dir, f'{i}_yaw.png')
    plt.savefig(filename)
    plt.clf()


def draw_v(save_dir, i, pred_v):
    x = range(pred_v.shape[0])

    pred_v = pred_v.detach().numpy()
    plt.scatter(x, pred_v, color='blue')

    plt.xlabel('planning steps')
    plt.ylabel('v')

    # 显示图形
    filename = osp.join(save_dir, f'{i}_v.png')
    plt.savefig(filename)
    plt.clf()

def draw_curvature(save_dir, i, pred_curvature):
    x = range(pred_curvature.shape[0])
    pred_curvature = pred_curvature.detach().numpy()
    plt.scatter(x, pred_curvature, color='blue')

    plt.xlabel('planning steps')
    plt.ylabel('curvature')

    # 显示图形
    filename = osp.join(save_dir, f'{i}_curvature.png')
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':
    draw_cover_circle(offset=1.6, rc=1.2)
