# -*- coding: utf-8 -*- 

import os
import sys
sys.path.insert(0, os.getcwd())     # /mnt/lustre/lizhongzhu/dlp
sys.path.insert(0, os.getcwd()+ '/tools')

import time
import os.path as osp
import json
import argparse
from tqdm import tqdm
import cv2
import pickle
import shutil

import torch
import multiprocessing as mp

from datetime import datetime

from visualization.visualization import visualization_dataset, draw_heading, draw_v, draw_curvature, visualization_path_compare
from visualization.st_graph.draw_st_graph import DrawSTGraph


def _get_date_from_data(data):
    """从 pickle 数据中提取 date 字符串，优先使用原始 date 字段，否则从 timestamp 推导。"""
    date_str = data.get('date', None)
    if date_str:
        return str(date_str)
    if 'timestamp' in data:
        ts = data['timestamp']
        if isinstance(ts, torch.Tensor):
            ts = ts.item()
        return datetime.fromtimestamp(int(ts) / 1e9).strftime('%Y%m%d')
    return None


def _resolve_img_save_dir(video_save_dir, date_str, date_split_folder):
    """若开启 date_split_folder 且能解析出 date，则保存到 video_save_dir/date_str/，否则为 video_save_dir。"""
    if date_split_folder and date_str:
        return osp.join(video_save_dir, date_str)
    return video_save_dir


def vis(args):
    print('Start to visualize infer video.')
    length, width = 1500, 1200
    video_save_dir = osp.join(args.work_dir, 'infer')
    print('save dir: ', video_save_dir)
    frame_size = (length, width)
    if args.copy_img:
        frame_size = (length + length, width)
    video_path_mp4 = osp.join(video_save_dir, 'full.mp4')
    video_writer = cv2.VideoWriter(video_path_mp4, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, frame_size)
    video_path_mp4 = osp.join(video_save_dir, 'full_fixed.mp4')
    video_writer_fixed = cv2.VideoWriter(video_path_mp4, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, frame_size)
    pkl_dir = osp.join(args.work_dir, 'infer', 'frames')
    
    if not os.path.exists(pkl_dir):
        print(f"{pkl_dir} is not exist")
        return
    frames = sorted([int(file.split('.')[0]) for file in os.listdir(pkl_dir)])
    for i, frame in enumerate(tqdm(frames, desc='Visualizing')):
        pkl_file = osp.join(pkl_dir, f'{frame}.pkl')
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            data_feaNotEmb = {
                'egolight_ori' : data['egolight_ori'],
                'navitopo_pts_ori' : data['navitopo_pts_ori'],
                'del_accLight_mask' : data['del_accLight_mask'],
            }
            img = visualization_dataset(data, data_feaNotEmb, i, video_save_dir, 'infer', length=length, width=width, write_img=False, draw_pred=True, fix_dist=False)
            video_writer.write(img)
            if not args.no_draw_fixed:
                img_fixed = visualization_dataset(data, data_feaNotEmb, i, video_save_dir, 'infer', length=length, width=width,write_img=False, draw_pred=True, fix_dist=True)
                video_writer_fixed.write(img_fixed)
    shutil.rmtree(pkl_dir)
    video_writer.release()
    video_writer_fixed.release()
    cv2.destroyAllWindows()

def cal_heading_tan(data):
    # dict_keys(['agent_attrs', 'agent_status', 'agent_time_mask', 'laneline_pts', 'laneline_attrs', 'laneline_mask', 'ego_curr_status', 'ego_future_status', 'ego_future_mask', 'ego_future_status_fixed', 'ego_future_mask_fixed', 'pred_traj', 'pred_traj_fixed', 'agent_prediction', 'pred_v', 'pred_yaw'])
    if 'pred_prob' in data:
        max_idx = torch.argmax(data['pred_prob'])
        pred_traj = data['pred_traj'][max_idx]
        pred_yaw = data['pred_yaw'][max_idx]
    else:
        pred_traj = data['pred_traj']
        pred_yaw = data['pred_yaw'] if 'pred_yaw' in data else None
    delta_pos = pred_traj[1:] - pred_traj[:-1]
    delta_pos_tan = delta_pos[:,1] / delta_pos[:,0]
    # delta_pos_tan = pred_traj[1:,1] / pred_traj[1:,0]
    if 'pred_yaw' in data:
        heading_tan = torch.tan(pred_yaw[1:])
        # headning_gt_tan = torch.tan(data['ego_future_status'][1:,4])
    else:
        heading_tan = None
    return delta_pos_tan, heading_tan

def cal_v(data):
    if 'pred_prob' in data:
        max_idx = torch.argmax(data['pred_prob'])
        pred_traj = data['pred_traj'][max_idx]
        pred_v = data['pred_v'][max_idx]
    else:
        pred_traj = data['pred_traj']
        pred_v = data['pred_v'] if 'pred_v' in data else None
    delta_pos = pred_traj[1:] - pred_traj[:-1]
    delta_pos_vx = delta_pos[:,0] / 0.1
    delta_pos_vy = delta_pos[:,1] / 0.1
    
    if 'pred_v' in data:
        pred_vx = pred_v[1:, 0]
        pred_vy = pred_v[1:, 1]
    else:
        pred_vx = pred_vy = None
    return delta_pos_vx, delta_pos_vy, pred_vx, pred_vy


def process_pickle(args, frames_dir="frames", res_pkl_name="res"):
    pkl_dir = osp.join(args.work_dir, 'infer', frames_dir)
    print("process pkl dir:", pkl_dir)
    if os.path.exists(pkl_dir):
        tmp = {}
        frames = sorted([int(file.split('.')[0]) for file in os.listdir(pkl_dir)])
        for i, frame in enumerate(frames):
            pkl_file = osp.join(pkl_dir, f'{frame}.pkl')
            with open(pkl_file, 'rb') as f:
                input_pkl = pickle.load(f)
            tmp[i] = input_pkl
        with open(osp.join(args.work_dir, 'infer', f'{res_pkl_name}.pkl'), 'wb') as f:
            pickle.dump(tmp, f)
        shutil.rmtree(pkl_dir)
    else:
        with open(osp.join(args.work_dir, 'infer', f'{res_pkl_name}.pkl'), 'rb') as f:
            tmp = pickle.load(f)
    return tmp

def vis(args):
    print('Start to visualize infer video.')
    length, width = 1500, 1200
    video_save_dir = osp.join(args.work_dir, 'infer')
    print('save dir: ', video_save_dir)
    frame_size = (length, width)
    if args.copy_img:
        frame_size = (length + length, width)
    video_path_mp4 = osp.join(video_save_dir, 'full.mp4')
    video_writer = cv2.VideoWriter(video_path_mp4, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, frame_size)
    video_path_mp4 = osp.join(video_save_dir, 'full_fixed.mp4')
    video_writer_fixed = cv2.VideoWriter(video_path_mp4, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, frame_size)
    if args.copy_img:
        tmp = process_pickle(args, "frames_sdk_dump", "res_sdk_dump")
    else:
        tmp = process_pickle(args)
    n = len(tmp)
    if args.st_end == 0:
        args.st_end = n

    if args.only_path_compare:
        for i in tqdm(range(n), desc='Path compare'):
            data = tmp[i]
            if 'pred_traj_fixed' in data and torch.any(torch.isnan(data['pred_traj_fixed'])):
                continue
            data_feaNotEmb = {
                'egolight_ori': data['egolight_ori'],
                'navitopo_pts_ori': data['navitopo_pts_ori'],
                'del_accLight_mask': data['del_accLight_mask'],
            }
            date_str = _get_date_from_data(data)
            img_save_dir = _resolve_img_save_dir(video_save_dir, date_str, args.date_split_folder)
            visualization_path_compare(data, data_feaNotEmb, i, img_save_dir, length=length, width=width, write_img=True)
        grouped = 'grouped by date' if args.date_split_folder else 'flat under infer'
        print(f'Path compare images saved to {video_save_dir} ({grouped})')
        return

    def main_draw(start, end, interval):
        st_graph_save_dir = osp.join(args.work_dir, 'st_graph')
        painter = DrawSTGraph(st_graph_save_dir)
        for i in tqdm(range(start, end, interval)):
            data = tmp[i]
            painter.main_draw_st_graph(data, i)
    if args.draw_st_graph:
        ps = []
        n_process = 12
        for i in range(n_process):
            p = mp.Process(target=main_draw, args=(i + args.st_start, args.st_end, n_process))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

    # st_graph_save_dir = osp.join(args.work_dir, 'st_graph')
    # painter = DrawSTGraph(st_graph_save_dir)

    for i in tqdm(range(n)):
        data = tmp[i]
        if torch.any(torch.isnan(data["pred_traj_fixed"])):
            continue
        data_feaNotEmb = {
            'egolight_ori' : data['egolight_ori'],
            'navitopo_pts_ori' : data['navitopo_pts_ori'],
            'del_accLight_mask' : data['del_accLight_mask'],
        }
        date_str = _get_date_from_data(data)
        img_save_dir = _resolve_img_save_dir(video_save_dir, date_str, args.date_split_folder)
        img = visualization_dataset(data, data_feaNotEmb, i, img_save_dir, 'infer', length=length, width=width, write_img=True, draw_pred=True, fix_dist=False)
        video_writer.write(img)
        img_fixed = visualization_dataset(data, data_feaNotEmb, i, img_save_dir, 'infer', length=length, width=width,write_img=False, draw_pred=True, fix_dist=True)
        video_writer_fixed.write(img_fixed)
        visualization_path_compare(data, data_feaNotEmb, i, img_save_dir, length=length, width=width, write_img=True)
        
    video_writer.release()
    video_writer_fixed.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--no_draw_fixed', action='store_true')
    parser.add_argument('--draw_st_graph', action="store_true")
    parser.add_argument('--st_start', type=int, default=0)
    parser.add_argument('--st_end', type=int, default=0)
    # parser.add_argument('--gpus', type=int)
    parser.add_argument('--copy_img', action="store_true")
    parser.add_argument('--only_path_compare', action='store_true', help='仅生成 path GT vs path pred 对比图，不写视频')
    date_dir = parser.add_mutually_exclusive_group()
    date_dir.add_argument(
        '--date_split_folder',
        dest='date_split_folder',
        action='store_true',
        help='在 infer 下按 date 建子目录保存（默认开启）',
    )
    date_dir.add_argument(
        '--no_date_split_folder',
        dest='date_split_folder',
        action='store_false',
        help='不按 date 分子目录，图片等直接保存在 infer 根目录下',
    )
    parser.set_defaults(date_split_folder=True)
    args = parser.parse_args()
    print(f"Is draw_fixed: {not args.no_draw_fixed}")
    return args


if __name__ == '__main__':
    torch.set_printoptions(precision=4, threshold=float('inf'))
    args = parse_args()
    vis(args)
    