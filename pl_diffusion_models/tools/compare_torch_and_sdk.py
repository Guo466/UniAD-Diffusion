# -*- coding: utf-8 -*- 

import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd()+'/configs')

import cv2
import time
import os.path as osp
import json
import argparse
from tqdm import tqdm
import importlib
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import bisect
from datetime import datetime, timezone
from infer_visualization import process_pickle
from joblib import Parallel, delayed

length, width = 4800, 5000
bar_length, bar_width = 100, 20
rectangle_length, rectangle_width = 20, 10
color_cv2 = {'red' : (0, 0, 205), 'orange' : (0, 140, 255), 'yellow' : (0, 215, 255), 'green' : (170,  205, 102), 'blue' : (255, 0, 0 ), 'purple' : (226, 43, 138), 'pink' : (180, 110, 255), 'grey' : (190, 190, 190), 'dark_grey' : (105, 105, 105), 'violet' : (238, 130, 238), 'dark_violet' : (211, 0, 148), 'black' : (0, 0, 0), 'white' : (255, 255, 255), 'brown' : (63, 133, 205), 'light_blue' : (255, 245, 0), 'olive' : (35, 142, 107), 'gold' :(0, 215, 255)}
start_pos_dict = {
    'agent_attrs': (50, 50),
    'agent_time_mask': (240, 50),
    'agent_status': (700, 50),

    'laneline_pts': (700, 600),
    'laneline_attrs': (50, 600),
    'laneline_mask': (350, 600),

    'navitopo_pts': (700, 2150),
    'navitopo_mask': (350, 2150),

    'pred_traj': (700, 2350),
    'pred_traj_fixed': (700, 2600),

    'sdk_pred_traj': (700, 2850),
    'sdk_pred_traj_fixed':(700, 3100),

    'input_noise': (700, 3400),
    'input_noise_fix_distance': (700, 3750),

    'mean_std': (700, 4100),
    'mean_std_fixed': (700, 4200),

    'occ_polygons_attrs': (1650, 2350),
    'occ_polygons_mask': (1950, 2350),
    'occ_polygons_pts': (2200, 2350),
}


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

def initialize_plot(img, rand_idx):
    draw_label(img, (30, 30), 'frame: ' + str(rand_idx), font_scale=0.5)

def blue_white_to_deep_blue_with_threshold(norm, threshold=0.1, gamma=1.8):
    """
    norm ∈ [0, 1]
    norm < threshold : 白色
    norm >= threshold: 蓝白 -> 深蓝
    """
    if norm < threshold:
        return (255, 255, 255)  # 白色 (BGR)

    # 重新归一化到 [0, 1]
    n = (norm - threshold) / (1.0 - threshold)
    n = max(0.0, min(1.0, n))

    # gamma 调整，增强中高值对比
    n = n ** gamma

    b = 255
    rg = int(255 * (1 - n))

    return (b, rg, rg)

def draw_colorbar(img, x0, y0, bar_length = 50, bar_width = 20, bar_height = 10):
    threshold, gamma = 0.1, 1.8
    norm_list = [i / bar_length for i in range(bar_length + 1)]
    dy, dx = 0, 0
    for i, norm in enumerate(norm_list):
        color = blue_white_to_deep_blue_with_threshold(
            norm, threshold=threshold, gamma=gamma
        )
        x_start = x0 + i * bar_width
        x_end   = x_start + bar_width
        img[y0:y0 + bar_height, x_start:x_end] = color
        img[y0:y0 + bar_height, x_start:x_start+1] = (0, 0, 0)

        if norm in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            cv2.putText(
                img,
                f"{norm}",
                (x_start, y0 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    img[y0:y0 + bar_height, x_start + bar_width:x_start + bar_width +1] = (0, 0, 0)
    img[y0:y0 + 1, x0:x_start + bar_width] = (0, 0, 0)
    img[y0 + bar_height:y0 + bar_height + 1, x0:x_start + bar_width + 1] = (0, 0, 0)

def draw_2d_grid(img, diff_2d, start_x, start_y):
    H, W = diff_2d.shape
    max_diff = 1 # 最大可接受差异
    for i in range(H):
        for j in range(W):
            norm = np.clip(diff_2d[i, j] / max_diff, 0, 1)
            color = blue_white_to_deep_blue_with_threshold(norm)
            top_left = (
                start_x + j * rectangle_length,
                start_y + i * rectangle_width
            )
            bottom_right = (
                start_x + (j + 1) * rectangle_length,
                start_y + (i + 1) * rectangle_width
            )
            cv2.rectangle(img, top_left, bottom_right, color, -1)
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 1)

def cal_similarity(key, tensor1, tensor2):
    diff = torch.abs(tensor1.float() - tensor2.float()).mean().item()
    denominator = torch.abs(tensor1.float()).mean().item()
    sim = 1 - (diff / (denominator + 1e-6))
    # print(f"{key} diff: {diff:.6f}, denom: {denominator:.6f}, sim: {sim:.4f}")
    return sim

def draw_three_dim(img, key, tensor1, tensor2, start_x, start_y, slices_per_row = 1):
    diff = np.abs(tensor1.numpy() - tensor2.numpy())  # (D, H, W)
    D, H, W = diff.shape
    sim = cal_similarity(key, tensor1, tensor2)
    custom_label = f"{key}: {diff.shape} sim: {sim:.2f}"
    draw_label(img, (start_x, start_y), custom_label)
    for d in range(D):  # 第 0 维
        slice_gap = 10
        row_idx = d // slices_per_row
        col_idx = d % slices_per_row

        slice_start_x = (
            start_x + col_idx * (W * rectangle_length + slice_gap)
        )
        slice_start_y = (
            start_y + row_idx * (H * rectangle_width + slice_gap)
        )

        draw_2d_grid(
            img,
            diff[d],
            slice_start_x,
            slice_start_y,
        )

def draw_compare_res(img, torch_infer_res, dump_infer_res):
    for key in start_pos_dict.keys():
        if key not in torch_infer_res and key not in dump_infer_res:
            print(f"key {key} not in results")
            continue
        if key in ['pred_yaw', 'sdk_pred_yaw']: continue
        start_x, start_y = start_pos_dict[key]
        tensor1 = torch_infer_res[key]
        tensor2 = dump_infer_res[key]
        if tensor1.dim() == 1:
            tensor1 = tensor1.unsqueeze(0).unsqueeze(0)
            tensor2 = tensor2.unsqueeze(0).unsqueeze(0)
        if tensor1.dim() == 2:
            tensor1 = tensor1.unsqueeze(0)
            tensor2 = tensor2.unsqueeze(0)
        if key in [ 'laneline_pts', 'laneline_mask', 
                    'navitopo_pts', 'navitopo_mask',
                    'pred_traj', 'pred_traj_fixed',
                    'sdk_pred_traj', 'sdk_pred_traj_fixed',
                    'input_noise', 'input_noise_fix_distance',
                    'occ_polygons_mask', 'occ_polygons_pts']: 
            tensor1 = tensor1.permute(0, 2, 1)
            tensor2 = tensor2.permute(0, 2, 1)
        slices_per_row = 25 if key in ['agent_status'] else 1
        draw_three_dim(img, key, tensor1, tensor2, start_x, start_y, slices_per_row)
    return img

def fmt(arr, width=8, prec=4):
    arr = np.asarray(arr)
    def _fmt(x):
        if np.isclose(x, int(x)):
            # return f"{int(x):{width - 5}d}"
            return f"{int(x):{width}d}"
        else:
            return f"{x:{width}.{prec}f}"

    return np.array2string(
        arr,
        formatter={'float_kind': _fmt},
        separator=' ',
        max_line_width=np.inf
    )

def compare_tensor_ond_dim(key, tensor1, tensor2):
    string = ""
    return "\n".join(
        f"{str(row1).ljust(5)}\t{str(row2).ljust(5)}\t{abs(row1 - row2)}" for row1, row2 in zip(tensor1.numpy(), tensor2.numpy())
    ) + '\n'

def compare_tensor_two_dim(key, tensor1, tensor2):
    string = ""
    if key in ['mean_std', 'mean_std_fixed', 'pred_yaw', 'sdk_pred_yaw']:
        for i in range(tensor1.shape[0]):
            row1 = tensor1[i].numpy()
            string += f'{fmt(row1)}\n'
        string += '\n'
        for i in range(tensor2.shape[0]):
            row2 = tensor2[i].numpy()
            string += f'{fmt(row2)}\n'
        string += '\n'
        for i in range(tensor1.shape[0]):
            row1 = tensor1[i].numpy()
            row2 = tensor2[i].numpy()
            diff = abs(tensor1[i] - tensor2[i]).numpy()
            string += f'{fmt(diff)}\n'
        string += '\n'
    elif key in ['agent_time_mask']:
        for i in range(tensor1.shape[0]):
            row1 = tensor1[i].numpy()
            row2 = tensor2[i].numpy()
            diff = (tensor1[i] ^ tensor2[i]).numpy()
            string += f'{fmt(row1)}\t{fmt(row2)}\t{fmt(diff)}\n'
    elif key in ['agent_attrs']:
        for i in range(tensor1.shape[0]):
            row1 = tensor1[i].numpy()
            row2 = tensor2[i].numpy()
            diff = abs(tensor1[i] - tensor2[i]).numpy()
            string += f'{fmt(row1).ljust(30)}\t{fmt(row2).ljust(30)}\t{fmt(diff)}\n'
    elif key in ['occ_polygons_attrs']:
        for i in range(tensor1.shape[0]):
            row1 = tensor1[i].numpy()
            row2 = tensor2[i].numpy()
            diff = abs(tensor1[i] - tensor2[i]).numpy()
            string += f'{fmt(row1, prec=3).ljust(40)}\t{fmt(row2, prec=3).ljust(40)}\t{fmt(diff, prec=3)}\n'
    else:
        for i in range(tensor1.shape[0]):
            row1 = tensor1[i].numpy()
            row2 = tensor2[i].numpy()
            diff = abs(tensor1[i] - tensor2[i]).numpy()
            string += f'{fmt(row1)}\t{fmt(row2)}\t{fmt(diff)}\n'
    return string

def compare_tensor_three_dim(key, tensor1, tensor2):
    string = ""
    if key in ['agent_status']:
        for i in range(tensor1.shape[0]):
            string += f"{key} {i}\n"
            for j in range(tensor1.shape[1]):
                row1 = tensor1[i, j].numpy()
                row2 = tensor2[i, j].numpy()
                diff = abs(tensor1[i, j] - tensor2[i, j]).numpy()
                string += f'{fmt(row1).ljust(60)}\t{fmt(row2).ljust(60)}\t{fmt(diff)}\n'
            string += '\n'
    else:
        for i in range(tensor1.shape[0]):
            string += f"{key} {i}\n"
            for j in range(tensor1.shape[1]):
                row1 = tensor1[i, j].numpy()
                row2 = tensor2[i, j].numpy()
                diff = abs(tensor1[i, j] - tensor2[i, j]).numpy()
                string += f'{fmt(row1).ljust(20)}\t{fmt(row2).ljust(20)}\t{fmt(diff)}\n'
            string += '\n'
    return string

def compare_res(torch_infer_res, dump_infer_res):
    string = ""
    for key in start_pos_dict.keys():
        if key not in torch_infer_res or key not in dump_infer_res:
            print(f"key {key} not in results")
            continue
        tensor1 = torch_infer_res[key]
        tensor2 = dump_infer_res[key]
        # print(key, tensor1.shape, tensor2.shape, tensor1.dim())
        string += f'{key} {tensor1.shape} {tensor2.shape} {tensor1.dim()}\n'
        if tensor1.dim() == 1:
            value_str = compare_tensor_ond_dim(key, tensor1, tensor2)
            string += value_str + '\n'
        if tensor1.dim() == 2:
            value_str = compare_tensor_two_dim(key, tensor1, tensor2)
            string += value_str + '\n'
        if tensor1.dim() == 3:
            value_str = compare_tensor_three_dim(key, tensor1, tensor2)
            string += value_str + '\n'
    return string

def get_timestamp_ns_index(timestamp_ns_list, timestamp_ns):
    if timestamp_ns_list is None:
        print("Timestamp list is not available.")
        return -1
    index = bisect.bisect_left(timestamp_ns_list, timestamp_ns)
    if index != len(timestamp_ns_list) and abs(timestamp_ns_list[index] - timestamp_ns) < 50000000:  # 0.05s
        return index
    else:
        return -1

def get_utc_time(timestamp_ns):
    utc_time = datetime.fromtimestamp(np.array(timestamp_ns) / 1e9, tz=timezone.utc)
    return utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')

def process_frame_joblib(i, torch_infer_res, timestamp_ns_list, dump_infer_res, compare_res_dir):
    timestamp_ns = torch_infer_res[i]['timestamp'].item()
    index = get_timestamp_ns_index(timestamp_ns_list, timestamp_ns)
    
    if index == -1:
        return None
    
    string = f'torch timestamp_ns:, {get_utc_time(timestamp_ns)} \n'
    string += f'sdk   timestamp_ns:, {get_utc_time(timestamp_ns_list[index])} \n\n'
    string += compare_res(torch_infer_res[i], dump_infer_res[index])
    
    filename = osp.join(compare_res_dir, f'{i}.txt')
    with open(filename, 'w') as f:
        f.write(string)
    
    img = np.full((width, length, 3), 255, dtype="uint8")
    initialize_plot(img, i)
    draw_colorbar(img, (length // 2 - bar_length * bar_width // 2), 20, bar_length, bar_width)
    draw_compare_res(img, torch_infer_res[i], dump_infer_res[index])
    cv2.imwrite(osp.join(compare_res_dir, f'{i}.png'), img)
    return filename

def get_pickle(res_pkl_name):
    with open(res_pkl_name, 'rb') as f:
        tmp = pickle.load(f)
    return tmp

def process_torch_res(torch_infer_res, dump_infer_res):
    for _, value in torch_infer_res.items():
        for key in ['sdk_pred_traj', 'sdk_pred_traj_fixed', 'sdk_pred_yaw']: 
            value[key] = value[key[4:]]
        value['agent_status'] = value['agent_status'][..., :21, :]
        value['agent_time_mask'] = value['agent_time_mask'][..., :21].int()
    
    for _, value in dump_infer_res.items():
        value['agent_time_mask'] = value['agent_time_mask'].int()
        value['laneline_attrs'] = value['laneline_attrs'].int()

def main(args):
    torch_infer_res = process_pickle(args, "frames", "res")
    dump_infer_res = process_pickle(args, "frames_sdk_dump", "res_sdk_dump")
    process_torch_res(torch_infer_res, dump_infer_res)
    timestamp_ns_list = [data['timestamp'].item() for data in dump_infer_res.values()]

    compare_res_dir = osp.join(args.work_dir, 'infer', 'compare_frames')
    if not os.path.exists(compare_res_dir):
        os.makedirs(compare_res_dir)

    n = len(torch_infer_res)
    # for i in range(n):
    #     print(f"===== compare frame {i} =====")
    #     timestamp_ns = torch_infer_res[i]['timestamp'].item()
        # index = get_timestamp_ns_index(timestamp_ns_list, timestamp_ns)
        # if index == -1:
        #     continue
        # print("----- timestamp ns:", timestamp_ns, "-----")
        # print("torch timestamp_ns:", i, get_utc_time(timestamp_ns))
        # print("sdk   timestamp_ns:", index, get_utc_time(timestamp_ns_list[index]))
        # string = f'torch timestamp_ns:,  {i} {get_utc_time(timestamp_ns)} \n'
        # string += f'sdk   timestamp_ns:, {index} {get_utc_time(timestamp_ns_list[index])} \n\n'

        # string += compare_res(torch_infer_res[i], dump_infer_res[index])
        # filename = osp.join(compare_res_dir, f'{i}.txt')
        # with open(filename, 'w') as f:
        #     f.write(string)
        # img = np.full((width, length, 3), 255, dtype="uint8")
        # initialize_plot(img, i)
        # draw_colorbar(img, (length // 2 - bar_length * bar_width // 2), 20, bar_length, bar_width)
        # draw_compare_res(img, torch_infer_res[i], dump_infer_res[index])
        # cv2.imwrite(osp.join(compare_res_dir, f'{i}_{index}.png'), img)
        # break
    n_jobs = -1
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_frame_joblib)(
            i, torch_infer_res, timestamp_ns_list, dump_infer_res, compare_res_dir
        ) 
        for i in range(n)
    )
    
    successful_files = [r for r in results if r is not None]
    print(f"处理完成，成功处理 {len(successful_files)} 个文件")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("===== compare torch infer and sdk dump infer =====")
    args = parse_args()
    main(args)
