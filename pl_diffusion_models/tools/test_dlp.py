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
import pickle
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from datasets.diffusion_dataset_ego_navi_fix_distance_path_oss_dlp import DlpDataset, DistributedSampler
from utils.misc_batch import plantf_collate_fn_eval, init_distributed_mode, get_world_size, custom_to_cuda
from models import LITMODEL
from random_generator import XorShiftRandom

# from petrel_client.client import Client
from aoss_client.client import Client
from datasets.dump_data import DumpData
from datasets.remote_client import create_remote_client

# p res['trajectory'][0,0,1:,:] - res['trajectory'][0,0,:-1,:]
def infer(model, dataloader, param):
    model.eval()
    world_size = get_world_size()
    assert args.max_frame % world_size == 0, 'max_frame % world_size should be 0'
    frame_per_gpu = args.max_frame // world_size
    save_dir = osp.join(args.work_dir, 'infer', 'frames')
    if args.rank == 0 and not osp.exists(save_dir):
        os.makedirs(save_dir)
    dist.barrier()

    if args.eval_half:
        torch.set_grad_enabled(False)
        save_dir = osp.join(args.work_dir, 'infer')
    all_save_dict = {} if args.eval_half else None

    use_fixed_noise_flag = False
    if args.noise_dir:
        input_noise_path = os.path.join(args.noise_dir, './input_noise.txt')
        input_noise_fixed_path = os.path.join(args.noise_dir, './input_noise_fix_distance.txt')
        if (not os.path.isfile(input_noise_path)) or (not os.path.isfile(input_noise_fixed_path)):
            print(f"{input_noise_path}: {os.path.isfile(input_noise_path)}")
            print(f"{input_noise_fixed_path}: {os.path.isfile(input_noise_fixed_path)}")
        else:
            input_noise = np.loadtxt(input_noise_path, comments='#', dtype=np.float32)
            input_noise = torch.from_numpy(input_noise).to(device=model.device, dtype=torch.float32)
            input_noise = input_noise.reshape(-1, 1, 80, 3)  # -> (1,1,80,3)

            input_noise_fix_distance = np.loadtxt(input_noise_fixed_path, comments='#', dtype=np.float32)
            input_noise_fix_distance = torch.from_numpy(input_noise_fix_distance).to(device=model.device, dtype=torch.float32)
            input_noise_fix_distance = input_noise_fix_distance.reshape(-1, 1, 80, 3)  # -> (1,1,80,3)
            use_fixed_noise_flag = True
            if args.rank == 0:
                print(f"[noise_dir] loaded input_noise num_samples={input_noise.shape[0]}")
                if hasattr(args, "num_samples") and args.num_samples is not None:
                    print(f"[noise_dir] NOTE: ignoring --num_samples={args.num_samples} (using noise_dir samples)")

    if args.rank == 0:
        print(f"[infer] sample_stride={args.sample_stride}, "
              f"max_frame(total_frames)={args.max_frame}, "
              f"len(dataloader_per_rank)={len(dataloader)}")

    for i, (data, data_label_path) in enumerate(tqdm(dataloader, desc='Infer')):
        if i >= args.max_frame:
            break
        data_feaNotEmb = data.pop("data_feaNotEmb")
        date_str = data.pop("date", [None])[0]
        data = custom_to_cuda(data)
        if not use_fixed_noise_flag:
            num_samples = args.num_samples
            # 和sdk推理对齐：采用Xorshift随机数生成方法，利用时间戳作为种子，保持和sdk推理时的一致性
            rng = XorShiftRandom(data['model_input']['timestamp'].item())
            noise_array = rng.normal_vector_1d(num_samples * 1 * data['model_input']['ego_future_mask'].shape[1] * 3, mean=0.0, stddev=1.0)
            input_noise = torch.tensor(noise_array, dtype=torch.float32, device=model.device)
            input_noise = input_noise.view(num_samples, 1, data['model_input']['ego_future_mask'].shape[1], 3)
            rng.seed(data['model_input']['timestamp'].item() + 1)
            noise_array_fixed = rng.normal_vector_1d(num_samples * 1 * data['model_input']['ego_future_mask_fixed'].shape[1] * 3, mean=0.0, stddev=1.0)
            input_noise_fix_distance = torch.tensor(noise_array_fixed, dtype=torch.float32, device=model.device)
            input_noise_fix_distance = input_noise_fix_distance.view(num_samples, 1, data['model_input']['ego_future_mask_fixed'].shape[1], 3)
            pred_trajs_tensor, pred_fix_distance_path_tensor = model.model.sample(data['model_input'], input_noise, input_noise_fix_distance)
            if args.rank == 0 and i == 0:
                print(f"[random_noise] input_noise num_samples={input_noise.shape[0]}")
        input_noise = input_noise[:, :, :data['model_input']['ego_future_mask'].shape[1], :]
        input_noise_fix_distance = input_noise_fix_distance[:, :, :data['model_input']['ego_future_mask_fixed'].shape[1], :]

        input_noise[0] = 0.0
        input_noise_fix_distance[0] = 0.0

        pred_trajs_tensor, pred_fix_distance_path_tensor = model.model.sample(
            data['model_input'], input_noise, input_noise_fix_distance,
            sample_steps=args.sample_steps
        )
        pred_trajs_tensor = pred_trajs_tensor[:, 0, ...]
        pred_fix_distance_path_tensor = pred_fix_distance_path_tensor[:, 0, ...]
        tmp = {}
        tmp['data_label_path'] = data_label_path
        tmp['date'] = date_str
        tmp['input_noise'] = input_noise.cpu()
        tmp['input_noise_fix_distance'] = input_noise_fix_distance.cpu()
        for key, val in data['model_input'].items():
            tmp[key] = val[0].cpu()

        tmp.update({
            'pred_traj' : pred_trajs_tensor[:, :, :2].cpu(),
            'pred_traj_fixed' : pred_fix_distance_path_tensor[:, :, :2].cpu(),
            'pred_yaw' : pred_trajs_tensor[:, :, 2].cpu(),
            'pred_prob' : None,
            'pred_prob_fixed' : None,
            "agents_importance": None,
            "laneline_importance": None,
            'pred_v' : None,
            })
        
        #if ('motion_mode_setting' in param['model']) and (param['model']['motion_mode_setting'] is not None):
        #    tmp.update({
        #        'agent_prediction_multimode_cls' : res['agent_prediction_multimode_cls'][0].cpu(), # [bs, na, mode_num]
        #        'agent_prediction_multimode_reg' : res['agent_prediction_multimode_reg'][0].cpu(), # [bs, na, mode_num, future_steps, out_channels]
        #    })
        #else:
        #    tmp.update({
        #        'agent_prediction' : res['agent_prediction'][0].cpu()    # [bs, na, future_steps, out_channels]
        #    })
        tmp.update({
            'egolight_ori' : data_feaNotEmb[0]['egolight_ori'],
            'navitopo_pts_ori' : data_feaNotEmb[0]['navitopo_pts_ori'],
            'del_accLight_mask' : data_feaNotEmb[0]['del_accLight_mask'],
        })
        
        if args.eval_half:
            all_save_dict[i] = tmp
        else:
            filename = osp.join(save_dir, f'{args.rank * frame_per_gpu + i}.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(tmp, f)
    if args.eval_half:
        all_data_filename = osp.join(save_dir, f'res_{args.rank}_{args.eval_half}.pkl')
        with open(all_data_filename, 'wb') as f:
            pickle.dump(all_save_dict, f)
            print(f'save pkl to {all_data_filename}')

def infer_dump_datas(model, dump_datas, param):
    model.eval()
    world_size = get_world_size()
    assert args.max_frame % world_size == 0, 'max_frame % world_size should be 0'
    frame_per_gpu = args.max_frame // world_size
    save_dir = osp.join(args.work_dir, 'infer', 'frames_sdk_dump')
    if args.rank == 0 and not osp.exists(save_dir):
        os.makedirs(save_dir)
    dist.barrier()

    for i in tqdm(range(args.max_frame)):
        data = dump_datas.iter_input_datas(i)
        data = custom_to_cuda(data)
        input_noise = data['model_input']['input_noise']
        input_noise_fix_distance = data['model_input']['input_noise_fix_distance']
        pred_trajs_tensor, pred_fix_distance_path_tensor = model.model.sample(data['model_input'], input_noise, input_noise_fix_distance, sample_steps=args.sample_steps)
        pred_trajs_tensor = pred_trajs_tensor[:, 0, ...]
        pred_fix_distance_path_tensor = pred_fix_distance_path_tensor[:, 0, ...]
        
        tmp = {}
        for key, val in data["model_input"].items():
            if key == 'navitopo_pts_ori':
                tmp[key] = val.cpu()
            else:
                tmp[key] = val[0].cpu()
        
        tmp.update({
            'pred_traj' : pred_trajs_tensor[:, :, :2].cpu(),
            'pred_traj_fixed' : pred_fix_distance_path_tensor[:, :, :2].cpu(),
            'pred_yaw' : pred_trajs_tensor[:, :, 2].cpu(),
            'pred_prob' : None,
            'pred_prob_fixed' : None,
            "agents_importance": None,
            "laneline_importance": None,
            'pred_v' : None,
            })
        sdk_traj = dump_datas.iter_output_datas(i)
        raw_data = sdk_traj.get("model_input", {})  # 确保获取raw字典，不存在则返回空字典

        tmp.update(
            {
                'sdk_pred_traj' : raw_data.get("trajectory")[:, :, :2],
                'sdk_pred_traj_fixed' : raw_data.get("trajectory_fixed")[:, :, :2],
                'sdk_pred_yaw' : raw_data.get("trajectory")[:, :, 3],
                'sdk_pred_prob' : None,
                'sdk_pred_prob_fixed' : None,
                "sdk_agents_importance": None,
                "sdk_laneline_importance": None,
                'sdk_pred_v' : None,
            }
        )
        filename = osp.join(save_dir, f'{args.rank * frame_per_gpu + i}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(tmp, f)


def build_model(param, args):
    """加载模型。先按 config 实例化，再加载 checkpoint state_dict，避免 Lightning CLI 解析 checkpoint 时的 module.config 报错。"""
    print(f"加载模型: {args.ckpt}")
    config_path = os.path.join(os.getcwd(), "config", "model", "LitMMDiTDiffusionModel.yaml")
    model_cls = LITMODEL.module_dict['LitMMDiTDiffusionModel']
    # 先用 config 实例化，避免 load_from_checkpoint 走 CLI parser 导致 NSKeyError: module.config
    model = model_cls(config=config_path)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model = model.eval()
    model = model.to('cuda')

    # 打印模型信息
    print(f"运行设备: {model.device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型加载完成, 参数数量: {total_params:,}")
    return model


def main(param, args):

    client = create_remote_client(param)

    # 有些配置中 date_split_file_test 允许为空，此时直接从数据目录推断，
    # 这里仅在显式提供路径时才去加载，避免 None 触发 os.path.exists 报错。
    date_split = None
    if param.get('date_split_file_test'):
        date_split = client.load_json(param['date_split_file_test'])

    open_eval_mode = False
    if args.eval_half:
        open_eval_mode = True

        print(f'args.eval_half: {args.eval_half}, eval_split_mode: {args.eval_split_mode}')

        eval_split_idx = int(param['eval_split'][args.eval_split_mode])

        tmp_data_dir = param['tmp_data_dir_eval'][eval_split_idx] + f'_{args.eval_half}'
        date_split_file = param['date_split_file_eval'][eval_split_idx].split('.json')[0] + f'_{args.eval_half}' + '.json'
        date_split = client.load_json(param['date_split_file_test'])
        print(f'tmp_data_dir: {tmp_data_dir}, date_split_file: {date_split_file} ')

        date_split = date_split
        param['tmp_data_dir_test'] = tmp_data_dir
        param['scene_extract_test'] = param['scene_extract_eval'][eval_split_idx]
        param['from_ceph_test'] = True
        param['change_light_file'] = param['change_light_file'][eval_split_idx]
        
    # 原始完整数据集
    full_dataset = DlpDataset(param, is_train=False)
    args.max_frame = len(full_dataset)
    print(f'args.max_frame (total frames): {args.max_frame}')

    # 根据 sample_stride 决定“读哪些帧 + infer 哪些帧”
    # 当 sample_stride > 1 时，仅对按步长采样到的帧构建子数据集，避免对被跳过的帧做读取与推理。
    if args.sample_stride > 1:
        all_indices = np.arange(args.max_frame)
        sampled_indices = all_indices[::args.sample_stride]
        if args.rank == 0:
            print(f"[main] sample_stride={args.sample_stride}, "
                  f"using {len(sampled_indices)} / {args.max_frame} frames for infer.")
        data = torch.utils.data.Subset(full_dataset, sampled_indices)
    else:
        data = full_dataset

    sampler_test = DistributedSampler(data, shuffle=False)
    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, batch_size=1, drop_last=False)    
    dataloader = DataLoader(
        data,
        batch_size=1,
        batch_sampler=batch_sampler_test,
        collate_fn=plantf_collate_fn_eval, 
        num_workers=0,
    )
    #data是就绪的

    model = build_model(param, args)

    infer(
        model,
        dataloader,
        param
    )
    # dist.destroy_process_group()


def sdk_dump_infer_main(param, args):
    dump_datas = DumpData(args.sdk_dump_dir)
    args.max_frame = dump_datas.get_max_frame()
    model = build_model(param, args)
    infer_dump_datas(model, dump_datas, param)
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--eval_half', type=str, default='')
    parser.add_argument('--eval_split_mode', type=str, default='eval_wuhan_v3')
    parser.add_argument("--sdk_dump_dir", default=None, type=str)
    parser.add_argument('--case_dir', type=str, default='')
    parser.add_argument('--noise_dir', type=str, default='')
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of trajectory samples to generate when NOT using --noise_dir.",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=5,
        help="Number of diffusion sampling steps (T).",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=1,
        help="Uniform sampling stride over dataset (1 means use every sample).",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.set_printoptions(precision=4, threshold=float('inf'))
    args = parse_args()
    init_distributed_mode(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    # config = args.config.split('/')[-1].rstrip('.py')        # e.g. configs/dlp_v1.py
    cfg_path = args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        param = yaml.safe_load(f)
    if args.case_dir:
        param['tmp_data_dir_test'] = os.path.join(args.case_dir, 'scene')
        param['date_split_file_test'] = os.path.join(args.case_dir, 'date_split.json')
    if args.rank == 0:
        print(param)
    main(param, args)
    if args.sdk_dump_dir:
        print("===== sdk dump infer =====")
        sdk_dump_infer_main(param, args)
    os.system(f'rm -rf configs/tmp.py')