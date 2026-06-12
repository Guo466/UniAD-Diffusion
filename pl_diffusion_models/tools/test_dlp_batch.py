# -*- coding: utf-8 -*-
"""
多 batch 推理脚本：与 test_dlp.py 命令行参数一致，并增加 --infer_batch_size。
默认 infer_batch_size=1 时行为与 test_dlp 对齐；>1 时对同一 collate batch 内各样本独立生成噪声并落盘。
使用 --noise_dir 时仅支持 infer_batch_size=1（与 test_dlp 相同，噪声文件未按帧切片索引）。
"""

import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/configs")

import argparse
import os.path as osp
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

from datasets.diffusion_dataset_ego_navi_fix_distance_path_oss_dlp import DlpDataset, DistributedSampler
from datasets.dump_data import DumpData
from datasets.remote_client import create_remote_client
from models import LITMODEL
from random_generator import XorShiftRandom
from utils.misc_batch import init_distributed_mode, get_world_size, custom_to_cuda

args = None  # 由 parse_args / __main__ 赋值，infer 中读取


def plantf_collate_fn_eval_multibatch(batch: List[dict]) -> Tuple[dict, List[Optional[str]]]:
    """与 plantf_collate_fn_eval 一致，但为 batch 内每条样本收集 data_label_path。"""
    data_label_paths: List[Optional[str]] = []
    for i in range(len(batch)):
        if "data_label_path" in batch[i]:
            data_label_paths.append(batch[i].pop("data_label_path", None))
        else:
            data_label_paths.append(None)

    res: Dict[str, Any] = {}
    for key in batch[0].keys():
        if key not in ["model_input", "pos", "neg"]:
            res[key] = [batch[i][key] for i in range(len(batch))]
            continue
        res[key] = {}
        for k in batch[0][key].keys():
            stacked = torch.stack([batch[i][key][k] for i in range(len(batch))], dim=0)
            res[key][k] = stacked
    return res, data_label_paths


def _expand_model_input_for_num_samples(
    model_input: Dict[str, Any], num_samples: int
) -> Tuple[Dict[str, Any], int]:
    """与 LitMMDiTDiffusionModel.test_step 一致：B -> B*num_samples。"""
    b = None
    for value in model_input.values():
        if isinstance(value, torch.Tensor):
            b = value.shape[0]
            break
    if b is None:
        raise RuntimeError("model_input 中无 Tensor，无法推断 batch 大小")

    expanded: Dict[str, Any] = {}
    for key, value in model_input.items():
        if isinstance(value, torch.Tensor):
            nd = value.dim()
            exp = value.unsqueeze(1).expand(-1, num_samples, *([-1] * (nd - 1)))
            exp = exp.reshape(b * num_samples, *value.shape[1:])
            expanded[key] = exp
        else:
            expanded[key] = value
    return expanded, b


def _build_xor_noise_for_batch(
    model_input: Dict[str, torch.Tensor],
    num_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 batch 维上每个样本用 timestamp / timestamp+1 种子生成噪声，与单条 test_dlp 一致。"""
    ts = model_input["timestamp"]
    B = int(ts.shape[0])
    T = int(model_input["ego_future_mask"].shape[1])
    Tf = int(model_input["ego_future_mask_fixed"].shape[1])

    traj_blocks: List[torch.Tensor] = []
    fix_blocks: List[torch.Tensor] = []
    for b in range(B):
        t0 = int(ts[b].item())
        rng = XorShiftRandom(t0)
        noise_array = rng.normal_vector_1d(
            num_samples * 1 * T * 3, mean=0.0, stddev=1.0
        )
        traj_blocks.append(
            torch.tensor(noise_array, dtype=torch.float32, device=device).view(
                num_samples, 1, T, 3
            )
        )
        rng.seed(t0 + 1)
        noise_fixed = rng.normal_vector_1d(
            num_samples * 1 * Tf * 3, mean=0.0, stddev=1.0
        )
        fix_blocks.append(
            torch.tensor(noise_fixed, dtype=torch.float32, device=device).view(
                num_samples, 1, Tf, 3
            )
        )

    input_noise = torch.cat(traj_blocks, dim=0)
    input_noise_fix_distance = torch.cat(fix_blocks, dim=0)
    return input_noise, input_noise_fix_distance


def infer(model, dataloader, param):
    model.eval()
    world_size = get_world_size()
    assert args.max_frame % world_size == 0, "max_frame % world_size should be 0"
    frame_per_gpu = args.max_frame // world_size
    save_dir = osp.join(args.work_dir, "infer", "frames")
    if args.rank == 0 and not osp.exists(save_dir):
        os.makedirs(save_dir)
    dist.barrier()

    if args.eval_half:
        torch.set_grad_enabled(False)
        save_dir = osp.join(args.work_dir, "infer")
    all_save_dict = {} if args.eval_half else None

    use_fixed_noise_flag = False
    if args.noise_dir:
        input_noise_path = os.path.join(args.noise_dir, "./input_noise.txt")
        input_noise_fixed_path = os.path.join(
            args.noise_dir, "./input_noise_fix_distance.txt"
        )
        if (not os.path.isfile(input_noise_path)) or (
            not os.path.isfile(input_noise_fixed_path)
        ):
            print(f"{input_noise_path}: {os.path.isfile(input_noise_path)}")
            print(f"{input_noise_fixed_path}: {os.path.isfile(input_noise_fixed_path)}")
        else:
            input_noise = np.loadtxt(input_noise_path, comments="#", dtype=np.float32)
            input_noise = torch.from_numpy(input_noise).to(
                device=model.device, dtype=torch.float32
            )
            input_noise = input_noise.reshape(-1, 1, 80, 3)

            input_noise_fix_distance = np.loadtxt(
                input_noise_fixed_path, comments="#", dtype=np.float32
            )
            input_noise_fix_distance = torch.from_numpy(input_noise_fix_distance).to(
                device=model.device, dtype=torch.float32
            )
            input_noise_fix_distance = input_noise_fix_distance.reshape(-1, 1, 80, 3)
            use_fixed_noise_flag = True
            if args.rank == 0:
                print(f"[noise_dir] loaded input_noise num_samples={input_noise.shape[0]}")
                if hasattr(args, "num_samples") and args.num_samples is not None:
                    print(
                        f"[noise_dir] NOTE: ignoring --num_samples={args.num_samples} (using noise_dir samples)"
                    )

    if args.rank == 0:
        print(
            f"[infer] infer_batch_size={args.infer_batch_size}, sample_stride={args.sample_stride}, "
            f"max_frame(total_frames)={args.max_frame}, len(dataloader_per_rank)={len(dataloader)}"
        )

    if use_fixed_noise_flag and args.infer_batch_size > 1:
        raise ValueError(
            "使用 --noise_dir 时请设置 --infer_batch_size 1（噪声文件未按 batch 切片，与 test_dlp 行为一致）。"
        )

    # 本 rank 上已落盘的样本序号（0..num_samples-1），与 test_dlp 在 batch_size=1 时的 enumerate(i) 一致
    local_sample_idx = 0

    for i, (data, data_label_paths) in enumerate(tqdm(dataloader, desc="Infer")):
        if i >= args.max_frame:
            break
        data_feaNotEmb = data.pop("data_feaNotEmb")
        dates = data.pop("date", [None] * len(data_label_paths))
        data = custom_to_cuda(data)
        model_input = data["model_input"]
        B = int(model_input["timestamp"].shape[0])
        num_samples = args.num_samples

        if not use_fixed_noise_flag:
            input_noise, input_noise_fix_distance = _build_xor_noise_for_batch(
                model_input, num_samples, model.device
            )
            if args.rank == 0 and i == 0:
                print(f"[random_noise] input_noise shape={tuple(input_noise.shape)}")
        else:
            # 与 test_dlp 相同：整段噪声张量，每步复用（仅 batch=1）
            pass

        expanded_mi, B_exp = _expand_model_input_for_num_samples(model_input, num_samples)
        assert B_exp == B

        Tm = expanded_mi["ego_future_mask"].shape[1]
        Tmf = expanded_mi["ego_future_mask_fixed"].shape[1]
        input_noise = input_noise[:, :, :Tm, :]
        input_noise_fix_distance = input_noise_fix_distance[:, :, :Tmf, :]

        for b in range(B):
            input_noise[b * num_samples] = 0.0
            input_noise_fix_distance[b * num_samples] = 0.0

        pred_trajs_tensor, pred_fix_distance_path_tensor = model.model.sample(
            expanded_mi,
            input_noise,
            input_noise_fix_distance,
            sample_steps=args.sample_steps,
        )

        # [B*ns, 1, T, C] -> [B, ns, T, C]（与单 batch 时 pred[:, 0, ...] 语义一致）
        _bns, _one, _T, _C = pred_trajs_tensor.shape
        pred_trajs_tensor = pred_trajs_tensor.reshape(B, num_samples, _one, _T, _C)[
            :, :, 0, :, :
        ]
        _bns2, _one2, _T2, _C2 = pred_fix_distance_path_tensor.shape
        pred_fix_distance_path_tensor = pred_fix_distance_path_tensor.reshape(
            B, num_samples, _one2, _T2, _C2
        )[:, :, 0, :, :]

        for b in range(B):
            date_str = dates[b] if b < len(dates) else None
            dlp = data_label_paths[b] if b < len(data_label_paths) else None

            tmp = {}
            tmp["data_label_path"] = dlp
            tmp["date"] = date_str
            tmp["input_noise"] = input_noise[b * num_samples : (b + 1) * num_samples].cpu()
            tmp["input_noise_fix_distance"] = input_noise_fix_distance[
                b * num_samples : (b + 1) * num_samples
            ].cpu()
            for key, val in data["model_input"].items():
                tmp[key] = val[b].cpu()

            tmp.update(
                {
                    "pred_traj": pred_trajs_tensor[b, :, :, :2].cpu(),
                    "pred_traj_fixed": pred_fix_distance_path_tensor[b, :, :, :2].cpu(),
                    "pred_yaw": pred_trajs_tensor[b, :, :, 2].cpu(),
                    "pred_prob": None,
                    "pred_prob_fixed": None,
                    "agents_importance": None,
                    "laneline_importance": None,
                    "pred_v": None,
                }
            )
            tmp.update(
                {
                    "egolight_ori": data_feaNotEmb[b]["egolight_ori"],
                    "navitopo_pts_ori": data_feaNotEmb[b]["navitopo_pts_ori"],
                    "del_accLight_mask": data_feaNotEmb[b]["del_accLight_mask"],
                }
            )

            if args.eval_half:
                all_save_dict[local_sample_idx] = tmp
            else:
                filename = osp.join(
                    save_dir, f"{args.rank * frame_per_gpu + local_sample_idx}.pkl"
                )
                with open(filename, "wb") as f:
                    pickle.dump(tmp, f)
            local_sample_idx += 1

    if args.eval_half:
        all_data_filename = osp.join(save_dir, f"res_{args.rank}_{args.eval_half}.pkl")
        with open(all_data_filename, "wb") as f:
            pickle.dump(all_save_dict, f)
            print(f"save pkl to {all_data_filename}")


def infer_dump_datas(model, dump_datas, param):
    model.eval()
    world_size = get_world_size()
    assert args.max_frame % world_size == 0, "max_frame % world_size should be 0"
    frame_per_gpu = args.max_frame // world_size
    save_dir = osp.join(args.work_dir, "infer", "frames_sdk_dump")
    if args.rank == 0 and not osp.exists(save_dir):
        os.makedirs(save_dir)
    dist.barrier()

    for i in tqdm(range(args.max_frame)):
        data = dump_datas.iter_input_datas(i)
        data = custom_to_cuda(data)
        input_noise = data["model_input"]["input_noise"]
        input_noise_fix_distance = data["model_input"]["input_noise_fix_distance"]
        pred_trajs_tensor, pred_fix_distance_path_tensor = model.model.sample(
            data["model_input"],
            input_noise,
            input_noise_fix_distance,
            sample_steps=args.sample_steps,
        )
        pred_trajs_tensor = pred_trajs_tensor[:, 0, ...]
        pred_fix_distance_path_tensor = pred_fix_distance_path_tensor[:, 0, ...]

        tmp = {}
        for key, val in data["model_input"].items():
            if key == "navitopo_pts_ori":
                tmp[key] = val.cpu()
            else:
                tmp[key] = val[0].cpu()

        tmp.update(
            {
                "pred_traj": pred_trajs_tensor[:, :, :2].cpu(),
                "pred_traj_fixed": pred_fix_distance_path_tensor[:, :, :2].cpu(),
                "pred_yaw": pred_trajs_tensor[:, :, 2].cpu(),
                "pred_prob": None,
                "pred_prob_fixed": None,
                "agents_importance": None,
                "laneline_importance": None,
                "pred_v": None,
            }
        )
        sdk_traj = dump_datas.iter_output_datas(i)
        raw_data = sdk_traj.get("model_input", {})

        tmp.update(
            {
                "sdk_pred_traj": raw_data.get("trajectory")[:, :, :2],
                "sdk_pred_traj_fixed": raw_data.get("trajectory_fixed")[:, :, :2],
                "sdk_pred_yaw": raw_data.get("trajectory")[:, :, 3],
                "sdk_pred_prob": None,
                "sdk_pred_prob_fixed": None,
                "sdk_agents_importance": None,
                "sdk_laneline_importance": None,
                "sdk_pred_v": None,
            }
        )
        filename = osp.join(save_dir, f"{args.rank * frame_per_gpu + i}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(tmp, f)


def build_model(param, args_):
    print(f"加载模型: {args_.ckpt}")
    config_path = os.path.join(os.getcwd(), "config", "model", "LitMMDiTDiffusionModel.yaml")
    model_cls = LITMODEL.module_dict["LitMMDiTDiffusionModel"]
    model = model_cls(config=config_path)
    ckpt = torch.load(args_.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.eval()
    model = model.to("cuda")

    print(f"运行设备: {model.device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型加载完成, 参数数量: {total_params:,}")
    return model


def main(param, args_):
    global args
    args = args_

    client = create_remote_client(param)

    if param.get("date_split_file_test"):
        _ = client.load_json(param["date_split_file_test"])

    if args.eval_half:
        print(f"args.eval_half: {args.eval_half}, eval_split_mode: {args.eval_split_mode}")

        eval_split_idx = int(param["eval_split"][args.eval_split_mode])

        tmp_data_dir = param["tmp_data_dir_eval"][eval_split_idx] + f"_{args.eval_half}"
        _ = param["date_split_file_eval"][eval_split_idx].split(".json")[0] + f"_{args.eval_half}" + ".json"
        _ = client.load_json(param["date_split_file_test"])
        print(f"tmp_data_dir: {tmp_data_dir}, date_split_file: eval half mode ")

        param["tmp_data_dir_test"] = tmp_data_dir
        param["scene_extract_test"] = param["scene_extract_eval"][eval_split_idx]
        param["from_ceph_test"] = True
        param["change_light_file"] = param["change_light_file"][eval_split_idx]

    full_dataset = DlpDataset(param, is_train=False)
    args.max_frame = len(full_dataset)
    print(f"args.max_frame (total frames): {args.max_frame}")

    if args.sample_stride > 1:
        all_indices = np.arange(args.max_frame)
        sampled_indices = all_indices[:: args.sample_stride]
        if args.rank == 0:
            print(
                f"[main] sample_stride={args.sample_stride}, "
                f"using {len(sampled_indices)} / {args.max_frame} frames for infer."
            )
        data = torch.utils.data.Subset(full_dataset, sampled_indices)
    else:
        data = full_dataset

    bs = max(1, int(args.infer_batch_size))
    sampler_test = DistributedSampler(data, shuffle=False)
    batch_sampler_test = torch.utils.data.BatchSampler(
        sampler_test, batch_size=bs, drop_last=False
    )
    dataloader = DataLoader(
        data,
        batch_size=1,
        batch_sampler=batch_sampler_test,
        collate_fn=plantf_collate_fn_eval_multibatch,
        num_workers=4,
    )

    model = build_model(param, args)
    infer(model, dataloader, param)


def sdk_dump_infer_main(param, args_):
    global args
    args = args_
    dump_datas = DumpData(args.sdk_dump_dir)
    args.max_frame = dump_datas.get_max_frame()
    model = build_model(param, args)
    infer_dump_datas(model, dump_datas, param)
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--eval_half", type=str, default="")
    parser.add_argument("--eval_split_mode", type=str, default="eval_wuhan_v3")
    parser.add_argument("--sdk_dump_dir", default=None, type=str)
    parser.add_argument("--case_dir", type=str, default="")
    parser.add_argument("--noise_dir", type=str, default="")
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
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=1,
        help="DataLoader batch size for infer (>1 加速；--noise_dir 时仅支持 1)。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_printoptions(precision=4, threshold=float("inf"))
    args = parse_args()
    init_distributed_mode(args)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    cfg_path = args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        param = yaml.safe_load(f)
    if args.case_dir:
        param["tmp_data_dir_test"] = os.path.join(args.case_dir, "scene")
        param["date_split_file_test"] = os.path.join(args.case_dir, "date_split.json")
    if args.rank == 0:
        print(param)
    main(param, args)
    if args.sdk_dump_dir:
        print("===== sdk dump infer =====")
        sdk_dump_infer_main(param, args)
    os.system("rm -rf configs/tmp.py")
