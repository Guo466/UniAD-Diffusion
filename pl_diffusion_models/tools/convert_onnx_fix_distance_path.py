#!/usr/bin/env python3
import argparse
import os
import torch
import yaml
import numpy as np
import json
# import onnxruntime as ort
import pickle
from typing import Dict, Any, Tuple
import datetime
import plotly.graph_objects as go
from collections import OrderedDict
import logging
logger = logging.getLogger('Deployment')
import time
# Setup project root and imports
import sys
# import pdb;pdb.set_trace()
current_dir = os.getcwd() # /iag_ad_01/ad/pwb_deploy_diffusion/v1_0/dlp_diffusion_rl/tools'
# root_dir = current_dir + '/../pl_diffusion_models_v2-bak/'
# root_dir = current_dir
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
os.chdir(root_dir)

# from models import LitMMDiTDiffusionModelPointnet
# from datasets import LitDiffusionDatasetOSSNocut
# from datasets.diffusion_dataset_ego_navi_fix_distance_path_oss_dlp import DlpDataset

# from models.scene_model import SceneModel
# from models.mmdit_diffusion_model_fix_distance_path_fp16 import MMDiTDiffusionModel
from models.mmdit_diffusion_model_fix_distance_path import MMDiTDiffusionModel
from utils.data_utils import load_config

def read_noise_txt(txt_path):
    """读取噪声文件"""
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    return np.array([[float(x) for x in line.split(',')] for line in lines])

def build_model_from_config(config_path: str) -> MMDiTDiffusionModel:
    full_cfg = load_config(config_path)
    model_cfg = full_cfg["model"] if "model" in full_cfg else full_cfg
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = MMDiTDiffusionModel(model_cfg)
    return model


def load_checkpoint_strict(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # support both direct state_dict and PL checkpoints
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace('model.', ''):v for k,v in state_dict.items()}
    # If exported from LightningModule, keys may be prefixed with 'model.'
    # Try direct load first, then try to strip 'model.'
    # try:
    #     model.load_state_dict(state_dict, strict=True)
    #     return
    # except Exception:
    #     pass
    # import pdb; pdb.set_trace()
    model.load_state_dict(state_dict, strict=True)
    # stripped = {}
    # for k, v in state_dict.items():
    #     nk = k
    #     if nk.startswith("model."):
    #         nk = nk[len("model.") :]
    #     stripped[nk] = v
    # model.load_state_dict(stripped, strict=True)


def create_dummy_inputs(
    batch_size: int,
    num_agents: int,
    num_lanes: int,
    num_lane_points: int,
    # num_tl: int,
    # num_tl_steps: int,
    num_routes: int,
    num_route_points: int,
    num_navis: int,
    num_navi_points: int,
    **kwargs
):
    device = torch.device("cpu")
    # Encoder forward expects 9 elements as per SceneEncoder.forward unpacking
    # raw_agent_features: [B,A,T,C_a]  where C_a>=? model uses indices up to 27, last channel is valid flag
    T = kwargs.get('T', 11) # 21
    C_a = kwargs.get('C_a', 33) # 43  # matches agent_feature_encode_mlp input
    raw_agent_features = torch.zeros(batch_size, num_agents, T, C_a, device=device)

    # raw_agent_pos: [B,A,3]  (x, y, heading)
    raw_agent_pos = torch.zeros(batch_size, num_agents, 3, device=device)

    # raw_static_map_features: [B,L,N,C_m] last channel is valid flag; C_m must be 8 per map_feature_encode_mlp
    C_m = 8
    raw_static_map_features = torch.zeros(batch_size, num_lanes, num_lane_points, C_m, device=device)

    # raw_map_pl_pos: [B,L,2]
    raw_map_pl_pos = torch.zeros(batch_size, num_lanes, 2, device=device)

    # raw_dynamic_map_features: [B,TL,T,C_tl], last channel valid flag; C_tl must be 10 per tl_feature_encode_mlp
    # C_tl = 10
    # raw_dynamic_map_features = torch.zeros(batch_size, num_tl, num_tl_steps, C_tl, device=device)

    # raw_ego_routing_features: [B,R,N,C_r], last channel valid flag; C_r must be 7 per ego_routing_feature_encode_mlp
    C_r = 7
    raw_ego_routing_features = torch.zeros(batch_size, num_routes, num_route_points, C_r, device=device)

    # raw_ego_turn_light_features: [B,3]
    raw_ego_turn_light_features = torch.zeros(batch_size, 3, device=device)

    # raw_ego_traffic_light_features: [B,1,10]
    raw_ego_traffic_light_features = torch.zeros(batch_size, 1, 10, device=device)

    # raw_ego_navi_features: [B,1,10]
    raw_ego_navi_features = torch.zeros(batch_size, num_navis , num_navi_points, C_r, device=device)

    # mask_info: components_num -> [B,4]  (A_real, L_real, TL_real, R_real)
    mask_info = torch.tensor(
        [[num_agents, num_lanes, 0, num_routes, num_navis, 0] for _ in range(batch_size)],
        dtype=torch.float32,
        device=device,
    )

    input_noise = torch.randn((batch_size, 1, 80, 3), device=device)
    # import pdb; pdb.set_trace()
    # sample_steps = torch.tensor([5], device=device, dtype=torch.float32)
    sample_steps = torch.full((batch_size,1), 5.0, device=device, dtype=torch.float32)

    return (
        raw_agent_features,
        # raw_agent_pos,
        raw_static_map_features,
        # raw_map_pl_pos,
        # raw_dynamic_map_features,
        raw_ego_routing_features,
        # raw_ego_turn_light_features,
        raw_ego_traffic_light_features,
        raw_ego_navi_features,
        mask_info,
        input_noise,
        sample_steps,
    )

def get_input_info(self):
        input_names = []
        input_data = [] 

        T = 21
        batch_size = 1
        agent_status = torch.Tensor(batch_size, 50, T, 6).cuda()
        agent_status = agent_status.to(torch.float).cuda()
        input_data.append(agent_status)
        input_names.append("agent_status")

        agent_attrs = torch.Tensor(batch_size, 1, 50, 3)
        agent_attrs[:, :, :, -1] = torch.randint(self.model.encoder.agent_encoder.type_emb.weight.shape[0], size=(1, 1, 50))
        agent_attrs = agent_attrs.to(torch.float).cuda()
        input_data.append(agent_attrs)
        input_names.append("agent_attrs")

        agent_time_mask = torch.ones(batch_size, 1, 50, T, dtype=torch.float).cuda()
        input_data.append(agent_time_mask)
        input_names.append("agent_time_mask")

        laneline_pts = torch.Tensor(batch_size, 50, 200, 2).cuda()
        input_data.append(laneline_pts)
        input_names.append("laneline_pts")

        laneline_attrs = torch.Tensor(batch_size, 1, 50, 3)
        laneline_attrs[:, :, :, 0] = torch.randint(self.model.encoder.map_encoder.color_emb.weight.shape[0], size=(1, 1, 50))
        laneline_attrs[:, :, :, 1] = torch.randint(self.model.encoder.map_encoder.laneline_type_emb.weight.shape[0], size=(1, 1, 50))
        laneline_attrs[:, :, :, 2] = torch.randint(self.model.encoder.map_encoder.laneline_style_emb.weight.shape[0], size=(1, 1, 50))
        #laneline_attrs[:, :, :, 3] = torch.randint(self.model.encoder.map_encoder.other_type_emb.weight.shape[0], size=(1, 1, 50))
        laneline_attrs = laneline_attrs.cuda()

        laneline_mask = torch.zeros(batch_size, 1, 1, 50, dtype=torch.float).cuda()
        input_data.append(laneline_mask)
        input_names.append("laneline_mask")

        ego_curr_status = torch.Tensor(batch_size, 1, 1, 4).cuda()
        ego_curr_status[:, :, :, 2] = torch.randint(self.model.encoder.agent_encoder.traffice_light_embed.weight.shape[0], size=(1, 1, 1))
        ego_curr_status[:, :, :, 3] = torch.randint(self.model.encoder.agent_encoder.car_light_embed.weight.shape[0], size=(1, 1, 1))
        input_data.append(ego_curr_status)
        input_names.append("ego_curr_status")

        np.savetxt(os.path.join(self.save_path, 'ae_type_emb.txt'), self.model.encoder.agent_encoder.type_emb.weight.detach().cpu().numpy())
        category = torch.cat([
            torch.zeros((batch_size, 1), dtype=torch.int).cuda(),
            agent_attrs[:, :, :, -1].squeeze(0).to(torch.int)], dim=1)   # torch.Size([bs, 101])
        category_feature = self.model.encoder.agent_encoder.type_emb(category)
        input_data.append(category_feature.unsqueeze(0))
        input_names.append("category_feature")

        np.savetxt(os.path.join(self.save_path, 'ae_traffic_light_emb.txt'), self.model.encoder.agent_encoder.traffice_light_embed.weight.detach().cpu().numpy())
        traffic_light = torch.cat([
            ego_curr_status[..., 2].squeeze(0).to(torch.int),
            torch.zeros((batch_size, 50), dtype=torch.int).cuda()], dim=1)
        traffic_light_feature = self.model.encoder.agent_encoder.traffice_light_embed(traffic_light)
        input_data.append(traffic_light_feature.unsqueeze(0))
        input_names.append("traffic_light_feature")

        np.savetxt(os.path.join(self.save_path, 'ae_car_light_emb.txt'), self.model.encoder.agent_encoder.car_light_embed.weight.detach().cpu().numpy())
        car_light = torch.cat([
            ego_curr_status[..., 3].squeeze(0).to(torch.int),
            torch.zeros((batch_size, 50), dtype=torch.int).cuda()], dim=1)
        car_light_feature = self.model.encoder.agent_encoder.car_light_embed(car_light)
        input_data.append(car_light_feature.unsqueeze(0))
        input_names.append("car_light_feature")

        np.savetxt(os.path.join(self.save_path, 'me_color_emb.txt'), self.model.encoder.map_encoder.color_emb.weight.detach().cpu().numpy())
        polygon_color_feature = self.model.encoder.map_encoder.color_emb(laneline_attrs[..., 0].squeeze(0).to(torch.int))
        input_data.append(polygon_color_feature.unsqueeze(0))
        input_names.append("polygon_color_feature")

        np.savetxt(os.path.join(self.save_path, 'me_laneline_style_emb.txt'), self.model.encoder.map_encoder.laneline_style_emb.weight.detach().cpu().numpy())
        polygon_laneline_style_feature = self.model.encoder.map_encoder.laneline_style_emb(laneline_attrs[..., 2].squeeze(0).to(torch.int))
        input_data.append(polygon_laneline_style_feature.unsqueeze(0))
        input_names.append("polygon_laneline_style_feature")

        np.savetxt(os.path.join(self.save_path, 'me_laneline_type_emb.txt'), self.model.encoder.map_encoder.laneline_type_emb.weight.detach().cpu().numpy())
        polygon_laneline_type_feature = self.model.encoder.map_encoder.laneline_type_emb(laneline_attrs[..., 1].squeeze(0).to(torch.int))
        input_data.append(polygon_laneline_type_feature.unsqueeze(0))
        input_names.append("polygon_laneline_type_feature")

        
        #np.savetxt(os.path.join(self.save_path, 'me_other_type_emb.txt'), self.model.encoder.map_encoder.other_type_emb.weight.detach().cpu().numpy())
        #polygon_other_type_feature = self.model.encoder.map_encoder.other_type_emb(laneline_attrs[..., 3].squeeze(0).to(torch.int))
        #input_data.append(polygon_other_type_feature.unsqueeze(0))
        #input_names.append("polygon_other_type_feature")

        np.savetxt(os.path.join(self.save_path, 'ae_query_emb.txt'), self.model.agent_query.weight.detach().cpu().numpy())
        agent_query = self.model.agent_query
        agent_query = agent_query.weight[1:2].unsqueeze(0).repeat(1, 1+50, 1)
        agent_attrs_index = input_names.index("agent_attrs")
        agent_attrs_tensor = input_data[agent_attrs_index]  # [1, 1, max_agents, 3]
        p_mask = agent_attrs_tensor[..., 2].squeeze(1) == 1
        b_mask = agent_attrs_tensor[..., 2].squeeze(1) == 3
        agent_query[:,1:, :][p_mask] = self.model.agent_query.weight[:1]
        agent_query[:,1:, :][b_mask] = self.model.agent_query.weight[2:3] 
        input_data.append(agent_query.unsqueeze(0))
        input_names.append("agent_query")
        occ_map=torch.zeros(1, 1, 120, 275, dtype = torch.float).cuda()  
        input_data.append(occ_map)
        input_names.append("occ_map")
        navitopo_pts = torch.Tensor(1, 5, 200, 2).cuda()
        input_data.append(navitopo_pts)
        input_names.append("navitopo_pts")

        navitopo_mask = torch.zeros(1, 1, 1, 5, dtype=torch.float).cuda()
        input_data.append(navitopo_mask)
        input_names.append("navitopo_mask")
        input_noise = torch.randn(batch_size, 1, 80, 3, dtype=torch.float).cuda()
        sample_steps = torch.full((batch_size,1), 5.0, dtype=torch.float)
        input_data.append(input_noise)
        input_names.append("input_noise")
        input_data.append(sample_steps)
        input_names.append("sample_steps")
        
        return input_data, input_names



def load_real_data_from_dataset_mapping(dataset_mapping_path: str, test_data_dir: str, frame_idx: int = 0):
    """从dataset_mapping.pkl加载真实数据"""
    print(f"=== 加载真实数据：{dataset_mapping_path} ===")
    
    # 加载dataset_mapping.pkl
    with open(dataset_mapping_path, 'rb') as f:
        test_data_file_paths = pickle.load(f)
    
    print(f"找到 {len(test_data_file_paths)} 个数据文件")

    # 加载指定帧的数据
    if frame_idx >= len(test_data_file_paths):
        print(f"警告：帧索引 {frame_idx} 超出范围，使用第0帧")
        frame_idx = 0
    
    files = list(test_data_file_paths.keys())
    # import pdb; pdb.set_trace()
    test_data_file_path = files[frame_idx]
    data_root = '/'.join(test_data_dir.split('/'))
    full_data_path = os.path.join(data_root, test_data_file_paths[test_data_file_path], test_data_file_path)
    
    print(f"加载数据文件：{full_data_path}")
    with open(full_data_path, 'rb') as f:
        test_data = pickle.load(f)
    return test_data, full_data_path


def create_real_inputs_from_data_bak(test_data, device: torch.device):
    import pdb; pdb.set_trace()
    """从真实数据创建模型输入"""
    # 提取特征数据
    static_map_features = test_data[0]['static_map_features']  # [L, N, C_m]
    ego_routing_features = test_data[0]['ego_routing_features']  # [R, N, C_r]
    ego_traffic_light_features = test_data[0]['ego_traffic_light_features']  # [1, 10]
    valid_agents_features = test_data[0]['valid_agents_features']  # [A, T, C_a]
    
    # 转换为tensor并添加batch维度
    batch_size = 8
    
    # raw_agent_features: [B, A, T, C_a]
    raw_agent_features = torch.from_numpy(valid_agents_features).unsqueeze(0).to(device).to(torch.float32)
    num_agents = raw_agent_features.shape[1]
    
    # raw_static_map_features: [B, L, N, C_m]
    raw_static_map_features = torch.from_numpy(static_map_features).unsqueeze(0).to(device).to(torch.float32)
    num_lanes = raw_static_map_features.shape[1]
    num_lane_points = raw_static_map_features.shape[2]
    
    # raw_ego_routing_features: [B, R, N, C_r]
    raw_ego_routing_features = torch.from_numpy(ego_routing_features).unsqueeze(0).to(device).to(torch.float32)
    num_routes = raw_ego_routing_features.shape[1]
    num_route_points = raw_ego_routing_features.shape[2]
    
    # raw_ego_traffic_light_features: [B, 1, 10]
    raw_ego_traffic_light_features = torch.from_numpy(ego_traffic_light_features).unsqueeze(0).to(device).to(torch.float32)
    
    # mask_info: [B, 4] (A_real, L_real, TL_real, R_real)
    mask_info = torch.tensor(
        [[num_agents, num_lanes, 0, num_routes]],
        dtype=torch.float32,
        device=device,
    )
    
    # input_noise: [B, 1, 80, 3]
    input_noise = torch.randn((batch_size, 1, 80, 3), device=device)
    
    # sample_steps: [B, 1]
    sample_steps = torch.full((batch_size, 1), 5.0, device=device, dtype=torch.float32)
    
    return (
        raw_agent_features,
        raw_static_map_features,
        raw_ego_routing_features,
        raw_ego_traffic_light_features,
        mask_info,
        input_noise,
        sample_steps,
    ), {
        'num_agents': num_agents,
        'num_lanes': num_lanes,
        'num_lane_points': num_lane_points,
        'num_routes': num_routes,
        'num_route_points': num_route_points
    }

def create_real_inputs_from_data(data_path, device: torch.device, dataset_config, args, return_context: bool = False):
    batch_size = 1 #8

    viz_config = dataset_config
    # viz_config = 'LitDiffusionDataset_full_oss_nocut_agent21.yaml' # 'LitDiffusionDataset_viz.yaml'
    print(f"=== 加载数据配置：{os.path.join(root_dir, 'config', 'dataset', viz_config)} ===")
    data_config_path = os.path.join(root_dir, 'config', 'dataset', viz_config) # 'LitDiffusionDataset_viz.yaml'
    data_config = load_config(data_config_path)
    dataset = DiffusionDataset(data_config=data_config, is_train=False, cache_data_to_disk=False, num_workers=1)
    dataset.data_list = [data_path]
    # import pdb; pdb.set_trace()
    test_data = dataset.__getitem__(10)
    # test_data = dataset.process_raw_data(0)
    data_tensor_dict = DiffusionDataset.collate_fn([test_data])
    # 解析数据字典
    model_input_dict, gt_info_dict, metrics_input = \
        data_tensor_dict['model_input'], data_tensor_dict['gt_info'], data_tensor_dict['motion_metrics_input']
    # import pdb; pdb.set_trace()
    raw_agent_features = model_input_dict['valid_agents_features']
    raw_static_map_features = model_input_dict['static_map_features']
    raw_ego_routing_features = model_input_dict['ego_routing_features']
    raw_ego_traffic_light_features = model_input_dict['ego_traffic_light_features']
    raw_ego_navi_features = model_input_dict['ego_navi_features']
    mask_info = model_input_dict['components_num']
    # import pdb; pdb.set_trace()
    if batch_size > 1:
        raw_agent_features = raw_agent_features.repeat(batch_size, *([1]*len(raw_agent_features[0].shape)))
        raw_static_map_features = raw_static_map_features.repeat(batch_size, *([1]*len(raw_static_map_features[0].shape)))
        raw_ego_routing_features = raw_ego_routing_features.repeat(batch_size, *([1]*len(raw_ego_routing_features[0].shape)))
        raw_ego_traffic_light_features = raw_ego_traffic_light_features.repeat(batch_size, *([1]*len(raw_ego_traffic_light_features[0].shape)))
        raw_ego_navi_features = raw_ego_navi_features.repeat(batch_size, *([1]*len(raw_ego_navi_features[0].shape)))
        mask_info = mask_info.repeat(batch_size, *([1]*len(mask_info[0].shape)))

    # import pdb; pdb.set_trace()
    # input_noise: [B, 1, 80, 3]
    # input_noise = torch.randn((batch_size, 1, 80, 3), device=device)
    noise_txt_path = 'visualization/noise_to_save.txt'
    input_noise = read_noise_txt(noise_txt_path)
    input_noise = torch.from_numpy(input_noise).to(device).to(torch.float32).reshape(8, 1, 80, 3)[:batch_size]

    # sample_steps: [B, 1]
    sample_steps = torch.full((batch_size, 1), 5.0, device=device, dtype=torch.float32)
    inputs = (
        raw_agent_features,
        raw_static_map_features,
        raw_ego_routing_features,
        raw_ego_traffic_light_features,
        raw_ego_navi_features,
        mask_info,
        input_noise,
        sample_steps,
    )

    if args.visualize_results and not args.use_real_data:
        print("⚠️  Visualization requires --use_real_data. 自动关闭可视化。")
        args.visualize_results = False

    viz_context = None
    if return_context:
        viz_context = {
            "test_data": test_data,
            "data_tensor_dict": data_tensor_dict,
            "data_path": data_path,
            "dataset_config": dataset_config
        }

    return inputs, viz_context


# def save_tensor_dict(tensor_dict: Dict[str, torch.Tensor], save_dir: str, prefix: str = "") -> Dict[str, Any]:
#     """保存tensor字典为npy文件和shape信息"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     info_dict = {}
#     for name, tensor in tensor_dict.items():
#         # 转换为numpy数组
#         numpy_array = tensor.detach().cpu().numpy()
        
#         # 保存npy文件
#         npy_path = os.path.join(save_dir, f"{prefix}{name}.npy")
#         np.save(npy_path, numpy_array)
        
#         # 保存shape信息
#         info_dict[name] = {
#             "shape": list(tensor.shape),
#             "dtype": str(tensor.dtype),
#             "npy_path": npy_path,
#             "min": float(numpy_array.min()),
#             "max": float(numpy_array.max()),
#             "mean": float(numpy_array.mean()),
#             "std": float(numpy_array.std())
#         }
    
#     # 保存信息字典为JSON
#     info_path = os.path.join(save_dir, f"{prefix}tensor_info.json")
#     with open(info_path, 'w') as f:
#         json.dump(info_dict, f, indent=2)
    
#     return info_dict

def save_tensor_dict(tensor_dict: Dict[str, torch.Tensor], save_dir: str, prefix: str = "", frame_idx: int = 0) -> Dict[str, Any]:
    """保存tensor字典为JSON文件，包含所有tensor数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建包含所有tensor数据的字典
    tensor_data_dict = {}
    
    for name, tensor in tensor_dict.items():
        # 转换为numpy数组
        numpy_array = tensor.detach().cpu().numpy()
        
        # 直接保存所有tensor数据到字典，不检查大小
        tensor_data_dict[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "data": numpy_array.tolist(),  # 直接保存所有数据
            "min": float(numpy_array.min()),
            "max": float(numpy_array.max()),
            "mean": float(numpy_array.mean()),
            "std": float(numpy_array.std())
        }
    
    # 保存到单个JSON文件
    json_path = os.path.join(save_dir, f"{prefix}tensor_data.json")
    with open(json_path, 'w') as f:
        json.dump(tensor_data_dict, f, indent=2)
    
    print(f"已保存 {len(tensor_dict)} 个tensor到 {json_path}")
    
    return tensor_data_dict


def save_tensor_to_bin(tensor: torch.Tensor, bin_path: str) -> None:
    """将tensor保存为float32 bin文件"""
    numpy_array = tensor.detach().cpu().numpy().astype(np.float32)
    numpy_array.tofile(bin_path)


def save_io_as_bins(pytorch_results: Dict[str, torch.Tensor], save_dir: str, frame_idx: int = 0) -> Dict[str, Dict[str, str]]:
    """
    将输入输出tensor分别保存到bin文件。
    目录结构：
        save_dir/
            inputs/<feature>/<frame_idx>.bin
            outputs/<feature>/<frame_idx>.bin
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths: Dict[str, Dict[str, str]] = {"inputs": {}, "outputs": {}}
    
    for group in ["inputs", "outputs"]:
        group_dir = os.path.join(save_dir, group)
        os.makedirs(group_dir, exist_ok=True)
        for name, tensor in pytorch_results[group].items():
            feature_dir = os.path.join(group_dir, name)
            os.makedirs(feature_dir, exist_ok=True)
            bin_path = os.path.join(feature_dir, f"{0}.bin") # frame_idx
            save_tensor_to_bin(tensor, bin_path)
            saved_paths[group][name] = bin_path
            print(f"已保存{group[:-1]}特征 {name} 到 {bin_path}")
    
    return saved_paths


def tensor_to_ndarry(tensor: torch.Tensor, dtype=np.float32):
    """Tensor转numpy"""
    return tensor.detach().cpu().numpy().astype(dtype)


def add_static_map_to_figure(fig: go.Figure, test_data: Any) -> None:
    """在Plotly图中绘制静态地图折线"""
    static_map_features = test_data[0]["static_map_features"]
    static_map_xy = static_map_features[..., :2]
    static_map_valid = static_map_features[..., -1] > 0
    for map_seg_id in range(static_map_xy.shape[0]):
        valid_mask = static_map_valid[map_seg_id]
        coords = static_map_xy[map_seg_id][valid_mask]
        if coords.shape[0] == 0:
            continue
        fig.add_trace(
            go.Scatter(
                y=coords[:, 0],
                x=coords[:, 1],
                mode="lines",
                line=dict(color="silver", width=1),
                name=f"map_{map_seg_id}",
                showlegend=False,
            )
        )


def visualize_pytorch_vs_onnx(
    frame_idx: int,
    viz_context: Dict[str, Any],
    pytorch_results: Dict[str, torch.Tensor],
    onnx_results: Dict[str, np.ndarray],
    save_dir: str,
    max_agents: int = 5,
) -> str:
    """使用Plotly可视化PyTorch与ONNX预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    test_data = viz_context["test_data"]
    data_tensor_dict = viz_context["data_tensor_dict"]
    model_input_dict = data_tensor_dict["model_input"]
    gt_info_dict = data_tensor_dict["gt_info"]
    metrics_input = data_tensor_dict["motion_metrics_input"]

    fig = go.Figure()
    add_static_map_to_figure(fig, test_data)

    valid_agents_features = tensor_to_ndarry(model_input_dict["valid_agents_features"][0])
    object_ids_raw = metrics_input["object_ids"][0]
    if isinstance(object_ids_raw, torch.Tensor):
        valid_agents_ids = [str(x.item()) for x in object_ids_raw]
    else:
        valid_agents_ids = list(object_ids_raw)
    target_agents_num = int(gt_info_dict["target_agents_num"][0])

    past_trajs = valid_agents_features[:, :, -5:-3]
    past_valid = valid_agents_features[:, :, -1] > 0

    gt_tensor = gt_info_dict["per_agent_centric_target_gt_for_loss"]
    if isinstance(gt_tensor, torch.Tensor):
        gt_trajs_all = tensor_to_ndarry(gt_tensor)
    else:
        gt_trajs_all = np.asarray(gt_tensor)
    gt_trajs = gt_trajs_all[:, :, :2] # 0, 

    pt_preds_all = tensor_to_ndarry(pytorch_results["outputs"]["pred_trajs"])
    onnx_preds_all = onnx_results["pred_trajs"].astype(np.float32)

    pred_agent_dim = min(pt_preds_all.shape[1], onnx_preds_all.shape[1])
    available_agents = min(
        target_agents_num,
        max_agents,
        len(valid_agents_ids),
        pred_agent_dim,
        gt_trajs.shape[0],
    )

    pt_preds = pt_preds_all[0, :available_agents, :, :2]
    onnx_preds = onnx_preds_all[0, :available_agents, :, :2]
    gt_trajs = gt_trajs[:available_agents]

    if available_agents == 0:
        print("⚠️  没有可用于可视化的目标智能体。")
        return ""

    agents_to_plot = available_agents

    for agent_idx in range(agents_to_plot):
        agent_id = str(valid_agents_ids[agent_idx])
        history_mask = past_valid[agent_idx]
        history = past_trajs[agent_idx][history_mask]
        if history.shape[0] > 1:
            fig.add_trace(
                go.Scatter(
                    y=history[:, 0],
                    x=history[:, 1],
                    mode="lines",
                    line=dict(color="gray", width=2),
                    name=f"{agent_id}_history",
                )
            )

        pt_traj = pt_preds[agent_idx]
        fig.add_trace(
            go.Scatter(
                y=pt_traj[:, 0],
                x=pt_traj[:, 1],
                mode="lines+markers",
                line=dict(color="royalblue", width=3),
                marker=dict(size=4),
                name=f"{agent_id}_pytorch",
            )
        )

        onnx_traj = onnx_preds[agent_idx]
        fig.add_trace(
            go.Scatter(
                y=onnx_traj[:, 0],
                x=onnx_traj[:, 1],
                mode="lines+markers",
                line=dict(color="orangered", width=3, dash="dash"),
                marker=dict(size=4),
                name=f"{agent_id}_onnx",
            )
        )

        gt_traj = gt_trajs[agent_idx]
        if gt_traj.shape[0] > 0:
            fig.add_trace(
                go.Scatter(
                    y=gt_traj[:, 0],
                    x=gt_traj[:, 1],
                    mode="lines",
                    line=dict(color="seagreen", width=2),
                    name=f"{agent_id}_gt",
                )
            )

    fig.update_layout(
        title=f"Frame {frame_idx}: PyTorch vs ONNX predictions",
        xaxis_title="Y",
        yaxis_title="X",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        height=800,
        width=800,
        plot_bgcolor="black",
    )

    save_path = os.path.join(save_dir, f"viz_frame_{frame_idx:04d}.html")
    fig.write_html(save_path, auto_play=False)
    print(f"=== 可视化结果保存到: {save_path} ===")
    return save_path


def save_all_frames_combined_data(all_frames_data: Dict[int, Dict[str, Any]], save_dir: str) -> str:
    """保存所有frame的输入输出数据到同一个JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建包含所有frame数据的字典
    all_frames_combined = {
        "total_frames": len(all_frames_data),
        "timestamp": datetime.datetime.now().isoformat(),
        "frames": {}
    }
    
    for frame_idx, frame_data in all_frames_data.items():
        frame_dict = {
            "inputs": {},
            "outputs": {}
        }
        
        # 保存输入数据
        for tensor_name, tensor_info in frame_data.get("inputs", {}).items():
            frame_dict["inputs"][tensor_name] = {
                "shape": tensor_info.get("shape", []),
                "dtype": tensor_info.get("dtype", ""),
                "data": tensor_info.get("data", []),
                "min": tensor_info.get("min", 0),
                "max": tensor_info.get("max", 0),
                "mean": tensor_info.get("mean", 0),
                "std": tensor_info.get("std", 0)
            }
        
        # 保存输出数据
        for tensor_name, tensor_info in frame_data.get("outputs", {}).items():
            frame_dict["outputs"][tensor_name] = {
                "shape": tensor_info.get("shape", []),
                "dtype": tensor_info.get("dtype", ""),
                "data": tensor_info.get("data", []),
                "min": tensor_info.get("min", 0),
                "max": tensor_info.get("max", 0),
                "mean": tensor_info.get("mean", 0),
                "std": tensor_info.get("std", 0)
            }
        
        all_frames_combined["frames"][f"frame_{frame_idx:04d}"] = frame_dict
    
    # 保存到单个JSON文件
    combined_path = os.path.join(save_dir, "all_frames_combined_io_data.json")
    with open(combined_path, 'w') as f:
        json.dump(all_frames_combined, f, indent=2)
    
    print(f"已保存所有frame的输入输出数据到 {combined_path}")
    print(f"总frame数量: {len(all_frames_data)}")
    
    return combined_path

# def save_all_frames_tensor_data(save_dir: str, all_frames_data: Dict[int, Dict[str, Any]]) -> str:
#     """保存所有frame_idx的tensor数据到同一个JSON文件"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 构建包含所有frame数据的字典
#     all_frames_dict = {}
    
#     for frame_idx, frame_data in all_frames_data.items():
#         frame_dict = {}
#         for tensor_name, tensor_info in frame_data.items():
#             if not isinstance(tensor_info, dict):
#                 continue
#             # 只保存关键信息，避免文件过大
#             frame_dict[tensor_name] = {
#                 "shape": tensor_info.get("shape", []),
#                 "dtype": tensor_info.get("dtype", ""),
#                 "min": tensor_info.get("min", 0),
#                 "max": tensor_info.get("max", 0),
#                 "mean": tensor_info.get("mean", 0),
#                 "std": tensor_info.get("std", 0),
#                 "frame_idx": frame_idx
#             }
        
#         all_frames_dict[f"frame_{frame_idx:04d}"] = frame_dict
    
#     # 保存到单个JSON文件
#     combined_path = os.path.join(save_dir, "all_frames_tensor_data.json")
#     with open(combined_path, 'w') as f:
#         json.dump(all_frames_dict, f, indent=2)
    
#     return combined_path


def process_all_frames(args, model, input_names, device):
    """处理所有frame_idx的输入输出和对齐检查"""
    if not args.use_real_data:
        print("警告：多frame处理仅支持真实数据模式")
        return None
    
    dataset_mapping_path = os.path.join(args.test_data_dir, 'dataset_mapping.pkl')
    if not os.path.exists(dataset_mapping_path):
        print(f"错误：dataset_mapping.pkl不存在于 {dataset_mapping_path}")
        return None
    
    # 加载dataset_mapping获取所有frame信息
    with open(dataset_mapping_path, 'rb') as f:
        test_data_file_paths = pickle.load(f)
    # import pdb; pdb.set_trace()
    total_frames = len(test_data_file_paths)
    print(f"=== 开始处理所有 {total_frames} 个frame ===")
    
    all_frames_results = {}
    all_frames_alignment = {}
    
    # 创建ONNX wrapper
    class MMDiTDiffusionModelONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, *model_input):
            pred_trajs = self.model.sample_onnx(model_input) 
            return pred_trajs
    
    wrapper = MMDiTDiffusionModelONNXWrapper(model)
    
    # 处理每个frame
    for frame_idx in range(min(total_frames, 10)):  # 限制处理前10个frame，避免时间过长
        print(f"处理frame {frame_idx}/{total_frames-1}")
        
        # try:
        # 加载当前frame的数据
        test_data, data_path = load_real_data_from_dataset_mapping(
            dataset_mapping_path, args.test_data_dir, frame_idx
        )
        inputs, _ = create_real_inputs_from_data(data_path, device, args.dataset_config, args, return_context=False)
        
        # 转换设备
        inputs = tuple(x.to(device).to(torch.float32) for x in inputs)
        
        # 运行PyTorch推理
        pytorch_results = run_pytorch_inference(wrapper, inputs, input_names)
        
        # 保存当前frame的tensor数据
        frame_save_dir = os.path.join(args.io_save_dir, f"frame_{frame_idx:04d}")
        os.makedirs(frame_save_dir, exist_ok=True)
        
        # 保存输入tensor
        input_info = save_tensor_dict(pytorch_results["inputs"], frame_save_dir, "pytorch_input_", frame_idx)
        # 保存输出tensor
        output_info = save_tensor_dict(pytorch_results["outputs"], frame_save_dir, "pytorch_output_", frame_idx)

        # 运行ONNX推理（如果已导出）
        if os.path.exists(args.onnx):
            onnx_results = run_onnx_inference(args.onnx, pytorch_results["inputs"])
            
            # 对齐检查
            alignment_results = compare_results(pytorch_results, onnx_results, args.tolerance)
            
            # 保存对齐结果
            alignment_path = os.path.join(frame_save_dir, "alignment_results.json")
            with open(alignment_path, 'w') as f:
                json.dump(alignment_results, f, indent=2)
            
            all_frames_alignment[frame_idx] = alignment_results
            
            # 打印当前frame的对齐结果
            print(f"Frame {frame_idx} 对齐结果:")
            all_passed = True
            for output_name, result in alignment_results.items():
                status = result["status"]
                max_diff = result["max_diff"]
                print(f"  {output_name}: {status} (max_diff: {max_diff:.2e})")
                if status != "PASS":
                    all_passed = False
            
            if all_passed:
                print(f"  Frame {frame_idx}: 🎉 所有输出通过对齐检查!")
            else:
                print(f"  Frame {frame_idx}: ⚠️ 部分输出未通过对齐检查!")
        
        # 保存当前frame的结果
        all_frames_results[frame_idx] = {
            "inputs": input_info,
            "outputs": output_info,
            "data_path": data_path
        }
            
        # except Exception as e:
        #     print(f"处理frame {frame_idx}时出错: {e}")
        #     continue
    
    # # 保存所有frame的汇总数据
    # if all_frames_results:
    #     import pdb; pdb.set_trace()
    #     combined_path = save_all_frames_tensor_data(args.io_save_dir, all_frames_results)
    #     print(f"所有frame的汇总数据已保存到: {combined_path}")
    
    # 保存所有frame的对齐结果汇总
    if all_frames_alignment:
        alignment_summary_path = os.path.join(args.io_save_dir, "all_frames_alignment_summary.json")
        with open(alignment_summary_path, 'w') as f:
            json.dump(all_frames_alignment, f, indent=2)
        print(f"所有frame的对齐结果汇总已保存到: {alignment_summary_path}")
    
    return all_frames_results

def run_pytorch_inference(model: torch.nn.Module, inputs: Tuple[torch.Tensor], input_names: list) -> Dict[str, torch.Tensor]:
    """运行PyTorch模型推理并返回结果"""
    model.eval()
    with torch.no_grad():
        # 运行模型推理
        output = model.sample_onnx(inputs)
        
        # 构建输出字典
        output_dict = {"pred_trajs": output[0], "fix_distance_pred_paths": output[1]}
        
        # 构建输入字典
        input_dict = {}
        for i, name in enumerate(input_names):
            input_dict[name] = inputs[i]
        
        return {"inputs": input_dict, "outputs": output_dict}


def run_onnx_inference(onnx_path: str, input_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """运行ONNX模型推理"""
    # 创建ONNX Runtime session
    session = ort.InferenceSession(onnx_path)
    
    # 准备输入
    ort_inputs = {}
    for name, tensor in input_dict.items():
        ort_inputs[name] = tensor.detach().cpu().numpy().astype(np.float32)
    # import pdb; pdb.set_trace()
    # 运行推理
    ort_outputs = session.run(None, ort_inputs)
    
    # 构建输出字典
    output_names = [output.name for output in session.get_outputs()]
    output_dict = {}
    for i, name in enumerate(output_names):
        output_dict[name] = ort_outputs[i]
    
    return output_dict


def compare_results(pytorch_results: Dict[str, torch.Tensor], 
                   onnx_results: Dict[str, np.ndarray], 
                   tolerance: float = 1e-5) -> Dict[str, Any]:
    """比较PyTorch和ONNX推理结果"""
    comparison_results = {}
    
    for output_name in pytorch_results["outputs"].keys():
        if output_name not in onnx_results:
            print(f"Warning: {output_name} not found in ONNX outputs")
            continue
            
        pytorch_tensor = pytorch_results["outputs"][output_name]
        onnx_array = onnx_results[output_name]
        
        # 转换为numpy数组进行比较
        pytorch_numpy = pytorch_tensor.detach().cpu().numpy()
        
        # 检查shape是否一致
        if pytorch_numpy.shape != onnx_array.shape:
            comparison_results[output_name] = {
                "status": "ERROR",
                "message": f"Shape mismatch: PyTorch {pytorch_numpy.shape} vs ONNX {onnx_array.shape}"
            }
            continue
        
        # 计算差异
        diff = np.abs(pytorch_numpy - onnx_array)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rmse = np.sqrt(np.mean(diff ** 2))
        
        # 检查是否在容忍范围内
        is_close = bool(max_diff <= tolerance)
        
        comparison_results[output_name] = {
            "status": "PASS" if is_close else "FAIL",
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "rmse": float(rmse),
            "tolerance": tolerance,
            "pytorch_shape": list(pytorch_numpy.shape),
            "onnx_shape": list(onnx_array.shape),
            "is_close": is_close
        }
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Export MMDiTDiffusionModel to ONNX")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.getcwd(), "config", "model", "LitMMDiTDiffusionModel.yaml"),
        help="Path to LitMMDiTDiffusionModel YAML config",
    )
    parser.add_argument("--ckpt", type=str, default="sampled_full_data_v1_map_pointnet_hs512_epoch28.ckpt", help="Optional checkpoint to load")
    parser.add_argument("--onnx", type=str, default="bs1_sampled_full_data_v1_map_pointnet_hs512_epoch28.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch", type=int, default=8, help="Dummy batch size")
    parser.add_argument("--model_version", type=str, default="DLP_v1", help="dlp_model_version")
    parser.add_argument("--agents", type=int, default=128, help="Dummy number of agents A")
    parser.add_argument("--lanes", type=int, default=512, help="Dummy number of lane segments L")
    parser.add_argument("--lane_points", type=int, default=11, help="Dummy number of lane points N")
    parser.add_argument("--navis", type=int, default=16, help="Dummy number of navi instances N") # ego navi num
    parser.add_argument("--navi_points", type=int, default=11, help="Dummy number of navi points N")
    # parser.add_argument("--tl", type=int, default=64, help="Dummy number of traffic light instances TL")
    # parser.add_argument("--tl_steps", type=int, default=1, help="Dummy TL history length T (should be 11)")
    parser.add_argument("--routes", type=int, default=64, help="Dummy number of routing instances R")
    parser.add_argument("--route_points", type=int, default=11, help="Dummy number of routing points N")
    parser.add_argument("--fixed_axes", default=True, action="store_true", help="Export with fixed (static) axes")
    parser.add_argument("--use_gpu", default=True, action="store_true", help="Export with CUDA if available")
    parser.add_argument("--T", type=int, default=11, help="Dummy batch size")
    parser.add_argument("--C_a", type=int, default=33, help="Dummy batch size")
    parser.add_argument("--save_io", action="store_true", help="Save model inputs and outputs for alignment check")
    parser.add_argument("--io_save_dir", type=str, default="./alignment_check", help="Directory to save IO tensors")
    parser.add_argument("--check_alignment", action="store_true", help="Run alignment check between PyTorch and ONNX")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Tolerance for alignment check")
    # 新增参数：使用真实数据
    parser.add_argument("--use_real_data", action="store_true", help="Use real data from dataset_mapping.pkl for alignment check")
    parser.add_argument("--test_data_dir", type=str, default="./test_data", help="Directory containing test data and dataset_mapping.pkl")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to use from dataset_mapping.pkl")
    parser.add_argument("--dataset_config", required=False, type=str, default='LitDiffusionDataset_ego_navi_fix_distance_path_oss.yaml', help="可视化结果保存目录（如：/path/to/viz_results）")
    parser.add_argument("--process_all_frames", action="store_true", help="Process all frame_idx from dataset_mapping.pkl")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum number of frames to process (to avoid excessive processing time)")
    parser.add_argument("--visualize_results", action="store_true", help="Visualize PyTorch vs ONNX trajectories")
    parser.add_argument("--viz_save_dir", type=str, default="./alignment_viz", help="Directory to save visualization html files")
    parser.add_argument("--viz_max_agents", type=int, default=5, help="Maximum number of agents to visualize")
    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    model = build_model_from_config(args.config)
    model.eval()
    def load_config(config_path: str):
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config

    # if args.ckpt is not None and os.path.isfile(args.ckpt):
    load_checkpoint_strict(model, args.ckpt)

    viz_context = None

    class MMDiTDiffusionModelONNXWrapper(torch.nn.Module):
        def __init__(self, model: MMDiTDiffusionModel, param, args, save_path=None):        
            super().__init__()
            self.model = model
            self.save_path = save_path
            self.model_name = "DLP"
            self.adela_name = args.model_version
            self.param = param
            self.args = args
        def forward(self, *model_input):
            # import pdb; pdb.set_trace()
            # pred_trajs = self.model.sample_onnx(model_input)
            pred_trajs = self.model.multi_mode_sample(model_input)
            return pred_trajs
        def get_input_info(self):
            input_names = []
            input_data = [] 

            # import pdb;pdb.set_trace()
            T = self.param['model']['encoder']['agent_hist_steps'] + 1
            batch_size = 1
            # max_n_agent = self.param['model_param']['max_n_agent']
            agent_status = torch.Tensor(batch_size, self.param['model_param']['max_n_agent'], T, 6).cuda()
            agent_status = agent_status.to(torch.float).cuda()
            input_data.append(agent_status)
            input_names.append("agent_status")

            agent_attrs = torch.Tensor(batch_size, 1, self.param['model_param']['max_n_agent'], 3)
            agent_attrs[:, :, :, -1] = torch.randint(self.model.encoder.agent_encoder.type_emb.weight.shape[0], size=(1, 1, self.param['model_param']['max_n_agent']))
            agent_attrs = agent_attrs.to(torch.float).cuda()
            input_data.append(agent_attrs)
            input_names.append("agent_attrs")

            agent_time_mask = torch.ones(batch_size, 1, self.param['model_param']['max_n_agent'], T, dtype=torch.float).cuda()
            input_data.append(agent_time_mask)
            input_names.append("agent_time_mask")

            laneline_pts = torch.Tensor(batch_size, self.param['model_param']['max_n_laneline'], 200, 2).cuda()
            input_data.append(laneline_pts)
            input_names.append("laneline_pts")

            laneline_attrs = torch.Tensor(batch_size, 1, self.param['model_param']['max_n_laneline'], 3)
            laneline_attrs[:, :, :, 0] = torch.randint(self.model.encoder.map_encoder.color_emb.weight.shape[0], size=(1, 1, self.param['model_param']['max_n_laneline']))
            laneline_attrs[:, :, :, 1] = torch.randint(self.model.encoder.map_encoder.laneline_type_emb.weight.shape[0], size=(1, 1, self.param['model_param']['max_n_laneline']))
            laneline_attrs[:, :, :, 2] = torch.randint(self.model.encoder.map_encoder.laneline_style_emb.weight.shape[0], size=(1, 1, self.param['model_param']['max_n_laneline']))
            #laneline_attrs[:, :, :, 3] = torch.randint(self.model.encoder.map_encoder.other_type_emb.weight.shape[0], size=(1, 1, self.param['model_param']['max_n_laneline']))
            laneline_attrs = laneline_attrs.cuda()

            laneline_mask = torch.zeros(batch_size, 1, 1, self.param['model_param']['max_n_laneline'], dtype=torch.float).cuda()
            input_data.append(laneline_mask)
            input_names.append("laneline_mask")

            ego_curr_status = torch.Tensor(batch_size, 1, 1, 4).cuda()
            ego_curr_status[:, :, :, 2] = torch.randint(self.model.encoder.agent_encoder.traffice_light_embed.weight.shape[0], size=(1, 1, 1))
            ego_curr_status[:, :, :, 3] = torch.randint(self.model.encoder.agent_encoder.car_light_embed.weight.shape[0], size=(1, 1, 1))
            input_data.append(ego_curr_status)
            input_names.append("ego_curr_status")

            np.savetxt(os.path.join(self.save_path, 'ae_type_emb.txt'), self.model.encoder.agent_encoder.type_emb.weight.detach().cpu().numpy())
            category = torch.cat([
                torch.zeros((batch_size, 1), dtype=torch.int).cuda(),
                agent_attrs[:, :, :, -1].squeeze(0).to(torch.int)], dim=1)   # torch.Size([bs, 101])
            category_feature = self.model.encoder.agent_encoder.type_emb(category)
            input_data.append(category_feature.unsqueeze(0))
            input_names.append("category_feature")

            np.savetxt(os.path.join(self.save_path, 'ae_traffic_light_emb.txt'), self.model.encoder.agent_encoder.traffice_light_embed.weight.detach().cpu().numpy())
            traffic_light = torch.cat([
                ego_curr_status[..., 2].squeeze(0).to(torch.int),
                torch.zeros((batch_size, self.param['model_param']['max_n_agent']), dtype=torch.int).cuda()], dim=1)
            traffic_light_feature = self.model.encoder.agent_encoder.traffice_light_embed(traffic_light)
            input_data.append(traffic_light_feature.unsqueeze(0))
            input_names.append("traffic_light_feature")

            np.savetxt(os.path.join(self.save_path, 'ae_car_light_emb.txt'), self.model.encoder.agent_encoder.car_light_embed.weight.detach().cpu().numpy())
            car_light = torch.cat([
                ego_curr_status[..., 3].squeeze(0).to(torch.int),
                torch.zeros((batch_size, self.param['model_param']['max_n_agent']), dtype=torch.int).cuda()], dim=1)
            car_light_feature = self.model.encoder.agent_encoder.car_light_embed(car_light)
            input_data.append(car_light_feature.unsqueeze(0))
            input_names.append("car_light_feature")

            np.savetxt(os.path.join(self.save_path, 'me_color_emb.txt'), self.model.encoder.map_encoder.color_emb.weight.detach().cpu().numpy())
            polygon_color_feature = self.model.encoder.map_encoder.color_emb(laneline_attrs[..., 0].squeeze(0).to(torch.int))
            input_data.append(polygon_color_feature.unsqueeze(0))
            input_names.append("polygon_color_feature")

            np.savetxt(os.path.join(self.save_path, 'me_laneline_style_emb.txt'), self.model.encoder.map_encoder.laneline_style_emb.weight.detach().cpu().numpy())
            polygon_laneline_style_feature = self.model.encoder.map_encoder.laneline_style_emb(laneline_attrs[..., 2].squeeze(0).to(torch.int))
            input_data.append(polygon_laneline_style_feature.unsqueeze(0))
            input_names.append("polygon_laneline_style_feature")

            np.savetxt(os.path.join(self.save_path, 'me_laneline_type_emb.txt'), self.model.encoder.map_encoder.laneline_type_emb.weight.detach().cpu().numpy())
            polygon_laneline_type_feature = self.model.encoder.map_encoder.laneline_type_emb(laneline_attrs[..., 1].squeeze(0).to(torch.int))
            input_data.append(polygon_laneline_type_feature.unsqueeze(0))
            input_names.append("polygon_laneline_type_feature")

        
            navitopo_pts = torch.Tensor(batch_size, 5, 200, 2).cuda()
            input_data.append(navitopo_pts)
            input_names.append("navitopo_pts")

            navitopo_mask = torch.zeros(batch_size, 1, 1, 5, dtype=torch.float).cuda()
            input_data.append(navitopo_mask)
            input_names.append("navitopo_mask")

            # add mean std

            mean_std = torch.Tensor(1, 6, 25).cuda()
            input_data.append(mean_std)
            input_names.append("mean_std")

            mean_std_fixed = torch.Tensor(1, 6, 40).cuda()
            input_data.append(mean_std_fixed)
            input_names.append("mean_std_fixed")

            occ_polygons_pts=torch.zeros(1, 80, 100, 2, dtype = torch.float).cuda()  
            input_data.append(occ_polygons_pts)
            input_names.append("occ_polygons_pts")

            occ_polygons_attrs=torch.zeros(1, 80, 5, dtype = torch.float).cuda()  
            input_data.append(occ_polygons_attrs)
            input_names.append("occ_polygons_attrs")

            occ_polygons_mask=torch.zeros(1, 80, dtype = torch.float).cuda()  
            input_data.append(occ_polygons_mask)
            input_names.append("occ_polygons_mask")

            # delta_x_mean = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_x_mean)
            # input_names.append("delta_x_mean")

            # delta_x_std = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_x_std)
            # input_names.append("delta_x_std")

            # delta_y_mean = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_y_mean)
            # input_names.append("delta_y_mean")

            # delta_y_std = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_y_std)
            # input_names.append("delta_y_std")

            # delta_fixed_x_mean = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_fixed_x_mean)
            # input_names.append("delta_fixed_x_mean")

            # delta_fixed_x_std = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_fixed_x_std)
            # input_names.append("delta_fixed_x_std")

            # delta_fixed_y_mean = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_fixed_y_mean)
            # input_names.append("delta_fixed_y_mean")

            # delta_fixed_y_std = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_fixed_y_std)
            # input_names.append("delta_fixed_y_std")

            # delta_yaw_mean = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_yaw_mean)
            # input_names.append("delta_yaw_mean")

            # delta_yaw_std = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_yaw_std)
            # input_names.append("delta_yaw_std")

            # delta_fixed_yaw_mean = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_fixed_yaw_mean)
            # input_names.append("delta_fixed_yaw_mean")

            # delta_fixed_yaw_std = torch.Tensor(1, 80).cuda()
            # input_data.append(delta_fixed_yaw_std)
            # input_names.append("delta_fixed_yaw_std")

            # input_noise_np = torch.randn(1, 1, 80, 3, dtype=torch.float).detach().cpu().numpy()
            # input_noise_2d = input_noise_np.reshape(-1, input_noise_np.shape[-1])  # 形状变为 (80, 3)
            # np.savetxt(os.path.join(self.save_path, 'input_noise.txt'), input_noise_2d)
            # np.savetxt(os.path.join(self.save_path, 'input_noise.txt'), torch.randn(1, 1, 80, 3, dtype=torch.float).detach().cpu().numpy())
            input_noise = torch.randn(self.args.batch, 1, 25, 3, dtype=torch.float).cuda()
            input_noise_fix_distance = torch.randn(self.args.batch, 1, 40, 3, dtype=torch.float).cuda()
            #  mm
            input_noise_np = input_noise.detach().cpu().numpy()
            input_noise_2d = input_noise_np.reshape(-1, input_noise_np.shape[-1])
            np.savetxt(os.path.join(self.save_path, 'input_noise.txt'), input_noise_2d)

            input_noise_fix_distance_np = input_noise_fix_distance.detach().cpu().numpy()  
            input_noise_fix_distance_2d = input_noise_fix_distance_np.reshape(-1, input_noise_fix_distance_np.shape[-1])
            np.savetxt(os.path.join(self.save_path, 'input_noise_fix_distance.txt'), input_noise_fix_distance_2d)
            sample_steps = torch.full((batch_size,1), 5.0, dtype=torch.float).cuda()
            input_data.append(input_noise)
            input_names.append("input_noise")
            input_data.append(input_noise_fix_distance)
            input_names.append("input_noise_fix_distance")
            input_data.append(sample_steps)
            input_names.append("sample_steps")


            return input_data, input_names

        def generate_meta_info(self, input_names, output_names):
            # generate parameters.json
            parameters = {
                "model_files": {
                    "net": {
                        "net": "{}.onnx".format(self.model_name),
                        "backend": "kestrel_nart",
                        "max_batch_size": 1,
                        "input": OrderedDict(),
                        "output":  OrderedDict()
                    }
                },
                "max_n_agent" : self.param['model_param']['max_n_agent'],
                "max_n_laneline" : self.param['model_param']['max_n_laneline'],
                "input_h": 544,
                "input_w": 960,
            }
            for input_name in input_names:
                parameters["model_files"]["net"]["input"][input_name] = input_name
            for output_name in output_names:
                parameters["model_files"]["net"]["output"][output_name] = output_name
            parameters_save_file = os.path.join(self.save_path, 'parameters.json')
            with open(parameters_save_file, 'w') as fp:
                json.dump(parameters, fp)
            logger.info(
                'parameters is saved to: {}'.format(parameters_save_file))

            # generate meta.json
            meta = {
                "model_name": "{}".format(self.adela_name),
                "model_type": "DLP",
                "version": {
                    "major": 1,
                    "minor": 0,
                    "patch": 0,
                    "train_date": "{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
                },
                "properties": {
                    "description": ""
                }
            }

            meta_save_file = os.path.join(self.save_path, 'meta.json')
            with open(meta_save_file, 'w') as fp:
                json.dump(meta, fp)
            logger.info('meta is saved to: {}'.format(meta_save_file))

            # tar: spetr.onnx parameters.json meta.json
            cmd = 'cd {} && tar cvf DLP.tar DLP.onnx parameters.json meta.json'.format(
                self.save_path)
            os.system(cmd)
            logger.info('[cmd] tar file: {}'.format(cmd))
            return
    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)
    save_path = os.path.join(os.path.dirname(args.ckpt), "deploy")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    param = load_config(args.config)
    wrapper = MMDiTDiffusionModelONNXWrapper(model, param, args, save_path)
    inputs, input_names = wrapper.get_input_info()
    
    # 根据参数选择使用虚拟数据或真实数据
    # if args.use_real_data:
    #     # 使用真实数据
    #     dataset_mapping_path = os.path.join(args.test_data_dir, 'dataset_mapping.pkl')
    #     if not os.path.exists(dataset_mapping_path):
    #         print(f"错误：dataset_mapping.pkl不存在于 {dataset_mapping_path}")
    #         return

    #     test_data, data_path = load_real_data_from_dataset_mapping(dataset_mapping_path, args.test_data_dir, args.frame_idx)
    #     inputs, viz_context = create_real_inputs_from_data(
    #         data_path, torch.device("cpu"), args.dataset_config, args, return_context=args.visualize_results
    #     )
        
    #     # real_data_info
    #     # print(f"=== 使用真实数据信息 ===")
    #     # print(f"智能体数量: {real_data_info['num_agents']}")
    #     # print(f"车道数量: {real_data_info['num_lanes']}")
    #     # print(f"车道点数: {real_data_info['num_lane_points']}")
    #     # print(f"路径数量: {real_data_info['num_routes']}")
    #     # print(f"路径点数: {real_data_info['num_route_points']}")

    # else:
        # 使用虚拟数据
        # inputs = create_dummy_inputs(
        #     args.batch,
        #     args.agents,
        #     args.lanes,
        #     args.lane_points,
        #     # args.tl,
        #     # args.tl_steps,
        #     args.routes,
        #     args.route_points,
        #     args.navis,
        #     args.navi_points,
        #     **vars(args)
        # )

    # import pdb; pdb.set_trace()
    # device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    # model.to(device)
    inputs = tuple(x.to(device).to(torch.float32) for x in inputs)

    # input_names = [
    #     "raw_agent_features",
    #     # "raw_agent_pos",
    #     "raw_static_map_features",
    #     # "raw_map_pl_pos",
    #     # "raw_dynamic_map_features",
    #     "raw_ego_routing_features",
    #     # "raw_ego_turn_light_features",
    #     "raw_ego_traffic_light_features",
    #     "raw_ego_navi_features",
    #     "mask_info",
    #     "input_noise",
    #     "sample_steps"
    # ]
    # import pdb; pdb.set_trace()
    # Output from SceneModel.forward is a tuple/list: (pred_trajs_list, pred_logits_list, agent_SSL, map_SSL, intent_mask)
    # We export the final layer outputs: last trajs [B,A,M,T,6], last logits [B,M]
    # To make that stable for ONNX, we wrap a small module.
    # class MMDiTDiffusionModelONNXWrapper(torch.nn.Module):
    #     def __init__(self, model: MMDiTDiffusionModel):        
    #         super().__init__()
    #         self.model = model
    #     def forward(self, *model_input):
    #         # import pdb; pdb.set_trace()
    #         pred_trajs = self.model.sample_onnx(model_input)
    #         return pred_trajs
    #     def get_input_info(self):
    #         input_names = []
    #         input_data = [] 

    #         T = 21

    #         agent_status = torch.Tensor(1, 50, T, 6).cuda()
    #         agent_status = agent_status.to(torch.float).cuda()
    #         input_data.append(agent_status)
    #         input_names.append("agent_status")

    #         agent_attrs = torch.Tensor(1, 1, 50, 3)
    #         agent_attrs[:, :, :, -1] = torch.randint(self.model.encoder.agent_encoder.type_emb.weight.shape[0], size=(1, 1, 50))
    #         agent_attrs = agent_attrs.to(torch.float).cuda()
    #         input_data.append(agent_attrs)
    #         input_names.append("agent_attrs")

    #         agent_time_mask = torch.ones(1, 1, 50, T, dtype=torch.float).cuda()
    #         input_data.append(agent_time_mask)
    #         input_names.append("agent_time_mask")

    #         laneline_pts = torch.Tensor(1, 50, 200, 2).cuda()
    #         input_data.append(laneline_pts)
    #         input_names.append("laneline_pts")

    #         laneline_attrs = torch.Tensor(1, 1, 50, 3)
    #         laneline_attrs[:, :, :, 0] = torch.randint(self.model.encoder.map_encoder.color_emb.weight.shape[0], size=(1, 1, 50))
    #         laneline_attrs[:, :, :, 1] = torch.randint(self.model.encoder.map_encoder.laneline_type_emb.weight.shape[0], size=(1, 1, 50))
    #         laneline_attrs[:, :, :, 2] = torch.randint(self.model.encoder.map_encoder.laneline_style_emb.weight.shape[0], size=(1, 1, 50))
    #         #laneline_attrs[:, :, :, 3] = torch.randint(self.model.encoder.map_encoder.other_type_emb.weight.shape[0], size=(1, 1, 50))
    #         laneline_attrs = laneline_attrs.cuda()

    #         laneline_mask = torch.zeros(1, 1, 1, 50, dtype=torch.float).cuda()
    #         input_data.append(laneline_mask)
    #         input_names.append("laneline_mask")

    #         ego_curr_status = torch.Tensor(1, 1, 1, 4).cuda()
    #         ego_curr_status[:, :, :, 2] = torch.randint(self.model.encoder.agent_encoder.traffice_light_embed.weight.shape[0], size=(1, 1, 1))
    #         ego_curr_status[:, :, :, 3] = torch.randint(self.model.encoder.agent_encoder.car_light_embed.weight.shape[0], size=(1, 1, 1))
    #         input_data.append(ego_curr_status)
    #         input_names.append("ego_curr_status")

    #         np.savetxt(os.path.join(self.save_path, 'ae_type_emb.txt'), self.model.encoder.agent_encoder.type_emb.weight.detach().cpu().numpy())
    #         category = torch.cat([
    #             torch.zeros((1, 1), dtype=torch.int).cuda(),
    #             agent_attrs[:, :, :, -1].squeeze(0).to(torch.int)], dim=1)   # torch.Size([bs, 101])
    #         category_feature = self.model.encoder.agent_encoder.type_emb(category)
    #         input_data.append(category_feature.unsqueeze(0))
    #         input_names.append("category_feature")

    #         np.savetxt(os.path.join(self.save_path, 'ae_traffic_light_emb.txt'), self.model.encoder.agent_encoder.traffice_light_embed.weight.detach().cpu().numpy())
    #         traffic_light = torch.cat([
    #             ego_curr_status[..., 2].squeeze(0).to(torch.int),
    #             torch.zeros((1, 50), dtype=torch.int).cuda()], dim=1)
    #         traffic_light_feature = self.model.encoder.agent_encoder.traffice_light_embed(traffic_light)
    #         input_data.append(traffic_light_feature.unsqueeze(0))
    #         input_names.append("traffic_light_feature")

    #         np.savetxt(os.path.join(self.save_path, 'ae_car_light_emb.txt'), self.model.encoder.agent_encoder.car_light_embed.weight.detach().cpu().numpy())
    #         car_light = torch.cat([
    #             ego_curr_status[..., 3].squeeze(0).to(torch.int),
    #             torch.zeros((1, 50), dtype=torch.int).cuda()], dim=1)
    #         car_light_feature = self.model.encoder.agent_encoder.car_light_embed(car_light)
    #         input_data.append(car_light_feature.unsqueeze(0))
    #         input_names.append("car_light_feature")

    #         np.savetxt(os.path.join(self.save_path, 'me_color_emb.txt'), self.model.encoder.map_encoder.color_emb.weight.detach().cpu().numpy())
    #         polygon_color_feature = self.model.encoder.map_encoder.color_emb(laneline_attrs[..., 0].squeeze(0).to(torch.int))
    #         input_data.append(polygon_color_feature.unsqueeze(0))
    #         input_names.append("polygon_color_feature")

    #         np.savetxt(os.path.join(self.save_path, 'me_laneline_style_emb.txt'), self.model.encoder.map_encoder.laneline_style_emb.weight.detach().cpu().numpy())
    #         polygon_laneline_style_feature = self.model.encoder.map_encoder.laneline_style_emb(laneline_attrs[..., 2].squeeze(0).to(torch.int))
    #         input_data.append(polygon_laneline_style_feature.unsqueeze(0))
    #         input_names.append("polygon_laneline_style_feature")

    #         np.savetxt(os.path.join(self.save_path, 'me_laneline_type_emb.txt'), self.model.encoder.map_encoder.laneline_type_emb.weight.detach().cpu().numpy())
    #         polygon_laneline_type_feature = self.model.encoder.map_encoder.laneline_type_emb(laneline_attrs[..., 1].squeeze(0).to(torch.int))
    #         input_data.append(polygon_laneline_type_feature.unsqueeze(0))
    #         input_names.append("polygon_laneline_type_feature")

            
    #         #np.savetxt(os.path.join(self.save_path, 'me_other_type_emb.txt'), self.model.encoder.map_encoder.other_type_emb.weight.detach().cpu().numpy())
    #         #polygon_other_type_feature = self.model.encoder.map_encoder.other_type_emb(laneline_attrs[..., 3].squeeze(0).to(torch.int))
    #         #input_data.append(polygon_other_type_feature.unsqueeze(0))
    #         #input_names.append("polygon_other_type_feature")

    #         # np.savetxt(os.path.join(self.save_path, 'ae_query_emb.txt'), self.model.agent_query.weight.detach().cpu().numpy())
    #         # agent_query = self.model.agent_query
    #         # agent_query = agent_query.weight[1:2].unsqueeze(0).repeat(1, 1+50, 1)
    #         # agent_attrs_index = input_names.index("agent_attrs")
    #         # agent_attrs_tensor = input_data[agent_attrs_index]  # [1, 1, max_agents, 3]
    #         # p_mask = agent_attrs_tensor[..., 2].squeeze(1) == 1
    #         # b_mask = agent_attrs_tensor[..., 2].squeeze(1) == 3
    #         # agent_query[:,1:, :][p_mask] = self.model.agent_query.weight[:1]
    #         # agent_query[:,1:, :][b_mask] = self.model.agent_query.weight[2:3] 
    #         # input_data.append(agent_query.unsqueeze(0))
    #         # input_names.append("agent_query")
    #         # occ_map=torch.zeros(1, 1, 120, 275, dtype = torch.float).cuda()  
    #         # input_data.append(occ_map)
    #         # input_names.append("occ_map")
    #         navitopo_pts = torch.Tensor(1, 5, 200, 2).cuda()
    #         input_data.append(navitopo_pts)
    #         input_names.append("navitopo_pts")

    #         navitopo_mask = torch.zeros(1, 1, 1, 5, dtype=torch.float).cuda()
    #         input_data.append(navitopo_mask)
    #         input_names.append("navitopo_mask")
            
    #         return input_data, input_names

    # wrapper = MMDiTDiffusionModelONNXWrapper(model)
    # get_input_info

    dynamic_axes = None
    if not args.fixed_axes:
        dynamic_axes = {
            # inputs
            "raw_agent_features": {0: "B", 1: "A", 2: "T"},
            # "raw_agent_pos": {0: "B", 1: "A"},
            "raw_static_map_features": {0: "B", 1: "L", 2: "N_l"},
            # "raw_map_pl_pos": {0: "B", 1: "L"},
            # "raw_dynamic_map_features": {0: "B", 1: "TL", 2: "T_tl"},
            "raw_ego_routing_features": {0: "B", 1: "R", 2: "N_r"},
            # "raw_ego_turn_light_features": {0: "B"},
            "raw_ego_traffic_light_features": {0: "B"},
            "raw_ego_navi_features": {0: "B", 1: "N_n", 2: "C_r"},
            "mask_info": {0: "B"},
            "input_noise": {0: "B", 1: "1", 2: "80", 3: "3"},
            "sample_steps": {0: "B"},
            # outputs
            "pred_trajs": {0: "B", 1: "A", 2: "T_pred"},
            # "pred_logits": {0: "B", 1: "M"},
        }

    # output_names = ["pred_trajs", "fix_distance_pred_paths"]
    output_names = ["trajectory", "trajectory_fixed"]
    # wrapper.generate_meta_info(input_names, output_names)
    # wrapper.generate_meta_info(input_names, output_names)

    # 导出ONNX模型
    # import pdb; pdb.set_trace()
    onnx_save_file = os.path.join(
    save_path, '{}.onnx'.format(wrapper.model_name))
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            onnx_save_file,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
    print(f"Exported ONNX to {args.onnx}")
    wrapper.generate_meta_info(input_names, output_names)

    # 如果启用了处理所有frame，则调用process_all_frames函数
    if args.process_all_frames:
        device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
        model.to(device)
        
        input_names = [
            "raw_agent_features",
            "raw_static_map_features",
            "raw_ego_routing_features",
            "raw_ego_traffic_light_features",
            "mask_info",
            "input_noise",
            "sample_steps"
        ]
        
        # 处理所有frame
        all_frames_results = process_all_frames(args, model, input_names, device)
        
        if all_frames_results:
            print("=== 所有frame处理完成 ===")
            print(f"成功处理了 {len(all_frames_results)} 个frame")
        else:
            print("=== 处理所有frame时出现错误 ===")

        if args.save_io:
            print(f"=== 保存所有帧的输入输出数据到 {args.io_save_dir} ===")
            json_path = save_all_frames_combined_data(all_frames_results, args.io_save_dir)
            print(f"=== 所有帧数据保存成功：{json_path} ===")
        
        return  # 处理完所有frame后直接返回，不执行后续的单个frame处理
    else:
        # # 运行PyTorch推理并保存输入输出（如果需要）
        # if args.save_io or args.check_alignment:
        #     os.makedirs(args.io_save_dir, exist_ok=True)
        #     print("=== Running PyTorch inference for alignment check ===")
        #     pytorch_results = run_pytorch_inference(wrapper, inputs, input_names)
            
        # 运行PyTorch推理并保存输入输出（如果需要）
        need_pytorch_outputs = args.save_io or args.check_alignment or args.visualize_results
        pytorch_results = None
        onnx_results = None
        if need_pytorch_outputs:
            os.makedirs(args.io_save_dir, exist_ok=True)
            print("=== Running PyTorch inference for alignment check ===")
            pytorch_results = run_pytorch_inference(wrapper, inputs, input_names)
            
            if args.save_io:
                print(f"=== Saving PyTorch inputs/outputs as bin files to {args.io_save_dir} ===")
                saved_paths = save_io_as_bins(pytorch_results, args.io_save_dir, args.frame_idx)
                total_inputs = len(saved_paths["inputs"])
                total_outputs = len(saved_paths["outputs"])
                print(f"=== 成功保存 {total_inputs} 个输入特征与 {total_outputs} 个输出特征 ===")

        # 运行对齐检查（如果需要）
        need_onnx_outputs = args.check_alignment or args.visualize_results
        if need_onnx_outputs and pytorch_results is not None:
            print("=== Running ONNX inference ===")
            onnx_results = run_onnx_inference(args.onnx, pytorch_results["inputs"])
            
            if args.check_alignment:
                print("=== Comparing PyTorch and ONNX results ===")
                comparison_results = compare_results(
                    pytorch_results, 
                    onnx_results, 
                    tolerance=args.tolerance
                )
                
                # 保存比较结果
                comparison_path = os.path.join(args.io_save_dir, "alignment_comparison.json")
                # import pdb; pdb.set_trace()
                with open(comparison_path, 'w') as f:
                    json.dump(comparison_results, f, indent=2, default=str)
                
                # 打印比较结果
                print("\n=== Alignment Check Results ===")
                all_passed = True
                for output_name, result in comparison_results.items():
                    status = result["status"]
                    max_diff = result["max_diff"]
                    print(f"{output_name}: {status} (max_diff: {max_diff:.2e})")
                    if status != "PASS":
                        all_passed = False
                
                if all_passed:
                    print("🎉 All outputs passed alignment check!")
                else:
                    print("⚠️  Some outputs failed alignment check!")
                
                print(f"Detailed results saved to: {comparison_path}")

            if args.visualize_results and viz_context is not None and onnx_results is not None:
                visualize_pytorch_vs_onnx(
                    args.frame_idx,
                    viz_context,
                    pytorch_results,
                    onnx_results,
                    args.viz_save_dir,
                    max_agents=args.viz_max_agents,
                )
            elif args.visualize_results and onnx_results is None:
                print("⚠️  可视化需要ONNX输出，但推理结果为空。")


if __name__ == "__main__":
    main()
