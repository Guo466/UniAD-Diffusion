# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List, Dict
import psutil

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch.utils.data._utils.collate import default_collate



def add_missing_hyperparametrs(param):
    """
    The model hyperparameters may vary across different versions. 
    Therefore, in this function, some missing hyperparameters are filled with default values. 
    Note that these default hyperparameters should not affect the inference results of the model.
    """
    # if 'n_date_file' not in param['dataset']:
    #     print("add n_date_file=empty.json in param['dataset']")
    #     param['dataset']['n_date_file'] = 'empty.json'    
    
    if 'laneline_shift_dist' not in param['dataset']['cost_map']:
        print("add laneline_shift_dist=0.1 in param['dataset']['cost_map']")
        param['dataset']['cost_map']['laneline_shift_dist'] = 0.1
    
    if 'use_new_laneline' not in param['dataset']:
        print("add use_new_laneline=False in param['dataset']")
        param['dataset']['use_new_laneline'] = False

    if 'route_cost_thre' not in param['dataset']:
        print("add route_cost_thre=99999 in param['dataset']")
        print("add use_route=False in param['dataset']")
        param['dataset']['route_cost_thre'] = 99999
        param['dataset']['use_route'] = False
        
    if 'from_ceph_test' not in param['dataset']:
        print("add from_ceph_test in param['dataset'], which equals from_ceph")
        param['dataset']['from_ceph_test'] = param['dataset']['from_ceph']
        
    if "cost_map_resolution" not in param['model']:
        print("add cost_map_resolution=0.4 in param['model']")
        param['model']['cost_map_resolution'] = 0.4
        
    if "offset" not in param['model']:
        print("add offset=1.7 in param['model']")
        param['model']['offset'] = 1.6
        
    if "rc" not in param['model']:
        print("add rc=1.3 in param['model']")
        param['model']['rc'] = 1.3
        
    if 'ceph_conf_file' not in param:
        print("add ceph_conf_file=aoss.conf in param")
        param['ceph_conf_file'] = 'aoss.conf'

    if "offset_scale_with_v" not in param['model']:
        print("add offset_scale_with_v=False in param['model']")
        param['model']['offset_scale_with_v'] = False
    
    if "late_pt_lower_weight" not in param['model']:
        print("add late_pt_lower_weight=False in param['model']")
        param['model']['late_pt_lower_weight'] = False

    if "contrastive_learning_funcs" not in param:
        print("add contrastive_learning_funcs in param")
        param['contrastive_learning_funcs'] = {
            "positive_func":{
                "pos_perturbation":False,
                "yaw_perturbation":False,
                "y_flip":False,
                "backward_agents_dropout": False,
                "non_interactive_agents_dropout": False,
            },
            "negative_func":{
                "leading_agents_dropout": False,
                "leading_agents_deceleration": False,
                "leading_agent_insertion": False,
                "static_obstacle_insertion": False,
                "vru_insertion": False,
                "interactive_agents_dropout": False,
                "traffic_light_inversion": False,
                "ego_light_switch": False
            }
        }
    param_cl = param['contrastive_learning_funcs'] 
    if param_cl is None:
        param['model']['use_contrastive_learning'] = False
    else:
        param['model']['use_contrastive_learning'] = True in [v for v in param_cl['positive_func'].values()] and True in [v for v in param_cl['negative_func'].values()]
    if param['model']['use_contrastive_learning']:
        print('use contrastive learning')
    else:
        print('not use contrastive learning')
    
    if 'cl_temperature' not in param['model']:
        print("add cl_temperature=0.1 in param['model']")
        param['model']['cl_temperature'] = 0.1
        
    if 'cat_enc_out' not in param['model']:
        print("add cat_enc_out=False in param['model']")
        param['model']['cat_enc_out'] = False

    if 'return_intermediate_dec' not in param['model']:
        print("add return_intermediate_dec=False in param['model']")
        param['model']['return_intermediate_dec'] = False

    if 'curvature_related_lateral_loss_factor' not in param['model']:
        print("add curvature_related_lateral_loss_factor=False in param['model']")
        param['model']['curvature_related_lateral_loss_factor'] = False
    
    if 'velo_weight_decay_factor' not in param['model']:
        print("add velo_weight_decay_factor=-0.5 in param['model']")
        param['model']['velo_weight_decay_factor'] = -0.5   

    if 'max_velo_weight' not in param['model']:
        print("add max_velo_weight=5 in param['model']")
        param['model']['max_velo_weight'] = 5

    if 'predict_agents_importance' not in param['model']:
        print("add predict_agents_importance=False in param['model']")
        param['model']['predict_agents_importance'] = False
    
    if 'predict_laneline_importance' not in param['model']:
        print("add predict_laneline_importance=False in param['model']")
        param['model']['predict_laneline_importance'] = False
    
    if 'important_agent_prediction_weight' not in param['model']:
        print("add important_agent_prediction_weight=0.2 in param['model']")
        param['model']['important_agent_prediction_weight'] = 0.2
    
    if "reconstruct_laneline" not in param['model']:
        print("add reconstruct_laneline=True in param['model']")
        param['model']['reconstruct_laneline'] = True

    if "cls_loss_use_softmax" not in param['model']:
        print("add cls_loss_use_softmax=True in param['model']")
        param['model']['cls_loss_use_softmax'] = False
    
    if 'planning_interval' not in param['dataset']:
        print("add planning_interval=0.1 in param['dataset']")
        param['dataset']['planning_interval'] = 0.1
    
    if "ego_care_normal_laneline"  not in param['dataset']:
        print("add ego_care_normal_laneline=True in param['dataset']")
        param['dataset']['ego_care_normal_laneline'] = True

    return param




if float(torchvision.__version__[:3]) < 0.5:
    import math
    # from torchvision.ops.misc import _NewEmptyTensorOp
    def _check_size_scale_factor(dim, size, scale_factor):
        # type: (int, Optional[List[int]], Optional[float]) -> None
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if not (scale_factor is not None and len(scale_factor) != dim):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )
    def _output_size(dim, input, size, scale_factor):
        # type: (int, Tensor, Optional[List[int]], Optional[float]) -> List[int]
        assert dim == 2
        _check_size_scale_factor(dim, size, scale_factor)
        if size is not None:
            return size
        # if dim is not 2 or scale_factor is iterable use _ntuple instead of concat
        assert scale_factor is not None and isinstance(scale_factor, (int, float))
        scale_factors = [scale_factor, scale_factor]
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]
elif float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0], size_divisibility=32)
    return tuple(batch)


def mot_collate_fn(batch: List[dict]) -> dict:
    # 将从数据集中获取的不同大小的数据（batch）按照相同的尺寸组合成一个batch
    # len(batch) == batch_size
    # len(batch[0]['sensors']) == frames_per_batch
    res = {}
    for key in list(batch[0].keys()):        
        res[key] = []
        if key == 'sensors':
            for i in range(len(batch[0][key])):
                temp = [batch[j][key][i] for j in range(len(batch))]
                # res[key].append(nested_tensor_from_tensor_list(temp).to('cuda'))
                res[key].append(nested_tensor_from_tensor_list(temp))


        elif key == 'maps':
            device = 'cuda'
            res[key] = []
            for i in range(len(batch[0][key])):  # i: frame-idx
                temp_dic = {}
                for s_key in list(batch[0][key][i].keys()):
                    # print('s_key:', s_key)
                    if s_key == 'boundary_point':  # 每个batch中 maps['boundary_point']都是一个List[Tensor]，直接合并List即可
                        temp_dic[s_key] = []
                        for j in range(len(batch)):
                            temp_dic[s_key] += batch[j][key][i][s_key]
                    
                    elif s_key == ('map_point', 'to', 'map_polygon'):
                        temp_dic[s_key] = {}
                        ss_key = 'edge_index'
                        # print('ss_key:', ss_key)
                        base_pt_idx_list = [0] + [batch[j][key][i]['map_point']['num_nodes'] for j in range(len(batch))]
                        # [0, 186, 302, 250, 78, 37, 260, 101, 159]
                        base_pt_idx_list = [sum(base_pt_idx_list[:i+1]) for i in range(len(base_pt_idx_list))]
                        # [0, 186, 488, 738, 816, 853, 1113, 1214, 1373]
                        base_pt_idx_tensor = torch.cat([torch.full_like(batch[j][key][i][s_key][ss_key][0:1], base_pt_idx_list[j], device=device) for j in range(len(batch))], dim=1)
                        
                        base_pl_idx_list = [0] + [batch[j][key][i]['map_polygon']['num_nodes'] for j in range(len(batch))]
                        # [0, 4, 4, 4, 3, 3, 3, 3, 4]
                        base_pl_idx_list = [sum(base_pl_idx_list[:i+1]) for i in range(len(base_pl_idx_list))]
                        # [0, 4, 8, 12, 15, 18, 21, 24, 28]
                        base_pl_idx_tensor = torch.cat([torch.full_like(batch[j][key][i][s_key][ss_key][1:2], base_pl_idx_list[j], device=device) for j in range(len(batch))], dim=1)
                        
                        base_edge_idx_tensor = torch.cat([base_pt_idx_tensor, base_pl_idx_tensor], dim=0)
                        
                        temp_dic[s_key][ss_key] = torch.cat([batch[j][key][i][s_key][ss_key] for j in range(len(batch))], dim=1).to(device) + base_edge_idx_tensor
                    
                    elif s_key == ('map_polygon', 'to', 'map_polygon'):
                        temp_dic[s_key] = {}
                        for ss_key in list(batch[0][key][i][s_key].keys()):
                            # print('ss_key:', ss_key)
                            if ss_key == 'edge_index':
                                base_pl_idx_list = [0] + [batch[j][key][i]['map_polygon']['num_nodes'] for j in range(len(batch))]
                                # [0, 4, 4, 4, 3, 3, 3, 3, 4]
                                base_pl_idx_list = [sum(base_pl_idx_list[:i+1]) for i in range(len(base_pl_idx_list))]
                                # [0, 4, 8, 12, 15, 18, 21, 24, 28]
                                base_edge_idx_tensor = torch.cat([torch.full_like(batch[j][key][i][s_key][ss_key], base_pl_idx_list[j], device=device) for j in range(len(batch))], dim=1)
                                temp_dic[s_key][ss_key] = torch.cat([batch[j][key][i][s_key][ss_key] for j in range(len(batch))], dim=1).to(device) + base_edge_idx_tensor
                            elif ss_key == 'type':
                                tensor_list = [batch[j][key][i][s_key][ss_key] for j in range(len(batch))]  # j: batch-idx
                                tensor_cat, _ = cat_tensor_from_tensor_list(tensor_list)
                                temp_dic[s_key][ss_key] = tensor_cat
                            else:
                                raise ValueError('{}.{} is not a valid key'.format(s_key, ss_key))
                    else:
                        temp_dic[s_key] = {}

                        for ss_key in list(batch[0][key][i][s_key].keys()):
                            # print('ss_key:', ss_key)
                            if ss_key == 'num_nodes':
                                try:
                                    num_nodes_all_batch = sum([batch[j][key][i][s_key][ss_key] for j in range(len(batch))])
                                    temp_dic[s_key][ss_key] = num_nodes_all_batch
                                except TypeError:
                                    print('get TypeError', batch[0][key])
                                    raise TypeError
                                    
                            else:
                                tensor_list = [batch[j][key][i][s_key][ss_key] for j in range(len(batch))]  # j: batch-idx
                                tensor_cat, batch_idx = cat_tensor_from_tensor_list(tensor_list)
                                temp_dic[s_key][ss_key] = tensor_cat
                                if not 'batch' in temp_dic[s_key]:
                                    temp_dic[s_key]['batch'] = batch_idx
                res[key].append(temp_dic)

        elif key == 'gt_instances':
            for i in range(len(batch[0][key])):
                res[key].append([])
                for j in range(len(batch)):
                    res[key][i].append(batch[j][key][i])
                    # res['gt_instances'][i][j]表示：第i帧，第j个batch的gt instance

        elif key in ['ego_hists', 'future_trajs', 'control_gts'] :
            device = 'cuda'
            for i in range(len(batch[0][key])):
                temp_dic = {}
                for s_key in list(batch[0][key][i].keys()):
                    temp_dic[s_key] = torch.cat([batch[j][key][i][s_key].unsqueeze(0) for j in range(len(batch))], dim=0).to(device)
                res[key].append(temp_dic)
                
        else:
            raise ValueError('not collate_fn for key: {}'.format(key))
    # s_key = ('map_polygon', 'to', 'map_polygon')
    # ss_key = 'edge_index'
    # for i in range(len(batch)):
    #     print('batch %d maps:' % i, batch[i]['maps'][0][s_key][ss_key])
    # print('batched maps:', res['maps'][0][s_key][ss_key])
    # if 'batch' in res['maps'][0][s_key]:
    #     print('batch-idx:', res['maps'][0][s_key]['batch'].view(-1))
    
    return res

def plantf_collate_fn(batch: List[dict]) -> dict:
    # len(batch) == batch_size
    # len(batch[0]['sensors']) == frames_per_batch
    # res = {}
    # for key in batch[0].keys():   # raw, pos, neg     
    #     if key not in ["raw","pos","neg"]:
    #         res[key] = []
    #         for i in range(len(batch)):
    #             res[key].append(batch[i][key])
    #         continue
    #     res[key] = {}
    #     for k in batch[0][key].keys():
    #         res[key][k] = []
    #         for i in range(len(batch)):
    #             res[key][k].append(batch[i][key][k])
    #         tmp = torch.stack(res[key][k], dim=0)
    #         res[key][k] = tmp
    #         # res[key][k] = tmp.cuda()
    # return res

    # 合并各样本的键：避免某条为 backup / 异常路径时缺少 'date' 等与 batch[0] 不一致导致 KeyError
    keys = set()
    for b in batch:
        keys.update(b.keys())
    res = {}
    for key in sorted(keys):
        if key in ['model_input', 'pos', 'neg']:
            res[key] = default_collate([b[key] for b in batch])
        else:
            res[key] = [b.get(key) for b in batch]
    return res

def plantf_collate_fn_eval(batch: List[dict]) -> dict:
    # len(batch) == batch_size
    # len(batch[0]['sensors']) == frames_per_batch
    if 'data_label_path' in batch[0]:
        data_label_path = batch[0]['data_label_path']
        batch[0].pop('data_label_path', None)
    else:
        data_label_path = None
    res = {}
    for key in batch[0].keys():   # raw, pos, neg     
        if key not in ["model_input","pos","neg"]:
            res[key] = []
            for i in range(len(batch)):
                res[key].append(batch[i][key])
            continue
        res[key] = {}
        for k in batch[0][key].keys():
            res[key][k] = []
            for i in range(len(batch)):
                res[key][k].append(batch[i][key][k])
            tmp = torch.stack(res[key][k], dim=0)
            res[key][k] = tmp
            # res[key][k] = tmp.cuda()
    return res, data_label_path


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def cat_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    example:
    input:
        tensor_list = [
            tensor.Size(1, dim),
            tensor.Size(2, dim),
            tensor.Size(3, dim),
            tensor.Size(4, dim)
        ]
    output:
        tensor_cat = tensor.Size(1+2+3+4, dim)
        batch_idx = tensor([0, 1,1, 2,2,2, 3,3,3,3])
    """
    device = 'cuda'
    try:
        tensor_cat = torch.cat(tensor_list, dim=0).to(device)
        if tensor_list[0].dim() == 2:
            batch_idx = torch.cat([torch.full_like(t[:, 0], i, dtype=torch.long, device=device) for i, t in enumerate(tensor_list)], dim=0)
        elif tensor_list[0].dim() == 1:
            batch_idx = torch.cat([torch.full_like(t, i, dtype=torch.long, device=device) for i, t in enumerate(tensor_list)], dim=0)
        else:
            raise ValueError('{} is not a valid dim'.format(tensor_list[0].dim()))
    except IndexError:
        print([t.shape for t in tensor_list])
        raise IndexError
    
    return tensor_cat, batch_idx

def nested_tensor_from_tensor_list(tensor_list: List[Tensor], size_divisibility: int = 0):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        if size_divisibility > 0:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride
            max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride

        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    elif tensor_list[0].ndim == 2:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, n_sensor, feature = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        device = 'cuda'
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # print(tensor.shape)
        mask = torch.ones((b, n_sensor), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # print('pad_img.shape:',pad_img.shape)
            # print('img.shape:',img.shape)
            # print(pad_img[: img.shape[0], : img.shape[1]].shape, img.shape)
            pad_img[: img.shape[0], : img.shape[1]].copy_(img)
            # True表示图像padding区域；False表示实际图像区域
            m[: img.shape[0]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    '''
    for a nested tensor, not all dimensions have regular sizes, 
    e.g. in CV, images can have variable shapes, so a batch of images forms a nested tensor
    '''
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        # LOCAL_RANK 可能未设置（如仅用 python 跑单卡时），用 RANK 作为回退
        args.gpu = int(os.environ.get('LOCAL_RANK', os.environ['RANK']))
        args.local_rank = args.gpu
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    #setup_for_distributed(True)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        # if float(torchvision.__version__[:3]) < 0.5:
        #     return _NewEmptyTensorOp.apply(input, output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        
    @torch.amp.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x, flag=False):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # if flag:
            #     print(i, x)
        return x


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.75, gamma: float = 3, mean_in_dim1=True):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid() # shape (1, n_query * n_cls)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # ​*​​表示两个矩阵对应位置处的两个元素相乘
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)   # shape: (1, n_query * n_cls)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if mean_in_dim1:
        return loss.mean(1).sum() / num_boxes
    else:
        # loss.shape (1, 3600)
        return loss.sum() / num_boxes

def huber_loss(error, delta=1.0):
    """
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss


def custom_to_cuda(data):
    # for key, val in data.items():
    #     for k in val.keys():
    #         data[key][k] = data[key][k].cuda(non_blocking=True)
    # return data
    return {
        key: {
            k: v.cuda(non_blocking=True) for k, v in val.items()
        } for key, val in data.items()
    }

def bilinear_interpolation_batch(values, coords):
    # 确保 values 的形状为 (bs, 1, height, length)
    values = values.unsqueeze(1)  # 形状变为 (bs, 1, height, length)

    # 将 coords 从 [0, 1] 范围转换到 [-1, 1] 范围
    coords = 2 * coords - 1
    coords = coords.unsqueeze(1).unsqueeze(1)  # 形状变为 (bs, 1, 1, 2)

    # 使用 grid_sample 进行双线性插值
    output = F.grid_sample(values, coords, mode='bilinear', align_corners=True)

    # 获取结果
    result = output.squeeze()
    return result


def log_memory_usage():
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / (1024 ** 3):.2f} GB")
    print(f"Available: {mem.available / (1024 ** 3):.2f} GB")
    print(f"Used: {mem.used / (1024 ** 3):.2f} GB")
    print(f"Percentage: {mem.percent}%")

