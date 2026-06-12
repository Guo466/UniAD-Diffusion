#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# ============================================================
# 本文件是 UniAD 自动驾驶系统的核心数据集类
# 负责：
#   1. 加载 NuScenes 数据集（图像、点云、标注）
#   2. 构建时序帧队列（用于BEV感知的时序融合）
#   3. 提供目标检测/跟踪/运动预测/规划的标注信息
#   4. 提供占用预测(Occ)所需的时序帧信息
#   5. 评估检测/跟踪/运动/规划等多个任务指标
# ============================================================

import copy
import numpy as np
import torch
import mmcv
from mmdet.datasets import DATASETS                              # mmdet数据集注册器，用于@DATASETS.register_module()
from mmdet.datasets.pipelines import to_tensor                   # 将numpy数组转为Tensor的工具函数
from mmdet3d.datasets import NuScenesDataset                     # mmdet3d提供的NuScenes基础数据集类，本类继承自它
from mmdet3d.core.bbox import LiDARInstance3DBoxes               # LiDAR坐标系下的3D目标框类

from os import path as osp                                       # 路径操作工具
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion  # 四元数工具：quaternion_yaw提取偏航角，Quaternion表示旋转
from .eval_utils.nuscenes_eval import NuScenesEval_custom        # 自定义的NuScenes检测评估器
from nuscenes.eval.tracking.evaluate import TrackingEval         # NuScenes官方跟踪评估器
from .eval_utils.nuscenes_eval_motion import MotionEval          # 自定义的运动预测评估器
from nuscenes.eval.common.config import config_factory           # 评估配置工厂函数
import tempfile                                                  # 用于创建临时目录
from mmcv.parallel import DataContainer as DC                    # mmcv的数据容器，用于多GPU数据传输
import random                                                    # 随机数模块（用于时序增强时打乱顺序）
import pickle                                                    # 用于反序列化pkl文件（加载标注数据）
from prettytable import PrettyTable                              # 用于打印格式化表格（显示规划评估结果）

from nuscenes import NuScenes                                    # NuScenes数据集API（用于查询场景/样本信息）
from projects.mmdet3d_plugin.datasets.data_utils.vector_map import VectorizedLocalMap   # 向量化局部地图工具
from projects.mmdet3d_plugin.datasets.data_utils.rasterize import preprocess_map        # 将向量地图栅格化为mask
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap             # NuScenes地图API
from projects.mmdet3d_plugin.datasets.data_utils.trajectory_api import NuScenesTraj     # NuScenes轨迹API（获取gt轨迹）
from .data_utils.data_utils import lidar_nusc_box_to_global, obtain_map_info, output_to_nusc_box, output_to_nusc_box_det
# lidar_nusc_box_to_global: 将LiDAR坐标系下的框转换到全局坐标系
# obtain_map_info: 获取地图信息（车道线等）
# output_to_nusc_box: 将模型输出转换为NuScenes格式的Box（含轨迹）
# output_to_nusc_box_det: 将模型输出转换为NuScenes格式的Box（纯检测）
from nuscenes.prediction import convert_local_coords_to_global   # 将局部坐标转换为全局坐标（用于轨迹坐标变换）


@DATASETS.register_module()  # 将此类注册到mmdet的数据集注册器中，config文件中可用类名字符串来引用
class NuScenesE2EDataset(NuScenesDataset):
    r"""NuScenes E2E Dataset.
    端到端自动驾驶数据集，继承自mmdet3d的NuScenesDataset。
    在父类基础上扩展了：
    - 时序帧队列（queue_length帧同时输入，支持BEV时序融合）
    - 地图信息（向量化地图 + 栅格化地图mask）
    - 轨迹标注（历史轨迹/未来轨迹/SDC轨迹/规划标签）
    - 占用预测所需的时序帧信息
    """

    def __init__(self,
                 queue_length=4,              # 时序队列长度：一次输入多少帧（当前帧+之前帧），用于BEV时序特征融合
                 bev_size=(200, 200),         # BEV（鸟瞰图）特征图分辨率，单位：格子数
                 patch_size=(102.4, 102.4),   # 地图patch的物理大小（单位：米），即以ego车为中心的感知范围
                 canvas_size=(200, 200),      # 地图栅格化后的画布像素大小
                 overlap_test=False,          # 测试时是否使用重叠检测（NMS相关）
                 predict_steps=12,            # 运动预测步数（预测未来12帧 = 6秒，0.5s/帧）
                 planning_steps=6,            # 规划步数（规划未来6帧 = 3秒）
                 past_steps=4,               # 历史轨迹步数（向前看4帧历史）
                 fut_steps=4,                # 未来轨迹步数（此处与predict_steps有区别，用于部分子模块）
                 use_nonlinear_optimizer=False,  # 是否使用非线性优化器来平滑SDC规划轨迹
                 lane_ann_file=None,          # 车道标注文件路径（语义格式的地图标注，可选）
                 eval_mod=None,              # 评估模式列表，例如['det', 'track', 'map', 'motion']

                 # For debug
                 is_debug=False,             # 是否开启调试模式（只加载少量样本）
                 len_debug=30,              # 调试模式下的样本数量

                 # Occ dataset（占用预测相关参数）
                 enbale_temporal_aug=False,   # 是否开启时序增强（当前版本强制False，未实现）
                 occ_receptive_field=3,       # 占用预测的感受野大小（过去帧数+当前帧 = 3帧）
                 occ_n_future=4,             # 占用预测需要的未来帧数（预测未来4帧）
                 occ_filter_invalid_sample=False,   # 是否过滤含无效帧的样本
                 occ_filter_by_valid_flag=False,    # 是否按有效标志过滤目标框

                 file_client_args=dict(backend='disk'),  # 文件读取方式，'disk'表示本地磁盘，'petrel'表示对象存储
                 *args,                      # 传给父类NuScenesDataset的位置参数
                 **kwargs):                  # 传给父类NuScenesDataset的关键字参数（如ann_file, data_root等）

        # ---- 注意：file_client 和 is_debug 必须在super().__init__()之前初始化 ----
        # 因为父类__init__会调用load_annotations()，而load_annotations()需要用到file_client
        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient(**file_client_args)  # 创建文件客户端（支持disk/petrel等后端）

        self.is_debug = is_debug
        self.len_debug = len_debug
        super().__init__(*args, **kwargs)    # 调用父类NuScenesDataset的初始化，完成基础属性初始化（data_root, classes等）

        # ---- 时序相关参数 ----
        self.queue_length = queue_length     # 时序队列长度
        self.overlap_test = overlap_test     # 重叠测试标志

        # ---- BEV地图相关参数 ----
        self.bev_size = bev_size             # BEV特征图大小

        # ---- 轨迹预测相关参数 ----
        self.predict_steps = predict_steps   # 运动预测步数
        self.planning_steps = planning_steps # 规划步数
        self.past_steps = past_steps         # 历史轨迹步数
        self.fut_steps = fut_steps           # 未来轨迹步数
        self.scene_token = None              # 当前场景token（用于判断帧是否属于同一场景）

        # ---- 加载额外的车道标注（语义格式，可选）----
        self.lane_infos = self.load_annotations(lane_ann_file) \
            if lane_ann_file else None       # 如果提供了lane_ann_file则加载，否则为None

        self.eval_mod = eval_mod             # 评估模式

        self.use_nonlinear_optimizer = use_nonlinear_optimizer

        # ---- 初始化NuScenes API ----
        # NuScenes API用于查询sample/annotation/场景/log等信息
        self.nusc = NuScenes(version=self.version,
                             dataroot=self.data_root, verbose=True)

        # ---- 地图配置 ----
        self.map_num_classes = 3             # 地图语义类别数：分隔线/人行道/道路边缘等（共3类）
        # 根据画布大小决定绘制地图时线条的粗细（像素宽度）
        if canvas_size[0] == 50:
            self.thickness = 1              # 小分辨率用细线
        elif canvas_size[0] == 200:
            self.thickness = 2              # 大分辨率用粗线
        else:
            assert False                    # 不支持其他分辨率
        self.angle_class = 36               # 方向角的离散化类别数（360度/36 = 每10度一类）
        self.patch_size = patch_size        # 地图物理范围（米）
        self.canvas_size = canvas_size      # 地图栅格化画布大小

        # 加载NuScenes 4个城市的地图API（NuScenes数据集采集于4个城市）
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot=self.data_root, map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot=self.data_root, map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot=self.data_root, map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot=self.data_root, map_name='singapore-queenstown'),
        }

        # 向量化地图：将NuScenes地图API中的线段/多边形转为向量格式（用于MapTR等地图预测任务）
        self.vector_map = VectorizedLocalMap(
            self.data_root,
            patch_size=self.patch_size,
            canvas_size=self.canvas_size)

        # 轨迹API：负责获取所有目标和ego车辆的历史/未来轨迹标签
        self.traj_api = NuScenesTraj(self.nusc,
                                     self.predict_steps,    # 运动预测步数
                                     self.planning_steps,   # 规划步数
                                     self.past_steps,       # 历史步数
                                     self.fut_steps,        # 未来步数
                                     self.with_velocity,    # 是否带速度（父类属性）
                                     self.CLASSES,          # 目标类别列表
                                     self.box_mode_3d,      # 3D框模式（LiDAR/Depth/Camera）
                                     self.use_nonlinear_optimizer)

        # ---- 占用预测(Occ)相关参数 ----
        self.enbale_temporal_aug = enbale_temporal_aug
        assert self.enbale_temporal_aug is False  # 当前版本不支持时序增强，强制关闭

        self.occ_receptive_field = occ_receptive_field  # 感受野（过去帧数+当前帧）
        self.occ_n_future = occ_n_future                # 需要预测的未来帧数
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        # 占用预测评估时固定使用7帧（hardcode），不受planning_steps影响
        self.occ_only_total_frames = 7  # NOTE: hardcode, not influenced by planning

    def __len__(self):
        """返回数据集的样本总数。
        调试模式下只返回少量样本，加快调试速度。
        """
        if not self.is_debug:
            return len(self.data_infos)  # 正常模式：返回全部样本数
        else:
            return self.len_debug        # 调试模式：只返回len_debug个样本

    def load_annotations(self, ann_file):
        """从标注文件中加载数据信息。
        
        NuScenes数据集的标注通常预处理为pkl格式（包含infos列表和metadata）。
        本函数读取pkl文件并按时间戳排序。

        Args:
            ann_file (str): 标注文件路径（.pkl格式）

        Returns:
            list[dict]: 按时间戳排序后的数据信息列表，每个元素包含：
                token, lidar_path, cams, gt_boxes, gt_names, gt_velocity等
        """
        if self.file_client_args['backend'] == 'disk':
            # 从本地磁盘读取pkl文件并反序列化
            # ann_file.name 是因为传入的可能是Path对象
            data = pickle.loads(self.file_client.get(ann_file.name))
            # 按时间戳排序（确保时序正确）
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            # load_interval是父类属性，表示加载间隔（=1则全部加载，=2则每隔一帧加载）
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']       # 元数据（包含version等信息）
            self.version = self.metadata['version'] # 数据集版本（v1.0-mini / v1.0-trainval）
        elif self.file_client_args['backend'] == 'petrel':
            # 从对象存储（Petrel/Ceph）读取pkl文件
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        else:
            assert False, 'Invalid file_client_args!'  # 不支持其他后端
        return data_infos

    def prepare_train_data(self, index):
        """
        准备训练数据，构建时序帧队列。
        
        核心逻辑：
        - 以当前帧（index）为最后一帧，向前取 queue_length-1 帧，构成时序队列
        - 所有帧必须来自同一场景（scene_token相同）
        - 将队列中所有帧通过 union2one 合并为一个样本返回
        
        Args:
            index (int): 当前帧的索引（训练时的target帧）
        Returns:
            dict | None: 训练数据字典，若无效则返回None
                img:          (queue_length, 6, 3, H, W) 时序帧图像
                img_metas:    各帧的元数据
                gt_bboxes_3d: 各帧的3D目标框
                gt_labels_3d: 各帧的目标类别标签
                gt_inds:      各帧的目标实例ID
                ...（更多字段见union2one函数）
        """
        data_queue = []                          # 存放多帧数据的列表
        self.enbale_temporal_aug = False         # 强制关闭时序增强

        if self.enbale_temporal_aug:
            # ---- 时序增强模式（当前未使用） ----
            # 随机打乱历史帧顺序来做数据增强
            prev_indexs_list = list(range(index-self.queue_length, index))
            random.shuffle(prev_indexs_list)
            prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
            input_dict = self.get_data_info(index)
        else:
            # ---- 正常时序模式 ----
            final_index = index                           # 当前帧（最新帧）
            first_index = index - self.queue_length + 1  # 队列中最早帧的索引

            # 边界检查：最早帧索引不能小于0
            if first_index < 0:
                return None

            # 场景一致性检查：队列中第一帧和最后一帧必须在同一场景中
            # 如果不在同一场景（即跨场景），则丢弃该样本
            if self.data_infos[first_index]['scene_token'] != \
                    self.data_infos[final_index]['scene_token']:
                return None

            # 获取当前帧（最新帧）的数据信息
            input_dict = self.get_data_info(final_index)
            # 历史帧列表：从倒数第二帧到第一帧（逆序，最近的帧先处理）
            prev_indexs_list = list(reversed(range(first_index, final_index)))

        if input_dict is None:
            return None

        # 记录当前帧的帧号和场景token（用于后续验证历史帧合法性）
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']

        # 执行数据预处理流水线（pre_pipeline负责初始化一些字段）
        self.pre_pipeline(input_dict)
        # 执行完整的数据增强和格式化流水线（归一化、裁剪、转tensor等）
        example = self.pipeline(input_dict)

        # 一致性断言：目标数量要与未来轨迹/历史轨迹的数量一致
        assert example['gt_labels_3d'].data.shape[0] == example['gt_fut_traj'].shape[0]
        assert example['gt_labels_3d'].data.shape[0] == example['gt_past_traj'].shape[0]

        # 如果设置了过滤空标注，且当前帧没有有效目标，则丢弃该样本
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None

        # 将当前帧插入队列头部（因为后续还要插入历史帧到更前面）
        data_queue.insert(0, example)

        # ---- 处理历史帧 ----
        for i in prev_indexs_list:
            if self.enbale_temporal_aug:
                i = max(0, i)              # 时序增强时，确保索引不越界

            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None

            # 确保历史帧的帧号严格小于当前帧（保证时序正确），且在同一场景
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']  # 更新"上一帧"的帧号

            # 一致性断言
            assert example['gt_labels_3d'].data.shape[0] == example['gt_fut_traj'].shape[0]
            assert example['gt_labels_3d'].data.shape[0] == example['gt_past_traj'].shape[0]

            # 深拷贝后插入队列头部（保证每帧数据独立）
            data_queue.insert(0, copy.deepcopy(example))

        # 将队列中的多帧数据合并为一个样本字典
        data_queue = self.union2one(data_queue)
        return data_queue

    def prepare_test_data(self, index):
        """
        准备测试数据。
        
        注意：测试时只处理单帧（不构建时序队列），
        时序信息由模型在推理时自己维护（通过prev_bev）。
        
        Args:
            index (int): 样本索引
        Returns:
            dict: 测试数据字典
        """
        input_dict = self.get_data_info(index)       # 获取该帧的数据信息
        self.pre_pipeline(input_dict)                # 预处理
        example = self.pipeline(input_dict)          # 完整流水线处理

        data_dict = {}
        for key, value in example.items():
            # l2g（lidar to global）变换矩阵需要特殊处理
            # 取第一个元素（value[0]）是因为训练时是列表，测试时只取当前帧
            if 'l2g' in key:
                data_dict[key] = to_tensor(value[0])
            else:
                data_dict[key] = value
        return data_dict

    def union2one(self, queue):
        """
        将时序帧队列（list of dict）合并为一个单一的样本字典。
        
        这是构建时序训练数据的关键函数：
        - 把queue_length帧的图像堆叠为 (L, 6, 3, H, W) 的Tensor
        - 把各帧的元数据整理为字典（key为时序索引）
        - 对can_bus数据做差分处理（转为相对位移/角度）
        - 用DataContainer(DC)包装各个字段（用于mmcv多GPU并行）
        
        Args:
            queue (list[dict]): 长度为queue_length的帧数据列表，按时间顺序排列
                                queue[0]是最早帧，queue[-1]是最新帧（当前帧）
        Returns:
            dict: 合并后的单样本字典
        """
        # ---- 从各帧中提取对应字段，组成列表 ----
        imgs_list = [each['img'].data for each in queue]                    # 每帧的图像 (6, 3, H, W)
        gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]   # 每帧的目标类别ID
        gt_sdc_label_list = [each['gt_sdc_label'].data for each in queue]   # 每帧的SDC（自车）类别
        gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]       # 每帧的目标实例ID
        gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]   # 每帧的3D目标框
        gt_past_traj_list = [to_tensor(each['gt_past_traj']) for each in queue]         # 每帧的历史轨迹
        gt_past_traj_mask_list = [
            to_tensor(each['gt_past_traj_mask']) for each in queue]                     # 历史轨迹有效mask
        gt_sdc_bbox_list = [each['gt_sdc_bbox'].data for each in queue]     # 每帧的SDC（自车）bounding box
        l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]   # 每帧的LiDAR到全局旋转矩阵
        l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]           # 每帧的LiDAR到全局平移向量

        # 时间戳列表：每帧一个float64时间戳，形状为 (L, 1)
        timestamp_list = [torch.tensor([each["timestamp"]], dtype=torch.float64) for each in queue]

        # ---- 只取最新帧（queue[-1]）的未来信息 ----
        # 未来轨迹和规划标签只对当前帧有意义
        gt_fut_traj = to_tensor(queue[-1]['gt_fut_traj'])                    # 所有目标的未来轨迹
        gt_fut_traj_mask = to_tensor(queue[-1]['gt_fut_traj_mask'])          # 未来轨迹有效mask
        gt_sdc_fut_traj = to_tensor(queue[-1]['gt_sdc_fut_traj'])            # SDC（自车）未来轨迹
        gt_sdc_fut_traj_mask = to_tensor(queue[-1]['gt_sdc_fut_traj_mask'])  # SDC未来轨迹mask
        gt_future_boxes_list = queue[-1]['gt_future_boxes']                  # 未来帧的目标框（用于Occ评估）
        gt_future_labels_list = [to_tensor(each)
                                 for each in queue[-1]['gt_future_labels']]  # 未来帧的目标类别

        # ---- 处理各帧的元数据（img_metas），并计算相对位移/角度 ----
        # can_bus是车辆CAN总线数据，包含位置([:3])和角度([-1])等信息
        # BEVFormer需要知道相邻帧之间ego车辆的相对位移，以便做BEV特征对齐
        metas_map = {}     # key=帧索引(0~L-1)，value=该帧的img_metas
        prev_pos = None    # 上一帧的绝对位置
        prev_angle = None  # 上一帧的绝对偏航角

        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data

            if i == 0:
                # 第一帧（最早帧）：没有历史帧，prev_bev = False
                metas_map[i]['prev_bev'] = False
                # 记录第一帧的绝对位置和角度（作为基准）
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])    # 前3个是xyz位置
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])  # 最后一个是偏航角（度）
                # 将第一帧的can_bus位移和角度设为0（因为没有参考帧）
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                # 其他帧：有历史帧，prev_bev = True
                metas_map[i]['prev_bev'] = True
                # 保存当前帧的绝对位置和角度
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # 将当前帧can_bus改为相对于上一帧的增量（相对位移和相对角度）
                metas_map[i]['can_bus'][:3] -= prev_pos      # 位置差分
                metas_map[i]['can_bus'][-1] -= prev_angle    # 角度差分
                # 更新"上一帧"的绝对值
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        # ---- 将图像和元数据存回queue[-1]（最新帧，作为最终输出的"容器"）----
        # 把所有帧的图像堆叠成 (L, 6, 3, H, W)，存入最新帧
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)   # cpu_only=False表示需要送到GPU
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)  # 元数据只在CPU上使用

        # 取最新帧作为最终输出字典
        queue = queue[-1]

        # ---- 将各个字段用DataContainer包装后存入最终字典 ----
        # DC（DataContainer）是mmcv的容器，用于处理多GPU数据分发时不同形状的数据
        queue['gt_labels_3d'] = DC(gt_labels_3d_list)                      # 各帧3D目标类别
        queue['gt_sdc_label'] = DC(gt_sdc_label_list)                      # 各帧SDC类别
        queue['gt_inds'] = DC(gt_inds_list)                                # 各帧实例ID
        queue['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)       # 各帧3D目标框（CPU）
        queue['gt_sdc_bbox'] = DC(gt_sdc_bbox_list, cpu_only=True)         # 各帧SDC bbox（CPU）
        queue['l2g_r_mat'] = DC(l2g_r_mat_list)                            # 各帧LiDAR->全局旋转矩阵
        queue['l2g_t'] = DC(l2g_t_list)                                    # 各帧LiDAR->全局平移向量
        queue['timestamp'] = DC(timestamp_list)                            # 各帧时间戳
        queue['gt_fut_traj'] = DC(gt_fut_traj)                             # 当前帧目标未来轨迹
        queue['gt_fut_traj_mask'] = DC(gt_fut_traj_mask)                   # 未来轨迹有效mask
        queue['gt_past_traj'] = DC(gt_past_traj_list)                      # 各帧目标历史轨迹
        queue['gt_past_traj_mask'] = DC(gt_past_traj_mask_list)            # 历史轨迹有效mask
        queue['gt_future_boxes'] = DC(gt_future_boxes_list, cpu_only=True) # 未来帧3D目标框（CPU）
        queue['gt_future_labels'] = DC(gt_future_labels_list)             # 未来帧目标类别
        return queue

    def get_ann_info(self, index):
        """获取某帧的标注信息（3D目标框、类别、轨迹、规划等）。

        这是数据准备的核心函数，负责：
        1. 读取3D目标框（gt_bboxes_3d）和类别（gt_labels_3d）
        2. 获取目标的历史轨迹（gt_past_traj）和未来轨迹（gt_fut_traj）
        3. 获取SDC（自动驾驶车辆/ego车）的信息和轨迹
        4. 获取规划标签（sdc_planning）和驾驶命令（command）

        Args:
            index (int): 样本索引

        Returns:
            dict: 标注信息字典，包含：
                - gt_bboxes_3d: LiDAR坐标系下的3D目标框 (N, 9) [x,y,z,w,l,h,yaw,vx,vy]
                - gt_labels_3d: 目标类别ID (N,)
                - gt_names: 目标类别名称列表 (N,)
                - gt_inds: 目标实例ID (N,)  用于跟踪任务
                - gt_fut_traj: 目标未来轨迹 (N, T, 2)
                - gt_fut_traj_mask: 未来轨迹有效mask (N, T)
                - gt_past_traj: 目标历史轨迹 (N, T, 2)
                - gt_past_traj_mask: 历史轨迹有效mask (N, T)
                - gt_sdc_bbox: SDC的边界框
                - gt_sdc_label: SDC的类别标签
                - gt_sdc_fut_traj: SDC的未来轨迹
                - gt_sdc_fut_traj_mask: SDC未来轨迹mask
                - sdc_planning: SDC的规划轨迹
                - sdc_planning_mask: 规划轨迹有效mask
                - command: 驾驶命令 (0=右转, 1=左转, 2=直行)
        """
        info = self.data_infos[index]  # 获取该帧的原始数据信息字典

        # ---- 筛选有效目标框 ----
        # 根据配置选择过滤条件：
        # use_valid_flag=True时使用NuScenes的valid_flag（标注质量标志）
        # use_valid_flag=False时使用LiDAR点数 > 0来过滤（至少有一个激光点才有效）
        if self.use_valid_flag:
            mask = info['valid_flag']              # bool数组，True表示有效
        else:
            mask = info['num_lidar_pts'] > 0       # 激光点数大于0才有效

        # 用mask筛选目标信息
        gt_bboxes_3d = info['gt_boxes'][mask]      # 3D目标框 (N, 7) [x,y,z,w,l,h,yaw]
        gt_names_3d = info['gt_names'][mask]       # 目标类别名称
        gt_inds = info['gt_inds'][mask]            # 目标实例ID（跟踪任务使用）

        # 从NuScenes API获取该帧的annotation tokens（标注token列表）
        sample = self.nusc.get('sample', info['token'])  # 获取该sample的详细信息
        ann_tokens = np.array(sample['anns'])[mask]      # 筛选有效目标的annotation token
        assert ann_tokens.shape[0] == gt_bboxes_3d.shape[0]  # 确保数量一致

        # ---- 获取目标的历史和未来轨迹标签 ----
        # get_traj_label返回：未来轨迹, 未来轨迹mask, 历史轨迹, 历史轨迹mask
        # 形状分别为 (N, T, 2)，其中T为预测步数，2为(x,y)坐标
        gt_fut_traj, gt_fut_traj_mask, gt_past_traj, gt_past_traj_mask = self.traj_api.get_traj_label(
            info['token'], ann_tokens)

        # ---- 获取SDC（ego车辆）的信息 ----
        # SDC = Self-Driving Car，即自动驾驶车辆本身
        sdc_vel = self.traj_api.sdc_vel_info[info['token']]  # SDC当前速度
        gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(sdc_vel)  # SDC的bbox和类别
        gt_sdc_fut_traj, gt_sdc_fut_traj_mask = self.traj_api.get_sdc_traj_label(
            info['token'])  # SDC的未来轨迹和mask

        # ---- 获取SDC的规划标签和驾驶命令 ----
        # sdc_planning: SDC的规划轨迹（ground truth）
        # sdc_planning_mask: 规划轨迹有效mask
        # command: 高层驾驶命令 (0=右转, 1=左转, 2=直行)
        sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(
            info['token'])

        # ---- 将类别名称转换为类别ID ----
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))  # 类别名→类别ID（0开始）
            else:
                gt_labels_3d.append(-1)  # 不在CLASSES中的类别标记为-1（背景/忽略）
        gt_labels_3d = np.array(gt_labels_3d)

        # ---- 拼接速度信息到目标框 ----
        # NuScenes提供目标的二维速度(vx, vy)，拼接到框的最后
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]   # 速度 (N, 2)
            nan_mask = np.isnan(gt_velocity[:, 0])    # 找出速度为NaN的目标（无速度标注）
            gt_velocity[nan_mask] = [0.0, 0.0]        # NaN速度用0.0替代
            # 拼接后gt_bboxes_3d形状变为 (N, 9)：[x,y,z,w,l,h,yaw,vx,vy]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # ---- 转换目标框中心点格式 ----
        # NuScenes的box中心在 (0.5, 0.5, 0.5)（即box的几何中心）
        # KITTI的box中心在 (0.5, 0.5, 0)（即box的底面中心）
        # 这里统一转换为目标坐标系格式
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],         # 框的维度（7或9）
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)  # 转为指定格式

        # ---- 组装返回字典 ----
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,               # 3D目标框
            gt_labels_3d=gt_labels_3d,               # 目标类别ID
            gt_names=gt_names_3d,                    # 目标类别名称
            gt_inds=gt_inds,                         # 目标实例ID（跟踪用）
            gt_fut_traj=gt_fut_traj,                 # 目标未来轨迹
            gt_fut_traj_mask=gt_fut_traj_mask,       # 未来轨迹有效mask
            gt_past_traj=gt_past_traj,               # 目标历史轨迹
            gt_past_traj_mask=gt_past_traj_mask,     # 历史轨迹有效mask
            gt_sdc_bbox=gt_sdc_bbox,                 # SDC bbox
            gt_sdc_label=gt_sdc_label,               # SDC类别
            gt_sdc_fut_traj=gt_sdc_fut_traj,         # SDC未来轨迹
            gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,  # SDC未来轨迹mask
            sdc_planning=sdc_planning,               # SDC规划轨迹（GT）
            sdc_planning_mask=sdc_planning_mask,     # 规划轨迹mask
            command=command,   # command=0-Right, command=1-Left, command=2-Forward
        )
        # 确保轨迹数量与目标数量一致
        assert gt_fut_traj.shape[0] == gt_labels_3d.shape[0]
        assert gt_past_traj.shape[0] == gt_labels_3d.shape[0]
        return anns_results

    def get_data_info(self, index):
        """获取某帧的完整数据信息，供数据预处理流水线使用。
        
        本函数是最核心的数据组装函数，负责：
        1. 获取地图信息（向量化地图→栅格化mask）
        2. 获取LiDAR数据信息（路径、sweep等）
        3. 计算各种坐标变换矩阵（LiDAR→ego→全局）
        4. 获取相机内参和外参
        5. 获取标注信息（调用get_ann_info）
        6. 获取占用预测所需的时序信息（调用get_occ_data_infos）
        7. 更新can_bus数据（位置、姿态）
        
        Args:
            index (int): 样本索引
        Returns:
            dict: 完整的数据信息字典，将传入数据预处理流水线
        """
        info = self.data_infos[index]  # 原始数据信息

        # ---- 加载车道线语义标注（可选，语义分割格式）----
        lane_info = self.lane_infos[index] if self.lane_infos else None

        # ---- 获取该帧所在位置（城市名），用于加载对应城市的地图 ----
        # NuScenes数据结构：scene -> log -> location
        location = self.nusc.get('log', self.nusc.get(
            'scene', info['scene_token'])['log_token'])['location']

        # ---- 生成向量化地图（以ego车为中心的局部地图） ----
        # vectors是一个向量列表，每个向量代表一个地图元素（车道线、人行横道等）
        vectors = self.vector_map.gen_vectorized_samples(location,
                                                         info['ego2global_translation'],
                                                         info['ego2global_rotation'])

        # ---- 将向量化地图栅格化为多种mask ----
        # semantic_masks: 语义分割mask  (num_classes, H, W)
        # instance_masks: 实例分割mask  (num_classes, H, W)，每个实例用不同整数标记
        # forward_masks:  前向方向mask   (num_angle_classes, H, W)
        # backward_masks: 后向方向mask   (num_angle_classes, H, W)
        semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(vectors,
                                                                                       self.patch_size,
                                                                                       self.canvas_size,
                                                                                       self.map_num_classes,
                                                                                       self.thickness,
                                                                                       self.angle_class)

        # 旋转instance_masks：栅格化坐标系→图像坐标系（顺时针旋转90度）
        instance_masks = np.rot90(instance_masks, k=-1, axes=(1, 2))
        instance_masks = torch.tensor(instance_masks.copy())  # 转为Tensor

        # ---- 从instance_masks中提取每个实例的标签、bbox和mask ----
        gt_labels = []  # 存储每个地图实例的类别ID
        gt_bboxes = []  # 存储每个地图实例的包围框 [xmin, ymin, xmax, ymax]
        gt_masks = []   # 存储每个地图实例的二值mask

        for cls in range(self.map_num_classes):           # 遍历3种地图类别
            for i in np.unique(instance_masks[cls]):      # 遍历该类别下的所有实例
                if i == 0:
                    continue                              # 0是背景，跳过
                gt_mask = (instance_masks[cls] == i).to(torch.uint8)  # 该实例的二值mask
                ys, xs = np.where(gt_mask)               # 找出mask中所有True位置的坐标
                gt_bbox = [min(xs), min(ys), max(xs), max(ys)]  # 从坐标中计算bbox [x1,y1,x2,y2]
                gt_labels.append(cls)
                gt_bboxes.append(gt_bbox)
                gt_masks.append(gt_mask)

        # ---- 获取额外的地图信息（车道分隔线、道路分隔线）----
        map_mask = obtain_map_info(self.nusc,
                                   self.nusc_maps,
                                   info,
                                   patch_size=self.patch_size,
                                   canvas_size=self.canvas_size,
                                   layer_names=['lane_divider', 'road_divider'])  # 只取这两类

        # 对map_mask进行翻转和旋转，与instance_masks对齐坐标系
        map_mask = np.flip(map_mask, axis=1)                        # 沿axis=1翻转（上下翻转）
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))            # 顺时针旋转90度
        map_mask = torch.tensor(map_mask.copy())

        # 将车道分隔线和道路分隔线的实例也加入到gt_labels/gt_bboxes/gt_masks中
        # 注意：map_mask最后一个通道是总体mask，不使用（[:-1]切片）
        for i, gt_mask in enumerate(map_mask[:-1]):
            ys, xs = np.where(gt_mask)
            gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
            gt_labels.append(i + self.map_num_classes)  # 类别ID在原来3类之后追加
            gt_bboxes.append(gt_bbox)
            gt_masks.append(gt_mask)

        # 转为Tensor
        gt_labels = torch.tensor(gt_labels)
        gt_bboxes = torch.tensor(np.stack(gt_bboxes))
        gt_masks = torch.stack(gt_masks)

        # ---- 组装输入字典 ----
        input_dict = dict(
            sample_idx=info['token'],                              # 样本唯一token（NuScenes使用token作为ID）
            pts_filename=info['lidar_path'],                       # LiDAR点云文件路径
            sweeps=info['sweeps'],                                 # LiDAR sweep信息（多扫描聚合用）
            ego2global_translation=info['ego2global_translation'], # ego车到全局坐标系的平移向量
            ego2global_rotation=info['ego2global_rotation'],       # ego车到全局坐标系的旋转四元数
            prev_idx=info['prev'],                                 # 前一帧的token
            next_idx=info['next'],                                 # 后一帧的token
            scene_token=info['scene_token'],                       # 所属场景的token
            can_bus=info['can_bus'],                               # CAN总线数据（位置、速度、加速度等18维向量）
            frame_idx=info['frame_idx'],                           # 该帧在场景中的帧序号（0-based）
            timestamp=info['timestamp'] / 1e6,                    # 时间戳（微秒→秒，除以1e6）
            map_filename=lane_info['maps']['map_mask'] if lane_info else None,  # 语义地图文件路径（可选）
            gt_lane_labels=gt_labels,                             # 地图实例类别
            gt_lane_bboxes=gt_bboxes,                             # 地图实例bbox
            gt_lane_masks=gt_masks,                               # 地图实例mask
        )

        # ---- 计算LiDAR到全局坐标系的变换矩阵 ----
        # 坐标变换链：LiDAR → ego车 → 全局坐标系
        # l2e: LiDAR to Ego（LiDAR坐标系到车身坐标系）
        # e2g: Ego to Global（车身坐标系到全局坐标系）
        l2e_r = info['lidar2ego_rotation']        # LiDAR→ego的旋转四元数
        l2e_t = info['lidar2ego_translation']     # LiDAR→ego的平移向量
        e2g_r = info['ego2global_rotation']       # ego→全局的旋转四元数
        e2g_t = info['ego2global_translation']    # ego→全局的平移向量

        l2e_r_mat = Quaternion(l2e_r).rotation_matrix   # 四元数→3x3旋转矩阵
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # LiDAR→全局的旋转矩阵：先旋转到ego，再旋转到全局
        # 注意：.T是转置（这里利用了旋转矩阵的特性）
        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T   # lidar to global rotation matrix
        # LiDAR→全局的平移向量
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t     # lidar to global translation matrix

        input_dict.update(
            dict(
                l2g_r_mat=l2g_r_mat.astype(np.float32),   # 转为float32节省内存
                l2g_t=l2g_t.astype(np.float32)))

        # ---- 获取相机信息（多相机设置，NuScenes有6个相机）----
        if self.modality['use_camera']:
            image_paths = []      # 各相机图像文件路径
            lidar2img_rts = []    # LiDAR→图像的投影变换矩阵 (4x4)
            lidar2cam_rts = []    # LiDAR→相机坐标系的变换矩阵 (4x4)
            cam_intrinsics = []   # 相机内参矩阵（扩展为4x4）

            for cam_type, cam_info in info['cams'].items():
                # cam_type: 相机名称（'CAM_FRONT', 'CAM_FRONT_LEFT'等）
                # cam_info: 该相机的详细信息（data_path, sensor2lidar_rotation等）
                image_paths.append(cam_info['data_path'])   # 图像文件路径

                # ---- 计算LiDAR到相机坐标系的变换矩阵 ----
                # sensor2lidar_rotation 是「相机→LiDAR」的旋转，取逆得到「LiDAR→相机」的旋转
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])  # LiDAR→相机的旋转矩阵
                # sensor2lidar_translation @ lidar2cam_r.T 得到LiDAR→相机的平移向量
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                # 构建4x4的LiDAR→相机的齐次变换矩阵
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T    # 旋转部分
                lidar2cam_rt[3, :3] = -lidar2cam_t      # 平移部分（取负，因为是齐次坐标形式）

                # 相机内参矩阵（3x3），扩展为4x4（padding with zeros and 1 on diagonal）
                intrinsic = cam_info['cam_intrinsic']    # 3x3内参矩阵
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic  # 嵌入到左上角

                # LiDAR→图像的投影矩阵：先变换到相机坐标系，再用内参投影到图像
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)  # 4x4矩阵乘法
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,          # 6个相机的图像路径列表
                    lidar2img=lidar2img_rts,            # LiDAR→图像变换矩阵列表（用于点云投影）
                    cam_intrinsic=cam_intrinsics,       # 相机内参列表
                    lidar2cam=lidar2cam_rts,            # LiDAR→相机变换矩阵列表
                ))

        # ---- 获取标注信息（目标框、类别、轨迹等）----
        # 注意：原代码有注释 "if not self.test_mode:"，但当前实现训练和测试都获取标注
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        # 将规划相关标签提升到input_dict顶层（方便后续pipeline访问）
        if 'sdc_planning' in input_dict['ann_info'].keys():
            input_dict['sdc_planning'] = input_dict['ann_info']['sdc_planning']
            input_dict['sdc_planning_mask'] = input_dict['ann_info']['sdc_planning_mask']
            input_dict['command'] = input_dict['ann_info']['command']

        # ---- 更新can_bus数据（用ego2global信息填充位置和姿态字段）----
        # can_bus是一个18维向量，这里更新其中的位置和朝向信息
        rotation = Quaternion(input_dict['ego2global_rotation'])  # ego车朝向（四元数）
        translation = input_dict['ego2global_translation']        # ego车位置（全局坐标）
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation                    # can_bus前3个元素 = 全局位置 (x, y, z)
        # NOTE(lty): fix can_bus format, in https://github.com/OpenDriveLab/UniAD/pull/214
        can_bus[3:7] = rotation.elements            # can_bus[3:7] = 四元数 (w, x, y, z)
        # 计算偏航角（yaw angle），从四元数提取绕z轴的旋转角度
        patch_angle = quaternion_yaw(rotation) / np.pi * 180  # 弧度→度（范围：-180~180）
        if patch_angle < 0:
            patch_angle += 360                       # 转换为 0~360 度
        can_bus[-2] = patch_angle / 180 * np.pi     # 存储弧度值（0~2π）
        can_bus[-1] = patch_angle                    # 存储度数值（0~360）

        # ---- 获取占用预测（Occ）所需的时序信息 ----
        all_frames, has_invalid_frame, occ_transforms, occ_future_ann_infos = \
            self.get_occ_data_infos(index)

        # 记录帧有效性信息（某些帧可能因跨场景而无效，用-1表示）
        # NOTE: This can only represent 7 frames in total as it influence evaluation
        input_dict['occ_has_invalid_frame'] = has_invalid_frame            # 是否存在无效帧（bool）
        input_dict['occ_img_is_valid'] = np.array(all_frames) >= 0        # 每帧是否有效（bool数组）
        input_dict.update(occ_transforms)                                  # 各帧的坐标变换矩阵

        # 为当前帧及未来帧提供检测标签（供占用预测任务使用）
        input_dict['occ_future_ann_infos'] = occ_future_ann_infos

        return input_dict

    def get_occ_data_infos(self, index):
        """获取占用预测（Occ）任务所需的时序信息。

        占用预测需要：
        - 过去帧的信息（receptive_field帧，包含当前帧）
        - 未来帧的信息（n_future帧）

        默认配置：occ_receptive_field=3（2过去帧+当前帧），occ_n_future=4
        共涉及 3+4=7 帧的时序信息。

        Args:
            index (int): 当前帧索引

        Returns:
            tuple:
                all_frames (list[int]): 所有相关帧的索引列表，无效帧用-1表示
                has_invalid_frame (bool): 前occ_only_total_frames帧中是否含无效帧
                occ_transforms (dict): 各帧的坐标变换矩阵（l2e, e2g的旋转和平移）
                occ_future_ann_infos (list): 各未来帧的检测标注信息
        """
        # 获取过去帧和未来帧的索引（无效帧用-1表示）
        prev_indices, future_indices = self.occ_get_temporal_indices(
            index, self.occ_receptive_field, self.occ_n_future)

        # 合并所有帧的索引：[过去帧..., 当前帧, 未来帧...]
        all_frames = prev_indices + [index] + future_indices

        # 检查前 occ_only_total_frames(=7) 帧中是否有无效帧（-1）
        # 有无效帧时，某些Occ评估指标会受影响
        has_invalid_frame = -1 in all_frames[:self.occ_only_total_frames]

        # 只取当前帧和未来帧（Occ预测是对当前和未来的占用进行预测）
        future_frames = [index] + future_indices

        # 获取当前帧和未来帧的坐标变换矩阵（l2e, e2g旋转和平移）
        # 无效帧（-1）对应的变换矩阵为None
        occ_transforms = self.occ_get_transforms(future_frames)

        # 获取当前帧和未来帧的检测标注（3D目标框）
        occ_future_ann_infos = self.get_future_detection_infos(future_frames)

        return all_frames, has_invalid_frame, occ_transforms, occ_future_ann_infos

    def get_future_detection_infos(self, future_frames):
        """获取未来帧（包含当前帧）的检测标注信息。

        对于有效帧（索引>=0），调用occ_get_detection_ann_info获取检测标注；
        对于无效帧（索引=-1，即跨场景或越界），返回None。

        Args:
            future_frames (list[int]): 未来帧索引列表，无效帧为-1

        Returns:
            list[dict|None]: 每帧的检测标注，无效帧对应None
        """
        detection_ann_infos = []
        for future_frame in future_frames:
            if future_frame >= 0:
                # 有效帧：获取检测标注
                detection_ann_infos.append(
                    self.occ_get_detection_ann_info(future_frame),
                )
            else:
                # 无效帧（跨场景/越界）：用None占位
                detection_ann_infos.append(None)
        return detection_ann_infos

    def occ_get_temporal_indices(self, index, receptive_field, n_future):
        """计算占用预测所需的过去帧和未来帧索引。

        规则：
        - 如果帧在同一场景内且索引合法，返回实际索引
        - 否则返回-1（无效帧）

        Args:
            index (int): 当前帧索引
            receptive_field (int): 感受野大小（含当前帧的过去帧数，如3表示当前帧+2过去帧）
            n_future (int): 需要预测的未来帧数

        Returns:
            tuple:
                previous_indices (list[int]): 过去帧索引列表，长度=receptive_field-1
                future_indices (list[int]): 未来帧索引列表，长度=n_future
        """
        current_scene_token = self.data_infos[index]['scene_token']  # 当前帧的场景token

        # ---- 生成过去帧索引 ----
        # t的范围：-(receptive_field-1) ~ -1（例如receptive_field=3时，t=-2,-1）
        previous_indices = []
        for t in range(- receptive_field + 1, 0):
            index_t = index + t           # 过去帧的绝对索引
            # 有效条件：1）索引>=0（不越界）；2）与当前帧同场景
            if index_t >= 0 and self.data_infos[index_t]['scene_token'] == current_scene_token:
                previous_indices.append(index_t)
            else:
                previous_indices.append(-1)  # 无效帧标记为-1

        # ---- 生成未来帧索引 ----
        # t的范围：1 ~ n_future（例如n_future=4时，t=1,2,3,4）
        future_indices = []
        for t in range(1, n_future + 1):
            index_t = index + t           # 未来帧的绝对索引
            # 有效条件：1）索引<数据集长度（不越界）；2）与当前帧同场景
            if index_t < len(self.data_infos) and self.data_infos[index_t]['scene_token'] == current_scene_token:
                future_indices.append(index_t)
            else:
                # NOTE: How to deal the invalid indices???
                future_indices.append(-1)  # 无效帧标记为-1

        return previous_indices, future_indices

    def occ_get_transforms(self, indices, data_type=torch.float32):
        """获取各帧的坐标变换矩阵（LiDAR->Ego->Global）。

        坐标变换用于在占用预测中把不同帧的目标坐标对齐到同一坐标系。

        Args:
            indices (list[int]): 帧索引列表，-1表示无效帧
            data_type: Tensor数据类型，默认float32

        Returns:
            dict: 包含4个列表的字典：
                occ_l2e_r_mats: LiDAR→ego的旋转矩阵列表，无效帧对应None
                occ_l2e_t_vecs: LiDAR→ego的平移向量列表，无效帧对应None
                occ_e2g_r_mats: ego→全局的旋转矩阵列表，无效帧对应None
                occ_e2g_t_vecs: ego→全局的平移向量列表，无效帧对应None
        """
        l2e_r_mats = []  # LiDAR→ego旋转矩阵列表 (3x3 Tensor)
        l2e_t_vecs = []  # LiDAR→ego平移向量列表 (3,  Tensor)
        e2g_r_mats = []  # ego→全局旋转矩阵列表  (3x3 Tensor)
        e2g_t_vecs = []  # ego→全局平移向量列表  (3,  Tensor)

        for index in indices:
            if index == -1:
                # 无效帧：所有变换矩阵设为None
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
            else:
                info = self.data_infos[index]
                # 读取该帧的旋转四元数和平移向量
                l2e_r = info['lidar2ego_rotation']        # LiDAR→ego旋转四元数
                l2e_t = info['lidar2ego_translation']     # LiDAR→ego平移向量
                e2g_r = info['ego2global_rotation']       # ego→全局旋转四元数
                e2g_t = info['ego2global_translation']    # ego→全局平移向量

                # 四元数→3x3旋转矩阵，转为Tensor
                l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix)
                e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix)

                l2e_r_mats.append(l2e_r_mat.to(data_type))
                l2e_t_vecs.append(torch.tensor(l2e_t).to(data_type))
                e2g_r_mats.append(e2g_r_mat.to(data_type))
                e2g_t_vecs.append(torch.tensor(e2g_t).to(data_type))

        res = {
            'occ_l2e_r_mats': l2e_r_mats,  # LiDAR→ego旋转矩阵
            'occ_l2e_t_vecs': l2e_t_vecs,  # LiDAR→ego平移向量
            'occ_e2g_r_mats': e2g_r_mats,  # ego→全局旋转矩阵
            'occ_e2g_t_vecs': e2g_t_vecs,  # ego→全局平移向量
        }

        return res

    def occ_get_detection_ann_info(self, index):
        """获取占用预测任务所需的检测标注信息。

        与get_ann_info类似，但：
        1. 不过滤LiDAR点数（保留所有目标，包括远处没有点云的目标）
        2. 额外返回可见性token（gt_vis_tokens）
        3. 不包含轨迹信息（占用预测只需要位置/类别）

        Args:
            index (int): 样本索引

        Returns:
            dict: 标注信息字典，包含：
                gt_bboxes_3d: 3D目标框
                gt_labels_3d: 目标类别ID
                gt_inds: 目标实例ID
                gt_vis_tokens: 目标可见性token（NuScenes提供，表示目标可见程度）
        """
        info = self.data_infos[index].copy()  # 深拷贝，避免修改原始数据

        # 取所有目标（不做mask过滤）
        gt_bboxes_3d = info['gt_boxes'].copy()    # 3D目标框
        gt_names_3d = info['gt_names'].copy()     # 目标类别名称
        gt_ins_inds = info['gt_inds'].copy()      # 目标实例ID

        # 可见性token（NuScenes对每个目标标注了可见程度：v0~v3）
        gt_vis_tokens = info.get('visibility_tokens', None)

        # 计算有效标志（虽然下面通过 occ_filter_by_valid_flag 控制是否使用）
        if self.use_valid_flag:
            gt_valid_flag = info['valid_flag']
        else:
            gt_valid_flag = info['num_lidar_pts'] > 0

        # 占用预测中强制不按有效标志过滤（保留所有目标，包括被遮挡的）
        assert self.occ_filter_by_valid_flag is False
        if self.occ_filter_by_valid_flag:
            # 如果开启了按有效标志过滤（当前不会执行到这里）
            gt_bboxes_3d = gt_bboxes_3d[gt_valid_flag]
            gt_names_3d = gt_names_3d[gt_valid_flag]
            gt_ins_inds = gt_ins_inds[gt_valid_flag]
            gt_vis_tokens = gt_vis_tokens[gt_valid_flag]

        # ---- 将类别名称转为类别ID ----
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)  # 不在CLASSES中的类别标为-1
        gt_labels_3d = np.array(gt_labels_3d)

        # ---- 拼接速度信息 ----
        if self.with_velocity:
            gt_velocity = info['gt_velocity']         # 不做mask过滤，保留所有目标速度
            nan_mask = np.isnan(gt_velocity[:, 0])    # 速度为NaN的目标
            gt_velocity[nan_mask] = [0.0, 0.0]        # NaN→0.0
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # ---- 转换目标框中心格式（NuScenes→KITTI）----
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,       # 3D目标框
            gt_labels_3d=gt_labels_3d,       # 目标类别ID
            # gt_names=gt_names_3d,          # 类别名称（已注释掉）
            gt_inds=gt_ins_inds,             # 目标实例ID
            gt_vis_tokens=gt_vis_tokens,     # 可见性token
        )

        return anns_results

    def __getitem__(self, idx):
        """根据索引获取一个训练/测试样本。

        这是PyTorch Dataset的标准接口，DataLoader会调用此函数。

        逻辑：
        - 测试模式：直接返回prepare_test_data的结果
        - 训练模式：在while循环中调用prepare_train_data，
          若返回None（无效样本），则随机换一个索引重试，
          直到获得有效样本为止

        Args:
            idx (int): 样本索引

        Returns:
            dict: 训练或测试数据字典
        """
        if self.test_mode:
            return self.prepare_test_data(idx)  # 测试模式：单帧处理

        while True:
            data = self.prepare_train_data(idx)  # 训练模式：构建时序队列
            if data is None:
                # 该样本无效（跨场景、空标注等），随机选择另一个样本
                idx = self._rand_another(idx)    # _rand_another是父类方法，随机返回一个新索引
                continue
            return data

    def _format_bbox(self, results, jsonfile_prefix=None):
        """将模型输出结果转换为NuScenes标准评估格式（含跟踪和运动预测）。

        NuScenes评估要求结果以JSON格式提交，每个检测结果包含：
        位置、大小、旋转、速度、类别、得分、属性等字段。
        本函数额外包含跟踪ID（tracking_id）和预测轨迹（predict_traj）。

        Args:
            results (list[dict]): 模型对每帧的预测结果列表
            jsonfile_prefix (str): 输出JSON文件的目录前缀

        Returns:
            str: 输出JSON文件的完整路径
        """
        nusc_annos = {}      # 存储各帧的检测/跟踪结果，key=sample_token
        nusc_map_annos = {}  # 存储各帧的地图评估结果，key=sample_token
        mapped_class_names = self.CLASSES  # 类别名称列表

        print('Start to convert detection format...')
        # 遍历所有帧的预测结果（mmcv.track_iter_progress会显示进度条）
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']  # 该帧的唯一标识token

            # ---- 处理地图评估结果 ----
            if 'map' in self.eval_mod:
                map_annos = {}
                # ret_iou包含各地图类别的交叉区域和并集区域（用于计算IoU）
                for key, value in det['ret_iou'].items():
                    map_annos[key] = float(value.numpy()[0])
                    nusc_map_annos[sample_token] = map_annos

            # 如果该帧没有3D目标框预测，跳过（空帧）
            if 'boxes_3d' not in det:
                nusc_annos[sample_token] = annos
                continue

            # ---- 将模型输出转换为NuScenes Box格式 ----
            boxes = output_to_nusc_box(det)           # 模型输出→NuScenes Box列表
            boxes_ego = copy.deepcopy(boxes)          # 保存ego坐标系下的Box（用于轨迹变换）
            # 将LiDAR坐标系的Box转换到全局坐标系（NuScenes评估在全局坐标系下进行）
            boxes, keep_idx = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                                       mapped_class_names,
                                                       self.eval_detection_configs,
                                                       self.eval_version)
            # keep_idx记录了哪些Box通过了NMS（非极大值抑制），用于与boxes_ego对应

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]  # 类别名称

                # ---- 根据速度和类别确定目标属性（attribute） ----
                # NuScenes评估需要为每个目标指定属性（如vehicle.moving, pedestrian.standing等）
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    # 速度>0.2 m/s，认为目标在运动
                    if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                        attr = 'vehicle.moving'      # 车辆：运动中
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'    # 自行车/摩托：有骑手
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]  # 其他类别使用默认属性
                else:
                    # 速度<=0.2 m/s，认为目标静止
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'  # 行人：站立
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'      # 公交车：停车
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]  # 其他类别使用默认属性

                # 过滤掉不参与最终评估的类别（只保留7类主要目标）
                if name not in ['car', 'truck', 'bus', 'trailer', 'motorcycle',
                                'bicycle', 'pedestrian']:
                    continue

                # ---- 处理预测轨迹（运动预测任务）----
                box_ego = boxes_ego[keep_idx[i]]  # 该目标在ego坐标系下的Box
                trans = box_ego.center             # ego坐标系下的目标中心位置（用于轨迹变换）

                if 'traj' in det:
                    # 局部坐标系下的预测轨迹 (K, T, 2)，K为轨迹数量，T为预测步数
                    traj_local = det['traj'][keep_idx[i]].numpy()[..., :2]
                    traj_scores = det['traj_scores'][keep_idx[i]].numpy()  # 各轨迹的置信度分数
                else:
                    # 没有轨迹预测时，用空数组占位
                    traj_local = np.zeros((0,))
                    traj_scores = np.zeros((0,))

                # ---- 将局部坐标系的轨迹转换到全局坐标系 ----
                traj_ego = np.zeros_like(traj_local)
                # 以目标中心为原点，旋转90度（局部坐标系→ego坐标系的角度差）
                rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi/2)
                for kk in range(traj_ego.shape[0]):
                    # convert_local_coords_to_global将局部坐标点转换为全局坐标
                    traj_ego[kk] = convert_local_coords_to_global(
                        traj_local[kk], trans, rot)

                # ---- 构建NuScenes格式的标注字典 ----
                nusc_anno = dict(
                    sample_token=sample_token,                      # 帧token
                    translation=box.center.tolist(),                # 目标中心坐标 [x, y, z]（全局坐标系）
                    size=box.wlh.tolist(),                         # 目标尺寸 [w, l, h]
                    rotation=box.orientation.elements.tolist(),    # 目标朝向（四元数）
                    velocity=box.velocity[:2].tolist(),            # 速度 [vx, vy]
                    detection_name=name,                           # 检测类别名称
                    detection_score=box.score,                     # 检测置信度分数
                    attribute_name=attr,                           # 目标属性
                    tracking_name=name,                            # 跟踪类别名称（同detection_name）
                    tracking_score=box.score,                      # 跟踪置信度
                    tracking_id=box.token,                         # 跟踪ID（用于多帧关联）
                    predict_traj=traj_ego,                         # 预测轨迹（全局坐标系）
                    predict_traj_score=traj_scores,                # 各条轨迹的概率分数
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos  # 保存该帧的所有检测结果

        # ---- 构建最终提交格式 ----
        nusc_submissions = {
            'meta': self.modality,           # 模态信息（camera/lidar/radar等）
            'results': nusc_annos,           # 检测/跟踪/运动预测结果
            'map_results': nusc_map_annos,   # 地图分割结果
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)  # 确保输出目录存在
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)  # 序列化为JSON文件
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """将结果格式化为JSON文件（供NuScenes官方评估工具使用）。

        如果没有指定输出路径，则创建临时目录。

        Args:
            results (list[dict]): 模型预测结果列表（每帧一个dict）
            jsonfile_prefix (str|None): 输出文件路径前缀，None则自动创建临时目录

        Returns:
            tuple: (result_files, tmp_dir)
                result_files (str): JSON文件路径
                tmp_dir (TemporaryDirectory|None): 临时目录对象（用完后需cleanup）
        """
        assert isinstance(results, list), 'results must be a list'
        # 验证结果数量与数据集大小一致
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            # 未指定路径时，创建临时目录
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None  # 指定了路径，不需要临时目录

        result_files = self._format_bbox(results, jsonfile_prefix)

        return result_files, tmp_dir

    def _format_bbox_det(self, results, jsonfile_prefix=None):
        """将模型输出结果转换为NuScenes纯检测格式（不含跟踪和轨迹）。

        与_format_bbox的区别：
        - 不包含tracking_id、tracking_score等跟踪字段
        - 不包含predict_traj等轨迹预测字段
        - 专门用于检测任务评估

        Args:
            results (list[dict]): 模型预测结果列表
            jsonfile_prefix (str): 输出JSON文件的目录前缀

        Returns:
            str: 输出JSON文件的完整路径（results_nusc_det.json）
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']

            # 若该帧预测结果为None，则存空列表
            if det is None:
                nusc_annos[sample_token] = annos
                continue

            # 将模型输出转换为NuScenes Box格式（纯检测版本）
            boxes = output_to_nusc_box_det(det)           # 与output_to_nusc_box的区别是不含traj
            boxes_ego = copy.deepcopy(boxes)
            # 将LiDAR坐标系的Box转换到全局坐标系
            boxes, keep_idx = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                                       mapped_class_names,
                                                       self.eval_detection_configs,
                                                       self.eval_version)

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]

                # ---- 根据速度和类别确定目标属性 ----
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                # 构建纯检测格式的标注字典（比_format_bbox少了跟踪和轨迹字段）
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc_det.json')  # 注意文件名不同
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results_det(self, results, jsonfile_prefix=None):
        """将检测结果格式化为JSON文件（纯检测版本）。

        与format_results的区别：调用_format_bbox_det而不是_format_bbox。

        Args:
            results (list[dict]): 模型预测结果列表
            jsonfile_prefix (str|None): 输出文件路径前缀

        Returns:
            tuple: (result_files, tmp_dir)
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results_det')
        else:
            tmp_dir = None

        result_files = self._format_bbox_det(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 planning_evaluation_strategy="uniad"):
        """完整的评估函数，支持多任务评估（检测/跟踪/地图/运动/占用/规划）。

        UniAD是一个端到端系统，本函数统一处理所有任务的评估：
        1. 占用预测（Occ）指标：IoU, PQ, SQ, RQ
        2. 规划指标：L2误差, 碰撞率（在0.5s~3.0s各时间段）
        3. 检测/跟踪/运动指标（通过_evaluate_single处理）
        4. 地图分割指标：各类别IoU

        Args:
            results (list[dict] | dict): 模型预测结果。
                如果是dict，则包含多个任务的结果（occ_results_computed, planning_results_computed, bbox_results）
                如果是list，则为标准的bbox预测结果列表
            metric (str): 评估指标名称（默认'bbox'）
            logger: 日志器
            jsonfile_prefix (str|None): JSON输出路径前缀
            result_names (list[str]): 结果名称列表
            show (bool): 是否可视化
            out_dir (str): 可视化结果输出目录
            pipeline: 数据加载流水线（可视化时使用）
            planning_evaluation_strategy (str): 规划评估策略
                'uniad': 每个时间点独立计算（UniAD论文中的方式）
                'stp3': 累积平均计算（STP3论文中的方式）

        Returns:
            dict[str, float]: 各评估指标及其数值
        """
        # ---- 处理多任务结果（dict格式） ----
        if isinstance(results, dict):

            # ---- 打印占用预测评估结果 ----
            if 'occ_results_computed' in results.keys():
                occ_results_computed = results['occ_results_computed']
                out_metrics = ['iou']  # 最少输出IoU指标

                # 如果有全景分割指标（PQ/SQ/RQ），也一并输出
                if occ_results_computed.get('pq', None) is not None:
                    out_metrics = ['iou', 'pq', 'sq', 'rq']

                print("Occ-flow Val Results:")
                for panoptic_key in out_metrics:
                    print(panoptic_key)
                    # 用 & 分隔各类别的指标值（格式化为1位小数）
                    print(' & '.join(
                        [f'{x:.1f}' for x in occ_results_computed[panoptic_key]]))

                # 打印评估样本数量和比例
                if 'num_occ' in occ_results_computed.keys() and 'ratio_occ' in occ_results_computed.keys():
                    print(f"num occ evaluated:{occ_results_computed['num_occ']}")
                    print(f"ratio occ evaluated: {occ_results_computed['ratio_occ'] * 100:.1f}%")

            # ---- 打印规划评估结果（PrettyTable格式）----
            if 'planning_results_computed' in results.keys():
                planning_results_computed = results['planning_results_computed']
                planning_tab = PrettyTable()
                planning_tab.title = f"{planning_evaluation_strategy}'s definition planning metrics"
                # 表头：指标名称 + 6个时间点（0.5s~3.0s）
                planning_tab.field_names = [
                    "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
                for key in planning_results_computed.keys():
                    value = planning_results_computed[key]   # 每个时间点的指标值，长度为6
                    row_value = []
                    row_value.append(key)  # 第一列：指标名称
                    for i in range(len(value)):
                        if planning_evaluation_strategy == "stp3":
                            # STP3方式：取前i+1个时间点的平均值（累积平均）
                            row_value.append("%.4f" % float(value[: i + 1].mean()))
                        elif planning_evaluation_strategy == "uniad":
                            # UniAD方式：取第i个时间点的值（各点独立）
                            row_value.append("%.4f" % float(value[i]))
                        else:
                            raise ValueError(
                                "planning_evaluation_strategy should be uniad or spt3"
                            )
                    planning_tab.add_row(row_value)
                print(planning_tab)

            # 提取bbox评估结果（后续统一处理）
            results = results['bbox_results']  # get bbox_results

        # ---- 格式化结果为JSON文件 ----
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)       # 含跟踪和轨迹
        result_files_det, tmp_dir = self.format_results_det(results, jsonfile_prefix)  # 纯检测

        # ---- 调用_evaluate_single执行核心评估 ----
        if isinstance(result_files, dict):
            # 多个检测头时分别评估
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(
                    result_files[name], result_files_det[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            # 单个检测头时直接评估
            results_dict = self._evaluate_single(
                result_files, result_files_det)

        # ---- 计算地图分割的IoU指标 ----
        if 'map' in self.eval_mod:
            # 累计所有帧的交集和并集（用于计算全局IoU = 总交集/总并集）
            drivable_intersection = 0    # 可行驶区域 交集像素数
            drivable_union = 0           # 可行驶区域 并集像素数
            lanes_intersection = 0       # 车道线 交集像素数
            lanes_union = 0              # 车道线 并集像素数
            divider_intersection = 0     # 分隔线 交集像素数
            divider_union = 0            # 分隔线 并集像素数
            crossing_intersection = 0    # 人行横道 交集像素数
            crossing_union = 0           # 人行横道 并集像素数
            contour_intersection = 0     # 轮廓 交集像素数
            contour_union = 0            # 轮廓 并集像素数

            for i in range(len(results)):
                # 累加每帧的交集和并集
                drivable_intersection += results[i]['ret_iou']['drivable_intersection']
                drivable_union += results[i]['ret_iou']['drivable_union']
                lanes_intersection += results[i]['ret_iou']['lanes_intersection']
                lanes_union += results[i]['ret_iou']['lanes_union']
                divider_intersection += results[i]['ret_iou']['divider_intersection']
                divider_union += results[i]['ret_iou']['divider_union']
                crossing_intersection += results[i]['ret_iou']['crossing_intersection']
                crossing_union += results[i]['ret_iou']['crossing_union']
                contour_intersection += results[i]['ret_iou']['contour_intersection']
                contour_union += results[i]['ret_iou']['contour_union']

            # 计算各类别的IoU = 总交集 / 总并集
            results_dict.update({
                'drivable_iou': float(drivable_intersection / drivable_union),
                'lanes_iou': float(lanes_intersection / lanes_union),
                'divider_iou': float(divider_intersection / divider_union),
                'crossing_iou': float(crossing_intersection / crossing_union),
                'contour_iou': float(contour_intersection / contour_union)
            })
            print(results_dict)

        # 清理临时目录
        if tmp_dir is not None:
            tmp_dir.cleanup()

        # 可视化（如果需要）
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return results_dict

    def _evaluate_single(self,
                         result_path,
                         result_path_det,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """对单个模型执行完整的NuScenes评估（检测/跟踪/运动预测）。

        根据self.eval_mod中包含的模式，分别执行对应的评估：
        - 'det':    目标检测评估（mAP, NDS等）
        - 'track':  多目标跟踪评估（AMOTA, AMOTP等）
        - 'motion': 运动预测评估（minADE, minFDE, EPA等）

        Args:
            result_path (str): 含跟踪和轨迹的结果JSON文件路径（_format_bbox的输出）
            result_path_det (str): 纯检测结果JSON文件路径（_format_bbox_det的输出）
            logger: 日志器
            metric (str): 指标类型（默认'bbox'）
            result_name (str): 指标前缀名称

        Returns:
            dict: 各评估指标的详细数值字典
        """
        # ---- 创建各任务的输出目录 ----
        output_dir = osp.join(*osp.split(result_path)[:-1])        # 根输出目录
        output_dir_det = osp.join(output_dir, 'det')               # 检测结果目录
        output_dir_track = osp.join(output_dir, 'track')           # 跟踪结果目录
        output_dir_motion = osp.join(output_dir, 'motion')         # 运动预测结果目录
        mmcv.mkdir_or_exist(output_dir_det)
        mmcv.mkdir_or_exist(output_dir_track)
        mmcv.mkdir_or_exist(output_dir_motion)

        # 数据集版本到评估集的映射
        eval_set_map = {
            'v1.0-mini': 'mini_val',       # mini版：用mini_val集评估
            'v1.0-trainval': 'val',        # 完整版：用val集评估
        }
        detail = dict()  # 存储所有评估指标

        # ---- 目标检测评估 ----
        if 'det' in self.eval_mod:
            # 使用自定义的NuScenesEval（在官方基础上做了一些修改）
            self.nusc_eval = NuScenesEval_custom(
                self.nusc,
                config=self.eval_detection_configs,     # 评估配置（IoU阈值等）
                result_path=result_path_det,            # 使用纯检测结果文件
                eval_set=eval_set_map[self.version],    # 评估集名称
                output_dir=output_dir_det,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos
            )
            self.nusc_eval.main(plot_examples=0, render_curves=False)  # 不画图，只计算指标

            # 读取评估结果（NuScenes评估器会将结果保存为metrics_summary.json）
            metrics = mmcv.load(osp.join(output_dir_det, 'metrics_summary.json'))
            metric_prefix = f'{result_name}_NuScenes'

            # 将各类别的AP（Average Precision）和TP误差记录到detail
            for name in self.CLASSES:
                # label_aps: 各距离阈值下的AP值 {0.5: ap, 1.0: ap, 2.0: ap, 4.0: ap}
                for k, v in metrics['label_aps'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
                # label_tp_errors: TP检测的各类误差（位移、尺寸、朝向等）
                for k, v in metrics['label_tp_errors'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
                # tp_errors: 全局TP误差
                for k, v in metrics['tp_errors'].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}'.format(metric_prefix, self.ErrNameMapping[k])] = val

            detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']   # NuScenes Detection Score
            detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']    # 平均AP

        # ---- 多目标跟踪评估 ----
        if 'track' in self.eval_mod:
            # 使用NuScenes官方提供的NIPS2019跟踪评估配置
            cfg = config_factory("tracking_nips_2019")
            self.nusc_eval_track = TrackingEval(
                config=cfg,
                result_path=result_path,                # 使用含tracking_id的结果文件
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_track,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root
            )
            self.nusc_eval_track.main()

            # 读取跟踪评估结果
            metrics = mmcv.load(osp.join(output_dir_track, 'metrics_summary.json'))
            # 记录所有跟踪指标
            keys = ['amota', 'amotp', 'recall', 'motar',     # AMOTA: Average Multi-Object Tracking Accuracy
                    'gt', 'mota', 'motp', 'mt', 'ml', 'faf', # MOTA: Multi-Object Tracking Accuracy
                    'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
            for key in keys:
                detail['{}/{}'.format(metric_prefix, key)] = metrics[key]

        # ---- 运动预测评估 ----
        if 'motion' in self.eval_mod:
            # ---- 第一轮：按运动类别评估（将vehicle/pedestrian合并为两大类）----
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type='motion_category'    # 按运动类别（粗粒度）
            )
            print('-'*50)
            print('Evaluate on motion category, merge class for vehicles and pedestrians...')

            # 标准运动预测指标（minADE, minFDE, Miss Rate等）
            print('evaluate standard motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode='standard')

            # 运动预测mAP-minFDE指标（结合检测质量和轨迹预测质量）
            print('evaluate motion mAP-minFDE metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode='motion_map')

            # EPA（Expected Prediction Accuracy）指标（UniAD主要用的运动预测指标）
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode='epa')

            # ---- 第二轮：按检测类别评估（细粒度，10个NuScenes类别）----
            print('-'*50)
            print('Evaluate on detection category...')
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type='detection_category'  # 按检测类别（细粒度）
            )

            # 重复以上三种评估模式
            print('evaluate standard motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode='standard')
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode='motion_map')
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode='epa')

        return detail
    