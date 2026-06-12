# =====================================================================================
# UniAD Stage2 端到端训练配置文件 (base_e2e.py)
#
# 本文件是 UniAD（Planning-oriented Autonomous Driving）第二阶段的基础配置，
# 在第一阶段（检测+地图）预训练权重的基础上，联合训练所有任务头：
#   - 检测/跟踪 (Detection / Tracking)       → pts_bbox_head (BEVFormerTrackHead)
#   - 地图分割   (Map Segmentation)           → seg_head (PansegformerHead)
#   - 占用预测   (Occupancy Flow Prediction)  → occ_head (OccHead)
#   - 运动预测   (Motion Prediction)          → motion_head (MotionHead)
#   - 轨迹规划   (Planning)                   → planning_head (PlanningHeadSingleMode)
#
# 配置文件采用 mmdetection 的层级继承机制：
#   _base_ 指定基础配置，本文件对其做覆盖和扩展。
# =====================================================================================

# ---- 继承基础配置 ----
# nus-3d.py: NuScenes数据集通用配置（坐标系、点云范围等基础参数）
# default_runtime.py: 默认运行时配置（日志、检查点保存等）
_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]

# Update-2023-06-12: 
# [Enhance] Update some freezing args of UniAD 

# ---- 插件配置 ----
# mmdetection支持外部插件，UniAD的自定义模块放在 projects/mmdet3d_plugin/
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# =============================================================================
# ① 点云与空间参数
# =============================================================================

# 点云范围：[x_min, y_min, z_min, x_max, y_max, z_max]，单位：米
# 以自车为中心，前后左右各51.2m，高度-5m~3m（LiDAR坐标系）
# 注意：如果修改此范围，模型中所有用到 pc_range 的地方也要同步修改
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# 体素大小：[dx, dy, dz]，单位：米
# 0.2m分辨率（xy），z方向8m（整个z范围压缩到一层，即伪2D处理）
voxel_size = [0.2, 0.2, 8]

# BEV地图分割任务中的图像块大小（米），对应 102.4m × 102.4m 的感知范围
patch_size = [102.4, 102.4]

# 图像归一化参数（BGR格式，Caffe风格的ImageNet均值）
# mean: BGR通道均值（对应 [B=103.530, G=116.280, R=123.675]）
# std: 全为1，即不做方差归一化
# to_rgb=False: 保持BGR通道顺序（不转RGB）
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# =============================================================================
# ② 类别与分组配置
# =============================================================================

# NuScenes 10大目标类别（按类别ID 0~9排列）
class_names = [
    "car",                  # 0: 小汽车
    "truck",                # 1: 卡车
    "construction_vehicle", # 2: 工程车辆（挖掘机、起重机等）
    "bus",                  # 3: 公交车
    "trailer",              # 4: 拖车
    "barrier",              # 5: 路障（锥桶、隔离墩）
    "motorcycle",           # 6: 摩托车
    "bicycle",              # 7: 自行车
    "pedestrian",           # 8: 行人
    "traffic_cone",         # 9: 交通锥
]

# 车辆类别ID列表（用于运动预测时区分"可移动车辆"）
# 包含：car(0), truck(1), construction_vehicle(2), bus(3), trailer(4), motorcycle(6), bicycle(7)
# 注意：没有包含 pedestrian(8), barrier(5), traffic_cone(9)
vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]

# 运动预测的类别分组（用于按组分别建模运动模式）
# [0,1,2,3,4]: 大型车辆组（car/truck/construction_vehicle/bus/trailer）
# [6,7]:       两轮车组（motorcycle/bicycle）
# [8]:         行人组
# [5,9]:       静态障碍物组（barrier/traffic_cone，通常不做运动预测）
group_id_list = [[0,1,2,3,4], [6,7], [8], [5,9]]

# 输入模态配置：UniAD只使用纯视觉（6个环视摄像头），不使用LiDAR/雷达
# use_external=True 表示使用外部数据（如 can_bus 自车运动信息）
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

# =============================================================================
# ③ Transformer 维度参数
# =============================================================================

_dim_ = 256           # 主特征维度（Transformer的embed_dims）
_pos_dim_ = _dim_ // 2   # 位置编码维度（128），用于sin/cos位置编码
_ffn_dim_ = _dim_ * 2    # FFN（前馈网络）中间层维度（512）
_num_levels_ = 4          # 多尺度特征层数（FPN输出的4个尺度）
bev_h_ = 200              # BEV特征图高度（对应 y 方向，200格 × 0.512m/格 ≈ 102.4m）
bev_w_ = 200              # BEV特征图宽度（对应 x 方向，200格 × 0.512m/格 ≈ 102.4m）
_feed_dim_ = _ffn_dim_    # 分割头FFN维度（与_ffn_dim_相同，=512）
_dim_half_ = _pos_dim_    # 分割头位置编码维度（与_pos_dim_相同，=128）
canvas_size = (bev_h_, bev_w_)  # BEV画布尺寸，元组形式 (200, 200)

# 时序队列长度：每次输入包含 queue_length 帧（当前帧+2历史帧）
# BEVFormer通过时序注意力融合历史帧的BEV特征
queue_length = 3  # each sequence contains `queue_length` frames.

# =============================================================================
# ④ 轨迹预测参数
# =============================================================================

### traj prediction args ###
predict_steps = 12     # 运动预测的未来时间步数（12步 × 0.5s/步 = 6秒）
predict_modes = 6      # 多模态轨迹数量（预测6条候选轨迹，模拟多种可能的运动意图）
fut_steps = 4          # 训练时GT轨迹的未来步数（4步 × 0.5s = 2秒）
past_steps = 4         # 历史轨迹步数（4步 × 0.5s = 2秒的历史轨迹）
use_nonlinear_optimizer = True  # 使用非线性优化器后处理运动预测结果（提升轨迹平滑性）

# =============================================================================
# ⑤ 占用流预测参数（OccFlow）
# =============================================================================

## occflow setting	
occ_n_future = 4       # 占用预测的未来帧数（预测未来4帧，即2秒的占用状态）
occ_n_future_plan = 6  # 规划任务用到的未来帧数（规划需要更长视野，6帧=3秒）
# 取两者最大值，确保数据集同时满足占用预测和规划任务的时序需求
occ_n_future_max = max([occ_n_future, occ_n_future_plan])  # = 6

# =============================================================================
# ⑥ 规划参数
# =============================================================================

### planning ###
planning_steps = 6     # 规划输出的未来步数（6步 × 0.5s = 3秒轨迹）
use_col_optim = True   # 使用碰撞避免优化（在后处理中利用OccFlow预测结果做碰撞约束）

# 规划评估策略说明（两种不同定义）：
# - "uniad": 在特定时间点计算（如L2距离计算的是第3.0s时的预测与GT之差）
# - "stp3":  计算到某时间点的平均值（如计算0~3.0s内所有步的平均L2距离）
# there exists multiple interpretations of the planning metric, where it differs between uniad and stp3/vad
# uniad: computed at a particular time (e.g., L2 distance between the predicted and ground truth future trajectory at time 3.0s)
# stp3: computed as the average up to a particular time (e.g., average L2 distance between the predicted and ground truth future trajectory up to 3.0s)
planning_evaluation_strategy = "uniad"  # uniad or stp3

# =============================================================================
# ⑦ 占用预测网格配置
# =============================================================================

### Occ args ### 
# 占用预测的BEV网格参数（与规划/分割的空间范围保持一致）
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],   # x轴范围[-50m, 50m]，分辨率0.5m，共200格
    'ybound': [-50.0, 50.0, 0.5],   # y轴范围[-50m, 50m]，分辨率0.5m，共200格
    'zbound': [-10.0, 10.0, 20.0],  # z轴范围[-10m, 10m]，单个bin高度20m（BEV压缩z维）
}

# Other settings
# 训练时GT框与预测框匹配的IoU阈值（低于此阈值的预测不算TP）
train_gt_iou_threshold=0.3

# =============================================================================
# ⑧ 模型配置
# =============================================================================

model = dict(
    type="UniAD",                          # 注册的模型类名（在 detectors/ 目录下定义）
    gt_iou_threshold=train_gt_iou_threshold,  # 训练匹配阈值（=0.3）
    queue_length=queue_length,             # 时序队列长度（=3），决定输入几帧图像
    use_grid_mask=True,                    # 使用GridMask数据增强（随机遮挡网格区域，提升泛化）
    video_test_mode=True,                  # 测试时使用视频模式（维护跨帧的跟踪状态）
    num_query=900,                         # 全局检测query数量（900个object query）
    num_classes=10,                        # 目标类别数（NuScenes 10类）
    vehicle_id_list=vehicle_id_list,       # 运动预测关注的车辆类别ID
    pc_range=point_cloud_range,            # 点云/BEV范围

    # ----------------------------------------------------------------
    # 图像骨干网络：ResNet-101（Caffe预训练，DCNv2增强）
    # ----------------------------------------------------------------
    img_backbone=dict(
        type="ResNet",
        depth=101,                         # ResNet-101
        num_stages=4,                      # 4个Stage（res1~res4）
        out_indices=(1, 2, 3),             # 输出第2、3、4个stage的特征（给FPN）
        frozen_stages=4,                   # 冻结全部4个stage（Stage2阶段骨干网络不训练）
        norm_cfg=dict(type="BN2d", requires_grad=False),  # BatchNorm不训练（frozen）
        norm_eval=True,                    # BN设为eval模式（使用统计量，不更新）
        style="caffe",                     # 使用Caffe风格（第一层conv stride在3×3，非PyTorch风格）
        dcn=dict(
            type="DCNv2", deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
        # 可变形卷积（DCN）：在 stage3 和 stage4 启用，增强感受野
        stage_with_dcn=(False, False, True, True),
    ),

    # ----------------------------------------------------------------
    # 特征金字塔网络（FPN）：融合多尺度特征
    # ----------------------------------------------------------------
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],     # ResNet stage2/3/4 的输出通道数
        out_channels=_dim_,                # 统一输出 256 通道
        start_level=0,                     # 从第0个输入特征层开始
        add_extra_convs="on_output",       # 通过额外卷积在最大尺度上生成更多层
        num_outs=4,                        # 输出4个尺度（P2~P5）
        relu_before_extra_convs=True,      # 额外卷积前加ReLU
    ),

    # ----------------------------------------------------------------
    # 模块冻结策略（Stage2：只训练新增任务头，固定特征提取器）
    # ----------------------------------------------------------------
    freeze_img_backbone=True,   # 冻结图像骨干网络（ResNet-101）
    freeze_img_neck=True,       # 冻结FPN颈部网络
    freeze_bn=True,             # 冻结所有BatchNorm层（使用固定统计量）
    freeze_bev_encoder=True,    # 冻结BEV编码器（BEVFormer Encoder，Stage1已训练好）

    # ----------------------------------------------------------------
    # 检测/跟踪 Query 筛选阈值
    # ----------------------------------------------------------------
    score_thresh=0.4,           # Query激活阈值：检测得分>0.4的query才会成为"活跃目标"
    filter_score_thresh=0.35,   # 输出过滤阈值：最终输出时过滤掉得分<0.35的目标

    # ----------------------------------------------------------------
    # Query交互模块（QIM）参数：用于跨帧的跟踪query传播（来自MOTR）
    # ----------------------------------------------------------------
    qim_args=dict(
        qim_type="QIMBase",        # QIM类型（基础版本）
        merger_dropout=0,          # 历史query与当前query合并时的Dropout率
        update_query_pos=True,     # 是否用预测的新位置更新query的位置编码
        fp_ratio=0.3,              # False Positive Query比例（训练时混入的假阳性query，提升鲁棒性）
        random_drop=0.1,           # 随机丢弃跟踪query的概率（防止过拟合）
    ),  # hyper-param for query dropping mentioned in MOTR

    # ----------------------------------------------------------------
    # 记忆库（Memory Bank）：存储历史帧的目标query，用于长期跟踪
    # ----------------------------------------------------------------
    mem_args=dict(
        memory_bank_type="MemoryBank",     # 记忆库类型
        memory_bank_score_thresh=0.0,      # 记忆库中query的保留阈值（得分>0保留）
        memory_bank_len=4,                 # 记忆库最多保存4帧的历史query
    ),

    # ----------------------------------------------------------------
    # 跟踪损失配置（ClipMatcher：片段级多帧匹配）
    # ----------------------------------------------------------------
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=10,
        weight_dict=None,   # None表示使用默认权重
        # 目标框8个参数的编码权重：[x, y, z, w, l, h, sin(θ), cos(θ), vx, vy]
        # 速度 vx/vy 权重为0.2（比位置/尺寸权重低）
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",  # 匈牙利算法做GT-预测框匹配（跟踪版本）
            cls_cost=dict(type="FocalLossCost", weight=2.0),     # 分类代价权重
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),     # 回归代价权重
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),  # 分类损失：Focal Loss（解决正负样本不平衡）
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),  # 回归损失：L1 Loss
    ),  # loss cfg for tracking

    # ================================================================
    # 检测/跟踪头：BEVFormerTrackHead
    # 负责：BEV特征生成 + 3D目标检测 + 多目标跟踪
    # ================================================================
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,          # BEV特征图高（200）
        bev_w=bev_w_,          # BEV特征图宽（200）
        num_query=900,         # 检测query数量（900个slot，DETR风格）
        num_classes=10,
        in_channels=_dim_,     # 输入特征通道数（=256）
        sync_cls_avg_factor=True,  # 多GPU训练时同步分类损失的平均因子
        with_box_refine=True,  # 使用级联bbox精修（每层decoder都会精化预测框）
        as_two_stage=False,    # 不使用两阶段（使用one-stage DETR-style）
        past_steps=past_steps, # 历史轨迹步数（=4）
        fut_steps=fut_steps,   # 未来轨迹步数（=4）

        # ---- Transformer主体 ----
        transformer=dict(
            type="PerceptionTransformer",
            rotate_prev_bev=True,   # 将历史帧BEV特征旋转对齐到当前帧坐标系（时序融合关键）
            use_shift=True,         # 使用位移对齐（结合can_bus信息补偿自车位移）
            use_can_bus=True,       # 使用CAN Bus信息（自车运动信息：速度、角速度等）
            embed_dims=_dim_,       # embedding维度=256

            # ---- BEV Encoder：将多视角图像特征提升为BEV特征 ----
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=6,                    # 6层BEV Encoder
                pc_range=point_cloud_range,
                num_points_in_pillar=4,          # 每个BEV柱子内采样4个3D参考点（沿z轴均匀分布）
                return_intermediate=False,       # 不返回中间层结果
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        # 第1个注意力：时序自注意力（当前BEV与历史BEV做cross-attention）
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        # 第2个注意力：空间交叉注意力（BEV query与多视角图像特征做deformable cross-attn）
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,          # 每个参考点周围采样8个点
                                num_levels=_num_levels_,  # 4个FPN特征尺度
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,  # FFN隐层=512
                    ffn_dropout=0.1,                 # FFN的Dropout率
                    # Encoder层的操作顺序：时序注意力→归一化→空间交叉注意力→归一化→FFN→归一化
                    operation_order=(
                        "self_attn",   # 时序自注意力
                        "norm",
                        "cross_attn",  # 空间交叉注意力（图像→BEV）
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),

            # ---- 目标检测 Decoder：BEV特征 → 3D目标框 ----
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,              # 6层decoder（级联精化）
                return_intermediate=True,  # 返回每层的中间预测（用于辅助损失）
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        # 第1个注意力：object query间的自注意力（建模目标间关系）
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        # 第2个注意力：object query对BEV特征的可变形交叉注意力
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,  # BEV只有1个尺度
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    # Decoder层操作顺序：自注意力→归一化→交叉注意力→归一化→FFN→归一化
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),

        # ---- 目标框解码器（BEV坐标→3D实际坐标）----
        bbox_coder=dict(
            type="NMSFreeCoder",                             # 无NMS的目标框解码器（DETR风格）
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 后处理保留范围（略大于pc_range）
            pc_range=point_cloud_range,
            max_num=300,        # 最终输出的最大目标数量
            voxel_size=voxel_size,
            num_classes=10,
        ),

        # ---- 位置编码：可学习的行列嵌入 ----
        positional_encoding=dict(
            type="LearnedPositionalEncoding",  # 可学习位置编码（相比Sinusoidal更灵活）
            num_feats=_pos_dim_,               # 位置特征维度=128
            row_num_embed=bev_h_,              # 行方向嵌入数=200（对应BEV高度）
            col_num_embed=bev_w_,              # 列方向嵌入数=200（对应BEV宽度）
        ),

        # ---- 检测头的损失函数 ----
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),  # IoU损失权重为0（未启用）
    ),

    # ================================================================
    # 地图分割头：PansegformerHead
    # 负责：将BEV特征解析为语义地图（道路、车道线、人行横道等）
    # 采用全景分割（Things+Stuff）框架
    # ================================================================
    seg_head=dict(
        type='PansegformerHead',
        bev_h=bev_h_,        # BEV特征图高（200）
        bev_w=bev_w_,        # BEV特征图宽（200）
        canvas_size=canvas_size,          # 输出地图尺寸 (200, 200)
        pc_range=point_cloud_range,
        num_query=300,        # 地图分割用的query数量（300个，少于检测的900）
        num_classes=4,        # 地图类别总数（Things 3类 + Stuff 1类）
        num_things_classes=3, # "Things"类别数（可数实体：车辆/行人/自行车等的占据区域）
        num_stuff_classes=1,  # "Stuff"类别数（背景区域：可行驶区域）
        in_channels=2048,     # 分割头接收ResNet最后层特征（2048通道）
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,

        # ---- 分割Transformer ----
        transformer=dict(
            type='SegDeformableTransformer',
            # Encoder：多尺度可变形注意力，提取全局特征
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',  # 多尺度可变形注意力
                        embed_dims=_dim_,
                        num_levels=_num_levels_,  # 4个尺度
                         ),
                    feedforward_channels=_feed_dim_,  # FFN隐层=512
                    ffn_dropout=0.1,
                    # Encoder只做自注意力（no cross-attn）
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            # Decoder：query对图像特征的交叉注意力，获取地图特征
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,  # 返回中间层结果（辅助损失）
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',     # 自注意力（query间）
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',  # 交叉注意力（query→图像）
                            embed_dims=_dim_,
                            num_levels=_num_levels_,
                        )
                    ],
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                ),
            ),
        ),

        # ---- 分割头的位置编码（正弦位置编码）----
        positional_encoding=dict(
            type='SinePositionalEncoding',  # 正弦位置编码（标准DETR使用）
            num_feats=_dim_half_,           # 位置编码维度=128
            normalize=True,                # 坐标归一化到[0,1]
            offset=-0.5),                  # 坐标偏移（中心化）

        # ---- 分割损失函数 ----
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),       # 分类损失（权重2.0）
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),    # 边界框回归损失（权重5.0，较大）
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),  # IoU损失（权重2.0）
        loss_mask=dict(type='DiceLoss', loss_weight=2.0),  # Mask损失（Dice Loss，权重2.0）

        # Things/Stuff 分别使用独立的Mask生成头（SegMaskHead）
        thing_transformer_head=dict(type='SegMaskHead',d_model=_dim_,nhead=8,num_decoder_layers=4),
        stuff_transformer_head=dict(type='SegMaskHead',d_model=_dim_,nhead=8,num_decoder_layers=6,self_attn=True),

        # ---- 训练配置（匈牙利匹配）----
        train_cfg=dict(
            # 不含mask的标准匈牙利匹配（用于bbox分配）
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                ),
            # 含mask的匈牙利匹配（用于全景分割，同时考虑bbox和mask质量）
            assigner_with_mask=dict(
                type='HungarianAssigner_multi_info',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                mask_cost=dict(type='DiceCost', weight=2.0),  # mask的Dice代价
                ),
            sampler =dict(type='PseudoSampler'),               # 伪采样器（DETR不需要实际采样）
            sampler_with_mask =dict(type='PseudoSampler_segformer'),
        ),
    ),

    # ================================================================
    # 占用流预测头：OccHead
    # 负责：预测未来多帧的BEV占用状态（哪些格子被哪类目标占据）
    # 同时预测运动流（目标的像素级位移）
    # ================================================================
    occ_head=dict(
        type='OccHead',

        grid_conf=occflow_grid_conf,   # BEV网格配置（范围和分辨率）
        ignore_index=255,              # 忽略类别index（255=未知/无效区域）

        bev_proj_dim=256,              # BEV特征投影维度（将BEV特征映射到此维度后输入Occ头）
        bev_proj_nlayers=4,            # BEV特征投影MLP层数

        # Transformer解码器（query对BEV特征做交叉注意力，生成占用预测）
        # attn_mask_thresh: 注意力掩码阈值，低于此值的注意力权重被屏蔽（稀疏注意力）
        attn_mask_thresh=0.3,
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,   # 返回中间层结果（辅助损失）
            num_layers=5,               # 5层Transformer Decoder
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,      # 注意力Dropout=0（不丢弃）
                    proj_drop=0.0,      # 投影Dropout=0
                    dropout_layer=None,
                    batch_first=False),  # 序列维度在第0维（标准PyTorch格式）
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,  # change to 512（注释说明曾考虑改为512）
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),  # 加入残差连接
                feedforward_channels=2048,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),

        # ---- Occupancy Query 配置 ----
        query_dim=256,         # 每个占用query的特征维度
        query_mlp_layers=3,    # query初始化MLP的层数

        # 辅助损失权重（中间层输出的监督信号）
        aux_loss_weight=1.,

        # ---- 占用分割损失（前景/背景二分类）----
        loss_mask=dict(
            type='FieryBinarySegmentationLoss',  # 来自FIERY论文的二值分割损失
            use_top_k=True,        # 只对最难的top-k像素计算损失（困难样本挖掘）
            top_k_ratio=0.25,      # 选取25%最难的像素
            future_discount=0.95,  # 未来帧的损失衰减因子（越远的未来帧权重越低：0.95^t）
            loss_weight=5.0,       # 整体损失权重（较大，说明占用预测是重要任务）
            ignore_index=255,
        ),
        # ---- Dice损失（更好地优化分割Mask的重叠度）----
        loss_dice=dict(
            type='DiceLossWithMasks',
            use_sigmoid=True,      # 先过Sigmoid再计算Dice
            activate=True,
            reduction='mean',
            naive_dice=True,       # 使用简单Dice计算（不用平滑版）
            eps=1.0,               # 防止除零的epsilon
            ignore_index=255,
            loss_weight=1.0),

        # ---- 评估配置 ----
        pan_eval=True,             # 使用全景分割评估（计算PQ/SQ/RQ指标）
        test_seg_thresh=0.1,       # 测试时的分割阈值（预测概率>0.1才算占用）
        test_with_track_score=True,# 测试时结合跟踪置信度分数（提升占用预测质量）
    ),

    # ================================================================
    # 运动预测头：MotionHead
    # 负责：为每个被跟踪目标预测未来多条轨迹（多模态运动预测）
    # 使用可变形注意力在BEV特征图上采样目标未来位置的特征
    # ================================================================
    motion_head=dict(
        type='MotionHead',
        bev_h=bev_h_,            # BEV特征图高（200）
        bev_w=bev_w_,            # BEV特征图宽（200）
        num_query=300,           # 运动预测query数量（从检测的900中选300个目标做运动预测）
        num_classes=10,
        predict_steps=predict_steps,  # 预测未来12步（6秒）
        predict_modes=predict_modes,  # 预测6条候选轨迹（多模态）
        embed_dims=_dim_,        # embedding维度=256

        # ---- 轨迹损失函数 ----
        loss_traj=dict(type='TrajLoss', 
            use_variance=True,           # 预测轨迹的方差（高斯NLL损失）
            cls_loss_weight=0.5,         # 轨迹分类损失权重（哪条轨迹更可能）
            nll_loss_weight=0.5,         # 负对数似然损失权重（轨迹概率分布）
            loss_weight_minade=0.,       # minADE损失权重（最小平均位移误差，设0=不用）
            loss_weight_minfde=0.25),    # minFDE损失权重（最小终点位移误差，0.25）

        num_cls_fcs=3,           # 分类头的全连接层数
        pc_range=point_cloud_range,
        group_id_list=group_id_list,    # 按类别分组（车辆/两轮车/行人分别建模）
        num_anchor=6,            # 锚点轨迹数量（每类目标有6条预定义的轨迹模板）
        use_nonlinear_optimizer=use_nonlinear_optimizer,  # 非线性后处理（True）
        # 预训练的运动锚点信息（6种运动模式的先验轨迹，从训练集统计得到）
        anchor_info_path='data/others/motion_anchor_infos_mode6.pkl',

        # ---- 运动预测Transformer ----
        transformerlayers=dict(
            type='MotionTransformerDecoder',
            pc_range=point_cloud_range,
            embed_dims=_dim_,
            num_layers=3,        # 3层Motion Decoder
            transformerlayers=dict(
                type='MotionTransformerAttentionLayer',
                batch_first=True,  # batch维度在第0维
                attn_cfgs=[
                    dict(
                        type='MotionDeformableAttention',  # 运动感知的可变形注意力
                        num_steps=predict_steps,           # 对12个未来时间步分别采样特征
                        embed_dims=_dim_,
                        num_levels=1,      # BEV只有1个尺度
                        num_heads=8,
                        num_points=4,      # 每个参考点采样4个邻域点
                        sample_index=-1),  # 从最后一层BEV特征采样（-1=最后一层）
                ],

                feedforward_channels=_ffn_dim_,   # FFN隐层=512
                ffn_dropout=0.1,
                # Motion Decoder操作顺序：只有cross-attn（query对BEV），无self-attn
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
        ),
    ),

    # ================================================================
    # 规划头：PlanningHeadSingleMode
    # 负责：预测自车（ego）未来6步（3秒）的行驶轨迹
    # 结合运动预测结果和命令（直行/左转/右转）做端到端规划
    # ================================================================
    planning_head=dict(
        type='PlanningHeadSingleMode',   # 单模态规划（直接输出一条轨迹，不做多假设）
        embed_dims=256,
        planning_steps=planning_steps,   # 规划6步（3秒）

        # ---- 规划损失：L2轨迹误差（ADE，平均位移误差）----
        loss_planning=dict(type='PlanningLoss'),

        # ---- 运动学可行性损失（新增）----
        # 约束预测轨迹满足车辆物理限制，三个子项：
        #   weight_accel: 加速度平滑（相邻步速度变化不能过大），权重0.5
        #   weight_jerk:  jerk平滑（加速度变化率，影响舒适性），权重0.1
        #   weight_fde:   终点位移误差（FDE），强制模型关注终点精度，权重1.0
        # dt=0.5: UniAD中每步时间间隔为0.5秒
        loss_kinematic=dict(
            type='KinematicFeasibilityLoss',
            dt=0.5,
            weight_accel=0.5,
            weight_jerk=0.1,
            weight_fde=1.0,
        ),

        # ---- 碰撞损失：惩罚规划轨迹与预测占用的重叠 ----
        # 三个不同的碰撞边界（delta=0为严格碰撞，delta>0为安全边界扩展）
        loss_collision=[dict(type='CollisionLoss', delta=0.0, weight=2.5),   # 严格碰撞，权重2.5
                        dict(type='CollisionLoss', delta=0.5, weight=1.0),   # 扩展0.5m，权重1.0
                        dict(type='CollisionLoss', delta=1.0, weight=0.25)], # 扩展1.0m，权重0.25

        use_col_optim=use_col_optim,   # 使用碰撞优化（True）
        planning_eval=True,            # 训练时同步评估规划指标
        with_adapter=True,             # 使用Adapter模块（轻量级特征适配层，提升跨任务迁移）
    ),

    # ================================================================
    # 训练配置（pts点云分支的匹配器，虽然UniAD不用LiDAR，但配置保留兼容性）
    # ================================================================
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],             # 体素网格尺寸
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,                   # BEV特征图相对于体素网格的下采样倍数
            assigner=dict(
                type="HungarianAssigner3D",      # 3D检测的匈牙利匹配器
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                # IoU代价权重为0（形式上保留，实际不使用）
                pc_range=point_cloud_range,
            ),
        )
    ),
)

# =============================================================================
# ⑨ 数据集配置
# =============================================================================

dataset_type = "NuScenesE2EDataset"   # 自定义数据集类（在 datasets/ 目录下定义）
data_root = "data/nuscenes/"           # NuScenes原始数据根目录
info_root = "data/infos/"              # 预处理后的pkl标注文件目录
file_client_args = dict(backend="disk")  # 文件读取后端（disk=本地磁盘，也支持HDFS/OSS等）

# 标注文件路径（pkl格式，包含场景/帧信息、标注、传感器参数等）
ann_file_train=info_root + f"nuscenes_infos_temporal_train.pkl"  # 训练集（700个场景）
ann_file_val=info_root + f"nuscenes_infos_temporal_val.pkl"      # 验证集（150个场景）
ann_file_test = info_root + f"nuscenes_infos_temporal_val.pkl"   # 测试集（用val集代替）


# =============================================================================
# ⑩ 数据预处理流程（Pipeline）
# =============================================================================

# ----------------------------------------------------------------------------
# 训练流程（含数据增强）
# ----------------------------------------------------------------------------
train_pipeline = [
    # 步骤1：从文件加载6个视角的图像，并转为float32
    # 支持Ceph分布式文件系统，img_root=''表示使用绝对路径
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=file_client_args, img_root=''),

    # 步骤2：多视角图像的光度畸变增强（随机调整亮度/对比度/饱和度/色调）
    # 提升模型对光照变化的鲁棒性
    dict(type="PhotoMetricDistortionMultiViewImage"),

    # 步骤3：加载多任务标注信息（3D框 + 轨迹 + 实例ID等）
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,         # 加载3D目标框（用于检测/跟踪）
        with_label_3d=True,        # 加载3D类别标签
        with_attr_label=False,     # 不加载属性标签（如moving/stopped）

        with_future_anns=True,     # 加载未来帧标注（用于occ_flow GT生成）
        with_ins_inds_3d=True,     # 加载实例ID（用于跟踪关联）
        ins_inds_add_1=True,       # 实例ID从1开始（0保留为背景/无效）
    ),

    # 步骤4：生成占用流GT标签
    # 根据未来帧的3D框，在BEV网格上光栅化生成：
    #   - gt_segmentation: 语义分割图（哪类目标占据哪个格子）
    #   - gt_instance:     实例分割图（每个格子属于哪个具体目标实例）
    #   - gt_centerness:   中心度图（格子到最近目标中心的距离）
    #   - gt_offset:       偏移图（格子到所属目标中心的像素偏移）
    #   - gt_flow:         前向光流图（目标的运动方向和速度）
    #   - gt_backward_flow: 后向光流图
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                    filter_invisible=False),  # NOTE: Currently vis_token is not in pkl 
    # only_vehicle=True: 只生成车辆类目标的占用标签（不包含行人、障碍物等）
    # filter_invisible=False: 不过滤不可见目标（即使被遮挡也生成标签）

    # 步骤5：按点云范围过滤目标（只保留在point_cloud_range内的目标）
    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),

    # 步骤6：按类别名称过滤目标（只保留class_names中的10个类别）
    dict(type="ObjectNameFilterTrack", classes=class_names),

    # 步骤7：图像归一化（减均值）
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),

    # 步骤8：图像Padding（确保尺寸是32的倍数，满足网络下采样要求）
    dict(type="PadMultiViewImage", size_divisor=32),

    # 步骤9：将各字段转换为Tensor格式
    dict(type="DefaultFormatBundle3D", class_names=class_names),

    # 步骤10：收集需要的字段，传给模型（只保留模型需要的key）
    dict(
        type="CustomCollect3D",
        keys=[
            # ---- 检测/跟踪 GT ----
            "gt_bboxes_3d",        # 3D目标框 (N, 9): [x,y,z,w,l,h,θ,vx,vy]
            "gt_labels_3d",        # 目标类别ID (N,)
            "gt_inds",             # 目标实例ID (N,)（用于跨帧跟踪匹配）
            "img",                 # 6视角图像 (6, C, H, W)

            # ---- 自车运动信息 ----
            "timestamp",           # 当前帧时间戳（秒）
            "l2g_r_mat",           # LiDAR→全局坐标系的旋转矩阵（3×3）
            "l2g_t",               # LiDAR→全局坐标系的平移向量（3,）

            # ---- 运动预测 GT ----
            "gt_fut_traj",         # 目标未来轨迹GT (N, fut_steps, 2)
            "gt_fut_traj_mask",    # 未来轨迹有效掩码 (N, fut_steps)（越界/场景切换时为0）
            "gt_past_traj",        # 目标历史轨迹GT (N, past_steps, 2)
            "gt_past_traj_mask",   # 历史轨迹有效掩码 (N, past_steps)

            # ---- 自车（SDC = Self-Driving Car）相关 GT ----
            "gt_sdc_bbox",         # 自车的3D框（用于可视化和碰撞检测）
            "gt_sdc_label",        # 自车类别标签（通常为car）
            "gt_sdc_fut_traj",     # 自车未来轨迹GT (planning_steps, 2)
            "gt_sdc_fut_traj_mask",# 自车轨迹有效掩码 (planning_steps,)

            # ---- 地图分割 GT ----
            "gt_lane_labels",      # 车道/地图元素类别标签
            "gt_lane_bboxes",      # 车道/地图元素的边界框
            "gt_lane_masks",       # 车道/地图元素的二值掩码（BEV栅格图）

            # ---- 占用流 GT（由GenerateOccFlowLabels生成）----
            "gt_segmentation",     # 语义分割图 (T, H, W)，T=未来帧数
            "gt_instance",         # 实例分割图 (T, H, W)
            "gt_centerness",       # 中心度图   (T, H, W)
            "gt_offset",           # 偏移图     (T, H, W, 2)
            "gt_flow",             # 前向光流图  (T, H, W, 2)
            "gt_backward_flow",    # 后向光流图  (T, H, W, 2)
            "gt_occ_has_invalid_frame",  # 占用GT的时序帧中是否有无效帧（bool）
            "gt_occ_img_is_valid",       # 各时序帧的图像是否有效（bool数组）

            # ---- 规划 GT ----
            "gt_future_boxes",     # 未来各帧的目标框（规划碰撞检测用）
            "gt_future_labels",    # 未来各帧的目标类别
            "sdc_planning",        # 自车规划轨迹GT (planning_steps, 2)
            "sdc_planning_mask",   # 规划轨迹有效掩码 (planning_steps,)
            "command",             # 驾驶命令（0=右转, 1=直行, 2=左转）
        ],
    ),
]

# ----------------------------------------------------------------------------
# 测试/验证流程（无数据增强，加入MultiScaleFlipAug3D封装）
# ----------------------------------------------------------------------------
test_pipeline = [
    # 步骤1：加载6视角图像
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
            file_client_args=file_client_args, img_root=''),

    # 步骤2：图像归一化（与训练相同，但无光度增强）
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),

    # 步骤3：图像Padding
    dict(type="PadMultiViewImage", size_divisor=32),

    # 步骤4：加载未来帧标注（测试时仍需GT用于评估OccFlow和规划）
    # 注意：测试时 with_bbox_3d=False（不加载当前帧3D框GT，避免泄露）
    dict(type='LoadAnnotations3D_E2E', 
         with_bbox_3d=False,       # 测试时不加载当前帧3D框（避免信息泄露）
         with_label_3d=False,      # 不加载类别GT
         with_attr_label=False,

         with_future_anns=True,    # 仍需加载未来帧（生成OccFlow评估GT）
         with_ins_inds_3d=False,   # 测试时不需要实例ID（无跟踪GT监督）
         ins_inds_add_1=True,      # ins_inds start from 1
         ),

    # 步骤5：生成占用流GT（测试时也需要，用于评估OccFlow性能）
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                       filter_invisible=False),

    # 步骤6：MultiScaleFlipAug3D封装（测试时的标准封装）
    # img_scale: 测试图像分辨率 1600×900（NuScenes原始分辨率）
    # flip=False: 测试时不做翻转增强（TTA关闭）
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),     # 测试图像尺寸
        pts_scale_ratio=1,         # 点云缩放比例（=1，不缩放）
        flip=False,                # 不做翻转TTA
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
                # with_label=False：测试时不需要打包标签
            ),
            dict(
                type="CustomCollect3D", keys=[
                    # 测试时收集的字段（比训练少：无GT检测框和实例ID）
                    "img",             # 6视角图像
                    "timestamp",       # 时间戳
                    "l2g_r_mat",       # 坐标变换矩阵
                    "l2g_t",
                    "gt_lane_labels",  # 地图GT（评估地图分割指标用）
                    "gt_lane_bboxes",
                    "gt_lane_masks",
                    # 占用流GT（评估用）
                    "gt_segmentation",
                    "gt_instance", 
                    "gt_centerness", 
                    "gt_offset", 
                    "gt_flow",
                    "gt_backward_flow",
                    "gt_occ_has_invalid_frame",	
                    "gt_occ_img_is_valid",	
                    # 规划GT（评估规划L2和碰撞率用）
                    "sdc_planning",	
                    "sdc_planning_mask",	
                    "command",         # 驾驶命令（评估时作为条件输入）
                ]
            ),
        ],
    ),
]

# =============================================================================
# ⑪ 数据加载器配置（DataLoader）
# =============================================================================

data = dict(
    samples_per_gpu=1,     # 每个GPU的batch size = 1（UniAD显存需求大，只能batch=1）
    workers_per_gpu=8,     # 每个GPU的数据加载工作进程数

    # ---- 训练集配置 ----
    train=dict(
        type=dataset_type,                # NuScenesE2EDataset
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,          # 训练集pkl标注文件
        pipeline=train_pipeline,          # 使用训练流程（含数据增强）
        classes=class_names,
        modality=input_modality,
        test_mode=False,                  # 训练模式（会构建时序队列）
        use_valid_flag=True,              # 使用valid_flag过滤（只保留有LiDAR点的目标）
        patch_size=patch_size,            # BEV地图块大小 [102.4, 102.4]
        canvas_size=canvas_size,          # BEV画布尺寸 (200, 200)
        bev_size=(bev_h_, bev_w_),        # BEV特征图尺寸 (200, 200)
        queue_length=queue_length,        # 时序队列长度=3（当前帧+2历史帧）
        predict_steps=predict_steps,      # 运动预测步数=12
        past_steps=past_steps,            # 历史轨迹步数=4
        fut_steps=fut_steps,              # 未来轨迹步数=4
        use_nonlinear_optimizer=use_nonlinear_optimizer,  # 非线性优化=True

        occ_receptive_field=3,            # 占用预测感受野（含当前帧的过去帧数=3，即过去2帧+当前帧）
        occ_n_future=occ_n_future_max,    # 占用预测未来帧数=6
        occ_filter_invalid_sample=False,  # 不过滤含无效帧的样本（保留跨场景边界样本）
        
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",   # 目标框类型（NuScenes使用LiDAR坐标系）
    ),

    # ---- 验证集配置 ----
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,            # 验证集pkl标注文件
        pipeline=test_pipeline,           # 使用测试流程（无增强）
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,                # 验证时batch=1（逐帧推理）
        # 评估模式：同时评估检测/地图/跟踪/运动预测四个任务
        # 'det': 目标检测（mAP, NDS）
        # 'map': 地图分割（各类别IoU）
        # 'track': 多目标跟踪（AMOTA, AMOTP）
        # 'motion': 运动预测（minADE, minFDE, EPA）
        eval_mod=['det', 'map', 'track','motion'],
        

        occ_receptive_field=3,            # 占用感受野=3（与训练保持一致）
        occ_n_future=occ_n_future_max,    # 未来帧数=6
        occ_filter_invalid_sample=False,  # 不过滤无效帧样本
    ),

    # ---- 测试集配置 ----
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,                   # 测试模式（不构建时序队列，逐帧处理）
        ann_file=ann_file_test,           # 测试集（使用val集）
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=['det', 'map', 'track','motion'],
    ),

    # ---- 采样器配置 ----
    # 训练时使用分布式分组采样（按序列长度分组，减少padding）
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    # 验证/测试时使用分布式均匀采样（不打乱顺序）
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

# =============================================================================
# ⑫ 优化器配置
# =============================================================================

optimizer = dict(
    type="AdamW",      # AdamW优化器（Adam + Weight Decay，常用于Transformer训练）
    lr=2e-4,           # 基础学习率 2×10^-4
    paramwise_cfg=dict(
        custom_keys={
            # 图像骨干网络使用更小的学习率（0.1×基础学习率 = 2×10^-5）
            # 因为骨干网络已经预训练好，不需要大幅更新
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,  # 权重衰减（L2正则化）
)

# 梯度裁剪配置（防止梯度爆炸）
# max_norm=35: 梯度L2范数超过35时进行裁剪
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# =============================================================================
# ⑬ 学习率调度策略
# =============================================================================

# learning policy
lr_config = dict(
    policy="CosineAnnealing",  # 余弦退火策略（学习率从初始值平滑下降到min_lr）
    warmup="linear",           # 线性预热（训练初期缓慢增大学习率，避免不稳定）
    warmup_iters=500,          # 预热迭代数（前500个iter从warmup_ratio×lr线性增大到lr）
    warmup_ratio=1.0 / 3,      # 预热起始比例（从 lr/3 开始预热）
    min_lr_ratio=1e-3,         # 最小学习率比例（cosine最低降至 lr×0.001 = 2×10^-7）
)

# 总训练轮次
total_epochs = 20

# =============================================================================
# ⑭ 评估配置
# =============================================================================

evaluation = dict(
    interval=20,               # 每20个epoch评估一次（仅在训练结束时评估）
    pipeline=test_pipeline,    # 评估时使用测试流程
    planning_evaluation_strategy=planning_evaluation_strategy,  # 规划评估策略="uniad"
)

# =============================================================================
# ⑮ 训练运行器与日志配置
# =============================================================================

# 使用基于Epoch的训练器（每个epoch遍历完整训练集）
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

# 日志配置：同时输出到文本和TensorBoard
log_config = dict(
    interval=10,   # 每10个iter打印一次日志
    hooks=[
        dict(type="TextLoggerHook"),       # 文本日志（输出到控制台和log文件）
        dict(type="TensorboardLoggerHook") # TensorBoard日志（可视化训练曲线）
    ]
)

# 模型检查点保存配置（每个epoch保存一次）
checkpoint_config = dict(interval=1)

# =============================================================================
# ⑯ 预训练权重与其他设置
# =============================================================================

# 从Stage1预训练权重加载（包含已训练好的检测+地图分割模型）
# Stage2在此基础上继续训练，添加运动预测、占用预测和规划头
load_from = "ckpts/uniad_base_track_map.pth"

# 允许模型中存在未使用的参数（多任务模型中某些分支在特定步骤可能不参与）
find_unused_parameters = True

# 日志器名称（使用mmdet的日志系统）
logger_name = 'mmdet'