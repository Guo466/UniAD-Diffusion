# =====================================================================================
# UniAD-Diffusion: 融合 UniAD + pl_diffusion_models 的配置文件
#
# 相比 base_e2e.py，本配置仅替换 planning_head：
#   PlanningHeadSingleMode  →  DiffusionPlanningHead
#
# DiffusionPlanningHead 使用 Rectified Flow（Flow Matching）范式：
#   - 训练时：GT 轨迹 → 加噪 → DiT 预测去噪方向 → MSE loss
#   - 推理时：高斯噪声 → Euler ODE 积分（5步）→ 去噪轨迹
#
# 其他所有模块（BEVFormer、TrackHead、SegHead、MotionHead、OccHead）与 Stage2 完全相同，
# 可直接加载 uniad_base_track_map.pth 作为初始化权重。
#
# 使用方法：
#   训练：bash tools/dist_train.sh projects/configs/stage2_e2e/base_e2e_diffusion.py 8
#   测试：bash tools/dist_test.sh  projects/configs/stage2_e2e/base_e2e_diffusion.py ckpts/xxx.pth 8
# =====================================================================================

# ---- 继承 base_e2e.py 的所有配置 ----
_base_ = ["./base_e2e.py"]

# =====================================================================================
# 覆盖：将 planning_head 替换为 DiffusionPlanningHead
# =====================================================================================

# =====================================================================================
# 笔记本单卡 + nuScenes Mini 适配配置
# =====================================================================================

# ---- Mini 数据集专用 pkl 路径 ----
_mini_info_root = "data/infos/"
_mini_data_root  = "data/nuscenes/"

# ---- 合并模型配置：planning_head 替换 + 显存优化（两者必须在同一个 model dict 中！）----
# 注意：mmcv 配置系统中，同名变量的后一次赋值会完全覆盖前一次，
#       因此 planning_head 和 显存优化配置必须合并到同一个 model = dict(...)
model = dict(
    # ==== 显存优化（RTX 4060 Laptop, 7.75 GB）====
    queue_length=1,     # 只使用当前帧（省去历史帧 BEV 缓存，节省约 1.5 GB）
    num_query=300,      # track query 从 900 减到 300（节省约 0.5 GB）
    seg_head=dict(
        num_query=100,  # 地图分割 query 从 300 减到 100
        num_things_classes=3,
        num_stuff_classes=1,
    ),
    occ_head=dict(
        # BEV 特征投影大幅降维（256→64），节省最多显存
        bev_proj_dim=64,
        bev_proj_nlayers=1,     # 只用1层卷积投影（原来4层）
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=5,           # 保持5层（需能整除 n_future_blocks=5）
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=64,          # 256→64（主要省显存来源）
                    num_heads=4,            # 8→4（头数同步减少）
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=64,
                    feedforward_channels=256,   # 2048→256（FFN 是显存大户）
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=256,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
    ),

    # ==== 替换 planning_head 为 DiffusionPlanningHead ====
    planning_head=dict(
        type='DiffusionPlanningHead',

        # ---- 基础参数（与原 PlanningHeadSingleMode 对齐）----
        embed_dims=256,
        planning_steps=6,          # 6步 × 0.5s = 3秒规划视野（与 UniAD 原始配置一致）
        output_dim=2,              # 输出 (x, y) 坐标（不含 heading，与 nuScenes 评估对齐）

        # ---- BEV 桥接参数 ----
        bev_h=200,
        bev_w=200,
        n_bev_tokens=64,           # 200×200 → 8×8=64 BEV context tokens
        max_agents=50,             # 最多 50 个 agent（TrackHead 的 num_query=900，取活跃 agent）

        # ---- DiT 扩散解码器参数 ----
        dit_depth=4,               # 4 层 DiT block
        dit_heads=8,               # 8 头 multi-head attention
        sample_steps=5,            # 推理时 Euler ODE 积分步数（越大越精确，但更慢）

        # ---- Flow Matching 损失 ----
        flow_matching_loss_weight=1.0,

        # ---- 碰撞损失（与原接口保持一致，DiT 端到端优化中也保留碰撞约束）----
        loss_collision=[
            dict(type='CollisionLoss', delta=0.0, weight=2.5),
            dict(type='CollisionLoss', delta=0.5, weight=1.0),
            dict(type='CollisionLoss', delta=1.0, weight=0.25),
        ],

        # ---- 兼容性参数（对接原 PlanningHeadSingleMode 接口）----
        loss_planning=None,        # 不使用原 ADE 损失（被 FM loss 替代）
        loss_kinematic=None,       # 不使用运动学损失（DiT 学习分布隐式保证平滑性）
        planning_eval=True,        # 训练时同步评估规划指标（L2 / 碰撞率）
        use_col_optim=False,       # 推理时不使用 CasADi 碰撞优化（DiT 本身学习避障）
        with_adapter=True,         # 启用 BEV Adapter（轻量级特征适配，与原配置一致）
        n_commands=3,              # 3 类驾驶命令（右转/直行/左转）

        # ---- 默认碰撞优化参数（use_col_optim=False 时不生效，保留接口兼容性）----
        col_optim_args=dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0),
    ),
)

# =====================================================================================
# 覆盖：训练超参数（Diffusion 模型通常需要更小的学习率和更多 epochs）
# =====================================================================================

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            "bev_track_bridge": dict(lr_mult=1.0),
            "ego_bridge": dict(lr_mult=1.0),
            "dit_blocks": dict(lr_mult=1.0),
            "t_embedder": dict(lr_mult=1.0),
        }
    ),
    weight_decay=0.01,
)

# =====================================================================================
# 加载权重
# =====================================================================================
load_from = "ckpts/uniad_base_track_map.pth"
find_unused_parameters = True

data = dict(
    # 笔记本单卡，显存约 8GB，batch=1 必须
    samples_per_gpu=1,
    # 笔记本 CPU 核心数有限，降低 worker 数避免内存溢出
    workers_per_gpu=2,

    train=dict(
        # 切换为 Mini 训练 pkl（由 create_data.py --version v1.0-mini 生成）
        ann_file=_mini_info_root + "nuscenes_infos_temporal_mini_infos_temporal_train.pkl",
        data_root=_mini_data_root,
    ),
    val=dict(
        ann_file=_mini_info_root + "nuscenes_infos_temporal_mini_infos_temporal_val.pkl",
        data_root=_mini_data_root,
    ),
    test=dict(
        ann_file=_mini_info_root + "nuscenes_infos_temporal_mini_infos_temporal_val.pkl",
        data_root=_mini_data_root,
    ),
)

# ---- 训练轮次：Mini 数据少，跑更多 epoch ----
total_epochs = 20
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

# ---- 评估间隔：每 5 个 epoch 评估一次（Mini 数据快）----
evaluation = dict(
    interval=5,
    planning_evaluation_strategy="uniad",
)

# ---- 学习率预热步数适配（Mini 数据集 iter 数很少）----
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=50,       # Mini 数据集每 epoch 约 60 iter，50 iter 预热即可
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

# ---- 日志间隔：更频繁打印（Mini 数据 iter 少）----
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)