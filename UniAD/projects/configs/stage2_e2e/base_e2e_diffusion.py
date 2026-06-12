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

model = dict(
    # 只覆盖 planning_head，其余字段（BEVFormer、TrackHead、MotionHead 等）继承自 base_e2e.py
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
# 覆盖：任务损失权重
# 与 base_e2e.py 保持一致，planning 权重不变
# =====================================================================================
# 注：task_loss_weight 在 UniADTrack 中定义，不在 model dict 中直接暴露
# 如需调整可在此处通过 model.task_loss_weight 覆盖：
# model.update({'task_loss_weight': dict(track=1.0, map=1.0, motion=1.0, occ=1.0, planning=1.0)})

# =====================================================================================
# 覆盖：训练超参数（Diffusion 模型通常需要更小的学习率和更多 epochs）
# =====================================================================================

# DiffusionPlanningHead 的新参数使用正常 lr，冻结层使用更小的 lr
# 与 base_e2e.py 的 optimizer 配置合并（只覆盖 lr）
optimizer = dict(
    type="AdamW",
    lr=1e-4,           # 比原版（2e-4）略小，适应 diffusion 训练的稳定性要求
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),    # 骨干网络使用 0.1× lr
            "bev_track_bridge": dict(lr_mult=1.0), # 桥接层正常 lr（新模块，需要充分训练）
            "ego_bridge": dict(lr_mult=1.0),
            "dit_blocks": dict(lr_mult=1.0),
            "t_embedder": dict(lr_mult=1.0),
        }
    ),
    weight_decay=0.01,
)

# =====================================================================================
# 覆盖：评估配置（添加扩散规划特有的评估策略）
# =====================================================================================

# 保持与 base_e2e.py 一致，规划评估使用 "uniad" 策略
planning_evaluation_strategy = "uniad"

evaluation = dict(
    interval=20,
    planning_evaluation_strategy=planning_evaluation_strategy,
)

# =====================================================================================
# 加载权重说明：
# - 推荐从 Stage2 的 base_e2e 预训练权重初始化
#   （BEVFormer + TrackHead + SegHead + MotionHead + OccHead 均已训练好）
# - DiffusionPlanningHead 的新参数（桥接层 + DiT）将从随机初始化开始训练
# - 若无 Stage2 权重，也可从 Stage1 权重（track+map）开始（但需要更多 epoch）
# =====================================================================================
load_from = "ckpts/uniad_base_track_map.pth"   # Stage1 权重（或改为 Stage2 权重）
find_unused_parameters = True