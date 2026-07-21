# =====================================================================================
# UniAD-Diffusion（服务器 / 全量数据集版）
#
# 目标：与原版 UniAD（base_e2e.py）做“公平对比实验”——
#   除 planning_head 外，所有配置（数据集、batch size、优化器、学习率、epoch数、
#   grad_clip 等）与 base_e2e.py 完全一致，只把 planning_head 从
#   PlanningHeadSingleMode 替换为 DiffusionPlanningHead（Flow Matching DiT）。
#
# 适用环境：8×4090（24GB/卡），nuScenes 全量数据集（Full Dataset，700 train + 150 val）。
#
# 与笔记本版 base_e2e_diffusion.py 的区别：
#   - 不再缩减 queue_length / num_query / seg_head.num_query（保持原版 3 / 900 / 300）
#   - 不再禁用 occ_head（保持原版完整 OccHead，用于对比 occ 指标 & 后续开启碰撞优化）
#   - DiT 结构使用默认（更强）配置：dit_depth=4, dit_heads=8（笔记本版为 2, 4）
#   - 优化器/学习率/梯度裁剪/epoch数/warmup 全部还原为 base_e2e.py 原始值，
#     确保与 UniAD 基线的训练配方完全相同，只有 planning_head 结构不同
#   - data.samples_per_gpu / workers_per_gpu 与原版一致（=1 / =8），
#     8 卡训练时有效 batch size = 8（与 UniAD 论文设置相同）
#
# 使用方法：
#   训练（8卡）：bash tools/uniad_dist_train.sh projects/configs/stage2_e2e/base_e2e_diffusion_full.py 8
#   测试（8卡）：bash tools/uniad_dist_eval.sh  projects/configs/stage2_e2e/base_e2e_diffusion_full.py ckpts/xxx.pth 8
# =====================================================================================

# ---- 继承 base_e2e.py 的全部配置（数据集/优化器/学习率/epoch等均不改动）----
_base_ = ["./base_e2e.py"]

# =====================================================================================
# 覆盖：仅替换 planning_head 为 DiffusionPlanningHead
# 注意：不修改 queue_length / num_query / seg_head / occ_head，均保持 base_e2e.py 原值
# =====================================================================================
model = dict(
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

        # ---- DiT 扩散解码器参数（服务器显存充足，使用完整规模）----
        dit_depth=4,               # 完整 4 层 DiT block（笔记本版为 2）
        dit_heads=8,               # 完整 8 头 multi-head attention（笔记本版为 4）
        sample_steps=5,            # 推理时 Euler ODE 积分步数

        # ---- Flow Matching 损失 ----
        flow_matching_loss_weight=1.0,

        # ---- ADE 辅助损失 ----
        ade_loss_weight=0.5,

        # ---- 碰撞损失（显存充足，恢复与原版一致的三档碰撞损失）----
        loss_collision=[
            dict(type='CollisionLoss', delta=0.0, weight=2.5),
            dict(type='CollisionLoss', delta=0.5, weight=1.0),
            dict(type='CollisionLoss', delta=1.0, weight=0.25),
        ],

        # ---- 兼容性参数（对接原 PlanningHeadSingleMode 接口）----
        loss_planning=None,        # 不使用原 ADE 损失（被 FM loss 替代）
        loss_kinematic=None,       # 不使用运动学损失（DiT 学习分布隐式保证平滑性）
        planning_eval=True,        # 与原版一致：训练时同步评估规划指标
        # 注意：DiffusionPlanningHead.forward_test 当前实现中并未真正调用
        # CasADi 碰撞优化后处理（outs_occflow 参数只做接口兼容，未使用），
        # 无论此处设为 True/False 都不影响推理结果，如实设为 False 避免误导。
        # 如需真正对齐 UniAD 的碰撞优化后处理，需要在 forward_test 中补充实现。
        use_col_optim=False,
        with_adapter=True,         # 与原版一致：启用 BEV Adapter
        n_commands=3,              # 3 类驾驶命令（右转/直行/左转）

        col_optim_args=dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0),
    ),
)

# =====================================================================================
# 训练超参数：与 base_e2e.py 完全一致（不做任何覆盖），
# 以下参数均继承自 _base_，此处仅注释说明，便于对比实验时核对：
#   optimizer      = AdamW, lr=2e-4, img_backbone lr_mult=0.1, weight_decay=0.01
#   optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
#   lr_config      = CosineAnnealing, warmup_iters=500, warmup_ratio=1/3, min_lr_ratio=1e-3
#   total_epochs   = 20
#   evaluation     = interval=20, planning_evaluation_strategy="uniad"
#   data.samples_per_gpu = 1, workers_per_gpu = 8
#   load_from      = "ckpts/uniad_base_track_map.pth"
# 如需针对 DiT 做超参微调（如更小学习率），单独在对照组之外的消融配置中覆盖，
# 不要修改本文件，以保证与 UniAD 基线的可比性。
# =====================================================================================

# ---- 数据集：使用全量数据集 pkl（由 create_data.py 默认（非 --version v1.0-mini）生成）----
# ann_file / data_root 均继承自 base_e2e.py 的默认值：
#   data_root = "data/nuscenes/"
#   info_root = "data/infos/"
#   ann_file_train = data/infos/nuscenes_infos_temporal_train.pkl
#   ann_file_val   = data/infos/nuscenes_infos_temporal_val.pkl
# 此处无需覆盖，保留 _base_ 默认值即可。

find_unused_parameters = True

# =====================================================================================
# 2卡训练 / 6 epoch 快速验证版覆盖
# 用途：在 2×4090 上以 6 epoch 与原版 UniAD（base_e2e_full_6ep.py）做公平对比。
# 注意：若之后改为 8 卡 20 epoch 完整训练，把下面这段注释掉即可。
# =====================================================================================
total_epochs = 6
runner = dict(type="EpochBasedRunner", max_epochs=6)
evaluation = dict(interval=6, planning_evaluation_strategy="uniad")
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,   # 全量数据集每 epoch 约 3500 iter，500 iter 预热合理
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)