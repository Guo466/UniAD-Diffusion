# =====================================================================================
# UniAD 原版 Stage2 — nuScenes Mini 适配配置
#
# 用途：在 mini 数据集上训练 UniAD 原版规划头（PlanningHeadSingleMode），
#       作为 base_e2e_diffusion.py（DiT 版本）的公平对比基线。
#
# 实验设计原则（唯一变量只有规划头）：
#   - 相同起点权重：uniad_base_track_map.pth
#   - 相同数据集：nuScenes mini
#   - 相同训练轮次：20 epoch
#   - 相同显存优化：queue_length=1, num_query=300, occ_head=None
#   - 唯一区别：planning_head 使用原版 PlanningHeadSingleMode
#
# 使用方法：
#   训练：python tools/train.py projects/configs/stage2_e2e/base_e2e_mini.py \
#             --no-validate
#   测试：python tools/test.py projects/configs/stage2_e2e/base_e2e_mini.py \
#             work_dir/uniad_mini/latest.pth --out output/uniad_mini_results.pkl \
#             --launcher none
# =====================================================================================

# 继承原版 Stage2 完整配置
_base_ = ["./base_e2e.py"]

# ---- Mini 数据集路径 ----
_mini_info_root = "data/infos/"
_mini_data_root  = "data/nuscenes/"

# =====================================================================================
# 显存优化（与 base_e2e_diffusion.py 完全相同，保证公平对比）
# =====================================================================================
model = dict(
    # 只使用当前帧，省去历史帧 BEV 缓存（节省约 1.5 GB）
    queue_length=1,
    # track query 从 900 减到 300（节省约 0.5 GB）
    num_query=300,
    seg_head=dict(
        num_query=100,
        num_things_classes=3,
        num_stuff_classes=1,
    ),
    # 禁用 OccHead（节省约 1.0~1.5 GB，与 DiT 版本保持一致）
    occ_head=None,
    # 规划头：PlanningHeadSingleMode（唯一与 DiT 版本不同的地方）
    # 完整写出所有参数，避免继承 base_e2e.py 的默认值导致报错
    planning_head=dict(
        type='PlanningHeadSingleMode',
        embed_dims=256,
        planning_steps=6,
        # ADE 轨迹损失（必须提供，不能为 None）
        loss_planning=dict(type='PlanningLoss'),
        # 碰撞损失：occ_head=None 时无法使用，改为空列表
        # （不能传 None，PlanningHeadSingleMode 会对它做迭代）
        loss_collision=[],
        # 运动学损失：显存优化，禁用
        loss_kinematic=None,
        # 碰撞避免优化：occ_head=None 时无法启用
        use_col_optim=False,
        col_optim_args=dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0),
        # 训练时评估：显存优化，关闭
        planning_eval=False,
        # BEV Adapter：显存优化，关闭
        with_adapter=False,
    ),
)

# =====================================================================================
# 训练超参数（与 DiT 版本保持一致）
# =====================================================================================
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)

# 起点权重（与 DiT 版本相同，从 Stage1 fine-tune）
load_from = "ckpts/uniad_base_track_map.pth"
find_unused_parameters = True

# =====================================================================================
# 数据配置（与 DiT 版本完全相同）
# =====================================================================================
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file=_mini_info_root + "nuscenes_infos_temporal_mini_infos_temporal_train.pkl",
        data_root=_mini_data_root,
        queue_length=1,
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

# =====================================================================================
# 训练轮次（与 DiT 版本相同：20 epoch）
# =====================================================================================
total_epochs = 20
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

evaluation = dict(
    interval=5,
    planning_evaluation_strategy="uniad",
)

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)

# 输出目录（与 DiT 版本区分）
work_dir = "work_dir/uniad_mini"