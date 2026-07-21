# =====================================================================================
# UniAD 原版 Stage2 — 全量数据集 + 6 epoch 对照组配置
#
# 用途：作为 base_e2e_diffusion_full.py（DiT版本，6 epoch）的公平对比基线。
#
# 实验设计原则（唯一变量只有规划头）：
#   - 相同起点权重：uniad_base_track_map.pth
#   - 相同数据集：nuScenes 全量（700 train + 150 val）
#   - 相同训练轮次：6 epoch
#   - 相同训练超参数（lr=2e-4, grad_clip=35, batch=1/卡）
#   - 相同模型结构（queue_length=3, num_query=900, 完整 occ_head）
#   - 唯一区别：planning_head 使用原版 PlanningHeadSingleMode
#
# 使用方法（2卡）：
#   bash tools/uniad_dist_train.sh projects/configs/stage2_e2e/base_e2e_full_6ep.py 2
# =====================================================================================

_base_ = ["./base_e2e.py"]

# 唯一覆盖：把 epoch 从 20 改为 6，其余全部继承 base_e2e.py 原值
total_epochs = 6
runner = dict(type="EpochBasedRunner", max_epochs=6)
evaluation = dict(interval=6, planning_evaluation_strategy="uniad")
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

find_unused_parameters = True