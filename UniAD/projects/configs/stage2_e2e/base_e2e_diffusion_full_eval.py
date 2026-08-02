# =====================================================================================
# UniAD-Diffusion（DiT 规划头）全量数据集 6 epoch 的独立评测专用配置
#
# 背景：
#   与 base_e2e_full_6ep_eval.py 同理——为保证与 baseline 的评测条件完全一致、
#   公平可比，这里同样去掉容易在 nuscenes-devkit TrackingEval 中崩溃的 'track'，
#   只评测 det / map / motion (+ planning)。
#
#   规划头后处理对齐说明：
#   base_e2e_diffusion_full.py 中 planning_head.use_col_optim 已设为 True，
#   DiffusionPlanningHead.forward_test 会与 PlanningHeadSingleMode 一样，
#   对推理轨迹执行相同的 CasADi 碰撞避免后处理（collision_optimization），
#   本文件通过 _base_ 继承该设置，无需重复覆盖，从而保证与 baseline 的
#   planning 指标（L2 / 碰撞率）在完全相同的后处理条件下计算，避免不对等比较。
#
# 用途：
#   独立、稳定地对 epoch_6.pth（DiT 版本）做一次评测，
#   与 base_e2e_full_6ep_eval.py 评测 baseline 的 epoch_6.pth 结果做公平对比。
#
# 使用方法（2卡）：
#   bash tools/uniad_dist_eval.sh \
#       projects/configs/stage2_e2e/base_e2e_diffusion_full_eval.py \
#       projects/work_dirs/stage2_e2e/base_e2e_diffusion_full/epoch_6.pth \
#       2
# =====================================================================================

_base_ = ["./base_e2e_diffusion_full.py"]

# 仅覆盖评估模式：去掉 'track'，与 base_e2e_full_6ep_eval.py 保持一致，确保公平对比
data = dict(
    val=dict(eval_mod=['det', 'map', 'motion']),
    test=dict(eval_mod=['det', 'map', 'motion']),
)
