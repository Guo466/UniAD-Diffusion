# =====================================================================================
# UniAD-Diffusion（DiT 规划头）全量数据集 6 epoch —— sample_steps 消融：steps=10
#
# 目的：
#   验证 "推理时 ODE 积分步数不足" 是否是 full 数据集下 DiT 落后 baseline 的原因之一。
#   直接复用已训练好的 epoch_6.pth（sample_steps=5 训练），仅在推理阶段把 Euler ODE
#   积分步数从 5 提高到 10，不需要重新训练，成本极低（一次评测几分钟）。
#
# 说明：
#   Flow Matching / Rectified Flow 理论上训练与推理的步数可以解耦——训练时学到的是
#   连续时间的速度场 v(x, t)，推理时用多少步数值积分只影响离散化误差，不需要重新训练。
#   如果 steps=10/20 相比 steps=5 有明显提升，说明 5 步积分误差是 full 数据下 DiT 表现
#   不佳的部分原因；如果提升很小，则说明主要瓶颈在别处（训练量不足/超参未适配等）。
#
# 用途：
#   独立、稳定地用同一个 epoch_6.pth 做一次 sample_steps=10 的评测，
#   与 base_e2e_diffusion_full_eval.py（sample_steps=5）的结果做对比。
#
# 使用方法（2卡）：
#   bash tools/uniad_dist_eval.sh \
#       projects/configs/stage2_e2e/base_e2e_diffusion_full_eval_steps10.py \
#       projects/work_dirs/stage2_e2e/base_e2e_diffusion_full/epoch_6.pth \
#       2
# =====================================================================================

_base_ = ["./base_e2e_diffusion_full_eval.py"]

# 仅覆盖推理时的 ODE 积分步数：5 -> 10
model = dict(
    planning_head=dict(
        sample_steps=10,
    ),
)
