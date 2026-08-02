# =====================================================================================
# UniAD-Diffusion（DiT 规划头）全量数据集 6 epoch —— sample_steps 消融：steps=20
#
# 目的：与 base_e2e_diffusion_full_eval_steps10.py 相同，验证 ODE 积分步数是否是
#       瓶颈；本文件把步数进一步提高到 20，观察是否较 steps=10 有进一步提升或已饱和。
#
# 使用方法（2卡）：
#   bash tools/uniad_dist_eval.sh \
#       projects/configs/stage2_e2e/base_e2e_diffusion_full_eval_steps20.py \
#       projects/work_dirs/stage2_e2e/base_e2e_diffusion_full/epoch_6.pth \
#       2
# =====================================================================================

_base_ = ["./base_e2e_diffusion_full_eval.py"]

# 仅覆盖推理时的 ODE 积分步数：5 -> 20
model = dict(
    planning_head=dict(
        sample_steps=20,
    ),
)
