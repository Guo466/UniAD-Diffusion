# =====================================================================================
# sample_steps 消融实验：steps=20
#
# 仅用于评估阶段，验证 Euler ODE 积分步数对轨迹质量和推理耗时的影响。
# 继承 base_e2e_diffusion.py 的全部配置，只覆盖 planning_head.sample_steps。
# 使用同一个训练好的 checkpoint（latest.pth）进行评测，不需要重新训练。
#
# 使用方法：
#   PYTHONPATH=. python tools/test.py \
#       projects/configs/stage2_e2e/base_e2e_diffusion_steps20.py \
#       projects/work_dirs/stage2_e2e/base_e2e_diffusion/latest.pth \
#       --launcher none --out output/dit_results_steps20.pkl
# =====================================================================================

_base_ = ["./base_e2e_diffusion.py"]

model = dict(
    planning_head=dict(
        sample_steps=20,
    ),
)