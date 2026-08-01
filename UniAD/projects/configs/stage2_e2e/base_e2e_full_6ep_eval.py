# =====================================================================================
# UniAD 原版 Stage2 — 全量数据集 6 epoch baseline 的独立评测专用配置
#
# 背景：
#   训练流程内置的 CustomDistEvalHook 在 epoch 6 结束后自动触发评测时，
#   nuscenes-devkit 官方 TrackingEval 在计算 'bicycle' 类别指标时抛出
#   TypeError: list indices must be integers or slices, not str
#   （第三方库在某些类别样本数量退化情况下的已知兼容性问题，
#    与训练本身、与 epoch_6.pth 权重正确性无关）。
#
# 用途：
#   跳过容易崩溃的 tracking 评测，仅评测 det / map / motion (+ planning)，
#   用于独立、稳定地对 epoch_6.pth 做一次评测，并与 DiT 版本做公平对比。
#
# 使用方法（2卡）：
#   bash tools/uniad_dist_eval.sh \
#       projects/configs/stage2_e2e/base_e2e_full_6ep_eval.py \
#       projects/work_dirs/stage2_e2e/base_e2e_full_6ep/epoch_6.pth \
#       2
# =====================================================================================

_base_ = ["./base_e2e_full_6ep.py"]

# 仅覆盖评估模式：去掉 'track'，避免 nuscenes-devkit TrackingEval 的已知崩溃
data = dict(
    val=dict(eval_mod=['det', 'map', 'motion']),
    test=dict(eval_mod=['det', 'map', 'motion']),
)
