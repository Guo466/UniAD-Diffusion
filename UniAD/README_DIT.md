# UniAD + Flow Matching DiT 规划头改造说明

本文档记录基于开源项目 [UniAD](https://github.com/OpenDriveLab/UniAD) 所做的规划头（Planning Head）改造：
将原始的确定性回归式 `PlanningHeadSingleMode` 替换为基于 **Flow Matching（Rectified Flow）** 的
**Diffusion Transformer（DiT）** 生成式规划头 `DiffusionPlanningHead`，并提供了与原版 UniAD 的
公平对比评测工具链。

用于记录当前（nuScenes **mini** 数据集）版本的运行方式，方便后续在服务器上用
**完整（full）数据集** 重新训练后，快速复现同样的评测与对比流程。

---

## 1. 版本与分支说明

- 本次改造在当前 commit 上完成并保存，后续如需回退到 mini 数据集验证通过的版本，可直接：

  ```bash
  git log --oneline          # 找到本次 commit 的 hash
  git reset --hard <commit_hash>
  ```

- 核心新增/修改文件：
  - `projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py`：DiT 规划头实现（Flow Matching 训练 + Euler ODE 推理）
  - `projects/configs/stage2_e2e/base_e2e_diffusion.py`：DiT 版本配置（笔记本 8GB 显存优化 + mini 数据集）
  - `projects/configs/stage2_e2e/base_e2e_diffusion_steps10.py` / `..._steps20.py`：`sample_steps` 消融实验配置
  - `projects/configs/stage2_e2e/base_e2e_mini.py`：原版 UniAD 的 mini 数据集对比配置（用于公平 baseline）
  - `tools/test.py`：重构了单卡评测逻辑（`custom_single_gpu_test`），补全规划指标（`PlanningMetric`）在线累积，
    并新增单帧推理耗时统计（`[Inference Timing]`）
  - `tools/compare_metrics.py`：DiT vs UniAD 原版规划指标（L2 + 碰撞率）自动对比脚本
  - `tools/analysis_tools/visualize/run.py`：支持按 scene 分别生成视频、自定义 fps 等（相比原版增强）

---

## 2. 环境与数据准备

沿用原版 UniAD 的环境搭建流程，参见 `docs/INSTALL.md`、`docs/DATA_PREP.md`。

mini 数据集需要额外生成 mini 版本的 info 文件（若尚未生成）：

```bash
# 在项目根目录下
python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/infos \
    --extra-tag nuscenes_infos_temporal_mini --version v1.0-mini --canbus data
```

Full 数据集则使用标准的 `--version v1.0-trainval`，具体参见 `docs/DATA_PREP.md`。

---

## 3. 训练指令

### 3.1 原版 UniAD（baseline，用于对比）

```bash
# mini 数据集（笔记本 / 单卡显存优化版）
PYTHONPATH=. python tools/train.py \
    projects/configs/stage2_e2e/base_e2e_mini.py \
    --launcher none

# full 数据集（服务器 / 多卡，按需调整 GPU 数量与 batch size）
bash tools/uniad_dist_train.sh projects/configs/stage2_e2e/base_e2e.py 8
```

### 3.2 DiT（Flow Matching）版本

```bash
# mini 数据集（笔记本 / 单卡显存优化版，20 epoch）
PYTHONPATH=. python tools/train.py \
    projects/configs/stage2_e2e/base_e2e_diffusion.py \
    --launcher none

# full 数据集（服务器 / 多卡，按需调整 GPU 数量与 batch size）
# 注意：base_e2e_diffusion.py 当前包含大量“笔记本 8GB 显存优化”配置
# （queue_length=1、num_query=300、occ_head=None、dit_depth=2 等），
# 在服务器满显存环境下训练 full 数据集前，建议按需放开这些限制以获得更强性能，
# 或新建一个 base_e2e_diffusion_full.py 继承本文件后覆盖对应字段。
bash tools/uniad_dist_train.sh projects/configs/stage2_e2e/base_e2e_diffusion.py 8
```

训练权重默认输出到：
- 原版：`work_dir/uniad_mini/`（mini，自定义）或 `projects/work_dirs/stage2_e2e/base_e2e/`（full，默认）
- DiT：`projects/work_dirs/stage2_e2e/base_e2e_diffusion/`

---

## 4. 评测指令

两版模型均使用统一的 `tools/test.py`，单卡评测需加 `--launcher none`。

### 4.1 原版 UniAD

```bash
PYTHONPATH=. python tools/test.py \
    projects/configs/stage2_e2e/base_e2e_mini.py \
    work_dir/uniad_mini/latest.pth \
    --launcher none \
    --out output/uniad_mini_results.pkl
```

### 4.2 DiT 版本（默认 sample_steps=5）

```bash
PYTHONPATH=. python tools/test.py \
    projects/configs/stage2_e2e/base_e2e_diffusion.py \
    projects/work_dirs/stage2_e2e/base_e2e_diffusion/latest.pth \
    --launcher none \
    --out output/dit_results_steps5.pkl
```

评测结束后终端会打印：
- `[Inference Timing]`：单帧纯前向推理耗时（均值/标准差/FPS）
- `planning_results_computed`（保存在输出 pkl 顶层）：包含 `L2` / `obj_col` / `obj_box_col`
  三个 Tensor（各 6 个时刻，对应 0.5s~3s，2Hz），与官方 UniAD 评测协议完全一致

### 4.3 sample_steps 消融实验（可选）

```bash
# steps=10
PYTHONPATH=. python tools/test.py \
    projects/configs/stage2_e2e/base_e2e_diffusion_steps10.py \
    projects/work_dirs/stage2_e2e/base_e2e_diffusion/latest.pth \
    --launcher none --out output/dit_results_steps10.pkl

# steps=20
PYTHONPATH=. python tools/test.py \
    projects/configs/stage2_e2e/base_e2e_diffusion_steps20.py \
    projects/work_dirs/stage2_e2e/base_e2e_diffusion/latest.pth \
    --launcher none --out output/dit_results_steps20.pkl
```

---

## 5. 对比方法与指令

### 5.1 指标对比：DiT vs UniAD 原版（核心对比，用于验证改造效果）

使用 `tools/compare_metrics.py`，自动对比 L2 误差与碰撞率（obj_col / obj_box_col），
在 1s / 2s / 3s 三个时刻分别给出优化幅度：

```bash
python tools/compare_metrics.py \
    --uniad_pkl output/uniad_mini_results.pkl \
    --dit_pkl   output/dit_results_steps5.pkl \
    --out_csv   output/compare_mini_final.csv
```

脚本会优先读取 pkl 顶层的 `planning_results_computed`（评测时在线累积算出，最权威），
若某个 pkl 是旧版本没有该字段，会自动回退为逐样本重新计算 L2（此时碰撞率不可用，
脚本会给出明确提示）。

**mini 数据集（20 epoch）参考对比结果**（用于核对复现是否正常）：

| 指标 | UniAD 原版 | DiT (steps=5) | 优化幅度 |
|------|-----------|---------------|---------|
| L2@1s (m) | 1.530 | 1.337 | ↓ 12.6% |
| L2@2s (m) | 3.187 | 2.897 | ↓ 9.1% |
| L2@3s (m) | 4.943 | 4.714 | ↓ 4.6% |
| obj_col@3s | 0.0617 | 0.0000 | ↓ 100% |
| obj_box_col@3s | 0.1111 | 0.0494 | ↓ 55.5% |

结论：相同 mini 数据集、相同 20 epoch 训练条件下，DiT 版本在规划 L2 误差上平均降低约 9%，
在 3 秒碰撞率上降低 55%~100%，验证了 Flow Matching 生成式建模在多模态轨迹预测与
避障安全性上相比传统回归范式的优势。

**Full 数据集复现时，只需替换两个 pkl 路径重新跑一遍上述命令**，即可得到同样格式的对比表。

### 5.2 sample_steps 消融实验对比（附加分析，非核心对比）

```bash
python3 - << 'PYEOF'
import mmcv

configs = [
    ("steps=5",  "output/dit_results_steps5.pkl"),
    ("steps=10", "output/dit_results_steps10.pkl"),
    ("steps=20", "output/dit_results_steps20.pkl"),
]

for name, path in configs:
    data = mmcv.load(path)
    pm = data['planning_results_computed']
    print(f"\n{name}:")
    print(f"  L2         : {pm['L2'].cpu().numpy()}")
    print(f"  obj_col    : {pm['obj_col'].cpu().numpy()}")
    print(f"  obj_box_col: {pm['obj_box_col'].cpu().numpy()}")
PYEOF
```

结论（mini 数据集）：当前默认 `sample_steps=5` 已是 L2 精度最优选择；更多积分步数
（10/20）因训练阶段仅用单步 FM+MSE 目标（train-inference step mismatch）反而使 L2
略微上升，但能小幅改善碰撞率（10 步后饱和）；由于 `dit_depth=2` 很浅，
步数从 5→20 仅带来约 1.2% 的推理耗时增长，代价很小。

### 5.3 可视化视频对比

```bash
# 生成 DiT 可视化视频
PYTHONPATH=. python tools/analysis_tools/visualize/run.py \
    --predroot output/dit_results_steps5.pkl \
    --out_folder output/dit_viz \
    --demo_video output/dit_demo.avi \
    --project_to_cam True --fps 2

# 生成原版 UniAD 可视化视频
PYTHONPATH=. python tools/analysis_tools/visualize/run.py \
    --predroot output/uniad_mini_results.pkl \
    --out_folder output/uniad_viz \
    --demo_video output/uniad_demo.avi \
    --project_to_cam True --fps 2
```

两者会按 scene 分别渲染并生成独立视频，可用 `ffmpeg` 的 `vstack` 滤镜拼接成上下对比视频：

```bash
ffmpeg -y \
  -i output/uniad_demo_scene00_xxx.avi \
  -i output/dit_demo_scene00_xxx.avi \
  -filter_complex "\
[0:v]scale=1600:-1,drawtext=text='UniAD (baseline)':x=20:y=20:fontsize=32:fontcolor=yellow:box=1:boxcolor=black@0.5[top]; \
[1:v]scale=1600:-1,drawtext=text='DiT (Flow Matching)':x=20:y=20:fontsize=32:fontcolor=yellow:box=1:boxcolor=black@0.5[bottom]; \
[top][bottom]vstack=inputs=2[v]" \
  -map "[v]" -c:v libx264 -crf 20 -pix_fmt yuv420p \
  output/compare_scene00.mp4
```

---

## 6. 已知注意事项 / 踩坑记录

1. **单卡评测必须加 `--launcher none`**，否则会因为分布式环境变量缺失（`KeyError: 'RANK'`）而报错。
2. `tools/test.py` 的 `custom_single_gpu_test` 已补全 `PlanningMetric` / `OccMetric` 累积逻辑，
   若从更早的 commit 复现，注意确认该函数是否包含耗时统计与指标累积代码（可用
   `grep -n "Inference Timing\|planning_results_computed" tools/test.py` 检查）。
3. `tools/analysis_tools/visualize/run.py` 依赖 `bbox_results` 中的 `planning_traj` / `command` 字段，
   若用旧版本 `test.py` 生成的 pkl 会因为字段缺失而报错，需用最新版 `test.py` 重新生成 pkl。
4. `base_e2e_diffusion.py` 中大量参数是为**笔记本 8GB 显存**专门优化的（见文件内注释），
   在服务器满显存环境下用 full 数据集训练时，建议评估是否要放开这些限制
   （如恢复 `occ_head`、增大 `dit_depth`/`dit_heads`、开启 `loss_collision` 等）以获得更强效果，
   否则只是复现了"显存受限版"的 DiT，没有发挥完整模型能力。
5. 两个项目（原版 UniAD、DiT 版本）如果分别维护在不同目录/仓库，注意 `tools/test.py`、
   `tools/analysis_tools/visualize/run.py` 等公共脚本要保持同步（可用 `md5sum` 校验），
   否则会出现 pkl 结构不兼容导致的解析报错。