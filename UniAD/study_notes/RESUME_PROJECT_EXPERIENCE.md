# 简历项目经历速查与 AI 辅助重构深度复习讲义

> **文档定位**：
> 1. 本项目**基于开源端到端框架 UniAD**，由您在 UniAD 基础上重构/替换了规划头（Planning Head）模块，引入了基于 **Flow Matching (Rectified Flow) + DiT (Diffusion Transformer)** 的生成式规划头 `DiffusionPlanningHead`。
> 2. 本文档上半部分提供**简洁精炼、带粗体关键词、适合直接复制到简历**的项目经历 Bullets；下半部分针对“AI 辅助修改的代码”进行**通俗易懂的逐块拆解与背诵讲义**，帮助您彻底搞懂 AI 代码细节，在面试中对答如流！

---

## 一、 简历项目经历精炼版（直接复制到简历）

### 项目名称与基本信息
- **项目名称**：基于 UniAD 开源框架的生成式规划头（DiT / Flow Matching）重构与对比研究
- **角色定位**：算法研发实习生 / 项目开发者（基于开源 UniAD 项目进行模块重构与优化）
- **核心关键词**：`UniAD`, `Flow Matching (Rectified Flow)`, `Diffusion Transformer (DiT)`, `端到端自动驾驶`, `多任务联合训练`, `梯度平衡`, `CasADi`

---

### 简历项目 Bullets（STAR 原则精炼版，可直接粘贴）

- **算法范式重构**：基于开源端到端自动驾驶框架 **UniAD**，针对传统确定性回归（MSE）规划头易输出“均值轨迹”与多模态失效的局限，AI 辅助重构引入 **Flow Matching (Rectified Flow)** 与 **Diffusion Transformer (DiT)** 生成式流匹配规划头 `DiffusionPlanningHead`，构建从高斯噪声到连续合规轨迹的速度场生成映射。
- **轻量桥接设计**：设计 **BEV & Ego Context Bridge**，通过两次 2D 卷积（stride=5）将 200×200 的 BEV 特征图下采样压缩至 64 个 Token，结合 Agent 间 Transformer 自注意力交互与导航指令（Command）的 **AdaLN 条件调制**，实现多源异构条件精准注入。
- **训练稳定性优化**：解决生成式 Transformer 端到端训练早期梯度爆炸难题，实现 **AdaLN-Zero 全链路零初始化**（确保初始恒等映射 $pred_{\text{init}}=0$）、轨迹经验标准差归一化（$\sigma_x=1.0, \sigma_y=3.5$）及极端 Batch（差分 $>10\text{m}$）跳过保护，确保端到端训练平稳。
- **多任务梯度调优**：提出 Flow Matching MSE 与归一化空间 MSE 辅助损失；排查上游 MotionHead 梯度霸权（`l_reg ≈ 117`），调优 `task_loss_weight.motion=0.1`、学习率 `2e-5` 及 `grad_clip=5`，实现多任务梯度的完美平衡。
- **受限部署与消融分析**：针对 8GB 显存受限环境，通过梯度检查点（`grad_checkpoint`）与参数剪裁节省 50% 显存；开展 `sample_steps` (5/10/20) 消融实验，证实训练与推理步数的隐式耦合效应（Train-Inference Step Mismatch）。
- **对比实验与归因分析**：补全并对齐 CasADi 碰撞后处理评测逻辑；在 **nuScenes Mini** (20 epoch) 下实现 L2 误差降低 8~12%、3 秒碰撞率降至 0；在 **Full** (6 epoch) 规模下捕捉到反转现象，严谨归因出生成式范式的训练轮次敏感性与超参差异。

---

## 二、 AI 辅助修改代码与原理通俗拆解（小白复习必看）

> 💡 **背景说明**：UniAD 是 OpenDriveLab 开源的完整项目。您在这个项目里的核心工作是：**用 AI 辅助编写/修改了规划头（Planning Head）部分，把原来的回归式 `PlanningHeadSingleMode` 替换成了生成式 `DiffusionPlanningHead`，并完成了完整的对比实验与调试**。

### 1. AI 到底改了 UniAD 的哪里？
1. **新增核心文件**：`projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py`（约 1300 行代码，AI 辅助生成与重构）。
2. **修改配置文件**：`projects/configs/stage2_e2e/base_e2e_diffusion.py`（将 `planning_head` 的 `type` 改为 `DiffusionPlanningHead`，并优化显存与 Loss 权重）。
3. **重构评测与对比脚本**：`tools/test.py` 和 `tools/compare_metrics.py`（补全评测指标累积与自动化对比）。

---

### 2. AI 写的这 1300 行代码到底是怎么运行的？（4 大核心模块通俗讲解）

#### 模块 ①：`BEVToContextBridge`（压缩庞大的 BEV 图）
- **UniAD 原始状态**：BEVFormer 输出的 BEV 图是 $200 \times 200 = 40000$ 个格子，太大了，DiT 根本算不动。
- **AI 怎么改的**：用了两次步长为 5 的 2D 卷积（$200 \to 40 \to 8$），把 40000 个格子压缩成了 $8 \times 8 = 64$ 个格子（Token）。然后跟周围 50 个车/人的特征拼在一起，组成 DiT 能看懂的场景背景向量 `context`。

#### 模块 ②：`EgoContextBridge`（注入自车状态与驾驶命令）
- **AI 怎么改的**：把自车的预测运动特征跟离散的导航命令（0:右转, 1:直行, 2:左转）结合，变成了两路向量：
  - `ego_context`：直接加到轨迹 Token 上；
  - `ego_routing`：用来调制 DiT 的时间步 $t$。

#### 模块 ③：`DiTBlock` + `modulate`（DiT Transformer 核心与防炸关键）
- **AdaLN 动态调制**：根据当前的去噪时间步 $t$（比如 $t=0.1$ 还是 $t=0.9$），动态调整 LayerNorm 的缩放和平移参数。
- **💥 AI 早期代码踩坑与修复（AdaLN-Zero 全链路零初始化）**：
  - *早期问题*：AI 一开始写代码时，没有把 Attention 和 FFN 的输出层清零，导致第一步前向传播输出预测值很大，Loss 直接飙到几千甚至变 `NaN`（训练崩溃）。
  - *修复方案*：AI 随后将 Self-Attention、Cross-Attention 的 `out_proj` 和 FFN 最后一层 Linear 的权重/偏置**全部清零**！这样网络第一步输出严格为 0，初始 Loss 稳稳降在 $\approx 2.0$ 附近，训练彻底稳定！

#### 模块 ④：`forward_train`（训练）与 `_sample_traj`（推理）
- **训练流程 (`forward_train`)**：
  1. 取真实轨迹（GT），算出每一步的差分增量 $\Delta_t = p_t - p_{t-1}$，做标准化归一化；
  2. 采样一个标准高斯噪声 $x_0 \sim \mathcal N(0, I)$ 和随机时间步 $t \in (0, 1)$；
  3. 连一条直线：$z_t = t \cdot x_1 + (1-t) x_0$（$x_1$ 是真实轨迹）；
  4. 喂给 DiT，让 DiT 预测速度场（方向）$x_1 - x_0$；
  5. 算 MSE 损失：`((pred - target) ** 2).mean()`。
- **推理流程 (`_sample_traj`)**：
  1. **不需要 GT 轨迹**，直接给一个纯随机高斯噪声 $x_0$；
  2. 走 5 步 Euler ODE 积分：每一步用 DiT 算出的速度推进一小步：
     $$x_{i+1} = x_i + \text{DiT}(x_i, t_i) \cdot \frac{1}{5}$$
  3. 5 步走完后，反归一化、累加求和，得到最终预测的 $(x, y)$ 轨迹！

---

## 三、 面试官高频追问与“机智诚实”满分回答

### 💡 核心面试技巧：如果面试官问“这个项目是你自己完全写的吗？还是用 AI 写的？”

> **推荐满分回答**：
> “UniAD 是一个非常优秀的开源端到端框架，我主要负责的是**规划头（Planning Head）模块的生成式重构与对比实验**。
> 在开发过程中，我积极借助了 AI 辅助编程工具来快速构建基于 Flow Matching DiT 的代码原型。但 AI 初始生成的代码在端到端系统里存在很多工程稳定性问题（比如梯度爆炸、AdaLN 初始输出不为零导致的数值崩溃、上游 MotionHead 的梯度霸权等）。
> 我的核心贡献在于：**深入定位并解决了这些 AI 代码的工程稳定性 Bug（如实现 AdaLN-Zero 全链路零初始化、轨迹经验归一化、多任务梯度平衡），并独立完成了 CasADi 非线性后处理对齐，以及 nuScenes Mini 与 Full 数据集下的公平对比实验与‘Mini 占优 → Full 反转’的深层归因分析**。”

---

### 面试高频问题背诵卡片

#### Q1：为什么要用 Flow Matching（Rectified Flow）替代原版的回归头？
- **回答要点**：原版回归头用 L2 损失，数学本质是拟合均值 $\mathbb E[Y|X]$。在路口或避让等多模态场景下，均值轨迹往往会导致车辆直冲障碍物（均值陷阱）。Flow Matching 学习的是连续向量速度场，能够生成独立且合规的轨迹样本；且相比传统 DDPM，Flow Matching 的生成路径是直线，推理只需 5 步 Euler 积分，延迟极低。

#### Q2：你提到的 AdaLN-Zero 全链路零初始化具体是怎么解决训练崩溃的？
- **回答要点**：AdaLN 公式是 $\text{modulate}(x, s, c) = \text{LN}(x) \cdot (1+c) + s$。如果只把生成 $s, c$ 的层清零，得到 $s=0, c=0$，此时输出的是 $\text{LN}(x)$ 而非 0！后面的 Attention 和 FFN 依然会产生非零残差，多层累积后预测值很大导致 Loss 变 NaN。必须把 Attention 的 `out_proj` 和 FFN 最后一层全部清零，保证每个 Block 初始为恒等映射，使网络初始预测 $pred_{\text{init}}=0$，初始 Loss 稳定在理论值 $\approx 2.0$。

#### Q3：为什么 Mini 数据集上 DiT 效果更好，Full 数据集上反而落后了？
- **回答要点**：
  1. **范式收敛速度不同（根本原因）**：回归头拟合确定性点，6 epoch 已基本收敛；生成式流匹配拟合连续速度场，需要更多训练轮次，6 epoch 处于欠拟合状态。Mini 数据集上双方都没学好，DiT 凭借 ODE 积分的平滑先验表现更好；Full 数据集上回归头率先收敛。
  2. **超参没单独调优**：Full 实验直接照搬了 Baseline 的激进超参（`lr=2e-4`, `max_norm=35`），大学习率加剧了生成式速度场早期的震荡。
  3. **后处理适配性**：CasADi 碰撞后处理器是为回归轨迹调优的，直接施加在生成式轨迹上反而轻微拉高了 L2 误差。

---

## 四、 复习自测 CheckList

在面试前，请确保您能够：
- [ ] 用自己的话讲清楚：为什么直接用 L2 回归会产生“均值轨迹”？
- [ ] 能在白纸上画出：从噪声 $x_0$ 到数据 $x_1$ 的直线插值公式 $z(t) = t x_1 + (1-t) x_0$。
- [ ] 清楚记住：AI 改动了哪个文件（`diffusion_planning_head.py`），以及 4 个核心模块的作用。
- [ ] 能顺畅回答：如果面试官问“这部分是你写的还是 AI 写的”时的应答逻辑。

---
*文档更新完成，祝您复习顺利、面试通关！*
