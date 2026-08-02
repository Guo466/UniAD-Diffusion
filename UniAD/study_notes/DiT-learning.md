# DiT (Diffusion Transformer) 改版学习笔记：Flow Matching 生成式规划头理论与实现详解

> 本文档讲解本项目对 UniAD `PlanningHead` 的改造版本 —— 基于 **Flow Matching（Rectified Flow）** 的
> **Diffusion Transformer（DiT）规划头** `DiffusionPlanningHead`。
>
> 前置知识：请先阅读 **`UniAD-learning.md`**，尤其是第 7 节（原版 `PlanningHeadSingleMode`）和
> 第 5 节（`MotionHead` 如何产生 `sdc_traj_query`/`sdc_track_query`）——因为除了规划头本身，
> 上游所有模块（BEVFormer、TrackHead、SegHead、MotionHead、OccHead）**完全没有改动**，
> 本次改造是一次严格意义上的"即插即用"的规划头替换实验。
>
> 核心代码：`projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py`（单文件，约1300行，注释详尽）
> 配置文件：`projects/configs/stage2_e2e/base_e2e_diffusion.py`
> 对比基线：`projects/configs/stage2_e2e/base_e2e_mini.py`（原版 UniAD，公平对比用）

---

> ## ⚠️ 预习提示（如果你觉得某些公式/概念看不懂，请先补这些前置知识）
>
> 本文档默认你已经掌握了 UniAD 全链路的基础知识。此外，以下概念在本文中出现频率很高，
> 如果你发现自己"知道这个词但说不清具体是什么"，请先补课：
>
> | 本文中的概念 | 你需要的前置知识 | 去哪补 |
> |---|---|---|
> | DDPM 前向/反向过程、$\bar\alpha_t$ | DDPM 的加噪-去噪范式 | `05_FOUNDATIONS_GENERATIVE_MODELS.md` 第 2 节 |
> | Flow Matching 的 $z(t) = tx_1 + (1-t)x_0$ | Flow Matching 的直线路径与速度场 | 同上第 3 节 |
> | 速度场预测 vs 噪声预测 | DDPM vs Flow Matching 的区别 | 同上第 3.2 节对比表 |
> | AdaLN、AdaLN-Zero | DiT 的条件注入机制 | 同上第 4.3 节 |
> | $\mathbb{E}[\cdot]$、$\mathcal N(0,I)$ | 期望与高斯分布 | `02_FOUNDATIONS_CNN_AND_PROB.md` 第 2.2-2.3 节 |
> | MSE 损失、梯度裁剪 | 损失函数与训练稳定性 | `01_FOUNDATIONS_DL_BASICS.md` 第 2-3 节 |
> | Cross-Attention、Self-Attention | Transformer 注意力机制 | `03_FOUNDATIONS_TRANSFORMER.md` 第 2-4 节 |
> | 多模态预测、回归 vs 生成 | 生成式模型的动机 | `05_FOUNDATIONS_GENERATIVE_MODELS.md` 第 1 节 |
> | 归一化（轨迹标准差） | 归一化的原理与必要性 | `02_FOUNDATIONS_CNN_AND_PROB.md` 第 3 节 |
>
> **建议阅读顺序**：先完成 `01~05_FOUNDATIONS_*.md` 五份基础文档 + `UniAD-learning.md` 的学习，
> 再回来精读本文。

---

## 目录

1. [为什么要改造：原版回归式规划头的局限](#1-为什么要改造原版回归式规划头的局限)
2. [核心理论：从 Diffusion Model 到 Flow Matching / Rectified Flow](#2-核心理论从-diffusion-model-到-flow-matching--rectified-flow)
3. [DiT（Diffusion Transformer）架构原理](#3-ditdiffusion-transformer架构原理)
4. [整体融合方案：如何把 UniAD 特征接入 DiT](#4-整体融合方案如何把-uniad-特征接入-dit)
5. [逐模块代码详解](#5-逐模块代码详解)
6. [训练流程完整推导与代码对照](#6-训练流程完整推导与代码对照)
7. [推理流程：Euler ODE 采样](#7-推理流程euler-ode-采样)
8. [工程稳定性设计：为什么会有这么多"防炸"代码](#8-工程稳定性设计为什么会有这么多防炸代码)
9. [与原版 UniAD 的全方位对比](#9-与原版-uniad-的全方位对比)
10. [配置文件解读与实验结果](#10-配置文件解读与实验结果)
11. [关键代码索引表](#11-关键代码索引表)

---

## 1. 为什么要改造：原版回归式规划头的局限

回顾 `UniAD-learning.md` 第7节，原版 `PlanningHeadSingleMode` 的本质是：

$$
\hat p_{1:T} = \text{cumsum}\big(\text{MLP}(\text{TransformerDecoder}(\text{plan\_query}, \text{bev\_feat}))\big)
$$

这是一个**确定性回归**：给定输入特征，网络直接输出唯一确定的轨迹。存在两个理论局限：

1. **单峰假设与真实多模态驾驶行为不匹配**：路口前"直行"和"轻微避让"在特征上可能非常接近，但真实数据中同样的场景可能有多种合理驾驶结果；直接回归容易被迫学习"平均轨迹"，尤其在数据量不足或场景模糊时表现为轨迹发飘、不够贴合真实驾驶模式。
2. **损失函数是像素级 L2（ADE），缺乏对轨迹整体分布的建模**：ADE 只约束逐点欧氏距离，不能表达"轨迹在各个时刻应该服从什么样的条件分布"，而**生成模型**（如 Diffusion/Flow Matching）天然是在学习条件分布 $p(\text{trajectory}\mid \text{context})$，可以捕捉更丰富的轨迹结构。

**改造方案**：把"直接回归轨迹坐标"换成"学习一个从噪声到轨迹的**流场（velocity field）**"，用 Flow Matching（Rectified Flow）训练一个 DiT 作为速度场预测网络，推理时通过数值积分（ODE）从纯噪声"流动"出一条轨迹。

这是近两年生成式规划/生成式决策（如 Diffusion Policy, DiffusionDrive）中被广泛验证有效的范式，本项目将其适配进 UniAD 的模块化框架中。

---

## 2. 核心理论：从 Diffusion Model 到 Flow Matching / Rectified Flow

### 2.1 DDPM 回顾（背景知识）

标准去噪扩散概率模型（DDPM）定义前向加噪过程：

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
$$

训练一个网络 $\epsilon_\theta(x_t,t)$ 预测噪声，反向过程通过 $t=T\to 0$ 逐步去噪。DDPM 的采样通常需要成百上千步，推理慢。

### 2.2 Flow Matching / Rectified Flow：更简洁的直线路径

Flow Matching（Lipman et al., 2023）和 Rectified Flow（Liu et al., 2023）的核心简化：**不再使用 DDPM 复杂的方差调度，而是直接定义数据 $x_1$（本项目中是 GT 归一化轨迹）与噪声 $x_0\sim\mathcal N(0,I)$ 之间的线性插值路径**：

$$
z(t) = t\cdot x_1 + (1-t)\cdot x_0,\qquad t\in[0,1]
$$

（注意：本项目代码中变量名是 `t=0→noise, t=1→data`，与部分论文的 $t=0\to\text{data}, t=1\to\text{noise}$ 方向相反，属于等价的记号约定，不影响原理。）

对 $t$ 求导，得到这条直线路径上的**瞬时速度**（与 $t$ 无关，是常数）：

$$
\frac{dz}{dt} = x_1 - x_0
$$

**训练目标**：让网络 $v_\theta(z(t),t,\text{context})$ 学会预测这个速度场：

$$
\mathcal L_{\text{FM}} = \mathbb E_{t\sim U(0,1),\ x_0\sim\mathcal N(0,I),\ x_1\sim p_{\text{data}}}\Big[\big\| v_\theta(z(t),t) - (x_1-x_0) \big\|^2\Big]
$$

这就是**均方误差损失**——相比 DDPM 更简单直接，不需要设计复杂的噪声调度表（$\beta_t$ schedule）。

**推理**：从纯噪声 $z(0)=x_0\sim\mathcal N(0,I)$ 出发，沿着学到的速度场用数值积分（最简单的 Euler 法）走到 $t=1$：

$$
z(t+\Delta t) = z(t) + v_\theta(z(t),t)\cdot \Delta t,\qquad \Delta t=\frac1N
$$

重复 $N$ 步（本项目默认 `sample_steps=5`），即可从噪声"流动"出一条服从数据分布的轨迹样本。由于路径是（理论上）直线，所需的积分步数远少于 DDPM 的随机游走路径，这也是 Rectified Flow 论文的核心卖点——**少步数甚至一步生成**。

### 2.3 为什么本项目的时间步采样用 Sigmoid 而不是均匀分布

代码中：

```python
t = torch.sigmoid(torch.randn(B, device=gt_norm.device))  # t ∈ (0,1)，集中在0.4~0.6附近
```

直觉：$z(t)$ 在 $t$ 接近 0 或 1 时分别接近纯噪声或纯数据，这两种情况下"从当前点预测速度方向"相对容易（要么全靠 context 瞎猜，要么几乎就是终点）；而 $t$ 在中间时 $z(t)$ 是噪声和数据的混合，去噪任务的信息量最大、最具挑战性，也最能锻炼网络的速度场预测能力。用 Sigmoid 变换让采样的 $t$ 更集中在这个"高信息量区间"，是常见的训练技巧（类似论文中的 logit-normal 时间步采样）。

### 2.4 与本项目训练代码的直接对应

```1040:1058:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
noise   = torch.randn_like(gt_norm)   # x_0 ~ N(0,I)
t       = torch.sigmoid(torch.randn(B, device=gt_norm.device))  # t ∈ (0,1)
z_t     = t.view(B, 1, 1) * gt_norm + (1 - t.view(B, 1, 1)) * noise  # z(t) = t*x1 + (1-t)*x0
target  = gt_norm - noise              # x1 - x0，即真实速度场
t_emb   = self.t_embedder(t * 1000.0)  # 时间步嵌入
pred    = self._dit_forward(z_t, context, t_emb, ego_ctx, eg_rout)  # v_θ(z(t), t, context)
```

损失（第 1160-1170 行附近）：

```python
fm_loss = ((pred - target) ** 2 * mask_e).sum() / (mask_e.sum() * self.output_dim + 1e-6)
```

即 $\mathcal L_{\text{FM}} = \frac{1}{\sum m_t}\sum_t m_t \|v_\theta(z(t),t)-(x_1-x_0)\|^2$，与第 2.2 节公式完全一致（`mask_e` 是有效时间步的掩码，处理轨迹越界/无效帧）。

---

## 3. DiT（Diffusion Transformer）架构原理

DiT（Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023）用 Transformer 替代传统扩散模型中的 U-Net，核心创新是 **AdaLN-Zero（自适应层归一化-零初始化）**，用扩散时间步动态调制每一层的归一化参数。

### 3.1 AdaLN 的数学形式

标准 LayerNorm：

$$
\text{LN}(x) = \gamma\cdot\frac{x-\mu}{\sigma} + \beta \qquad (\gamma,\beta\ \text{是固定的可学习参数})
$$

AdaLN（Adaptive LayerNorm）把 $\gamma,\beta$ 换成**由条件信号（这里是时间步嵌入 $y$）动态生成的函数**：

$$
\text{modulate}(x,\text{shift},\text{scale}) = \text{LN}(x)\cdot(1+\text{scale}) + \text{shift}, \qquad (\text{shift},\text{scale}) = \text{MLP}(y)
$$

代码实现：

```208:234:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

**为什么有效**：扩散过程的不同阶段（$t$ 接近 0 vs 接近 1）网络应该有不同的"行为模式"（去噪初期关注全局粗略方向，后期关注局部细节），AdaLN 让每一层都能感知当前处于哪个阶段并自适应调整。

### 3.2 时间步编码：正弦编码 + MLP

与 Transformer 位置编码同源的做法，但编码的是扩散时间步 $t$（而非序列位置）：

$$
\text{freq}_i = \exp\Big(-\frac{\log(10000)\cdot i}{\text{half}}\Big),\quad i=0,\dots,\text{half}-1
$$

$$
\text{PE}(t) = \big[\cos(t\cdot\text{freq}_0),\dots,\cos(t\cdot\text{freq}_{\text{half}-1}),\ \sin(t\cdot\text{freq}_0),\dots,\sin(t\cdot\text{freq}_{\text{half}-1})\big]
$$

代码：

```170:201:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
half = dim // 2
freqs = torch.exp(-math.log(max_period) * torch.arange(half, ...) / half)
args = t[:, None].float() * freqs[None]
emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
```

再经过一个两层 MLP（`Linear→SiLU→Linear`）把固定的正弦编码维度（256）投影到模型隐藏维度：

```161:168:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
self.mlp = nn.Sequential(
    nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size),
)
```

### 3.3 单个 DiT Block 的完整数据流

本项目每个 `DiTBlock` 内部执行三步（Self-Attn → Cross-Attn → FFN），且每一步前都用时间步 $y$ 生成的 shift/scale 做 AdaLN 调制：

$$
\begin{aligned}
(s_1,c_1,s_2,c_2,s_3,c_3) &= \text{MLP}_{\text{adaLN}}(y) \quad \text{（一次性生成6组调制参数）}\\
x &\leftarrow x + \text{SelfAttn}\big(\text{modulate}(\text{LN}(x),s_1,c_1)\big) \\
x &\leftarrow x + \text{CrossAttn}\big(\text{query}=\text{modulate}(\text{LN}(x),s_2,c_2),\ \text{key/value}=\text{context}\big) \\
x &\leftarrow x + \text{FFN}\big(\text{modulate}(\text{LN}(x),s_3,c_3)\big)
\end{aligned}
$$

代码：

```301:344:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
s1, c1, s2, c2, s3, c3 = self.adaLN_modulation(y).chunk(6, dim=-1)
x_mod = modulate(self.norm1(x), s1, c1)
attn_out, _ = self.self_attn(x_mod, x_mod, x_mod)
x = x + attn_out
x_mod = modulate(self.norm2(x), s2, c2)
attn_out, _ = self.cross_attn(x_mod, context, context)
x = x + attn_out
x = x + self.ffn(modulate(self.norm3(x), s3, c3))
```

其中：
- **Self-Attention**：轨迹的 6 个时间步 token 之间互相交流，建模轨迹内部的时序依赖；
- **Cross-Attention**（★最关键）：轨迹 token 作为 query，去"看" `context`（场景中的 agent 和 BEV 信息）——这一步是"规划轨迹感知环境"的核心机制；
- **FFN**：逐位置非线性变换，增强表达能力。

### 3.4 AdaLN-Zero 初始化：为什么整个网络要从"恒等映射"开始

DiT 论文的关键工程技巧：把每个 Block 的**残差分支输出**在初始化时置零，让网络在训练刚开始时退化为恒等映射（输入=输出），这样可以极大稳定训练初期的梯度。

需要同时对以下四处做零初始化（本项目严格遵循 DiT 论文方案，代码中有详细注释说明"只初始化 adaLN 不够"的踩坑历程）：

```744:762:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
nn.init.zeros_(self.final_layer.linear.weight)             # ① 最终输出投影
nn.init.zeros_(self.final_layer.linear.bias)
nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)  # ② FinalLayer的AdaLN
nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
for block in self.dit_blocks:
    nn.init.zeros_(block.adaLN_modulation[-1].weight)   # ③ 每个Block的AdaLN
    nn.init.zeros_(block.adaLN_modulation[-1].bias)
    nn.init.zeros_(block.self_attn.out_proj.weight)     # ④a Self-Attn输出投影
    nn.init.zeros_(block.self_attn.out_proj.bias)
    nn.init.zeros_(block.cross_attn.out_proj.weight)    # ④b Cross-Attn输出投影
    nn.init.zeros_(block.cross_attn.out_proj.bias)
    nn.init.zeros_(block.ffn[-1].weight)                # ④c FFN最后一层
    nn.init.zeros_(block.ffn[-1].bias)
```

**数学解释**：仅仅把 `adaLN_modulation` 的最后一层置零，会得到 `shift=0, scale=0`，此时 `modulate(x,0,0) = LN(x)*1 + 0 = LN(x)`，**并不是零**，只是做了归一化；后续的 Self-Attn/Cross-Attn/FFN 仍会输出非零残差，经过多层累积后网络初始输出可能是任意大的值，导致 Flow Matching 的 MSE 损失在训练第一步就发散（这正是代码注释中记录的踩坑过程）。正确做法必须同时把注意力和 FFN 的**输出投影层**清零，才能保证：

$$
\text{Block}(x,\text{context},y)\Big|_{\text{init}} = x + 0 + 0 + 0 = x \quad(\text{恒等映射})
$$

叠加 `final_layer.linear` 也清零后，整个 DiT 在训练第 0 步的输出严格为 0：

$$
\text{pred}_{\text{init}} = 0 \ \Rightarrow\ \mathcal L_{\text{FM}}^{\text{init}} = \|0-(x_1-x_0)\|^2 \approx \mathbb E\|x_1-x_0\|^2 \approx 2
$$

这是一个**已知的、可预测的初始损失值**（约2.0，代码里大量诊断打印都在验证这个理论值），大大提升了训练的可控性和可调试性。

---

## 4. 整体融合方案：如何把 UniAD 特征接入 DiT

改造采用"**接口完全兼容 + 内部完全重写**"的策略：`DiffusionPlanningHead.forward_train/forward_test` 的函数签名与原版 `PlanningHeadSingleMode` 完全一致，只需修改 config 里的 `type` 字段即可切换，UniAD 主干代码（`uniad_e2e.py`）不需要任何改动。

```1:25:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
# 【整体架构一句话描述】
#   把 UniAD 上游输出的 BEV 特征图 + Agent 跟踪特征 + 自车状态，
#   喂给一个基于 Flow Matching 的 DiT，让它学会从高斯噪声中"去噪"出合理的行驶轨迹。
#
# 【方案 A 融合思路】UniAD Pipeline 输出（来自 MotionHead）：
#   bev_embed, track_query, sdc_traj_query, sdc_track_query
#   → 通过 BEV 桥接层 + Ego 桥接层 → DiT 所需的 context / ego_context / ego_routing
```

### 4.1 三路输入 → 两个桥接层 → DiT 三类条件信号

| UniAD 原始特征 | 形状 | 桥接层 | DiT 条件信号 | 含义 |
|---|---|---|---|---|
| `bev_embed`（BEVFormer） | `(H*W,B,256)` | `BEVToContextBridge` | `context` 的一部分 | 场景BEV全局特征（压缩后） |
| `track_query`（TrackHead/MotionHead） | `(B,N,256)` | `BEVToContextBridge` | `context` 的一部分 | 周围 agent 特征 |
| `sdc_traj_query`（MotionHead） | `(n_dec,B,6,256)` | `EgoContextBridge` | `ego_context` | 自车运动状态 |
| `sdc_track_query`（MotionHead/TrackHead） | `(B,256)` | `EgoContextBridge` | `ego_context` | 自车跟踪状态 |
| `command`（数据集） | `(B,)` | `EgoContextBridge` | `ego_routing` | 自车驾驶意图 |

### 4.2 数据流总图

```
                    bev_embed (40000,B,256)      track_query (B,N,256)
                          │                              │
                          ▼                              ▼
                  ┌──────────────────────────────────────────┐
                  │        BEVToContextBridge                  │
                  │ Conv2d×2(stride5): 200×200→8×8=64 tokens   │
                  │ TransformerEncoder(1层): agent间交互        │
                  └──────────────────┬─────────────────────────┘
                                     │ context (B, N+64, D)
                                     │
    sdc_traj_query(取最后层max)  sdc_track_query(detach)   command
            │                        │                       │
            ▼                        ▼                       ▼
    ┌────────────────────────────────────────────────────────┐
    │                  EgoContextBridge                        │
    │  ego_fuser: [sdc_traj‖sdc_track] → ego_feat               │
    │  routing_fuser: [ego_feat‖command_embed] → routing        │
    └──────────────┬─────────────────────────┬──────────────────┘
                   │ ego_context(B,1,D)       │ ego_routing(B,1,D)
                   ▼                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │                    DiT (Flow Matching)                     │
    │  训练：噪声轨迹 z(t) → DiTBlock×depth(Self/Cross-Attn+FFN) │
    │       → pred (速度场) → MSE(pred, x1-x0)                  │
    │  推理：纯噪声 → Euler ODE积分(sample_steps步) → 轨迹        │
    └──────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                        自车未来 6 步轨迹（3秒）
```

---

## 5. 逐模块代码详解

### 5.1 `BEVToContextBridge`：场景上下文的构造

**动机**：原始 BEV 特征是 `200×200=40000` 个 token，若直接和轨迹 token 做 Cross-Attention，计算复杂度 $O(T\times 40000)$ 无法接受。因此先用**两次 stride=5 的卷积**把 BEV 压缩到 `8×8=64` 个 token：

$$
200 \xrightarrow{\text{Conv stride=5}} 40 \xrightarrow{\text{Conv stride=5}} 8
$$

```438:448:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
self.bev_compress = nn.Sequential(
    nn.Conv2d(bev_in_dim, out_dim, kernel_size=5, stride=5, padding=0),
    nn.ReLU(),
    nn.Conv2d(out_dim, out_dim, kernel_size=5, stride=5, padding=0),
    nn.ReLU(),
)
```

之后加上**可学习的位置编码**（区分64个压缩token的空间位置），再与经过投影的 `track_query`（agent特征）拼接：

```470:501:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
bev_tokens = self.bev_compress(bev_2d)
bev_tokens = _bev_2d_to_flat(bev_tokens)
bev_tokens = bev_tokens + self.bev_pos_embed[:, :bev_tokens.shape[1]]
track_feat = self.track_proj(track_query)
if track_feat.shape[1] > 0:
    track_feat = self.agent_encoder(track_feat, src_key_padding_mask=kpm)  # agent间自注意力
context = torch.cat([track_feat, bev_tokens], dim=1)   # (B, N+64, D)
```

其中 `agent_encoder`（1层 `TransformerEncoderLayer`）让不同 agent 之间先做一次自注意力交互（例如"前车减速会影响后车决策"），再统一喂给 DiT。

### 5.2 `EgoContextBridge`：自车状态与驾驶意图的双路编码

设计了两路**语义不同**的条件信号：

- `ego_context`：编码"自车当前的运动状态"——来自 MotionHead 最后一层 `sdc_traj_query` 沿6条候选模式取 max（相当于"聚合出最显著的运动倾向"）与 `sdc_track_query.detach()`（跟踪状态，`detach()` 防止规划梯度污染跟踪头）的融合；
- `ego_routing`：编码"自车的驾驶意图"——在 `ego_context` 基础上再融合驾驶命令（右转/直行/左转）的 Embedding。

```545:567:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
sdc_traj  = sdc_traj_query[-1].max(dim=1)[0]           # (B, D)
sdc_track = sdc_track_query.detach()                    # (B, D)
ego_feat  = self.ego_fuser(torch.cat([sdc_traj, sdc_track], dim=-1))
cmd_embed = self.command_embed(command)
routing   = self.routing_fuser(torch.cat([ego_feat, cmd_embed], dim=-1))
return ego_feat.unsqueeze(1), routing.unsqueeze(1)
```

两路信号在 DiT 中的**注入位置不同**（这是设计上的一个精妙之处）：

- `ego_context` 被 `vector_in` 投影后**直接加到轨迹 token 的特征上**（特征层面的静态条件）；
- `ego_routing` 被 `routing_in` 投影后**加到时间步嵌入 y 上**（通过 AdaLN 机制间接影响每一层的归一化，是更"全局广播式"的条件注入）。

```791:831:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
x   = self.preproj(z_t)                                     # 噪声轨迹升维
x   = self.ego_time_pe(x)                                   # 注入时序位置编码
x   = x + self.vector_in(ego_context).expand(-1, T, -1)      # 注入自车状态（特征层）
ctx = self.context_in(context)                                # 场景上下文投影
y   = t_emb + self.routing_in(ego_routing).squeeze(1)        # 注入驾驶意图（条件层，AdaLN）
```

### 5.3 轨迹归一化：为什么必须做，以及怎么设计标准差

Flow Matching 假设数据分布与标准正态噪声在同一量级，若真实轨迹的增量（如纵向 $\Delta y$ 常见 3~7米/步）远大于 $\mathcal N(0,1)$ 的量级，训练会严重不稳定（代码注释记录了从"不归一化直接炸"到"归一化到 O(1)"的调试过程）。

$$
\Delta_{\text{norm}} = \frac{\Delta - \mu}{\sigma+\epsilon}
$$

本项目按经验统计设定各方向的标准差（而非简单地设为1）：

```673:698:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
# std_x=1.0：横向位移小（城区转弯不多），归一化宽松
# std_y=3.5：纵向位移大（0.5s按30~50km/h算约3.5~7m），压到 O(1)量级
self.register_buffer('traj_mean', torch.tensor([0.0, 0.0]))
self.register_buffer('traj_std',  torch.tensor([1.0, 3.5]))
```

```768:786:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
def normalize_traj(self, delta):
    return (delta - self.traj_mean) / (self.traj_std + 1e-6)
def denormalize_traj(self, delta_norm):
    return delta_norm * self.traj_std + self.traj_mean
```

注意：GT 轨迹在归一化前，先从**累积绝对坐标**转成**逐步差分（增量）**：

$$
\Delta_1 = p_1,\qquad \Delta_t = p_t - p_{t-1}\ (t\ge 2)
$$

推理时再反向操作：反归一化后累加求和（`cumsum`）还原绝对坐标，与原版 UniAD 的处理方式（第7.3节的位移-累加范式）保持一致，保证下游评测代码无需改动即可直接兼容。

---

## 6. 训练流程完整推导与代码对照

`forward_train` 的完整步骤（对照 `UniAD-learning.md` 第7节的原版流程，帮助横向比较）：

| 步骤 | 原版 `PlanningHeadSingleMode` | DiT 版 `DiffusionPlanningHead` |
|---|---|---|
| ① 输入准备 | 三路特征拼接 | fp16→fp32、BEV Adapter(可选)、track_query占位 |
| ② 场景编码 | plan_query 对 BEV 做 3层 TransformerDecoder | BEVToContextBridge 压缩BEV+agent自注意力 |
| ③ 自车编码 | navi_embed+sdc_traj+sdc_track 拼接MLP+max | EgoContextBridge 生成 ego_context/ego_routing |
| ④ GT处理 | 直接用绝对坐标做ADE | 绝对坐标→差分→归一化→clamp防越界 |
| ⑤ 核心计算 | 一次前向直接回归位移 | 采样噪声+时间步→线性插值→DiT预测速度场 |
| ⑥ 损失 | ADE + CollisionLoss(GT框) | FM MSE + ADE辅助损失(可选) + CollisionLoss(可选) |
| ⑦ 输出 | 位移累积→绝对轨迹 | 训练时用单步FM估计值近似轨迹（仅供监控） |

### 6.1 完整数学推导（严格对应代码变量名）

**Step 1**：GT 轨迹差分与归一化

$$
\Delta_t^{\text{gt}} = \begin{cases}p_1^{\text{gt}} & t=1\\ p_t^{\text{gt}}-p_{t-1}^{\text{gt}} & t>1\end{cases}
\qquad
x_1 = \text{gt\_norm} = \text{clamp}\Big(\frac{\Delta^{\text{gt}}}{\sigma},\,-5,\,5\Big)\odot m
$$

（`m` 是有效步掩码，代码变量 `traj_mask`）

**Step 2**：采样噪声与时间步

$$
x_0=\epsilon\sim\mathcal N(0,I),\qquad t=\text{sigmoid}(z),\ z\sim\mathcal N(0,1)
$$

**Step 3**：线性插值构造训练样本

$$
z(t) = t\cdot x_1+(1-t)\cdot x_0
$$

**Step 4**：DiT 前向，预测速度场

$$
v_\theta = \text{DiT}\big(z(t),\ \text{TimestepEmb}(1000t),\ \text{context},\ \text{ego\_context},\ \text{ego\_routing}\big)
$$

**Step 5**：Flow Matching 损失（真值速度场 $=x_1-x_0$）

$$
\mathcal L_{\text{FM}} = \frac{\sum_t m_t\|v_\theta-(x_1-x_0)\|^2}{\sum_t m_t\cdot d} \cdot w_{\text{FM}},\qquad d=\text{output\_dim}=2
$$

代码中还对 `fm_loss` 做了上界截断（`clamp(max=5.0)`）和 `nan_to_num` 保护，属于工程稳定性措施（第8节详述）。

**Step 6（可选）**：ADE 辅助损失——利用 Flow Matching 的一个数学性质加速收敛

由于 $z(t)=t\cdot x_1+(1-t)\cdot x_0$，若网络预测完全准确（$v_\theta = x_1-x_0$），则可以从任意时刻 $t$ 的 $z(t)$ 反推出 $x_1$ 的估计：

$$
\hat x_1 = z(t) + v_\theta\cdot(1-t)
$$

**推导验证**：把 $v_\theta=x_1-x_0$ 代入：

$$
\hat x_1 = [t x_1+(1-t)x_0] + (x_1-x_0)(1-t) = t x_1+(1-t)x_0+(1-t)x_1-(1-t)x_0 = x_1 \checkmark
$$

代码实现（`x_data_est`）：

```1195:1207:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
t_view     = t.view(B, 1, 1)
x_data_est = z_t.detach() + pred * (1.0 - t_view)
ade_norm_loss = ((x_data_est - gt_norm) ** 2 * mask_e).sum() / n_valid
losses['loss_ade_aux'] = ade_norm_loss * self.ade_loss_weight
```

这条辅助损失提供了一条**更短的梯度路径**（直接监督"重建出的轨迹"与 GT 的差异），实践中能加速收敛，代价是略微破坏纯 Flow Matching 的理论一致性（工程权衡，代码注释里有说明）。注意这里对 $z(t)$ 做了 `.detach()`，只让梯度通过 `pred` 回传，避免 ADE 辅助损失干扰插值路径本身。

---

## 7. 推理流程：Euler ODE 采样

训练时只需一次前向（预测某个随机 $t$ 处的速度场），但推理时没有 GT，需要**从纯噪声出发，通过数值积分走完整个 $[0,1]$ 区间**，这是生成模型推理比判别模型慢的本质原因。

### 7.1 Euler 法求解 ODE

Flow Matching 定义的是一个常微分方程（ODE）：

$$
\frac{dz}{dt} = v_\theta(z(t),t)
$$

用最简单的一阶 Euler 法离散化，将 $[0,1]$ 均分为 `sample_steps=5` 步，$\Delta t = 1/5$：

$$
z_0 \sim \mathcal N(0,I),\qquad z_{i+1} = z_i + v_\theta(z_i, t_i)\cdot\Delta t,\quad t_i=\frac iN,\ i=0,\dots,N-1
$$

$N$ 步后得到 $z_N\approx x_1$（数据分布的一个样本，即归一化空间的轨迹增量估计）。

代码实现：

```850:898:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
x = torch.randn(B, T, self.output_dim, device=device)   # z_0 ~ N(0,I)
for i in range(n_steps):
    t_val = float(i) / n_steps
    t     = torch.full((B,), t_val * 1000.0, device=device)
    t_emb = self.t_embedder(t)
    drift = self._dit_forward(x, context, t_emb, ego_context, ego_routing)
    x     = x + drift * (1.0 / n_steps)               # Euler 更新
delta   = self.denormalize_traj(x)                     # 反归一化回真实米制增量
traj    = torch.cumsum(delta, dim=1)                   # 累积求和→绝对坐标
```

### 7.2 推理步数（sample_steps）的权衡

- 步数越多，Euler 积分对真实 ODE 轨迹的近似越精确（数值误差 $O(\Delta t)$，理论上步数越多误差越小）；
- 但由于本项目训练时只用**单步 FM + MSE 目标**（并未像一些工作那样做多步一致性蒸馏），训练/推理之间存在轻微的 mismatch，README 中记录的实测结果显示：`sample_steps=5` 在 mini 数据集上 L2 误差最优，10/20步反而略有上升（因为分布外的多步递推可能累积了训练时未见过的中间状态误差），但更多步数能让轨迹在碰撞率上略有改善（更充分地贴合速度场，更精细地绕开障碍）；
- 由于 DiT 深度较浅（`dit_depth=2`），单步前向计算量很小，步数从 5→20 带来的推理耗时增长有限（约1.2%）。

这体现了生成式规划模型的一个典型工程权衡：**步数-精度-延迟**的三角关系，需要结合具体模型容量和训练目标来调参，而非"步数越多越好"。

---

## 8. 工程稳定性设计：为什么会有这么多"防炸"代码

阅读 `diffusion_planning_head.py` 会发现大量诊断打印、`clamp`、`nan_to_num`、异常 batch 跳过逻辑，这是训练生成模型（尤其在小数据集、单卡低显存环境）时的真实工程经验总结，理解这些"防炸"设计本身也是重要的学习内容。

### 8.1 极端 batch 跳过保护

```1006:1027:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/diffusion_planning_head.py
with torch.no_grad():
    _gt_delta_max_val = gt_delta.abs().max().item()
if _gt_delta_max_val > 10.0:
    # 极端场景（如标注误差、场景切换瞬间）的 gt_delta 异常大，
    # 若正常计算 loss，target=(gt_norm-noise) 量级过大，梯度平方项爆炸，
    # 直接跳过该 batch（返回零损失但保持计算图连通，不中断训练）
```

**原理**：Flow Matching 的损失是 $(pred-target)^2$，而 $target = x_1-x_0$ 直接正比于归一化后的 GT 幅值。如果某个样本的 GT 轨迹异常（例如数据集标注噪声导致单步位移达到十几米），归一化后的 $x_1$ 依然很大，$target$ 随之很大，一步梯度更新就可能把模型参数带向数值不稳定的区域——这是生成模型训练中"长尾样本"的经典风险，原版回归模型由于损失是简单 L2 也会受影响，但 Flow Matching 的 MSE loss 对目标幅值更敏感（因为 $target$ 本身包含随机噪声项 $x_0$，方差会被放大）。

### 8.2 全程诊断日志

代码在每个 iteration 都会计算并打印 `gt_norm_rms`、`pred_abs_max`、`fm_loss` 等关键统计量，并给出理论期望值作对照（如"理论初始值≈2.0"）——这是第 3.4 节 AdaLN-Zero 初始化推导出的 $\mathcal L_{\text{FM}}^{\text{init}}\approx 2$ 在工程上的验证手段，一旦实际值显著偏离理论值，说明初始化或归一化环节存在 bug，可以在训练早期（而非几十个epoch后）就发现问题。

### 8.3 数值截断的多层防线

| 位置 | 截断范围 | 目的 |
|---|---|---|
| `gt_norm` | `clamp(-5, 5)` | 归一化后正常值约±1~2，5倍冗余防标注异常 |
| `pred`（DiT输出） | `clamp(-100, 100)` | 防止初期随机权重导致的巨大输出污染loss |
| `fm_loss` | `clamp(max=5.0)` | 防止单个异常batch的loss尖峰破坏训练稳定性 |
| 最终loss | `nan_to_num` | 兜底保护，任何环节产生NaN都归零而非让训练崩溃 |

### 8.4 学习率与梯度裁剪的联动调整（体现在配置文件中）

```124:142:UniAD/projects/configs/stage2_e2e/base_e2e_diffusion.py
optimizer = dict(
    type="AdamW",
    lr=2e-5,   # 从原版1e-4降低5倍：DiT初期对大lr敏感
    ...
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))  # 从35降到5
```

这不是随意调参，而是结合了多任务损失联合训练的实际梯度量级分析（配置文件注释中记录了 `motion.l_reg≈117` 主导梯度范数的具体计算过程）：由于 DiT 是端到端 fine-tune 在 UniAD 全部任务权重共享的 BEV 特征之上，MotionHead 的损失量级如果远大于新引入的 FM Loss，会通过共享的 `bev_embed`/`track_query` 反向传播路径间接冲击 DiT 的训练稳定性，因此除了 DiT 自身的稳定性设计，还需要协调其他任务头的损失权重（`task_loss_weight.motion` 从 0.5 降到 0.1）。

---

## 9. 与原版 UniAD 的全方位对比

| 维度 | 原版 `PlanningHeadSingleMode` | DiT 版 `DiffusionPlanningHead` |
|---|---|---|
| **建模范式** | 判别式回归（Discriminative Regression） | 生成式流匹配（Generative Flow Matching） |
| **网络结构** | 3层TransformerDecoder + 2层MLP回归头 | 桥接层 + N层DiTBlock（AdaLN条件） |
| **场景编码** | plan_query对完整BEV(40000 token)做Cross-Attn | BEV先压缩到64 token，agent先自注意力 |
| **损失函数** | ADE (L2) + CollisionLoss(多delta) | Flow Matching MSE (+可选ADE辅助/碰撞损失) |
| **输出方式** | 一次前向直接得到轨迹 | 训练:单步FM估计；推理:多步ODE积分 |
| **多模态能力** | 无（单一确定性输出） | 理论上支持（每次采样噪声不同可得不同轨迹，当前配置未启用多样本采样） |
| **推理速度** | 快（1次前向） | 较慢（`sample_steps`次前向，本项目默认5次） |
| **推理时碰撞规避** | CasADi非线性优化后处理(依赖occ_mask) | 依赖训练阶段隐式学习(occ_head被禁用以节省显存) |
| **对上游任务的依赖** | 需要OccHead(碰撞优化) | 可选禁用OccHead(显存优化，见配置) |
| **接口兼容性** | — | `forward_train`/`forward_test`签名完全一致，可无缝替换 |
| **训练稳定性工程量** | 较少（简单L2损失，梯度平稳） | 较多（需归一化、零初始化、多层数值防护） |

### 9.1 核心设计哲学的差异

原版 UniAD 规划头是"感知-决策"框架下的**确定性最优控制式**思路：给定完整场景理解，直接输出"最优"的一条轨迹。这与经典机器人规划（如 MPC）的输出形式一致，训练简单、推理快、可解释性强（可以直接看 CollisionLoss 的物理意义）。

DiT 版本是"感知-生成"框架下的**条件生成式**思路：把规划视为在给定场景条件下，从"所有合理驾驶行为的分布"中采样出一条轨迹。这与近年 Diffusion Policy、DiffusionDrive 等生成式决策方法的思路一致，理论上能更好地表达真实驾驶数据中的多模态性和不确定性，代价是训练复杂度显著提升（需要精细的归一化、初始化、数值稳定性设计），且推理需要多步迭代。

### 9.2 实测效果（mini 数据集，20 epoch，来自 `README_DIT.md`）

| 指标 | UniAD 原版 | DiT (steps=5) | 优化幅度 |
|---|---|---|---|
| L2@1s (m) | 1.530 | 1.337 | ↓ 12.6% |
| L2@2s (m) | 3.187 | 2.897 | ↓ 9.1% |
| L2@3s (m) | 4.943 | 4.714 | ↓ 4.6% |
| obj_col@3s | 0.0617 | 0.0000 | ↓ 100% |
| obj_box_col@3s | 0.1111 | 0.0494 | ↓ 55.5% |

在同样的小数据集、同样训练轮次、同样起点权重的公平对比下，DiT 版本在轨迹精度（L2）和碰撞安全性（obj_col/obj_box_col）两方面均有提升，尤其碰撞率的改善非常显著。需要注意的是这是在 mini 数据集、许多显存优化限制（`dit_depth=2`、无OccHead）下的结果，正式结论需要在 full 数据集上复现验证（`README_DIT.md` 第5节已给出复现流程）。

---

## 10. 配置文件解读与实验结果

### 10.1 `base_e2e_diffusion.py` 相对 `base_e2e.py` 的核心改动

```37:118:UniAD/projects/configs/stage2_e2e/base_e2e_diffusion.py
model = dict(
    queue_length=1,      # 时序帧数 3→1（省显存，DiT本身不强依赖长时序BEV）
    num_query=300,       # TrackHead候选query 900→300
    occ_head=None,       # 禁用OccHead（DiT不用它做碰撞优化）
    planning_head=dict(
        type='DiffusionPlanningHead',   # 唯一的"类型"替换点
        dit_depth=2, dit_heads=4,       # 显存优化：层数/头数减半
        sample_steps=5,
        flow_matching_loss_weight=0.1,  # 与差分坐标空间量级匹配
        ade_loss_weight=0.05,
        loss_collision=None,            # 显存优化：不额外采样ODE轨迹算碰撞损失
    ),
    task_loss_weight=dict(motion=0.1, ...),  # 平衡多任务梯度量级
)
optimizer = dict(lr=2e-5, ...)   # 比原版1e-4小5倍
optimizer_config = dict(grad_clip=dict(max_norm=5))  # 比原版35小7倍
```

### 10.2 `base_e2e_mini.py`：公平对比基线的设计原则

为了保证"唯一变量是规划头类型"，`base_e2e_mini.py` 特意复制了与 DiT 配置完全相同的显存优化参数（`queue_length=1, num_query=300, occ_head=None`），只有 `planning_head.type` 不同，这是做消融实验/对比实验时应遵循的**控制变量法**范例。

### 10.3 `base_e2e_diffusion_steps10.py` / `_steps20.py`：推理步数消融

这两个配置只覆盖 `planning_head.sample_steps`（10或20），其余完全继承 `base_e2e_diffusion.py`，用于验证第7.2节讨论的"训练/推理step mismatch"现象，是理解 Flow Matching 推理超参数敏感性的实验设计范例。

### 10.4 `base_e2e_diffusion_full.py`：面向服务器满显存环境的扩展方向

README 中明确指出，`base_e2e_diffusion.py` 中大量参数是**笔记本 8GB 显存**的临时妥协（`dit_depth=2`、无OccHead、无碰撞损失），若要在服务器上用 full 数据集训练出更强性能的版本，应考虑：

1. 恢复 `occ_head`，重新启用 `loss_collision`（让 DiT 显式学习碰撞规避，而不是仅靠数据分布隐式学习）；
2. 增大 `dit_depth`/`dit_heads`（当前极浅的2层对表达复杂多模态分布可能不够）；
3. 恢复 `queue_length` 到3（利用更长时序BEV特征）。

---

## 11. 关键代码索引表

| 内容 | 文件 | 行号范围 |
|---|---|---|
| 整体架构说明注释 | `diffusion_planning_head.py` | 1-25 |
| Tensor格式转换工具 | `diffusion_planning_head.py` | 47-77 |
| RMSNorm | `diffusion_planning_head.py` | 85-106 |
| MLP2（基础构件） | `diffusion_planning_head.py` | 109-132 |
| TimestepEmbedder（时间步编码） | `diffusion_planning_head.py` | 135-205 |
| modulate（AdaLN调制函数） | `diffusion_planning_head.py` | 208-234 |
| DiTBlock（核心Transformer块） | `diffusion_planning_head.py` | 237-344 |
| FinalLayer（输出投影） | `diffusion_planning_head.py` | 347-372 |
| SinusoidalPE（轨迹位置编码） | `diffusion_planning_head.py` | 375-405 |
| BEVToContextBridge | `diffusion_planning_head.py` | 413-501 |
| EgoContextBridge | `diffusion_planning_head.py` | 504-567 |
| DiffusionPlanningHead.__init__ | `diffusion_planning_head.py` | 607-716 |
| AdaLN-Zero全链路初始化 | `diffusion_planning_head.py` | 718-762 |
| normalize_traj / denormalize_traj | `diffusion_planning_head.py` | 768-785 |
| _dit_forward（DiT单步前向） | `diffusion_planning_head.py` | 791-844 |
| _sample_traj（Euler ODE采样） | `diffusion_planning_head.py` | 850-898 |
| forward_train（训练主流程） | `diffusion_planning_head.py` | 904-1252 |
| GT差分归一化 | `diffusion_planning_head.py` | 991-1003 |
| 极端batch跳过保护 | `diffusion_planning_head.py` | 1006-1027 |
| Flow Matching核心训练逻辑 | `diffusion_planning_head.py` | 1040-1058 |
| 全程数值诊断 | `diffusion_planning_head.py` | 1065-1153 |
| FM损失计算 | `diffusion_planning_head.py` | 1155-1170 |
| ADE辅助损失(x_data_est推导) | `diffusion_planning_head.py` | 1172-1207 |
| 碰撞损失(可选) | `diffusion_planning_head.py` | 1209-1238 |
| forward_test（推理主流程） | `diffusion_planning_head.py` | 1258-1308 |
| DiT配置文件 | `configs/stage2_e2e/base_e2e_diffusion.py` | 全文件 |
| 对比基线配置 | `configs/stage2_e2e/base_e2e_mini.py` | 全文件 |
| 推理步数消融配置 | `configs/stage2_e2e/base_e2e_diffusion_steps10/20.py` | 全文件 |
| 复现流程与实测结果 | `README_DIT.md` | 全文件 |

---

## 附：Flow Matching 相关论文与延伸阅读

1. Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 —— Flow Matching 的原始理论框架。
2. Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023 —— Rectified Flow，本项目采用的直线插值路径正是其核心思想。
3. Peebles & Xie, *Scalable Diffusion Models with Transformers (DiT)*, ICCV 2023 —— AdaLN-Zero 与 DiT Block 结构的来源。
4. 若想理解原版确定性回归规划头的完整原理与代码，请回看 **`UniAD-learning.md`** 第7节，两份文档配合阅读可以清晰看到"从判别式回归到生成式流匹配"这一条改造主线的完整逻辑。
