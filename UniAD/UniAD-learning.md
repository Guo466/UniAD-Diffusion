# UniAD 原版学习笔记：理论、公式推导与代码实现详解

> 本文档面向"想彻底搞懂 UniAD 每一行代码在做什么、为什么这么做"的读者。
> 论文：*Planning-oriented Autonomous Driving*（CVPR 2023 Best Paper Candidate）
> 代码：`UniAD/projects/mmdet3d_plugin/`
>
> 阅读建议：按顺序读完第 1~8 节，会形成"图像 → BEV → 检测跟踪 → 地图 → 运动预测 → 占用预测 → 规划"的完整闭环认知。每一节都会给出：**这一步在解决什么问题 → 数学原理 → 对应代码位置**。

---

## 目录

1. [整体设计哲学：Planning-oriented](#1-整体设计哲学planning-oriented)
2. [第零层：BEVFormer —— 把多相机图像变成鸟瞰图特征](#2-第零层bevformer--把多相机图像变成鸟瞰图特征)
3. [Stage1：TrackHead —— 端到端检测与多目标跟踪](#3-stage1trackhead--端到端检测与多目标跟踪)
4. [SegHead：在线地图分割（略讲）](#4-seghead在线地图分割略讲)
5. [MotionHead —— 多模态运动预测](#5-motionhead--多模态运动预测)
6. [OccHead —— 占用流预测](#6-occhead--占用流预测)
7. [PlanningHead —— 自车轨迹规划（原版回归式）](#7-planninghead--自车轨迹规划原版回归式)
8. [损失函数与评测指标全景](#8-损失函数与评测指标全景)
9. [训练/推理数据流总览图](#9-训练推理数据流总览图)
10. [关键代码索引表](#10-关键代码索引表)

---

## 1. 整体设计哲学：Planning-oriented

### 1.1 传统自动驾驶 pipeline 的问题

传统模块化自动驾驶系统：`检测 → 跟踪 → 预测 → 规划`，每个模块独立训练、独立优化自己的指标（检测看 mAP，预测看 minADE），但这些指标之间可能存在**目标不一致（misalignment）**：一个 mAP 很高的检测器，未必能给规划任务提供最有用的特征；一个 minADE 很低的预测器，也未必真的有利于最终"开得又快又安全"。

### 1.2 UniAD 的解法：单一 Transformer 网络端到端训练，但保留任务分解

UniAD 并没有把所有任务塞进一个黑盒，而是保留了任务分解（检测、跟踪、地图、运动、占用、规划各自成头），但是：

- **所有任务头共享同一个 BEV 特征**（`bev_embed`），检测跟踪产生的 `track_query` 被后续所有任务复用；
- **前序任务的 Query 输出直接作为后续任务的输入**（Query 携带的是学到的特征，而非人工规则的检测框坐标），实现了**特征级别的信息传递**，而非传统的"结果级别"传递；
- **所有任务的损失同时反向传播**，因此上游任务（检测/跟踪）会被下游任务（规划）的梯度"塑形"——学到的检测特征天然向着"更有利于规划"的方向偏移。

对应总控代码：

```314:325:UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(
                bev_embed,
                outs_motion,          # 包含 sdc_traj_query（自车轨迹查询）和 occ 相关信息
                sdc_planning,         # GT 自车轨迹（监督信号）
                sdc_planning_mask,    # GT 自车轨迹有效性
                command,              # 高层指令（直行/左转/右转）
                gt_future_boxes       # GT 未来目标框（碰撞检测用）
            )
            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)
```

整体流水线（严格串行依赖）：

```
多相机图像
   │
   ▼ (BEVFormer, 见第2节)
BEV 特征 bev_embed (200×200×256)
   │
   ▼ (TrackHead, 见第3节)
检测框 + 跟踪 Query track_query, sdc_embedding
   │
   ▼ (SegHead, 见第4节)
车道线/地图要素 lane_query
   │
   ▼ (MotionHead, 见第5节)
多模态轨迹预测 + sdc_traj_query, sdc_track_query
   │
   ▼ (OccHead, 见第6节)
未来占用网格 occ_mask
   │
   ▼ (PlanningHead, 见第7节)
自车未来轨迹 sdc_traj
```

每个箭头旁边就是本节要讲的模块。

---

## 2. 第零层：BEVFormer —— 把多相机图像变成鸟瞰图特征

虽然 BEVFormer 不是 UniAD 论文的创新点（它是前作），但它是**一切下游任务的地基**，必须理解。核心文件：

- `projects/mmdet3d_plugin/uniad/modules/encoder.py`（`BEVFormerEncoder`, `BEVFormerLayer`）
- `projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py`（空间交叉注意力，图像→BEV）
- `projects/mmdet3d_plugin/uniad/modules/temporal_self_attention.py`（时序自注意力，历史BEV→当前BEV）
- `projects/mmdet3d_plugin/uniad/modules/transformer.py`（顶层调度）

### 2.1 核心思想

BEV（Bird's-Eye-View，鸟瞰图）特征是一个 `200×200` 的网格，每个格子对应真实世界 `0.5m×0.5m` 的区域（感知范围 `pc_range=[-51.2,-51.2,-5.0, 51.2,51.2,3.0]` 米）。每个格子用一个可学习的 **BEV Query** 向量表示，初始值是随机初始化的 `nn.Embedding`：

```python
# bevformer_head.py 第217-218行附近
self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
```

BEVFormer 用 6 层 Encoder，让每个 BEV Query 通过注意力机制去"看"多相机图像的对应区域，把 2D 图像特征"提拉"到 3D/BEV 空间。每层做两件事：

1. **Temporal Self-Attention（TSA）**：融合上一帧的 BEV 特征（时序建模，类似 RNN 的隐状态传递）；
2. **Spatial Cross-Attention（SCA）**：融合当前帧 6 个相机的图像特征（空间建模，2D→3D 的核心）。

### 2.2 Deformable Attention 基础

标准注意力 $\text{Attn}(Q,K,V)=\text{softmax}(QK^T/\sqrt{d})V$ 的复杂度是 $O(N_q N_k)$，BEV 有 40000 个 Query，图像特征点也有几万个，直接算不现实。BEVFormer 使用 **Deformable Attention**：每个 Query 不看全部 Key，只在参考点附近采样少量点：

$$
\text{DeformAttn}(\mathbf q,\mathbf p) = \sum_{m=1}^{M} W_m \sum_{l=1}^{L}\sum_{k=1}^{K} A_{mlk}\cdot W_m' \cdot \phi\big(\mathbf f_l,\ \mathbf p + \Delta\mathbf p_{mlk}\big)
$$

- $M$：注意力头数；$L$：特征金字塔层数；$K$：每层采样点数
- $\Delta \mathbf p_{mlk}$：由 query 经线性层预测的**采样偏移量**（可学习，"deformable" 的含义）
- $A_{mlk}$：采样点对应的注意力权重，满足 $\sum_{l,k}A_{mlk}=1$（对 `num_levels*num_points` 维度做 softmax）
- $\phi(\mathbf f_l,\cdot)$：在特征图 $\mathbf f_l$ 上做**双线性插值**采样

代码对应（以 3D 版本 `MSDeformableAttention3D` 为例）：

```333:347:UniAD/projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py
value = self.value_proj(value)
if key_padding_mask is not None:
    value = value.masked_fill(key_padding_mask[..., None], 0.0)
value = value.view(bs, num_value, self.num_heads, -1)
sampling_offsets = self.sampling_offsets(query).view(
    bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
attention_weights = self.attention_weights(query).view(
    bs, num_query, self.num_heads, self.num_levels * self.num_points)
attention_weights = attention_weights.softmax(-1)
```

复杂度对比：标准注意力 $O(40000\times 6\times H_fW_f\times 256)$ → Deformable Attention $O(40000\times 8\times 4\times 256)$，降了几个数量级。

**偏移量初始化技巧**（让训练更稳定）：初始时每个头指向圆周上不同方向，不同采样点按距离递增缩放：

```252:269:UniAD/projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py
constant_init(self.sampling_offsets, 0.)
thetas = torch.arange(
    self.num_heads,
    dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
```

### 2.3 Spatial Cross-Attention：3D→2D 透视投影公式（图像特征 → BEV）

**问题**：BEV Query 对应真实世界的一个 `(x,y)` 网格位置，但图像特征在 2D 像素平面，二者之间隔着相机的内外参。SCA 要做的是：把 BEV 网格的 3D 参考点投影到每个相机的图像平面，找到对应的图像特征去做 Deformable Attention 采样。

**Step 1：BEV 网格 → 3D 参考点（多个高度采样，即 pillar）**

每个 BEV 格子在真实世界是一根"柱子"（因为不知道该格子上方到底有多高的物体），沿高度方向采样 `num_points_in_pillar=4` 个点：

```48:73:UniAD/projects/mmdet3d_plugin/uniad/modules/encoder.py
# 摘要：在 [z_min, z_max] 范围内等间隔采样 4 个高度层，
# 与 (x,y) 网格坐标组合成 3D 参考点 ref_3d: (bs, num_points_in_pillar, H*W, 3)
```

**Step 2：归一化坐标 → 真实米制坐标**

$$
P_{\text{lidar}} = P_{\text{norm}} \odot (\text{pc\_max} - \text{pc\_min}) + \text{pc\_min}
$$

```100:105:UniAD/projects/mmdet3d_plugin/uniad/modules/encoder.py
reference_points[..., 0:1] = reference_points[..., 0:1] * \
    (pc_range[3] - pc_range[0]) + pc_range[0]
```

**Step 3：齐次坐标 + 相机投影矩阵**

$$
\begin{pmatrix}u'\\v'\\d'\\1\end{pmatrix} = \mathbf{P}_{\text{lidar2img}}\begin{pmatrix}x\\y\\z\\1\end{pmatrix},\qquad
\mathbf{P}_{\text{lidar2img}} = \mathbf K \cdot [\mathbf R_{l2c}\,|\,\mathbf t_{l2c}]
$$

**Step 4：透视除法（perspective division）**

$$
u = \frac{u'}{d'},\qquad v = \frac{v'}{d'}
$$

**Step 5：归一化到 [0,1] + 可见性判断**

$$
\hat u = u / W_{\text{img}},\quad \hat v = v / H_{\text{img}}
$$

一个 3D 点只有同时满足**深度为正**（在相机前方）且**落在图像范围内**，才算对这个相机"可见"：

$$
\text{visible} = (d' > \epsilon)\ \wedge\ (0<\hat u<1)\ \wedge\ (0<\hat v<1)
$$

```114:134:UniAD/projects/mmdet3d_plugin/uniad/modules/encoder.py
reference_points_cam = torch.matmul(
    lidar2img.to(torch.float32), reference_points.to(torch.float32)
).squeeze(-1)
eps = 1e-5
bev_mask = (reference_points_cam[..., 2:3] > eps)
reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
    reference_points_cam[..., 2:3],
    torch.ones_like(reference_points_cam[..., 2:3]) * eps)
reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                     & (reference_points_cam[..., 1:2] < 1.0)
                     & (reference_points_cam[..., 0:1] < 1.0)
                     & (reference_points_cam[..., 0:1] > 0.0))
```

由于每根"柱子"上有 4 个高度点，同一个 BEV Query 通常只在部分相机中可见（比如靠近车身的点可能被自车遮挡）；SCA 只对每个 BEV Query **实际可见的相机**做加权平均：

```164:171:UniAD/projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py
slots += queries
count = bev_mask.sum(-1) > 0
count = count.permute(1, 2, 0).sum(-1)
slots = slots / torch.clamp(count[..., None], min=1.0)
```

### 2.4 Temporal Self-Attention：历史帧 BEV 的时序对齐

**问题**：上一帧的 BEV 特征是在"上一帧自车位置"为原点的坐标系下计算的，但自车在两帧之间发生了运动（位移+转向），必须先把上一帧 BEV 平移旋转对齐到"当前帧自车位置"为原点的坐标系，才能和当前帧融合。

这就是很多初学者最容易迷惑的地方——**为什么要用 can_bus 信号做坐标对齐**。

**位移对齐**（利用 CAN 总线记录的自车位移）：

$$
\Delta P_{\text{lidar}} = \mathbf R_{l2g}^{-1}\cdot \Delta P_{\text{global}}, \qquad
\text{shift} = \frac{\Delta P_{\text{lidar}}}{[\text{real\_w},\ \text{real\_h}]}
$$

```121:135:UniAD/projects/mmdet3d_plugin/uniad/modules/transformer.py
# 摘要：can_bus 中的全局位移 → 转到 LiDAR 坐标系 → 归一化为 BEV 网格坐标的偏移量 shift
```

**旋转对齐**（把历史 BEV 特征图按 Δ航向角旋转）：

```140:149:UniAD/projects/mmdet3d_plugin/uniad/modules/transformer.py
# 摘要：以 BEV 中心 (100,100) 为轴心，将 prev_bev 特征图旋转 Δθ（度）
```

对齐之后，当前帧的 BEV 参考点 `ref_2d` 与历史帧的偏移参考点 `shift_ref_2d = ref_2d + shift` 被拼在一起，分别去查询 `prev_bev`（历史特征）和 `bev_query`（当前特征）：

```193:208:UniAD/projects/mmdet3d_plugin/uniad/modules/encoder.py
if prev_bev is not None:
    prev_bev = prev_bev.permute(1, 0, 2)
    prev_bev = torch.stack(
        [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
    hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
        bs*2, len_bev, num_bev_level, 2)
```

最终，两路 Deformable Attention 的输出取**平均**，作为时序融合结果：

```254:259:UniAD/projects/mmdet3d_plugin/uniad/modules/temporal_self_attention.py
output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
output = output.mean(-1)
```

第一帧（没有历史）时，`prev_bev=None`，代码退化为"自己与自己"的自注意力（`hybird_ref_2d` 两路都用 `ref_2d`）。

### 2.5 BEVFormerLayer 的整体数据流

每层执行顺序为 `TSA → LayerNorm → SCA → LayerNorm → FFN → LayerNorm`，堆叠 6 层，最终输出 `bev_embed: (H*W, B, 256) = (40000, B, 256)`，这就是后续所有任务头共享的"世界模型"。

---

## 3. Stage1：TrackHead —— 端到端检测与多目标跟踪

文件：`uniad_track.py`（跨帧调度逻辑）+ `track_head.py`（单帧检测网络）+ `track_head_plugin/`（Query 生命周期管理）。

### 3.1 核心思想：DETR-Track 范式

传统跟踪：先逐帧检测，再用 IoU/外观特征做数据关联（匈牙利匹配）——检测和关联是两个独立步骤。

UniAD 采用 **Query-based 跟踪**：每个目标对应一个持续存在的向量（Track Query），这个向量从目标第一次出现开始就存在，每一帧都被网络更新，天然地包含了跨帧关联信息，不需要额外的关联算法。

- `num_query=900` 个候选 Query，用于探测新目标
- 额外 **1 个专属 Query**（索引第 900 个，"Ego Query"）固定表示自车（SDC）：

```202:206:UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py
self.query_embedding = nn.Embedding(self.num_query + 1, self.embed_dims * 2)
self.reference_points = nn.Linear(self.embed_dims, 3)
```

### 3.2 检测头：3D 框如何从 BEV 特征解码

每个 Query 通过 Deformable Attention（`decoder.py`）在 BEV 特征上采样，经过 6 层 Decoder 迭代优化，最终每层输出：

- `cls_branches`：分类 logit（10类）
- `reg_branches`：`code_size=10` 维的框参数 `[cx,cy,cz,w,l,h,sinθ,cosθ,vx,vy]`
- `past_traj_reg_branches`：历史+未来轨迹偏移（额外的辅助监督，帮助 Query 学习运动信息）

```140:153:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/track_head.py
self.code_weights = [1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
#                    cx   cy   cz   w    l    h   sinθ cosθ  vx   vy
```
（速度维度权重更小，因为速度真值噪声更大）

### 3.3 Query 生命周期管理（跟踪的核心）

每帧结束后，`uniad_track.py` 执行以下步骤：

1. **匈牙利匹配**（`criterion.match_for_single_frame`）：把 901 个 Query 的预测和 GT 目标做二分图匹配，分配 `obj_idxes`（全局唯一 ID）
2. **速度更新参考点**（`velo_update`）：假设目标做匀速直线运动，用预测速度把参考点"预测"到下一帧的位置，减轻下一帧检测头的搜索负担：

$$
P_{t+1} = \underbrace{\mathbf R_{g2l_2}\big(\mathbf R_{l2g_1}(P_t + \mathbf v\cdot \Delta t) + \mathbf t_{l2g_1} - \mathbf t_{l2g_2}\big)}_{\text{Local}_1 \to \text{Global} \to \text{Local}_2}
$$

```393:467:UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py
# velo_update: 反sigmoid→sigmoid→真实坐标→速度外推→Local1→Global→Local2→重新归一化→反sigmoid
```

3. **QIM（Query Interaction Module）**：决定哪些 Query 继续跟踪（高置信度）、哪些新目标被激活、哪些消失目标被回收（`miss_tolerance=5` 帧未检测到则删除）。
4. **MemoryBank**：为每个活跃目标保存过去 4 帧的特征，提供长时记忆，缓解遮挡问题。

### 3.4 输出给下游任务的关键产物

```750:786:UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py
# select_active_track_query: 筛选出有效跟踪目标，输出 track_query_embeddings
# select_sdc_track_query:    专门提取自车 Query 的状态，输出 sdc_embedding
```

- `track_query_embeddings`：所有存活目标的特征向量，喂给 MotionHead 做运动预测
- `sdc_embedding`：自车的特征向量，是后续 `sdc_track_query` 的直接来源

---

## 4. SegHead：在线地图分割（略讲）

SegHead 基于 BEV 特征预测车道线、可行驶区域等地图元素，输出 `lane_query`（地图要素的 Query 表示）。它的产出主要被 MotionHead 用作"地图上下文"（agent 需要知道自己在哪条车道上、车道怎么走），本文档不做重点展开，读者可参考 `panseg_head.py` 和 `seg_head_plugin/` 目录。

---

## 5. MotionHead —— 多模态运动预测

文件：`motion_head.py` + `motion_head_plugin/`（`base_motion_head.py`, `modules.py`, `motion_deformable_attn.py`, `motion_optimization.py`）

### 5.1 问题定义与设计动机

给定当前帧所有被跟踪目标（含自车）的特征，预测它们未来 `predict_steps` 步的运动轨迹。**关键困难**：未来是不确定的——同一个十字路口前的车辆，可能直行、左转、右转，这是**多模态**问题，如果用单一轨迹回归会导致模型学出"平均轨迹"（往往是不合理的居中轨迹）。

UniAD 采用 **MTP（Multimodal Trajectory Prediction）** 范式：为每个 agent 预测 `num_anchor` 条候选轨迹 + 每条轨迹的概率，训练时只对"最接近 GT 的那条"（winner）做回归监督（Winner-Take-All），分类分支学习给 winner 最高概率。

### 5.2 Anchor（锚点）机制

Anchor 轨迹模板是**离线用 K-Means 对训练集轨迹聚类**得到的（不是随机初始化），按粗粒度类别分组（车辆/行人等各自一套模板）：

```33:45:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/base_motion_head.py
def _load_anchors(self, anchor_info_path):
    anchor_infos = pickle.load(open(anchor_info_path, 'rb'))
    self.kmeans_anchors = torch.stack(
        [torch.from_numpy(a) for a in anchor_infos["anchors_all"]])
    # shape: (num_groups, num_anchor, predict_steps, 2)
```

这些 anchor 本质上是"典型运动模式的先验知识"——比如车辆组的 6 个 anchor 可能对应"直行快""直行慢""左转""右转""掉头""静止"等模式，网络不需要从零学习这些基本模式，只需要在此基础上做偏移量微调。

### 5.3 Intention Query：四种嵌入的融合

每个 anchor 会转化为一个 **Intention Query**（意图查询向量），由四种嵌入相加/融合而成：

| 嵌入 | 坐标系 | 直觉含义 |
|---|---|---|
| `agent_level_embedding` | 目标自身局部坐标系 | "这类目标典型的运动模式长什么样" |
| `scene_level_ego_embedding` | 全局/BEV 坐标系（含平移+旋转） | "这个 anchor 轨迹在场景中的绝对空间位置" |
| `scene_level_offset_embedding` | 仅旋转（不含平移） | "anchor 相对目标当前朝向的偏移方向" |
| `learnable_embed` | 无坐标系，纯可学习参数 | 让网络自己学一些数据驱动的补充信息 |

融合方式（`modules.py`）：

```python
static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed
dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
    [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))
query_embed_intention = self.static_dynamic_fuser(torch.cat(
    [static_intention_embed, dynamic_query_embed], dim=-1))
```

`static` 只在第 0 层算一次（代表先验、不随迭代变化），`dynamic` 每层重新计算（代表当前场景实时感知）。

### 5.4 MotionFormer：四路交互机制

MotionHead 内部堆叠若干层 `MotionTransformerDecoder`（`modules.py`），每层做四种交互：

1. **IntentionInteraction**（模式间自注意力）：让 6 条候选轨迹之间互相"商量"，避免多条 anchor 收敛到同一个模式；
2. **TrackAgentInteraction**（agent-agent 交叉注意力）：query=某 agent 的意图向量，key/value=场景内所有 agent 的 track_query，建模**博弈行为**（如"前车减速，我也要减速"）；
3. **MapInteraction**（agent-map 交叉注意力）：query=意图向量，key/value=`lane_query`，建模**遵循车道约束**；
4. **Motion Deformable Attention**（agent-BEV 交叉注意力）：让预测轨迹的每个时间步分别去 BEV 特征图上采样局部环境信息。

**Motion Deformable Attention 与标准 Deformable Attention 的本质区别**：标准版本的参考点是固定的（query 自身位置），而这里的参考点是**沿着预测轨迹的每个时间步移动的**——即"轨迹上第几步，就去 BEV 图对应位置附近采样"：

```430:447:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/motion_deformable_attn.py
reference_trajs = reference_trajs[:, :, :, [self.sample_index], :, :]
reference_trajs_ego = self.agent_coords_to_ego_coords(reference_trajs, bbox_results)
reference_trajs_ego[..., 0] = (reference_trajs_ego[..., 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
sampling_locations = reference_trajs_ego + sampling_offsets / offset_normalizer
```

四路交互的输出经过融合 MLP（`out_query_fuser`），再迭代更新参考轨迹坐标，如此循环 `num_layers` 次，逐层refine。

### 5.5 SDC Query 是如何"混"在其中被产生的

**巧妙设计**：SDC（自车）并没有单独的一套网络，而是被当作"第 901 个 agent"，和其他被跟踪目标一起走完整个 MotionFormer 流程：

```191:211:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py
# 摘要：把 sdc_embedding 拼接到 track_query 序列的最后一个位置，
# track_query: (B, A_track+1, D)，最后一个是 SDC
```

跑完 MotionFormer 之后，从输出序列的**最后一个位置**取出，就是 SDC 专属的轨迹 query：

```273:278:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py
outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]          # (n_dec, B, num_anchor=6, D)
outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]           # (B, D)
outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]   # (B, D)
```

这两个变量正是 **PlanningHead 的核心输入**（第7节会用到）。

### 5.6 损失函数：高斯 NLL + Winner-Take-All

#### (a) 二元高斯激活函数

回归分支对每条候选轨迹的每个时间步输出 5 个数：`(Δx, Δy, log σx, log σy, ρ_logit)`，需要转换为合法的高斯分布参数（标准差必须为正、相关系数必须在 `[-1,1]`）：

```5:24:UniAD/projects/mmdet3d_plugin/models/utils/functional.py
def bivariate_gaussian_activation(ip):
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)   # exp 保证 σ>0
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)      # tanh 保证 |ρ|<1
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out
```

#### (b) 双变量高斯 NLL 推导

标准双变量高斯分布：

$$
f(x,y) = \frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_x)^2}{\sigma_x^2}+\frac{(y-\mu_y)^2}{\sigma_y^2}-\frac{2\rho(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y}\right]\right)
$$

负对数似然（NLL，即损失）：

$$
\mathcal L_{\text{NLL}} = -\log f = \frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_x)^2}{\sigma_x^2}+\frac{(y-\mu_y)^2}{\sigma_y^2}-\frac{2\rho(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y}\right] + \log(\sigma_x\sigma_y) + \tfrac12\log(1-\rho^2) + \log 2\pi
$$

代码里等价的实现（令 $\text{ohr}=1/\sqrt{1-\rho^2}$，注意代码中的 `sig_x` 已经是网络回归量**取倒数**意义上的用法，本质上和上式等价，只是把 $1/\sigma$ 吸收进了系数里）：

```python
# traj_loss.py 第141-155行（近似摘录）
ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
nll = 0.5 * ohr**2 * (
        sig_x**2 * (x - mu_x)**2 + sig_y**2 * (y - mu_y)**2
        - 2*rho*sig_x*sig_y*(x - mu_x)*(y - mu_y)
      ) - torch.log(sig_x * sig_y * ohr) + 1.8379   # 1.8379 ≈ log(2π)
```

#### (c) 多模态最近邻匹配（Winner-Take-All）

$$
k^\ast = \arg\min_{k=1,\dots,K}\ \big\|\hat{p}_T^{(k)} - p_T^{gt}\big\|_2 \quad (\text{minFDE 匹配，以终点误差为准})
$$

只对赢家 $k^\ast$ 那条轨迹计算回归 NLL 损失；分类损失让赢家概率最大化：

$$
\mathcal L_{\text{cls}} = -\log p_{k^\ast} = -\log\frac{e^{s_{k^\ast}}}{\sum_j e^{s_j}}
$$

对应的评测指标定义：

$$
\text{minADE}=\min_k \frac1T\sum_{t=1}^T\|\hat p_t^{(k)}-p_t^{gt}\|_2,\qquad
\text{minFDE}=\min_k \|\hat p_T^{(k)}-p_T^{gt}\|_2
$$

$$
\text{MissRate}=\frac1N\sum_i \mathbb 1\Big[\min_k\max_t\|\hat p_{i,t}^{(k)}-p_{i,t}^{gt}\|_2>2\text{m}\Big]
$$

#### (d) GT 轨迹的运动学平滑（Motion Optimization）

在计算损失前，UniAD 还会用一个基于 CasADi 的**非线性最优控制问题**对 GT 轨迹做平滑（`motion_optimization.py`），本质上是把带噪声的标注轨迹投影到"物理可行"的轨迹流形上：

- 状态量：$\mathbf x=[x,y,\psi,v]$（位置、朝向、速度）
- 控制量：$\mathbf u=[\kappa,a]$（曲率、加速度）
- 运动学方程（自行车模型，RK4 积分）：

$$
\dot x=v\cos\psi,\quad \dot y=v\sin\psi,\quad \dot\psi=v\kappa,\quad \dot v=a
$$

- 目标函数：跟踪参考轨迹的误差 + 控制量平滑（曲率变化率、jerk）+ 终端朝向误差加权：

$$
J=\alpha_{xy}\|\text{pos}-\text{pos}_{\text{ref}}\|^2+\alpha_\psi\|\psi-\psi_{\text{ref}}\|^2+\alpha_{\text{rate}}(\|\dot\kappa\|^2+\|\text{jerk}\|^2)+\alpha_{\text{abs}}(\|\kappa\|^2+\|a\|^2)
$$

注意 SDC 自身的 GT 轨迹**不参与**此平滑（因为规划任务有自己的 GT，见第7节）。

---

## 6. OccHead —— 占用流预测

文件：`occ_head.py` + `occ_head_plugin/`（`modules.py`, `metrics.py`, `utils.py`）

### 6.1 与 MotionHead 的互补关系

MotionHead 预测的是"目标中心点未来在哪"（一条曲线），但没有告诉你目标的**空间范围**（车有多大、朝向如何、会不会和自车重叠）。OccHead 预测的是"未来每个时刻，BEV 网格中哪些像素被占据"（`n_future=4` 步，2秒），是**逐像素稠密预测**，直接可用于碰撞检测。

### 6.2 网络结构：类 U-Net 时序传播

1. **BEV 特征降维**：`(200×200×256) → SimpleConv2d → (200×200×64)`，再经 2 次 `Bottleneck` 下采样到 `50×50`；
2. **Query 融合**：把 MotionHead 输出的 `traj_query`（轨迹特征）、`track_query`（跟踪特征）通过 MLP 融合为 `ins_query`（每个 agent 一个 64 维向量），代表"这个目标的未来会怎样占用空间"；
3. **逐时间步循环**（5 个未来时间步）：每步做 `downscale → Transformer cross-attn（用 ins_query 条件化 BEV 特征）→ UpsamplingAdd（上采样+跳连，类似U-Net decoder）`，并将上一步的状态递推给下一步（类似 ConvGRU 的隐状态传递）；
4. **实例占用解码**：把每个 agent 的 query 特征和逐时间步的 dense BEV 特征做**内积**（`einsum`），得到每个 agent 在每个时刻、每个像素上的占用 logit：

$$
\text{occ\_logit}_{b,q,t,h,w} = \sum_{c} \text{ins\_query}_{b,q,t,c}\cdot \text{feat}_{b,t,c,h,w}
$$

### 6.3 损失函数

- **Mask Loss**（`FieryBinarySegmentationLoss`）：像素级二分类 BCE，配合 **Top-K 困难样本挖掘**（只对最难的 25% 像素回传梯度）和**未来帧折扣**（`future_discount=0.95`，越远的时间步权重越低，因为越难预测）；
- **Dice Loss**：衡量预测掩码与 GT 掩码的整体重叠程度，缓解正负样本极度不均衡的问题（大部分 BEV 格子是空的）：

$$
\text{Dice} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
$$

其中 $p_i$ 是预测概率，$g_i$ 是 GT 标签（0/1）。

### 6.4 评测：全景分割指标（PQ/SQ/RQ）

`metrics.py` 中的 `PanopticMetric` 把语义分割（是否被占用）与实例分割（是哪个目标占用）结合，用 IoU 匹配预测实例和 GT 实例，计算：

$$
\text{PQ}=\underbrace{\frac{\sum_{(p,g)\in \text{TP}} \text{IoU}(p,g)}{|\text{TP}|}}_{SQ\ (\text{分割质量})}\times \underbrace{\frac{|\text{TP}|}{|\text{TP}|+\tfrac12|\text{FP}|+\tfrac12|\text{FN}|}}_{RQ\ (\text{识别质量})}
$$

OccHead 最终产出 `seg_out`（占用掩码），会传给 PlanningHead 用于碰撞避免。

---

## 7. PlanningHead —— 自车轨迹规划（原版回归式）

文件：`planning_head.py` + `planning_head_plugin/`（`collision_optimization.py`, `planning_metrics.py`）

这是 UniAD 的**最终任务头**，也是本项目改造的对象（改造版见 `DiT-learning.md`）。先理解原版怎么做，才能理解改造版"改进了什么"。

### 7.1 三路信息融合

$$
\text{plan\_query} = \text{MLP}\Big(\big[\ \text{sdc\_traj\_query}\ \|\ \text{sdc\_track\_query}\ \|\ \text{navi\_embed}\ \big]\Big)
$$

三路输入分别是：

| 输入 | 来源 | 含义 |
|---|---|---|
| `sdc_traj_query` | MotionHead 最后一层 | 自车的运动预测特征（6 条候选模式） |
| `sdc_track_query` | TrackHead（`detach()`） | 自车的跟踪状态特征 |
| `navi_embed` | 可学习 `nn.Embedding(3, D)` | 高层导航命令（右转/直行/左转）的向量化 |

拼接后维度 $256\times3=768$，经 MLP 压缩到 256，并沿 6 个候选模式维度取 `max`（相当于"自动挑选最显著的那个运动倾向"）：

```343:350:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py
plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)
plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[0]
```

### 7.2 Transformer Decoder 查询 BEV 全局特征

$$
\text{plan\_query}' = \text{TransformerDecoder}(\text{query}=\text{plan\_query},\ \text{memory}=\text{bev\_feat})
$$

（共 3 层 Decoder）

这一步让"1 个规划向量"通过交叉注意力去看**整张 BEV 图**（40000个格子），获得全局场景理解（车道走向、周围车辆分布等）。

### 7.3 回归输出：位移序列 → 累积绝对坐标

```401:407:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py
sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
sdc_traj_all[..., :2] = torch.cumsum(sdc_traj_all[..., :2], dim=1)
```

网络直接回归的是**每一步相对上一步的位移** $\Delta_t=(\Delta x_t,\Delta y_t)$，通过累加得到绝对轨迹：

$$
p_t = \sum_{i=1}^{t}\Delta_i
$$

之后再经 `bivariate_gaussian_activation`（同第5.6节公式）转成高斯分布参数，兼容概率化的下游使用（虽然规划头默认只用均值 $\mu_x,\mu_y$ 作为最终输出坐标）。

### 7.4 损失函数

$$
\mathcal L_{\text{plan}} = \underbrace{\frac{1}{T}\sum_t m_t\|\hat p_t - p_t^{gt}\|_2}_{\text{loss\_ade（PlanningLoss）}} + \sum_{i}\underbrace{w_i\cdot\text{IoU}_{\text{AABB}}(\text{box}(\hat p_t),\ \text{GT box}_j)}_{\text{loss\_collision\_i}}
$$

**PlanningLoss（ADE）**：

```21:26:UniAD/projects/mmdet3d_plugin/losses/planning_loss.py
def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
    err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=-1)
    err = torch.pow(err, exponent=0.5)
    return torch.sum(err * mask)/(torch.sum(mask) + 1e-5)
```

即标准的（掩码平均）欧氏距离误差 ADE：$\mathcal L_{\text{ade}}=\dfrac{\sum_t m_t\|\hat p_t-p_t^{gt}\|_2}{\sum_t m_t}$。

**CollisionLoss（多安全边界的框重叠面积惩罚）**：

对预测轨迹每一步，构造一个尺寸略微膨胀过的自车包围盒（膨胀量 `delta`），与该时刻所有 GT 目标框计算 AABB（轴对齐包围盒）重叠面积，重叠越大惩罚越大：

$$
w=1.85+\delta,\quad h=4.084+\delta,\qquad
\mathcal L_{\text{col}}=\sum_{t}\sum_{j}\text{Area}\big(\text{AABB}(\text{box}_{\text{sdc},t})\cap \text{AABB}(\text{box}_{j,t})\big)
$$

```30:65:UniAD/projects/mmdet3d_plugin/losses/planning_loss.py
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        self.w = 1.85 + delta
        self.h = 4.084 + delta
    def inter_bbox(self, corners_a, corners_b):
        xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        ...
        intersect = max((xi1 - xi2), 0) * max((yi1 - yi2), 0)
        return intersect
```

配置中通常用 3 个不同的 `delta`（0 / 0.5 / 1.0m）叠加，构成**多尺度安全边界**：越严格的边界（delta 越小）权重越大（如 2.5），越宽松的边界权重越小（如 0.25），这样既有硬约束（不能真的撞上）又有软约束（尽量保持安全距离）。

### 7.5 推理时的碰撞避免后处理（非线性优化）

训练阶段用 GT 框做碰撞损失，但推理阶段没有 GT，只能用 OccHead 预测的占用图 `occ_mask`。UniAD 用一个基于 **CasADi + IPOPT** 的非线性优化器 `CollisionNonlinearOptimizer` 对初始规划轨迹做后处理微调：

$$
\min_{\{x_t,y_t\}} \quad \underbrace{\sum_t\big[(x_t-x_t^{\text{ref}})^2+(y_t-y_t^{\text{ref}})^2\big]}_{\text{跟踪原始规划轨迹}} \ +\ \underbrace{\alpha\sum_t\sum_{i} \frac{1}{2.507\sigma}\exp\left(-\frac{(x_t-c_{x,i})^2+(y_t-c_{y,i})^2}{2\sigma^2}\right)}_{\text{高斯排斥场，远离每个障碍物像素}}
$$

- 第一项保证优化后的轨迹不会偏离网络原始预测太远；
- 第二项是以每个占用像素 $(c_{x,i},c_{y,i})$ 为中心的高斯"排斥力场"，距离越近代价越高；
- $\sigma$ 控制排斥场的作用范围，$\alpha$（`alpha_collision`）控制避障的"激进程度"。

```216:280:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/collision_optimization.py
cost_stage = alpha_xy * sumsqr(self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y))
normalizer = 1 / (2.507 * self.sigma)
for t in range(len(self.obj_pixel_pos)):
    x, y = self.position_x[t], self.position_y[t]
    for i in range(len(self.obj_pixel_pos[t])):
        col_x, col_y = self.obj_pixel_pos[t][i]
        cost_collision += alpha_collision * normalizer * exp(
            -((x - col_x)**2 + (y - col_y)**2) / 2 / self.sigma**2)
self._optimizer.minimize(cost_stage + cost_collision)
```

障碍物像素来源（`planning_head.py` 的 `collision_optimization` 方法）：从 `occ_mask` 中取值为 1 的网格，转换为 ego 坐标系下的米制坐标，并**只保留距预测轨迹点 5m 以内的**（`occ_filter_range=5.0`）以降低噪声干扰。

**求解方式**：直接多重打靶法（Direct Multiple Shooting）——把每个时间步的 (x,y) 都设为独立决策变量，一次性联合优化，用 IPOPT 内点法求解，用原始网络预测轨迹做 warm start（初始猜测）以加速收敛。

---

## 8. 损失函数与评测指标全景

### 8.1 全部任务损失一览

| 任务 | 损失 | 核心公式 |
|---|---|---|
| Track | Focal Loss + L1 | DETR 式集合匹配损失 |
| Map (Seg) | 分割损失 | 略 |
| Motion | `loss_cls`（NLL分类）+ `loss_reg`（高斯NLL）+ `minADE`/`minFDE` | 第5.6节 |
| Occ | `loss_mask`（BCE+TopK+时间折扣）+ `loss_dice` | 第6.3节 |
| Planning | `loss_ade`（PlanningLoss）+ `loss_collision_i`（多delta） | 第7.4节 |

各任务损失最终按 `task_loss_weight` 加权求和（默认全部为 1.0）：

```335:355:UniAD/projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py
def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
    loss_factor = self.task_loss_weight[prefix]
    loss_dict = {f"{prefix}.{k}": v * loss_factor for k, v in loss_dict.items()}
    return loss_dict
```

### 8.2 规划任务的评测指标（PlanningMetric）

推理阶段最终关心的三个指标（`planning_head_plugin/planning_metrics.py`），均按 `n_future=6` 个时间步（0.5s~3s）分别统计：

**① L2 误差**：

$$
\text{L2}_t = \sqrt{m_t\cdot\big[(x_t^{\text{pred}}-x_t^{\text{gt}})^2+(y_t^{\text{pred}}-y_t^{\text{gt}})^2\big]}
$$

**② obj_col（点级碰撞率）**：把预测轨迹点转换为 BEV 网格索引，检查该像素在占用图中是否为 1（有障碍物），且**排除 GT 轨迹自身就会碰撞的时间步**（避免把"数据集本身的极端场景"误判为模型的错）：

```84:118:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py
m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))
obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()
```

**③ obj_box_col（框级碰撞率）**：不只检查轨迹中心点，而是把完整的自车包围盒（`W=1.85m, H=4.084m`）光栅化为像素集合，平移到每个时间步的预测位置，只要**框内任意像素**碰到障碍物即判定为碰撞——比 obj_col 更严格、更贴近真实碰撞语义：

```43:82:UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py
pts = np.array([[-H/2+0.5, W/2], [H/2+0.5, W/2], [H/2+0.5, -W/2], [-H/2+0.5, -W/2]])
rr, cc = polygon(pts[:,1], pts[:,0])   # 光栅化矩形框为像素坐标集合
```

| 指标 | 粒度 | 严格程度 |
|---|---|---|
| obj_col | 轨迹中心点 | 宽松 |
| obj_box_col | 完整车身框 | 严格 |

三个指标都在 1s/2s/3s（对应第2/4/6步）汇报，这是论文 Table 中 `L2(m)` 和 `Col. Rate(%)` 的直接来源。

---

## 9. 训练/推理数据流总览图

```
                              ┌─────────────────────────────┐
                              │   多相机图像 (B,Ncam,3,H,W)   │
                              └───────────────┬───────────────┘
                                              │ ResNet + FPN
                                              ▼
                              ┌─────────────────────────────┐
                              │  BEVFormer Encoder (6层)      │
                              │  TSA(历史帧融合) + SCA(图像→BEV)│
                              └───────────────┬───────────────┘
                                              │ bev_embed (40000,B,256)
                     ┌────────────────────────┼─────────────────────────┐
                     ▼                        ▼                         │
        ┌─────────────────────┐   ┌─────────────────────┐              │
        │  TrackHead (6层Dec)  │   │   SegHead            │              │
        │ 900+1个 Query        │   │  地图/车道线要素       │              │
        │ 匈牙利匹配+QIM+MemBank│   └──────────┬───────────┘              │
        └──────────┬───────────┘              │ lane_query               │
                    │ track_query, sdc_embedding                          │
                    ▼                          ▼                         │
        ┌─────────────────────────────────────────────┐                 │
        │           MotionHead (MotionFormer)           │                 │
        │  Anchor(K-Means)+4路交互(意图/agent/map/BEV)   │                 │
        │  Winner-Take-All + 高斯NLL                     │                 │
        └───────────────┬───────────────┬───────────────┘                 │
                         │ traj_query(车辆)│ sdc_traj_query, sdc_track_query│
                         ▼               │                                │
        ┌─────────────────────┐          │                                │
        │  OccHead (类U-Net)    │          │                                │
        │ 5步时序ConvGRU式传播  │          │                                │
        └──────────┬───────────┘          │                                │
                    │ occ_mask             │                                │
                    ▼                      ▼                                ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │                    PlanningHead                                    │
        │  三路融合(轨迹+跟踪+命令) → TransformerDecoder查询BEV → 回归位移累积  │
        │  训练: ADE + CollisionLoss(GT框)                                    │
        │  推理: CasADi非线性优化(occ_mask高斯排斥场)                          │
        └───────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              自车未来 6 步轨迹 (3秒)
```

---

## 10. 关键代码索引表

| 模块 | 功能 | 文件 |
|---|---|---|
| 总控 | `forward_train`/`forward_test` 串联五大任务 | `uniad/detectors/uniad_e2e.py` |
| BEV底座 | 检测+跟踪主循环，Query生命周期管理 | `uniad/detectors/uniad_track.py` |
| BEV底座 | 6层Encoder，TSA+SCA堆叠 | `uniad/modules/encoder.py` |
| BEV底座 | 图像→BEV 3D-2D投影注意力 | `uniad/modules/spatial_cross_attention.py` |
| BEV底座 | 历史帧时序自注意力 | `uniad/modules/temporal_self_attention.py` |
| BEV底座 | ego-motion坐标对齐、can_bus注入 | `uniad/modules/transformer.py` |
| Track | 检测头网络结构与损失 | `uniad/dense_heads/track_head.py` |
| Motion | 主流程、SDC query提取 | `uniad/dense_heads/motion_head.py` |
| Motion | Anchor加载、分支初始化 | `motion_head_plugin/base_motion_head.py` |
| Motion | MotionFormer四路交互 | `motion_head_plugin/modules.py` |
| Motion | 轨迹型Deformable Attention | `motion_head_plugin/motion_deformable_attn.py` |
| Motion | GT轨迹运动学平滑(CasADi) | `motion_head_plugin/motion_optimization.py` |
| Occ | 主流程、类U-Net时序解码 | `uniad/dense_heads/occ_head.py` |
| Occ | BEV裁剪/卷积/上采样模块 | `occ_head_plugin/modules.py` |
| Occ | PQ/SQ/RQ全景分割指标 | `occ_head_plugin/metrics.py` |
| Planning | 三路融合+Transformer+回归 | `uniad/dense_heads/planning_head.py` |
| Planning | 推理时CasADi碰撞规避优化 | `planning_head_plugin/collision_optimization.py` |
| Planning | L2/obj_col/obj_box_col评测 | `planning_head_plugin/planning_metrics.py` |
| Losses | PlanningLoss/CollisionLoss/KinematicLoss | `losses/planning_loss.py` |
| Losses | 高斯激活函数、坐标变换工具 | `models/utils/functional.py` |

---

## 附：延伸阅读建议

1. 想深入 Deformable DETR 的原理，可阅读 Zhu et al., *Deformable DETR* (ICLR 2021)。
2. 想理解 BEVFormer 的设计初衷，可阅读 Li et al., *BEVFormer* (ECCV 2022)。
3. 想理解本项目如何把 PlanningHead 从"确定性回归"改造为"生成式 Flow Matching"，请阅读配套的 **`DiT-learning.md`**，其中会与本文档的第7节逐点对比。
