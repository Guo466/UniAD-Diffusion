# 第 3 讲：Transformer / Attention / DETR——整个项目的骨架技术

> **预计学习时长**：4-6 小时（这是全路线最核心、最值得花时间的一讲）
> **前置要求**：已读完 `01_FOUNDATIONS_DL_BASICS.md` 和 `02_FOUNDATIONS_CNN_AND_PROB.md`。
> **学完后你应该能回答**：
> 1. Self-Attention 机制具体是怎么算的？Query/Key/Value 分别代表什么？
> 2. 为什么 Transformer 比 RNN/LSTM 更适合处理长序列？
> 3. Multi-Head Attention 是什么？为什么要"多头"？
> 4. 位置编码在做什么？为什么 Transformer 需要它而 CNN 不需要？
> 5. Transformer Encoder 和 Decoder 有什么区别？
> 6. DETR 是什么？它怎么用 Transformer 做目标检测的？
> 7. Deformable Attention 和标准 Attention 有什么区别？为什么 BEVFormer 需要它？

---

## 1. 从"为什么需要 Attention"开始

在 Transformer 出现之前，处理序列数据（比如一句话的多个词、一段视频的多帧图像）
的主流方法是 RNN（循环神经网络）或 LSTM。RNN 的做法是：按时间顺序逐个处理输入，
每处理一个就更新一个"隐状态"，这个隐状态携带着"之前看过的内容的总结"。

RNN 的问题：
1. **只能串行处理，不能并行**：必须先处理第 1 个词，才能处理第 2 个词，
   训练速度慢，GPU 的并行算力发挥不出来；
2. **长距离依赖弱**：隐状态从第 1 个词一路传到第 100 个词，信息会逐渐"稀释"，
   模型很难记住很早之前的输入（虽然 LSTM 有门控机制缓解，但本质上还是有这个问题）。

Transformer 的核心创新是**Self-Attention（自注意力）机制**：
让序列中的每个位置都能**直接、一次到位地"看到"序列中所有其他位置**，
不需要像 RNN 那样一步一步传。这解决了上述两个问题——可以并行计算，
也不存在"远距离信息稀释"的问题。

---

## 2. Self-Attention：最核心的机制，逐行拆解

### 2.1 核心直觉："每个位置都在问一个问题，所有位置来回答"

假设你有一句话："The cat sat on the mat because it was tired."

当模型处理到 "it" 这个词时，"it" 指代的是 "cat" 还是 "mat"？
人类很轻松就能根据上下文判断，但模型怎么做到？Self-Attention 的设计思路是：
让每个位置都**主动发出一个"查询"（Query）**，同时每个位置都**提供自己的"信息"（Key 和 Value）**，
通过 Query 和 Key 的匹配程度（"相关度"），决定每个位置应该从其他位置那里
"吸收"多少信息（Value 的加权组合）。

- **Query (Q)**："我在找什么？"——每个位置根据自己的内容生成一个"查询向量"，
  表示"我想知道和我相关的内容"；
- **Key (K)**："我有什么信息？"——每个位置生成一个"键向量"，
  表示"我提供的信息可以用这个特征来检索"；
- **Value (V)**："我的具体内容是什么？"——每个位置生成一个"值向量"，
  是真正要被传递的信息内容。

这个 Q/K/V 类比图书馆检索：
你拿着一张写着关键词的纸条（Query），去和每本书的标签（Key）做比对，
比对结果越匹配的书，你就越认真地读它的内容（Value）。

### 2.2 具体计算步骤

假设输入序列有 $N$ 个位置，每个位置已经被映射成了一个 $d$ 维向量（比如词嵌入，
或者你项目里的 BEV Query / Track Query），记作矩阵 $X \in \mathbb R^{N \times d}$。

**Step 1：生成 Q, K, V**

通过三个可学习的线性变换矩阵 $W^Q, W^K, W^V$，把输入 $X$ 映射成 Q, K, V：

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

这里 $W^Q, W^K \in \mathbb R^{d \times d_k}$，$W^V \in \mathbb R^{d \times d_v}$，
所以 $Q, K \in \mathbb R^{N \times d_k}$，$V \in \mathbb R^{N \times d_v}$。
（$d_k$ 和 $d_v$ 通常相等，记为 $d_{\text{model}}/h$，$h$ 是头数，下面会讲。）

**Step 2：计算注意力权重（Attention Weights）**

用 Q 和 K 的点积来衡量"每个 Query 和每个 Key 有多匹配"：

$$
\text{scores} = Q K^T \in \mathbb R^{N \times N}
$$

这个矩阵的 $(i,j)$ 元素就是第 $i$ 个位置的 Query 和第 $j$ 个位置的 Key 的点积，
点积越大说明越"匹配/相关"。

然后除以 $\sqrt{d_k}$（防止点积数值太大导致 softmax 梯度消失，这是一个工程上的
缩放技巧，所以也叫 **Scaled Dot-Product Attention**），再过 softmax 归一化：

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb R^{N \times N}
$$

softmax 的作用是让每一行的权重**加起来等于 1**，可以理解为"第 $i$ 个位置
分配给所有位置的注意力比例"（关注谁多少）。

**Step 3：加权求和得到输出**

用注意力权重对 Value 做加权求和：

$$
\text{Output} = A \cdot V \in \mathbb R^{N \times d_v}
$$

输出的第 $i$ 行就是"第 $i$ 个位置，从所有位置的 Value 中，按注意力权重加权组合出来的新表示"。

### 2.3 完整公式（一句话版）

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

这就是你在 UniAD-learning.md 第 2.2 节看到的那个公式。现在你应该能完全读懂了：
$QK^T/\sqrt{d_k}$ 是计算相关度，softmax 归一化成权重，乘以 $V$ 做加权聚合。

---

## 3. Multi-Head Attention：为什么不只用一个注意力，而要用"多头"

### 3.1 动机：一个注意力头只能关注一种模式

如果只用一个注意力头（即只算一次 $QKV$），那么每个位置只能输出一种
"对所有位置的加权组合"，这意味着它只能捕捉一种类型的关联模式
（比如可能只学会了关注"距离最近的对象"，或者只学会了关注"同一类别的对象"）。

但现实中，一个词/一个对象可能同时需要关注多种不同类型的信息
（比如 "it" 既需要关注语法上的主语 "cat"，又需要关注语义上的位置 "on the mat"）。

### 3.2 做法：并行算多个头，拼接后投影

把 $d_{\text{model}}$ 维的特征，拆成 $h$ 个头（head），每个头的维度是
$d_k = d_{\text{model}} / h$，每个头**独立**地算一遍 Attention：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

然后把 $h$ 个头的输出拼起来，过一个线性投影 $W^O$：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

**效果**：不同的头可以自动学会关注不同类型的关联模式
（类似 CNN 里不同的卷积核学会检测不同的纹理），
最后拼接投影相当于把这些不同视角的信息融合起来，表达能力远强于单头注意力。

你项目里的 `num_heads=8`（BEVFormer 的 Deformable Attention）和
`dit_heads=8`（DiT 的 Self-Attention）就是在设置这个 $h$ 值。

---

## 4. Transformer 的完整结构

### 4.1 Transformer Encoder（编码器）：理解输入

一个 Encoder Block 由两个子层组成：

```
输入 → Multi-Head Self-Attention → Add & Norm → Feed-Forward Network (FFN) → Add & Norm → 输出
```

- **Add & Norm**：**残差连接**（Residual Connection，把输入直接加到输出上）+
  **Layer Normalization**（对特征做归一化，稳定训练）。残差连接的核心思想是：
  让网络学习的是"和输入的差值"（残差）而不是"从零开始变换"，
  这大大缓解了深层网络训练时的梯度消失问题（梯度可以通过残差连接的"捷径"
  直接传回浅层），是让几十层的 Transformer 能被训练出来的关键技术。
- **FFN**：一个两层的 MLP（线性→ReLU→线性），对每个位置独立做非线性变换，
  增加表达能力。

堆叠 $L$ 个这样的 Encoder Block（比如 BERT 用 12 层，GPT 用 96 层），
就得到完整的 Transformer Encoder，输出是对输入序列的"深层理解"表示。

### 4.2 Transformer Decoder（解码器）：生成输出

Decoder Block 比 Encoder Block 多一个子层——**Cross-Attention（交叉注意力）**：

```
输入 → Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm → 输出
            ↑ 自己看自己            ↑ 用 Encoder 的输出做 K, V
```

- **Masked Self-Attention**：和普通 Self-Attention 一样，但加了**掩码**（mask），
  防止当前位置"偷看"后面的位置（因为在生成任务中，后面的内容还没生成出来，
  不能用来做参考）；
- **Cross-Attention**：Query 来自 Decoder 自身，但 **Key 和 Value 来自 Encoder 的输出**，
  相当于 Decoder 在"问" Encoder："和当前要生成的内容相关的输入信息是什么？"
  这是 Encoder-Decoder 结构中**两个模块交换信息的桥梁**。

你项目里 UniAD 的 MotionHead 就使用了这种 Cross-Attention——
Motion Query（来自跟踪的 Agent）作为 Query，BEV 特征作为 Key/Value，
让每个 Agent 的查询去 BEV 特征图中"找和自己运动相关的信息"。

---

## 5. 位置编码（Positional Encoding）：为什么 Transformer 需要它

Self-Attention 的计算是**位置无关**的——对它来说，"猫坐在垫子上"和"垫子上坐在猫"
是一样的（因为 QK 点积不区分顺序），但显然这两句话意思完全不同。

为了弥补这个缺陷，Transformer 在输入 Attention 之前，给每个位置的特征向量
**加上一个位置编码向量**（Positional Encoding），让模型知道"这个词在第几个位置"。
原始 Transformer 用的是正弦/余弦函数的固定编码，后来的变体（比如你项目中
BEVFormer 的可学习位置编码 `nn.Embedding`）让位置编码也成为可训练的参数。

**为什么 CNN 不需要位置编码？** 因为卷积操作天然保持空间结构——
卷积核在图像上的滑动位置本身就是位置信息，每个输出值天然对应输入的一个空间位置。
而 Self-Attention 是"全局看到所有位置"，不区分位置差异，所以需要额外注入位置信息。

---

## 6. DETR：用 Transformer 做目标检测的革命性方法

### 6.1 传统检测方法的流程

在 DETR 之前，目标检测的主流方法（如 Faster R-CNN、YOLO 系列）都是"两步走"：
1. 用一个区域提议网络（RPN）生成大量候选框（"这里可能有个东西"）；
2. 对每个候选框做分类和回归精修。

这种方法的缺点：需要手工设计的锚框（Anchor）、非极大值抑制（NMS）等启发式步骤，
不够优雅，也很难和端到端训练完美兼容。

### 6.2 DETR 的核心思想：把检测变成"集合预测"问题

DETR（Detection Transformer，2020 年 Facebook 提出）的思路非常简洁：

1. 用 CNN 骨干网络提取图像特征；
2. 用 Transformer Encoder 对特征做 Self-Attention 增强；
3. **关键创新**：引入一组**可学习的 Object Query**（比如 100 个），
   每个 Query 通过 Transformer Decoder 的 Cross-Attention 去"查询"图像特征，
   每个被"激活"的 Query 对应图像中的一个检测目标，输出它的类别和边界框坐标；
4. 用**匈牙利匹配**（Hungarian Matching）把预测结果和真实标注做最优一对一匹配，
   不需要 NMS（因为每个 Query 最多预测一个目标，天然不会重复检测）。

**Object Query 是理解 DETR 的关键**，也是理解你整个项目的关键——
UniAD 中的 Track Query、Motion Query、SDC Query，本质上都是 DETR 的 Object Query
思想在不同任务上的延伸和变体。它们都是**可学习的向量**，通过 Cross-Attention
从更大的特征图中"提取出"自己关心的信息，然后被一个预测头（MLP）解码成
具体的输出（检测框、轨迹、规划路径等）。

### 6.3 DETR 与 UniAD 的联系

UniAD 可以看作是 DETR 思想在多任务自动驾驶场景下的极致扩展：

| DETR | UniAD 对应物 |
|---|---|
| 图像特征 | BEV 特征（BEVFormer 把多相机图像统一到鸟瞰图） |
| Object Query | Track Query（跟踪目标）、Motion Query（运动预测）、SDC Query（自车） |
| Transformer Decoder | 多个 Decoder 堆叠，每个任务有专属的 Cross-Attention |
| 分类头 + 回归头 | 检测头 + 跟踪头 + 运动预测头 + 占用预测头 + 规划头 |
| 单任务（检测） | 多任务联合（检测→跟踪→运动→占用→规划） |

理解了 DETR，你就能理解 UniAD 里为什么每个任务都用"Query + Cross-Attention"的模式——
这是 DETR 范式的自然延续，Query 就是"主动的信息索取者"，特征图就是"信息源"，
Cross-Attention 就是"按需提取"的机制。

---

## 7. Deformable Attention：让 Attention 更高效的变体

### 7.1 标准 Attention 的计算瓶颈

标准 Self-Attention 的计算量是 $O(N^2)$（$N$ 是序列长度/特征点数），
因为每个位置都要和所有其他位置算一次点积。在 BEVFormer 的场景下，
BEV 网格有 $200 \times 200 = 40000$ 个 Query，图像特征也有几万个点，
$40000^2 = 16$ 亿次点积运算，直接用标准 Attention 是不现实的。

### 7.2 Deformable Attention 的思路：每个 Query 只看"关键少数"个点

Deformable Attention（Deformable DETR, 2021）的核心简化：
**不让每个 Query 看所有位置，而是让网络自己学会"应该看哪几个点"**——
每个 Query 预测出 $K$ 个采样偏移量（offset），只在参考点附近这 $K$ 个偏移位置
去采样特征，注意力权重也只在这 $K$ 个采样点上做 softmax。

计算量从 $O(N^2)$ 降到 $O(N \cdot K)$（$K$ 通常是 4-8，远小于 $N$），
代价是每个位置只能看到局部信息，但实践中效果往往足够好
（因为 Transformer 堆叠多层后，信息可以通过逐层传播间接"看到"全局）。

**这也是为什么 UniAD-learning.md 里 Deformable Attention 公式中出现了
$\Delta\mathbf p_{mlk}$（采样偏移量）和 $K$（采样点数）——
它们就是 Deformable Attention 相比标准 Attention 新增的概念。**

---

## 8. 本讲小结：把 Transformer 和你的项目联系起来

1. **Self-Attention**：每个位置通过 Q/K/V 机制，直接从所有位置按需提取信息，
   是 Transformer 的核心操作，也是 UniAD 中所有"Query 交叉注意力"的理论基础。
2. **Multi-Head Attention**：多头并行，让模型能同时关注多种关联模式，
   你项目中的 `num_heads` 就是在控制这个。
3. **Encoder/Decoder 结构**：Encoder 做输入理解，Decoder 做"按需查询"，
   UniAD 的每个任务头（Track/Motion/Planning）都使用了 Decoder 模式。
4. **DETR 的 Object Query**：可学习的查询向量，通过 Cross-Attention 从特征图中
   提取信息并解码成具体输出，UniAD 中的各种 Query 都是这一思想的延伸。
5. **Deformable Attention**：标准 Attention 的高效变体，只在参考点附近采样
   少量点做注意力，是 BEVFormer 在 40000 个 BEV Query 场景下能跑起来的关键。
6. **位置编码 / 残差连接 / LayerNorm**：这些"基础设施"确保了 Transformer
   的稳定训练和位置感知能力。

---

## 自检问题（这是全路线最重要的一组自检题，务必认真对待）

1. 用自己的话解释：Self-Attention 中 Query、Key、Value 分别代表什么？
   为什么需要三个而不是两个或一个？
2. 写出 Scaled Dot-Product Attention 的完整公式，并解释每一步的含义。
3. 为什么要除以 $\sqrt{d_k}$？如果不除会怎样？
4. 多头注意力（Multi-Head Attention）相比单头有什么好处？
5. Transformer Encoder 和 Decoder 的主要区别是什么？Cross-Attention 在做什么？
6. 残差连接（Residual Connection）解决了什么问题？为什么深层网络必须要有它？
7. DETR 的 Object Query 是什么？它怎么替代了传统检测方法中的 Anchor 和 NMS？
8. Deformable Attention 相比标准 Attention 做了什么简化？为什么这种简化在实践中是可接受的？
9. 请把 DETR 的 Object Query 和 UniAD 中的 Track Query / Motion Query / SDC Query
   做类比，解释它们的共同设计思想。

如果以上问题你都能合上文档、用自己的话讲清楚，恭喜，你已经掌握了整个项目的
架构骨架。接下来可以进入自动驾驶领域知识的补充：
`04_FOUNDATIONS_AUTONOMOUS_DRIVING.md`。
