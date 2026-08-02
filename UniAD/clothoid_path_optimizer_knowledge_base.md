# ClothoidPathOptimizer 完整知识体系：从数学基础到工程实现

> 本文档由四篇独立笔记合并整理而成，目标是把 `modules/planning/path/clothoid_path_optimizer/` 这个模块从**最底层的数学原理**到**最终的工程代码**，构建成一条完整、无断层的学习链路。
>
> 参考的学城文档：
> - [《变分法》](https://km.sankuai.com/collabpage/2778046310) —— 数学地基
> - [《AL-iLQR》](https://km.sankuai.com/collabpage/2778214068) —— 核心算法框架
> - [《Line Search方法》](https://km.sankuai.com/collabpage/2777844970) —— 算法内部关键子步骤
> - [《DesignDoc】20251011 - Clothoid Path Optimizer》](https://km.sankuai.com/collabpage/2777203995) —— 工程设计文档
>
> 本文假设你是**完全零基础**的小白，不需要任何优化理论或最优控制背景，我们会从最直觉的例子讲起，一步步搭建到能看懂全部代码的程度。

---

## 全文知识体系导览

在开始之前，先建立一个"鸟瞰图"，理解四块内容是怎么串起来的：

```
┌─────────────────────────────────────────────────────────────────┐
│  第一部分：数学地基 —— 变分法                                        │
│  "如何从无穷多条候选曲线里，找出让某个指标最优的那一条？"                    │
│  核心结论：欧拉-拉格朗日方程                                          │
└─────────────────────────────────────────────────────────────────┘
                              │ 应用于"控制系统"这一具体场景，
                              │ 加入动力学约束，就演化成……
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  第二部分：核心算法 —— 从变分法到 AL-iLQR                             │
│  2.1 Clothoid 曲线与状态空间建模（问题怎么描述）                        │
│  2.2 离散化（连续问题怎么变成计算机能算的问题）                          │
│  2.3 LQR 与 HJB 方程（无约束线性二次问题的解析解）                      │
│  2.4 iLQR（非线性问题的迭代局部近似解法）                              │
│  2.5 增强拉格朗日法 AL（怎么处理不等式约束）                            │
│  核心结论：AL 外层循环 + iLQR 内层循环，交替求解                        │
└─────────────────────────────────────────────────────────────────┘
                              │ iLQR 内层的 Forward Pass 里，
                              │ 有一个关键子步骤需要单独深入……
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  第三部分：算法细节 —— 线搜索（Line Search）                          │
│  "沿着算出来的下降方向，这一步到底该走多远？"                            │
│  核心结论：回溯线搜索 + Wolfe 条件保证收敛性                           │
└─────────────────────────────────────────────────────────────────┘
                              │ 理论落地，映射到实际工程代码……
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  第四部分：工程实现全景                                              │
│  4.1 整体架构与文件地图                                              │
│  4.2 约束与代价的完整清单（曲率、转速、加速度、jerk、半平面……）            │
│  4.3 几何工具库（让"折线距离"处处可导：LogSumExp + SmoothPolyline）    │
│  4.4 utils/：决策层信息 → 约束代价对象 的组装流水线                     │
│  4.5 完整求解主流程走读                                              │
│  4.6 端到端数值例子 + 关键设计取舍 FAQ                                │
└─────────────────────────────────────────────────────────────────┘
```

**一句话概括整个知识体系**：变分法告诉我们"最优解要满足什么样的必要条件"；把这个思想应用到"状态随时间/弧长演化的控制系统"上，并针对非线性、有约束的实际问题做迭代化、数值化处理，就得到了 AL-iLQR 算法；这个算法每一步迭代都要"选方向、选步长"，其中"选步长"这个子问题由线搜索理论负责；最后，所有这些理论都要落地成 `clothoid_path_optimizer` 文件夹里几十个文件的具体代码，包括如何描述车辆运动学、如何把车道边界/障碍物翻译成约束代价、如何处理折线不可导的数值难题等一系列工程问题。

### 阅读建议

- **如果你只想知道"这个模块大概是干什么、怎么跑起来的"**：直接跳到第四部分，重点看 4.1（架构）和 4.5（主流程走读）。
- **如果你想彻底搞懂算法原理、能自己推导公式**：按顺序从第一部分读到第三部分，每一节都配有对应的代码引用，建议对照源码一起看。
- **如果你想做参数调优或排查求解失败问题**：重点看第二部分 2.5 节（AL 收敛判定）、第三部分（线搜索失败的原因）、第四部分 4.6 节（FAQ 里有大量调参相关的问答）。

### 全文目录

**第一部分：数学地基 —— 变分法**
1.1 先建立直觉：什么是"泛函"，变分法在解决什么问题
1.2 热身：普通函数求极值 vs 泛函求极值
1.3 变分法基本引理
1.4 变分法基本方法：微小扰动
1.5 欧拉-拉格朗日方程完整推导
1.6 直观理解与一个具体例子

**第二部分：核心算法 —— 从变分法到 AL-iLQR**
2.1 Clothoid 曲线与状态空间建模
2.2 离散化：从连续微分方程到 iLQR 能用的离散系统
2.3 LQR 与 HJB 方程
2.4 iLQR：非线性问题的迭代解法
2.5 增强拉格朗日法：怎么处理约束
2.6 代码总览：一次 `ComputePath` 调用做了什么
2.7 逐文件代码对应

**第三部分：线搜索（Line Search）详解**
3.1 为什么需要线搜索
3.2 下降方向
3.3 Wolfe 条件与回溯法
3.4 收敛性理论
3.5 代码对应：`solver.h` 里的回溯线搜索

**第四部分：工程实现全景**
4.1 整体架构与文件地图
4.2 约束与代价完整清单
4.3 几何工具库：让"折线距离"处处可导
4.4 utils/：决策层信息组装流水线
4.5 完整求解主流程走读
4.6 端到端数值例子与设计取舍 FAQ

---

# 第一部分：数学地基 —— 变分法

> 变分法研究的是**连续、无限维**的优化问题（"从无穷多条曲线里找最优的那条"）。它不会在 `clothoid_path_optimizer` 代码里直接出现（因为该模块用的是离散化后的数值方法，见第二部分），但它是**整个最优控制理论的地基**——第二部分里 HJB 方程、iLQR 的推导，本质上都是这里讲的思想的延伸和离散化版本。

## 1.1 先建立直觉：什么是"泛函"，变分法在解决什么问题

### 从"函数"到"泛函"

你从小学到的函数，比如 $f(x)=x^2$，是"**输入一个数字，输出一个数字**"。

现在换一个更抽象的东西：**输入一整条曲线（一个函数），输出一个数字**。这种"函数的函数"，专业名词叫**泛函（functional）**。

生活中的例子：
- 给你一条从家到公司的路线（一条曲线 $F(x)$），"输出"这条路线的**总长度**——这是一个泛函。
- 给你一条赛车的路径，"输出"跑完这条路径需要的**总时间**——同样是泛函。

**变分法（Calculus of Variations）** 就是专门研究"**在所有可能的曲线里，找出让某个泛函取得最大值或最小值的那一条**"的数学工具。

### 为什么这很难

普通函数求极值，是在一堆**数字**里找最优的那个数字。而泛函求极值，是在一堆**曲线（函数）**里找最优的那一条——这是"**无穷维**"的问题。变分法要解决的核心问题就是：**怎么把这个看起来无从下手的"无穷维找最优"问题，转化成可以用微积分求解的方程？**

答案的核心工具：**变分法基本引理** + **微小扰动法**，最终推导出**欧拉-拉格朗日方程**——一个"普通的微分方程"，解出它就找到了让泛函取极值的那条曲线。

## 1.2 热身：普通函数求极值 vs 泛函求极值

回忆普通函数 $g(x)$ 求极值：求导数 $g'(x)$，令 $g'(x_0)=0$ 解出候选点（极值的**必要条件**）。

**为什么成立？** 在极值点附近，稍微挪动 $x_0\to x_0+h$（$h$ 很小），函数值变化 $g(x_0+h)-g(x_0)\approx g'(x_0)h$。要让这个一阶变化量对**任意小的 $h$** 都不产生下降/上升趋势，只能是 $g'(x_0)=0$。

**变分法把这个思路原封不动地搬到"曲线"上**：不是挪动一个数字，而是"挪动"整条曲线 $F(x)$，然后要求这个挪动不会让泛函继续变化（一阶变化量为零），从而反推出 $F(x)$ 必须满足的方程。

## 1.3 变分法基本引理：整篇推导的"地基"

### 引理内容

> 如果函数 $M(x)$ 在区间 $[a,b]$ 上连续，并且对于**任意**一个满足边界条件 $h(a)=h(b)=0$ 的连续（可微）函数 $h(x)$，都有：
>
> $$\int_a^b M(x)\cdot h(x)\,dx=0 \tag{1}$$
>
> 那么，$M(x)$ 在整个区间 $[a,b]$ 上必须恒等于 $0$。

后面推导欧拉-拉格朗日方程时，会得到一个形如"某个式子乘以任意函数 $\eta(x)$，积分等于 0"的等式。要从这里得出"某个式子本身就是 0"，靠的正是这个引理——**它是变分法能把"无穷维问题化简为微分方程"的关键跳板**。

### 为什么成立？（直观证明——反证法）

1. **假设** $M(x)$ 在某点 $x_0$ 不等于 0，比如 $M(x_0)>0$。
2. 因为 $M(x)$ 连续，$x_0$ 附近的小邻域内 $M(x)$ 也都大于 0（连续函数的局部保号性）。
3. 既然 $h(x)$ 可以任意构造，就专门构造一个"凸起"的 $h(x)$：在这个小邻域内 $h(x)>0$，其余地方 $h(x)=0$。
4. 乘积 $M(x)h(x)$ 在邻域内为正，邻域外为 0，整个积分**必然大于 0**。
5. 但这与前提"式(1)对任意 $h(x)$ 都等于 0"**矛盾**！

同理 $M(x_0)<0$ 也会导致矛盾。**因此唯一不产生矛盾的可能性：$M(x)$ 处处等于 0。**

这个证明的精髓在于"**任意**"两个字给了极大的自由度——我们可以"精准打击"到 $M(x)$ 不为零的地方，如果矛盾出现，说明假设不成立。

## 1.4 变分法基本方法：如何"轻轻扰动"一个函数

### 泛函问题的一般形式

$$
J[F]=\int_a^bL\big(x,F(x),F'(x)\big)\,dx
$$

- $F(x)$：候选曲线，是这个问题的**自变量**。
- $L(x,F,F')$："**拉格朗日量**"，综合"位置、曲线取值、曲线斜率"算出一个数。
- $J[F]$：整条曲线的拉格朗日量沿 $x$ 积分起来的总值。

### 构造微小扰动

$$
\widetilde F(x)=F(x)+\varepsilon\,\eta(x) \tag{2}
$$

| 符号 | 含义 |
|---|---|
| $F(x)$ | 候选最优曲线 |
| $\eta(x)$ | "变分函数"/"扰动方向"，把曲线往哪个"形状"上挪 |
| $\varepsilon$ | 挪动的幅度大小 |
| $\widetilde F(x)$ | 挪动后的新曲线 |

**边界条件**：若原问题要求 $F(a)=\alpha,F(b)=\beta$（端点固定），则要求 $\eta(a)=\eta(b)=0$，这样 $\widetilde F$ 也自动满足同样的边界值。

### 转化成普通函数问题

固定 $F,\eta$ 后，$J[F+\varepsilon\eta]$ 就只是一个关于 $\varepsilon$ 的普通函数：

$$
\Phi(\varepsilon):=J[F+\varepsilon\eta]
$$

**这是整个变分法最巧妙的地方**：把"无穷维函数空间里找极值"的问题，通过"沿特定方向 $\eta$ 切一刀"，转化成了高中就会解的"一元函数求极值"问题！若 $F$ 是最优曲线，则 $\Phi(\varepsilon)$ 在 $\varepsilon=0$ 处必须是极值点。

## 1.5 欧拉-拉格朗日方程：完整推导

**第一步：极值必要条件** $\Phi'(0)=0$。

**第二步：求导展开**。用链式法则：

$$
\Phi'(0)=\int_a^b\left[\frac{\partial L}{\partial F}\eta(x)+\frac{\partial L}{\partial F'}\eta'(x)\right]dx=0 \tag{3}
$$

**第三步：分部积分**（关键技巧，把 $\eta'(x)$ 也转化成 $\eta(x)$ 形式）：

$$
\int_a^b\frac{\partial L}{\partial F'}\eta'(x)\,dx=\left[\frac{\partial L}{\partial F'}\eta(x)\right]_a^b-\int_a^b\frac{d}{dx}\left(\frac{\partial L}{\partial F'}\right)\eta(x)\,dx
$$

因为 $\eta(a)=\eta(b)=0$，边界项直接消失！这正是为什么变分法一定要给 $\eta(x)$ 加"端点为零"这个条件。

**第四步：合并**：

$$
\int_a^b\left[\frac{\partial L}{\partial F}-\frac{d}{dx}\left(\frac{\partial L}{\partial F'}\right)\right]\eta(x)\,dx=0 \tag{4}
$$

**第五步：套用基本引理**。式 (4) 对任意 $\eta(x)$ 都成立，所以方括号里的式子必须恒等于 0：

$$
\boxed{\frac{\partial L}{\partial F}-\frac{d}{dx}\left(\frac{\partial L}{\partial F'}\right)=0}
$$

这就是**欧拉-拉格朗日方程（Euler-Lagrange Equation）**！

## 1.6 直观理解与一个具体例子

### 逻辑链条总结

```
① 想找让泛函 J[F] 取极值的曲线 F(x)
       ↓
② 假设 F(x) 就是最优解，往任意方向 η(x) 挪一点点 ε
       ↓
③ 把 J[F+εη] 看成关于 ε 的普通函数 Φ(ε)
       ↓
④ 极值必要条件：Φ'(0) = 0
       ↓
⑤ 展开求导，出现 η(x) 和 η'(x) 两项
       ↓
⑥ 分部积分，把 η'(x) 转化成 η(x) 形式
       ↓
⑦ 边界条件 η(a)=η(b)=0 让分部积分的边界项消失
       ↓
⑧ 套用变分法基本引理
       ↓
⑨ 得出欧拉-拉格朗日方程
```

**一句话总结**：把"在无穷多条曲线里找最优"的问题，通过"任意方向的微小扰动 + 一阶变化率必须为零"，转化成了曲线本身必须满足的微分方程。

### 例子：两点间最短路径是直线

设曲线 $y=F(x)$，连接 $(a,y_a)$ 和 $(b,y_b)$，弧长泛函 $J[F]=\int_a^b\sqrt{1+F'(x)^2}\,dx$，拉格朗日量 $L=\sqrt{1+F'^2}$（不显式依赖 $x,F$）。

$$
\frac{\partial L}{\partial F}=0,\qquad \frac{\partial L}{\partial F'}=\frac{F'}{\sqrt{1+F'^2}}
$$

代入欧拉-拉格朗日方程：$-\dfrac{d}{dx}\left(\dfrac{F'}{\sqrt{1+F'^2}}\right)=0 \Rightarrow \dfrac{F'}{\sqrt{1+F'^2}}=$ 常数 $\Rightarrow F'(x)=k$（常数）$\Rightarrow F(x)=kx+c$，**正是一条直线**！这就用变分法严格证明了"两点之间线段最短"。

### 为什么 `clothoid_path_optimizer` 代码里没有直接对应

欧拉-拉格朗日方程能被解析求解的场景非常有限——通常只在拉格朗日量简单、且没有不等式约束的情况下才有解析解。真实路径规划问题有大量不等式约束（车道边界、避障、曲率限制），会让最优解条件变得复杂（需要 KKT 条件等），因此工程上普遍采用**离散化 + 数值迭代**（第二部分的 AL-iLQR）来逼近最优解。**变分法给出的是"理想情况下最优解应满足什么条件"的理论指导，AL-iLQR 是"现实世界有约束、非线性场景下，数值逼近类似最优性条件的解"的工程实现**——这条传承关系会在第二部分反复出现。

---

# 第二部分：核心算法 —— 从变分法到 AL-iLQR

> 第一部分讲了变分法的思想："在无穷多条曲线里找最优，等价于找满足特定微分方程的那条"。本部分把这个思想具体化到"控制系统"这一场景，加入**动力学约束**和**不等式约束**，一步步推导出工程上真正使用的 AL-iLQR 算法。
>
> 变分法与本部分的联系：
> - 欧拉-拉格朗日方程 → HJB 方程（连续版本的最优性必要条件）
> - HJB 方程 → Riccati 方程/LQR（线性二次情形的解析解）
> - LQR → iLQR（非线性情形的迭代近似）
> - 无约束 → 有约束（AL 方法处理不等式约束）

## 2.1 Clothoid 曲线与状态空间建模

### 为什么需要一个新的 Path Optimizer

在此之前，Planning 里跑的是 Frenet Frame（基于参考线 s-l 坐标系）下的 path optimizer，在几类场景下天然存在困难：

| 场景 | 具体问题 |
|---|---|
| 大曲率 U-turn | s-l 坐标系下很难严格保证 path 曲率满足车辆最小转弯半径；曲率作为 cost 而非约束，容易超限 |
| 狭窄空间掉头/会车 | 只把车辆看作一个点来避障，没有精确的"四角碰撞"描述 |
| 大曲率参考线 | s-l 坐标系下 curvature、dkappa 的近似计算在参考线曲率很大时会病态 |

**根本原因**：Frenet frame 下的很多变量本质上是对参考线曲率的近似，参考线曲率越大，近似误差越大。

Clothoid Path Optimizer 的思路是**完全抛弃 Frenet 坐标系**，直接在 Cartesian 坐标系下用一个包含曲率变化率的运动学模型描述 path。

### 为什么叫 "Clothoid"

Clothoid（回旋曲线/Euler 螺线）是一条曲率随弧长线性变化的曲线，常用于道路缓和曲线设计。这里的模型更进一步：不仅 $\kappa$ 对弧长的一阶导 $\dot\kappa$ 可控，还引入二阶导 $\ddot\kappa$ 作为控制量，本质上是一个更高阶的曲线族。

### 状态方程

设 path 上一点的状态为 $(x, y, \theta, \kappa, \omega)$，控制量为 $\alpha = \ddot\kappa$：

$$
\left\{
\begin{aligned}
\dot{x} &= \cos\theta \\
\dot{y} &= \sin\theta \\
\dot{\theta} &= \kappa \\
\dot{\kappa} &= \omega \\
\dot{\omega} &= \alpha
\end{aligned}
\right.
$$

简记 $\dot{\bm x} = \bm f(\bm x, \bm u)$，$\bm x \in \mathbb{R}^5$，$\bm u \in \mathbb{R}^1$。

**为什么要引入 $\omega$ 和 $\alpha$**：
- 只有 $(x,y,\theta,\kappa)$ 4 维的话，$\kappa$ 可以在相邻采样点之间瞬间跳变，对应无穷大的转向速率，物理不可实现。
- 引入 $\omega=\dot\kappa$ 后，$\omega$ 直接对应前轮转向速率，限制 $\omega$ 就能限制横向 jerk。
- 进一步引入 $\alpha=\ddot\kappa$ 作为控制量，限制方向盘转速的变化率，避免小幅度高频抖动。

### 代码对照

```cpp:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer_common.h
struct ClothoidPathStateIndex {
  static constexpr int kX = 0;
  static constexpr int kY = 1;
  static constexpr int kTheta = 2;
  static constexpr int kKappa = 3;
  static constexpr int kDKappa = 4;   // ω = dκ/ds
  static constexpr int kSize = 5;
};

struct ClothoidPathControlIndex {
  static constexpr int kDdKappa = 0;  // α = ddκ/ds²
  static constexpr int kSize = 1;
};
```

动力学方程 $\bm f$ 的实现：

```cpp:modules/planning/path/clothoid_path_optimizer/clothoid_path_dynamic.cc
ClothoidPathDynamic::VecX ClothoidPathDynamic::Evaluate(const VecX& x, const VecU& u) const {
  VecX x_dot;
  x_dot(kX) = math::Cos(x(kTheta));
  x_dot(kY) = math::Sin(x(kTheta));
  x_dot(kTheta) = x(kKappa);
  x_dot(kKappa) = x(kDKappa);
  x_dot(kDKappa) = u(kDdKappa);
  return x_dot;
}
```

### Jacobian（用于线性化）

iLQR 需要动力学方程关于状态和控制的 Jacobian $\mathbf{J}_{\bm f} \in \mathbb{R}^{5\times 6}$：

$$
\mathbf{J}_{\bm f} =
\begin{bmatrix}
0 & 0 & -\sin\theta & 0 & 0 & 0 \\
0 & 0 & \cos\theta & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

代码：

```cpp:modules/planning/path/clothoid_path_optimizer/clothoid_path_dynamic.cc
ClothoidPathDynamic::Jacobians ClothoidPathDynamic::EvaluateJacobians(const VecX& x, const VecU& u) const {
  Jacobians jacobians = Jacobians::Zero();
  jacobians(kX, kTheta) = -math::Sin(x(kTheta));
  jacobians(kY, kTheta) =  math::Cos(x(kTheta));
  jacobians(kTheta, kKappa) = 1.0;
  jacobians(kKappa, kDKappa) = 1.0;
  jacobians(kDKappa, kSize + kDdKappa) = 1.0;
  return jacobians;
}
```

这个 Jacobian 非常稀疏（只有 5 个非零元素），因为模型除了 $\dot x, \dot y$ 依赖 $\theta$ 是非线性之外，其余都是线性关系。

## 2.2 离散化：从连续微分方程到 iLQR 能用的离散系统

iLQR 处理的是离散系统 $\bm x_{k+1} = \bm f_d(\bm x_k, \bm u_k)$，所以需要数值积分把连续模型离散化。

### 欧拉法（最简单的离散化）

$$
\bm x_{k+1} = \bm x_k + h\, \bm f(\bm x_k, \bm u_k)
$$

线性化后：$[\bm A\ \ \bm B] = [\bm I_{5\times5}\ \ \bm 0_{5\times1}] + h\,\mathbf{J}_{\bm f}$

### 四阶龙格库塔法（RK4，实际使用的方法）

$$
\begin{aligned}
\bm k_1 &= \bm f(\bm x_k, \bm u_k) \\
\bm k_2 &= \bm f(\bm x_k + \tfrac{h}{2}\bm k_1, \bm u_k) \\
\bm k_3 &= \bm f(\bm x_k + \tfrac{h}{2}\bm k_2, \bm u_k) \\
\bm k_4 &= \bm f(\bm x_k + h\bm k_3, \bm u_k) \\
\bm x_{k+1} &= \bm x_k + \tfrac{h}{6}(\bm k_1 + 2\bm k_2 + 2\bm k_3 + \bm k_4)
\end{aligned}
$$

Jacobian 通过链式法则递推各 $\bm k_i$ 对 $\bm x_k, \bm u_k$ 的 Jacobian，最终合成。

### 代码组织：ContinuousDynamics → Integrator → DiscretizedModel

- `ContinuousDynamics<X,U>`：定义 `Evaluate`（算 $\bm f$）和 `EvaluateJacobians`（算 $\mathbf J_{\bm f}$）接口，`ClothoidPathDynamic` 是它的实现。
- `Integrator<X,U>`：定义积分方法，`RungeKutta4` 是其中一种实现。
- `DiscretizedModel<X,U>`：组合 `ContinuousDynamics` + `Integrator` + 步长 $h$，对外暴露统一的 `DynamicModel` 接口。

```cpp:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
std::unique_ptr<DynamicModel<kNumStates, kNumControls>>
ClothoidPathOptimizer::GetDynamicModelAtSample(int index) const {
  auto dynamics = std::make_unique<const ClothoidPathDynamic>();
  auto integrator = std::make_unique<const RungeKutta4<kNumStates, kNumControls>>();
  return std::make_unique<DiscretizedModel<kNumStates, kNumControls>>(
      std::move(dynamics), std::move(integrator), steps_[index]);
}
```

注意每个采样点的步长 `steps_[index]` 可以不同（近处密、远处疏），所以每个点都要单独构造一个 `DynamicModel`。

### 采样步长的选择

代码里分"密采样段"和"稀疏采样段"：近处用较密的步长（正比于速度，设了 0.5m 下限保证低速时也有足够分辨率），远处用更稀疏的步长补足总长度——因为近处的路径精度对当前决策更重要。

## 2.3 LQR 与 HJB 方程

在讲 iLQR 之前，先理解它的"简化版祖先"：LQR。因为 iLQR 本质上是"在每一步都用 LQR 的思路，处理一个非线性问题的局部线性近似"。

### LQR 问题定义

假设动力学是**线性**的 $x_{k+1}=A_kx_k+B_ku_k$，代价是**二次型**的：

$$
\min_{u_k} \; \frac12 x_N^T Q_N x_N + \frac12\sum_{k=0}^{N-1}\left[x_k^TQ_kx_k + u_k^TR_ku_k\right]
$$

### 为什么能求出解析解：HJB 方程

定义**价值函数** $J^*(x,t)$ = "从状态 $x$、时刻 $t$ 出发的最小代价"。Bellman 最优性原理：**当前最优 = 这一步的代价 + 从下一步开始的最优代价**。对 LQR，假设价值函数也是二次型 $J^* = \frac12 x^TPx$，代入 HJB 方程可以解出：

$$
u^* = -Kx, \quad K_k = (R_k + B_k^TP_{k+1}B_k)^{-1}B_k^TP_{k+1}A_k
$$

$P$ 满足**离散黎卡提方程**：

$$
P_k = Q_k + A_k^TP_{k+1}A_k - A_k^TP_{k+1}B_k(R_k+B_k^TP_{k+1}B_k)^{-1}B_k^TP_{k+1}A_k
$$

**记住这个"从后往前递推"的模式**——这就是 iLQR 的 Backward Pass 的原型。LQR 是"一步到位"的解析解，因为问题是线性二次的；而 Clothoid 路径问题是非线性的，没有解析解，只能靠 iLQR 反复迭代逼近。

## 2.4 iLQR：非线性问题的迭代解法

### 核心思想

在当前"名义轨迹"附近，把动力学和代价都做局部二次/线性近似，用 LQR 的方法求出"改进方向"，然后沿着这个方向小步前进，反复迭代，直到收敛到局部最优解。

### Q-function 展开

$$
Q(x_k,u_k) = \ell_k(x_k,u_k) + V_{k+1}(f(x_k,u_k))
$$

对 Q 做二阶泰勒展开，得到分块矩阵：

$$
\begin{aligned}
Q_x &= \ell_x + f_x^T V_x', & Q_u &= \ell_u + f_u^T V_x' \\
Q_{xx} &= \ell_{xx} + f_x^T V_{xx}' f_x, & Q_{uu} &= \ell_{uu} + f_u^T V_{xx}' f_u, & Q_{ux} &= \ell_{ux} + f_u^T V_{xx}' f_x
\end{aligned}
$$

对应代码：

```cpp:modules/planning/common/math/solver/solver.h
VecX Q_x = l_x + f_x.transpose() * V_x;
VecU Q_u = l_u + f_u.transpose() * V_x;
MatXX Q_xx = l_xx + f_x.transpose() * V_xx * f_x;
MatUU Q_uu = l_uu + f_u.transpose() * V_xx * f_u;
MatUX Q_ux = l_ux + f_u.transpose() * V_xx * f_x;
```

> **iLQR vs DDP**：DDP 保留 $f$ 的二阶导数项，更精确但计算贵；iLQR 只用一阶导数（雅可比），用 Gauss-Newton 近似 Hessian，更快——这是本模块采用的方法，也是为什么 `ClothoidPathDynamic` 只需提供 `EvaluateJacobians`。

### 求最优控制增量

对 $\delta Q$ 关于 $\delta u$ 求导为零，加正则化 $\rho I$（Levenberg-Marquardt 式，保证 $Q_{uu}$ 正定可逆）：

$$
\delta u_k^* = -(Q_{uu}+\rho I)^{-1}(Q_{ux}\delta x_k + Q_u) = \underbrace{K_k}_{\text{反馈增益}}\delta x_k + \underbrace{d_k}_{\text{前馈修正}}
$$

代码（无控制量约束的简单情形）：

```cpp:modules/planning/common/math/solver/solver.h
meta->k_trajectory[i] = -Q_uu_reg_inv * Q_u;     // 前馈增益 d_k
meta->K_trajectory[i] = -Q_uu_reg_inv * Q_ux;    // 反馈增益 K_k
```

价值函数递推回代：

```cpp:modules/planning/common/math/solver/solver.h
V_x = Q_x + K^T * Q_uu * k + K^T * Q_u + Q_ux^T * k;
V_xx = Q_xx + K^T * Q_uu * K + K^T * Q_ux + Q_ux^T * K;
```

## 2.5 增强拉格朗日法：怎么处理约束

### 为什么是 AL + iLQR 这个组合

- 这是一个**带不等式约束的非线性最优控制问题**。
- **iLQR** 擅长无约束的轨迹优化，但不直接支持不等式约束。
- **AL** 把约束优化转化为一系列无约束优化。

两者结合：**外层 AL 负责"满足约束"，内层 iLQR 负责"在当前惩罚下求最优轨迹"**。

### 为什么朴素的罚函数法不够好

$$
\min_x \; f(x) + \frac{\mu}{2} c(x)^2
$$

要让约束精确满足，理论上需要 $\mu \to \infty$，但 $\mu$ 太大会导致数值病态。

### AL 的改进

加上拉格朗日乘子项：

$$
\mathcal{L}_A = f(x) + \lambda^T c(x) + \frac{1}{2} c(x)^T I_\mu\, c(x)
$$

- $\lambda$：拉格朗日乘子（对偶变量），在迭代中不断更新，逐渐逼近真实约束力。
- $\mu$：罚因子，不需要趋于无穷大。
- $I_\mu$：对角矩阵，对不等式约束且已被满足（$c_i<0$）且 $\lambda_i=0$ 的项置零（不罚已经满足的约束）。

直觉：$\lambda$ 提前把约束往正确方向推一把，$\mu$ 只是"锦上添花"的惩罚。

### AL 外层循环

代码位于 `ALSolver::Solve`：

```cpp:modules/planning/common/math/solver/augmented_lagrangian/al_solver.h
template <int X, int U, int C1, int C2>
typename ALSolver<X, U, C1, C2>::Solution ALSolver<X, U, C1, C2>::Solve(
    const ConstrainedProblem<X, U, C1, C2>& constrained_problem) const {
  ALProblem problem(constrained_problem);
  ResetDuals(&problem);      // 初始化 λ
  ResetPenalties(&problem);  // 初始化 μ
  ...
  for (int iter = 0; iter < config_.max_outer_iterations; ++iter) {
    const IlqrSolution inner_solution = ilqr_solver_.Solve(problem);  // 内层 iLQR
    UpdateStatus(problem, inner_solution, ...);
    const State state = CheckTerminationCondition(status);
    if (state != State::kUnsolved) return Solution{...};
    UpdateDuals(inner_solution, &problem);   // 更新 λ
    UpdatePenalties(&problem);               // 更新 μ（× penalty_scaling_factor）
  }
}
```

每一轮：
1. 内层 iLQR 求出一条轨迹；
2. 检查最大约束违反量 `max_violation`，若 < `constraint_tolerance`（如 1e-4）则成功退出；
3. 否则更新 $\lambda$（投影梯度上升）和 $\mu$（乘以 `penalty_scaling_factor`，如 10 倍）；
4. 失败退出条件：内层不收敛 / 外层超最大迭代 / $\mu$ 超上限 / 超时。

### 乘子更新公式（不等式约束）

$$
\lambda_i^+ = \max\{0,\; \lambda_i + \mu_i\, c_i(x^*)\}
$$

不等式约束的乘子必须非负（KKT 互补松弛性要求）。

### iLQR 内层循环

代码位于 `Solver::Solve`，核心是 **Backward Pass + Forward Pass + Line Search** 交替进行：

**Backward Pass**（2.4 节已讲）：从终端往回递推，算出 $\{K_k\}, \{d_k\}$。

**Forward Pass**：拿到每步的 $(K_k, d_k)$ 后，按步长系数 $\alpha$（从 1 开始回溯）滚动出新轨迹：

$$
\bm u_i^{new} = \bm u_i + \alpha\, d_i + K_i(\bm x_i^{new} - \bm x_i), \quad \bm x_{i+1}^{new} = f_d(\bm x_i^{new}, \bm u_i^{new})
$$

```cpp:modules/planning/common/math/solver/solver.h
(*u_trajectory_updated)[i] =
    u_trajectory[i] + context.alpha * context.k_trajectory[i] +
    context.K_trajectory[i] * ((*x_trajectory_updated)[i] - x_trajectory[i]);
(*x_trajectory_updated)[i + 1] =
    problem.dynamic_model(i).Evaluate((*x_trajectory_updated)[i], (*u_trajectory_updated)[i]);
```

**Line Search**：如果 $\alpha=1$ 时新轨迹代价没有显著下降，就把 $\alpha$ 缩小，重新做 Forward Pass，直到代价下降满足要求或 $\alpha$ 太小放弃（详见第三部分）。

### 内外循环嵌套关系（一图流）

```
AL 外层循环（更新 λ, μ）
  └── iLQR 内层循环（在固定 λ, μ 下求最优轨迹）
        └── Backward Pass（从后往前，算 K, d）
        └── Forward Pass（从前往后，滚动新轨迹 + 线搜索）
        └── 判断是否收敛，没收敛就再来一次 Backward+Forward
  └── 内层收敛后，更新 λ 和 μ
  └── 判断外层是否收敛（约束是否被满足）
```

### iLQR 内层收敛判定

- **梯度过小**：`normalized_grad < min_grad_thresh` 且 `mu` 足够小 → 收敛。
- **代价改进量过小**：`cost_improvement < min_cost_improvement_threshold` → 收敛。
- **正则化系数超上限** `mu > max_mu` → 求解失败。

## 2.6 代码总览：一次 `ComputePathCurve` 调用做了什么

```cpp:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
base::Optional<math::Curve2d> ClothoidPathOptimizer::ComputePathCurve(...) const {
  // ① 生成 s 采样点
  const std::vector<double> restriction_s_samples = GenerateRestrictionAndRepulsionSSample();
  // ② 生成约束/代价参数
  GenerateRestrictionAndRepulsionCostParams(...);
  // ③ 生成引导线
  const std::vector<Eigen::Vector2d> guide_line = ...;
  // ④ 生成初始猜测路径
  *initial_path_set = GenerateInitialPaths(&reference_path);
  // ⑤（可选）两阶段求解：先求可行解
  if (config_->enable_two_stage_solve()) {
    const Solver::Solution feasible_solution = GenerateFeasiblePath(...);
  }
  // ⑥ 生成完整约束代价集合
  ConstraintAndCostSet constraint_and_cost_set = GenerateConstraintsAndCosts(...);
  // ⑦ 组装成 PathProblem
  const PathProblem path_problem = ConstructPathProblem(*initial_path_set, &constraint_and_cost_set);
  // ⑧ 调用 AL-iLQR 求解器
  const Solver::Solution solution = solver.Solve(path_problem);
  // ⑨ 转换回连续曲线
  return ConvertToCurve(solution.ilqr_solution.x_trajectory);
}
```

一句话：**①~⑦ 都是"把现实路况翻译成数学问题"，⑧ 才是 AL-iLQR 算法，⑨ 是把数学解翻译回曲线**。

## 2.7 逐文件代码对应

| 文件 | 数学对应 | 作用 |
|---|---|---|
| `clothoid_path_optimizer_common.h` | 定义 $x, u$ 的下标 | 状态/控制向量的"字典" |
| `clothoid_path_dynamic.h/.cc` | $\dot x=f(x,u)$ 及 Jacobian | 连续动力学模型 |
| `clothoid_path_optimizer.h/.cc` | 整体优化问题构造与求解 | "总指挥" |
| `constraints/` | $g_k(x_k,u_k) \le 0$ | 各类硬约束（曲率、转速、半平面、光滑折线） |
| `cost_functions/` | $\ell_k(x_k,u_k)$ | 各类软代价（横向加速度/jerk、半平面、光滑折线） |
| `utils/` | 生成器 | 把决策层信息转换成 constraint/cost 对象 |
| `common/math/solver/` | AL + iLQR 求解器 | 通用求解框架 |

---

## 第二部分核心公式速查

| 概念 | 公式 |
|---|---|
| 连续动力学 | $\dot x = f(x,u)$ |
| 离散动力学 | $x_{k+1} = f_d(x_k,u_k,\Delta s)$（RK4 积分） |
| 优化问题 | $\min \ell_N(x_N)+\sum \ell_k(x_k,u_k)$ s.t. 动力学、$g_k\le0$ |
| 增强拉格朗日函数 | $\mathcal{L}_A = f(x)+\lambda^Tc(x)+\frac12c(x)^TI_\mu c(x)$ |
| 乘子更新（不等式） | $\lambda_i^+=\max\{0,\lambda_i+\mu_ic_i(x^*)\}$ |
| 罚因子更新 | $\mu^+=\phi\mu,\ \phi>1$ |
| LQR 最优反馈 | $u^*=-Kx$，$K=(R+B^TP'B)^{-1}B^TP'A$ |
| iLQR 控制增量 | $\delta u_k^*=K_k\delta x_k+d_k$ |
| 反馈增益 | $K_k=-(Q_{uu}+\rho I)^{-1}Q_{ux}$ |
| 前馈修正 | $d_k=-(Q_{uu}+\rho I)^{-1}Q_u$ |

---

# 第三部分：算法细节 —— 线搜索（Line Search）

> 第二部分讲了 AL-iLQR 的整体框架：Backward Pass 算方向、Forward Pass 走新轨迹。但"走多远"这个看似简单的问题，其实藏着大量理论细节——步长太大可能迈过头导致代价不降反升，步长太小又会导致收敛巨慢甚至不收敛。本部分专门拆解 Forward Pass 里"线搜索"这一子步骤的数学原理和代码实现。
>
> 本部分的核心结论：**回溯线搜索 + 自适应正则化 $\mu$ 构成了一套"方向-步长"协同调节机制，从理论上保证了算法的全局收敛性**。

## 3.1 为什么需要线搜索

iLQR 每一步迭代的更新公式是：

$$
\bm u_i^{new} = \bm u_i + \alpha\, d_i + K_i(\bm x_i^{new} - \bm x_i)
$$

其中 $(K_i, d_i)$ 由 Backward Pass 算出（第二部分 2.4 节），$\alpha$ 是步长系数。问题来了：**$\alpha$ 该取多少？**

- $\alpha$ 太大 → 一步迈太远，局部二次近似不再准确，代价可能不降反升，算法震荡。
- $\alpha$ 太小 → 每一步进展极小，收敛极慢，甚至可能出现"每步都下降，但永远到不了最优"的病态情况。

**线搜索（line search）** 就是专门研究"给定了方向之后，步长该取多少"的子问题的完整理论体系。

## 3.2 下降方向：往哪个方向走

### 什么是下降方向

如果沿着方向 $p_k$ 走一个很小的步子，代价函数 $f$ 的值会变小，就说 $p_k$ 是一个**下降方向（descent direction）**。数学条件：

$$
p_k^T \nabla f_k < 0
$$

直觉：$\nabla f_k$（梯度）指向函数增长最快的方向，只要 $p_k$ 和梯度方向夹角大于 90°（点积为负），沿着 $p_k$ 走一点点函数值就会减小。

### 常见的方向选法：$p_k = -B_k^{-1}\nabla f_k$

- **最速下降法**：$B_k = I$（单位矩阵），方向就是 $-\nabla f_k$。简单但收敛慢（容易"之字形"震荡）。
- **牛顿法**：$B_k = \nabla^2 f(x_k)$（Hessian 矩阵）。收敛快但计算量大。
- **拟牛顿法**：$B_k$ 是 Hessian 的近似，是速度和精度的折中。

### 为什么 $B_k$ 正定就一定是下降方向

如果 $B_k$ 正定，则：

$$
p_k^T\nabla f_k = -\nabla f_k^T B_k^{-1}\nabla f_k < 0
$$

正定矩阵的逆也是正定的，而非零向量与正定矩阵做内积结果必然大于 0，加负号后小于 0。**只要 $B_k$ 正定，$p_k$ 就一定是下降方向**——这是后续"正则化 $\mu$ 保证收敛性"的理论基石。

> **对应到代码**：在 iLQR 里，方向 $p_k$ 就是 Backward Pass 算出的 $(K_k, d_k)$，其中 $d_k = -(Q_{uu}+\mu I)^{-1}Q_u$。这里的 $Q_{uu}+\mu I$ 就是 $B_k$，$Q_u$ 对应当前步的局部梯度。

## 3.3 Wolfe 条件与回溯法

### 为什么不能只要求"下降就行"

只要求 $f(x_k+\alpha_k p_k) < f(x_k)$ 是不够的。反例：设真正的最小值是 $f^*=-1$，但迭代序列 $f(x_k) = 5/k$，虽然每步都下降，但极限是 0 而非 $-1$。**每一步下降得太少，永远到不了真正的最优点**。

### Armijo 条件（充分下降条件）——防止步长"太大"

$$
f(x_k+\alpha_k p_k) \le f(x_k) + c_1\alpha_k\nabla f_k^T p_k, \quad c_1\in(0,1)
$$

右边是一条向下倾斜的直线 $l(\alpha) = f(x_k) + c_1\alpha\nabla f_k^T p_k$。$c_1$ 通常很小（如 $10^{-4}$），这条线比真正的切线更平缓。**只要新函数值落在这条线下方，就说明下降"足够多"**。

### 曲率条件——防止步长"太小"

Armijo 条件有个漏洞：$\alpha$ 取得足够小时几乎总能满足。所以加一个条件排除"过小"的步长：

$$
\nabla f(x_k+\alpha_k p_k)^T p_k \ge c_2\nabla f_k^T p_k, \quad c_2\in(c_1,1)
$$

左边 $\nabla f(x_k+\alpha_k p_k)^T p_k$ 就是 $\phi'(\alpha_k)$——走完这一步之后沿着 $p_k$ 方向的剩余坡度。如果坡度仍然很负，说明可以继续走，停得太早了。

### Wolfe 条件 = Armijo + 曲率条件

两者合在一起就是 **Wolfe 条件**：既不让步长太大导致下降不够，也不让步长太小导致效率低下。

**Strong Wolfe 条件**是更严格的版本，把曲率条件改成绝对值形式：

$$
|\nabla f(x_k+\alpha_k p_k)^T p_k| \le c_2|\nabla f_k^T p_k|
$$

这样无论坡度是很负还是很正，只要绝对值太大就说明离驻点还太远。**Strong Wolfe 更稳健，在拟牛顿法里更常用**。

### 理论保证：一定存在满足 Wolfe 条件的步长

只要 $f$ 连续可微、$p_k$ 是下降方向、$f$ 沿射线有下界，那么**一定存在**一段区间的 $\alpha$ 同时满足 Wolfe 条件。证明核心技巧：
1. $\phi(\alpha) = f(x_k+\alpha p_k)$ 有下界，而 Armijo 直线无下界，两者**必然相交**。
2. 取第一个交点 $\alpha'$，用均值定理在 $(0,\alpha')$ 之间找到一点 $\alpha''$，能证明它满足 Wolfe 条件的两部分。

**这告诉我们回溯线搜索一定能找到解，不会陷入无解死循环。**

### 回溯法（Backtracking）：工程上最常用的简化方案

如果只需满足 Armijo 条件，有一种非常简单的实用方法：

1. 选初始步长 $\bar\alpha$（牛顿法通常从 1 开始）。
2. 设缩小因子 $\rho\in(0,1)$（如 0.5）和常数 $c\in(0,1)$。
3. 从 $\alpha=\bar\alpha$ 开始，检查是否满足 Armijo 条件。
4. 不满足就缩小 $\alpha \leftarrow \rho\alpha$，重新检查。
5. 一旦满足，采用该 $\alpha$。

**这正是 `clothoid_path_optimizer` 的 iLQR 求解器实际采用的策略！** 只不过它不是用固定 $\rho$ 反复相乘，而是**预先算好了一串候选步长表**（`alpha_values_`），依次尝试，第一个满足条件的就采用。

## 3.4 收敛性理论

### Zoutendijk 定理：线搜索理论的核心结论

设 $\theta_k$ 是搜索方向 $p_k$ 与负梯度 $-\nabla f_k$ 之间的夹角。在较宽松的假设下（$f$ 连续可微、有下界、梯度 Lipschitz 连续），**只要步长满足 Wolfe 条件**，就能推出：

$$
\sum_{k=0}^\infty \cos^2\theta_k\,\|\nabla f_k\|^2 < \infty
$$

这个级数收敛意味着它的每一项必须**趋向于 0**。也就是说：

> **要么梯度范数 $\|\nabla f_k\|\to 0$（走到了局部最优附近），要么 $\cos\theta_k\to 0$（方向和负梯度越来越垂直，方向"失效"了）。**

### 实用结论

如果能保证搜索方向和负梯度的夹角不会退化（$\cos\theta_k$ 有正下界 $\delta$），则推出 $\|\nabla f_k\|\to 0$，即**算法一定收敛到驻点**。

对于牛顿类方法 $p_k=-B_k^{-1}\nabla f_k$（$B_k$ 正定），如果条件数一致有界 $\|B_k\|\cdot\|B_k^{-1}\|\le M$，就能推出 $\cos\theta_k \ge 1/M$，结合 Zoutendijk 条件得到全局收敛。

**这就是为什么"控制矩阵条件数"在数值优化里如此重要**——如果 $B_k$（对应到 iLQR 里的 $Q_{uu}+\mu I$）病态，方向会失真，算法可能不收敛。这也解释了为什么 iLQR 代码里要加正则化项 $\mu I$：**加正则化本质上就是在改善条件数，让 $B_k$ 保持良好的正定性和条件数，从而保证收敛性**。

## 3.5 代码对应：`solver.h` 里的回溯线搜索

### 候选步长表

```cpp:modules/planning/common/math/solver/solver.h
// Values computed by "np.power(10,np.linspace(0,-3,11))"
std::vector<double> alpha_values_{1.,
                                  0.50118723,
                                  0.25118864,
                                  0.12589254,
                                  0.06309573,
                                  0.03162278,
                                  0.01584893,
                                  0.00794328,
                                  0.00398107,
                                  0.00199526,
                                  0.001};
```

这串数字是 $10^{\text{linspace}(0,-3,11)}$，从 $10^0=1$ 对数均匀递减到 $10^{-3}=0.001$，共 11 个值。**本质就是回溯法"每次乘以缩小因子"的思想，只是提前把衰减序列打好了表**。

用对数均匀分布（等比数列）而非等差数列的好处：候选步长在"大步长"和"小步长"两个量级都有足够密的采样，兼顾搜索效率和精度。

### Forward Pass：给定 $\alpha$，滚动出候选轨迹

```cpp:modules/planning/common/math/solver/solver.h
void Solver<X, U, Problem>::ForwardIteration(...) const {
  ...
  for (int i = 0; i < num_steps; ++i) {
    (*u_trajectory_updated)[i] =
        u_trajectory[i] + context.alpha * context.k_trajectory[i] +
        context.K_trajectory[i] * ((*x_trajectory_updated)[i] - x_trajectory[i]);
    ...
    (*x_trajectory_updated)[i + 1] =
        problem.dynamic_model(i).Evaluate((*x_trajectory_updated)[i], (*u_trajectory_updated)[i]);
  }
}
```

注意：**$\alpha$ 只缩放前馈项 $d_k$，不缩放反馈项 $K_k$**——反馈项负责实时修正状态偏差，不需要也不应该被步长衰减。

### 判断 $\alpha$ 是否可接受：Armijo 条件的比值化实现

```cpp:modules/planning/common/math/solver/solver.h
for (double alpha : alpha_values_) {
  ...
  meta.alpha = alpha;
  ForwardIteration(...);           // 用当前 alpha 滚动出候选轨迹
  cost_updated = ComputeCost(...); // 算出候选轨迹的真实代价
  meta.cost_improvement = cost_history.back() - cost_updated;               // 实际下降量
  meta.expected_cost_improvement = -meta.alpha * (meta.dV[0] + meta.alpha * meta.dV[1]); // 预期下降量
  meta.cost_improvement_ratio = meta.cost_improvement / meta.expected_cost_improvement;
  if (meta.cost_improvement_ratio > line_search_min_cost_improvement_ratio_) {
    is_forward_succeeded = true;
    break;   // 接受这个 alpha
  }
}
```

| 代码变量 | 数学含义 |
|---|---|
| `cost_history.back()` | $f(x_k)$，走这一步之前的代价 |
| `cost_updated` | $f(x_k+\alpha p_k)$，走这一步之后的代价 |
| `meta.cost_improvement` | 实际下降量 $f(x_k) - f(x_k+\alpha p_k)$ |
| `meta.expected_cost_improvement` | 预期下降量（由 Backward Pass 的二阶近似预测） |
| `meta.cost_improvement_ratio` | 实际下降 / 预期下降，本质是 Armijo 条件的"比值"版本 |
| `line_search_min_cost_improvement_ratio_` | 判定阈值，对应 Armijo 条件里的常数 $c_1$ |

`cost_improvement_ratio > line_search_min_cost_improvement_ratio_` 这一行代码，本质上就是 **Armijo 充分下降条件的比值化实现**：不是简单要求"下降"，而是要求"实际下降量占预期下降量的比例超过某个阈值"，用来避免"下降太慢导致不收敛"的病态情况。

> **注**：这里没有单独实现"曲率条件"，因为在信赖域/正则化风格的 iLQR 里，"步长过小导致效率低"是通过**动态调整正则化 $\mu$** 来解决的——$\mu$ 变小时 $\alpha=1$ 更容易被接受，天然避免了"每次都用很小步长"的低效问题。

### 正则化 $\mu$ 如何充当"信赖域"

```cpp:modules/planning/common/math/solver/solver.h
void Solver::IncreaseRegularization(IterationMeta* meta) const {
  meta->delta_mu = std::max(delta_mu_factor_, meta->delta_mu * delta_mu_factor_);
  meta->mu = std::max(min_mu_, meta->mu * meta->delta_mu);
}

void Solver::ReduceRegularization(IterationMeta* meta) const {
  meta->delta_mu = std::min(1.0 / delta_mu_factor_, meta->delta_mu / delta_mu_factor_);
  if (meta->mu * meta->delta_mu > min_mu_) {
    meta->mu *= meta->delta_mu;
  } else {
    meta->mu = 0.0;
  }
}
```

- **$\mu$ 越大** → $Q_{uu,reg} = Q_{uu} + \mu I$ 越接近"单位矩阵的倍数" → 方向 $p_k = -Q_{uu,reg}^{-1}Q_u$ 越接近**最速下降方向**。最速下降方向虽然收敛慢，但**极其稳健**，适合在离最优点远、非线性强的地方使用。
- **$\mu$ 越小（趋于 0）** → $Q_{uu,reg}$ 越接近真实的 $Q_{uu}$ → 方向越接近**纯牛顿方向**，收敛快，但要求局部近似足够准。

**自适应正则化策略**：线搜索连续失败 → 说明局部近似不准 → 增大 $\mu$，退化成更保守的方向；线搜索成功 → 减小 $\mu$，下次尝试更接近牛顿法的"激进"步子。这本质上和数值优化里经典的**信赖域方法（trust region method）**思想相通：$\mu$ 越大等价于信赖域越小。

### 一次迭代的完整流程

```
外层 for 循环（最多 max_num_iter_ 次）
  │
  ├─① Backward Pass（选方向）
  │    ├─ 计算 Q_uu_reg = Q_uu + mu*I
  │    ├─ 尝试 Cholesky 分解求逆
  │    │    ├─ 失败（不正定）→ IncreaseRegularization(mu 变大) → 重新算
  │    │    └─ 成功 → 得到 k_trajectory(d_k), K_trajectory(K_k)
  │    └─ 得到本轮的下降方向 p_k = (k, K)
  │
  ├─ 检查梯度是否已经足够小 → 是则收敛退出
  │
  ├─② Forward Pass + 回溯线搜索（选步长）
  │    └─ 遍历 alpha_values_ = [1, 0.5, 0.25, ..., 0.001]
  │         ├─ 用当前 alpha 做 ForwardIteration，得到候选轨迹
  │         ├─ 计算 cost_improvement_ratio = 实际下降/预期下降
  │         ├─ 满足阈值 → 接受这个 alpha，break（对应 Armijo 条件）
  │         └─ 不满足 → 换下一个更小的 alpha，继续试
  │
  ├─③ 根据 Forward Pass 是否成功，更新状态
  │    ├─ 成功 → 采用新轨迹，ReduceRegularization(mu 变小，下次更激进)
  │    └─ 失败 → IncreaseRegularization(mu 变大，下次更保守)，mu 超上限则放弃
  │
  └─ 判断是否收敛 → 是则退出，否则回到①
```

### 常见疑问

**Q1：为什么不用完整的 Wolfe 条件（Armijo + 曲率条件），而是只用"下降比值"？**

`cost_improvement_ratio` 已经隐含了 Armijo 条件的核心精神。而曲率条件在这套 AL-iLQR 框架里，被**外层自适应调整 $\mu$** 等价替代：$\mu$ 小的时候 $\alpha=1$ 通常就能被接受，天然避免了"每次都用很小步长"的低效问题。

**Q2：`alpha_values_` 为什么是等比数列而非等差数列？**

等差数列在需要非常小步长的场景覆盖不到。等比数列（对数均匀）让候选步长在"大步长"和"小步长"两个量级都有足够密的采样，这是数值优化里选取候选步长表的常见技巧。

**Q3：所有 $\alpha$ 都不满足条件会怎样？**

`is_forward_succeeded` 保持 `false`，回到主循环执行 `IncreaseRegularization`，把 $\mu$ 调大后重新做 Backward Pass。如果 $\mu$ 最终超过 `max_mu_` 还是不行，则求解失败，上层走 Fallback 逻辑。

**Q4：`mu` 怎么保证 Zoutendijk 定理里的"$\cos\theta_k$ 不能太小"？**

只要 $\mu$ 在合理范围内，$Q_{uu,reg}$ 的条件数就不会失控，从而保证方向 $p_k$ 和负梯度的夹角不会退化。**正则化不仅是"防止矩阵不可逆"的数值技巧，更是从理论上保证算法全局收敛性的关键环节。**

---

## 第三部分核心公式速查

| 概念 | 公式 |
|---|---|
| 迭代更新 | $x_{k+1}=x_k+\alpha_kp_k$ |
| 下降方向条件 | $p_k^T\nabla f_k<0$ |
| 常见方向 | $p_k=-B_k^{-1}\nabla f_k$ |
| Armijo 条件 | $f(x_k+\alpha_kp_k)\le f(x_k)+c_1\alpha_k\nabla f_k^Tp_k$ |
| 曲率条件 | $\nabla f(x_k+\alpha_kp_k)^Tp_k\ge c_2\nabla f_k^Tp_k$ |
| Strong Wolfe | $|\nabla f(x_k+\alpha_kp_k)^Tp_k|\le c_2|\nabla f_k^Tp_k|$ |
| 回溯法 | $\alpha\leftarrow\rho\alpha$ 直到满足 Armijo 条件 |
| 方向与负梯度夹角 | $\cos\theta_k=\dfrac{-\nabla f_k^Tp_k}{\|\nabla f_k\|\|p_k\|}$ |
| Zoutendijk 条件 | $\sum_{k=0}^\infty\cos^2\theta_k\|\nabla f_k\|^2<\infty$ |
| 全局收敛结论 | $\cos\theta_k\ge\delta>0 \Rightarrow \|\nabla f_k\|\to0$ |
| 代码中的下降比值 | `cost_improvement_ratio = 实际下降 / 预期下降` |

---

# 第四部分：工程实现全景

> 前三部分讲了从变分法到 AL-iLQR 的完整理论，但"理论怎么变成代码"是一个独立的工程问题。本部分聚焦于 `clothoid_path_optimizer/` 目录下的实际代码，回答：代码怎么组织的？约束和代价具体有哪些？折线不可导的数值难题怎么解？决策层给的高层信息怎么翻译成求解器能吃的输入？完整求解流程走一遍？
>
> 核心线索：**几何层**提供可导距离函数 → **约束/代价层**用这些距离函数定义具体约束和代价 → **生成器层**把决策层信息批量转换成约束代价对象 → **主类**把所有东西打包成 `ConstrainedProblem` 交给求解器。

## 4.1 整体架构与文件地图

```
clothoid_path_optimizer/
├── clothoid_path_optimizer.h/.cc        # 主类：串联全流程的入口
├── clothoid_path_optimizer_common.h     # 状态/控制量维度定义、公共数据结构
├── clothoid_path_dynamic.h/.cc          # Clothoid 连续动力学模型 f(x,u)（ILQR 用）
├── decision_input.h                     # 决策层输入的数据结构（车道线/guide line/障碍物决策）
├── geometry/                            # 处处可导的几何距离工具（折线、坐标平移）
│   ├── differentiable_line_segment.*    # 单条线段的可导距离函数
│   ├── point_translator.h               # 车体坐标平移（后轴→四角/前轴等）
│   └── smooth_polyline.h                # 多线段折线的光滑距离场（核心几何库）
├── math/
│   └── log_sum_exp.h                    # softmax/soft-min 数学工具（光滑折线的基石）
├── constraints/                         # 各类硬约束（曲率、转速、半平面、光滑折线）
├── cost_functions/                      # 各类软代价（横向加速度/jerk、半平面、光滑折线）
└── utils/                               # "生成器"：把决策层的高层信息转成 constraint/cost 对象
    ├── kinematic_constraint_and_cost_generator.*   # 车辆能力+舒适性
    ├── restriction_constraint_and_cost_generator.* # 可行驶区域边界
    ├── repulsion_cost_generator.*                  # 软避让区域
    ├── lane_boundary_cost_generator.*               # 车道边界
    ├── obstacle_constraint_and_cost_generator.*     # 障碍物
    ├── box_collision_constraint_and_cost_generator.* / generate_box_collision_params.* # 四角碰撞
    ├── guide_line_cost_generator.*                  # 引导线（贴近参考路径）
    ├── generate_restriction_cost_params.* / generate_repulsion_cost_params.*           # 参数预处理
    ├── polyline_util.*                              # 折线基础工具
    └── clothoid_path_optimizer_debugger.*           # 调试信息填充
```

**依赖的通用求解器框架**（不在本目录，位于 `modules/planning/common/math/solver/`）：

```
common/math/solver/
├── continuous_dynamics.h        # 连续动力学模型基类接口
├── integrator.h                 # 数值积分器（RungeKutta4 等）
├── discretized_model.h          # 把 连续模型+积分器 封装成离散 DynamicModel
├── cost_function.h              # 代价函数基类接口
├── solver.h                     # ILQR 求解器本体
└── augmented_lagrangian/
    ├── constraint.h / inequality_constraint.h  # 约束基类，内置 AL 惩罚项计算
    ├── al_problem.h / constrained_problem.h    # 问题描述（把约束+代价打包）
    └── al_solver.h                              # AL 外层循环，反复调用 ILQR 求解器
```

**一句话概括整体分层**：**几何层（geometry/math）** 提供可导的距离函数 → **约束/代价层（constraints/cost_functions）** 用这些距离函数加上运动学公式定义具体的约束和代价 → **生成器层（utils）** 把决策层给的道路元素批量转换成一组约束和代价对象 → **主类（clothoid_path_optimizer.cc）** 把所有约束代价按采样点打包成一个 `ConstrainedProblem`，交给 **求解器（common/math/solver）** 用 AL+iLQR 求解。

## 4.2 约束与代价完整清单

下表汇总了本模块用到的所有约束/代价，包含来源、数学公式、代码文件：

| 来源 | 名称 | 类型 | 公式 | 代码文件 |
|---|---|---|---|---|
| 车辆能力 | 曲率约束 | 硬约束 | $-\kappa_{max}\le\kappa\le\kappa_{max}$ | `constraints/clothoid_curvature_constraint.*` |
| 车辆能力 | 转向速率约束 | 硬约束 | $-\delta'_{max}\le\delta'\le\delta'_{max}$ | `constraints/clothoid_steering_rate_constraint.*` |
| 舒适性 | 横向加速度代价 | 软代价（死区二次） | 见 4.2.3 | `cost_functions/clothoid_lateral_acceleration_cost_function.*` |
| 舒适性 | 横向 jerk 代价 | 软代价（死区二次） | 见 4.2.4 | `cost_functions/clothoid_lateral_jerk_cost_function.*` |
| 舒适性 | dκ/ds 代价 | 软代价 | $J=\tfrac12 w\omega^2,\ w\propto v^6$ | `utils/kinematic_constraint_and_cost_generator.*` 内联生成 |
| 舒适性 | ddκ/dds 代价 | 软代价 | $J=\tfrac12 w\alpha^2,\ w\propto v^8$ | 同上 |
| 道路元素 | 半平面约束/代价 | 硬约束+软代价 | 见 4.2.5 | `constraints/half_plane_constraint.h`, `cost_functions/half_plane_cost_function.h` |
| 道路元素 | 光滑折线约束/代价 | 硬约束+软代价 | 见 4.3 | `constraints/smooth_polyline_constraint.h`, `cost_functions/smooth_polyline_cost_function.h` |

### 4.2.1 曲率约束（Curvature Constraint）

$$
-\kappa_{max}\le\kappa\le\kappa_{max}
$$

拆成两个 $c\le0$ 形式的不等式：

$$
c_0 = \kappa - \kappa_{max} \le 0, \qquad c_1 = -\kappa - \kappa_{max} \le 0
$$

```cpp:modules/planning/path/clothoid_path_optimizer/constraints/clothoid_curvature_constraint.cc
ClothoidCurvatureConstraint::VecC ClothoidCurvatureConstraint::Evaluate(const VecX& x,
                                                                        const VecU& u) const {
  VecC c(kDimension);
  c(0) = x(ClothoidPathStateIndex::kKappa) - kappa_max_;
  c(1) = -x(ClothoidPathStateIndex::kKappa) - kappa_max_;
  return c;
}
```

雅可比非常简单：$\partial c_0/\partial\kappa=1$，$\partial c_1/\partial\kappa=-1$，其余全为 0。这个约束的 `PenaltyScale`（AL 初始惩罚缩放）设得比较大（`1000.0`），说明这是"绝对不能违反"的硬约束优先级最高。

### 4.2.2 转向速率约束（Steering Rate Constraint）

单车模型下前轮转角 $\delta$ 与曲率的关系为 $\kappa = \tan\delta / L$（$L$ 为轴距）。对时间求导：

$$
\frac{d\kappa}{ds}\frac{ds}{dt} = \frac{1}{L\cos^2\delta}\frac{d\delta}{dt}
$$

代入 $ds/dt=v$，以及 $\kappa^2L^2+1 = 1/\cos^2\delta$，整理得到前轮转角速率：

$$
\delta' = \frac{vL\omega}{\kappa^2L^2+1}
$$

约束为 $-\delta'_{max}\le\delta'\le\delta'_{max}$，及偏导数：

$$
\frac{\partial\delta'}{\partial\kappa} = -\frac{2vL^3\omega\kappa}{(\kappa^2L^2+1)^2}, \qquad \frac{\partial\delta'}{\partial\omega} = \frac{vL}{\kappa^2L^2+1}
$$

代码实现（变量名与公式符号一一对应：`wheel_base_`=$L$，`reference_speed_`=$v$，`x(kDKappa)`=$\omega$）：

```cpp:modules/planning/path/clothoid_path_optimizer/constraints/clothoid_steering_rate_constraint.cc
const double kappa_wheelbase = x(ClothoidPathStateIndex::kKappa) * wheel_base_;
const double steering_rate =
    (reference_speed_ * wheel_base_ * x(ClothoidPathStateIndex::kDKappa)) /
    (kappa_wheelbase * kappa_wheelbase + 1.0);
VecC c(kDimension);
c(0) = (steering_rate - steering_rate_max_);
c(1) = (-steering_rate - steering_rate_max_);
```

**数值验证**（来自单测）：$L=2.5,\ \delta'_{max}=2.0,\ v=1.7,\ \kappa=0.6,\ \omega=0.7$：

- $\kappa L = 1.5$，分母 $\kappa^2L^2+1=3.25$
- $\delta' = 1.7\times2.5\times0.7/3.25 = 0.91538$
- $c_0=0.91538-2.0=-1.08462\le0$（可行），$c_1=-2.91538\le0$（可行）

### 4.2.3 横向加速度代价（Lateral Acceleration Cost）

$$
a_{lat} = \kappa v^2
$$

选定阈值 $\tilde a_{lat}$（如 $2\,\text{m/s}^2$），超过阈值才惩罚（死区二次，deadzone quadratic）：

$$
J=\begin{cases}
\tfrac12 w(a_{lat}+\tilde a_{lat})^2, & a_{lat}<-\tilde a_{lat}\\
0, & -\tilde a_{lat}\le a_{lat}\le\tilde a_{lat}\\
\tfrac12 w(a_{lat}-\tilde a_{lat})^2, & a_{lat}>\tilde a_{lat}
\end{cases}
$$

```cpp:modules/planning/path/clothoid_path_optimizer/cost_functions/clothoid_lateral_acceleration_cost_function.cc
const double kappa = x(ClothoidPathStateIndex::kKappa);
const double lateral_acceleration = kappa * reference_speed_ * reference_speed_;
double cost = 0.0;
if (lateral_acceleration > lateral_acceleration_limit_) {
  const double delta_lateral_acceleration = lateral_acceleration - lateral_acceleration_limit_;
  cost = 0.5 * lateral_acceleration_weight_ * delta_lateral_acceleration * delta_lateral_acceleration;
} else if (lateral_acceleration < -lateral_acceleration_limit_) {
  const double delta_lateral_acceleration = lateral_acceleration + lateral_acceleration_limit_;
  cost = 0.5 * lateral_acceleration_weight_ * delta_lateral_acceleration * delta_lateral_acceleration;
}
```

梯度/Hessian（对 $\kappa$ 求导）：$\partial J/\partial\kappa = w\Delta a\cdot v^2$，$\partial^2 J/\partial\kappa^2 = wv^4$。

### 4.2.4 横向 Jerk 代价（Lateral Jerk Cost）

对 $a_{lat}$ 关于时间求导得到横向 jerk：

$$
j_{lat} = \omega v^3 + 2\kappa v\, a_{lon}
$$

由于优化器里没有精确的纵向加速度 $a_{lon}$，忽略第二项，近似为：

$$
j_{lat}\approx \omega v^3
$$

同样是死区二次代价，形式与横向加速度完全同构（阈值改为 $\tilde j_{lat}$）。梯度/Hessian：$\partial J/\partial\omega = w\Delta j\cdot v^3$，$\partial^2 J/\partial\omega^2=wv^6$。

### 4.2.5 dκ/ds 与 ddκ/dds 代价

为了减小方向盘转速，增加：

$$
J = \tfrac12 w\omega^2,\quad w\propto v^6
$$

为了减小方向盘小幅高频抖动，增加：

$$
J = \tfrac12 w\alpha^2,\quad w\propto v^8
$$

**理解这两个比例关系的物理意义**：如果把 $w=c\cdot v^6$ 代入第一个公式，$J = \tfrac12 c(v^3\omega)^2 = \tfrac12 c\, j_{lat}^2$，也就是说这本质上**等价于对横向 jerk 做了一个（无死区的）二次惩罚**——用一个简单的二次型近似替代了 4.2.4 节的死区二次代价，双管齐下让曲率变化更平顺。同理第二个公式等价于对 $j_{lat}$ 的变化率做惩罚。

### 4.2.6 半平面约束/代价（Half Plane）

**几何概念**：一个半平面由一个参考点 $\bm p_0$ 和单位法向量 $\hat{\bm n}$ 定义，把平面切成两侧：

$$
H=\{\bm p \mid \hat{\bm n}\cdot(\bm p-\bm p_0)\le0\}
$$

$\hat{\bm n}$ 指向的一侧是"禁止/违反"侧。典型应用场景是把一段近似笔直的车道边界表达成一个半平面。

**约束形式**：$c=\hat{\bm n}\cdot(\bm p-\bm p_0)\le0$，其中 $\bm p=(x,y)$ 可以是后轴中心，也可以通过 `PointTranslator`（4.3.4 节介绍）平移到车辆的前轴中心、角点等任意部位。

```cpp:modules/planning/path/clothoid_path_optimizer/constraints/half_plane_constraint.h
const Eigen::Vector2d position = x.template head<2>();
c(0) = normal_.dot(position - point_);
```

**代价形式**（死区二次，可行侧为零）：

$$
J=\begin{cases}0,& d\le0\\ \tfrac12 w d^2, & d>0\end{cases}, \quad d=\hat{\bm n}\cdot(\bm p-\bm p_0)
$$

**平移版本**：三种预定义别名对应车辆的三个关键位置：

| 别名 | 平移类型 | 用途 |
|---|---|---|
| `FrontCenterHalfPlaneConstraint` | 仅纵向偏移 | 约束前轴中心 |
| `RearWheelHalfPlaneConstraint` | 仅横向偏移 | 约束后轮 |
| `CornerHalfPlaneConstraint` | 纵向+横向偏移 | 约束车辆四角 |

半平面约束/代价的 `PenaltyScale`（`1.0`）比曲率/转速约束（`1000.0`）小很多，因为半平面约束会在折线上密集采样生成大量约束点，单个约束不宜给太重的初始惩罚。

## 4.3 几何工具库：让"折线距离"处处可导

### 4.3.1 为什么需要专门做这件事

道路边界、限制区域、排斥区域等绝大多数道路元素都可以用一条**折线（polyline）**来描述。最自然的想法是直接用"点到折线的欧氏距离"构造约束/代价——但这个距离函数在折线的**拐点（两条线段的连接处）**是不可导的（分段函数在连接点两侧梯度方向不同），而 ILQR 需要代价函数的解析 Hessian。如果直接用不可导的距离函数，求解器在轨迹经过拐点附近时会数值不稳定甚至失败。

解决思路分两层：
1. **单条线段**（`DifferentiableLineSegment`）：先把"点到单条线段"的距离写成分段解析函数，各段内部精确可导（但线段自身在两端连接的地方仍不连续）。
2. **多条线段组合**（`SmoothPolyline`）：用 **LogSumExp（对数和指数）** 技巧把多条线段的距离做**软最小值（soft-min）**融合，从而在整条折线上获得**处处一阶、二阶连续可导**的距离场。

### 4.3.2 LogSumExp：数学地基

标准 soft-max（当 $\lambda\to\infty$ 时逼近真正的 $\max$）：

$$
\text{LSE}_\lambda(\bm x) = \frac1\lambda\ln\sum_i e^{\lambda x_i}
$$

代码里实现了一个数值更稳定、也更贴近真值的变体：

```cpp:modules/planning/path/clothoid_path_optimizer/math/log_sum_exp.h
double Evaluate(const Eigen::Matrix<double, N, 1>& x) const {
  const int size = x.size();
  DCHECK_GE(size, 1);
  const double max_coeff = x.maxCoeff();
  const double normalized_sum_exp = ((lambda_ * (x.array() - max_coeff)).exp()).sum();
  return max_coeff + std::log1p(normalized_sum_exp - 1.0) / lambda_;
}
```

数学上：

$$
\text{LSE}_\lambda(\bm x) = m + \frac1\lambda\ln\Big(\sum_i e^{\lambda(x_i-m)} - 1\Big), \quad m=\max(\bm x)
$$

**两个工程细节**：
1. **减去最大值 $m$ 再算指数**：这是经典的 log-sum-exp 数值稳定技巧，避免 $e^{\lambda x_i}$ 在 $x_i$ 较大时溢出（此时 $e^{\lambda(x_i-m)}\le1$，恒定不溢出）。
2. **减 1 再用 `log1p`**：因为 $\sum_i e^{\lambda(x_i-m)}$ 中必有一项（最大值对应项）恰好为 $e^0=1$，减去这个 1 再用 `log1p`（比直接 `log(1+y)` 数值更精确）能让结果更贴近真实的 $\max$（因为其余各项都是"多余"的贡献）。

**梯度就是标准 softmax 权重**：

$$
\frac{\partial\text{LSE}_\lambda}{\partial x_i} = \frac{e^{\lambda(x_i-m)}}{\sum_j e^{\lambda(x_j-m)}} = \text{softmax}(\lambda\bm x)_i
$$

**Hessian 是 softmax 的标准 Hessian**：

$$
\mathbf H_{\text{LSE}} = \lambda\big(\text{diag}(\bm s)-\bm s\bm s^T\big), \quad \bm s=\text{softmax}(\lambda\bm x)
$$

**数值直觉**：输入 `[1,2,3,4]`，$\lambda=5$，最大值 4 占绝对主导（$e^{5\times(-1)}=0.0067$ 这类项很小），所以 $\text{LSE}\approx4.0014$，非常接近真实最大值 4，梯度几乎是 one-hot（接近 `[0,0,0,1]`）。

### 4.3.3 DifferentiableLineSegment：单条线段的可导距离

设线段端点为 $\bm s,\bm e$，方向 $\hat{\bm t}=(\bm e-\bm s)/L$，法线 $\hat{\bm n}=(-\hat t_y,\hat t_x)$（逆时针转 90°，指向线段左侧）。

给定查询点 $\bm p$，先算它在线段方向上的投影长度 $t=\hat{\bm t}\cdot(\bm p-\bm s)$，分三种情况：

| 情况 | 条件 | 最近点 | 距离 | 梯度 |
|---|---|---|---|---|
| 垂足在起点前 | $t\le0$ | $\bm s$ | $\lVert\bm p-\bm s\rVert$ | $(\bm p-\bm s)/\lVert\bm p-\bm s\rVert$ |
| 垂足在线段内 | $0<t<L$ | $\bm s+\frac{t}{L}(\bm e-\bm s)$ | $\lvert\hat{\bm n}\cdot(\bm p-\bm s)\rvert$ | $\text{sign}(\cdot)\hat{\bm n}$ |
| 垂足在终点后 | $t\ge L$ | $\bm e$ | $\lVert\bm p-\bm e\rVert$ | $(\bm p-\bm e)/\lVert\bm p-\bm e\rVert$ |

```cpp:modules/planning/path/clothoid_path_optimizer/geometry/differentiable_line_segment.cc
Eigen::Vector2d DifferentiableLineSegment::GetNearestPoint(const Eigen::Vector2d& point,
                                                           ProjectionType* projection_type) const {
  const double inner_prod = direction_.dot(point - start_);
  if (inner_prod <= 0.0) { ...; return start_; }
  if (inner_prod >= length_) { ...; return end_; }
  ...
  return math::Lerp(start_, end_, inner_prod / length_);
}
```

**Hessian**：线段内部（垂足落在线段上）时距离是位置的线性函数，Hessian 恒为零矩阵；端点外侧则退化为"点到点距离"的标准 Hessian $\frac{\bm I - \hat{\bm v}\hat{\bm v}^T}{\lVert\bm v\rVert}$。

**关键认知**：单条线段本身的距离梯度/Hessian 在 $t=0$ 和 $t=L$ 处是不连续的（跳变），这个不连续性正是留给 `SmoothPolyline` 用 LogSumExp 去抹平的。

### 4.3.4 SmoothPolyline：把多条线段拼成一条处处光滑的折线

给定一串点 $\{p_0,\ldots,p_n\}$，构造 $n-1$ 条 `DifferentiableLineSegment`。核心操作是把点到各线段的距离向量 $\bm d(\bm p)=(d_1,\ldots,d_N)$ 用 **soft-min** 融合：

$$
d_{\text{smooth}}(\bm p) = -\text{LSE}_\lambda(-\bm d(\bm p)) = \min(\bm d) + \frac1\lambda\ln\Big(\sum_ie^{\lambda(\min(\bm d)-d_i)}-1\Big)
$$

```cpp:modules/planning/path/clothoid_path_optimizer/geometry/smooth_polyline.h
double SmoothPolyline<N>::SmoothDistance(const Eigen::Vector2d& point) const {
  const Eigen::Matrix<double, N, 1> minus_distance = -Distances(point);
  return -log_sum_exp_.Evaluate(minus_distance);
}
```

温度系数取 `kLambdaForLSE = 10.0`（λ 越大，soft-min 越逼近真实最小值，但拐点附近可导的过渡区间越窄；λ 越小反之），这是一个经过验证的工程折中值。

**梯度**：链式法则展开后是各线段梯度按 softmax 权重加权求和——最近的那条线段权重接近 1，远处线段权重接近 0：

$$
\nabla_{\bm p}d_{\text{smooth}} = \sum_i w_i\nabla_{\bm p}d_i, \qquad w_i=\text{softmax}(-\lambda\bm d)_i
$$

**Hessian**：由两部分组成——各线段 Hessian 的加权和，加上 softmax 权重本身随位置变化带来的额外曲率项（这一项恰好在拐点附近起主导作用，负责把两条线段梯度方向的"跳变"抹平成光滑过渡）。

**左右侧判断**：`SmoothDistance` 返回的是无符号（非负）距离，是否在折线的"违反侧"要另外调用 `IsPointOnRightSide`，本质是找到最近线段后看点相对该线段法线的位置。拐角处等距归属的判断用了**角平分线**修正（避免最近线段判断在拐角处出现"跳变式"的错误归属）。

**数值直觉**（折线形如 "⊂"，见 `smooth_polyline_test.cc`）：点 (1,0) 距最近线段的欧氏距离恰为 1.0，`SmoothDistance` 算出约 0.999995，几乎无偏差，因为此时只有一条线段占绝对主导权重；点 (3,0) 靠近拐角，两条线段距离接近，`SmoothDistance` 约为 0.996769，比真实最小值 1.0 略小——这就是 soft-min 在拐角附近产生的"温和低估"，正是为了让梯度/Hessian 保持连续所付出的代价。

### 4.3.5 PointTranslator：把约束/代价施加到车辆任意部位

车辆状态里的 $(x,y,\theta)$ 描述的是后轴中心，但很多约束（比如碰撞约束）需要施加在车辆的四个角点或前轴中心上。`PointTranslator` 提供从车体坐标系偏移 $(\Delta x,\Delta y)$ 到全局坐标的变换：

$$
\bm f(\bm q) = \begin{pmatrix}x+\cos\theta\,\Delta x-\sin\theta\,\Delta y\\ y+\sin\theta\,\Delta x+\cos\theta\,\Delta y\end{pmatrix}, \quad \bm q=(x,y,\theta)
$$

Jacobian：

$$
\mathbf J = \frac{\partial\bm f}{\partial\bm q} = \begin{pmatrix}1&0&-\Delta y_w\\0&1&\Delta x_w\end{pmatrix}
$$

三种平移类型 `kLongitudinal`（只有 $\Delta x$，用于前轴中心）、`kLateral`（只有 $\Delta y$，用于后轮/车道边缘）、`kBoth`（两者都有，用于四角点）。因为坐标变换本身含有 $\theta$ 的三角函数，是非线性的，`PointTranslator` 还额外提供了 Jacobian 对状态量的二阶导，配合链式法则可以推出完整的二阶 Hessian，而不是仅用 Gauss-Newton 近似。

**用途**：所有 `constraints/` 和 `cost_functions/` 里带 `TranslatedPoint` 前缀或 `FrontCenter/RearWheel/Corner` 前缀的类，都是"基础版约束/代价 + `PointTranslator`"的组合，通过链式法则把对目标点坐标的梯度/Hessian 转换回对车辆状态 $(x,y,\theta)$ 的梯度/Hessian。

## 4.4 utils/：决策层信息组装流水线

`utils/` 目录是"胶水层"：上游决策模块给出的都是高层语义信息（车道边界在哪、这块区域要不要避让、这个障碍物往左还是往右绕开、留多少 buffer），这一层负责把它们转换成 ILQR 能直接使用的 `Constraint`/`CostFunction` 对象数组，按每个采样点组织起来。

### 4.4.1 两层结构：参数生成 → 约束代价生成

- **参数生成层**（`generate_restriction_cost_params.*`、`generate_repulsion_cost_params.*`、`generate_box_collision_params.*`）：把原始输入（Restriction/Repulsion/box collision 可行区间函数）在一批 s 采样点上转换成结构化的中间参数（`RestrictionCostParams`、`RepulsionCostParams`、`BoxCollisionParams`，定义在 `clothoid_path_optimizer_common.h`），主要做**坐标转换**（s-l → x-y）和**采样**。
- **约束代价生成层**（各个 `*_generator.*`）：拿着这些中间参数（或直接拿原始决策输入），针对每个轨迹采样点生成具体的 `Constraint`/`CostFunction` 对象。

### 4.4.2 KinematicConstraintAndCostGenerator：车辆能力 + 舒适性打包

这是"车辆能力+舒适性"这一大类的统一组装入口，对每个采样点（跳过初始点 $i=0$，因为初始状态是给定的边界条件不需要约束）分别生成：

- 曲率约束（`ClothoidCurvatureConstraint`）
- 转向速率约束（`ClothoidSteeringRateConstraint`，需要用到该采样点对应的参考速度）
- 横向加速度代价（`ClothoidLateralAccelerationCostFunction`）
- 横向 jerk 代价（`ClothoidLateralJerkCostFunction`）
- dκ/ds、ddκ/dds 二次代价（权重按速度高次幂预先计算好）

每个采样点用的参考速度来自 `reference_speed_profile`（见 4.4.6 节），这就是为什么这一步需要传入 `reference_path` 和 `reference_speed_profile` 两个参数一起使用——速度决定了转向速率约束的具体数值、也决定了舒适性代价的具体数值。

### 4.4.3 五类道路元素 generator 的区别

| Generator | 对应元素 | 硬约束/软代价 | 车辆模型 |
|---|---|---|---|
| `RestrictionConstraintAndCostGenerator` | 可行驶区域左右边界（如车道边界、路口边界） | 硬约束 + 软代价 | 后轴单点 |
| `RepulsionCostGenerator` | 软性避让区域（如靠近但不禁止） | 纯软代价 | 后轴单点 |
| `LaneBoundaryCostGenerator` | 车道线（区分实线/虚线） | 纯软代价 | 四角模型 |
| `BoxCollisionConstraintAndCostGenerator` | 精确碰撞边界（四角避障） | 硬约束 + 软代价 | 四角模型 |
| `ObstacleConstraintAndCostGenerator` | 具体障碍物（带 nudge 方向和 buffer） | 硬约束 + 软代价 | 四角模型 |

**Restriction vs Repulsion 的区别**：Restriction 是"绝对不能越过的边界"（比如道路物理边界），必须同时有硬约束兜底；Repulsion 是"希望离得越远越好，但不是不可逾越"（比如临时性的软避让区域），只用代价表达，不设硬约束。

**为什么车道边界、碰撞边界、障碍物都用"四角模型"**：车辆是一个有长宽的矩形，只用后轴中心（一个点）去判断是否碰撞，在狭窄空间或大转弯时会严重低估实际占用空间；用车辆四个角点分别去检查与边界/障碍物的距离，能显著提升狭窄空间的通过能力和安全性。

### 4.4.4 四角碰撞（Box Collision）的具体做法

`generate_box_collision_params.*` 先把车辆矩形轮廓换算成四个关键点（左前、右前、左后、右后），每个点通过 `PointTranslator` 用 `kBoth` 类型从后轴中心偏移得到。然后为每个角点分别在左右两侧生成一条对应的碰撞边界折线（`right_box_collision_boundaries` / `left_box_collision_boundaries`）。

`BoxCollisionConstraintAndCostGenerator` 再基于这些折线，对每个角点分别生成 `SmoothPolyline` 约束（硬性不可碰撞）和代价（软性远离边界）。

### 4.4.5 GuideLineCostGenerator：贴近引导线的代价

引导线（guide line）是一个泛指：可以是参考线、车道中心线，也可以是路口跟车/变道时决策层给的引导线。目标很简单——path 上的每个采样点尽可能贴近这条线：

$$
J = \tfrac12 w\, d^2, \quad d=\text{点到guide line的距离}
$$

代码里对起点附近和终点分别有特殊处理（比如终点用更大的权重收紧，保证 path 末端与引导线对齐），并且用 Huber 型代价避免远离引导线时代价增长过快压制其他更重要的约束。

### 4.4.6 Reference Speed Profile：piecewise jerk 速度预估

优化器只知道起始点的精确速度（车辆当前速度），后续采样点没有精确速度信息，但舒适性代价（横向加速度/jerk）恰恰需要速度参与计算。解决方法是预先用一个简化模型（**piecewise jerk**）估计一条速度曲线：用恒定 jerk 把加速度积分到 0，然后匀速行驶（如果当前是刹车状态，则先用恒定 jerk 到达匀减速、再到低速保持），得到的是 v-t 曲线，再通过时间积分转换成 v-s 曲线供后续插值使用。

有一个例外情况：当参考线上探测到大曲率（意味着接下来速度可能有较大变化，比如进入急转弯），直接复用上一帧规划出的速度曲线，避免预估模型在这种场景下产生不合理的远端速度估计。

## 4.5 完整求解主流程走读

主入口 `ComputePathCurve` 大致按以下顺序执行：

```cpp:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
base::Optional<math::Curve2d> ClothoidPathOptimizer::ComputePathCurve(...) const {
  // ① 生成道路元素采样点（近处密、远处疏）
  const std::vector<double> restriction_s_samples = GenerateRestrictionAndRepulsionSSample();
  // ② 生成限制/排斥/碰撞的中间参数
  GenerateRestrictionAndRepulsionCostParams(...);
  // ③ 生成引导线
  const std::vector<Eigen::Vector2d> guide_line = ...;
  // ④ 生成初始猜测路径
  *initial_path_set = GenerateInitialPaths(&reference_path);
  // ⑤（可选）两阶段求解：先求可行解
  if (config_->enable_two_stage_solve()) {
    const Solver::Solution feasible_solution = GenerateFeasiblePath(...);
  }
  // ⑥ 生成完整约束代价集合
  ConstraintAndCostSet constraint_and_cost_set = GenerateConstraintsAndCosts(...);
  // ⑦ 组装成 PathProblem
  const PathProblem path_problem = ConstructPathProblem(*initial_path_set, &constraint_and_cost_set);
  // ⑧ 调用 AL-iLQR 求解器
  const Solver::Solution solution = solver.Solve(path_problem);
  // ⑨ 失败兜底
  if (solution.state != State::kSolved) {
    return GetFallbackSolution(...);
  }
  // ⑩ 转换回连续曲线
  return ConvertToCurve(solution.ilqr_solution.x_trajectory);
}
```

逐步解释：

1. **生成道路元素采样点**（`GenerateRestrictionAndRepulsionSSample`）：从近到远，分辨率从 0.5m→1m→10m 逐级放粗，兼顾精度与计算量。
2. **生成限制/排斥/碰撞的中间参数**（`GenerateRestrictionAndRepulsionCostParams`）：调用 4.4.1 节的参数生成层。
3. **生成引导线**（`GenerateDefaultGuideLine` 或 `GenerateGuideLine`）：如果决策层没给引导线，用限制区间中点（结合特定的 repulsion 做微调）自动生成一条默认引导线。
4. **生成初始解**（`GenerateInitialPaths`）：以参考路径曲线为基准采样出一条初始状态轨迹，控制量用相邻两点 dkappa 差分近似；再用一个独立的 `FeasibleTrajectoryGenerator`（本质是只考虑运动学约束的简化 AL+ILQR 问题）求一条更"可行"的初始解。
5. **（可选）两阶段求解**：先用只含运动学约束+guide line+边界的"简化问题"求一条可行轨迹，替换初始解里的轨迹部分，降低完整问题的求解难度。
6. **组装完整的约束代价集合**（`GenerateConstraintsAndCosts`）：调用 4.4.2/4.4.3/4.4.5 节的所有 generator，把结果汇总进 `ConstraintAndCostSet`。
7. **按采样点打包成 Problem**（`ConstructPathProblem`）：把每个采样点上散落在多个集合里的约束/代价对象合并成该采样点唯一的一个约束向量和一个代价函数对象。
8. **调用 AL+ILQR 求解器**（`Solver::Solve`，见第二部分 2.5 节）。
9. **失败兜底**：如果没求解成功，记录失败原因并返回上一帧路径或初始解里代价最小的一条，保证下游模块始终能拿到一条可用的 path。
10. **转换回连续曲线**（`ConvertToCurve`）。

一句话：**①~⑦ 都是"把现实路况翻译成数学问题"，⑧ 才是 AL-iLQR 算法，⑨⑩ 是把数学解翻译回曲线 + 兜底保障**。

## 4.6 端到端数值例子与设计取舍 FAQ

### 端到端数值例子

用简化数字走一遍单个采样点上"曲率约束 + 横向加速度代价"是怎样共同影响 ILQR 一步迭代的（省略动力学 Jacobian 与其它代价，只展示核心思路）：

假设某采样点当前状态 $\kappa=0.6,\ \omega=0.7$，参考速度 $v=2.5$：

1. **曲率约束**：$\kappa_{max}=2.0$，$c_0=0.6-2.0=-1.4<0$，可行，AL 惩罚为 0（不在违反侧，且假设乘子 $\lambda=0$）。
2. **横向加速度代价**：$a_{lat}=0.6\times2.5^2=3.75$，阈值 $\tilde a=2.0$，超出 $\Delta a=1.75$，代价 $J=0.5\times0.5\times1.75^2=0.765625$，梯度 $\partial J/\partial\kappa=5.46875$。
3. 这个梯度会作为 $l_x$ 的一部分，参与 Backward Pass 里 $Q_x=l_x+f_x^TV_x$ 的计算，进而影响该采样点反馈增益 $K$ 和前馈增益 $k$ 的求解——直观理解就是：由于当前曲率导致横向加速度超标产生了正的代价梯度，ILQR 会倾向于把该点及附近的 $\kappa$ 往更小的方向调整（沿负梯度方向），从而在下一次 Forward Pass 里让新轨迹的横向加速度代价降低。
4. 如果外层 AL 迭代中曲率约束的乘子 $\lambda_{c_0}$ 已经被上一轮更新为正值（说明历史上出现过违反），即使当前 $c_0<0$ 仍然可行，只要 $\lambda_{c_0}>0$ 就仍然会有一次性的 AL 惩罚项参与（让约束满足有一定"记忆"，防止在临界值附近反复震荡）。

多个采样点、多种约束代价的叠加最终通过 ILQR 反复的 Backward+Forward Pass 收敛到一条同时兼顾"不超曲率限制"和"横向加速度尽量小"的轨迹。

### 关键设计取舍 FAQ

**Q1：为什么不直接用曲率作代价，而是作硬约束？**

因为曲率超限对应车辆物理上转不过来的弯，这是绝对不能违反的边界，用硬约束（AL 方法逐步增大惩罚直至满足）比软代价（只是"倾向于"限制）更可靠。

**Q2：为什么横向加速度/jerk 用死区二次代价，而不是硬约束？**

横向加速度/jerk 是舒适性指标，不是物理硬限制——稍微超过一点阈值不代表不可行，只是不够舒适，所以用带死区的软代价（阈值内零惩罚，超出后二次增长）更合适，也给了优化器更大的可行空间去平衡其它约束。

**Q3：为什么要专门设计 SmoothPolyline 而不是直接用折线距离？**

详见 4.3 节：ILQR 需要处处可导（至少二阶）的代价函数，折线原始距离函数在拐点不可导，必须用 LogSumExp 光滑化。

**Q4：为什么要有"两阶段求解"（先求可行解，再求完整最优解）？**

完整问题包含大量非凸的碰撞类约束，直接从粗糙的初始解开始求解容易不收敛或收敛到很差的局部最优。先用一个只含运动学约束的简化问题求一条运动学可行的轨迹作为更好的初始解，能显著提升完整问题的收敛速度和成功率，这是非凸轨迹优化里常见的"热启动（warm start）"思路。

**Q5：为什么求解失败还要返回 fallback？**

Planning 模块每个规划周期都要有输出，不能因为一次数值求解失败就让车辆"没有路径可走"。返回上一帧路径或者候选初始解里代价最小的一条，保证系统的鲁棒性，同时把失败信息上报供后续排查。

**Q6：为什么 `min_alpha` 是一个需要调的参数？**

`min_alpha` 决定了线搜索能接受的最小步长。设得太大，可能在还有改进空间时就放弃搜索，导致收敛提前终止或频繁触发正则化增大；设得太小，会在明显无法改进的方向上浪费过多计算尝试很小的步长。它本质上是"收敛判定严格程度"与"求解耗时"之间的权衡参数。

**Q7：为什么约束要写成 $\le 0$ 的标准形式？**

统一的标准形式方便通用的求解器代码（`InequalityConstraint` 基类）用同一套逻辑处理所有约束——不管是曲率约束、转向速率约束还是避障约束，只要实现 `Evaluate` 返回 $g(x,u)$、以及对应雅可比即可，上层 AL 框架完全不需要关心每个约束具体是什么物理含义。这是一种典型的**面向对象的接口抽象**设计。

**Q8：Huber 代价函数是什么，为什么引导线代价要用它？**

Huber 代价是一种"分段"代价：误差较小时是二次函数（梯度平滑、靠近最优点时收敛快），误差较大时是线性函数（梯度是常数，不会因为个别偏离很远的点产生过大的梯度、把整条轨迹"带偏"）。引导线上如果因为障碍物需要绕行，路径会短暂远离引导线，这时如果用纯二次代价，梯度会大到把优化过程"拉扯"得很剧烈；用 Huber 代价就能让这种"合理的远离"不至于产生过大的修正力度。

**Q9：`ClothoidPathDynamic` 只提供了一阶雅可比，AL-iLQR 不是需要二阶信息吗？**

这正是 iLQR（相对于更精确的 DDP）的简化之处：iLQR 只用动力学的一阶导数（雅可比 $A,B$），配合代价函数自身的二阶导数，通过 **Gauss-Newton 近似**拼出 Q-function 的 Hessian，从而避免显式计算动力学的二阶导数。这是精度和计算效率之间的工程权衡。

---

## 第四部分核心公式速查

| 概念 | 公式 |
|---|---|
| 曲率约束 | $c_0=\kappa-\kappa_{max}\le0,\ c_1=-\kappa-\kappa_{max}\le0$ |
| 转向速率 | $\delta'=\frac{vL\omega}{\kappa^2L^2+1}$ |
| 横向加速度 | $a_{lat}=\kappa v^2$ |
| 横向 jerk | $j_{lat}\approx\omega v^3$ |
| dκ/ds 代价 | $J=\tfrac12 w\omega^2,\ w\propto v^6$ |
| ddκ/dds 代价 | $J=\tfrac12 w\alpha^2,\ w\propto v^8$ |
| 半平面约束 | $c=\hat{\bm n}\cdot(\bm p-\bm p_0)\le0$ |
| LogSumExp | $\text{LSE}_\lambda(\bm x)=m+\frac1\lambda\ln(\sum_i e^{\lambda(x_i-m)}-1)$ |
| SmoothDistance | $d_{smooth}=-\text{LSE}_\lambda(-\bm d)$ |
| PointTranslator | $\bm f(\bm q)=(x+\cos\theta\Delta x-\sin\theta\Delta y,\ y+\sin\theta\Delta x+\cos\theta\Delta y)^T$ |
| 引导线代价 | $J=\tfrac12 w\, d^2$（Huber 型） |

---
