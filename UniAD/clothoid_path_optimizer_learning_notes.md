# ClothoidPathOptimizer 从零学起：公式推导 + 代码对应 + 逻辑流程

> 参考资料：[《AL-iLQR》学城文档](https://km.sankuai.com/collabpage/2778214068)
> 对应代码：`modules/planning/path/clothoid_path_optimizer/`
>
> 本文假设你是**完全零基础**的小白：不知道什么是轨迹优化，没学过最优控制，甚至连"状态""控制量"这些词都觉得陌生。我们会从最直觉的例子开始，一步步搭到能看懂代码的程度。

---

## 目录

1. [先搞懂问题是什么：为什么要"优化"一条路径](#1-先搞懂问题是什么为什么要优化一条路径)
2. [Clothoid 曲线：为什么车的路径要用它来描述](#2-clothoid-曲线为什么车的路径要用它来描述)
3. [状态空间建模：把"车怎么走"写成数学式子](#3-状态空间建模把车怎么走写成数学式子)
4. [离散化：让计算机能算](#4-离散化让计算机能算)
5. [轨迹优化问题的标准形式](#5-轨迹优化问题的标准形式)
6. [约束怎么处理：增强拉格朗日法（Augmented Lagrangian）](#6-约束怎么处理增强拉格朗日法augmented-lagrangian)
7. [没有约束怎么解：LQR 与 HJB 方程](#7-没有约束怎么解lqr-与-hjb-方程)
8. [非线性怎么办：iLQR（迭代 LQR）](#8-非线性怎么办ilqr迭代-lqr)
9. [AL + iLQR 合体：Backward Pass / Forward Pass](#9-al--ilqr-合体backward-pass--forward-pass)
10. [代码总览：一次 `ComputePath` 调用做了什么](#10-代码总览一次computepath调用做了什么)
11. [逐个文件对应：state/control/dynamics](#11-逐个文件对应statecontroldynamics)
12. [逐个文件对应：约束（constraints）](#12-逐个文件对应约束constraints)
13. [逐个文件对应：代价函数（cost_functions & utils）](#13-逐个文件对应代价函数cost_functions--utils)
14. [从"猜一条初始路径"到"求解"的全流程](#14-从猜一条初始路径到求解的全流程)
15. [常见疑问 FAQ](#15-常见疑问-faq)

---

## 1. 先搞懂问题是什么：为什么要"优化"一条路径

想象你在开车，前方要绕过一个障碍物，同时要贴着车道行驶、不能急打方向盘。你的大脑其实在做一件事：

> **在满足"不撞车""不出车道""方向盘转动不能太猛"这些要求（约束）的前提下，找一条尽量平顺、尽量贴近理想路线（代价最小）的路。**

这就是"路径优化"要干的事。计算机没有直觉，所以我们必须把"平顺""贴近理想路线""不能急打方向盘"这些模糊的感觉，翻译成**数学上可以计算的公式**，然后用算法去"搜索"出最好的那条路径。

`ClothoidPathOptimizer` 就是 walle2 里专门做这件事的模块。它接收：
- 车道边界、障碍物位置（约束从哪来）
- 一条参考引导线（大致想往哪走）
- 车辆当前状态（在哪、朝向、当前转弯半径等）

输出：
- 一条从当前位置出发、面向未来若干米、满足所有约束、代价最小的路径。

---

## 2. Clothoid 曲线：为什么车的路径要用它来描述

### 2.1 直觉：为什么不能用折线或者圆弧拼接？

- **折线**：转弯处曲率突变（从 0 突然变成无穷大），方向盘要瞬间打死，物理上不可能。
- **圆弧拼接**：圆弧本身曲率是常数，但两段圆弧拼接处曲率会有"跳变"，方向盘还是要瞬间转到某个角度，不平滑。

### 2.2 Clothoid（回旋曲线）的特点

Clothoid 曲线的核心特性是：**曲率 $\kappa$ 随弧长 $s$ 线性变化**，即：

$$
\kappa(s) = \kappa_0 + \dot\kappa \cdot s
$$

对应到开车的直觉就是：**匀速转动方向盘**，曲率平滑地从一个值变化到另一个值，不会有跳变。这正是真实车辆能做到的动作（方向盘转动是连续的，不会瞬间跳到某个角度）。

在 `ClothoidPathOptimizer` 里，我们不仅让曲率线性变化，还进一步把"曲率的变化率" $\dot\kappa$（代码里叫 `dkappa`）也作为一个状态量，让它也能被约束和优化，"曲率的变化率的变化率" $\ddot\kappa$（代码里叫 `ddkappa`）则作为控制输入。这样车辆路径在数学上是 **$C^2$ 连续**的（曲率本身连续，曲率的导数也连续），对应到物理世界就是：方向盘角度连续、方向盘转动速度也连续，开起来非常平顺。

---

## 3. 状态空间建模：把"车怎么走"写成数学式子

### 3.1 什么是"状态"、"控制"

- **状态 $x$**：描述系统"此刻是什么样子"的一组数字。比如车此刻的位置 $(x, y)$、朝向 $\theta$、曲率 $\kappa$……
- **控制 $u$**：我们能主动施加、去改变状态的量。比如方向盘转动的力度。

一个系统只要满足"未来只取决于当前状态和当前控制，与更早的历史无关"（这叫**马尔可夫性质**），就可以写成：

$$
\dot{x} = f(x, u)
$$

意思是：**状态的变化速度，是当前状态和当前控制的函数**。

### 3.2 Clothoid 路径的状态空间模型

对应代码 `clothoid_path_optimizer_common.h`：

```19:31:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer_common.h
struct ClothoidPathStateIndex {
  static constexpr int kX = 0;
  static constexpr int kY = 1;
  static constexpr int kTheta = 2;
  static constexpr int kKappa = 3;
  static constexpr int kDKappa = 4;
  static constexpr int kSize = 5;
};

struct ClothoidPathControlIndex {
  static constexpr int kDdKappa = 0;
  static constexpr int kSize = 1;
};
```

也就是说，状态向量是 5 维：

$$
x = \begin{bmatrix} x \\ y \\ \theta \\ \kappa \\ \dot\kappa \end{bmatrix}
\quad\quad
\text{（位置 x、位置 y、朝向 heading、曲率 kappa、曲率变化率 dkappa）}
$$

控制量是 1 维（曲率的二阶导数，即方向盘"转动加速度"）：

$$
u = \begin{bmatrix} \ddot\kappa \end{bmatrix}
$$

### 3.3 动力学方程 $f(x, u)$ 对应代码

车辆沿着弧长 $s$ 前进（这里把 $s$ 当作类似"时间"的自变量），几何关系告诉我们：

$$
\begin{aligned}
\frac{dx}{ds} &= \cos\theta \\
\frac{dy}{ds} &= \sin\theta \\
\frac{d\theta}{ds} &= \kappa \\
\frac{d\kappa}{ds} &= \dot\kappa \\
\frac{d\dot\kappa}{ds} &= \ddot\kappa = u
\end{aligned}
$$

直觉解释：
- **位置变化率 = 沿当前朝向前进**（$\cos\theta, \sin\theta$ 就是单位方向向量的分量）。
- **朝向变化率 = 曲率**（曲率越大，转向越快，这是曲率的定义）。
- **曲率变化率 = $\dot\kappa$**（定义）。
- **$\dot\kappa$ 的变化率 = 控制量 $u$**（我们主动控制的就是这个"曲率加速度"）。

这几乎是逐字对应到代码 `clothoid_path_dynamic.cc` 的 `Evaluate` 函数：

```11:19:modules/planning/path/clothoid_path_optimizer/clothoid_path_dynamic.cc
ClothoidPathDynamic::VecX ClothoidPathDynamic::Evaluate(const VecX& x, const VecU& u) const {
  VecX x_dot;
  x_dot(ClothoidPathStateIndex::kX) = math::Cos(x(ClothoidPathStateIndex::kTheta));
  x_dot(ClothoidPathStateIndex::kY) = math::Sin(x(ClothoidPathStateIndex::kTheta));
  x_dot(ClothoidPathStateIndex::kTheta) = x(ClothoidPathStateIndex::kKappa);
  x_dot(ClothoidPathStateIndex::kKappa) = x(ClothoidPathStateIndex::kDKappa);
  x_dot(ClothoidPathStateIndex::kDKappa) = u(ClothoidPathControlIndex::kDdKappa);
  return x_dot;
}
```

一一对照：
| 数学公式 | 代码 |
|---|---|
| $\dot x = \cos\theta$ | `x_dot(kX) = math::Cos(x(kTheta))` |
| $\dot y = \sin\theta$ | `x_dot(kY) = math::Sin(x(kTheta))` |
| $\dot\theta = \kappa$ | `x_dot(kTheta) = x(kKappa)` |
| $\dot\kappa_{\text{state}} = \dot\kappa$ | `x_dot(kKappa) = x(kDKappa)` |
| $\ddot\kappa = u$ | `x_dot(kDKappa) = u(kDdKappa)` |

### 3.4 雅可比矩阵（Jacobian）：为什么需要它

后面求解算法（iLQR）要对动力学方程做**线性近似**（泰勒展开的一阶项），所以要提前把 $f(x,u)$ 对 $x$、对 $u$ 的偏导数（雅可比矩阵）算出来：

$$
A = \frac{\partial f}{\partial x}, \quad B = \frac{\partial f}{\partial u}
$$

对应代码：

```21:33:modules/planning/path/clothoid_path_optimizer/clothoid_path_dynamic.cc
ClothoidPathDynamic::Jacobians ClothoidPathDynamic::EvaluateJacobians(const VecX& x,
                                                                      const VecU& u) const {
  Jacobians jacobians = Jacobians::Zero();
  jacobians(ClothoidPathStateIndex::kX, ClothoidPathStateIndex::kTheta) =
      -math::Sin(x(ClothoidPathStateIndex::kTheta));
  jacobians(ClothoidPathStateIndex::kY, ClothoidPathStateIndex::kTheta) =
      math::Cos(x(ClothoidPathStateIndex::kTheta));
  jacobians(ClothoidPathStateIndex::kTheta, ClothoidPathStateIndex::kKappa) = 1.0;
  jacobians(ClothoidPathStateIndex::kKappa, ClothoidPathStateIndex::kDKappa) = 1.0;
  jacobians(ClothoidPathStateIndex::kDKappa,
            ClothoidPathStateIndex::kSize + ClothoidPathControlIndex::kDdKappa) = 1.0;
  return jacobians;
}
```

比如 $\partial \dot x / \partial \theta = -\sin\theta$，正是 `-math::Sin(x(kTheta))`。这个矩阵就是简单地对第 3.3 节的 5 个方程分别求偏导，没有任何"魔法"，纯粹是高中/大学微积分的链式法则。

> 小贴士：`jacobians(row, ClothoidPathStateIndex::kSize + u_index)` 这种写法是因为代码把 $A$（对 $x$ 的偏导）和 $B$（对 $u$ 的偏导）拼在了同一个矩阵里，状态维度之后紧跟着控制维度的偏导。

---

## 4. 离散化：让计算机能算

### 4.1 为什么要离散化

计算机无法处理连续的微分方程和积分，必须把"连续的时间/弧长"切成一段一段的。这就像你不能对着一条连续曲线直接编程，只能取一堆采样点来逼近它。

### 4.2 离散化怎么做（对应学城文档 1.1～1.4 节）

把总长度 $s\in[0, s_f]$ 切成 $N-1$ 段，每段步长 $\Delta s$（代码里叫 `steps_[i]`，注意这里步长可以不均匀），得到 $N$ 个采样点（节点，knot points）：

$$
x_0, x_1, \dots, x_N
$$

原来的微分方程：

$$
\dot x = f(x, u)
$$

离散化为差分方程：

$$
x_{k+1} = f_d(x_k, u_k, \Delta s)
$$

这里的 $f_d$ 就是"从 $x_k$ 出发，沿着动力学方程 $f$ 走一小步 $\Delta s$ 后到达的状态"，具体走法用**数值积分**（比如四阶龙格库塔 RK4）来近似求解这一小段的微分方程。

对应代码里：

```536:547:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
std::unique_ptr<
    DynamicModel<ClothoidPathOptimizer::kNumStates, ClothoidPathOptimizer::kNumControls>>
ClothoidPathOptimizer::GetDynamicModelAtSample(int index) const {
  std::unique_ptr<const ContinuousDynamics<kNumStates, kNumControls>> dynamics =
      std::make_unique<const ClothoidPathDynamic>();
  std::unique_ptr<const Integrator<kNumStates, kNumControls>> integrator =
      std::make_unique<const RungeKutta4<kNumStates, kNumControls>>();
  std::unique_ptr<DynamicModel<kNumStates, kNumControls>> dynamic_model =
      std::make_unique<DiscretizedModel<kNumStates, kNumControls>>(
          std::move(dynamics), std::move(integrator), steps_[index]);
  return dynamic_model;
}
```

- `ClothoidPathDynamic` = 连续动力学 $f(x,u)$（第 3 节讲的那个）。
- `RungeKutta4` = 数值积分器，把连续方程离散成一步一步的差分方程 $f_d$。
- `steps_[index]` = 这一步的步长 $\Delta s$。
- `DiscretizedModel` 把上面两者结合起来，就是离散动力学模型。

### 4.3 目标函数（代价）的离散化

连续的积分：

$$
\int_0^{s_f} \ell(x(s), u(s))\, ds
$$

离散化成求和：

$$
\sum_{k=0}^{N-1} \ell_k(x_k, u_k, \Delta s)
$$

终点单独留一个终端代价 $\ell_N(x_N)$（比如最后一个点要贴近参考线）。

---

## 5. 轨迹优化问题的标准形式

把第 3、4 节拼起来，完整问题就是：

$$
\min_{x_{0:N},\, u_{0:N-1}} \;\; \ell_N(x_N) + \sum_{k=0}^{N-1} \ell_k(x_k, u_k, \Delta s)
$$

subject to（约束条件）：

$$
\begin{aligned}
x_{k+1} &= f_d(x_k, u_k, \Delta s), && k = 0,\dots,N-1 &&\text{（动力学约束，必须精确满足）}\\
g_k(x_k, u_k) &\le 0, && \forall k &&\text{（不等式约束，比如曲率不能超限）}\\
h_k(x_k, u_k) &= 0, && \forall k &&\text{（等式约束，本模块基本不用）}
\end{aligned}
$$

- **优化变量**：所有采样点的状态 $x_0,\dots,x_N$ 和控制 $u_0,\dots,u_{N-1}$。
- **动力学约束**：必须严格满足，因为这是"物理规律"（车不可能瞬移）。
- **不等式约束** $g_k \le 0$：曲率限制、方向盘转速限制、车道边界、避障等。

在代码里，这些约束和代价分别对应 `constraints/` 和 `cost_functions/` 文件夹下的类，之后第 12、13 节详细讲。

求解这个问题的核心算法是 **AL-iLQR**（增强拉格朗日 + 迭代LQR），下面几节专门讲它。

---

## 6. 约束怎么处理：增强拉格朗日法（Augmented Lagrangian）

### 6.1 最朴素的想法：罚函数法，为什么不够好

一个直觉的做法：把约束"塞进"目标函数里，谁违反约束就罚谁：

$$
\min_x \; f(x) + \frac{\mu}{2} c(x)^2
$$

$\mu$ 越大，违反约束的代价越高，逼着解往可行域里挤。

**问题**：要让约束真正精确满足（$c(x) = 0$），理论上需要 $\mu \to \infty$。但数值计算里，$\mu$ 太大会导致目标函数"病态"（condition number 爆炸），优化器根本收敛不了、或者数值抖动得很厉害。这就像你想用一个特别硬的弹簧把两块板压在一起——弹簧越硬，稍微一点误差就会产生巨大的力，数值上很不稳定。

### 6.2 增强拉格朗日法的改进

在罚函数的基础上，**再加一个拉格朗日乘子项**：

$$
\mathcal{L}_A(x, \lambda, \mu) = f(x) + \lambda^T c(x) + \frac{1}{2} c(x)^T I_\mu\, c(x)
$$

- $f(x)$：原始目标（比如路径要平顺、贴近参考线）。
- $\lambda$：拉格朗日乘子，可以理解成"约束力"的估计值，会在迭代中不断更新，逐渐逼近真实值。
- $\mu$：罚因子，依然存在，但**不需要趋于无穷大**。
- $I_\mu$：一个对角矩阵，作用是"聪明地决定要不要罚"：
  - 如果是不等式约束，且这个约束**已经被严格满足**（$c_i(x) < 0$）并且当前乘子 $\lambda_i = 0$，那就不罚（这一项系数置零）——没必要为没有违反的约束浪费惩罚力度。
  - 否则，正常用 $\mu_i > 0$ 惩罚。

直觉理解：$\lambda$ 就像一只"手"，提前把约束往正确方向推一把；$\mu$ 只是"锦上添花"的惩罚，不需要非常大也能配合 $\lambda$ 一起把解推到可行域。这样数值上就稳定得多。

### 6.3 迭代步骤（外层循环）

1. **固定 $\lambda,\mu$**，最小化 $\mathcal{L}_A(x,\lambda,\mu)$（这一步用 iLQR 来做，见第 8、9 节）。
2. **更新乘子**：

$$
\lambda_i^+ = \begin{cases}
\lambda_i + \mu_i\, c_i(x^*) & i \in \text{等式约束} \\
\max\{0,\; \lambda_i + \mu_i\, c_i(x^*)\} & i \in \text{不等式约束}
\end{cases}
$$

（不等式约束的乘子必须非负，这是 KKT 条件里的"互补松弛性"要求。）

3. **更新罚因子**：$\mu^+ = \phi\, \mu$，其中 $\phi > 1$（一般取 2~10）。如果约束还没收敛，就把惩罚力度进一步加大。
4. **检查是否收敛**（约束违反程度是否小于容忍值）。
5. 没收敛就回到第 1 步。

这就是为什么代码里 `ALSolver::Solve` 会有一个 `for (int iter = 0; iter < config_.max_outer_iterations; ++iter)` 的外层循环——这正是上面 1~5 步的循环：

```156:205:modules/planning/common/math/solver/augmented_lagrangian/al_solver.h
ALSolver<X, U, C1, C2>::Solve(
    const ConstrainedProblem<X, U, C1, C2>& constrained_problem) const {
  ALProblem problem(constrained_problem);
  ResetDuals(&problem);      // 初始化 lambda
  ResetPenalties(&problem);  // 初始化 mu
  ...
  for (int iter = 0; iter < config_.max_outer_iterations; ++iter) {
    const IlqrSolution inner_solution = ilqr_solver_.Solve(problem, ...);  // 第1步：内层iLQR求解
    UpdateStatus(problem, inner_solution, solve_start_time, &status);
    const State state = CheckTerminationCondition(status);   // 第4步：检查收敛
    ...
    if (state != State::kUnsolved) {
      return Solution{...};   // 收敛了，返回结果
    }
    UpdateDuals(inner_solution, &problem);     // 第2步：更新 lambda
    UpdatePenalties(&problem);                 // 第3步：更新 mu
    ...
  }
}
```

一句话总结：**外层是"调整惩罚力度"的循环（AL），内层是在给定惩罚下"求一个尽量优"的轨迹（iLQR）**。

### 6.4 为什么要"逐步"增大 $\mu$，而不是一步到位

- $\mu$ 太小 → 惩罚力度不够，解可能一直"赖"在不可行域（约束没被满足）。
- $\mu$ 一开始就很大 → 数值病态，优化器一步都迈不动或者震荡。

所以采取"温水煮青蛙"策略：每轮迭代把 $\mu$ 稍微放大一点（$\mu^+ = \phi\mu$），逐渐收紧约束，给优化器一个"逐渐适应"的过程。

---

## 7. 没有约束怎么解：LQR 与 HJB 方程

在讲 iLQR 之前，先理解它的"简化版祖先"：LQR（线性二次调节器）。因为 iLQR 本质上就是"在每一步都用 LQR 的思路，处理一个非线性问题的局部线性近似"。

### 7.1 LQR 问题定义

假设：
- 动力学是**线性**的：$\dot x = Ax + Bu$
- 代价是**二次型**的（只有平方项，没有三次方等）：

$$
\min_{u(t)} \; \frac12 x^T(t_f) Q(t_f) x(t_f) + \frac12\int_0^{t_f}\left[x^TQx + u^TRu\right]dt
$$

- $Q \succeq 0$：状态权重矩阵（越大表示越不希望状态偏离目标）。
- $R \succ 0$：控制权重矩阵（越大表示越不希望使劲打方向盘/踩油门）。

离散版本（我们实际用的）：

$$
\min_{u_k} \; \frac12 x_N^T Q_N x_N + \frac12\sum_{k=0}^{N-1}\left[x_k^TQ_kx_k + u_k^TR_ku_k\right]
\quad\text{s.t.}\quad x_{k+1} = A_kx_k+B_ku_k
$$

### 7.2 为什么能求出解析解：HJB 方程

定义"从状态 $x$、时刻 $t$ 出发，走到终点的最小代价"为**价值函数** $J^*(x,t)$。

Bellman 最优性原理告诉我们：**当前最优 = 这一步的代价 + 从下一步开始的最优代价**。用无穷小时间步 $\Delta t$ 展开、做泰勒展开、两边抵消化简，最终得到 **HJB 方程**：

$$
0 = J_t^*(x,t) + \min_u\Big[\ell(x,u,t) + J_x^{*T}(x,t) f(x,u,t)\Big]
$$

对 LQR（线性动力学 + 二次代价），假设价值函数也是二次型 $J^*(x,t) = \frac12 x^TK(t)x$，代入 HJB 方程、对 $u$ 求导为零，可以解出：

$$
u^* = -R^{-1}B^T J_x^* = -Kx
$$

即：**最优控制是状态的线性反馈**。$K$ 满足一个叫**黎卡提方程**的微分方程。

离散情形下，反馈增益和价值函数矩阵有递推公式：

$$
K_k = (R_k + B_k^TP_{k+1}B_k)^{-1}B_k^TP_{k+1}A_k
$$

$$
P_k = Q_k + A_k^TP_{k+1}A_k - A_k^TP_{k+1}B_k(R_k+B_k^TP_{k+1}B_k)^{-1}B_k^TP_{k+1}A_k
\quad\quad P_N = Q_N
$$

**记住这个"从后往前递推"的模式**——这就是下面 iLQR 的 Backward Pass 的原型。LQR 是"一步到位"的解析解，因为问题是线性二次的；而我们的 Clothoid 路径问题是非线性的（有障碍物、复杂代价函数），没有解析解，只能靠 iLQR 反复迭代逼近。

---

## 8. 非线性怎么办：iLQR（迭代 LQR）

### 8.1 核心思想

真实问题是非线性的：动力学 $f(x,u)$ 里有 $\cos\theta,\sin\theta$；代价函数里有分段函数（超过阈值才罚）。iLQR 的思路是：

> **在当前的一条"名义轨迹"附近，把动力学和代价都做局部二次/线性近似，用 LQR 的方法求出一个"改进方向"，然后沿着这个方向小步前进，反复迭代，直到收敛到一个局部最优解。**

这就像下山：你不知道山谷在哪，但你可以在脚下这一小块地方，看一下坡度和曲率（局部近似），往下坡方向走一步，再重新看一次坡度……反复迭代，最后走到谷底。

### 8.2 Q-function：一步的"代价+未来代价"

定义 Q-function（不是强化学习里的 Q，是这里的记号）：

$$
Q(x_k,u_k) = \ell_k(x_k,u_k) + V_{k+1}(f(x_k,u_k))
$$

意思是：**这一步的代价，加上走到下一步之后剩余的最优代价（cost-to-go）**。

对 Q-function 做二阶泰勒展开：

$$
\delta Q_k = \frac12\begin{bmatrix}\delta x_k\\ \delta u_k\end{bmatrix}^T
\begin{bmatrix}Q_{xx} & Q_{xu}\\ Q_{ux} & Q_{uu}\end{bmatrix}
\begin{bmatrix}\delta x_k\\ \delta u_k\end{bmatrix}
+ \begin{bmatrix}Q_x\\ Q_u\end{bmatrix}^T\begin{bmatrix}\delta x_k\\ \delta u_k\end{bmatrix}
$$

其中这些分块矩阵由动力学的雅可比 $A=\partial f/\partial x, B=\partial f/\partial u$，以及下一时刻价值函数的梯度 $p'=V_x'$、Hessian $P'=V_{xx}'$ 组合而成：

$$
\begin{aligned}
Q_x &= \ell_x + A^Tp' \\
Q_u &= \ell_u + B^Tp' \\
Q_{xx} &= \ell_{xx} + A^TP'A \\
Q_{uu} &= \ell_{uu} + B^TP'B \\
Q_{ux} &= \ell_{ux} + B^TP'A
\end{aligned}
$$

- **DDP**：连 $f$ 的二阶导数项也保留，更精确但计算贵。
- **iLQR**：只用 $f$ 的一阶导数（雅可比），把动力学**线性化**，用 Gauss-Newton 近似 Hessian，更快，是本模块采用的方法（这也是为什么 `ClothoidPathDynamic` 只需要提供 `EvaluateJacobians`，不需要二阶导数）。

### 8.3 求最优控制增量

对 $\delta Q_k$ 关于 $\delta u_k$ 求导为零：

$$
Q_{uu}\delta u_k + Q_{ux}\delta x_k + Q_u = 0
$$

解得：

$$
\delta u_k^* = -Q_{uu}^{-1}(Q_{ux}\delta x_k + Q_u)
$$

为了防止 $Q_{uu}$ 不可逆（奇异），加一个小的正则化 $\rho I$：

$$
\delta u_k^* = -(Q_{uu}+\rho I)^{-1}(Q_{ux}\delta x_k + Q_u) = \underbrace{K_k}_{\text{反馈增益}}\delta x_k + \underbrace{d_k}_{\text{前馈修正}}
$$

- $K_k = -(Q_{uu}+\rho I)^{-1}Q_{ux}$：**根据状态偏差调整控制**（跟 LQR 的反馈增益一模一样的形式）。
- $d_k = -(Q_{uu}+\rho I)^{-1}Q_u$：**即使没有状态偏差，也要做的固定修正**（因为当前轨迹本身还不是最优的）。

再把这个最优 $\delta u_k^*$ 代回去，可以递推出上一步（$k-1$）的价值函数近似：

$$
P_k = Q_{xx} + K_k^TQ_{uu}K_k + K_k^TQ_{ux} + Q_{xu}K_k
$$
$$
p_k = Q_x + K_k^TQ_{uu}d_k + K_k^TQ_u + Q_{xu}d_k
$$
$$
\Delta V_k = d_k^TQ_u + \frac12 d_k^TQ_{uu}d_k \quad\text{（预期的代价下降量）}
$$

---

## 9. AL + iLQR 合体：Backward Pass / Forward Pass

把第 6 节的增强拉格朗日目标函数 $\mathcal{L}_A$，当作 iLQR 里的代价函数 $\ell_k$ 来用，就得到了 **AL-iLQR**：内层用 iLQR 处理非线性+局部二次近似，外层用 AL 处理约束。

### 9.1 Backward Pass（从后往前）

1. **初始化终端**：根据终端代价，算出 $p_N, P_N$：
   $$p_N = (\ell_N)_x + (c_N)_x^T(\lambda+I_{\mu,N}c_N), \quad P_N = (\ell_N)_{xx}+(c_N)_x^TI_{\mu,N}(c_N)_x$$
2. **从 $k=N-1$ 到 $k=0$ 依次**：
   - 用当前的 $A_k, B_k$（动力学雅可比）和下一步的 $p_{k+1}, P_{k+1}$，展开 Q-function 的各个分块矩阵。
   - 计算这一步的反馈增益 $K_k$、前馈修正 $d_k$、期望代价改进 $\Delta V_k$。
   - 如果 $Q_{uu}$ 病态（不可逆/奇异），增大正则化 $\rho$，重新算这一步。
3. 遍历完所有时间步，得到完整的 $\{K_k\}, \{d_k\}, \Delta V$。

### 9.2 Forward Pass（从前往后，正向模拟）

有了每一步的反馈增益，就可以正向"滚动"出一条新的候选轨迹：

- 状态偏差：$\delta x_k = \bar x_k - x_k$（新轨迹与旧轨迹在这一步的差异）
- 控制偏差：$\delta u_k = K_k \delta x_k + \alpha d_k$（$\alpha$ 是线搜索步长因子，初始为 1）
- 更新控制：$\bar u_k = u_k + \delta u_k$
- 更新状态：$\bar x_{k+1} = f(\bar x_k, \bar u_k)$（用真实非线性动力学往前推，不是线性近似）

**线搜索**：算出新旧代价的比值

$$
z = \frac{J(X,U) - J(\bar X,\bar U)}{-\Delta V(\alpha)}
$$

- 如果 $z \in [10^{-4}, 10]$（实际下降量和预期下降量比例合理），接受这条新轨迹；
- 否则把 $\alpha$ 减半，重新按 Forward Pass 的公式再滚动一次，直到满意或 $\alpha$ 太小放弃。

如果多次线搜索都失败，或代价爆炸，就放弃这次前向传播，**增大正则化 $\rho$**，重新做一次 Backward Pass（正则化越大，$K,d$ 越接近保守的梯度下降方向，牺牲一点效率换取稳定性）。只有 Backward Pass 成功一次之后，$\rho$ 才会减小。

### 9.3 内外循环嵌套关系（一图流）

```
AL 外层循环（更新 λ, μ）
  └── iLQR 内层循环（在固定 λ, μ 下求最优轨迹）
        └── Backward Pass（从后往前，算 K, d）
        └── Forward Pass（从前往后，滚动新轨迹 + 线搜索）
        └── 判断是否收敛，没收敛就再来一次 Backward+Forward
  └── 内层收敛后，更新 λ（拉格朗日乘子）和 μ（罚因子）
  └── 判断外层是否收敛（约束是否被满足）
```

---

## 10. 代码总览：一次 `ComputePath` 调用做了什么

现在有了理论基础，我们来看 `clothoid_path_optimizer.cc` 里最核心的入口函数 `ComputePathCurve`，按顺序梳理它做的事情（对应第 5 节讲的"问题构造 → 求解"两大步骤）：

```109:222:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
base::Optional<math::Curve2d> ClothoidPathOptimizer::ComputePathCurve(...) const {
  ...
  // ① 生成 s 采样点（哪些弧长位置需要考虑车道边界/避障约束）
  const std::vector<double> restriction_and_repulsiton_s_samples =
      GenerateRestrictionAndRepulsionSSample();

  // ② 根据车道边界、避障需求、box collision 生成对应的代价/约束参数
  GenerateRestrictionAndRepulsionCostParams(...);

  // ③ 生成引导线 guide_line（大致想让路径贴近的曲线）
  const std::vector<Eigen::Vector2d> guide_line = ...;

  // ④ 生成初始猜测路径（warm start，见第14节）
  *initial_path_set = GenerateInitialPaths(&reference_path);

  // ⑤（可选）先解一个简化的"可行性"问题，保证初始猜测满足运动学约束
  if (config_->enable_two_stage_solve()) {
    const Solver::Solution feasible_solution = GenerateFeasiblePath(...);
    ...
  }

  // ⑥ 生成完整的约束集合和代价函数集合
  ConstraintAndCostSet constraint_and_cost_set = GenerateConstraintsAndCosts(...);

  // ⑦ 组装成 AL 求解器能识别的 PathProblem
  const PathProblem path_problem = ConstructPathProblem(*initial_path_set, &constraint_and_cost_set);

  // ⑧ 调用 AL-iLQR 求解器求解（就是第6~9节讲的算法！）
  const Solver solver(al_solver_config_, ilqr_solver_config_);
  const Solver::Solution solution = solver.Solve(path_problem);

  // ⑨ 把解出来的离散状态轨迹转换回连续曲线
  return ConvertToCurve(solution.ilqr_solution.x_trajectory);
}
```

一句话总结：**①②③④⑤⑥⑦ 都是"把现实世界的路况翻译成数学问题（构造 $\ell_k, g_k$）"，⑧才是真正调用第 6~9 节讲的 AL-iLQR 算法去求解，⑨是把数学解翻译回现实世界的一条曲线**。

---

## 11. 逐个文件对应：state/control/dynamics

| 文件 | 数学对应 | 作用 |
|---|---|---|
| `clothoid_path_optimizer_common.h` | 定义 $x=(x,y,\theta,\kappa,\dot\kappa)$，$u=(\ddot\kappa)$ 的下标 | 状态/控制向量的"字典"，其他所有文件都靠这几个下标去存取状态里的哪个分量 |
| `clothoid_path_dynamic.h/.cc` | $\dot x=f(x,u)$ 及其雅可比 $A,B$（第 3 节） | 描述车辆沿弧长演化的物理规律，供 iLQR 的 Backward Pass 用来算 $Q_{xx},Q_{uu}$ 等 |
| `clothoid_path_optimizer.h/.cc` | 整体优化问题的构造与求解（第 5、10 节） | "总指挥"：把约束、代价拼起来，调用求解器，转换结果 |

`ClothoidPath` 结构体（在 `clothoid_path_optimizer_common.h` 里）：

```38:45:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer_common.h
struct ClothoidPath {
  using VectorOfVecX = ...;
  using VectorOfVecU = ...;

  VectorOfVecX x_trajectory;   // 对应 x_0, x_1, ..., x_N
  VectorOfVecU u_trajectory;   // 对应 u_0, u_1, ..., u_{N-1}
};
```

就是数学里 $x_{0:N}, u_{0:N-1}$ 这两个序列的代码实体。

---

## 12. 逐个文件对应：约束（constraints）

约束对应第 5 节里的不等式 $g_k(x_k,u_k) \le 0$。**注意所有约束都被写成"$\le 0$"的标准形式**，这是为了让求解器统一处理。

### 12.1 曲率约束 `clothoid_curvature_constraint.cc`

数学上：曲率不能超过车辆最大转弯能力 $\kappa_{max}$，即 $-\kappa_{max} \le \kappa \le \kappa_{max}$。

拆成两条 $\le 0$ 的约束：

$$
g_1 = \kappa - \kappa_{max} \le 0, \quad g_2 = -\kappa - \kappa_{max} \le 0
$$

```10:16:modules/planning/path/clothoid_path_optimizer/constraints/clothoid_curvature_constraint.cc
ClothoidCurvatureConstraint::VecC ClothoidCurvatureConstraint::Evaluate(const VecX& x,
                                                                        const VecU& u) const {
  VecC c(kDimension);
  c(0) = x(ClothoidPathStateIndex::kKappa) - kappa_max_;
  c(1) = -x(ClothoidPathStateIndex::kKappa) - kappa_max_;
  return c;
}
```

雅可比也很直白：$\partial g_1/\partial \kappa = 1$，$\partial g_2 /\partial \kappa=-1$，其余偏导都是 0，对应代码里 `jacobians.dcdx(0, kKappa) = 1.0; jacobians.dcdx(1, kKappa) = -1.0;`。

### 12.2 转向速率约束 `clothoid_steering_rate_constraint.cc`

方向盘转动速度不能太快。根据阿克曼转向几何，前轮转角 $\delta$ 与曲率关系为 $\delta = \arctan(\kappa L)$（$L$ 是轴距 wheel_base）。对时间求导，并把 $ds/dt$ 换成参考车速 $v$，可以得到转向速率：

$$
\dot\delta = \frac{v\, L\, \dot\kappa}{(\kappa L)^2+1}
$$

约束：$-\dot\delta_{max} \le \dot\delta \le \dot\delta_{max}$，同样拆成两条：

```10:19:modules/planning/path/clothoid_path_optimizer/constraints/clothoid_steering_rate_constraint.cc
ClothoidSteeringRateConstraint::VecC ClothoidSteeringRateConstraint::Evaluate(const VecX& x,
                                                                              const VecU& u) const {
  const double kappa_wheelbase = x(ClothoidPathStateIndex::kKappa) * wheel_base_;
  const double steering_rate =
      (reference_speed_ * wheel_base_ * x(ClothoidPathStateIndex::kDKappa)) /
      (kappa_wheelbase * kappa_wheelbase + 1.0);
  VecC c(kDimension);
  c(0) = (steering_rate - steering_rate_max_);
  c(1) = (-steering_rate - steering_rate_max_);
  return c;
}
```

对应关系一目了然：`kappa_wheelbase` $=\kappa L$，`steering_rate` $=\dot\delta$，分母 `kappa_wheelbase * kappa_wheelbase + 1.0` $=(\kappa L)^2+1$。雅可比部分是对 $\dot\delta$ 关于 $\kappa$ 和 $\dot\kappa$ 求偏导（用商法则展开），细节不必逐行推，只要理解"这是链式法则+商法则的直接展开"即可。

### 12.3 其他约束/代价生成器（utils 文件夹）

| 文件 | 对应的现实需求 |
|---|---|
| `restriction_constraint_and_cost_generator.*` | 车道边界（不能压出边界之外） |
| `box_collision_constraint_and_cost_generator.*` | 车身包络盒不能与障碍物边界碰撞 |
| `obstacle_constraint_and_cost_generator.*` | 具体障碍物（车辆、行人）的避让约束/代价 |
| `kinematic_constraint_and_cost_generator.*` | 把上面第 12.1、12.2 节的运动学约束和二次代价打包生成 |

这些生成器本质上都是"根据感知/决策的输入，在每个采样点 $k$ 上，实例化一批 `InequalityConstraint`/`CostFunction` 对象"，最终汇总到 `ConstraintAndCostSet` 结构体里（见 `clothoid_path_optimizer_common.h` 55~67 行）。

---

## 13. 逐个文件对应：代价函数（cost_functions & utils）

代价函数对应第 5 节的 $\ell_k(x_k,u_k)$，它们不是硬性约束，而是"软性偏好"：不满足也不会直接判失败，但会让总代价变高，求解器会尽量避免。

### 13.1 侧向加速度代价 `clothoid_lateral_acceleration_cost_function.cc`

物理关系：侧向加速度 $a_{lat} = \kappa v^2$（转弯半径越小/车速越快，侧向加速度越大，这就是你坐车过弯时感觉到的"离心力"）。

如果超过舒适阈值 $a_{lim}$，就用**分段二次罚函数**惩罚（类似一个"死区+二次增长"的曲线，超过阈值才开始罚，且罚得越多代价平方增长）：

$$
\ell = \begin{cases}
\frac12 w (a_{lat}-a_{lim})^2 & a_{lat} > a_{lim} \\
\frac12 w (a_{lat}+a_{lim})^2 & a_{lat} < -a_{lim} \\
0 & \text{otherwise}
\end{cases}
$$

```12:32:modules/planning/path/clothoid_path_optimizer/cost_functions/clothoid_lateral_acceleration_cost_function.cc
double ClothoidLateralAccelerationCostFunction::EvaluateCost(...) const {
  const double kappa = x(ClothoidPathStateIndex::kKappa);
  const double lateral_acceleration = kappa * reference_speed_ * reference_speed_;
  double cost = 0.0;
  if (lateral_acceleration > lateral_acceleration_limit_) {
    const double delta_lateral_acceleration = lateral_acceleration - lateral_acceleration_limit_;
    cost = 0.5 * lateral_acceleration_weight_ * delta_lateral_acceleration * delta_lateral_acceleration;
  } else if (lateral_acceleration < -lateral_acceleration_limit_) {
    ...
  }
  ...
}
```

`EvaluateDerivatives` 函数则是手动算出这个分段二次函数对状态 $x$ 的一阶导（`dfdx`）和二阶导（`d2fdxdx`），这是 iLQR 做二阶展开时必须的信息（第 8.2 节的 $\ell_x,\ell_{xx}$）。因为只有 $\kappa$ 这一维有关，其余分量导数都是 0。

### 13.2 侧向加加速度（Jerk）代价 `clothoid_lateral_jerk_cost_function.cc`

原理完全一样，只是把 $\kappa v^2$（加速度）换成 $\dot\kappa v^3$（加加速度，jerk），衡量的是"侧向力变化有多剧烈"，让乘坐体验更平顺。数学结构和 13.1 一模一样，只是作用在状态的 `kDKappa` 分量上而不是 `kKappa`。

### 13.3 引导线代价 `guide_line_cost_generator.cc`

对应"路径要贴近参考引导线"这个软约束。核心思路：
1. 把引导线在起点、终点各自延长一段（`ExtendGuideLine`），避免路径首尾因为找不到最近点而出问题。
2. 对轨迹上除起点外的每个采样点，找到引导线上离它最近的一小段折线（`SamplePolyline`）。
3. 用"点到折线距离"的平方作为代价（对最后一个点用纯二次代价，中间点用 Huber 代价——即距离较小时是二次代价、距离较大时退化成线性代价，防止个别异常点把梯度带偏，这是鲁棒统计里常见的手法）。
4. 同时为车辆后轴中心和前保险杠中心各生成一条代价（`is_front` 参数），让车头和车身整体都尽量贴合引导线，而不只是后轴。

```17:73:modules/planning/path/clothoid_path_optimizer/utils/guide_line_cost_generator.cc
CostFunctionSet GuideLineCostGenerator::GenerateCostFunction(...) const {
  ...
  for (int i = 1; i < num_path_points; ++i) {
    cost_function_set[i].reserve(2);
    generate_single_cost_function(i, false);  // 后轴中心 to 引导线
    generate_single_cost_function(i, true);   // 车头中心 to 引导线
  }
  return cost_function_set;
}
```

### 13.4 运动学代价打包 `kinematic_constraint_and_cost_generator.cc`

这个文件把第 12.1/12.2 节的**硬约束**（曲率、转向速率不能超限）和第 13.1/13.2 节的**软代价**（超过舒适阈值才罚）、以及一个基础的二次型正则代价（惩罚 $\dot\kappa,\ddot\kappa$ 过大，让路径整体更平滑，对应第 7 节 LQR 里的 $x^TQx+u^TRu$）打包在一起，按采样点位置生成对应的 `ConstraintSet` 和 `CostFunctionSet`：

```31:58:modules/planning/path/clothoid_path_optimizer/utils/kinematic_constraint_and_cost_generator.cc
CostFunctionSet KinematicConstraintAndCostGenerator::GenerateCostFunction(...) const {
  ...
  cost_function_set[i].push_back(std::make_unique<ClothoidLateralAccelerationCostFunction>(...));
  cost_function_set[i].push_back(std::make_unique<ClothoidLateralJerkCostFunction>(...));
  cost_function_set[i].push_back(std::make_unique<QuadCostType>(x_weight, u_weight, x_target));
  ...
}
```

其中 `QuadCostType` 就是标准的二次代价 $\frac12(x-x_{target})^TQ(x-x_{target}) + \frac12 u^TRu$，对应第 7.1 节 LQR 的代价形式——这正是"轨迹优化=LQR的非线性推广"这句话在代码里的直接体现。

---

## 14. 从"猜一条初始路径"到"求解"的全流程

iLQR/AL-iLQR 是**局部优化算法**，需要一个初始猜测轨迹（"名义轨迹"）才能开始做 Backward/Forward Pass。初始猜测的好坏直接影响能否收敛、收敛到哪个局部最优。这就是 `GenerateInitialPaths` 函数要做的事：

```549:650:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
std::vector<ClothoidPath> ClothoidPathOptimizer::GenerateInitialPaths(
    ClothoidPath* reference_path) const {
  ...
  // ① 把参考路径（上一帧结果或车道中心线）采样成初始的 x_trajectory
  for (int i = 1; i < num_samples; ++i) {
    ...
    reference_path->x_trajectory[i](0) = position.x;
    reference_path->x_trajectory[i](1) = position.y;
    reference_path->x_trajectory[i](2) = nullable_reference_path_curve_->Heading(s);
    reference_path->x_trajectory[i](3) = nullable_reference_path_curve_->SignedCurvature(s);
    reference_path->x_trajectory[i](4) = nullable_reference_path_curve_->SignedCurvatureDerivative(s);
    ...
  }
  // ② 这条 reference_path 只是"点位"上贴合参考线，但不一定满足动力学方程（不连续、不可行）
  //    于是再用一个独立的 FeasibleTrajectoryGenerator，在满足动力学约束的前提下，
  //    生成一条尽量贴近 reference_path 的"可行"初始轨迹 initial_path
  const FeasibleTrajectoryGenerator<...> feasible_path_generator(...);
  const base::Optional<...> initial_u_trajectory = feasible_path_generator.Generate(
      steps_, reference_path->x_trajectory, reference_path->u_trajectory);
  ...
  // ③ 用真实动力学模型把 u_trajectory 滚动出对应的 x_trajectory（保证一定满足动力学约束）
  for (int i = 1; i < num_samples; ++i) {
    ...
    initial_path.x_trajectory[i] =
        dynamic_model->Evaluate(initial_path.x_trajectory[i - 1], initial_path.u_trajectory[i - 1]);
  }
  return {initial_path};
}
```

直觉理解：
- **①** 先"拍脑袋"地把参考线（上一帧路径或车道中心线）在各个采样弧长处的坐标、朝向、曲率抄一遍，作为大致方向。但这样抄出来的曲率序列可能是"跳变"的、不满足我们第 3 节定义的动力学方程（比如相邻两点曲率差不满足 $\Delta\kappa = \dot\kappa\cdot\Delta s$）。
- **②③** 所以再单独跑一个小型的可行轨迹生成器（也是基于 iLQR，只不过代价函数很简单——只惩罚"离参考点太远"），把①中"抄"来的轨迹**投影**到"物理上真实可行"的轨迹空间里，确保作为 AL-iLQR 主问题初始解时，动力学约束是天然满足的（不会因为初始解都不可行，导致算法一上来就在"修复动力学违反"上浪费大量迭代）。

之后 `ComputePathCurve` 里，如果开启了 `enable_two_stage_solve()`，还会再做一次"简化版可行性求解"（`GenerateFeasiblePath`，只包含运动学约束+guide line+box collision+障碍物约束，不含车道边界/repulsion 等更精细的代价），进一步把初始解调整到一个更接近最终解、且大概率满足硬约束的状态，这是为了让最终的完整问题（约束更多、代价更复杂）更容易收敛，是一种**分阶段热启动（warm start）策略**。

最终调用 `ConstructPathProblem` 把 `initial_path_set` 和完整的 `ConstraintAndCostSet` 组装成 `PathProblem`，交给 `Solver`（`ALSolver`）求解，也就回到了第 6~9 节讲的 AL-iLQR 算法主体。

---

## 15. 常见疑问 FAQ

**Q1：为什么状态里要把 $\dot\kappa$ 也放进去，而不是直接把 $\kappa$ 当控制量？**

如果直接把 $\kappa$ 当控制量，那 $\kappa$ 可以在相邻两个采样点之间瞬间跳变（因为控制量通常允许分段常数、可以突变），这样路径就不是 $C^1$ 连续的（曲率不连续，方向盘要瞬间打转）。把 $\dot\kappa$ 放进状态、$\ddot\kappa$ 当控制量，可以保证 $\kappa$ 本身是连续、光滑变化的（因为它是状态量，状态量在这套离散动力学下天然连续演化），这正是"Clothoid 路径" $C^2$ 连续性的来源。

**Q2：为什么约束要写成 $\le 0$ 的标准形式，而不是直接写 $\kappa \le \kappa_{max}$？**

统一的标准形式方便通用的求解器代码（`InequalityConstraint` 基类）用同一套逻辑处理所有约束——不管是曲率约束、转向速率约束还是避障约束，只要实现 `Evaluate` 返回 $g(x,u)$、以及对应雅可比即可，上层 AL 框架完全不需要关心每个约束具体是什么物理含义。这是一种典型的**面向对象的接口抽象**设计。

**Q3：为什么要先解一个"可行性问题"（`enable_two_stage_solve`），再解完整问题？**

完整问题的代价函数非常复杂（引导线代价、车道边界代价、避障代价……），如果初始解本身就严重违反硬约束（比如曲率超限），iLQR 的线性化近似在这种"远离可行域"的点上可能非常不准，容易导致收敛失败或收敛到很差的局部解。先解一个"轻量版"问题（只考虑最基础的运动学约束+粗略的引导线代价），能得到一个更接近可行域、更适合做局部近似的起点，相当于给主问题做了一次**热启动**，提高收敛成功率。

**Q4：Huber 代价函数是什么，为什么引导线代价要用它？**

Huber 代价是一种"分段"代价：误差较小时是二次函数（梯度平滑、靠近最优点时收敛快），误差较大时是线性函数（梯度是常数，不会因为个别偏离很远的点产生过大的梯度、把整条轨迹"带偏"）。这在统计学里叫**鲁棒回归**，目的是降低异常值（outlier）的影响。引导线上如果因为障碍物需要绕行，路径会短暂远离引导线，这时如果用纯二次代价，梯度会大到把优化过程"拉扯"得很剧烈；用 Huber 代价就能让这种"合理的远离"不至于产生过大的修正力度。

**Q5：`ClothoidPathDynamic` 只提供了一阶雅可比，AL-iLQR 不是需要二阶信息吗？**

这正是 iLQR（相对于更精确的 DDP）的简化之处：iLQR 只用动力学的一阶导数（雅可比 $A,B$），配合代价函数自身的二阶导数（`d2fdxdx` 等），通过 **Gauss-Newton 近似**拼出 Q-function 的 Hessian（见第 8.2 节公式），从而避免显式计算动力学的二阶导数（这一项在 DDP 里存在，但计算量大、往往对结果影响不大，尤其是当代价函数占主导时）。这是精度和计算效率之间的工程权衡。

---

## 附：核心公式速查表

| 概念 | 公式 |
|---|---|
| 连续动力学 | $\dot x = f(x,u)$ |
| 离散动力学 | $x_{k+1} = f_d(x_k,u_k,\Delta s)$（RK4 积分） |
| 优化问题 | $\min \ell_N(x_N)+\sum \ell_k(x_k,u_k)$ s.t. 动力学、$g_k\le0$、$h_k=0$ |
| 增强拉格朗日函数 | $\mathcal{L}_A = f(x)+\lambda^Tc(x)+\frac12c(x)^TI_\mu c(x)$ |
| 乘子更新（不等式） | $\lambda_i^+=\max\{0,\lambda_i+\mu_ic_i(x^*)\}$ |
| 罚因子更新 | $\mu^+=\phi\mu,\ \phi>1$ |
| LQR 最优反馈 | $u^*=-Kx$，$K=(R+B^TP'B)^{-1}B^TP'A$ |
| iLQR 控制增量 | $\delta u_k^*=K_k\delta x_k+d_k$ |
| 反馈增益 | $K_k=-(Q_{uu}+\rho I)^{-1}Q_{ux}$ |
| 前馈修正 | $d_k=-(Q_{uu}+\rho I)^{-1}Q_u$ |
| 线搜索判据 | $z=\dfrac{J(X,U)-J(\bar X,\bar U)}{-\Delta V(\alpha)}\in[10^{-4},10]$ |

祝学习顺利！建议阅读顺序：先通读一遍本文抓住"是什么问题、为什么这样设计"，再对照代码把每个公式的变量名和代码变量名对应起来读一遍源码，最后可以尝试自己在纸上，对着 `clothoid_path_dynamic.cc` 手推一遍雅可比矩阵，加深理解。
