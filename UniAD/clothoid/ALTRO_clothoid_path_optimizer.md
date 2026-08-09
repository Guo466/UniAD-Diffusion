# ALTRO 论文精讲：从零开始，并与 `clothoid_path_optimizer` 代码逐节对照

> 论文：*ALTRO: A Fast Solver for Constrained Trajectory Optimization*（Taylor A. Howell, Brian E. Jackson, Zachary Manchester, IROS 2019）
> 对照代码：`modules/planning/path/clothoid_path_optimizer/` 及其依赖的求解器 `modules/planning/common/math/solver/`

本文档目标：假设你**没有任何轨迹优化 / 最优控制背景**，把论文中涉及的所有数学概念从最基础的地方讲起，然后把论文的每一个算法模块对应到本仓库 `clothoid_path_optimizer` 里的具体代码实现，让你既懂"论文讲了什么"，也懂"代码是怎么落地这套理论的"。

---

## 目录

1. [写在最前面：这篇论文到底在解决什么问题](#1-写在最前面这篇论文到底在解决什么问题)
2. [预备知识 0：状态、控制、动力学系统是什么](#2-预备知识-0状态控制动力学系统是什么)
3. [预备知识 1：什么是"轨迹优化"（Trajectory Optimization）](#3-预备知识-1什么是轨迹优化trajectory-optimization)
4. [预备知识 2：泰勒展开与二次近似——为什么要"线性化+二次化"](#4-预备知识-2泰勒展开与二次近似为什么要线性化二次化)
5. [预备知识 3：动态规划与"cost-to-go"](#5-预备知识-3动态规划与cost-to-go)
6. [核心算法 A：iLQR（迭代线性二次调节器）逐行推导](#6-核心算法-ailqr迭代线性二次调节器逐行推导)
7. [核心算法 B：增广拉格朗日法（Augmented Lagrangian）从零讲起](#7-核心算法-b增广拉格朗日法augmented-lagrangian从零讲起)
8. [ALTRO 主体：把 iLQR 塞进 AL 框架里](#8-altro-主体把-ilqr-塞进-al-框架里)
9. [ALTRO 的四个"加分项"技巧](#9-altro-的四个加分项技巧)
10. [论文实验结果速览](#10-论文实验结果速览)
11. [总对照表：论文符号 ↔ 代码符号 ↔ 文件位置](#11-总对照表论文符号--代码符号--文件位置)
12. [`clothoid_path_optimizer` 全貌：从论文算法到无人车路径规划](#12-clothoid_path_optimizer-全貌从论文算法到无人车路径规划)
13. [一个完整的手工算例：单步 iLQR 反向传播](#13-一个完整的手工算例单步-ilqr-反向传播)
14. [常见疑问 FAQ](#14-常见疑问-faq)

---

## 1. 写在最前面：这篇论文到底在解决什么问题

假设你要控制一辆车（或者机械臂、无人机）在有限的时间内，从起点走到终点，同时：

- 要尽量"省力"（控制量小、路径平滑）；
- 不能撞到障碍物（约束）；
- 不能超过车轮转角变化率上限（约束）；
- 必须严格服从车辆的运动学规律（动力学方程，不能瞬间移动或瞬间转向）。

这就是一个**带约束的轨迹优化问题**。论文里把它写成公式 (1)：

```
minimize     l_f(x_N) + Σ_{k=0}^{N-1} l(x_k, u_k)
x_{0:N}, u_{0:N-1}

subject to   x_{k+1} = f(x_k, u_k)         <- 动力学约束（等式）
             g_k(x_k, u_k) ≤ 0             <- 不等式约束（比如车道边界、避障）
             h_k(x_k, u_k) = 0             <- 等式约束（比如终点必须精确到达）
```

- `x_k`：第 `k` 个时间步的**状态**（state），比如车辆的位置、朝向、曲率。
- `u_k`：第 `k` 步施加的**控制量**（control/input），比如方向盘转速。
- `l(x_k,u_k)`：**代价函数**（stage cost），你希望它尽量小。
- `l_f(x_N)`：**终端代价**，对最后一步状态的惩罚（比如离终点还有多远）。
- `f(x_k,u_k)`：**动力学方程**，描述"在状态 `x_k` 下施加控制 `u_k`，下一步状态会变成什么"。

解决这类问题历史上有两条路：

| 方法类型 | 代表算法 | 优点 | 缺点 |
|---|---|---|---|
| **直接法**（Direct，如 DIRCOL） | 把所有 `x_k, u_k` 都当成决策变量，丢给通用非线性规划求解器（SNOPT/Ipopt） | 鲁棒、通用、能处理任意约束 | 慢，问题规模大 |
| **间接法**（Indirect，如 iLQR/DDP） | 只把控制量 `u_k` 当决策变量，用动态规划反向传播出最优策略 | 快，天然满足动力学（"anytime feasible"） | 传统上难以处理复杂约束，初始猜测要求高 |

**ALTRO 的核心贡献**：把间接法（iLQR）套进增广拉格朗日框架里，让间接法也能像直接法一样优雅地处理各种约束，同时保持"快"这个优点。论文标题里的 "Augmented Lagrangian TRajectory Optimizer" 就是这个意思。

对应到我们的代码：`ClothoidPathOptimizer` 要求解的问题就是"如何在满足车道边界、避障、曲率/转向速率限制的前提下，规划出一条从当前车辆位置到未来若干米处的平滑路径"——这跟论文的问题设定（公式 1）是完全同构的。

---

## 2. 预备知识 0：状态、控制、动力学系统是什么

如果你完全没接触过控制理论，先建立最基本的直觉。

### 2.1 状态 State

"状态"就是**完整描述系统当前情况所需的最少变量集合**。比如描述一辆沿平面行驶的车，常见状态是：

```
x = [位置x, 位置y, 朝向角θ, 曲率κ, 曲率变化率dκ]
```

这正是 `clothoid_path_optimizer` 采用的状态定义（见第 12 节的代码引用）。为什么要加入"曲率"和"曲率变化率"？因为方向盘不能瞬间打死，需要用平滑的曲率变化来描述真实可行驶的路径（这就是"回旋曲线/Clothoid"的物理意义，后面第 12 节详细讲）。

### 2.2 控制 Control / Input

"控制"是你能**主动施加**的量，用来改变系统未来的状态。比如汽车的油门、刹车、方向盘转速。在 clothoid 路径问题中，控制量是：

```
u = [曲率的二阶导数 ddκ]  （也就是"曲率变化率的变化率"）
```

### 2.3 动力学方程 Dynamics

动力学方程描述了状态如何随时间演化，最一般的写法是微分方程：

```
ẋ(t) = f_c(x(t), u(t))     <- 连续时间动力学
```

或离散形式：

```
x_{k+1} = f(x_k, u_k)      <- 离散时间动力学（每隔 Δt 采样一次）
```

对 clothoid 路径而言，连续动力学非常直观（对应代码 `clothoid_path_dynamic.cc`）：

```
ẋ     = cos(θ)
ẏ     = sin(θ)
θ̇     = κ
κ̇     = dκ
d(dκ) = u   (即 ddκ)
```

直观理解：

- 位置的变化率就是朝向角的余弦/正弦（沿着车头方向走）；
- 朝向角的变化率就是曲率（曲率越大转弯越急）；
- 曲率的变化率是 dκ；
- dκ 的变化率是你能控制的量 `u`。

这五个方程会随时间反复积分，就"滚出"了整条路径。为什么用积分（离散化）而不是直接解析求解？因为真实问题里还叠加了各种代价和约束，无法解析求解，必须用数值优化。

**离散化方法**：连续时间的 `ẋ = f_c(x,u)` 要变成离散的 `x_{k+1} = f(x_k, u_k)`，中间需要"积分器"。最简单的是前向欧拉：

```
x_{k+1} ≈ x_k + f_c(x_k, u_k) · Δt
```

更精确的是四阶龙格库塔（RK4），用 4 个采样点加权平均斜率，精度是欧拉法的 4 阶（误差量级 O(Δt⁵) vs O(Δt²)）：

```
k1 = f_c(x_k, u_k)
k2 = f_c(x_k + 0.5·Δt·k1, u_k)
k3 = f_c(x_k + 0.5·Δt·k2, u_k)
k4 = f_c(x_k + Δt·k3, u_k)
x_{k+1} = x_k + Δt/6 · (k1 + 2k2 + 2k3 + k4)
```

**代码对应**：

```37:126:modules/planning/common/math/solver/integrator.h
template <int X, int U>
class ForwardEuler final : public Integrator<X, U> {
  ...
  VecX Integrate(...) const override {
    const VecX x_dot = dynamics.Evaluate(x, u);
    return x + x_dot * step;
  }
};

template <int X, int U>
class RungeKutta4 final : public Integrator<X, U> {
  ...
  VecX Integrate(...) const override {
    const VecX k1 = dynamics.Evaluate(x, u);
    const VecX k2 = dynamics.Evaluate(x + k1 * 0.5 * step, u);
    const VecX k3 = dynamics.Evaluate(x + k2 * 0.5 * step, u);
    const VecX k4 = dynamics.Evaluate(x + k3 * step, u);
    return x + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
  }
};
```

`clothoid_path_optimizer` 使用的就是 RK4：

```589:600:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
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

`DiscretizedModel` 就是把"连续动力学 + 积分器 + 步长"打包成一个可以直接调用 `Evaluate(x,u)` 得到 `x_{k+1}` 的对象：

```19:58:modules/planning/common/math/solver/discretized_model.h
template <int X, int U>
class DiscretizedModel final : public DynamicModel<X, U> {
 public:
  ...
  VecX Evaluate(const VecX& x, const VecU& u) const override {
    return integrator_->Integrate(*dynamics_, x, u, step_);
  }

  Jacobians EvaluateJacobians(const VecX& x, const VecU& u) const override {
    ...
  }
};
```

---

## 3. 预备知识 1：什么是"轨迹优化"（Trajectory Optimization）

有了状态和动力学的概念，轨迹优化就是：

> 给定初始状态 `x_0`，找一串控制量 `u_0, u_1, ..., u_{N-1}`（进而通过动力学方程滚出状态序列 `x_0, x_1, ..., x_N`），使得总代价 `Σ l(x_k,u_k) + l_f(x_N)` 最小，同时满足所有约束。

这是一个典型的**非线性、高维、带约束的优化问题**。直接用通用求解器（梯度下降/牛顿法）很难处理，因为：

1. 变量维度高（`N` 个时间步 × 状态维度）；
2. 动力学约束是非线性的（`x_{k+1} = f(x_k, u_k)` 里 `f` 通常非线性，比如上面的 `cos(θ)`）；
3. 还有额外的不等式约束（避障、限速）。

这就是论文第二章要引入 iLQR（间接法）和增广拉格朗日（约束处理）这两把"武器"的原因。

---

## 4. 预备知识 2：泰勒展开与二次近似——为什么要"线性化+二次化"

在展开 iLQR 之前，必须先理解一个贯穿全文的数学套路：**用二阶泰勒展开局部近似一个复杂的非线性函数**。

### 4.1 一元函数的泰勒展开

对于一个光滑函数 `f(x)`，在点 `x₀` 附近可以展开为：

```
f(x₀+δ) ≈ f(x₀) + f'(x₀)·δ + (1/2)·f''(x₀)·δ²
```

- 一次项 `f'(x₀)` 告诉你函数在这一点"往哪个方向变化最快"（梯度/导数）；
- 二次项 `f''(x₀)` 告诉你这个变化趋势"弯曲"得多厉害（曲率/二阶导数，也叫 Hessian 的一维版本）。

如果只用一次项近似（线性近似），你能做梯度下降；如果用到二次项（二次近似），你能做牛顿法——牛顿法通常收敛快得多，因为它同时利用了"往哪走"和"走多远合适"的信息。

### 4.2 多元函数的泰勒展开（论文的记号）

论文开篇的记号约定（Notation 部分）就是在讲这个：

```
f_x  ≡ ∂f(x,u)/∂x |_{x_k,u_k}         <- 一阶偏导（梯度/雅可比）
f_xx ≡ ∂²f(x,u)/∂x² |_{x_k,u_k}       <- 二阶偏导（Hessian）
f_xu ≡ ∂²f(x,u)/∂x∂u |_{x_k,u_k}      <- 交叉二阶偏导
```

对一个多元函数 `f(x,u)`，在 `(x_k,u_k)` 附近做二阶泰勒展开：

```
f(x_k+δx, u_k+δu) ≈ f(x_k,u_k)
                   + f_x·δx + f_u·δu                              (一次项)
                   + (1/2)[δx;δu]ᵀ [f_xx f_xu; f_xu^T f_uu] [δx;δu]  (二次项)
```

**为什么整篇论文都在做这件事？** 因为原问题的代价函数、动力学方程、约束函数都是非线性的，直接优化非常困难。但如果我们只关心"当前解附近一个很小邻域内怎么改进"，那么用二阶泰勒展开去局部近似，就把一个难的非线性问题变成了一个简单的**二次规划问题（QP）**——而二次规划是有解析解的（对无约束情况，就是解一个线性方程组）。

这就是 iLQR 的核心思想：**在当前轨迹附近，把动力学线性化、把代价二次化，求解这个近似问题得到一个"改进方向"，然后沿着这个方向走一小步，再重新线性化/二次化……如此反复迭代，直到收敛**。这跟"用很多段切线/切面拼接近似一条曲线"是同一个思路。

**代码对应**：`CostFunction::EvaluateDerivatives` 和 `DynamicModel::EvaluateJacobians` 就是在计算这些一阶、二阶导数：

```278:298:modules/planning/common/math/solver/solver.h
typename CostFunction<X, U>::Derivatives terminal_cost_derivatives;
problem.terminal_cost_function().EvaluateDerivatives(
    x_trajectory[num_steps], VecU::Zero(), &terminal_cost_derivatives);
MatXX V_xx = terminal_cost_derivatives.d2fdxdx;
VecX V_x = terminal_cost_derivatives.dfdx;
...
typename CostFunction<X, U>::Derivatives cost_derivatives;
problem.cost_function(i).EvaluateDerivatives(
    x_trajectory[i], u_trajectory[i], &cost_derivatives);
const VecX& l_x = cost_derivatives.dfdx;
const MatXX& l_xx = cost_derivatives.d2fdxdx;
const VecU& l_u = cost_derivatives.dfdu;
const MatUU& l_uu = cost_derivatives.d2fdudu;
const MatUX& l_ux = cost_derivatives.d2fdudx;
typename DynamicModel<X, U>::Jacobians jacobians =
    problem.dynamic_model(i).EvaluateJacobians(x_trajectory[i], u_trajectory[i]);
const MatXX& f_x = jacobians.dfdx;
const MatXU& f_u = jacobians.dfdu;
```

变量名 `l_x, l_xx, l_u, l_uu, l_ux, f_x, f_u` 与论文记号 `l_x, l_xx, ...` 完全一一对应，这是刻意为了方便对照论文而保留的命名风格（代码注释里也明说了 "We follow the variable namings in the paper"）。

---

## 5. 预备知识 3：动态规划与"cost-to-go"

### 5.1 动态规划的核心思想

动态规划（Dynamic Programming, DP）解决的是"多阶段决策"问题：如果一个问题可以被分解成一系列子问题，且"整体最优解"必然包含"子问题的最优解"（最优子结构），那么可以从最后一步往前递推，逐步求出每一步的最优策略。

对轨迹优化问题，"阶段"就是时间步 `k=0,1,...,N`。定义**最优代价函数（cost-to-go / value function）**：

```
V_k(x) = 从状态 x 出发，走完剩下所有步骤（k 到 N），能达到的最小总代价
```

这满足贝尔曼方程（Bellman equation），也就是论文的公式 (4)(5)(6)：

```
V_N(x_N) = l_f(x_N)                                          式(4) 终端条件
V_k(x_k) = min_{u_k} { l(x_k,u_k) + V_{k+1}(f(x_k,u_k)) }     式(5)
         = min_{u_k} Q_k(x_k,u_k)                             式(6)
```

`Q_k(x_k,u_k) = l(x_k,u_k) + V_{k+1}(f(x_k,u_k))` 叫做**动作价值函数**（action-value function，跟强化学习里的 Q 函数是同一个概念）：它表示"在状态 `x_k` 下，先执行动作 `u_k`，然后按最优策略走完剩下的路，一共要花多少代价"。

**直觉类比**：想象你在下棋，`V_k(x)` 是"从当前棋局 `x` 出发，双方都按最优策略走，我方最终能赢多少分"；`Q_k(x,u)` 是"如果我现在走这一步 `u`，之后再按最优策略走，能赢多少分"。动态规划就是从终局往前"倒推"，把每个局面下"走哪一步最优"都算出来。

### 5.2 为什么要从后往前算（Backward Pass）？

因为 `V_k` 的定义依赖 `V_{k+1}`（"剩下的最优代价"），所以必须先知道最后一步的代价 `V_N`，再倒推 `V_{N-1}`，再 `V_{N-2}`……一直推到 `V_0`。这个"从后往前"的过程就叫**反向传播 / Backward Pass**（论文 Algorithm 2，代码里是 `BackwardIteration`）。

问题是：对于连续状态空间、非线性系统，`V_k(x)` 是一个定义在整个状态空间上的函数，没法精确表示（除非是线性二次系统）。这就是为什么要用第 4 节的"二阶泰勒展开"技巧：**用二次函数局部近似 `V_k(x)`**（论文式 7）：

```
δV_k(x_k) ≈ p_k^T δx_k + (1/2) δx_k^T P_k δx_k
```

这里 `p_k`（向量）和 `P_k`（矩阵）分别是 `V_k` 在当前轨迹点附近的一阶、二阶近似系数（可以理解为"局部梯度"和"局部曲率矩阵"）。整个反向传播就是递推地计算这一串 `p_k, P_k`（从 `k=N` 到 `k=0`）。这正是 iLQR 算法的核心。

---

## 6. 核心算法 A：iLQR（迭代线性二次调节器）逐行推导

现在把第 4、5 节的工具组装起来，完整推导 iLQR。

### 6.1 什么是 LQR（不迭代的版本）

如果动力学是**线性**的（`x_{k+1} = A_k x_k + B_k u_k`），代价是**二次**的（`l = x^T Q x + u^T R u`），那么这个问题叫"线性二次调节器"问题（LQR），它有**精确解析解**——不需要迭代，直接反向递推一遍 Riccati 方程就能得到全局最优解。

但现实中的动力学几乎总是非线性的（比如 clothoid 动力学里的 `cos(θ)`）。iLQR 的想法是：

> 在当前的一条轨迹（状态-控制序列）附近，把非线性动力学**线性化**、把代价函数**二次化**，得到一个近似的 LQR 问题，解出这个 LQR 问题得到一个"修正量"，用这个修正量更新轨迹，然后在新的轨迹上重复这个过程，直到收敛。

这就是"迭代"的 LQR，即 iLQR。

### 6.2 反向传播（Backward Pass）——论文式 (10)~(19)，Algorithm 2

从终端条件开始（论文式 8、9）：

```
p_N = ∂l_f(x)/∂x |_{x_N}          <- 终端代价的梯度
P_N = ∂²l_f(x)/∂x² |_{x_N}        <- 终端代价的 Hessian
```

然后对每个 `k` 从 `N-1` 递减到 `0`，展开动作价值函数 `Q_k`（论文式 10）：

```
δQ_k = (1/2)[δx_k;δu_k]^T [Q_xx Q_xu; Q_ux Q_uu] [δx_k;δu_k] + [Q_x;Q_u]^T [δx_k;δu_k]
```

其中各分块矩阵是（论文式 11~15）：

```
Q_xx = l_xx + A_k^T P_{k+1} A_k
Q_uu = l_uu + B_k^T P_{k+1} B_k
Q_ux = l_ux + B_k^T P_{k+1} A_k
Q_x  = l_x  + A_k^T p_{k+1}
Q_u  = l_u  + B_k^T p_{k+1}
```

这里 `A_k = ∂f/∂x`，`B_k = ∂f/∂u` 是动力学在当前点的雅可比（一阶导数），可以理解为"把下一步的 cost-to-go 二次型，通过线性化的动力学，拉回到当前时刻的状态-控制空间里"。

**直觉理解**：`Q_uu` 告诉你"在这一步稍微改变一点控制量 `u`，对总代价的二阶影响有多大"（越大说明这个方向的控制对代价越敏感）；`Q_ux` 告诉你"状态的扰动和控制的扰动如何耦合影响代价"。

**求最优控制修正量**：对 `δQ_k` 关于 `δu_k` 求最小值（这是一个无约束二次函数求最小值，直接令梯度为零），得到（论文式 16）：

```
δu_k* = -(Q_uu + ρI)^{-1} (Q_ux·δx_k + Q_u) ≡ K_k·δx_k + d_k
```

拆成两部分：

- `d_k = -(Q_uu+ρI)^{-1} Q_u`：**前馈项**（feedforward），不依赖 `δx_k`，是"标称"的修正量；
- `K_k = -(Q_uu+ρI)^{-1} Q_ux`：**反馈增益矩阵**（feedback gain），乘以状态偏差 `δx_k` 后加到控制上，起到"实时纠偏"的作用（类似 PID 里的比例反馈）。

`ρI` 是**正则化项**（regularization），加上它是为了保证 `Q_uu+ρI` 一定可逆（数值稳定性），`ρ` 太小可能矩阵接近奇异，太大又会让每步更新过于保守。代码里对应 `mu`（即 `Q_uu_reg`）。

把 `δu_k*` 代回 `δQ_k`，可以解出当前时刻的 `p_k, P_k`（论文式 17、18）：

```
p_k = Q_x + K_k^T Q_uu d_k + K_k^T Q_u + Q_xu d_k
P_k = Q_xx + K_k^T Q_uu K_k + K_k^T Q_ux + Q_xu K_k
```

以及**预期的代价改进量**（论文式 19，用于后面的线搜索判断）：

```
ΔV_k = d_k^T Q_u + (1/2) d_k^T Q_uu d_k
```

**代码对应**（`BackwardIteration`，几乎是论文公式的逐字翻译）：

```264:348:modules/planning/common/math/solver/solver.h
template <int X, int U, typename Problem>
bool Solver<X, U, Problem>::BackwardIteration(...) const {
  ...
  MatXX V_xx = terminal_cost_derivatives.d2fdxdx;   // 对应 P_N
  VecX V_x = terminal_cost_derivatives.dfdx;         // 对应 p_N
  ...
  for (int i = num_steps - 1; i >= 0; --i) {
    ...
    VecX Q_x = l_x + f_x.transpose() * V_x;
    VecU Q_u = l_u + f_u.transpose() * V_x;
    MatXX Q_xx = l_xx + f_x.transpose() * V_xx * f_x;
    MatUU Q_uu = l_uu + f_u.transpose() * V_xx * f_u;
    MatUX Q_ux = l_ux + f_u.transpose() * V_xx * f_x;
    MatUU Q_uu_reg = l_uu + f_u.transpose() * V_xx * f_u + meta->mu * MatUU::Identity();
    ...
    meta->k_trajectory[i] = -Q_uu_reg_inv * Q_u;      // 对应 d_k
    meta->K_trajectory[i] = -Q_uu_reg_inv * Q_ux;     // 对应 K_k
    meta->dV[0] += meta->k_trajectory[i].transpose() * Q_u;
    meta->dV[1] += (0.5 * meta->k_trajectory[i].transpose() * Q_uu * meta->k_trajectory[i]).value();
    V_x = Q_x + meta->K_trajectory[i].transpose() * Q_uu * meta->k_trajectory[i] +
          meta->K_trajectory[i].transpose() * Q_u + Q_ux.transpose() * meta->k_trajectory[i];
    V_xx = Q_xx + meta->K_trajectory[i].transpose() * Q_uu * meta->K_trajectory[i] +
           meta->K_trajectory[i].transpose() * Q_ux + Q_ux.transpose() * meta->K_trajectory[i];
    V_xx = 0.5 * (V_xx + V_xx.transpose());  // 强制对称，消除数值误差
  }
  return true;
}
```

一一对应关系：`V_x↔p_k, V_xx↔P_k, meta->k_trajectory↔d_k（前馈）, meta->K_trajectory↔K_k（反馈增益）, meta->mu↔ρ（正则化）`。

注意代码用的是 `mu` 命名（来自另一篇经典论文 "Control-Limited DDP"），而 ALTRO 论文里用 `ρ`，两者是同一个东西——数值稳定性正则化系数。

**关于控制量上下界的处理**（代码里 `action_limit` 分支）：如果控制量有边界约束（比如方向盘转速上限），代码会用一个内层的"投影牛顿法"来求解带边界的二次规划（`OptimizeActionModification`），这是论文之外，Tassa et al. "Control-Limited DDP" 论文的技巧，本仓库也吸收了进来，用于处理 box 约束的控制量。这不属于 ALTRO 论文本身讨论的范围，但属于同一套 iLQR/DDP 体系的扩展。

### 6.3 前向传播（Forward Pass）——论文式 Algorithm 3

反向传播算出了每一步的 `K_k, d_k`，接下来要用它们**真正地更新一遍轨迹**：

```
ū_k = u_k + K_k(x̄_k - x_k) + α·d_k       <- 论文 Algorithm 3 第4行
x̄_{k+1} = f(x̄_k, ū_k)                     <- 用真实（非线性）动力学滚出下一状态
```

注意这里：

- `x̄_k - x_k` 是"新旧轨迹在这一步的状态偏差"，`K_k` 把这个偏差转换为控制的反馈修正——这保证了新轨迹即使跟旧轨迹有偏差，依然能沿着"最优反馈策略"走，而不是死板地套用旧的控制量；
- `α` 是**线搜索步长**（line search step size），从 `1` 开始尝试，如果这一步导致代价没有充分下降，就把 `α` 减小再试（比如乘 0.5），直到代价确实下降为止。这保证了 iLQR 的**全局收敛性**（每次迭代代价单调不增）。
- 用的是**真实的非线性动力学** `f(x̄_k,ū_k)` 来滚出新轨迹（而不是线性化后的近似），这保证了每一步迭代后的轨迹永远是"动力学可行"的——这也是间接法相比直接法的天然优势（"anytime dynamically feasible"，论文反复强调的一点）。

**代码对应**：

```238:262:modules/planning/common/math/solver/solver.h
template <int X, int U, typename Problem>
void Solver<X, U, Problem>::ForwardIteration(...) const {
  ...
  (*x_trajectory_updated)[0] = x_trajectory[0];
  const int num_steps = problem.num_steps();
  for (int i = 0; i < num_steps; ++i) {
    (*u_trajectory_updated)[i] =
        u_trajectory[i] + context.alpha * context.k_trajectory[i] +
        context.K_trajectory[i] * ((*x_trajectory_updated)[i] - x_trajectory[i]);
    if (action_limit) {
      ClampAction(*action_limit, &((*u_trajectory_updated)[i]));
    }
    (*x_trajectory_updated)[i + 1] =
        problem.dynamic_model(i).Evaluate((*x_trajectory_updated)[i], (*u_trajectory_updated)[i]);
  }
}
```

线搜索（尝试一系列 `alpha_values_`，从 1 逐渐缩小到 0.001）：

```588:756:modules/planning/common/math/solver/solver.h
for (double alpha : alpha_values_) {
  ...
  meta.alpha = alpha;
  ForwardIteration(problem, meta, x_trajectory, u_trajectory, action_limit,
                   &x_trajectory_updated, &u_trajectory_updated);
  cost_updated = ComputeCost(problem, x_trajectory_updated, u_trajectory_updated, ...);
  meta.cost_improvement = cost_history.back() - cost_updated;
  meta.expected_cost_improvement = -meta.alpha * (meta.dV[0] + meta.alpha * meta.dV[1]);
  ...
  if (meta.cost_improvement_ratio > line_search_min_cost_improvement_ratio_) {
    is_forward_succeeded = true;
    ...
    break;
  }
}
```

`meta.expected_cost_improvement = -α(ΔV_0 + αΔV_1)` 正是论文式 (19) `ΔV_k = d_k^T Q_u + (1/2)d_k^T Q_uu d_k` 在乘以步长 `α` 之后的形式（因为 `d_k` 要乘 `α` 才是实际施加的修正量，二次项要乘 `α²`）——**这一步是判断"这次改进是否符合预期"的关键**，如果实际代价下降比预期差太多（`cost_improvement_ratio` 太小），就认为这个二次近似不可信，需要缩小步长重试，或增大正则化 `mu` 重新做反向传播。

### 6.4 完整 iLQR 循环——论文 Algorithm 1

```
1: 初始化 x0, U（控制序列的初值）, tolerance
2: 用初始控制滚出初始状态轨迹 X
3: 循环：
4:    J ← 计算当前总代价
5:    do:
6:       J_prev ← J
7:       K, d, ΔV ← BackwardPass(X, U)     <- 第 6.2 节
8:       X, U, J ← ForwardPass(...)         <- 第 6.3 节（含线搜索）
9:    while |J - J_prev| > tolerance         <- 收敛判据：代价不再显著下降
10:   返回 X, U, J
```

**代码对应**：`Solver<X,U,Problem>::Solve()` 的主循环，正是这个 do-while 结构（用 `for (; iter < max_num_iter_; ++iter)` 实现，退出条件在循环体内部判断）：

```641:819:modules/planning/common/math/solver/solver.h
for (; iter < max_num_iter_; ++iter) {
  ...
  // 反向传播（若失败则增大正则化重试）
  while (meta.mu < max_mu_) {
    is_backward_succeeded = BackwardIteration(...);
    if (is_backward_succeeded) { ... break; }
    IncreaseRegularization(&meta);
  }
  // 梯度足够小 -> 收敛退出
  meta.normalized_grad = ComputeNormalizedGrad(meta.k_trajectory, u_trajectory);
  if (meta.normalized_grad < min_grad_thresh_ && meta.mu < grad_exit_mu_thresh_) {
    solution.is_solved = true;
    break;
  }
  // 前向传播 + 线搜索
  if (is_backward_succeeded) {
    for (double alpha : alpha_values_) { ... }
  }
  if (is_forward_succeeded) {
    x_trajectory = std::move(x_trajectory_updated);
    u_trajectory = std::move(u_trajectory_updated);
    cost_history.push_back(cost_updated);
    if (meta.cost_improvement < min_cost_improvement_threshold_ || ...) {
      solution.is_solved = true;   // 代价改进量足够小 -> 收敛
      break;
    }
    ReduceRegularization(&meta);   // 成功了就放松正则化
  } else {
    IncreaseRegularization(&meta); // 失败了就加强正则化
    if (meta.mu > max_mu_) { solution.is_solved = false; break; }
  }
}
```

这里额外多了两个工程细节，是论文没有细讲但代码里非常重要的部分：

1. **多个初始猜测的竞争**：`FindBestInitSolution` 会同时评估多条候选初始轨迹（`candidate_init_solutions`），选代价最小的作为起点——这是因为非凸问题对初值敏感，工程上常常准备几条"候选路径"（比如贴左边界走、贴右边界走、居中走）分别跑一遍，取更优的。

2. **正则化的自适应调整**（`IncreaseRegularization`/`ReduceRegularization`）：

```350:364:modules/planning/common/math/solver/solver.h
template <int X, int U, typename Problem>
void Solver<X, U, Problem>::IncreaseRegularization(IterationMeta* meta) const {
  meta->delta_mu = std::max(delta_mu_factor_, meta->delta_mu * delta_mu_factor_);
  meta->mu = std::max(min_mu_, meta->mu * meta->delta_mu);
}

template <int X, int U, typename Problem>
void Solver<X, U, Problem>::ReduceRegularization(IterationMeta* meta) const {
  meta->delta_mu = std::min(1.0 / delta_mu_factor_, meta->delta_mu / delta_mu_factor_);
  if (meta->mu * meta->delta_mu > min_mu_) {
    meta->mu *= meta->delta_mu;
  } else {
    meta->mu = 0.0;
  }
}
```

这是一种"自适应步长"策略：如果这次迭代成功（代价确实下降了），就减小正则化 `mu`（下次更信任二次近似，走大步）；如果失败了（`Q_uu` 不正定或线搜索找不到合适步长），就增大 `mu`（相当于把问题变得更像梯度下降，更保守但更稳）。这跟 Levenberg-Marquardt 算法里阻尼因子的自适应调整逻辑是同一套思想。

---

## 7. 核心算法 B：增广拉格朗日法（Augmented Lagrangian）从零讲起

iLQR 本身只能解决**无约束**问题（论文式 2、3，只有动力学约束，没有 `g_k, h_k`）。但真实问题有大量不等式/等式约束（避障、车道边界）。第七节要讲的就是怎么把约束"塞进"代价函数里，让 iLQR 依然能用。

### 7.1 为什么不能直接把约束丢掉，简单加个大权重的惩罚项？

一种朴素想法：把约束 `c(x)≤0` 变成惩罚项 `μ·max(0,c(x))²` 加到代价里，`μ` 取一个很大的数，这样违反约束时代价会暴涨，优化器自然会避开。这叫**纯罚函数法（penalty method）**。

问题：`μ` 太小则约束满足不精确（优化器会为了降低其他代价，"稍微"违反约束换取更低的总代价）；`μ` 太大则代价函数变得病态（数值上极度敏感，一阶/二阶导数在惩罚区域附近爆炸式增长，导致优化器数值不稳定，收敛困难）。这是纯罚函数法的经典缺陷。

### 7.2 增广拉格朗日法的改进

增广拉格朗日（Augmented Lagrangian, AL）法巧妙地结合了"拉格朗日乘子法"和"罚函数法"，用一个**逐渐更新的乘子 `λ`** 来避免把 `μ` 无限增大。原问题（论文式 20）：

```
minimize f(x)
subject to c_I(x) ≤ 0,  c_E(x) = 0
```

定义增广拉格朗日函数（论文式 21）：

```
L_A(x,λ,μ) = f(x) + λ^T c(x) + (1/2) c(x)^T I_μ c(x)
```

拆开看：

- `f(x)`：原始代价；
- `λ^T c(x)`：**一阶拉格朗日项**，`λ` 是拉格朗日乘子（可正可负，等式约束下），作用是"精确地"把约束的影子价格编码进代价里；
- `(1/2)c(x)^T I_μ c(x)`：**二阶罚项**，`μ` 是罚参数，作用是让违反约束的代价随违反程度平方增长，帮助数值稳定并加速收敛。

对于不等式约束，`I_μ` 是一个对角矩阵，定义为（论文式 22）：

```
I_μ,ii = 0                  如果 c_i(x)<0 且 λ_i=0（约束未激活，且乘子也是0，不需要惩罚）
       = μ_i               否则（约束被违反，或虽然满足但乘子非零——说明该约束在边界上"起作用"）
```

这个条件判断的意义是：不等式约束只有在"被触碰"或"曾经被触碰"（乘子非零）时才需要惩罚，如果本来就满足得很宽松（松弛变量为负且乘子为零），就不需要对它加惩罚——这跟 KKT 条件里的"互补松弛性"是一致的。

### 7.3 外层迭代：交替更新原变量、乘子、罚参数

AL 方法是一个**双层循环**：

- **内层**：固定 `λ, μ`，用某个求解器（这里就是 iLQR）最小化 `L_A(x,λ,μ)`，得到近似最优解 `x̂*`；
- **外层**：根据 `x̂*` 更新 `λ` 和 `μ`，再重新做内层优化，如此反复，直到约束违反量足够小。

乘子更新公式（论文式 23，本质是对 `L_A` 关于 `x` 求梯度为零的一阶必要条件推出）：

```
λ_i^{new} = λ_i + μ_i·c_i(x̂*)                    对等式约束 i∈E
          = max(0, λ_i + μ_i·c_i(x̂*))            对不等式约束 i∈I（保证乘子非负）
```

**直觉理解**：如果约束被违反（`c_i(x̂*) > 0`），说明惩罚力度不够，下一轮就把乘子调大，加重惩罚；如果约束满足得很好（`c_i(x̂*)` 很负），且当前乘子已经是 0，`max(0,...)` 会让它保持在 0（不需要惩罚一个本来就没问题的约束）。

罚参数更新（论文式 24，单调递增的调度策略）：

```
μ_i^{new} = φ_i · μ_i         其中 φ_i > 1（比如取 10）
```

每轮外层迭代都把罚参数放大 `φ` 倍，让约束惩罚越来越"严格"。这个单调递增的策略保证了理论上的收敛性，但正如论文所说，"AL methods make rapid initial progress, but suffer from slow constraint convergence once the penalty is capped at a maximum finite value"——这也是论文后面要引入"投影牛顿法作为第二阶段"的动机（第 9.4 节）。

### 7.4 Algorithm 4（论文原文）与代码对照

```
1: function AL(x0, SOLVER, tolerance)
2:   初始化 λ, μ, φ
3:   while max(c) > tolerance:
4:      用 SOLVER（这里是iLQR）最小化 L_A(x,λ,μ)
5:      更新 λ (式23), 更新 μ (式24)
6:   返回 X, λ
```

**代码对应**：`ALSolver<X,U,C1,C2>::Solve`

```155:206:modules/planning/common/math/solver/augmented_lagrangian/al_solver.h
template <int X, int U, int C1, int C2>
typename ALSolver<X, U, C1, C2>::Solution ALSolver<X, U, C1, C2>::Solve(
    const ConstrainedProblem<X, U, C1, C2>& constrained_problem) const {
  ALProblem problem(constrained_problem);
  ResetDuals(&problem);       // 初始化 λ = initial_dual (通常是0)
  ResetPenalties(&problem);   // 初始化 μ = initial_penalty
  Status status;
  ...
  for (int iter = 0; iter < config_.max_outer_iterations; ++iter) {
    // ---- 内层：调用 iLQR 求解当前 λ, μ 下的无约束问题 ----
    const IlqrSolution inner_solution = ilqr_solver_.Solve(problem, ...);
    UpdateStatus(problem, inner_solution, solve_start_time, &status);
    const State state = CheckTerminationCondition(status);
    ...
    if (state != State::kUnsolved) {
      return Solution{...};   // 约束违反量已足够小，或达到失败条件，退出
    }
    // ---- 外层：更新乘子和罚参数 ----
    UpdateDuals(inner_solution, &problem);      // 对应式(23)
    UpdatePenalties(&problem);                  // 对应式(24)
    ...
  }
}
```

乘子更新的具体实现（在 `ConstraintPenalty::UpdateDual` 里，对不等式约束会有 `max(0,...)` 逻辑，等式约束没有这个截断）：

```63:70:modules/planning/common/math/solver/augmented_lagrangian/constraint_penalty.h
void UpdateDual(const VecX& x, const VecU& u) {
  lambda_ = constraint_->UpdateDual(x, u, lambda_, scaled_mu_);
}

void UpdatePenalty(double scale) {
  mu_ = constraint_->UpdatePenalty(mu_, scale);
  scaled_mu_ = constraint_->UpdatePenalty(scaled_mu_, scale);
}
```

`scale` 就是论文里的 `φ`，对应配置项 `penalty_scaling_factor`（默认值可以在 `ALSolverConfig` 里看到，注释里明确写着 "\phi in ALTRO paper"）：

```47:57:modules/planning/common/math/solver/augmented_lagrangian/al_solver.h
struct Config {
  int max_outer_iterations = 30;
  int max_total_iterations = 300;
  double max_solve_time_ms = 1e5;        // Max total solve time in ms;
  double constraint_tolerance = 1e-4;    // Maximum constraint violation threshold
  double initial_dual = 0.0;             // Initial lambda
  double initial_penalty = 1.0;          // Inital mu
  double penalty_scaling_factor = 10.0;  // \phi in ALTRO paper
  double max_penalty = 1e8;              // Maximum penalty parameter allowed
  int verbose_level = 0;                 // 0=>low, 1=>medium, 2=>high
};
```

**终止条件判断**（对应论文 `while max(c)>tolerance`，但工程上还要加上"迭代次数、超时、罚参数超上限"等失败保护）：

```257:279:modules/planning/common/math/solver/augmented_lagrangian/al_solver.h
template <int X, int U, int C1, int C2>
typename ALSolver<X, U, C1, C2>::State ALSolver<X, U, C1, C2>::CheckTerminationCondition(
    const Status& status) const {
  if (status.is_inner_timed_out) {
    return State::kFailedTimeout;
  }
  if (!status.is_ilqr_solved) {
    return State::kFailedInnerIlqrNotConverge;
  }
  if (status.max_violation < config_.constraint_tolerance) {
    return State::kSolved;              // <- 论文的核心判据: max(c) < tolerance
  }
  if (status.max_penalty > config_.max_penalty) {
    return State::kFailedMaxPenalty;
  }
  if (status.outer_iterations >= config_.max_outer_iterations) {
    return State::kFailedMaxOuterIterations;
  }
  if (status.total_iterations >= config_.max_total_iterations) {
    return State::kFailedMaxTotalIterations;
  }
  return State::kUnsolved;
}
```

---

## 8. ALTRO 主体：把 iLQR 塞进 AL 框架里

论文第三章开头一句话点明了 ALTRO 的整体架构（Algorithm 6）：

```
ALTRO 分两个阶段：
第一阶段：在 AL 框架下用 iLQR 求解无约束子问题，快速收敛到"粗糙容差"
第二阶段（可选）：用第一阶段的解热启动一个"投影牛顿法"，达到高精度约束满足
```

```
1: procedure ALTRO
2:   初始化 x0, U, tolerance; X̃（期望状态轨迹，可选）
3:   if 使用不可行初始化 then
4:      X ← X̃, W ← 由式(37)计算的"虚拟控制"
5:   else
6:      X ← 用 U 滚出的可行轨迹
7:   (X,U), λ ← AL_iLQR((X,U), ILQR, tol.)      <- 第一阶段（本文第7、8节讲的东西）
8:   (X,U) ← ProjectedNewton((X,U,λ), tol.)      <- 第二阶段（第9.4节，可选）
9:   return X, U
```

**代码对应**：`ClothoidPathOptimizer` 里对应第一阶段的调用是这样的（`ComputePathCurve` 函数）：

```223:248:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
// Solve prblem.
const PathProblem path_problem =
    ConstructPathProblem(*initial_path_set, &constraint_and_cost_set);
const absl::Time timestamp_3 = absl::Now();
const Solver solver(al_solver_config_, ilqr_solver_config_);
const Solver::Solution solution = solver.Solve(path_problem);
...
if (solution.state != Solver::State::kSolved) {
  ReportSolveFailedEvent("Failed to solve: " + std::to_string(static_cast<int>(solution.state)));
  return base::none;
}
return ConvertToCurve(solution.ilqr_solution.x_trajectory);
```

`Solver` 就是 `ALSolver<kNumStates, kNumControls, Eigen::Dynamic, Eigen::Dynamic>` 的别名：

```38:41:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.h
using Solver = ALSolver<kNumStates, kNumControls, Eigen::Dynamic, Eigen::Dynamic>;
using PathProblem = ConstrainedProblem<kNumStates, kNumControls, Eigen::Dynamic, Eigen::Dynamic>;
using ALSolverConfig = Solver::Config;
using IlqrSolverConfig = Solver::IlqrSolver::Config;
```

也就是说，`ALSolver::Solve` 内部会自动调用 `IlqrSolver::Solve`（第 6 节的 iLQR）——这正是 Algorithm 6 中的第 7 行 `AL_iLQR`。

值得注意的是：`clothoid_path_optimizer` **没有实现论文的第二阶段（投影牛顿法/Algorithm 5）**，只用了第一阶段（AL + iLQR）。这是工程上的合理取舍——车辆路径规划对约束满足精度的要求（比如 `constraint_tolerance`）远没有论文里机械臂避障场景（要求 `1e-8`）那么苛刻，`1e-4` 量级的容差配合足够的安全 buffer 已经足够安全；而且投影牛顿法需要显式构造大规模稀疏 KKT 矩阵求逆，在车载实时计算资源上代价过高。这一点会在第 9.4 节详细展开原因。

### 8.1 `ClothoidPathProblem`（状态/控制维度、约束类型）与论文记号对照

```35:37:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.h
class ClothoidPathOptimizer final {
  static constexpr int kNumStates = ClothoidPathStateIndex::kSize;      // X = 5
  static constexpr int kNumControls = ClothoidPathControlIndex::kSize; // U = 1
```

```19:26:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer_common.h
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

对照论文公式 (1) 的记号：`x_k = [x,y,θ,κ,dκ]^T ∈ R^5`（对应论文的 `x∈R^n`），`u_k = [ddκ] ∈ R^1`（对应论文的 `u∈R^m`）。`C1`（等式约束数）和 `C2`（不等式约束数）在本工程里都取 `Eigen::Dynamic`，因为不同路径规划场景下约束数量是动态变化的（障碍物数量、车道边界数量都不固定）。

`ConstrainedProblem` 就是论文式 (1) 的直接映射（`equality_constraints` 对应 `h_k`，`inequality_constraints` 对应 `g_k`）：

```967:1002:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
ClothoidPathOptimizer::PathProblem ClothoidPathOptimizer::ConstructPathProblem(
    const std::vector<ClothoidPath>& initial_path_set,
    ConstraintAndCostSet* constraint_and_cost_set) const {
  PathProblem path_problem;
  path_problem.problem =
      std::make_unique<Problem<kNumStates, kNumControls>>(initial_path_set[0].x_trajectory[0]);
  const int num_steps = steps_.size();
  path_problem.equality_constraints.resize(num_steps + 1);   // h_k：这里全为空指针（无等式约束）
  path_problem.inequality_constraints.reserve(num_steps + 1); // g_k：来自各种约束生成器
  for (int i = 0; i < num_steps; ++i) {
    std::unique_ptr<DynamicModel<kNumStates, kNumControls>> dynamic_model =
        GetDynamicModelAtSample(i);                            // 对应 f(x_k,u_k)
    std::unique_ptr<CostFunction<kNumStates, kNumControls>> cost_function_vector =
        CollectCostsAtSample(i, constraint_and_cost_set);       // 对应 l(x_k,u_k)
    path_problem.problem->AddStep(std::move(cost_function_vector), std::move(dynamic_model));
    std::unique_ptr<InequalityConstraint<kNumStates, kNumControls, Eigen::Dynamic>>
        constraint_vector = CollectConstraintsAtSample(i, constraint_and_cost_set);
    path_problem.inequality_constraints.push_back(std::move(constraint_vector));
  }
  ...
}
```

值得注意：`clothoid_path_optimizer` 在实践中**没有使用等式约束**（`equality_constraints` 全部是空指针），所有硬性限制（车道边界、避障、曲率范围）都建模成不等式约束 `g_k(x_k,u_k)≤0`。这在实际工程中很常见——因为等式约束往往可以放宽为"落在一个很窄区间内"的不等式约束，这样数值上更鲁棒（避免因为浮点误差导致等式约束永远无法精确满足）。

---

## 9. ALTRO 的四个"加分项"技巧

论文第三章除了核心的 AL+iLQR 结合外，还提出了四个独立的技巧改进（III.A ~ III.D）。逐一讲解，并说明本仓库用了哪些。

### 9.1 平方根反向传播（Square-Root Backward Pass）——论文 III.A

**问题**：AL 方法要让约束收敛得快，罚参数 `μ` 必须变得很大（比如 `1e8`）。但 `Q_uu` 里包含了 `μ` 相关的二次惩罚项，`μ` 太大会导致 `Q_uu` 的**条件数**（最大特征值/最小特征值）急剧恶化，进而在矩阵求逆（`Q_uu^{-1}`）时产生严重的数值误差（甚至变成负定，导致 Cholesky 分解失败）。

**解决方案**：不直接对 `P_k, Q_xx, Q_uu` 这些矩阵做数值运算，而是维护它们的 **Cholesky 平方根因子**（`S=√P, Z_xx=√Q_xx, Z_uu=√Q_uu`），所有递推关系都改写成对这些"平方根"矩阵的运算（用 QR 分解代替直接矩阵乘法/求逆）。这是数值线性代数里的经典技巧（源自"平方根卡尔曼滤波器"），核心好处是：**矩阵的条件数在平方根空间里是原来的平方根**（比如原矩阵条件数是 `1e16`，平方根矩阵的条件数只有 `1e8`），大幅提升数值稳定性，代价是每步反向传播的计算量略微增加（多做几次 QR 分解）。

论文公式 (25)~(35) 给出了具体的 QR 分解递推公式。**本仓库的求解器（`solver.h`）没有实现这个平方根版本**，而是采用了更简单直接的做法：

```319:344:modules/planning/common/math/solver/solver.h
MatUU Q_uu_reg_inv;
if (U == 1) {
  const double q_uu_reg_inv = 1.0 / Q_uu_reg(0);
  ...
} else {
  Eigen::LLT<MatUU> llt_of_Q_uu_reg(Q_uu_reg);
  if (llt_of_Q_uu_reg.info() == Eigen::NumericalIssue) {
    return false;    // Cholesky 分解失败（矩阵非正定），触发正则化增大重试
  }
  const MatUU Q_uu_reg_chol = llt_of_Q_uu_reg.matrixL().transpose();
  Q_uu_reg_inv = Q_uu_reg_chol.inverse() * Q_uu_reg_chol.transpose().inverse();
}
```

代码里通过 `Eigen::LLT`（标准 Cholesky 分解）判断数值问题，一旦分解失败就走 `IncreaseRegularization` 增大 `mu`（也就是论文里的 `ρ`）重试，用"失败重试+自适应正则化"的工程手段来规避平方根 BP 要解决的数值稳定性问题。这是一种更简单但鲁棒性略逊的替代方案——对于本工程状态维度只有 5、控制维度只有 1 的小规模问题，这个简化通常是够用的；如果未来遇到病态问题频发，平方根 BP 会是一个值得引入的优化方向。

### 9.2 不可行状态轨迹初始化（Infeasible State Trajectory Initialization）——论文 III.B

**动机**：很多时候我们容易猜出一条"看起来合理"的状态轨迹（比如参考线、专家演示轨迹、上一帧的路径），但很难猜出与之精确匹配的控制序列——因为要让状态轨迹严格满足非线性动力学方程 `x_{k+1}=f(x_k,u_k)`，需要精确反解控制量，这本身就很难。

**解决方案**：人为地在动力学里加一个"虚拟控制" `w_k∈R^n`（维度等于状态维度，让系统变得"完全能控/fully actuated"）：

```
x_{k+1} = f(x_k, u_k) + w_k              式(36)
```

给定期望轨迹 `x̃` 和某个粗糙的控制猜测 `U`，可以直接算出让状态轨迹精确匹配的 `w_k`（式37）：

```
w_k = x̃_{k+1} - f(x_k, u_k)
```

然后把 `w_k` 也当作待优化的"控制量"，加入代价 `Σ(1/2)w_k^T R_inf w_k`（式38）并附加约束 `w_k=0`。随着优化的进行（`w_k` 的惩罚权重 `R_inf` 通过 AL 框架不断加大，或作为等式约束被 AL 逐步逼近 0），最终 `w` 趋近于 0，问题退化回原始的严格动力学可行问题。

**直觉理解**：`w_k` 就像是"允许暂时作弊——让轨迹先大致对上期望形状，再逐步把作弊的部分磨平，直到完全符合物理规律"。

**代码对应**：`FeasibleTrajectoryGenerator`，专门用来生成"满足动力学的初始路径"：

```92:153:modules/planning/common/math/solver/feasible_trajectory_generator.h
template <typename DynamicsType, int X, int U>
base::Optional<typename FeasibleTrajectoryGenerator<DynamicsType, X, U>::VectorOfVecU>
FeasibleTrajectoryGenerator<DynamicsType, X, U>::Generate(...) const {
  ...
  for (int i = 0; i < num_steps; ++i) {
    ...
    std::unique_ptr<DynamicModel<X, U + X>> dynamic_model =
        std::make_unique<InfeasibleModel<X, U>>(std::move(original_model));  // 增广动力学: f(x,u)+w
    // 计算 w_k 的初始猜测（式37）
    augmented_reference_u_trajectory[i].template head<U>() = reference_u_trajectory[i];
    augmented_reference_u_trajectory[i].template tail<X>() =
        reference_x_trajectory[i + 1] -
        dynamic_model->Evaluate(reference_x_trajectory[i], augmented_reference_u_trajectory[i]);
    ...
    // 代价函数中对 w_k 加权惩罚（对应式38）
    std::unique_ptr<CostFunction<X, U + X>> quad_cost = std::make_unique<QuadCost<X, U + X>>(
        x_weight_ * steps[i], u_weight_ * steps[i], target_x_trajectory[i]);
    feasible_problem.problem->AddStep(std::move(quad_cost), std::move(dynamic_model));
    // 约束 w_k = 0（对应论文里"附加约束 w_k=0"）
    std::unique_ptr<EqualityConstraint<X, U + X, X>> model_feasibility_constraint =
        std::make_unique<ModelFeasibilityConstraint<X, U>>();
    feasible_problem.equality_constraints.push_back(std::move(model_feasibility_constraint));
  }
  ...
  const Solver solver(*al_solver_config_, *ilqr_solver_config_);
  typename Solver::Solution solution = solver.Solve(feasible_problem);
  ...
}
```

注意这里控制维度从 `U` 增广到 `U+X`（`AugmentedVecU`），正是把 `[u_k; w_k]` 拼在一起当作新的"控制量"，让 `ALSolver` 用同一套 AL+iLQR 机制去求解——这是论文式 (36)~(38) 的直接工程实现，`w_k=0` 的约束通过 `ALProblem` 里的等式约束（`C1=X`）来强制满足，随着 AL 外层迭代乘子和罚参数的更新，`w_k` 会被逐步压到 0。

在 `ClothoidPathOptimizer::GenerateInitialPaths` 里可以看到具体调用（用参考线/上一帧路径作为期望轨迹 `target_x_trajectory`，生成一条动力学可行的初始 clothoid 路径）：

```677:712:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
ClothoidPathProblem::VecX x_weight;
x_weight << kPositionCostWeight, kPositionCostWeight, 0.0, 0.0, kDkappaCostWeight;
ClothoidPathProblem::VecU u_weight;
u_weight << kDDkappaCostWeight;
const FeasibleTrajectoryGenerator<ClothoidPathDynamic, kNumStates, kNumControls>
    feasible_path_generator(&al_solver_config, &ilqr_solver_config, x_weight, u_weight);
ClothoidPath initial_path;
initial_path.x_trajectory.resize(num_samples);
initial_path.x_trajectory[0] = reference_path->x_trajectory[0];
// 只关心位置(x,y)是否匹配目标轨迹，θ/κ/dκ 目标值设为0（不强行约束朝向和曲率）
FeasibleTrajectoryGenerator<ClothoidPathDynamic, kNumStates, kNumControls>::VectorOfVecX
    target_x_trajectory = reference_path->x_trajectory;
for (auto& target_x : target_x_trajectory) {
  target_x[2] = 0.0;
  target_x[3] = 0.0;
  target_x[4] = 0.0;
}
const base::Optional<...> initial_u_trajectory = feasible_path_generator.Generate(
    steps_, reference_path->x_trajectory, reference_path->u_trajectory, target_x_trajectory);
```

这里的权重设计很有意思：`x_weight` 只在位置 `(x,y)` 和 `dκ` 上给了非零权重，`θ, κ` 权重为 0——这是因为参考路径的朝向和曲率往往不是"硬性目标"，只需要位置贴合参考线，让动力学自己算出合理的朝向和曲率即可，这样生成的初始路径更平滑、更符合车辆运动学。这是工程上对论文思想的灵活变通。

### 9.3 最小时间问题（Minimum Time）——论文 III.C

把每一步的时间步长 `τ_k=√dt_k` 也当作待优化的控制量之一，让求解器自己决定"每一步该花多长时间"，从而实现"用最短总时间完成任务"的优化目标（同时约束所有 `τ_k` 相等，避免求解器通过任意扭曲时间离散化来"作弊降低代价"）。

**本仓库未使用此技巧**——`clothoid_path_optimizer` 解决的是纯粹的**空间路径规划**问题（不含时间维度，`steps_` 是预先按照参考速度和固定的空间/时间分辨率算好的采样间隔，见 `GenerateSampleSteps`），不涉及"求解最优时间分配"，所以论文这一节的技巧在当前代码库中没有对应实现。速度规划是路径规划之后的独立环节（Speed Optimizer），不在本文档讨论范围内。

### 9.4 投影牛顿法（Projected Newton Method）——论文 III.D

**动机**：论文指出 AL 方法有个先天缺陷——罚参数 `μ` 不能无限增大（数值上会爆炸），所以 AL 方法只能收敛到一个"粗糙"的容差（比如约束违反量 `1e-3`），如果要达到机器精度级别的约束满足（`1e-8`），AL 方法收敛会变得极慢。

**解决方案**：论文提出用 AL+iLQR 阶段的解作为"热启动"，切换到一个**有效集投影牛顿法**（active-set projected Newton method），对当前激活的约束直接求解 KKT 系统的牛顿步，并把每一步的搜索方向**投影**到约束流形上（论文式 41）：

```
δY_p ← δY_p - H^T(HH^T)^{-1} h
```

其中 `h,H` 分别是约束违反量和约束雅可比。这个投影操作保证了迭代过程中始终严格满足约束（"anytime feasible" 在约束层面的体现），通常只需要 1~2 步牛顿迭代就能把约束违反降到机器精度。

**本仓库未实现此阶段**。原因主要有：

1. **精度需求不同**：论文的机器臂避障场景需要极高精度（因为要在物理机器人上执行，容差稍大就可能碰撞），而车辆路径规划任务通常在几何约束外还留有安全 buffer（车道边界的 `buffer`、避障的安全距离等，见 `restriction_constraint_config().buffer()` 等配置），`1e-4`~`1e-3` 量级的约束违反完全在安全裕度内，不需要追求机器精度。
2. **工程复杂度与实时性**：投影牛顿法需要显式组装稀疏 KKT 矩阵并求解线性系统 `HH^T`，对于车载实时控制场景（要求毫秒级求解），这个额外阶段的收益和成本不成正比。
3. 本仓库的 `ALSolver` 直接以 AL 内层 iLQR 收敛作为终态（`State::kSolved`），是论文中"仅第一阶段"的精简版本，在工程上是合理的取舍。

---

## 10. 论文实验结果速览

论文第四章用几个经典 benchmark 问题比较 ALTRO 和 DIRCOL（用 Ipopt 求解的直接法）：

1. **平行泊车（Reeds-Shepp 车模型）**：ALTRO 在有约束场景下比 DIRCOL 更快；在最小时间问题上两者都能收敛到 bang-bang 控制（最优时间控制的一种表现形式：控制量在上下界之间"开关"式切换），但 DIRCOL 的解在拐点处震荡剧烈，ALTRO 更平滑。
2. **Car Escape（走出迷宫式障碍区）**：标准约束 iLQR（不带 AL）无法找到无碰撞路径，但 ALTRO 可以；配合投影牛顿法（ALTRO*）能把约束违反精度从 `1e-3` 提升到 `1e-8`。
3. **四旋翼穿越迷宫**：ALTRO 能找到无碰撞轨迹，而 DIRCOL 即使用 ALTRO 的解作为初值也未能收敛。
4. **Kuka iiwa 机械臂避障**：验证了 ALTRO 在高维状态/控制空间下的有效性。

核心结论（论文 Table I）：ALTRO 在**含约束、含障碍物**的场景下相比 DIRCOL 有明显的速度和鲁棒性优势，这也解释了为什么 `clothoid_path_optimizer` 这种需要频繁处理动态障碍物、车道边界约束的场景选择基于 iLQR+AL 的方案，而不是通用 NLP 求解器。

---

## 11. 总对照表：论文符号 ↔ 代码符号 ↔ 文件位置

| 论文符号/概念 | 含义 | 代码符号 | 代码位置 |
|---|---|---|---|
| `x_k` | 状态 | `VecX x`, `x_trajectory[i]` | `solver.h` |
| `u_k` | 控制 | `VecU u`, `u_trajectory[i]` | `solver.h` |
| `f(x_k,u_k)` | 离散动力学 | `DynamicModel::Evaluate` | `dynamic_model.h`, `discretized_model.h` |
| `l(x_k,u_k)`, `l_f(x_N)` | 阶段/终端代价 | `CostFunction::EvaluateCost` | `cost_function.h` |
| `g_k, h_k` | 不等式/等式约束 | `InequalityConstraint`, `EqualityConstraint` | `augmented_lagrangian/inequality_constraint.h` 等 |
| `V_k(x)`, `p_k`, `P_k` | cost-to-go 及其一/二阶近似 | `V_x`, `V_xx` | `solver.h::BackwardIteration` |
| `Q_xx,Q_uu,Q_ux,Q_x,Q_u` | 动作价值函数二次展开系数 | 同名变量 | `solver.h::BackwardIteration` |
| `K_k, d_k` | 反馈增益、前馈项 | `meta->K_trajectory[i]`, `meta->k_trajectory[i]` | `solver.h::BackwardIteration` |
| `ρ`（正则化） | 保证 `Q_uu` 可逆 | `meta->mu` | `solver.h` |
| `α`（线搜索步长） | 前向传播步长 | `meta.alpha`, `alpha_values_` | `solver.h::Solve` |
| `ΔV_k` | 预期代价改进量 | `meta.dV[0], meta.dV[1]` | `solver.h` |
| Algorithm 1 iLQR | 内层迭代 | `Solver<X,U,Problem>::Solve` | `solver.h` |
| Algorithm 2 Backward Pass | 反向传播 | `Solver::BackwardIteration` | `solver.h` |
| Algorithm 3 Forward Pass | 前向传播+线搜索 | `Solver::ForwardIteration` + `Solve` 内线搜索循环 | `solver.h` |
| `L_A(x,λ,μ)` | 增广拉格朗日函数 | `ALCostFunction::EvaluateAugmentedLagrangian` | `al_cost_function.h` |
| `λ`（拉格朗日乘子） | 对偶变量 | `ConstraintPenalty::lambda_` | `constraint_penalty.h` |
| `μ`（罚参数） | 惩罚权重 | `ConstraintPenalty::mu_` | `constraint_penalty.h` |
| `φ`（罚参数放大因子） | 罚参数调度系数 | `config.penalty_scaling_factor` | `al_solver.h` |
| 式(23) 乘子更新 | 对偶上升 | `ALSolver::UpdateDuals` → `ConstraintPenalty::UpdateDual` | `al_solver.h`, `constraint_penalty.h` |
| 式(24) 罚参数更新 | 罚参数调度 | `ALSolver::UpdatePenalties` → `ConstraintPenalty::UpdatePenalty` | `al_solver.h`, `constraint_penalty.h` |
| Algorithm 4 AL 外层循环 | 双层优化主循环 | `ALSolver<X,U,C1,C2>::Solve` | `al_solver.h` |
| III.A 平方根反向传播 | 数值稳定性优化 | 未实现（用 `Eigen::LLT` + 失败重试代替） | `solver.h::BackwardIteration` |
| III.B 不可行轨迹初始化 | 状态轨迹热启动 | `FeasibleTrajectoryGenerator`, `InfeasibleModel` | `feasible_trajectory_generator.h` |
| III.C 最小时间问题 | 时间也作为优化变量 | 未使用（本工程无时间维度优化需求） | — |
| III.D 投影牛顿法 | 高精度约束满足的第二阶段 | 未实现（AL 精度已足够） | — |
| Algorithm 6 ALTRO 主流程 | 整体两阶段算法 | `ClothoidPathOptimizer::ComputePathCurve` 中调用 `Solver::Solve`（仅第一阶段） | `clothoid_path_optimizer.cc` |

---

## 12. `clothoid_path_optimizer` 全貌：从论文算法到无人车路径规划

理解了 ALTRO 论文后，再完整梳理一遍 `ClothoidPathOptimizer::ComputePathCurve` 这个函数的完整流程，看它是如何把抽象算法应用到具体的自动驾驶路径规划场景的。

### 12.1 为什么用"回旋曲线"（Clothoid）作为路径表示？

回旋曲线（Clothoid，又叫 Euler 螺旋）的定义特点是：**曲率沿弧长线性变化**。这正是真实车辆在转弯时应该走的路径形状——因为方向盘转动需要时间，不可能瞬间从直行切换到某个固定曲率的圆弧（那样会导致车辆在切换瞬间产生极大的横向加加速度冲击，乘坐体验差且不安全）。

在状态里额外引入 `κ`（曲率）和 `dκ`（曲率变化率）作为独立的状态量（而不仅仅是 `x,y,θ` 三个基本量），本质上就是把"曲率变化率的变化率"`ddκ`（控制量）作为最高阶的自由变量，让积分出来的路径曲率天然连续、光滑变化——这正是 Clothoid 路径的核心特性，也是为什么这个状态空间设计（`kX,kY,kTheta,kKappa,kDKappa`）会被称为"clothoid path"的原因。

### 12.2 整体流程（对照代码逐段解释）

```
ComputePathCurve() 的执行步骤：
┌─────────────────────────────────────────────────────────────┐
│ 1. 生成时间/空间采样步长 steps_（构造函数阶段，GenerateSampleSteps） │
│    -> 对应论文里离散化的时间步长 Δt（这里是空间步长 Δs）           │
├─────────────────────────────────────────────────────────────┤
│ 2. 生成约束/代价采样点，构造 RestrictionCostParams 等参数         │
│    -> 车道边界、避障框、guide line 等，对应论文的 g_k(x,u)         │
├─────────────────────────────────────────────────────────────┤
│ 3. GenerateInitialPaths()：生成动力学可行的初始路径               │
│    -> 对应论文 III.B "不可行状态轨迹初始化" + FeasibleTrajectory   │
│       Generator（用 AL+iLQR 求一遍简化子问题，得到可行的 u 序列）  │
├─────────────────────────────────────────────────────────────┤
│ 4.（可选）GenerateFeasiblePath()：两阶段求解的第一小阶段            │
│    -> 用简化的约束/代价先求一版粗糙可行解，作为下一步的更好初值      │
├─────────────────────────────────────────────────────────────┤
│ 5. GenerateConstraintsAndCosts()：构造完整的约束和代价集合          │
│    -> 运动学约束(曲率/转向速率限制)、车道边界、障碍物、guide line、  │
│       lane boundary、repulsion 斥力代价等，对应论文的 l(x,u), g(x,u)│
├─────────────────────────────────────────────────────────────┤
│ 6. ConstructPathProblem()：组装成 PathProblem (ConstrainedProblem)  │
│    -> 对应论文公式(1)完整问题定义                                 │
├─────────────────────────────────────────────────────────────┤
│ 7. Solver::Solve()：调用 ALSolver -> 内部循环调用 iLQR             │
│    -> 对应论文 Algorithm 6 的第一阶段（AL_iLQR）                  │
├─────────────────────────────────────────────────────────────┤
│ 8. ConvertToCurve()：把优化出的状态轨迹 x_trajectory 转成路径曲线   │
│    Curve2d，再投影到 Frenet 坐标系，供下游速度规划/控制模块使用     │
└─────────────────────────────────────────────────────────────┘
```

### 12.3 关键代码段精读

**步骤1：采样步长生成**（`GenerateSampleSteps`）——为什么要"密集采样+稀疏采样"两段式？

```288:317:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
std::vector<double> ClothoidPathOptimizer::GenerateSampleSteps(...) const {
  ...
  // dense sample info
  constexpr double kMinDenseSamplingStep = 0.5;
  const double dense_sampling_step =
      std::max(config.time_resolution_dense_sampling() * reference_speed, kMinDenseSamplingStep);
  const int num_dense_samples = ...;
  // sparse sample info
  const double sparse_sample_length = max_sampling_length - dense_sampling_step * num_dense_samples;
  double sparse_sampling_step = ...;
  ...
}
```

近处密集采样（更高的时间/空间分辨率）能更精细地控制车辆近期行为（对避障、跟车等精度要求高），远处稀疏采样能在不显著增加优化变量数目（进而不显著增加计算量，因为 iLQR 的单次迭代复杂度是 `O(N)`，`N` 是采样点数）的前提下把规划视野拉得更远。这是纯粹的工程效率考量，论文本身没有涉及"变步长采样"的技巧（论文用的是等间隔离散化），属于本仓库结合实际场景做的优化。

**步骤3-4：两阶段求解**（对应 `config_->enable_two_stage_solve()`）：

```191:201:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
// Solving feasible path.
if (config_->enable_two_stage_solve()) {
  const Solver::Solution feasible_solution = GenerateFeasiblePath(guide_line,
                                                                  restriction_cost_params,
                                                                  box_collision_params,
                                                                  decision_input,
                                                                  reference_speed_profile,
                                                                  *initial_path_set);
  initial_path_set->at(0).x_trajectory = std::move(feasible_solution.ilqr_solution.x_trajectory);
  initial_path_set->at(0).u_trajectory = std::move(feasible_solution.ilqr_solution.u_trajectory);
}
```

这是工程上对论文思想的又一次灵活运用：先用一版"简化的约束和代价"（`GenerateFeasiblePathConstraintsAndCosts`，只保留最基本的运动学约束和guide line代价，避障/车道边界等复杂代价被清空）跑一次完整的 AL+iLQR，得到一个更接近可行域、更平滑的中间解，再把它作为**第二次**（完整约束/代价）求解的初始猜测。这可以理解成是对论文"用粗糙解热启动精细阶段"思想（Algorithm 6 中第一阶段热启动第二阶段）的变体应用——只不过这里两个阶段都是"AL+iLQR"，只是约束/代价的复杂度不同，而不是论文里"AL+iLQR"热启动"投影牛顿"。

**步骤7：求解与结果提取**

```226:247:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
const PathProblem path_problem =
    ConstructPathProblem(*initial_path_set, &constraint_and_cost_set);
const absl::Time timestamp_3 = absl::Now();
const Solver solver(al_solver_config_, ilqr_solver_config_);
const Solver::Solution solution = solver.Solve(path_problem);
...
if (solution.state != Solver::State::kSolved) {
  ReportSolveFailedEvent("Failed to solve: " + std::to_string(static_cast<int>(solution.state)));
  return base::none;
}
return ConvertToCurve(solution.ilqr_solution.x_trajectory);
```

如果求解失败（比如超时、达到最大罚参数、迭代不收敛），会走 `GetFallbackSolution` 逻辑（用初始路径或上一帧路径兜底），这是自动驾驶系统对"优化求解器失败"这一异常情况必须具备的鲁棒性设计——毕竟车辆不能因为一次优化不收敛就完全停止规划。

**代价函数与约束的模块化组装**：`ConstraintAndCostSet` 把各类代价/约束（运动学、guide line、车道边界、避障、斥力等）按"生成器模式"（`XxxConstraintAndCostGenerator`）解耦，再按采样点索引重新汇总成每一步的 `CostFunctionVector`/`InequalityConstraintVector`（`CollectCostsAtSample`/`CollectConstraintsAtSample`）。这跟论文里"`l(x,u)` 可以是任意标量函数、`g(x,u)` 可以是任意向量值函数"的抽象设定完全兼容——论文只关心"存在一个代价函数和约束函数"，而工程实现则把它们拆成一堆更小的、语义清晰的子模块（对应不同的驾驶场景考量），再线性叠加/拼接成完整的代价/约束向量喂给求解器。

---

## 13. 一个完整的手工算例：单步 iLQR 反向传播

为了彻底吃透第 6 节的公式，我们用一个极简的一维例子手算一遍。

**设定**：状态 `x∈R`，控制 `u∈R`，一步的线性动力学 `x_{k+1} = x_k + u_k`（`A=1, B=1`），代价 `l(x,u) = x² + u²`（`l_x=2x, l_xx=2, l_u=2u, l_uu=2, l_ux=0`），只有一步（`N=1`），终端代价 `l_f(x)=x²`（`p_N=2x_N, P_N=2`）。

**第0步（终端）**：假设 `x_1=1`（这是当前迭代的名义轨迹），则：

```
p_1 = 2·x_1 = 2
P_1 = 2
```

**反向递推到 k=0**：假设当前名义 `x_0=0.5, u_0=0.5`（满足 `x_1=x_0+u_0=1`）。

```
Q_x  = l_x + A·p_1 = 2·0.5 + 1·2 = 3
Q_u  = l_u + B·p_1 = 2·0.5 + 1·2 = 3
Q_xx = l_xx + A²·P_1 = 2 + 1·2 = 4
Q_uu = l_uu + B²·P_1 = 2 + 1·2 = 4
Q_ux = l_ux + B·P_1·A = 0 + 1·2·1 = 2
```

加正则化 `ρ=0`（假设数值良好不需要正则化）：

```
d_0 = -(Q_uu)^{-1}·Q_u = -3/4 = -0.75
K_0 = -(Q_uu)^{-1}·Q_ux = -2/4 = -0.5
```

**含义解读**：`d_0=-0.75` 说明当前的 `u_0=0.5` 偏大了，应该减小 0.75（前馈修正）；`K_0=-0.5` 说明如果状态出现正向偏差 `δx_0`，应该反向调整控制（负反馈，这跟直觉一致——比如 `x` 偏大就应该减小 `u` 让下一步 `x` 不要更大）。

**前向传播**（`α=1`，假设新旧轨迹此时状态一致 `δx_0=0`）：

```
ū_0 = u_0 + α·d_0 + K_0·0 = 0.5 - 0.75 = -0.25
x̄_1 = x̄_0 + ū_0 = 0.5 - 0.25 = 0.25
```

**验证是否真的更优**：原代价 `J_old = l(0.5,0.5)+l_f(1) = (0.25+0.25)+1 = 1.5`；新代价 `J_new = l(0.5,-0.25)+l_f(0.25) = (0.25+0.0625)+0.0625 = 0.375`。代价从 `1.5` 降到 `0.375`，确实下降了，说明这一步 iLQR 迭代是有效的。如果这个问题多迭代几轮，会继续收敛到解析最优解 `x_0=0.5`（固定，是初始条件的一部分，这里简化为固定值不作为变量）附近的真正最优 `u_0`。

这就是 iLQR 单次迭代"反向传播算增益 → 前向传播更新轨迹 → 代价下降"的完整闭环，跟第 6 节代码逻辑完全一致，只是这里维度是 1、只有一步，方便手算验证。

---

## 14. 常见疑问 FAQ

**Q1：iLQR 和 DDP 有什么区别？**
DDP（Differential Dynamic Programming）在展开 `Q_k` 时，会保留动力学函数 `f` 的二阶导数项（`f_xx` 等），而 iLQR 为了简化计算，只用动力学的一阶导数（雅可比 `A_k,B_k`），把动力学当作局部线性来处理，只有代价函数保留二阶信息。iLQR 可以看作是 DDP 的一个近似简化版本，计算更快，在大多数工程场景下精度损失可以接受。本仓库的 `solver.h` 用的正是 iLQR 简化（`DynamicModel::EvaluateJacobians` 只返回一阶雅可比 `dfdx, dfdu`，没有二阶导数接口）。

**Q2：为什么代码里既有 `mu`（iLQR 正则化）又有 AL 里的 `mu`（罚参数），这是同一个东西吗？**
不是。这是历史上两篇不同论文用了相同符号的巧合（iLQR/DDP 领域的正则化系数、AL 方法的罚参数，恰好都习惯用 `μ` 表示）。代码里通过命名空间区分：`Solver::IterationMeta::mu` 是 iLQR 反向传播里保证 `Q_uu` 可逆的正则化系数（对应论文 `ρ`，定义在 `solver.h` 里），而 `ConstraintPenalty::mu_` 是增广拉格朗日框架里的约束惩罚权重（对应论文式 21 里的 `μ`，定义在 `constraint_penalty.h` 里）。两者作用的层级不同：前者服务于"内层 iLQR 单次反向传播的数值稳定性"，后者服务于"外层 AL 循环里约束满足程度的调度"。阅读代码时看它属于 `Solver<...>` 还是 `ALCostFunction/ConstraintPenalty` 就能分辨。

**Q3：为什么状态里要放 `κ`（曲率）和 `dκ`（曲率变化率），而不是直接用 `x,y,θ` 三维状态加"转向角"作为控制？**
如果只用 `x,y,θ` 作为状态、"转向角"（等价于瞬时曲率）作为控制，那么优化出来的曲率序列可能是不连续跳变的（因为控制量在 iLQR 里逐步更新时没有显式的平滑性约束），对应到真实车辆就是要求方向盘瞬间打到某个角度，物理上不可实现。把 `κ, dκ` 也纳入状态、把 `ddκ` 作为控制，相当于对"曲率"这个量又做了两次积分（`ddκ→dκ→κ`），天然保证了曲率的连续性和曲率变化率的连续性（因为状态是积分出来的，积分天然连续），这正是"回旋曲线路径"名称的由来，也是第 12.1 节详细解释的内容。

**Q4：`ALSolver` 每次外层迭代都要重新跑一次完整的 iLQR，会不会很慢？**
论文和代码都对此做了优化：其一，AL 外层迭代次数通常不多（`max_outer_iterations` 默认 30 次，实际收敛往往个位数次就够了）；其二，`ALSolver` 支持"热启动"（`enable_warm_start_`，见 `al_solver.h` 第 199~203 行），把上一轮外层迭代收敛的控制序列作为下一轮 iLQR 的初始猜测，避免每次都从零开始迭代，大幅减少内层 iLQR 需要的迭代次数；其三，因为约束是以"软惩罚项"的形式叠加进代价函数里的，iLQR 内部的反向/前向传播计算复杂度不会因为约束数量增多而显著增加（只是 `EvaluateDerivatives` 里要多算几个约束的梯度/海森，属于常数级的额外开销）。

**Q5：`InequalityConstraintVector`/`CostFunctionVector` 这种"Vector"包装是做什么用的？**
因为每个采样点上可能同时存在多种约束（运动学、车道边界、避障框、目标障碍物……）和多种代价（引导线、车道边界代价、斥力代价……），`InequalityConstraintVector`/`CostFunctionVector` 就是把"同一个采样点上的所有约束/代价函数"打包成一个复合对象，对外表现为单个 `InequalityConstraint`/`CostFunction` 接口（内部循环调用所有子约束/子代价求和），这样 `ALProblem`/`Solver` 的核心算法代码不需要关心"某一步到底叠加了多少种代价"，只需要按统一接口调用 `EvaluateCost`/`EvaluateDerivatives` 等方法即可。这是一种典型的组合模式（Composite Pattern），让论文里抽象的"单一代价函数 `l(x,u)`"在工程上可以灵活扩展为任意多个子代价的叠加，而不需要修改核心求解器代码。

**Q6：读完这篇文档，我应该重点记住哪几件事？**
1. 轨迹优化的本质是"在满足动力学和约束的前提下，寻找一串代价最小的控制序列"；
2. iLQR 通过"局部线性化动力学 + 局部二次化代价 + 动态规划反向传播 + 前向传播线搜索"的迭代循环求解无约束问题，核心产物是每步的前馈量 `d_k` 和反馈增益矩阵 `K_k`；
3. 增广拉格朗日法通过"乘子 `λ` + 罚参数 `μ`"的组合，把约束优雅地转化为可以用无约束求解器（iLQR）处理的软惩罚，并通过外层迭代逐步逼近约束的精确满足；
4. ALTRO = AL 框架 + iLQR 内核 + 若干工程增强（平方根 BP、不可行轨迹初始化、最小时间、投影牛顿精修），本仓库完整实现了前两者核心思想，并结合自动驾驶路径规划场景做了大量工程化的定制（多阶段求解、动态维度约束、多候选初值竞争等）；
5. `clothoid_path_optimizer` 是这套理论在无人车路径规划领域的一个具体而完整的落地案例——状态 `[x,y,θ,κ,dκ]`、控制 `[ddκ]`、约束覆盖车道边界/避障/运动学限制、代价覆盖引导线/平滑性/舒适性，通过 `ALSolver::Solve` 一行代码驱动整个 AL+iLQR 求解流程。