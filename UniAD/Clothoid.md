# Clothoid Path Optimizer 原理与实现详解

> 本文档面向希望**从零理解**这个模块的工程师/学习者，目标是把"为什么这么设计"和"代码具体怎么写"对应起来讲清楚。
> 理论部分主要参考学城设计文档《[DesignDoc】20251011 - Clothoid Path Optimizer》(collabpage/2777203995)，并与本目录下的实际代码逐一核对。

---

## 目录

1. [背景：为什么需要一个新的 Path Optimizer](#1-背景为什么需要一个新的-path-optimizer)
2. [整体架构与文件地图](#2-整体架构与文件地图)
3. [Clothoid 运动学模型](#3-clothoid-运动学模型)
4. [模型离散化：从连续微分方程到 ILQR 能用的离散系统](#4-模型离散化从连续微分方程到-ilqr-能用的离散系统)
5. [ILQR + 增广拉格朗日（AL）求解框架](#5-ilqr--增广拉格朗日al求解框架)
6. [Constraint 与 Cost 的完整清单与公式推导](#6-constraint-与-cost-的完整清单与公式推导)
7. [几何工具库：让"折线距离"处处可导](#7-几何工具库让折线距离处处可导)
8. [utils/：把决策层信息组装成约束与代价](#8-utils把决策层信息组装成约束与代价)
9. [完整求解主流程走读（ClothoidPathOptimizer::ComputePathCurve）](#9-完整求解主流程走读clothoidpathoptimizercomputepathcurve)
10. [一个简化的端到端数值例子](#10-一个简化的端到端数值例子)
11. [关键设计取舍 FAQ](#11-关键设计取舍-faq)

---

## 1. 背景：为什么需要一个新的 Path Optimizer

在这个模块之前，Planning 里跑的是 Frenet Frame（基于参考线 s-l 坐标系）下的 path optimizer。它在大多数场景下工作良好，但在几类场景下天然存在困难：

| 场景 | 具体问题 |
|---|---|
| 大曲率 U-turn | s-l 坐标系下很难严格保证 path 曲率满足车辆最小转弯半径；曲率作为 cost 而非约束，容易超限；path 帧间摆动明显 |
| 狭窄空间掉头/会车 | 只把车辆看作一个点/圆来避障，没有精确的"四角碰撞"描述，容易贴障碍物太近 |
| 大曲率参考线（弯心附近） | s-l 坐标系下 curvature、dkappa 的近似计算在参考线曲率很大时会病态，导致优化结果异常（比如 heading 偏差过大） |
| 交换区（掉头区域） | 轨迹不符合人类驾驶习惯，横向舒适性不足 |

根本原因是：**Frenet frame 下的很多变量（尤其曲率相关量）本质上是对参考线曲率的近似**，参考线曲率越大，这个近似误差就越大。

Clothoid Path Optimizer 的思路是**完全抛弃 Frenet 坐标系**，直接在 **Cartesian（世界）坐标系**下用一个包含曲率变化率的运动学模型描述 path，所有车辆能力约束（转弯半径、转向速率）和舒适性代价（横向加速度、jerk）都基于这个模型精确表达，不再依赖参考线曲率做近似。

因为工作量和整体稳定性的考虑，第一期只用于 **U-turn 场景**（曲率超限、弯心病态、四角避障问题集中爆发的场景），验证稳定后再推广到其他场景。

---

## 2. 整体架构与文件地图

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

一句话概括整体分层：**几何层（geometry/math）** 提供可导的距离函数 → **约束/代价层（constraints/cost_functions）** 用这些距离函数加上运动学公式定义具体的约束和代价 → **生成器层（utils）** 把决策层给的道路元素批量转换成一组约束和代价对象 → **主类（clothoid_path_optimizer.cc）** 把所有约束代价按采样点打包成一个 `ConstrainedProblem`，交给 **求解器（common/math/solver）** 用 AL+ILQR 求解。

---

## 3. Clothoid 运动学模型

### 3.1 为什么叫 "Clothoid"

Clothoid（回旋曲线，也叫 Euler 螺线）是一条曲率随弧长线性变化的曲线，常用于道路/铁路的缓和曲线设计，因为曲率连续变化能让车辆转向盘转动更平顺。这里的模型比标准 clothoid 更进一步：不仅曲率 $\kappa$ 对弧长的一阶导 $\dot\kappa$ 可控，还引入了二阶导 $\ddot\kappa$ 作为控制量，本质上是一个更高阶的、可以任意分段调整曲率变化率的曲线族。

### 3.2 状态方程

设 path 上一点的状态为 $(x, y, \theta, \kappa, \omega)$，其中：

- $x, y$：Cartesian 坐标
- $\theta$：航向角（heading）
- $\kappa$：曲率（曲率半径的倒数）
- $\omega = \dot\kappa$：曲率对弧长 $s$ 的一阶导数

控制量为 $\alpha = \dot\omega = \ddot\kappa$：曲率对弧长的二阶导数。

模型是关于弧长 $s$ 的微分方程组：

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

简记为 $\dot{\bm x} = \bm f(\bm x, \bm u)$，其中 $\bm x = (x,y,\theta,\kappa,\omega)^T \in \mathbb{R}^5$，$\bm u = (\alpha) \in \mathbb{R}^1$。

**为什么要引入 $\omega$ 和 $\alpha$：**
- 只有 $(x,y,\theta,\kappa)$ 4 维状态的话，$\kappa$ 是被优化器"瞬间"改变的（一个采样点到下一个采样点曲率可以跳变），这对应无穷大的转向速率，物理上不可实现，而且不平顺。
- 引入 $\omega=\dot\kappa$ 后，$\omega$ 直接对应前轮转向速率（方向盘转动的快慢），限制 $\omega$ 就能限制横向 jerk 和转向速率。
- 进一步引入 $\alpha=\ddot\kappa$ 作为控制量（而不是把 $\omega$ 直接作为控制量），是为了限制**方向盘转速的变化率**，避免方向盘小幅度高频抖动——如果 $\omega$ 直接被当作可以任意跳变的控制量，那么它自己也可能出现不平顺的跳变。

### 3.3 代码对照

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

状态量 5 维、控制量 1 维，与公式完全对应（$\omega$ 对应 `kDKappa`，$\alpha$ 对应控制量 `kDdKappa`）。

动力学方程 $\bm f$ 的实现：

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

逐行对照公式，完全一致。

### 3.4 Jacobian（用于线性化）

ILQR 求解器需要动力学方程关于状态和控制的 Jacobian $\mathbf{J}_{\bm f} \in \mathbb{R}^{5\times 6}$（5 行状态导数、前 5 列对 $\bm x$ 求导、第 6 列对 $\bm u$ 求导）：

$$
\mathbf{J}_{\bm f}(\bm x_k, \bm u_k) =
\begin{bmatrix}
0 & 0 & -\sin\theta_k & 0 & 0 & 0 \\
0 & 0 & \cos\theta_k & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

代码实现：

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

这个 Jacobian 非常稀疏（只有 5 个非零元素），这是因为模型本身除了 $\dot x, \dot y$ 依赖 $\theta$ 是非线性（三角函数）之外，其余都是线性关系（直接等于下一个状态量或控制量）。

---

## 4. 模型离散化：从连续微分方程到 ILQR 能用的离散系统

ILQR 处理的是离散时间系统 $\bm x_{k+1} = \bm f_d(\bm x_k, \bm u_k)$，而上面写的是连续弧长域的微分方程，所以需要数值积分把连续模型离散化，同时还要拿到离散后方程的 Jacobian（因为是非线性方程，需要做线性化）。

### 4.1 欧拉法（最简单的离散化）

$$
\bm x_{k+1} = \bm x_k + h\, \bm f(\bm x_k, \bm u_k)
$$

其中 $h$ 是步长（这里等价于弧长增量 $\Delta s$）。线性化后得到状态矩阵 $\bm A$、输入矩阵 $\bm B$：

$$
[\bm A\ \ \bm B] = \begin{bmatrix} \bm I_{5\times5} & \bm 0_{5\times1}\end{bmatrix} + h\,\mathbf{J}_{\bm f}(\bm x_k,\bm u_k)
$$

### 4.2 四阶龙格库塔法（RK4，实际使用的方法）

欧拉法精度较低，代码里实际用的是 RK4：

$$
\begin{aligned}
\bm k_1 &= \bm f(\bm x_k, \bm u_k) \\
\bm k_2 &= \bm f(\bm x_k + \tfrac{h}{2}\bm k_1, \bm u_k) \\
\bm k_3 &= \bm f(\bm x_k + \tfrac{h}{2}\bm k_2, \bm u_k) \\
\bm k_4 &= \bm f(\bm x_k + h\bm k_3, \bm u_k) \\
\bm x_{k+1} &= \bm x_k + \tfrac{h}{6}(\bm k_1 + 2\bm k_2 + 2\bm k_3 + \bm k_4)
\end{aligned}
$$

因为 ILQR 需要离散方程的 Jacobian，而 RK4 是 4 个点的加权组合，所以需要用链式法则把每个 $\bm k_i$ 对 $\bm x_k, \bm u_k$ 的 Jacobian 递推出来（详见设计文档中的完整公式），最终合成：

$$
[\bm A\ \ \bm B] = \begin{bmatrix}\bm I_{5\times5} & \bm 0_{5\times1}\end{bmatrix} + \frac{h}{6}\left[\mathbf J_{\bm k_1} + 2\mathbf J_{\bm k_2} + 2\mathbf J_{\bm k_3} + \mathbf J_{\bm k_4}\right]
$$

### 4.3 代码组织：ContinuousDynamics → Integrator → DiscretizedModel

这一层是通用求解器框架的一部分，不属于 Clothoid 专属代码，好处是**任何符合 `ContinuousDynamics` 接口的运动学模型都可以复用同一套离散化和 RK4 积分逻辑**。

- `ContinuousDynamics<X,U>`：定义 `Evaluate`（算 $\bm f$）和 `EvaluateJacobians`（算 $\mathbf J_{\bm f}$）两个接口，`ClothoidPathDynamic` 就是它的一个具体实现。
- `Integrator<X,U>`：定义如何把一个 `ContinuousDynamics` 在给定步长下积分一步，`RungeKutta4` 是其中一种实现，内部按上面公式计算 $\bm k_1..\bm k_4$ 并组合 Jacobian。
- `DiscretizedModel<X,U>`：组合一个 `ContinuousDynamics` + 一个 `Integrator` + 步长 $h$，对外暴露统一的 `DynamicModel` 接口（`Evaluate`、`EvaluateJacobians`），ILQR 求解器只依赖这个统一接口，不关心底层用的是欧拉法还是 RK4。

代码里生成每个采样点的离散模型：

```513:524:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
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

注意每个采样点的步长 `steps_[index]` 是可以不同的（见下一节的采样策略），这也是为什么每个采样点都要单独构造一个 `DynamicModel`。

### 4.4 采样步长的选择

设计文档提到：步长可以根据车辆当前速度 $v$ 选择，比如用 $v\times10\text{s}$ 作为 path 总长度，步长选 $v\times0.1\text{s}$，这样能保证 path 足够长、采样点足够密，同时计算量不会太大。

代码里的实现更精细，分成"密采样段"和"稀疏采样段"两部分（`GenerateSampleSteps`）：

```239:268:modules/planning/path/clothoid_path_optimizer/clothoid_path_optimizer.cc
std::vector<double> ClothoidPathOptimizer::GenerateSampleSteps(
    const ClothoidPathOptimizerConfig& config,
    const math::Vector2d& ego_position,
    double reference_speed) const {
  const double max_sampling_length =
      std::min(config.path_length(), reference_line_curve_->curve().max_s());
  // dense sample info
  constexpr double kMinDenseSamplingStep = 0.5;
  const double dense_sampling_step =
      std::max(config.time_resolution_dense_sampling() * reference_speed, kMinDenseSamplingStep);
  const int num_dense_samples =
      std::min(static_cast<int>(std::ceil(config.duration_dense_sampling() /
                                          config.time_resolution_dense_sampling())),
               static_cast<int>(std::floor(max_sampling_length / dense_sampling_step)));
  // sparse sample info
  const double sparse_sample_length = max_sampling_length - dense_sampling_step * num_dense_samples;
  double sparse_sampling_step =
      std::max(dense_sampling_step,
               sparse_sample_length / static_cast<double>(config.number_sparse_sampling_steps()));
  ...
}
```

思路：近处（比如未来几秒）用较密的步长（正比于速度，速度越快步长越大，但设了 `kMinDenseSamplingStep=0.5m` 下限保证低速时也有足够分辨率），远处用更稀疏的步长补足总长度——因为近处的路径精度对当前决策更重要，远处只需要保证大致形状合理。

---

## 5. ILQR + 增广拉格朗日（AL）求解框架

### 5.1 为什么是 ILQR + AL 这个组合

- 这是一个**带不等式约束的非线性最优控制问题**：目标是找一条 $(\bm x_{0:N}, \bm u_{0:N-1})$ 轨迹，最小化各种代价的加和，同时满足动力学约束和一堆不等式约束（曲率、转速、不能碰撞等）。
- **ILQR**（Iterative LQR）擅长求解无约束的、轨迹优化型的非线性最优控制问题，通过反复线性化动力学、二次近似代价，用动态规划（Riccati 递推）算出解析最优增益，收敛快。但它本身不直接支持不等式约束。
- **增广拉格朗日法（AL）** 是一种把约束优化转化为一系列无约束优化的通用方法：给每个约束配一个拉格朗日乘子和一个惩罚系数，构造增广目标函数，反复求解无约束子问题、更新乘子和惩罚系数，直至约束违反量收敛到容差以内。

两者结合：**外层 AL 循环负责"满足约束"，内层 ILQR 循环负责"在当前惩罚下求出最优轨迹"**。

### 5.2 增广拉格朗日函数

对于不等式约束 $c(\bm x,\bm u) \le 0$，增广拉格朗日惩罚项为：

$$
L_A = \lambda \cdot c + \frac{1}{2}\mu \cdot c^2 \quad (\text{仅当 } c \ge 0 \text{ 即违反约束，或 } \lambda > 0 \text{ 时计入})
$$

其中 $\lambda$ 是拉格朗日乘子（对偶变量），$\mu$ 是惩罚系数。这一逻辑对应代码里 `InequalityConstraint::EvaluateAugmentedLagrangian`（只对违反约束或乘子非零的分量计入惩罚，这是不等式约束 AL 方法的标准处理，保证在可行域内部时惩罚为 0，梯度连续）。

### 5.3 AL 外层循环

代码位于 `ALSolver::Solve`：

```140:184:modules/planning/common/math/solver/augmented_lagrangian/al_solver.h
template <int X, int U, int C1, int C2>
typename ALSolver<X, U, C1, C2>::Solution ALSolver<X, U, C1, C2>::Solve(
    const ConstrainedProblem<X, U, C1, C2>& constrained_problem) const {
  ALProblem problem(constrained_problem);
  ResetDuals(&problem);
  ResetPenalties(&problem);
  Status status;
  ...
  for (int iter = 0; iter < config_.max_outer_iterations; ++iter) {
    const IlqrSolution inner_solution = ilqr_solver_.Solve(problem);
    UpdateStatus(problem, inner_solution, solve_start_time, &status);
    const State state = CheckTerminationCondition(status);
    ...
    if (state != State::kUnsolved) {
      return Solution{...};
    }
    UpdateDuals(inner_solution, &problem);
    UpdatePenalties(&problem);
  }
  LOG(FATAL) << "Should not reach here.";
}
```

每一轮外层迭代：
1. 用当前的 $\lambda,\mu$ 调用内层 ILQR 求解器求出一条完整轨迹（`ilqr_solver_.Solve(problem)`）；
2. 检查这条轨迹的最大约束违反量 `max_violation`，若小于 `constraint_tolerance`（如 `1e-4`）则判定为 `kSolved`，成功退出；
3. 否则更新每个约束的拉格朗日乘子 $\lambda$（`UpdateDuals`，本质是投影梯度上升）和惩罚系数 $\mu$（`UpdatePenalties`，通常乘以一个 `penalty_scaling_factor`，如 10 倍，见 `Config::penalty_scaling_factor`），继续下一轮外层迭代；
4. 还有几种失败退出条件：`kFailedInnerIlqrNotConverge`（内层没收敛）、`kFailedMaxOuterIterations`、`kFailedMaxTotalIterations`、`kFailedMaxPenalty`（惩罚系数超过 `max_penalty` 还没满足约束，说明问题可能不可行）、`kFailedTimeout`。

### 5.4 ILQR 内层循环

代码位于 `Solver::Solve`（`modules/planning/common/math/solver/solver.h`），核心是 **Backward Pass（反向传递）** 和 **Forward Pass + Line Search（前向传递+线搜索）** 交替进行。

#### Backward Pass：Riccati 递推

从终端时刻往回递推，维护"cost-to-go"的二次近似 $V(\bm x) \approx \tfrac12 \bm x^T V_{xx}\bm x + V_x^T\bm x$。在每一步 $i$，先算伪哈密顿量 $Q$ 函数（把当前步代价 $l$ 和下一步 cost-to-go 通过动力学 Jacobian 关联起来）：

$$
\begin{aligned}
Q_x &= l_x + f_x^T V_x, \quad Q_u = l_u + f_u^T V_x \\
Q_{xx} &= l_{xx} + f_x^T V_{xx} f_x, \quad Q_{uu} = l_{uu} + f_u^T V_{xx} f_u, \quad Q_{ux} = l_{ux} + f_u^T V_{xx} f_x
\end{aligned}
$$

对应代码：

```290:294:modules/planning/common/math/solver/solver.h
VecX Q_x = l_x + f_x.transpose() * V_x;
VecU Q_u = l_u + f_u.transpose() * V_x;
MatXX Q_xx = l_xx + f_x.transpose() * V_xx * f_x;
MatUU Q_uu = l_uu + f_u.transpose() * V_xx * f_u;
MatUX Q_ux = l_ux + f_u.transpose() * V_xx * f_x;
```

然后对 $Q_{uu}$ 加正则项 `mu * I`（Levenberg-Marquardt 式正则化，保证矩阵正定可逆，处理非凸性/数值问题），求解使 $Q$ 最小的控制修正量：

$$
\delta\bm u^* = -Q_{uu,reg}^{-1}Q_u - Q_{uu,reg}^{-1}Q_{ux}\,\delta\bm x = \bm k + \bm K \delta\bm x
$$

对应代码（无控制量约束的简单情形）：

```326:327:modules/planning/common/math/solver/solver.h
meta->k_trajectory[i] = -Q_uu_reg_inv * Q_u;
meta->K_trajectory[i] = -Q_uu_reg_inv * Q_ux;
```

其中 $\bm k$ 是"前馈增益"（对整体的修正），$\bm K$ 是"反馈增益"（根据实际状态偏差做补偿）。如果控制量有上下界（比如 `min_u/max_u`），会退化成一个带边界约束的二次规划子问题，用投影牛顿法求解（`OptimizeActionModification`），这一细节属于工程健壮性处理，非核心原理。

再更新 $V_x, V_{xx}$ 往上一步传递：

```331:335:modules/planning/common/math/solver/solver.h
V_x = Q_x + meta->K_trajectory[i].transpose() * Q_uu * meta->k_trajectory[i] +
      meta->K_trajectory[i].transpose() * Q_u + Q_ux.transpose() * meta->k_trajectory[i];
V_xx = Q_xx + meta->K_trajectory[i].transpose() * Q_uu * meta->K_trajectory[i] +
       meta->K_trajectory[i].transpose() * Q_ux + Q_ux.transpose() * meta->K_trajectory[i];
```

#### Forward Pass + Line Search：前向传递

拿到每一步的 $(\bm k_i, \bm K_i)$ 后，按步长系数 $\alpha$（从 1 开始，逐步减半式回溯）滚动出新轨迹：

$$
\bm u_i^{new} = \bm u_i + \alpha \bm k_i + \bm K_i(\bm x_i^{new} - \bm x_i), \quad \bm x_{i+1}^{new} = f_d(\bm x_i^{new}, \bm u_i^{new})
$$

对应代码：

```244:251:modules/planning/common/math/solver/solver.h
(*u_trajectory_updated)[i] =
    u_trajectory[i] + context.alpha * context.k_trajectory[i] +
    context.K_trajectory[i] * ((*x_trajectory_updated)[i] - x_trajectory[i]);
...
(*x_trajectory_updated)[i + 1] =
    problem.dynamic_model(i).Evaluate((*x_trajectory_updated)[i], (*u_trajectory_updated)[i]);
```

**Line Search（回溯线搜索）**：如果 $\alpha=1$ 时新轨迹代价没有显著下降（或者变差了），就把 $\alpha$ 缩小（代码里预先算好了一组从 1 到 0.001 的候选值 `alpha_values_`），重新做 Forward Pass，直到代价下降比例满足要求（`cost_improvement_ratio > line_search_min_cost_improvement_ratio_`）或 $\alpha$ 小于 `min_alpha` 阈值放弃。这一步是保证 ILQR 全局收敛性的关键——纯粹的二次近似（$\alpha=1$）在非线性问题里不一定总让代价下降，需要线搜索兜底。

**`min_alpha` 的含义**：这是线搜索里允许尝试的最小步长。如果连最小的 $\alpha$ 都无法让代价下降或约束改进，就认为这一步迭代失败，进而触发增大正则化系数 `mu` 重新做 Backward Pass，或者判定整体求解失败。它是判断"这条搜索方向已经无法继续优化"的一个阈值型超参数。

### 5.5 收敛判定

ILQR 内层的退出条件（`Solver::Solve` 主循环里）：

- **梯度过小**：`normalized_grad < min_grad_thresh` 且正则化系数 `mu` 足够小 → 判定 `is_solved = true`，收敛退出（说明已经到了局部最优附近，几乎不需要再调整控制量）。
- **代价改进量过小**：`cost_improvement < min_cost_improvement_threshold` → 同样判定收敛退出。
- **正则化系数超过上限** `mu > max_mu` → 判定求解失败退出。

外层 AL 的退出条件已在 5.3 节列出。

---

## 6. Constraint 与 Cost 的完整清单与公式推导

下表汇总了本模块用到的所有约束/代价，包含来源、数学公式、代码文件、变量映射：

| 来源 | 名称 | 类型 | 公式 | 代码文件 |
|---|---|---|---|---|
| 车辆能力 | 曲率约束 | 硬约束 | $-\kappa_{max}\le\kappa\le\kappa_{max}$ | `constraints/clothoid_curvature_constraint.*` |
| 车辆能力 | 转向速率约束 | 硬约束 | $-\delta'_{max}\le\delta'\le\delta'_{max}$ | `constraints/clothoid_steering_rate_constraint.*` |
| 舒适性 | 横向加速度代价 | 软代价（死区二次） | 见 6.3 | `cost_functions/clothoid_lateral_acceleration_cost_function.*` |
| 舒适性 | 横向 jerk 代价 | 软代价（死区二次） | 见 6.4 | `cost_functions/clothoid_lateral_jerk_cost_function.*` |
| 舒适性 | dκ/ds 代价 | 软代价 | $J=\tfrac12 w\omega^2$ | `utils/kinematic_constraint_and_cost_generator.*` 内联生成 |
| 舒适性 | ddκ/dds 代价 | 软代价 | $J=\tfrac12 w\alpha^2$ | 同上 |
| 道路元素 | 半平面约束/代价 | 硬约束+软代价 | 见 6.5 | `constraints/half_plane_constraint.h`, `cost_functions/half_plane_cost_function.h` |
| 道路元素 | 光滑折线约束/代价 | 硬约束+软代价 | 见第 7 章 | `constraints/smooth_polyline_constraint.h`, `cost_functions/smooth_polyline_cost_function.h` |

### 6.1 曲率约束（Curvature Constraint）

$$
-\kappa_{max}\le\kappa\le\kappa_{max}
$$

拆成两个 $c\le0$ 形式的不等式：

$$
c_0 = \kappa - \kappa_{max} \le 0, \qquad c_1 = -\kappa - \kappa_{max} \le 0
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

雅可比非常简单：$\partial c_0/\partial\kappa=1$，$\partial c_1/\partial\kappa=-1$，其余全为 0。这个约束的 `PenaltyScale`（AL 初始惩罚缩放）设得比较大（`1000.0`），说明这是"绝对不能违反"的硬约束优先级最高。

### 6.2 转向速率约束（Steering Rate Constraint）

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

```10:19:modules/planning/path/clothoid_path_optimizer/constraints/clothoid_steering_rate_constraint.cc
const double kappa_wheelbase = x(ClothoidPathStateIndex::kKappa) * wheel_base_;
const double steering_rate =
    (reference_speed_ * wheel_base_ * x(ClothoidPathStateIndex::kDKappa)) /
    (kappa_wheelbase * kappa_wheelbase + 1.0);
VecC c(kDimension);
c(0) = (steering_rate - steering_rate_max_);
c(1) = (-steering_rate - steering_rate_max_);
```

雅可比逐项对照公式，完全一致：

```22:41:modules/planning/path/clothoid_path_optimizer/constraints/clothoid_steering_rate_constraint.cc
const double denom = 1.0 + kappa_wheelbase * kappa_wheelbase;  // κ²L²+1
const double denom_2 = denom * denom;                           // (κ²L²+1)²
jacobians.dcdx(0, kKappa) = -2.0 * speed_wheelbase * wheel_base_ * kappa_wheelbase *
                              x(kDKappa) / denom_2;  // ∂δ'/∂κ
jacobians.dcdx(0, kDKappa) = speed_wheelbase / denom;        // ∂δ'/∂ω
jacobians.dcdx(1, kKappa) = -jacobians.dcdx(0, kKappa);
jacobians.dcdx(1, kDKappa) = -jacobians.dcdx(0, kDKappa);
```

**数值验证**（来自单测）：$L=2.5,\ \delta'_{max}=2.0,\ v=1.7,\ \kappa=0.6,\ \omega=0.7$：

- $\kappa L = 1.5$，分母 $\kappa^2L^2+1=3.25$
- $\delta' = 1.7\times2.5\times0.7/3.25 = 0.91538$
- $c_0=0.91538-2.0=-1.08462\le0$（可行），$c_1=-2.91538\le0$（可行）
- $\partial\delta'/\partial\kappa = -2.11243$，$\partial\delta'/\partial\omega=1.30769$，与代码输出完全一致。

### 6.3 横向加速度代价（Lateral Acceleration Cost）

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

```12:32:modules/planning/path/clothoid_path_optimizer/cost_functions/clothoid_lateral_acceleration_cost_function.cc
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

梯度/Hessian（对 $\kappa$ 求导）：$\partial J/\partial\kappa = w\Delta a\cdot v^2$，$\partial^2 J/\partial\kappa^2 = wv^4$，代码：

```34:57:modules/planning/path/clothoid_path_optimizer/cost_functions/clothoid_lateral_acceleration_cost_function.cc
derivatives_ptr->dfdx(kKappa) = lateral_acceleration_weight_ * delta_lateral_acceleration * speed_squared;
derivatives_ptr->d2fdxdx(kKappa, kKappa) = lateral_acceleration_weight_ * speed_squared * speed_squared;
```

**数值验证**：$v=2.5,\ \tilde a=2.0,\ w=0.5,\ \kappa=0.6$：$a_{lat}=0.6\times6.25=3.75$，$\Delta a=1.75$，$J=0.5\times0.5\times1.75^2=0.765625$，$\partial J/\partial\kappa=5.46875$，$\partial^2 J/\partial\kappa^2=19.53125$，与单测一致。

### 6.4 横向 Jerk 代价（Lateral Jerk Cost）

对 $a_{lat}$ 关于时间求导得到横向 jerk：

$$
j_{lat} = \omega v^3 + 2\kappa v\, a_{lon}
$$

由于优化器里没有精确的纵向加速度 $a_{lon}$，忽略第二项，近似为：

$$
j_{lat}\approx \omega v^3
$$

同样是死区二次代价，形式与横向加速度完全同构（阈值改为 $\tilde j_{lat}$）：

```12:31:modules/planning/path/clothoid_path_optimizer/cost_functions/clothoid_lateral_jerk_cost_function.cc
const double dkappa = x(ClothoidPathStateIndex::kDKappa);
const double lateral_jerk = dkappa * reference_speed_ * reference_speed_ * reference_speed_;
// ... 与横向加速度代价相同结构的 deadzone quadratic ...
```

梯度/Hessian：$\partial J/\partial\omega = w\Delta j\cdot v^3$，$\partial^2 J/\partial\omega^2=wv^6$。

**数值验证**：$v=2.5,\tilde j=2.0,w=0.5,\omega=0.7$：$j_{lat}=0.7\times15.625=10.9375$，$\Delta j=8.9375$，$J=19.96973$，梯度 $69.82422$，Hessian $122.07031$，与单测一致。

### 6.5 dκ/ds 与 ddκ/dds 代价

为了减小方向盘转速，增加：

$$
J = \tfrac12 w\omega^2,\quad w\propto v^6
$$

为了减小方向盘小幅高频抖动，增加：

$$
J = \tfrac12 w\alpha^2,\quad w\propto v^8
$$

**理解这两个比例关系的物理意义**：如果把 $w=c\cdot v^6$ 代入第一个公式，$J = \tfrac12 c(v^3\omega)^2 = \tfrac12 c\, j_{lat}^2$，也就是说这本质上**等价于对横向 jerk 做了一个（无死区的）二次惩罚**——用一个简单的二次型近似替代了 6.4 节的死区二次代价，双管齐下让曲率变化更平顺。同理第二个公式等价于对 $j_{lat}$ 的变化率做惩罚。

代码里这两个代价函数是在 `KinematicConstraintAndCostGenerator` 里内联实现的一个简单二次代价函数对象（权重在生成时按速度的高次幂预先算好乘入 `weight`，具体计算逻辑见 `utils/kinematic_constraint_and_cost_generator.cc`），保持了与设计文档一致的比例关系。

### 6.6 半平面约束/代价（Half Plane）

**几何概念**：一个半平面由一个参考点 $\bm p_0$ 和单位法向量 $\hat{\bm n}$ 定义，把平面切成两侧：

$$
H=\{\bm p \mid \hat{\bm n}\cdot(\bm p-\bm p_0)\le0\}
$$

$\hat{\bm n}$ 指向的一侧是"禁止/违反"侧。典型应用场景是把一段近似笔直的车道边界表达成一个半平面：车辆不可以越过这条线。

**约束形式**：$c=\hat{\bm n}\cdot(\bm p-\bm p_0)\le0$，其中 $\bm p=(x,y)$ 可以是后轴中心，也可以通过 `PointTranslator`（第 7 章介绍）平移到车辆的前轴中心、角点等任意部位。

```62:69:modules/planning/path/clothoid_path_optimizer/constraints/half_plane_constraint.h
const Eigen::Vector2d position = x.template head<2>();
c(0) = normal_.dot(position - point_);
```

**代价形式**（死区二次，可行侧为零）：

$$
J=\begin{cases}0,& d\le0\\ \tfrac12 w d^2, & d>0\end{cases}, \quad d=\hat{\bm n}\cdot(\bm p-\bm p_0)
$$

```54:63:modules/planning/path/clothoid_path_optimizer/cost_functions/half_plane_cost_function.h
const double d = SignedDistance(x.template head<2>());
const double cost = d <= 0.0 ? 0.0 : 0.5 * weight_ * d * d;
```

**平移版本**：三种预定义别名对应车辆的三个关键位置：

| 别名 | 平移类型 | 用途 |
|---|---|---|
| `FrontCenterHalfPlaneConstraint` | 仅纵向偏移 | 约束前轴中心 |
| `RearWheelHalfPlaneConstraint` | 仅横向偏移 | 约束后轮 |
| `CornerHalfPlaneConstraint` | 纵向+横向偏移 | 约束车辆四角 |

半平面约束/代价的 `PenaltyScale`（`1.0`）比曲率/转速约束（`1000.0`）小很多，因为半平面约束会在折线上密集采样生成大量约束点，单个约束不宜给太重的初始惩罚。

---

## 7. 几何工具库：让"折线距离"处处可导

### 7.1 为什么需要专门做这件事

道路边界、限制区域、排斥区域等绝大多数道路元素都可以用一条**折线（polyline）**来描述。最自然的想法是直接用"点到折线的欧氏距离"构造约束/代价——但这个距离函数在折线的**拐点（两条线段的连接处）**是不可导的（分段函数在连接点两侧梯度方向不同），而 ILQR 需要代价函数的解析 Hessian。如果直接用不可导的距离函数，求解器在轨迹经过拐点附近时会数值不稳定甚至失败。

解决思路分两层：
1. **单条线段**（`DifferentiableLineSegment`）：先把"点到单条线段"的距离写成分段解析函数，各段内部精确可导（但线段自身在两端连接的地方仍不连续）。
2. **多条线段组合**（`SmoothPolyline`）：用 **LogSumExp（对数和指数）** 技巧把多条线段的距离做**软最小值（soft-min）**融合，从而在整条折线上获得**处处一阶、二阶连续可导**的距离场。

### 7.2 LogSumExp：数学地基

标准 soft-max（当 $\lambda\to\infty$ 时逼近真正的 $\max$）：

$$
\text{LSE}_\lambda(\bm x) = \frac1\lambda\ln\sum_i e^{\lambda x_i}
$$

代码里实现了一个数值更稳定、也更贴近真值的变体：

```19:25:modules/planning/path/clothoid_path_optimizer/math/log_sum_exp.h
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

```27:34:modules/planning/path/clothoid_path_optimizer/math/log_sum_exp.h
Eigen::Matrix<double, N, 1> Gradient(const Eigen::Matrix<double, N, 1>& x) const {
  const double max_coeff = x.maxCoeff();
  const Eigen::Array<double, N, 1> normalized_coeff_exp = (lambda_ * (x.array() - max_coeff)).exp();
  return normalized_coeff_exp / normalized_coeff_exp.sum();
}
```

**Hessian 是 softmax 的标准 Hessian**：

$$
\mathbf H_{\text{LSE}} = \lambda\big(\text{diag}(\bm s)-\bm s\bm s^T\big), \quad \bm s=\text{softmax}(\lambda\bm x)
$$

```36:42:modules/planning/path/clothoid_path_optimizer/math/log_sum_exp.h
Eigen::Matrix<double, N, N> Hessian(const Eigen::Matrix<double, N, 1>& x) const {
  const Eigen::Matrix<double, N, 1> softmax = Gradient(x);
  Eigen::Matrix<double, N, N> hessian;
  hessian.noalias() = -softmax * softmax.transpose();
  hessian.diagonal().array() += softmax.array();
  return lambda_ * hessian;
}
```

**数值直觉**：输入 `[1,2,3,4]`，$\lambda=5$，最大值 4 占绝对主导（$e^{5\times(-1)}=0.0067$ 这类项很小），所以 $\text{LSE}\approx4.0014$，非常接近真实最大值 4，梯度几乎是 one-hot（接近 `[0,0,0,1]`）。

### 7.3 DifferentiableLineSegment：单条线段的可导距离

设线段端点为 $\bm s,\bm e$，方向 $\hat{\bm t}=(\bm e-\bm s)/L$，法线 $\hat{\bm n}=(-\hat t_y,\hat t_x)$（逆时针转 90°，指向线段左侧）。

给定查询点 $\bm p$，先算它在线段方向上的投影长度 $t=\hat{\bm t}\cdot(\bm p-\bm s)$，分三种情况：

| 情况 | 条件 | 最近点 | 距离 | 梯度 |
|---|---|---|---|---|
| 垂足在起点前 | $t\le0$ | $\bm s$ | $\lVert\bm p-\bm s\rVert$ | $(\bm p-\bm s)/\lVert\bm p-\bm s\rVert$ |
| 垂足在线段内 | $0<t<L$ | $\bm s+\frac{t}{L}(\bm e-\bm s)$ | $\lvert\hat{\bm n}\cdot(\bm p-\bm s)\rvert$ | $\text{sign}(\cdot)\hat{\bm n}$ |
| 垂足在终点后 | $t\ge L$ | $\bm e$ | $\lVert\bm p-\bm e\rVert$ | $(\bm p-\bm e)/\lVert\bm p-\bm e\rVert$ |

```22:65:modules/planning/path/clothoid_path_optimizer/geometry/differentiable_line_segment.cc
Eigen::Vector2d DifferentiableLineSegment::GetNearestPoint(const Eigen::Vector2d& point,
                                                           ProjectionType* projection_type) const {
  const double inner_prod = direction_.dot(point - start_);
  if (inner_prod <= 0.0) { ...; return start_; }
  if (inner_prod >= length_) { ...; return end_; }
  ...
  return math::Lerp(start_, end_, inner_prod / length_);
}
```

**Hessian**：线段内部（垂足落在线段上）时距离是位置的线性函数（$\lvert\hat{\bm n}\cdot(\cdot)\rvert$ 的绝对值内部是线性的），所以 Hessian 恒为零矩阵；端点外侧则退化为"点到点距离"的标准 Hessian $\frac{\bm I - \hat{\bm v}\hat{\bm v}^T}{\lVert\bm v\rVert}$（$\bm v$ 为查询点到端点的向量）：

```67:76:modules/planning/path/clothoid_path_optimizer/geometry/differentiable_line_segment.cc
Eigen::Matrix2d DifferentiableLineSegment::DistanceHessian(const Eigen::Vector2d& point) const {
  const double inner_prod = direction_.dot(point - start_);
  if (math::IsInside(inner_prod, 0.0, length_, math::kEpsilon)) {
    return Eigen::Matrix2d::Zero();
  }
  const Eigen::Vector2d tmp_vector = inner_prod < 0.0 ? point - start_ : point - end_;
  const double squared_length = tmp_vector.squaredNorm();
  return (Eigen::Matrix2d::Identity() - tmp_vector * tmp_vector.transpose() / squared_length) /
         std::sqrt(squared_length);
}
```

**关键认知**：单条线段本身的距离梯度/Hessian 在 $t=0$ 和 $t=L$ 处是不连续的（跳变），这个不连续性正是留给 `SmoothPolyline` 用 LogSumExp 去抹平的。

### 7.4 SmoothPolyline：把多条线段拼成一条处处光滑的折线

给定一串点 $\{p_0,\ldots,p_n\}$，构造 $n-1$ 条 `DifferentiableLineSegment`。核心操作是把点到各线段的距离向量 $\bm d(\bm p)=(d_1,\ldots,d_N)$ 用 **soft-min** 融合：

$$
d_{\text{smooth}}(\bm p) = -\text{LSE}_\lambda(-\bm d(\bm p)) = \min(\bm d) + \frac1\lambda\ln\Big(\sum_ie^{\lambda(\min(\bm d)-d_i)}-1\Big)
$$

```209:214:modules/planning/path/clothoid_path_optimizer/geometry/smooth_polyline.h
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

```217:221:modules/planning/path/clothoid_path_optimizer/geometry/smooth_polyline.h
Eigen::Vector2d SmoothPolyline<N>::SmoothDistanceGradient(const Eigen::Vector2d& point) const {
  const Eigen::Matrix<double, N, 1> minus_distances = -Distances(point);
  const Eigen::Matrix<double, N, 1> log_sum_exp_gradient = log_sum_exp_.Gradient(minus_distances);
  return DistanceGradients(point) * log_sum_exp_gradient;
}
```

**Hessian**：由两部分组成——各线段 Hessian 的加权和，加上 softmax 权重本身随位置变化带来的额外曲率项（这一项恰好在拐点附近起主导作用，负责把两条线段梯度方向的"跳变"抹平成光滑过渡）：

```233:247:modules/planning/path/clothoid_path_optimizer/geometry/smooth_polyline.h
smooth_hessian.noalias() =
    -distance_gradients * log_sum_exp_hessian * distance_gradients.transpose();
...
for (int i = 0; i < line_segments_.size(); ++i) {
  smooth_hessian += log_sum_exp_gradient[i] * distance_hessians[i];
}
```

**左右侧判断**：`SmoothDistance` 返回的是无符号（非负）距离，是否在折线的"违反侧"要另外调用 `IsPointOnRightSide`，本质是找到最近线段后看点相对该线段法线的位置：

```110:117:modules/planning/path/clothoid_path_optimizer/geometry/smooth_polyline.h
bool SmoothPolyline<N>::IsPointOnRightSide(const Eigen::Vector2d& point, int* nearest_index) const {
  const int tmp_nearest_index = GetNearestLineSegmentIndex(point);
  return line_segments_[tmp_nearest_index].IsPointOnRightSide(point);
}
```

拐角处等距归属的判断用了**角平分线**修正（避免最近线段判断在拐角处出现"跳变式"的错误归属）。

**数值直觉**（折线形如 "⊂"，见 `smooth_polyline_test.cc`）：点 (1,0) 距最近线段的欧氏距离恰为 1.0，`SmoothDistance` 算出约 0.999995，几乎无偏差，因为此时只有一条线段占绝对主导权重；点 (3,0) 靠近拐角，两条线段距离接近，`SmoothDistance` 约为 0.996769，比真实最小值 1.0 略小——这就是 soft-min 在拐角附近产生的"温和低估"，正是为了让梯度/Hessian 保持连续所付出的代价。

### 7.5 PointTranslator：把约束/代价施加到车辆任意部位

车辆状态里的 $(x,y,\theta)$ 描述的是后轴中心，但很多约束（比如碰撞约束）需要施加在车辆的四个角点或前轴中心上。`PointTranslator` 提供从车体坐标系偏移 $(\Delta x,\Delta y)$ 到全局坐标的变换：

$$
\bm f(\bm q) = \begin{pmatrix}x+\cos\theta\,\Delta x-\sin\theta\,\Delta y\\ y+\sin\theta\,\Delta x+\cos\theta\,\Delta y\end{pmatrix}, \quad \bm q=(x,y,\theta)
$$

Jacobian：

$$
\mathbf J = \frac{\partial\bm f}{\partial\bm q} = \begin{pmatrix}1&0&-\Delta y_w\\0&1&\Delta x_w\end{pmatrix}
$$

```119:128:modules/planning/path/clothoid_path_optimizer/geometry/point_translator.h
jacobian.topLeftCorner<2, 2>().setIdentity();
jacobian(0, 2) = -world_frame_offset(1);
jacobian(1, 2) = world_frame_offset(0);
```

三种平移类型 `kLongitudinal`（只有 $\Delta x$，用于前轴中心）、`kLateral`（只有 $\Delta y$，用于后轮/车道边缘）、`kBoth`（两者都有，用于四角点）。因为坐标变换本身含有 $\theta$ 的三角函数，是非线性的，`PointTranslator` 还额外提供了 Jacobian 对状态量的二阶导（`derivative_jacobian_wrt_x/y`），配合链式法则可以推出完整的二阶 Hessian，而不是仅用 Gauss-Newton 近似。

**用途**：所有 `constraints/` 和 `cost_functions/` 里带 `TranslatedPoint` 前缀或 `FrontCenter/RearWheel/Corner` 前缀的类，都是"基础版约束/代价 + `PointTranslator`"的组合，通过链式法则把对目标点坐标的梯度/Hessian 转换回对车辆状态 $(x,y,\theta)$ 的梯度/Hessian。

---

## 8. utils/：把决策层信息组装成约束与代价

`utils/` 目录是"胶水层"：上游决策模块给出的都是高层语义信息（车道边界在哪、这块区域要不要避让、这个障碍物往左还是往右绕开、留多少 buffer），这一层负责把它们转换成 ILQR 能直接使用的 `Constraint`/`CostFunction` 对象数组，按每个采样点组织起来。

### 8.1 两层结构：参数生成 → 约束代价生成

- **参数生成层**（`generate_restriction_cost_params.*`、`generate_repulsion_cost_params.*`、`generate_box_collision_params.*`）：把原始输入（Restriction/Repulsion/box collision 可行区间函数）在一批 s 采样点上转换成结构化的中间参数（`RestrictionCostParams`、`RepulsionCostParams`、`BoxCollisionParams`，定义在 `clothoid_path_optimizer_common.h`），主要做**坐标转换**（s-l → x-y）和**采样**。
- **约束代价生成层**（各个 `*_generator.*`）：拿着这些中间参数（或直接拿原始决策输入），针对每个轨迹采样点生成具体的 `Constraint`/`CostFunction` 对象。

### 8.2 KinematicConstraintAndCostGenerator：车辆能力 + 舒适性打包

这是"车辆能力+舒适性"这一大类的统一组装入口，对每个采样点（跳过初始点 $i=0$，因为初始状态是给定的边界条件不需要约束）分别生成：

- 曲率约束（`ClothoidCurvatureConstraint`）
- 转向速率约束（`ClothoidSteeringRateConstraint`，需要用到该采样点对应的参考速度）
- 横向加速度代价（`ClothoidLateralAccelerationCostFunction`）
- 横向 jerk 代价（`ClothoidLateralJerkCostFunction`）
- dκ/ds、ddκ/dds 二次代价（权重按速度高次幂预先计算好）

每个采样点用的参考速度来自 `reference_speed_profile`（见 8.6 节），这就是为什么这一步需要传入 `reference_path` 和 `reference_speed_profile` 两个参数一起使用——速度决定了转向速率约束的具体数值、也决定了舒适性代价的具体数值。

### 8.3 五类道路元素 generator 的区别

| Generator | 对应元素 | 硬约束/软代价 | 车辆模型 |
|---|---|---|---|
| `RestrictionConstraintAndCostGenerator` | 可行驶区域左右边界（如车道边界、路口边界） | 硬约束 + 软代价 | 后轴单点 |
| `RepulsionCostGenerator` | 软性避让区域（如靠近但不禁止） | 纯软代价 | 后轴单点 |
| `LaneBoundaryCostGenerator` | 车道线（区分实线/虚线） | 纯软代价 | 四角模型 |
| `BoxCollisionConstraintAndCostGenerator` | 精确碰撞边界（四角避障） | 硬约束 + 软代价 | 四角模型 |
| `ObstacleConstraintAndCostGenerator` | 具体障碍物（带 nudge 方向和 buffer） | 硬约束 + 软代价 | 四角模型 |

**Restriction vs Repulsion 的区别**：Restriction 是"绝对不能越过的边界"（比如道路物理边界），必须同时有硬约束兜底；Repulsion 是"希望离得越远越好，但不是不可逾越"（比如临时性的软避让区域，如对向来车的安全裕度），只用代价表达，不设硬约束。

**为什么车道边界、碰撞边界、障碍物都用"四角模型"**：车辆是一个有长宽的矩形，只用后轴中心（一个点）去判断是否碰撞，在狭窄空间或大转弯时会严重低估实际占用空间；用车辆四个角点分别去检查与边界/障碍物的距离，能显著提升狭窄空间的通过能力和安全性，这也正是设计文档里提到的"提供更加精确的车辆碰撞描述方法（四角碰撞）"。

### 8.4 四角碰撞（Box Collision）的具体做法

`generate_box_collision_params.*` 先把车辆矩形轮廓换算成四个关键点（左前、右前、左后、右后），每个点通过 `PointTranslator` 用 `kBoth` 类型从后轴中心偏移得到（前角需要纵向+横向偏移，后角只需要横向偏移，因为后轴中心本身就在纵向上和后角对齐）。然后为每个角点分别在左右两侧生成一条对应的碰撞边界折线（`right_box_collision_boundaries` / `left_box_collision_boundaries`，见 `BoxCollisionParams` 结构体）。

`BoxCollisionConstraintAndCostGenerator` 再基于这些折线，对每个角点分别生成 `SmoothPolyline` 约束（硬性不可碰撞）和代价（软性远离边界）。

### 8.5 GuideLineCostGenerator：贴近引导线的代价

引导线（guide line）是一个泛指：可以是参考线、车道中心线，也可以是路口跟车/变道时决策层给的引导线。目标很简单——path 上的每个采样点尽可能贴近这条线：

$$
J = \tfrac12 w\, d^2, \quad d=\text{点到guide line的距离}
$$

代码里对起点附近和终点分别有特殊处理（比如终点用更大的权重收紧，保证 path 末端与引导线对齐），并且用 Huber 型代价避免远离引导线时代价增长过快压制其他更重要的约束。

### 8.6 Reference Speed Profile：piecewise jerk 速度预估

优化器只知道起始点的精确速度（车辆当前速度），后续采样点没有精确速度信息，但舒适性代价（横向加速度/jerk）恰恰需要速度参与计算。解决方法是预先用一个简化模型（**piecewise jerk**）估计一条速度曲线：用恒定 jerk 把加速度积分到 0，然后匀速行驶（如果当前是刹车状态，则先用恒定 jerk 到达匀减速、再到低速保持），得到的是 v-t 曲线，再通过时间积分转换成 v-s 曲线供后续插值使用。

有一个例外情况：当参考线上探测到大曲率（意味着接下来速度可能有较大变化，比如进入急转弯），直接复用上一帧规划出的速度曲线，避免预估模型在这种场景下产生不合理的远端速度估计。

### 8.7 ClothoidPathOptimizerDebugger：调试信息

填充 `ClothoidPathOptimizerDebug` proto，包括车辆参数、s 采样步长、速度曲线、引导线、限制区/排斥区/碰撞边界折线的几何形状、参考路径/初始路径/最终优化路径三条曲线、各类约束代价的可视化数据，以及编译问题耗时、求解耗时等性能数据。这些信息最终会喂给可视化插件（如 mviz），方便调参和排查问题。

---

## 9. 完整求解主流程走读（ClothoidPathOptimizer::ComputePathCurve）

主入口 `ComputePathCurve` 大致按以下顺序执行（对应 `clothoid_path_optimizer.cc`）：

1. **生成道路元素采样点**（`GenerateRestrictionAndRepulsionSSample`）：从近到远，分辨率从 0.5m→1m→10m 逐级放粗，兼顾精度与计算量。
2. **生成限制/排斥/碰撞的中间参数**（`GenerateRestrictionAndRepulsionCostParams`）：调用 8.1 节的参数生成层。
3. **生成引导线**（`GenerateDefaultGuideLine` 或 `GenerateGuideLine`）：如果决策层没给引导线，用限制区间中点（结合特定的 repulsion 做微调）自动生成一条默认引导线。
4. **生成初始解**（`GenerateInitialPaths`）：以参考路径曲线为基准采样出一条初始状态轨迹，控制量用相邻两点 dkappa 差分近似；如果开启了 `enable_two_stage_solve`，还会先用一个独立的 `FeasibleTrajectoryGenerator`（本质是只考虑运动学约束的简化 AL+ILQR 问题）求一条更"可行"的初始解，再用它替换粗糙的差分近似，帮助后续完整问题更快更稳地收敛。
5. **生成参考速度曲线**（`GenerateReferenceSpeedProfile`，见 8.6 节）。
6. **（可选）两阶段求解**：先用只含运动学约束+guide line+边界的"简化问题"（`GenerateFeasiblePath`）求一条可行轨迹，替换初始解里的轨迹部分，降低完整问题的求解难度。
7. **组装完整的约束代价集合**（`GenerateConstraintsAndCosts`）：调用 8.2/8.3/8.5 节的所有 generator，把结果汇总进 `ConstraintAndCostSet`。
8. **按采样点打包成 Problem**（`ConstructPathProblem`）：把每个采样点上散落在多个集合里的约束/代价对象，通过 `CollectConstraintsAtSample`/`CollectCostsAtSample` 合并成该采样点唯一的一个约束向量和一个代价函数对象。
9. **调用 AL+ILQR 求解器**（`Solver::Solve`，见第 5 章）。
10. **失败兜底**：如果没求解成功（`solution.state != kSolved`），记录失败原因（`ReportSolveFailedEvent`）并返回上一帧路径或初始解里代价最小的一条（`GetFallbackSolution`），保证下游模块始终能拿到一条可用的 path，不会因为求解失败而完全没有输出。

---

## 10. 一个简化的端到端数值例子

为了建立直观理解，这里用简化数字走一遍单个采样点上"曲率约束 + 横向加速度代价"是怎样共同影响 ILQR 一步迭代的（省略动力学 Jacobian 与其它代价，只展示核心思路）：

假设某采样点当前状态 $\kappa=0.6,\ \omega=0.7$，参考速度 $v=2.5$：

1. **曲率约束**：$\kappa_{max}=2.0$，$c_0=0.6-2.0=-1.4<0$，可行，AL 惩罚为 0（不在违反侧，且假设乘子 $\lambda=0$）。
2. **横向加速度代价**：$a_{lat}=0.6\times2.5^2=3.75$，阈值 $\tilde a=2.0$，超出 $\Delta a=1.75$，代价 $J=0.5\times0.5\times1.75^2=0.765625$，梯度 $\partial J/\partial\kappa=5.46875$。
3. 这个梯度会作为 $l_x$ 的一部分，参与 Backward Pass 里 $Q_x=l_x+f_x^TV_x$ 的计算，进而影响该采样点反馈增益 $K$ 和前馈增益 $k$ 的求解——直观理解就是：由于当前曲率导致横向加速度超标产生了正的代价梯度，ILQR 会倾向于把该点及附近的 $\kappa$ 往更小的方向调整（沿负梯度方向），从而在下一次 Forward Pass 里让新轨迹的横向加速度代价降低。
4. 如果外层 AL 迭代中曲率约束的乘子 $\lambda_{c_0}$ 已经被上一轮更新为正值（说明历史上出现过违反），即使当前 $c_0<0$ 仍然可行，只要 $\lambda_{c_0}>0$ 就仍然会有一次性的 AL 惩罚项参与（这是不等式约束 AL 方法的标准处理，让约束满足有一定"记忆"，防止在临界值附近反复震荡）。

多个采样点、多种约束代价的叠加最终通过 ILQR 反复的 Backward+Forward Pass 收敛到一条同时兼顾"不超曲率限制"和"横向加速度尽量小"的轨迹。

---

## 11. 关键设计取舍 FAQ

**为什么不直接用曲率作代价，而是作硬约束？**
因为曲率超限对应车辆物理上转不过来的弯，这是绝对不能违反的边界，用硬约束（AL 方法逐步增大惩罚直至满足）比软代价（只是"倾向于"限制）更可靠。

**为什么横向加速度/jerk 用死区二次代价，而不是硬约束？**
横向加速度/jerk 是舒适性指标，不是物理硬限制——稍微超过一点阈值不代表不可行，只是不够舒适，所以用带死区的软代价（阈值内零惩罚，超出后二次增长）更合适，也给了优化器更大的可行空间去平衡其它约束。

**为什么要专门设计 SmoothPolyline 而不是直接用折线距离？**
详见第 7 章：ILQR 需要处处可导（至少二阶）的代价函数，折线原始距离函数在拐点不可导，必须用 LogSumExp 光滑化。

**为什么要有"两阶段求解"（先求可行解，再求完整最优解）？**
完整问题包含大量非凸的碰撞类约束，直接从粗糙的初始解开始求解容易不收敛或收敛到很差的局部最优。先用一个只含运动学约束的简化问题求一条运动学可行的轨迹作为更好的初始解，能显著提升完整问题的收敛速度和成功率，这是非凸轨迹优化里常见的"热启动（warm start）"思路。

**为什么求解失败还要返回 fallback？**
Planning 模块每个规划周期都要有输出，不能因为一次数值求解失败就让车辆"没有路径可走"。返回上一帧路径或者候选初始解里代价最小的一条，保证系统的鲁棒性，同时把失败信息上报（`ReportSolveFailedEvent`）供后续排查。

**为什么 `min_alpha` 是一个需要调的参数？**
`min_alpha` 决定了线搜索能接受的最小步长。设得太大，可能在还有改进空间时就放弃搜索，导致收敛提前终止或频繁触发正则化增大；设得太小，会在明显无法改进的方向上浪费过多计算尝试很小的步长。它本质上是"收敛判定严格程度"与"求解耗时"之间的权衡参数，也是本模块调参实践中经常需要根据实际收敛效果微调的参数之一。
