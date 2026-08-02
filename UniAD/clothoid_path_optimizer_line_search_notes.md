# Line Search（线搜索）从零学起：公式推导 + 代码对应 + 逻辑流程

> 参考资料：[《Line Search方法》学城文档](https://km.sankuai.com/collabpage/2777844970)
> 对应模块：迭代式轨迹优化求解器（iLQR）内层求解逻辑
>
> 本文聚焦于 AL-iLQR 算法 Forward Pass 里"线搜索"这一步到底在做什么、为什么这么设计，把数值优化理论和求解器的实现逻辑逐段对应起来。
>
> 本文假设你是**完全零基础**的小白，不需要任何优化理论背景，仅凭本文内容即可完整理解线搜索机制，无需查阅其他资料。

---

## 目录

1. [先建立直觉：什么是"线搜索"](#1-先建立直觉什么是线搜索)
2. [下降方向：往哪个方向走](#2-下降方向往哪个方向走)
3. [步长：走多远——为什么不能简单地"只要下降就行"](#3-步长走多远为什么不能简单地只要下降就行)
4. [Wolfe 条件：判断一个步长"好不好"的黄金标准](#4-wolfe-条件判断一个步长好不好的黄金标准)
5. [Goldstein 条件：另一种思路](#5-goldstein-条件另一种思路)
6. [回溯法（Backtracking）：工程上最常用的简化方案](#6-回溯法backtracking工程上最常用的简化方案)
7. [线搜索方法的收敛性：为什么这样做能保证算法不失败](#7-线搜索方法的收敛性为什么这样做能保证算法不失败)
8. [牛顿法/拟牛顿法的全局收敛性](#8-牛顿法拟牛顿法的全局收敛性)
9. [代码总览：求解器里的一次迭代做了什么](#9-代码总览求解器里的一次迭代做了什么)
10. [逐段代码对应：Backward Pass 里的下降方向从哪来](#10-逐段代码对应backward-pass-里的下降方向从哪来)
11. [逐段代码对应：Forward Pass 里的回溯线搜索](#11-逐段代码对应forward-pass-里的回溯线搜索)
12. [逐段代码对应：正则化 mu 如何充当"信赖域"](#12-逐段代码对应正则化-mu-如何充当信赖域)
13. [完整迭代流程图](#13-完整迭代流程图)
14. [常见疑问 FAQ](#14-常见疑问-faq)

---

## 1. 先建立直觉：什么是"线搜索"

想象你在爬山（这里我们要"下山"，即让代价函数变小），蒙着眼睛，只能感觉到脚下的坡度。你每一步要做两件事：

1. **往哪个方向迈** —— 这叫**搜索方向** $p_k$。
2. **这一步迈多大** —— 这叫**步长** $\alpha_k$。

数学上，从当前位置 $x_k$ 走到下一个位置 $x_{k+1}$ 的更新公式是：

$$
x_{k+1} = x_k + \alpha_k p_k
$$

**线搜索（line search）** 就是专门研究"**给定了方向 $p_k$ 之后，$\alpha_k$ 到底该取多少**"这个子问题的一整套理论和方法。

为什么这很重要？因为：
- $\alpha_k$ 太大 → 一步迈太远，可能"迈过头"，甚至代价不降反升，算法震荡不收敛。
- $\alpha_k$ 太小 → 每一步进展都很小，算法要跑很久很久才能收敛，效率低。

我们需要一套"聪明"的规则，能自动找到一个"刚刚好"的步长。这正是本文要讲的内容，也正是 `clothoid_path_optimizer` 底层 iLQR 求解器里 `alpha_values_` 那部分代码在做的事。

---

## 2. 下降方向：往哪个方向走

### 2.1 什么样的方向算"下降方向"

如果沿着方向 $p_k$ 走一个很小的步子，代价函数 $f$ 的值会变小，我们就说 $p_k$ 是一个**下降方向（descent direction）**。数学条件是：

$$
p_k^T \nabla f_k < 0
$$

直觉理解：$\nabla f_k$（梯度）指向"函数增长最快"的方向，那么只要 $p_k$ 和梯度方向**夹角大于 90°**（点积为负），沿着 $p_k$ 走一点点，函数值就会减小。

### 2.2 常见的方向选法：$p_k = -B_k^{-1}\nabla f_k$

- **最速下降法（steepest descent）**：$B_k = I$（单位矩阵），方向就是 $-\nabla f_k$，即"哪个方向下降最快就往哪走"。简单，但收敛慢（容易"之字形"震荡）。
- **牛顿法（Newton's method）**：$B_k = \nabla^2 f(x_k)$，即 Hessian 矩阵（二阶导数）。用二阶信息判断"这个方向下降后还能再下降多少"，收敛快，但每一步都要算二阶导数、求逆，计算量大。
- **拟牛顿法（quasi-Newton）**：$B_k$ 是 Hessian 的近似（通过历史梯度信息估计出来，不用真的算二阶导数），是速度和精度的折中。

### 2.3 为什么 $B_k$ 正定就一定是下降方向

如果 $B_k$ 是**正定矩阵**（可以理解成"处处都是凸的、像碗一样的形状"），那么：

$$
p_k^T\nabla f_k = -\nabla f_k^T B_k^{-1}\nabla f_k < 0
$$

这是因为正定矩阵的逆也是正定的，而正定矩阵夹在任何非零向量两边做内积，结果必然大于 0（这是正定矩阵的定义），所以加上负号后必然小于 0。**这就保证了只要 $B_k$ 正定，$p_k$ 就一定是下降方向**——这是后面第 8 节"牛顿法全局收敛性"的理论基石。

> **对应到实现**：在 iLQR 求解器里，方向 $p_k$ 就是 Backward Pass 算出来的 $(K_k, d_k)$（反馈增益和前馈修正），第 10 节会详细对应实现逻辑。

---

## 3. 步长：走多远——为什么不能简单地"只要下降就行"

### 3.1 最朴素的想法为什么不够

理想情况下，我们希望找到让 $\phi(\alpha) = f(x_k+\alpha p_k)$（沿着方向 $p_k$ 这条射线上，函数值随 $\alpha$ 的变化）**取得全局最小值**的那个 $\alpha$。但这样做太贵了——要反复计算 $f$（甚至梯度），计算成本高。

那退而求其次：只要求 $f(x_k+\alpha_k p_k) < f(x_k)$（下降就行），是不是就够了？

**答案是：不够。** 书中给了一个反例：假设真正的最小值是 $f^*=-1$，但迭代序列是

$$
f(x_k) = \frac{5}{k},\quad k=1,2,3,\dots
$$

虽然每一步 $f(x_k)$ 确实都在变小（$5, 2.5, 1.67,\dots$），但它的极限是 $0$，而不是 $-1$！**每一步下降得"太少太少"，永远也到不了真正的最优点。**

### 3.2 解决办法：充分下降条件（sufficient decrease condition）

要避免"下降太慢"的陷阱，需要一个更严格的条件：**不仅要下降，下降的幅度还要"足够大"**（和步长、当前下降方向的陡峭程度成比例）。这就引出了下一节的 Wolfe 条件。

---

## 4. Wolfe 条件：判断一个步长"好不好"的黄金标准

Wolfe 条件由两部分组成，分别防止步长"太大"和"太小"。

### 4.1 Armijo 条件（充分下降条件）——防止步长"太大"/下降太慢

$$
f(x_k+\alpha_k p_k) \le f(x_k) + c_1\alpha_k\nabla f_k^Tp_k, \quad c_1\in(0,1)
$$

直觉：右边是一条**直线** $l(\alpha) = f(x_k) + c_1\alpha\nabla f_k^Tp_k$，因为 $\nabla f_k^Tp_k<0$（下降方向），这条线是向下倾斜的。$c_1$ 通常取一个很小的数（比如 $10^{-4}$），所以这条线比"真正的切线"更平缓一点。

**只要新的函数值落在这条线的下方，就说明这一步"下降得足够多"**（不会出现前面 5/k 那种"每步都降，但降得不够"的病态情况）。

### 4.2 曲率条件（Curvature Condition）——防止步长"太小"

Armijo 条件有个漏洞：只要 $\alpha$ 取得足够小，几乎总能满足它（因为线性近似在很小的邻域内总是准的）。这样算法可能会一直用特别小的步长，导致收敛巨慢。

所以再加一个条件，专门排除"过小"的步长：

$$
\nabla f(x_k+\alpha_k p_k)^Tp_k \ge c_2\nabla f_k^Tp_k, \quad c_2\in(c_1,1)
$$

直觉：左边 $\nabla f(x_k+\alpha_kp_k)^Tp_k$ 其实就是 $\phi'(\alpha_k)$——**走到这一步之后，沿着 $p_k$ 方向的坡度**。如果这个坡度还是**很负**（说明沿着这个方向继续走还能大幅下降），那就说明你停得太早了，步子迈得不够大。曲率条件要求这个"剩余坡度"不能太陡，逼着算法多走一点。

### 4.3 为什么叫"Wolfe 条件"

把 4.1（Armijo，充分下降）+ 4.2（曲率条件，防止太小）合在一起，就是 **Wolfe 条件**：既不让步子太大导致下降不够，也不让步子太小导致效率低下——**恰到好处**。

### 4.4 Strong Wolfe 条件：更严格的版本

普通 Wolfe 条件的曲率条件只限制了"不能太负"，但没限制"不能太正"——万一步子迈过了头，坡度变成很大的正数（意味着已经冲过了谷底往上爬了），普通 Wolfe 条件是允许的。

**Strong Wolfe 条件** 把曲率条件改成绝对值形式，更严格：

$$
\big|\nabla f(x_k+\alpha_kp_k)^Tp_k\big| \le c_2\big|\nabla f_k^Tp_k\big|
$$

这样无论坡度是很负还是很正，只要绝对值太大，就说明离"驻点"（坡度为 0 的点，即这条射线上的局部最优点）还太远，不接受这个步长。**Strong Wolfe 更稳健，在拟牛顿法里更常用**。

### 4.5 理论保证：一定存在满足 Wolfe 条件的步长

学城文档里的引理证明了：只要 $f$ 连续可微、$p_k$ 是下降方向、$f$ 沿着这条射线有下界（不会跌到 $-\infty$），那么**一定存在**一段区间的 $\alpha$ 同时满足 Wolfe 条件和 Strong Wolfe 条件。证明的核心技巧：

1. $\phi(\alpha)=f(x_k+\alpha p_k)$ 有下界，而 Armijo 条件里的直线 $l(\alpha)$ 是无下界的（一直往下延伸），所以两条曲线**必然相交**。
2. 取第一个交点 $\alpha'$，用**均值定理**（区间内总存在一点，其瞬时斜率等于整个区间的平均斜率）在 $(0,\alpha')$ 之间找到一点 $\alpha''$，能证明它同时满足 Wolfe 条件的两部分。

这个理论保证很重要：**它告诉我们"回溯线搜索"这种简单的算法一定能找到解**，不会陷入无解的死循环——这也是为什么代码里的回溯循环（第 11 节）设了一个步数上限就足够安全。

### 4.6 Wolfe 条件的一个好性质：尺度不变性

把目标函数 $f$ 乘上任意正的常数，或者对变量做仿射变换，Wolfe 条件依然成立、不会被破坏。这个性质让 Wolfe 条件特别适合各种优化算法，尤其是拟牛顿法（不需要根据具体问题手动调整超参数）。

---

## 5. Goldstein 条件：另一种思路

$$
f(x_k) + (1-c)\alpha_k\nabla f_k^Tp_k \;\le\; f(x_k+\alpha_kp_k) \;\le\; f(x_k) + c\alpha_k\nabla f_k^Tp_k, \quad 0<c<1/2
$$

- **右边不等式**：和 Armijo 条件一样，保证下降幅度足够。
- **左边不等式**：换一种方式限制步长不能太小——**直接给函数值设一个下界**，而不像 Wolfe 条件那样看"坡度"。

两者殊途同归，都是在"步长太大"和"步长太小"之间找平衡点。**Goldstein 条件常用在牛顿法里，但不适合拟牛顿法**（因为拟牛顿法需要保持 Hessian 近似矩阵的正定性，Goldstein 条件可能会排除掉一些能保持正定性的合理步长）。

---

## 6. 回溯法（Backtracking）：工程上最常用的简化方案

如果只需要满足 Armijo 条件（不追求曲率条件那么复杂），有一种非常简单实用的方法——**回溯法**：

**算法步骤：**
1. 选一个初始步长 $\bar\alpha$（牛顿法里通常从 $\bar\alpha=1$ 开始，因为牛顿法的"满步"通常已经很接近最优）。
2. 设一个缩小因子 $\rho\in(0,1)$（比如 0.5）和常数 $c\in(0,1)$。
3. 从 $\alpha=\bar\alpha$ 开始，检查是否满足 Armijo 条件。
4. 如果不满足，缩小步长：$\alpha \leftarrow \rho\alpha$，回到第 3 步重新检查。
5. 一旦满足，就用这个 $\alpha$ 作为最终步长 $\alpha_k$。

**这正是 `clothoid_path_optimizer` 的 iLQR 求解器里实际采用的策略！** 只不过它不是用一个固定的缩小因子 $\rho$ 反复相乘，而是**预先算好了一串逐渐变小的候选步长表**（`alpha_values_`），依次尝试，第一个满足条件的就采用。本质思想完全一致：**从大步长开始尝试，不行就换更小的，直到满足下降条件为止**。第 11 节会展开讲代码细节。

---

## 7. 线搜索方法的收敛性：为什么这样做能保证算法不失败

### 7.1 方向与负梯度的夹角

设 $\theta_k$ 是搜索方向 $p_k$ 与负梯度 $-\nabla f_k$ 之间的夹角：

$$
\cos\theta_k = \frac{-\nabla f_k^Tp_k}{\|\nabla f_k\|\,\|p_k\|}
$$

- $\theta_k$ 越小（$\cos\theta_k$ 越接近 1），说明 $p_k$ 越接近"最陡下降方向"。
- 只要 $\cos\theta_k>0$，$p_k$ 就是一个合法的下降方向。

### 7.2 Zoutendijk 定理：线搜索理论的核心结论

在比较宽松的假设下（$f$ 连续可微、有下界、梯度 Lipschitz 连续，即梯度变化不会突变），**只要步长满足 Wolfe 条件**，就能推出：

$$
\sum_{k=0}^\infty \cos^2\theta_k\,\|\nabla f_k\|^2 < \infty
$$

这个无穷级数是有限的（收敛的），意味着它的每一项必须**趋向于 0**。也就是说：

> **要么梯度范数 $\|\nabla f_k\|\to 0$（已经走到了坡度接近于 0 的地方，即局部最优附近），要么 $\cos\theta_k \to 0$（方向和负梯度越来越垂直，方向"失效"了）。**

### 7.3 实用结论

如果我们能保证搜索方向和负梯度的夹角"不要太离谱"（$\cos\theta_k$ 有一个正的下界 $\delta$，即方向不会退化成和梯度垂直），那么就能推出：

$$
\|\nabla f_k\| \to 0
$$

**这意味着算法一定会收敛到一个驻点**（坡度为 0 的点），不会永远在某个"高原"上无意义地震荡下去。这正是为什么"选一个合理的搜索方向 + 用 Wolfe 条件选步长"这套组合拳，在理论上是靠谱的。

---

## 8. 牛顿法/拟牛顿法的全局收敛性

### 8.1 "全局收敛"的准确含义

这里"全局收敛（globally convergent）"的定义是：

$$
\lim_{k\to\infty}\|\nabla f_k\| = 0
$$

**注意：这不代表一定收敛到全局最优解**，只是保证收敛到一个**驻点**（可能是极小点、极大点，甚至鞍点）。要进一步保证是极小点，还需要额外利用 Hessian 的负曲率信息。

### 8.2 牛顿类方法收敛的充分条件

对于形如 $p_k=-B_k^{-1}\nabla f_k$ 的方向（$B_k$ 对称正定），如果：

1. 每个 $B_k$ 都是正定的（保证 $p_k$ 是下降方向，见第 2.3 节）。
2. **条件数一致有界**：$\|B_k\|\cdot\|B_k^{-1}\|\le M$（矩阵不会"病态"，方向不会偏得离谱）。

那么可以推出 $\cos\theta_k \ge 1/M$（方向和负梯度的夹角有一个固定的上限），结合 Zoutendijk 条件，就能得到：

$$
\lim_{k\to\infty}\|\nabla f_k\|=0
$$

**这就是为什么"控制矩阵条件数"在数值优化里如此重要**——如果 $B_k$（对应到 iLQR 里就是 $Q_{uu}+\rho I$）病态，方向会失真，算法可能不收敛。这也解释了为什么 iLQR 代码里要在 $Q_{uu}$ 上加正则化项 $\rho I$（第 12 节详解）：**加正则化本质上就是在人为地改善条件数，让 $B_k$ 保持良好的正定性和条件数，从而保证收敛性**。

---

## 9. 代码总览：求解器里的一次迭代做了什么

现在有了理论基础，来看 `Solver::Solve` 主循环（这是驱动 Clothoid 路径优化底层 iLQR 求解的核心函数）。每一次外层 `for` 循环迭代，做的事情严格对应"选方向 → 选步长"两大步：

```cpp
// iLQR 求解器主循环：每轮迭代 = 选方向（Backward Pass）+ 选步长（Forward Pass + 线搜索）
// 变量说明：x_trajectory, u_trajectory = 当前轨迹；meta = 迭代元信息（含正则化 mu、下降方向等）
//           alpha_values_ = 预计算的候选步长表 [1.0, 0.5, 0.25, 0.125, 0.0625, 0.031, 0.016, 0.008, 0.004, 0.001]
//           cost_history = 历史代价记录；max_mu_ = 正则化上限
for (int iter = 0; iter < max_num_iter_; ++iter) {

  // ① Backward Pass：计算下降方向 (k_trajectory, K_trajectory)，对应第2、10节
  bool is_backward_succeeded = false;
  while (meta.mu < max_mu_) {
    is_backward_succeeded = BackwardIteration(
        problem, x_trajectory, u_trajectory, action_limit, &meta);
    if (is_backward_succeeded) {
      meta.is_backward_succeeded = true;
      break;
    }
    IncreaseRegularization(&meta);  // 方向求解失败，加大正则化 mu 再试
  }

  // 梯度足够小，提前收敛退出
  meta.normalized_grad = ComputeNormalizedGrad(meta.k_trajectory, u_trajectory);
  if (meta.normalized_grad < min_grad_thresh_ && meta.mu < grad_exit_mu_thresh_) {
    solution.is_solved = true;
    break;
  }

  // ② Forward Pass + 回溯线搜索：在①算出的方向上，找一个合适的步长 alpha
  bool is_forward_succeeded = false;
  if (is_backward_succeeded) {
    for (double alpha : alpha_values_) {
      meta.alpha = alpha;
      // 用当前 alpha 做 Forward Pass，滚动出候选轨迹
      ForwardIteration(problem, meta, x_trajectory, u_trajectory, action_limit,
                       &x_trajectory_updated, &u_trajectory_updated);
      // 计算候选轨迹的真实总代价
      double cost_updated = ComputeCost(problem, x_trajectory_updated, u_trajectory_updated);
      meta.cost_improvement = cost_history.back() - cost_updated;               // 实际下降量
      meta.expected_cost_improvement = -meta.alpha * (meta.dV[0] + meta.alpha * meta.dV[1]); // 预期下降量
      // 计算"实际下降/预期下降"的比值（即 Armijo 充分下降条件的比值化实现）
      if (meta.expected_cost_improvement > min_expect_cost_improvement_threshold_) {
        meta.cost_improvement_ratio = meta.cost_improvement / meta.expected_cost_improvement;
      } else {
        meta.cost_improvement_ratio = (meta.cost_improvement > 0) ? 1.0 : -1.0;  // 预期下降量太小时的兜底
      }
      if (meta.cost_improvement_ratio > line_search_min_cost_improvement_ratio_) {
        is_forward_succeeded = true;
        break;   // 找到合适的 alpha，接受这次更新
      }
    }
  }

  // ③ 根据这一轮成功/失败，更新轨迹 + 调整正则化 mu
  if (is_forward_succeeded) {
    x_trajectory = std::move(x_trajectory_updated);
    u_trajectory = std::move(u_trajectory_updated);
    cost_history.push_back(ComputeCost(problem, x_trajectory, u_trajectory));  // 记录新代价
    ReduceRegularization(&meta);   // 成功了，放松一点正则化，下次更接近牛顿法
  } else {
    IncreaseRegularization(&meta); // 失败了，加大正则化，下次方向更保守
    if (meta.mu > max_mu_) { solution.is_solved = false; break; }
  }
}
```

一句话总结：**每轮迭代 = Backward Pass 决定"往哪走"（方向） + 回溯线搜索决定"走多远"（步长），走成功了就放松正则化，走失败了就收紧正则化重来**。

---

## 10. 逐段代码对应：Backward Pass 里的下降方向从哪来

对应第 2 节的"方向选择"。`BackwardIteration` 函数（实现的正是 Backward Pass 这一步）算出来的 $(k_i, K_i)$，就是这一轮的"下降方向"：

```cpp
// Backward Pass：从最后一个时间步往前递推，计算每一步的下降方向 (前馈项 d_k, 反馈增益 K_k)
// 变量说明：num_steps = 轨迹步数 N；V_xx = 下一时刻价值函数的 Hessian 近似（P_{k+1}）
//           l_x, l_u = 当前步代价的梯度；l_xx, l_uu, l_ux = 当前步代价的 Hessian 分块
//           f_x = 动力学雅可比 A_k, f_u = 动力学雅可比 B_k
//           MatUU = 控制维度×控制维度的矩阵类型（这里 U=1，所以是 1×1）
for (int i = num_steps - 1; i >= 0; --i) {
  // 从下一步的价值函数 V_{k+1} 和当前步的代价展开，计算 Q-function 的各个分块
  VecU Q_u  = l_u + f_u.transpose() * V_x;              // Q_u = ℓ_u + B^T p'
  VecX Q_x  = l_x + f_x.transpose() * V_x;              // Q_x = ℓ_x + A^T p'
  MatXU Q_ux = l_ux + f_u.transpose() * V_xx * f_x;     // Q_ux = ℓ_ux + B^T P' A
  MatUU Q_uu = l_uu + f_u.transpose() * V_xx * f_u;     // Q_uu = ℓ_uu + B^T P' B
  MatXX Q_xx = l_xx + f_x.transpose() * V_xx * f_x;     // Q_xx = ℓ_xx + A^T P' A

  // 在 Q_uu 上加正则化 mu*I，保证矩阵正定可逆（这就是第8.2节说的"控制条件数"）
  MatUU Q_uu_reg = Q_uu + meta->mu * MatUU::Identity();

  // 尝试 Cholesky 分解求逆（如果失败说明 Q_uu_reg 不正定，需要更大 mu）
  Eigen::LLT<MatUU> llt_of_Q_uu_reg(Q_uu_reg);
  if (llt_of_Q_uu_reg.info() == Eigen::NumericalIssue) {
    return false;  // 分解失败，外层会加大 mu 再试
  }
  MatUU Q_uu_reg_inv = MatUU(llt_of_Q_uu_reg.solve(MatUU::Identity()));

  // 求解前馈项 d_k 和反馈增益 K_k（即 p_k = -B_k^{-1} ∇f_k 的分解形式）
  meta->k_trajectory[i] = -Q_uu_reg_inv * Q_u;     // 前馈项 d_k = -(Q_uu+ρI)^{-1} Q_u
  meta->K_trajectory[i] = -Q_uu_reg_inv * Q_ux;    // 反馈增益 K_k = -(Q_uu+ρI)^{-1} Q_ux

  // 递推更新价值函数近似，供下一步（i-1）使用
  V_x = Q_x + meta->K_trajectory[i].transpose() * Q_uu * meta->k_trajectory[i]
            + meta->K_trajectory[i].transpose() * Q_u + Q_ux.transpose() * meta->k_trajectory[i];
  V_xx = Q_xx + meta->K_trajectory[i].transpose() * Q_uu * meta->K_trajectory[i]
              + meta->K_trajectory[i].transpose() * Q_ux
              + Q_ux.transpose() * meta->K_trajectory[i];

  // 记录预期代价下降量 ΔV(α) 的系数（供线搜索时计算 expected_cost_improvement）
  meta->dV[0] += meta->k_trajectory[i].transpose() * Q_u;                 // 一阶项 ∑α d_k^T Q_u
  meta->dV[1] += 0.5 * meta->k_trajectory[i].transpose() * Q_uu * meta->k_trajectory[i];  // 二阶项
}
```

- `Q_uu_reg = Q_uu + mu * I` 正是第 2.2 节里说的 $B_k$（这里 $B_k = Q_{uu}+\rho I$，对应第 8.2 节讨论的"矩阵条件数"）。
- `meta->k_trajectory[i] = -Q_uu_reg_inv * Q_u` 正是 $p_k = -B_k^{-1}\nabla f_k$ 的具体实现：`Q_u` 相当于这一步的局部梯度，`Q_uu_reg_inv` 相当于 $B_k^{-1}$。
- **只要 `mu`（即 $\rho$）取得足够大，`Q_uu_reg` 一定是正定的**（因为对角线加上足够大的正数，一定能压过原本可能存在的负特征值），这就保证了 $k_{\text{trajectory}}$ 一定是下降方向——这正是第 2.3 节"$B_k$ 正定 ⟹ 下降方向"的具体应用。

如果 `Q_uu_reg` 仍然不可逆（Cholesky 分解失败，代码里 `llt_of_Q_uu_reg.info() == Eigen::NumericalIssue`），说明当前 `mu` 还不够大，函数直接返回 `false`，外层 `while (meta.mu < max_mu_)` 循环就会调用 `IncreaseRegularization` 加大 `mu` 重新尝试——这是第 8.2 节"通过加大正则化改善条件数"的直接代码实现。

---

## 11. 逐段代码对应：Forward Pass 里的回溯线搜索

对应第 6 节"回溯法"。先看预先算好的候选步长表：

```cpp
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

这串数字是 $10^{\text{linspace}(0,-3,11)}$，也就是从 $10^0=1$ 均匀（对数尺度）递减到 $10^{-3}=0.001$，一共 11 个值——**本质上就是第 6 节回溯法里"每次乘以缩小因子 $\rho$"的思想，只是提前把整个衰减序列表打好了**，代码只需要顺序遍历，而不用每次都做乘法。这样做的好处是：不同量级的步长（1 到 0.001）能覆盖足够宽的搜索范围，同时又是提前算好的常量表，避免运行时反复计算 $\rho^n$ 的浮点误差累积。

### 11.1 Forward Pass：给定 $\alpha$，滚动出候选轨迹

```cpp
// Forward Pass：给定步长 alpha 和 Backward Pass 算出的 (d_k, K_k)，正向滚动出候选轨迹
// 变量说明：x_trajectory, u_trajectory = 旧轨迹（参考线）
//           context.alpha = 当前尝试的步长因子
//           context.k_trajectory[i] = 第 i 步的前馈修正 d_k
//           context.K_trajectory[i] = 第 i 步的反馈增益 K_k
void ForwardIteration(const Problem& problem, const IterationMeta& context,
                      const Trajectory& x_trajectory, const Trajectory& u_trajectory,
                      const ActionLimit& action_limit,
                      Trajectory* x_trajectory_updated,
                      Trajectory* u_trajectory_updated) const {
  (*x_trajectory_updated)[0] = x_trajectory[0];  // 起点状态固定不变
  const int num_steps = u_trajectory.size();
  for (int i = 0; i < num_steps; ++i) {
    // 更新公式：ū_k = u_k + K_k(x̄_k - x_k) + α * d_k
    //   其中 K_k(x̄_k - x_k) 是反馈项（根据实际状态偏差实时修正），α * d_k 是前馈项（步长缩放的固定修正）
    (*u_trajectory_updated)[i] =
        u_trajectory[i] + context.alpha * context.k_trajectory[i] +
        context.K_trajectory[i] * ((*x_trajectory_updated)[i] - x_trajectory[i]);
    // 控制量截断（物理限制，比如方向盘最大转速）
    (*u_trajectory_updated)[i] = ClampControl((*u_trajectory_updated)[i], action_limit);
    // 用真实非线性动力学模型正推到下一步状态（不使用线性近似，保证动力学约束严格满足）
    (*x_trajectory_updated)[i + 1] =
        problem.dynamic_model(i).Evaluate((*x_trajectory_updated)[i], (*u_trajectory_updated)[i]);
  }
}
```

这正是更新公式 $\bar u_k = u_k + K_k(\bar x_k-x_k) + \alpha d_k$ 的具体实现（这里 `k_trajectory` 就是 $d_k$，`K_trajectory` 就是 $K_k$）——**$\alpha$ 只缩放前馈项 $d_k$，不缩放反馈项**，这是标准 iLQR 线搜索的做法（反馈项负责实时修正状态偏差，不需要也不应该被步长衰减）。

### 11.2 判断这个 $\alpha$ 是否可接受：本质是 Armijo 条件的变体

```cpp
// 回溯线搜索：遍历候选步长表 alpha_values_，第一个满足 Armijo 充分下降条件的 alpha 就采用
// 变量说明：alpha_values_ = 预计算步长表 [1.0, 0.501, 0.251, 0.126, 0.063, 0.031, 0.016, 0.008, 0.004, 0.001]
//           meta.dV[0], meta.dV[1] = Backward Pass 算出的预期下降量系数
//           line_search_min_cost_improvement_ratio_ = Armijo 阈值（对应 c_1 的角色）
bool is_forward_succeeded = false;
for (double alpha : alpha_values_) {
  meta.alpha = alpha;
  // 用当前 alpha 做 Forward Pass，滚动出候选轨迹
  ForwardIteration(problem, meta, x_trajectory, u_trajectory, action_limit,
                   &x_trajectory_updated, &u_trajectory_updated);
  // 计算候选轨迹的真实总代价
  double cost_updated = ComputeCost(problem, x_trajectory_updated, u_trajectory_updated);
  // 实际下降量 = 旧代价 - 新代价
  meta.cost_improvement = cost_history.back() - cost_updated;
  // 预期下降量 = -ΔV(α) = -α * (dV[0] + α * dV[1])
  //   其中 dV[0] = ∑ d_k^T Q_u（一阶项），dV[1] = ½∑ d_k^T Q_uu d_k（二阶项）
  meta.expected_cost_improvement = -meta.alpha * (meta.dV[0] + meta.alpha * meta.dV[1]);
  // 计算"实际下降/预期下降"的比值（Armijo 充分下降条件的比值化实现）
  if (meta.expected_cost_improvement > min_expect_cost_improvement_threshold_) {
    meta.cost_improvement_ratio = meta.cost_improvement / meta.expected_cost_improvement;
  } else {
    // 预期下降量太小（接近 0）时直接看实际下降是否为正
    meta.cost_improvement_ratio = (meta.cost_improvement > 0) ? 1.0 : -1.0;
  }
  if (meta.cost_improvement_ratio > line_search_min_cost_improvement_ratio_) {
    is_forward_succeeded = true;
    break;   // 接受这个 alpha
  }
}
```

**逐项对应学城文档里的概念：**

| 代码变量 | 数学含义 |
|---|---|
| `cost_history.back()` | $f(x_k)$，走这一步之前的代价 |
| `cost_updated` | $f(x_k+\alpha p_k)$，走这一步之后的代价 |
| `meta.cost_improvement` | 实际下降量 $f(x_k) - f(x_k+\alpha p_k)$ |
| `meta.dV[0] + alpha*meta.dV[1]` | 由 Backward Pass 预测出的、该步长下"理论上应该下降多少"（对应公式 $\Delta V(\alpha)=\sum\alpha d_k^TQ_u+\frac12\alpha^2d_k^TQ_{uu}d_k$） |
| `meta.expected_cost_improvement` | 预期下降量 $-\Delta V(\alpha)$ |
| `meta.cost_improvement_ratio` | 就是学城 AL-iLQR 文档里的 $z=\dfrac{\text{实际下降}}{\text{预期下降}}$（本质上是 Wolfe/Armijo 条件的"比值"版本：既要求下降（分子>0），也隐含约束下降不能偏离预期太多） |
| `line_search_min_cost_improvement_ratio_` | 判定阈值，$z$ 超过它才接受这个 $\alpha$（对应 Armijo 条件里的常数 $c_1$ 扮演的角色——"至少要下降到预期的多少比例才算数"） |

也就是说，`cost_improvement_ratio > line_search_min_cost_improvement_ratio_` 这一行代码，本质上就是学城文档第 4.1 节 **Armijo 充分下降条件的比值化实现**：不是简单要求"下降"，而是要求"实际下降量占预期下降量的比例，超过某个阈值"，这正是第 3.2 节讲的"充分下降条件"，用来避免"下降 5/k 那种下降太慢导致不收敛"的病态情况。

如果所有候选 $\alpha$ 都不满足条件（`alpha_values_` 遍历完还没 break，或触发 `alpha < min_alpha_` 提前退出），`is_forward_succeeded` 保持 `false`，说明这次线搜索**彻底失败**——回到主循环，会执行 `IncreaseRegularization`，加大 `mu`，下一轮重新做一次更保守的 Backward Pass。

> **注**：这里没有单独实现"曲率条件"（4.2 节），因为在信赖域/正则化风格的 iLQR 里，"步长过小导致效率低"这个问题是通过**动态调整正则化 `mu`**（而不是要求曲率条件）来解决的——`mu` 变小时相当于允许更大的牛顿步（$\alpha=1$ 时更容易被接受），本质上起到了类似"避免步子一直很小"的效果。这是工程实现上对理论的一种等价简化。

---

## 12. 逐段代码对应：正则化 mu 如何充当"信赖域"

```cpp
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

结合第 8.2 节的理论：
- **`mu` 越大** → `Q_uu_reg = Q_uu + mu*I` 越接近"单位矩阵的倍数" → 方向 $p_k=-Q_{uu,reg}^{-1}Q_u$ 越接近**最速下降方向**（第 2.2 节里 $B_k=I$ 的情形）。最速下降方向虽然收敛慢，但**极其稳健**，几乎总是下降方向，适合在离最优点很远、非线性很强的地方使用。
- **`mu` 越小（趋于 0）** → `Q_uu_reg` 越接近真实的 `Q_uu`（类似 Hessian）→ 方向越接近**纯牛顿方向**，收敛快，但要求当前点离最优点足够近、局部近似足够准，否则容易"迈过头"导致线搜索失败。

**这就是所谓的"自适应正则化"策略**：线搜索连续失败 → 说明局部近似不准 → 增大 `mu`，退化成更保守的方向，重新试；线搜索成功 → 说明这个区域近似还不错 → 减小 `mu`，下次尝试更接近牛顿法的"激进"步子，加快收敛。这本质上和数值优化里经典的**信赖域方法（trust region method）**思想是相通的：`mu` 越大等价于信赖域越小（只信任局部很小范围内的二次近似）。

---

## 13. 完整迭代流程图

```
外层 for 循环（最多 max_num_iter_ 次）
  │
  ├─① Backward Pass（选方向，第10节）
  │    │
  │    ├─ 计算 Q_uu_reg = Q_uu + mu*I
  │    ├─ 尝试 Cholesky 分解求逆
  │    │    ├─ 失败（不正定）→ IncreaseRegularization(mu 变大) → 重新算
  │    │    └─ 成功 → 得到 k_trajectory(d_k), K_trajectory(K_k)
  │    └─ 得到本轮的下降方向 p_k = (k, K)
  │
  ├─ 检查梯度是否已经足够小 → 是则收敛退出
  │
  ├─② Forward Pass + 回溯线搜索（选步长，第11节）
  │    │
  │    └─ 遍历 alpha_values_ = [1, 0.5, 0.25, 0.125, 0.0625, 0.031, 0.016, 0.008, 0.004, 0.001]
  │         ├─ 用当前 alpha 做 ForwardIteration，得到候选轨迹
  │         ├─ 计算 cost_improvement_ratio = 实际下降/预期下降
  │         ├─ 满足阈值 → 接受这个 alpha，break（对应 Armijo 条件）
  │         └─ 不满足 → 换下一个更小的 alpha，继续试
  │
  ├─③ 根据 Forward Pass 是否成功，更新状态
  │    ├─ 成功 → 采用新轨迹，ReduceRegularization(mu 变小，下次更激进)
  │    └─ 失败 → IncreaseRegularization(mu 变大，下次更保守)，mu 超上限则放弃
  │
  └─ 判断是否收敛（cost_improvement 是否足够小）→ 是则退出，否则回到①
```

---

## 14. 常见疑问 FAQ

**Q1：为什么不直接用理论上最标准的 Wolfe 条件（同时检查 Armijo + 曲率条件），而是只用一个"下降比值"？**

工程实现上，`cost_improvement_ratio`（实际下降/预期下降）这个设计已经隐含了 Armijo 条件的核心精神——要求下降幅度达到预期的某个比例。而曲率条件本来是为了防止"步长太小、效率低"，但在这套 AL-iLQR 框架里，"步子大小"的合理性是通过**外层自适应调整 `mu`（正则化）**来保证的：`mu` 小的时候，$\alpha=1$（最大候选步长）通常就能被接受，天然避免了"每次都用很小步长"的低效问题。这是理论条件在具体工程框架下的一种简化和等价替代，属于常见的工程实践。

**Q2：`alpha_values_` 为什么是等比数列（对数均匀），而不是等差数列？**

如果用等差数列（比如 $1, 0.9, 0.8,\dots$），在需要非常小步长的场景（比如高度非线性、离最优点很远时）就覆盖不到（数列很快就到 0 但中间没有足够小的值）。用对数均匀分布（等比数列）能让候选步长在"大步长"和"小步长"两个数量级都有足够密的采样，兼顾了大范围搜索的效率和小范围搜索的精度，这是数值优化里选取候选步长表的常见技巧。

**Q3：如果 `alpha_values_` 里所有步长都不满足条件，会发生什么？**

对应实现里 `is_forward_succeeded` 保持 `false`，回到主循环会执行 `IncreaseRegularization`，把 `mu` 调大后回到 Backward Pass 重新计算一个更保守的方向，再重新做一遍线搜索。如果 `mu` 最终超过了 `max_mu_` 还是不行，则认为这个问题在当前设置下"无法进一步优化"，`solution.is_solved = false`，求解失败，上层调用方通常会走 Fallback 逻辑（例如复用上一帧路径等）。

**Q4：`min_alpha_` 这个配置项是干什么的？**

它给线搜索的步长设了一个"下限"，如果连 `min_alpha_` 都不满足下降条件，就直接放弃这一轮的线搜索（不再尝试更小的 $\alpha$），提前退出遍历，避免浪费计算资源在"注定要失败、极小步长依然不能改善"的场景上，配合外层 `IncreaseRegularization` 更快地切换到保守方向重来。

**Q5：这里的"下降方向"$p_k$跟 Zoutendijk 定理里要求的"$\cos\theta_k$不能太小"是怎么保证的？**

对应第 8.2 节：只要 `mu` 保持在合理范围内，`Q_uu_reg` 的条件数就不会失控（正则化本身就是为了控制条件数），从而保证方向 $p_k$ 和负梯度的夹角不会退化到接近 90°。这也是为什么正则化策略（第 12 节）在这个算法里如此重要——它不仅仅是"防止矩阵不可逆"的数值技巧，更是从理论上保证算法**全局收敛性**的关键环节。

---

## 附：核心公式速查表

| 概念 | 公式 |
|---|---|
| 迭代更新 | $x_{k+1}=x_k+\alpha_kp_k$ |
| 下降方向条件 | $p_k^T\nabla f_k<0$ |
| 常见方向 | $p_k=-B_k^{-1}\nabla f_k$ |
| Armijo 条件 | $f(x_k+\alpha_kp_k)\le f(x_k)+c_1\alpha_k\nabla f_k^Tp_k$ |
| 曲率条件 | $\nabla f(x_k+\alpha_kp_k)^Tp_k\ge c_2\nabla f_k^Tp_k$ |
| Strong Wolfe | $\|\nabla f(x_k+\alpha_kp_k)^Tp_k\|\le c_2\|\nabla f_k^Tp_k\|$ |
| Goldstein 条件 | $f(x_k)+(1-c)\alpha_k\nabla f_k^Tp_k \le f(x_k+\alpha_kp_k)\le f(x_k)+c\alpha_k\nabla f_k^Tp_k$ |
| 回溯法 | $\alpha\leftarrow\rho\alpha$ 直到满足 Armijo 条件 |
| 方向与负梯度夹角 | $\cos\theta_k=\dfrac{-\nabla f_k^Tp_k}{\|\nabla f_k\|\|p_k\|}$ |
| Zoutendijk 条件 | $\sum_{k=0}^\infty\cos^2\theta_k\|\nabla f_k\|^2<\infty$ |
| 全局收敛结论 | $\cos\theta_k\ge\delta>0 \Rightarrow \|\nabla f_k\|\to0$ |
| 牛顿类方法收敛条件 | $B_k\succ0$ 且 $\|B_k\|\|B_k^{-1}\|\le M \Rightarrow \cos\theta_k\ge1/M$ |
| 代码中的下降比值 | `cost_improvement_ratio = 实际下降 / 预期下降` |

---

## 建议学习路径

1. 先读第 1~8 节，把"方向"和"步长"这两个独立的概念、以及 Wolfe 条件的三种变体理解清楚（不用死记公式，记住"为什么需要它"）。
2. 再读第 9~13 节，把每一段实现逻辑和前面的理论概念连起来看，理解"选方向 → 选步长 → 调整正则化"这一整套迭代机制。
3. 结合第 13 节的完整迭代流程图，在脑海中模拟一遍 `alpha`、`cost_improvement_ratio`、`mu` 等关键量在迭代过程中是如何变化的，会对"正则化收紧/放松、线搜索接受/拒绝"这套机制有更直观的体感。
4. 如果还想深入，可以结合第 2 部分"下降方向"的推导，理解 Backward Pass 里 $Q_{uu}, Q_{ux}, Q_u$ 是怎么算出来的，从而更完整地打通"从约束、代价函数，到方向、步长，再到最终路径"的全链路。
