#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# ================================================================================
# 文件概述：CollisionNonlinearOptimizer —— 基于非线性优化的碰撞避免轨迹平滑器
#
# 功能：
#   在测试阶段（inference），对规划头预测的初始轨迹进行后处理优化。
#   核心思想：在"尽量不偏离原始轨迹"和"避开占用预测中的障碍物"之间找平衡。
#
# 使用的优化框架：CasADi（开源数值优化库，支持自动微分和非线性规划）
#   - Opti(): 创建优化问题实例
#   - IPOPT:  内点法求解器（Interior Point OPTimizer），适合非线性优化
#
# 优化方法：直接多重打靶法（Direct Multiple Shooting）
#   - 将轨迹的每个时间步作为独立的决策变量
#   - 同时优化所有时间步，比逐步优化更全局
#
# 目标函数（最小化）：
#   min  alpha_xy × ||预测轨迹 - 参考轨迹||²          ← 跟踪损失（不要偏太远）
#      + alpha_collision × Σ 高斯碰撞代价(t)           ← 碰撞代价（远离障碍物）
#
# 改编自：nuplan-devkit（Motional公司的自动驾驶开发工具包）
# ================================================================================

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt  # numpy 的类型注解工具

# CasADi 数值优化库的核心组件
from casadi import (
    DM,       # Dense Matrix：CasADi 的密集矩阵类型（类似 numpy array）
    Opti,     # 优化问题构建器（声明变量、参数、约束、目标函数）
    OptiSol,  # 优化求解结果对象（从中提取最优解）
    cos,      # CasADi 的余弦函数（支持自动微分）
    diff,     # CasADi 的差分函数
    sin,      # CasADi 的正弦函数
    sumsqr,   # 所有元素的平方和（sum of squares）
    vertcat,  # 垂直拼接矩阵（vertical concatenate）
    exp,      # 自然指数函数 e^x（用于高斯碰撞代价）
)

# 类型别名：Pose = (x坐标, y坐标, 朝向角yaw)，描述一个2D位姿
Pose = Tuple[float, float, float]  # (x, y, yaw)


class CollisionNonlinearOptimizer:
    """基于非线性规划的轨迹碰撞避免优化器。

    工作流程：
    1. 接收规划头预测的初始轨迹（参考轨迹）
    2. 接收 OccFlow 预测的障碍物位置（各时间步的障碍物坐标列表）
    3. 建立优化问题：以轨迹的 (x, y) 坐标为决策变量
    4. 目标函数 = 与参考轨迹的偏差 + 对障碍物的高斯排斥代价
    5. 用 IPOPT 求解，得到在安全性和准确性之间平衡的最优轨迹

    碰撞代价的直觉理解（高斯排斥场）：
        每个障碍物在周围形成一个"排斥力场"，距离越近代价越高：
        cost = alpha × (1/2.507σ) × exp(-dist² / (2σ²))
        这是一个高斯函数，障碍物中心代价最大，向外指数衰减。

    Optimize planned trajectory with predicted occupancy
    Solved with direct multiple-shooting.
    modified from https://github.com/motional/nuplan-devkit
    :param trajectory_len: trajectory length
    :param dt: timestep (sec)
    """

    def __init__(self, trajectory_len: int, dt: float, sigma, alpha_collision, obj_pixel_pos):
        """初始化优化器，构建优化问题的结构（变量、参数、目标函数）。

        注意：此函数只定义优化问题的"结构"，不包含具体数值。
        具体数值（参考轨迹）在 set_reference_trajectory() 中设置。

        Args:
            trajectory_len (int): 轨迹长度（时间步数），对应 planning_steps=6
            dt (float): 相邻时间步的时间间隔（秒），UniAD 中为 0.5s
            sigma (float): 高斯碰撞代价的标准差，控制排斥力场的扩散范围
                           sigma 越大，障碍物的"影响范围"越大（越保守）
            alpha_collision (float): 碰撞代价的权重系数（越大越保守，越不靠近障碍物）
            obj_pixel_pos (list[list]): 各时间步的障碍物坐标列表
                obj_pixel_pos[t] = [(x₁,y₁), (x₂,y₂), ...]，ego坐标系，单位米
                由 planning_head.py 的 collision_optimization() 函数准备好传入
        """
        self.dt = dt                              # 时间步长（0.5s）
        self.trajectory_len = trajectory_len      # 轨迹步数（6步）
        self.current_index = 0                    # 当前时间步索引（备用）
        self.sigma = sigma                        # 高斯标准差
        self.alpha_collision = alpha_collision    # 碰撞代价权重
        self.obj_pixel_pos = obj_pixel_pos        # 障碍物坐标列表

        # 构建各时间步的时间间隔数组（目前所有步均相同=dt，留出扩展接口）
        # Use a array of dts to make it compatible to situations with varying dts across different time steps.
        self._dts: npt.NDArray[np.float32] = np.asarray([[dt] * trajectory_len])
        # shape: (1, trajectory_len)，例如 [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]

        # 初始化优化问题（声明变量、参数、目标函数）
        self._init_optimization()

    def _init_optimization(self) -> None:
        """初始化 CasADi 优化问题的各个组成部分。

        CasADi 的优化问题构建步骤：
        1. 创建 Opti() 实例（优化问题容器）
        2. 声明决策变量（要求解的未知量，即轨迹坐标）
        3. 声明参数（已知但可变的量，即参考轨迹）
        4. 设置目标函数（最小化的表达式）
        5. 配置求解器选项
        """
        self.nx = 2  # 状态维度：每个时间步只优化 (x, y) 两个坐标（不含朝向）

        self._optimizer = Opti()  # 创建 CasADi 优化问题实例

        # 声明决策变量（优化器要找的最优解）
        self._create_decision_variables()

        # 声明参数（优化器需要知道的外部输入，但不是要求解的）
        self._create_parameters()

        # 设置目标函数（告诉优化器"什么是好的轨迹"）
        self._set_objective()

        # 配置 IPOPT 求解器选项（静默模式，不打印中间过程）
        # "ipopt.print_level": 0  → 不打印迭代日志
        # "print_time": 0         → 不打印求解时间
        # "ipopt.sb": "yes"       → 抑制启动横幅（suppress banner）
        self._optimizer.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})

    def set_reference_trajectory(self, reference_trajectory: Sequence[Pose]) -> None:
        """设置参考轨迹（即规划头输出的初始轨迹）并初始化求解器的初始猜测。

        在 planning_head.py 中，每次推理前调用此函数，传入当前帧的规划轨迹。
        优化器会在这条轨迹附近搜索更安全的替代轨迹。

        Args:
            reference_trajectory (Sequence[Pose]): 参考轨迹，shape (planning_steps, 3)
                每行为 (x, y, yaw)，但优化器只使用前两维 (x, y)
                坐标为 ego 坐标系，单位米

        注意：
            DM(reference_trajectory).T 将 (6, 3) 的 numpy 数组转置为 (3, 6)，
            但参数 ref_traj 定义为 (2, 6)，所以只取前两行 (x, y)。
        """
        # 将参考轨迹数值赋给 CasADi 参数（触发优化器知道"目标轨迹"在哪）
        self._optimizer.set_value(self.ref_traj, DM(reference_trajectory).T)
        # DM(...).T: 将 numpy 数组转为 CasADi DM 矩阵，并转置为列向量格式

        # 用参考轨迹初始化决策变量的初始猜测（warm start）
        # warm start 的作用：给优化器一个好的起点，加速收敛
        self._set_initial_guess(reference_trajectory)

    def set_solver_optimizerons(self, options: Dict[str, Any]) -> None:
        """覆盖默认的求解器配置（如需要更详细的日志或调整精度时使用）。

        Args:
            options (dict): IPOPT 求解器的配置字典，例如：
                {'ipopt.print_level': 5}  → 打印详细迭代信息（调试时有用）
                {'ipopt.tol': 1e-6}       → 设置收敛容差
        """
        self._optimizer.solver("ipopt", options)

    def solve(self) -> OptiSol:
        """执行优化求解，返回最优解对象。

        调用此函数前必须先调用 set_reference_trajectory() 设置参考轨迹。

        Returns:
            OptiSol: CasADi 的求解结果对象。
                通过 sol.value(变量名) 提取最优值，例如：
                - sol.value(optimizer.position_x) → 最优 x 坐标序列（长度6的数组）
                - sol.value(optimizer.position_y) → 最优 y 坐标序列（长度6的数组）

        :return Casadi optimization class
        """
        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
        """声明优化问题的决策变量（即"要求解的未知量"）。

        决策变量就是优化器会调整的量，通过不断调整这些值来最小化目标函数。
        这里的决策变量是轨迹的 (x, y) 坐标序列。

        变量形状：(nx=2, trajectory_len=6)，即：
            [[x₁, x₂, x₃, x₄, x₅, x₆],
             [y₁, y₂, y₃, y₄, y₅, y₆]]
        优化器要找到最优的 x₁~x₆ 和 y₁~y₆。
        """
        # 声明状态变量矩阵，形状 (2, 6)
        # 第0行 = x坐标序列，第1行 = y坐标序列
        self.state = self._optimizer.variable(self.nx, self.trajectory_len)

        # 提取行视图，方便后续引用
        self.position_x = self.state[0, :]  # x坐标序列，shape (1, 6)
        self.position_y = self.state[1, :]  # y坐标序列，shape (1, 6)

    def _create_parameters(self) -> None:
        """声明优化问题的参数（"已知的外部输入，但不是要求解的"）。

        参数与决策变量的区别：
        - 决策变量：优化器主动调整的未知量（轨迹坐标）
        - 参数：外部给定的已知量（参考轨迹），每次推理可以更新值

        这里只有一个参数：参考轨迹（规划头的初始预测轨迹）。

        参数形状：(2, trajectory_len) = (2, 6)，即：
            [[x_ref₁, x_ref₂, ..., x_ref₆],
             [y_ref₁, y_ref₂, ..., y_ref₆]]
        """
        # 声明参考轨迹参数，只使用 (x, y) 两维（不含朝向 yaw）
        self.ref_traj = self._optimizer.parameter(2, self.trajectory_len)  # (x, y)

    def _set_objective(self) -> None:
        """定义优化目标函数（最小化的表达式）。

        目标函数由两项组成，体现了两个相互竞争的目标：

        ① 跟踪损失（cost_stage）：最小化与参考轨迹的偏差
            cost_stage = alpha_xy × Σ[(x_ref_t - x_t)² + (y_ref_t - y_t)²]
            = alpha_xy × ||ref_traj - predicted_traj||²（Frobenius范数的平方）
            → 防止优化后的轨迹偏离原始规划太远

        ② 碰撞代价（cost_collision）：对每个障碍物施加高斯排斥力
            对于时间步 t 的每个障碍物位置 (col_x, col_y)：
            cost += alpha × normalizer × exp(-[(x_t-col_x)² + (y_t-col_y)²] / (2σ²))
            这是一个以障碍物为中心的负高斯函数，越靠近障碍物代价越高。

        直觉理解：
            高斯排斥代价就像在每个障碍物周围放一座"小山"，
            优化器会自动绕开这些"小山"，同时尽量不偏离参考路径太远。

        Set the objective function. Use care when modifying these weights.
        """
        # ---- 跟踪损失 ----
        alpha_xy = 1.0  # 跟踪权重（固定为1.0，碰撞权重由 alpha_collision 控制）

        # sumsqr: CasADi 的平方和函数（等价于 ||·||²_F）
        # vertcat(position_x, position_y): 将 x 和 y 序列垂直拼接为 (2, 6) 矩阵
        cost_stage = (
            alpha_xy * sumsqr(
                self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y)
            )
        )
        # cost_stage = Σ_t [(x_ref_t - x_t)² + (y_ref_t - y_t)²]

        # ---- 碰撞代价 ----
        alpha_collision = self.alpha_collision  # 碰撞惩罚权重（配置文件中设为5.0）

        cost_collision = 0

        # 高斯归一化系数：1/(2.507×σ)
        # 2.507 ≈ √(2π)，使高斯函数在积分后近似为1（概率归一化）
        # 这个系数让不同 sigma 下的碰撞代价量级大致可比
        normalizer = 1 / (2.507 * self.sigma)

        # 遍历每个时间步的每个障碍物，累加碰撞代价
        # TODO: vectorize this（当前是循环实现，后续可向量化提速）
        for t in range(len(self.obj_pixel_pos)):
            # 取第 t 步的决策变量（CasADi 符号变量，非具体数值）
            x, y = self.position_x[t], self.position_y[t]

            for i in range(len(self.obj_pixel_pos[t])):
                # 第 t 步第 i 个障碍物的坐标（ego坐标系，单位米）
                col_x, col_y = self.obj_pixel_pos[t][i]

                # 高斯碰撞代价：
                #   dist² = (x - col_x)² + (y - col_y)²  （欧式距离的平方）
                #   cost  = alpha × normalizer × exp(-dist² / (2σ²))
                # 当 dist=0（刚好在障碍物上）时代价最大 = alpha × normalizer
                # 当 dist=σ  时代价降为最大值的 e^(-0.5) ≈ 60%
                # 当 dist=2σ 时代价降为最大值的 e^(-2)  ≈ 13%
                cost_collision += alpha_collision * normalizer * exp(
                    -((x - col_x)**2 + (y - col_y)**2) / 2 / self.sigma**2
                )

        # 设置最终目标函数（两项相加，优化器将最小化此表达式）
        self._optimizer.minimize(cost_stage + cost_collision)

    def _set_initial_guess(self, reference_trajectory: Sequence[Pose]) -> None:
        """为优化器设置决策变量的初始猜测值（warm start）。

        为什么需要初始猜测？
        非线性优化（如 IPOPT）是迭代求解的，需要一个起始点。
        - 好的起始点：靠近最优解，收敛快，不容易陷入局部最优
        - 差的起始点（如全零）：可能收敛慢或得到次优解

        这里用参考轨迹（规划头的原始预测）作为起点，直觉上：
        参考轨迹已经是一个"不错"的轨迹，优化只是在它附近微调。

        Set a warm-start for the solver based on the reference trajectory.

        Args:
            reference_trajectory (Sequence[Pose]): 参考轨迹 (planning_steps, 3)
                格式同 set_reference_trajectory()
        """
        # 将参考轨迹的前两维 (x, y) 作为 state 变量的初始值
        # DM(reference_trajectory).T: (6, 3) → 转置 → (3, 6)，取前两行 → (2, 6)
        self._optimizer.set_initial(
            self.state[:2, :],              # 对应 position_x 和 position_y
            DM(reference_trajectory).T      # 参考轨迹的 (x, y, yaw) 转置后取前两行
        )  # (x, y, yaw)