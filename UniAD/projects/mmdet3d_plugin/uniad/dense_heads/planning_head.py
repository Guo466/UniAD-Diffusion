#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# ================================================================================
# 文件概述：PlanningHeadSingleMode —— UniAD 的端到端轨迹规划头
#
# 功能：
#   给定 BEV 特征图、运动预测结果（SDC 轨迹 query + 跟踪 query）和驾驶命令
#   （直行/左转/右转），预测自车（SDC, Self-Driving Car）未来 6 步（3 秒）的
#   行驶轨迹，并在测试时可选地使用占用预测结果（OccFlow）做碰撞避免优化。
#
# 主要流程（forward）：
#   1. 融合三路信息：轨迹 query + 跟踪 query + 导航命令嵌入  →  plan_query (1×256)
#   2. plan_query 对 BEV 特征图做 Transformer 交叉注意力
#   3. 回归分支输出 6 步位移（累积求和转为绝对坐标）
#   4. 测试时用 CollisionNonlinearOptimizer 做碰撞避免后处理
#
# 损失函数：
#   - loss_planning：L2/ADE 轨迹误差
#   - loss_collision：基于未来 GT 框的碰撞惩罚（三个安全边界 delta=0/0.5/1.0m）
# ================================================================================

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from einops import rearrange  # 张量维度重排工具（比 view/permute 更直观）
from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation
# bivariate_gaussian_activation: 将网络输出解码为二元高斯分布参数（均值+协方差）
from .planning_head_plugin import CollisionNonlinearOptimizer
# CollisionNonlinearOptimizer: 非线性碰撞避免优化器（基于 CasADi 的轨迹优化）
import numpy as np
import copy


@HEADS.register_module()  # 注册到 mmdet 的 HEADS 注册表，配置文件中用 type='PlanningHeadSingleMode' 调用
class PlanningHeadSingleMode(nn.Module):
    """单模态规划头：预测自车（SDC）未来 planning_steps 步的行驶轨迹。

    "单模态"是指直接输出一条确定性轨迹（与多模态方法相比，不预测多条候选轨迹）。
    轨迹规划是 UniAD 的最终目标任务，它整合了检测、跟踪、地图、运动预测、占用
    预测等所有上游任务的信息。

    核心信息融合方式：
        - sdc_traj_query: 运动预测头为 SDC 生成的轨迹查询（包含 SDC 的运动信息）
        - sdc_track_query: 跟踪头维护的 SDC 跟踪查询（包含 SDC 的位置/外观信息）
        - navi_embed:      导航命令的可学习嵌入（0=右转, 1=直行, 2=左转）
        → 三路信息拼接后经 MLP 融合为单个 plan_query
        → plan_query 对 BEV 特征图做注意力，获取环境信息
        → 回归头输出未来轨迹坐标
    """

    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=6,
                 loss_planning=None,
                 loss_collision=None,
                 loss_kinematic=None,       # 运动学可行性损失配置（新增）
                 planning_eval=False,
                 use_col_optim=False,
                 col_optim_args=dict(
                    occ_filter_range=5.0,   # 碰撞优化的感知范围（米）
                    sigma=1.0,              # 高斯碰撞代价的标准差
                    alpha_collision=5.0,    # 碰撞代价的缩放系数
                 ),
                 with_adapter=False,
                ):
        """
        初始化规划头的所有网络模块和损失函数。

        Args:
            bev_h (int): BEV特征图高度，默认200（对应100m范围，0.5m分辨率）
            bev_w (int): BEV特征图宽度，默认200
            embed_dims (int): 特征嵌入维度，默认256（与BEVFormer主干保持一致）
            planning_steps (int): 规划步数，默认6步（每步0.5s，共3秒）
            loss_planning (dict): 规划轨迹损失配置（如 PlanningLoss，计算ADE）
            loss_collision (list[dict]): 碰撞损失配置列表（三个不同安全边界的CollisionLoss）
            planning_eval (bool): 是否在训练过程中同步计算规划评估指标
            use_col_optim (bool): 测试时是否启用碰撞避免后处理优化器
            col_optim_args (dict): 碰撞优化器的超参数
                - occ_filter_range: 只考虑距自车预测位置5m以内的障碍物（减少噪声）
                - sigma: 碰撞高斯代价的扩散范围
                - alpha_collision: 碰撞代价权重
            with_adapter (bool): 是否使用BEV Adapter（轻量级卷积适配层）
                - True 时插入一个 Conv2d→ReLU→Conv2d 的残差块，
                  用于在 Stage2 微调时快速适配 BEV 特征分布
        """
        super(PlanningHeadSingleMode, self).__init__()

        # ---- BEV 特征图尺寸（用于坐标转换）----
        self.bev_h = bev_h   # 200
        self.bev_w = bev_w   # 200

        # ---- 导航命令嵌入（可学习的 Embedding）----
        # NuScenes 有 3 种驾驶命令：右转(0)、直行(1)、左转(2)
        # 每个命令映射为一个 embed_dims 维的可学习向量
        self.navi_embed = nn.Embedding(3, embed_dims)

        # ---- 轨迹回归分支 ----
        # 输入: plan_query (embed_dims=256)
        # 输出: planning_steps*2 = 12 个数值（每步 x, y 位移）
        # 网络结构: Linear → ReLU → Linear（两层MLP）
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2),
        )

        # ---- 规划损失（ADE: Average Displacement Error）----
        self.loss_planning = build_loss(loss_planning)
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval  # 是否在训练时评估

        # ---- 运动学可行性损失（新增）----
        # 约束预测轨迹满足车辆物理限制（加速度平滑 + jerk 平滑 + FDE 终点精度）
        # 当配置文件未指定 loss_kinematic 时，self.loss_kinematic 为 None，损失计算跳过
        if loss_kinematic is not None:
            self.loss_kinematic = build_loss(loss_kinematic)
        else:
            self.loss_kinematic = None

        # ================================================================
        # 规划查询融合模块
        # 将三路信息（轨迹query + 跟踪query + 导航嵌入）融合为单个 plan_query
        # ================================================================

        # fuser_dim=3: 三路输入的拼接倍数
        # 拼接后维度 = embed_dims * 3 = 256*3 = 768
        fuser_dim = 3

        # Transformer Decoder 层（plan_query 作为 target，BEV 特征作为 memory）
        # - num_heads=8: 多头注意力头数
        # - dim_feedforward=512: FFN 中间层维度
        # - dropout=0.1: Dropout 率
        # - batch_first=False: 序列维度在第0维（PyTorch 默认格式）
        attn_module_layer = nn.TransformerDecoderLayer(
            embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False
        )
        # 3层 Transformer Decoder（plan_query 迭代查询 BEV 特征 3 次）
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)

        # 三路信息的 MLP 融合器
        # 输入: 768（3×256）→ 输出: 256
        # 包含 LayerNorm（稳定训练）+ ReLU
        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*fuser_dim, embed_dims),  # 768 → 256
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )

        # plan_query 的可学习位置编码（1 个位置，embed_dims 维）
        # 加在融合后的 plan_query 上，告知 Transformer "这是规划 query"
        self.pos_embed = nn.Embedding(1, embed_dims)

        # ---- 碰撞损失（支持多个不同安全边界）----
        # loss_collision 是一个列表，包含多个 CollisionLoss 配置
        # 每个 CollisionLoss 对应不同的 delta（障碍物膨胀边界）
        # 例如：delta=0（严格碰撞）、delta=0.5m、delta=1.0m
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        # 转为 nn.ModuleList，使参数被正确注册和管理
        self.loss_collision = nn.ModuleList(self.loss_collision)

        # ---- 碰撞避免优化器参数 ----
        self.use_col_optim = use_col_optim                         # 是否启用碰撞优化（测试时）
        self.occ_filter_range = col_optim_args['occ_filter_range'] # 障碍物感知范围（5m）
        self.sigma = col_optim_args['sigma']                       # 碰撞代价高斯扩散半径
        self.alpha_collision = col_optim_args['alpha_collision']   # 碰撞代价权重（5.0）

        # ================================================================
        # BEV Adapter（可选）
        # 用于 Stage2 微调时适配 BEV 特征分布的轻量级卷积残差模块
        # Stage2 冻结了 BEV Encoder，但 BEV 特征的分布对规划头来说可能不理想，
        # BEV Adapter 通过少量卷积参数对特征进行自适应变换
        # ================================================================
        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            # 单个 Adapter Block: Conv2d(256→128, 3×3) → ReLU → Conv2d(128→256, 1×1)
            # 先降维再升维（类似 Bottleneck），减少参数量
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),  # 降维: 256→128
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),             # 升维: 128→256
            )
            N_Blocks = 3  # 堆叠 3 个 Adapter Block（提升适配能力）
            # 深拷贝确保每个 block 有独立的参数（不共享权重）
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)

    def forward_train(self,
                      bev_embed,
                      outs_motion={},
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      ):
        """训练阶段的前向传播入口。

        训练时不使用 occ_mask（occ_mask=None），因为训练时 OccFlow 头的输出
        不稳定，直接使用可能引入噪声。碰撞损失用 GT 未来框（gt_future_boxes）代替。

        Args:
            bev_embed (torch.Tensor): BEV 特征图，形状 (H*W, B, C) = (40000, 1, 256)
                由 BEVFormer Encoder 生成，包含当前帧的全局鸟瞰图特征
            outs_motion (dict): 运动预测头的输出，包含：
                - 'sdc_traj_query': SDC 轨迹查询 (num_layers, B, P, C)
                  运动预测头为 SDC 生成的多模态轨迹特征（最后一层用于规划）
                - 'sdc_track_query': SDC 跟踪查询 (B, C)
                  跟踪头维护的 SDC 状态向量
                - 'bev_pos': BEV 位置编码 (B, C, H, W)
            sdc_planning (torch.Tensor): 自车规划轨迹 GT (B, 1, planning_steps, 3)
                最后一维为 [x, y, heading]（位置+朝向）
            sdc_planning_mask (torch.Tensor): 规划轨迹有效掩码 (B, 1, planning_steps, 1)
                False=无效步（场景边界/越界），损失计算时忽略
            command (torch.Tensor): 驾驶命令 (B,)，值为 0/1/2（右转/直行/左转）
            gt_future_boxes (list): 未来各帧的 GT 目标框（用于碰撞损失计算）

        Returns:
            dict: 包含：
                - 'losses': 损失字典（loss_ade + loss_collision_0/1/2）
                - 'outs_motion': 规划输出（与 forward() 返回值相同）
        """
        # 从运动预测输出中提取 SDC 相关的查询和位置编码
        sdc_traj_query = outs_motion['sdc_traj_query']   # (num_layers, B, P, C)
        sdc_track_query = outs_motion['sdc_track_query'] # (B, C)
        bev_pos = outs_motion['bev_pos']                 # (B, C, H, W)

        # 训练时不使用 OccFlow mask（设为 None），避免引入训练初期 OccFlow 的不稳定噪声
        occ_mask = None

        # 调用 forward() 执行规划推理
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)

        # 计算损失
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs)

        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None):
        """测试阶段的前向传播入口。

        测试时使用 OccFlow 头的输出（seg_out）作为 occ_mask，
        在 use_col_optim=True 时对规划轨迹做碰撞避免后处理优化。

        Args:
            bev_embed (torch.Tensor): BEV 特征图 (H*W, B, C)
            outs_motion (dict): 运动预测头输出（同 forward_train）
            outs_occflow (dict): 占用流预测头输出，包含：
                - 'seg_out': 未来帧的占用掩码 (B, T, num_classes, H, W) 或 (B, T, 1, H, W)
                  表示未来各帧 BEV 网格中哪些位置被障碍物占据
            command (torch.Tensor): 驾驶命令 (B,)

        Returns:
            dict: 规划结果（同 forward() 的返回值）
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        # 测试时用 OccFlow 的分割输出作为 occ_mask（用于碰撞避免优化）
        occ_mask = outs_occflow['seg_out']

        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        return outs_planning

    def forward(self,
                bev_embed,
                occ_mask,
                bev_pos,
                sdc_traj_query,
                sdc_track_query,
                command):
        """规划头核心前向传播逻辑（被 forward_train 和 forward_test 共同调用）。

        完整流程：
        ① 准备三路信息（轨迹query / 跟踪query / 导航嵌入）
        ② MLP 融合 → plan_query (1×1×256)
        ③ BEV Adapter 微调 BEV 特征（如启用）
        ④ Transformer Decoder（plan_query 查询 BEV 特征）
        ⑤ 回归分支 → 位移序列 → 累积求和 → 绝对坐标
        ⑥ 高斯激活解码（bivariate_gaussian_activation）
        ⑦ 碰撞优化后处理（测试时且 use_col_optim=True）

        Args:
            bev_embed (torch.Tensor): BEV 特征 (H*W, B, C)，即 (40000, 1, 256)
                注意：这里是展平后的序列形式，BEVFormer Encoder 的输出
            occ_mask (torch.Tensor | None): 占用掩码 (B, T, 1, H, W)
                训练时为 None；测试时为 OccFlow 头的分割预测结果
            bev_pos (torch.Tensor): BEV 位置编码 (B, C, H, W)，即 (1, 256, 200, 200)
            sdc_traj_query (torch.Tensor): SDC 轨迹查询 (num_layers, B, P, C)
                - num_layers: 运动预测 Decoder 层数（取最后一层 [-1]）
                - P: 预测模式数（predict_modes，如 6 条轨迹的 query）
                - C: 特征维度 (256)
            sdc_track_query (torch.Tensor): SDC 跟踪查询 (B, C)，即 (1, 256)
                跟踪头为 SDC 维护的状态向量，包含位置/速度/外观信息
            command (torch.Tensor): 驾驶命令 (B,)，0=右转, 1=直行, 2=左转

        Returns:
            dict: {
                'sdc_traj':     规划轨迹 (1, planning_steps, 2) 或优化后的版本,
                'sdc_traj_all': 同上（保留两个 key 是为了与其他任务头的接口兼容）
            }
            轨迹坐标格式：以自车当前位置为原点的 ego 坐标系，单位：米
        """
        # ================================================================
        # ① 准备输入：从运动预测的多层输出中取最后一层
        # ================================================================

        # sdc_track_query 不参与梯度计算（detach），避免规划损失反向传播影响跟踪头
        sdc_track_query = sdc_track_query.detach()

        # 取运动预测 Decoder 最后一层的 SDC 轨迹 query
        # sdc_traj_query: (num_layers, B, P, C) → 取 [-1] → (B, P, C) = (1, 6, 256)
        sdc_traj_query = sdc_traj_query[-1]

        # P: 运动预测的模式数（predict_modes=6，即 6 条候选轨迹的特征）
        P = sdc_traj_query.shape[1]   # P = 6

        # 将跟踪 query 扩展到 P 份，与 6 条轨迹 query 对应
        # (B, C) → (B, 1, C) → (B, P, C)，即把 SDC 跟踪状态复制 P 次
        sdc_track_query = sdc_track_query[:, None].expand(-1, P, -1)   # (1, 6, 256)

        # ================================================================
        # ② 三路信息融合：轨迹query + 跟踪query + 导航嵌入
        # ================================================================

        # 查找当前 batch 的导航命令对应的可学习嵌入向量
        # command: (B,) → navi_embed.weight[command]: (B, C) = (1, 256)
        navi_embed = self.navi_embed.weight[command]  # (1, 256)

        # 扩展到 P 份，与轨迹/跟踪 query 维度对齐
        navi_embed = navi_embed[None].expand(-1, P, -1)   # (1, 6, 256)

        # 在最后一维拼接三路信息：256 + 256 + 256 = 768
        plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)
        # plan_query: (1, 6, 768)

        # MLP 融合：768 → 256，然后沿 P 维取 max（选出最显著的规划模式特征）
        # max(1, keepdim=True): 在 P=6 维取最大值，得到 (1, 1, 256)
        # 这相当于从 6 条候选轨迹特征中"选出"最优的那条
        plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[0]
        # plan_query: (1, 6, 768) → mlp → (1, 6, 256) → max → (1, 1, 256)

        # 重排维度：(B, P, C) → (P, B, C) = (1, 1, 256)
        # PyTorch Transformer 接口要求序列维在第 0 位（batch_first=False 时）
        plan_query = rearrange(plan_query, 'b p c -> p b c')   # (1, 1, 256)

        # ================================================================
        # ③ 准备 BEV 特征（加位置编码 + 可选 Adapter）
        # ================================================================

        # 将 BEV 位置编码从 (B, C, H, W) 展平为序列 (H*W, B, C)
        bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')   # (40000, 1, 256)

        # BEV 特征 + 位置编码（加法融合空间位置信息）
        bev_feat = bev_embed + bev_pos   # (40000, 1, 256)

        # ---- BEV Adapter（可选）----
        # 在 BEV 特征上叠加一个轻量卷积残差块，微调 BEV 特征分布
        ##### Plugin adapter #####
        if self.with_adapter:
            # 需要先转回 2D 空间格式才能做卷积
            bev_feat = rearrange(bev_feat, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
            # (1, 256, 200, 200) → bev_adapter → (1, 256, 200, 200) → 残差加
            bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
            # 转回序列格式
            bev_feat = rearrange(bev_feat, 'b c h w -> (h w) b c')  # (40000, 1, 256)
        ##########################

        # ================================================================
        # ④ Transformer Decoder：plan_query 查询 BEV 特征
        # ================================================================

        # 给 plan_query 加上可学习的位置编码（区分"规划 query"与"BEV 序列"）
        pos_embed = self.pos_embed.weight   # (1, 256)，可学习参数
        plan_query = plan_query + pos_embed[None]  # (1, 1, 256) + (1, 1, 256) = (1, 1, 256)

        # 3 层 Transformer Decoder
        # - target（query）:  plan_query (1, 1, 256)
        # - memory（context）: bev_feat (40000, 1, 256)
        # 每层 Decoder 先做 plan_query 的自注意力，再做对 BEV 特征的交叉注意力
        # 通过注意力机制，plan_query "看" 到整个 BEV 地图，关注与规划相关的区域
        # plan_query: [1, 1, 256]
        # bev_feat: [40000, 1, 256]
        plan_query = self.attn_module(plan_query, bev_feat)   # 输出: (1, 1, 256)

        # ================================================================
        # ⑤ 回归输出：plan_query → 轨迹坐标
        # ================================================================

        # reg_branch 将 256 维特征映射为 planning_steps*2 = 12 个标量
        # view 重塑为 (batch, planning_steps, 2)，其中 2 = (x_offset, y_offset)
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
        # sdc_traj_all: (1, 6, 2) — 这里是相对位移（每步相对于上一步的偏移）

        # 累积求和：将相对位移转换为相对于当前位置的绝对坐标
        # 例如：位移序列 [0.5, 0.5, 0.5, ...] → 坐标序列 [0.5, 1.0, 1.5, ...]
        sdc_traj_all[..., :2] = torch.cumsum(sdc_traj_all[..., :2], dim=1)
        # sdc_traj_all: (1, 6, 2) — 现在是绝对坐标（ego坐标系）

        # ================================================================
        # ⑥ 二元高斯激活（解码为概率分布参数）
        # ================================================================

        # bivariate_gaussian_activation 将原始回归输出（可能包含负值/大值）
        # 转换为合法的二元高斯分布参数：
        #   - 均值 (mu_x, mu_y): 轨迹点的期望位置
        #   - 标准差 (sigma_x, sigma_y) 和相关系数 rho: 不确定性估计
        # 注意：这里只处理第 0 个样本（batch_size=1 的场景）
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])

        # ================================================================
        # ⑦ 碰撞避免后处理（仅测试时且 use_col_optim=True）
        # ================================================================

        if self.use_col_optim and not self.training:
            # post process, only used when testing
            # 使用非线性优化器根据 OccFlow 预测结果对轨迹做碰撞避免调整
            assert occ_mask is not None  # 测试时 occ_mask 不能为 None
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)

        return dict(
            sdc_traj=sdc_traj_all,      # 规划轨迹（可能经过碰撞优化）
            sdc_traj_all=sdc_traj_all,  # 同上（两个 key 是为接口兼容性保留）
        )

    def collision_optimization(self, sdc_traj_all, occ_mask):
        """测试时对规划轨迹做碰撞避免非线性优化。

        核心思想：
        1. 从 OccFlow 的占用预测中提取每个时间步的障碍物位置（BEV 坐标）
        2. 只保留与自车预测位置距离在 occ_filter_range（5m）以内的障碍物（减少噪声）
        3. 使用 CollisionNonlinearOptimizer（基于 CasADi 的数值优化器）：
           - 以初始规划轨迹为参考，在满足碰撞避免约束的情况下最小化与参考轨迹的偏差
           - 碰撞代价使用高斯核函数：cost = alpha × exp(-dist²/(2σ²))

        Args:
            sdc_traj_all (torch.Tensor): 初始规划轨迹 (1, planning_steps, 2)
                坐标为 ego 坐标系，单位米
            occ_mask (torch.Tensor): 占用预测掩码 (B, T, [1,] H, W)
                B=1（batch），T=未来帧数（5帧），H×W=BEV网格（200×200）
                值为 1 的位置表示该格子被障碍物占据

        Returns:
            torch.Tensor: 优化后的规划轨迹 (1, planning_steps, 2)
                如果没有有效障碍物（valid_occupancy_num==0），直接返回原始轨迹
        """
        pos_xy_t = []           # 各时间步的障碍物位置列表（ego坐标系，单位米）
        valid_occupancy_num = 0  # 当前帧附近有效障碍物的总数量

        # occ_mask 有时含冗余维度 (B, T, 1, H, W)，需要压缩
        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)  # (B, T, H, W)
        occ_horizon = occ_mask.shape[1]  # 占用预测的时间跨度（T=5帧）
        assert occ_horizon == 5          # 确认是5帧（0.5s, 1.0s, 1.5s, 2.0s, 2.5s）

        # 对每个规划步骤，从对应时间的 occ_mask 中提取障碍物位置
        for t in range(self.planning_steps):  # t = 0, 1, 2, 3, 4, 5（6步规划）
            # 映射规划步到占用帧：规划步 t 对应 occ 帧 t+1（偏移1帧）
            # 当 t >= occ_horizon-1 时，用最后一帧的 occ（占用预测覆盖不到那么远）
            cur_t = min(t+1, occ_horizon-1)   # occ 帧索引 (1~4)

            # 提取 occ_mask 中值为 1（有障碍物）的网格坐标（像素坐标）
            # pos_xy: (N, 2)，格式为 [row, col]（PyTorch nonzero 输出行列格式）
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)  # (N, 2)

            # 交换行列顺序：从 [row, col] → [col, row]，使得：
            # pos_xy[:, 0] = col（对应 x 轴，即横向位置）
            # pos_xy[:, 1] = row（对应 y 轴，即纵向位置）
            pos_xy = pos_xy[:, [1, 0]]   # (N, 2)，现在是 [col, row] = [x_pix, y_pix]

            # 将像素坐标转换为 ego 坐标系（米）
            # BEV 网格以自车为中心：第 bev_h//2 行 = y=0，第 bev_w//2 列 = x=0
            # 分辨率为 0.5m/格，+0.25 是格子中心偏移
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25  # x: 像素→米
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25  # y: 像素→米

            # 距离过滤：只保留与当前规划点距离 < occ_filter_range(5m) 的障碍物
            # 原因：距离太远的障碍物对当前时刻的规划几乎无影响，过滤掉减少噪声
            keep_index = torch.sum(
                (sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1
            ) < self.occ_filter_range**2   # 欧式距离平方 < 5² = 25

            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())  # 转 numpy 备用
            valid_occupancy_num += torch.sum(keep_index > 0)            # 统计有效障碍物数

        # 如果附近没有任何障碍物，直接返回原始轨迹（无需优化）
        if valid_occupancy_num == 0:
            return sdc_traj_all

        # ---- 非线性轨迹优化 ----
        # CollisionNonlinearOptimizer 使用 CasADi 数值优化框架：
        # 目标函数：minimize ||traj - ref_traj||² + alpha_collision × Σ collision_cost(t)
        # 其中 collision_cost 使用高斯核：exp(-dist²/(2σ²))
        col_optimizer = CollisionNonlinearOptimizer(
            self.planning_steps,    # 规划步数=6
            0.5,                    # 每步时间间隔=0.5s
            self.sigma,             # 高斯核标准差（碰撞代价扩散范围）
            self.alpha_collision,   # 碰撞代价权重（越大越保守）
            pos_xy_t               # 各时间步的障碍物位置列表
        )
        # 设置参考轨迹（网络预测的初始轨迹，优化从这里出发）
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())

        # 求解优化问题
        sol = col_optimizer.solve()

        # 提取优化后的轨迹坐标（x 序列 + y 序列 → 按列堆叠为 (6, 2)）
        sdc_traj_optim = np.stack(
            [sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)],
            axis=-1
        )  # (6, 2)

        # 转回 Tensor，保持与原始轨迹相同的 device 和 dtype
        return torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
        # 返回: (1, 6, 2)

    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None):
        """计算规划头的总损失（ADE 损失 + 多尺度碰撞损失）。

        损失由两部分组成：
        1. loss_ade (PlanningLoss)：预测轨迹与 GT 轨迹的 L2 误差
           衡量规划轨迹与人类驾驶员轨迹的接近程度
        2. loss_collision_i (CollisionLoss)：规划轨迹与未来 GT 框的碰撞惩罚
           使用不同 delta（安全边界）强制规划轨迹与障碍物保持安全距离：
           - delta=0.0: 严格碰撞（轨迹点落在 GT 框内），权重最大（2.5）
           - delta=0.5: 扩展 0.5m 安全边界，权重中等（1.0）
           - delta=1.0: 扩展 1.0m 安全边界，权重较小（0.25）

        Args:
            sdc_planning (torch.Tensor): 自车规划轨迹 GT (B, 1, T_plan, 3)
                最后维度：[x(m), y(m), heading(rad)]
            sdc_planning_mask (torch.Tensor): 规划有效掩码 (B, 1, T_plan, 1)
                False=无效（场景切换/越界帧），损失计算时忽略这些时间步
            outs_planning (dict): forward() 的输出，包含 'sdc_traj_all'
            future_gt_bbox (list[list]): 未来各帧的 GT 目标框
                future_gt_bbox[0][t] 为第 t 帧的所有障碍物 GT 框（用于碰撞检测）

        Returns:
            dict: 损失字典，包含：
                - 'loss_collision_0': delta=0.0 的碰撞损失
                - 'loss_collision_1': delta=0.5 的碰撞损失
                - 'loss_collision_2': delta=1.0 的碰撞损失
                - 'loss_ade':         轨迹 L2/ADE 损失
        """
        # 从输出字典中取出规划轨迹
        # sdc_traj_all: (1, planning_steps, 2) 或含高斯参数的 (1, planning_steps, 5)
        sdc_traj_all = outs_planning['sdc_traj_all'] # b, p, t, 5

        loss_dict = dict()

        # ---- 计算多尺度碰撞损失 ----
        for i in range(len(self.loss_collision)):
            # loss_collision[i] 对应第 i 个安全边界（delta=0/0.5/1.0）
            # 参数说明：
            #   sdc_traj_all:                           预测轨迹 (1, T, 2)
            #   sdc_planning[0, :, :planning_steps, :3]: GT 轨迹 (1, T, 3)（取x,y,heading）
            #   torch.any(sdc_planning_mask[0, :, :planning_steps], dim=-1): 有效帧掩码
            #   future_gt_bbox[0][1:planning_steps+1]:  未来帧 GT 框（从第1帧开始取T帧）
            loss_collision = self.loss_collision[i](
                sdc_traj_all,
                sdc_planning[0, :, :self.planning_steps, :3],       # GT 轨迹（x,y,heading）
                torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1),  # 有效掩码
                future_gt_bbox[0][1:self.planning_steps+1]           # 未来障碍物 GT 框
            )
            loss_dict[f'loss_collision_{i}'] = loss_collision

        # ---- 计算轨迹 ADE 损失（L2 误差）----
        # loss_planning 计算预测轨迹与 GT 轨迹各时间步的平均欧式距离（ADE）
        # 参数说明：
        #   sdc_traj_all:                           预测轨迹 (1, T, 2)
        #   sdc_planning[0, :, :planning_steps, :2]: GT 轨迹 (1, T, 2)（只取x,y，不含heading）
        #   torch.any(sdc_planning_mask[...], dim=-1): 有效帧掩码（忽略无效步的损失）
        loss_ade = self.loss_planning(
            sdc_traj_all,
            sdc_planning[0, :, :self.planning_steps, :2],           # GT 轨迹（只用x,y）
            torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1)
        )
        loss_dict.update(dict(loss_ade=loss_ade))

        # ---- 计算运动学可行性损失（新增）----
        # 只有在配置了 loss_kinematic 时才计算，保持对原始配置的向后兼容
        if self.loss_kinematic is not None:
            # 复用与 ADE 损失完全相同的输入格式：
            #   sdc_traj_all:  预测轨迹 (1, T, 2)，累积坐标（非位移）
            #   GT 轨迹:       (1, T, 2)，只取 x,y
            #   有效掩码:       (1, T)，True=有效时间步
            valid_mask = torch.any(
                sdc_planning_mask[0, :, :self.planning_steps], dim=-1
            )  # (1, T)
            loss_kinematic = self.loss_kinematic(
                sdc_traj_all,                                           # 预测轨迹 (1, T, 2)
                sdc_planning[0, :, :self.planning_steps, :2],           # GT 轨迹 (1, T, 2)
                valid_mask                                              # 有效掩码 (1, T)
            )
            loss_dict.update(dict(loss_kinematic=loss_kinematic))

        return loss_dict