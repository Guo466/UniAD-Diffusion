#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# ================================================================================
# 文件概述：OccHead —— 占用流预测头（Occupancy & Flow Head）
#
# 功能：
#   预测未来多个时间步（n_future=4步，即2秒）内，场景中每个被跟踪目标
#   在 BEV（鸟瞰图）空间中的"占用区域"（Occupancy），形成类似视频分割的预测。
#
# 核心问题：
#   "未来 2 秒内，场景中的每辆车/行人会占据哪些位置？"
#   → 输出：每个目标在 BEV 网格中每个时间步的二值掩码（占用=1，空=0）
#
# 与 MotionHead 的区别：
#   MotionHead：预测每个目标的轨迹（中心点坐标序列，1D）
#   OccHead：预测每个目标的占用区域（BEV 网格的像素级别掩码，2D）
#   两者互补：轨迹给出运动方向，占用给出空间范围（车有多大、占了哪些格子）
#
# 在 UniAD 流水线中的位置：
#   TrackHead → SegHead → MotionHead → OccHead → PlanningHead
#   OccHead 的输出（occ_mask）直接传给 PlanningHead，
#   用于碰撞避免（告诉规划头"这些区域将来有目标，不要去"）
#
# 核心机制：实例级占用预测（Instance-level Occupancy）
#   - 不是预测"整个场景的占用网格"，而是对每个目标单独预测其占用掩码
#   - 每个目标的 Query（来自 MotionHead）通过 Transformer 与 BEV 特征交互
#   - 最终对每个目标生成 (T, H, W) 的占用序列，T=未来时间步数
#
# 损失函数：
#   - Dice Loss（loss_dice）：衡量预测掩码与 GT 掩码的重叠程度
#   - Mask Loss（loss_mask）：像素级别的二分类交叉熵
# ================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule
from einops import rearrange         # 张量维度重排工具（比 reshape/permute 更直观）
from mmdet.core import reduce_mean   # 分布式训练中同步平均值
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
import copy
# 从 occ_head_plugin 导入各子模块：
from .occ_head_plugin import (
    MLP,                                          # 多层感知机（全连接网络）
    BevFeatureSlicer,                             # BEV 特征图裁剪器（从大 BEV 中取小范围）
    SimpleConv2d,                                 # 简单卷积层
    CVT_Decoder,                                  # 跨视图 Transformer 解码器（上采样恢复分辨率）
    Bottleneck,                                   # 瓶颈卷积块（下采样）
    UpsamplingAdd,                                # 上采样 + 跳连加法（类似 U-Net）
    predict_instance_segmentation_and_trajectories  # 后处理：生成实例分割结果
)

def _get_clones(module, N):
    """将一个模块深拷贝 N 份，返回 ModuleList。
    
    用途：每个时间步需要独立的参数（不共享权重），
         通过深拷贝创建 N 个结构相同但参数独立的模块。
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@HEADS.register_module()
class OccHead(BaseModule):
    """占用流预测头。
    
    对场景中每个被跟踪目标，预测其在未来 n_future 个时间步的 BEV 占用掩码。
    
    核心流程：
      1. merge_queries()：融合 MotionHead 输出的三种 query（轨迹query、track query、位置query）
      2. forward()：以融合后的 query 为条件，逐时间步生成未来 BEV 特征
      3. 用 einsum 将 query 特征与 BEV 特征做点积，得到每个目标的占用 logit
      4. 计算 Dice Loss + Mask Loss
    
    Args:
        receptive_field (int): 使用过去多少帧（含当前帧）作为输入，默认3帧（1.5秒历史）
        n_future (int): 预测未来多少帧，默认4帧（2秒）
        spatial_extent (tuple): BEV 空间范围（米），如 (50, 50) 表示前后左右各50米
        ignore_index (int): 损失计算中忽略的标签值（255 = 无效区域）
        grid_conf (dict): 目标 BEV 网格配置（分辨率、范围）
        bev_size (tuple): BEV 特征图尺寸，如 (200, 200)
        bev_emb_dim (int): 输入 BEV 特征的通道数（来自 BEVFormer，256维）
        bev_proj_dim (int): BEV 特征投影后的维度（64维，降维减少计算量）
        bev_proj_nlayers (int): BEV 投影卷积层数
        query_dim (int): 输入 query 的维度（来自 MotionHead，256维）
        query_mlp_layers (int): query → occ特征 的 MLP 层数
        detach_query_pos (bool): 是否断开位置 query 的梯度（防止位置信息影响特征学习）
        temporal_mlp_layer (int): 时序 MLP 的层数（用于对 query 做时序感知变换）
        transformer_decoder (dict): Transformer decoder 层的配置
        attn_mask_thresh (float): 注意力掩码的阈值（sigmoid > 此值 = 前景区域）
        aux_loss_weight (float): 辅助损失的权重
        loss_mask (dict): Mask Loss 的配置
        loss_dice (dict): Dice Loss 的配置
        pan_eval (bool): 是否使用全景评估模式（实例分割评估）
        test_seg_thresh (float): 测试时的分割阈值（sigmoid > 此值 = 占用）
        test_with_track_score (bool): 测试时是否用跟踪置信度加权占用预测
    """
    def __init__(self, 
                 # General
                 receptive_field=3,       # 过去帧数（含当前帧），用于判断序列是否有效
                 n_future=4,              # 预测未来帧数（4帧 × 0.5s = 2秒）
                 spatial_extent=(50, 50), # BEV 空间范围（米）
                 ignore_index=255,        # GT 标签中的忽略值

                 # BEV
                 grid_conf=None,          # 目标 BEV 网格配置（分辨率、范围）

                 bev_size=(200, 200),     # BEV 特征图的空间尺寸（200×200格子）
                 bev_emb_dim=256,         # 输入 BEV 特征的通道数（来自 BEVFormer）
                 bev_proj_dim=64,         # 投影后的 BEV 特征通道数（降维）
                 bev_proj_nlayers=1,      # BEV 投影的卷积层数

                 # Query
                 query_dim=256,           # query 向量的维度
                 query_mlp_layers=3,      # query → occ特征 MLP 的层数
                 detach_query_pos=True,   # 是否截断位置 query 的梯度
                 temporal_mlp_layer=2,    # 时序 MLP 的层数

                 # Transformer
                 transformer_decoder=None,  # Transformer decoder 配置

                 attn_mask_thresh=0.5,    # 注意力掩码阈值（前景/背景分界线）
                 
                 # Loss
                 sample_ignore_mode='all_valid',  # 序列有效性过滤模式
                 aux_loss_weight=1.,              # 辅助损失权重

                 loss_mask=None,   # Mask Loss 配置（像素级二分类交叉熵）
                 loss_dice=None,   # Dice Loss 配置（掩码重叠度损失）

                 # Cfgs
                 init_cfg=None,

                 # Eval
                 pan_eval=False,              # 是否使用全景分割评估
                 test_seg_thresh: float=0.5,  # 测试时的分割阈值
                 test_with_track_score=False, # 是否用跟踪分数加权占用预测
                 ):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)
        
        # ---- 基本参数 ----
        self.receptive_field = receptive_field  # 用于判断历史帧是否全部有效（在 loss 中使用）
        self.n_future = n_future                # 预测未来帧数
        self.spatial_extent = spatial_extent    # BEV 空间范围（米），用于坐标归一化
        self.ignore_index = ignore_index        # GT 标签忽略值（255 = 无效/被遮挡）

        # ---- BEV 特征图裁剪器 ----
        # BEVFormer 生成的 BEV 特征图范围通常比 OccHead 需要的大
        # BevFeatureSlicer 将大 BEV 特征图裁剪/重采样到目标分辨率
        # 源 BEV 配置（BEVFormer 的范围：±51.2m，分辨率0.512m/格）
        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],  # x轴：从-51.2m到51.2m，每格0.512m
            'ybound': [-51.2, 51.2, 0.512],  # y轴：同上
            'zbound': [-10.0, 10.0, 20.0],   # z轴：高度范围
        }
        # grid_conf 是 OccHead 目标配置（来自配置文件，可能范围更小）
        self.bev_sampler = BevFeatureSlicer(bevformer_bev_conf, grid_conf)
        
        self.bev_size = bev_size          # BEV 特征图尺寸（200×200）
        self.bev_proj_dim = bev_proj_dim  # 投影后的特征通道数（64）

        # ---- BEV 特征投影：将 256 维 BEV 特征降维到 64 维 ----
        # 降维目的：减少后续卷积和 Transformer 的计算量
        if bev_proj_nlayers == 0:
            self.bev_light_proj = nn.Sequential()  # 不做投影，直接透传
        else:
            self.bev_light_proj = SimpleConv2d(
                in_channels=bev_emb_dim,    # 输入：256维（BEVFormer 输出）
                conv_channels=bev_emb_dim,  # 中间：256维
                out_channels=bev_proj_dim,  # 输出：64维（降维）
                num_conv=bev_proj_nlayers,  # 卷积层数
            )

        # ---- 基础下采样：将 BEV 特征图空间分辨率缩小 4 倍 ----
        # 200×200 → 50×50（通过两个 Bottleneck 各下采样2倍）
        # 目的：减少 Transformer 中 token 数量，降低计算量
        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=bev_proj_dim, downsample=True),  # 200×200 → 100×100
            Bottleneck(in_channels=bev_proj_dim, downsample=True)   # 100×100 → 50×50
        )

        # ---- 时序 Transformer 的时间步数 ----
        # n_future_blocks = n_future + 1 = 5（包含当前帧 t=0 和未来4帧 t=1,2,3,4）
        self.n_future_blocks = self.n_future + 1

        # ---- Transformer Decoder 配置 ----
        self.attn_mask_thresh = attn_mask_thresh  # 注意力掩码阈值（0.5）
        
        self.num_trans_layers = transformer_decoder.num_layers  # Transformer 总层数
        # 每个时间步分配相同数量的 Transformer 层
        assert self.num_trans_layers % self.n_future_blocks == 0
        
        # 提取注意力头数（用于后续生成注意力掩码时对齐头数）
        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        
        # 构建 Transformer Decoder（多层 cross-attention）
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)

        # ---- 时序感知 MLP（每个时间步一个独立的 MLP）----
        # 将 query 从"无时序意识"变换为"感知当前是第几步"
        # 每个时间步的 MLP 参数独立，让模型区分不同时间步
        # 注意：ins_query 在进入 forward() 之前已经被 merge_queries 压缩到 bev_proj_dim(64) 维
        # 所以 input_dim = bev_proj_dim，而不是 query_dim
        temporal_mlp = MLP(bev_proj_dim, bev_proj_dim, bev_proj_dim, num_layers=temporal_mlp_layer)
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)  # 5个独立 MLP
            
        # ---- 逐时间步下采样卷积（每步一个独立卷积）----
        # 在 Transformer cross-attention 之前，将 BEV 特征再下采样一次：
        # 50×50 → 25×25（/8 总分辨率）
        # 目的：进一步减少 Transformer 的 token 数量
        downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)  # 5个独立卷积
        
        # ---- 逐时间步上采样（每步一个独立上采样模块）----
        # Transformer 结束后，将特征从 25×25 恢复到 50×50
        # 使用 UpsamplingAdd（上采样 + 跳连加法，类似 U-Net 的跳连）
        upsample_add = UpsamplingAdd(in_channels=bev_proj_dim, out_channels=bev_proj_dim)
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)  # 5个独立上采样

        # ---- 密集解码器：将多时间步特征上采样到完整分辨率 ----
        # 将 50×50 的特征解码回 200×200（全分辨率用于输出）
        self.dense_decoder = CVT_Decoder(
            dim=bev_proj_dim,
            blocks=[bev_proj_dim, bev_proj_dim],  # 两个上采样块
        )

        # ---- Query 融合模块 ----
        # mode_fuser：将多模态轨迹 query（6条轨迹取最大值）从256→64
        # 作用：从 MotionHead 的 num_anchor 条轨迹中，提取最具代表性的特征
        self.mode_fuser = nn.Sequential(
            nn.Linear(query_dim, bev_proj_dim),    # 256 → 64
            nn.LayerNorm(bev_proj_dim),
            nn.ReLU(inplace=True)
        )
        
        # multi_query_fuser：融合三种 query（轨迹query + track query + 位置query）
        # 输入：mode_fuser(traj_query)[bev_proj_dim] + track_query[query_dim] + track_query_pos[query_dim]
        #       = bev_proj_dim + query_dim * 2
        # 注意：traj_query 经 mode_fuser 后已从 query_dim 降为 bev_proj_dim，
        #       因此拼接维度是 bev_proj_dim + query_dim*2（而非 query_dim*3）
        _fuser_in_dim = bev_proj_dim + query_dim * 2
        _fuser_mid_dim = max(query_dim * 2, bev_proj_dim * 4)  # 中间层保持足够宽度
        self.multi_query_fuser = nn.Sequential(
            nn.Linear(_fuser_in_dim, _fuser_mid_dim),
            nn.LayerNorm(_fuser_mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(_fuser_mid_dim, bev_proj_dim),
        )

        self.detach_query_pos = detach_query_pos  # 是否截断位置 query 的梯度

        # query_to_occ_feat：将 query 映射为用于生成占用掩码的特征向量
        # 最终会与 BEV 特征做点积得到占用 logit
        # 注意：ins_query 经 merge_queries 压缩到 bev_proj_dim(64) 维后，
        # 再经 temporal_mlps 变换，此处输入依然是 bev_proj_dim 维
        self.query_to_occ_feat = MLP(
            bev_proj_dim, bev_proj_dim, bev_proj_dim, num_layers=query_mlp_layers
        )
        
        # temporal_mlp_for_mask：用于生成注意力掩码的 query 变换
        # 与 query_to_occ_feat 结构相同但参数独立（深拷贝）
        self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)
        
        # ---- 损失函数配置 ----
        self.sample_ignore_mode = sample_ignore_mode  # 样本有效性过滤模式
        assert self.sample_ignore_mode in ['all_valid', 'past_valid', 'none']

        self.aux_loss_weight = aux_loss_weight  # 辅助损失权重（辅助掩码损失的缩放系数）

        # 构建损失函数对象
        self.loss_dice = build_loss(loss_dice)  # Dice Loss：衡量掩码重叠度
        self.loss_mask = build_loss(loss_mask)  # Mask Loss：像素级二分类交叉熵

        # ---- 评估相关参数 ----
        self.pan_eval = pan_eval                          # 是否使用全景分割评估
        self.test_seg_thresh = test_seg_thresh            # 分割阈值（默认0.5）
        self.test_with_track_score = test_with_track_score  # 是否用跟踪分数加权
        
        self.init_weights()  # 初始化权重

    def init_weights(self):
        """初始化 Transformer decoder 的权重（Xavier 正态分布初始化）。
        
        只对维度 > 1 的参数（权重矩阵）初始化，偏置不处理。
        Xavier 初始化能让梯度在网络层间保持稳定，避免梯度消失/爆炸。
        """
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_attn_mask(self, state, ins_query):
        """生成 Transformer cross-attention 的注意力掩码。
        
        核心思想：
            先用 query 对当前 BEV 特征做一次"粗预测"，得到每个目标的大致位置，
            然后把这个大致位置作为注意力掩码，引导 Transformer 只关注相关区域。
            
            这类似于"先粗定位，再精确关注"的两阶段注意力机制，
            避免 Transformer 在无关区域浪费注意力。
        
        Args:
            state (Tensor): 当前时间步的 BEV 特征，shape (B, C, H', W')
                H'=W'=25（/8 下采样后的特征图）
            ins_query (Tensor): 当前时间步的实例 query，shape (B, Q, C)
                Q = 被跟踪的目标数量
        
        Returns:
            attn_mask (Tensor): 注意力掩码，True 的位置被屏蔽（不关注）
                shape: (B*num_heads, H'*W', Q)
                True 表示背景（query 不应该关注），False 表示前景（query 应该关注）
            upsampled_mask_pred (Tensor): 上采样到原始 BEV 分辨率的掩码预测
                shape: (B, Q, H, W)，用于辅助监督
            ins_embed (Tensor): 用于生成掩码的 query 嵌入，shape (B, Q, C)
        """
        # temporal_mlp_for_mask: 对 64 维 query 做非线性变换（维度保持 bev_proj_dim=64）
        # ins_embed: shape (B, Q, C)，C=bev_proj_dim=64
        ins_embed = self.temporal_mlp_for_mask(ins_query)
        
        # einsum("bqc,bchw->bqhw"): 每个 query 向量与每个 BEV 空间位置做点积
        # 点积越大 → 该位置越可能被这个 query 对应的目标占用
        # mask_pred: shape (B, Q, H', W')，每个目标在每个位置的"粗占用得分"
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)
        
        # sigmoid < 0.5 的位置认为是背景，设为 True（屏蔽）
        # 即：只让 Transformer 关注预测为前景（sigmoid >= 0.5）的区域
        attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
        
        # 变形为 Transformer 注意力掩码所需的格式
        # rearrange: (B, Q, H', W') → (B, H'*W', Q)，方便矩阵运算
        # unsqueeze(1).repeat(1, num_heads, 1, 1): 复制 num_heads 份（多头注意力每头一个掩码）
        # flatten(0, 1): (B, num_heads, H'*W', Q) → (B*num_heads, H'*W', Q)
        attn_mask = rearrange(attn_mask, 'b q h w -> b (h w) q').unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = attn_mask.detach()  # 不通过掩码传递梯度（掩码仅作为条件，不参与优化）
        
        # 特殊处理：如果某个 query 对应的掩码全为 True（全背景），
        # 则取消掩码（设为全 False），防止该 query 无法关注任何位置
        attn_mask[torch.where(
            attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        # 将掩码预测上采样到完整 BEV 分辨率（200×200），用于辅助损失计算
        upsampled_mask_pred = F.interpolate(
            mask_pred,
            self.bev_size,      # 目标尺寸：200×200
            mode='bilinear',    # 双线性插值
            align_corners=False
        )  # shape: (B, Q, 200, 200)，用于辅助 GT 监督

        return attn_mask, upsampled_mask_pred, ins_embed

    def forward(self, x, ins_query):
        """核心前向传播：逐时间步生成未来 BEV 特征，并预测每个目标的占用掩码。
        
        整体流程（循环 n_future_blocks=5 次，对应 t=0,1,2,3,4）：
          对每个时间步 i：
            1. downscale_convs[i]：BEV 特征再下采样 50×50 → 25×25
            2. temporal_mlps[i]：将 query 做时序感知变换（感知"我是第几步"）
            3. get_attn_mask()：用 query 生成注意力掩码（粗定位）
            4. Transformer cross-attention：query 关注 BEV 特征中的相关区域，更新 BEV 特征
            5. upsample_adds[i]：上采样 25×25 → 50×50，并与上一步特征加和（跳连）
            6. 保存当前步的 BEV 特征和掩码预测
        
        循环结束后：
            - dense_decoder：将所有时间步的 BEV 特征 50×50 → 200×200（全分辨率）
            - query_to_occ_feat：将 query 映射为占用特征向量
            - einsum：query 与 BEV 特征点积 → 占用 logit（每个目标在每个位置的占用分数）
        
        Args:
            x (Tensor): BEV 特征图，shape (H*W, B, D)
                H*W = 200*200 = 40000，B=batch_size，D=256（BEVFormer 输出格式）
            ins_query (Tensor): 融合后的实例 query，shape (B, Q, C)
                Q = 车辆目标数量，C = bev_proj_dim = 64（经 merge_queries 处理后）
        
        Returns:
            mask_preds (Tensor): 辅助注意力掩码预测（用于辅助损失）
                shape: (B, Q, T, H, W) = (B, Q, 5, 200, 200)
            ins_occ_logits (Tensor): 最终占用 logit（主要输出）
                shape: (B, Q, T, H, W) = (B, Q, 5, 200, 200)
                正值 → 该目标在该位置/时间步占用，负值 → 不占用
        """
        # ============================================================
        # Step 1: BEV 特征预处理
        # ============================================================
        # 将 BEV 特征从 (H*W, B, D) 变形为图像格式 (B, D, H, W)
        # rearrange: (h*w, b, d) → (b, d, h, w)
        base_state = rearrange(x, '(h w) b d -> b d h w', h=self.bev_size[0])
        # base_state: shape (B, 256, 200, 200)

        # 裁剪到目标 BEV 范围（从 ±51.2m 范围裁剪到配置的范围）
        base_state = self.bev_sampler(base_state)
        
        # 降维：256 → 64
        base_state = self.bev_light_proj(base_state)
        
        # 下采样：200×200 → 50×50（/4 总倍率）
        base_state = self.base_downscale(base_state)
        # base_state: shape (B, 64, 50, 50)
        
        base_ins_query = ins_query  # 保存初始 query，供后续时序循环使用

        # ============================================================
        # Step 2: 逐时间步 Transformer 处理（5个时间步：t=0,1,2,3,4）
        # ============================================================
        last_state = base_state       # 上一时间步的 BEV 特征（初始为当前帧）
        last_ins_query = base_ins_query  # 上一时间步的实例 query
        
        future_states = []              # 存储每步的 BEV 特征（用于最终解码）
        mask_preds = []                 # 存储每步的辅助掩码预测（用于辅助损失）
        temporal_query = []             # 存储每步时序变换后的 query
        temporal_embed_for_mask_attn = []  # 存储每步用于生成掩码的嵌入

        # 每个时间步分配的 Transformer 层数
        n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
        assert n_trans_layer_each_block >= 1
        
        for i in range(self.n_future_blocks):  # i = 0, 1, 2, 3, 4（对应 t=0 到 t=4）
            # ---- 2a: 将 BEV 特征再下采样：50×50 → 25×25 ----
            # 在 Transformer 处理前再下采样，进一步减少 token 数量
            cur_state = self.downscale_convs[i](last_state)  # shape: (B, 64, 25, 25)

            # ---- 2b: 时序感知 query 变换 ----
            # temporal_mlps[i]：将 query 变换为"感知当前是第 i 步"的特征
            # 每个时间步有独立的 MLP 参数，让模型自动学习时序差异
            # cur_ins_query: shape (B, Q, 64)（输入输出都是 bev_proj_dim=64 维）
            cur_ins_query = self.temporal_mlps[i](last_ins_query)
            temporal_query.append(cur_ins_query)

            # ---- 2c: 生成注意力掩码（粗定位引导精确注意力）----
            # attn_mask: (B*num_heads, H'*W', Q)，True = 背景（屏蔽），False = 前景（关注）
            # mask_pred: (B, Q, 200, 200)，上采样后的掩码，用于辅助监督
            # cur_ins_emb_for_mask_attn: (B, Q, 64)，用于生成掩码的 query 嵌入
            attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(cur_state, cur_ins_query)
            
            # attn_masks 列表：[None, attn_mask]
            # None 对应自注意力（BEV 内部），attn_mask 对应 cross-attention（BEV-query 交互）
            attn_masks = [None, attn_mask] 

            mask_preds.append(mask_pred)  # 保存辅助掩码
            temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)

            # ---- 2d: 变形为 Transformer 所需的格式 ----
            # Transformer 要求输入格式为 (序列长度, batch, 特征维度)
            cur_state = rearrange(cur_state, 'b c h w -> (h w) b c')
            # cur_state: shape (25*25=625, B, 64)，BEV 格子数=625

            cur_ins_query = rearrange(cur_ins_query, 'b q c -> q b c')
            # cur_ins_query: shape (Q, B, 64)

            # ---- 2e: 逐 Transformer 层处理（每个时间步分配 n_trans_layer_each_block 层）----
            for j in range(n_trans_layer_each_block):
                trans_layer_ind = i * n_trans_layer_each_block + j  # 全局层索引
                trans_layer = self.transformer_decoder.layers[trans_layer_ind]
                
                # Cross-attention：BEV 特征（query）关注实例 query（key/value）
                # 注意：这里的角色是"BEV 特征被 query 条件化"
                # - query: BEV 格子特征（接收信息的一方）
                # - key/value: 实例 query（提供目标信息的一方）
                # attn_masks 引导 BEV 特征只在目标可能出现的区域更新
                cur_state = trans_layer(
                    query=cur_state,          # (H'*W', B, C)：BEV 格子特征
                    key=cur_ins_query,        # (Q, B, C)：实例 query（key）
                    value=cur_ins_query,      # (Q, B, C)：实例 query（value）
                    query_pos=None,           # BEV 无位置编码（已通过 sampler 隐含位置）
                    key_pos=None,             # query 无额外位置编码
                    attn_masks=attn_masks,    # [None（自注意力）, attn_mask（交叉注意力）]
                    query_key_padding_mask=None,
                    key_padding_mask=None
                )  # 输出: (H'*W', B, C)，BEV 特征被目标信息"污染"（更新）

            # ---- 2f: 变形回图像格式，并上采样恢复分辨率 ----
            cur_state = rearrange(cur_state, '(h w) b c -> b c h w', h=self.bev_size[0]//8)
            # cur_state: shape (B, 64, 25, 25)

            # UpsamplingAdd：将 25×25 上采样回 50×50，并与上一步的 50×50 特征相加
            # 跳连连接（类似 U-Net）防止信息在下采样时丢失
            cur_state = self.upsample_adds[i](cur_state, last_state)
            # cur_state: shape (B, 64, 50, 50)

            # 保存当前时间步的结果
            future_states.append(cur_state)  # 形状: (B, 64, 50, 50)
            last_state = cur_state           # 更新为下一步的输入（时序传递）

        # ============================================================
        # Step 3: 汇总多时间步特征
        # ============================================================
        # 将5个时间步的 BEV 特征堆叠成时序张量
        # future_states: (B, T, D, H/4, W/4) = (B, 5, 64, 50, 50)
        future_states = torch.stack(future_states, dim=1)
        
        # 将5个时间步的 query 堆叠
        # temporal_query: (B, T, Q, D) = (B, 5, Q, 64)
        temporal_query = torch.stack(temporal_query, dim=1)
        
        # 将5个时间步的辅助掩码堆叠
        # mask_preds: (B, Q, T, H, W) = (B, Q, 5, 200, 200)
        mask_preds = torch.stack(mask_preds, dim=2)
        
        # 将5个时间步的掩码生成嵌入堆叠
        # ins_query: (B, T, Q, D) = (B, 5, Q, 64)
        ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)

        # ============================================================
        # Step 4: 解码到全分辨率
        # ============================================================
        # dense_decoder：将 BEV 特征从 50×50 上采样回 200×200
        # future_states: (B, T, D, H, W) = (B, 5, 64, 200, 200)
        future_states = self.dense_decoder(future_states)
        
        # query_to_occ_feat：将 query 特征映射为占用预测所需的嵌入
        # ins_occ_query: (B, T, Q, C) = (B, 5, Q, 64)
        ins_occ_query = self.query_to_occ_feat(ins_query)
        
        # ============================================================
        # Step 5: 生成最终占用 logit
        # ============================================================
        # einsum("btqc,btchw->bqthw"):
        #   对每个时间步 t，每个目标 q，计算其 query 向量与每个 BEV 格子特征的点积
        #   点积越大 → 该目标在该格子/时间步的占用概率越高
        # ins_occ_logits: (B, Q, T, H, W) = (B, Q, 5, 200, 200)
        ins_occ_logits = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)
        
        return mask_preds, ins_occ_logits

    def merge_queries(self, outs_dict, detach_query_pos=True):
        """融合 MotionHead 输出的三种 query，生成 OccHead 的输入。
        
        MotionHead 输出了三种不同的 query，各自携带不同信息：
          1. traj_query（轨迹 query）：包含多模态轨迹预测的特征
             shape: (n_dec, B, Q, n_modes, D)，取最后一层和最优模式
          2. track_query：目标的身份/外观特征（来自 TrackHead）
             shape: (B, Q, D)
          3. track_query_pos：目标的位置编码（目标在哪里）
             shape: (B, Q, D)
        
        融合步骤：
          1. traj_query：取最后一层 decoder 的输出，在 n_modes 维度取 max（最优模式）
          2. 将三种 query 拼接：[traj_query || track_query || track_query_pos]（768维）
          3. 通过 multi_query_fuser MLP 压缩到 64 维
        
        Args:
            outs_dict (dict): MotionHead 的输出字典，包含：
                'traj_query': shape (n_dec, B, Q, n_modes, D)
                'track_query': shape (B, Q, D)
                'track_query_pos': shape (B, Q, D)
            detach_query_pos (bool): 是否截断位置 query 的梯度
                True：位置信息只作为条件，不反向传播到位置编码（防止过拟合）
        
        Returns:
            ins_query (Tensor): 融合后的实例 query，shape (B, Q, 64)
                作为 OccHead.forward() 的输入
        """
        # 取轨迹 query（n_dec, B, Q, n_modes, D）和其他两种 query
        ins_query = outs_dict.get('traj_query', None)   # (n_dec, B, Q, n_modes, D)
        track_query = outs_dict['track_query']           # (B, Q, D)
        track_query_pos = outs_dict['track_query_pos']   # (B, Q, D)

        # 可选：截断位置 query 的梯度
        # 位置信息（在哪里）不应该影响到轨迹预测的特征学习
        if detach_query_pos:
            track_query_pos = track_query_pos.detach()

        # 取最后一层 decoder 的轨迹 query
        # ins_query: (n_dec, B, Q, n_modes, D) → (B, Q, n_modes, D)
        ins_query = ins_query[-1]
        
        # mode_fuser：256→64，然后在 n_modes 维取 max（取最优模式的特征）
        # ins_query: (B, Q, n_modes, D) → (B, Q, n_modes, 64) → max → (B, Q, 64)
        # .max(2)[0]：在第2维（n_modes）取最大值，[0] 取值（[1] 是索引）
        ins_query = self.mode_fuser(ins_query).max(2)[0]
        
        # multi_query_fuser：融合三种 query
        # torch.cat([ins_query, track_query, track_query_pos], dim=-1)
        # 拼接后：(B, Q, 64+256+256) = (B, Q, 576)
        # 注意：这里 ins_query 已经是64维，track_query 是256维，track_query_pos 是256维
        # 实际上拼接后是 64+256+256=576，但代码中写的是 query_dim*3=768
        # （因为 merge 前 ins_query 仍是 256 维，mode_fuser 之前）
        # → MLP → (B, Q, 64)
        ins_query = self.multi_query_fuser(
            torch.cat([ins_query, track_query, track_query_pos], dim=-1)
        )
        
        return ins_query  # shape: (B, Q, 64)

    def forward_train(
                    self,
                    bev_feat,
                    outs_dict,
                    gt_inds_list=None,
                    gt_segmentation=None,
                    gt_instance=None,
                    gt_img_is_valid=None,
                ):
        """训练模式前向传播，返回损失字典。
        
        流程：
          1. get_occ_labels()：将 GT 标签整理为正确格式
          2. merge_queries()：融合三种 query
          3. forward()：核心前向传播，得到占用预测
          4. 对每个样本，按匹配索引对齐 GT 实例掩码与预测掩码
          5. 计算 Dice Loss + Mask Loss（主损失 + 辅助损失）
        
        Args:
            bev_feat (Tensor): BEV 特征图，shape (H*W, B, D)
            outs_dict (dict): MotionHead 的输出（包含各种 query）
            gt_inds_list (list): 每帧每个目标的 GT 实例 ID
            gt_segmentation (Tensor): GT 语义分割标签，shape (B, T, H, W)
                0 = 背景，1 = 前景（有目标）
            gt_instance (Tensor): GT 实例分割标签，shape (B, T, H, W)
                0 = 背景，其他正整数 = 目标实例 ID（如第3个目标 ID=3）
            gt_img_is_valid (Tensor): 每帧是否有效（由于数据增强，部分帧可能无效）
                shape (B, receptive_field + n_future) = (B, 7)
        
        Returns:
            dict: 损失字典，包含：
                'loss_dice': 主 Dice 损失
                'loss_mask': 主 Mask 损失
                'loss_aux_dice': 辅助 Dice 损失（来自 get_attn_mask 的粗预测）
                'loss_aux_mask': 辅助 Mask 损失
        """
        # ---- 整理 GT 标签 ----
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid)
        
        # all_matched_gt_ids：每个 track query 对应的 GT 实例 ID（来自匈牙利匹配）
        # 用于将 GT 实例掩码与预测掩码对齐
        all_matched_gt_ids = outs_dict['all_matched_idxes']  # list，长度=batch_size

        # ---- 融合三种 query，生成 OccHead 的输入 ----
        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
        # ins_query: (B, Q, 64)

        # ---- 核心前向传播 ----
        # mask_preds_batch: 辅助掩码预测，(B, Q, T, 200, 200)
        # ins_seg_preds_batch: 主占用预测 logit，(B, Q, T, 200, 200)
        mask_preds_batch, ins_seg_preds_batch = self(bev_feat, ins_query=ins_query)
        
        # ---- GT 实例分割标签 ----
        # gt_instance: (B, T, H, W)，每个格子的值 = 占用它的目标实例 ID（0=背景）
        ins_seg_targets_batch = gt_instance  # (B, 5, 200, 200)
        
        # ---- 有效帧过滤 ----
        # img_is_valid 标记每帧是否有效（数据增强时边界帧可能无效）
        # 只有当过去3帧全部有效时，才对该样本计算损失
        img_is_valid = gt_img_is_valid  # shape (B, 7)
        assert img_is_valid.size(1) == self.receptive_field + self.n_future, \
            f"Img_is_valid can only be 7 as for loss calculation and evaluation!!! Don't change it"
        
        frame_valid_mask = img_is_valid.bool()  # (B, 7) 布尔掩码
        past_valid_mask = frame_valid_mask[:, :self.receptive_field]    # (B, 3) 过去帧
        future_frame_mask = frame_valid_mask[:, (self.receptive_field-1):]  # (B, 5) 含当前帧的未来帧
        
        # 只有当所有过去帧都有效时，才计算该样本的损失
        past_valid = past_valid_mask.all(dim=1)  # (B,) 布尔值
        future_frame_mask[~past_valid] = False    # 过去帧无效的样本，所有未来帧也标为无效
        
        # ---- 逐样本计算损失 ----
        loss_dict = dict()
        loss_dice = ins_seg_preds_batch.new_zeros(1)[0].float()      # 主 Dice 损失累计
        loss_mask = ins_seg_preds_batch.new_zeros(1)[0].float()      # 主 Mask 损失累计
        loss_aux_dice = ins_seg_preds_batch.new_zeros(1)[0].float()  # 辅助 Dice 损失累计
        loss_aux_mask = ins_seg_preds_batch.new_zeros(1)[0].float()  # 辅助 Mask 损失累计

        bs = ins_query.size(0)
        assert bs == 1  # 目前只支持 batch_size=1

        for ind in range(bs):
            # cur_gt_inds: 当前样本最后一帧的 GT 实例 ID 列表
            # [-1]：取时序中的最后一帧（因为每个样本包含多帧，track 匹配用最后一帧）
            cur_gt_inds = gt_inds_list[ind][-1]

            # cur_matched_gt: 当前样本每个 track query 对应的 GT 实例 ID
            # 这是 MotionHead 中匈牙利匹配的结果
            cur_matched_gt = all_matched_gt_ids[ind]  # shape (Q,)
            
            # 按匹配索引重新排序 GT 实例 ID
            # 让第 j 个预测结果对应正确的 GT 实例
            cur_gt_inds = cur_gt_inds[cur_matched_gt]
            
            # 处理特殊匹配情况：
            # cur_matched_gt == -1：该 query 没有匹配到 GT（误检/新目标）→ 标记为背景
            # cur_matched_gt == -2：该 query 是"无 query"占位符 → 标记为忽略
            cur_gt_inds[cur_matched_gt == -1] = -1   # 未匹配 → 背景（忽略）
            cur_gt_inds[cur_matched_gt == -2] = -2   # 无 query → 特殊忽略

            frame_mask = future_frame_mask[ind]  # (T,) 当前样本各帧的有效性

            # ---- 预测结果和 GT ----
            ins_seg_preds = ins_seg_preds_batch[ind]   # (Q, T, H, W)：预测的占用 logit
            ins_seg_targets = ins_seg_targets_batch[ind]  # (T, H, W)：GT 实例 ID 图
            mask_preds = mask_preds_batch[ind]         # (Q, T, H, W)：辅助掩码预测

            # ---- 将 GT 实例图转换为每个 query 的二值掩码 ----
            # 对每个 track query，找到其对应的 GT 实例 ID，
            # 然后在 GT 实例图中找到该 ID 对应的区域，生成二值掩码
            ins_seg_targets_ordered = []
            for ins_id in cur_gt_inds:
                if (ins_seg_targets == self.ignore_index).all().item() is True:
                    # GT 全为忽略值（该样本完全无效）→ 直接用原始 GT
                    ins_tgt = ins_seg_targets.long()
                elif ins_id.item() in [-1, -2]:
                    # 未匹配的 query（误检）→ 目标 GT 全设为 255（忽略损失）
                    ins_tgt = torch.ones_like(ins_seg_targets).long() * self.ignore_index
                else:
                    SPECIAL_INDEX = -20  # 特殊目标（如遮挡目标）的内部标记
                    if ins_id.item() == self.ignore_index:
                        # GT 中该目标被标记为 255（遮挡/不可见），改为 -20 区分
                        ins_id = torch.ones_like(ins_id) * SPECIAL_INDEX
                    # 在 GT 实例图中找到 ins_id 的区域，生成 0/1 二值掩码
                    # ins_tgt: (T, H, W)，值为 0（背景）或 1（该目标占用该格子）
                    ins_tgt = (ins_seg_targets == ins_id).long()
                
                ins_seg_targets_ordered.append(ins_tgt)
            
            # 堆叠为 (Q, T, H, W) 格式（每个 query 一张时序占用图）
            ins_seg_targets_ordered = torch.stack(ins_seg_targets_ordered, dim=0)
            
            # ---- 尺寸合法性检查 ----
            t, h, w = ins_seg_preds.shape[-3:]
            assert t == 1 + self.n_future, f"{ins_seg_preds.size()}"
            assert ins_seg_preds.size() == ins_seg_targets_ordered.size(), \
                f"{ins_seg_preds.size()}, {ins_seg_targets_ordered.size()}"
            
            # 有效目标数（用于 loss 平均，防止 loss 随目标数量变化）
            num_total_pos = ins_seg_preds.size(0)
            num_total_pos = ins_seg_preds.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
            
            # ---- 计算损失 ----
            # 主 Dice 损失（使用预测 logit 与 GT 二值掩码）
            cur_dice_loss = self.loss_dice(
                ins_seg_preds, ins_seg_targets_ordered, 
                avg_factor=num_total_pos, frame_mask=frame_mask)

            # 主 Mask 损失（像素级二分类交叉熵）
            cur_mask_loss = self.loss_mask(
                ins_seg_preds, ins_seg_targets_ordered, frame_mask=frame_mask)

            # 辅助 Dice 损失（使用 get_attn_mask 的粗预测 mask_preds）
            # 辅助损失监督更早的预测，帮助模型更快收敛
            cur_aux_dice_loss = self.loss_dice(
                mask_preds, ins_seg_targets_ordered, 
                avg_factor=num_total_pos, frame_mask=frame_mask)
            
            # 辅助 Mask 损失
            cur_aux_mask_loss = self.loss_mask(
                mask_preds, ins_seg_targets_ordered, frame_mask=frame_mask)

            loss_dice += cur_dice_loss
            loss_mask += cur_mask_loss
            loss_aux_dice += cur_aux_dice_loss * self.aux_loss_weight  # 辅助损失乘以权重
            loss_aux_mask += cur_aux_mask_loss * self.aux_loss_weight

        # 除以 batch_size 取平均
        loss_dict['loss_dice'] = loss_dice / bs
        loss_dict['loss_mask'] = loss_mask / bs
        loss_dict['loss_aux_dice'] = loss_aux_dice / bs
        loss_dict['loss_aux_mask'] = loss_aux_mask / bs

        return loss_dict

    def forward_test(
                    self,
                    bev_feat,
                    outs_dict,
                    no_query=False,
                    gt_segmentation=None,
                    gt_instance=None,
                    gt_img_is_valid=None,
                ):
        """测试/推理模式前向传播，返回预测结果和 GT 对比字典。
        
        与 forward_train 的区别：
          - 不计算损失，直接输出预测结果
          - 同时输出 GT（用于评估指标计算）
          - 支持 no_query 模式（场景中没有目标时，直接输出全零）
          - 支持用跟踪置信度对占用预测加权
        
        Args:
            bev_feat (Tensor): BEV 特征图
            outs_dict (dict): MotionHead 的输出
            no_query (bool): 是否没有 query（场景中没有被跟踪的目标）
            gt_segmentation: GT 语义分割（用于评估）
            gt_instance: GT 实例分割（用于评估）
            gt_img_is_valid: 帧有效性标记
        
        Returns:
            dict: 包含以下键：
                'seg_gt': GT 语义分割，shape (B, T, 1, H, W)
                'ins_seg_gt': GT 实例分割（连续 ID 版本），shape (B, T, H, W)
                'seg_out': 预测的语义分割结果，shape (B, T, 1, H, W)，值 0/1
                'ins_seg_out': 预测的实例分割结果（全景评估模式下），shape (B, T, H, W)
                'pred_ins_logits': 原始占用 logit，shape (B, Q, T, H, W)
                'pred_ins_sigmoid': sigmoid 后的占用概率，shape (B, Q, T, H, W)
        """
        # ---- 整理 GT 标签 ----
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid)

        out_dict = dict()
        
        # 保存 GT（只取未来 n_future+1 帧）
        out_dict['seg_gt'] = gt_segmentation[:, :1+self.n_future]   # (B, T, 1, H, W)
        out_dict['ins_seg_gt'] = self.get_ins_seg_gt(gt_instance[:, :1+self.n_future])  # (B, T, H, W)
        
        if no_query:
            # 场景中没有被跟踪目标，直接返回全零预测（没有占用）
            out_dict['seg_out'] = torch.zeros_like(out_dict['seg_gt']).long()
            out_dict['ins_seg_out'] = torch.zeros_like(out_dict['ins_seg_gt']).long()
            return out_dict

        # ---- 融合 query 并执行前向传播 ----
        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
        
        # 只需要主占用预测，辅助掩码预测（_）在测试时不使用
        _, pred_ins_logits = self(bev_feat, ins_query=ins_query)
        # pred_ins_logits: (B, Q, T, H, W)

        out_dict['pred_ins_logits'] = pred_ins_logits  # 保存原始 logit 供后续分析

        # 只取未来 n_future+1 帧（T=5）的预测，截掉多余部分
        pred_ins_logits = pred_ins_logits[:, :, :1+self.n_future]  # (B, Q, T, H, W)
        
        # sigmoid 将 logit 转换为 [0,1] 的占用概率
        # pred_ins_sigmoid: (B, Q, T, H, W)，每个值 ∈ (0,1)
        pred_ins_sigmoid = pred_ins_logits.sigmoid()

        if self.test_with_track_score:
            # 用跟踪置信度分数对占用概率加权
            # 跟踪置信度低的目标（可能是误检），其占用预测也应该降低权重
            track_scores = outs_dict['track_scores'].to(pred_ins_sigmoid)  # (B, Q)
            track_scores = track_scores[:, :, None, None, None]  # 扩展维度以便广播
            pred_ins_sigmoid = pred_ins_sigmoid * track_scores   # (B, Q, T, H, W)

        out_dict['pred_ins_sigmoid'] = pred_ins_sigmoid
        
        # 生成语义分割结果：对所有目标取最大占用概率，超过阈值则认为该格子被占用
        # pred_seg_scores: (B, T, H, W)，在 Q 维取最大值（任一目标占用即为占用）
        pred_seg_scores = pred_ins_sigmoid.max(1)[0]
        
        # 二值化：> test_seg_thresh（默认0.5）则为占用（1），否则为背景（0）
        # unsqueeze(2)：增加通道维度，变为 (B, T, 1, H, W) 与 seg_gt 格式一致
        seg_out = (pred_seg_scores > self.test_seg_thresh).long().unsqueeze(2)
        out_dict['seg_out'] = seg_out  # (B, T, 1, H, W)
        
        if self.pan_eval:
            # 全景评估模式：额外生成实例分割结果（每个格子标注属于哪个实例）
            # predict_instance_segmentation_and_trajectories：
            #   基于语义分割 + 各目标的 sigmoid 概率，生成实例级别的分割图
            #   背景 = 0，前景从 1 开始连续编号
            pred_consistent_instance_seg = \
                predict_instance_segmentation_and_trajectories(seg_out, pred_ins_sigmoid)
            # pred_consistent_instance_seg: (B, T, H, W)，实例分割图
            out_dict['ins_seg_out'] = pred_consistent_instance_seg

        return out_dict

    def get_ins_seg_gt(self, gt_instance):
        """将 GT 实例分割图的实例 ID 整理为连续编号（从1开始）。
        
        问题背景：
            原始 GT 实例 ID 是非连续的（如 ID 为 3、7、15……），
            这在评估时会带来不便（空间浪费、不一致）。
            此函数将其映射为连续编号（1、2、3……），背景保持为 0。
        
        Args:
            gt_instance (Tensor): 原始 GT 实例图，shape (B, T, H, W)
                0 = 背景，其他正整数 = 实例 ID（不连续）
        
        Returns:
            ins_gt_new (Tensor): 整理后的 GT 实例图，shape (B, T, H, W)
                0 = 背景，1, 2, 3, ... = 实例 ID（连续）
        """
        ins_gt_old = gt_instance  # 原始非连续实例 ID
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)  # 初始化全零
        
        ins_inds_unique = torch.unique(ins_gt_old)  # 获取所有出现过的实例 ID
        new_id = 1  # 新 ID 从1开始
        
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:  # 跳过背景(0)和忽略值(255)
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id  # 将原 ID 替换为连续新 ID
            new_id += 1
        
        return ins_gt_new  # 连续实例 ID 图

    def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
        """整理 GT 标签的格式，使其符合 OccHead 的输入要求。
        
        主要做两件事：
          1. 测试时从 list 中解包（训练时 GT 已经是 Tensor，测试时被包在列表里）
          2. 截取帧数：只取前 n_future+1 帧（当前帧 + 未来4帧 = 5帧）
          3. 调整维度：为 gt_segmentation 增加通道维度
        
        Args:
            gt_segmentation: GT 语义分割，训练时 shape (B, T_total, H, W)
            gt_instance: GT 实例分割，训练时 shape (B, T_total, H, W)
            gt_img_is_valid: 帧有效性，训练时 shape (B, T_total)
        
        Returns:
            tuple: 整理后的三个 GT 张量
                gt_segmentation: (B, n_future+1, 1, H, W)，增加了通道维度
                gt_instance: (B, n_future+1, H, W)
                gt_img_is_valid: (B, receptive_field + n_future)
        """
        if not self.training:
            # 测试时，GT 被包在列表中（dataset 返回格式不同），需要解包
            gt_segmentation = gt_segmentation[0]
            gt_instance = gt_instance[0]
            gt_img_is_valid = gt_img_is_valid[0]

        # 只取前 n_future+1 帧（5帧），转为 long 类型，增加通道维度
        # unsqueeze(2)：(B, T, H, W) → (B, T, 1, H, W)，第2维是通道
        gt_segmentation = gt_segmentation[:, :self.n_future+1].long().unsqueeze(2)
        
        # 只取前 n_future+1 帧实例标签
        gt_instance = gt_instance[:, :self.n_future+1].long()
        
        # 帧有效性：取 receptive_field + n_future = 3+4 = 7 帧
        gt_img_is_valid = gt_img_is_valid[:, :self.receptive_field + self.n_future]
        
        return gt_segmentation, gt_instance, gt_img_is_valid
    