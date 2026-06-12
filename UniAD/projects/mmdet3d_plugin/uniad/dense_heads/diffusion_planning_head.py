# ================================================================================
# DiffusionPlanningHead —— 融合 UniAD + pl_diffusion_models 的扩散规划头
#
# 方案 A 融合思路（Quick Validation）：
#   UniAD Pipeline 输出（来自 MotionHead）：
#     - bev_embed       : (H*W, B, D=256)  BEV 场景特征
#     - track_query     : (B, N, D=256)    所有 agent 的跟踪特征
#     - sdc_traj_query  : (num_layers, B, P, D)  自车候选轨迹特征
#     - sdc_track_query : (B, D)           自车跟踪特征
#
#   通过 BEV 桥接层 + Track 桥接层，将上述特征转为 DiT 所需的：
#     - context         : (B, N_ctx, D)    场景上下文（agent + BEV token）
#     - ego_context     : (B, 1, D)        自车状态特征
#     - ego_routing     : (B, 1, D)        自车意图/路径特征（由 command 嵌入生成）
#
#   DiT 扩散解码器（Rectified Flow / Flow Matching）输出：
#     - 自车未来 planning_steps 步轨迹（训练：FM MSE loss；推理：Euler ODE 5步）
#
# 接口与 PlanningHeadSingleMode 保持完全兼容，可直接替换配置中的 planning_head
# ================================================================================

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss


# ---- 用原生 PyTorch 实现 tensor 变形（消除 einops 依赖）----

def _bev_seq_to_2d(x, h, w):
    """(h*w, B, D) → (B, D, h, w)"""
    seq, b, d = x.shape
    return x.reshape(h, w, b, d).permute(2, 3, 0, 1).contiguous()


def _bev_2d_to_seq(x):
    """(B, D, h, w) → (h*w, B, D)"""
    b, d, h, w = x.shape
    return x.permute(2, 3, 0, 1).reshape(h * w, b, d).contiguous()


def _bev_2d_to_flat(x):
    """(B, D, h, w) → (B, h*w, D)"""
    b, d, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, d).contiguous()


# ============================================================
# 子模块（来自 pl_diffusion_models，解耦依赖后直接内嵌）
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MLP2(nn.Module):
    """两层 MLP，带 GELU 激活和可选 RMSNorm."""
    def __init__(self, in_features, hidden_features=None, out_features=None, use_norm=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.norm = RMSNorm(hidden_features) if use_norm else nn.Identity()
        self.fc2  = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.norm(self.act(self.fc1(x))))


class TimestepEmbedder(nn.Module):
    """将扩散时间步 t ∈ [0, 1000] 嵌入为向量."""
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.sinusoidal_embedding(t, self.freq_dim))


def modulate(x, shift, scale):
    """DiT adaptive layer norm modulation."""
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    单个 DiT Block：AdaNorm Self-Attn → AdaNorm Cross-Attn → AdaNorm FFN。
    AdaNorm 的 shift/scale 由时间步嵌入 y 动态生成。
    """
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.num_heads = heads
        self.self_attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        ffn_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim),
        )
        # y → (shift1, scale1, shift2, scale2, shift3, scale3)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, context, y, self_attn_mask=None, cross_attn_mask=None):
        """
        Args:
            x            : (B, T, D)   ego 轨迹 token
            context      : (B, N, D)   场景上下文
            y            : (B, D)      时间步 conditioning
            self_attn_mask  : (B, T, T)    可选
            cross_attn_mask : (B, T, N)    可选
        """
        s1, c1, s2, c2, s3, c3 = self.adaLN_modulation(y).chunk(6, dim=-1)

        # Self-Attention
        x_mod = modulate(self.norm1(x), s1, c1)
        if self_attn_mask is not None:
            B, T = x.shape[0], x.shape[1]
            sa = (self_attn_mask.unsqueeze(1)
                  .expand(-1, self.num_heads, -1, -1)
                  .reshape(B * self.num_heads, T, T)
                  .float()
                  .masked_fill(self_attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                               .reshape(B * self.num_heads, T, T).bool(), float('-inf')))
            attn_out, _ = self.self_attn(x_mod, x_mod, x_mod, attn_mask=sa)
        else:
            attn_out, _ = self.self_attn(x_mod, x_mod, x_mod)
        x = x + attn_out

        # Cross-Attention
        x_mod = modulate(self.norm2(x), s2, c2)
        attn_out, _ = self.cross_attn(x_mod, context, context)
        x = x + attn_out

        # FFN
        x = x + self.ffn(modulate(self.norm3(x), s3, c3))
        return x


class FinalLayer(nn.Module):
    """DiT 最终输出层：AdaNorm → Linear."""
    def __init__(self, hidden_size, output_size=2):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, y):
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class SinusoidalPE(nn.Module):
    """正弦位置编码（用于 ego 轨迹 token 的时间维度）."""
    def __init__(self, d_model, max_len=80):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


# ============================================================
# 桥接模块：UniAD 特征 → DiT 输入格式
# ============================================================

class BEVToContextBridge(nn.Module):
    """
    将 UniAD 的 BEV 特征图和 track_query 融合为 DiT 的 context。

    输入：
        bev_embed   : (H*W, B, D)   展平的 BEV 特征
        track_query : (B, N, D)     所有 agent 的跟踪特征

    输出：
        context  : (B, N+N_bev, D)   合并的场景上下文
        mask_ctx : (B, N+N_bev, N+N_bev)   全可见 mask
    """
    def __init__(self, bev_in_dim=256, track_in_dim=256, out_dim=256,
                 n_bev_tokens=64, bev_h=200, bev_w=200, max_agents=50):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        # BEV 特征压缩：200×200 → 8×8=64 tokens（两次 5× stride Conv）
        self.bev_compress = nn.Sequential(
            nn.Conv2d(bev_in_dim, out_dim, kernel_size=5, stride=5, padding=0),  # 200→40
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=5, stride=5, padding=0),     # 40→8
            nn.ReLU(),
        )
        compressed_h = bev_h // 5 // 5   # 8
        compressed_w = bev_w // 5 // 5   # 8
        self.actual_bev_tokens = compressed_h * compressed_w  # 64

        # Track 特征投影
        self.track_proj = nn.Sequential(nn.Linear(track_in_dim, out_dim), nn.LayerNorm(out_dim))

        # BEV token 可学习位置编码
        self.bev_pos_embed = nn.Parameter(torch.randn(1, self.actual_bev_tokens, out_dim) * 0.02)

        # Agent 内部 Transformer（agent 间交互，1层，显存优化：nhead 4→8/4，FFN 缩小）
        # nhead=4: 256/4=64，整除；FFN=256（原512）；层数=1（原2）→ 节省约50%显存
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=out_dim,
            dropout=0.1, batch_first=True
        )
        self.agent_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, bev_embed, track_query, track_mask=None):
        B = track_query.shape[0]
        H, W = self.bev_h, self.bev_w

        # 1. 压缩 BEV 特征
        bev_2d     = _bev_seq_to_2d(bev_embed, H, W)          # (B, D, H, W)
        bev_tokens = self.bev_compress(bev_2d)                 # (B, D, 8, 8)
        bev_tokens = _bev_2d_to_flat(bev_tokens)               # (B, 64, D)
        bev_tokens = bev_tokens + self.bev_pos_embed[:, :bev_tokens.shape[1]]

        # 2. 投影 track 特征
        track_feat = self.track_proj(track_query)              # (B, N, D)

        # 3. Agent 内部交互（N=0 时跳过，避免 Transformer 空序列报错）
        if track_feat.shape[1] > 0:
            kpm = (~track_mask) if track_mask is not None else None
            track_feat = self.agent_encoder(track_feat, src_key_padding_mask=kpm)

        # 4. 拼接
        context = torch.cat([track_feat, bev_tokens], dim=1)   # (B, N+64, D)
        N_ctx   = context.shape[1]
        mask_ctx = torch.ones(B, N_ctx, N_ctx, device=context.device, dtype=torch.bool)
        return context, mask_ctx


class EgoContextBridge(nn.Module):
    """
    从 UniAD MotionHead 输出中提取自车上下文特征。

    输入：
        sdc_traj_query  : (num_layers, B, P, D)
        sdc_track_query : (B, D)
        command         : (B,)

    输出：
        ego_context : (B, 1, D)
        ego_routing : (B, 1, D)
    """
    def __init__(self, in_dim=256, out_dim=256, n_commands=3):
        super().__init__()
        self.ego_fuser = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )
        self.command_embed = nn.Embedding(n_commands, out_dim)
        self.routing_fuser = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )

    def forward(self, sdc_traj_query, sdc_track_query, command):
        sdc_traj  = sdc_traj_query[-1].max(dim=1)[0]          # (B, D)
        sdc_track = sdc_track_query.detach()                   # (B, D)
        ego_feat  = self.ego_fuser(torch.cat([sdc_traj, sdc_track], dim=-1))  # (B, D)
        cmd_embed = self.command_embed(command)                 # (B, D)
        routing   = self.routing_fuser(torch.cat([ego_feat, cmd_embed], dim=-1))  # (B, D)
        return ego_feat.unsqueeze(1), routing.unsqueeze(1)     # (B,1,D), (B,1,D)


# ============================================================
# 主模块：DiffusionPlanningHead
# ============================================================

@HEADS.register_module()
class DiffusionPlanningHead(nn.Module):
    """
    基于 Flow Matching（Rectified Flow）的自车规划头。

    替换 PlanningHeadSingleMode，接口完全兼容：
        forward_train(bev_embed, outs_motion, sdc_planning, sdc_planning_mask, command, gt_future_boxes)
        forward_test (bev_embed, outs_motion, outs_occflow, command)

    核心流程（方案 A）：
        1. BEV(200×200) → Conv 压缩 → 64 BEV token
        2. track_query(N个agent) → Agent Encoder → agent 特征
        3. 拼接 → DiT context (N+64, D)
        4. sdc_traj/track_query + command → ego_context(1,D) + ego_routing(1,D)
        5. DiT Flow Matching：
           训练：GT δ轨迹 + 噪声插值 → DiT 预测 drift → MSE loss
           推理：高斯噪声 → Euler ODE(5步) → 累积求和 → 绝对坐标轨迹
    """

    def __init__(
        self,
        embed_dims=256,
        planning_steps=6,
        n_bev_tokens=64,
        dit_depth=4,
        dit_heads=8,
        sample_steps=5,
        bev_h=200,
        bev_w=200,
        max_agents=50,
        output_dim=2,
        flow_matching_loss_weight=1.0,
        loss_planning=None,        # 兼容接口（不使用）
        loss_collision=None,       # 碰撞损失（与原接口兼容）
        loss_kinematic=None,       # 兼容接口（不使用）
        planning_eval=False,
        use_col_optim=False,
        col_optim_args=dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0),
        with_adapter=False,
        n_commands=3,
    ):
        super().__init__()
        self.planning_steps            = planning_steps
        self.sample_steps              = sample_steps
        self.output_dim                = output_dim
        self.flow_matching_loss_weight = flow_matching_loss_weight
        self.planning_eval             = planning_eval
        self.use_col_optim             = use_col_optim
        self.embed_dims                = embed_dims

        # ---- 桥接层 ----
        self.bev_track_bridge = BEVToContextBridge(
            bev_in_dim=embed_dims, track_in_dim=embed_dims, out_dim=embed_dims,
            n_bev_tokens=n_bev_tokens, bev_h=bev_h, bev_w=bev_w, max_agents=max_agents,
        )
        self.ego_bridge = EgoContextBridge(in_dim=embed_dims, out_dim=embed_dims, n_commands=n_commands)

        # ---- DiT 扩散解码器 ----
        self.t_embedder   = TimestepEmbedder(embed_dims)
        self.ego_time_pe  = SinusoidalPE(embed_dims, max_len=planning_steps + 10)
        self.preproj      = MLP2(output_dim, embed_dims, embed_dims)
        self.vector_in    = MLP2(embed_dims, embed_dims, embed_dims)
        self.routing_in   = MLP2(embed_dims, embed_dims, embed_dims)
        self.context_in   = MLP2(embed_dims, embed_dims, embed_dims)
        self.dit_blocks   = nn.ModuleList([
            DiTBlock(dim=embed_dims, heads=dit_heads, mlp_ratio=2.0)
            for _ in range(dit_depth)
        ])
        self.final_layer  = FinalLayer(hidden_size=embed_dims, output_size=output_dim)

        # 轨迹统计量（nuScenes ego 城市场景估计值）
        self.register_buffer(
            'traj_mean', torch.tensor([0.4, 0.0] if output_dim == 2 else [0.4, 0.0, 0.0])
        )
        self.register_buffer(
            'traj_std',  torch.tensor([0.3, 0.15] if output_dim == 2 else [0.3, 0.15, 0.1])
        )

        # ---- 碰撞损失（兼容接口）----
        self.loss_collision = nn.ModuleList()
        if loss_collision is not None:
            for cfg in loss_collision:
                self.loss_collision.append(build_loss(cfg))

        # ---- BEV Adapter（可选）----
        self.with_adapter = with_adapter
        if with_adapter:
            blk = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, 1),
            )
            self.bev_adapter = nn.Sequential(*[copy.deepcopy(blk) for _ in range(3)])

    # ------------------------------------------------------------------
    # 辅助：轨迹归一化 / 反归一化
    # ------------------------------------------------------------------

    def normalize_traj(self, delta):
        return (delta - self.traj_mean) / (self.traj_std + 1e-6)

    def denormalize_traj(self, delta_norm):
        return delta_norm * self.traj_std + self.traj_mean

    # ------------------------------------------------------------------
    # DiT 单步前向（预测 drift）
    # ------------------------------------------------------------------

    def _dit_forward(self, z_t, context, t_emb, ego_context, ego_routing):
        """
        Args:
            z_t         : (B, T, output_dim)  噪声轨迹
            context     : (B, N_ctx, D)
            t_emb       : (B, D)
            ego_context : (B, 1, D)
            ego_routing : (B, 1, D)
        Returns:
            pred : (B, T, output_dim)  预测 drift
        """
        B, T, _ = z_t.shape
        x   = self.preproj(z_t)                                     # (B, T, D)
        x   = self.ego_time_pe(x)
        x   = x + self.vector_in(ego_context).expand(-1, T, -1)     # 加 ego 状态
        ctx = self.context_in(context)                               # (B, N_ctx, D)
        y   = t_emb + self.routing_in(ego_routing).squeeze(1)       # (B, D)
        for block in self.dit_blocks:
            x = block(x, ctx, y)
        return self.final_layer(x, y)                                # (B, T, output_dim)

    # ------------------------------------------------------------------
    # ODE 采样（推理）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_traj(self, context, ego_context, ego_routing, B, device, n_steps=5):
        """从高斯噪声出发，Euler ODE 积分采样，返回绝对坐标轨迹 (B, T, output_dim)。"""
        T = self.planning_steps
        x = torch.randn(B, T, self.output_dim, device=device)
        for i in range(n_steps):
            t_val = float(i) / n_steps
            t     = torch.full((B,), t_val * 1000.0, device=device)
            t_emb = self.t_embedder(t)
            drift = self._dit_forward(x, context, t_emb, ego_context, ego_routing)
            x     = x + drift * (1.0 / n_steps)
        delta   = self.denormalize_traj(x)
        traj    = torch.cumsum(delta, dim=1)
        return traj

    # ------------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------------

    def forward_train(self, bev_embed, outs_motion, sdc_planning, sdc_planning_mask,
                      command, gt_future_boxes=None):
        """
        训练入口（接口与 PlanningHeadSingleMode.forward_train 完全一致）。

        Args:
            bev_embed        : (H*W, B, D)
            outs_motion      : dict  含 sdc_traj_query/sdc_track_query/bev_pos/track_query
            sdc_planning     : (B, 1, planning_steps, 3)  GT 轨迹 [x,y,heading]
            sdc_planning_mask: (B, 1, planning_steps, 1)  有效掩码
            command          : (B,)  驾驶命令
            gt_future_boxes  : list  碰撞损失用（可选）

        Returns:
            dict: {'losses': {...}, 'outs_motion': {...}}
        """
        sdc_traj_query  = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        track_query     = outs_motion.get('track_query', None)
        B               = sdc_track_query.shape[0]

        # BEV Adapter（可选）
        if self.with_adapter:
            bh, bw = self.bev_track_bridge.bev_h, self.bev_track_bridge.bev_w
            bev_2d   = _bev_seq_to_2d(bev_embed, bh, bw)
            bev_2d   = bev_2d + self.bev_adapter(bev_2d)
            bev_embed = _bev_2d_to_seq(bev_2d)

        if track_query is None:
            track_query = torch.zeros(B, 1, self.embed_dims, device=bev_embed.device)

        context, _  = self.bev_track_bridge(bev_embed, track_query)
        ego_ctx, eg_rout = self.ego_bridge(sdc_traj_query, sdc_track_query, command)

        # 准备 GT 轨迹 delta（累积坐标 → 逐步增量 → 归一化）
        gt_traj    = sdc_planning[:, 0, :, :self.output_dim]         # (B, T, 2)
        traj_mask  = sdc_planning_mask[:, 0, :, 0].bool()            # (B, T)
        gt_delta   = torch.zeros_like(gt_traj)
        gt_delta[:, 0, :]  = gt_traj[:, 0, :]
        gt_delta[:, 1:, :] = gt_traj[:, 1:, :] - gt_traj[:, :-1, :]
        gt_norm    = self.normalize_traj(gt_delta) * traj_mask.unsqueeze(-1).float()

        # Flow Matching：插值 + DiT 预测
        noise   = torch.randn_like(gt_norm)
        t       = torch.sigmoid(torch.randn(B, device=gt_norm.device))
        z_t     = t.view(B, 1, 1) * gt_norm + (1 - t.view(B, 1, 1)) * noise
        target  = gt_norm - noise
        t_emb   = self.t_embedder(t * 1000.0)
        pred    = self._dit_forward(z_t, context, t_emb, ego_ctx, eg_rout)

        # Flow Matching MSE Loss（仅在有效步计算）
        mask_e  = traj_mask.unsqueeze(-1).float()
        fm_loss = ((pred - target) ** 2 * mask_e).sum() / (mask_e.sum() * self.output_dim + 1e-6)
        fm_loss = fm_loss * self.flow_matching_loss_weight
        losses  = {'loss_flow_matching': fm_loss}

        # 碰撞损失（可选）
        # CollisionLoss.forward(sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox)
        # - sdc_traj_all: 预测轨迹 (B, T, 2)（绝对坐标）
        # - sdc_planning_gt: GT 轨迹 (B, T, 3)（含 heading，用于 sdc_yaw）
        # - sdc_planning_gt_mask: GT 轨迹有效掩码 (B, T)
        # - future_gt_bbox: GT 未来障碍物框（list）
        if len(self.loss_collision) > 0 and gt_future_boxes is not None:
            with torch.no_grad():
                s_traj = self._sample_traj(context, ego_ctx, eg_rout, B, gt_norm.device, n_steps=3)
            # gt_traj: (B, T, 2)，需要扩充为 (B, T, 3)（补零 heading）
            gt_traj_3d = torch.cat([gt_traj, torch.zeros_like(gt_traj[..., :1])], dim=-1)
            for i, col_fn in enumerate(self.loss_collision):
                losses[f'loss_collision_{i}'] = col_fn(
                    s_traj,           # (B, T, 2): 预测轨迹（绝对坐标）
                    gt_traj_3d,       # (B, T, 3): GT 轨迹（含伪 heading=0）
                    traj_mask,        # (B, T): 有效掩码
                    gt_future_boxes   # list: GT 未来障碍物框
                )

        # 构造与 PlanningHeadSingleMode 兼容的返回格式
        # pred_traj: (B, T, 2)，与原 PlanningHeadSingleMode 输出格式完全一致
        with torch.no_grad():
            pred_traj = self._sample_traj(context, ego_ctx, eg_rout, B,
                                           gt_norm.device, n_steps=self.sample_steps)
        outs_planning = {
            'sdc_traj':     pred_traj,   # (B, T, 2) — 与评估接口兼容
            'sdc_traj_all': pred_traj,
        }
        return dict(losses=losses, outs_motion=outs_planning)

    # ------------------------------------------------------------------
    # 推理前向
    # ------------------------------------------------------------------

    def forward_test(self, bev_embed, outs_motion, outs_occflow=None, command=None):
        """
        推理入口（接口与 PlanningHeadSingleMode.forward_test 完全一致）。

        Returns:
            dict: {'sdc_traj': (1, T, 2), 'sdc_traj_all': (1, T, 2)}
        """
        sdc_traj_query  = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        track_query     = outs_motion.get('track_query', None)
        B               = sdc_track_query.shape[0]
        device          = sdc_track_query.device

        if self.with_adapter:
            bh, bw = self.bev_track_bridge.bev_h, self.bev_track_bridge.bev_w
            bev_2d   = _bev_seq_to_2d(bev_embed, bh, bw)
            bev_2d   = bev_2d + self.bev_adapter(bev_2d)
            bev_embed = _bev_2d_to_seq(bev_2d)

        if track_query is None:
            track_query = torch.zeros(B, 1, self.embed_dims, device=device)

        context, _ = self.bev_track_bridge(bev_embed, track_query)
        ego_ctx, eg_rout = self.ego_bridge(sdc_traj_query, sdc_track_query, command)

        pred_traj = self._sample_traj(context, ego_ctx, eg_rout, B, device,
                                       n_steps=self.sample_steps)
        # pred_traj: (B, T, 2)，与原 PlanningHeadSingleMode 输出格式一致
        return {
            'sdc_traj':     pred_traj,   # (B, T, 2)
            'sdc_traj_all': pred_traj,
        }