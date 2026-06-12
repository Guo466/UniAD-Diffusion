import torch
import math
from typing import Optional, Tuple, Union
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np

#***********************************
#******** DeepseekMOE utils ********
#***********************************
class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss
    
# ****************
# ***CoPE utils***
# ****************
class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.parameter.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        pos_ceil = pos.ceil().long() # [B,H,L]
        pos_floor = pos.floor().long() # [B,H,L]
        logits_int = torch.matmul(query, self.pos_emb) # [B,L,D] -> [B,L,P]
        logits_ceil = logits_int.gather(-1, pos_ceil)# [B,L,L] -> [B,L,L]
        logits_floor = logits_int.gather(-1, pos_floor) # [B,H,L,D] -> [B,H,L,D]
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w )


@torch.no_grad()
def compute_intent_mask(attn_logits, npos_max, use_cope: bool = False):
    '''
    generate mask with cope method
    attn_logits: [B,M,A,T,D]
    return: [B,M,A,I,T]
    '''
    B,M,A,T = attn_logits.shape[:4]
    if use_cope:
        sim_TT = attn_logits @ attn_logits.transpose(-2, -1)  # [B,M,A,T,T]
        scale = attn_logits.size(-1) ** -0.5
        log_p_T = F.log_softmax(sim_TT * scale, dim=-1)
        sim_TT_next = torch.roll(sim_TT, shifts=-1, dims=-2)
        sim_TT_next[..., -1, :] = sim_TT[..., -1, :]
        q_T_next = F.softmax(sim_TT_next * scale, dim=-1)
        scaled_attn_similarity = F.kl_div(log_p_T, q_T_next, reduction='none').sum(dim=-1)
        thresh = scaled_attn_similarity.mean(dim=-1, keepdim=True) # + 0.5 * scaled_attn_similarity.std(dim=-1, keepdim=True)
        is_boundary = (scaled_attn_similarity > thresh).to(torch.int)    # 1 = different, 0 = same
        clip_idx = is_boundary.cumsum(dim=-1)
        clip_idx_start_shift, _ = clip_idx.min(dim=-1, keepdim=True) 
        clip_idx = clip_idx - clip_idx_start_shift
        clip_idx = clip_idx.clamp_max(npos_max - 1)
        intent_mask = torch.zeros(attn_logits.shape[:-2] + (npos_max,attn_logits.shape[-2]),dtype=int,device=attn_logits.device,requires_grad=False) # [B,M,A,I,T]
        for i in range(npos_max):
            intent_mask[...,i,:] = (clip_idx == i).int()
    else:
        intent_mask = torch.eye((npos_max), device=attn_logits.device,requires_grad=False) # [I,I]
        intent_mask = intent_mask.repeat_interleave(int(T/npos_max), dim=1) # [I,T], every I can only attend to T/I points
        intent_mask = intent_mask.repeat(B,M,A,1,1) # [B,M,A,I,T]
    return intent_mask

# ***************************   
# ***Transformer RPE utils***
# ***************************
# KNARPE
class AttentionRPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout_p: float = 0.0,
        bias: bool = True,
        d_rpe: int = -1,
        apply_q_rpe: bool = False,
    ) -> None:
        """
        Always batch first. Always src and tgt have the same d_model.
        """
        super(AttentionRPE, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.apply_q_rpe = apply_q_rpe
        self.d_rpe = d_rpe

        assert self.d_head * n_head == d_model, "d_model must be divisible by n_head"

        if self.d_rpe > 0:
            n_project_rpe = 3 if apply_q_rpe else 2
            self.mlp_rpe = nn.Linear(d_rpe, n_project_rpe * d_model, bias=bias)

        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_model, d_model)))
        self.out_proj_weight = nn.Parameter(torch.empty((d_model, d_model)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
            self.out_proj_bias = nn.Parameter(torch.empty(d_model))
        else:
            self.register_parameter("in_proj_bias", None)
            self.register_parameter("out_proj_bias", None)

        self.dropout = nn.Dropout(p=dropout_p, inplace=False) if dropout_p > 0 else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
        if self.out_proj_bias is not None:
            nn.init.constant_(self.out_proj_bias, 0.0)

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
        need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            tgt: [n_batch, (n_src), n_tgt, d_model], None for self attention, (n_src) if using rpe.
            tgt_padding_mask: [n_batch, (n_src), n_tgt], bool, if True, tgt is invalid, (n_src) if using rpe.
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.
            rpe: [n_batch, n_src, n_tgt, d_rpe]

        Returns:
            out: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None

        Remarks:
            absoulte_pe should be already added to src/tgt.
            if for a batch entry all tgt are invalid, then returns 0 for that batch entry.
        """
        n_batch, n_src, _ = src.shape
        if tgt is None:
            n_tgt = n_src
            # self-attention
            qkv = F.linear(src, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            n_tgt = tgt.shape[-2]
            # encoder-decoder attention
            w_src, w_tgt = self.in_proj_weight.split([self.d_model, self.d_model * 2])
            b_src, b_tgt = None, None
            if self.in_proj_bias is not None:
                b_src, b_tgt = self.in_proj_bias.split([self.d_model, self.d_model * 2])
            q = F.linear(src, w_src, b_src)
            kv = F.linear(tgt, w_tgt, b_tgt)
            k, v = kv.chunk(2, dim=-1)
        # q: [n_batch, n_src, d_model], k,v: [n_batch, (n_src), n_tgt, d_model]

        attn_invalid_mask = None  # [n_batch, n_src, n_tgt]
        if tgt_padding_mask is not None:  # [n_batch, n_tgt], bool
            attn_invalid_mask = tgt_padding_mask
            if attn_invalid_mask.dim() == 2:
                attn_invalid_mask = attn_invalid_mask.unsqueeze(1).expand(-1, n_src, -1)
        if attn_mask is not None:  # [n_batch, n_src, n_tgt], bool
            if attn_invalid_mask is None:
                attn_invalid_mask = attn_mask
            else:
                attn_invalid_mask = attn_invalid_mask | attn_mask

        mask_no_tgt_valid = None  # [n_batch, n_src]
        if attn_invalid_mask is not None:
            mask_no_tgt_valid = attn_invalid_mask.all(-1)
            if mask_no_tgt_valid.any():
                attn_invalid_mask = attn_invalid_mask & (~mask_no_tgt_valid.unsqueeze(-1))  # to avoid softmax nan
            else:
                mask_no_tgt_valid = None

        # get attn: [n_batch, n_head, n_src, n_tgt]
        if rpe is None:
            if k.dim() == 3:
                # ! normal attention; q: [n_batch, n_src, d_model], k,v: [n_batch, n_tgt, d_model]
                q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).contiguous()
                k = k.view(n_batch, n_tgt, self.n_head, self.d_head).transpose(1, 2).contiguous()
                v = v.view(n_batch, n_tgt, self.n_head, self.d_head).transpose(1, 2).contiguous()
                attn = torch.matmul(q, k.transpose(-2, -1))  # [n_batch, n_head, n_src, n_tgt]
                # q: [n_batch, n_head, n_src, d_head], k,v: [n_batch, n_head, n_tgt, d_head]
            else:
                # ! KNN attention; q: [n_batch, n_src, d_model], k,v: [n_batch, n_src, n_tgt, d_model]
                # k,v: [n_batch, n_src, n_tgt, d_model] -> [n_batch, n_head, n_src, n_tgt_knn, d_head]
                k = k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
                v = v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
                # [n_batch, n_src, d_model] -> [n_batch, n_head, n_src, 1, d_head]
                q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).unsqueeze(3)
                attn = torch.sum(q * k, dim=-1)  # [n_batch, n_head, n_src, n_tgt_knn]
        else:
            # ! rpe attention; q: [n_batch, n_src, d_model], k,v: [n_batch, n_tgt, d_model]
            assert self.d_rpe > 0
            # k,v: [n_batch, n_src, n_tgt, d_model] -> [n_batch, n_head, n_src, n_tgt_knn, d_head]
            k = k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            v = v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            # [n_batch, n_src, d_model] -> [n_batch, n_head, n_src, 1, d_head]
            q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).unsqueeze(3)

            # project rpe to rpe_q, rpe_k, rpe_v: [n_batch, n_head, n_src, n_tgt, d_head]
            rpe = self.mlp_rpe(rpe)
            if self.apply_q_rpe:
                rpe_q, rpe_k, rpe_v = rpe.chunk(3, dim=-1)
                rpe_q = rpe_q.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            else:
                rpe_k, rpe_v = rpe.chunk(2, dim=-1)
            rpe_k = rpe_k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            rpe_v = rpe_v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)

            # get attn: [n_batch, n_head, n_src, n_tgt]
            if self.apply_q_rpe:
                attn = torch.sum((q + rpe_q) * (k + rpe_k), dim=-1)
                # attn = torch.sum((q + rpe_q) * (k + rpe_k) - rpe_q * rpe_k, dim=-1)
            else:
                attn = torch.sum(q * (k + rpe_k), dim=-1)
            # q: [n_batch, n_head, n_src, 1, d_head]
            # k,v: [n_batch, n_head, n_src, n_tgt, d_head]
            # rpe_q, rpe_k, rpe_v: [n_batch, n_head, n_src, n_tgt, d_head]

        if attn_invalid_mask is not None:
            # attn_invalid_mask: [n_batch, n_src, n_tgt], attn: [n_batch, n_head, n_src, n_tgt]
            attn = attn.masked_fill(attn_invalid_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(attn / math.sqrt(self.d_head), dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # attn: [n_batch, n_head, n_src, n_tgt]
        if rpe is None:
            if v.dim() == 4:
                out = torch.matmul(attn, v)  # v, [n_batch, n_head, n_tgt, d_head]
            else:
                out = torch.sum(v * attn.unsqueeze(-1), dim=3)  # v: [n_batch, n_head, n_src, n_tgt, d_head]
        else:
            # v, rpe_v: [n_batch, n_head, n_src, n_tgt, d_head]
            out = torch.sum((v + rpe_v) * attn.unsqueeze(-1), dim=3)

        # out: [n_batch, n_head, n_src, d_head]
        out = out.transpose(1, 2).flatten(2, 3)  # [n_batch, n_src, d_model]
        out = F.linear(out, self.out_proj_weight, self.out_proj_bias)

        if mask_no_tgt_valid is not None:
            # mask_no_tgt_valid: [n_batch, n_src], out: [n_batch, n_src, d_model]
            out = out.masked_fill(mask_no_tgt_valid.unsqueeze(-1), 0)

        if need_weights:
            attn_weights = attn.mean(1)  # [n_batch, n_src, n_tgt]
            if mask_no_tgt_valid is not None:
                attn_weights = attn_weights.masked_fill(mask_no_tgt_valid.unsqueeze(-1), 0)
            return out, attn_weights
        else:
            return out, None


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int,
        dropout_p: float,
        activation: str,
        norm_first: bool,
        decoder_self_attn: bool,
        bias: bool,
        d_rpe: int = -1,
        apply_q_rpe: bool = False,
    ) -> None:
        super(TransformerCrossAttention, self).__init__()
        self.norm_first = norm_first
        self.d_feedforward = d_feedforward
        self.decoder_self_attn = decoder_self_attn
        inplace = False

        self.dropout = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
        self.activation = _get_activation_fn(activation)
        self.norm1 = nn.LayerNorm(d_model)

        if self.decoder_self_attn:
            self.attn_src = AttentionRPE(
                d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias, d_rpe=d_rpe, apply_q_rpe=apply_q_rpe
            )
            self.norm_src = nn.LayerNorm(d_model)
            self.dropout_src = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

        if self.norm_first:
            self.norm_tgt = nn.LayerNorm(d_model)

        self.attn = AttentionRPE(
            d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias, d_rpe=d_rpe, apply_q_rpe=apply_q_rpe
        )
        if self.d_feedforward > 0:
            self.linear1 = nn.Linear(d_model, d_feedforward)
            self.linear2 = nn.Linear(d_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
            self.dropout2 = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        decoder_rpe: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            src_padding_mask: [n_batch, n_src], bool, if True, src is invalid.
            tgt: [n_batch, (n_src), n_tgt, d_model], None for self attention, (n_src) if using rpe.
            tgt_padding_mask: [n_batch, (n_src), n_tgt], bool, if True, tgt is invalid, (n_src) if using rpe.
            rpe: [n_batch, n_src, n_tgt, d_rpe]
            decoder_tgt: [n_batch, n_src, n_tgt_decoder, d_model], when use decoder_rpe
            decoder_tgt_padding_mask: [n_batch, n_src, n_tgt_decoder], when use decoder_rpe
            decoder_rpe: [n_batch, n_src, n_tgt_decoder, d_rpe]
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.

        Returns:
            out: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None

        Remarks:
            absoulte_pe should be already added to src/tgt.
        """
        if self.decoder_self_attn:
            # transformer decoder
            if self.norm_first:
                _s = self.norm_src(src)
                if decoder_tgt is None:
                    _s = self.attn_src(_s, tgt_padding_mask=src_padding_mask)[0]
                else:
                    decoder_tgt = self.norm_src(decoder_tgt)
                    _s = self.attn_src(_s, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask, rpe=decoder_rpe)[0]

                if self.dropout_src is None:
                    src = src + _s
                else:
                    src = src + self.dropout_src(_s)
            else:
                if decoder_tgt is None:
                    _s = self.attn_src(src, tgt_padding_mask=src_padding_mask)[0]
                else:
                    _s = self.attn_src(src, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask, rpe=decoder_rpe)[0]

                if self.dropout_src is None:
                    src = self.norm_src(src + _s)
                else:
                    src = self.norm_src(src + self.dropout_src(_s))

        if tgt is None:
            tgt_padding_mask = src_padding_mask

        if self.norm_first:
            src2 = self.norm1(src)
            if tgt is not None:
                tgt = self.norm_tgt(tgt)
        else:
            src2 = src

        # [n_batch, n_src, d_model]
        src2, attn_weights = self.attn(
            src=src2,
            tgt=tgt,
            tgt_padding_mask=tgt_padding_mask,
            attn_mask=attn_mask,
            rpe=rpe,
            need_weights=need_weights,
        )

        if self.d_feedforward > 0:
            if self.dropout1 is None:
                src = src + src2
            else:
                src = src + self.dropout1(src2)

            if self.norm_first:
                src2 = self.norm2(src)
            else:
                src = self.norm1(src)
                src2 = src

            src2 = self.activation(self.linear1(src2))
            if self.dropout is None:
                src2 = self.linear2(src2)
            else:
                src2 = self.linear2(self.dropout(src2))

            if self.dropout2 is None:
                src = src + src2
            else:
                src = src + self.dropout2(src2)

            if not self.norm_first:
                src = self.norm2(src)
        else:
            # densetnt vectornet
            src2 = self.activation(src2)
            if self.dropout is None:
                src = src + src2
            else:
                src = src + self.dropout(src2)
            if not self.norm_first:
                src = self.norm1(src)

        if src_padding_mask is not None:
            src.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
            if need_weights:
                attn_weights.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
        return src, attn_weights

# ************************
# ***Transformer utils***
# ************************
   
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num=8, bias=True, depth=1, pos_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size  = hidden_size
        self.head_num     = head_num
        assert self.hidden_size % self.head_num == 0
        self.head_size    = self.hidden_size // self.head_num
        self.pos_dim = pos_dim
        if self.pos_dim is not None:
            self.rope_enc = RotatePosEnc(head_size = self.head_size, pos_dim = self.pos_dim)   
        self.scale_factor = self.head_size ** -0.5
        # W_q, W_k, W_v
        self.q_proj       = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.k_proj       = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.v_proj       = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        # Output projection
        self.o_proj       = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.o_proj.RESIDUAL_SCALE = depth

    def forward(self, q, k, v, mask=None, query_pos=None, key_pos=None):
        original_q_shape = q.shape # [B, T, q_numel, D] or [B, q_numel, D]
        pseudo_batch = original_q_shape[:-2] # [B, T] or [B, ]

        # Project input query, key, value by W_q, W_k, W_v
        # and reshape hidden size to multi-head size
        q = self.q_proj(q).view(*pseudo_batch, -1, self.head_num, self.head_size).transpose(-3, -2) # [B, T, h, q_numel, d] or [B, h, q_numel, d]
        k = self.k_proj(k).view(*pseudo_batch, -1, self.head_num, self.head_size).transpose(-3, -2) # [B, T, h, k_numel, d] or [B, h, k_numel, d]
        v = self.v_proj(v).view(*pseudo_batch, -1, self.head_num, self.head_size).transpose(-3, -2) # [B, T, h, k_numel, d] or [B, h, k_numel, d]
        if self.pos_dim is not None:
            # Apply RoPE to q and k with global xy
            q, k = self.rope_enc(q,k,query_pos,key_pos)
        # Scaled dot product attention. original version
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale_factor # [B, T, h, q_numel, k_numel] or [B, h, q_numel, k_numel]
        if mask is not None:
            mask = mask.unsqueeze(dim=-3)                                     # [B, T, 1, q_numel, k_numel] or [B, 1, q_numel, k_numel]
            attn_scores = attn_scores.masked_fill(mask==0, value=-1e16)
        attn_probs = torch.softmax(attn_scores, dim=-1)                       # [B, T, h, q_numel, k_numel] or [B, h, q_numel, k_numel]
        output = torch.matmul(attn_probs, v)                                  # [B, T, h, q_numel, d] or [B, h, q_numel, d]

        # flash version
        # if mask is not None:
        #     # float mask with 1.0 or 0.0
        #     mask = mask.unsqueeze(dim=-3) # for broadcast
        #     mask = mask.masked_fill(mask==0, value=torch.finfo(mask.dtype).min/2) # '0' -> -inf
        #     mask = mask.masked_fill(mask==1, value=torch.tensor(0, dtype=mask.dtype)) # '1' -> 0
        #     output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,scale=self.scale_factor)
        # else:
        #     output = torch.nn.functional.scaled_dot_product_attention(q, k, v,scale=self.scale_factor)
        
        # Reshape multi-head size to hidden size and project output 
        output = output.transpose(-3,-2).contiguous().view(original_q_shape)  # [B, T, q_numel, D] or [B, q_numel, D]
        return self.o_proj(output)
    
class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, bias=True, depth=1):
        super(PointWiseFeedForward, self).__init__()
        self.w_1  = nn.Linear(d_model, d_ffn, bias=bias)
        self.w_2  = nn.Linear(d_ffn, d_model, bias=bias)
        self.w_2.RESIDUAL_SCALE = depth
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, depth=1, pos_dim=None):
        super(TransformerEncoderLayer, self).__init__()
        self.pos_dim = pos_dim
        if self.pos_dim is not None:
            self.self_attn      = MultiHeadAttention(hidden_size=hidden_size, depth=depth, pos_dim=self.pos_dim)
        else:
            self.self_attn      = MultiHeadAttention(hidden_size=hidden_size, depth=depth)
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.ffn            = PointWiseFeedForward(d_model=hidden_size,d_ffn=hidden_size, depth=depth)
        self.ffn_norm       = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None, query_pos=None, key_pos=None):
        # Self attention and residual connection
        if self.pos_dim is not None:
            self_attn_layer = lambda x: self.self_attn(q=x, k=x, v=x, mask=mask, query_pos=query_pos, key_pos=key_pos)
        else:   
            self_attn_layer = lambda x: self.self_attn(q=x, k=x, v=x, mask=mask)
        x = x + self_attn_layer(self.self_attn_norm(x))
        # Feed forward network and residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x
class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, enable_self_attn=False, depth=3, pos_dim=None):
        super(TransformerDecoderLayer, self).__init__()
        self.enable_self_attn   = enable_self_attn
        self.pos_dim = pos_dim
        if enable_self_attn:
            if self.pos_dim is not None:
                self.self_attn      = MultiHeadAttention(hidden_size=hidden_size, depth=depth, pos_dim=self.pos_dim)
            else:
                self.self_attn      = MultiHeadAttention(hidden_size=hidden_size, depth=depth)
            self.self_attn_norm = nn.LayerNorm(hidden_size)
        if self.pos_dim is not None:
            self.cross_attn         = MultiHeadAttention(hidden_size=hidden_size, depth=depth, pos_dim=self.pos_dim)
        else:
            self.cross_attn         = MultiHeadAttention(hidden_size=hidden_size, depth=depth)
        self.cross_attn_norm    = nn.LayerNorm(hidden_size)
        self.ffn                = PointWiseFeedForward(d_model=hidden_size,d_ffn=hidden_size, depth=depth)
        self.ffn_norm           = nn.LayerNorm(hidden_size)

    def forward(self, x, memory, x_mask=None, memory_mask=None, query_pos=None, key_pos=None):
        # Self attention and residual connection
        if self.enable_self_attn:
            if self.pos_dim is not None:
                self_attn_layer = lambda x: self.self_attn(q=x, k=x, v=x, mask=x_mask, query_pos=query_pos, key_pos=key_pos)
            else:
                self_attn_layer = lambda x: self.self_attn(q=x, k=x, v=x, mask=x_mask)
            x = x + self_attn_layer(self.self_attn_norm(x))
        # Cross attention and residual connection
        if self.pos_dim is not None:
            cross_attn_layer = lambda x: self.cross_attn(q=x, k=memory, v=memory, mask=memory_mask, query_pos=query_pos, key_pos=key_pos)
        else:
            cross_attn_layer = lambda x: self.cross_attn(q=x, k=memory, v=memory, mask=memory_mask)
        x = x + cross_attn_layer(self.cross_attn_norm(x))
        # Feed forward network and redisual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, depth, pos_dim=None):
        super(TransformerEncoder, self).__init__()
        self.pos_dim = pos_dim
        if self.pos_dim is not None:
            self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size, depth=depth, pos_dim=self.pos_dim) for _ in range(depth)])
        else:
            self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size, depth=depth) for _ in range(depth)])

    def forward(self, x, mask=None, query_pos=None, key_pos=None):
        for layer in self.layers:
            if self.pos_dim is not None:
                x = layer(x, mask=mask, query_pos=query_pos, key_pos=key_pos)
            else:
                x = layer(x, mask=mask)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, depth, enable_self_attn=False, pos_dim=None):
        super(TransformerDecoder, self).__init__()
        self.pos_dim = pos_dim
        if self.pos_dim is not None:
            self.layers = nn.ModuleList([TransformerDecoderLayer(hidden_size, enable_self_attn, depth=depth, pos_dim=self.pos_dim) for _ in range(depth)])
        else:
            self.layers = nn.ModuleList([TransformerDecoderLayer(hidden_size, enable_self_attn, depth=depth) for _ in range(depth)])
        self.memory_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, memory, x_mask=None, memory_mask=None, query_pos=None, key_pos=None):
        memory = self.memory_norm(memory)
        for layer in self.layers:
            if self.pos_dim is not None:
                x = layer(x, memory, x_mask=x_mask, memory_mask=memory_mask, query_pos=query_pos, key_pos=key_pos)
            else:
                x = layer(x, memory, x_mask=x_mask, memory_mask=memory_mask)
        return x

class RotatePosEnc(nn.Module):
    """
    2D RoPE (Rotary Position Embedding) for continuous XY coordinates
    """
    def __init__(self, head_size: int = 128, theta: float = 10000.0, pos_dim: int = 2):
        super(RotatePosEnc, self).__init__()
        self.head_size = head_size
        self.theta = theta
        self.pos_dim = pos_dim
        # Precompute frequencies for both x and y dimensions
        # Each dimension gets half of the head_size
        if self.pos_dim == 2:
            freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_size, 4, dtype=torch.float32)[: (self.head_size // 4)] / self.head_size))
        else:
            freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_size, 2)[: (self.head_size // 2)].float() / self.head_size))
        
        # Create frequencies for x and y coordinates
        self.register_buffer('freqs', freqs, persistent=False)


    def precompute_freqs_cis_1d(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Precompute frequency complex numbers for given positions
        
        Args:
            positions: [B, N, 1] tensor of X coordinates
        """
        B, N, _ = positions.shape
        d_coords = torch.sqrt(positions[...,0]**2 + positions[...,1]**2)
        freqs = torch.outer(d_coords.reshape(-1), self.freqs)  # [B*N, head_size//2]
        freqs = freqs.reshape(B, N, -1)  # [B, N, head_size//2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [B, N, head_size//2]
        return freqs_cis
        
    def precompute_freqs_cis_2d(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Precompute frequency complex numbers for given positions
        
        Args:
            positions: [B, N, 2] tensor of XY coordinates
            
        Returns:
            freqs_cis: [B, N, head_size//2] complex tensor
        """
        B, N, _ = positions.shape
        
        # Separate x and y coordinates
        x_coords = positions[..., 0]  # [B, N]
        y_coords = positions[..., 1]  # [B, N]
        
        # Compute outer product: positions × frequencies
        # For x dimension
        x_freqs = torch.outer(x_coords.reshape(-1), self.freqs)  # [B*N, head_size//4]
        x_freqs = x_freqs.reshape(B, N, -1)  # [B, N, head_size//4]
        
        # For y dimension  
        y_freqs = torch.outer(y_coords.reshape(-1), self.freqs)  # [B*N, head_size//4]
        y_freqs = y_freqs.reshape(B, N, -1)  # [B, N, head_size//4]

        # Convert to complex numbers
        freqs_cis_x = torch.polar(torch.ones_like(x_freqs), x_freqs)  
        freqs_cis_y = torch.polar(torch.ones_like(y_freqs), y_freqs)  

        # Convert to complex numbers
        freqs_cis = torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)  # concatenate complex numbers
        
        return freqs_cis # [B,N, head_size//2]
    
    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, D], freqs_cis: [B, N, head_size//2]
        b, h, n, _ = x.shape
        assert freqs_cis.shape[0] == b and freqs_cis.shape[1] == n, "freqs_cis must match [B,N]"
        return freqs_cis.view(b, 1, n, freqs_cis.size(-1))
    
    def apply_rotary_emb(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor
        
        Args:
            x: [B, H, N, D] input tensor
            freqs_cis: [B, N, head_size//2] complex frequency tensor
            
        Returns:
            Rotated tensor with same shape as x
        """
        # Reshape x to complex numbers
        x_ = x.float()
        x_reshaped = x_.reshape(*x_.shape[:-1], -1, 2)  # [B, H, N, D//2, 2]
        x_complex = torch.view_as_complex(x_reshaped)  # [B, H, N, D//2]
        
        # Reshape freqs_cis for broadcasting
        freqs_cis = self.reshape_for_broadcast(freqs_cis, x_complex)
        
        # Apply rotation
        x_rotated = x_complex * freqs_cis  # [B, H, N, D//2]
        
        # Convert back to real representation
        x_out = torch.view_as_real(x_rotated)  # [B, H, N, D//2, 2]
        x_out = x_out.flatten(-2)  # [B, H, N, D]
        
        return x_out.type_as(x)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                query_pos: Optional[torch.Tensor] = None, 
                key_pos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotary position embedding to query and key
        
        Args:
            query: [B, H, N, D] query tensor
            key: [B, H, N, D] key tensor  
            query_pos: [B, N, 2] query positions (XY coordinates)
            key_pos: [B, N, 2] key positions (XY coordinates)
            
        Returns:
            Rotated query and key tensors
        """
        # Precompute frequency complex numbers
        if self.pos_dim == 2:
            query_freqs_cis = self.precompute_freqs_cis_2d(query_pos) if query_pos is not None else None
            key_freqs_cis = self.precompute_freqs_cis_2d(key_pos) if key_pos is not None else None
        elif self.pos_dim == 1:
            query_freqs_cis = self.precompute_freqs_cis_1d(query_pos) if query_pos is not None else None
            key_freqs_cis = self.precompute_freqs_cis_1d(key_pos) if key_pos is not None else None
        else:
            raise ValueError(f"Invalid pos_dim: {self.pos_dim}")

        # Apply rotary embedding
        if query_freqs_cis is not None:
            query = self.apply_rotary_emb(query, query_freqs_cis)
        
        if key_freqs_cis is not None:
            key = self.apply_rotary_emb(key, key_freqs_cis)
        
        return query, key



# first we define a rope embedding class.
# class RotatePosEnc(nn.Module):
#     r"""
#     """

#     def __init__(self,head_size = 128, dimention=2):
#         super(RotatePosEnc, self).__init__()
#         self.head_size = head_size
#         self.dimention = dimention
#         # 1. use cat theta to learn pos embedding
#         if self.dimention == 2:
#             angle = 1.0 / (10000 ** torch.linspace(0,1,self.head_size // 4))
#             self.angle = angle.unsqueeze(-1).repeat(1,2).flatten() # [head_size//2]
#         elif self.dimention == 1:
#             angle = 1.0 / (10000 ** torch.linspace(0,1,self.head_size // 2)) # [head_size//2]
#             self.angle = angle.unsqueeze(-1).repeat(1,2).flatten() # [head_size]
#         # 2. use add theta to learn pos embedding
#         # angle = 1.0 / (10000 ** torch.linspace(0,1,self.head_size // 2)) # [head_size//2]
#         # self.angle = angle.unsqueeze(-1).repeat(1,2).flatten() # [head_size]


#     def rotate_every_two(self,x):
#         x1 = x[..., ::2] # [B, head_num, A, head_size//2]
#         x2 = x[..., 1::2]
#         x = torch.stack((-x2, x1), dim=-1)
#         return x.flatten(-2) # [B, head_num, A, head_size]

#     def theta_shift(self, x, sin, cos):
#         return (x * cos) + (self.rotate_every_two(x)*sin) # [B, head_num, A, head_size]
    

#     def get_angle(self, query_pos, key_pos, theta):
#         if self.dimention == 2:
#             query_pos_x, key_pos_x = query_pos[:,:,0], key_pos[:,:,0] # [B,A]
#             query_pos_y, key_pos_y = query_pos[:,:,1], key_pos[:,:,1] # [B,A]
#             query_x_theta, key_x_theta = query_pos_x[:,:,None] * theta, key_pos_x[:,:,None]*theta # [B,A,head_size/2]
#             query_y_theta, key_y_theta = query_pos_y[:,:,None] * theta, key_pos_y[:,:,None]*theta # [B,A,head_size/2]
#             # 1. use cat theta to learn pos embedding
#             query_n_theta, key_n_theta = torch.cat((query_y_theta, query_x_theta), dim=-1), torch.cat((key_y_theta, key_x_theta), dim=-1) # [B,A,head_size]
#             # 2. use add theta to learn pos embedding
#             # query_n_theta, key_n_theta = query_y_theta + query_x_theta, key_y_theta + key_x_theta # [B,A,head_size]
#             query_sin, key_sin = torch.sin(query_n_theta).unsqueeze(-3), torch.sin(key_n_theta).unsqueeze(-3) # [B, 1, A, head_size],[B, 1, L, head_size]
#             query_cos, key_cos = torch.cos(query_n_theta).unsqueeze(-3), torch.cos(key_n_theta).unsqueeze(-3) # [B, 1, A, head_size],[B, 1, L, head_size]
#         elif self.dimention == 1:
#             # query_pos, key_pos: [B,A,T]
#             query_x_theta, key_x_theta = query_pos[...,None] * theta, key_pos[...,None] * theta # [B,A,T,head_size] 
#             query_sin, key_sin = torch.sin(query_x_theta).unsqueeze(-3), torch.sin(key_x_theta).unsqueeze(-3) # [B, A, 1, T, head_size],[B, L, 1, N, head_size]
#             query_cos, key_cos = torch.cos(query_x_theta).unsqueeze(-3), torch.cos(key_x_theta).unsqueeze(-3) # [B, A, 1, T, head_size],[B, L, 1, N, head_size]    
#         return query_sin, query_cos, key_sin, key_cos
    
#     def forward(self,query,key,query_pos = None,key_pos = None):
#         theta = self.angle.to(query.device)
#         query_sin, query_cos, key_sin, key_cos = self.get_angle(query_pos, key_pos, theta)
#         query = self.theta_shift(query, query_sin, query_cos)
#         key = self.theta_shift(key, key_sin, key_cos)
#         return query,key

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, mask=None, start_idx=0):
        """
        Arguments:
            x: Tensor, shape ``[B,A,T,D]`` or ``[B,A,D]``
            mask: Optional mask tensor with same shape as x
        Returns:
            x: Tensor, shape same as input x
        """
        # Get the sequence length from the input tensor
        seq_len = x.size(-2)
        pe_slice = self.pe[start_idx:start_idx+seq_len]
        if x.dim() == 4:  # [B,A,T,D]
            pe_expanded = pe_slice.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,D]
        elif x.dim() == 3:  # [B,A,D]
            pe_expanded = pe_slice.unsqueeze(0)  # [1,seq_len,D]
        else:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {x.dim()}")
        
        if mask is None:
            x = x + pe_expanded
        else:
            x = x + pe_expanded * mask
        return self.dropout(x)
    
class MLP_3(nn.Module):
    def __init__(self, dims):
        super(MLP_3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]), 
            nn.LayerNorm(dims[1]), nn.ReLU(), 
            nn.Linear(dims[1], dims[2]),
            nn.LayerNorm(dims[2]), nn.ReLU(), 
            nn.Linear(dims[2], dims[3])
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

# *************************
# ***Relative Pose utils***
# *************************
def cast_rad(angle: Union[float, np.ndarray, Tensor]) -> Union[float, np.ndarray, Tensor]:
    """Cast angle such that they are always in the [-pi, pi) range."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# transformation for torch
def torch_rad2rot(rad: Tensor) -> Tensor:
    """
    Args:
        rad: [n_batch] or [n_scene, n_agent] or etc.

    Returns:
        rot_mat: [{rad.shape}, 2, 2]
    """
    _cos = torch.cos(rad)
    _sin = torch.sin(rad)
    return torch.stack([torch.stack([_cos, -_sin], dim=-1), torch.stack([_sin, _cos], dim=-1)], dim=-2)


def torch_sincos2rot(in_sin: Tensor, in_cos: Tensor) -> Tensor:
    """
    Args:
        in_sin: [n_batch] or [n_scene, n_agent] or etc.
        in_cos: [n_batch] or [n_scene, n_agent] or etc.

    Returns:
        rot_mat: [{in_sin.shape}, 2, 2]
    """
    return torch.stack([torch.stack([in_cos, -in_sin], dim=-1), torch.stack([in_sin, in_cos], dim=-1)], dim=-2)


def torch_pos2local(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    """Transform M position to the local coordinates.

    Args:
        in_pos: [..., M, 2]
        local_pos: [..., 1, 2]
        local_rot: [..., 2, 2]

    Returns:
        out_pos: [..., M, 2]
    """
    return torch.matmul(in_pos - local_pos, local_rot)


def torch_pos2global(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    """Reverse torch_pos2local

    Args:
        in_pos: [..., M, 2]
        local_pos: [..., 1, 2]
        local_rot: [..., 2, 2]

    Returns:
        out_pos: [..., M, 2]
    """
    return torch.matmul(in_pos.double(), local_rot.transpose(-1, -2).double()) + local_pos.double()


def torch_dir2local(in_dir: Tensor, local_rot: Tensor) -> Tensor:
    """Transform M dir to the local coordinates.

    Args:
        in_dir: [..., M, 2]
        local_rot: [..., 2, 2]

    Returns:
        out_dir: [..., M, 2]
    """
    return torch.matmul(in_dir, local_rot)


def torch_dir2global(in_dir: Tensor, local_rot: Tensor) -> Tensor:
    """Reverse torch_dir2local

    Args:
        in_dir: [..., M, 2]
        local_rot: [..., 2, 2]

    Returns:
        out_dir: [..., M, 2]
    """
    return torch.matmul(in_dir, local_rot.transpose(-1, -2))


def torch_rad2local(in_rad: Tensor, local_rad: Tensor, cast: bool = True) -> Tensor:
    """Transform M rad angles to the local coordinates.

    Args:
        in_rad: [..., M]
        local_rad: [...]

    Returns:
        out_rad: [..., M]
    """
    out_rad = in_rad - local_rad.unsqueeze(-1)
    if cast:
        out_rad = cast_rad(out_rad)
    return out_rad


def torch_rad2global(in_rad: Tensor, local_rad: Tensor) -> Tensor:
    """Reverse torch_rad2local

    Args:
        in_rad: [..., M]
        local_rad: [...]

    Returns:
        out_rad: [..., M]
    """
    return cast_rad(in_rad + local_rad.unsqueeze(-1))

@torch.no_grad()
def get_rel_pose(pose: torch.Tensor, invalid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        pose: [n_scene, n_emb, 3], (x,y,yaw), in global coordinate
        invalid: [n_scene, n_emb]

    Returns:
        rel_pose: [n_scene, n_emb, n_emb, 3] (x,y,yaw)
        rel_dist: [n_scene, n_emb, n_emb]
    """
    xy = pose[:, :, :2]  # [n_scene, n_emb, 2]
    yaw = pose[:, :, -1]  # [n_scene, n_emb]
    rel_pose = torch.cat(
        [
            torch_pos2local(xy.unsqueeze(1), xy.unsqueeze(2), torch_rad2rot(yaw)),
            torch_rad2local(yaw.unsqueeze(1), yaw, cast=False).unsqueeze(-1),
        ],
        dim=-1,
    )  # [n_scene, n_emb, n_emb, 3]
    rel_dist = torch.norm(rel_pose[..., :2], dim=-1)  # [n_scene, n_emb, n_emb]
    rel_dist.masked_fill_(invalid.unsqueeze(1) | invalid.unsqueeze(2), float("inf"))
    return rel_pose, rel_dist


@torch.no_grad()
def get_rel_dist(xy: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
    """
    Args:
        xy: [n_scene, n_emb, 2], in global coordinate
        invalid: [n_scene, n_emb]

    Returns:
        rel_dist: [n_scene, n_emb, n_emb]
    """
    rel_dist = torch.norm(xy.unsqueeze(1) - xy.unsqueeze(2), dim=-1)  # [n_scene, n_emb, n_emb]
    rel_dist.masked_fill_(invalid.unsqueeze(1) | invalid.unsqueeze(2), float("inf"))
    return rel_dist


@torch.no_grad()
def get_tgt_knn_idx(
    tgt_invalid: torch.Tensor, rel_pose: Optional[torch.Tensor], rel_dist: torch.Tensor, n_tgt_knn: int, dist_limit: Union[float, torch.Tensor],
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        tgt_invalid: [n_scene, n_tgt]
        rel_pose: [n_scene, n_src, n_tgt, 3]
        rel_dist: [n_scene, n_src, n_tgt]
        knn: int, set to <=0 to skip knn, i.e. n_tgt_knn=n_tgt
        dist_limit: float, or torch.Tensor [n_scene, n_tgt, 1]

    Returns:
        idx_tgt: [n_scene, n_src, n_tgt_knn], or None
        tgt_invalid_knn: [n_scene, n_src, n_tgt_knn]
        rpe: [n_scene, n_src, n_tgt_knn, 3]
    """
    n_scene, n_src, _ = rel_dist.shape
    idx_scene = torch.arange(n_scene)[:, None, None]  # [n_scene, 1, 1]
    idx_src = torch.arange(n_src)[None, :, None]  # [1, n_src, 1]

    if 0 < n_tgt_knn < tgt_invalid.shape[1]:
        # [n_scene, n_src, n_tgt_knn]
        dist_knn, idx_tgt = torch.topk(rel_dist, n_tgt_knn, dim=-1, largest=False, sorted=False)
        # [n_scene, n_src, n_tgt_knn]
        tgt_invalid_knn = tgt_invalid.unsqueeze(1).expand(-1, n_src, -1)[idx_scene, idx_src, idx_tgt]
        # [n_batch, n_src, n_tgt_knn, 3]
        if rel_pose is None:
            rpe = None
        else:
            rpe = rel_pose[idx_scene, idx_src, idx_tgt]
    else:
        dist_knn = rel_dist
        tgt_invalid_knn = tgt_invalid.unsqueeze(1).expand(-1, n_src, -1)  # [n_scene, n_src, n_tgt]
        rpe = rel_pose
        idx_tgt = None

    tgt_invalid_knn = tgt_invalid_knn | (dist_knn > dist_limit)
    if rpe is not None:
        rpe = rpe.masked_fill(tgt_invalid_knn.unsqueeze(-1), 0)

    return idx_tgt, tgt_invalid_knn, rpe

