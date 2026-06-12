import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch.distributed as dist
from utils.loss_utils import batch_label_ADE, ego_planning_loss_preprocess
from losses.motion_prediction_loss import DirectGmmLoss, BarlowTwins, SmoothL1LossMasked
from losses.motion_planning_loss import RepeatedIndicesMinkowskiCollisionLoss, SoftplusHingeComfortLoss, AdaptiveProgressLoss
import numpy as np
from metrics.torchmetrics import MotionForecastMetric,JointMotionForecastMetric,EgoMotionPlanningMetric
from models.builder import LITMODEL
from utils.model_utils import (PositionalEncoding,
                                TransformerEncoder,
                                TransformerEncoderLayer,
                                TransformerDecoder,
                                TransformerDecoderLayer,
                                MLP_3)
from utils.model_utils import (compute_intent_mask,
                                TransformerCrossAttention,
                                get_rel_dist,
                                get_tgt_knn_idx)
from utils.model_utils import DeepseekV2MLP, MoEGate
from utils.data_utils import load_config
from typing import Tuple,Optional
from torch import Tensor

#******************************************
#******** Pytorch Lightning Model *********
#******************************************
@LITMODEL.register_module()
class LitMMDiTDiffusionModel(L.LightningModule):
    def __init__(self, config=None, lr=0.001, warmup_steps=0,warmup_interval='step'):
        super(LitMMDiTDiffusionModel, self).__init__()
        if config is None:
            config_path = os.path.join(os.getcwd(), 'config', 'model', self.__class__.__name__+'.yaml')
            config = load_config(config_path)
            config = config['model']
        elif isinstance(config, str):
            config_path = os.path.join(os.getcwd(), 'config', 'model', config)
            config = load_config(config_path)
            config = config['model']
        if config['compile']:
            self.model = torch.compile(MMDiTDiffusionModel(config))
        else:   
            self.model = MMDiTDiffusionModel(config)
        self.gmm_loss = DirectGmmLoss()
        self.test_metric_ego = EgoMotionPlanningMetric(is_waymo_dataset=True)
        self.future_length = 80
        self.mode_num = 6
        self.lr = lr
        self.pred_trajs_shape = None
        self.warmup_steps = warmup_steps
        self.warmup_interval = warmup_interval
        self.save_hyperparameters()

    def forward(self, x):
        return self.model.forward(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr) # self.lr is the max lr
        total_num = self.trainer.estimated_stepping_batches
        
        # main scheduler: CosineAnnealingLR
        warmup_steps = self.warmup_steps if self.warmup_interval == 'step' else int(self.warmup_steps * total_num / self.trainer.max_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_num - warmup_steps
        )
        
        # warmup scheduler: LinearLR
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0,total_iters=warmup_steps)
        
    
        # combine Warmup and Cosine
        scheduler = {
            "scheduler" :torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]),
            "interval": 'step',
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        model_input = batch['model_input']
        loss = self.model((model_input))
        self.log_dict({"train_loss":loss}, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        model_input = batch['model_input']
        loss = self.model((model_input))
        self.log_dict({"val_loss":loss}, sync_dist=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        model_input = batch['model_input']
        pred_trajs = self.model.sample(model_input)
        pred_trajs = pred_trajs.unsqueeze(2).repeat(1,1,6,1,1)# [B, 1, 6, 80, 3]
        batch_target_agents_pred_trajs_list = [pred_trajs[i, :1][..., :2].detach().cpu().numpy().astype(np.float32)
                                                    for i in range(pred_trajs.shape[0])]
        batch_target_agents_pred_scores_list = [np.ones(6, dtype=np.float32)
                                                for i in range(pred_trajs.shape[0])]
        model_output_dict = {'pred_trajs': batch_target_agents_pred_trajs_list,
                             'pred_scores': batch_target_agents_pred_scores_list}
        self.test_metric_ego.update(model_output_dict, batch['ego_planning_metrics_input'])

    def on_test_epoch_end(self):
        metrics_output = self.test_metric_ego.compute()
        if self.trainer.is_global_zero:
            import pprint
            pp = pprint.PrettyPrinter(depth=4, sort_dicts=False)
            pp.pprint(metrics_output)
        self.test_metric_ego.reset()
    
    def calc_loss(self, pred_trajs, pred_scores, target_agents_gt, target_agents_gt_valid, target_agents_num):
        B, A, M, _, _ = pred_trajs.shape
        valid_agents_indices = torch.arange(0, A, device=pred_trajs.device).expand(B, -1)
        target_agents_mask = (valid_agents_indices < target_agents_num.unsqueeze(-1))
        target_agents_pred_trajs, target_agents_pred_scores = pred_trajs[target_agents_mask], pred_scores[target_agents_mask] # [B*N_targets, 6, points_num, 5], [B*N_targets, 6]
        
        min_ade_value, min_ade_idx = batch_label_ADE(target_agents_pred_trajs.unsqueeze(1)[..., :2].detach(),
                                                     target_agents_gt.unsqueeze(1)[..., :2],
                                                     target_agents_num,
                                                     target_agents_gt_valid.unsqueeze(1)) # [B*N_targets,1], [B*N_targets,1]
        min_ade_mode_mask = torch.arange(0, self.mode_num, device=target_agents_pred_trajs.device).expand(
            target_agents_pred_trajs.shape[0], self.mode_num
        ) == min_ade_idx # [B*N_targets, 6]
        nearest_pred_trajs = target_agents_pred_trajs[min_ade_mode_mask] # [B*N_targets, points_num, 5]
        target_agents_gt_xy = target_agents_gt[..., :2] # [B*N_targets, points_num, 2]
        reg_loss = self.gmm_loss.update(nearest_pred_trajs, target_agents_gt_xy, target_agents_gt_valid)
        cls_loss = F.cross_entropy(target_agents_pred_scores, min_ade_idx.squeeze(-1))
        loss = reg_loss + cls_loss
        min_ade_value = min_ade_value.mean()
        return loss, min_ade_value

#********************************
#******** Multi TF Model ********
#********************************
class MMDiTDiffusionModel(nn.Module):
    def __init__(self, config):
        super(MMDiTDiffusionModel, self).__init__()
        # self.encoder = MMDiTDiffusionEncoder(config['encoder'])
        self.encoder = SceneEncoder(config['encoder'])
        self.decoder = DiTDiffusionDecoder(config['decoder'])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.075
            if hasattr(module, "RESIDUAL_SCALE"):
                std *= (2 * module.RESIDUAL_SCALE) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.075)

    def forward(self, x):
        model_input = x
        encoder_output = self.encoder(model_input)
        prediction = self.decoder(model_input, *encoder_output)
        return prediction
    
    @torch.no_grad()
    def sample(self, model_input, sample_steps=5):
        encoder_output = self.encoder(model_input)
        pred_trajs = self.decoder.sample(sample_steps, *encoder_output)[-1] # [B, 1, 81, 3]
        # pred_trajs = pred_trajs[:, :, 1:, :] # [B, 1, 80, 3]
        pred_trajs = self.decoder.denormalize_state(pred_trajs)

        pred_trajs = torch.cumsum(pred_trajs, dim=-2) # [B, 1, 80, 3]
        # print(f'pred_trajs: {pred_trajs[0,0, :, :2]}')


        return pred_trajs
    
#******************************************
#******** Multi TF Model Components********
#******************************************
class MultiTFMask(nn.Module):
    def __init__(self):
        super(MultiTFMask, self).__init__()
    
    def forward(self, input):
        x_t, x_t_valid, x_m, x_m_valid, x_r, x_r_valid, x_nv,input_length = input # [B,A,T,C_a], [B,A,T,1], [B,L,N,1], [B,L,N,C_m], [B,2]
        B,A,L, R, NV = x_t.shape[0], x_t.shape[1], x_m.shape[1], x_r.shape[1], x_nv.shape[1] # B=batch size, A=agent num, L=lane segment num, R=routing num, NV=ego navi num

        # Mask Initialization
        T2T_mask = torch.matmul(x_t_valid, x_t_valid.transpose(-2,-1))          # [B,A,T,T]
        T2T_mask.requires_grad = False
        N2N_mask = torch.matmul(x_m_valid, x_m_valid.transpose(-2,-1))          # [B,L,N,N]
        N2N_mask.requires_grad = False
        A2A_mask = torch.zeros((B,A,A), device=x_t.device, requires_grad=False) # [B,A,A]
        L2L_mask = torch.zeros((B,L,L), device=x_m.device, requires_grad=False) # [B,L,L]
        A2L_mask = torch.zeros((B,A,L), device=x_t.device, requires_grad=False) # [B,A,L]
        AL2AL_mask = torch.zeros((B,A+L,A+L), device=x_t.device, requires_grad=False) # [B,A+L,A+L]
        A2AL_mask = torch.zeros((B,A,A+L), device=x_t.device, requires_grad=False) # [B,A,A+L]
        R2R_mask = torch.zeros((B,R,R), device=x_t.device, requires_grad=False) # [B,R,R]
        EGO2R_mask = torch.zeros((B,1,R), device=x_t.device, requires_grad=False) # [B,1,R]
        Nv2Nv_mask = torch.zeros((B,NV,NV), device=x_t.device, requires_grad=False) # [B,NV,NV]
        

        R_N2N_mask = torch.matmul(x_r_valid, x_r_valid.transpose(-2,-1))          # [B,R,N,N]
        R_N2N_mask.requires_grad = False

        A_real_length = input_length[:,0].reshape(B,1,1)
        L_real_length = input_length[:,1].reshape(B,1,1)
        R_real_length  = input_length[:,3].reshape(B,1,1)
        Nv_real_length = input_length[:,4].reshape(B,1,1)

        # Agents' time dimension casual mask
        T2T_mask = torch.tril(T2T_mask) # [B,A,T,T], lower triangular part of original T2T_mask

        # A2A mask filling
        A2A_mask_row_index = torch.arange(A, dtype=torch.float32, device=input_length.device).reshape(1,A,1).repeat(B,1,A) # [B,A,A]
        A2A_mask_col_index = A2A_mask_row_index.transpose(1,2)

        row_mask_a         = A2A_mask_row_index < A_real_length
        col_mask_a         = A2A_mask_col_index < A_real_length

        A2A_mask[row_mask_a&col_mask_a] = 1.0

        # L2L mask filling
        L2L_mask_row_index = torch.arange(L, dtype=torch.float32, device=input_length.device).reshape(1,L,1).repeat(B,1,L) # [B,L,L]
        L2L_mask_col_index = L2L_mask_row_index.transpose(1,2)

        row_mask_l         = L2L_mask_row_index < L_real_length
        col_mask_l         = L2L_mask_col_index < L_real_length

        L2L_mask[row_mask_l&col_mask_l] = 1.0

        # A2L mask filling
        A2L_mask_row_index = torch.arange(A, dtype=torch.float32, device=input_length.device).reshape(1,A,1).repeat(B,1,L) # [B,A,L]
        A2L_mask_col_index = torch.arange(L, dtype=torch.float32, device=input_length.device).reshape(1,1,L).repeat(B,A,1) # [B,A,L]

        row_mask_al        = A2L_mask_row_index < A_real_length
        col_mask_al        = A2L_mask_col_index < L_real_length

        A2L_mask[row_mask_al&col_mask_al] = 1.

        # A2all maskfilling
        A2AL_mask_row_index = torch.arange(A, dtype=torch.float32, device=input_length.device).reshape(1,A,1).repeat(B,1,A+L) # [B,A,A+L]
        A2AL_mask_col_index = torch.arange(A+L, dtype=torch.float32, device=input_length.device).reshape(1,1,A+L).repeat(B,A,1) # [B,A,A+L]

        row_mask_a2all = A2AL_mask_row_index < A_real_length
        col_mask_a2all = torch.logical_or(A2AL_mask_col_index < A_real_length, torch.logical_and(A < A2AL_mask_col_index, A2AL_mask_col_index < A + L_real_length))

        A2AL_mask[row_mask_a2all&col_mask_a2all] = 1.

        # Global mask filling
        AL2AL_mask_row_index = torch.arange(A+L, dtype=torch.float32, device=input_length.device).reshape(1,A+L,1).repeat(B,1,A+L) # [B,A,L])
        AL2AL_mask_col_index = AL2AL_mask_row_index.transpose(1,2)

        row_mask_alal = torch.logical_or(AL2AL_mask_row_index < A_real_length, torch.logical_and(A < AL2AL_mask_row_index, AL2AL_mask_row_index < A + L_real_length))
        col_mask_alal = torch.logical_or(AL2AL_mask_col_index < A_real_length, torch.logical_and(A < AL2AL_mask_col_index, AL2AL_mask_col_index < A + L_real_length))

        AL2AL_mask[row_mask_alal&col_mask_alal] = 1.

        # ego routing mask filling
        R2R_mask_row_index = torch.arange(R, dtype=torch.float32, device=input_length.device).reshape(1,R,1).repeat(B,1,R) # [B,R,R]
        R2R_mask_col_index = R2R_mask_row_index.transpose(1,2)
        row_mask_r        = R2R_mask_row_index < R_real_length
        col_mask_r        = R2R_mask_col_index < R_real_length
        R2R_mask[row_mask_r&col_mask_r] = 1.0

        # ego to routing mask filling
        EGO2R_mask_col_index = torch.arange(R, dtype=torch.float32, device=input_length.device).reshape(1,1,R) # [B,1,R]
        col_mask_ego2r = EGO2R_mask_col_index < R_real_length
        EGO2R_mask[col_mask_ego2r] = 1.0

        # ego navi to ego navi mask filling
        Nv2Nv_mask_row_index = torch.arange(NV, dtype=torch.float32, device=input_length.device).reshape(1,NV,1).repeat(B,1,NV) # [B,NV,NV]
        Nv2Nv_mask_col_index = Nv2Nv_mask_row_index.transpose(1,2)
        row_mask_nv2nv = Nv2Nv_mask_row_index < Nv_real_length
        col_mask_nv2nv = Nv2Nv_mask_col_index < Nv_real_length
        Nv2Nv_mask[row_mask_nv2nv&col_mask_nv2nv] = 1.0

        return T2T_mask, N2N_mask, A2A_mask, L2L_mask, A2L_mask, A2AL_mask, AL2AL_mask, R2R_mask, R_N2N_mask, EGO2R_mask ,Nv2Nv_mask # [B,A,T,T], [B,L,N,N], [B,A,A], [B,L,L], [B,A,L], [B,A,A+L], [B,A+L,A+L], [B,R,R], [B,R,N,N], [B,1,R], [B,NV,NV]

class SceneEncoder(nn.Module):
    def __init__(self, config):
        super(SceneEncoder, self).__init__()
        self.hidden_size              = config['hidden_size']
        self.depth                    = config['depth']
        self.mask_generator           = MultiTFMask()

        self.agent_feature_encode_mlp = MLP_2(in_features=33, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        # self.map_feature_encode_mlp   = MLP_2(in_features=26, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.map_point_net = SubgraphNet(input_size=26, hidden_size=self.hidden_size, depth=3)
        self.map_general_type_embedding = MLP_2(in_features=7, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.ego_routing_point_net = SubgraphNet(input_size=7, hidden_size=self.hidden_size, depth=3)
        # self.ego_routing_feature_encode_mlp = MLP_2(in_features=7, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)

        self.ego_traffic_light_feature_encode_mlp = MLP_2(in_features=10, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.agent_type_embedding     = MLP_2(in_features=6, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.agent_time_pe            = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=11)
        # self.agent_instance_pe        = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=128) # TODO: Move to  config
        # self.map_segment_pe           = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=11)
        # self.map_sequence_pe          = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=512) # TODO: Move to  config
        # self.ego_routing_segment_pe   = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=11) # point-level positional encoding
        self.ego_routing_sequence_pe  = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=64) # instance-level positional encoding

        self.ego_navi_point_net = SubgraphNet(input_size=7, hidden_size=self.hidden_size, depth=3)
        self.agent_self_attn_on_T     = TransformerEncoder(hidden_size=self.hidden_size, depth=self.depth)
        self.agent_self_attn_on_A_with_T = TransformerEncoder(hidden_size=self.hidden_size, depth=self.depth)
        # self.map_self_attn_on_N       = TransformerEncoder(hidden_size=self.hidden_size, depth=self.depth)
        # self.ego_routing_self_attn_on_N = TransformerEncoder(hidden_size=self.hidden_size, depth=self.depth)

        self.ego_current_status_feature_encode_mlp = MLP_2(in_features=6, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        # if self.rope_dim is not None:
        #     print(f"Using {self.rope_dim}D RoPE")
        #     self.agent_self_attn_on_A     = TransformerEncoder(hidden_size=self.hidden_size, depth=3, pos_dim=self.rope_dim)
        #     self.map_self_attn_on_L       = TransformerEncoder(hidden_size=self.hidden_size, depth=6, pos_dim=self.rope_dim)
        #     self.agent_map_cross_attn     = TransformerDecoder(hidden_size=self.hidden_size, depth=self.depth, pos_dim=self.rope_dim)
        # else:
        #     print("Using no RoPE")
        self.agent_self_attn_on_A     = TransformerEncoder(hidden_size=self.hidden_size, depth=4) # 3
        self.map_self_attn_on_L       = TransformerEncoder(hidden_size=self.hidden_size, depth=4) # 6
        self.agent_map_cross_attn     = TransformerDecoder(hidden_size=self.hidden_size, depth=4) # 2
        self.ego_routing_self_attn_on_R = TransformerEncoder(hidden_size=self.hidden_size, depth=4) # 3
        self.ego_navi_self_attn_on_Nv = TransformerEncoder(hidden_size=self.hidden_size, depth=4) # 3
        # self.agents_projection        = MLP_3(dims=[self.hidden_size, self.hidden_size*4, self.hidden_size*8, self.hidden_size])
        # self.map_projection           = MLP_3(dims=[self.hidden_size, self.hidden_size*4, self.hidden_size*8, self.hidden_size])
        # if self.use_dynamic_map:
        #     print("Using dynamic map")
        #     self.tl_feature_encode_mlp    = MLP_3(dims=[10, self.hidden_size, self.hidden_size*2, self.hidden_size])
        #     self.tl_time_pe               = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=11)
        #     self.tl_sequence_pe           = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=64)
        #     self.tl_self_attn_on_T        = TransformerEncoder(hidden_size=self.hidden_size, depth=self.depth)
        #     self.tl_self_attn_on_TL       = TransformerEncoder(hidden_size=self.hidden_size, depth=1)
        #     self.agent_tl_cross_attn      = TransformerDecoder(hidden_size=self.hidden_size, depth=self.depth)

    def forward(self, input):
        # Input unpacking
        # [B,A,T,C_a], [B,A,3],[B,L,N,C_m], [B,L,2]
        # [B,TL,T,C_tl], [B,R,N,C_r], [B,3],[B,1,10], [B,4]
        #raw_dynamic_map_features, 
        raw_agent_features, raw_static_map_features = input['valid_agents_features'], input['static_map_features']
        raw_ego_routing_features, raw_ego_traffic_light_features,raw_ego_navi_features, mask_info = input['ego_routing_features'], input['ego_traffic_light_features'], input['ego_navi_features'], input['components_num']
        
        raw_agents_lw = raw_agent_features[:,:,-1,25:27] # [B,A,2]
        target_agents_num = mask_info[:,0]

        # Mask info unpacking
        # [B,A,T,1], [B,A], [B,L,N,1], [B,L]
        # [B,TL,T,1], [B,TL]
        # [B,R,N,1], [B,R]
        # agent and traffic light use the last point to check valid; map instance use the first point to check valid
        # agents_valid_on_T, agents_valid_on_A, map_valid_on_N, map_valid_on_L, \
        #     traffic_light_valid_on_T, traffic_light_valid_on_TL, \
        #     ego_routing_valid_on_N, ego_routing_valid_on_R = raw_agent_features[..., [-1]], raw_agent_features[:,:,-1,-1], \
        #                                                      raw_static_map_features[..., [-1]], raw_static_map_features[:,:,0,-1], \
        #                                                      raw_dynamic_map_features[..., [-1]], raw_dynamic_map_features[:,:,-1,-1], \
        #                                                      raw_ego_routing_features[..., [-1]], raw_ego_routing_features[:,:,0, -1]
        agents_valid_on_T, agents_valid_on_A, map_valid_on_N, map_valid_on_L, \
            ego_routing_valid_on_N, ego_routing_valid_on_R, ego_navi_valid_on_N, ego_navi_valid_on_Nv = raw_agent_features[..., [-1]], raw_agent_features[:,:,-1,-1], \
                                                             raw_static_map_features[..., [-1]], raw_static_map_features[:,:,0,-1], \
                                                             raw_ego_routing_features[..., [-1]], raw_ego_routing_features[:,:,0, -1], \
                                                             raw_ego_navi_features[..., [-1]], raw_ego_navi_features[:,:,0, -1]
        # Masks Generation
        # mask_a_t2t, mask_n2n, mask_a2a, mask_l2l, mask_a2l, mask_tl_t2t, \
        #     mask_tl2tl ,mask_a2tl,mask_r2r,mask_r_n2n = self.mask_generator((raw_agent_features,agents_valid_on_T,
        #                                                                         raw_static_map_features, map_valid_on_N,
        #                                                                         raw_dynamic_map_features, traffic_light_valid_on_T,
        #                                                                         raw_ego_routing_features, ego_routing_valid_on_N, 
        #                                                                         mask_info))   mask_a_t2t, mask_n2n, mask_a2a, mask_l2l, mask_a2l, mask_tl_t2t, \
        mask_t2t, mask_n2n, mask_a2a, mask_l2l, mask_a2l, mask_a2al, mask_al2al, mask_r2r, mask_r_n2n, mask_ego2r, mask_nv2nv = self.mask_generator((raw_agent_features,agents_valid_on_T,
                                                                                raw_static_map_features, map_valid_on_N,
                                                                                raw_ego_routing_features, ego_routing_valid_on_N,
                                                                                raw_ego_navi_features,
                                                                                mask_info))

        raw_map_type = raw_static_map_features[...,[6]] # [B, L, N, 1]
        B,L,N,_ = raw_static_map_features.shape
        # Map general type one hot
        raw_map_polyline_specific_type = raw_static_map_features[:, :, 0,[6]]
        raw_map_general_type = torch.zeros_like(raw_map_polyline_specific_type)
        raw_map_general_type[raw_map_polyline_specific_type == 0] = 0
        raw_map_general_type[torch.logical_and(raw_map_polyline_specific_type>=1, raw_map_polyline_specific_type<=3)] = 1
        raw_map_general_type[raw_map_polyline_specific_type==6] = 2
        raw_map_general_type[torch.logical_or(raw_map_polyline_specific_type==9, raw_map_polyline_specific_type==10)] = 2
        raw_map_general_type[raw_map_polyline_specific_type==13] = 2
        raw_map_general_type[torch.logical_or(raw_map_polyline_specific_type==7, raw_map_polyline_specific_type==8)] = 3
        raw_map_general_type[torch.logical_or(raw_map_polyline_specific_type==11, raw_map_polyline_specific_type==12)] = 3
        raw_map_general_type[raw_map_polyline_specific_type==14] = 4
        raw_map_general_type[torch.logical_and(raw_map_polyline_specific_type>=15, raw_map_polyline_specific_type<=16)] = 5
        raw_map_general_type[raw_map_polyline_specific_type==18] = 6
        raw_map_general_type_one_hot = torch.zeros_like(raw_map_general_type, dtype=raw_static_map_features.dtype,device = raw_static_map_features.device, requires_grad=False).repeat(1,1,7)
        raw_map_general_type_one_hot.scatter_(-1, raw_map_general_type.to(torch.int64), 1)
        raw_map_general_type_one_hot = raw_map_general_type_one_hot * map_valid_on_L.unsqueeze(-1)
        # Map specific type one hot
        raw_map_one_hot_18 = torch.zeros((B,L,N,19), dtype=raw_static_map_features.dtype,device = raw_static_map_features.device, requires_grad=False)
        raw_map_one_hot_18.scatter_(-1, raw_map_type.to(torch.int64), 1)# convert to one hot
        # mask 
        raw_static_map_features = torch.cat([raw_static_map_features[...,:6],raw_map_one_hot_18,raw_static_map_features[...,[7]]], dim=-1)
        raw_static_map_features = raw_static_map_features * map_valid_on_N
         
        
        # Initial Feature Embedding and positional embedding
        agent_features = self.agent_feature_encode_mlp(raw_agent_features)              # [B,A,T,D]
        agent_features = self.agent_time_pe(agent_features, mask=agents_valid_on_T)     # [B,A,T,D]
        # static_map_features   = self.map_feature_encode_mlp(raw_static_map_features)                     # [B,L,N,D]
        # static_map_features   = self.map_segment_pe(static_map_features, mask=map_valid_on_N)            # [B,L,N,D]
        # if self.use_dynamic_map:
        #     dynamic_map_features  = self.tl_feature_encode_mlp(raw_dynamic_map_features)                     # [B,TL,T,D]
        #     dynamic_map_features  = self.tl_time_pe(dynamic_map_features, mask=traffic_light_valid_on_T)     # [B,TL,T,D]
        # ego_routing_features  = self.ego_routing_feature_encode_mlp(raw_ego_routing_features)            # [B,R,N,D]
        # ego_routing_features  = self.ego_routing_segment_pe(ego_routing_features, mask=ego_routing_valid_on_N) # [B,R,N,D]
        # ego_turn_light_features    = self.ego_turn_light_feature_encode_mlp(raw_ego_turn_light_features)                # [B,D]
        ego_traffic_light_features = self.ego_traffic_light_feature_encode_mlp(raw_ego_traffic_light_features)    # [B,1,D]

        # First-level self-attn to aggregate info of dim T/N
        agent_features = self.agent_self_attn_on_T(agent_features, mask=mask_t2t)                 # [B,A,T,D]
        mask_a2a_on_T = torch.matmul(agents_valid_on_T.permute(0,2,1,3), agents_valid_on_T.permute(0,2,3,1)) # [B,T,A,A]
        agent_features = self.agent_self_attn_on_A_with_T(agent_features.transpose(1,2),mask=mask_a2a_on_T) # [B,T,A,D]
        agent_features = agent_features.transpose(1,2) # [B,A,T,D]
        # static_map_features   = self.map_self_attn_on_N(static_map_features, mask=mask_n2n)         # [B,L,N,D]
        
        # if self.use_dynamic_map:
        #     dynamic_map_features   = self.tl_self_attn_on_T(dynamic_map_features, mask=mask_tl_t2t)     # [B,TL,T,D]
        # ego_routing_features  = self.ego_routing_self_attn_on_N(ego_routing_features, mask=mask_r_n2n)   # [B,R,N,D]
        
        ## Max aggregation
        agent_features = torch.max(agent_features, dim=-2)[0]                       # [B,A,D]
        # static_map_features   = torch.max(static_map_features, dim=-2)[0]           # [B,L,D]
        static_map_features = self.map_point_net(raw_static_map_features) # [B,L,D]
        # if self.use_dynamic_map:
        #     dynamic_map_features  = torch.max(dynamic_map_features, dim=-2)[0]          # [B,TL,D]
        # ego_routing_features  = torch.max(ego_routing_features, dim=-2)[0]          # [B,R,D]
        ego_routing_features  = self.ego_routing_point_net(raw_ego_routing_features)   # [B,R,D]
        

        # Second-level self-attn to gather info from dim A/L
        # if self.rope_dim is not None:
        #     # agent_features = self.agent_instance_pe(agent_features, mask=agents_valid_on_A.unsqueeze(-1))        # [B,A,D]
        #     agent_features = self.agent_self_attn_on_A(agent_features, mask=mask_a2a, query_pos=raw_agent_pos[...,:2], key_pos=raw_agent_pos[...,:2])                            # [B,A,D]
        #     # static_map_features   = self.map_sequence_pe(static_map_features, mask=map_valid_on_L.unsqueeze(-1)) # [B,L,D]
        #     static_map_features   = self.map_self_attn_on_L(static_map_features, mask=mask_l2l, query_pos=raw_map_pl_pos[...,:2], key_pos=raw_map_pl_pos[...,:2])                  # [B,L,D]
        # else:
        agent_features = self.agent_self_attn_on_A(agent_features, mask=mask_a2a)                            # [B,A,D]
        static_map_features   = self.map_self_attn_on_L(static_map_features, mask=mask_l2l)                  # [B,L,D] 

        # fuse agent type and agent_features
        agent_types_emb = self.agent_type_embedding(raw_agent_features[:,:,-1,7:13]) # [B,A,6] ---> [B,A,D]
        agent_features = agent_features + agent_types_emb
        # fuse map general type and map_features
        map_general_type_emb = self.map_general_type_embedding(raw_map_general_type_one_hot) # [B,L,6] ---> [B,L,D]
        static_map_features = static_map_features + map_general_type_emb
        # if self.use_dynamic_map:
        #     dynamic_map_features  = self.tl_sequence_pe(dynamic_map_features, mask=traffic_light_valid_on_TL.unsqueeze(-1))  # [B,TL,D] 
        #     dynamic_map_features  = self.tl_self_attn_on_TL(dynamic_map_features, mask=mask_tl2tl)               # [B,TL,D] 
        # ego routing is sequence. use PE
        ego_routing_features  = self.ego_routing_sequence_pe(ego_routing_features, mask=ego_routing_valid_on_R.unsqueeze(-1))  # [B,R,D]
        ego_routing_features  = self.ego_routing_self_attn_on_R(ego_routing_features, mask=mask_r2r)         # [B,R,D]

        
        ## Reserve self-context-aware agent_features
        self_aware_agent_features = agent_features.clone()                          # [B,A,D]

        # Final cross-attn for agents to gather info from maps
        # if self.rope_dim is not None:
        #     map_aware_agent_features  = self.agent_map_cross_attn(x=agent_features, 
        #                                                           memory=static_map_features,
        #                                                           x_mask=None, 
        #                                                           memory_mask=mask_a2l,
        #                                                           query_pos=raw_agent_pos[...,:2],
        #                                                           key_pos=raw_map_pl_pos[...,:2]) # [B,A,D]
        # else:
        map_aware_agent_features  = self.agent_map_cross_attn(x=agent_features,
                                                                memory=static_map_features,
                                                                x_mask=None, 
                                                                memory_mask=mask_a2l) # [B,A,D]
        # if self.use_dynamic_map:
        #     map_aware_agent_features  = self.agent_tl_cross_attn(x=map_aware_agent_features, 
        #                                                      memory=dynamic_map_features,
        #                                                      x_mask=None, 
        #                                                      memory_mask=mask_a2tl) # [B,A,D]

        # TODO: test local attn
        # Aggregate self-context-aware and map-context-aware agent features
        # to generate full-context-aware agent features
        ## NOTE: here we use star-operation + to perform aggregation, 
        ## other aggregation functions also acceptable, such as *, torch.cat, etc.
        context_aware_agent_features = self_aware_agent_features + map_aware_agent_features      # [B,A,D]

        # ego current status features
        raw_ego_current_status_features = torch.cat([raw_agent_features[:,[0], 0, 3:7], raw_agents_lw[:,[0]]], dim=-1) # [B,1,6], 6->vx,vy, heading_sin, heading_cos, l, w
        ego_current_status_features = self.ego_current_status_feature_encode_mlp(raw_ego_current_status_features) # [B,1,D]

        ego_routing_features  = torch.max(ego_routing_features, dim=-2)[0].unsqueeze(1)          # [B,1,D]

        raw_ego_navi_features = raw_ego_navi_features * ego_navi_valid_on_N
        ego_navi_features = self.ego_navi_point_net(raw_ego_navi_features) # [B,Nv,D]
        ego_navi_features = self.ego_navi_self_attn_on_Nv(ego_navi_features, mask=mask_nv2nv) # [B,Nv,D]
        ego_navi_features = torch.max(ego_navi_features, dim=-2)[0].unsqueeze(1) # [B,1,D]

        # ego_context_features  = ego_current_status_features + ego_routing_features + ego_traffic_light_features # [B,1,D]
        ego_context_features  = ego_current_status_features + ego_traffic_light_features# [B,1,D]

        ego_routing_features = ego_routing_features + ego_navi_features

        context_features = torch.cat((context_aware_agent_features, static_map_features), dim=1) # [B, A+L, D]
        return context_features, ego_context_features,ego_routing_features, mask_a2al #, agent_features_for_SSL, map_features_for_SSL

class DiTDiffusionDecoder(nn.Module):
    def __init__(self, config):
        super(DiTDiffusionDecoder, self).__init__()
        self.hidden_size = config['hidden_size']
        self.depth = config['depth']
        self.output_dim = 3
        self.num_heads = config['num_heads']
        self.preproj = MLP_2(in_features=self.output_dim, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.t_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        self.vector_in = MLP_2(in_features=self.hidden_size, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.context_in = MLP_2(in_features=self.hidden_size, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.blocks = nn.ModuleList([DiTBlock(dim=self.hidden_size, heads=self.num_heads, mlp_ratio=2.0) for i in range(self.depth)])
        self.final_layer = FinalLayer(hidden_size=self.hidden_size, output_size=self.output_dim)
        self.timesteps = 1000
        # self.register_buffer("input_tokens_mask", torch.tril(torch.ones(80, 80)).view(1, 80, 80))
        self.ego_time_pe = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=80)
        self.ego_routing_in = MLP_2(in_features=self.hidden_size, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)

    # @staticmethod
    # def normalize_state(state):
    #     state[...,0] = (state[...,0] - 0.85865366) / 0.47746402
    #     state[...,1] = (state[...,1] + 0.0041765613) / 0.11964693
    #     state[...,2] = state[...,2] / (math.pi/2)
    #     return state
    
    # @staticmethod
    # def denormalize_state(state):
    #     state[...,0] = state[...,0] * 0.47746402 + 0.85865366
    #     state[...,1] = state[...,1] * 0.11964693 - 0.0041765613
    #     state[...,2] = state[...,2] * (math.pi/2)
    #     return state


    # # 860 mine2
    @staticmethod
    def normalize_state(state):
        state[...,0] = (state[...,0] - 1.0528826) / 0.37231752
        state[...,1] = (state[...,1] + 0.009744605) / 0.09104148
        state[...,2] = state[...,2] / (math.pi/2)
        return state
    
    @staticmethod
    def denormalize_state(state):
        state[...,0] = state[...,0] * 0.37231752 + 1.0528826
        state[...,1] = state[...,1] * 0.09104148 - 0.009744605
        state[...,2] = state[...,2] * (math.pi/2)
        return state

    def prepare_input(self, model_input):
        valid_agents_future_gt, valid_agents_future_gt_valid = model_input['valid_agents_future_gt'], model_input['valid_agents_future_gt_valid'] # [B, A, T_f, 3], [B, A, T_f]
        valid_agents_past_features = model_input['valid_agents_features'] # [B, A, T, 33]
        B, A = valid_agents_past_features.shape[:2]
        # valid_agents_current_global_position = valid_agents_past_features[:, :, [-1], -5:-3] # [B, A, 1, 2]
        # valid_agents_current_global_heading = valid_agents_past_features[:, :, [-1], [-2]].unsqueeze(-1) # [B, A, 1, 1]
        valid_agents_current_per_agent_centric_position = valid_agents_past_features[:, :, [-1], 0:2] # [B, A, 1, 2]
        valid_agents_current_per_agent_centric_heading = torch.zeros((B,A,1,1), device=valid_agents_past_features.device, requires_grad=False) #valid_agents_past_features[:, :, [-1], [-2]].unsqueeze(-1) # [B, A, 1, 1]
        valid_agents_current_gt_valid = valid_agents_past_features[:, :, [-1], -1] # [B, A, 1]
        valid_agents_current_gt = torch.cat([valid_agents_current_per_agent_centric_position, valid_agents_current_per_agent_centric_heading], dim=-1) # [B, A, 1, 3]
        tokens = torch.cat([valid_agents_current_gt, valid_agents_future_gt], dim=-2) # [B, A, T_f+1, 3]
        # NOTE: apply state normalization
        delta_tokens = tokens[:, :, 1:, :] - tokens[:, :, :-1, :] # [B, A, T_f, 3]
        ## Normalized to normal distribution
        
        output_tokens = delta_tokens
        output_tokens = self.normalize_state(output_tokens)
        tokens_valid = torch.cat([valid_agents_current_gt_valid, valid_agents_future_gt_valid], dim=-1).unsqueeze(-1) # [B, A, T_f+1, 1]
        tokens_valid = tokens_valid[:, :, 1:, :] * tokens_valid[:, :, :-1, :] # [B, A, T_f, 1]  
        output_tokens = output_tokens * tokens_valid
        B, A, T_f, _ = output_tokens.shape

        ego_output_tokens = output_tokens[:, [0], :, :] # [B, 1, T_f, 3]
        ego_output_tokens_valid = tokens_valid[:, [0], :, :] # [B, 1, T_f, 1]
        input_noise = torch.randn_like(ego_output_tokens) # [B, 1, T_f, 3]
        input_tokens_mask = torch.matmul(ego_output_tokens_valid, ego_output_tokens_valid.transpose(-2,-1)).squeeze(1) # [B, T_f, T_f]
        return ego_output_tokens, ego_output_tokens_valid, input_noise, input_tokens_mask

    def model_forward(self, z_t, context, t, vec, ego_routing_features, input_tokens_mask, mask_ego2al):
        B, _, T_f, _ = z_t.shape
        z_t = self.preproj(z_t).squeeze(1) # [B, 80, D]
        z_t = self.ego_time_pe(z_t) # [B, 80, D]
        z_t = z_t + self.vector_in(vec) # [B, 80, D]

        y = self.t_embedder(t.reshape((B, )) * self.timesteps) # [B, D]

        y = y + self.ego_routing_in(ego_routing_features[:,0,:]) # [B, D]
        
        context = self.context_in(context) # [B, A, D]
        for block in self.blocks:
            z_t = block(z_t, context, y, input_tokens_mask, mask_ego2al)
        pred = self.final_layer(z_t, y)
        pred = pred.reshape(B, 1, T_f, -1) # [B, 1, T_f+1, 3]
        return pred

    def forward(self, model_input, context, ego_context_features, ego_routing_features, mask_a2al):
        ego_output_tokens, ego_output_tokens_valid, input_noise, input_tokens_mask = self.prepare_input(model_input)
        B, _, T_f, _ = ego_output_tokens.shape
        

        # rectified flow
        nt = torch.randn((B,)).to(input_noise.device)
        t = torch.sigmoid(nt)
        # t = torch.where(torch.rand_like(t) <= 0.9, t, torch.rand_like(t))

        t = t.view([B, *([1] * len(input_noise.shape[1:]))]) # [B, 1, 1, 1]
        z_t = t * ego_output_tokens + (1-t) * input_noise # [B, 1, T_f+1, 3]
        target = ego_output_tokens - input_noise
        pred = self.model_forward(z_t, context, t, ego_context_features, ego_routing_features, input_tokens_mask, mask_a2al[:, :1, :].repeat(1, 80, 1))
        loss = (pred - target) ** 2
        mask = ego_output_tokens_valid.repeat(1, 1, 1, 3) # [B, 1, T_f+1, 3]
        # loss_list = []
        # for i in range(B):
        #     loss_list.append(loss[i][mask[i].nonzero(as_tuple=True)].mean())
        # return torch.stack(loss_list).mean()
        loss = loss * mask
        return loss.mean()
    
    @torch.no_grad()
    def sample(self, T, context, ego_context_features,ego_routing_features, mask_a2al):
        B = ego_context_features.shape[0]
        input_noise = torch.randn((B, 1, 80, 3), device=ego_context_features.device)

        samples = [input_noise.clone()]
        for i in range(T):
            t = torch.ones((B, ), device=input_noise.device) * i / T
            drift_pred = self.model_forward(samples[-1], context, t, ego_context_features, ego_routing_features, None, mask_a2al[:, :1, :].repeat(1, 80, 1))
            pred_trajs = samples[-1] + drift_pred * 1. / T
            samples.append(pred_trajs)
        # for t_next, t_cur in zip(T[1:], T[:-1]):
        #     t = torch.full((B, ), t_cur, device=input_noise.device)
        #     z_t = self.preproj(samples[-1].reshape(B, 1, -1))
        #     y = self.t_embedder(t * self.timesteps)
        #     y = y + ego_context_features[:,0,:] # [B, D]
        #     for block in self.blocks:
        #         z_t = block(z_t, context, y, None, mask_a2al[:, :1, :])
        #     drift_pred = self.final_layer(z_t, y)
        #     drift_pred = drift_pred.reshape(B, 1, T_f, 3)
        #     pred_trajs = samples[-1] + drift_pred * (t_next - t_cur)
        #     # pred_trajs[:, :, [0], :] = ego_output_tokens[:, :1, :1, :]
        #     samples.append(pred_trajs)
        return samples

#*****************************
#******** Model utils ********
#*****************************
class SubgraphNet(nn.Module):
    def __init__(self, input_size, hidden_size, depth=1):
        super().__init__()
        half_hidden_size = hidden_size // 2
        self.sub_layers = nn.ModuleList([
            SubgraphLayer(input_size, half_hidden_size)]
            + [SubgraphLayer(hidden_size, half_hidden_size) for _ in range(depth-1)])
    
    def forward(self, x):
        for layer in self.sub_layers:
            x = layer(x) # [B, L, N ,D]
        x = torch.max(x, dim=-2)[0] # [B, L, D]
        return x

class SubgraphLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        x = self.fc1(x) # [B,L,N,D]
        x = F.relu(F.layer_norm(x, x.shape[-2:]))
        x = self.fc2(x) # [B,L,N,D]
        pooled_x = torch.max(x, dim=-2)[0].unsqueeze(-2) # [B,L,1,D/2]
        repeated_pooled_x = pooled_x.repeat(1,1,x.shape[-2],1) # [B,L,N,D/2]
        output = torch.cat([x, repeated_pooled_x], dim=-1) # [B,L,N,D]
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num=8, bias=True, depth=1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size  = hidden_size
        self.head_num     = head_num
        assert self.hidden_size % self.head_num == 0
        self.head_size    = self.hidden_size // self.head_num
        self.scale_factor = self.head_size ** -0.5
        # W_q, W_k, W_v
        self.q_proj       = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.k_proj       = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.v_proj       = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        # Output projection
        self.o_proj       = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.o_proj.RESIDUAL_SCALE = depth

    def forward(self, q, k, v, mask=None):
        original_q_shape = q.shape # [B, T, q_numel, D] or [B, q_numel, D]
        pseudo_batch = original_q_shape[:-2] # [B, T] or [B, ]

        # Project input query, key, value by W_q, W_k, W_v
        # and reshape hidden size to multi-head size
        q = self.q_proj(q).view(*pseudo_batch, -1, self.head_num, self.head_size).transpose(-3, -2) # [B, T, h, q_numel, d] or [B, h, q_numel, d]
        k = self.k_proj(k).view(*pseudo_batch, -1, self.head_num, self.head_size).transpose(-3, -2) # [B, T, h, k_numel, d] or [B, h, k_numel, d]
        v = self.v_proj(v).view(*pseudo_batch, -1, self.head_num, self.head_size).transpose(-3, -2) # [B, T, h, k_numel, d] or [B, h, k_numel, d]

        # Scaled dot product attention. original version
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale_factor # [B, T, h, q_numel, k_numel] or [B, h, q_numel, k_numel]
        if mask is not None:
            mask = mask.unsqueeze(dim=-3)                                     # [B, T, 1, q_numel, k_numel] or [B, 1, q_numel, k_numel]
            attn_scores = attn_scores.masked_fill(mask==0, value=-6e4)
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
    def __init__(self, hidden_size, depth=1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn      = MultiHeadAttention(hidden_size=hidden_size, depth=depth)
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.ffn            = PointWiseFeedForward(d_model=hidden_size,d_ffn=hidden_size, depth=depth)
        self.ffn_norm       = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        # Self attention and residual connection
        self_attn_layer = lambda x: self.self_attn(q=x, k=x, v=x, mask=mask)
        x = x + self_attn_layer(self.self_attn_norm(x))
        # Feed forward network and residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, enable_self_attn=False, depth=3):
        super(TransformerDecoderLayer, self).__init__()
        self.enable_self_attn   = enable_self_attn
        if enable_self_attn:
            self.self_attn      = MultiHeadAttention(hidden_size=hidden_size, depth=depth)
            self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.cross_attn         = MultiHeadAttention(hidden_size=hidden_size, depth=depth)
        self.cross_attn_norm    = nn.LayerNorm(hidden_size)
        self.ffn                = PointWiseFeedForward(d_model=hidden_size,d_ffn=hidden_size, depth=depth)
        self.ffn_norm           = nn.LayerNorm(hidden_size)

    def forward(self, x, memory, x_mask=None, memory_mask=None):
        # Self attention and residual connection
        if self.enable_self_attn:
            self_attn_layer = lambda x: self.self_attn(q=x, k=x, v=x, mask=x_mask)
            x = x + self_attn_layer(self.self_attn_norm(x))
        # Cross attention and residual connection
        cross_attn_layer = lambda x: self.cross_attn(q=x, k=memory, v=memory, mask=memory_mask)
        x = x + cross_attn_layer(self.cross_attn_norm(x))
        # Feed forward network and redisual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, depth):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size, depth=depth) for _ in range(depth)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, depth, enable_self_attn=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(hidden_size, enable_self_attn, depth=depth) for _ in range(depth)])
        self.memory_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, memory, x_mask=None, memory_mask=None):
        memory = self.memory_norm(memory)
        for layer in self.layers:
            x = layer(x, memory, x_mask=x_mask, memory_mask=memory_mask)
        return x


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

    def forward(self, x, mask=None, t=None):
        """
        Arguments:
            x: Tensor, shape ``[B,A,T,D]``
        """
        if t is None:
            pe = self.pe
        else:
            pe = self.pe[:t]
        if mask is None:
            if len(x.shape) == 4:
                x = x + pe.unsqueeze(0).unsqueeze(0)
            elif len(x.shape) == 3:
                x = x + pe.unsqueeze(0)
        else:
            if len(x.shape) == 4:
                x = x + pe.unsqueeze(0).unsqueeze(0) * mask
            elif len(x.shape) == 3:
                x = x + pe.unsqueeze(0) * mask
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
    
from functools import partial
class MLP_2(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features=None, 
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
#*****************************
#******** MMDit utils ********
#*****************************
class RMSNorm(nn.Module):
    def __init__(
            self,
            dim: int,
            elementwise_affine: bool = False,
            eps: float = 1e-6,
            device=None,
            dtype=None
    ):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x

#*****************************
#********* Dit utils *********
#*****************************
def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
# def modulate(x, shift, scale, only_first=False):
#     if only_first:
#         x_first, x_rest = x[:, :1], x[:, 1:]
#         x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
#     else:
#         x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#     return x

def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))
    return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            # nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape
        
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x
    
class DiTBlock(nn.Module):
    def __init__(self, dim=128, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.modulate_norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, heads)
        self.modulate_norm2 = nn.LayerNorm(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp1 = MLP_3([dim, self.mlp_hidden_dim, self.mlp_hidden_dim, dim])
        self.mlp1 = MLP_2(in_features=dim, hidden_features=self.mlp_hidden_dim, out_features=dim, act_layer=nn.GELU, norm_layer=None)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.pre_cross_attn_norm = nn.LayerNorm(dim)
        self.pre_memory_norm = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, heads)
        self.post_cross_attn_norm = nn.LayerNorm(dim)
        # self.mlp2 = MLP_3([dim, self.mlp_hidden_dim, self.mlp_hidden_dim, dim])
        self.mlp2 = MLP_2(in_features=dim, hidden_features=self.mlp_hidden_dim, out_features=dim, act_layer=nn.GELU, norm_layer=None)

    def forward(self, x, cross_c, y, self_attn_mask, cross_attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)

        modulated_x = modulate(self.modulate_norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(modulated_x, modulated_x, modulated_x, self_attn_mask)

        modulated_x = modulate(self.modulate_norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        x = x + self.cross_attn(self.pre_cross_attn_norm(x), self.pre_memory_norm(cross_c), cross_c, cross_attn_mask)
        x = x + self.mlp2(self.post_cross_attn_norm(x))
        return x
