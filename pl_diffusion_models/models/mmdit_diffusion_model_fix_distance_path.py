import contextlib
import math
import os
import pickle
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning as L
from utils.loss_utils import batch_label_ADE
from scipy import interpolate
from losses.motion_prediction_loss import DirectGmmLoss
import numpy as np
np.set_printoptions(legacy='1.21')  # 仅支持 '1.13'/'1.21'/False，避免 FutureWarning
from tqdm import tqdm
from metrics.torchmetrics_mean_std_fix_distance_path import EgoMotionPlanningMetric_mean_std_fix_distance_path 
from metrics.torchmetrics import EgoSimpleMetric
from models.builder import LITMODEL
from utils.model_utils import (PositionalEncoding,
                                TransformerEncoder,
                                TransformerEncoderLayer,
                                TransformerDecoder,
                                TransformerDecoderLayer,
                                MLP_3)
from utils.data_utils import load_config
from models.modules.agent_encoder import AgentEncoder_path, AgentEncoder_traj, EgoEncoder_path
from models.modules.map_encoder import MapEncoder, MapEncoder_navi, MapEncoder_route, MapEncoder_occ
from models.layers.transformer import Transformer
from utils.misc_batch import MLP
from utils.utils import img2real, save_bad_batch
from utils.guidance_utils import costmap_guidance_func, L2_guidance_func, soft_L2_guidance_func, vis_guidance_grad, costmap_train_loss, soft_L2_train_loss

# ******************************************
# ******** Pytorch Lightning Model *********
# ******************************************
@LITMODEL.register_module()
class LitMMDiTDiffusionModel(L.LightningModule):
    def __init__(self, config=None, lr=0.001, warmup_steps=0,warmup_interval='step', enable_epoch_end_callback=False, enable_save_bad_batch=False, load_weights_only=False):
        super(LitMMDiTDiffusionModel, self).__init__()
        if config is None:
            config_path = os.path.join(os.getcwd(), 'config', 'model', self.__class__.__name__+'.yaml')
            config = load_config(config_path)
            config = config['model']
        elif isinstance(config, str):
            config_path = os.path.join(os.getcwd(), 'config', 'model', config)
            config = load_config(config_path)
            config = config['model']

        self.load_weights_only = load_weights_only
        self.future_steps = config['future_steps']
        self.future_steps_fixed = config['future_steps_fixed']
        self.planning_interval = config['planning_interval']
        self.planning_interval_fixed = config['planning_interval_fixed']
        self.use_aux_loss = config['decoder']['use_aux_loss']

        if config['compile']:
            self.model = torch.compile(MMDiTDiffusionModel(config))
        else:   
            self.model = MMDiTDiffusionModel(config)
            self.model.freeze_fix_distance()
        self.gmm_loss = DirectGmmLoss()
        self.statistic_mean_std = EgoMotionPlanningMetric_mean_std_fix_distance_path(is_waymo_dataset=True)
        self.test_metric_ego = EgoSimpleMetric(config['planning_interval'])
        self.is_multi_mode_sample = True
        self.mode_num = 8
        self.lr = lr
        self.pred_trajs_shape = None
        self.warmup_steps = warmup_steps
        self.warmup_interval = warmup_interval
        self.save_hyperparameters()
        self.enable_epoch_end_callback = enable_epoch_end_callback
        # enable saving bad batches when loss exceeds threshold (for debugging)
        self.enable_save_bad_batch = enable_save_bad_batch
        self.trainStage = config['decoder']['trainStage']

    def forward(self, x):
        return self.model.forward(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  # self.lr is the max lr

        # 估算总的训练 step 数，至少为 1，避免边界问题
        total_num = max(int(self.trainer.estimated_stepping_batches), 1)

        # 计算 warmup 的 step 数：按 step 或按 epoch 比例
        if self.warmup_interval == "step":
            warmup_steps = int(self.warmup_steps)
        else:
            max_epochs = max(int(self.trainer.max_epochs or 1), 1)
            warmup_steps = int(self.warmup_steps * total_num / max_epochs)

        # 保证 0 <= warmup_steps < total_num，避免 T_max <= 0
        warmup_steps = min(max(warmup_steps, 0), max(total_num - 1, 0))
        T_max = max(total_num - warmup_steps, 1)

        if warmup_steps > 0:
            # warmup scheduler: LinearLR
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            # main scheduler: CosineAnnealingLR
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
            )

            # combine Warmup and Cosine
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps],
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            # 无 warmup，仅使用 CosineAnnealingLR
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                ),
                "interval": "step",
                "frequency": 1,
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @classmethod
    def load_weights_from_checkpoint(cls, checkpoint_path, config=None, **kwargs):
        """
        手动加载模型权重，完全绕过Lightning的checkpoint加载机制
        
        Args:
            checkpoint_path: checkpoint文件路径
            config: 模型配置
            **kwargs: 其他初始化参数（如lr, use_constant_lr等）
        
        Returns:
            加载了权重的模型实例
        """
        import torch
        
        # 创建模型实例
        model = cls(config=config, **kwargs)
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取state_dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 处理state_dict的key，移除可能的前缀
        # Lightning checkpoint中，state_dict的key通常有"model."前缀
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     # 移除"model."前缀（如果存在）
        #     if k.startswith('model.'):
        #         new_key = k[6:]  # 移除"model."前缀
        #     else:
        #         new_key = k
        #     new_state_dict[new_key] = v
        
        # 加载权重（使用strict=False以允许部分匹配）
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if missing_keys:
            print(f"Warning: Missing keys when loading weights: {missing_keys[:5]}...")  # 只显示前5个
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")  # 只显示前5个
        
        return model

    def on_after_backward(self):
        # 遍历所有参数检查未参与反传的参数比较耗时，仅用于调试。
        # 默认关闭；如需开启请设置环境变量：export ENABLE_AFTER_BACKWARD_UNUSED_PARAMS_CHECK=true
        if os.getenv("ENABLE_AFTER_BACKWARD_UNUSED_PARAMS_CHECK", "").strip().lower() != "true":
            return
        unused = []
        for name, p in self.named_parameters():
            if p.requires_grad and p.grad is None:
                unused.append(name)
        if len(unused):
            print(">>> UNUSED PARAMS (no grad):")
            for n in unused:
                print("  -", n)
        return

    def training_step(self, batch, batch_idx):
        model_input = batch['model_input']
        # Expose trainer state to decoder for conditional batch saving/debugging.
        # (Decoder is a plain nn.Module, so it cannot directly access Lightning's epoch/step.)
        loss_dict = self.model((model_input))
        loss = loss_dict['loss']
        # Save bad batches for debugging when loss is abnormally high after warmup epochs.
        # Save model_input from all ranks with loss > threshold (0.5), each rank saves its own model_input.
        if not self.enable_save_bad_batch:
            # skip saving if disabled in config
            pass
        else:
            try:
                cur_epoch = int(getattr(self, "current_epoch", -1))
                if cur_epoch <= 10:
                    # skip saving in early epochs
                    pass
                else:
                    # use detached scalar for distributed checks (avoid autograd sync overhead)
                    if 'stage_1' == self.trainStage:
                        loss_for_check = loss_dict['loss_path'].detach()
                    if 'stage_2' == self.trainStage:
                        loss_for_check = loss_dict['loss_traj'].detach()
                    loss_threshold = 0.5
                    save_root = str(getattr(self.trainer, "default_root_dir", os.getcwd()))
                    cur_step = int(getattr(self, "global_step", -1))
                    cur_batch_idx = int(batch_idx)
                    # check if any rank has high loss (quick global max check to avoid all_gather on most steps)
                    is_distributed = dist.is_available() and dist.is_initialized()
                    if is_distributed:
                        max_loss_global = loss_for_check.clone()
                        dist.all_reduce(max_loss_global, op=dist.ReduceOp.MAX)
                        # only proceed if global max exceeds threshold
                        if max_loss_global <= loss_threshold:
                            pass  # skip saving
                        else:
                            # check if current rank's loss exceeds threshold
                            # each rank with loss > threshold saves its own model_input
                            if loss_for_check.item() > loss_threshold:
                                current_rank = dist.get_rank()
                                save_bad_batch(
                                    model_input=model_input,
                                    loss_for_check=loss_for_check,
                                    rank=current_rank,
                                    cur_epoch=cur_epoch,
                                    cur_step=cur_step,
                                    cur_batch_idx=cur_batch_idx,
                                    save_root=save_root,
                                )
                    else:
                        # single-process fallback: check local loss only
                        if loss_for_check > loss_threshold:
                            rank = 0
                            save_bad_batch(
                                model_input=model_input,
                                loss_for_check=loss_for_check,
                                rank=rank,
                                cur_epoch=cur_epoch,
                                cur_step=cur_step,
                                cur_batch_idx=cur_batch_idx,
                                save_root=save_root,
                            )
            except Exception as e:
                # never fail training due to debug-saving errors, but log the error
                import traceback
                print(f"[ERROR] Failed to save bad batch: {e}")
                traceback.print_exc()                

        if not torch.isfinite(loss):
            print(f"nan in loss: {loss}")
            zero_loss = sum(p.sum() for p in self.parameters()) * 0.0
            if self.use_aux_loss:
                self.log_dict({
                    "train_loss": zero_loss, "train_loss_traj": zero_loss, 
                    "train_loss_path": zero_loss,"loss_path_navi_x": zero_loss, "loss_path_navi_y": zero_loss,
                    "loss_path_norm": zero_loss, "loss_path_navi_x_norm": zero_loss, "loss_path_navi_y_norm": zero_loss,
                    "ema_path": zero_loss, "ema_path_navi_x": zero_loss, "ema_path_navi_y": zero_loss,               
                }, sync_dist=True, prog_bar=True)
            else:
                self.log_dict({"train_loss":zero_loss, "train_loss_traj":zero_loss, "train_loss_path":zero_loss}, sync_dist=True, prog_bar=True)
            return zero_loss

        if self.use_aux_loss:
            self.log_dict({
                        "train_loss": loss, "train_loss_traj": loss_dict['loss_traj'], 
                        "train_loss_path": loss_dict['loss_path'], "loss_path_navi_x": loss_dict['loss_path_navi_x'], "loss_path_navi_y": loss_dict['loss_path_navi_y'],
                        "loss_path_norm": loss_dict['loss_path_norm'],"loss_path_navi_x_norm": loss_dict['loss_path_navi_x_norm'], "loss_path_navi_y_norm": loss_dict['loss_path_navi_y_norm'],
                        "ema_path": loss_dict['ema_path'],  "ema_path_navi_x": loss_dict['ema_path_navi_x'], "ema_path_navi_y": loss_dict['ema_path_navi_y'],
            }, sync_dist=True, prog_bar=True)
        else:
            self.log_dict({"train_loss":loss, "train_loss_traj": loss_dict['loss_traj'], "train_loss_path": loss_dict['loss_path']}, sync_dist=True, prog_bar=True)
        if self.enable_epoch_end_callback:
            self.statistic_mean_std.update(model_input)
        return loss
    
    def validation_step(self, batch, batch_idx):
        model_input = batch['model_input']
        loss_dict = self.model((model_input))
        loss = loss_dict["loss"]
        # 只记录标量 val_loss，避免将 dict 直接传给 logger
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        model_input = batch['model_input']
        B = model_input['agent_attrs'].shape[0]
        device = model_input['agent_attrs'].device
        num_samples = self.mode_num if self.is_multi_mode_sample else 1
        
        # [B, ...] -> [B*num_samples, ...]
        expanded_model_input = {}
        for key, value in model_input.items():
            if isinstance(value, torch.Tensor):
                expanded = value.unsqueeze(1).expand(-1, num_samples, *[-1]*(value.dim()-1))
                expanded = expanded.reshape(B * num_samples, *value.shape[1:])
                expanded_model_input[key] = expanded
            else:
                expanded_model_input[key] = value
        
        input_noise = torch.randn((B * num_samples, 1, self.future_steps//int(self.planning_interval/0.1), 3), device=device)
        input_noise_fix_distance = torch.randn((B * num_samples, 1, self.future_steps_fixed//int(self.planning_interval_fixed/1.0), 3), device=device)
        
        pred_trajs, fix_distance_pred_paths = self.model.sample(
            expanded_model_input, input_noise, input_noise_fix_distance
        )
        
        pred_trajs = pred_trajs.reshape(B, num_samples, 1, self.future_steps//int(self.planning_interval/0.1), 3)
        fix_distance_pred_paths = fix_distance_pred_paths.reshape(B, num_samples, 1, self.future_steps_fixed//int(self.planning_interval_fixed/1.0), 3)
        
        model_output_dict = {
            'pred_trajs': pred_trajs,  # [B, num_samples, 1, 80, 3]
            'pred_fix_distace_path': fix_distance_pred_paths
        }
        ego_future_dict = {
            'ego_future_status': model_input['ego_future_status'],  # [B, 80, 5]
            'ego_future_mask': model_input['ego_future_mask'],  # [B, 80]
            'ego_future_status_fixed': model_input['ego_future_status_fixed'],
            'ego_future_mask_fixed': model_input['ego_future_mask_fixed']
        }
        self.test_metric_ego.update(model_output_dict, ego_future_dict)

    def on_train_epoch_end(self):
        if not self.enable_epoch_end_callback:
            return
        if hasattr(self, 'statistic_mean_std') and len(self.statistic_mean_std.ego_gt_xy) > 0:
            metrics_output = self.statistic_mean_std.compute()
            if self.trainer.is_global_zero:
                import pprint
                pp = pprint.PrettyPrinter(depth=4, sort_dicts=False)
                print(f"Train Epoch {self.current_epoch} metrics:")
                pp.pprint(metrics_output)
            self.statistic_mean_std.reset()
    
    def on_test_epoch_end(self):
        metrics_output = self.test_metric_ego.compute()
        if self.trainer.is_global_zero:
            print("\n" + "=" * 60)
            print(f"{'Ego Simple Metrics':^60}")
            print("=" * 60)
            print(f"{'Metric':<20} {'ADE':>15} {'FDE':>15}")
            print("-" * 60)
            print(f"{'Trajectory 3s':<20} {metrics_output['ade_3s']:>15.4f} {metrics_output['fde_3s']:>15.4f}")
            print(f"{'Trajectory 5s':<20} {metrics_output['ade_5s']:>15.4f} {metrics_output['fde_5s']:>15.4f}")
            print(f"{'Trajectory 8s':<20} {metrics_output['ade_8s']:>15.4f} {metrics_output['fde_8s']:>15.4f}")
            print("-" * 60)
            print(f"{'Path 3s':<20} {metrics_output['ade_3s_path']:>15.4f} {metrics_output['fde_3s_path']:>15.4f}")
            print(f"{'Path 5s':<20} {metrics_output['ade_5s_path']:>15.4f} {metrics_output['fde_5s_path']:>15.4f}")
            print(f"{'Path 8s':<20} {metrics_output['ade_8s_path']:>15.4f} {metrics_output['fde_8s_path']:>15.4f}")
            print("-" * 60)
            print(f"{'Multi Traj Cosline':<20} {metrics_output['multi_traj_mean_cosline']:>15.4f}")
            print("=" * 60)
            print(f"{'Num Samples':<20} {metrics_output['num_samples']:>15}")
            print("=" * 60 + "\n")
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

# ********************************
# ******** Multi TF Model ********
# ********************************
class MMDiTDiffusionModel(nn.Module):
    def __init__(self, config):
        super(MMDiTDiffusionModel, self).__init__()
        # self.encoder = MMDiTDiffusionEncoder(config['encoder'])
        self.encoder_path = SceneEncoder_path(config['encoder'])
        self.encoder_traj = SceneEncoder_traj(config['encoder'])
        self.decoder = DiTDiffusionDecoder(config['decoder'])
        self.navi_drop_prob = config['encoder'].get('navi_drop_prob', 0.5)
        self.infer_fake_route_with_navitopo = config['encoder'].get('infer_fake_route_with_navitopo', True)
        self.use_guidance = config['decoder'].get('use_guidance', 0)
        self.use_navi_guidance = config['decoder'].get('use_navi_guidance', 0)
        self.use_obs_guidance = config['decoder'].get('use_obs_guidance', 0)
        self.std_scaler_traj = config['decoder'].get('std_scaler_traj', 1)
        self.std_scaler_path = config['decoder'].get('std_scaler_path', 1)
        self.apply(self._init_weights)
        self.trainStage = config['decoder'].get('trainStage', 'stage_1')

    def freeze_fix_distance(self):
        if self.trainStage == 'stage_1':
            # 先暂时不加，path的部分直接没有traj的loss，再动梯度反而会慢
            #for name, p in self.named_parameters():
            #    if 'encoder' in name:
            #        p.requires_grad = True
            #    elif 'decoder' in name and ('fix_distance' in name or 'fixdistance' in name):
            #        p.requires_grad = True
            #    else:
            #        p.requires_grad = False
            pass

        if self.trainStage == 'stage_2':
            for name, p in self.named_parameters():
                if 'encoder_path' in name:
                    p.requires_grad = False
                elif 'decoder' in name and ('fix_distance' in name or 'fixdistance' in name):
                    p.requires_grad = False
                else:
                    p.requires_grad = True

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

    def _replace_route_with_nearest_navitopo(self, model_input_path):
        """推理时：将 route 替换为距离 route 最远的 navitopo，实现输入层面的 fake。"""
        if "route_pts" not in model_input_path or "navitopo_pts" not in model_input_path:
            return
        route_pts = model_input_path["route_pts"]   # (bs, max_poly, n_pt, 2)
        navitopo_pts = model_input_path["navitopo_pts"]  # (bs, N, P, 2)
        navitopo_mask = model_input_path.get("navitopo_mask")  # (bs, N), 1=invalid
        bs, max_poly, n_pt, _ = route_pts.shape
        device = route_pts.device
        N, P = navitopo_pts.shape[1], navitopo_pts.shape[2]
        # 计算每条 navitopo 到当前 route 的最小距离（route 点与 navitopo 点对之间的最小距离）
        min_dist_per_line = torch.full((bs, N), float("inf"), device=device, dtype=route_pts.dtype)
        for b in range(bs):
            route_b = route_pts[b, 0]   # (n_pt, 2)，取第一段 route
            navi_b = navitopo_pts[b]     # (N, P, 2)
            # 忽略全零的 route，避免无意义距离
            if route_b.abs().sum() < 1e-6:
                continue
            # diff: (N, n_pt, P, 2) -> norm -> (N, n_pt, P)
            diff = route_b.unsqueeze(0).unsqueeze(2) - navi_b.unsqueeze(1)
            dist_mat = torch.norm(diff, dim=-1)   # (N, n_pt, P)
            min_dist_per_line[b] = dist_mat.view(N, -1).min(dim=1).values  # (N,)
        # 选最远：invalid 置为 -inf，argmax 得到距离 route 最远的有效 navitopo
        if navitopo_mask is not None:
            min_dist_per_line = torch.where(
                navitopo_mask.bool(),
                torch.full_like(min_dist_per_line, float("-inf"), device=device, dtype=min_dist_per_line.dtype),
                min_dist_per_line,
            )
        farthest_idx = min_dist_per_line.argmax(dim=1)   # (bs,)
        new_route_pts = torch.zeros_like(route_pts)
        for b in range(bs):
            idx = farthest_idx[b].item()
            # route 全零时未更新距离，全为 inf，argmax 无意义，不替换
            if torch.isinf(min_dist_per_line[b].max()):
                continue
            line = navitopo_pts[b, idx]   # (P, 2)
            if navitopo_mask is not None and navitopo_mask[b, idx] != 0:
                # 无有效 navitopo，保留全零
                continue
            if P >= n_pt:
                new_route_pts[b, 0, :, :] = line[:n_pt]
            else:
                new_route_pts[b, 0, :P, :] = line
        if max_poly > 1:
            # 只填第一段 route，其余保持零
            pass
        model_input_path["route_pts"] = new_route_pts
        # 替换为 navitopo 后清空 route 的 lane 属性，避免混用真实 route 的语义
        if "route_lane_attrs" in model_input_path:
            model_input_path["route_lane_attrs"] = torch.zeros_like(model_input_path["route_lane_attrs"])
        if "route_lane_recommend_mask" in model_input_path:
            model_input_path["route_lane_recommend_mask"] = torch.zeros_like(model_input_path["route_lane_recommend_mask"])

    def _model_input_zeronavi(self, model_input):
        """训练阶段按概率随机drop navitopo（仅 encoder 输入）；推理阶段固定将 model_input 中所有 navitopo 置零，route 始终保留；若启用则先将 route 替换为最近 navitopo。"""
        model_input_path = dict(model_input)

        def _zero_navitopo(m, include_rs=False):
            if "navitopo_pts" in m:
                m["navitopo_pts"] = torch.zeros_like(m["navitopo_pts"])
            if "navitopo_mask" in m:
                m["navitopo_mask"] = torch.ones_like(m["navitopo_mask"])
            if "navitopo_attrs" in m:
                m["navitopo_attrs"] = torch.zeros_like(m["navitopo_attrs"])
            if include_rs:
                if "navitopo_rs" in m:
                    m["navitopo_rs"] = torch.zeros_like(m["navitopo_rs"])
                if "navitopo_rs_mask" in m:
                    m["navitopo_rs_mask"] = torch.zeros_like(m["navitopo_rs_mask"])

        if self.training:
            if self.navi_drop_prob <= 0.0:
                return model_input_path
            if "navitopo_pts" in model_input_path:
                navitopo_pts = model_input_path["navitopo_pts"]
                if torch.sum(navitopo_pts) != 0 and random.random() < self.navi_drop_prob:
                    _zero_navitopo(model_input_path, include_rs=False)
        else:
            # pass
            # 推理：可选将 route 替换为距离自车最近的 navitopo（输入层面 fake）
            # if self.infer_fake_route_with_navitopo:
            #     self._replace_route_with_nearest_navitopo(model_input_path)
            _zero_navitopo(model_input_path, include_rs=True)
        return model_input_path

    def forward(self, x):
        model_input = x
        model_input_path = self._model_input_zeronavi(model_input)
        if self.trainStage == 'stage_1':
            encoder_output_path = self.encoder_path(model_input_path)
            encoder_output_traj = self.encoder_traj(model_input)
            # decoder 用原始 model_input，使 path loss（navitopo_rs / navitopo_rs_mask）仍基于真实 navitopo
            prediction = self.decoder(model_input, encoder_output_path, encoder_output_traj)
        elif self.trainStage == 'stage_2':
            encoder_output_path = self.encoder_path(model_input_path)
            encoder_output_traj = self.encoder_traj(model_input)
            # path loss 仍使用真实 navitopo
            prediction = self.decoder(model_input, encoder_output_path, encoder_output_traj)
        else:
            return None
        return prediction

    @torch.no_grad()
    def sample(self, model_input, input_noise, input_noise_fix_distance, sample_steps=5):
        model_input_path = self._model_input_zeronavi(model_input)
        encoder_output_path = self.encoder_path(model_input_path)
        encoder_output_traj = self.encoder_traj(model_input)
        repeated_encoder_output_path = []
        repeated_encoder_output_traj = []
        for i in range(len(encoder_output_path)):
            # 只在 batch size 不匹配时才进行 repeat（用于可视化场景）
            # 在测试时 model_input 已经被扩展，encoder_output 的 batch size 已经匹配，不需要 repeat
            if encoder_output_path[i].shape[0] != input_noise.shape[0]:
                repeated_encoder_output_path.append(encoder_output_path[i].repeat(input_noise.shape[0], 1, 1))
                repeated_encoder_output_traj.append(encoder_output_traj[i].repeat(input_noise.shape[0], 1, 1))
            else:
                repeated_encoder_output_path.append(encoder_output_path[i])
                repeated_encoder_output_traj.append(encoder_output_traj[i])

        if self.use_guidance and (self.use_navi_guidance or self.use_obs_guidance):
            pred_trajs, fix_distance_pred_paths = self.decoder.path_guidance_sample(input_noise, input_noise_fix_distance, sample_steps, repeated_encoder_output_path, repeated_encoder_output_traj, model_input_path, use_navi_guidance=self.use_navi_guidance, use_obs_guidance=self.use_obs_guidance) # [B, 1, 81, 3]
        else:
            pred_trajs, fix_distance_pred_paths = self.decoder.sample(input_noise, input_noise_fix_distance, sample_steps, repeated_encoder_output_path, repeated_encoder_output_traj, 'infer') # [B, 1, 81, 3]

        pred_trajs, fix_distance_pred_paths = pred_trajs[-1], fix_distance_pred_paths[-1]
        pred_trajs = self.decoder.denormalize_state(pred_trajs, model_input, self.std_scaler_traj)
        fix_distance_pred_paths = self.decoder.denormalize_fix_distance_path_state(fix_distance_pred_paths, model_input)
        pred_trajs = torch.cumsum(pred_trajs, dim=-2) # [B, 1, 80, 3]
        fix_distance_pred_paths = torch.cumsum(fix_distance_pred_paths, dim=-2) # [B, 1, 80, 3]

        return pred_trajs, fix_distance_pred_paths

    def _convert_tuple_to_dict_14(self, data):
        """将14个输入元组转换为编码器期望的字典格式"""
        # 根据 get_input_info 中的顺序定义14个键
        keys = [
            'agent_status',           # 0: 智能体状态
            'agent_attrs',            # 1: 智能体属性
            'agent_time_mask',        # 2: 智能体时间掩码
            'laneline_pts',           # 3: 车道线点
            'laneline_mask',          # 4: 车道线掩码
            'ego_curr_status',         # 5: 自车当前状态
            'category_feature',       # 6: 类别特征
            'traffic_light_feature',  # 7: 交通灯特征
            'car_light_feature',      # 8: 车灯特征
            'polygon_color_feature',  # 9: 多边形颜色特征
            'polygon_laneline_style_feature',  # 10: 车道线样式特征
            'polygon_laneline_type_feature',   # 11: 车道线类型特征
            'navitopo_pts',           # 12: 导航拓扑点
            'navitopo_mask',           # 13: 导航拓扑掩码
            'mean_std',
            'mean_std_fixed',
            'occ_polygons_pts',
            'occ_polygons_attrs',
            'occ_polygons_mask'
            # 'delta_x_mean',
            # 'delta_x_std',
            # 'delta_y_mean',
            # 'delta_y_std',
            # 'delta_fixed_x_mean',
            # 'delta_fixed_x_std',
            # 'delta_fixed_y_mean',
            # 'delta_fixed_y_std',
            # 'delta_yaw_mean',
            # 'delta_yaw_std',
            # 'delta_fixed_yaw_mean',
            # 'delta_fixed_yaw_std'
        ]

        # 确保长度匹配
        if len(data) != len(keys):
            print(f"⚠️  输入元组长度({len(data)})与期望键数({len(keys)})不匹配")
            # return self._adaptive_tuple_to_dict_14(input_tuple, keys)
        data_dict = {
            "agent_status": data[0],
            "agent_attrs": data[1].squeeze(0),
            "agent_time_mask": data[2].squeeze(0),
            "laneline_pts": data[3],
            "laneline_mask": data[4].squeeze(0).squeeze(0),
            "ego_curr_status": data[5].squeeze(0).squeeze(0),
            "category_feature": data[6].squeeze(0),
            "traffic_light_feature": data[7].squeeze(0),
            "car_light_feature": data[8].squeeze(0),
            "polygon_color_feature": data[9].squeeze(0),
            "polygon_laneline_style_feature": data[10].squeeze(0),
            "polygon_laneline_type_feature": data[11].squeeze(0),
            "navitopo_pts": data[12],
            "navitopo_mask": data[13].squeeze(0).squeeze(0),
            "mean_std": data[14],
            "mean_std_fixed": data[15],
            "occ_polygons_pts": data[16],
            "occ_polygons_attrs": data[17],
            "occ_polygons_mask": data[18]
            # "delta_x_mean": data[14],
            # "delta_x_std": data[15],
            # "delta_y_mean": data[16],
            # "delta_y_std": data[17],
            # "delta_fixed_x_mean": data[18],
            # "delta_fixed_x_std": data[19],
            # "delta_fixed_y_mean": data[20],
            # "delta_fixed_y_std": data[21],
            # "delta_yaw_mean": data[22],
            # "delta_yaw_std": data[23],
            # "delta_fixed_yaw_mean": data[24],
            # "delta_fixed_yaw_std": data[25]
        }

        return data_dict 

    @torch.no_grad()
    def sample_onnx(self, sample_input):
        model_input = sample_input[:18]
        # import pdb;pdb.set_trace()
        input_noise = sample_input[18]
        input_noise_fix_distance = sample_input[19]
        sample_steps = sample_input[20]
        model_input_dict = self._convert_tuple_to_dict_14(model_input)
        encoder_output = self.encoder(model_input_dict)
        sample_steps = sample_steps[0].to(torch.int32)

        batch_size = input_noise.shape[0]
        pred_trajs, fix_distance_pred_paths = self.decoder.sample(input_noise, input_noise_fix_distance, sample_steps[0], *encoder_output) # [B, 1, 81, 3]
        # import pdb;pdb.set_trace()
        pred_trajs, fix_distance_pred_paths = pred_trajs[-1], fix_distance_pred_paths[-1]
        # pred_trajs = self.decoder.denormalize_state(pred_trajs)
        # fix_distance_pred_paths = self.decoder.denormalize_fix_distance_path_state(fix_distance_pred_paths)
        pred_trajs = self.decoder.denormalize_state(pred_trajs, model_input_dict, self.std_scaler_traj)
        fix_distance_pred_paths = self.decoder.denormalize_fix_distance_path_state(fix_distance_pred_paths, model_input_dict)
        # pred_trajs = torch.cumsum(pred_trajs, dim=-2) # [B, 1, 80, 3]
        # fix_distance_pred_paths = torch.cumsum(fix_distance_pred_paths, dim=-2) # [B, 1, 80, 3]
        pred_trajs_last = torch.cumsum(pred_trajs, dim=-2) # [B, 1, 80, 3]
        fix_distance_pred_paths_last = torch.cumsum(fix_distance_pred_paths, dim=-2) # [B, 1, 80, 3]
        # import pdb;pdb.set_trace()
        pred_trajs_final = expand_and_repeat_trajectory(pred_trajs_last)
        fix_distance_pred_paths_final = expand_and_repeat_trajectory(fix_distance_pred_paths_last)

        return pred_trajs_final, fix_distance_pred_paths_final

    @torch.no_grad()
    def multi_mode_sample(self, sample_input):
        model_input = sample_input[:19]
        input_noise = sample_input[19]
        input_noise_fix_distance = sample_input[20]
        sample_steps = sample_input[21]
        model_input_dict = self._convert_tuple_to_dict_14(model_input)
        encoder_output = self.encoder(model_input_dict)
        sample_steps = sample_steps[0].to(torch.int32)

        # 获取批次大小
        batch_size = input_noise.shape[0]
        # refactor encoder output
        mm_encoder_output = []
        for e_out in encoder_output:
            mm_e_out = e_out.unsqueeze(dim=1).repeat(1, batch_size, 1, 1)
            mm_e_out = mm_e_out.reshape(torch.Size([-1]) + mm_e_out.shape[2:])
            mm_encoder_output.append(mm_e_out)

        # B = model_input['components_num'].shape[0]
        # input_noise = torch.randn((B * sample_num, 1, 80, 3), device=model_input['components_num'].device)

        pred_trajs, fix_distance_pred_paths = self.decoder.sample(input_noise, input_noise_fix_distance, sample_steps[0], *mm_encoder_output) # [B, 1, 81, 3]
        # import pdb;pdb.set_trace()
        pred_trajs, fix_distance_pred_paths = pred_trajs[-1], fix_distance_pred_paths[-1]
        # pred_trajs = self.decoder.denormalize_state(pred_trajs)
        # fix_distance_pred_paths = self.decoder.denormalize_fix_distance_path_state(fix_distance_pred_paths)
        pred_trajs = self.decoder.denormalize_state(pred_trajs, model_input_dict, self.std_scaler_traj)
        fix_distance_pred_paths = self.decoder.denormalize_fix_distance_path_state(fix_distance_pred_paths, model_input_dict)
        # pred_trajs = torch.cumsum(pred_trajs, dim=-2) # [B, 1, 80, 3]
        # fix_distance_pred_paths = torch.cumsum(fix_distance_pred_paths, dim=-2) # [B, 1, 80, 3]
        pred_trajs_last = torch.cumsum(pred_trajs, dim=-2) # [B, 1, 80, 3]
        fix_distance_pred_paths_last = torch.cumsum(fix_distance_pred_paths, dim=-2) # [B, 1, 80, 3]
        # import pdb;pdb.set_trace()
        # pred_trajs_last = extend_xy_yaw(pred_trajs_last, new_len=30, dt=0.2, method='cubic', extrap_mode='constant_velocity')
        # pred_trajs_last = extend_xy_yaw_torch_batch(
        #     pred_trajs_last, new_len=30, dt=0.2, method="linear", extrap_mode="constant_velocity"
        # )
        # pred_trajs_last = extend_xy_yaw_onnx_compatible(
        #     pred_trajs_last,
        #     new_len=30,
        #     dt=0.2,
        #     method="linear",
        #     extrap_mode="constant_velocity",
        #     yaw_calc_method="differential",
        # )
        # import pdb;pdb.set_trace()
        pred_trajs_final = expand_and_repeat_trajectory(pred_trajs_last)
        fix_distance_pred_paths_final = expand_and_repeat_trajectory(fix_distance_pred_paths_last)

        return pred_trajs_final, fix_distance_pred_paths_final

    # For visualization
    # @torch.no_grad()
    # def sample(self, model_input, input_noise, sample_steps=5):
    #     encoder_output = self.encoder(model_input)
    #     B = input_noise.shape[0]
    #     repeated_encoder_output = []
    #     for i in range(len(encoder_output)):
    #         repeated_encoder_output.append(encoder_output[i].repeat(B, 1, 1))
    #     pred_trajs, fix_distance_pred_paths = self.decoder.sample(input_noise, sample_steps, *repeated_encoder_output) # [B, 1, 81, 3]
    #     pred_trajs = pred_trajs[-1]
    #     fix_distance_pred_paths = fix_distance_pred_paths[-1]
    #     # pred_trajs = pred_trajs[:, :, 1:, :] # [B, 1, 80, 3]
    #     pred_trajs = self.decoder.denormalize_state(pred_trajs)
    #     fix_distance_pred_paths = self.decoder.denormalize_fix_distance_path_state(fix_distance_pred_paths)
    #     pred_trajs = torch.cumsum(pred_trajs, dim=-2) # [B, 1, 80, 3]
    #     fix_distance_pred_paths = torch.cumsum(fix_distance_pred_paths, dim=-2) # [B, 1, 80, 3]
    #     # print(f'pred_trajs: {pred_trajs[0,0, :, :2]}')

    #     return pred_trajs, fix_distance_pred_paths

# ******************************************
# ******** Multi TF Model Components********
# ******************************************
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

class SceneEncoder_path(nn.Module):
    def __init__(self, config):
        super(SceneEncoder_path, self).__init__()
        self.hidden_size              = config['hidden_size']
        self.depth                    = config['depth']
        self.mask_generator           = MultiTFMask()
        self.agent_hist_steps = config['agent_hist_steps']
        dim = config['hidden_size']
        self.transformer = Transformer(
            d_model = dim,
            nhead = config['nhead'],
            num_encoder_layers = config['transformer_layers'],
            num_decoder_layers = None,
            dim_feedforward = dim * 4,
            dropout=0.0,
            return_intermediate_dec = False
        )

        self.agent_encoder = AgentEncoder_path(
                    **config['agent_encoder'],
                    dim=dim,
                    nhead = config['nhead'],
                    num_encoder_layers = config['transformer_layers'],
                    num_decoder_layers = None,
                    hist_steps=config['agent_hist_steps'],
                )
        ego_hist_steps = int(config.get('ego_hist_steps', config['agent_hist_steps']))
        ego_time_max_len = max(ego_hist_steps + 2, config['agent_hist_steps'] + 2)
        self.ego_encoder = EgoEncoder_path(
                    **config['agent_encoder'],
                    dim=dim,
                    nhead = config['nhead'],
                    num_encoder_layers = config['transformer_layers'],
                    num_decoder_layers = None,
                    hist_steps=config['agent_hist_steps'],
                    ego_time_max_len=ego_time_max_len,
                )
        self.agent_ln = nn.LayerNorm(dim)
        self.ego_ln = nn.LayerNorm(dim)

        self.map_encoder = MapEncoder(
            **config['map_encoder'],
            dim=dim,
        )
        self.map_ln = nn.LayerNorm(dim)


        self.map_encoder_navi = MapEncoder_navi(
            **config['map_encoder_navi'],
            dim=dim,
        )
        self.map_ln_navi = nn.LayerNorm(dim)

        self.map_encoder_route = MapEncoder_route(
            **config['map_encoder_route'],
            dim=dim,
        )
        self.map_ln_route = nn.LayerNorm(dim)
        self.map_ln_lane = nn.LayerNorm(dim)

        self.map_encoder_occ = MapEncoder_occ(
            **config['map_encoder_occ'],
            dim=dim,
        )
        self.map_ln_occ = nn.LayerNorm(dim)

        self.pos_emb = MLP(4, dim, dim, 2)

        # 200m 内最近一条 MAIN_SIDE（左/右择路）单 token，与 navi_link_ms_* 对齐
        self.navi_link_ms_max_dist_m = float(config.get("navi_link_ms_max_dist_m", 200.0))
        self.navi_link_ms_kind_emb = nn.Embedding(3, dim, padding_idx=0)
        self.navi_link_ms_mlp = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.navi_link_ms_ln = nn.LayerNorm(dim)

        # 对比实验：最近一条 DEDICATED（左转/右转专用道）独立单 token，与 navi_link_ded_* 对齐（逻辑同 MAIN_SIDE）
        self.navi_link_ded_max_dist_m = float(
            config.get("navi_link_ded_max_dist_m", config.get("navi_link_ms_max_dist_m", 200.0))
        )
        self.navi_link_ded_kind_emb = nn.Embedding(3, dim, padding_idx=0)
        self.navi_link_ded_mlp = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.navi_link_ded_ln = nn.LayerNorm(dim)

    def forward(self, data):
        '''
        agent_attrs torch.Size([bs, n_max_agent, 3]): size_y, size_x, cls
        agent_status torch.Size([bs, n_max_agent, hist+1+future, 6]):  x, y, vx, vy, yaw, score
        agent_time_mask torch.Size([bs, n_max_agent, hist+1+future]) 
        laneline_pts torch.Size([bs, 10, 20, 2])
        laneline_attrs torch.Size([bs, 10, 4])
        laneline_mask torch.Size([bs, 10])
        ego_curr_status torch.Size([bs, 2])
        ego_future_status torch.Size([bs, 50, 5])
        ego_future_mask torch.Size([bs, 50])
        '''
        bs, n_max_agent, _ = data['agent_attrs'].shape
        # agent & map encoding
        x_polygon = self.map_encoder(data)
        x_polygon = self.map_ln(x_polygon)
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_polygon).any():
            print("There is nan in map encoder")

        x_agent, object_mask , x_agent_his ,his_mask= self.agent_encoder(data)     # torch.Size([bs, 1 + n_agent, 128]); torch.Size([bs, 1 + n_agent])
        x_ego, ego_mask, _, _ = self.ego_encoder(data)
        x_agent = self.agent_ln(x_agent)
        x_ego = self.ego_ln(x_ego)
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_agent).any():
            print("There is nan in agent encoder")

        # navi encoding
        x_navitopo = self.map_encoder_navi(data)
        x_navitopo = self.map_ln_navi(x_navitopo)
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_navitopo).any():
            print("There is nan in navi encoder")

        x_route, x_lane, route_mask = self.map_encoder_route(data)
        x_route = self.map_ln_route(x_route)
        # x_lane = self.map_ln_lane(x_lane)  # this is redundant

        condition_path = x_lane
        # occ encoding
        x_polygon_occ = self.map_encoder_occ(data)
        x_polygon_occ = self.map_ln_occ(x_polygon_occ)

        x_navi_ms, navi_ms_pad = self._encode_navi_link_ms_token(data, x_lane.dtype, x_lane.device)
        x_navi_ms = self.navi_link_ms_ln(x_navi_ms)
        x_navi_ded, navi_ded_pad = self._encode_navi_link_ded_token(data, x_lane.dtype, x_lane.device)
        x_navi_ded = self.navi_link_ded_ln(x_navi_ded)

        # obtain transformer encoder input
        pos_embed, agent_his_embed = self.obtain_global_pos_embedding(data, path_flag=True)
        x = torch.cat([x_ego, x_agent, x_polygon, x_polygon_occ, x_navitopo, x_route, x_lane, x_navi_ms, x_navi_ded], dim=1) + pos_embed      # 首 token 为 ego 历史编码
        key_padding_mask = torch.cat([
            ego_mask,
            object_mask,
            data['laneline_mask'],
            data['occ_polygons_mask'],
            data['navitopo_mask'],
            route_mask,
            route_mask,
            navi_ms_pad,
            navi_ded_pad,
            ],
        dim=-1).bool()  # torch.Size([bs, 1 + n_agent + n_poly + 1])
        enc_out = self.transformer(
            src=x, 
            tgt=None, 
            src_padding_mask=key_padding_mask,
            tgt_padding_mask=None,
            only_compute_encoder = True,
        )   # enc_out: torch.Size([bs, 100, dim]); dec_out: torch.Size([n_dec_layer, bs, n_query, dim])

        # mask不同地方的01含义不同，注意
        mask = ~key_padding_mask
        mask_a2al = mask.unsqueeze(1).expand(-1, n_max_agent, -1)
        mask_a2al = mask_a2al & mask[:, :n_max_agent].unsqueeze(-1)

        return enc_out, condition_path, mask_a2al
        # path 的返回值不一样。没有自车的特征了。只有全局fea和条件y
        # return 全局fea，条件（navi），某些mask

    def _encode_navi_link_ms_token(self, data, dtype, device):
        """最近 MAIN_SIDE 单 token；无效时特征置零且 padding=True。"""
        bs = data["agent_attrs"].shape[0]
        max_d = self.navi_link_ms_max_dist_m
        if max_d <= 1e-6:
            max_d = 200.0
        if "navi_link_ms_valid" not in data:
            pad = torch.ones((bs, 1), device=device, dtype=torch.bool)
            return torch.zeros((bs, 1, self.hidden_size), device=device, dtype=dtype), pad
        v = data["navi_link_ms_valid"].to(device=device, dtype=dtype).view(bs, 1)
        d = data["navi_link_ms_dist"].to(device=device, dtype=dtype).view(bs, 1).clamp(0.0, max_d) / max_d
        kind = data["navi_link_ms_kind"].long().to(device=device).view(bs).clamp(0, 2)
        valid = (v.squeeze(-1) > 0.5).float()
        e = self.navi_link_ms_kind_emb(kind)
        h = self.navi_link_ms_mlp(torch.cat([e, d], dim=-1)).unsqueeze(1)
        h = h * v.unsqueeze(-1)
        pad = (v.squeeze(-1) < 0.5).unsqueeze(1).bool()
        return h, pad

    def _encode_navi_link_ded_token(self, data, dtype, device):
        """最近 DEDICATED（专用道左/右）单 token；无效时特征置零且 padding=True。键缺失时与 ms 分支一致回退。"""
        bs = data["agent_attrs"].shape[0]
        max_d = float(getattr(self, "navi_link_ded_max_dist_m", getattr(self, "navi_link_ms_max_dist_m", 200.0)))
        if max_d <= 1e-6:
            max_d = 200.0
        if "navi_link_ded_valid" not in data:
            pad = torch.ones((bs, 1), device=device, dtype=torch.bool)
            return torch.zeros((bs, 1, self.hidden_size), device=device, dtype=dtype), pad
        v = data["navi_link_ded_valid"].to(device=device, dtype=dtype).view(bs, 1)
        d = data["navi_link_ded_dist"].to(device=device, dtype=dtype).view(bs, 1).clamp(0.0, max_d) / max_d
        kind = data["navi_link_ded_kind"].long().to(device=device).view(bs).clamp(0, 2)
        e = self.navi_link_ded_kind_emb(kind)
        h = self.navi_link_ded_mlp(torch.cat([e, d], dim=-1)).unsqueeze(1)
        h = h * v.unsqueeze(-1)
        pad = (v.squeeze(-1) < 0.5).unsqueeze(1).bool()
        return h, pad

    def obtain_global_pos_embedding(self, data, path_flag=False):
        '''
        agent_attrs torch.Size([bs, n_max_agent, 3]): size_y, size_x, cls
        agent_status torch.Size([bs, n_max_agent, hist+1+future, 6]):  x, y, vx, vy, yaw, score
        agent_time_mask torch.Size([bs, n_max_agent, hist+1+future]) 
        # obtain global pos embedding
        '''
        device = data['agent_status'].device
        bs = data['agent_status'].shape[0]

        agent_pos = data["agent_status"][:, :, self.agent_hist_steps, :2]     # torch.Size([bs, n_agent, 2]), 当前帧的agent位置
        agent_heading = data["agent_status"][:, :, self.agent_hist_steps, 4]  # torch.Size([bs, n_agent])
        agent_angle = torch.stack([agent_heading.cos(), agent_heading.sin()], dim=-1)

        n_pt = data["laneline_pts"].shape[2]
        #polygon_center_pos = data["laneline_pts"][:,:,n_pt//2,:]
        polygon_strat_pos = data["laneline_pts"][:, :, 0, :]
        polygon_vector = data["laneline_pts"][:, :, n_pt - 1, :] - data["laneline_pts"][:, :, 0, :]
        polygon_vector_norm = torch.norm(polygon_vector, dim=-1)
        polygon_mask = data["laneline_mask"]
        polygon_vector_norm += polygon_mask + 1e-6 * torch.ones_like(polygon_vector_norm)  # avoid division 0
        polygon_angle = torch.stack(
            [
                polygon_vector[:, :, 0] / polygon_vector_norm,
                polygon_vector[:, :, 1] / polygon_vector_norm,
            ],
            dim=-1,
        )
        if path_flag:
            # navitopo pos encoding
            navitopo_strat_pos = data["navitopo_pts"][:, :, 0, :]
            navitopo_vector = data["navitopo_pts"][:, :, n_pt - 1, :] - data["navitopo_pts"][:, :, 0, :]
            navitopo_vector_norm = torch.norm(navitopo_vector, dim=-1)
            navitopo_mask = data["navitopo_mask"]
            navitopo_vector_norm += navitopo_mask + 1e-6 * torch.ones_like(navitopo_vector_norm)  # avoid division 0
            navitopo_angle = torch.stack(
                [
                    navitopo_vector[:, :, 0] / navitopo_vector_norm,
                    navitopo_vector[:, :, 1] / navitopo_vector_norm,
                ],
                dim=-1,
            )

            # sd_route pos encoding
            route_strat_pos = data["route_pts"][:, :, 0, :]
            route_vector = data["route_pts"][:, :, n_pt - 1, :] - data["route_pts"][:, :, 0, :]
            route_vector_norm = torch.norm(route_vector, dim=-1)
            route_mask = torch.all(data["route_pts"] == 0, dim=(2, 3)).float()
            route_vector_norm += route_mask + 1e-6 * torch.ones_like(route_vector_norm)  # avoid division 0
            route_angle = torch.stack(
                [
                    route_vector[:, :, 0] / route_vector_norm,
                    route_vector[:, :, 1] / route_vector_norm,
                ],
                dim=-1,
            )
        
        # occ pos encoding
        occ_n_pt = data["occ_polygons_pts"].shape[2]
        occ_polygon_center_pos = data["occ_polygons_attrs"][:, :, 0:2]
        occ_polygon_angle_rad = data["occ_polygons_attrs"][:, :, 4]
        occ_polygon_angle = torch.stack(
            [
                torch.cos(occ_polygon_angle_rad),
                torch.sin(occ_polygon_angle_rad),
            ],
            dim=-1,
        )
        # 与序列 x 中 [x_ego, x_agent, ...] 对齐：首槽位为 ego 历史 token 的占位几何
        ego_pos_pe = torch.zeros((bs, 1, 2), device=device)
        ego_heading_pe = torch.zeros((bs, 1, 1), device=device)
        ego_angle_pe = torch.cat([ego_heading_pe.cos(), ego_heading_pe.sin()], dim=-1)
        position_list = [ego_pos_pe, agent_pos, polygon_strat_pos, occ_polygon_center_pos]
        angle_list = [ego_angle_pe, agent_angle, polygon_angle, occ_polygon_angle]
        if path_flag:
            position_list += [navitopo_strat_pos, route_strat_pos, route_strat_pos]
            angle_list += [navitopo_angle, route_angle, route_angle]
            max_d = float(getattr(self, "navi_link_ms_max_dist_m", 200.0))
            if max_d <= 1e-6:
                max_d = 200.0
            if "navi_link_ms_valid" in data:
                vv = data["navi_link_ms_valid"].float().squeeze(-1)
                dd = data["navi_link_ms_dist"].float().squeeze(-1).clamp(0.0, max_d)
                ms_x = dd * (vv > 0.5).float()
            else:
                ms_x = torch.zeros(bs, device=device, dtype=torch.float32)
            ms_pos = torch.stack([ms_x, torch.zeros_like(ms_x)], dim=-1).unsqueeze(1)
            ms_ang = torch.zeros(bs, 1, 2, device=device, dtype=ms_pos.dtype)
            ms_ang[:, 0, 0] = 1.0
            position_list += [ms_pos]
            angle_list += [ms_ang]
            # DEDICATED token：用 y 轴放归一化距离，与 MAIN_SIDE 的 x 轴区分，避免两槽 PE 完全重合
            if "navi_link_ded_valid" in data:
                vv_d = data["navi_link_ded_valid"].float().squeeze(-1)
                dd_d = data["navi_link_ded_dist"].float().squeeze(-1).clamp(0.0, max_d)
                ded_y = dd_d * (vv_d > 0.5).float()
            else:
                ded_y = torch.zeros(bs, device=device, dtype=torch.float32)
            ded_pos = torch.stack([torch.zeros_like(ded_y), ded_y], dim=-1).unsqueeze(1)
            ded_ang = torch.zeros(bs, 1, 2, device=device, dtype=ded_pos.dtype)
            ded_ang[:, 0, 0] = 1.0
            position_list += [ded_pos]
            angle_list += [ded_ang]
        position = torch.cat(position_list, dim=1)   # torch.Size([bs, 1 + n_agent + n_poly, 2])
        angle = torch.cat(angle_list, dim=1)   # torch.Size([bs, 1 + n_agent + n_poly, 2])
        pos = torch.cat([position, angle], dim=-1)   # torch.Size([bs, 1 + n_agent + n_poly, 4])
        pos_embed = self.pos_emb(pos)   # mlp

        agent_pos_his = data["agent_status"][:, :, :self.agent_hist_steps, :2]     # torch.Size([bs, n_agent, 2]), 当前帧的agent位置
        agent_heading_his = data["agent_status"][:, :, :self.agent_hist_steps, 4]  # torch.Size([bs, n_agent])
        agent_angle_his = torch.stack([agent_heading_his.cos(), agent_heading_his.sin()], dim=-1)
        agent_his_mask = data["agent_time_mask"][:, :, :self.agent_hist_steps]
        agent_pos = torch.cat([agent_pos_his, agent_angle_his], dim=-1)
        agent_pos_embed = self.pos_emb(agent_pos)
        if os.getenv("DEPLOY") != 'True' and torch.isnan(pos_embed).any():
            print('Error, nan in pos_embed')
            # raise ValueError("There is nan in pos_embed")
        if os.getenv("DEPLOY") != 'True' and torch.isnan(agent_pos_embed).any():
            print('Error, nan in agent_pos_embed')
        #B,n,128
        return pos_embed, agent_pos_embed
class SceneEncoder_traj(nn.Module):
    def __init__(self, config):
        super(SceneEncoder_traj, self).__init__()
        self.hidden_size              = config['hidden_size']
        self.depth                    = config['depth']
        self.mask_generator           = MultiTFMask()
        self.agent_hist_steps = config['agent_hist_steps']
        dim = config['hidden_size']
        self.transformer = Transformer(
            d_model = dim,
            nhead = config['nhead'],
            num_encoder_layers = config['transformer_layers'],
            num_decoder_layers = None,
            dim_feedforward = dim * 4,
            dropout=0.0,
            return_intermediate_dec = False
        )

        self.agent_encoder = AgentEncoder_traj(
                    **config['agent_encoder'],
                    dim=dim,
                    nhead = config['nhead'],
                    num_encoder_layers = config['transformer_layers'],
                    num_decoder_layers = None,
                    hist_steps=config['agent_hist_steps'],
                )
        self.agent_ln = nn.LayerNorm(dim)

        self.map_encoder = MapEncoder(
            **config['map_encoder'],
            dim=dim,
        )
        self.map_ln = nn.LayerNorm(dim)

        self.map_encoder_occ = MapEncoder_occ(
            **config['map_encoder_occ'],
            dim=dim,
        )
        self.map_ln_occ = nn.LayerNorm(dim)

        self.pos_emb = MLP(4, dim, dim, 2)

    def forward(self, data):
        '''
        agent_attrs torch.Size([bs, n_max_agent, 3]): size_y, size_x, cls
        agent_status torch.Size([bs, n_max_agent, hist+1+future, 6]):  x, y, vx, vy, yaw, score
        agent_time_mask torch.Size([bs, n_max_agent, hist+1+future]) 
        laneline_pts torch.Size([bs, 10, 20, 2])
        laneline_attrs torch.Size([bs, 10, 4])
        laneline_mask torch.Size([bs, 10])
        ego_curr_status torch.Size([bs, 2])
        ego_future_status torch.Size([bs, 50, 5])
        ego_future_mask torch.Size([bs, 50])
        '''
        bs, n_max_agent, _ = data['agent_attrs'].shape
        # agent & map encoding
        x_polygon = self.map_encoder(data)
        x_polygon = self.map_ln(x_polygon)
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_polygon).any():
            print("There is nan in map encoder")

        x_agent, object_mask , x_agent_his ,his_mask= self.agent_encoder(data)     # torch.Size([bs, 1 + n_agent, 128]); torch.Size([bs, 1 + n_agent])
        x_agent = self.agent_ln(x_agent)
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_agent).any():
            print("There is nan in agent encoder")

        # occ encoding
        x_polygon_occ = self.map_encoder_occ(data)
        x_polygon_occ = self.map_ln_occ(x_polygon_occ)

        # obtain transformer encoder input
        pos_embed, agent_his_embed = self.obtain_global_pos_embedding(data)
        x = torch.cat([x_agent, x_polygon, x_polygon_occ], dim=1) + pos_embed      # (bs, 1 + n_agent + n_poly, dim)
        key_padding_mask = torch.cat([
            object_mask, 
            data['laneline_mask'],
            data['occ_polygons_mask']], 
        dim=-1).bool()  # torch.Size([bs, 1 + n_agent + n_poly])


        #只输出enc，navi没在这里，navi拿走当path的condition
        enc_out = self.transformer(
            src=x, 
            tgt=None, 
            src_padding_mask=key_padding_mask,
            tgt_padding_mask=None,
            only_compute_encoder = True,
        )   # enc_out: torch.Size([bs, 101, dim]); dec_out: torch.Size([n_dec_layer, bs, n_query, dim])

        # mask不同地方的01含义不同，注意
        mask = ~key_padding_mask
        mask_a2al = mask.unsqueeze(1).expand(-1, n_max_agent + 1, -1)
        mask_a2al = mask_a2al & mask[:, :n_max_agent + 1].unsqueeze(-1)

        return enc_out, enc_out[:, :1, :], mask_a2al
        #return 全局fea，自车fea，某些mask
        #return context_features, ego_context_features,ego_routing_features, mask_a2al #, agent_features_for_SSL, map_features_for_SSL
    
    def obtain_global_pos_embedding(self, data):
        '''
        agent_attrs torch.Size([bs, n_max_agent, 3]): size_y, size_x, cls
        agent_status torch.Size([bs, n_max_agent, hist+1+future, 6]):  x, y, vx, vy, yaw, score
        agent_time_mask torch.Size([bs, n_max_agent, hist+1+future]) 
        # obtain global pos embedding
        '''
        device = data['agent_status'].device
        bs = data['agent_status'].shape[0]

        ego_pos = torch.zeros((bs, 1, 2), device=device)
        ego_heading = torch.zeros((bs, 1, 1), device=device)
        ego_angle = torch.cat([ego_heading.cos(), ego_heading.sin()], dim=-1)

        agent_pos = data["agent_status"][:, :, self.agent_hist_steps, :2]     # torch.Size([bs, n_agent, 2]), 当前帧的agent位置
        agent_heading = data["agent_status"][:, :, self.agent_hist_steps, 4]  # torch.Size([bs, n_agent])
        agent_angle = torch.stack([agent_heading.cos(), agent_heading.sin()], dim=-1)

        n_pt = data["laneline_pts"].shape[2]
        #polygon_center_pos = data["laneline_pts"][:,:,n_pt//2,:]
        polygon_strat_pos = data["laneline_pts"][:,:,0,:]
        polygon_vector = data["laneline_pts"][:, :, n_pt-1, :] - data["laneline_pts"][:, :, 0, :] # polyline从起点到终点的vector, torch.Size([bs, n_poly, 2])
        polygon_vector_norm = torch.norm(polygon_vector, dim=-1)  # ([2, n_poly, 1])
        polygon_mask = data["laneline_mask"]
        polygon_vector_norm += polygon_mask + 1e-6 * torch.ones_like(polygon_vector_norm)  # avoid division 0
        polygon_angle = torch.stack([
            polygon_vector[:,:,0] / polygon_vector_norm, 
            polygon_vector[:,:,1] / polygon_vector_norm, 
            ], dim=-1)
        
        # occ pos encoding
        occ_n_pt = data["occ_polygons_pts"].shape[2]
        occ_polygon_center_pos = data["occ_polygons_attrs"][:, :, 0:2]
        occ_polygon_angle_rad = data["occ_polygons_attrs"][:, :, 4]
        occ_polygon_angle = torch.stack(
            [
                torch.cos(occ_polygon_angle_rad),
                torch.sin(occ_polygon_angle_rad),
            ],
            dim=-1,
        )

        position = torch.cat([ego_pos, agent_pos, polygon_strat_pos, occ_polygon_center_pos], dim=1)   # torch.Size([bs, 1 + n_agent + n_poly, 2])
        angle = torch.cat([ego_angle, agent_angle, polygon_angle, occ_polygon_angle], dim=1)   # torch.Size([bs, 1 + n_agent + n_poly, 2])
        pos = torch.cat([position, angle], dim=-1)   # torch.Size([bs, 1 + n_agent + n_poly, 4])
        pos_embed = self.pos_emb(pos)   # mlp

        agent_pos_his = data["agent_status"][:, :, :self.agent_hist_steps, :2]     # torch.Size([bs, n_agent, 2]), 当前帧的agent位置
        agent_heading_his = data["agent_status"][:, :, :self.agent_hist_steps, 4]  # torch.Size([bs, n_agent])
        agent_angle_his = torch.stack([agent_heading_his.cos(), agent_heading_his.sin()], dim=-1)
        agent_his_mask = data["agent_time_mask"][:, :, :self.agent_hist_steps]
        agent_pos = torch.cat([agent_pos_his, agent_angle_his], dim=-1)
        agent_pos_embed = self.pos_emb(agent_pos)
        if os.getenv("DEPLOY") != 'True' and torch.isnan(pos_embed).any():
            print('Error, nan in pos_embed')
            # raise ValueError("There is nan in pos_embed")
        if os.getenv("DEPLOY") != 'True' and torch.isnan(agent_pos_embed).any():
            print('Error, nan in agent_pos_embed')
        #B,n,128
        return pos_embed, agent_pos_embed

class DiTDiffusionDecoder(nn.Module):
    def __init__(self, config):
        #zhihui
        #代办：
        #1. 注意normal
        #2. 条件变成navi
        super(DiTDiffusionDecoder, self).__init__()
        self.use_guidance = config['use_guidance']
        self.use_navi_guidance = config['use_navi_guidance']
        self.use_obs_guidance = config['use_obs_guidance']
        self.hidden_size = config['hidden_size']
        self.hidden_size_fix_distance = config['hidden_size_fix_distance']
        self.hidden_size_in = config['hidden_size_in']
        self.depth = config['depth']
        self.trainStage = config['trainStage']
        self.depth_fix_distance = config['depth_fix_distance']
        self.output_dim = 3
        self.num_heads = config['num_heads']
        self.future_steps = config['future_steps']
        self.future_steps_fixed = config['future_steps_fixed']
        self.planning_interval = config['planning_interval']
        self.planning_interval_fixed = config['planning_interval_fixed']
        self.guidance_config = config['guidance_config']

        self.preproj = MLP_2(in_features=self.output_dim, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.fix_distance_path_preproj = MLP_2(in_features=self.output_dim, hidden_features=self.hidden_size_fix_distance, out_features=self.hidden_size_fix_distance, act_layer=nn.GELU, norm_layer=RMSNorm)
        
        self.t_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        self.t_embedder_fix_distance = TimestepEmbedder(hidden_size=self.hidden_size_fix_distance)
        
        self.vector_in = MLP_2(in_features=self.hidden_size_in, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        #self.fix_distance_path_in = MLP_2(in_features=self.hidden_size_in, hidden_features=self.hidden_size_fix_distance, out_features=self.hidden_size_fix_distance, act_layer=nn.GELU, norm_layer=RMSNorm)
        
        self.context_in = MLP_2(in_features=self.hidden_size_in, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.context_in_fix_distance = MLP_2(in_features=self.hidden_size_in, hidden_features=self.hidden_size_fix_distance, out_features=self.hidden_size_fix_distance, act_layer=nn.GELU, norm_layer=RMSNorm)

        self.blocks = nn.ModuleList([DiTBlock(dim=self.hidden_size, heads=self.num_heads, mlp_ratio=2.0) for i in range(self.depth)])
        self.fix_distance_path_blocks = nn.ModuleList([DiTBlock(dim=self.hidden_size_fix_distance, heads=self.num_heads, mlp_ratio=2.0) for i in range(self.depth_fix_distance)])
        
        self.final_layer = FinalLayer(hidden_size=self.hidden_size, output_size=self.output_dim)
        self.fix_distance_path_final_layer = FinalLayer(hidden_size=self.hidden_size_fix_distance, output_size=self.output_dim)
        
        self.timesteps = 1000
        # self.register_buffer("input_tokens_mask", torch.tril(torch.ones(80, 80)).view(1, 80, 80))
        self.ego_time_pe = PositionalEncoding(d_model=self.hidden_size, dropout=0.0, max_len=self.future_steps//int(self.planning_interval/0.1))
        self.ego_fix_distance_path_pe = nn.Embedding(self.future_steps_fixed//int(self.planning_interval_fixed/1.0), self.hidden_size_fix_distance)
        
        # 条件编码
        self.ego_routing_in = MLP_2(in_features = (self.future_steps_fixed // int(self.planning_interval_fixed/1.0)) * 2, hidden_features=self.hidden_size, out_features=self.hidden_size, act_layer=nn.GELU, norm_layer=RMSNorm)
        self.ego_routing_in_fixdistance = MLP_2(in_features=self.hidden_size_in, hidden_features=self.hidden_size_fix_distance, out_features=self.hidden_size_fix_distance, act_layer=nn.GELU, norm_layer=RMSNorm)

        self.uniform_t = config.get('uniform_t', False)
        print(f"uniform_t: {self.uniform_t}")

        self.std_scaler_traj = config.get('std_scaler_traj', 1)
        print(f"std_scaler_traj: {self.std_scaler_traj}")

        self.std_scaler_path = config.get('std_scaler_path', 1)
        print(f"std_scaler_path: {self.std_scaler_path}")

        # aux loss 相关参数（EMA + scale）
        self.use_aux_loss = config['use_aux_loss']
        self.aux_loss_config = config['aux_loss_config']
        # self.use_lane_condition = config.get('use_lane_condition', False)
        self.use_lane_condition = False
        print(f"use_lane_condition: {self.use_lane_condition}")
        if self.use_aux_loss:
            self.norm_momentum = self.aux_loss_config['norm_momentum']
            self.loss_path_scale = self.aux_loss_config['loss_path_scale']
            self.loss_navi_scale = self.aux_loss_config['loss_navi_scale']
            ema_in_state_dict = self.aux_loss_config['ema_in_state_dict']
            self.register_buffer("loss_path_ema", torch.tensor(1.0), persistent=ema_in_state_dict)
            self.register_buffer("loss_navi_x_ema", torch.tensor(1.0), persistent=ema_in_state_dict)
            self.register_buffer("loss_navi_y_ema", torch.tensor(1.0), persistent=ema_in_state_dict)

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
    # @staticmethod
    # def normalize_state(state):
    #     state[...,0] = (state[...,0] - 1.0528826) / 0.37231752
    #     state[...,1] = (state[...,1] + 0.009744605) / 0.09104148
    #     state[...,2] = state[...,2] / (math.pi/2)
    #     return state
    
    # @staticmethod
    # def denormalize_state(state):
    #     state[...,0] = state[...,0] * 0.37231752 + 1.0528826
    #     state[...,1] = state[...,1] * 0.09104148 - 0.009744605
    #     state[...,2] = state[...,2] * (math.pi/2)
    #     return state

    def normalize_state(self, state, model_input, std_scaler_traj=1):
        # import pdb;pdb.set_trace()
        valid_delta_x_mean = model_input['mean_std'][:, 0].unsqueeze(1)
        valid_delta_x_std = (model_input['mean_std'][:, 1].unsqueeze(1) * std_scaler_traj).clamp(min=1e-6)
        valid_delta_y_mean = model_input['mean_std'][:, 2].unsqueeze(1)
        valid_delta_y_std = (model_input['mean_std'][:, 3].unsqueeze(1) * std_scaler_traj).clamp(min=1e-6)
        valid_delta_yaw_mean = model_input['mean_std'][:, 4].unsqueeze(1)
        valid_delta_yaw_std = (model_input['mean_std'][:, 5].unsqueeze(1) * std_scaler_traj).clamp(min=1e-6)
        state[...,0] = (state[...,0] - valid_delta_x_mean) / valid_delta_x_std
        state[...,1] = (state[...,1] - valid_delta_y_mean) / valid_delta_y_std
        state[...,2] = (state[...,2] - valid_delta_yaw_mean) / (valid_delta_yaw_std * (math.pi/2))
        return state

    def denormalize_state(self, state, model_input, std_scaler_traj=1):
        # import pdb;pdb.set_trace()
        valid_delta_x_mean = model_input['mean_std'][:, 0].unsqueeze(1)
        valid_delta_x_std = (model_input['mean_std'][:, 1].unsqueeze(1) * std_scaler_traj).clamp(min=1e-6)
        valid_delta_y_mean = model_input['mean_std'][:, 2].unsqueeze(1)
        valid_delta_y_std = (model_input['mean_std'][:, 3].unsqueeze(1) * std_scaler_traj).clamp(min=1e-6)
        valid_delta_yaw_mean = model_input['mean_std'][:, 4].unsqueeze(1)
        valid_delta_yaw_std = (model_input['mean_std'][:, 5].unsqueeze(1) * std_scaler_traj).clamp(min=1e-6)
        x_denorm = state[...,0] * valid_delta_x_std + valid_delta_x_mean
        y_denorm = state[...,1] * valid_delta_y_std +valid_delta_y_mean
        yaw_denorm = state[...,2] * (math.pi/2) * valid_delta_yaw_std + valid_delta_yaw_mean
        return torch.cat([x_denorm.unsqueeze(-1), y_denorm.unsqueeze(-1), yaw_denorm.unsqueeze(-1)], dim=-1)

    def normalize_fix_distance_path_state(self, state, model_input):
        valid_delta_x_mean = model_input['mean_std_fixed'][:, 0].unsqueeze(1)
        valid_delta_x_std = model_input['mean_std_fixed'][:, 1].unsqueeze(1).clamp(min=1e-6)
        valid_delta_y_mean = model_input['mean_std_fixed'][:, 2].unsqueeze(1)
        valid_delta_y_std = model_input['mean_std_fixed'][:, 3].unsqueeze(1).clamp(min=1e-6)
        valid_delta_yaw_mean = model_input['mean_std_fixed'][:, 4].unsqueeze(1)
        valid_delta_yaw_std = model_input['mean_std_fixed'][:, 5].unsqueeze(1).clamp(min=1e-6)
        state[...,0] = (state[...,0] - valid_delta_x_mean) / valid_delta_x_std
        state[...,1] = (state[...,1] - valid_delta_y_mean) / valid_delta_y_std
        state[...,2] = (state[...,2] - valid_delta_yaw_mean) / (valid_delta_yaw_std * (math.pi/2))
        return state

    def denormalize_fix_distance_path_state(self, state, model_input):
        valid_delta_x_mean = model_input['mean_std_fixed'][:, 0].unsqueeze(1)
        valid_delta_x_std = model_input['mean_std_fixed'][:, 1].unsqueeze(1).clamp(min=1e-6)
        valid_delta_y_mean = model_input['mean_std_fixed'][:, 2].unsqueeze(1)
        valid_delta_y_std = model_input['mean_std_fixed'][:, 3].unsqueeze(1).clamp(min=1e-6)
        valid_delta_yaw_mean = model_input['mean_std_fixed'][:, 4].unsqueeze(1)
        valid_delta_yaw_std = model_input['mean_std_fixed'][:, 5].unsqueeze(1).clamp(min=1e-6)
        x_denorm = state[...,0] * valid_delta_x_std + valid_delta_x_mean
        y_denorm = state[...,1] * valid_delta_y_std +valid_delta_y_mean
        yaw_denorm = state[...,2] * (math.pi/2) * valid_delta_yaw_std + valid_delta_yaw_mean
        return torch.cat([x_denorm.unsqueeze(-1), y_denorm.unsqueeze(-1), yaw_denorm.unsqueeze(-1)], dim=-1)

    def prepare_input(self, model_input):
        #整理ego相关输入。
        #xy，heading，delta输入
        ego_gt, egp_gt_valid = model_input['ego_future_status'], model_input['ego_future_mask'] # [B, 1, T_f, 3], [B, 1, T_f]
        B = ego_gt.shape[0]
        ego_gt_pos = ego_gt[:, :, :2]
        ego_gt_heading = ego_gt[:, :, 4:5]
        ego_gt = torch.cat([ego_gt_pos, ego_gt_heading], dim=-1).unsqueeze(1)

        ego_current_position = torch.tensor([0.0, 0.0], device=ego_gt.device, dtype = torch.float32)
        ego_current_position = ego_current_position.repeat(B, 1, 1, 1)  # (B,1,1,2)
        ego_current_heading = torch.zeros((B,1,1,1), device=ego_gt.device, requires_grad=False) #valid_agents_past_features[:, :, [-1], [-2]].unsqueeze(-1) # [B, A, 1, 1]
        ego_current_gt = torch.cat([ego_current_position, ego_current_heading], dim=-1) # [B, 1, 1, 3]

        ego_tokens = torch.cat([ego_current_gt, ego_gt], dim=-2) # [B, 1, T_f+1, 3]
        delta_ego_tokens = ego_tokens[:, :, 1:, :] - ego_tokens[:, :, :-1, :] # [B, 1, T_f, 3]

        # norm注意
        # ego_tokens = self.normalize_state(delta_ego_tokens)
        ego_tokens = self.normalize_state(delta_ego_tokens, model_input, self.std_scaler_traj)


        ego_tokens_valid = egp_gt_valid.unsqueeze(1).unsqueeze(-1).to(dtype=torch.float32, device=ego_gt.device)  # [B,1,T_f,1]
        input_noise = torch.randn_like(ego_tokens) # [B, 1, T_f, 3]
        input_tokens_mask = torch.matmul(ego_tokens_valid, ego_tokens_valid.transpose(-2,-1)).squeeze(1) # [B, T_f, T_f]
        
        return ego_tokens, ego_tokens_valid, input_noise, input_tokens_mask

    def prepare_fix_distance_path_input(self, model_input):
        #整理ego相关输入
        #xy，heading，delta
        ego_fixDis_gt = model_input.get('ego_future_status_fixed_train_gt', model_input['ego_future_status_fixed'])
        egp_fixDis_gt_valid = model_input.get('ego_future_mask_fixed_train_gt', model_input['ego_future_mask_fixed']) # [B, 1, T_f, 3], [B, 1, T_f]
        B = ego_fixDis_gt.shape[0]
        ego_fixDis_gt_pos = ego_fixDis_gt[:, :, :2]
        ego_fixDis_gt_heading = ego_fixDis_gt[:, :, 4:5]
        ego_fixDis_gt = torch.cat([ego_fixDis_gt_pos, ego_fixDis_gt_heading], dim=-1).unsqueeze(1)

        ego_fixDis_current_position = torch.tensor([0.0, 0.0], device=ego_fixDis_gt.device, dtype = torch.float32)
        ego_fixDis_current_position = ego_fixDis_current_position.repeat(B, 1, 1, 1)  # (B,1,1,2)
        ego_fixDis_current_heading = torch.zeros((B,1,1,1), device=ego_fixDis_gt.device, requires_grad=False) #valid_agents_past_features[:, :, [-1], [-2]].unsqueeze(-1) # [B, A, 1, 1]
        ego_fixDis_current_gt = torch.cat([ego_fixDis_current_position, ego_fixDis_current_heading], dim=-1) # [B, A, 1, 3]

        ego_fixDis_tokens = torch.cat([ego_fixDis_current_gt, ego_fixDis_gt], dim=-2) # [B, A, T_f+1, 3]
        delta_ego_fixDis_tokens = ego_fixDis_tokens[:, :, 1:, :] - ego_fixDis_tokens[:, :, :-1, :] # [B, A, T_f, 3]

        #norm注意
        # ego_fixDis_tokens = self.normalize_fix_distance_path_state(delta_ego_fixDis_tokens)
        ego_fixDis_tokens = self.normalize_fix_distance_path_state(delta_ego_fixDis_tokens, model_input)


        ego_fixDis_tokens_valid = egp_fixDis_gt_valid.unsqueeze(1).unsqueeze(-1).to(dtype=torch.float32, device=ego_fixDis_gt.device)  # [B,1,T_f,1]
        input_fixDis_noise = torch.randn_like(ego_fixDis_tokens) # [B, 1, T_f, 3]
        input_fixDis_tokens_mask = torch.matmul(ego_fixDis_tokens_valid, ego_fixDis_tokens_valid.transpose(-2,-1)).squeeze(1) # [B, T_f, T_f]

        return ego_fixDis_tokens, ego_fixDis_tokens_valid, input_fixDis_noise, input_fixDis_tokens_mask

    def model_forward(self, z_t, context, t, vec, condition, input_tokens_mask, mask_ego2al):
        B, _, _, _ = z_t.shape
        z_t = self.preproj(z_t).squeeze(1) # [B, 80, D]
        z_t = self.ego_time_pe(z_t) # [B, 80, D]
        z_t = z_t + self.vector_in(vec) # [B, 80, D]

        y = self.t_embedder(t.reshape((B, )) * self.timesteps) # [B, D]

        condition = condition.reshape(B, 1, -1)
        y = y + self.ego_routing_in(condition[:,0,:])
        
        context = self.context_in(context) # [B, A, D]
        for block in self.blocks:
            z_t = block(z_t, context, y, input_tokens_mask, mask_ego2al)
        pred = self.final_layer(z_t, y)
        pred = pred.reshape(B, 1, self.future_steps//int(self.planning_interval/0.1), -1) # [B, 1, T_f+1, 3]
        return pred

    def fix_distance_path_model_forward(self, z_t, context, t, ego_routing_features, input_tokens_mask, mask_ego2al):
        B, _, _, _ = z_t.shape
        z_t = self.fix_distance_path_preproj(z_t).squeeze(1) # [B, 80, D]
        z_t = self.ego_fix_distance_path_pe.weight.unsqueeze(0) + z_t # [B, 80, D]

        y = self.t_embedder_fix_distance(t.reshape((B, )) * self.timesteps) # [B, D]
        if self.use_lane_condition:
            y = y + self.ego_routing_in_fixdistance(ego_routing_features[:,0,:]) # [B, D]
        context = self.context_in_fix_distance(context) # [B, A, D]
        for block in self.fix_distance_path_blocks:
            z_t = block(z_t, context, y, input_tokens_mask, mask_ego2al)
        pred = self.fix_distance_path_final_layer(z_t, y)
        pred = pred.reshape(B, 1, self.future_steps_fixed//int(self.planning_interval_fixed/1.0), -1) # [B, 1, T_f+1, 3]
        return pred

    def forward(self, model_input, enc_path_out, enc_traj_out):
        if self.trainStage == 'stage_1':
            env_context_features, ego_routing_features, mask_a2al_agent = enc_path_out
            context, ego_context_features, mask_a2al_all = enc_traj_out
        if self.trainStage == 'stage_2':
            env_context_features, ego_routing_features, mask_a2al_agent = enc_path_out
            context, ego_context_features, mask_a2al_all = enc_traj_out
        ego_output_tokens, ego_output_tokens_valid, input_noise, input_tokens_mask = self.prepare_input(model_input)
        ego_fix_distance_path_output_tokens, ego_fix_distance_path_output_tokens_valid, ego_fix_distance_path_input_noise, ego_fix_distance_path_input_tokens_mask = self.prepare_fix_distance_path_input(model_input)
        B, _, T_f, _ = ego_output_tokens.shape
        dict_loss = {}
        # rectified flow
        if self.uniform_t:
            t = torch.rand((B,)).to(input_noise.device)
        else:
            nt = torch.randn((B,)).to(input_noise.device)
            t = torch.sigmoid(nt)
        # t = torch.where(torch.rand_like(t) <= 0.9, t, torch.rand_like(t))

        t = t.view([B, *([1] * len(input_noise.shape[1:]))]) # [B, 1, 1, 1]
        z_t_fix_distance_path = t * ego_fix_distance_path_output_tokens + (1-t) * ego_fix_distance_path_input_noise
        target_fix_distance_path = ego_fix_distance_path_output_tokens - ego_fix_distance_path_input_noise
        z_t = t * ego_output_tokens + (1-t) * input_noise # [B, 1, T_f+1, 3]
        target = ego_output_tokens - input_noise
            
        if 'stage_1' == self.trainStage:
            #stage 1: only path
            pred_fix_distance_path = self.fix_distance_path_model_forward(z_t_fix_distance_path, env_context_features, t, ego_routing_features, ego_fix_distance_path_input_tokens_mask, mask_a2al_agent[:, :1, :].repeat(1, self.future_steps_fixed//int(self.planning_interval_fixed/1.0), 1))
            loss_fix_distance_path = (pred_fix_distance_path - target_fix_distance_path) ** 2
            mask_fix_distance_path = ego_fix_distance_path_output_tokens_valid.repeat(1, 1, 1, 3) # [B, 1, T_f+1, 3]
            mask_fix_distance_path = mask_fix_distance_path.bool() & model_input['trainFlag'].view(-1, 1, 1, 1)
            loss_fix_distance_path = loss_fix_distance_path * mask_fix_distance_path
            dict_loss['loss_path'] = loss_fix_distance_path.sum() / (mask_fix_distance_path.sum().clamp(min=1.0))
            dict_loss['loss_traj'] = 0

            if self.use_aux_loss:
                T = 5
                pred_trajs_fix_distance_path = ego_fix_distance_path_input_noise.clone()
                for i in range(T):
                    sample_t = torch.ones((B, ), device = ego_fix_distance_path_output_tokens.device) * i / T
                    drift_pred_fix_distance_path = self.fix_distance_path_model_forward(pred_trajs_fix_distance_path, env_context_features, sample_t, ego_routing_features, None, mask_a2al_agent[:, :1, :].repeat(1, self.future_steps_fixed//int(self.planning_interval_fixed/1.0), 1))
                    pred_trajs_fix_distance_path = pred_trajs_fix_distance_path + drift_pred_fix_distance_path * 1. / T

                fix_distance_path_denorm = torch.cumsum(self.denormalize_fix_distance_path_state(pred_trajs_fix_distance_path, model_input), dim=-2)
                loss_path = dict_loss['loss_path']
                # 多候选 navi 目标：navitopo_rs (B, 1+N, T, 2), navitopo_rs_mask (B, 1+N, T)
                navitopo_rs = model_input.get('navitopo_rs', None)
                navitopo_rs_mask = model_input.get('navitopo_rs_mask', None)

                if navitopo_rs is not None and navitopo_rs.dim() == 4:
                    navi_target = navitopo_rs[..., :2]           # (B, K, T, 2)
                    navi_mask = navitopo_rs_mask.bool() & model_input['trainFlag'].view(-1, 1, 1)
                else:
                    # 回退到 GT path 作为单候选 target
                    target_path = model_input.get('ego_future_status_fixed_train_gt', model_input['ego_future_status_fixed'])
                    target_mask = model_input.get('ego_future_mask_fixed_train_gt', model_input['ego_future_mask_fixed'])
                    navi_target = target_path[..., :2].unsqueeze(1)  # (B, 1, T, 2)
                    navi_mask = (target_mask.bool() & model_input['trainFlag'].view(-1, 1)).unsqueeze(1)

                laneline_pw = model_input.get('laneline_point_weight', None)  # (B, T), 在 dataset 端预计算
                loss_path_navi_x, loss_path_navi_y = soft_L2_train_loss(
                    fix_distance_path_denorm,
                    navi_target,
                    navi_mask,
                    laneline_point_weight=laneline_pw,
                )

                if self.training:
                    with torch.no_grad():
                        m = self.norm_momentum
                        self.loss_path_ema.mul_(m).add_(loss_path.detach() * (1.0 - m))
                        self.loss_navi_x_ema.mul_(m).add_(loss_path_navi_x.detach() * (1.0 - m))
                        self.loss_navi_y_ema.mul_(m).add_(loss_path_navi_y.detach() * (1.0 - m))

                eps = 1e-6
                loss_path_norm = loss_path / (self.loss_path_ema + eps)
                loss_path_navi_x_norm = loss_path_navi_x / (self.loss_navi_x_ema + eps)
                loss_path_navi_y_norm = loss_path_navi_y / (self.loss_navi_y_ema + eps)
                dict_loss.update({
                    'loss_path': loss_path, 'loss_path_navi_x': loss_path_navi_x, 'loss_path_navi_y': loss_path_navi_y,
                    'loss_path_norm': loss_path_norm, 'loss_path_navi_x_norm': loss_path_navi_x_norm, 'loss_path_navi_y_norm': loss_path_navi_y_norm,
                    'ema_path': self.loss_path_ema, 'ema_path_navi_x': self.loss_navi_x_ema, 'ema_path_navi_y': self.loss_navi_y_ema,
                })

        if 'stage_2' == self.trainStage:
            #stage 2: 1. path, 2. traj
            # 因为forward和sample的预测量不一样，所以还得走一次path的forward，用作loss标准。然后再走一次path的sample，作为traj的condition
            
            #1.1. 走path的forward
            pred_fix_distance_path = self.fix_distance_path_model_forward(z_t_fix_distance_path, env_context_features, t, ego_routing_features, ego_fix_distance_path_input_tokens_mask, mask_a2al_agent[:, :1, :].repeat(1, self.future_steps_fixed//int(self.planning_interval_fixed/1.0), 1))
            loss_fix_distance_path = (pred_fix_distance_path - target_fix_distance_path) ** 2
            mask_fix_distance_path = ego_fix_distance_path_output_tokens_valid.repeat(1, 1, 1, 3) # [B, 1, T_f+1, 3]
            mask_fix_distance_path = mask_fix_distance_path.bool() & model_input['trainFlag'].view(-1, 1, 1, 1)
            loss_fix_distance_path = loss_fix_distance_path * mask_fix_distance_path
            dict_loss['loss_path'] = loss_fix_distance_path.sum() / (mask_fix_distance_path.sum().clamp(min=1.0))

            # 1.2. 走path的sample
            if self.use_guidance and (self.use_navi_guidance or self.use_obs_guidance):
                _, pred_fix_distance_paths_sample = self.path_guidance_sample(input_noise, ego_fix_distance_path_input_noise, 5, enc_path_out, enc_traj_out, model_input, use_navi_guidance=self.use_navi_guidance, use_obs_guidance=self.use_obs_guidance) # [B, 1, 81, 3]
                pred_fix_distance_path_sample = pred_fix_distance_paths_sample[-1]
            else:
                _, pred_fix_distance_paths_sample = self.sample(input_noise, ego_fix_distance_path_input_noise, 5, enc_path_out, enc_traj_out, 'train') # [B, 1, 81, 3]
                pred_fix_distance_path_sample = pred_fix_distance_paths_sample[-1]
            pred_fix_distance_path_sample = pred_fix_distance_path_sample[..., :2] # only use x, y
            pred = self.model_forward(z_t, context, t, ego_context_features, pred_fix_distance_path_sample, input_tokens_mask, mask_a2al_all[:, :1, :].repeat(1, self.future_steps//int(self.planning_interval/0.1), 1))
            loss = (pred - target) ** 2
            mask = ego_output_tokens_valid.repeat(1, 1, 1, 3) # [B, 1, T_f+1, 3]
            mask = mask.bool() & model_input['trainFlag'].view(-1, 1, 1, 1)
            loss = loss * mask
            dict_loss['loss_traj'] = loss.sum() / (mask.sum().clamp(min=1.0))

        if 'stage_1' == self.trainStage:
            if self.use_aux_loss:
                # loss = self.loss_path_scale * loss_path_norm + \
                #        self.loss_navi_scale * loss_path_navi_x_norm + \
                #        self.loss_navi_scale * loss_path_navi_y_norm
                loss = self.loss_path_scale * loss_path_norm + \
                       self.loss_navi_scale * loss_path_navi_y_norm
            else:
                loss = loss_fix_distance_path.sum() / (mask_fix_distance_path.sum().clamp(min=1.0))
        if 'stage_2' == self.trainStage:
            # loss = (loss.sum() + loss_fix_distance_path.sum()) / (mask.sum() + mask_fix_distance_path.sum()).clamp(min=1.0)
            loss = loss.sum() / (mask.sum()).clamp(min=1.0)
        dict_loss['loss'] = loss
        return dict_loss
    
    @torch.no_grad()
    def sample(self, input_noise, input_noise_fix_distance, T, enc_path, enc_traj, mode):
        env_context_features, ego_routing_features, mask_a2al_agent = enc_path
        context, ego_context_features, mask_a2al_all = enc_traj
        B = ego_context_features.shape[0]
        samples = [input_noise.clone()]
        fix_distance_path_samples = [input_noise_fix_distance.clone()]
        for i in range(T):
            t = torch.ones((B, ), device=input_noise.device) * i / T
            drift_pred_fix_distance_path = self.fix_distance_path_model_forward(fix_distance_path_samples[-1], env_context_features, t, ego_routing_features, None, mask_a2al_agent[:, :1, :].repeat(1, self.future_steps_fixed//int(self.planning_interval_fixed/1.0), 1))
            pred_trajs_fix_distance_path = fix_distance_path_samples[-1] + drift_pred_fix_distance_path * 1. / T
            fix_distance_path_samples.append(pred_trajs_fix_distance_path)
        if mode != 'train':
            for i in range(T):
                t = torch.ones((B, ), device=input_noise.device) * i / T
                drift_pred = self.model_forward(samples[-1], context, t, ego_context_features, fix_distance_path_samples[-1][...,:2], None, mask_a2al_all[:, :1, :].repeat(1, self.future_steps//int(self.planning_interval/0.1), 1))
                pred_trajs = samples[-1] + drift_pred * 1. / T
                samples.append(pred_trajs)

        return samples, fix_distance_path_samples

    def path_guidance_sample(self, input_noise, input_noise_fix_distance, T, enc_path, enc_traj, model_input=None, use_navi_guidance=True, use_obs_guidance=True):
        env_context_features, ego_routing_features, mask_a2al_agent = enc_path
        context, ego_context_features, mask_a2al_all = enc_traj
        
        guidance_info = self.guidance_config
        B = ego_context_features.shape[0]
        samples = [input_noise.clone()]
        fix_distance_path_samples = [input_noise_fix_distance.clone()]
        for i in range(T):
            t = torch.ones((B, ), device=input_noise.device) * i / T
            drift_pred_fix_distance_path = self.fix_distance_path_model_forward(fix_distance_path_samples[-1], env_context_features, t, ego_routing_features, None, mask_a2al_agent[:, :1, :].repeat(1, self.future_steps_fixed//int(self.planning_interval_fixed/1.0), 1))
            pred_trajs_fix_distance_path = fix_distance_path_samples[-1] + drift_pred_fix_distance_path * 1. / T
            with torch.enable_grad():
                # Global guidance parameters
                gmax, occ_guidance_params, guidance_start_ratio = guidance_info['gmax'], guidance_info['costmap'], guidance_info['guidance_start_ratio']
                alpha = max(0.0, min(1.0, ((i + 1) / T - guidance_start_ratio) / (1.0 - guidance_start_ratio)))
                # alpha = float(i + 1 < T) * max(0.0, min(1.0, (i / T - guidance_start_ratio) / (1.0 - guidance_start_ratio)))
                if use_navi_guidance:
                    pred_trajs_fix_distance_path = pred_trajs_fix_distance_path.clone().detach().requires_grad_(True)
                    fix_distance_path_denorm = torch.cumsum(self.denormalize_fix_distance_path_state(pred_trajs_fix_distance_path, model_input), dim=-2)
                    # loss = costmap_guidance_func(fix_distance_path_denorm, model_input['path_rewardmap'], occ_guidance_params, i)
                    loss_navi = soft_L2_guidance_func(fix_distance_path_denorm, model_input['navitopo_pts'], model_input['navitopo_mask'])
                    guidance_grad_navi = torch.autograd.grad(outputs=loss_navi, inputs=fix_distance_path_denorm, allow_unused=True)[0]
                    gn = guidance_grad_navi.norm(p=2, dim=(-1,-2,-3), keepdim=True) + 1e-6
                    norm_scale = torch.clamp(gmax / gn, min=0.0, max=1.0)
                    scaled_guidance_grad_navi = norm_scale * guidance_grad_navi * guidance_info['hard_scale_navi']
                    guidanced_fix_distance_path_denorm = fix_distance_path_denorm - alpha * scaled_guidance_grad_navi
                    # tqdm.write(f"i: {i}, Navimap_loss: {loss_navi.item():<8.2f}, alpha: {alpha:<6.2f}, norm_scale: {norm_scale.item():<8.2f}, gn: {gn.item():<6.2f}, grad: {guidance_grad_navi.mean():<6.2f}")

                    ego_current_position = torch.tensor([0.0, 0.0], device=guidanced_fix_distance_path_denorm.device, dtype = torch.float32)
                    ego_current_position = ego_current_position.repeat(B, 1, 1, 1)  # (B,1,1,2)
                    ego_current_heading = torch.zeros((B,1,1,1), device=guidanced_fix_distance_path_denorm.device, requires_grad=False) #valid_agents_past_features[:, :, [-1], [-2]].unsqueeze(-1) # [B, A, 1, 1]
                    ego_current_gt = torch.cat([ego_current_position, ego_current_heading], dim=-1) # [B, 1, 1, 3]
                    ego_tokens = torch.cat([ego_current_gt, guidanced_fix_distance_path_denorm], dim=-2) # [B, 1, T_f+1, 3]
                    delta_ego_tokens = ego_tokens[:, :, 1:, :] - ego_tokens[:, :, :-1, :] # [B, 1, T_f, 3]
                    pred_trajs_fix_distance_path = self.normalize_fix_distance_path_state(delta_ego_tokens, model_input)

                if use_obs_guidance:
                    pred_trajs_fix_distance_path = pred_trajs_fix_distance_path.clone().detach().requires_grad_(True)
                    fix_distance_path_denorm = torch.cumsum(self.denormalize_fix_distance_path_state(pred_trajs_fix_distance_path, model_input), dim=-2)
                    loss_costmap = guidance_info['scale_costmap'] * costmap_guidance_func(fix_distance_path_denorm, model_input['path_costmap'], occ_guidance_params, i)
                    guidance_grad_costmap = torch.autograd.grad(outputs=loss_costmap, inputs=pred_trajs_fix_distance_path, allow_unused=True)[0]
                    gn = guidance_grad_costmap.norm(p=2, dim=(-1,-2,-3), keepdim=True) + 1e-6
                    norm_scale = torch.clamp(gmax / gn, min=0.0, max=1.0)
                    scaled_guidance_grad_costmap = norm_scale * guidance_grad_costmap * guidance_info['hard_scale_costmap']
                    pred_trajs_fix_distance_path = pred_trajs_fix_distance_path - alpha * scaled_guidance_grad_costmap
                # tqdm.write(f"i: {i}, Costmap_loss: {loss_costmap.item():<8.2f}, alpha: {alpha:<6.2f}, norm_scale: {norm_scale.item():<8.2f}, gn: {gn.item():<6.2f}, grad: {guidance_grad_costmap.mean():<6.2f}")
            fix_distance_path_samples.append(pred_trajs_fix_distance_path)
        for i in range(T):
            t = torch.ones((B, ), device=input_noise.device) * i / T
            drift_pred = self.model_forward(samples[-1], context, t, ego_context_features, fix_distance_path_samples[-1], None, mask_a2al_all[:, :1, :].repeat(1, self.future_steps//int(self.planning_interval/0.1), 1))
            pred_trajs = samples[-1] + drift_pred * 1. / T
            samples.append(pred_trajs)
    
        debug = False
        if debug:
            save_path="/mnt/afs/liuzhaoyang1/diffusion_codes/dif_17_guidance/tmp/grad"
            os.makedirs(save_path, exist_ok=True)
            free_samples, free_fix_distance_path_samples = self.sample(input_noise, input_noise_fix_distance, T, context, ego_context_features, ego_routing_features, mask_a2al)
            free_path_denorm = torch.cumsum(self.denormalize_fix_distance_path_state(free_fix_distance_path_samples[-1], model_input), dim=-2)
            guidanced_fix_distance_path_denorm = torch.cumsum(self.denormalize_fix_distance_path_state(fix_distance_path_samples[-1], model_input), dim=-2)
            diff = torch.norm(free_path_denorm[..., :2] - guidanced_fix_distance_path_denorm[..., :2], dim=-1).mean().item()
            # tqdm.write(f"Navimap_loss: {loss_navi.item():<6.2f}, Costmap_loss: {loss_costmap.item():<6.2f}, navi_grad: {guidance_grad_navi.mean():<6.2f}, cost_grad: {guidance_grad_costmap.mean():<6.2f}")

            # save_path = os.path.join(save_path, f"count{int(model_input['infer_conut'].item())}_diff{diff:.2f}_loss{loss_navi.item():.2f}_vs_{loss_costmap.item():.2f}.png")
            # vis_guidance_grad(free_path_denorm, guidanced_fix_distance_path_denorm, [-g for g in [guidance_grad_navi, guidance_grad_costmap]], model_input, save_path, safe_range=1.4)
            import pdb; pdb.set_trace()
        return samples, fix_distance_path_samples

# *****************************
# ******** Model utils ********
# *****************************
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

        # Attention scores + softmax in float32 (stable under outer AMP fp16/bf16). Avoid downcasting scores
        # to q.dtype before softmax — that was redundant work and could hurt stability.
        attn_dt = torch.float32
        q_f = q.to(attn_dt)
        k_f = k.to(attn_dt)
        v_f = v.to(attn_dt)
        attn_scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * self.scale_factor
        attn_scores = torch.clamp(attn_scores, min=-1e4, max=1e4)
        if mask is not None:
            mask = mask.unsqueeze(dim=-3)  # [B, T, 1, q_numel, k_numel] or [B, 1, q_numel, k_numel]
            attn_scores = attn_scores.masked_fill(mask == 0, value=-6e4)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v_f)
        output = output.to(q.dtype)

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

    @torch.amp.autocast(device_type="cuda", dtype=torch.float32)
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

    @torch.amp.autocast(device_type="cuda", dtype=torch.float32)
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
    
    @torch.amp.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# *****************************
# ******** MMDit utils ********
# *****************************
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

# *****************************
# ********* Dit utils *********
# *****************************
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

    @torch.amp.autocast(device_type="cuda", dtype=torch.float32)
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
        # Force fp32 inside DiT blocks on CUDA so AdaLN (1+scale), gates, and MLP match FinalLayer — avoids
        # fp16 blow-ups around mid-depth blocks when conditioning enlarges y (e.g. lane residual).
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float32)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with amp_ctx:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)

            modulated_x = modulate(self.modulate_norm1(x), shift_msa, scale_msa)
            x = x + gate_msa.unsqueeze(1) * self.self_attn(modulated_x, modulated_x, modulated_x, self_attn_mask)

            modulated_x = modulate(self.modulate_norm2(x), shift_mlp, scale_mlp)
            x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

            x = x + self.cross_attn(self.pre_cross_attn_norm(x), self.pre_memory_norm(cross_c), cross_c, cross_attn_mask)
            x = x + self.mlp2(self.post_cross_attn_norm(x))
        return x

def expand_and_repeat_trajectory(trajectory_3d, repeat_dim=1, repeat_count=5):
    """将轨迹扩展到5D并在指定维度重复"""
    batch_size, num_agents, seq_len, _ = trajectory_3d.shape
    
    # 创建5D张量
    trajectory_5d = torch.zeros(batch_size, num_agents, seq_len, 5,
                            dtype=trajectory_3d.dtype,
                            device=trajectory_3d.device)
    
    # 填充数据
    trajectory_5d[..., 0] = trajectory_3d[..., 0]  # x
    trajectory_5d[..., 1] = trajectory_3d[..., 1]  # y  
    trajectory_5d[..., 3] = trajectory_3d[..., 2]  # heading
    
    # 在指定维度重复
    # if repeat_dim == 1:  # 智能体维度
    #     trajectory_expanded = trajectory_5d.repeat(1, repeat_count, 1, 1)
    # elif repeat_dim == 0:  # 批次维度
    #     trajectory_expanded = trajectory_5d.repeat(repeat_count, 1, 1, 1)
    # else:
    #     raise ValueError(f"不支持的重复维度: {repeat_dim}")
    
    return trajectory_5d

# def extend_xy_yaw(traj,
#                  new_len=30,
#                  dt=0.2,
#                  method='cubic',            # 'cubic' or 'linear'
#                  extrap_mode='constant_velocity',  # 'extrapolate' | 'constant_velocity' | 'hold'
#                  t_orig=None,
#                  yaw_calc_method='differential'):  # 'differential' or 'spline'
#     """
#     Extend/interpolate batch of 2D trajectories with yaw calculation.

#     Args:
#         traj: torch.Tensor, shape (B, 1, T, 3) with [x, y, yaw]
#         new_len: int, desired temporal length (default 30)
#         dt: float, time step between points (default 0.2)
#         method: 'cubic' or 'linear' interpolation
#         extrap_mode: how to handle times > last original time
#         t_orig: optional 1D array-like of length T with timestamps
#         yaw_calc_method: 'differential' (from x,y) or 'spline' (interpolate yaw directly)

#     Returns:
#         torch.Tensor of shape (B, 1, new_len, 3) with [x, y, yaw]
#     """
#     # 输入验证
#     assert traj.ndim == 4, f"traj must be 4D, got {traj.ndim}D"
#     assert traj.shape[1] == 1, f"Second dim should be 1, got {traj.shape[1]}"
#     assert traj.shape[3] == 3, f"Last dim should be 3 (x,y,yaw), got {traj.shape[3]}"

#     B, _, T, C = traj.shape
#     device = traj.device
#     dtype = traj.dtype

#     # 如果t_orig未提供，使用默认时间戳
#     if t_orig is None:
#         t_orig = (np.arange(T) + 1) * dt
#     else:
#         t_orig = np.asarray(t_orig, dtype=float)
#         assert t_orig.shape[0] == T, f"t_orig length {t_orig.shape[0]} must equal T {T}"

#     # 新时间点
#     t_new = (np.arange(new_len) + 1) * dt

#     # 转换为numpy进行处理
#     traj_np = traj.detach().cpu().numpy()  # (B, 1, T, 3)

#     # 分离x, y, yaw
#     x_orig = traj_np[:, 0, :, 0]  # (B, T)
#     y_orig = traj_np[:, 0, :, 1]  # (B, T)
#     yaw_orig = traj_np[:, 0, :, 2]  # (B, T)

#     # 输出数组
#     out_np = np.zeros((B, 1, new_len, 3), dtype=traj_np.dtype)

#     for b in range(B):
#         # 提取当前批次的x, y, yaw
#         x_b = x_orig[b].astype(float)
#         y_b = y_orig[b].astype(float)
#         yaw_b = yaw_orig[b].astype(float)

#         # 1. 插值x和y
#         def interp_component(vals):
#             """插值/外推一个1D分量"""
#             if method == 'cubic':
#                 try:
#                     # 检查是否有足够点进行三次插值
#                     if len(t_orig) >= 4:
#                         f = interpolate.interp1d(t_orig, vals, kind='cubic',
#                                                fill_value="extrapolate", assume_sorted=True)
#                         base = f(t_new)
#                     else:
#                         # 点太少，退回到线性插值
#                         base = np.interp(t_new, t_orig, vals)
#                 except (ValueError, TypeError):
#                     # 如果三次插值失败，使用线性插值
#                     base = np.interp(t_new, t_orig, vals)
#             else:  # linear
#                 base = np.interp(t_new, t_orig, vals)

#             # 处理外推
#             if extrap_mode == 'extrapolate':
#                 return base
#             elif extrap_mode == 'hold':
#                 mask_after = t_new > t_orig[-1]
#                 if np.any(mask_after):
#                     base[mask_after] = vals[-1]
#                 return base
#             elif extrap_mode == 'constant_velocity':
#                 mask_after = t_new > t_orig[-1]
#                 res = base.copy()
#                 if np.any(mask_after):
#                     if T >= 2:
#                         dt_seg = t_orig[-1] - t_orig[-2]
#                         if dt_seg == 0:
#                             vel = 0.0
#                         else:
#                             vel = (vals[-1] - vals[-2]) / dt_seg
#                         res[mask_after] = vals[-1] + vel * (t_new[mask_after] - t_orig[-1])
#                     else:
#                         # 只有一个点，无法计算速度
#                         res[mask_after] = vals[-1]
#                 return res
#             else:
#                 raise ValueError(f"Unknown extrap_mode: {extrap_mode}")

#         # 插值x和y
#         x_interp = interp_component(x_b)
#         y_interp = interp_component(y_b)

#         # 2. 计算航向角yaw
#         if yaw_calc_method == 'spline':
#             # 方法1: 直接插值yaw
#             yaw_interp = interp_component(yaw_b)
#         else:  # 'differential'
#             # 方法2: 从插值后的x,y计算yaw（更常用）
#             yaw_interp = calculate_yaw_from_xy(x_interp, y_interp, dt)

#         # 存储结果
#         out_np[b, 0, :, 0] = x_interp
#         out_np[b, 0, :, 1] = y_interp
#         out_np[b, 0, :, 2] = yaw_interp

#     # 转换回torch张量
#     out_t = torch.from_numpy(out_np).to(device=device, dtype=dtype)
#     return out_t


# def calculate_yaw_from_xy(x, y, dt=0.2):
#     """
#     从x,y坐标计算航向角yaw

#     Args:
#         x: np.array, x坐标
#         y: np.array, y坐标
#         dt: float, 时间步长

#     Returns:
#         np.array, 航向角yaw (弧度)
#     """
#     n = len(x)
#     yaw = np.zeros(n, dtype=x.dtype)

#     if n == 1:
#         # 只有一个点，无法计算方向
#         return yaw

#     # 计算差分
#     dx = np.diff(x)
#     dy = np.diff(y)

#     # 计算方向角度
#     angles = np.arctan2(dy, dx)

#     # 处理角度跳变（2π跳变）
#     angles = np.unwrap(angles)

#     # 第一个点的yaw用第二个点的值
#     yaw[0] = angles[0]
#     yaw[1:] = angles

#     return yaw


# def calculate_yaw_from_xy_smooth(x, y, dt=0.2, window=3):
#     """
#     从x,y坐标计算平滑的航向角yaw，使用滑动窗口

#     Args:
#         x: np.array, x坐标
#         y: np.array, y坐标
#         dt: float, 时间步长
#         window: int, 滑动窗口大小

#     Returns:
#         np.array, 平滑后的航向角yaw (弧度)
#     """
#     n = len(x)
#     yaw = np.zeros(n, dtype=x.dtype)

#     if n < 2:
#         return yaw

#     # 使用滑动窗口计算更平滑的方向
#     for i in range(n):
#         if i < window - 1:
#             # 使用可用的点数
#             valid_window = i + 1
#             start_idx = 0
#         elif i >= n - window + 1:
#             # 接近末尾
#             valid_window = n - i
#             start_idx = i - valid_window + 1
#         else:
#             # 正常情况
#             valid_window = window
#             start_idx = i - window + 1

#         # 计算窗口内的平均方向
#         if valid_window >= 2:
#             dx = x[i] - x[start_idx]
#             dy = y[i] - y[start_idx]
#             distance = np.sqrt(dx**2 + dy**2)

#             if distance > 1e-6:  # 避免除以零
#                 yaw[i] = np.arctan2(dy, dx)
#             else:
#                 # 如果距离太小，保持前一个值
#                 yaw[i] = yaw[i-1] if i > 0 else 0.0
#         else:
#             yaw[i] = 0.0

#     # 解缠绕角度
#     yaw = np.unwrap(yaw)

#     return yaw

# def _batch_linear_interp_torch(values: torch.Tensor,
#                               t_orig: torch.Tensor,
#                               t_new: torch.Tensor,
#                               extrap_mode: str = 'extrapolate') -> torch.Tensor:
#     """
#     批量线性插值
#     """
#     B, T = values.shape
#     L = len(t_new)

#     # 扩展维度以便广播
#     t_orig_exp = t_orig.unsqueeze(0)  # (1, T)
#     t_new_exp = t_new.unsqueeze(0)    # (1, L)

#     # 找到每个t_new在t_orig中的插入位置
#     indices = torch.searchsorted(t_orig, t_new)
#     indices = torch.clamp(indices, 1, T-1)

#     idx_low = indices - 1
#     idx_high = indices

#     # 收集相邻点的值
#     batch_idx = torch.arange(B, device=values.device).unsqueeze(1)  # (B, 1)

#     values_low = values[batch_idx, idx_low.unsqueeze(0).expand(B, -1)]
#     values_high = values[batch_idx, idx_high.unsqueeze(0).expand(B, -1)]

#     # 获取时间
#     t_low = t_orig[idx_low]
#     t_high = t_orig[idx_high]

#     # 计算插值权重
#     dt_seg = t_high - t_low
#     dt_safe = torch.where(dt_seg.abs() < 1e-6, torch.ones_like(dt_seg), dt_seg)
#     weight_high = (t_new - t_low) / dt_safe
#     weight_low = 1.0 - weight_high

#     # 线性插值
#     interp_values = weight_low.unsqueeze(0) * values_low + weight_high.unsqueeze(0) * values_high

#     # 处理外推
#     if extrap_mode != 'extrapolate':
#         mask_before = t_new < t_orig[0]
#         mask_after = t_new > t_orig[-1]

#         if extrap_mode == 'hold':
#             if mask_before.any():
#                 interp_values[:, mask_before] = values[:, 0:1].expand(-1, mask_before.sum())
#             if mask_after.any():
#                 interp_values[:, mask_after] = values[:, -1:].expand(-1, mask_after.sum())

#         elif extrap_mode == 'constant_velocity' and T >= 2:
#             if mask_before.any():
#                 dt_seg = t_orig[1] - t_orig[0]
#                 if dt_seg.abs() > 1e-6:
#                     velocities = (values[:, 1] - values[:, 0]) / dt_seg
#                     t_extrap = t_new[mask_before] - t_orig[0]
#                     extrap_values = values[:, 0:1] + velocities.unsqueeze(1) * t_extrap.unsqueeze(0)
#                     interp_values[:, mask_before] = extrap_values

#             if mask_after.any():
#                 dt_seg = t_orig[-1] - t_orig[-2]
#                 if dt_seg.abs() > 1e-6:
#                     velocities = (values[:, -1] - values[:, -2]) / dt_seg
#                     t_extrap = t_new[mask_after] - t_orig[-1]
#                     extrap_values = values[:, -1:] + velocities.unsqueeze(1) * t_extrap.unsqueeze(0)
#                     interp_values[:, mask_after] = extrap_values

#     return interp_values


# def _batch_calculate_yaw_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     批量计算yaw
#     """
#     B, L = x.shape

#     if L < 2:
#         return torch.zeros_like(x)

#     # 计算差分
#     dx = x[:, 1:] - x[:, :-1]
#     dy = y[:, 1:] - y[:, :-1]

#     # 计算角度
#     angles = torch.atan2(dy, dx)

#     # 第一个点用第二个点的角度
#     yaw = torch.zeros((B, L), device=x.device, dtype=x.dtype)
#     yaw[:, 0] = angles[:, 0]
#     yaw[:, 1:] = angles

#     return yaw

# def _linear_interpolate_1d_torch(values: torch.Tensor,
#                                 t_orig: torch.Tensor,
#                                 t_new: torch.Tensor,
#                                 extrap_mode: str = 'extrapolate') -> torch.Tensor:
#     """
#     1D线性插值，支持外推模式
#     """
#     T = len(t_orig)
#     L = len(t_new)

#     # 使用torch的内置interp函数（如果有）或手动实现
#     if hasattr(torch, 'interp') and torch.__version__ >= '1.10.0':
#         # 使用torch.interp
#         interp_values = torch.interp(t_new, t_orig, values)

#         # 处理外推
#         if extrap_mode != 'extrapolate':
#             mask_before = t_new < t_orig[0]
#             mask_after = t_new > t_orig[-1]

#             if extrap_mode == 'hold':
#                 if mask_before.any():
#                     interp_values[mask_before] = values[0]
#                 if mask_after.any():
#                     interp_values[mask_after] = values[-1]

#             elif extrap_mode == 'constant_velocity' and T >= 2:
#                 if mask_before.any():
#                     dt_seg = t_orig[1] - t_orig[0]
#                     if dt_seg.abs() > 1e-6:
#                         vel = (values[1] - values[0]) / dt_seg
#                         interp_values[mask_before] = values[0] + vel * (t_new[mask_before] - t_orig[0])

#                 if mask_after.any():
#                     dt_seg = t_orig[-1] - t_orig[-2]
#                     if dt_seg.abs() > 1e-6:
#                         vel = (values[-1] - values[-2]) / dt_seg
#                         interp_values[mask_after] = values[-1] + vel * (t_new[mask_after] - t_orig[-1])

#         return interp_values
#     else:
#         # 手动实现线性插值
#         interp_values = torch.zeros_like(t_new)

#         for i, t in enumerate(t_new):
#             # 找到插值区间
#             if t <= t_orig[0]:
#                 # 在第一个点之前
#                 if extrap_mode == 'hold':
#                     interp_values[i] = values[0]
#                 elif extrap_mode == 'constant_velocity' and T >= 2:
#                     dt_seg = t_orig[1] - t_orig[0]
#                     if dt_seg.abs() > 1e-6:
#                         vel = (values[1] - values[0]) / dt_seg
#                         interp_values[i] = values[0] + vel * (t - t_orig[0])
#                     else:
#                         interp_values[i] = values[0]
#                 else:  # 'extrapolate' 或其他
#                     interp_values[i] = values[0]

#             elif t >= t_orig[-1]:
#                 # 在最后一个点之后
#                 if extrap_mode == 'hold':
#                     interp_values[i] = values[-1]
#                 elif extrap_mode == 'constant_velocity' and T >= 2:
#                     dt_seg = t_orig[-1] - t_orig[-2]
#                     if dt_seg.abs() > 1e-6:
#                         vel = (values[-1] - values[-2]) / dt_seg
#                         interp_values[i] = values[-1] + vel * (t - t_orig[-1])
#                     else:
#                         interp_values[i] = values[-1]
#                 else:  # 'extrapolate' 或其他
#                     interp_values[i] = values[-1]

#             else:
#                 # 在中间，进行线性插值
#                 # 使用searchsorted找到插入位置
#                 idx = torch.searchsorted(t_orig, t)
#                 if idx == 0:
#                     idx = 1

#                 t0, t1 = t_orig[idx-1], t_orig[idx]
#                 v0, v1 = values[idx-1], values[idx]

#                 # 线性插值
#                 alpha = (t - t0) / (t1 - t0)
#                 interp_values[i] = v0 + alpha * (v1 - v0)

#         return interp_values

# def _cubic_interpolate_1d_torch(values: torch.Tensor,
#                               t_orig: torch.Tensor,
#                               t_new: torch.Tensor,
#                               extrap_mode: str = 'extrapolate') -> torch.Tensor:
#     """
#     三次样条插值，使用PyTorch实现
#     注意：这是一个简化的实现，对于精确的三次样条，建议使用scipy
#     """
#     T = len(t_orig)
#     L = len(t_new)

#     if T < 4:
#         # 点太少，退回到线性插值
#         return _linear_interpolate_1d_torch(values, t_orig, t_new, extrap_mode)

#     # 使用grid_sample进行三次插值
#     # 将1D数据转换为4D格式 (1, 1, 1, T)
#     values_4d = values.view(1, 1, 1, T)

#     # 创建采样网格 (-1 到 1)
#     # 计算t_new在t_orig范围内的归一化位置
#     t_min, t_max = t_orig[0], t_orig[-1]
#     t_range = t_max - t_min

#     if t_range > 1e-6:
#         # 归一化到 [-1, 1]
#         t_norm = 2 * (t_new - t_min) / t_range - 1
#     else:
#         t_norm = torch.zeros_like(t_new)

#     # 创建采样网格
#     grid = torch.stack([
#         torch.zeros_like(t_norm),  # 不需要height维度
#         t_norm.unsqueeze(-1)       # width维度
#     ], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, L, 2)

#     # 使用grid_sample进行三次插值
#     interp_values_4d = F.grid_sample(
#         values_4d,
#         grid,
#         mode='bicubic',
#         padding_mode='border',  # 使用边界值进行外推
#         align_corners=True
#     )

#     interp_values = interp_values_4d.squeeze()  # (L,)

#     # 处理外推模式
#     if extrap_mode != 'extrapolate':
#         mask_before = t_new < t_orig[0]
#         mask_after = t_new > t_orig[-1]

#         if extrap_mode == 'hold':
#             if mask_before.any():
#                 interp_values[mask_before] = values[0]
#             if mask_after.any():
#                 interp_values[mask_after] = values[-1]

#         elif extrap_mode == 'constant_velocity' and T >= 2:
#             if mask_before.any():
#                 dt_seg = t_orig[1] - t_orig[0]
#                 if dt_seg.abs() > 1e-6:
#                     vel = (values[1] - values[0]) / dt_seg
#                     interp_values[mask_before] = values[0] + vel * (t_new[mask_before] - t_orig[0])

#             if mask_after.any():
#                 dt_seg = t_orig[-1] - t_orig[-2]
#                 if dt_seg.abs() > 1e-6:
#                     vel = (values[-1] - values[-2]) / dt_seg
#                     interp_values[mask_after] = values[-1] + vel * (t_new[mask_after] - t_orig[-1])

#     return interp_values

# # 批量处理的高性能版本

# def extend_xy_yaw_torch_batch(traj: torch.Tensor,
#                              new_len: int = 30,
#                              dt: float = 0.2,
#                              method: str = 'linear',
#                              extrap_mode: str = 'constant_velocity',
#                              yaw_calc_method: str = 'differential') -> torch.Tensor:
#     """
#     批量处理版本的轨迹延长，使用PyTorch向量化操作
#     """
#     B, _, T, C = traj.shape
#     device = traj.device
#     dtype = traj.dtype

#     # 时间戳
#     t_orig = (torch.arange(T, device=device, dtype=dtype) + 1) * dt
#     t_new = (torch.arange(new_len, device=device, dtype=dtype) + 1) * dt

#     # 分离x, y
#     x = traj[:, 0, :, 0]  # (B, T)
#     y = traj[:, 0, :, 1]  # (B, T)
#     yaw = traj[:, 0, :, 2]  # (B, T)

#     # 批量线性插值
#     if method == 'linear':
#         x_interp = _batch_linear_interp_torch(x, t_orig, t_new, extrap_mode)
#         y_interp = _batch_linear_interp_torch(y, t_orig, t_new, extrap_mode)
#     else:
#         # 三次样条插值（对每个batch单独处理）
#         x_interp = torch.zeros((B, new_len), device=device, dtype=dtype)
#         y_interp = torch.zeros((B, new_len), device=device, dtype=dtype)

#         for b in range(B):
#             x_interp[b] = _cubic_interpolate_1d_torch(x[b], t_orig, t_new, extrap_mode)
#             y_interp[b] = _cubic_interpolate_1d_torch(y[b], t_orig, t_new, extrap_mode)

#     # 计算yaw
#     if yaw_calc_method == 'spline':
#         if method == 'linear':
#             yaw_interp = _batch_linear_interp_torch(yaw, t_orig, t_new, extrap_mode)
#         else:
#             yaw_interp = torch.zeros((B, new_len), device=device, dtype=dtype)
#             for b in range(B):
#                 yaw_interp[b] = _cubic_interpolate_1d_torch(yaw[b], t_orig, t_new, extrap_mode)
#     else:  # 'differential'
#         yaw_interp = _batch_calculate_yaw_torch(x_interp, y_interp)

#     # 组合结果
#     result = torch.stack([x_interp, y_interp, yaw_interp], dim=-1).unsqueeze(1)  # (B, 1, L, 3)

#     return result

# def extend_xy_yaw_onnx_compatible(traj: torch.Tensor,
#                                   new_len: int = 30,
#                                   dt: float = 0.2,
#                                   method: str = 'linear',
#                                   extrap_mode: str = 'constant_velocity',
#                                   yaw_calc_method: str = 'differential') -> torch.Tensor:
#     """
#     ONNX兼容的轨迹延长函数
#     完全避免Python控制流和torch.searchsorted
#     """
#     B, C, T, D = traj.shape
    
#     # 提取x, y
#     x = traj[:, 0, :, 0]  # (B, T)
#     y = traj[:, 0, :, 1]  # (B, T)
#     yaw = traj[:, 0, :, 2]  # (B, T)
    
#     # 创建时间网格
#     device = traj.device
#     dtype = traj.dtype
    
#     # 使用tensor的shape获取维度，避免len()
#     t_orig = (torch.arange(T, device=device, dtype=dtype) + 1) * dt
#     t_new = (torch.arange(new_len, device=device, dtype=dtype) + 1) * dt
    
#     # 批量线性插值
#     if method == 'linear':
#         x_interp = _onnx_batch_linear_interp(x, t_orig, t_new, extrap_mode)
#         y_interp = _onnx_batch_linear_interp(y, t_orig, t_new, extrap_mode)
#     else:
#         # 使用torch的grid_sample进行插值
#         x_interp = _onnx_batch_grid_interp(x, t_orig, t_new, method='bicubic')
#         y_interp = _onnx_batch_grid_interp(y, t_orig, t_new, method='bicubic')
    
#     # 计算yaw
#     if yaw_calc_method == 'spline':
#         if method == 'linear':
#             yaw_interp = _onnx_batch_linear_interp(yaw, t_orig, t_new, extrap_mode)
#         else:
#             yaw_interp = _onnx_batch_grid_interp(yaw, t_orig, t_new, method='bicubic')
#     else:  # 'differential'
#         yaw_interp = _onnx_calculate_yaw(x_interp, y_interp)
    
#     # 组合结果
#     result = torch.stack([x_interp, y_interp, yaw_interp], dim=-1).unsqueeze(1)
    
#     return result


# def _onnx_batch_linear_interp(values: torch.Tensor,
#                               t_orig: torch.Tensor,
#                               t_new: torch.Tensor,
#                               extrap_mode: str = 'extrapolate') -> torch.Tensor:
#     """
#     ONNX兼容的批量线性插值
#     使用grid_sample替代searchsorted
#     """
#     B, T = values.shape
#     L = t_new.shape[0]
    
#     # 将1D数据转换为4D格式以使用grid_sample
#     # values: (B, T) -> (B, 1, 1, T)
#     values_4d = values.view(B, 1, 1, T)
    
#     # 归一化t_new到[-1, 1]范围
#     t_min, t_max = t_orig[0], t_orig[-1]
#     t_range = t_max - t_min
    
#     # 避免除零
#     t_range_safe = torch.where(t_range.abs() < 1e-6, torch.ones_like(t_range), t_range)
    
#     # 归一化
#     t_norm = 2.0 * (t_new - t_min) / t_range_safe - 1.0
    
#     # 创建采样网格
#     # grid_sample期望的网格格式: (N, H_out, W_out, 2)
#     # 对于1D插值，我们使用height=1, width=L
#     zeros_1 = torch.zeros((B, L), device=values.device, dtype=values.dtype)  # 形状 (B, L)
#     grid = torch.stack([
#         zeros_1,  # height坐标（保持为0）
#         t_norm.unsqueeze(0).expand(B, L)  # width坐标
#     ], dim=-1).unsqueeze(1)  # (B, 1, L, 2)
    
#     # 使用grid_sample进行线性插值
#     interp_4d = F.grid_sample(
#         values_4d,
#         grid,
#         mode='bilinear',
#         padding_mode='border',  # 使用边界值进行外推
#         align_corners=True
#     )
    
#     interp_values = interp_4d.squeeze(2).squeeze(1)  # (B, L)
    
#     # 处理外推模式
#     if extrap_mode == 'hold':
#         # 创建掩码
#         mask_before = t_new < t_orig[0]
#         mask_after = t_new > t_orig[-1]
        
#         # 使用torch.where处理边界
#         interp_values = torch.where(
#             mask_before.unsqueeze(0).expand(B, -1),
#             values[:, 0:1].expand(-1, L),
#             interp_values
#         )
        
#         interp_values = torch.where(
#             mask_after.unsqueeze(0).expand(B, -1),
#             values[:, -1:].expand(-1, L),
#             interp_values
#         )
    
#     elif extrap_mode == 'constant_velocity':
#         # 恒定速度外推
#         mask_before = t_new < t_orig[0]
#         mask_after = t_new > t_orig[-1]
        
#         # 计算速度
#         dt_seg = t_orig[1] - t_orig[0]
#         dt_seg_safe = torch.where(dt_seg.abs() < 1e-6, torch.ones_like(dt_seg), dt_seg)
#         velocities = (values[:, 1] - values[:, 0]) / dt_seg_safe
        
#         # 前向外推
#         if mask_before.any():
#             t_extrap_before = t_new[mask_before] - t_orig[0]
#             extrap_before = values[:, 0:1] + velocities.unsqueeze(1) * t_extrap_before.unsqueeze(0)
#             interp_values[:, mask_before] = extrap_before
        
#         # 后向外推
#         if mask_after.any():
#             dt_seg_end = t_orig[-1] - t_orig[-2]
#             dt_seg_end_safe = torch.where(dt_seg_end.abs() < 1e-6, torch.ones_like(dt_seg_end), dt_seg_end)
#             velocities_end = (values[:, -1] - values[:, -2]) / dt_seg_end_safe
            
#             t_extrap_after = t_new[mask_after] - t_orig[-1]
#             extrap_after = values[:, -1:] + velocities_end.unsqueeze(1) * t_extrap_after.unsqueeze(0)
#             interp_values[:, mask_after] = extrap_after
    
#     return interp_values


# def _onnx_batch_grid_interp(values: torch.Tensor,
#                            t_orig: torch.Tensor,
#                            t_new: torch.Tensor,
#                            method: str = 'bicubic') -> torch.Tensor:
#     """
#     使用grid_sample进行批量插值，支持bicubic
#     """
#     B, T = values.shape
#     L = t_new.shape[0]
    
#     # 将1D数据转换为4D格式
#     values_4d = values.view(B, 1, 1, T)
    
#     # 归一化t_new
#     t_min, t_max = t_orig[0], t_orig[-1]
#     t_range = t_max - t_min
    
#     # 避免除零
#     t_range_safe = torch.where(t_range.abs() < 1e-6, torch.ones_like(t_range), t_range)
    
#     # 归一化到[-1, 1]
#     t_norm = 2.0 * (t_new - t_min) / t_range_safe - 1.0
    
#     # 创建采样网格
#     grid = torch.stack([
#         torch.zeros_like(t_norm),
#         t_norm.unsqueeze(0).expand(B, -1)
#     ], dim=-1).unsqueeze(1)  # (B, 1, L, 2)
    
#     # 选择插值模式
#     mode = 'bilinear' if method == 'linear' else 'bicubic'
    
#     # 使用grid_sample插值
#     interp_4d = F.grid_sample(
#         values_4d,
#         grid,
#         mode=mode,
#         padding_mode='border',
#         align_corners=True
#     )
    
#     interp_values = interp_4d.squeeze(2).squeeze(1)  # (B, L)
    
#     return interp_values


# def _onnx_calculate_yaw(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     ONNX兼容的yaw计算
#     避免Python控制流
#     """
#     B, L = x.shape
    
#     # 使用torch.where处理边界条件
#     # 当L<2时，返回零张量
#     yaw = torch.zeros((B, L), device=x.device, dtype=x.dtype)
    
#     # 使用条件掩码
#     mask = torch.tensor(L >= 2, device=x.device, dtype=torch.bool)
    
#     if mask:
#         # 计算差分
#         dx = x[:, 1:] - x[:, :-1]
#         dy = y[:, 1:] - y[:, :-1]
        
#         # 计算角度
#         angles = torch.atan2(dy, dx)
        
#         # 第一个点用第二个点的角度
#         yaw[:, 0] = angles[:, 0]
#         yaw[:, 1:] = angles
    
#     return yaw
