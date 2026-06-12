import torch
import torch.nn as nn


class BarlowTwins(nn.Module):
    def __init__(self, lambda_red=0.005):
        """
        Args:
            lambda_red (float): Weight for the redundancy reduction term
        """
        super(BarlowTwins, self).__init__()
        self.lambda_red = lambda_red

        # normalization layer for the representations z1 and z2

    def forward(self, y1, y2):
        # empirical cross-correlation matrix
        c = y1.T @ y2
        batch_size = y1.size(0)
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        n,m = c.shape
        off_diag = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten().pow_(2).sum()
        loss = on_diag + self.lambda_red * off_diag
        return loss

class DirectGmmLoss():
    def __init__(self):
        pass
    def update(self,nearest_trajs, gt_trajs,valid_mask,
                    log_std_range=(-6.609, 5.0), rho_limit=0.5):

        # assert nearest_trajs.shape[-1] == 6 # for joint model last dim is 6; for multi-modal model last dim is 5.

        res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        # time-dependent lower bound for log_std: std(t=0)=1e-4 -> std(t=80)=0.2
        T= nearest_trajs.shape[1]
        device = nearest_trajs.device
        dtype = nearest_trajs.dtype
        t_idx = torch.arange(T).to(device, dtype)
        log_min_0 = torch.tensor(-9.2103).to(device, dtype)  # ln(1e-4) at 0
        log_min_T = torch.tensor(-1.609).to(device, dtype)   # ln(0.2) at T
        log_max_0 = torch.tensor(1.1939).to(device, dtype)   # ln(3.3) at 0
        log_max_T = torch.tensor(5).to(device, dtype)        # ln(150) at T
        alpha = t_idx / T
        log_std_min_t = (log_min_0 + (log_min_T - log_min_0) * alpha).view(1, T)
        log_std_max = (log_max_0 + (log_max_T - log_max_0) * alpha).view(1, T)
        log_std1 = torch.clip(nearest_trajs[:, :, 3], min=log_std_min_t, max=log_std_max)
        log_std2 = torch.clip(nearest_trajs[:, :, 4], min=log_std_min_t, max=log_std_max)
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

        #cls_loss = self.loss_fn_c(pred_scores,label_score) #[B*A, ]
        reg_loss = (reg_gmm_log_coefficient + reg_gmm_exp)
        if valid_mask is None:
            reg_loss = reg_loss.sum(dim=-1) #[B*A, ]
        else:
            T = reg_loss.shape[-1]
            reg_loss =T * (reg_loss * valid_mask).sum(dim = -1) / torch.clip(valid_mask.sum(dim = -1),min=1.)
        return reg_loss.mean() 


class SmoothL1LossMasked(nn.Module):
    def __init__(self):
        super(SmoothL1LossMasked, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target, valid_mask):
        '''
        pred: [B*A,T,2] or [B*A,T]
        target: [B*A,T,2] or [B*A,T]
        valid_mask: [B*A,T]
        '''
        if valid_mask is None:
            loss = self.smooth_l1_loss(pred, target)
            return loss.mean()
        if valid_mask.shape[-1] != pred.shape[-1]:
            # 2d case
            valid_mask = valid_mask.unsqueeze(-1).expand_as(pred)
        loss = self.smooth_l1_loss(pred*valid_mask, target*valid_mask)
        return loss.mean()