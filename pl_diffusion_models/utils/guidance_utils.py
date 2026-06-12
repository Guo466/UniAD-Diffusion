import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from utils.utils import img2real
from typing import List, Dict
import os

colors = [
    'cyan', 'green', 'purple', 'magenta', 'blue', 'orange', 
]

def build_sdf_from_occ_map(occ_map: torch.Tensor, resolution: float) -> torch.Tensor:
    """
    Args:
        occ_map: (1,H,W) int/uint8/bool, 1=ground (FREE), others (0 unknown / 2 occupied / 3 construction) are OBSTACLE
    Returns:  
        (1,1,H,W) signed distance field, >0 in free, ~=0 boundary, <0 inside obstacle
    """
    occ_np = occ_map.detach().cpu().numpy()
    free_np = (occ_np == 1)  # bool, (B,H,W)

    B, H, W = free_np.shape
    sdf = np.empty((B, H, W), dtype=np.float32)

    for b in range(B):
        free = free_np[b]
        obst = ~free

        # Outside distance: free cells to nearest obstacle
        dist_out = distance_transform_edt(free)

        # Inside distance: obstacle cells to nearest free
        dist_in = distance_transform_edt(obst)

        sdf_b = dist_out.astype(np.float32)
        sdf_b[obst] = -dist_in[obst].astype(np.float32)
        sdf[b] = sdf_b * float(resolution)          # meters

    return torch.from_numpy(sdf).unsqueeze(1).to(occ_map.device)  # (1,1,H,W)

def batch_bilinear_interpolation(costmap, coords):
    """
    costmap:    (vaild, H, W)
    coords :    (vaild, 2)  in [-1, 1]
    return :    (vaild)
    """
    costmap = costmap.unsqueeze(1)                                                  # (vaild, 1, H, W)
    coords = coords.unsqueeze(1).unsqueeze(1)                                       # (vaild, 1, 1, 2)
    output = F.grid_sample(costmap, coords, mode='bilinear', align_corners=True)    # (vaild, 1, 1, 1)
    result = output[:, 0, 0, 0]                                                     # (vaild,)
    return result

def costmap_guidance_func(traj: torch.Tensor, costmap: torch.Tensor, occ_guidance_params: dict, dif_timestep: int) -> torch.Tensor:
    """
    OCC collision energy E_occ (scalar).
    Args:
        traj: size(N, 1, T, 3), Multi-modal prediction
        occ_map: (1, T, H, W)
        occ_guidance_params: parameters for guidance
    Returns:
        scaled cost: scalar
    """
    res = float(occ_guidance_params["resolution"])     # 0.4
    i0 = float(occ_guidance_params["i0"])
    j0 = float(occ_guidance_params["j0"])
    r = float(occ_guidance_params["r"])
    omega = float(occ_guidance_params["omega"])
    inflate = float(occ_guidance_params.get("inflate", 0.0))
    eps = 1e-6
    
    N, _, T, _ = traj.shape
    costmap = costmap.expand(N, -1, -1, -1)                     # (N, T, H, W)
    _, _, H, W = costmap.shape

    traj_xy = traj[..., 0:2].squeeze(1)                         # (N, T, 2), real -> grid coord
    j = traj_xy[..., 0] / (res + eps) + j0
    i = i0 - traj_xy[..., 1] / (res + eps)

    gx = (j / (max(W - 1, 1))) * 2.0 - 1.0
    gy = (i / (max(H - 1, 1))) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)                        # (N, T, 2), -> [-1, 1]

    valid_mask = ((gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1))
    d = batch_bilinear_interpolation(               
        costmap[valid_mask],                        # (vaild, H, W)
        grid[valid_mask]                            # (vaild, 2)
    )                                               # (vaild)

    d_eff = d - inflate
    danger = torch.clamp(1.0 - d_eff / (r + eps), min=0.0)             # (N,T)
    psi = (torch.exp(omega * danger) - omega * danger) / (omega + eps)
    return psi.mean()

def costmap_train_loss(traj: torch.Tensor, costmap: torch.Tensor, occ_guidance_params: dict, dif_timestep: int) -> torch.Tensor:
    """
    OCC collision energy E_occ (scalar).
    Args:
        traj: size(bs, 1, T, 3), Multi-modal prediction
        occ_map: (bs, T, H, W)
        occ_guidance_params: parameters for guidance
    Returns:
        scaled cost: scalar
    """
    res = float(occ_guidance_params["resolution"])     # 0.4
    i0 = float(occ_guidance_params["i0"])
    j0 = float(occ_guidance_params["j0"])
    r = float(occ_guidance_params["r"])
    eps = 1e-6
    
    N, _, T, _ = traj.shape
    _, _, H, W = costmap.shape
    costmap = costmap.expand(-1, T, -1, -1)

    traj_xy = traj[..., 0:2].squeeze(1)                         # (N, T, 2), real -> grid coord
    j = traj_xy[..., 0] / (res + eps) + j0
    i = i0 - traj_xy[..., 1] / (res + eps)

    gx = (j / (max(W - 1, 1))) * 2.0 - 1.0
    gy = (i / (max(H - 1, 1))) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)                        # (N, T, 2), -> [-1, 1]

    valid_mask = ((gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1))
    if not valid_mask.any():
        return traj.new_zeros(())

    d = batch_bilinear_interpolation(               
        costmap[valid_mask],                        # (vaild, H, W)
        grid[valid_mask]                            # (vaild, 2)
    )                                               # (vaild)
    cost = (r - d).clamp_min(0)                     # (vaild)
    loss = cost.sum() / ( (cost > 0).sum() + 1e-6 )
    return loss

def compute_laneline_point_weight(traj_pts, laneline_pts, laneline_attrs, laneline_mask,
                                  dist_threshold=3.0):
    """
    根据轨迹点与最近车道线的类型，计算逐点权重。

    规则（取距离 < dist_threshold 的最近 2 条车道线/curb）：
      - 没有车道线 → 0.5
      - 都是虚线   → 1.0
      - 1条实线/curb → 1.5
      - 2条实线/curb → 2.0

    Args:
        traj_pts:       (B, T, 2)   轨迹 xy
        laneline_pts:   (B, N, P, 2) 车道线采样点
        laneline_attrs: (B, N, 3)   [color, laneline_type, laneline_style]
        laneline_mask:  (B, N)      0=valid, 1=invalid
        dist_threshold: float       距离阈值，默认 3.0m
    Returns:
        point_weight:   (B, T)
    """
    B, T, _ = traj_pts.shape
    N, P = laneline_pts.shape[1], laneline_pts.shape[2]
    device = traj_pts.device
    dtype = traj_pts.dtype

    ltype  = laneline_attrs[..., 1]   # (B, N)
    lstyle = laneline_attrs[..., 2]   # (B, N)

    is_boundary = (ltype == 1) | (ltype == 2)                          # (B, N)
    is_solid_or_curb = (ltype == 1) | ((ltype == 2) & (lstyle == 2))   # (B, N)
    valid_lane = (~laneline_mask.bool()) & is_boundary                 # (B, N)

    min_dist_to_lane = torch.full((B, T, N), float('inf'), device=device, dtype=dtype)
    chunk_size = 10
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        lane_chunk = laneline_pts[:, i:j, :, :]                                 # (B, chunk, P, 2)
        diff = traj_pts[:, :, None, None, :] - lane_chunk[:, None, :, :, :]     # (B, T, chunk, P, 2)
        dist2 = (diff * diff).sum(-1)                                           # (B, T, chunk, P)
        min_dist_to_lane[:, :, i:j] = dist2.min(dim=-1).values.sqrt()           # (B, T, chunk)

    min_dist_to_lane.masked_fill_(~valid_lane.unsqueeze(1), float('inf'))

    k = min(2, N)
    top2_dist, top2_idx = min_dist_to_lane.topk(k, dim=-1, largest=False)      # (B, T, k)
    within = top2_dist < dist_threshold                                          # (B, T, k)

    is_sc_exp = is_solid_or_curb.unsqueeze(1).expand(-1, T, -1)                 # (B, T, N)
    top2_sc = torch.gather(is_sc_exp, 2, top2_idx) & within                    # (B, T, k)

    n_valid = within.sum(-1)                                                     # (B, T)
    n_solid = top2_sc.sum(-1)                                                    # (B, T)

    point_weight = torch.full((B, T), 0.5, device=device, dtype=dtype)
    has_lane = n_valid > 0
    point_weight[has_lane & (n_solid == 0)] = 1.0
    point_weight[has_lane & (n_solid == 1)] = 1.5
    point_weight[has_lane & (n_solid >= 2)] = 2.0

    return point_weight


def soft_L2_train_loss(traj, target_navi, target_mask_navi=None, laneline_point_weight=None):
    """
    支持多候选 navi 目标，选 loss 最小的那条回传梯度。
    向后兼容旧的单候选输入格式。

    traj: (B, 1, T, 3)
    target_navi: (B, K, T, 2) 或 (B, T, 2)（自动升维）
    target_mask_navi: (B, K, T) 或 (B, T)（自动升维）
    laneline_point_weight: (B, T) or None  — 逐点车道线权重
    return :
        loss_x, loss_y
    """
    traj = traj.squeeze(1)[..., :2]     # (B, T, 2)
    B, T, _ = traj.shape
    device = traj.device
    dtype = traj.dtype

    # 兼容旧的 (B, T, 2) 输入，自动升维为 (B, 1, T, 2)
    if target_navi.dim() == 3:
        target_navi = target_navi.unsqueeze(1)
    if target_mask_navi is not None and target_mask_navi.dim() == 2:
        target_mask_navi = target_mask_navi.unsqueeze(1)

    K = target_navi.shape[1]

    if target_mask_navi is None:
        target_mask_navi = torch.ones((B, K, T), device=device, dtype=dtype)

    # 误差：(B, K, T, 2)
    diff = traj.unsqueeze(1) - target_navi

    dx = diff[..., 0]  # (B, K, T)
    dy = diff[..., 1]  # (B, K, T)

    # mask: (B, K, T)
    weight = target_mask_navi.to(dtype)
    if laneline_point_weight is not None:
        weight = weight * laneline_point_weight.unsqueeze(1)  # broadcast (B, 1, T)

    denom = weight.sum(dim=-1).clamp(min=1)  # (B, K)

    # 每个候选的 loss: (B, K)
    loss_x_all = (dx ** 2 * weight).sum(dim=-1) / denom
    loss_y_all = (dy ** 2 * weight).sum(dim=-1) / denom

    # 选择最优候选（total loss 最小的那条）
    loss_total_all = loss_x_all + loss_y_all  # (B, K)

    is_valid = target_mask_navi.any(dim=-1)   # (B, K)
    huge = torch.finfo(dtype).max
    loss_total_masked = loss_total_all.masked_fill(~is_valid, huge)
    best_idx = loss_total_masked.argmin(dim=1)  # (B,)

    # gather 最优候选的 loss_x, loss_y
    loss_x_best = loss_x_all.gather(1, best_idx[:, None]).squeeze(1)  # (B,)
    loss_y_best = loss_y_all.gather(1, best_idx[:, None]).squeeze(1)  # (B,)

    has_valid = is_valid.any(dim=-1).to(dtype)  # (B,)
    loss_x_best = loss_x_best * has_valid
    loss_y_best = loss_y_best * has_valid

    loss_x = loss_x_best.mean()
    loss_y = loss_y_best.mean()

    return loss_x, loss_y

def vis_guidance_grad(
    free_pred_trajs_denorm: torch.Tensor,
    guidanced_path_denorm: torch.Tensor,
    grad_list: List[torch.Tensor],
    model_input,
    save_path: str,
    safe_range=1.4,
    xlim=(-10, 80),
    ylim=(-30, 30),
):
    import matplotlib.pyplot as plt

    # (B, ...) -> (...): only use first batch
    occ_map = model_input['occ_map'].detach()[0].cpu().numpy()                 # (H, W)     
    # curb_pts = model_input['curb_pts'].detach()[0].cpu().numpy()               # (N, P, 2)
    costmap = model_input['path_costmap'].detach()[0].cpu().numpy()            # (T, H, W)
    navitopo_pts = model_input['navitopo_pts'].detach()[0].cpu().numpy()       # (N, P, 2)
    rewardmap = model_input['path_rewardmap'].detach()[0].cpu().numpy()        # (T, H, W)
    curb_pts = navitopo_pts # tmp

    B = guidanced_path_denorm.shape[0]
    traj = guidanced_path_denorm.detach().cpu().squeeze(1)                      # [B, T, 3]
    free_traj = free_pred_trajs_denorm.detach().cpu().squeeze(1)                # [B, T, 3]
    grad_list = [grad.detach().cpu().squeeze(1) for grad in grad_list]

    nrows, ncols = B, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5 * B), squeeze=False)

    for i in range(B):
        guidanced_x, guidanced_y = traj[i, :, 0], traj[i, :, 1]
        free_x, free_y = free_traj[i, :, 0], free_traj[i, :, 1]

        # =======================
        # Column 0: ALL
        # =======================
        ax = axes[i, 0]

        # draw occ_map
        occ_map_img = occ_map.copy()                                    # (H, W)
        occ_map_img[occ_map_img > 1] = 0
        H, W = occ_map_img.shape
        ii, jj = np.meshgrid(np.arange(H), np.arange(W),indexing="ij")
        real_pts = img2real(np.stack([jj, ii], axis=-1), resolution=0.4, i0=60, j0=25)  # (H, W, 2)
        x_map, y_map = real_pts[..., 0], real_pts[..., 1]
        pcm = ax.pcolormesh(x_map, y_map, occ_map_img, shading="auto", cmap="coolwarm_r")
        fig.colorbar(pcm, ax=ax, fraction=0.046, label="occmap type")

        # draw curb
        for pts in curb_pts: ax.plot(pts[:, 0], pts[:, 1], 'b-')
        # draw navitopo
        for pts in navitopo_pts: ax.plot(pts[:, 0], pts[:, 1], color='cyan')
        # draw Trajectory
        ax.plot(free_x, free_y, '-o', color='red', markersize=2)
        ax.plot(guidanced_x, guidanced_y, '-o', color='green', markersize=2)

        ax.set_title(f"Traj {i} | ALL")
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)

        # =======================
        # Column 1: occ_map
        # =======================
        ax = axes[i, 1]

        # draw occ_map
        occ_map_img = occ_map.copy()                                    # (H, W)
        occ_map_img[occ_map_img > 1] = 0
        H, W = occ_map_img.shape
        ii, jj = np.meshgrid(np.arange(H), np.arange(W),indexing="ij")
        real_pts = img2real(np.stack([jj, ii], axis=-1), resolution=0.4, i0=60, j0=25)  # (H, W, 2)
        x_map, y_map = real_pts[..., 0], real_pts[..., 1]
        pcm = ax.pcolormesh(x_map, y_map, occ_map_img, shading="auto", cmap="coolwarm_r")
        fig.colorbar(pcm, ax=ax, fraction=0.046, label="occmap type")

        # draw navitopo
        for pts in navitopo_pts: ax.plot(pts[:, 0], pts[:, 1], color='cyan')
        # draw Trajectory
        ax.plot(free_x, free_y, '-o', color='red', markersize=2)
        ax.plot(guidanced_x, guidanced_y, '-o', color='green', markersize=2)
        # Finally, draw Gradient
        for idx, grad in enumerate(grad_list):
            dx, dy = grad[i, :, 0], grad[i, :, 1]
            ax.quiver(guidanced_x, guidanced_y, dx, dy, angles='xy', scale_units='xy', scale=0.1, color=colors[idx], width=0.003)

        ax.set_title(f"Traj {i} | OccMap")
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)

        # =======================
        # Column 2: curb_pts
        # =======================
        ax = axes[i, 2]

        # draw curb
        for pts in curb_pts: ax.plot(pts[:, 0], pts[:, 1], 'b-')

        # draw navitopo
        for pts in navitopo_pts: ax.plot(pts[:, 0], pts[:, 1], color='cyan')
        # draw Trajectory
        ax.plot(free_x, free_y, '-o', color='red', markersize=2)
        ax.plot(guidanced_x, guidanced_y, '-o', color='green', markersize=2)
        # Finally, draw Gradient
        for idx, grad in enumerate(grad_list):
            dx, dy = grad[i, :, 0], grad[i, :, 1]
            ax.quiver(guidanced_x, guidanced_y, dx, dy, angles='xy', scale_units='xy', scale=0.1, color=colors[idx], width=0.003)

        ax.set_title(f"Traj {i} | Crub")
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)

        # =======================
        # Column 3: costmap
        # =======================
        ax = axes[i, 3]

        # draw Costmap
        costmap_img = costmap.copy()                                    # [T, H, W]
        costmap_img = costmap_img[0] if costmap_img.ndim == 3 else costmap_img          # (T, H, W) -> (H, W): only draw first
        H, W = costmap_img.shape
        ii, jj = np.meshgrid(np.arange(H), np.arange(W),indexing="ij")
        real_pts = img2real(np.stack([jj, ii], axis=-1), resolution=0.4, i0=50, j0=0)   # (H, W, 2)
        x_map, y_map = real_pts[..., 0], real_pts[..., 1]                               # (H, W)
        pcm = ax.pcolormesh(x_map, y_map, costmap_img, shading="auto", cmap="coolwarm_r")
        fig.colorbar(pcm, ax=ax, fraction=0.046, label="SDF (m)")

        # draw navitopo
        for pts in navitopo_pts: ax.plot(pts[:, 0], pts[:, 1], color='cyan')
        # draw Trajectory
        ax.plot(free_x, free_y, '-o', color='red', markersize=2)
        ax.plot(guidanced_x, guidanced_y, '-o', color='green', markersize=2)
        # Finally, draw Gradient
        for idx, grad in enumerate(grad_list):
            dx, dy = grad[i, :, 0], grad[i, :, 1]
            ax.quiver(guidanced_x, guidanced_y, dx, dy, angles='xy', scale_units='xy', scale=0.1, color=colors[idx], width=0.003)

        ax.set_title(f"Traj {i} | CostMap")
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)

        # =======================
        # Column 4: rewardmap
        # =======================
        ax = axes[i, 4]

        # draw Rewardmap
        rewardmap_img = rewardmap.copy()                                    # [T, H, W]
        rewardmap_img = rewardmap_img[0] if rewardmap_img.ndim == 3 else rewardmap_img          # (T, H, W) -> (H, W): only draw first
        H, W = rewardmap_img.shape
        ii, jj = np.meshgrid(np.arange(H), np.arange(W),indexing="ij")
        real_pts = img2real(np.stack([jj, ii], axis=-1), resolution=0.4, i0=50, j0=0)   # (H, W, 2)
        x_map, y_map = real_pts[..., 0], real_pts[..., 1]                               # (H, W)
        pcm = ax.pcolormesh(x_map, y_map, rewardmap_img, shading="auto", cmap="coolwarm_r")
        fig.colorbar(pcm, ax=ax, fraction=0.046, label="SDF (m)")

        # draw navitopo
        for pts in navitopo_pts: ax.plot(pts[:, 0], pts[:, 1], color='cyan')
        # draw Trajectory
        ax.plot(free_x, free_y, '-o', color='red', markersize=2)
        ax.plot(guidanced_x, guidanced_y, '-o', color='green', markersize=2)
        # Finally, draw Gradient
        for idx, grad in enumerate(grad_list):
            dx, dy = grad[i, :, 0], grad[i, :, 1]
            ax.quiver(guidanced_x, guidanced_y, dx, dy, angles='xy', scale_units='xy', scale=0.1, color=colors[idx], width=0.003)

        ax.set_title(f"Traj {i} | RewardMap")
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)

        # =======================
        # Column 5: Distance
        # =======================
        ax = axes[i, 5]

        # draw Distance
        distance = costmap.copy()
        distance = distance[0] if distance.ndim == 3 else distance
        H, W = distance.shape
        distance[(safe_range - distance)<=0] = 0
        ii, jj = np.meshgrid(np.arange(H), np.arange(W),indexing="ij")
        real_pts = img2real(np.stack([jj, ii], axis=-1), resolution=0.4, i0=50, j0=0)       # (H, W, 2)
        x_map, y_map = real_pts[..., 0], real_pts[..., 1]                                   # (H, W)
        pcm = ax.pcolormesh(x_map, y_map, distance, shading="auto", cmap="coolwarm_r")
        fig.colorbar(pcm, ax=ax, fraction=0.046, label="Danger (m)")

        # draw navitopo
        for pts in navitopo_pts: ax.plot(pts[:, 0], pts[:, 1], color='cyan')
        # draw Trajectory
        ax.plot(free_x, free_y, '-o', color='red', markersize=2)
        ax.plot(guidanced_x, guidanced_y, '-o', color='green', markersize=2)
        # Finally, draw Gradient
        for idx, grad in enumerate(grad_list):
            dx, dy = grad[i, :, 0], grad[i, :, 1]
            ax.quiver(guidanced_x, guidanced_y, dx, dy, angles='xy', scale_units='xy', scale=0.1, color=colors[idx], width=0.003)

        ax.set_title(f"Traj {i} | Danger")
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def L2_guidance_func(traj: torch.Tensor, target: torch.Tensor, target_mask=None):
    """
    traj: (S, 1, T, 3)
    target: (1, N, P, 2)
    """
    # TODO: target_mask & closest
    traj = traj.squeeze(1)[..., :2]         # (S, T, 2)
    target = target[:, 0, :, :]             # (1, P, 2): only first navi

    traj = traj.unsqueeze(2)                # (S, T, 1, 2)
    target = target.unsqueeze(1)            # (1, 1, P, 2)

    dist2 = torch.sum((traj - target) ** 2, dim=-1)  # (S, T, P)
    min_dist2, _ = dist2.min(dim=-1)                  # (S, T)
    return min_dist2.mean()

def soft_L2_guidance_func(traj, target, target_mask=None, beta=10.0):
    """
    traj: (S, 1, T, 3)
    target: (1, N, P, 2)
    target_mask: (1, N,)       True = invalid, False = valid
    """
    # TODO: target_mask & closest
    # only use first valid navi
    target_mask = target_mask.squeeze(1).bool()
    valid_idx = (~target_mask).nonzero(as_tuple=True)[0]
    idx = valid_idx[0].item() if valid_idx.numel() else None

    if idx is None:
        return traj.new_tensor(0.0)

    traj = traj.squeeze(1)[..., :2]         # (S, T, 2)
    target = target[:, idx, :, :]           # (1, P, 2): only use first valid navi

    traj = traj.unsqueeze(2)                # (S, T, 1, 2)
    target = target.unsqueeze(1)            # (1, 1, P, 2)

    dist2 = torch.sum((traj - target) ** 2, dim=-1)  # (S,T,P)
    weight = torch.softmax(-beta * dist2, dim=-1)
    loss = (weight * dist2).sum(dim=-1)

    return loss.mean()

# def cost_func_speed(delta_traj: torch.Tensor, speed_guidance_params: dict):
#     scale = float(speed_guidance_params.get("scale", 0.0))
#     planning_interval = float(speed_guidance_params.get("planning_interval", 0.2))
#     target_v = float(speed_guidance_params.get("target_v", 5.0))
#     eps = 1e-6

#     anchor_v = torch.norm(delta_traj[:, :, :, :2], dim=-1) / (planning_interval + eps)
#     # import pdb
#     # pdb.set_trace()
#     # print("anchor_v1", anchor_v)
#     reward = (anchor_v - target_v)**2
#     # import pdb
#     # pdb.set_trace()
#     # reward.sum().backward(retain_graph=False) 
#     # guidance_grad = x.grad
#     # return scale * reward.sum()
#     return scale * reward.mean()
