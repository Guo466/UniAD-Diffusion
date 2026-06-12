import torch
import math

class EgoHeadingAugmentation():
    def __init__(self, config):
        self.max_angle = config.get("max_angle", 20)
        self.apply_prob = config.get("apply_probaility", 0.5)

    def _get_rotation_matrix(self, angle_rad, device, dtype):
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)
        row1 = torch.stack([c, -s])
        row2 = torch.stack([s,  c])
        R = torch.stack([row1, row2])   # -> shape [2,2]
        return R

    def _get_ego_speed(self, model_input):
        return model_input["ego_curr_status"][0]

    def _apply_aug(self, model_input, scene=None, train_mode=True):
        """
        单样本版本（用于 Dataset.__getitem__）
        所有元素绕自车旋转
        """
        if not train_mode:
            return model_input, scene

        required_keys = ["ego_curr_status", "agent_status", "trainFlag"]
        if not all(k in model_input for k in required_keys):
            return model_input, scene

        if not bool(model_input["trainFlag"]):
            return model_input, scene

        # probability
        if torch.rand(1).item() >= self.apply_prob:
            return model_input, scene

        device = model_input["ego_curr_status"].device
        dtype = model_input["ego_curr_status"].dtype

        # speed
        speed = self._get_ego_speed(model_input)

        # theta limit
        max_theta = torch.tensor(
            math.pi * self.max_angle / 180,
            device=device,
            dtype=dtype
        )
        # zhihui 速度越大绕动越小
        # 但是：1. 目前主要是针对path用，其实跟速度无关。2. 要解决的case场景往往都是高速的，至少是落在上边公式范围外的
        # 不过：1. traj train的时候也用旋转的么？
        # 所以：1. 暂时先用全部的角度，暂时train path
        # theta = torch.min(1.225 / (speed + 1e-6) / 2, max_theta)
        theta = max_theta


        # sample angle
        angle = ((torch.rand(1, device=device, dtype=dtype) * 2 * theta) - theta).squeeze()

        # rotation matrix
        R = self._get_rotation_matrix(angle, device, dtype)

        # ---- agent ----
        # [pos, v, yaw, score]
        model_input["agent_status"][..., :2] = torch.matmul(
            model_input["agent_status"][..., :2], R
        )
        model_input["agent_status"][..., 2:4] = torch.matmul(
            model_input["agent_status"][..., 2:4], R
        )
        model_input["agent_status"][..., 4] -= angle

        # ---- map ----
        # [pos]
        if "laneline_pts" in model_input:
            model_input["laneline_pts"][..., :2] = torch.matmul(
                model_input["laneline_pts"][..., :2], R
            )
        if "navitopo_pts" in model_input:
            model_input["navitopo_pts"][..., :2] = torch.matmul(
                model_input["navitopo_pts"][..., :2], R
            )
        if "route_pts" in model_input:
            model_input["route_pts"][..., :2] = torch.matmul(
                model_input["route_pts"][..., :2], R
            )

        # ---- ego future ----
        # 这里拿的v是tensor，要转。cur拿的v是标量，不用转
        for key in [
            "ego_future_status",
            "ego_future_status_fixed",
            "ego_future_status_fixed_full",
        ]:
            if key in model_input:
                model_input[key][..., :2] = torch.matmul(
                    model_input[key][..., :2], R
                )
                model_input[key][..., 2:4] = torch.matmul(
                    model_input[key][..., 2:4], R
                )
                model_input[key][..., 4] -= angle

        # ---- ego history ----
        if "ego_status" in model_input:
            model_input["ego_status"][..., :2] = torch.matmul(
                model_input["ego_status"][..., :2], R
            )
            model_input["ego_status"][..., 2:4] = torch.matmul(
                model_input["ego_status"][..., 2:4], R
            )
            model_input["ego_status"][..., 4] -= angle
        # ---- ego cur ----
        # 这里只有traj会用到。xy都是0不用管，yaw是特意置1不用管，v是标量不用管
        # ....

        # ---- occ ----
        if "occ_map" in model_input:
            occ_map = model_input["occ_map"]
            H, W = occ_map.shape
            x_range = (-10, 100)
            y_range = (-24, 24)
            x_coords = torch.linspace(x_range[0], x_range[1], W, device=device, dtype=dtype)
            y_coords = torch.linspace(y_range[0], y_range[1], H, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
            coords_rot = coords @ R.T
            xx_rot, yy_rot = coords_rot[:, 0], coords_rot[:, 1]
            x_idx = ((xx_rot - x_range[0]) / (x_range[1] - x_range[0]) * W).long().clamp(0, W-1)
            y_idx = ((yy_rot - y_range[0]) / (y_range[1] - y_range[0]) * H).long().clamp(0, H-1)
            occ_map_rot = torch.zeros_like(occ_map)
            occ_map_flat = occ_map.reshape(-1)
            occ_map_rot[y_idx, x_idx] = occ_map_flat[torch.arange(H*W)]
            model_input["occ_map"] = occ_map_rot

        if "occ_polygons_pts" in model_input and "occ_polygons_attrs" in model_input:
            polygons = model_input["occ_polygons_pts"]           # (N, P, 2)
            polygons_attrs = model_input["occ_polygons_attrs"]  # (N, 5)

            # 构建旋转矩阵，用 -angle
            c = torch.cos(-angle)
            s = torch.sin(-angle)
            R_poly = torch.tensor([[c, -s], [s, c]], dtype=polygons.dtype, device=polygons.device)

            # polygon points
            polygons_rot = polygons @ R_poly.T

            # polygon centers
            cycx = polygons_attrs[..., :2]  # [cy, cx]
            centers_xy = torch.stack([cycx[..., 1], cycx[..., 0]], dim=-1)  # 转成 [x, y]
            centers_xy_rot = centers_xy @ R_poly.T
            attrs_rot = polygons_attrs.clone()
            attrs_rot[..., 0] = centers_xy_rot[..., 1]  # cy
            attrs_rot[..., 1] = centers_xy_rot[..., 0]  # cx
            # 注意，旋转用的-angle
            attrs_rot[..., 4] -= -angle                  # theta_pix = theta_pix - (-angle) = theta_pix + angle

            model_input["occ_polygons_pts"] = polygons_rot
            model_input["occ_polygons_attrs"] = attrs_rot
        
        return model_input, scene