import torch
import numpy as np
import bezier
import torch.nn.functional as F

map_ego_light_yflip = {
    2 : 3,  # left -> right
    3 : 2,  # right -> left
    1 : 1
}
def safety_check(
    ego_position,
    ego_heading,
    agents_position,
    agents_heading,
    agents_shape,
):
    
    if len(agents_position) == 0:
        return True
    
    _rear_to_cog = compute_rear_to_cog()
    ego_center = (
        ego_position
        + np.stack([np.cos(ego_heading), np.sin(ego_heading)], axis=-1)*
        _rear_to_cog
    )
    ego_state = torch.from_numpy(
        np.concatenate([ego_center, np.array([ego_heading])], axis=-1)
    ).unsqueeze(0)
    objects_state = torch.from_numpy(
        np.concatenate([agents_position, agents_heading[..., None]], axis=-1)
    ).unsqueeze(0)
    
    collisions = collision_check(
        ego_state=ego_state,
        objects=objects_state,
        objects_width=agents_shape[:, 1].unsqueeze(0),
        objects_length=agents_shape[:, 0].unsqueeze(0),
    )
    return not collisions.any()

def compute_rear_to_cog():
    front_length = 4.049
    rear_length = 1.127
    length = front_length + rear_length
    return length / 2.0 - rear_length

def random_offset_generator(low, high):
    op = np.random.default_rng()
    # x, y, yaw, vx, angular_vel
    return op.uniform(low, high)

def collision_check(ego_state,objects,objects_width,objects_length):
    bs, N = objects.shape[:2]

    # rotate object to ego's local frame
    cos, sin = torch.cos(ego_state[:, 2]), torch.sin(ego_state[:, 2])
    rotate_mat = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(bs, 2, 2)

    rotated_objects = objects.clone()
    rotated_objects[..., :2] = torch.matmul(
        rotated_objects[..., :2] - ego_state[:, :2].unsqueeze(1), rotate_mat
    )
    rotated_objects[..., 2] -= ego_state[..., 2].unsqueeze(1)

    # [bs, N, 4, 2], [bs, N, 2], [bs, N, 2]
    
    object_corners, axis1, axis2 = build_bbox_from_center(
        rotated_objects[..., :2],
        rotated_objects[..., 2],
        objects_width,
        objects_length,
    )
    _sdc_normalized_corners = comput_sdc_normalized_corners()
    ego_corners = _sdc_normalized_corners.reshape(1, 1, 4, 2).repeat(
        bs, N, 1, 1
    )  # [bs, N, 4, 2]

    all_corners = torch.cat(
        [object_corners, ego_corners], dim=-2
    )  # [bs, N, 8, 2]
    
    x_projection = object_corners[..., 0]
    y_projection = object_corners[..., 1]
    axis1_projection = torch.matmul(all_corners, axis1.unsqueeze(-1)).squeeze(-1)
    axis2_projection = torch.matmul(all_corners, axis2.unsqueeze(-1)).squeeze(-1)
    _sdc_half_length,_sdc_half_width = compute_sdc_half_shape()
    x_separated = (x_projection.max(-1)[0] < -_sdc_half_length) | (
        x_projection.min(-1)[0] > _sdc_half_length
    )
    y_separated = (y_projection.max(-1)[0] < -_sdc_half_width) | (
        y_projection.min(-1)[0] > _sdc_half_width
    )
    axis1_separated = (
        axis1_projection[..., :4].max(-1)[0] < axis1_projection[..., 4:].min(-1)[0]
    ) | (
        axis1_projection[..., :4].min(-1)[0] > axis1_projection[..., 4:].max(-1)[0]
    )
    axis2_separated = (
        axis2_projection[..., :4].max(-1)[0] < axis2_projection[..., 4:].min(-1)[0]
    ) | (
        axis2_projection[..., :4].min(-1)[0] > axis2_projection[..., 4:].max(-1)[0]
    )

    collision = ~(x_separated | y_separated | axis1_separated | axis2_separated)

    return collision

def build_bbox_from_center(center, heading, width, length):
    """
    params:
        center: [bs, N, (x, y)]
        heading: [bs, N]
        width: [bs, N]
        length: [bs, N]
    return:
        corners: [bs, 4, (x, y)]
        heading_vec, tanh_vec: [bs, 2]
    """
    cos = torch.cos(heading)
    sin = torch.sin(heading)

    heading_vec = torch.stack([cos, sin], dim=-1) * length.unsqueeze(-1) / 2
    tanh_vec = torch.stack([-sin, cos], dim=-1) * width.unsqueeze(-1) / 2

    corners = torch.stack(
        [
            center + heading_vec + tanh_vec,
            center - heading_vec + tanh_vec,
            center - heading_vec - tanh_vec,
            center + heading_vec - tanh_vec,
        ],
        dim=-2,
    )

    return corners, heading_vec, tanh_vec

def comput_sdc_normalized_corners():
    # length = 4.049+1.127
    # width = 1.1485 * 2.0
    length = 5
    width = 2
    return torch.stack(
            [
                torch.tensor([length / 2, width / 2]),
                torch.tensor([length / 2, -width / 2]),
                torch.tensor([-length / 2, -width / 2]),
                torch.tensor([-length / 2, width / 2]),
            ],
            dim=0,
        )

def compute_sdc_half_shape():
    length = 4.049+1.127
    width = 1.1485 * 2.0
    return length/2,width/2 

# rotate_occ_grid
def rotate_occupancy_grid(occ_map, rotate_mat, center=None, threshold=False):
    """
    旋转占据图（支持绕任意点旋转）

    Args:
        occ_map: torch.Tensor, shape (H, W) 或 (1, 1, H, W)
        rotate_mat: torch.Tensor, shape (2, 2)
        center: (cx, cy) 旋转中心，像素坐标，默认为图像中心
        threshold: 是否将结果二值化 (True -> 0/1)
    Returns:
        torch.Tensor, same shape as occ_map
    """
    if occ_map.ndim == 2:
        occ_map = occ_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    H, W = occ_map.shape[-2:]
    device = occ_map.device
    dtype = occ_map.dtype

    # 默认绕图像中心旋转
    if center is None:
        center = (W / 2.0, H / 2.0)

    # 生成归一化坐标 [-1, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=dtype),
        indexing='ij'
    )
    grid = torch.stack([xx, yy], dim=-1)  # (H, W, 2)

    # 平移到中心，旋转，再平移回去
    center_tensor = torch.tensor(center, device=device, dtype=dtype)
    grid_centered = grid - center_tensor
    grid_rot = torch.matmul(grid_centered, rotate_mat)
    grid_rot += center_tensor

    # 转换为 grid_sample 的 [-1, 1] 坐标系
    grid_rot[..., 0] = 2.0 * grid_rot[..., 0] / (W - 1) - 1.0
    grid_rot[..., 1] = 2.0 * grid_rot[..., 1] / (H - 1) - 1.0
    grid_rot = grid_rot.unsqueeze(0)  # (1, H, W, 2)

    rotated = F.grid_sample(
        occ_map, grid_rot, mode='nearest', align_corners=True, padding_mode='zeros'
    )

    rotated = rotated.squeeze()

    if threshold:
        rotated = (rotated > 0.5).float()
    rotated[rotated == 0] = 1
    return rotated


def renormalization(res, yaw_change, keep_nums=3, control_len=6):
    center_angle = float(yaw_change)
    rotate_mat = torch.from_numpy(np.array(
        [
            [np.cos(center_angle), -np.sin(center_angle)],
            [np.sin(center_angle), np.cos(center_angle)],
        ],
        dtype=np.float32,
    ))
    res["agent_status"][:,:,:2] = torch.matmul(res["agent_status"][:,:,:2], rotate_mat)   # pos
    res["agent_status"][:,:,2:4] = torch.matmul(res["agent_status"][:,:,2:4], rotate_mat)   # v
    res["agent_status"][:,:,4] -= center_angle

    res["laneline_pts"] = torch.matmul(res["laneline_pts"], rotate_mat)
    res['navitopo_pts']= torch.matmul(res["navitopo_pts"], rotate_mat)
    center_ego = (60,25) #occ grid 旋转中心
    res["occ_map"] = rotate_occupancy_grid(res["occ_map"], rotate_mat,center_ego, threshold=False)
    target = res["ego_future_status"] # [x, y, vx, vy, yaw]
    target[:,:2] = torch.matmul(target[:,:2], rotate_mat)   # pos
    target[:,2:4] = torch.matmul(target[:,2:4], rotate_mat)   # v
    target[:,4] -= center_angle
    res["ego_future_status"] = target

    target = res["ego_future_status_fixed"]
    target[:,:2] = torch.matmul(target[:,:2], rotate_mat)   # pos
    target[:,2:4] = torch.matmul(target[:,2:4], rotate_mat)   # v
    target[:,4] -= center_angle
    res["ego_future_status_fixed"] = target

    return res

def renormalization_pos(res, pos_change, control_len=6):
    # tmp = torch.arange(19, 0, -1) / 20 * pos_change[1]      # 在2s之内，回到原来的gt

    pos_change = torch.from_numpy(pos_change[:2].copy()).float()
    res["agent_status"][:,:,:2] -= pos_change

    res["laneline_pts"] = res["laneline_pts"] - pos_change
    res["navitopo_pts"] = res["navitopo_pts"] - pos_change

    target = res["ego_future_status"] # [x, y, vx, vy, yaw]
    target[:,:2] = torch.matmul(target[:,:2], rotate_mat)   # pos
    target[:,2:4] = torch.matmul(target[:,2:4], rotate_mat)   # v
    target[:,4] -= center_angle
    res["ego_future_status"] = target

    target = res["ego_future_status_fixed"]
    target[:,:2] = torch.matmul(target[:,:2], rotate_mat)   # pos
    target[:,2:4] = torch.matmul(target[:,2:4], rotate_mat)   # v
    target[:,4] -= center_angle
    res["ego_future_status_fixed"] = target

    return res

def _normalization(res, new_state, first_time=False, radius=None):
    cur_state = new_state
    center_xy, center_angle = torch.from_numpy(cur_state[:2].copy()).type(torch.float32), cur_state[2].copy()
    rotate_mat = torch.from_numpy(np.array(
        [
            [np.cos(center_angle), -np.sin(center_angle)],
            [np.sin(center_angle), np.cos(center_angle)],
        ],
        dtype=np.float32,
    ))
    res["agents"][:,:2] = torch.matmul(res["agents"][:,:2] - center_xy, rotate_mat)
    res["agents"][:,5:7] = torch.matmul(res["agents"][:,5:7], rotate_mat)
    res["agents"][:,7] -= center_angle
    res["laneline_pts"] = torch.matmul(res["laneline_pts"] - center_xy, rotate_mat)
    target = res["ego_future_status"]
    target[:, :2] = torch.matmul(target[:, :2] - center_xy, rotate_mat)
    target[:, 2] -= center_angle
    res["ego_future_status"] = target

    # TODO 添加固定距离的plan gt @李中柱

    # if first_time:
    #     point_position = res["laneline_pts"]
    #     x_max, x_min = radius, -radius
    #     y_max, y_min = radius, -radius
    #     valid_mask = (
    #         (point_position[:, 0, :, 0] < x_max)
    #         & (point_position[:, 0, :, 0] > x_min)
    #         & (point_position[:, 0, :, 1] < y_max)
    #         & (point_position[:, 0, :, 1] > y_min)
    #     )
    #     valid_polygon = valid_mask.any(-1)
    #     res["map"]["valid_mask"] = valid_mask

    return res

def v_perturbation(res, min_v, max_v):
    change = random_offset_generator(low=[min_v], high=[max_v])   # yaw
    res['ego_curr_status'][0] += change[0]
    res['ego_curr_status'][0] = max(0, res['ego_curr_status'][0])   # 避免出现负速度
    return res

def yaw_rate_perturbation(res, min_yaw_rate, max_yaw_rate):
    change = random_offset_generator(low=[min_yaw_rate], high=[max_yaw_rate])   # yaw
    res['ego_curr_status'][1] += change[0]
    return res

def yaw_perturbation(res, min_yaw=-0.1, max_yaw=0.1):        
    # yaw_change = random_offset_generator(low=[-0.1], high=[0.1])   # yaw
    yaw_change = random_offset_generator(low=[min_yaw], high=[max_yaw])   # yaw
    # print('yaw_change: ', yaw_change)
    res = renormalization(res, yaw_change, keep_nums=0)
    return res

def pos_perturbation(res, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5,  normalize_gt_in_2s=False, hist_steps=20):        
    pos_change = random_offset_generator(low=[x_min, y_min], high=[x_max, y_max]) # x, y
    # print('pos_change: ', pos_change)

    consider_num = (res["agent_time_mask"][:,hist_steps]).sum()
    position = res["agent_status"][:consider_num,hist_steps,:2]
    heading = res["agent_status"][:consider_num,hist_steps,4]
    shape = res["agent_attrs"][:consider_num,:2]    # length, width

    num_tries, scale = 0, 1.0
    while num_tries < 5:
        pos_change = pos_change * scale

        if safety_check(
            ego_position=pos_change,
            ego_heading=0,
            agents_position=position,
            agents_heading=heading,
            agents_shape=shape,
        ):
            break
        num_tries += 1
        scale *= 0.5
    # if not safety_check(
    #     ego_position=pos_change,
    #     ego_heading=0,
    #     agents_position=position,
    #     agents_heading=heading,
    #     agents_shape=shape,
    # ):
    #     return res
    res = renormalization_pos(res, pos_change)
    return res

def y_flip(res):
    # agents
    res['agent_status'][:,:,1] *= -1            # y
    res['agent_status'][:,:,3] *= -1            # vy
    res['agent_status'][:,:,4] *= -1            # yaw
    
    # laneline
    res['laneline_pts'][:,:,1] *= -1    # point pos y

    # ego current status
    res['ego_curr_status'][1] *= -1     # yaw rate
    res['ego_curr_status'][3] = map_ego_light_yflip[int(res['ego_curr_status'][3])]  # 左转向灯变右转向灯

    # ego future trajectory
    res['ego_future_status'][:, 1] *= -1    # y
    res['ego_future_status'][:, 3] *= -1    # vy
    res['ego_future_status'][:, 4] *= -1    # yaw
    res['ego_future_status_fixed'][:, 1] *= -1    # y
    res['ego_future_status_fixed'][:, 3] *= -1    # vy
    res['ego_future_status_fixed'][:, 4] *= -1    # yaw
    return res