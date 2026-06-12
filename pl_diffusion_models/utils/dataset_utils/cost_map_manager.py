import numpy as np
import torch
import cv2
import math
from scipy import ndimage
import torch
import torch.nn.functional as F


class CostMapManager:
    def __init__(
        self,
        length,   # 0 ~ 80m
        width,    # -20 ~ 20m
        resolution,
        hist_steps,
        future_steps,
        use_agent,
        use_laneline,
        laneline_shift_dist,
    ) -> None:
        self.length = length
        self.width = width
        self.resolution = resolution
        self.hist_steps = hist_steps
        self.future_steps = future_steps
        self.use_agent = use_agent
        self.use_laneline = use_laneline
        self.laneline_shift_dist = laneline_shift_dist
        self.range = [0, length*resolution, -width/2.0*resolution, width/2.0*resolution] # x_min, x_max, y_min, y_max, e.g. [0, 50, -10, 10]

    def compute_edt_frame(i: int, mask: np.ndarray):
        """
        计算单个时间步的 EDT 距离场。
        """
        # 距离变换 1: 距离障碍物距离 (非零到零)
        dist = ndimage.distance_transform_edt(mask)

        # 距离变换 2: 距离可通行区域距离 (零到非零)
        inv_dist = ndimage.distance_transform_edt(1 - mask)

        return i, dist, inv_dist

    def build_collision_cost_maps(self, ego_future_status, agent_attrs, agent_status, agents_time_mask,
                                  laneline_pts, laneline_attrs, laneline_mask, occ_map, use_agent=True, use_agent_fixed=False, width=2, print_flag=False):
        # initialization
        drivable_area_mask = np.ones((self.future_steps, self.width, self.length), dtype=np.uint8)     # (50, 250, 100)，1是可行驶区域，0是不可行驶区域
        distance = np.full((self.future_steps, self.width, self.length), 100.)     # (50, 250, 100)
        inv_distance = np.zeros((self.future_steps, self.width, self.length))     # (50, 250, 100)
        laneline_offset = self.laneline_shift_dist

        # laneline
        # laneline_pts, laneline_attrs, laneline_mask: [50, 20, 2], [50, 4], [50]
        if self.use_laneline:
            # 提取 shape
            num_lines = laneline_mask.shape[0]
            num_pts = laneline_pts.shape[1]

            # 转为 numpy
            laneline_mask_np = laneline_mask.cpu().numpy() if isinstance(laneline_mask, torch.Tensor) else laneline_mask
            laneline_pts_np = laneline_pts.cpu().numpy() if isinstance(laneline_pts, torch.Tensor) else laneline_pts
            laneline_attrs_np = laneline_attrs.cpu().numpy() if isinstance(laneline_attrs, torch.Tensor) else laneline_attrs
            ego_future_np = ego_future_status.cpu().numpy() if isinstance(ego_future_status, torch.Tensor) else ego_future_status

            # 提前计算 ego_future 的 2D 位置矩阵
            ego_xy = ego_future_np[:, :2]

            for i in range(num_lines):
                if laneline_mask_np[i]:
                    # 遇到 padding，跳出
                    break

                pts = laneline_pts_np[i]  # [num_pts, 2]

                # 判断是否属于车前方路沿 / 黄实线 / 实线
                flag = False
                color, laneline_type, laneline_style = laneline_attrs_np[i]
                laneline_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) if len(pts) > 1 else 0.0
                if (laneline_type == 1 or color == 2) and (np.max(pts[:, 0]) > 0) and (laneline_len > 5.0):    # 自车前方的路沿, 黄线，实线；累计长度>5m
                    flag = True
                if flag:
                    # 计算 ego_future 与 laneline 所有点的欧氏距离矩阵
                    diff = ego_xy[:, None, :] - pts[None, :, :]  # [T, N, 2]
                    distances = np.linalg.norm(diff, axis=-1)    # [T, N]

                    # 最小距离与对应索引
                    min_dist = np.min(distances)
                    if min_dist < 1 or min_dist > 20:
                        continue

                    tmp = np.argmin(distances)
                    row = tmp // distances.shape[1]
                    col = tmp % distances.shape[1]
                    if col == num_pts - 1:
                        col -= 1

                    pt1 = pts[col]
                    pt2 = pts[col + 1]

                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    length = np.hypot(dx, dy)
                    if length < 1e-6:
                        continue

                    cos_theta = dx / length
                    sin_theta = dy / length

                    # 计算两侧偏移点，判断 factor 方向
                    pt3 = np.array([pt1[0] + width * sin_theta, pt1[1] - width * cos_theta])
                    pt4 = np.array([pt1[0] - width * sin_theta, pt1[1] + width * cos_theta])

                    ego_pt = ego_xy[row]
                    dist3 = np.sum((pt3 - ego_pt) ** 2)
                    dist4 = np.sum((pt4 - ego_pt) ** 2)
                    factor = 1 if dist3 > dist4 else -1

                    # 对当前 line 的每个相邻点画矩形
                    for j in range(1, num_pts):
                        pt1 = pts[j - 1]
                        pt2 = pts[j]
                        dx = pt2[0] - pt1[0]
                        dy = pt2[1] - pt1[1]
                        length = np.hypot(dx, dy)
                        if length < 1e-6:
                            continue

                        cos_theta = dx / length
                        sin_theta = dy / length

                        corners = np.array([
                            [pt1[0] + factor * (laneline_offset) * sin_theta, pt1[1] - factor * (laneline_offset) * cos_theta],
                            [pt2[0] + factor * (laneline_offset) * sin_theta, pt2[1] - factor * (laneline_offset) * cos_theta],
                            [pt2[0] + factor * (laneline_offset + width) * sin_theta, pt2[1] - factor * (laneline_offset + width) * cos_theta],
                            [pt1[0] + factor * (laneline_offset + width) * sin_theta, pt1[1] - factor * (laneline_offset + width) * cos_theta],
                        ])
                        corners = np.vstack((corners, corners[0]))  # 闭合

                        self.fill_convex_polygon(drivable_area_mask[0], corners, value=0)

        # ego, only for debug
        # for i in range(self.future_steps):
        #     if i % 10 == 0:
        #         corners = box_center_to_corners(ego_future_status[i,0], ego_future_status[i,1], 5, 2, ego_future_status[i,4])
        #         corners = np.vstack((corners, corners[0]))
        #         self.fill_convex_polygon(drivable_area_mask[0], corners, value=0)
        # for i in range(1, self.future_steps):
        #     drivable_area_mask[i] = drivable_area_mask[0]

        # occ_map 进costmap
        # costmap: 1) length: 250(0 ~ 50m); 2) width: 100(-10 ~ 10m); 3) resolution: 0.2
        # occ: x: 275(-10~100) y: 120(-24~24) res: 0.4
        occ_map[occ_map == 1] = 1 # 1是可行驶区域，0是不可行驶区域
        occ_map[occ_map > 1] = 0 
        occ_region = occ_map[35:85, 25:150]
        downsampled_occ = torch.flip(occ_region, dims=[0]).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1).cpu().numpy()
        drivable_area_mask[0][downsampled_occ == 0] = 0

        # drivable_area_mask[1:] = drivable_area_mask[0]    # 这种写法，经过测试：在gpu上很快；但是在cpu上很慢
        for i in range(1, self.future_steps):
            drivable_area_mask[i] = drivable_area_mask[0]

        # agent
        flag_agent_in_costmap = False
        if use_agent:
            agent_shape_np = agent_attrs[:, :2].cpu().numpy() if isinstance(agent_attrs, torch.Tensor) else agent_attrs[:, :2]
            agent_cls_np = agent_attrs[:, 2].cpu().numpy() if isinstance(agent_attrs, torch.Tensor) else agent_attrs[:, 2]
            agent_future_traj_np = agent_status[:, self.hist_steps+1:, [0, 1, 4]].cpu().numpy() if isinstance(agent_status, torch.Tensor) else agent_status[:, self.hist_steps+1:, [0,1,4]]
            agent_future_mask_np = agents_time_mask[:, self.hist_steps+1:].cpu().numpy() if isinstance(agents_time_mask, torch.Tensor) else agents_time_mask[:, self.hist_steps+1:]

            for t in range(self.future_steps):
                valid_mask = agent_future_mask_np[:, t]

                valid_agents = agent_future_traj_np[:, t, :3][valid_mask]  # x, y, yaw
                valid_shape = agent_shape_np[valid_mask]
                valid_cls = agent_cls_np[valid_mask]
                if valid_agents.shape[0] == 0:
                    continue

                n_agents = valid_agents.shape[0]
                for i_agent in range(n_agents):
                    shape_i = valid_shape[i_agent].copy()
                    if valid_cls[i_agent] in [1, 3]:
                        shape_i += 0.5

                    corners = box_center_to_corners(valid_agents[i_agent, 0], valid_agents[i_agent, 1],
                                                     shape_i[0], shape_i[1], valid_agents[i_agent, 2])

                    # 边界检查
                    if np.max(corners[:,0]) < 0 or np.min(corners[:,0]) > self.resolution * self.length or \
                       np.max(corners[:,1]) < -self.resolution * self.width / 2 or np.min(corners[:,1]) > self.resolution * self.width / 2:
                        continue

                    # 填充多边形
                    corners = np.vstack((corners, corners[0]))
                    self.fill_convex_polygon(drivable_area_mask[t], corners, value=0)
                    flag_agent_in_costmap = True

        if use_agent_fixed:
            # 转成 NumPy，保证索引统一
            agent_shape_np = agent_attrs[:, :2].cpu().numpy() if isinstance(agent_attrs, torch.Tensor) else agent_attrs[:, :2]
            agent_cls_np = agent_attrs[:, 2].cpu().numpy() if isinstance(agent_attrs, torch.Tensor) else agent_attrs[:, 2]
            agent_future_traj_np = agent_status[:, self.hist_steps+1:, [0,1,4]].cpu().numpy() if isinstance(agent_status, torch.Tensor) else agent_status[:, self.hist_steps+1:, [0,1,4]]
            agent_future_mask_np = agents_time_mask[:, self.hist_steps+1:].cpu().numpy() if isinstance(agents_time_mask, torch.Tensor) else agents_time_mask[:, self.hist_steps+1:]
            agent_curr_pos_np = agent_status[:, self.hist_steps, :2].cpu().numpy() if isinstance(agent_status, torch.Tensor) else agent_status[:, self.hist_steps, :2]
            keep_agent_mask_np = ~ignore_agent_costmap(agent_curr_pos_np)

            for i_agent in range(agent_future_traj_np.shape[0]):
                # 原版逻辑条件
                if not agents_time_mask[i_agent, self.hist_steps]:
                    continue
                if not agent_future_mask_np[i_agent, :].any():
                    continue
                if not keep_agent_mask_np[i_agent]:
                    continue

                # 筛选未来有效位置
                future_valid_pos = agent_future_traj_np[i_agent, :, :2][agent_future_mask_np[i_agent, :]]

                # ego_future_status 可能是 Tensor，需要转 NumPy
                ego_future_np = ego_future_status[:, :2].cpu().numpy() if isinstance(ego_future_status, torch.Tensor) else ego_future_status[:, :2]
                distances = np.linalg.norm(ego_future_np[:, None, :] - future_valid_pos[None, :, :], axis=-1)
                min_distance = np.min(distances)

                # 最大速度 & 最大位移
                agent_vel_np = agent_status[:, self.hist_steps+1:, 2:4].cpu().numpy() if isinstance(agent_status, torch.Tensor) else agent_status[:, self.hist_steps+1:, 2:4]
                max_vel = np.max(np.abs(agent_vel_np[i_agent][agent_future_mask_np[i_agent, :]]))
                max_future_disp = np.max(np.abs(future_valid_pos - agent_curr_pos_np[i_agent, :]))

                if min_distance <= 2.5 or max_vel >= 0.3 or max_future_disp >= 1:
                    continue

                # 构建 valid agent
                valid_agent = agent_status[i_agent, self.hist_steps, [0,1,4]].cpu().numpy() if isinstance(agent_status, torch.Tensor) else agent_status[i_agent, self.hist_steps, [0,1,4]]
                valid_shape = agent_shape_np[i_agent].copy()
                if agent_cls_np[i_agent] in [1, 3]:
                    valid_shape += 0.5

                # corners
                corners = box_center_to_corners(valid_agent[0], valid_agent[1], valid_shape[0], valid_shape[1], valid_agent[2])
                if np.max(corners[:,0]) < 0 or np.min(corners[:,0]) > self.resolution * self.length or \
                   np.max(corners[:,1]) < -self.resolution * self.width / 2 or np.min(corners[:,1]) > self.resolution * self.width / 2:
                    continue

                corners = np.vstack((corners, corners[0]))
                self.fill_convex_polygon(drivable_area_mask[0], corners, value=0)

        
        if flag_agent_in_costmap:
            # 有agent存在，每一时刻的costmap都可能不相同，需要逐帧生成
            for i in range(self.future_steps):
                # cv2.imwrite(f'tmp_{i}.png', drivable_area_mask[i]*200)
                distance[i] = ndimage.distance_transform_edt(drivable_area_mask[i])     # 二值图像中每个非零像素到最近的零像素的欧几里得距离，越大越好
                inv_distance[i] = ndimage.distance_transform_edt(1 - drivable_area_mask[i])     # 二值图像中每个零像素到最近的非零像素的欧几里得距离，越小越好
        else:
            # 没有agent存在，每一时刻的costmap都相同，生成第一帧的即可
            # cv2.imwrite(f'tmp.png', drivable_area_mask[0]*200)
            distance[0] = ndimage.distance_transform_edt(drivable_area_mask[0])   # 二值图像中每个非零像素到最近的零像素的欧几里得距离，越大越好
            inv_distance[0] = ndimage.distance_transform_edt(1 - drivable_area_mask[0])  # 二值图像中每个零像素到最近的非零像素的欧几里得距离，越小越好
            for i in range(1, self.future_steps):
                distance[i] = distance[0]
                inv_distance[i] = inv_distance[0]
        drivable_area_sdf = (distance - inv_distance) * self.resolution
        # import pdb; pdb.set_trace()
        return torch.tensor(drivable_area_sdf, dtype=torch.float32)


    def fill_convex_polygon(self, mask, polygon, value=0):
        polygon = self.bev_to_pixel(polygon)
        cv2.fillConvexPoly(mask, np.floor(polygon).astype(np.int32), value)
        # cv2.imwrite('tmp.png', mask*200)
        '''
        NOTE
        e.g. mask.shape: (100, 250)
        polygon = np.array([[200,50], [203,50], [203,53], [200,53], [200,50]])
        np.where(mask == 0):    array([50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53, 53]), 
                                array([200, 201, 202, 203, 200, 201, 202, 203, 200, 201, 202, 203, 200,
        201, 202, 203])
        more details, see tools/util/fillConvexPoly.png
        '''
        return 0

    def bev_to_pixel(self, coord: np.ndarray):
        coord[:,0] /= self.resolution
        coord[:,1] = coord[:,1] / self.resolution + self.width / 2
        return coord

def ignore_agent_costmap(agent_pos, angle_threshold=135):
    # 计算 agent 的相对位置向量
    agent_vector = np.array(agent_pos) - np.array([0,0])
    ego_direction = np.array([1,0])
    # 归一化agent方向向量
    agent_vec_norm = np.linalg.norm(agent_vector,axis=1)
    
    # 计算夹角的余弦值
    epsilon = 1e-8
    cos_angle = (agent_vector @ ego_direction) / (agent_vec_norm + epsilon)
    angle_radians = np.arccos(cos_angle)

    # 角度范围是0~180°
    angle_degrees = np.degrees(angle_radians)
    
    # 判断是否在屏蔽角度范围内
    return angle_degrees > angle_threshold

def box_center_to_corners(x_c, y_c, w, h, theta):
    """
    Converts (x_c, y_c, w, h, theta) to 4 corner points representation [(x0, y0) , ..., (x3, y3)].

    """
    center = np.array([x_c, y_c])
    center = np.tile(center, 4)
    
    dx = 0.5 * w
    dy = 0.5 * h
    cos = np.cos(theta)
    sin = np.sin(theta)

    dxcos = dx * cos
    dxsin = dx * sin
    dycos = dy * cos
    dysin = dy * sin

    dxy = np.array([- dxcos + dysin, - dxsin - dycos,
                      dxcos + dysin,   dxsin - dycos,
                      dxcos - dysin,   dxsin + dycos,
                    - dxcos - dysin, - dxsin + dycos])
    corners = center + dxy
    return corners.reshape((4,2))
