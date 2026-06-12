import cv2
import numpy as np
import math
import torch
import time
import os
import functools
import os.path as osp
from typing import List, Dict
from visualization.st_graph.stclass import StObjectContainer, StTrajectoryPoint, StPoint2D, StObject, StUsedPoint
from visualization.st_graph.stpolygon2d import StPolygon2D

class DrawSTGraph:
    """ST图绘制类"""
    def __init__(self, save_dir=''):
        self.image = None
        self.rows = 1000.0
        self.cols = 1000.0
        self.t_scale = 100.0
        self.s_scale = 4
        self.coordinate_col = 950.0
        self.save_dir = save_dir
        self.agent_hist_steps = 20
        self.ego_length = 4.886 # 自车长
        self.ego_width = 2.285  # 自车宽
        self.lf = 3.903         # 自车后轴到车头的距离
        self.lr = 0.983         # 自车后轴到车尾的距离
        self.buffer = 0.0       # 碰撞缓冲距离
        self.time_interval = 0.1
        self.obj_end_time = 8.0

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """标准化角度到[-π, π)"""
        while angle >= math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def st_data_process(self, preds_data) -> None:
        """处理数据生成障碍物和ego轨迹"""
        object_trajs = []
        agent_id = preds_data['agent_id']
        agent_attrs = preds_data['agent_attrs']
        agents_mask = preds_data['agent_time_mask']
        agent_prediction_cls = preds_data['agent_prediction_multimode_cls']
        agent_prediction_reg = preds_data['agent_prediction_multimode_reg']
        # 处理障碍物轨迹
        max_n_agent = agents_mask.shape[0]
        for i in range(max_n_agent):
            if not agents_mask[i,self.agent_hist_steps]:
                break
            agent_max_idx = torch.argmax(agent_prediction_cls[i])
            st_object = StObject(id=int(agent_id[i]), length=float(agent_attrs[i][0]), width=float(agent_attrs[i][1]))
            obj_trajectory_raw = agent_prediction_reg[i, agent_max_idx,:,:2]
            
            for j in range(len(obj_trajectory_raw)):
                current_point = obj_trajectory_raw[j]
                theta = self.normalize_angle(float(agent_prediction_reg[i, agent_max_idx,j,3]))
                
                traj_point = StTrajectoryPoint(
                    position=StPoint2D(current_point[0], current_point[1]),
                    theta=theta,
                    time_difference=j * self.time_interval
                )
                st_object.trajectory.append(traj_point)
            
            object_trajs.append(st_object)

        
        # 处理障碍物真值轨迹
        object_gt = []
        agent_status = preds_data['agent_status']
        for i in range(max_n_agent):
            if not agents_mask[i, self.agent_hist_steps]:
                break
            st_object = StObject(id=int(agent_id[i]), length=float(agent_attrs[i][0]), width=float(agent_attrs[i][1]))
            obj_trajectory_raw = agent_status[i, self.agent_hist_steps+1:,:2]
            
            for j in range(len(obj_trajectory_raw)):
                if agents_mask[i, self.agent_hist_steps+1+j]:
                    current_point = obj_trajectory_raw[j]
                    theta = self.normalize_angle(float(agent_status[i, self.agent_hist_steps+1+j,3]))
                    
                    traj_point = StTrajectoryPoint(
                        position=StPoint2D(current_point[0], current_point[1]),
                        theta=theta,
                        time_difference=j * self.time_interval
                    )
                    st_object.trajectory.append(traj_point)
            
            object_gt.append(st_object)

        # 处理ego轨迹
        max_idx = torch.argmax(preds_data['pred_prob'])
        ego_traj = StObject(id="ego", length=self.ego_length, width=self.ego_width)
        sum_distance = 0.0
        
        for i in range(len(preds_data['pred_traj'][max_idx])):
            ego_pt = preds_data['pred_traj'][max_idx][i]
            traj_point = StTrajectoryPoint(
                position=StPoint2D(float(ego_pt[0]), float(ego_pt[1])),
                theta=self.normalize_angle(float(preds_data['pred_yaw'][max_idx][i])),
                time_difference=i * self.time_interval,
                sum_distance=sum_distance
            )
            
            if i > 0:
                prev_pt = preds_data['pred_traj'][max_idx][i-1]
                dx = float(ego_pt[0] - prev_pt[0])
                dy = float(ego_pt[1] - prev_pt[1])
                sum_distance += math.hypot(dx, dy)
                traj_point.sum_distance = sum_distance
            
            ego_traj.trajectory.append(traj_point)
        return object_trajs, object_gt, ego_traj

    def st_calc_polygon(self, x: float, y: float, theta: float, 
                       r2f: float, r2r: float, width: float) -> List[StPoint2D]:
        """计算多边形顶点"""
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        # 计算四个顶点
        x_fr = x + r2f * cos_theta + 0.5 * width * sin_theta
        y_fr = y + r2f * sin_theta - 0.5 * width * cos_theta
        
        x_fl = x + r2f * cos_theta - 0.5 * width * sin_theta
        y_fl = y + r2f * sin_theta + 0.5 * width * cos_theta
        
        x_rl = x - r2r * cos_theta - 0.5 * width * sin_theta
        y_rl = y - r2r * sin_theta + 0.5 * width * cos_theta
        
        x_rr = x - r2r * cos_theta + 0.5 * width * sin_theta
        y_rr = y - r2r * sin_theta - 0.5 * width * cos_theta

        return [
            StPoint2D(x_fr, y_fr),
            StPoint2D(x_fl, y_fl),
            StPoint2D(x_rl, y_rl),
            StPoint2D(x_rr, y_rr)
        ]

    def st_check_polygon_collision(self, ego_center: StPoint2D, ego_sqr: float,
                                  obs_center: StPoint2D, obs_sqr: float,
                                  ego_polygon: StPolygon2D, obs_polygon: StPolygon2D) -> bool:
        """检查两个多边形是否碰撞"""
        # 快速距离检测
        if (ego_center - obs_center).squared_norm() > ego_sqr + obs_sqr:
            return False
        
        # 边界框检测
        if (ego_polygon.max_x < obs_polygon.min_x or
            ego_polygon.min_x > obs_polygon.max_x or
            ego_polygon.max_y < obs_polygon.min_y or
            ego_polygon.min_y > obs_polygon.max_y):
            return False
        
        # 多边形相交检测
        return ego_polygon.intersect(obs_polygon)

    def st_collision_check(self, object_trajs, ego_traj) -> Dict[str, StObjectContainer]:
        """碰撞检查主函数"""
        obs_relations = dict()
        filtered_ignores: List[str] = []

        # 初始化障碍物关系映射
        for obs in object_trajs:
            if not obs.trajectory:
                filtered_ignores.append(obs.id)
                continue
            
            obs_start_x = obs.trajectory[0].position.x
            if obs_start_x < 0:
                filtered_ignores.append(obs.id)
                continue
            
            obj_sqr = (obs.length / 2.0) **2 + (obs.width / 2.0)** 2
            obs_relations[obs.id] = StObjectContainer(obs.id, obj_sqr)

        filtered_ignores_set = set(filtered_ignores)
        ego_sqr = (self.ego_length / 2.0) **2 + (self.ego_width / 2.0)** 2

        # 检查每个障碍物与ego的碰撞
        for obs in object_trajs:
            if obs.id in filtered_ignores_set:
                continue
            
            if obs.id not in obs_relations:
                continue
            
            relation = obs_relations[obs.id]

            # 遍历ego轨迹点
            for ego_pt in ego_traj.trajectory:
                # 计算ego的多边形
                ego_corners = self.st_calc_polygon(
                    ego_pt.position.x, ego_pt.position.y, ego_pt.theta,
                    self.lf + self.buffer, self.lr + self.buffer, self.ego_width + self.buffer
                )
                ego_polygon = StPolygon2D(ego_corners)
                # 遍历障碍物轨迹点
                for j, obs_pt in enumerate(obs.trajectory):
                    obj_time_diff = obs_pt.time_difference
                    if obj_time_diff > self.obj_end_time:
                        break

                    # 计算障碍物的多边形
                    obs_corners = self.st_calc_polygon(
                        obs_pt.position.x, obs_pt.position.y, obs_pt.theta,
                        obs.length / 2.0, obs.length / 2.0, obs.width
                    )
                    obs_polygon = StPolygon2D(obs_corners)
                    # 检查碰撞
                    has_interaction = False
                    if relation.obj_sqr > 0.0:
                        has_interaction = self.st_check_polygon_collision(
                            ego_pt.position, ego_sqr,
                            obs_pt.position, relation.obj_sqr,
                            ego_polygon, obs_polygon
                        )
                    if has_interaction:
                        
                        collision_point = StUsedPoint(
                            t=obj_time_diff,
                            s_min=ego_pt.sum_distance,
                            s_max=ego_pt.sum_distance
                        )

                        merged = False
                        # 合并相同时间的碰撞点
                        for point in relation.collision_points:
                            if abs(point.t - obj_time_diff) < 1e-4:
                                point.s_min = min(point.s_min, ego_pt.sum_distance)
                                point.s_max = max(point.s_max, ego_pt.sum_distance)
                                merged = True
                                break
                        if not merged:
                            obs_relations[obs.id].collision_points.append(collision_point)


        def cmp_time(point: StUsedPoint):
            return point.t
        
        for obj_id, obs_relation in obs_relations.items():
            obs_relations[obj_id].collision_points.sort(key=cmp_time)

        return obs_relations


    # 主处理流程
    def main_draw_st_graph(self, preds_data, frame_idx) -> None:
        """处理帧数据的主函数"""
        object_trajs, object_gt, ego_traj = self.st_data_process(preds_data)
        obs_relations = self.st_collision_check(object_trajs, ego_traj)
        vec_boundary = list(obs_relations.values())

        obs_gt_relations = self.st_collision_check(object_gt, ego_traj)
        gt_vec_boundary = list(obs_gt_relations.values())
        # for it in vec_boundary:
        #     print(it.collision_points)
        self.save_st_graph(vec_boundary, gt_vec_boundary, ego_traj.trajectory, frame_idx)
    

    def show_st_graph(self, 
                     st_boundary_vec: List[StObjectContainer],
                     speed_data: List[StTrajectoryPoint]):
        """显示ST图"""
        # 初始化图像
        self.image = np.ones((int(self.rows), int(self.cols), 3), dtype=np.uint8) * 255
        # 绘制坐标和内容
        self.draw_coordinate()
        self.draw_boundaries(st_boundary_vec, 50, 550)
        self.draw_speed_profiles(speed_data, 50, 550, (150, 50, 50))
        # 显示图像
        cv2.imshow("S-T graph", self.image)
        cv2.waitKey(1)

    def save_st_graph(self, 
                     st_boundary_vec: List[StObjectContainer],
                     st_gt_vec_boundary: List[StObjectContainer],
                     speed_data: List[StTrajectoryPoint],
                     frame_idx: int):
        """显示ST图"""
        # 初始化图像
        self.image = np.ones((int(self.rows), int(self.cols), 3), dtype=np.uint8) * 255
        # 绘制坐标和内容
        self.draw_coordinate(frame_idx)
        self.draw_boundaries(st_boundary_vec, 50, 950)
        self.draw_boundaries(st_gt_vec_boundary, 50, 950, color=(255, 0, 0))
        self.draw_speed_profiles(speed_data, 50, 950, (150, 50, 50))
        # 保存图像
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        img_name = osp.join(self.save_dir, f'{frame_idx}_st.png')
        cv2.imwrite(img_name, self.image)
        # cv2.imshow("S-T graph", self.image)
        # cv2.waitKey(1)

    def to_pixel(self, s: float, t: float, x_off: float, x_limit: float) -> tuple:
        """将(s,t)转换为像素坐标"""
        row = self.coordinate_col - s * self.s_scale
        col = t * self.t_scale + x_off

        # 边界检查
        row = min(max(row, 50), self.coordinate_col - 1)
        col = min(max(col, x_off), x_limit - 1)
        return (int(col), int(row))

    def to_pixel_from_point(self, local_pt: StTrajectoryPoint, x_off: float, x_limit: float) -> tuple:
        """从轨迹点转换为像素坐标"""
        return self.to_pixel(local_pt.sum_distance, local_pt.time_difference, x_off, x_limit)

    def draw_coordinate(self, frame_idx):
        """绘制坐标轴"""
        # 绘制轴线
        cv2.line(self.image, (50, 950), (950, 950), (0, 0, 0), 1)
        cv2.line(self.image, (50, 950), (50, 50), (0, 0, 0), 1)

        # 绘制箭头
        self.arrow((940, 950), (960, 950), (0, 0, 0), 1, 30)
        self.arrow((50, 60), (50, 40), (0, 0, 0), 1, 30)

        # 绘制坐标轴标签
        cv2.putText(self.image, "t", (950, 970), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(self.image, "s", (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(self.image, "0", (30, 970), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(self.image, f"{frame_idx}", (950, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

        # 绘制t轴刻度
        for i in range(1, 9):
            x = 50 + 100 * i
            cv2.line(self.image, (x, 950), (x, 945), (0, 0, 0), 1)
            cv2.putText(self.image, str(i), (x - 5, 970), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # 绘制s轴刻度
        for i in range(1, 21):
            y = 950 - 40 * i
            s_val = 10 * i
            cv2.line(self.image, (50, y), (55, y), (0, 0, 0), 1)
            cv2.putText(self.image, str(s_val), (10 if s_val >= 100 else 20, y + 5), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    def arrow(self, p1: tuple, p2: tuple, color: tuple, thickness: float, alpha: float):
        """绘制箭头"""
        cv2.line(self.image, p1, p2, color, int(thickness))
        
        # 计算箭头角度
        len_vec = math.hypot(p2[1] - p1[1], p2[0] - p1[0])
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        pi = math.pi

        # 绘制箭头分支
        arrow1_x = p2[0] - len_vec * math.cos(angle + pi * alpha / 180)
        arrow1_y = p2[1] - len_vec * math.sin(angle + pi * alpha / 180)
        cv2.line(self.image, (int(arrow1_x), int(arrow1_y)), p2, color, int(thickness))

        arrow2_x = p2[0] - len_vec * math.cos(angle - pi * alpha / 180)
        arrow2_y = p2[1] - len_vec * math.sin(angle - pi * alpha / 180)
        cv2.line(self.image, (int(arrow2_x), int(arrow2_y)), p2, color, int(thickness))

    def draw_boundaries(self, st_boundary_vec: List[StObjectContainer], x_off: float, x_limit: float, color=(0, 0, 255)):
        """绘制边界"""
        thickness = 1

        for boundary in st_boundary_vec:
            if len(boundary.collision_points) < 1:
                continue

            # 绘制边界线
            for i in range(len(boundary.collision_points) - 1):
                # 下边界
                pt1 = self.to_pixel(boundary.collision_points[i].s_min, 
                                   boundary.collision_points[i].t, x_off, x_limit)
                pt2 = self.to_pixel(boundary.collision_points[i+1].s_min, 
                                   boundary.collision_points[i+1].t, x_off, x_limit)
                cv2.line(self.image, pt1, pt2, color, thickness)

                # 垂直线
                pt1 = self.to_pixel(boundary.collision_points[i].s_min, 
                                   boundary.collision_points[i].t, x_off, x_limit)
                pt2 = self.to_pixel(boundary.collision_points[i].s_max, 
                                   boundary.collision_points[i].t, x_off, x_limit)
                cv2.line(self.image, pt1, pt2, color, thickness)

                # 上边界
                pt1 = self.to_pixel(boundary.collision_points[i].s_max, 
                                   boundary.collision_points[i].t, x_off, x_limit)
                pt2 = self.to_pixel(boundary.collision_points[i+1].s_max, 
                                   boundary.collision_points[i+1].t, x_off, x_limit)
                cv2.line(self.image, pt1, pt2, color, thickness)

                # 闭合最后一条边
                if i == len(boundary.collision_points) - 2:
                    pt1 = self.to_pixel(boundary.collision_points[i+1].s_min, 
                                       boundary.collision_points[i+1].t, x_off, x_limit)
                    pt2 = self.to_pixel(boundary.collision_points[i+1].s_max, 
                                       boundary.collision_points[i+1].t, x_off, x_limit)
                    cv2.line(self.image, pt1, pt2, color, thickness)

            # 绘制障碍物ID
            pt = self.to_pixel(boundary.collision_points[0].s_max, 
                              boundary.collision_points[0].t, x_off, x_limit)
            cv2.putText(self.image, str(boundary.obs_id), pt, 
                       cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    def draw_speed_profiles(self, speed_data: List[StTrajectoryPoint], 
                           x_off: float, x_limit: float, color: tuple):
        """绘制速度曲线"""
        thickness = 2
        if len(speed_data) < 2:
            return

        # 绘制轨迹线
        for i in range(len(speed_data) - 1):
            pt1 = self.to_pixel_from_point(speed_data[i], x_off, x_limit)
            pt2 = self.to_pixel_from_point(speed_data[i+1], x_off, x_limit)
            cv2.line(self.image, pt1, pt2, color, thickness)

        # 绘制标签
        pt = self.to_pixel_from_point(speed_data[-1], x_off, x_limit)
        cv2.putText(self.image, "speed_profile", pt, 
                   cv2.FONT_HERSHEY_PLAIN, 1, color, 1)