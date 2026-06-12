import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import copy
import torch
import random
import math
import numpy as np
import cv2
import os.path as osp
from enum import Enum
from scipy.interpolate import interp1d
from scipy.spatial import distance
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.base import CAP_STYLE
from dataset_utils.state_perburbation import safety_check, compute_rear_to_cog, random_offset_generator, _normalization, renormalization, renormalization_pos, pos_perturbation, yaw_perturbation, v_perturbation, yaw_rate_perturbation, y_flip
from dataset_utils.geometric_utils import random_shape_generator, generate_rotated_bbox_points, approximate_agent_cls_shape, cal_convex_iou, CustomPathPoint
#from visualization.visualization import visualization_dataset
PEDESTRIAN_TYPE = 1
VEHICLE_TYPE = 2
BIKE_TYPE = 3
class ContrastiveScenarioGenerator:
    def __init__(
        self,
        positive_func_active,
        negative_func_active,
        future_steps,
        hist_steps,
        total_steps,
        planning_interval,
        n_feature_agent_status,
        max_n_agent,
        work_dir,
        ego_care_normal_laneline,
        dec_acc = None,
        visualize_generated_sample = False
    ) -> None:
        self.positive_func_active = positive_func_active
        self.negative_func_active = negative_func_active
        self.future_steps = future_steps
        self.hist_steps = hist_steps
        self.total_steps = total_steps
        self.planning_interval = planning_interval
        self.n_feature_agent_status = n_feature_agent_status
        self.max_n_agent = max_n_agent
        self.work_dir = work_dir
        self.ego_care_normal_laneline = ego_care_normal_laneline
        self.visualize_generated_sample = visualize_generated_sample
        self.positive_func_set = {
                            # 'pos_perturbation': pos_perturbation,
                            # 'yaw_perturbation': yaw_perturbation,
                            # 'y_flip': self.y_flip,
                            'non_interactive_agents_dropout': self.non_interactive_agents_dropout,
                            "side_non_interactive_agents_dropout": self.side_non_interactive_agents_dropout,
                            "lanekeep_centerline_dropout": self.lanekeep_centerline_dropout,
                            "non_target_centerline_laneline_dropout": self.non_target_centerline_laneline_dropout,}
        self.negative_func_set = {
                            'leading_agent_insertion': self.leading_agent_insertion,
                            "static_obstacle_insertion":self.static_obstacle_insertion,
                            "vru_insertion": self.vru_insertion,
                            'interactive_agents_dropout': self.interactive_agents_dropout,
                            'traffic_light_inversion': self.traffic_light_inversion,
                            'ego_light_switch': self.ego_light_switch,
                            'side_interactive_agents_insertion': self.side_interactive_agents_insertion}
    
        self.ego_corridor = None
        self.ego_path = None
        self.free_path_points = []
    
    def scenario_casual_reasoning_preprocess(self, data, scene, date, i):
        self.get_ego_path(data)
        leading_agents_mask, risky_leading_agents_mask, leading_distance, is_waiting_for_red_light_without_lead = self.get_leading_agents(data)
        #backward_agents_mask = self.get_backward_agents_mask(data)
        interactive_agents_mask = self.get_interactive_agents_mask(data)
        ego_relevant_laneline_mask = self.get_ego_relevant_laneline_mask(data)
        ego_target_laneline_mask = self.get_target_laneline_mask(data, date, i)
        side_agent_mask = self.get_side_agent_mask(data)
        valid_agents_mask = data['agent_time_mask'][:,20].numpy()
        non_interactive_agents_mask = valid_agents_mask & (interactive_agents_mask <= 0) & (risky_leading_agents_mask <= 0)
        side_non_interactive_agents_mask = non_interactive_agents_mask & side_agent_mask
        # if non_interactive_agents_mask.sum() > 1:
        #     non_interactive_agents_mask[leading_agents_mask] = False
            
        preprocess_result = {
            "leading_agents_mask": leading_agents_mask,
            "risky_leading_agents_mask": risky_leading_agents_mask,
            "leading_distance":leading_distance,
            "is_waiting_for_red_light_without_lead": is_waiting_for_red_light_without_lead,
            "interactive_agents_mask": interactive_agents_mask,
            "non_interactive_agents_mask": non_interactive_agents_mask,
            "ego_relevant_laneline_mask": ego_relevant_laneline_mask,
            "ego_target_laneline_mask": ego_target_laneline_mask,
            "side_non_interactive_agents_mask":side_non_interactive_agents_mask
        }
        return preprocess_result

    def generate_contrastive_samples(self, data, preprocess_result, scene):
        # contrastive sample generation
        positive_set_available = []
        negative_set_available = []
        ##### DETERMINE AVAILABLE POSITIVE FUNCTIONS #####
        if 'lat_pos_perturbation' in self.positive_func_active:
            positive_set_available.append('lat_pos_perturbation')

        if 'lon_pos_perturbation' in self.positive_func_active:
            positive_set_available.append('lon_pos_perturbation')
       
        if 'yaw_perturbation' in self.positive_func_active:
            positive_set_available.append('yaw_perturbation')
        
        if 'y_flip' in self.positive_func_active:
            positive_set_available.append('y_flip')
        
        if 'v_perturbation' in self.positive_func_active:
            positive_set_available.append('v_perturbation')
            
        if 'yaw_rate_perturbation' in self.positive_func_active:
            positive_set_available.append('yaw_rate_perturbation')
        
        if 'queued_vehicle_perturbation' in self.positive_func_active:
            positive_set_available.append('queued_vehicle_perturbation') 

        if 'non_interactive_agents_dropout' in self.positive_func_active:
            if np.any(preprocess_result["non_interactive_agents_mask"]):
                positive_set_available.append('non_interactive_agents_dropout')
        if 'side_non_interactive_agents_dropout' in self.positive_func_active:
            if np.any(preprocess_result["side_non_interactive_agents_mask"]):
                positive_set_available.append('side_non_interactive_agents_dropout')
        if 'lanekeep_centerline_dropout' in self.positive_func_active:
            if 'lane_keeping' in scene:
                positive_set_available.append('lanekeep_centerline_dropout')
        if 'non_target_centerline_laneline_dropout' in self.positive_func_active:
            positive_set_available.append('non_target_centerline_laneline_dropout')
        ##### DETERMINE AVAILABLE NEGATIVE FUNCTIONS #####
        if 'interactive_agents_dropout' in self.negative_func_active:
            # donnot dropout all interactive agent if there is no non-interactive agent
            if (not preprocess_result["is_waiting_for_red_light_without_lead"] and 
                (np.any(preprocess_result["interactive_agents_mask"]) 
                or np.any(preprocess_result["leading_agents_mask"])) and 
                np.any(preprocess_result["non_interactive_agents_mask"])
            ):
                negative_set_available.append('interactive_agents_dropout')
        
        if 'traffic_light_inversion' in self.negative_func_active:
            if preprocess_result["is_waiting_for_red_light_without_lead"]:
                negative_set_available.append('traffic_light_inversion')
        
        if 'leading_agent_insertion' in self.negative_func_active:
            if len(self.free_path_points) > 0:
                negative_set_available.append('leading_agent_insertion')

        if 'static_obstacle_insertion' in self.negative_func_active:
            negative_set_available.append('static_obstacle_insertion')
        
        if 'vru_insertion' in self.negative_func_active:
            negative_set_available.append('vru_insertion')
        
        if 'ego_light_switch' in self.negative_func_active:
            if 'change_lane' in scene:
                negative_set_available.append('ego_light_switch')

        if 'side_interactive_agents_insertion' in self.negative_func_active and 'efficient' in scene:
            lane_centerlines = [
                data['laneline_pts'][i] 
                for i in range(len(data['laneline_attrs'])) 
                if data['laneline_attrs'][i] == [0, 0, 0, 1]
            ]
            
            if not lane_centerlines:
                target_lane_centerline_pts = None  # No valid lanes found

            # Find the target lane centerline in the specified direction
            target_lane_centerline_pts = self.find_adjacent_lane_centerline(lane_centerlines, data['ego_curr_status'][3])
            
            if target_lane_centerline_pts is not None:
                preprocess_result['target_lane_centerline_pts'] = target_lane_centerline_pts
            
                # Extract agent positions for the specified timestep
                agents_curr_pos = data['agent_status'][:, (self.hist_steps + 1), :2]
                
                # Define masks for left and right side agents
                side_masks = {
                    'left': ((abs(agents_curr_pos[:, 0]) < 10) & (agents_curr_pos[:, 1] < 4) & (agents_curr_pos[:, 1] > 2)),
                    'right': ((abs(agents_curr_pos[:, 0]) < 10) & (agents_curr_pos[:, 1] > -4) & (agents_curr_pos[:, 1] < -2))
                }
                # Check side-specific conditions
                for side in ['left', 'right']:
                    if side in scene:
                        side_agent_mask = side_masks[side].numpy()
                        side_interactive_agent_mask = side_agent_mask & preprocess_result["interactive_agents_mask"]
                        if not np.any(side_interactive_agent_mask):
                            negative_set_available.append('side_interactive_agents_insertion')

        res_pos = self.transform_pos(data, positive_set_available, preprocess_result, scene)
        res_neg = self.transform_neg(data, negative_set_available, preprocess_result, scene)

        return res_pos, res_neg
   
    def transform_pos(self, res, positive_set_available, preprocess_result, scene):
        res_pos = copy.deepcopy(res)
        # state perturbation methods
        perturbation_positive_set_available = [pos_func for pos_func in positive_set_available if pos_func in ['lat_pos_perturbation','lon_pos_perturbation','yaw_perturbation', 'v_perturbation', 'yaw_rate_perturbation', 'yaw_rate_perturbation_v2', 'queued_vehicle_perturbation']]
        # if 'turn' not in res['scene']: 
        #     perturbation_positive_set_available.append('y_flip')
        assert len(perturbation_positive_set_available) > 0
        # at least choose 1 from perturbation operations
        # ops_num_to_select = random.randint(1, len(perturbation_positive_set_available))
        # selected_perturbation_ops = random.sample(perturbation_positive_set_available, ops_num_to_select)
        #selected_perturbation_ops = [perturbation_ops for perturbation_ops in perturbation_positive_set_available if random.random() < 0.5]
        selected_perturbation_ops = []
        for op in perturbation_positive_set_available:
            if op in ['queued_vehicle_perturbation']:
                if random.random() < 0.8:  # queued_vehicle_perturbation 0.8 概率
                    selected_perturbation_ops.append(op)
            else:
                if random.random() < 0.5:  # 其他操作 0.5 概率
                    selected_perturbation_ops.append(op)

        # Ensure at least one option is selected
        while len(selected_perturbation_ops) == 0:
            selected_perturbation_ops = [perturbation_ops for perturbation_ops in perturbation_positive_set_available if random.random() < 0.5]
        if 'y_flip' in selected_perturbation_ops and res_pos['scene'] in ["lane_keeping","large_curvature_lane_keeping"]:
            res_pos = y_flip(res_pos)
            # # 场景字段左右替换
            # if 'left' in res_pos['scene']:
            #     index = res_pos['scene'].find('left')
            #     res_pos['scene'] = res_pos['scene'][:index] + 'right' + res_pos['scene'][index+4:]
            # elif 'right' in res_pos['scene']:
            #     index = res_pos['scene'].find('left')
            #     res_pos['scene'] = res_pos['scene'][:index] + 'left' + res_pos['scene'][index+5:]

        if 'lat_pos_perturbation' in selected_perturbation_ops or 'lon_pos_perturbation' in selected_perturbation_ops:
            lat_displacement, lon_displacement = 0, 0
            if 'lat_pos_perturbation' in selected_perturbation_ops:
                lat_displacement = abs(min(res_pos['ego_curr_status'][0] * self.planning_interval, 0.5))
                if 'lane_keeping' in scene or 'normal' in scene:
                    lat_displacement = 1.0
                elif 'change_lane' in scene:
                    lat_displacement = abs(min(res_pos['ego_curr_status'][0] * self.planning_interval, 0.1))
                elif 'turn' in scene:
                    lat_displacement = abs(min(res['ego_curr_status'][0] * self.planning_interval, 1.0))
            if 'lon_pos_perturbation' in selected_perturbation_ops and 'brake' in scene:
                lon_displacement = abs(min(res_pos['ego_curr_status'][0] * 20 * self.planning_interval, 2.5)) 
            # gt在2s内平滑过度，江涛在nuplan闭环里测了，效果不好，不用
            res_pos = pos_perturbation(res_pos, x_min=-0.5 * lon_displacement, x_max=lon_displacement, y_min=-lat_displacement, y_max=lat_displacement,hist_steps=self.hist_steps)
            
        if 'yaw_perturbation' in selected_perturbation_ops and 'static' not in scene:
            res_pos = yaw_perturbation(res_pos)
        
        if 'v_perturbation' in selected_perturbation_ops:
            # v_diff = min(abs(res_pos['ego_curr_status'][0]) * 0.1, 1)
            # if scene in ['brake2stop', 'acc']:
            #     v_diff *= 3
            # if scene in ['acc']:
            #     res_pos = v_perturbation(res_pos, min_v=-v_diff, max_v=0.2*v_diff)
            # else:
            #     res_pos = v_perturbation(res_pos, min_v=-v_diff, max_v=v_diff)
            v_diff = min(abs(res_pos['ego_curr_status'][0]) * 0.1, 1)
            if 'brake2stop' in scene:
                v_diff *= 3
            res_pos = v_perturbation(res_pos, min_v=-v_diff, max_v=v_diff)

        if 'yaw_rate_perturbation' in selected_perturbation_ops:
            yaw_rate_diff = abs(res_pos['ego_curr_status'][1] * 0.2)
            res_pos = yaw_rate_perturbation(res_pos, min_yaw_rate=-yaw_rate_diff, max_yaw_rate=yaw_rate_diff)
        
        if 'yaw_rate_perturbation_v2' in selected_perturbation_ops:
            if res_pos['scene'] in ['static', 'turn_left_static', 'interactive_turn_left_static', 'turn_right_static', 'interactive_turn_right_static', 'interactive_agent_cross', 'interactive_agent_large_dec', 'brake', 'brake2stop', 'acc', 'safe_acc', 'cross_lane_keeping', 'lane_keeping', 'near_intersection', 'normal']:
                curr_v = res_pos['ego_curr_status'][1]
                if curr_v < 1.5 and curr_v > -1.5:
                    if curr_v < 0:
                        min_v = -1.5
                        max_v = curr_v
                    elif curr_v > 0:
                        min_v = curr_v
                        max_v = 1.5
                    else:
                        min_v = -1.5
                        max_v = 1.5
                    res_pos = yaw_rate_perturbation(res_pos, min_yaw_rate=min_v-curr_v, max_yaw_rate=max_v-curr_v)
                else:
                    yaw_rate_diff = abs(res_pos['ego_curr_status'][1] * 0.2)
                    res_pos = yaw_rate_perturbation(res_pos, min_yaw_rate=-yaw_rate_diff, max_yaw_rate=yaw_rate_diff)
            else:
                yaw_rate_diff = abs(res_pos['ego_curr_status'][1] * 0.2)
                res_pos = yaw_rate_perturbation(res_pos, min_yaw_rate=-yaw_rate_diff, max_yaw_rate=yaw_rate_diff)
        
        if 'queued_vehicle_perturbation' in selected_perturbation_ops:
            res_pos, queued_vehicle_valid = self.queued_vehicle_perturbation(res_pos, scene, preprocess_result)

        res_pos['cl_status'][0] = True
        non_perturbation_positive_set_available = [pos_func for pos_func in positive_set_available if pos_func not in ['lat_pos_perturbation','lon_pos_perturbation','yaw_perturbation', 'v_perturbation','yaw_rate_perturbation','y_flip', 'queued_vehicle_perturbation']]
        if len(non_perturbation_positive_set_available) == 0:
            return res_pos
        
        if np.random.uniform(0, 1) < 0.5:
            # optional positive sampling methods
            method = random.choice(non_perturbation_positive_set_available)
            # print(f"using positive func: {method}.")
            self.positive_func_set[method](res_pos, preprocess_result)
        
        return res_pos

    def transform_neg(self, res, negative_set_available, preprocess_result, scene):
        res_neg = copy.deepcopy(res)
        if len(negative_set_available) == 0:
            return res_neg
        method = random.choice(negative_set_available)
        # print(f"using negative func: {method}.")
        self.negative_func_set[method](res_neg, preprocess_result)
        res_neg['cl_status'][0] = True
        return res_neg
    
    def set_fake_agent(self, length, width, cls,
                        x, y, v, yaw, score, n_hist_invalid):
        # TODO, merge with fake_agent func @zhangbowen
        # 目前仅支持fake静止、匀速运动目标
        
        agent_attr = torch.tensor([length, width, cls])
        agent_status = torch.zeros(self.total_steps, self.n_feature_agent_status)
        vx = v * torch.cos(yaw)
        vy = v * torch.sin(yaw)
        for i in range(self.total_steps):
            agent_status[i] = torch.tensor([x - vx * (self.hist_steps-i) * self.planning_interval, 
                                          y - vy * (self.hist_steps-i) * self.planning_interval, 
                                          vx, vy, yaw, score])    # x, y, vx, vy, yaw, score

        agent_time_mask = [True] * (self.total_steps)
        agent_time_mask[:n_hist_invalid] = [False] * n_hist_invalid
        agent_time_mask = torch.tensor(agent_time_mask)
        return agent_attr, agent_status, agent_time_mask
    
    def get_ego_path(self,data):
        ego_future_traj_fixed_full = data['ego_future_status_fixed_full'][data['ego_future_mask_fixed_full']]
        ego_future_pos_fixed_full = torch.cat((torch.tensor([[0.,0.]]),ego_future_traj_fixed_full[:,:2]),dim=0)
        if ego_future_traj_fixed_full.shape[0] < 2:
            return
        self.ego_corridor = LineString(ego_future_pos_fixed_full.numpy()).buffer(
                approximate_agent_cls_shape[VEHICLE_TYPE][1] / 2, cap_style=CAP_STYLE.square)

        self.ego_path = LineString([p for p in ego_future_pos_fixed_full])
        return
    
    def get_leading_agents(self, data):    
        leading_agents_mask = np.zeros(data['agent_status'].shape[0],dtype=bool)
        risky_leading_agents_mask = np.zeros(data['agent_status'].shape[0],dtype=bool)
        leading_distance = np.full(data['agent_status'].shape[0], 1e6, dtype=np.float32)
        is_waiting_for_red_light_without_lead = False 

        ego_future_traj_fixed_full = data['ego_future_status_fixed_full'][data['ego_future_mask_fixed_full']]
        if self.ego_path is None or self.ego_corridor is None:
            return leading_agents_mask, risky_leading_agents_mask, leading_distance, is_waiting_for_red_light_without_lead

        agent_bboxes = generate_rotated_bbox_points(data['agent_status'][:,self.hist_steps,:2],data['agent_attrs'][:,:2],data['agent_status'][:,self.hist_steps,4])
        min_ttc = 1e3
        for i, agent_bbox in enumerate(agent_bboxes):
            intersection = self.ego_corridor.intersects(Polygon(agent_bbox))
            if intersection:
                leading_agents_mask[i] = 1
                agent_position = Point([data['agent_status'][i,self.hist_steps,0], data['agent_status'][i,self.hist_steps,1]])
                agent_heading = data['agent_status'][i,self.hist_steps,4]
                leading_distance[i] = self.ego_path.project(agent_position)
                
                # calculate relative direction between agent and egopath
                projection_point = self.ego_path.interpolate(leading_distance[i])
                epsilon = 0.01  # small distance for tangent calculation

                point_before = self.ego_path.interpolate(leading_distance[i] - epsilon)
                point_after = self.ego_path.interpolate(leading_distance[i] + epsilon)
                tangent_vector = np.array([point_after.x - point_before.x, point_after.y - point_before.y])
                tangent_direction = np.arctan2(tangent_vector[1], tangent_vector[0])
                relative_angle = tangent_direction - agent_heading
                # Normalize the angle to be within [-pi, pi]
                relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

                agent_vnorm = torch.norm(data['agent_status'][i,self.hist_steps,2:4])
                ttc = leading_distance[i]/(data['ego_curr_status'][0] - agent_vnorm)
                
                if abs(relative_angle) < 0.3 and ttc > 0 and ttc < 6 :
                    risky_leading_agents_mask[i] = 1
                    if min_ttc > ttc:
                        min_ttc = ttc
        dist_to_stopline = 1e6
        if data['ego_curr_status'][2] == 5: # 0: unknown, 1: invalid, 2: off, 3: green, 4: yellow, 5: red
            stopline_attr = 4 # last value in attr: 0~None; 1~centerline; 2~stopline; 3~crosswalk; 4~route
            #merge 185
            if stopline_attr in data['laneline_attrs'][:,1]: 
                stopline_indices = (data['laneline_attrs'][:,1]==stopline_attr).nonzero(as_tuple=True)[0]
                for stopline_idx in stopline_indices:
                    stopline = LineString(data['laneline_pts'][stopline_idx].squeeze(0))
                    if self.ego_path.intersects(stopline):
                        #import ipdb;ipdb.set_trace()
                        dist_to_stopline = Point((0,0)).distance(stopline)
                        if all(leading_distance > dist_to_stopline):
                            is_waiting_for_red_light_without_lead = True
                            if self.visualize_generated_sample:
                                rand_idx = random.randint(1,10000)
                                img_dir = osp.join(self.work_dir, 'visualization')
                                visualization_dataset(data, rand_idx, img_dir, 'is_waiting_for_red_light_without_lead',corridor=self.ego_corridor)
                            break
        if self.visualize_generated_sample and any(risky_leading_agents_mask):
            rand_idx = random.randint(1,10000)
            img_dir = osp.join(self.work_dir, 'visualization')
            visualization_dataset(data, rand_idx, img_dir, 'risky_leading_agents',agent_highlight_mask=risky_leading_agents_mask,custom_note = f"ttc: {min_ttc}")
        elif self.visualize_generated_sample and any(leading_agents_mask):
            rand_idx = random.randint(1,10000)
            img_dir = osp.join(self.work_dir, 'visualization')
            visualization_dataset(data, rand_idx, img_dir, 'leading_agents_mask',agent_highlight_mask=leading_agents_mask)
            
        
        self.free_path_points.clear()
        free_path_start_progress = data['ego_curr_status'][0]**2 / (2 * 5)
        free_path_end_progress = max(7, data['ego_curr_status'][0]**2 / (2 * 1.5))
        if is_waiting_for_red_light_without_lead:
            free_path_end_progress = dist_to_stopline
        elif np.any(leading_agents_mask):
            free_path_end_progress = np.min(leading_distance)
        
        free_path_start_progress += 5
        free_path_end_progress -= 5
        for i, p in enumerate(self.ego_path.coords):
            progress = self.ego_path.project(Point([p[0],p[1]]))
            if progress > free_path_end_progress:
                break
            if progress <= free_path_end_progress and progress >= free_path_start_progress:
                dy = (self.ego_path.coords[i+1][1] if i + 1 < len(self.ego_path.coords) - 1 else self.ego_path.coords[i][1]) - self.ego_path.coords[i-1][1]
                dx = (self.ego_path.coords[i+1][0] if i + 1 < len(self.ego_path.coords) - 1 else self.ego_path.coords[i][0]) - self.ego_path.coords[i-1][0]
                self.free_path_points.append(CustomPathPoint(p[0],p[1], progress, math.atan2(dy, dx)))
        # if len(self.free_path_points) == 0:
        #     import ipdb;ipdb.set_trace()
        return leading_agents_mask, risky_leading_agents_mask, leading_distance, is_waiting_for_red_light_without_lead
    
    def get_interactive_agents_mask(self, data, max_interaction_horizon=40):
        """
        directly interactive agents:
        1. the agents whose future path cross with ego future path of fixed length
        other criterion are yet to define
        """
        interactive_agents_mask = np.zeros(data['agent_status'].shape[0],dtype=bool)
        #import ipdb;ipdb.set_trace()
        ego_heading = data['ego_future_status'][data['ego_future_mask'],4]
        ego_pos = data['ego_future_status'][data['ego_future_mask'],:2]
        ego_T = ego_pos.shape[0]
        valid_agent_future_mask = copy.deepcopy(data['agent_time_mask'][:,(self.hist_steps+1):])
        #valid_agent_future_mask[:,:self.hist_steps] = False
        agents_heading = data['agent_status'][:,(self.hist_steps+1):,4]
        agents_pos = data['agent_status'][:,(self.hist_steps+1):,:2]
        agents_shape = data['agent_attrs'][:,:2]
        for agent_idx in range(self.max_n_agent):
            if not (True in valid_agent_future_mask[agent_idx]):
                continue

            distances = distance.cdist(
                (agents_pos[agent_idx]).reshape(-1, 2),
                ego_pos.reshape(-1, 2),
            )
            distances[~valid_agent_future_mask[agent_idx]] = 1e6
            min_t_agent, min_t_ego = np.unravel_index(distances.argmin(), distances.shape)
            min_dist = distances[min_t_agent, min_t_ego]

            if min_dist > 4:  # coarse distance judgement
                continue
            if min_t_ego < min_t_agent or min_t_ego - min_t_agent > max_interaction_horizon: 
                continue
            # import ipdb;ipdb.set_trace()
            agent_bbox = generate_rotated_bbox_points(agents_pos[agent_idx,min_t_agent].reshape(-1, 2),agents_shape[agent_idx].reshape(-1, 2),agents_heading[agent_idx,min_t_agent].reshape(-1, 1)).squeeze(0)
            ego_bbox = generate_rotated_bbox_points(ego_pos[min_t_ego].reshape(-1, 2),torch.tensor(approximate_agent_cls_shape[VEHICLE_TYPE]).reshape(-1, 2),ego_heading[min_t_ego].reshape(-1, 1)).squeeze(0)
            intersection = Polygon(ego_bbox).intersects(Polygon(agent_bbox))
            if intersection:
                interactive_agents_mask[agent_idx] = 1

        if self.visualize_generated_sample and any(interactive_agents_mask):
            rand_idx = random.randint(1,10000)
            img_dir = osp.join(self.work_dir, 'visualization')
            visualization_dataset(data, rand_idx, img_dir, 'interactive_agents_mask',agent_highlight_mask=interactive_agents_mask)
            
        return interactive_agents_mask
    
    def get_side_agent_mask(self,data):
        agents_curr_pos = data['agent_status'][:,(self.hist_steps+1),:2]
        side_agent_mask = ((abs(agents_curr_pos[:,0]) < 10) & (abs(agents_curr_pos[:,1]) < 4) & (abs(agents_curr_pos[:,1]) > 2)).numpy()
        return side_agent_mask
    def get_target_laneline_mask(self, data, date, i):
        """
         the lane ego should reach in 50m, represented in centerline or laneline
        """
        ego_target_laneline_mask = np.zeros(data['laneline_attrs'].shape[0],dtype=bool)
        ego_future_traj_fixed_50m = torch.cat((torch.tensor([[0.,0.]]),data['ego_future_status_fixed_full'][data['ego_future_mask_fixed_full']][:50,:2]),dim=0)
        if ego_future_traj_fixed_50m.shape[0] < 2:
            return ego_target_laneline_mask
        # px, py = ego_future_traj_fixed_50m[-1,0],ego_future_traj_fixed_50m[-1,1]
        min_positive_dist = float('inf')
        min_negative_dist = float('inf')
        min_centerline_dist = float('inf')
        nearest_centerline_index = None
        nearest_positive_index = None
        nearest_negative_index = None
        for pt_meter_idx in [50,40,30,20,10,0]:
            if nearest_centerline_index is not None:
                break
            if pt_meter_idx >= ego_future_traj_fixed_50m.shape[0]:
                print("ego_future_traj_fixed_50m shape err", date, i)
                continue
            px, py = ego_future_traj_fixed_50m[pt_meter_idx,0],ego_future_traj_fixed_50m[pt_meter_idx,1]
            for i, laneline in enumerate(data['laneline_pts']):
                if data['laneline_attrs'][i,1] == 3: # 0:padding; 1:edge; 2:laneline; 3:centerline; 4:stopline; 5:route; 6:other
                    for j in range(laneline.shape[0]):
                        # 计算点到目标点的距离
                        # dist = np.sqrt(0.01*(laneline[j,0] - px)**2 + (laneline[j,1] - py)**2)
                        l1_x_dist = (laneline[j,0] - px)
                        l1_y_dist = (laneline[j,1] - py)
                        if abs(l1_x_dist) > 10:
                            continue
                        if abs(l1_y_dist) < min_centerline_dist:
                            min_centerline_dist = abs(l1_y_dist)
                            nearest_centerline_index = i

        if nearest_centerline_index is not None:
            ego_target_laneline_mask[nearest_centerline_index] = True
        # ego_target_laneline_mask[nearest_positive_index] = True
        # ego_target_laneline_mask[nearest_negative_index] = True
        if self.visualize_generated_sample and any(ego_target_laneline_mask):
            rand_idx = random.randint(1,10000)
            img_dir = osp.join(self.work_dir, 'visualization')
            visualization_dataset(data, rand_idx, img_dir,laneline_highlight_mask=ego_target_laneline_mask, custom_note='ego_target_laneline')
            import ipdb;ipdb.set_trace()
        return ego_target_laneline_mask
    def get_ego_relevant_laneline_mask(self, data):
        """
        make route side lanelines as important to introduce minimun inductive biases:
        1. the agents whose future path cross with ego future path of fixed length
        other criterion are yet to define
        """
        ego_relevant_laneline_mask = np.zeros(data['laneline_attrs'].shape[0],dtype=bool)
        ego_future_traj_fixed_50m = torch.cat((torch.tensor([[0.,0.]]),data['ego_future_status_fixed_full'][data['ego_future_mask_fixed_full']][:50,:2]),dim=0)
        if ego_future_traj_fixed_50m.shape[0] < 2:
            return ego_relevant_laneline_mask
        ego_lanewidth_3m_corridor_50m = LineString(ego_future_traj_fixed_50m.numpy()).buffer(
            3, cap_style=CAP_STYLE.square)
        
        if self.ego_care_normal_laneline:
            # laneline, centerline use 2m corridor
            ego_lanewidth_2m_corridor_50m = LineString(ego_future_traj_fixed_50m.numpy()).buffer(
                2, cap_style=CAP_STYLE.square)
            normal_laneline_mask = (data['laneline_attrs'][:, 1] == 2) | (data['laneline_attrs'][:, 1] == 3)
            normal_laneline_indices = torch.nonzero(normal_laneline_mask, as_tuple=False).squeeze()
            normal_laneline_indices = normal_laneline_indices.unsqueeze(0) if normal_laneline_indices.dim() == 0 else normal_laneline_indices
            for i in normal_laneline_indices:
                l = LineString(data['laneline_pts'][i].numpy())
                if l.intersects(ego_lanewidth_2m_corridor_50m):
                    ego_relevant_laneline_mask[i] = True
        
        # curb use 3m corridor
        curb_laneline_mask = (data['laneline_attrs'][:, 1]==1)
        curb_laneline_indices = torch.nonzero(curb_laneline_mask, as_tuple=False).squeeze()
        curb_laneline_indices = curb_laneline_indices.unsqueeze(0) if curb_laneline_indices.dim() == 0 else curb_laneline_indices

        for i in curb_laneline_indices:
            l = LineString(data['laneline_pts'][i].numpy())
            if l.intersects(ego_lanewidth_3m_corridor_50m):
                ego_relevant_laneline_mask[i] = True
        
        if self.visualize_generated_sample and any(ego_relevant_laneline_mask):
            rand_idx = random.randint(1,10000)
            img_dir = osp.join(self.work_dir, 'visualization')
            visualization_dataset(data, rand_idx, img_dir, 'ego_relevant_curb',laneline_highlight_mask=ego_relevant_laneline_mask)
        return ego_relevant_laneline_mask
    def queued_vehicle_perturbation(self, data, scene, preprocess_result):
        valid_agent_mask = torch.zeros(data['agent_status'].shape[0], dtype=torch.bool)
        if 'change_lane' not in scene and 'nudge' not in scene:
            vx, vy = data['ego_curr_status'][:2]
            vel = torch.sqrt(vx**2 + vy**2)
            dist_to_stopline = 1e6
            stopline_attr = 4 #  0~None; 1~edge; ; 2~laneline; 3~centerline; 4~stopline; 5~route; 6~other
            ego_point = Point(0.0, 0.0)
            leading_distance = np.full(data['agent_status'].shape[0], 1e6, dtype=np.float32)
            # 路口场景
            if stopline_attr in data['laneline_attrs'][:,1] and any(preprocess_result['leading_agents_mask']):
                stopline_indices = (data['laneline_attrs'][:,1] == stopline_attr).nonzero(as_tuple=True)[0]
                leading_mask = preprocess_result['leading_agents_mask']
                for i in np.where(leading_mask)[0]:
                    ax = data['agent_status'][i, self.hist_steps, 0]
                    ay = data['agent_status'][i, self.hist_steps, 1]
                    leading_distance[i] = self.ego_path.project(Point([ax, ay]))

                for stopline_idx in stopline_indices:
                    stopline = LineString(data['laneline_pts'][stopline_idx].squeeze(0))
                    if self.ego_path.intersects(stopline):
                        dist_to_stopline = Point((0,0)).distance(stopline)
                        # 有前车在停止线前
                        if any(leading_distance <= dist_to_stopline):
                            # 再加 costmap 逻辑：只保留符合条件的 leading agent
                            for i in np.where(leading_mask)[0]:
                                if leading_distance[i] > dist_to_stopline:  
                                    continue
                                # costmap条件
                                if not data['agent_time_mask'][i, self.hist_steps]:
                                    continue
                                if not data['agent_time_mask'][i, self.hist_steps+1:].any():
                                    continue
                                distances = torch.cdist(
                                    data['ego_future_status'][:,:2],
                                    data['agent_status'][i, self.hist_steps+1:, :2][data['agent_time_mask'][i, self.hist_steps+1:]]
                                )
                                if torch.min(distances) <= 2.5:
                                    continue
                                if torch.max(torch.abs(data['agent_status'][i, self.hist_steps+1:, 2:4][data['agent_time_mask'][i, self.hist_steps+1:]])) >= 0.3:
                                    continue
                                if torch.max(torch.abs(
                                    data['agent_status'][i, self.hist_steps+1:, :2][data['agent_time_mask'][i, self.hist_steps+1:]] -
                                    data['agent_status'][i, self.hist_steps, :2]
                                )) >= 1:
                                    continue

                                valid_agent_mask[i] = True

                            if any(valid_agent_mask):
                                #rand_idx = random.randint(1,10000)
                                #img_dir = osp.join(self.work_dir, 'visualization_ori')
                                #visualization_dataset(data, rand_idx, img_dir, scene, agent_highlight_mask=valid_agent_mask)

                                data = self.leading_perturbation(data, valid_agent_mask)
                                
                                #img_dir = osp.join(self.work_dir, 'visualization_per')
                                #visualization_dataset(data, rand_idx, img_dir, scene, agent_highlight_mask=valid_agent_mask)
                                break
        return data, valid_agent_mask

    def leading_perturbation(self, data, valid_agent_mask):
        his = self.hist_steps
        agent_status = data['agent_status']  # shape: (num_agents, T, 6)  [x, y, vx, vy, yaw, conf]

        y_mean = 0.2
        y_std = 0.5
        for i, is_valid in enumerate(valid_agent_mask):
            if not is_valid:
                continue

            # 高斯,上下两个0.2的峰
            noise_y = np.random.normal(loc = y_mean, scale = y_std)
            sign_y = np.random.choice([-1, 1])

            if np.random.rand() < 0.5:
                # case 1: 从一开始就压线开 -> 整段历史都加偏移
                agent_status[i, :his+1, 1] += noise_y * sign_y
            else:
                # case 2: 从中间某个时刻开始压线
                start = np.random.randint(0, max(1, his-5))
                length = his + 1 - start
                ramp = np.linspace(0, 1, length)
                agent_status[i, start:his+1, 1] += noise_y * sign_y * ramp

        return data

    def interactive_agents_dropout(self, data, preprocess_result):
        dropout_mask = (
            preprocess_result['interactive_agents_mask']
            | preprocess_result['leading_agents_mask']
        )
        if self.visualize_generated_sample and any(dropout_mask):
            rand_idx = random.randint(1,10000)
            img_dir = osp.join(self.work_dir, 'visualization')
            visualization_dataset(data, rand_idx, img_dir, 'interactive_agents_dropout',agent_highlight_mask=dropout_mask)
        data['agent_status'][dropout_mask] = 0
        data['agent_attrs'][dropout_mask] = 0
        data['agent_time_mask'][dropout_mask] = 0
        return
    
    def non_interactive_agents_dropout(self, data, preprocess_result):
        iter_times = 5
        # empty agent is not allowed in agent encoder
        for i in range(iter_times):
            drop_portion = np.random.uniform(low=0.1, high=1.0)
            noise = np.random.uniform(0, 1, len(preprocess_result["non_interactive_agents_mask"]))
            noise[~preprocess_result["non_interactive_agents_mask"]] = 2
            dropout_mask = noise <= drop_portion
            if not torch.any(data['agent_attrs'][~dropout_mask]):
                continue
            if self.visualize_generated_sample and any(dropout_mask):
                rand_idx = random.randint(1,10000)
                img_dir = osp.join(self.work_dir, 'visualization')
                visualization_dataset(data, rand_idx, img_dir, 'non_interactive_agents_dropout',agent_highlight_mask=dropout_mask)
            data['agent_status'][dropout_mask] = 0
            data['agent_attrs'][dropout_mask] = 0
            data['agent_time_mask'][dropout_mask] = 0
        return
    
    def side_non_interactive_agents_dropout(self, data, preprocess_result):
        data['agent_status'][preprocess_result["side_non_interactive_agents_mask"]] = 0
        data['agent_attrs'][preprocess_result["side_non_interactive_agents_mask"]] = 0
        data['agent_time_mask'][preprocess_result["side_non_interactive_agents_mask"]] = 0
        return
    
    def lanekeep_centerline_dropout(self, data, preprocess_result):
        
        # 条件 1: centerline
        mask_attr = data['laneline_attrs'][:, 1] == 1

        # 条件 2: centerline上存在点的 y 坐标 < 0.5
        mask_pts = torch.any(data['laneline_pts'][:, :, 1] < 0.5, dim=1)

        relevant_indices = torch.where(mask_attr & mask_pts)[0]

        data['laneline_pts'][relevant_indices] = 0
        data['laneline_attrs'][relevant_indices] = 0
        return
    
    def non_target_centerline_laneline_dropout(self, data, preprocess_result):
        
        # 条件 1: centerline
        non_target_centerline_mask = preprocess_result['ego_target_laneline_mask'] == 0

        # # 条件 2: centerline上存在点的 y 坐标 < 0.5
        # mask_pts = torch.any(data['laneline_pts'][:, :, 1] < 0.5, dim=1)

        # relevant_indices = torch.where(mask_attr & mask_pts)[0]

        data['laneline_pts'][non_target_centerline_mask] = 0
        data['laneline_attrs'][non_target_centerline_mask] = 0
        return
    def leading_agent_insertion(self, data, preprocess_result):
        """
        insert a leading agent in front of ego vehicle if there is no leading vehicle
        """
        path_point = random.choice(self.free_path_points)
        agents_velocity = torch.norm(
           data['agent_status'][:, self.hist_steps, 2:4], dim=-1
        )
        agents_velocity_diff = torch.abs(agents_velocity[1:] - data['ego_curr_status'][0])
        similar_agent_idx = np.argmin(agents_velocity_diff)
        if agents_velocity_diff[similar_agent_idx] < 2:
            copy_agent_idx = similar_agent_idx
        else:
            copy_agent_idx = None

        if copy_agent_idx is None:
            scale_coeff = 1.0
        elif agents_velocity[copy_agent_idx] < 0.1:
            scale_coeff = 1.0
        else:
            scale_coeff = data['ego_curr_status'][0] / agents_velocity[copy_agent_idx]
        ## If copy_agent_idx is None, directly copy ego behaviour with scaling operations

        agent_attr, agent_status, agent_time_mask = self._generate_agent_from_idx(data, copy_agent_idx, scale_coeff, path_point)
        # agent_attr, agent_status, agent_time_mask = self.set_fake_agent(
        #                                             length=random_shape_generator(2)[0].item(), 
        #                                             width=random_shape_generator(2)[1].item(), 
        #                                             cls=2,  
        #                                             x=torch.from_numpy(random_offset_generator(low=[5.0], high=[max(5.0,data['ego_curr_status'][0] * self.future_steps * self.planning_interval)])),
        #                                             y=torch.from_numpy(random_offset_generator(low=[-1.0], high=[1.0])), 
        #                                             v =min(data['ego_curr_status'][0] + torch.from_numpy(random_offset_generator(low=[-8.0], high=[-2.0])),0),
        #                                             yaw=torch.from_numpy(random_offset_generator(low=[-0.1], high=[0.1])), 
        #                                             score=0.9, 
        #                                             n_hist_invalid=0)
        # if exist_agent_idxs.shape[0] == 0:
        #     # no agent in scene, generate a leading agent from scratch     
        #     inserted_agent_idx = 0
        # elif empty_agent_idxs.shape[0] == 0:
        #     # if the real agent num reach max_agent_num, remove one randomly
        #     inserted_agent_idx = random.randint(0,self.max_n_agent - 1)
        # else:
        #     inserted_agent_idx = empty_agent_idxs[0]

        empty_agent_idxs = torch.nonzero((data['agent_status'][:,:,5] <= 1e-6).all(dim=1) | (data['agent_time_mask'] == 0).all(dim=1), as_tuple=True)[0] # the 6th agent feature score is zero for non-exist agent
        inserted_agent_idx = empty_agent_idxs[0] if empty_agent_idxs.shape[0] > 0 else data['agent_status'].shape[0] - 1
        data['agent_status'][inserted_agent_idx] = agent_status
        data['agent_attrs'][inserted_agent_idx] = agent_attr
        data['agent_time_mask'][inserted_agent_idx] = agent_time_mask
        if self.visualize_generated_sample:
            rand_idx = random.randint(1,10000)
            img_dir = osp.join(self.work_dir, 'visualization')
            inserted_agents_mask = np.zeros(data['agent_status'].shape[0],dtype=bool)
            inserted_agents_mask[inserted_agent_idx] = True
            visualization_dataset(data, rand_idx, img_dir, 'leading_agent_insertion',agent_highlight_mask=inserted_agents_mask,corridor=self.ego_corridor)
        return
    
    def static_obstacle_insertion(self, data, preprocess_result):
        """
        insert a static obstacle at a position need to be avoided by the ego vehicle
        """
        empty_agent_idxs = torch.nonzero((data['agent_status'][:,:7] == 0).all(dim=1), as_tuple=True)[0] # the 9th agent feature score is not zero for non-exist agent now
        exist_agent_idxs = torch.nonzero((data['agent_status'][:,:7] != 0).any(dim=1) & (data['agent_status'][:,8] > 1e-6), as_tuple=True)[0]
        agent_cls = random.choice([PEDESTRIAN_TYPE, VEHICLE_TYPE, BIKE_TYPE])
        agent_attr, agent_status, agent_time_mask = self.set_fake_agent(
                                                    length=random_shape_generator(agent_cls)[0].item(), 
                                                    width=random_shape_generator(agent_cls)[1].item(), 
                                                    cls=agent_cls, 
                                                    x=torch.from_numpy(random_offset_generator(low=[10.0], high=[max(10.0,data['ego_curr_status'][0] * self.future_steps * self.planning_interval)])),
                                                    y=torch.from_numpy(random_offset_generator(low=[-2.0], high=[2.0])), 
                                                    v = 0,
                                                    yaw= torch.from_numpy(random_offset_generator(low=[-1.0], high=[1.0])), 
                                                    score=0.9, 
                                                    n_hist_invalid=0)
        if exist_agent_idxs.shape[0] == 0:
            # no agent in scene, generate a leading agent from scratch     
            inserted_agent_idx = 0
        elif empty_agent_idxs.shape[0] == 0:
            # if the real agent num reach max_agent_num, remove one randomly
            inserted_agent_idx = random.randint(0,self.max_n_agent - 1)
        else:
            inserted_agent_idx = empty_agent_idxs[0]
        data['agent_status'][inserted_agent_idx] = agent_status
        data['agent_attrs'][inserted_agent_idx] = agent_attr
        data['agent_time_mask'][inserted_agent_idx] = agent_time_mask
        return  

    def vru_insertion(self, data, preprocess_result):
        """
         mimic the behaviour of another vru at a given position, if non-exist then generate a vru from scratch
        """
        empty_agent_idxs = torch.nonzero((data['agent_status'][:,:7] == 0).all(dim=1), as_tuple=True)[0] # the 9th agent feature score is not zero for non-exist agent now
        exist_agent_idxs = torch.nonzero((data['agent_status'][:,:7] != 0).any(dim=1) & (data['agent_status'][:,8] > 1e-6), as_tuple=True)[0]
        vru_agents = torch.where((data['agent_attrs'][:, 2] == 1) | (data['agent_attrs'][:, 2] == 3))[0]
        mimic_exist_agent = False
        if vru_agents.shape[0] > 0:
            mimic_exist_agent = True
            mimic_agent_idx = vru_agents[0]

        if mimic_exist_agent:
            agent_attr, agent_status, agent_time_mask = self.set_fake_agent(
                                                    length=random_shape_generator(int(data['agent_attrs'][mimic_agent_idx,2].item()))[0].item(), 
                                                    width=random_shape_generator(int(data['agent_attrs'][mimic_agent_idx,2].item()))[1].item(), 
                                                    cls=data['agent_attrs'][mimic_agent_idx,2].item(), 
                                                    x=torch.from_numpy(random_offset_generator(low=[10.0], high=[max(10.0,data['ego_curr_status'][0] * self.future_steps * self.planning_interval)])),
                                                    y=torch.from_numpy(random_offset_generator(low=[-2.0], high=[2.0])), 
                                                    v = torch.sqrt(data['agent_status'][mimic_agent_idx,self.hist_steps,2] ** 2 + data['agent_status'][mimic_agent_idx,self.hist_steps,3] ** 2),
                                                    yaw= data['agent_status'][mimic_agent_idx,self.hist_steps,4], 
                                                    score=0.9, 
                                                    n_hist_invalid=0)
            if empty_agent_idxs.shape[0] == 0:
                # if the real agent num reach max_agent_num, remove one randomly
                inserted_agent_idx = random.randint(0,self.max_n_agent - 1)
            else:
                inserted_agent_idx = empty_agent_idxs[0]
        else:
            vru_cls = random.choice([PEDESTRIAN_TYPE, BIKE_TYPE])
            agent_attr, agent_status, agent_time_mask = self.set_fake_agent(
                                                    length=random_shape_generator(vru_cls)[0].item(), 
                                                    width=random_shape_generator(vru_cls)[1].item(), 
                                                    cls=vru_cls, 
                                                    x=torch.from_numpy(random_offset_generator(low=[10.0], high=[max(10.0,data['ego_curr_status'][0] * self.future_steps * self.planning_interval)])),
                                                    y=torch.from_numpy(random_offset_generator(low=[-2.0], high=[2.0])), 
                                                    v = 0,
                                                    yaw= torch.from_numpy(random_offset_generator(low=[-1.0], high=[1.0])), 
                                                    score=0.9, 
                                                    n_hist_invalid=0)
            if empty_agent_idxs.shape[0] == 0:
                # if the real agent num reach max_agent_num, remove one randomly
                inserted_agent_idx = random.randint(0,self.max_n_agent - 1)
            else:    
                inserted_agent_idx = empty_agent_idxs[0]
        data['agent_status'][inserted_agent_idx] = agent_status
        data['agent_attrs'][inserted_agent_idx] = agent_attr
        data['agent_time_mask'][inserted_agent_idx] = agent_time_mask
        return    
    
    def traffic_light_inversion(self, data, preprocess_result):
        assert data['ego_curr_status'][2] == 5 # 0: unknown, 1: invalid, 2: off, 3: green, 4: yellow, 5: red
        possible_ego_traffic_light = [2, 3, 4] 
        data['ego_curr_status'][2] = random.choice(possible_ego_traffic_light)
        if self.visualize_generated_sample:
            rand_idx = random.randint(1,10000)
            img_dir = osp.join(self.work_dir, 'visualization')
            visualization_dataset(data, rand_idx, img_dir, 'traffic_light_inversion')
        return
    
    def ego_light_switch(self, data, preprocess_result):
        ## for lanechange scene, close the light
        data['ego_curr_status'][3] = 1
        # possible_ego_light = [1, 2, 3] # 1: off 2: left 3: right
        # possible_ego_light.remove(data['ego_curr_status'][3])
        # data['ego_curr_status'][3] = random.choice(possible_ego_light)
        return
    
    def generate_interactive_target(
        self, side_lane_centerline, ego_speed, hist_steps, future_steps, scale_coeff_range=(1.0, 1.4), lateral_tolerance=0.5
    ):
        """
        Generate an interactive target based on the defined criteria.

        Parameters:
        - side_lane_centerline: list of (x, y) points representing the side lane's centerline.
        - ego_speed: float, the current speed of the ego vehicle.
        - hist_steps: int, number of historical steps.
        - future_steps: int, number of future steps.
        - scale_coeff_range: tuple, range of scaling coefficients for velocity (default 1.0-1.4).
        - lateral_tolerance: float, allowable lateral deviation from the lane centerline.

        Returns:
        - agent_attr: tensor, attributes of the generated agent (length, width, type).
        - agent_status: tensor, agent status over time (x, y, vx, vy, yaw, score).
        - agent_time_mask: tensor, time mask for the agent's presence in each timestep.
        """
        # Scale coefficient: Random multiplier for ego_speed
        scale_coeff = random.uniform(*scale_coeff_range)
        target_speed = scale_coeff * ego_speed

        # Randomly sample a longitudinal position within (-10, 10)
        longitudinal_position = random.uniform(-10, 10)

        # Find the corresponding lateral position on the lane centerline
        lane_points = torch.tensor(side_lane_centerline)
        longitudinal_indices = lane_points[:, 0]
        lateral_position = torch.interp(
            torch.tensor(longitudinal_position), longitudinal_indices, lane_points[:, 1]
        )

        # Add a random lateral deviation within tolerance
        lateral_position += random.uniform(-lateral_tolerance, lateral_tolerance)

        # Generate historical and future positions near the lane centerline
        positions = lane_points[:, :2]  # Extract (x, y)
        closest_idx = torch.argmin(torch.abs(longitudinal_indices - longitudinal_position))
        hist_positions = positions[max(0, closest_idx - hist_steps):closest_idx]
        fut_positions = positions[closest_idx:min(len(positions), closest_idx + future_steps)]

        # Compute heading direction from lane centerline
        velocity_direction = torch.tensor([1.0, 0.0]) if len(hist_positions) < 2 else (
            hist_positions[-1] - hist_positions[-2]
        )
        velocity_direction /= torch.norm(velocity_direction)
        heading_angle = torch.atan2(velocity_direction[1], velocity_direction[0])

        # Compute velocity
        velocity = target_speed * velocity_direction

        # Generate agent status
        agent_status = torch.zeros(hist_steps + future_steps + 1, self.n_feature_agent_status)
        positions_all = torch.cat([hist_positions, torch.tensor([[longitudinal_position, lateral_position]]), fut_positions])
        for i, position in enumerate(positions_all):
            agent_status[i] = torch.tensor([
                position[0], position[1], velocity[0], velocity[1], heading_angle, 0.9
            ])

        # Generate agent attributes
        agent_attr = torch.tensor([
            random_shape_generator(VEHICLE_TYPE, 0.1)[0].item(),
            random_shape_generator(VEHICLE_TYPE, 0.1)[1].item(),
            VEHICLE_TYPE,
        ])

        # Generate time mask
        agent_time_mask = torch.tensor([True] * len(positions_all) + [False] * (hist_steps + future_steps + 1 - len(positions_all)))

        return agent_attr, agent_status, agent_time_mask

    def find_adjacent_lane_centerline(self, lane_centerlines, direction):
        """
        Find the nearest lane centerline on the left or right side of the ego lane.

        Parameters:
        - lane_centerlines: list, each element is a list of (x, y) points representing a lane centerline.
        - direction: int, 2 for finding the left lane, 3 for finding the right lane.

        Returns:
        - adjacent_lane_centerline: list, the centerline points of the nearest adjacent lane in the specified direction.
        Returns None if no such lane exists.
        """
        def find_nearest_lane_to_origin(lanes):
            # Find the lane centerline closest to the origin (0, 0)
            min_distance = float('inf')
            nearest_lane = None
            for lane in lanes:
                lane_points = np.array(lane)
                distances = np.linalg.norm(lane_points, axis=1)
                avg_distance = np.mean(distances)
                if avg_distance < min_distance:
                    min_distance = avg_distance
                    nearest_lane = lane
            return nearest_lane

        def is_in_direction(line, ref_point, target_direction):
            # Check if a centerline is in the target direction (left or right) relative to the reference point
            relative_points = np.array(line) - np.array(ref_point)
            if target_direction == 2:  # Left
                return np.sum(relative_points[:, 1] > 0) > len(relative_points) / 2
            elif target_direction == 3:  # Right
                return np.sum(relative_points[:, 1] < 0) > len(relative_points) / 2
            return False

        def compute_distance_to_ego(line, ref_point):
            # Compute the average distance from the centerline to the reference point
            line_points = np.array(line)
            distances = np.linalg.norm(line_points - np.array(ref_point), axis=1)
            return np.mean(distances)

        # Identify the ego lane centerline as the closest to the origin
        ego_lane_centerline = find_nearest_lane_to_origin(lane_centerlines)
        if ego_lane_centerline is None:
            return None  # No lanes available

        # Use the nearest point on the ego lane centerline as the reference point
        ref_point = np.array(ego_lane_centerline).mean(axis=0)  # Use the geometric center of the ego lane

        adjacent_lane_candidates = []
        for centerline in lane_centerlines:
            if centerline == ego_lane_centerline:
                continue  # Skip the ego lane centerline
            if is_in_direction(centerline, ref_point, direction):  # Check if the lane is in the specified direction
                distance = compute_distance_to_ego(centerline, ref_point)
                adjacent_lane_candidates.append((centerline, distance))
        
        if not adjacent_lane_candidates:
            return None  # No lane found in the specified direction

        # Find the nearest lane centerline in the specified direction
        adjacent_lane_centerline = min(adjacent_lane_candidates, key=lambda x: x[1])[0]
        return adjacent_lane_centerline

    def side_interactive_agents_insertion(self, data, preprocess_result):
        """
        Insert interactive agents based on the side lane.

        Parameters:
        - data: dict, contains information about lane attributes, points, and ego status.
        - preprocess_result: dict, stores preprocessed results, including the target lane centerline points.

        Returns:
        - agent_attr: tensor, attributes of the generated agent (length, width, type).
        - agent_status: tensor, agent status over time (x, y, vx, vy, yaw, score).
        - agent_time_mask: tensor, time mask for the agent's presence in each timestep.
        Returns None if no target lane exists or no interactive agent is generated.
        """

        # Ensure the target lane centerline points exist in preprocess_result
        if 'target_lane_centerline_pts' not in preprocess_result:
            import ipdb; ipdb.set_trace()

        # Extract necessary parameters
        ego_speed = data['ego_curr_status'][0]
        hist_steps = self.hist_steps
        future_steps = self.future_steps

        # Define configuration for the interactive target
        scale_coeff_range = (1.0, 1.4)  # Range of scaling coefficients for velocity
        lateral_tolerance = 0.5         # Allowable lateral deviation from the lane centerline

        # Generate an interactive agent based on the side lane centerline
        agent_attr, agent_status, agent_time_mask = self.generate_interactive_target(
            preprocess_result['target_lane_centerline_pts'],
            ego_speed,
            hist_steps,
            future_steps,
            scale_coeff_range,
            lateral_tolerance
        )

        return agent_attr, agent_status, agent_time_mask
    
    def _generate_agent_from_idx(
        self, data, mimic_agent_idx, scale_coeff, path_point, shape_offset=0.1
    ):
        scale_coeff = (scale_coeff * random_offset_generator(low=[0.0], high=[0.8])).item()
        if mimic_agent_idx is None:
            #mimic ego vehicle with scaling operation
            current_position = torch.tensor([0.0,0.0])
            hist_position = torch.zeros(self.hist_steps, 2)
            for i in range(self.hist_steps):
                hist_position[i] = torch.tensor([current_position[0] - data['ego_curr_status'][0] * (self.hist_steps-i) * self.planning_interval, 0.0]) 
            fut_position = data['ego_future_status'][:,:2]

            heading = torch.zeros(self.total_steps, 1)
            for i in range(self.hist_steps):
                heading[i] = torch.tensor([0.0])
            heading[self.hist_steps] = 0.0
            for i in range(self.future_steps):
                heading[self.hist_steps + i + 1] = data['ego_future_status'][i,4].unsqueeze(dim=0) 

            velocity = torch.zeros(self.total_steps, 2)
            for i in range(self.hist_steps):
                velocity[i] = scale_coeff * torch.tensor([data['ego_curr_status'][0],0.0])
            velocity[self.hist_steps] =  scale_coeff * torch.tensor([data['ego_curr_status'][0],0.0])
            for i in range(self.future_steps):
                velocity[self.hist_steps + i + 1] =  scale_coeff * data['ego_future_status'][i,2:4]
            score = torch.full((self.total_steps,), 0.9)
            agent_attr = torch.tensor([random_shape_generator(VEHICLE_TYPE,shape_offset)[0].item(), random_shape_generator(VEHICLE_TYPE,shape_offset)[1].item(), VEHICLE_TYPE])
            agent_time_mask = [True] * (self.total_steps)
            agent_time_mask = torch.tensor(agent_time_mask)
        else:
            current_position = data['agent_status'][mimic_agent_idx,self.hist_steps,:2]
            hist_position = data['agent_status'][mimic_agent_idx,: self.hist_steps,:2]
            fut_position = data['agent_status'][mimic_agent_idx,self.hist_steps + 1:,:2]
            
            heading = data['agent_status'][mimic_agent_idx,:,4]
            velocity = scale_coeff * data['agent_status'][mimic_agent_idx,:,2:4]
            score = data['agent_status'][mimic_agent_idx,:,5]
            
            agent_attr = data['agent_attrs'][mimic_agent_idx]
            agent_attr[0] = random_shape_generator(int(data['agent_attrs'][mimic_agent_idx,2].item()),shape_offset)[0].item()
            agent_attr[1] = random_shape_generator(int(data['agent_attrs'][mimic_agent_idx,2].item()),shape_offset)[1].item()
            
            agent_time_mask = data['agent_time_mask'][mimic_agent_idx]
            

        hist_diff = torch.cat(
            [scale_coeff * torch.diff(hist_position, dim=0), torch.zeros((1, 2))], dim=0
        )
        fut_diff = torch.cat(
            [torch.zeros((1, 2)), scale_coeff * torch.diff(fut_position, dim=0)], dim=0
        )
        #scale_coeff * torch.diff(fut_position, dim=0)
        scaled_position = torch.cat(
            [
                -torch.cumsum(hist_diff.flip(dims=[0]), dim=0).flip(dims=[0]) + current_position,
                current_position.reshape(1,-1),
                torch.cumsum(fut_diff, dim=0) + current_position,
            ],
            dim=0,
        )

        delta_angle = heading[self.hist_steps] - path_point.angle
        cos, sin = torch.cos(delta_angle), torch.sin(delta_angle)
        rot_mat = torch.tensor([[cos, -sin], [sin, cos]])

        new_position = (
            torch.matmul(scaled_position - current_position[None, :2], rot_mat)
            + torch.tensor([path_point.x,path_point.y])
        )
        new_heading = heading - heading[self.hist_steps] + path_point.angle
        new_heading = new_heading.reshape(-1)
        new_velocity = torch.matmul(velocity, rot_mat)

        agent_status = torch.zeros(self.total_steps, self.n_feature_agent_status)
        vx = new_velocity[0,:]
        vy = new_velocity[1,:]

        for i in range(self.total_steps):
            agent_status[i] = torch.tensor([new_position[i,0], 
                                          new_position[i,1], 
                                          new_velocity[i,0], new_velocity[i,1], new_heading[i], score[i]])    # x, y, vx, vy, yaw, score
        return agent_attr, agent_status, agent_time_mask
