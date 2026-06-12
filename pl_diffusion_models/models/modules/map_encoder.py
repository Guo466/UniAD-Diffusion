import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.embedding import PointsEncoder
from models.layers.common_layers import build_mlp


class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel,
        n_color,
        n_laneline_type,
        n_laneline_style,
        dim,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.polygon_encoder = PointsEncoder(polygon_channel, dim)
        # self.speed_limit_emb = nn.Sequential(
        #     nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim)
        # )
        self.color_emb = nn.Embedding(n_color + 1, dim)#+1 给none留一个
        self.laneline_type_emb = nn.Embedding(n_laneline_type + 1, dim)
        self.laneline_style_emb = nn.Embedding(n_laneline_style + 1, dim)      # 0~None; 1~centerline; 2~stopline; 3~crosswalk, 4~route

    def forward(self, data) -> torch.Tensor:
        '''
        laneline_pts torch.Size([2, 20, 20, 2])
        laneline_attrs torch.Size([2, 20, 4])
        laneline_mask torch.Size([2, 20])
        '''
        valid_mask = data["laneline_mask"]  # torch.Size([2, 10])
        n_pt = data["laneline_pts"].shape[2]    # 20
        # polygon_center = data["map"]["polygon_center"]
        #polygon_center = data["laneline_pts"][:,:,n_pt//2,:]   # torch.Size([bs, n_max_poly, 2])
        polygon_start = data["laneline_pts"][:,:,0:1,:]   # torch.Size([bs, n_max_poly, 1, 2])
        # polygon_type = data["map"]["polygon_type"].long()
        # polygon_id = data["laneline_attrs"][:,:,3].long()

        # polygon_on_route = data["map"]["polygon_on_route"].long()
        # polygon_tl_status = data["map"]["polygon_tl_status"].long()
        # polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        # polygon_speed_limit = data["map"]["polygon_speed_limit"]

        point_position = data["laneline_pts"]   # torch.Size([2, 20, 20, 2])
        point_vector = point_position[:,:,1:,:] - point_position[:,:,:-1,:] # torch.Size([2, 20, 19, 2])
        point_vector_norm = torch.norm(point_vector, dim=-1)  #  Size([2, 10, 19, 2])
        point_vector_norm += valid_mask.unsqueeze(-1).expand_as(point_vector_norm) + 1e-6 * torch.ones_like(point_vector_norm)
        # point_orientation = data["map"]["point_orientation"]
        point_orientation = torch.stack([
            point_vector[:,:,:,0] / point_vector_norm,
            point_vector[:,:,:,1] / point_vector_norm,
        ], dim=-1)
            
        rel_point_position = point_position - polygon_start
        polygon_feature = torch.cat(
            [
                rel_point_position, # (bs, max_poly, n_pt, 2)
                point_position,  # (bs, max_poly, n_pt, 2)
                torch.cat([point_orientation, point_orientation[:, :, -1:, :]], dim=2),      # (bs, max_poly, n_pt, 2)
            ],
            dim=-1
        )   # (bs, max_poly, n_pt, 6)

        bs, max_poly, n_pt, channel = polygon_feature.shape
        valid_mask = valid_mask.unsqueeze(-1).repeat(1, 1, n_pt).view(bs * max_poly, n_pt)
        polygon_feature = polygon_feature.reshape(bs * max_poly, n_pt, channel)
        x_polygon = self.polygon_encoder(polygon_feature, 1 - valid_mask).view(bs, max_poly, -1)    # torch.Size([2, 10, 128])
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_polygon).any():
            print('Error, nan in map encoder: polygon_encoder')

        if os.getenv("DEPLOY") != 'True':
            polygon_color = data["laneline_attrs"][:,:,0].long()  # torch.Size([bs, n_max_poly])
            polygon_laneline_type = data["laneline_attrs"][:,:,1].long()
            polygon_laneline_style = data["laneline_attrs"][:,:,2].long()
            x_color = self.color_emb(polygon_color)
            x_laneline_type = self.laneline_type_emb(polygon_laneline_type)
            x_laneline_style = self.laneline_style_emb(polygon_laneline_style)
        else:
            #####部署要看下#####
            x_color = data['polygon_color_feature']
            x_laneline_type = data['polygon_laneline_type_feature']
            x_laneline_style = data['polygon_laneline_style_feature']

        # x_pos_index = self.pos_index_emb(polygon_id)

        # x_type = self.type_emb(polygon_type)
        # x_on_route = self.on_route_emb(polygon_on_route)
        # x_tl_status = self.traffic_light_emb(polygon_tl_status)
        # x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        # x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
        #     polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        # )
        # x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        # x_polygon += x_color + x_edge_type + x_laneline_type + x_pos_index
        x_polygon = x_polygon + x_color + x_laneline_type + x_laneline_style
        # x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_polygon).any():
            print('Error, nan in map encoder')
            # raise ValueError("There is nan in map encoder")
        return x_polygon

class MapEncoder_navi(nn.Module):
    def __init__(
        self,
        polygon_channel,
        dim,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.polygon_encoder = PointsEncoder(polygon_channel+1, dim)

    def forward(self, data) -> torch.Tensor:
        '''
        laneline_pts torch.Size([B, 5, 200, 2])
        laneline_attrs torch.Size([B, 5, 3])
        laneline_mask torch.Size([B, 5])
        '''
        
        valid_mask = data["navitopo_mask"]  # torch.Size([B, 5])
        n_pt = data["navitopo_pts"].shape[2]    # 200
        # polygon_center = data["map"]["polygon_center"]
        #polygon_center = data["navitopo_pts"][:,:,n_pt//2,:]   # torch.Size([bs, n_max_poly, 2])
        polygon_start = data["navitopo_pts"][:,:,0:1,:]   # torch.Size([bs, n_max_poly, 1, 2])
        # polygon_type = data["map"]["polygon_type"].long()
        # polygon_id = data["laneline_attrs"][:,:,3].long()

        # polygon_on_route = data["map"]["polygon_on_route"].long()
        # polygon_tl_status = data["map"]["polygon_tl_status"].long()
        # polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        # polygon_speed_limit = data["map"]["polygon_speed_limit"]

        point_position = data["navitopo_pts"]   # torch.Size([2, 20, 20, 2])
        point_vector = point_position[:,:,1:,:] - point_position[:,:,:-1,:] # torch.Size([2, 20, 19, 2])
        point_vector_norm = torch.norm(point_vector, dim=-1)  #  Size([2, 10, 19, 2])
        point_vector_norm += valid_mask.unsqueeze(-1).expand_as(point_vector_norm) + 1e-6 * torch.ones_like(point_vector_norm)
        # point_orientation = data["map"]["point_orientation"]
        point_orientation = torch.stack([
            point_vector[:,:,:,0] / point_vector_norm,
            point_vector[:,:,:,1] / point_vector_norm,
        ], dim=-1)
            
        #rel_point_position = point_position - polygon_center.unsqueeze(2).expand_as(point_position)
        #torch.Size([32, 5, 200, 2]) torch.Size([32, 5, 1, 2])
        rel_point_position = point_position - polygon_start
        
        dist2change = data["navitopo_attrs"][:, :, 2:]
        point_position_x = point_position[:, :, :, 0]
        dist2change_mask = point_position_x > dist2change
        polygon_feature = torch.cat(
            [
                rel_point_position, # (bs, max_poly, n_pt, 2)
                point_position,  # (bs, max_poly, n_pt, 2)
                torch.cat([point_orientation, point_orientation[:, :, -1:, :]], dim=2),      # (bs, max_poly, n_pt, 2)
                dist2change_mask.unsqueeze(-1),
            ],
            dim=-1
        )   # (bs, max_poly, n_pt, 6)

        bs, max_poly, n_pt, channel = polygon_feature.shape
        valid_mask = valid_mask.unsqueeze(-1).repeat(1, 1, n_pt).view(bs * max_poly, n_pt)
        polygon_feature = polygon_feature.reshape(bs * max_poly, n_pt, channel)
        x_polygon = self.polygon_encoder(polygon_feature, 1 - valid_mask).view(bs, max_poly, -1)    # torch.Size([2, 10, 128]

        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_polygon).any():
            print('Error, nan in navi_map encoder')

        return x_polygon

class MapEncoder_route(nn.Module):
    def __init__(
        self,
        dim,
        polygon_channel,
        lane_attn_pool=True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.route_num_lanes = 8
        self.route_lane_emb_dim = 8
        self.polygon_encoder = PointsEncoder(polygon_channel, dim)
        # route_lane_recommend_mask: -1(invalid), 0(not recommended), 1(recommended)
        # value embedding + positional embedding, then project back to 8 dims
        self.route_lane_token_dim = 64  # lightweight internal width
        # Lane-slot / attribute embeddings
        self.route_lane_slot_emb = nn.Embedding(self.route_num_lanes, self.route_lane_token_dim)
        self.route_lane_rec_emb = nn.Embedding(2, self.route_lane_token_dim)   # recommended: 0/1
        self.route_lane_bus_emb = nn.Embedding(2, self.route_lane_token_dim)   # is_bus_lane: 0/1
        self.route_lane_type_num_groups = 9
        self.route_lane_type_emb = nn.Embedding(
            self.route_lane_type_num_groups,
            self.route_lane_token_dim,
            padding_idx=8,
        )
        self.route_lane_token_mlp = nn.Sequential(
            nn.Linear(4 * self.route_lane_token_dim, self.route_lane_token_dim),
            nn.GELU(),
            nn.Linear(self.route_lane_token_dim, self.route_lane_token_dim),
        )
        self.route_lane_out_proj = nn.Linear(self.route_lane_token_dim, self.dim)
        route_lane_type_group_map = self.create_lane_type_map()
        self.register_buffer('route_lane_type_group_map', route_lane_type_group_map)
        self.lane_attn_pool = lane_attn_pool
        if self.lane_attn_pool:
            self.route_lane_pool_score = nn.Sequential(
                nn.Linear(self.route_lane_token_dim, self.route_lane_token_dim),
                nn.GELU(),
                nn.Linear(self.route_lane_token_dim, 1),
            )

    def create_lane_type_map(self):
        # lane_type_main is remapped into coarse semantic groups + padding
        # group ids:
        #   0: straight
        #   1: left-related
        #   2: right-related
        #   3: u-turn-related
        #   4: mixed / both-direction
        #   5: bus
        #   6: empty / variable
        #   7: reserved / other
        #   8: padding for invalid lanes (NULL)
        lane_type_map = torch.full((256,), 7, dtype=torch.long)  # default: reserved/other
        lane_type_map[255] = 8  # NULL -> padding

        # straight
        lane_type_map[0] = 0

        # left-related
        for t in [1, 2, 11, 13, 14, 16, 20]:
            lane_type_map[t] = 1

        # right-related
        for t in [3, 4, 12, 17, 19]:
            lane_type_map[t] = 2

        # u-turn-related
        for t in [5, 8, 9, 10, 18]:
            lane_type_map[t] = 3

        # mixed / both-direction
        for t in [6, 7]:
            lane_type_map[t] = 4

        # bus
        lane_type_map[21] = 5

        # empty / variable
        for t in [22, 23]:
            lane_type_map[t] = 6
        return lane_type_map


    def forward(self, data) -> torch.Tensor:
        n_pt = data["route_pts"].shape[2]
        # invalid的时候，route pts均为0，dataset已经处理过了。不依赖mask
        valid_mask = torch.zeros((data["route_pts"].shape[0], data["route_pts"].shape[1]))
        polygon_start = data["route_pts"][:,:,0:1,:]   # torch.Size([bs, n_max_poly, 1, 2])
        point_position = data["route_pts"]   # torch.Size([2, 20, 20, 2])
        point_vector = point_position[:,:,1:,:] - point_position[:,:,:-1,:] # torch.Size([2, 20, 19, 2])
        point_vector_norm = torch.norm(point_vector, dim=-1)  #  Size([2, 10, 19, 2])
        point_vector_norm += 1e-6 * torch.ones_like(point_vector_norm)

        point_orientation = torch.stack([
            point_vector[:,:,:,0] / point_vector_norm,
            point_vector[:,:,:,1] / point_vector_norm,
        ], dim=-1)
            
        rel_point_position = point_position - polygon_start

        route_lane_attrs = data.get("route_lane_attrs", None)
        route_lane_recommend_mask = data.get("route_lane_recommend_mask", None)

        polygon_feature = torch.cat(
            [
                rel_point_position,  # (bs, max_poly, n_pt, 2)
                point_position,  # (bs, max_poly, n_pt, 2)
                torch.cat([point_orientation, point_orientation[:, :, -1:, :]], dim=2),  # (bs, max_poly, n_pt, 2)
            ],
            dim=-1,
        )  # (bs, max_poly, n_pt, 6)
        if (route_lane_attrs is not None):
            # route_lane_attrs: (bs, max_poly, 8, 4)
            # [valid, recommended, lane_type_main, is_bus_lane]
            valid = route_lane_attrs[..., 0].long()
            recommended = route_lane_attrs[..., 1].long()
            lane_type_main = route_lane_attrs[..., 2].long()
            is_bus_lane = route_lane_attrs[..., 3].long()

            bs, max_poly, num_lanes = valid.shape

            # -----------------------------------
            # Remap raw lane_type_main to groups
            # -----------------------------------
            # clamp first to be safe, then remap
            lane_type_main = lane_type_main.clamp(0, 255)
            lane_type_group = self.route_lane_type_group_map[lane_type_main]  # (bs, max_poly, 8)

            # invalid lanes -> padding category
            lane_type_group = lane_type_group.masked_fill(valid == 0, 8)

            # -----------------------------------
            # Per-lane embeddings
            # -----------------------------------
            slot_idx = torch.arange(num_lanes, device=valid.device).view(1, 1, -1)
            slot_emb = self.route_lane_slot_emb(slot_idx).expand(bs, max_poly, -1, -1)

            rec_emb = self.route_lane_rec_emb(recommended.clamp(0, 1))
            type_emb = self.route_lane_type_emb(lane_type_group)
            bus_emb = self.route_lane_bus_emb(is_bus_lane.clamp(0, 1))

            # -----------------------------------
            # Build lane tokens
            # -----------------------------------
            lane_token = torch.cat(
                [slot_emb, rec_emb, type_emb, bus_emb],
                dim=-1,
            )  # (bs, max_poly, 8, 4 * lane_token_dim)

            lane_token = self.route_lane_token_mlp(lane_token)  # (bs, max_poly, 8, lane_token_dim)

            # -----------------------------------
            # Mask invalid lanes
            # -----------------------------------
            lane_valid = valid.unsqueeze(-1).float()  # (bs, max_poly, 8, 1)
            lane_token = lane_token * lane_valid

            # -----------------------------------
            # Masked mean pooling over lane slots
            # -----------------------------------
            if self.lane_attn_pool:
                scores = self.route_lane_pool_score(lane_token).squeeze(-1)  # (bs, max_poly, 8)
                scores = scores.masked_fill(valid == 0, -1e4)
                attn = torch.softmax(scores, dim=2).unsqueeze(-1)  # (bs, max_poly, 8, 1)
                lane_pooled = (lane_token * attn).sum(dim=2)        # (bs, max_poly, token_dim)
            else:
                denom = lane_valid.sum(dim=2).clamp(min=1e-6)  # (bs, max_poly, 1)
                lane_pooled = lane_token.sum(dim=2) / denom    # (bs, max_poly, lane_token_dim)

            # -----------------------------------
            # Final projection to model width
            # -----------------------------------
            x_lane = self.route_lane_out_proj(lane_pooled)  # (bs, max_poly, 256)

        else:
            bs, max_poly, _, _ = point_position.shape
            x_lane = torch.zeros(
                (bs, max_poly, self.dim),
                dtype=point_position.dtype,
                device=point_position.device,
            )

        channel = polygon_feature.shape[-1]
        valid_mask = valid_mask.unsqueeze(-1).repeat(1, 1, n_pt).view(bs * max_poly, n_pt)
        polygon_feature = polygon_feature.reshape(bs * max_poly, n_pt, channel)
        x_polygon = self.polygon_encoder(polygon_feature, 1 - valid_mask).view(bs, max_poly, -1)    # torch.Size([2, 10, 128])

        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_polygon).any():
            print('Error, nan in navi_map encoder')
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_lane).any():
            print('Error, nan in route lane encoder')
        x_mask = torch.all(data["route_pts"] == 0, dim=(2, 3)).float()
        return x_polygon, x_lane, x_mask

class MapEncoder_occ(nn.Module):
    def __init__(
        self,
        polygon_channel,
        n_type,
        dim,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.polygon_encoder = PointsEncoder(polygon_channel, dim)
        # self.type_emb = nn.Embedding(n_type, dim)
        # self.attr_emb = nn.Sequential(
        #     nn.Linear(3, dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim)
        # )

    def forward(self, data) -> torch.Tensor:
        valid_mask = data["occ_polygons_mask"]  # (bs, N)
        occ_polygons_pts = data["occ_polygons_pts"]  # (bs, N, P, 2)
        
        bs, max_poly, n_pt, channel = occ_polygons_pts.shape
        valid_mask = valid_mask.unsqueeze(-1).repeat(1, 1, n_pt).view(bs * max_poly, n_pt)
        occ_polygons_pts = occ_polygons_pts.reshape(bs * max_poly, n_pt, channel)
        x_polygon = self.polygon_encoder(occ_polygons_pts, 1 - valid_mask).view(bs, max_poly, -1)
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_polygon).any():
            print('Error, nan in occ encoder: polygon_encoder')

        # if os.getenv("DEPLOY") != 'True':
        #     polygon_type = data["occ_polygons_attrs"][:, :, 2].long()
        #     polygon_center = data["occ_polygons_attrs"][:, :, :2]
        #     polygon_area = data["occ_polygons_attrs"][:, :, 3].unsqueeze(-1)
        #     x_type = self.type_emb(polygon_type)
        #     x_attr = self.attr_emb(torch.cat([polygon_center, polygon_area], dim=-1))
        # else:
        #     pass

        # x_polygon = x_polygon + x_type + x_attr
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_polygon).any():
            print("Error, nan in map occ encoder")
            # raise ValueError("There is nan in map encoder")
        return x_polygon