import os

import torch
import torch.nn as nn
import math

from models.layers.common_layers import build_mlp
from models.layers.transformer import Transformer

class AgentEncoder_path(nn.Module):
    def __init__(
        self,
        n_agent_cls,
        n_traffic_light_cls,  # 0: unknown | 1: invalid | 2: off | 3: green | 4: yellow | 5. red
        n_car_light_cls,      # 0: unknown | 1: off | 2: left | 3: right
        ego_channel,
        agent_channel,
        dim,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        use_agent_history,
        use_state_attn_encoder_ego,
        use_state_attn_encoder_agent,
        hist_steps,
        state_dropout=0.75,
        ego_state_emb_depth = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ego_channel = ego_channel
        self.agent_channel = agent_channel
        self.use_agent_history = use_agent_history
        self.use_state_attn_encoder_ego = use_state_attn_encoder_ego
        self.use_state_attn_encoder_agent = use_state_attn_encoder_agent
        self.hist_steps = hist_steps

        if not use_agent_history:
            if self.use_state_attn_encoder_agent:
                self.agent_feat_emb = StateAttentionEncoder(
                    agent_channel, dim, state_dropout
                ) # state_dropout=0.75
            else:
                # NOTE 有zero padding
                self.agent_feat_emb = nn.Sequential(
                    build_mlp(agent_channel, [dim] * 2),
                    nn.LayerNorm(dim)
                )
        else:
            self.agent_proj = build_mlp(agent_channel, [dim] * 2)
            self.agent_time_pe = PositionalEncoding(dim=dim, max_len=20)
            self.agent_feat_emb = Transformer(
                d_model = dim,
                nhead = nhead,
                num_encoder_layers = num_encoder_layers,
                num_decoder_layers = None,
                dim_feedforward = dim * 4,
                dropout=0.0,
            )

        if self.use_state_attn_encoder_ego:
            self.ego_state_emb = StateAttentionEncoder(
                ego_channel, dim, state_dropout
            ) # state_dropout=0.75            
        else:
            print(f"ego_state_emb_depth: {ego_state_emb_depth}")
            self.ego_state_emb = build_mlp(ego_channel, [dim] * ego_state_emb_depth)
            self.ego_state_norm = nn.LayerNorm(dim)

        self.type_emb = nn.Embedding(n_agent_cls, dim)    # ego is another cls
        self.traffice_light_embed = nn.Embedding(n_traffic_light_cls, dim)  # 0: unknown | 1: invalid | 2: off | 3: green | 4: yellow | 5. red
        self.car_light_embed = nn.Embedding(n_car_light_cls, dim)  # 0: unknown | 1: off | 2: left | 3: right

    @staticmethod
    def to_vector(feat, valid_mask):
        vec_mask = valid_mask[..., 1:]

        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)
        #差分会导致少一帧，现在丢掉0帧。mask只依赖当前t的mask，不再依赖t+1/t-1
        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    def forward(self, data):
        '''
        agent_attrs torch.Size([bs, n_max_agent, 3]): size_y, size_x, cls
        agent_status torch.Size([bs, n_max_agent, hist+1+future, 6]):  x, y, vx, vy, yaw, score
        agent_time_mask torch.Size([bs, n_max_agent, hist+1+future]) 
        '''
        # 1. agent时序融合
        T = self.hist_steps + 1     # 20 + 1
        position = data["agent_status"][:,:,:T,:2]   # torch.Size([bs, max_agent, 21, 2])
        velocity = data["agent_status"][:,:,:T,2:4]   # torch.Size([bs, max_agent, 21, 2])
        heading = data["agent_status"][:,:,:T,4]   # torch.Size([bs, max_agent, 21])
        score = data["agent_status"][:,:,1:T,5].unsqueeze(-1) # torch.Size([bs, max_agent, 20, 1])
        shape = data["agent_attrs"][:,:,:2].unsqueeze(2).repeat(1,1,T-1,1)  # torch.Size([bs, max_agent, 20, 2])
        agent_time_mask = data["agent_time_mask"][:,:,:T]   # [bs, n_max_agent, hist+1]，True表示有效
        if os.getenv("DEPLOY") != 'True':
            #差分在这里
            #新改mask的只依赖于t
            #这里虽然用了t，但还是丢了最早帧，因为后边要和ego做交互，ego是差分少一帧的
            agent_time_mask = agent_time_mask.bool()
            agent_time_mask_vec = agent_time_mask[..., 1:]
            heading_vec = self.to_vector(heading, agent_time_mask)  # [bs, n_max_agent, hist]
            agent_feature = torch.cat(
                [
                    self.to_vector(position, agent_time_mask),   #  torch.Size([bs, n_max_agent, 20, 2])
                    self.to_vector(velocity, agent_time_mask),   #  torch.Size([bs, n_max_agent, 20, 2])
                    torch.stack([heading_vec.cos(), heading_vec.sin()],dim=-1), #  torch.Size([bs, n_max_agent, 20, 2])
                    shape, # torch.Size([bs, max_agent, 20, 2])
                    score, # torch.Size([bs, max_agent, 20, 1])
                    agent_time_mask_vec.float().unsqueeze(-1), # torch.Size([bs, max_agent, 20, 1])
                ],
                dim=-1,
            )   # (bs, n_max_agent, 20, 10)
        else:
            #部署如果需要记得改。
            agent_time_mask_vec = agent_time_mask[..., 1:]  # 只依赖当前帧的mask [bs, n_max_agent, 20]
            agent_time_mask_vec_float = agent_time_mask_vec.float()
            # 直接使用下一帧的值，而不是差分
            position_next = position[:, :, 1:, ...] * agent_time_mask_vec_float.unsqueeze(-1)
            velocity_next = velocity[:, :, 1:, ...] * agent_time_mask_vec_float.unsqueeze(-1)
            heading_next = heading[:, :, 1:] * agent_time_mask_vec_float

            agent_feature = torch.cat(
                [
                    position_next,
                    velocity_next,
                    torch.stack([heading_next.cos(), heading_next.sin()], dim=-1),
                    shape,
                    score,
                    agent_time_mask_vec_float.unsqueeze(-1),
                ],
                dim=-1,
            )
        bs, n_agent, n_time, n_feat = agent_feature.shape

        if os.getenv("DEPLOY") != 'True':
            agent_valid_mask = agent_time_mask.any(-1)  # 选出非padding的部分, (bs, n_max_agent)
            # agent_feature[agent_valid_mask].shape:  torch.Size([n_valid_agent, 20, 10]))
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs = self.agent_proj(agent_feature[agent_valid_mask])  # torch.Size([n_valid_agent, 20, 128])
            agent_hs = self.agent_time_pe(agent_hs) # torch.Size([n_valid_agent, 20, 128])
            key_padding_mask = agent_time_mask_vec[agent_valid_mask]    # torch.Size([n_valid_agent, 20]), True表示有效，False表示无效
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs, _ = self.agent_feat_emb(
                    src=agent_hs, 
                    tgt=None, 
                    src_padding_mask=~key_padding_mask,
                    tgt_padding_mask=None,
                )   # 每一个agent，时序信息做融合 [289, 20, 128]

            device = agent_feature.device
            x_agent = torch.zeros(bs, n_agent, self.dim, device=device) #[32, 50, 128]
            x_agent[agent_valid_mask] = agent_hs[:,-1,:]    # 提取agent当前帧的hidden state
            
            x_agent_his =  torch.zeros(bs, n_agent, n_time, self.dim, device=device)  # [32, 50, 20, 128]
            x_agent_his[agent_time_mask_vec] = agent_hs[key_padding_mask]  # [32, 50, 20] [289, 20]    
        else:
            agent_valid_mask = torch.max(agent_time_mask, dim=-1, keepdim=True)[0] # [32, 50, 1]
            agent_feature = agent_feature.view(bs * n_agent, n_time, n_feat)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs = self.agent_proj(agent_feature)
            agent_hs = self.agent_time_pe(agent_hs) # [1600, 20, 128]
            agent_time_mask_vec = agent_time_mask_vec.view(bs * n_agent, n_time) #[1600, 20]
            
            # a. This method makes key_padding_mask contain all one rows and self.agent_feat_emb will output nan value.
            # key_padding_mask = 1.0 - agent_time_mask_vec.view(bs * n_agent, n_time)

            # b. This method results in coredump in running with C++.
            # add_tensor = torch.where(torch.sum(agent_time_mask_vec, dim=-1, keepdim=True) > 0.0, 1.0, 0.0)

            # c. This method makes deployed model running with C++.
            add_tensor = torch.max(agent_time_mask_vec, dim=-1, keepdim=True)[0]    # torch.Size([50, 1]), torch.float32

            key_padding_mask = add_tensor.expand_as(agent_time_mask_vec) - agent_time_mask_vec
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs, _ = self.agent_feat_emb(
                    src=agent_hs, 
                    tgt=None, 
                    src_padding_mask=key_padding_mask.bool(),
                    tgt_padding_mask=None,
                )   # 每一个agent，时序信息做融合
            agent_hs = agent_hs.view(bs, n_agent, n_time, -1) #[32, 50, 20, 128]
            key_padding_mask = key_padding_mask.view(bs, n_agent, n_time).unsqueeze(-1)
            x_agent = agent_hs[:, :, n_time-1, :] * agent_valid_mask

            agent_time_mask_vec = agent_time_mask_vec.view(bs, n_agent, n_time)
            x_agent_his = agent_hs * agent_time_mask_vec.unsqueeze(-1) # [32, 50, 20, 128]

        if os.getenv("DEPLOY") != 'True':
            category = data["agent_attrs"][:,:,2].long()
            x_type = self.type_emb(category)    # torch.Size([bs, 1 + max_agent, 128])

            object_mask = ~agent_valid_mask
        else:
            # @wenbo 部署这里要改category保留，traffic_light和car_light直接去掉。他车没有这个属性.
            x_type = data["category_feature"]    # torch.Size([bs, 1 + max_agent, 128])
            x_traffic_light = data["traffic_light_feature"] 
            x_car_light = data["car_light_feature"]       
            object_mask = torch.cat([
                torch.zeros((bs, 1), dtype=torch.float).cuda(), 
                1 - agent_valid_mask.squeeze(-1)], dim=1)

        x_type_agent_his = x_type[:, :].unsqueeze(2).expand_as(x_agent_his)
        x_agent_his = x_agent_his + x_type_agent_his
        x_agent = x_agent + x_type
        
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_agent).any():
            print('Error, nan in agent encoder')
            # raise ValueError("There is nan in agent encoder")
        if os.getenv("DEPLOY") != 'True':
            return x_agent, object_mask, x_agent_his, ~agent_time_mask_vec 
        else : 
            return x_agent, object_mask, x_agent_his, (1 - agent_time_mask_vec)


class EgoEncoder_path(nn.Module):
    """自车历史轨迹编码（与 dif_114f 一致），输入 data['ego_status'] 形状 [B, T, 7]。"""

    def __init__(
        self,
        n_agent_cls,
        n_traffic_light_cls,
        n_car_light_cls,
        ego_channel,
        agent_channel,
        dim,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        use_agent_history,
        use_state_attn_encoder_ego,
        use_state_attn_encoder_agent,
        hist_steps,
        state_dropout=0.75,
        ego_state_emb_depth=2,
        ego_time_max_len=128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ego_channel = ego_channel
        self.agent_channel = agent_channel
        self.use_agent_history = use_agent_history
        self.use_state_attn_encoder_ego = use_state_attn_encoder_ego
        self.use_state_attn_encoder_agent = use_state_attn_encoder_agent

        if not use_agent_history:
            if self.use_state_attn_encoder_agent:
                self.agent_feat_emb = StateAttentionEncoder(
                    agent_channel, dim, state_dropout
                )
            else:
                self.agent_feat_emb = nn.Sequential(
                    build_mlp(agent_channel, [dim] * 2),
                    nn.LayerNorm(dim),
                )
        else:
            self.agent_proj = build_mlp(agent_channel, [dim] * 2)
            self.agent_time_pe = PositionalEncoding(dim=dim, max_len=ego_time_max_len)
            self.agent_feat_emb = Transformer(
                d_model=dim,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=None,
                dim_feedforward=dim * 4,
                dropout=0.0,
            )

        if self.use_state_attn_encoder_ego:
            self.ego_state_emb = StateAttentionEncoder(ego_channel, dim, state_dropout)
        else:
            print(f"ego_state_emb_depth: {ego_state_emb_depth}")
            self.ego_state_emb = build_mlp(ego_channel, [dim] * ego_state_emb_depth)
            self.ego_state_norm = nn.LayerNorm(dim)

        self.type_emb = nn.Embedding(n_agent_cls, dim)
        self.traffice_light_embed = nn.Embedding(n_traffic_light_cls, dim)
        self.car_light_embed = nn.Embedding(n_car_light_cls, dim)

    @staticmethod
    def to_vector(feat, valid_mask):
        vec_mask = valid_mask[..., 1:]
        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)
        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    def forward(self, data):
        T = data["ego_status"].shape[1]
        ego_status = data["ego_status"][:, :T, :]
        pos_ego = ego_status[:, :, :2]
        v_ego = ego_status[:, :, 2:4]
        agent_time_mask = ego_status[:, :, -1]
        bs = pos_ego.shape[0]
        pos_ego = pos_ego.unsqueeze(1)
        v_ego = v_ego.unsqueeze(1)
        agent_time_mask = agent_time_mask.unsqueeze(1)
        device = pos_ego.device
        if os.getenv("DEPLOY") != "True":
            agent_time_mask = agent_time_mask.bool()
            agent_time_mask_vec = agent_time_mask[..., 1:]
            agent_feature = torch.cat(
                [
                    self.to_vector(pos_ego, agent_time_mask),
                    self.to_vector(v_ego, agent_time_mask),
                    torch.zeros(bs, 1, T - 1, 5, device=device),
                    agent_time_mask_vec.float().unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            agent_time_mask_vec = agent_time_mask[..., 1:]
            agent_time_mask_vec_float = agent_time_mask_vec.float()
            position_next = pos_ego[:, :, 1:, ...] * agent_time_mask_vec_float.unsqueeze(-1)
            agent_feature = torch.cat(
                [
                    position_next,
                    torch.zeros(bs, 1, T - 1, 7, device=device),
                    agent_time_mask_vec_float.unsqueeze(-1),
                ],
                dim=-1,
            )
        bs, n_agent, n_time, n_feat = agent_feature.shape

        if os.getenv("DEPLOY") != "True":
            agent_valid_mask = agent_time_mask.any(-1)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs = self.agent_proj(agent_feature[agent_valid_mask])
            agent_hs = self.agent_time_pe(agent_hs)
            key_padding_mask = agent_time_mask_vec[agent_valid_mask]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs, _ = self.agent_feat_emb(
                    src=agent_hs,
                    tgt=None,
                    src_padding_mask=~key_padding_mask,
                    tgt_padding_mask=None,
                )

            device = agent_feature.device
            x_agent = torch.zeros(bs, n_agent, self.dim, device=device)
            x_agent[agent_valid_mask] = agent_hs[:, -1, :]

            x_agent_his = torch.zeros(bs, n_agent, n_time, self.dim, device=device)
            x_agent_his[agent_time_mask_vec] = agent_hs[key_padding_mask]
        else:
            agent_valid_mask = torch.max(agent_time_mask, dim=-1, keepdim=True)[0]
            agent_feature = agent_feature.view(bs * n_agent, n_time, n_feat)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs = self.agent_proj(agent_feature)
            agent_hs = self.agent_time_pe(agent_hs)
            agent_time_mask_vec = agent_time_mask_vec.view(bs * n_agent, n_time)

            add_tensor = torch.max(agent_time_mask_vec, dim=-1, keepdim=True)[0]
            key_padding_mask = add_tensor.expand_as(agent_time_mask_vec) - agent_time_mask_vec
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs, _ = self.agent_feat_emb(
                    src=agent_hs,
                    tgt=None,
                    src_padding_mask=key_padding_mask.bool(),
                    tgt_padding_mask=None,
                )
            agent_hs = agent_hs.view(bs, n_agent, n_time, -1)
            key_padding_mask = key_padding_mask.view(bs, n_agent, n_time).unsqueeze(-1)
            x_agent = agent_hs[:, :, n_time - 1, :] * agent_valid_mask

            agent_time_mask_vec = agent_time_mask_vec.view(bs, n_agent, n_time)
            x_agent_his = agent_hs * agent_time_mask_vec.unsqueeze(-1)

        if os.getenv("DEPLOY") != "True":
            object_mask = torch.zeros((bs, 1), dtype=torch.float, device=device)
        else:
            object_mask = torch.zeros((bs, 1), dtype=torch.float).cuda()
        if os.getenv("DEPLOY") != "True" and torch.isnan(x_agent).any():
            print("Error, nan in ego encoder path")
        if os.getenv("DEPLOY") != "True":
            return x_agent, object_mask, x_agent_his, ~agent_time_mask_vec
        return x_agent, object_mask, x_agent_his, (1 - agent_time_mask_vec)


class AgentEncoder_traj(nn.Module):
    def __init__(
        self,
        n_agent_cls,
        n_traffic_light_cls,  # 0: unknown | 1: invalid | 2: off | 3: green | 4: yellow | 5. red
        n_car_light_cls,      # 0: unknown | 1: off | 2: left | 3: right
        ego_channel,
        agent_channel,
        dim,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        use_agent_history,
        use_state_attn_encoder_ego,
        use_state_attn_encoder_agent,
        hist_steps,
        state_dropout=0.75,
        ego_state_emb_depth = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ego_channel = ego_channel
        self.agent_channel = agent_channel
        self.use_agent_history = use_agent_history
        self.use_state_attn_encoder_ego = use_state_attn_encoder_ego
        self.use_state_attn_encoder_agent = use_state_attn_encoder_agent
        self.hist_steps = hist_steps

        if not use_agent_history:
            if self.use_state_attn_encoder_agent:
                self.agent_feat_emb = StateAttentionEncoder(
                    agent_channel, dim, state_dropout
                ) # state_dropout=0.75
            else:
                # NOTE 有zero padding
                self.agent_feat_emb = nn.Sequential(
                    build_mlp(agent_channel, [dim] * 2),
                    nn.LayerNorm(dim)
                )
        else:
            self.agent_proj = build_mlp(agent_channel, [dim] * 2)
            self.agent_time_pe = PositionalEncoding(dim=dim, max_len=20)
            self.agent_feat_emb = Transformer(
                d_model = dim,
                nhead = nhead,
                num_encoder_layers = num_encoder_layers,
                num_decoder_layers = None,
                dim_feedforward = dim * 4,
                dropout=0.0,
            )

        if self.use_state_attn_encoder_ego:
            self.ego_state_emb = StateAttentionEncoder(
                ego_channel, dim, state_dropout
            ) # state_dropout=0.75            
        else:
            print(f"ego_state_emb_depth: {ego_state_emb_depth}")
            self.ego_state_emb = build_mlp(ego_channel, [dim] * ego_state_emb_depth)
            self.ego_state_norm = nn.LayerNorm(dim)

        self.type_emb = nn.Embedding(n_agent_cls, dim)    # ego is another cls
        self.traffice_light_embed = nn.Embedding(n_traffic_light_cls, dim)  # 0: unknown | 1: invalid | 2: off | 3: green | 4: yellow | 5. red
        self.car_light_embed = nn.Embedding(n_car_light_cls, dim)  # 0: unknown | 1: off | 2: left | 3: right

    @staticmethod
    def to_vector(feat, valid_mask):
        vec_mask = valid_mask[..., 1:]

        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)
        #差分会导致少一帧，现在丢掉0帧。mask只依赖当前t的mask，不再依赖t+1/t-1
        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    def forward(self, data):
        '''
        agent_attrs torch.Size([bs, n_max_agent, 3]): size_y, size_x, cls
        agent_status torch.Size([bs, n_max_agent, hist+1+future, 6]):  x, y, vx, vy, yaw, score
        agent_time_mask torch.Size([bs, n_max_agent, hist+1+future]) 
        '''
        # 1. agent时序融合
        T = self.hist_steps + 1     # 20 + 1
        position = data["agent_status"][:,:,:T,:2]   # torch.Size([bs, max_agent, 21, 2])
        velocity = data["agent_status"][:,:,:T,2:4]   # torch.Size([bs, max_agent, 21, 2])
        heading = data["agent_status"][:,:,:T,4]   # torch.Size([bs, max_agent, 21])
        score = data["agent_status"][:,:,1:T,5].unsqueeze(-1) # torch.Size([bs, max_agent, 20, 1])
        shape = data["agent_attrs"][:,:,:2].unsqueeze(2).repeat(1,1,T-1,1)  # torch.Size([bs, max_agent, 20, 2])
        agent_time_mask = data["agent_time_mask"][:,:,:T]   # [bs, n_max_agent, hist+1]，True表示有效
        if os.getenv("DEPLOY") != 'True':
            #差分在这里
            #新改mask的只依赖于t
            #这里虽然用了t，但还是丢了最早帧，因为后边要和ego做交互，ego是差分少一帧的
            agent_time_mask = agent_time_mask.bool()
            agent_time_mask_vec = agent_time_mask[..., 1:]
            heading_vec = self.to_vector(heading, agent_time_mask)  # [bs, n_max_agent, hist]
            agent_feature = torch.cat(
                [
                    self.to_vector(position, agent_time_mask),   #  torch.Size([bs, n_max_agent, 20, 2])
                    self.to_vector(velocity, agent_time_mask),   #  torch.Size([bs, n_max_agent, 20, 2])
                    torch.stack([heading_vec.cos(), heading_vec.sin()],dim=-1), #  torch.Size([bs, n_max_agent, 20, 2])
                    shape, # torch.Size([bs, max_agent, 20, 2])
                    score, # torch.Size([bs, max_agent, 20, 1])
                    agent_time_mask_vec.float().unsqueeze(-1), # torch.Size([bs, max_agent, 20, 1])
                ],
                dim=-1,
            )   # (bs, n_max_agent, 20, 10)
        else:
            #部署如果需要记得改。
            agent_time_mask_vec = agent_time_mask[..., 1:]  # 只依赖当前帧的mask [bs, n_max_agent, 20]
            agent_time_mask_vec_float = agent_time_mask_vec.float()
            # 直接使用下一帧的值，而不是差分
            position_next = position[:, :, 1:, ...] * agent_time_mask_vec_float.unsqueeze(-1)
            velocity_next = velocity[:, :, 1:, ...] * agent_time_mask_vec_float.unsqueeze(-1)
            heading_next = heading[:, :, 1:] * agent_time_mask_vec_float

            agent_feature = torch.cat(
                [
                    position_next,
                    velocity_next,
                    torch.stack([heading_next.cos(), heading_next.sin()], dim=-1),
                    shape,
                    score,
                    agent_time_mask_vec_float.unsqueeze(-1),
                ],
                dim=-1,
            )
        bs, n_agent, n_time, n_feat = agent_feature.shape
        # if not self.use_agent_history:
        #     if self.use_state_attn_encoder_agent:
        #         x_agent = self.agent_feat_emb(agent_feature.reshape(-1, n_agent_feat), undrop_channel=2)    # torch.Size([bs * n, 128])
        #     else:
        #         device = agent_feature.device
        #         x_agent = torch.zeros(bs, n_agent, self.dim, device=device)
        #         x_agent[~valid_mask] = self.agent_feat_emb(agent_feature[~valid_mask])
        #         # x_agent = self.agent_feat_emb(agent_feature.reshape(-1, n_agent_feat))

        if os.getenv("DEPLOY") != 'True':
            agent_valid_mask = agent_time_mask.any(-1)  # 选出非padding的部分, (bs, n_max_agent)
            # agent_feature[agent_valid_mask].shape:  torch.Size([n_valid_agent, 20, 10]))
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs = self.agent_proj(agent_feature[agent_valid_mask])  # torch.Size([n_valid_agent, 20, 128])
            agent_hs = self.agent_time_pe(agent_hs) # torch.Size([n_valid_agent, 20, 128])
            key_padding_mask = agent_time_mask_vec[agent_valid_mask]    # torch.Size([n_valid_agent, 20]), True表示有效，False表示无效
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs, _ = self.agent_feat_emb(
                    src=agent_hs, 
                    tgt=None, 
                    src_padding_mask=~key_padding_mask,
                    tgt_padding_mask=None,
                )   # 每一个agent，时序信息做融合 [289, 20, 128]

            device = agent_feature.device
            x_agent = torch.zeros(bs, n_agent, self.dim, device=device) #[32, 50, 128]
            x_agent[agent_valid_mask] = agent_hs[:,-1,:]    # 提取agent当前帧的hidden state
            
            x_agent_his =  torch.zeros(bs, n_agent, n_time, self.dim, device=device)  # [32, 50, 20, 128]
            x_agent_his[agent_time_mask_vec] = agent_hs[key_padding_mask]  # [32, 50, 20] [289, 20]    
        else:
            agent_valid_mask = torch.max(agent_time_mask, dim=-1, keepdim=True)[0] # [32, 50, 1]
            agent_feature = agent_feature.view(bs * n_agent, n_time, n_feat)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs = self.agent_proj(agent_feature)
            agent_hs = self.agent_time_pe(agent_hs) # [1600, 20, 128]
            agent_time_mask_vec = agent_time_mask_vec.view(bs * n_agent, n_time) #[1600, 20]
            
            # a. This method makes key_padding_mask contain all one rows and self.agent_feat_emb will output nan value.
            # key_padding_mask = 1.0 - agent_time_mask_vec.view(bs * n_agent, n_time)

            # b. This method results in coredump in running with C++.
            # add_tensor = torch.where(torch.sum(agent_time_mask_vec, dim=-1, keepdim=True) > 0.0, 1.0, 0.0)

            # c. This method makes deployed model running with C++.
            add_tensor = torch.max(agent_time_mask_vec, dim=-1, keepdim=True)[0]    # torch.Size([50, 1]), torch.float32

            key_padding_mask = add_tensor.expand_as(agent_time_mask_vec) - agent_time_mask_vec
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                agent_hs, _ = self.agent_feat_emb(
                    src=agent_hs, 
                    tgt=None, 
                    src_padding_mask=key_padding_mask.bool(),
                    tgt_padding_mask=None,
                )   # 每一个agent，时序信息做融合
            agent_hs = agent_hs.view(bs, n_agent, n_time, -1) #[32, 50, 20, 128]
            key_padding_mask = key_padding_mask.view(bs, n_agent, n_time).unsqueeze(-1)
            x_agent = agent_hs[:, :, n_time-1, :] * agent_valid_mask

            agent_time_mask_vec = agent_time_mask_vec.view(bs, n_agent, n_time)
            x_agent_his = agent_hs * agent_time_mask_vec.unsqueeze(-1) # [32, 50, 20, 128]

        # 2. 自车特征投影
        ego_feature = data["ego_curr_status"]   # (bs, 2)
        ego_feature = torch.cat([
            torch.zeros(bs, 3).cuda(), 
            ego_feature], dim=-1)   # add x, y, yaw, from (bs, 4) to (bs, 7); (x, y, yaw, vx, yaw_rate, traffic_light, ego_light)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            if self.use_state_attn_encoder_ego:
                x_ego = self.ego_state_emb(ego_feature[:, :5], undrop_channel=3) # from torch.Size([bs, 5]) to torch.Size([bs, 128])
            else:
                x_ego = self.ego_state_emb(ego_feature[:, :5]) # from torch.Size([bs, 5]) to torch.Size([bs, 128])
            x_ego = self.ego_state_norm(x_ego)

        # 3. 拼接自车和agent特征
        x_agent = torch.cat([
            x_ego.unsqueeze(1), 
            x_agent], dim=1)   # torch.Size([bs, 1 + max_agent, 128])

        if os.getenv("DEPLOY") != 'True':
            # ego的cls设为0
            category = data["agent_attrs"][:,:,2].long()
            traffic_light = data["ego_curr_status"][...,2].long().unsqueeze(-1)     # v, yaw_rate, traffic_light, ego_light
            car_light = data["ego_curr_status"][...,3].long().unsqueeze(-1)         # v, yaw_rate, traffic_light, ego_light
            category = torch.cat([
                torch.zeros((bs, 1), dtype=torch.long).cuda(),
                category], dim=1)   # torch.Size([bs, 101])
            x_type = self.type_emb(category)    # torch.Size([bs, 1 + max_agent, 128])

            traffic_light = torch.cat([
                traffic_light,
                torch.zeros((bs, n_agent), dtype=torch.long).cuda()], dim=1)
            x_traffic_light = self.traffice_light_embed(traffic_light)

            car_light = torch.cat([
                car_light,
                torch.zeros((bs, n_agent), dtype=torch.long).cuda()], dim=1)
            x_car_light = self.car_light_embed(car_light)
            object_mask = torch.cat([
                torch.zeros((bs, 1), dtype=torch.float).cuda(), 
                ~agent_valid_mask], dim=1)
        else:
            x_type = data["category_feature"]    # torch.Size([bs, 1 + max_agent, 128])
            x_traffic_light = data["traffic_light_feature"] 
            x_car_light = data["car_light_feature"]       
            object_mask = torch.cat([
                torch.zeros((bs, 1), dtype=torch.float).cuda(), 
                1 - agent_valid_mask.squeeze(-1)], dim=1)

        x_type_agent_his = x_type[:,1:].unsqueeze(2).expand_as(x_agent_his)
        x_agent_his = x_agent_his + x_type_agent_his
        x_agent = x_agent + x_type + x_traffic_light + x_car_light
        if os.getenv("DEPLOY") != 'True' and torch.isnan(x_agent).any():
            print('Error, nan in agent encoder')
            # raise ValueError("There is nan in agent encoder")
        if os.getenv("DEPLOY") != 'True':
            return x_agent, object_mask, x_agent_his, ~agent_time_mask_vec 
        else : 
            return x_agent, object_mask, x_agent_his, (1 - agent_time_mask_vec)
    
class StateAttentionEncoder(nn.Module):
    def __init__(self, state_channel, dim, state_dropout=0.5) -> None:
        super().__init__()

        self.state_channel = state_channel
        self.state_dropout = state_dropout
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(state_channel)])
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)
        self.pos_embed = nn.Parameter(torch.Tensor(1, state_channel, dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, dim))

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        
    @torch.amp.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x, undrop_channel):
        # x.shape: (bs * n, n_feat)
        x_embed = []
        for i, linear in enumerate(self.linears):
            x_embed.append(linear(x[:, i, None]))
        x_embed = torch.stack(x_embed, dim=1)   # before: x_embed[0].shape: (bs * n, 128); len(x_embed) = n_feat; after: (bs * n, n_feat, 128) 

        # pos_embed对每一个特征做位置编码
        pos_embed = self.pos_embed.repeat(x_embed.shape[0], 1, 1)
        x_embed += pos_embed    # torch.Size([bs * n, state_channel, 128])
        if self.training and self.state_dropout > 0:
            visible_tokens = torch.zeros(
                (x_embed.shape[0], undrop_channel), device=x.device, dtype=torch.bool
            )
            dropout_tokens = (
                torch.rand((x_embed.shape[0], self.state_channel - undrop_channel), device=x.device)
                < self.state_dropout
            )   # self.state_dropout 的概率为false，即保留该特征
            key_padding_mask = torch.cat([visible_tokens, dropout_tokens], dim=1)   # torch.Size([bs * n, n_feat])
        else:
            key_padding_mask = None
        
        query = self.query.repeat(x_embed.shape[0], 1, 1)
        x_state = self.attn(
            query=query.permute(1,0,2),     # from torch.Size([bs * n, 1, 128]) to torch.Size([1, bs * n, , 128])
            key=x_embed.permute(1,0,2),     # torch.Size([state_channel, bs * n, 128])
            value=x_embed.permute(1,0,2),
            key_padding_mask=key_padding_mask,  # torch.Size([bs * n, n_feat])
        )[0]    # torch.Size([1, bs * n, 128])

        return x_state[0]   # torch.Size([bs * n, 128])


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=20):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        # 对于每个位置i，第2k维是sin(i/10000^(2k/dim))
        # 第2k+1维是cos(i/10000^(2k/dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加一个批量维度
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
