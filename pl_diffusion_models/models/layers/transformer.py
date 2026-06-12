import copy
import torch
from torch.functional import _return_inverse
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    def __init__(self, 
        d_model=256, 
        nhead=8,
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dim_feedforward=1024, 
        dropout=0.0,
        activation="relu", 
        return_intermediate_dec=False,
        extra_track_attn=False, 
        n_detect_query=50):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(
            d_model,
            dim_feedforward, 
            dropout, 
            activation, 
            nhead)

        self.encoder = TransformerEncoder(
            encoder_layer, 
            num_encoder_layers)

        if num_decoder_layers is not None:
            decoder_layer = TransformerDecoderLayer(
                d_model, 
                dim_feedforward, 
                dropout, 
                activation,
                nhead,)

            self.decoder = TransformerDecoder(
                decoder_layer, 
                num_decoder_layers,
                return_intermediate_dec)
            #先写死128
            self.navi_point_mlp = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.lateral_mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.lane_ln = nn.LayerNorm(128)

            self.query_longi = nn.Parameter(torch.Tensor(1, 1, 1, 128))
            nn.init.normal_(self.query_longi, mean=0.0, std=0.01)

            self.q_proj = nn.Linear(2 * 128, 128)

        else:
            self.decoder = None

        self._reset_parameters()
    

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt=None, src_padding_mask=None, tgt_padding_mask=None, only_compute_encoder=False):
        if not tgt is None:
            #横向tgt 初始 (B, 5, 128)
            B, n_lat, dim = tgt.shape  # n_lat == 5
            #-> (B, 5, 1, dim) -> (B, 5, n_longi, dim)
            tgt = tgt.unsqueeze(2)                # (B,5,1,dim)
            tgt = tgt.expand(-1, -1, 1, -1)      # (B,5,10,dim)

            #纵向(1,1,10,128)
            query_longi = self.query_longi.expand(B, n_lat, -1, -1)  # (B,5,10,dim)

            #cat
            tgt = self.q_proj(torch.cat([tgt, query_longi], dim=-1))  # (B,5,10,out_dim)
            tgt = tgt.contiguous().reshape(B, 1 * n_lat, dim)

            #(B,5) -> (B,5,1) -> (B,5,10) -> flatten (B,50)
            tgt_padding_mask = tgt_padding_mask.unsqueeze(-1)         # (B,5,1)
            tgt_padding_mask = tgt_padding_mask.expand(-1, -1, 1)    # (B,5,10)
            tgt_padding_mask =tgt_padding_mask.contiguous().reshape(B, 1 * n_lat)

        # encoder
        if self.encoder is not None:
            memory = self.encoder(src, src_padding_mask)
        else:
            memory = src
        
        if only_compute_encoder:
            return memory

        if self.decoder is not None:
            hidden_state = self.decoder(tgt, memory.clone(), tgt_padding_mask, src_padding_mask)
        else:
            hidden_state = memory

        return memory, hidden_state

class DualDecoderTransformer(nn.Module):
    def __init__(self, 
        d_model=256, 
        nhead=8,
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dim_feedforward=1024, 
        dropout=0.0,
        activation="relu", 
        return_intermediate_dec=False,
        extra_track_attn=False, 
        n_detect_query=50):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(
            d_model,
            dim_feedforward, 
            dropout, 
            activation, 
            nhead)

        self.encoder = TransformerEncoder(
            encoder_layer, 
            num_encoder_layers)

        if num_decoder_layers is not None:
            decoder_layer = TransformerDecoderLayer(
                d_model, 
                dim_feedforward, 
                dropout, 
                activation,
                nhead,)

            # tb: time-based
            # db: distance-based
            self.tb_decoder = TransformerDecoder(
                decoder_layer, 
                num_decoder_layers,
                return_intermediate_dec)

            self.db_decoder = TransformerDecoder(
                decoder_layer, 
                num_decoder_layers,
                return_intermediate_dec)
        else:
            self.tb_decoder = None
            self.db_decoder = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt1=None, tgt2=None, src_padding_mask=None, 
                tgt_padding_mask1=None, tgt_padding_mask2=None, only_compute_encoder=False):

        # encoder
        if self.encoder is not None:
            memory = self.encoder(src, src_padding_mask)
        else:
            memory = src

        if only_compute_encoder:
            return memory

        if self.tb_decoder is not None:
            tb_hidden_state = self.tb_decoder(tgt1, memory.clone(), tgt_padding_mask1, src_padding_mask)
            db_hidden_state = self.db_decoder(tgt2, memory.clone(), tgt_padding_mask2, src_padding_mask)
        else:
            tb_hidden_state = memory
            db_hidden_state = memory

        return memory, tb_hidden_state, db_hidden_state

class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
        d_model=256, 
        d_ffn=1024, 
        dropout=0.0, 
        activation="relu", 
        n_heads=8):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    @torch.amp.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, src, mask=None):
        # self attention
        # src.shape: torch.Size([batch_size, seq_len, n_feature])
        # mask.shape: torch.Size([batch_size, seq_len,])

        # print('mask.shape: ', mask.shape)
        src2 = self.self_attn(src.transpose(0, 1), src.transpose(0, 1), src.transpose(0, 1), key_padding_mask=mask)[0].transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask=None):
        output = src
        for _, layer in enumerate(self.layers):
            output = layer(output, mask=mask)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super().__init__()
        self.num_head = n_heads
        
        # cross attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, tgt_padding_mask):
        # q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(tgt.transpose(0, 1), 
                              tgt.transpose(0, 1), 
                              tgt.transpose(0, 1), 
                              key_padding_mask=tgt_padding_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def forward(self, tgt, src, tgt_padding_mask=None, src_padding_mask=None):
        # self attention
        tgt = self._forward_self_attn(tgt, tgt_padding_mask)

        # cross attention
        tgt2 = self.cross_attn(tgt.transpose(0, 1),
                               src.transpose(0, 1), 
                               src.transpose(0, 1),
                               key_padding_mask=src_padding_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, src, tgt_padding_mask=None, src_padding_mask=None):
        #  hidden_state = self.decoder(tgt, memory, query_embed, src_padding_mask, tgt_padding_mask)
        output = tgt
        # self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        intermediate = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, src, tgt_padding_mask, src_padding_mask)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



