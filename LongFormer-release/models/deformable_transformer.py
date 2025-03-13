import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.ms_deform_attn import MSDeformAttn
from models.deformable_attention import DeformableAttention3D


class VisualDecoder(nn.Module):
    def __init__(self, 
                image_dim=256,
                d_model=768, 
                nhead=2,
                num_decoder_layers=6, 
                dim_feedforward=1024, 
                dropout=0.1,
                activation="relu", 
                n_times=1,
                num_feature_scales=4, 
                dec_n_points=4,
                ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_scales = num_feature_scales
        

        decoder_layer = VisualDecoderLayer(
                                image_dim,
                                d_model, 
                                dim_feedforward,
                                dropout, 
                                activation,
                                n_times,
                                num_feature_scales , 
                                nhead, 
                                dec_n_points)
        
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()


    def forward(self, 
                img_embs,
                flow_embs,
                pos_embs,
                img_indicators,
                query_embeds, 
                diff,
                args):
        
        bs = img_indicators.shape[0]

        query_pos, query = torch.split(query_embeds, int(query_embeds.shape[-1]/2), dim=1) # TMP!!!!
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        if isinstance(img_embs,list): # multi-scale
            img_embed_flatten = []
            flow_embed_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for lvl, (img_embed, flow_embed, pos_embed) in enumerate(zip(img_embs, flow_embs, pos_embs)):
                bs, img_dim, d, h, w = img_embed.shape
                # t = bs_t//bs
                spatial_shape = (d, h, w)
                spatial_shapes.append(spatial_shape)

                # img_embed = img_embed.reshape(bs, img_dim, d, h, w)
                # pos_embed = pos_embed.reshape(bs, img_dim, d, h, w)
                img_embed = img_embed.flatten(2).transpose(1, 2)  # b t n dim
                flow_embed = flow_embed.flatten(2).transpose(1, 2)
                pos_embed = pos_embed.flatten(2).transpose(1, 2) # b t n dim

                img_embed_flatten.append(img_embed)
                flow_embed_flatten.append(flow_embed)

                lvl_pos_embed = pos_embed# + self.level_embed[lvl].view(1, 1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
            img_embed_flatten = torch.cat(img_embed_flatten, 2)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=img_embed_flatten.device)

        else:
            bs,img_dim, d, h, w = img_embs.shape
            # t = bs_t//bs
            spatial_shape = (d, h, w)
            
            img_embed = img_embs.flatten(2).transpose(1, 2)  # b t n dim
            flow_embed = flow_embs.flatten(2).transpose(1, 2)
            pos_embed = pos_embs.flatten(2).transpose(1, 2) # b t n dim
            img_embed_flatten = img_embed
            flow_embed_flatten = flow_embed
            lvl_pos_embed_flatten = pos_embed
            spatial_shapes = torch.as_tensor(spatial_shape, dtype=torch.long, device=img_embed_flatten.device)

        for _, layer in enumerate(self.layers):
            query = layer(
                query,
                query_pos,
                img_embed_flatten,
                flow_embed_flatten,
                lvl_pos_embed_flatten, #indicator_temporal_lvl_pos_embed_flatten,
                spatial_shapes,
                )
   
        return query
   

class VisualDecoderLayer(nn.Module):
    def __init__(self, 
                image_dim=256,
                d_model=256, 
                d_ffn=1024,
                dropout=0.1, 
                activation="relu",
                n_times=1,
                n_levels=1, 
                n_heads=8, 
                n_points=4):
        super().__init__()

        # import pdb;pdb.set_trace()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # self.cross_attn = MSDeformAttn(image_dim, d_model, n_times, n_levels, n_heads, n_points, 'decode')
        self.cross_attn = DeformableAttention3D(
            q_dim = d_model,
            kv_dim = image_dim,                     # feature dimensions
            n_times = n_times,
            n_levels = n_levels,
            dim_head = 64,                      # dimension per head
            heads = 8,                          # attention heads
            dropout = 0.,                       # dropout
            downsample_factor = (2,2,2),      # downsample factor (r in paper)
            offset_scale = (2,2,2),           # scale of offset, maximum offset
            offset_kernel_size = (8,8,8),   # offset kernel size
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self, 
        query,
        query_pos, 
        img_embed_flatten,
        flow_embed_flatten,
        indicator_temporal_lvl_pos_embed_flatten,
        spatial_shapes,
        ):
        
        q = k = self.with_pos_embed(query, query_pos) # 就是二者加起来
        self_attn_out = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), query.transpose(0, 1))[0].transpose(0, 1)
        query = query + self.dropout(self_attn_out)
        query = self.norm(query) # bs n_q 768

        
        value = img_embed_flatten
        key = indicator_temporal_lvl_pos_embed_flatten + img_embed_flatten
        flow_value = flow_embed_flatten
        flow_key = indicator_temporal_lvl_pos_embed_flatten + flow_embed_flatten

        cross_attn_out = self.cross_attn(query, key, value, flow_key, flow_value, spatial_shapes) # bs nq 768
        query = query + self.dropout2(cross_attn_out)
        query = self.norm2(query)

        query = self.forward_ffn(query)

        
        return query

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

