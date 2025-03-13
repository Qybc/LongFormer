from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

def ms_deform_attn(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, T_, S_, M_, D_ = value.shape
    _, Lq_, M_, T_, L_, P_, _ = sampling_locations.shape # 0~1

    # Split projection of input features back into individual levels
    value_list = value.split([D_ * H_ * W_ for D_, H_, W_ in value_spatial_shapes], dim=2)
    
    sampling_grids = 2 * sampling_locations - 1     # To get in range of [-1, 1], as necessary for F.grid_sample  bs 32 6 2 2 4 3
    sampling_value_list = []
    for lid_, (D, H, W) in enumerate(value_spatial_shapes):
        # import pdb;pdb.set_trace()
        value_l_ = value_list[lid_].flatten(3).transpose(2, 3).reshape(N_*T_*M_, D_, D, H, W) # bs t 21952 6 128 -> bs t 21952 768 -> bs t 768 21952 -> bs*t*6 128 7 7 7
        sampling_grid_l_ = sampling_grids[:, :, :, :, lid_].permute(0,3,2,1,4,5).flatten(0, 2)# bs 32 6 t 2 4 3 -> bs 32 6 t 4 3 -> bs t 6 32 4 3 -> bs*t*6 32 4 3
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_[:, None], mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(2) # bs*t*6 128 32 4
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.permute(0,2,1,3,4,5).reshape(N_*M_, 1, Lq_, T_*L_*P_) # bs*6 1 32 8
    output = (torch.stack(sampling_value_list, dim=-2).view(N_, T_, M_, D_, Lq_, L_, P_).permute(0,2,3,4,1,5,6).flatten(0,1).flatten(-3) * attention_weights).sum(-1).view(N_, M_*D_, Lq_) # b t m 128 32 l p -> b m 128 32 t l p -> (b m) 128 32 (t l p) -> b 768 32
    return output.transpose(1, 2).contiguous() # b 32 768

    


class MSDeformAttn(nn.Module):
    def __init__(self, image_dim=1408, d_model=768, n_times=1, n_levels=4, n_heads=8, n_points=4, mode='encode'):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))

        self.im2col_step = 64
        self.mode = mode
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_times = n_times

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_times * n_levels * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_times * n_levels * n_points)
        self.value_proj = nn.Linear(image_dim, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        if self.n_heads == 6:
            grid_init = torch.FloatTensor([[-1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.], [ 0., 0., 1.], [ 0., 1., 0.], [ 1., 0., 0.]])
        else:
            raise ValueError("Only n_heads == 6 supported.") # TODO: 26 directions
         
        grid_init = grid_init.view(self.n_heads, 1, 1, 1, 3).repeat(1, self.n_times, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)


    def forward(self, query, reference_points, value):
        
        
        bs, Len_q, _ = query.shape
        value = self.value_proj(value)# bs t n 256 -> bs t n 768
        value = value.view(bs, value.shape[1], value.shape[2], self.n_heads, value.shape[-1] // self.n_heads) # bs t n 6 128

        sampling_offsets = self.sampling_offsets(query).view(bs, Len_q, self.n_heads, self.n_times, self.n_levels, self.n_points, 3) # bs 32 6 2 2 4 3
        attention_weights = self.attention_weights(query).view(bs, Len_q, self.n_heads, self.n_times * self.n_levels * self.n_points) # bs 32 6 2*2*4
        attention_weights = F.softmax(attention_weights, -1).view(bs, Len_q, self.n_heads, self.n_times, self.n_levels, self.n_points) # bs 32 6 2 2 4

        offset_normalizer = torch.tensor([[28,28,28],[14,14,14],[7,7,7]]).to(sampling_offsets.device)
        sampling_locations = reference_points[:, :, None, None, None, None, :] + sampling_offsets / offset_normalizer[None, None, None, None, :, None, :]

        output = ms_deform_attn(value,offset_normalizer,sampling_locations,attention_weights)
        output = self.output_proj(output)
        return output
        


