import torch
import torch.nn.functional as F
from torch import nn, einsum

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def cast_tuple(x, length = 1):
    return x if isinstance(x, tuple) else ((x,) * length)

# tensor helpers

def create_grid_like(t, dim = 0):
    f, h, w, device = *t.shape[-3:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    f, h, w = grid.shape[-3:]
    grid_f, grid_h, grid_w = grid.unbind(dim = dim)

    grid_f = 2.0 * grid_f / max(f - 1, 1) - 1.0
    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_f, grid_h, grid_w), dim = out_dim)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype = torch.float32))

    def forward(self, x):
        return x * rearrange(self.scale, 'c -> 1 c 1 1 1')

# continuous positional bias from SwinV2

class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        # import pdb;pdb.set_trace()
        device, dtype = grid_q.device, grid_kv.dtype

        # grid_q: np np np 3
        # grid_kv: bs*head t np np np 3

        grid_q = rearrange(grid_q, '... c -> 1 (...) c')
        grid_kv = rearrange(grid_kv, 'b ... c -> b (...) c') # b t*np*np*np

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1) [b,np*np*np,t*np*np*np,3]

        

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups) # bs (head*1) np*np*np,t*np*np*np

        # import pdb;pdb.set_trace()

        return bias

# main class

class DeformableAttention3D(nn.Module):
    def __init__(
        self,
        *,
        q_dim,
        kv_dim,
        n_times,
        n_levels,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 1, # 文中的r
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 3,
        group_queries = True,
        group_key_values = True
    ):
        super().__init__()
        downsample_factor = cast_tuple(downsample_factor, length = 3)
        offset_scale = default(offset_scale, downsample_factor)

        self.n_times = n_times
        self.n_levels = n_levels
        self.dim_head = dim_head

        offset_conv_padding = tuple(map(lambda x: (x[0] - x[1]) / 2, zip(offset_kernel_size, downsample_factor)))
        assert all([(padding > 0 and padding.is_integer()) for padding in offset_conv_padding])

        offset_groups = default(offset_groups, heads) 
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5 # 1/head
        self.heads = heads
        self.offset_groups = offset_groups # 不设置的话就是head数

        offset_dims = inner_dim // offset_groups # 64

        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            # nn.Conv3d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = tuple(map(int, offset_conv_padding))),
            nn.Conv3d(offset_dims, offset_dims, 1, groups = offset_dims, stride = 1, padding = 0),
            nn.GELU(),
            nn.Conv3d(offset_dims, 3, 1, bias = False),      # self.n_levels               
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.rel_pos_bias = CPB(q_dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv3d(q_dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        self.to_k = nn.Conv3d(kv_dim*self.n_levels*self.n_times, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv3d(kv_dim*self.n_levels*self.n_times, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Conv3d(inner_dim, q_dim, 1)

    def forward(self, query, key, value, flow_key, flow_value, spatial_shapes, return_vgrid = False):
        """
        b - batch
        h - heads
        f - frames
        x - height
        y - width
        d - dimension
        g - offset groups
        """
        
        heads = self.heads
        device = query.device
        bs, nq, q_dim = query.shape
        np = round(pow(nq,1/3))
        bs, n_v, v_dim = value.shape

        # import pdb;pdb.set_trace()

        if len(spatial_shapes.shape)>1:
            key_list = key.split([D_ * H_ * W_ for D_, H_, W_ in spatial_shapes], dim=2)
            value_list = value.split([D_ * H_ * W_ for D_, H_, W_ in spatial_shapes], dim=2)
            query = query.transpose(1,2).reshape(bs, q_dim, np, np, np) # bs 768 3 3 3
            keys = []
            values = []
            for lvl in range(self.n_levels):
                keys.append(key_list[lvl].transpose(2,3).reshape(bs,t,v_dim,spatial_shapes[lvl][0],spatial_shapes[lvl][1],spatial_shapes[lvl][2]))
                values.append(value_list[lvl].transpose(2,3).reshape(bs,t,v_dim,spatial_shapes[lvl][0],spatial_shapes[lvl][1],spatial_shapes[lvl][2]))
        else:
            D_, H_, W_ = spatial_shapes
            query = query.transpose(1,2).reshape(bs, q_dim, np, np, np)
            # import pdb;pdb.set_trace()
            keys = key.transpose(1,2).reshape(bs,v_dim,D_,H_,W_)
            values = value.transpose(1,2).reshape(bs,v_dim,D_,H_,W_)
            flow_keys = flow_key.transpose(1,2).reshape(bs,v_dim,D_,H_,W_)
            flow_values = flow_value.transpose(1,2).reshape(bs,v_dim,D_,H_,W_)



        q = self.to_q(query) # bs inner_dim np np np


        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)
        grouped_queries = group(q) # bs*head inner_dim/head np np np

        # import pdb;pdb.set_trace()
        
        offsets = self.to_offsets(grouped_queries)
        np = offsets.shape[-1]
        offsets = offsets.reshape(bs*heads,3,np,np,np) # bs*head inner_dim/head np np np -> bs*head 【3】 np np np这里这个3是xyz的offset



        # calculate grid + offsets
        grid = create_grid_like(offsets)#.repeat(self.n_times,1,1,1).reshape(self.n_times,3,np,np,np) # t*3 np np np 采样的坐标，不可学习的
        vgrid = grid + offsets # [bs*head 3 np np np]
        vgrid_scaled = normalize_grid(vgrid,dim=1,out_dim=-1) #[bs*head np np np 3]

        '''
        input: [B, C, H_in, W_in]
        grid: [B, H_out, W_out, 2]
        output: [B, C, H_out, W_out]
        '''
        # import pdb;pdb.set_trace()
        if isinstance(keys,list):
            ks = []
            vs = []
            for lvl in range(self.n_levels):
                tmp_k = []
                tmp_v = []
                for time in range(self.n_times):

                    k_feats = F.grid_sample(
                        group(keys[lvl][:,time]),
                        vgrid_scaled[:,time],
                    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
                    tmp_k.append(k_feats)

                    v_feats = F.grid_sample(
                        group(values[lvl][:,time]),
                        vgrid_scaled[:,time],
                    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
                    tmp_v.append(v_feats)
                tmp_k = torch.stack(tmp_k,dim=2)
                tmp_v = torch.stack(tmp_v,dim=2)
                ks.append(tmp_k)
                vs.append(tmp_v)
            
            k_feats = torch.stack(ks,dim=3)
            v_feats = torch.stack(vs,dim=3)

            k = rearrange(k_feats, '(b g) d ... -> b (g d) ...', b = bs) # bs 256 t l np np np
            v = rearrange(v_feats, '(b g) d ... -> b (g d) ...', b = bs) # bs 256 t l np np np

            # derive key / values

            k = self.to_k(k.flatten(1,2)) # bs 512 np np np
            v = self.to_v(v.flatten(1,2)) # bs 512 np np np


            
        else:
            if self.n_times ==2:
                k_feats_img = F.grid_sample(
                    group(keys),
                    vgrid_scaled,
                    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
                k_feats_flow = F.grid_sample(
                    group(flow_keys),
                    vgrid_scaled,
                    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

                v_feats_img = F.grid_sample(
                    group(values),
                    vgrid_scaled,
                    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
                v_feats_flow = F.grid_sample(
                    group(flow_values),
                    vgrid_scaled,
                    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
                # 
                k_feats = torch.stack((k_feats_img,k_feats_flow),dim=2)
                v_feats = torch.stack((v_feats_img,v_feats_flow),dim=2)
                # import pdb;pdb.set_trace()

                k = rearrange(k_feats, '(b g) d ... -> b (g d) ...', b = bs) # bs 1024 np np np
                v = rearrange(v_feats, '(b g) d ... -> b (g d) ...', b = bs) # bs 1024 np np np
                k = self.to_k(k.flatten(1,2)) # bs 512 np np np
                v = self.to_v(v.flatten(1,2)) # bs 512 np np np
            else:
                k_feats_img = F.grid_sample(
                    group(keys[:,0]),
                    vgrid_scaled,
                    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
                v_feats_img = F.grid_sample(
                    group(values[:,0]),
                    vgrid_scaled,
                    mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
                k = rearrange(k_feats_img, '(b g) d ... -> b (g d) ...', b = bs) # bs 1024 t np np np
                v = rearrange(v_feats_img, '(b g) d ... -> b (g d) ...', b = bs) # bs 1024 t np np np
                k = self.to_k(k) # bs 512 np np np
                v = self.to_v(v) # bs 512 np np np

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k) # q: bs head np*np*np 64, k: bs head 27 64 -> sim: bs head 343 27

        # relative positional bias

        grid = create_grid_like(query) # 3 np np np
        grid_scaled = normalize_grid(grid, dim = 0,out_dim=-1) # np np np 3
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled) # bs head*o np*np*np t*np*np*np
        sim = sim + rel_pos_bias

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v) # attn: bs head 343 27, v: bs head 27 64 -> bs head 343 64
        # import pdb;pdb.set_trace()
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = np, y = np, z = np) # b 8 343 64 -> b 512 7 7 7
        out = self.to_out(out) # b 256 7 7 7

        


        if return_vgrid:
            return out, vgrid
        
        # import pdb;pdb.set_trace()

        return out.flatten(-3).transpose(1,2)