import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from util.misc import inverse_sigmoid
from models.resnet import ResNet, BasicBlock, Bottleneck, get_inplanes, convert_weights_to_fp16
from models.densenet import densenet121
from models.deformable_transformer import VisualDecoder
import numpy as np


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.orig_channels = num_pos_feats
        channels = int(np.ceil(num_pos_feats/6)*2)
        self.d_embed = nn.Embedding(50, channels)
        self.h_embed = nn.Embedding(50, channels)
        self.w_embed = nn.Embedding(50, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.d_embed.weight)
        nn.init.uniform_(self.h_embed.weight)
        nn.init.uniform_(self.w_embed.weight)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        i = torch.arange(d, device=x.device)
        j = torch.arange(h, device=x.device)
        k = torch.arange(w, device=x.device)
        x_emb = self.d_embed(i)  # 28 256
        y_emb = self.h_embed(j)
        z_emb = self.w_embed(k)
        pos = torch.cat([
            x_emb.unsqueeze(1).unsqueeze(2).repeat(1, h, w, 1),
            y_emb.unsqueeze(0).unsqueeze(2).repeat(d, 1, w, 1),
            z_emb.unsqueeze(0).unsqueeze(1).repeat(d, h, 1, 1),
        ], dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
        return pos[:,:self.orig_channels,...]

class Longformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.num_queries = args.num_queries
        self.hidden_dim = hidden_dim = args.hidden_dim

        # import pdb;pdb.set_trace()

        if args.vision_encoder=='densenet121':
            self.visual_encoder = densenet121(sample_size=224,sample_duration=16,num_classes=args.num_classes,get_features=True)
            self.visual_encoder_feature_dim = 1024
        elif args.vision_encoder=='res50':
            self.visual_encoder = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes())
            self.visual_encoder_feature_dim = 256

        self.pos_emb = PositionEmbeddingLearned(num_pos_feats=self.visual_encoder_feature_dim)
        
        self.visual_decoder = VisualDecoder(
            image_dim=self.visual_encoder_feature_dim,
            d_model=args.hidden_dim,
            nhead=args.nheads,
            num_decoder_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=0.1,
            activation="relu",
            n_times=args.n_times,
            num_feature_scales=args.num_feature_scales,
            dec_n_points=args.dec_n_points
            )
        # convert_weights_to_fp16(self.visual_decoder)
        
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim*2)

        self.classifier = nn.Linear(hidden_dim, args.num_classes)

        

    def forward(self, img, flow, img_indicators, args):

        query_embeds = self.query_embed.weight
        diff_embeds = None #self.diff_embed.weight
        print(img_indicators)

        # import pdb;pdb.set_trace()
        img_embs,flow_embs = self.visual_encoder(img, flow)
        if isinstance(img_embs,list):
            pos_embs = []
            for lvl in range(len(img_embs)):
                pos_emb = self.pos_emb(img_embs[lvl]).to(img_embs[lvl].dtype) # bs*t img_dim d h w
                pos_embs.append(pos_emb)
        else:
            pos_embs = self.pos_emb(img_embs).to(img_embs.dtype)

        query = self.visual_decoder(img_embs, flow_embs, pos_embs, img_indicators, query_embeds, diff_embeds, args)

        out = query[:,0,:] # first query
        outputs = self.classifier(out)

        return outputs

