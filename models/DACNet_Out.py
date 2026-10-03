import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Fusion import Cyclemap, Fusion
from layers.BackBone import iTransformer, Linear, MLP 
from layers.Norm import InstanceNorm


class Model(nn.Module):

    def __init__(self, configs, map_raw, anchor):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm

        self.norm = InstanceNorm(2, configs.use_norm)
        self.cyclemap = Cyclemap(map_raw, configs.intra_len, configs.inter_len, configs.enc_in, 
                                configs.D_cp, configs.D_mix, configs.D_de, compress=1, mix=configs.mix, denoise=1)
        self.fusion = Fusion(configs.pred_len, configs.enc_in, sim_mode=configs.sim_mode, scale=None)

        if configs.backbone == 'itransformer':
            self.backbone = iTransformer(configs)
        elif configs.backbone == 'linear':
            self.backbone = Linear(configs)
        elif configs.backbone == 'mlp':
            self.backbone = MLP(configs)          

    def forward(self, x, x_mark, indices):
        x = x.permute(0, 2, 1)                     # (B, C, L)        
        if self.use_norm:
            x = self.norm(x, 'norm')

        y = self.backbone(x, x_mark)               # (B, C, T)
        map_dy = self.cyclemap(indices)            # (B, C, H, L)
        dec_out = self.fusion(y, map_dy)           # (B, C, T)                                                                           

        if self.use_norm:
            dec_out = self.norm(dec_out, 'denorm')        
        dec_out = dec_out.permute(0, 2, 1)         # (B, T, C)

        return dec_out

