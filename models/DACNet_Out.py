import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Fusion import Fusion
from layers.BackBone import iTransformer, Linear, MLP 
from layers.Norm import InstanceNorm


class Model(nn.Module):

    def __init__(self, configs, map_raw):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm

        self.norm = InstanceNorm(2, configs.use_norm)
        self.fusion = Fusion(map_raw, configs.pred_len, configs.intra_len, configs.inter_len, configs.enc_in, 
                             configs.D_cp, configs.D_de, configs.D_mix, mix=configs.mix, sim_mode=configs.sim_mode, scale=0.05)
        self.encoder = nn.Linear(configs.seq_len, configs.seq_len)

        if configs.backbone == 'itransformer':
            self.backbone = iTransformer(configs)
        elif configs.backbone == 'linear':
            self.backbone = Linear(configs)
        elif configs.backbone == 'mlp':
            self.backbone = MLP(configs)          

    def forward(self, x, x_mark, indices):
        x = x.permute(0, 2, 1)            # (B, C, L)
        
        if self.use_norm:
            x = self.norm(x, 'norm')

        x = self.encoder(x)                    # (B, C, L)
        y_loc = self.backbone(x, x_mark)       # (B, C, T)                                                                           

        if self.use_norm:
            y_loc = self.norm(y_loc, 'denorm')
                                              
        y_fuse = self.fusion(y_loc, indices)   # (B, C, T)
        dec_out = y_fuse.permute(0, 2, 1)      # (B, T, C)

        return dec_out

