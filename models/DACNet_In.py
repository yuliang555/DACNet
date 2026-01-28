import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Fusion import Fusion
from layers.BackBone import iTransformer, Linear, MLP, DLinear, DMLP 
from layers.Norm import InstanceNorm



class Model(nn.Module):

    def __init__(self, configs, map_raw):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm

        self.norm = InstanceNorm(1, configs.use_norm)
        self.fusion = Fusion(map_raw, configs.seq_len, configs.intra_len, configs.inter_len, configs.enc_in, 
                             configs.D_cp, configs.D_de, configs.D_mix, mix=configs.mix, sim_mode=configs.sim_mode, scale=0.05)
        self.encoder = nn.Linear(configs.seq_len, configs.seq_len)

        if configs.backbone == 'itransformer':
            self.backbone = iTransformer(configs)
        elif configs.backbone == 'linear':
            self.backbone = Linear(configs)
        elif configs.backbone == 'dlinear':
            self.backbone = DLinear(configs)            
        elif configs.backbone == 'mlp':
            self.backbone = MLP(configs)
        elif configs.backbone == 'dmlp':
            self.backbone = DMLP(configs)

    def forward(self, x, x_mark, indices):
        
        if self.use_norm:
            x = self.norm(x, 'norm')

        x_loc = self.encoder(x.permute(0, 2, 1))      # (B, C, L)
        x_fuse = self.fusion(x_loc, indices)          # (B, C, L)                              
        dec_out = self.backbone(x_fuse, x_mark)       # (B, C, T)
        dec_out = dec_out.permute(0, 2, 1)            # (B, T, C)                                              
            
        if self.use_norm:
            dec_out = self.norm(dec_out, 'denorm')

        return dec_out
    


