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
        # self.seq_len = configs.seq_len
        # self.map_raw = map_raw

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

        # seq_x = x.permute(0, 2, 1)
        # seq_y = self.map_raw.permute(0, 2, 1).unfold(dimension=2, size=96, step=1) # C, S, P, L
        # seq_y = seq_y.permute(2, 0, 1, 3)                                          # P, C, S, L

        # # Pearson correlation
        # eps = 1e-8
        # mean_x = seq_x.mean(dim=2, keepdim=True)                # (B, C, 1)
        # std_x = seq_x.std(dim=2, unbiased=False, keepdim=True)  # (B, C, 1)        
        # mean_y = seq_y.mean(dim=3, keepdim=True)                # (P, C, S, 1)
        # std_y = seq_y.std(dim=3, unbiased=False, keepdim=True)  # (P, C, S, 1)
        # x_norm = (seq_x - mean_x) / (std_x + eps)               # (B, C, L)
        # y_norm = (seq_y - mean_y) / (std_y + eps)               # (P, C, S, L)
        
        # corr = torch.einsum('bcl,pcsl->bpcs', x_norm, y_norm) / self.seq_len    # (B, P, C, S) 
        
        # mask_x = (std_x.squeeze(-1) < eps).unsqueeze(1).unsqueeze(3)            # (B, 1, C, 1)
        # mask_y = (std_y.squeeze(-1) < eps).unsqueeze(0)                         # (1, P, C, S)
        # corr = torch.where(mask_x | mask_y, torch.zeros_like(corr), corr)

        # corr = corr.sum(dim=-1).permute(0, 2, 1)                                                # B, C, P
        # period_indices = torch.arange(self.seq_len).unsqueeze(0).unsqueeze(0).to(corr.device)   # 1, 1, L
        # indices = torch.argmax(corr, dim=-1, keepdim=True) + period_indices                     # B, C, L


        x_loc = self.encoder(x.permute(0, 2, 1))      # (B, C, L)
        x_fuse = self.fusion(x_loc, indices)          # (B, C, L)                              
        dec_out = self.backbone(x_fuse, x_mark)       # (B, C, T)
        dec_out = dec_out.permute(0, 2, 1)            # (B, T, C)                                              
            
        if self.use_norm:
            dec_out = self.norm(dec_out, 'denorm')

        return dec_out
    


