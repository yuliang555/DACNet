import torch
import torch.nn as nn
from layers.BackBone import iTransformer, Linear, MLP, DLinear, DMLP 
from layers.Norm import InstanceNorm
from utils.similarity import similarity


class Fusion(nn.Module):

    def __init__(self, map_raw, seq_len, intra_len, inter_len, enc_in, D_cp, D_de, D_mix, mix=0, sim_mode='l1', scale=0.05):
        super(Fusion, self).__init__()
        self.inter_len = inter_len
        self.D_cp = D_cp
        self.scale = scale
        self.map_raw = map_raw
        self.mix = mix

        self.similarity = similarity(sim_mode)                           

        self.denoise = nn.Sequential(
            nn.Linear(intra_len, D_de),
            nn.ReLU(),           
            nn.Linear(D_de, intra_len),                    
        )       
        self.compress = nn.Sequential(
            nn.Linear(inter_len, D_cp),
            nn.ReLU(),
            nn.Linear(D_cp, D_cp),           
        )

        if mix:
            self.mixing = nn.Sequential(
                nn.Linear(enc_in, D_mix),
                nn.ReLU(),
                nn.Linear(D_mix, enc_in),           
            )

        self.lamda1 = nn.Parameter(torch.zeros(enc_in, 1), requires_grad=True)
        self.lamda2 = nn.Parameter(torch.zeros(1, seq_len), requires_grad=True)
   

    def forward(self, x_loc, indices, lamda=None):
        B, C, L = x_loc.shape

        indices = indices.unsqueeze(2).repeat(1, 1, self.inter_len, 1)                              # (B, C, H, L)
        map_raw = torch.gather(self.map_raw.unsqueeze(0).repeat(B, 1, 1, 1), dim=3, index=indices)
        # print('map_dy shape:', map_dy.shape) # (B, C, H, L)

        # indices = torch.repeat_interleave(indices.unsqueeze(1).repeat(1, self.D_cp, 1), repeats=C, dim=0)
        # map_dy = torch.gather(map_st.repeat(B, 1, 1), dim=2, index=indices)     # (B*C, H, L)
        # map_dy = map_dy.reshape(B, C, self.D_cp, L)                             # (B, C, H, L)                

        map_tmp = self.compress(self.map_raw)                        # (C, P+L, H)

        if self.mix:
            map_tmp = self.mixing(map_tmp.permute(1, 2, 0))          # (P+L, H, C)
            map_tmp = map_tmp.permute(2, 1, 0)                       # (C, H, P+L)
        else:
            map_tmp = map_tmp.permute(0, 2, 1)                       # (C, H, P+L)

        map_st = self.denoise(map_tmp)                               # (C, H, P+L)       

        scores = self.similarity(x_loc, map_dy)               # (B, C, H)
        scores = torch.softmax(scores * self.scale, dim=2)    # (B, C, H)

        lamda = torch.matmul(torch.sigmoid(self.lamda1), torch.sigmoid(self.lamda2)) # (C, L)
        x_global = torch.einsum('bchl,bch->bcl', map_dy, scores)      # (B, C, L)
        x_fuse = x_global * lamda + x_loc * (1 - lamda)               # (B, C, L)

        return x_fuse


class Model(nn.Module):

    def __init__(self, configs, map_raw, baseline):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.seq_len = configs.seq_len
        self.baseline = baseline

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

        seq_x = x.permute(0, 2, 1)
        seq_y = self.baseline.clone().permute(1, 0, 2)                  # P, C, L

        eps = 1e-8
        mean_x = seq_x.mean(dim=2, keepdim=True)                # (B, C, 1)
        std_x = seq_x.std(dim=2, unbiased=False, keepdim=True)  # (B, C, 1)        
        mean_y = seq_y.mean(dim=2, keepdim=True)                # (P, C, 1)
        std_y = seq_y.std(dim=2, unbiased=False, keepdim=True)  # (P, C, 1)
        x_norm = (seq_x - mean_x) / (std_x + eps)               # (B, C, L)
        y_norm = (seq_y - mean_y) / (std_y + eps)               # (P, C, L)
        
        corr = torch.einsum('bcl,pcl->bpc', x_norm, y_norm) / self.seq_len    # (B, P, C) 
        
        mask_x = (std_x.squeeze(-1) < eps).unsqueeze(1)            # (B, 1, C)
        mask_y = (std_y.squeeze(-1) < eps).unsqueeze(0)            # (1, P, C)
        corr = torch.where(mask_x | mask_y, torch.zeros_like(corr), corr)

        corr = corr.permute(0, 2, 1)                                                            # B, C, P
        period_indices = torch.arange(self.seq_len).unsqueeze(0).unsqueeze(0).to(corr.device)   # 1, 1, L
        indices = torch.argmax(corr, dim=-1, keepdim=True) + period_indices                     # B, C, L


        x_loc = self.encoder(x.permute(0, 2, 1))      # (B, C, L)
        x_fuse = self.fusion(x_loc, indices)          # (B, C, L)                              
        dec_out = self.backbone(x_fuse, x_mark)       # (B, C, T)
        dec_out = dec_out.permute(0, 2, 1)            # (B, T, C)                                              
            
        if self.use_norm:
            dec_out = self.norm(dec_out, 'denorm')

        return dec_out
    


