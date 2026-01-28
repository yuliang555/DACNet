import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils.similarity import similarity


class Fusion(nn.Module):

    def __init__(self, map_raw, seq_len, intra_len, inter_len, enc_in, D_cp, D_de, D_mix, mix=0, sim_mode='l1', scale=0.05):
        super(Fusion, self).__init__()
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

        map_tmp = self.compress(self.map_raw)                        # (C, P+L, H)

        if self.mix:
            map_tmp = self.mixing(map_tmp.permute(1, 2, 0))          # (P+L, H, C)
            map_tmp = map_tmp.permute(2, 1, 0)                       # (C, H, P+L)
        else:
            map_tmp = map_tmp.permute(0, 2, 1)                       # (C, H, P+L)

        map_st = self.denoise(map_tmp)                               # (C, H, P+L)       

        indices = torch.repeat_interleave(indices.unsqueeze(1).repeat(1, self.D_cp, 1), repeats=C, dim=0)
        map_dy = torch.gather(map_st.repeat(B, 1, 1), dim=2, index=indices)     # (B*C, H, L)
        map_dy = map_dy.reshape(B, C, self.D_cp, L)                             # (B, C, H, L)

        scores = self.similarity(x_loc, map_dy)               # (B, C, H)
        scores = torch.softmax(scores * self.scale, dim=2)    # (B, C, H)

        lamda = torch.matmul(torch.sigmoid(self.lamda1), torch.sigmoid(self.lamda2)) # (C, L)
        x_global = torch.einsum('bchl,bch->bcl', map_dy, scores)      # (B, C, L)
        x_fuse = x_global * lamda + x_loc * (1 - lamda)               # (B, C, L)

        return x_fuse
