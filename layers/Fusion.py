import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils.similarity import similarity


class Cyclemap(nn.Module):

    def __init__(self, map_raw, intra_len, inter_len, enc_in, D_cp, D_mix, D_de, compress=1,mix=0, denoise=1):
        super(Cyclemap, self).__init__()
        self.map_raw = map_raw
        self.compress = compress
        self.mix = mix
        self.denoise = denoise                        

        if compress:
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

        if denoise:
            self.denoise = nn.Sequential(
                nn.Linear(intra_len, D_de),
                nn.ReLU(),           
                nn.Linear(D_de, intra_len),                    
            )               

    def forward(self, indices):      
        if self.compress:
            map_tmp = self.compress(self.map_raw)                    # (C, P+L, H)
        else:
            map_tmp = self.map_raw

        if self.mix:
            map_tmp = self.mixing(map_tmp.permute(1, 2, 0))          # (P+L, H, C)
            map_tmp = map_tmp.permute(2, 1, 0)                       # (C, H, P+L)
        else:
            map_tmp = map_tmp.permute(0, 2, 1)                       # (C, H, P+L)

        if self.denoise:    
            map_tmp = self.denoise(map_tmp)                            # (C, H, P+L)

        B, L = indices.shape
        C, H, _ = map_tmp.shape

        indices = torch.repeat_interleave(indices.unsqueeze(1).repeat(1, H, 1), repeats=C, dim=0)
        map_dy = torch.gather(map_tmp.repeat(B, 1, 1), dim=2, index=indices)                       # (B*C, H, L)
        map_dy = map_dy.reshape(B, C, H, L)                                                        # (B, C, H, L)

        return map_dy


class Fusion(nn.Module):

    def __init__(self, seq_len, enc_in, sim_mode='l1', scale=None):
        super(Fusion, self).__init__()
        self.scale = 1 / sqrt(seq_len) if scale is None else scale

        self.encoder = nn.Linear(seq_len, seq_len)
        self.similarity = similarity(sim_mode)                           

        self.lamda1 = nn.Parameter(torch.zeros(enc_in, 1), requires_grad=True)
        self.lamda2 = nn.Parameter(torch.zeros(1, seq_len), requires_grad=True)
   

    def forward(self, x, map_dy):
        x = self.encoder(x)

        scores = self.similarity(x, map_dy)                                          # (B, C, H)
        scores = torch.softmax(scores * self.scale, dim=2)                           # (B, C, H)

        lamda = torch.matmul(torch.sigmoid(self.lamda1), torch.sigmoid(self.lamda2)) # (C, L)
        x_global = torch.einsum('bchl,bch->bcl', map_dy, scores)                     # (B, C, L)
        x_fuse = x_global * lamda + x * (1 - lamda)                                  # (B, C, L)

        return x_fuse
