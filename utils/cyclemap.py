import pandas as pd
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def correlation(seq_x, seq_y, seq_len):
    # seq_x: (B, C, L), seq_y: (P, C, L)
    eps = 1e-8
    seq_x_centered = seq_x - seq_x.mean(dim=2, keepdim=True)
    seq_y_centered = seq_y - seq_y.mean(dim=2, keepdim=True)
    cov = torch.einsum('bcl,pcl->bpc', seq_x_centered, seq_y_centered)     # (B, P, C)
    # cov = torch.sum(seq_x_centered * seq_y_centered, dim=2)
    var_x = torch.sum(seq_x_centered**2, dim=2)
    var_y = torch.sum(seq_y_centered**2, dim=2)
    var = torch.einsum('bc,pc->bpc', var_x, var_y)                         # (B, P, C)
    
    acf = cov / (torch.sqrt(var) + eps)

    acf = acf.sum(dim=2)                                                            # B, P
    period_indices = torch.arange(seq_len).unsqueeze(0).to(acf.device)              # 1, L
    indices = torch.argmax(acf, dim=-1, keepdim=True) + period_indices              # B, L          

    return indices


def svd_compress_per_channel(tensor, target_s, denoise_strength):   
    C, L, S = tensor.shape
    device = tensor.device
    tensor_cpu = tensor.cpu()
    
    compressed_tensor = torch.zeros(C, L, target_s, device='cpu')
    
    for c in range(C):
        channel_data = tensor_cpu[c]
        U, S_values, V = torch.svd(channel_data, some=True)        

        total_energy = torch.sum(S_values ** 2)
        cumulative_energy = torch.cumsum(S_values ** 2, dim=0) / total_energy

        k_candidates = torch.where(cumulative_energy >= denoise_strength)[0]
        k_denoise = k_candidates[0].item() + 1 if len(k_candidates) > 0 else len(S_values)
        k_denoise = min(k_denoise, len(S_values))
        
        effective_k = min(k_denoise, target_s, L, S)
        
        S_diag = torch.diag(S_values[:effective_k])
        
        U_reduced = U[:, :effective_k]  # [L, effective_k]
        V_reduced = V[:target_s, :effective_k]  # [target_s, effective_k]
        
        channel_compressed = U_reduced @ S_diag @ V_reduced.t()  # [L, target_s]
        
        compressed_tensor[c] = channel_compressed
    
    compressed_tensor = compressed_tensor.to(device)
    
    return compressed_tensor



def cyclemap(data, seq_len, cycle, device, drift=0):
    data = torch.tensor(data).permute(1, 0).float().to(device)

    if drift:                
        max_len = 1000
        idx_len = max_len - seq_len + 1
        C, L = data.shape    
        max_lag = L - idx_len - seq_len

        baseline = data[:, :max_len].unfold(dimension=1, size=seq_len, step=1)

        acf_values = []
        
        for k in range(max_lag + 1):
            x_t = data[:, :L-k] 
            x_t_k = data[:, k:]   
            
            x_t_centered = x_t - x_t.mean(dim=1, keepdim=True)
            x_t_k_centered = x_t_k - x_t_k.mean(dim=1, keepdim=True)
            cov = torch.sum(x_t_centered * x_t_k_centered, dim=1)
            var_t = torch.sum(x_t_centered**2, dim=1)
            var_t_k = torch.sum(x_t_k_centered**2, dim=1)
            
            acf = cov / (torch.sqrt(var_t * var_t_k) + 1e-8)
            acf_values.append(acf)
            
        acf_tensor = torch.stack(acf_values, dim=1)
        acf_np = acf_tensor.cpu().numpy()
        acf_np = np.insert(acf_np, 0, -np.inf, axis=1)

        peaks_lags = []
        peaks_len = []
        for c in range(C):
            peaks, _ = find_peaks(acf_np[c], height=-1., distance=(cycle // 2))
            peaks_lags.append(peaks - 1)
            peaks_len.append(len(peaks))

        new_peaks_lags = []
        I = min(peaks_len)
        S = idx_len + seq_len
        for lags in peaks_lags:
            new_peaks_lags.append(lags[:I])

        lags_tensor = torch.tensor(new_peaks_lags).long().to(device)

        batch_indices = torch.arange(C, device=device).view(C, 1, 1).expand(C, S, I)
        base_indices = lags_tensor.view(C, 1, I).expand(C, S, I)
        offsets = torch.arange(S, device=device).view(1, S, 1).expand(C, S, I)
        final_indices = base_indices + offsets
        map_raw = data[batch_indices, final_indices]  # (C, S, I)
    else:
        map_raw = data.unfold(dimension=1, size=cycle+seq_len, step=cycle).permute(0, 2, 1)
        baseline = None

    return map_raw, baseline

    
