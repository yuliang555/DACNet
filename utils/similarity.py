import torch
import torch.nn.functional as F

EPS = 1e-8


def dot_similarity(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    """
    A1: (B, C, L), A2: (B, C, H, L) -> out: (B, C, H)
    out[b,c,h] = sum_l A1[b,c,l] * A2[b,c,h,l]
    """
    return torch.einsum('bcl,bchl->bch', A1, A2)


def cosine_similarity(A1: torch.Tensor, A2: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    A1: (B,C,L), A2: (B,C,H,L)
    """
    dot = torch.einsum('bcl,bchl->bch', A1, A2)       # (B,C,H)
    n1 = torch.norm(A1, dim=-1).unsqueeze(-1)         # (B,C,1)
    n2 = torch.norm(A2, dim=-1)                       # (B,C,H)
    denom = (n1 * n2).clamp_min(eps)                  # (B,C,H)
    return dot / denom


def pearson_correlation(A1: torch.Tensor, A2: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    A1: (B,C,L), A2: (B,C,H,L)
    """
    if A1.dim() != 3 or A2.dim() != 4:
        raise ValueError(f"A1 must be (B,C,L) and A2 must be (B,C,H,L), got {A1.shape} and {A2.shape}")

    mean1 = A1.mean(dim=-1, keepdim=True)         # (B,C,1)
    mean2 = A2.mean(dim=-1, keepdim=True)         # (B,C,1,H)

    x = A1 - mean1                                # (B,C,L)
    y = A2 - mean2                                # (B,C,H,L)

    cov_num = torch.einsum('bcl,bchl->bch', x, y)        # (B,C,H)
    denom_x = x.pow(2).sum(dim=-1).unsqueeze(-1)         # (B,C,1)
    denom_y = y.pow(2).sum(dim=-1)                       # (B,C,H)
    denom = torch.sqrt(denom_x * denom_y).clamp_min(eps) # (B,C,H)

    return cov_num / denom


def negative_l2_similarity(A1: torch.Tensor, A2: torch.Tensor, sqrt: bool = True, eps: float = EPS) -> torch.Tensor:
    """
    A1: (B,C,L), A2: (B,C,H,L) -> out: (B,C,H)
    """
    dot = torch.einsum('bcl,bchl->bch', A1, A2)     # (B,C,H)
    x2 = (A1 * A1).sum(dim=-1).unsqueeze(-1)        # (B,C,1)
    y2 = (A2 * A2).sum(dim=-1)                      # (B,C,H)
    dist2 = x2 + y2 - 2.0 * dot                     # (B,C,H)
    dist2 = dist2.clamp_min(0.0)
    if sqrt:
        return -torch.sqrt(dist2 + eps)
    else:
        return -dist2


def negative_l1_similarity(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    """
    A1: (B,C,L), A2: (B,C,H,L) -> out: (B,C,H)
    """
    if A1.dim() != 3 or A2.dim() != 4:
        raise ValueError(f"A1 must be (B,C,L) and A2 must be (B,C,H,L), got {A1.shape} and {A2.shape}")
    l1 = torch.abs(A1.unsqueeze(2) - A2).sum(dim=-1)  # (B,C,H)
    return -l1


# Convenience dispatcher
def similarity( mode: str = 'cosine') -> torch.Tensor:
    """mode in {'dot','cosine','pearson','l2','l1'}; inputs: A1 (B,L), A2 (B,H,L) -> (B,H)"""
    mode = mode.lower()
    if mode == 'dot':
        return dot_similarity
    if mode == 'cosine':
        return cosine_similarity
    if mode == 'pearson':
        return pearson_correlation
    if mode == 'l2':
        return negative_l2_similarity
    if mode == 'neg_l1' or mode == 'l1':
        return negative_l1_similarity
    raise ValueError(f"Unknown similarity mode: {mode}")
