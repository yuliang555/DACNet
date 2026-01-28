import numpy as np
import copy
from math import sin, pi
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from numpy.lib.stride_tricks import as_strided


def sb_generator(periods, sb_type, L):
    assert sb_type in ["sin", "swatooth", "reactangle", "pulse"], "sb_type must be one of sin, swatooth, reactangle, or pulse"
    
    period_list = copy.deepcopy(periods)
    
    # for period in periods:
    #     for i in range(2, 5):
    #         period_list.append(period // i)

    if sb_type == "sin":
        sb = [[sin(t * 2 * pi / period) for t in range(L)] for period in period_list]
        sb1 = [[sin(t * 2 * pi / period + pi) for t in range(L)] for period in period_list]
        # sb2 = [[sin((t + 2) * 2 * pi / period) for t in range(L)] for period in period_list]
    elif sb_type == "swatooth":
        sb = [[t % period for t in range(L)] for period in period_list]
    elif sb_type == "reactangle":
        sb = [[2 * t // period % 2 for t in range(L)] for period in period_list]
    elif sb_type == "pulse":
        sb = [[1 if t % period == 0 else 0 for t in range(L)] for period in period_list]
    
    sb = np.array(sb)
    sb1 = np.array(sb1)
    # sb2 = np.array(sb2)
    sb = np.concatenate([sb, sb1], axis=0).T
    
    # sb = np.array(sb).T
    print(sb.shape)
    
    return sb


def tb_generator(data, kernel, N):    
    L, C = data.shape
    stride = kernel
    
    tb_len = (L - kernel) // stride + 1
    shape = (tb_len, kernel, C)    
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    tb = as_strided(data, shape=shape, strides=strides)
    tb = tb.mean(axis=1)
    
    tb_len = tb_len // N
    tb = tb[:tb_len * N].reshape(N, tb_len, C)
    
    mean = tb.mean(axis=1, keepdims=True)
    std  = tb.std(axis=1, keepdims=True)
    (tb - mean) / (std + 1e-5)       
    print(tb.shape)

    return tb, tb_len


# def tb_generator(data, kernel):    
#     L, C = data.shape
#     stride = kernel
    
#     tb_len = (L - kernel) // stride + 1
#     shape = (tb_len, kernel, C)
    
#     strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
#     tb = as_strided(data, shape=shape, strides=strides)
    
#     mean = tb.mean(axis=0, keepdims=True)
#     std  = tb.std(axis=0, keepdims=True)
#     (tb - mean) / (std + 1e-5)       
#     print(tb.shape)

#     return tb, tb_len