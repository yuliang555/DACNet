import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, enc_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.enc_in = enc_in
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):

        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.concat([x, x_mark.permute(0, 2, 1)], dim=1))
            
        x = self.dropout(x)
        
        return x[:, :self.enc_in, :], x
    
    
class MFRS1_Embedding(nn.Module):
    def __init__(self, use_pos, seq_len, enc_in, d_model, dropout=0.1):
        super(MFRS1_Embedding, self).__init__()
        self.use_pos = use_pos

        self.pos = torch.nn.Parameter(torch.randn(seq_len, enc_in))        
        self.x_embedding = nn.Linear(seq_len, d_model)        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sb):
        if self.use_pos == 1:
            x = x + self.pos
        elif self.use_pos == 2:
            x = x * self.pos
                         
        x = self.dropout(self.x_embedding(x.permute(0, 2, 1)))        
        sb = self.dropout(self.x_embedding(sb.permute(0, 2, 1)))

        return x, sb
    
    
class MFRS2_Embedding(nn.Module):
    def __init__(self, use_pos, groups, seq_len, tb_len, d_model, dropout=0.1, data=None):
        super(MFRS2_Embedding, self).__init__()
        self.use_pos = use_pos
        self.seq_len = seq_len
        self.d_model = d_model
        self.groups = groups
        
        self.data = data
        C, P, H = data.shape        
        kernel, stride = 24, 12
        self.N = P // groups * groups         
        
        self.conv1d = nn.Conv1d(self.N, groups, kernel_size=kernel, stride=stride, groups=groups)             
        self.norm = nn.LayerNorm(tb_len)
        self.g_embedding = nn.Linear((H- kernel) // stride + 1, d_model)       
        
        self.pos = torch.nn.Parameter(torch.randn(seq_len, C))        
        self.x_embedding = nn.Linear(seq_len, d_model)
        
        # self.decomposition1 = series_decomp(15)
        # self.decomposition2 = series_decomp(35)
        # self.decomposition3 = series_decomp(35)
                       
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, indices):
        B, _, C = x.shape
        # _, M, H = indices.shape
        # s1, t1 = self.decomposition1(x)
        # s2, t2 = self.decomposition2(x)
        # cross1 = torch.stack([s1, s2, t1, t2], dim=1)
        # cross1 = cross1.permute(0, 3, 1, 2).reshape(-1, 4, self.seq_len)
        # cross1 = self.dropout(self.x_embedding(cross1))
                        
        if self.use_pos == 1:
            x = x + self.pos
        elif self.use_pos == 2:
            x = x * self.pos
        
        x = x.permute(0, 2, 1).reshape(-1, self.seq_len).unsqueeze(1)                     
        x = self.dropout(self.x_embedding(x))
        
        cross = self.dropout(self.g_embedding(self.conv1d(self.data[:, :self.N, :])))
        cross = cross.unsqueeze(0).repeat(B, 1, 1, 1).reshape(-1, self.groups, self.d_model)        
        # cross = []
        
        # for m in range(M):
        #     N = self.N[m]
        #     indice = torch.repeat_interleave(indices[:, m].view(B, 1, H).expand(B, N, H), repeats=C, dim=0)
        #     data = torch.gather(self.datas[m][:, :N, :].repeat(B, 1, 1), dim=2, index=indice)
        #     data = self.conv1d[m](self.norm(data))
        #     data = self.dropout(self.g_embedding(data))
        #     cross.append(data)
            
        # cross = torch.concat(cross, dim=1)      

        return x, cross
    
    
class MFRS3_Embedding(nn.Module):
    def __init__(self, seq_len, tb_len, enc_in, d_model, dropout=0.1):
        super(MFRS3_Embedding, self).__init__()
        
        self.pos = torch.nn.Parameter(torch.randn(seq_len, enc_in))        
        self.x_embedding = nn.Linear(seq_len, d_model)
        self.tb_embedding = nn.Linear(tb_len, d_model)        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, tb):
        x = x + self.pos             
        x = self.dropout(self.x_embedding(x.permute(0, 2, 1))).unsqueeze(2)
        
        cross = self.dropout(self.tb_embedding(tb.permute(0, 1, 3, 2)))

        return x, cross


class MFRS4_Embedding(nn.Module):
    def __init__(self, seq_len, tb_len, enc_in, d_model, dropout=0.1):
        super(MFRS4_Embedding, self).__init__()
        self.seq_len = seq_len
        self.tb_len = tb_len
        
        self.pos = torch.nn.Parameter(torch.randn(seq_len, enc_in))        
        self.x_embedding = nn.Linear(seq_len, d_model)
        self.tb_embedding = nn.Linear(tb_len, d_model)
        # self.tb_embedding = nn.Sequential( 
        #     nn.Linear(tb_len, seq_len),
        #     nn.ReLU(),
        #     self.x_embedding
        # )        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, tb):
        x = (x + self.pos).permute(0, 2, 1).reshape(-1, self.seq_len)             
        x = self.dropout(self.x_embedding(x)).unsqueeze(1)
        
        tb = tb.permute(0, 2, 1).reshape(-1, self.tb_len)             
        tb = self.dropout(self.tb_embedding(tb)).unsqueeze(1)

        return x, x, tb


class MFRS5_Embedding(nn.Module):
    def __init__(self, use_pos, seq_len, tb_len, enc_in, N, d_model, dropout=0.1):
        super(MFRS5_Embedding, self).__init__()
        self.use_pos = use_pos
        self.d_model = d_model
        self.tb_len = tb_len
        self.enc_in = enc_in
        self.N = N
        
        if use_pos:
            self.pos = torch.nn.Parameter(torch.randn(seq_len, enc_in))
        
        self.x_embedding = nn.Linear(seq_len, d_model)
        self.sb_embedding = nn.Linear(seq_len, d_model)
        # self.tb_embedding = nn.Linear(tb_len, d_model)
        self.tb_embedding = nn.Sequential( 
            nn.Linear(tb_len, seq_len),
            nn.ReLU(),
            self.x_embedding
        )        
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sb, tb):
        if self.use_pos:
            x = x + self.pos
                                                   
        x = self.dropout(self.x_embedding((x).permute(0, 2, 1)))
        x2 = x.reshape(-1, self.d_model).unsqueeze(1) 
                    
        sb = self.dropout(self.sb_embedding(sb.permute(0, 2, 1)))       
        
        tb = tb.permute(0, 3, 1, 2).reshape(-1, self.N, self.tb_len)             
        tb = self.dropout(self.tb_embedding(tb))

        return x, x2, sb, tb, x2
    
    
def pos_encoding(len, c_in):
    W_pos = torch.empty((len, c_in))
    nn.init.uniform_(W_pos, -0.02, 0.02)

    return nn.Parameter(W_pos, requires_grad=True)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean 

