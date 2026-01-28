import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import CrossEncoder, CrossEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


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

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(configs.seq_len,configs.pred_len)
        self.Linear_Trend = nn.Linear(configs.seq_len,configs.pred_len)

    def forward(self, x, x_mark_enc):
        sx, tx = self.decompsition(x)                   
        sy = self.Linear_Seasonal(sx)
        ty = self.Linear_Trend(tx)
        dec_out = (sy + ty)

        return dec_out


class Linear(nn.Module):

    def __init__(self, configs):
        super(Linear, self).__init__()        
        self.model = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x, x_mark_enc):
        # x: (B, C, L)                                                          
        dec_out = self.model(x)

        return dec_out


class MLP(nn.Module):

    def __init__(self, configs):
        super(MLP, self).__init__()        
        self.model = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )

    def forward(self, x, x_mark_enc):
        # x: (B, C, L)                                                  
        dec_out = self.model(x)

        return dec_out


class iTransformer(nn.Module):

    def __init__(self, configs):
        super(iTransformer, self).__init__()

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.enc_in, configs.d_model, configs.dropout)
        
        self.encoder = CrossEncoder(
            [
                CrossEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x, x_mark_enc):
        # x: (B, C, L)
        x, cross = self.enc_embedding(x, x_mark_enc)
        enc_out, attns = self.encoder(x, cross, attn_mask=None)                                                           
        dec_out = self.projector(enc_out)

        return dec_out
