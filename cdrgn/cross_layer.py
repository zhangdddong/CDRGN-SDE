import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, dim, dropout_rate=0.3, seg_dim=10):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.dim_linear = nn.Linear(1, seg_dim)
        self.linear = nn.Linear(seg_len, dim)
        self.res_linear = nn.Linear(dim * seg_dim * seg_dim, dim)

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.norm_layer = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.dim_linear(x)
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)     # batch_size, dsw_dim, seg_num, seg_dim
        x_embed = rearrange(x_embed, 'b d seg_num d_model -> b (d seg_num d_model)')
        x_embed = self.dropout(x_embed)
        x_embed = self.res_linear(x_embed)
        x_embed = self.norm_layer(x_embed)

        return x_embed