import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from timm.models.layers import DropPath, trunc_normal_
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model=128, n_head=4, drop_rate=0.1):
        super().__init__()
        assert dim_model % n_head == 0, "d_model must be divisible by heads"
        self.dim_head = dim_model // n_head
        self.num_head = n_head
        self.dim_model = dim_model

        self.q = nn.Linear(dim_model, dim_model, bias=False)
        self.k = nn.Linear(dim_model, dim_model, bias=False)
        self.v = nn.Linear(dim_model, dim_model, bias=False)

        self.dropout_att = nn.Dropout(drop_rate)
        self.dropout_fc = nn.Dropout(drop_rate)

        self.out = nn.Linear(dim_model, dim_model, bias=False)

    def attention(self, q, k, v, scale, mask=None, dropout=None):
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(scale)
        # mask the attention scores of padding operation
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            att = att.masked_fill(mask, -1e9)

        att = F.softmax(att, dim=-1)
        if dropout is not None:
            att = dropout(att)

        output = torch.matmul(att, v)
        return output, att
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # 进行线性操作划分为成 h 个头
        k = self.k(k).view(bs, -1, self.num_head, self.dim_head)
        q = self.q(q).view(bs, -1, self.num_head, self.dim_head)
        v = self.v(v).view(bs, -1, self.num_head, self.dim_head)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        out, att = self.attention(q, k, v, scale=self.dim_head, mask=mask, dropout=self.dropout_att)

        out = out.transpose(1,2).contiguous().view(bs, -1, self.dim_model)

        out = self.dropout_fc(self.out(out))

        return out, att
    
class FFN(nn.Module):
    def __init__(self, dim_model=128, dim_hidden=512, dim_out=128, drop_rate=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_hidden),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim_hidden, dim_out),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = self.ffn(x)
        return x
    
    
class TransformerLayer(nn.Module):
    def __init__(self, dim_model=128, n_head=4, drop_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.msa = MultiHeadAttention(dim_model=dim_model, n_head=n_head, drop_rate=drop_rate)
        self.ffn = FFN(dim_model, dim_model*4, dim_model, drop_rate=drop_rate)
        self.LN1 = nn.LayerNorm(dim_model)
        self.LN2 = nn.LayerNorm(dim_model)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask=None):
        res = x
        x = self.LN1(x)
        x_att, att_map = self.msa(x, x, x, mask=mask)
        x = x_att + res
        y = self.ffn(self.LN2(x)) + x
        return y, att_map
    
    
class CrossTransformerLayer(nn.Module):
    def __init__(self, dim_model=128, n_head=4, drop_rate=0.1):
        super(CrossTransformerLayer, self).__init__()
        self.msa = MultiHeadAttention(dim_model=dim_model, n_head=n_head, drop_rate=drop_rate)
        self.ffn = FFN(dim_model, dim_model*4, dim_model, drop_rate=drop_rate)
        self.LN1 = nn.LayerNorm(dim_model)
        self.LN2 = nn.LayerNorm(dim_model)
        self.LN3 = nn.LayerNorm(dim_model)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, target, source, mask=None):
        target_ln = self.LN1(target)
        source_ln = self.LN2(source)
        x_att, _ = self.msa(target_ln, source_ln, source_ln, mask=mask)
        x_att = x_att + target
        y = self.ffn(self.LN3(x_att)) + x_att
        return y