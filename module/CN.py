"""
The proposed conditional networks: BCFM & CCFM
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from .Transformer import  TransformerLayer, CrossTransformerLayer, MultiHeadAttention
from timm.models.layers import trunc_normal_
import math

class BCFM(nn.Module):
    """
    Bidirectional Conditional Feature Modulation (BCFM)
    """
    def __init__(self, dim_model=128):
        super(BCFM, self).__init__()
        self.rf_p = PFP(dim_model=dim_model, init_k_size=5, depth=4, scale_factor=4)
        self.tr_d = CrossTransformerLayer(dim_model=dim_model, n_head=4, drop_rate=0.1)
        
        self.rf_d = PFP(dim_model=dim_model, init_k_size=3, depth=4, scale_factor=2)
        self.tr_p = CrossTransformerLayer(dim_model=dim_model, n_head=4, drop_rate=0.1)

    def forward(self, x_p, x_d, p_mask, d_mask):
        x_p = self.rf_p(x_p, x_d, p_mask, d_mask)
        x_d = self.tr_d(x_d, x_p, p_mask)
        
        y_d = self.rf_d(x_d, x_p, d_mask, p_mask)
        y_p = self.tr_p(x_p, y_d, d_mask)
        
        return y_p, y_d


class CCFM(nn.Module):
    """
    Catalysis-aware Conditional Feature Modulation (CCFM)
    """
    def __init__(self, dim_model=128, num_head=4):
        super(CCFM, self).__init__()
        self.dim_model = dim_model
        
        self.W_p = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU())

        self.W_d = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU())
        
        self.all = nn.Linear(2*dim_model, dim_model, False)

        self.k_p = nn.Linear(dim_model, dim_model, False)
        self.k_d = nn.Linear(dim_model, dim_model, False)

        self.n_head = num_head
        self.head_dim = dim_model // num_head
        self.dim_model = dim_model
        
    def forward(self, x_p, x_d, p_mask, d_mask):
        bs = x_p.size(0)

        g_p = torch.mean(x_p, dim=1, keepdim=True)
        g_p = self.W_p(g_p)
        
        g_d = torch.mean(x_d, dim=1, keepdim=True)
        g_d = self.W_d(g_d)
        
        g_all = torch.cat([g_p, g_d], -1)

        q_all = self.all(g_all).view(bs, -1, self.n_head, self.head_dim)
        k_p = self.k_p(x_p).view(bs, -1, self.n_head, self.head_dim)
        k_d = self.k_d(x_d).view(bs, -1, self.n_head, self.head_dim)
        v_p = x_p.view(bs, -1, self.n_head, self.head_dim)
        v_d = x_d.view(bs, -1, self.n_head, self.head_dim)

        q_all, k_p, k_d, v_p, v_d = q_all.transpose(1, 2), k_p.transpose(1, 2), k_d.transpose(1, 2), v_p.transpose(1, 2), v_d.transpose(1, 2)

        p_scores = torch.matmul(k_p, q_all.transpose(-2, -1))/math.sqrt(self.head_dim)
        p_scores = p_scores.masked_fill(p_mask.unsqueeze(1).unsqueeze(-1), -1e9)
        
        d_scores = torch.matmul(k_d, q_all.transpose(-2, -1))/math.sqrt(self.head_dim)
        d_scores = d_scores.masked_fill(d_mask.unsqueeze(1).unsqueeze(-1), -1e9)
        
        attention_p = F.softmax(p_scores, dim=2)
        attention_d = F.softmax(d_scores, dim=2)

        rep_p = v_p * attention_p
        rep_p = rep_p.transpose(1, 2).contiguous().view(bs, -1, self.dim_model)
        rep_p = rep_p.sum(dim=1)

        rep_d = v_d * attention_d
        rep_d = rep_d.transpose(1, 2).contiguous().view(bs, -1, self.dim_model)
        rep_d = rep_d.sum(dim=1)

        rep = rep_d + rep_p

        return rep
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PFP(nn.Module):
    """
    Poly focal perception block
    """
    def __init__(self, dim_model=128, init_k_size=5, depth=4, scale_factor=4):
        super(PFP, self).__init__()
        self.dim_model = dim_model
        self.ffn = MLP(dim_model=dim_model, dim_hidden=dim_model*4, dim_out=dim_model)
        self.attn = PKA(dim_model=dim_model, init_k_size=init_k_size, depth=depth, scale_factor=scale_factor)
        self.norm1 = nn.LayerNorm(dim_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim_model, elementwise_affine=False, eps=1e-6)

        self.w_d = nn.Linear(dim_model, dim_model, bias=False)
        self.w_p = nn.Linear(dim_model, dim_model, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_model, 4 * dim_model)
        )

        self.pool = MaskedAveragePooling(keepdim=True)
        
        self.apply(self._init_weights)
    def forward(self, x_p, x_d, p_mask, d_mask):

        g_d = self.w_d(x_d)
        g_p = self.w_p(x_p)
        g_d = self.pool(g_d, d_mask)

        p_scores = torch.bmm(g_p, g_d.transpose(1, 2)) / math.sqrt(self.dim_model)
        p_scores = p_scores.masked_fill(p_mask.unsqueeze(-1), -1e9)
        attention_p = F.softmax(p_scores, dim=1)
        g_p = g_p * attention_p
        c = g_p.sum(dim=1)

        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)

        x = x_p
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

class PKA(nn.Module):
    """
    Poly kernel attention
    """
    def __init__(self, dim_model=128, init_k_size=3, depth=4, scale_factor=4):
        super(PKA, self).__init__()
        self.conv_list = nn.ModuleList()
        self.dim_model = dim_model
        for k in range(depth):
            k_size = init_k_size + k*scale_factor
            self.conv_list.append(
                nn.Sequential(
                    nn.Conv1d(dim_model, dim_model, kernel_size=k_size, groups=dim_model, stride=1, padding=k_size//2, bias=False),
                    nn.BatchNorm1d(dim_model),
                    nn.ReLU()
                )
            )

        self.out_ffn = nn.Linear(dim_model, dim_model)

    def forward(self, x_p):
        x_p = x_p.transpose(1, 2)
        x_all = 0
        for i, layer in enumerate(self.conv_list):
            x_p = layer(x_p)
            x_all = x_all + x_p
        x_all = x_all.transpose(1, 2)
        x_all = self.out_ffn(x_all)
        return x_all

class MLP(nn.Module):
    def __init__(self, dim_model=128, dim_hidden=512, dim_out=128, drop_rate=0.1):
        super(MLP, self).__init__()
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
        

class MaskedAveragePooling(nn.Module):
    def __init__(self, keepdim=False):
        super(MaskedAveragePooling, self).__init__()
        self.keepdim = keepdim

    def forward(self, x, mask):
        """
        x: [batch, seq, dim]
        mask: padding mask ; True represents padding token
        """
        mask_expanded = mask.unsqueeze(-1)
        x_masked = x.masked_fill(mask_expanded, 0)
        seq_lengths = mask.size(1) - mask.float().sum(dim=1, keepdim=True)
        seq_lengths = torch.clamp(seq_lengths, min=1)
        avg_pool = x_masked.sum(dim=1) / seq_lengths

        if self.keepdim:
            avg_pool = avg_pool.unsqueeze(1)

        return avg_pool




    