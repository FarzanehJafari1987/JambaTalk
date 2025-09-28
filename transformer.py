import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn

class LowRankLearnedRoPE(nn.Module):
    def __init__(self, d_model, rank=16):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model
        self.rank = rank

        # Low-rank projection instead of full dense
        self.proj = nn.Sequential(
            nn.Linear(d_model, rank, bias=False),
            nn.Linear(rank, d_model, bias=False)
        )

    def forward(self, x, pos: torch.LongTensor = None):
        B, T, D = x.shape
        pos_idx = pos if pos is not None else torch.arange(T, device=x.device)

        # Compute base RoPE angles only for needed positions
        theta = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device).float() / D))
        angles = pos_idx[:, None] * theta[None, :]  # [T, D/2]
        sin_pos, cos_pos = torch.sin(angles), torch.cos(angles)

        # Learnable modification
        base_emb = torch.cat([sin_pos, cos_pos], dim=-1)  # [T, D]
        learned_emb = self.proj(base_emb)  # [T, D]
        learned_sin, learned_cos = learned_emb.chunk(2, dim=-1)

        sin_final = sin_pos + learned_sin
        cos_final = cos_pos + learned_cos

        # Apply rotation
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        x_rot_even = x_even * cos_final.unsqueeze(0) - x_odd * sin_final.unsqueeze(0)
        x_rot_odd  = x_even * sin_final.unsqueeze(0) + x_odd * cos_final.unsqueeze(0)

        return torch.stack([x_rot_even, x_rot_odd], dim=-1).reshape(B, T, D)

class TransformerDecoderLayerGQA_RoPE(nn.Module):
    def __init__(self, d_model, n_query_heads, n_kv_heads, dim_feedforward=2048, dropout=0.1, max_seq_len=600):
        super().__init__()
        assert d_model % n_query_heads == 0
        assert d_model % n_kv_heads == 0

        self.d_model = d_model
        self.n_q = n_query_heads
        self.n_kv = n_kv_heads
        self.q_head_dim = d_model // n_query_heads
        self.kv_head_dim = d_model // n_kv_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Learnable positional embedding
        self.lrl_rope = LowRankLearnedRoPE(d_model)

    def forward(self, tgt, memory=None):
        x = tgt
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # # Apply learnable RoPE
        q = self.lrl_rope(q)  # [B, T, D]
        k = self.lrl_rope(k)  # [B, T, D]

        # Reshape for attention
        q = q.view(B, T, self.n_q, self.q_head_dim).transpose(1, 2)  # [B, n_q, T, d]
        k = k.view(B, T, self.n_kv, self.kv_head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.kv_head_dim).transpose(1, 2)

        # GQA logic: repeat k, v if needed
        if self.n_q != self.n_kv:
            repeat_factor = self.n_q // self.n_kv
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.q_head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.out_proj(attn_output)

        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x
