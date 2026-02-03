# -*- coding: utf-8 -*-
# model_components.py
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 1) RBF: e^{ij} = exp(-γ (||r_i - r_j|| - μ)^2) ============
class RBFEncoder(nn.Module):
    """
    学习型RBF层：
      输入: pairwise 距离 D (B,N,N)
      输出: RBF特征 E (B,N,N,K)
      参数: μ(K), logσ(K) 以及可选γ(标量)
    """
    def __init__(self, num_k: int = 64, mu_max: float = 10.0):
        super().__init__()
        self.K = num_k
        # 初始化μ为等距bins，σ初值为(间隔/2)
        mu = torch.linspace(0.0, mu_max, steps=num_k)
        sigma = torch.full((num_k,), (mu_max / (num_k - 1 + 1e-6)) * 0.5)
        self.mu = nn.Parameter(mu)           # (K,)
        self.log_sigma = nn.Parameter(sigma.log())  # (K,)
        self.log_gamma = nn.Parameter(torch.zeros(()))  # γ（可学习缩放）

        # 方便外部使用的 SmoothL1 / CE 等 loss 可以基于 hard_bucket 生成的标签

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (B,N,N)
        B, N, _ = dist.shape
        mu = self.mu.view(1, 1, 1, self.K)                     # (1,1,1,K)
        sigma = self.log_sigma.exp().view(1, 1, 1, self.K)     # (1,1,1,K)
        gamma = self.log_gamma.exp()                           # 标量 γ>0
        d = dist.unsqueeze(-1)                                 # (B,N,N,1)
        e = torch.exp(-gamma * ((d - mu) ** 2) / (2 * sigma ** 2))  # (B,N,N,K)
        return e

    @torch.no_grad()
    def hard_bucket(self, dist: torch.Tensor) -> torch.Tensor:
        """
        把连续距离离散到离μ最近的桶，作为成对分类标签（B,N,N）
        """
        d = dist.unsqueeze(-1)  # (B,N,N,1)
        mu = self.mu.view(1, 1, 1, self.K)
        idx = torch.argmin((d - mu).abs(), dim=-1)
        return idx  # long


# ============ 2) 位置编码：标准sin/cos（只在 decoder 使用） ============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)      # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        self.register_buffer('pe', pe)  # 不参与训练

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,D) -> 返回 (B,N,D) 加上位置编码
        """
        N = x.size(1)
        return x + self.pe[:N, :].unsqueeze(0).type_as(x)


# ============ 3) 带成对偏置并层间可更新的多头自注意力 ============
class MultiHeadSelfAttentionWithPair(nn.Module):
    """
    MOL-AE / Uni-Mol 风格注意力：
      注意力 logits = QK^T / sqrt(d_k) + b^l_{ij,h}
      层间偏置更新：b^{l+1}_{ij,h} = b^l_{ij,h} + Q_i^h (K_j^h)^T
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B,N,D) -> (B,H,N,d_k)
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.d_k).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,                    # (B,N,D)
        pair_bias: torch.Tensor,            # (B,N,N,H)
        pad_mask: Optional[torch.Tensor]    # (B,N) True为PAD
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        # 线性映射
        q = self._shape(self.q_proj(x))  # (B,H,N,d_k)
        k = self._shape(self.k_proj(x))  # (B,H,N,d_k)
        v = self._shape(self.v_proj(x))  # (B,H,N,d_k)

        # 注意力logits：QK^T / sqrt(dk) + b^l
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,N,N)
        # 将 pair_bias (B,N,N,H) -> (B,H,N,N)
        pb = pair_bias.permute(0, 3, 1, 2).contiguous()
        attn_logits = attn_logits + pb

        if pad_mask is not None:
            # key位置padding -> -inf
            key_mask = pad_mask.unsqueeze(1).unsqueeze(2)      # (B,1,1,N)
            attn_logits = attn_logits.masked_fill(key_mask, float("-inf"))

        # softmax
        attn = F.softmax(attn_logits, dim=-1)                  # (B,H,N,N)
        attn = self.dropout(attn)

        # 输出
        z = torch.matmul(attn, v)                              # (B,H,N,d_k)
        z = z.permute(0, 2, 1, 3).contiguous().view(B, N, D)   # (B,N,D)
        out = self.o_proj(z)

        # 成对偏置更新（不除sqrt(dk)）
        qk_unscaled = torch.matmul(q, k.transpose(-2, -1))     # (B,H,N,N)
        pair_next = pair_bias + qk_unscaled.permute(0, 2, 3, 1)  # (B,N,N,H)

        if pad_mask is not None:
            # 把 PAD 相关的 pair 置 0，避免数值积累
            m_i = (~pad_mask).float().unsqueeze(-1) * (~pad_mask).float().unsqueeze(1)  # (B,N,N)
            pair_next = pair_next * m_i.unsqueeze(-1)

        return out, pair_next, attn


# ============ 4) Transformer Block (残差 + LN) ============
class TransformerBlockWithPair(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadSelfAttentionWithPair(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pair_bias: torch.Tensor, pad_mask: Optional[torch.Tensor]):
        # 注意力
        attn_out, pair_next, _ = self.mha(x, pair_bias, pad_mask)  # Z_i^l
        # 残差 + LN
        z_tilde = self.ln1(x + attn_out)
        # 前馈 + 残差 + LN
        y = self.ffn(z_tilde)
        x_next = self.ln2(z_tilde + y)
        return x_next, pair_next
