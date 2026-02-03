import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json

# ================= 1. 基础组件 (完全匹配权重结构) =================

class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory=None, mask=None):
        if memory is None: memory = x
        B, L, _ = x.shape
        S = memory.shape[1]
        
        q = self.q_proj(x).reshape(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(B, S, self.n_head, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.o_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.mha = CustomMultiheadAttention(d_model, n_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = x + self.mha(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, n_head)
        self.cross_attn = CustomMultiheadAttention(d_model, n_head)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, memory, tgt_mask=None):
        x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
        x = x + self.cross_attn(self.norm2(x), memory=memory)
        x = x + self.ffn(self.norm3(x))
        return x

class RBF(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('log_sigma', torch.zeros(1))
        self.register_buffer('log_gamma', torch.zeros(1))

# ================= 2. 主结构 (包含修复的 ParameterDict) =================

class MoleculePretrainModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=8, n_layers=6):
        super().__init__()
        
        # 1. Embedding container
        self.atom_emb = nn.ModuleDict({
            'emb': nn.Embedding(vocab_size, d_model)
        })
        
        # 2. Positional Encoding (关键修复: 使用 ParameterDict)
        self.pe = nn.ParameterDict({
            'pe': nn.Parameter(torch.randn(1, 512, d_model))
        })
        
        # 3. RBF placeholders
        self.rbf = RBF()
        self.rbf_mlp = nn.ModuleDict({
            'net': nn.Sequential(nn.Linear(1,1), nn.Linear(1,1))
        })

        # 4. Encoder
        self.encoder = nn.ModuleDict({
            'layers': nn.ModuleList([EncoderLayer(d_model, n_head) for _ in range(n_layers)])
        })
        
        # 5. Heads (placeholders for loading)
        self.dist_head = nn.ModuleDict({
            'dense': nn.Linear(d_model, d_model),
            'layer_norm': nn.LayerNorm(d_model),
            'out_proj': nn.Linear(d_model, d_model) 
        })
        self.head_atom = nn.ModuleDict({
            'dense': nn.Linear(d_model, d_model),
            'layer_norm': nn.LayerNorm(d_model),
            'out_proj': nn.Linear(d_model, vocab_size)
        })
        self.pair2coord_proj = nn.ModuleDict({
            'net': nn.Sequential(nn.Linear(1,1), nn.Linear(1,1))
        })

class ChemicalReactionModelV2(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=8):
        super().__init__()
        self.d_model = d_model
        
        # Base
        self.base = MoleculePretrainModel(vocab_size, d_model, n_head, n_layers=6)
        
        # Decoder components
        self.product_query = nn.Parameter(torch.randn(1, 100, d_model))
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_head) for _ in range(6)])
        
        # Task Heads
        self.head_atom = nn.Linear(d_model, vocab_size)
        self.head_coord = nn.Linear(d_model, 3)
        self.head_dist = nn.Linear(d_model, 1)
        
        self.class_weights = nn.Parameter(torch.ones(vocab_size))

    def forward_encoder(self, src_seq):
        x = self.base.atom_emb['emb'](src_seq)
        L = x.size(1)
        # 从 ParameterDict 中取出 pe
        pe = self.base.pe['pe'][:, :L, :].to(x.device)
        x = x + pe
        
        for layer in self.base.encoder['layers']:
            x = layer(x)
        return x

    def forward_decoder(self, memory):
        B = memory.size(0)
        x = self.product_query.expand(B, -1, -1)
        for layer in self.decoder_layers:
            x = layer(x, memory=memory)
        return x

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['atom2id'], data['id2atom']
