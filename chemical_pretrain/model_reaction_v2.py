# -*- coding: utf-8 -*-
# model_reaction_v2.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from chemical_pretrain.constants import VOCAB_SIZE, PAD_ID, MASK_ID
from chemical_pretrain.utils_geometry import kabsch_rotate 
from pretrain_model import MoleculePretrainModel

# 超参数配置
D_MODEL = 256
N_HEADS = 8
D_FF = 1024
LAYER_ENC = 6
DROPOUT = 0.1
RBF_K = 128
RBF_MU_MAX = 30.0
MAX_LEN = 128

# =========================================================
# 升级后的预测头 (2层 MLP + LayerNorm + GELU)
# =========================================================
class MLPHead(nn.Module):
    """用于原子类型和坐标预测的 2 层 MLP 头"""
    def __init__(self, d_model, d_out, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = D_MODEL * 2 
            
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LayerNorm(d_ff), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_out)
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

# =========================================================
# Decoder Layer (现在只接受一个合并后的浮点型 attn_mask)
# =========================================================

class CrossAttentionDecoderLayer(nn.Module):
    """带 Cross-Attention 的解码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None):
        # 1. Self Attention (Masked)
        # 传入 (N_heads * B) x T x T 的浮点型合并掩码 (tgt_mask) 
        x2, _ = self.self_attn(
            x, x, x, 
            attn_mask=tgt_mask, 
            key_padding_mask=None # 掩码已合并到 attn_mask 中
        )
        x = self.norm1(x + self.dropout(x2))

        # 2. Cross Attention (Query=Decoder, Key/Val=Encoder)
        if memory is not None:
            # Cross-Attention 中传入布尔型 key_padding_mask 是兼容的
            x2, attn_weights = self.cross_attn(
                query=x, 
                key=memory, 
                value=memory, 
                key_padding_mask=memory_mask
            )
            x = self.norm2(x + self.dropout(x2))
        else:
            x = self.norm2(x)
            attn_weights = None

        # 3. FFN
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout(x2))
        return x, attn_weights

# =========================================================
# 主模型 ChemicalReactionModelV2 (实现掩码合并与重复)
# =========================================================

class ChemicalReactionModelV2(nn.Module):
    def __init__(self, pretrain_weights_path=None, class_weights=None):
        super().__init__()

        # === 1. 基础 Encoder (复用预训练模型结构) ===
        self.base = MoleculePretrainModel(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
            d_ff=D_FF, n_layers_enc=LAYER_ENC, n_layers_dec=0,
            rbf_k=RBF_K, rbf_mu_max=RBF_MU_MAX, dropout=DROPOUT
        )

        # 【修复点】：存储 N_HEADS
        self.n_heads = N_HEADS 

        # 加载预训练权重 (保持原有的智能加载逻辑)
        if pretrain_weights_path and os.path.exists(pretrain_weights_path):
            print(f"[ModelV2] 正在加载预训练权重: {pretrain_weights_path}")
            try:
                ckpt = torch.load(pretrain_weights_path, map_location='cpu')
                src_state = ckpt.get('model_state_dict', ckpt)
                new_state_dict = {}
                loaded_keys = 0
                tgt_state = self.state_dict()
                
                for k, v in src_state.items():
                    if k.startswith("layers."):
                        new_k = f"base.encoder.{k}"
                    elif k.startswith("atom_emb") or k.startswith("rbf"):
                        new_k = f"base.{k}"
                    else:
                        new_k = k

                    if new_k in tgt_state and v.shape == tgt_state[new_k].shape:
                        new_state_dict[new_k] = v
                        loaded_keys += 1
                
                if loaded_keys > 0:
                    self.base.load_state_dict(new_state_dict, strict=False)
                    print(f"  ✅ 成功映射并加载了 {loaded_keys} 个参数张量。")
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")

        # === 2. Decoder (改进版) ===
        self.decoder_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT)
            for _ in range(6)
        ])

        self.sos_emb = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        
        # 使用 MLPHead 替换单层 Linear
        self.head_atom = MLPHead(D_MODEL, VOCAB_SIZE, d_ff=D_FF, dropout=DROPOUT)
        self.head_coord = MLPHead(D_MODEL, 3, d_ff=D_FF, dropout=DROPOUT)
        self.head_dist = nn.Linear(D_MODEL, D_MODEL)

        self.max_prod_len = MAX_LEN

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head_dist.weight)
        nn.init.zeros_(self.head_dist.bias)

    def generate_square_subsequent_mask(self, sz, device):
        """生成因果掩码 (T x T)"""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        # 下三角（即当前及之前位置）为 0.0，上三角（即之后位置）为 -inf
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, r_ids, r_coords, r_mask, p_ids=None, p_coords=None, p_mask=None, coord_loss_weight=1.0):
        """
        r_ids: 反应物原子ID
        p_ids: 产物原子ID (Target)
        """
        B = r_ids.shape[0]
        device = r_ids.device

        # -------------------------
        # 1. Encoder (Reactants) - 代码省略，保持不变
        # -------------------------
        x_r = self.base.atom_emb(r_ids)
        dist_r = self.base.pairwise_distance(r_coords, r_mask)
        e_ij_r = self.base.rbf(dist_r)
        b_ij_r = self.base.rbf_mlp(e_ij_r)
        valid_r = ~r_mask
        pair_valid_r = valid_r.unsqueeze(1) & valid_r.unsqueeze(2)
        b0_r = b_ij_r * pair_valid_r.unsqueeze(-1).float()
        x_enc, _ = self.base.encoder(x_r, b0_r, r_mask)
        mask_float = valid_r.float().unsqueeze(-1)
        global_feat = (x_enc * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-6)

        if p_ids is not None:
            # === Training Mode (Teacher Forcing) ===
            
            T = min(self.max_prod_len, p_ids.shape[1])
            p_ids_slice = p_ids[:, :T]
            p_coords_slice = p_coords[:, :T, :]
            p_mask_slice = p_mask[:, :T]

            # 2. 构建 Decoder Input
            tgt_emb = self.base.atom_emb(p_ids_slice)
            sos = self.sos_emb.expand(B, -1, -1)
            dec_in_emb = torch.cat([sos, tgt_emb[:, :-1, :]], dim=1)
            dec_in = self.base.pe(dec_in_emb + global_feat.unsqueeze(1))

            # 3. 生成并合并掩码
            # Causal Mask (T, T) - 浮点型
            tgt_mask_causal = self.generate_square_subsequent_mask(T, device)
            
            # Padding Mask (B, T) - 布尔型
            sos_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            tgt_key_padding_mask_bool = torch.cat([sos_mask, p_mask_slice[:, :-1]], dim=1) 

            # 1. 转换 key_padding_mask 为浮点掩码 (True -> -inf, False -> 0.0)
            # (B, T) -> (B, 1, T)
            # True 标记的位置（即 PAD）会被设置为 -inf，从而在 Attention 中被忽略
            tgt_padding_mask_float = tgt_key_padding_mask_bool.float().masked_fill(tgt_key_padding_mask_bool, float('-inf')).unsqueeze(1) 

            # 2. 合并：(1, T, T) Causal Mask + (B, 1, T) Padding Mask -> (B, T, T)
            tgt_mask_combined = tgt_mask_causal.unsqueeze(0) + tgt_padding_mask_float

            # 3. 【核心修复】：重复 N_HEADS 次： (B, T, T) -> (N_heads * B, T, T)
            # 解决 PyTorch 对 3D 掩码的维度要求
            tgt_mask_final = tgt_mask_combined.repeat(self.n_heads, 1, 1)

            # -------------------------
            # 4. Decoder Forward
            # -------------------------
            x_dec = dec_in
            memory_key_padding_mask = r_mask

            for layer in self.decoder_layers:
                x_dec, _ = layer(
                    x_dec, 
                    memory=x_enc, 
                    tgt_mask=tgt_mask_final, # 传入 (N_heads * B) x T x T 浮点型
                    memory_mask=memory_key_padding_mask,
                    tgt_key_padding_mask=None # 设为 None，因为已合并
                )
            
            # -------------------------
            # 5. Output Heads & Loss
            # -------------------------
            logits_atom = self.head_atom(x_dec) 
            pred_coords = self.head_coord(x_dec)

            # --- Loss Calculation ---
            logits_flat = logits_atom.reshape(-1, VOCAB_SIZE)
            targets_flat = p_ids_slice.reshape(-1)

            # Atom Loss
            if self.class_weights is not None:
                safe_weights = self.class_weights.to(device).clamp(min=0.1)
                l_atom = F.cross_entropy(logits_flat, targets_flat, weight=safe_weights, ignore_index=PAD_ID)
            else:
                l_atom = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_ID)

            # Coord Loss
            valid_p = ~p_mask_slice
            coords_aligned = kabsch_rotate(pred_coords, p_coords_slice, p_mask_slice)
            l_coord = F.smooth_l1_loss(coords_aligned[valid_p], p_coords_slice[valid_p])

            # Dist Loss
            pred_dist = torch.cdist(pred_coords, pred_coords)
            true_dist = torch.cdist(p_coords_slice, p_coords_slice)
            pair_mask = valid_p.unsqueeze(1) & valid_p.unsqueeze(2)
            l_dist = F.huber_loss(pred_dist[pair_mask], true_dist[pair_mask], delta=1.0)

            total_loss = 2.0 * l_atom + coord_loss_weight * (1.0 * l_coord + 0.5 * l_dist)

            return total_loss, {
                "l_atom": l_atom.item(), 
                "l_coord": l_coord.item(), 
                "l_dist": l_dist.item()
            }, (logits_atom, coords_aligned)

        else:
            return None, None