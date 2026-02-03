# -*- coding: utf-8 -*-
# model_reaction_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dataset_sdf import PAD_ID
from utils_geometry import kabsch_rotate
from chemical_pretrain.constants import *
from pretrain_model import MoleculePretrainModel, valid_pair_from_pad_mask


class CrossAttentionDecoderLayer(nn.Module):
    """
    带 Cross-Attention 的解码器层。
    允许产物 query 直接关注反应物 encoder 的输出。
    """

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

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # 1. Self Attention (Query 之间的交互)
        # x: (B, T_prod, D)
        q = k = v = x
        x2, _ = self.self_attn(q, k, v, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(x2))

        # 2. Cross Attention (Query 关注 Encoder Memory)
        # memory: (B, T_react, D)
        # memory_mask: (B, T_react) True for PAD
        if memory is not None:
            key_padding_mask = memory_mask  # (B, T_react)
            x2, attn_weights = self.cross_attn(query=x, key=memory, value=memory, key_padding_mask=key_padding_mask)
            x = self.norm2(x + self.dropout(x2))
        else:
            x = self.norm2(x)
            attn_weights = None

        # 3. FFN
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout(x2))
        return x, attn_weights


class ChemicalReactionModelV2(nn.Module):
    def __init__(self, pretrain_weights_path=None, class_weights=None):
        super().__init__()

        # === 1. 复用预训练的 Encoder ===
        # 我们只需要 Encoder 部分，Decoder 我们自己重写
        self.base = MoleculePretrainModel(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
            d_ff=D_FF, n_layers_enc=LAYER_ENC, n_layers_dec=0,  # 不用 base 的 decoder
            rbf_k=RBF_K, rbf_mu_max=RBF_MU_MAX, dropout=DROPOUT
        )

        # 加载权重 (只加载 Encoder 部分)
        if pretrain_weights_path and os.path.exists(pretrain_weights_path):
            print(f"[ModelV2] Loading Encoder weights from {pretrain_weights_path}")
            ckpt = torch.load(pretrain_weights_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            # 过滤掉 decoder 相关的权重，因为我们这层不需要 base.decoder
            base_dict = self.base.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in base_dict and v.shape == base_dict[k].shape}
            self.base.load_state_dict(filtered_dict, strict=False)

        # === 2. 新的 Decoder (带 Cross Attention) ===
        # 这让产物预测能直接“看到”反应物原子
        self.decoder_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT)
            for _ in range(6)  # 6层 Decoder
        ])

        self.max_prod_len = MAX_LEN
        # Learnable Queries: 相当于产物的“槽位”
        self.product_query = nn.Parameter(torch.randn(1, self.max_prod_len, D_MODEL) * 0.02)

        # Heads
        self.head_atom = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.head_coord = nn.Linear(D_MODEL, 3)
        self.head_dist = nn.Linear(D_MODEL, D_MODEL)  # 简单映射用于dist loss辅助

        # Class Weights (处理 C/H 不平衡)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, r_ids, r_coords, r_mask, p_ids=None, p_coords=None, p_mask=None, coord_loss_weight=1.0):
        B = r_ids.shape[0]
        device = r_ids.device

        # === 1. Encoder (Reactants) ===
        # 获取反应物的特征表示
        x_r = self.base.atom_emb(r_ids)
        dist_r = self.base.pairwise_distance(r_coords, r_mask)
        e_ij_r = self.base.rbf(dist_r)
        b_ij_r = self.base.rbf_mlp(e_ij_r)

        valid_r = ~r_mask
        pair_valid_r = valid_r.unsqueeze(1) & valid_r.unsqueeze(2)
        b0_r = b_ij_r * pair_valid_r.unsqueeze(-1).float()

        # x_enc: (B, N_react, D) - 包含了每个反应物原子的丰富信息
        x_enc, _ = self.base.encoder(x_r, b0_r, r_mask)

        # === 2. Decoder (Products) ===
        # 初始化 Query
        # 以前你只加了 global_feat，现在我们通过 Cross-Attn 动态获取信息
        # 但我们仍然可以把 global context 加到 query 初始化里作为“底色”
        mask_float = valid_r.float().unsqueeze(-1)
        global_feat = (x_enc * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-6)

        dec_in = self.product_query.expand(B, -1, -1) + global_feat.unsqueeze(1)  # (B, T_prod, D)
        dec_in = self.base.pe(dec_in)  # 加位置编码

        # 逐层解码
        x_dec = dec_in
        # r_mask is True for PAD. CrossAttention expects True for PAD.
        memory_key_padding_mask = r_mask

        for layer in self.decoder_layers:
            # Cross Attention: x_dec 关注 x_enc
            x_dec, _ = layer(x_dec, memory=x_enc, memory_mask=memory_key_padding_mask)

        # === 3. Prediction Heads ===
        logits_atom = self.head_atom(x_dec)  # (B, T_prod, Vocab)
        pred_coords = self.head_coord(x_dec)  # (B, T_prod, 3)

        # === 4. Loss Calculation ===
        if p_ids is not None:
            T = min(self.max_prod_len, p_ids.shape[1])

            p_ids_slice = p_ids[:, :T]
            p_coords_slice = p_coords[:, :T, :]
            p_mask_slice = p_mask[:, :T]

            logits_slice = logits_atom[:, :T, :]
            coords_slice = pred_coords[:, :T, :]

            # Loss 1: Atom Classification (Weighted!)
            # Flatten
            logits_flat = logits_slice.reshape(-1, VOCAB_SIZE)
            targets_flat = p_ids_slice.reshape(-1)

            if self.class_weights is not None:
                l_atom = F.cross_entropy(logits_flat, targets_flat, weight=self.class_weights, ignore_index=PAD_ID)
            else:
                l_atom = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_ID)

            # Loss 2: Coordinates (Kabsch Aligned)
            # 只有在非 Mask 区域计算
            valid_p = ~p_mask_slice

            # 先对齐
            coords_aligned = kabsch_rotate(coords_slice, p_coords_slice, p_mask_slice)

            # 使用 MSE 或 L1，L1 对异常值更鲁棒
            l_coord = F.l1_loss(coords_aligned[valid_p], p_coords_slice[valid_p])

            # Loss 3: Distance Matrix Loss (辅助几何结构学习)
            # 计算预测坐标内部的距离矩阵 vs 真实距离矩阵
            # 这有助于模型学习内部结构，而不受绝对位置/旋转影响
            pred_dist = torch.cdist(coords_slice, coords_slice)
            true_dist = torch.cdist(p_coords_slice, p_coords_slice)

            # 创建 pair mask (B, T, T)
            pair_mask = valid_p.unsqueeze(1) & valid_p.unsqueeze(2)
            l_dist = F.mse_loss(pred_dist[pair_mask], true_dist[pair_mask])

            # === 组合 Loss ===
            # 动态调整 coord 权重，或者给 Atom 更高的权重
            total_loss = 2.0 * l_atom + coord_loss_weight * (1.0 * l_coord + 1.0 * l_dist)

            return total_loss, {
                "loss": total_loss.item(),
                "l_atom": l_atom.item(),
                "l_coord": l_coord.item(),
                "l_dist": l_dist.item()
            }, (logits_atom, coords_aligned)

        return logits_atom, pred_coords