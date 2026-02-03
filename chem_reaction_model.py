# -*- coding: utf-8 -*-
# chem_reaction_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dataset_sdf import PAD_ID
from utils_geometry import kabsch_rotate
from chemical_pretrain.constants import *
from pretrain_model import MoleculePretrainModel, valid_pair_from_pad_mask


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

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, tgt_is_causal=False):
        # 1. Self Attention
        # 注意：tgt_mask 需要是 causal mask 或者 None
        x2, _ = self.self_attn(
            x, x, x, 
            attn_mask=tgt_mask,
            is_causal=tgt_is_causal
        )
        x = self.norm1(x + self.dropout(x2))

        # 2. Cross Attention
        if memory is not None:
            key_padding_mask = memory_mask
            x2, attn_weights = self.cross_attn(
                x, memory, memory,
                key_padding_mask=key_padding_mask
            )
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
        self.base = MoleculePretrainModel(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
            d_ff=D_FF, n_layers_enc=LAYER_ENC, n_layers_dec=0,
            rbf_k=RBF_K, rbf_mu_max=RBF_MU_MAX, dropout=DROPOUT
        )

        # 修复：更好的权重加载逻辑
        if pretrain_weights_path and os.path.exists(pretrain_weights_path):
            print(f"[ModelV2] 从 {pretrain_weights_path} 加载预训练权重")
            try:
                ckpt = torch.load(pretrain_weights_path, map_location='cpu')
                
                # 尝试不同的键名
                if 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                else:
                    state_dict = ckpt
                
                # 修复：处理可能的前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 去掉 'module.' 前缀（如果是DataParallel保存的）
                    if k.startswith('module.'):
                        k = k[7:]
                    # 去掉 'base.' 前缀
                    if k.startswith('base.'):
                        k = k[5:]
                    new_state_dict[k] = v
                
                # 只加载匹配的权重
                base_dict = self.base.state_dict()
                filtered_dict = {}
                for k, v in new_state_dict.items():
                    if k in base_dict and v.shape == base_dict[k].shape:
                        filtered_dict[k] = v
                
                if len(filtered_dict) > 0:
                    self.base.load_state_dict(filtered_dict, strict=False)
                    print(f"[ModelV2] 成功加载 {len(filtered_dict)} 个权重")
                else:
                    print("[ModelV2] 警告：没有匹配的权重，使用随机初始化")
                    
            except Exception as e:
                print(f"[ModelV2] 加载权重失败: {e}")

        # === 2. 新的 Decoder (带 Cross Attention) ===
        self.decoder_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT)
            for _ in range(6)
        ])

        self.max_prod_len = MAX_LEN
        # Learnable Queries
        self.product_query = nn.Parameter(torch.randn(1, self.max_prod_len, D_MODEL) * 0.02)

        # Heads
        self.head_atom = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.head_coord = nn.Linear(D_MODEL, 3)
        self.head_dist = nn.Linear(D_MODEL, D_MODEL)

        # === 修复：更好的类别权重处理 ===
        if class_weights is not None:
            # 确保权重是张量
            if isinstance(class_weights, torch.Tensor):
                # 设置权重下限，防止H/C权重过低
                class_weights = class_weights.clone()
                # 确保PAD权重为0
                class_weights[PAD_ID] = 0.0
                # 设置最小权重为0.1
                valid_indices = (class_weights > 0)
                class_weights[valid_indices] = torch.clamp(class_weights[valid_indices], min=0.1, max=10.0)
                self.register_buffer('class_weights', class_weights)
            else:
                self.class_weights = None
            print(f"[ModelV2] 类别权重已设置，形状: {class_weights.shape}")
            
            # 打印重要原子的权重
            important_atoms = {2: "H", 7: "C", 8: "N", 9: "O", 16: "P", 17: "S", 18: "Cl"}
            for atom_id, symbol in important_atoms.items():
                if atom_id < len(class_weights):
                    weight_val = class_weights[atom_id].item() if torch.is_tensor(class_weights[atom_id]) else class_weights[atom_id]
                    print(f"  {symbol}(ID:{atom_id}): {weight_val:.4f}")
        else:
            self.class_weights = None
            print("[ModelV2] 警告: 未提供类别权重，使用标准交叉熵")

        # 初始化新增层的权重
        self._init_weights()

    def _init_weights(self):
        """初始化新增层的权重"""
        nn.init.xavier_uniform_(self.head_atom.weight)
        nn.init.zeros_(self.head_atom.bias)
        nn.init.xavier_uniform_(self.head_coord.weight, gain=0.01)  # 较小的初始化
        nn.init.zeros_(self.head_coord.bias)
        nn.init.xavier_uniform_(self.head_dist.weight)
        nn.init.zeros_(self.head_dist.bias)

    def _generate_causal_mask(self, seq_len, device):
        """生成因果掩码 (上三角为 -inf)"""
        # 创建上三角矩阵
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        # 将上三角部分设为 -inf，下三角和主对角线设为 0
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, r_ids, r_coords, r_mask, p_ids=None, p_coords=None, p_mask=None, coord_loss_weight=1.0):
        B = r_ids.shape[0]
        device = r_ids.device

        # === 1. Encoder (Reactants) ===
        x_r = self.base.atom_emb(r_ids)
        dist_r = self.base.pairwise_distance(r_coords, r_mask)
        e_ij_r = self.base.rbf(dist_r)
        b_ij_r = self.base.rbf_mlp(e_ij_r)

        valid_r = ~r_mask
        pair_valid_r = valid_r.unsqueeze(1) & valid_r.unsqueeze(2)
        b0_r = b_ij_r * pair_valid_r.unsqueeze(-1).float()

        x_enc, _ = self.base.encoder(x_r, b0_r, r_mask)

        # === 2. Decoder (Products) ===
        # 初始化 Query
        mask_float = valid_r.float().unsqueeze(-1)
        global_feat = (x_enc * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-6)

        dec_in = self.product_query.expand(B, -1, -1) + global_feat.unsqueeze(1)
        dec_in = self.base.pe(dec_in)

        # 生成因果掩码
        seq_len = dec_in.shape[1]
        causal_mask = self._generate_causal_mask(seq_len, device)

        # 逐层解码
        x_dec = dec_in
        memory_key_padding_mask = r_mask  # True for PAD

        for layer in self.decoder_layers:
            x_dec, _ = layer(
                x_dec, 
                memory=x_enc, 
                tgt_mask=causal_mask,  # 关键修复：添加因果掩码
                memory_mask=memory_key_padding_mask,
                tgt_is_causal=True  # 额外的因果标志
            )

        # === 3. Prediction Heads ===
        logits_atom = self.head_atom(x_dec)
        pred_coords = self.head_coord(x_dec)

        # === 4. Loss Calculation ===
        if p_ids is not None:
            T = min(self.max_prod_len, p_ids.shape[1])

            p_ids_slice = p_ids[:, :T]
            p_coords_slice = p_coords[:, :T, :]
            p_mask_slice = p_mask[:, :T]

            logits_slice = logits_atom[:, :T, :]
            coords_slice = pred_coords[:, :T, :]

            # Loss 1: Atom Classification
            logits_flat = logits_slice.reshape(-1, VOCAB_SIZE)
            targets_flat = p_ids_slice.reshape(-1)

            if self.class_weights is not None:
                # 确保权重在正确的设备上
                weights = self.class_weights.to(device)
                l_atom = F.cross_entropy(
                    logits_flat, 
                    targets_flat, 
                    weight=weights,
                    ignore_index=PAD_ID,
                    label_smoothing=0.1  # 添加标签平滑
                )
            else:
                l_atom = F.cross_entropy(
                    logits_flat, 
                    targets_flat, 
                    ignore_index=PAD_ID
                )

            # Loss 2: Coordinates
            valid_p = ~p_mask_slice
            
            # 先对齐
            coords_aligned = kabsch_rotate(coords_slice, p_coords_slice, p_mask_slice)
            
            # 使用 Smooth L1 Loss
            l_coord = F.smooth_l1_loss(
                coords_aligned[valid_p], 
                p_coords_slice[valid_p]
            )

            # Loss 3: Distance Matrix Loss
            pred_dist = torch.cdist(coords_slice, coords_slice)
            true_dist = torch.cdist(p_coords_slice, p_coords_slice)
            
            pair_mask = valid_p.unsqueeze(1) & valid_p.unsqueeze(2)
            l_dist = F.huber_loss(
                pred_dist[pair_mask], 
                true_dist[pair_mask],
                delta=1.0
            )

            # === 组合 Loss ===
            # 动态调整权重
            total_loss = (
                2.5 * l_atom +  # 原子分类损失权重最高
                coord_loss_weight * (1.0 * l_coord + 0.5 * l_dist)
            )

            # 计算准确率
            with torch.no_grad():
                pred_ids = logits_slice.argmax(dim=-1)
                valid_mask = ~p_mask_slice
                correct = (pred_ids == p_ids_slice) & valid_mask
                acc = correct.sum().float() / valid_mask.sum().float()

            return total_loss, {
                "loss": total_loss.item(),
                "l_atom": l_atom.item(),
                "l_coord": l_coord.item(),
                "l_dist": l_dist.item(),
                "atom_acc": acc.item()
            }, (logits_atom, coords_aligned)

        return logits_atom, pred_coords