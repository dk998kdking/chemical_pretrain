# -*- coding: utf-8 -*-
# pretrain_model.py
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from molecular_pretrain.model_components import RBFEncoder, PositionalEncoding, TransformerBlockWithPair
from molecular_pretrain.dataset_sdf import PAD_ID, MASK_ID


# ========== 工具函数 ==========
def valid_pair_from_pad_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """
    pad_mask: (B,N) True = PAD
    返回: (B,N,N) True = 两端都非 PAD
    """
    valid = ~pad_mask
    return valid.unsqueeze(1) & valid.unsqueeze(2)


# ========== 基础模块 ==========
class AtomEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.emb.weight[PAD_ID].zero_()

    def forward(self, x):
        return self.emb(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MaskLMHead(nn.Module):
    """
    只对 masked token 做原子类型的预测。
    """

    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, features: torch.Tensor, masked_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        features: (B,N,D)
        masked_tokens: (B,N) bool，True 表示该位置被 mask
        返回：如果 masked_tokens 不为 None，则形状为 (n_mask, vocab_size)
        """
        if masked_tokens is not None:
            features = features[masked_tokens]  # (n_mask, D)

        x = self.dense(features)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return x


class DistanceHead(nn.Module):
    """
    对 pair representation (B,N,N,H) 回归 pairwise distance (B,N,N)
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.dense = nn.Linear(num_heads, num_heads)
        self.layer_norm = nn.LayerNorm(num_heads)
        self.out_proj = nn.Linear(num_heads, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _, H = x.shape
        x = self.dense(x)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(B, N, N)  # (B,N,N)
        # 对称化
        x = 0.5 * (x + x.transpose(-1, -2))
        return x


class SharedEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlockWithPair(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, pair_bias: torch.Tensor, pad_mask: torch.Tensor):
        for blk in self.layers:
            x, pair_bias = blk(x, pair_bias, pad_mask)
        return x, pair_bias


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlockWithPair(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, pair_bias: torch.Tensor, pad_mask: torch.Tensor):
        for blk in self.layers:
            x, pair_bias = blk(x, pair_bias, pad_mask)
        return x, pair_bias


# ========== 主模型：只保留 3 个损失 + 3 个直观指标 ==========
class MoleculePretrainModel(nn.Module):
    """
    基于自动编码器的分子图表示学习：
      - Encoder：对部分原子做 drop，形成子结构（3D Cloze Test），不加 PE；
      - Decoder：在 latent 上插回 [MASK] 位置，只在 decoder 端加 PE，pair 表示从 0 开始；
      - 预训练目标只有 3 个：
          1) 原子类型重建（MLM）
          2) 3D 坐标重建
          3) pairwise 距离重建
      - 额外返回 3 个直观指标：
          * atom_acc: 被 mask 原子的类型预测准确率
          * coord_rmsd: 坐标 RMSD（Å）
          * dist_rmse: 距离 RMSE（Å）
    """

    def __init__(
        self,
        vocab_size: int = 120,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers_enc: int = 6,
        n_layers_dec: int = 4,
        rbf_k: int = 64,
        rbf_mu_max: float = 10.0,
        p_mask: float = 0.15,          # 原子类型 MLM 的 mask 比例
        max_len: int = 512,
        dropout: float = 0.1,
        drop_ratio: float = 0.15,      # 3D Cloze 的 drop 比例
        coord_loss_weight: float = 1.0,
        dist_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.rbf_k = rbf_k
        self.p_mask = p_mask
        self.drop_ratio = drop_ratio
        self.coord_loss_weight = coord_loss_weight
        self.dist_loss_weight = dist_loss_weight

        # 嵌入
        self.atom_emb = AtomEmbedding(vocab_size, d_model)

        # RBF & MLP(e_ij -> pair bias per head)
        self.rbf = RBFEncoder(num_k=rbf_k, mu_max=rbf_mu_max)
        self.rbf_mlp = MLP(rbf_k, n_heads, hidden=rbf_k, dropout=dropout)

        # 编码器 / 解码器
        self.encoder = SharedEncoder(d_model, n_heads, d_ff, n_layers_enc, dropout)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers_dec, dropout)

        # 预测头
        self.head_atom = MaskLMHead(d_model, vocab_size)
        self.pair2coord_proj = MLP(n_heads, 1, hidden=n_heads, dropout=dropout)
        self.dist_head = DistanceHead(n_heads)

        # 损失函数：SmoothL1 更稳一点
        self.coord_loss_fn = nn.SmoothL1Loss()
        self.dist_loss_fn = nn.SmoothL1Loss()

    # ---------- 掩码/采样相关 ----------
    def sample_atom_mask(self, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        原子类型 MLM 的 mask 位置。
        pad_mask: (B,N) True=PAD
        返回 masked_tokens: (B,N) True=该原子类型被 mask
        """
        B, N = pad_mask.shape
        keep = ~pad_mask  # 非 PAD 才能 mask

        probs = torch.full((B, N), self.p_mask, device=pad_mask.device)
        probs = probs * keep.float()

        sampled = torch.bernoulli(probs).bool()
        sampled = sampled & keep

        # 每个样本至少 mask 一个原子，避免 batch 里有样本 loss_atom 为 0
        any_mask = sampled.any(dim=1)  # (B,)
        if (~any_mask).any():
            for b in range(B):
                if not any_mask[b]:
                    nonpad = torch.nonzero(keep[b], as_tuple=False).view(-1)
                    if nonpad.numel() > 0:
                        sampled[b, nonpad[0]] = True

        return sampled

    def sample_drop_mask(self, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        3D Cloze Test：在 encoder 中被“丢弃”的原子。
        pad_mask: (B,N) True=PAD
        返回 drop_mask: (B,N) True=在 encoder 端 drop
        """
        B, N = pad_mask.shape
        keep = ~pad_mask  # 非 PAD 才有机会被 drop

        probs = torch.full((B, N), self.drop_ratio, device=pad_mask.device)
        probs = probs * keep.float()

        sampled = torch.bernoulli(probs).bool()
        sampled = sampled & keep

        # 确保不会把一个分子的所有非 PAD 都 drop 掉
        for b in range(B):
            if sampled[b].sum() >= keep[b].sum() and keep[b].any():
                # 保留一个非 PAD 不被 drop
                idxs = torch.nonzero(keep[b], as_tuple=False).view(-1)
                sampled[b, idxs[0]] = False

        return sampled

    @staticmethod
    def pairwise_distance(coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        coords: (B,N,3)
        pad_mask: (B,N) True=PAD
        返回 dist: (B,N,N)，PAD 相关的对置 0
        """
        diff = coords.unsqueeze(1) - coords.unsqueeze(2)  # (B,N,N,3)
        dist = diff.pow(2).sum(-1).sqrt()                 # (B,N,N)

        valid_atom = ~pad_mask
        pair_valid = valid_atom.unsqueeze(1) & valid_atom.unsqueeze(2)  # (B,N,N)
        dist = dist * pair_valid.float()
        return dist

    # ---------- 前向传播 ----------
    def forward(
        self,
        atom_ids: torch.Tensor,   # (B,N) 原子类型 id
        coords: torch.Tensor,     # (B,N,3) 原子坐标
        pad_mask: torch.Tensor,   # (B,N) True=PAD
    ):
        B, N = atom_ids.shape
        device = atom_ids.device

        # ===== 1) 原子类型 MLM 掩码 =====
        masked_tokens = self.sample_atom_mask(pad_mask)  # (B,N) bool
        atom_ids_input = atom_ids.clone()
        atom_ids_input[masked_tokens] = MASK_ID

        # Encoder 输入嵌入（不加 PE）
        x_full = self.atom_emb(atom_ids_input)  # (B,N,D)

        # ===== 2) 3D Cloze Test：对 encoder 做 drop =====
        drop_mask = self.sample_drop_mask(pad_mask)      # (B,N) bool
        pad_mask_enc = pad_mask | drop_mask              # encoder 中视为 PAD 的位置

        # 完整分子的 pairwise distance（用原始 pad_mask）
        dist_full = self.pairwise_distance(coords, pad_mask)  # (B,N,N)

        # 用 RBF + MLP 生成 encoder 初始 pair 表示
        e_ij = self.rbf(dist_full)      # (B,N,N,K)
        b_ij_full = self.rbf_mlp(e_ij)  # (B,N,N,H)

        valid_enc = ~pad_mask_enc
        pair_valid_enc = valid_enc.unsqueeze(1) & valid_enc.unsqueeze(2)  # (B,N,N)
        b0 = b_ij_full * pair_valid_enc.unsqueeze(-1).float()             # (B,N,N,H)

        x0 = x_full  # 不额外置零，pad_mask_enc 会在注意力里起作用

        # ===== 3) Shared Encoder =====
        x_enc, b_enc = self.encoder(x0, b0, pad_mask_enc)  # (B,N,D), (B,N,N,H)

        # ===== 4) Decoder 输入：插回 [MASK]，加 PE，pair 从 0 开始 =====
        x_latent = x_enc.clone()
        mask_vec = self.atom_emb.emb.weight[MASK_ID]  # (D,)
        x_latent[drop_mask] = mask_vec.to(x_latent.dtype)

        x_dec0 = self.pe(x_latent)                   # 只在 decoder 侧加位置编码
        b_dec0 = torch.zeros_like(b_enc)             # decoder pair 初始为 0

        x_dec, b_dec = self.decoder(x_dec0, b_dec0, pad_mask)

        # ===== 5) 三个预训练损失 + 三个直观指标 =====

        # ---- 5.1 原子类型重建（MLM）+ 准确率 atom_acc ----
        if masked_tokens.any():
            logits_atom = self.head_atom(x_dec, masked_tokens=masked_tokens)  # (n_mask, V)
            atom_tgt = atom_ids[masked_tokens]                                # (n_mask,)
            loss_atom = F.cross_entropy(logits_atom, atom_tgt)

            with torch.no_grad():
                pred_atom = logits_atom.argmax(dim=-1)                        # (n_mask,)
                atom_acc = (pred_atom == atom_tgt).float().mean()
        else:
            loss_atom = torch.tensor(0.0, device=device)
            atom_acc = torch.tensor(0.0, device=device)

        # ---- 5.2 坐标重建：对所有非 PAD 原子坐标重建 + coord_rmsd ----
        valid_atom = ~pad_mask  # (B,N)
        if valid_atom.any():
            coords_emb = coords  # (B,N,3)

            # delta_pos: (B,N,N,3)
            delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)

            pair_valid = valid_pair_from_pad_mask(pad_mask)        # (B,N,N)
            pair_valid_f = pair_valid.unsqueeze(-1).float()        # (B,N,N,1)

            # 每个分子中有效原子数（减 1），避免除以 0
            atom_counts = valid_atom.sum(dim=1).clamp(min=2).float()  # 至少 2
            atom_num = (atom_counts - 1.0).view(B, 1, 1, 1)           # (B,1,1,1)

            # pair 表示 -> 标量权重
            attn_scalar = self.pair2coord_proj(b_dec) * pair_valid_f  # (B,N,N,1)

            # 聚合更新
            coord_update = (delta_pos / atom_num) * attn_scalar   # (B,N,N,3)
            coord_update = coord_update.sum(dim=2)                # (B,N,3)

            coord_pred = coords_emb + coord_update                # (B,N,3)

            coord_pred_valid = coord_pred[valid_atom]             # (n_valid,3)
            coord_tgt_valid = coords_emb[valid_atom]              # (n_valid,3)

            loss_coord = (
                self.coord_loss_fn(coord_pred_valid, coord_tgt_valid)
                * self.coord_loss_weight
            )

            # 坐标 RMSD（不加权，方便理解）
            with torch.no_grad():
                diff = coord_pred_valid - coord_tgt_valid         # (n_valid,3)
                coord_rmsd = torch.sqrt((diff.pow(2).sum(-1).mean()).clamp(min=1e-12))
        else:
            loss_coord = torch.tensor(0.0, device=device)
            coord_rmsd = torch.tensor(0.0, device=device)
            coord_pred = coords  # 占位，extras 里用不到也没关系

        # ---- 5.3 距离重建：对所有非 PAD 原子对 + dist_rmse ----
        pair_valid = valid_pair_from_pad_mask(pad_mask)  # (B,N,N)
        if pair_valid.any():
            dist_pred = self.dist_head(b_dec)               # (B,N,N)
            dist_pred_valid = dist_pred[pair_valid]         # (n_pair,)
            dist_tgt_valid = dist_full[pair_valid]          # (n_pair,)

            loss_dist = (
                self.dist_loss_fn(dist_pred_valid, dist_tgt_valid)
                * self.dist_loss_weight
            )

            with torch.no_grad():
                err = dist_pred_valid - dist_tgt_valid
                dist_rmse = torch.sqrt((err.pow(2).mean()).clamp(min=1e-12))
        else:
            loss_dist = torch.tensor(0.0, device=device)
            dist_rmse = torch.tensor(0.0, device=device)
            dist_pred = dist_full  # 占位

        # ===== 6) 汇总（三个损失 + 三个指标） =====
        loss = loss_atom + loss_coord + loss_dist

        loss_dict: Dict[str, torch.Tensor] = {
            "loss_atom": loss_atom,
            "loss_coord": loss_coord,
            "loss_dist": loss_dist,
            "loss": loss,
            "atom_acc": atom_acc,
            "coord_rmsd": coord_rmsd,
            "dist_rmse": dist_rmse,
        }

        # 一些中间量，如果你想在训练日志里自己打印可以用到
        extras: Dict[str, torch.Tensor] = {
            "x_enc": x_enc,
            "b_enc": b_enc,
            "drop_mask": drop_mask,
            "masked_tokens": masked_tokens,
            "coord_pred": coord_pred,   # 完整预测坐标 (B,N,3)
            "dist_pred": dist_pred,     # 完整预测距离 (B,N,N)
        }

        return loss, loss_dict, extras

    # 供下游微调时保存/加载 encoder
    def shared_encoder_state(self):
        return self.encoder.state_dict()
