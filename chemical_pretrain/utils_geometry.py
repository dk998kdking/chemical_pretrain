# utils_geometry.py
import torch


def _to_atom_pad_mask(P: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    统一把各种形状的 mask 转成原子级 PAD mask：(B, N)，True=PAD
    P: (B, N, 3)
    mask:
      - None
      - (B, N)           原子级 PAD / valid 掩码
      - (B, N, N)        pair 掩码
      - (B, N, N, H)     多头 pair 掩码
    """
    B, N, _ = P.shape
    device = P.device

    if mask is None:
        return torch.zeros(B, N, dtype=torch.bool, device=device)

    m = mask.to(torch.bool)

    # ----- 情况 1：已经是 (B, N) -----
    if m.dim() == 2:
        if m.shape == (B, N):
            # 假设 True=PAD（和你 dataset 里保持一致）
            return m
        if m.shape == (N, B):
            # 如果不小心转置了，这里帮你转回来
            return m.t().contiguous()
        # 异常形状，直接当作“全是非 PAD”
        return torch.zeros(B, N, dtype=torch.bool, device=device)

    # ----- 情况 2：pair 掩码 (B, N, N) 或 (B, N, N, H) -----
    if m.dim() == 3:
        # (B, N, N)
        # 如果 True 的比例很高，一般是“有效 pair”，反之则可能是“PAD pair”
        true_ratio = m.float().mean()
        if true_ratio > 0.5:
            # True ≈ valid pair
            atom_valid = m.any(dim=-1)  # (B, N)
        else:
            # True ≈ pad/invalid pair
            atom_valid = (~m).any(dim=-1)
        return ~atom_valid  # True=PAD

    if m.dim() == 4:
        # (B, N, N, H)
        true_ratio = m.float().mean()
        if true_ratio > 0.5:
            atom_valid = m.any(dim=-1).any(dim=-1)  # (B, N)
        else:
            atom_valid = (~m).any(dim=-1).any(dim=-1)
        return ~atom_valid

    # 其它奇形怪状：保守地认为“全是非 PAD”
    return torch.zeros(B, N, dtype=torch.bool, device=device)


def kabsch_rotate(P, Q, mask=None):
    """
    Args:
        P: (B, N, 3)  预测坐标 (Source)
        Q: (B, N, 3)  真实坐标 (Target)
        mask:
            - None：全原子有效
            - (B, N)：True=PAD 的 PAD 掩码
            - (B, N, N)/(B, N, N, H)：pair 掩码（会自动化简为原子级 PAD 掩码）
    Returns:
        P_aligned: (B, N, 3)  旋转平移后的 P
    """
    # 1) 把各种 mask 统一成 (B, N) 的 PAD 掩码
    B, N, _ = P.shape
    device = P.device

    atom_pad_mask = _to_atom_pad_mask(P, mask)      # (B, N) True=PAD
    valid_mask = (~atom_pad_mask).float().unsqueeze(-1)  # (B, N, 1)

    # 2) 计算重心（对有效原子加权平均）
    num_valid = valid_mask.sum(dim=1, keepdim=True)           # (B, 1, 1)
    weight_sum = torch.clamp(num_valid, min=1e-6)

    p_mean = (P * valid_mask).sum(dim=1, keepdim=True) / weight_sum
    q_mean = (Q * valid_mask).sum(dim=1, keepdim=True) / weight_sum

    # 3) 去中心化
    P_c = (P - p_mean) * valid_mask
    Q_c = (Q - q_mean) * valid_mask

    # 4) 协方差矩阵 H (B, 3, 3)
    H = torch.matmul(P_c.transpose(1, 2), Q_c)

    # 5) SVD 分解（带 try 保护）
    try:
        U, S, V = torch.svd(H)
    except RuntimeError:
        # SVD 偶尔会炸（数值问题），这种情况下直接返回原坐标，loss 会很大，但训练还能继续
        return P

    # 6) 修正旋转矩阵，避免反射
    d = torch.det(torch.matmul(V, U.transpose(1, 2)))  # (B,)
    correction = torch.ones((B, 3), device=device)
    correction[:, 2] = torch.sign(d)
    S_corr = torch.diag_embed(correction)

    R = torch.matmul(torch.matmul(V, S_corr), U.transpose(1, 2))  # (B, 3, 3)

    # 7) 应用旋转 + 平移
    P_rot = torch.matmul(P_c, R.transpose(1, 2))
    P_aligned = P_rot + q_mean

    return P_aligned
