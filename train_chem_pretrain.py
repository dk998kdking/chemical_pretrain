# -*- coding: utf-8 -*-
# train_chem_pretrain.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from rdkit import Chem
from tqdm import tqdm

from chemical_pretrain.reaction_dataset import ReactionDataset, reaction_collate
from model_reaction_v2 import ChemicalReactionModelV2  # 引用上面那个修复后的文件
from chemical_pretrain.utils import set_seed, ensure_dir
from chemical_pretrain.constants import VOCAB_SIZE, PAD_ID, MASK_ID

# === 配置 ===
CONFIG = {
    "train_txt": "data/train_10.txt",
    "cache_path": "data/cached_reaction.pt",
    "pretrain_weights": "shared_encoder.pt",
    "out_dir": "runs_stage2_v2",
    "sample_dir": "runs_stage2_v2/samples",
    "epochs": 50,
    "batch_size": 16,
    "lr": 2e-4,
    "log_interval": 50,
    "eval_interval": 1,
}

# ========================================================================
# 1. 强化的类别权重计算 (带截断保护)
# ========================================================================
def compute_class_weights(dataset, device, vocab_size=VOCAB_SIZE):
    print("正在统计数据集原子分布以计算类别权重...")
    counts = np.zeros(vocab_size, dtype=np.float32)
    
    # 为了速度，随机采样 20% 的数据进行统计，或者统计前 20000 条
    sample_size = min(len(dataset), 20000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for i in tqdm(indices, desc="统计分布"):
        try:
            data = dataset[i]
            if 'p_atom_ids' in data: # 根据 Dataset 返回格式调整 key
                 p_ids = data['p_atom_ids']
            elif 'product' in data and 'atom_ids' in data['product']:
                p_ids = data['product']['atom_ids']
            else:
                continue

            unique_ids, counts_per_id = np.unique(p_ids, return_counts=True)
            for uid, cnt in zip(unique_ids, counts_per_id):
                if 0 <= uid < vocab_size:
                    counts[uid] += cnt
        except:
            continue
    
    # --- 权重计算逻辑 ---
    counts = np.maximum(counts, 1.0)
    total = np.sum(counts)
    freq = counts / total
    
    # 1. 基础反向频率
    weights = 1.0 / freq
    
    # 2. Mask/Pad 处理
    weights[PAD_ID] = 0.0
    weights[MASK_ID] = 1.0 
    
    # 3. 平滑 (Sqrt)
    weights = np.sqrt(weights)
    
    # 4. 归一化
    valid_mask = (weights > 0)
    if valid_mask.any():
        mean_w = np.mean(weights[valid_mask])
        weights = weights / mean_w
    
    # 5. === 关键修改: 权重截断 (Clamp) ===
    # 强制所有有效原子的权重至少为 0.1。
    # 这样 C/H 不会因为出现太多而被惩罚到 0.000x，导致模型学不会骨架。
    weights[valid_mask] = np.maximum(weights[valid_mask], 0.1)
    
    # 6. 给稀有原子额外加成 (N, O, F, P, S, Cl)
    boost_ids = [8, 9, 10, 16, 17, 18, 35, 53] 
    for idx in boost_ids:
        if idx < vocab_size:
            weights[idx] *= 1.5
            
    print("修正后的权重预览 (Top 5):")
    top_indices = np.argsort(weights)[-5:]
    pt = Chem.GetPeriodicTable()
    for idx in top_indices:
        sym = pt.GetElementSymbol(int(idx-1)) if idx > 1 else str(idx)
        print(f"  {sym} (ID {idx}): {weights[idx]:.4f}")

    return torch.from_numpy(weights).float().to(device)


def build_id2atom():
    pt = Chem.GetPeriodicTable()
    id2atom = {PAD_ID: "PAD", MASK_ID: "MASK"}
    for idx in range(2, VOCAB_SIZE):
        try:
            symbol = pt.GetElementSymbol(int(idx - 1))
        except:
            symbol = f"X{idx}"
        id2atom[idx] = symbol
    return id2atom

# ========================================================================
# 2. 辅助函数 (保存 XYZ)
# ========================================================================
def save_xyz(path, atom_ids, coords, id2atom, comment="Generated"):
    """保存为.xyz格式，自动过滤零坐标填充"""
    coords = coords.cpu().numpy()
    atom_ids = atom_ids.cpu().tolist()
    
    lines = []
    valid_count = 0
    
    for i, atom_id in enumerate(atom_ids):
        if atom_id in (PAD_ID, MASK_ID):
            continue
            
        x, y, z = coords[i]
        
        # 过滤掉坐标极小且不是第一个原子的点 (通常是 padding 预测错误)
        if i > 0 and abs(x) < 1e-3 and abs(y) < 1e-3 and abs(z) < 1e-3:
            continue
            
        symbol = id2atom.get(atom_id, "X")
        lines.append(f"{symbol} {x:.4f} {y:.4f} {z:.4f}")
        valid_count += 1
    
    content = [f"{valid_count}", comment] + lines
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(content))

def calculate_metrics(logits_atom, pred_coords, tgt_ids, tgt_coords, tgt_mask):
    with torch.no_grad():
        pred_ids = logits_atom.argmax(dim=-1)
        valid_mask = ~tgt_mask
        correct = (pred_ids == tgt_ids) & valid_mask
        acc = (correct.sum().float() / (valid_mask.sum().float() + 1e-8)).item()
        
        diff = (pred_coords - tgt_coords) * valid_mask.unsqueeze(-1).float()
        mse = diff.pow(2).sum() / (valid_mask.sum().float() * 3 + 1e-8)
        rmsd = torch.sqrt(mse).item()
    return acc, rmsd

# ========================================================================
# 主训练循环
# ========================================================================
def main():
    cfg = CONFIG
    set_seed(2024)
    ensure_dir(cfg['out_dir'])
    ensure_dir(cfg['sample_dir'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. Dataset
    full_ds = ReactionDataset(cfg['train_txt'], cfg['cache_path'])
    n_train = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, 
                              collate_fn=reaction_collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, 
                            collate_fn=reaction_collate, num_workers=2)
    
    # 2. Weights
    class_weights = compute_class_weights(full_ds, device)
    
    # 3. Model
    model = ChemicalReactionModelV2(
        pretrain_weights_path=cfg['pretrain_weights'],
        class_weights=class_weights
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=cfg['lr'], steps_per_epoch=len(train_loader), epochs=cfg['epochs'])
    id2atom = build_id2atom()
    
    best_rmsd = float("inf")
    
    # 4. Loop
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        steps = 0
        
        # 动态坐标权重: 随着训练进行，逐渐增加对坐标的关注
        coord_w = 0.5 if epoch < 5 else 2.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if batch is None: continue
            
            # Move to device
            r_ids = batch['r_atom_ids'].to(device)
            r_coords = batch['r_coords'].to(device)
            r_mask = batch['r_mask'].to(device)
            p_ids = batch['p_atom_ids'].to(device)
            p_coords = batch['p_coords'].to(device)
            p_mask = batch['p_mask'].to(device)
            
            optimizer.zero_grad()
            
            loss, logs, _ = model(r_ids, r_coords, r_mask, p_ids, p_coords, p_mask, coord_loss_weight=coord_w)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            steps += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'atom': f"{logs['l_atom']:.3f}"})
            
        # Validation
        if epoch % cfg['eval_interval'] == 0:
            model.eval()
            val_rmsd_sum = 0
            val_acc_sum = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None: continue
                    r_ids = batch['r_atom_ids'].to(device)
                    r_coords = batch['r_coords'].to(device)
                    r_mask = batch['r_mask'].to(device)
                    p_ids = batch['p_atom_ids'].to(device)
                    p_coords = batch['p_coords'].to(device)
                    p_mask = batch['p_mask'].to(device)
                    
                    _, _, (logits, coords) = model(r_ids, r_coords, r_mask, p_ids, p_coords, p_mask)
                    
                    # Truncate for metric calc
                    T = min(128, p_ids.shape[1])
                    acc, rmsd = calculate_metrics(logits[:,:T], coords[:,:T], p_ids[:,:T], p_coords[:,:T], p_mask[:,:T])
                    
                    val_rmsd_sum += rmsd
                    val_acc_sum += acc
                    val_steps += 1
                    
                    # Save sample (first batch only)
                    if val_steps == 1:
                        idx = 0
                        pred_ids = logits.argmax(dim=-1)
                        save_xyz(os.path.join(cfg['sample_dir'], f"ep{epoch}_real.xyz"), 
                                 p_ids[idx,:T], p_coords[idx,:T], id2atom)
                        save_xyz(os.path.join(cfg['sample_dir'], f"ep{epoch}_pred.xyz"), 
                                 pred_ids[idx,:T], coords[idx,:T], id2atom, comment=f"RMSD={rmsd:.2f}")

            val_rmsd = val_rmsd_sum / max(1, val_steps)
            val_acc = val_acc_sum / max(1, val_steps)
            print(f"Validation: RMSD={val_rmsd:.4f}, Atom Acc={val_acc:.4f}")
            
            if val_rmsd < best_rmsd:
                best_rmsd = val_rmsd
                torch.save(model.state_dict(), os.path.join(cfg['out_dir'], "best.pt"))
                print("Saved Best Model.")

if __name__ == "__main__":
    main()
