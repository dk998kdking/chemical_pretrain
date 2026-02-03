# -*- coding: utf-8 -*-
# train_continue.py - 加载 best.pt 继续训练

import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from rdkit import Chem
from tqdm import tqdm
import logging

# === 项目内引用 ===
from chemical_pretrain.reaction_dataset import ReactionDataset, reaction_collate
from model_reaction_v2 import ChemicalReactionModelV2
from chemical_pretrain.utils import set_seed, ensure_dir
from chemical_pretrain.constants import VOCAB_SIZE, PAD_ID, MASK_ID

# === 配置 ===
CONFIG = {
    "train_txt": "data/train_10.txt",
    "cache_path": "data/cached_reaction.pt",
    "resume_checkpoint": "best.pt",          # <--- 这里指定要加载的权重文件
    "out_dir": "runs_stage2_continue",       # 新的输出目录
    "sample_dir": "runs_stage2_continue/samples",
    "epochs": 50,                            # 继续训练多少轮
    "batch_size": 16,
    "lr": 1e-4,                              # 续训学习率建议稍小
    "log_interval": 50,
    "eval_interval": 1,
}

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ========================================================================
# 1. 辅助函数
# ========================================================================

def compute_class_weights(dataset, device, vocab_size=VOCAB_SIZE):
    """计算类别权重（保持原样）"""
    print("正在统计数据集原子分布以计算类别权重...")
    counts = np.zeros(vocab_size, dtype=np.float32)
    sample_size = min(len(dataset), 20000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for i in tqdm(indices, desc="统计分布"):
        try:
            data = dataset[i]
            if 'p_atom_ids' in data: p_ids = data['p_atom_ids']
            elif 'product' in data: p_ids = data['product']['atom_ids']
            else: continue
            unique_ids, counts_per_id = np.unique(p_ids, return_counts=True)
            for uid, cnt in zip(unique_ids, counts_per_id):
                if 0 <= uid < vocab_size: counts[uid] += cnt
        except: continue
    
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / (counts / np.sum(counts))
    weights[PAD_ID] = 0.0; weights[MASK_ID] = 1.0
    weights = np.sqrt(weights)
    weights = weights / np.mean(weights[weights > 0])
    weights[weights > 0] = np.maximum(weights[weights > 0], 0.1)
    
    # 稀有原子加成
    for idx in [8, 9, 10, 16, 17, 18, 35, 53]:
        if idx < vocab_size: weights[idx] *= 1.5
            
    return torch.from_numpy(weights).float().to(device)

def build_id2atom():
    pt = Chem.GetPeriodicTable()
    id2atom = {PAD_ID: "PAD", MASK_ID: "MASK"}
    for idx in range(2, VOCAB_SIZE):
        try: id2atom[idx] = pt.GetElementSymbol(int(idx - 1))
        except: id2atom[idx] = f"X{idx}"
    return id2atom

def load_checkpoint_safe(model, path, device):
    """安全加载权重"""
    if not os.path.exists(path):
        logging.error(f"权重文件不存在: {path}")
        return False, float('inf')
    
    logging.info(f"正在加载旧权重: {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # 兼容直接保存 state_dict 或保存 dict 的情况
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        best_metric = checkpoint.get('loss', float('inf')) # 尝试获取上次的loss
    else:
        state_dict = checkpoint
        best_metric = float('inf')

    # 处理 key
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        if "class_weights" in name: continue # 不加载类别权重 buffer
        new_state_dict[name] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=False)
        logging.info("权重加载成功！")
        return True, best_metric
    except Exception as e:
        logging.error(f"加载失败: {e}")
        return False, float('inf')

def extract_outputs_robust(extras):
    """智能提取 (coords, logits)，防止顺序错误"""
    coord_pred = None; atom_logits = None
    candidates = list(extras.values()) if isinstance(extras, dict) else (extras if isinstance(extras, (list, tuple)) else [extras])

    for item in candidates:
        if isinstance(item, torch.Tensor):
            if item.dim() == 3:
                if item.shape[-1] == 3: coord_pred = item
                elif item.shape[-1] > 10: atom_logits = item
    return coord_pred, atom_logits

def save_xyz(path, atom_ids, coords, id2atom, comment="Generated"):
    """保存样本"""
    coords = coords.cpu().numpy()
    atom_ids = atom_ids.cpu().tolist()
    lines = []
    valid_count = 0
    for i, atom_id in enumerate(atom_ids):
        if atom_id in (PAD_ID, MASK_ID): continue
        x, y, z = coords[i]
        # 过滤原点附近的Padding噪声
        if i > 0 and abs(x)<1e-3 and abs(y)<1e-3 and abs(z)<1e-3: continue
        
        symbol = id2atom.get(atom_id, "X")
        lines.append(f"{symbol} {x:.4f} {y:.4f} {z:.4f}")
        valid_count += 1
    
    content = [f"{valid_count}", f"{comment}"] + [l + "" for l in lines]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(content)

def calculate_metrics(logits_atom, pred_coords, tgt_ids, tgt_coords, tgt_mask):
    """计算 RMSD 和 Atom Acc"""
    if logits_atom is None or pred_coords is None:
        return 0.0, 0.0

    # 对齐长度
    min_len = min(logits_atom.shape[1], tgt_ids.shape[1], pred_coords.shape[1])
    logits_atom = logits_atom[:, :min_len]
    pred_coords = pred_coords[:, :min_len]
    tgt_ids = tgt_ids[:, :min_len]
    tgt_coords = tgt_coords[:, :min_len]
    tgt_mask = tgt_mask[:, :min_len]

    with torch.no_grad():
        pred_ids = logits_atom.argmax(dim=-1)
        valid_mask = ~tgt_mask.bool() # mask通常是1为pad，取反
        
        # Acc
        correct = (pred_ids == tgt_ids) & valid_mask
        acc = (correct.sum().float() / (valid_mask.sum().float() + 1e-8)).item()
        
        # RMSD
        diff = (pred_coords - tgt_coords)
        # 只计算有效原子的距离
        diff = diff[valid_mask] 
        if diff.shape[0] == 0:
            rmsd = 0.0
        else:
            mse = diff.pow(2).sum() / (diff.shape[0] * 3) # 平均到每个坐标分量
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
    id2atom = build_id2atom()
    
    # 3. Model Init
    print("初始化模型结构...")
    model = ChemicalReactionModelV2(
        pretrain_weights_path=None, # 这里设为None，因为我们要手动加载best.pt
        class_weights=class_weights
    ).to(device)
    
    # 4. === 加载 best.pt ===
    loaded, prev_loss = load_checkpoint_safe(model, cfg['resume_checkpoint'], device)
    if not loaded:
        print("警告：未加载到权重，将从头开始！")
    
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=cfg['lr'], steps_per_epoch=len(train_loader), epochs=cfg['epochs'])
    
    best_rmsd = float("inf")
    
    print(f"开始继续训练... (Epochs: {cfg['epochs']})")

    # 5. Loop
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        steps = 0
        
        coord_w = 2.0 # 续训阶段通常比较关注结构，权重可以大一点
        
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
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
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
                    
                    # 前向传播
                    _, _, extras = model(r_ids, r_coords, r_mask, p_ids, p_coords, p_mask)
                    
                    # === 稳健提取 (修复崩坏点) ===
                    coords, logits = extract_outputs_robust(extras)
                    
                    # 计算指标
                    acc, rmsd = calculate_metrics(logits, coords, p_ids, p_coords, p_mask)
                    
                    val_rmsd_sum += rmsd
                    val_acc_sum += acc
                    val_steps += 1
                    
                    # Save sample (每个epoch只存第一个batch)
                    if val_steps == 1 and logits is not None and coords is not None:
                        idx = 0
                        T = min(128, p_ids.shape[1], logits.shape[1])
                        pred_ids = logits.argmax(dim=-1)
                        
                        save_xyz(os.path.join(cfg['sample_dir'], f"ep{epoch}_real.xyz"), 
                                 p_ids[idx,:T], p_coords[idx,:T], id2atom)
                        save_xyz(os.path.join(cfg['sample_dir'], f"ep{epoch}_pred.xyz"), 
                                 pred_ids[idx,:T], coords[idx,:T], id2atom, comment=f"RMSD={rmsd:.2f}")

            val_rmsd = val_rmsd_sum / max(1, val_steps)
            val_acc = val_acc_sum / max(1, val_steps)
            
            # 打印你需要的格式
            print(f"Validation: RMSD={val_rmsd:.4f}, Atom Acc={val_acc:.4f}")
            
            # 只要有进步就保存 (或者你可以设一个阈值)
            if val_rmsd < best_rmsd:
                best_rmsd = val_rmsd
                torch.save(model.state_dict(), os.path.join(cfg['out_dir'], "best_retrained.pt"))
                print("Saved Best Model (Retrained).")

if __name__ == "__main__":
    main()
