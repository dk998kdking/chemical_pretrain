# -*- coding: utf-8 -*-
# dataset_reaction.py
import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from tqdm import tqdm
import multiprocessing
from chemical_pretrain.constants import PAD_ID, MASK_ID


# ===== 必须与 dataset_sdf.py 逻辑完全一致 =====
def z_to_id(z: int) -> int:
    """原子序数 1-118 -> ID 2-119, 0=PAD, 1=MASK"""
    if z is None or z <= 0: return MASK_ID
    return int(z) + 1


def mol_to_feat(smi, max_atoms=512):
    if not smi: return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol: return None
        mol = Chem.AddHs(mol)

        # 3D 坐标生成
        res = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=3)
        if res != 0:
            res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if res != 0: return None  # 无法生成构象则跳过

        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=10)
        except:
            pass

        n_atoms = mol.GetNumAtoms()
        if n_atoms > max_atoms or n_atoms == 0: return None

        conf = mol.GetConformer()

        # 使用与 Stage 1 相同的 z_to_id
        ids = np.array([z_to_id(a.GetAtomicNum()) for a in mol.GetAtoms()], dtype=np.int64)
        xyz = conf.GetPositions().astype(np.float32)

        return {'atom_ids': ids, 'coords': xyz}
    except Exception:
        return None


def process_line(line):
    line = line.strip()
    if not line: return None
    try:
        # 格式: "Reactant>Agent>Product" 或 "Reactant>>Product"
        # 假设第一列是 Reaction SMILES
        parts = line.split()
        smiles_raw = parts[0]
        if '>' not in smiles_raw: return None

        # 分割
        r_smi, _, p_smi = smiles_raw.partition('>')
        if '>' in p_smi:  # 处理 Agent 部分
            _, _, p_smi = p_smi.partition('>')

        r_feat = mol_to_feat(r_smi)
        p_feat = mol_to_feat(p_smi)

        if r_feat is not None and p_feat is not None:
            return {'reactant': r_feat, 'product': p_feat}
        return None
    except Exception:
        return None


class ReactionDataset(Dataset):
    def __init__(self, txt_path, cache_path=None, max_workers=8):
        self.data = []

        # 1. 尝试加载缓存
        if cache_path and os.path.exists(cache_path):
            print(f"[Dataset] 加载缓存: {cache_path}")
            try:
                # 修复: weights_only=False 允许加载 numpy 数组
                self.data = torch.load(cache_path, weights_only=False)
                print(f"[Dataset] 已加载 {len(self.data)} 条数据。")
                return
            except Exception as e:
                print(f"[Dataset] 缓存加载失败: {e}，重新处理...")

        # 2. 处理文本
        print(f"[Dataset] 处理文件: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [l for l in f.readlines() if l.strip()]

        print(f"[Dataset] 启动 {max_workers} 进程生成 3D 构象...")
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = list(tqdm(pool.imap(process_line, lines, chunksize=10), total=len(lines)))

        self.data = [r for r in results if r is not None]
        print(f"[Dataset] 有效数据: {len(self.data)} / {len(lines)}")

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def reaction_collate(batch):
    if not batch: return None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None

    r_list = [b['reactant'] for b in batch]
    p_list = [b['product'] for b in batch]

    def collate_mols(mol_list):
        B = len(mol_list)
        Ns = [len(m['atom_ids']) for m in mol_list]
        max_n = max(Ns)

        atom_ids = torch.full((B, max_n), PAD_ID, dtype=torch.long)
        coords = torch.zeros((B, max_n, 3), dtype=torch.float32)
        pad_mask = torch.ones((B, max_n), dtype=torch.bool)

        for i, m in enumerate(mol_list):
            n = Ns[i]
            atom_ids[i, :n] = torch.from_numpy(m['atom_ids'])
            coords[i, :n, :] = torch.from_numpy(m['coords'])
            pad_mask[i, :n] = False

        return atom_ids, coords, pad_mask

    r_ids, r_coords, r_mask = collate_mols(r_list)
    p_ids, p_coords, p_mask = collate_mols(p_list)

    return {
        "r_atom_ids": r_ids, "r_coords": r_coords, "r_mask": r_mask,
        "p_atom_ids": p_ids, "p_coords": p_coords, "p_mask": p_mask
    }