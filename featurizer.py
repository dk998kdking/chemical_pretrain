# -*- coding: utf-8 -*-
# chem_reaction_pretrain/featurizer.py
from typing import Tuple, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from .constants import PAD_ID, MAX_ATOMS

# 直接用“原子序号Z”作为类别（0=PAD）
def smiles_to_ids_coords(smiles: str,
                         max_atoms: int = MAX_ATOMS,
                         seed: int = 17) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    返回:
      atom_ids: (N,) long  —— 原子序号Z，范围 [1..127]，0=PAD
      coords:   (N,3) float —— 3D坐标（Å）
    出错返回 None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        ok = AllChem.EmbedMolecule(mol, params)
        if ok != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

        conf = mol.GetConformer()
        N = mol.GetNumAtoms()
        Z = []
        xyz = np.zeros((N, 3), dtype=np.float32)
        for i, a in enumerate(mol.GetAtoms()):
            z = a.GetAtomicNum()
            z = 127 if z >= 127 else z  # VOCAB_SIZE=128，0=PAD
            Z.append(z)
            p = conf.GetAtomPosition(i)
            xyz[i, 0], xyz[i, 1], xyz[i, 2] = p.x, p.y, p.z

        # 截断/保留前 max_atoms 个
        if N > max_atoms:
            Z = Z[:max_atoms]
            xyz = xyz[:max_atoms]

        atom_ids = torch.tensor(Z, dtype=torch.long)
        coords = torch.tensor(xyz, dtype=torch.float32)
        return atom_ids, coords
    except Exception:
        return None
