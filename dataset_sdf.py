# -*- coding: utf-8 -*-
# dataset_sdf.py
"""
从 SDF 文件中读取 3D 分子并转换为用于预训练的张量数据。

本版本包含：
1. **缓存机制**：
   - 第一次从 SDF 解析分子时，会把处理好的样本列表保存到 `cache_path`（.pt 文件）；
   - 下次再创建 SDFMolDataset 时，如果缓存存在且不要求重建，就直接从缓存加载，
     不再调用 RDKit 重新解析 SDF（不会再看到各种 RDKit 的 warning）。

2. **解析进度条（tqdm）**：
   - 当从 SDF 解析时，会用 tqdm 显示“正在解析第几个分子”，
     进度条的 total 使用 **SDF 中真实的分子条目数**，
     再和 max_mols 取 min，代表“本次实际要遍历的分子数”。

缓存内容包括：
  - atom_ids:  (n,)  numpy.int64
  - coords:    (n,3) numpy.float32
与原始实现完全兼容。
"""
import os
from typing import List, Dict, Any, Optional

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import Dataset
import torch
from tqdm import tqdm  # 用于解析进度条

# 关闭 RDKit 日志，避免每次解析 SDF 打印大量 warning（如 atropisomer 提示）
RDLogger.DisableLog("rdApp.*")

# 定义词汇表中的特殊标记ID
PAD_ID = 0       # 填充标记ID，用于序列长度对齐
MASK_ID = 1      # 掩码标记ID，用于掩码语言模型训练


def z_to_id(z: int) -> int:
    """
    将原子序数映射到词汇表ID

    此函数将元素周期表中的原子序数转换为连续的词汇表ID，
    其中原子序数1-118对应词汇表ID 2-119，保留0和1给特殊标记。

    参数:
        z (int): 原子序数(1-118)

    返回:
        int: 词汇表ID，其中:
             - 0 = PAD标记
             - 1 = MASK标记
             - 2-119 = 原子序数1-118
    """
    # 处理无效的原子序数，使用MASK_ID作为默认值
    if z is None or z <= 0:
        return MASK_ID  # 对于极少数异常情况，返回掩码标记作为兜底
    # 将原子序数偏移1位，为特殊标记预留位置
    return int(z) + 1   # 原子序数+1得到词汇表ID


class SDFMolDataset(Dataset):
    """
    从 SDF 文件读取分子数据的 PyTorch 数据集类（带缓存 + tqdm 进度条）

    该数据集从 SDF 格式文件中读取 3D 分子结构，并提供原子类型(词汇表ID)和 3D 坐标信息。

    每个样本包含:
      - atom_ids: (n,)   基于原子序数的词汇表ID (numpy.int64)
      - coords:   (n,3) 三维坐标 (numpy.float32)

    会过滤掉：
      - 无构象的分子
      - 无原子的分子
      - 原子数超过 max_atoms 的分子

    支持将预处理结果缓存到 .pt 文件，下次直接加载，避免反复调用 RDKit。
    """

    def __init__(
        self,
        sdf_path: str,
        max_mols: int = 10000,
        max_atoms: int = 128,
        sanitize: bool = True,
        remove_hs: bool = False,
        cache_path: Optional[str] = None,
        rebuild_cache: bool = False,
        verbose: bool = True,
    ):
        """
        参数:
            sdf_path (str):
                原始 SDF 文件路径。

            max_mols (int):
                最多读取的分子数（解析时的上限，过滤后实际样本数可能更少）。

            max_atoms (int):
                单个分子允许的最大原子数，超过此值的分子会被跳过。

            sanitize (bool):
                RDKit 读取 SDF 时是否进行 sanitize。

            remove_hs (bool):
                RDKit 读取时是否去掉显式氢。

            cache_path (str or None):
                缓存文件路径；若为 None，则默认使用
                `sdf_path + f".maxatoms{max_atoms}.pt"`。

            rebuild_cache (bool):
                若为 True，则忽略已有缓存，强制重新解析 SDF 并覆盖缓存。

            verbose (bool):
                是否打印数据加载与缓存相关的信息。
        """
        assert os.path.exists(sdf_path), f"SDF 文件不存在: {sdf_path}"
        self.max_atoms = max_atoms
        self.sdf_path = sdf_path

        # 默认缓存路径：sdf_path 后缀加 .maxatoms{max_atoms}.pt
        if cache_path is None:
            cache_path = f"{sdf_path}.maxatoms{max_atoms}.pt"
        self.cache_path = cache_path

        self.samples: List[Dict[str, Any]] = []

        # ========= 优先尝试从缓存加载 =========
        if (not rebuild_cache) and os.path.exists(self.cache_path):
            try:
                # PyTorch 2.6 默认 weights_only=True，会禁止包含 numpy 对象的老缓存
                # 我们这里明确告诉它：这是自己生成的缓存，信得过，按旧方式加载完整对象即可。
                try:
                    data = torch.load(
                        self.cache_path,
                        map_location="cpu",
                        weights_only=False,  # 关键修改：关闭“只加载权重”安全模式
                    )
                except TypeError:
                    # 兼容旧版本 PyTorch（没有 weights_only 参数）
                    data = torch.load(self.cache_path, map_location="cpu")

                # 兼容两种简单格式：直接 list 或 dict 包裹
                if isinstance(data, dict) and "samples" in data:
                    self.samples = data["samples"]
                elif isinstance(data, list):
                    self.samples = data
                else:
                    raise ValueError("缓存文件格式不符合预期，请删除后重建。")

                if verbose:
                    print(
                        f"[SDFMolDataset] 从缓存加载 {len(self.samples)} 个分子: "
                        f"{self.cache_path}"
                    )
                return
            except Exception as e:
                if verbose:
                    print(f"[SDFMolDataset] 读取缓存失败，将从 SDF 重建缓存: {e}")

        # ========= 若缓存不可用，则从 SDF 解析并构建缓存 =========
        if verbose:
            print(
                f"[SDFMolDataset] 从 SDF 解析分子: {sdf_path} "
                f"(max_mols={max_mols}, max_atoms={max_atoms})"
            )

        suppl = Chem.SDMolSupplier(sdf_path, sanitize=sanitize, removeHs=remove_hs)

        # 估计 SDF 中的分子总条目数，用于进度条 total
        total_in_sdf: Optional[int] = None
        try:
            # RDKit 的 SDMolSupplier 支持 len()，返回文件中的记录数
            total_in_sdf = len(suppl)
        except Exception:
            total_in_sdf = None

        if verbose:
            if total_in_sdf is not None:
                # 本次实际要遍历的条目数 = min(文件总数, max_mols)
                if max_mols is not None:
                    total_for_tqdm = min(total_in_sdf, max_mols)
                else:
                    total_for_tqdm = total_in_sdf
            else:
                # 拿不到总数时，退而求其次：用 max_mols 或 None
                total_for_tqdm = max_mols if max_mols is not None else None

            iterator = tqdm(
                suppl,
                total=total_for_tqdm,
                desc="[SDFMolDataset] Parsing molecules",
                ncols=100,
            )
        else:
            iterator = suppl

        num_loaded = 0
        num_total = 0

        for mol in iterator:
            num_total += 1
            if mol is None:
                # RDKit 解析失败的分子（可能有格式问题）
                continue

            n_atoms = mol.GetNumAtoms()
            if n_atoms == 0:
                continue
            if n_atoms > self.max_atoms:
                # 跳过过大的分子
                continue
            if mol.GetNumConformers() == 0:
                # 没有 3D 构象
                continue

            conf = mol.GetConformer()

            # 提取原子类型 -> 词汇表 ID
            atom_ids = np.array(
                [z_to_id(atom.GetAtomicNum()) for atom in mol.GetAtoms()],
                dtype=np.int64,
            )

            # 提取 3D 坐标
            coords = np.zeros((n_atoms, 3), dtype=np.float32)
            for idx in range(n_atoms):
                pos = conf.GetAtomPosition(idx)
                coords[idx, 0] = float(pos.x)
                coords[idx, 1] = float(pos.y)
                coords[idx, 2] = float(pos.z)

            self.samples.append({"atom_ids": atom_ids, "coords": coords})
            num_loaded += 1

            # 如果设置了 max_mols，上限达到就停止
            if max_mols is not None and num_loaded >= max_mols:
                break

        if verbose:
            print(
                f"[SDFMolDataset] 解析完成: 遍历记录 {num_total} 条，"
                f"保留 {num_loaded} 个分子 (<= {max_atoms} 原子)。"
            )

        # 保存缓存
        try:
            torch.save(
                {
                    "samples": self.samples,
                    "max_atoms": self.max_atoms,
                    "sdf_path": self.sdf_path,
                },
                self.cache_path,
            )
            if verbose:
                print(f"[SDFMolDataset] 已保存缓存到: {self.cache_path}")
        except Exception as e:
            if verbose:
                print(f"[SDFMolDataset] 保存缓存失败（不影响训练）: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回单个样本:
          - atom_ids: (n,) numpy.int64
          - coords:   (n,3) numpy.float32
        """
        return self.samples[idx]


def batch_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    将一批样本打包成批次张量，用于 DataLoader 的 collate_fn。

    输入 batch: List[{"atom_ids": np.ndarray[n], "coords": np.ndarray[n,3]}]

    输出:
      - atom_ids: (B, Nmax) long，PAD 位置填 PAD_ID
      - coords:   (B, Nmax, 3) float32，PAD 位置填 0
      - pad_mask: (B, Nmax) bool，True 表示 PAD 位置
      - valid_n:  (B,) long，每个分子实际的原子数
    """
    B = len(batch)
    if B == 0:
        raise ValueError("batch_collate 收到空 batch，请检查 DataLoader。")

    Ns = [b["atom_ids"].shape[0] for b in batch]
    Nmax = max(Ns)

    atom_ids = torch.full((B, Nmax), PAD_ID, dtype=torch.long)
    coords = torch.zeros(B, Nmax, 3, dtype=torch.float32)
    pad_mask = torch.ones(B, Nmax, dtype=torch.bool)  # 先默认全是 PAD=True

    for i, ex in enumerate(batch):
        n = ex["atom_ids"].shape[0]
        atom_ids[i, :n] = torch.from_numpy(ex["atom_ids"])
        coords[i, :n, :] = torch.from_numpy(ex["coords"])
        pad_mask[i, :n] = False  # 前 n 个为真实原子

    valid_n = torch.tensor(Ns, dtype=torch.long)
    return {"atom_ids": atom_ids, "coords": coords, "pad_mask": pad_mask, "valid_n": valid_n}
