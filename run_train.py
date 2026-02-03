# -*- coding: utf-8 -*-
# chemical_pretrain/run_train.py
from pathlib import Path
import sys

# 自动把“项目根目录”加入 sys.path（根=包的父目录）
PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 静默 RDKit 的 warning/error（可选）
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')

from chemical_pretrain.train_chem_pretrain import main

if __name__ == "__main__":
    main()
