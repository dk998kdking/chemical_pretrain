# -*- coding: utf-8 -*-
# chemical_pretrain/constants.py
from pathlib import Path

# ===== 必须与 Stage 1 run_train.py 完全一致 =====
# 词表：0=PAD, 1=MASK, 2~119=原子
VOCAB_SIZE = 120
D_MODEL    = 256
N_HEADS    = 8         # 保持与 Stage 1 一致
D_FF       = 1024
LAYER_ENC  = 15        # 保持与 Stage 1 一致
LAYER_DEC  = 5         # Stage 2 解码器层数

RBF_K      = 64
RBF_MU_MAX = 10.0
MAX_LEN    = 512       # 最大原子数
DROPOUT    = 0.1

# 路径配置
PAD_ID = 0
MASK_ID = 1