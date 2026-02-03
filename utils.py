# -*- coding: utf-8 -*-
# utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

class Meter:
    def __init__(self):
        self.sum = {}
        self.n = 0
    def update(self, d: dict):
        for k, v in d.items():
            v = float(v.detach().cpu().item())
            self.sum[k] = self.sum.get(k, 0.0) + v
        self.n += 1
    def mean(self):
        return {k: v / max(1, self.n) for k, v in self.sum.items()}
