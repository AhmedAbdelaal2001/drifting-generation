from __future__ import annotations

import os
import random
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (best-effort; may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(system_cfg: dict) -> torch.device:
    """
    Decide torch device from config.
    system_cfg:
      - device: "auto" | "cpu" | "cuda"
    """
    mode = str(system_cfg.get("device", "auto")).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")