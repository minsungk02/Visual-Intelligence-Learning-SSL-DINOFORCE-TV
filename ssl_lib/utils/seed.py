"""
Random seed 고정.

재현성 평가가 있는 챌린지라 모든 seed 소스를 통제.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    모든 random source의 seed 고정.

    Args:
        seed: 시드 값.
        deterministic: True면 cudnn deterministic 모드 + CUBLAS workspace 고정.
            성능 약간 손해 가능. 재현성 평가 통과가 우선이므로 True 권장.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 일부 CUDA 연산(예: matmul)의 deterministic 보장에 필요.
        # warn_only=True로 두면 deterministic alternative가 없을 때 학습이 멈추지 않음.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        # 비결정적 모드 — 학습 속도 우선
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def make_loader_generator(seed: int) -> torch.Generator:
    """DataLoader shuffle 재현용 generator."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
