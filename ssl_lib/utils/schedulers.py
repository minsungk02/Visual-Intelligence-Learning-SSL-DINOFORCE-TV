"""
Optimizer 빌더 + Cosine LR scheduler.

설계:
- MoCo v2: SGD + momentum.
- MoCo v3 (ViT): AdamW (bias/norm/pos_embed/cls는 weight_decay 제외).
- VICReg / MoCo-mc / Barlow Twins: LARS.
- predictor_lr_mult가 있으면 predictor group을 분리해서 다른 LR 적용.
- LR scheduler는 step-wise cosine + warmup.
"""
import math
from typing import List

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Config dict로부터 optimizer 빌드. SGD / LARS / AdamW 지원.

    - SGD:   기존 baseline (MoCo v2).
    - LARS:  VICReg, MoCo-mc, Barlow Twins 등에서 권장.
             bias/BN parameter는 layer adaptation에서 제외 + wd=0.
    - AdamW: MoCo v3 ViT 표준. bias/norm/pos_embed/cls는 wd 제외.

    cfg["optimizer"]["predictor_lr_mult"]가 있으면
    predictor parameter group을 분리해서 다른 LR 적용.
    """
    opt_cfg = cfg["optimizer"]
    name = opt_cfg.get("name", "sgd").lower()
    base_lr = opt_cfg["lr"]
    momentum = opt_cfg.get("momentum", 0.9)
    weight_decay = opt_cfg["weight_decay"]

    predictor_lr_mult = opt_cfg.get("predictor_lr_mult", None)

    if name == "adamw":
        # MoCo v3 ViT 표준 optimizer.
        # ndim<=1(bias, LayerNorm/BN) + pos_embed/cls_token은 weight decay 제외.
        # 동결된 patch_embed 등 requires_grad=False 파라미터는 자동 제외.
        decay, no_decay = [], []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "pos_embed" in pname or "cls_token" in pname:
                no_decay.append(p)
            else:
                decay.append(p)
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=base_lr,
            betas=betas,
        )
        return optimizer

    if name == "lars":
        from .lars import LARS, split_params_for_lars
        groups = split_params_for_lars(
            model,
            weight_decay=weight_decay,
            predictor_lr_mult=predictor_lr_mult if predictor_lr_mult else 1.0,
        )
        # 각 group에 base_lr 적용 (predictor group은 lr_mult로 스케일)
        for g in groups:
            g["lr"] = base_lr * g.pop("lr_mult", 1.0)
        optimizer = LARS(
            groups,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=opt_cfg.get("lars_eta", 0.001),
        )
        return optimizer

    # SGD path
    if predictor_lr_mult is None or not hasattr(model, "predictor"):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=base_lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        predictor_params = list(model.predictor.parameters())
        predictor_param_ids = {id(p) for p in predictor_params}

        other_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in predictor_param_ids
        ]

        optimizer = torch.optim.SGD(
            [
                {"params": other_params, "lr": base_lr},
                {"params": predictor_params, "lr": base_lr * predictor_lr_mult},
            ],
            momentum=momentum,
            weight_decay=weight_decay,
        )

    return optimizer


class CosineLRScheduler:
    """
    Cosine LR schedule with linear warmup.
    
    매 step마다 호출. epoch 기반이 아니라 step 기반.
    predictor_lr_mult가 적용된 param group의 비율을 유지하면서 base LR만 스케줄링.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        """
        Args:
            optimizer: torch optimizer.
            base_lr: peak LR (warmup 끝나는 시점의 LR).
            warmup_steps: warmup 동안의 step 수.
            total_steps: 전체 학습 step 수.
            min_lr: cosine 끝나는 시점의 LR.
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_count = 0
        
        # 각 param group의 LR ratio (base 대비) 저장
        # predictor group처럼 lr_mult 적용된 경우 그 비율 유지
        self.lr_ratios: List[float] = [
            g["lr"] / base_lr for g in optimizer.param_groups
        ]
    
    def step(self) -> float:
        """한 step 진행 후 현재 LR 반환 (첫 param group 기준)."""
        self.step_count += 1
        lr = self._compute_lr(self.step_count)
        for g, ratio in zip(self.optimizer.param_groups, self.lr_ratios):
            g["lr"] = lr * ratio
        return lr
    
    def _compute_lr(self, step: int) -> float:
        """현재 step에서의 base LR 계산."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * step / max(self.warmup_steps, 1)
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
