"""
LARS optimizer (Layer-wise Adaptive Rate Scaling).

You et al., "Large Batch Training of Convolutional Networks", 2017.

핵심 아이디어:
- 각 layer의 trust ratio = ||w|| / (||∇w|| + λ||w||) 로 layer 단위 LR 조정.
- 큰 batch size에서 SGD의 LR scaling 한계를 극복.
- VICReg / Barlow Twins / DINO 등 modern SSL이 표준으로 채택.

이 구현은 SGD-Momentum의 LARS 변형 (NVIDIA / facebookresearch/vicreg 식):
    update_t = momentum * update_{t-1} + grad + wd * w
    w_t = w - trust_ratio * lr * update_t

`exclude_from_layer_adapt=True`인 param group은 LARS 비활성 (bias/BN/predictor용).
"""
from typing import Iterable

import torch
from torch.optim import Optimizer


class LARS(Optimizer):
    """
    LARS optimizer.

    Args:
        params: parameter iterable 또는 param group list.
                각 group에 'exclude_from_layer_adapt' (bool) 키를 두면
                해당 group은 trust ratio 적용 없이 평범한 SGD-momentum.
        lr: base learning rate.
        momentum: momentum coefficient.
        weight_decay: L2 weight decay.
        eta: LARS trust coefficient (보통 0.001).
        eps: numerical stability.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 0.001,
        eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            eps=eps,
            exclude_from_layer_adapt=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            eta = group["eta"]
            eps = group["eps"]
            exclude = group.get("exclude_from_layer_adapt", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if exclude:
                    # 평범한 SGD-momentum (LARS 비활성)
                    if wd != 0.0:
                        grad = grad.add(p, alpha=wd)
                    buf = self.state.get(p, {}).get("momentum_buffer", None)
                    if buf is None:
                        buf = torch.clone(grad).detach()
                    else:
                        buf.mul_(momentum).add_(grad)
                    self.state.setdefault(p, {})["momentum_buffer"] = buf
                    p.add_(buf, alpha=-lr)
                    continue

                # LARS: trust ratio = ||w|| / (||g|| + wd*||w||)
                w_norm = torch.norm(p)
                g_norm = torch.norm(grad)
                trust = torch.where(
                    (w_norm > 0) & (g_norm > 0),
                    eta * w_norm / (g_norm + wd * w_norm + eps),
                    torch.ones_like(w_norm),
                )

                effective = grad
                if wd != 0.0:
                    effective = effective.add(p, alpha=wd)
                effective = effective.mul(trust)

                buf = self.state.get(p, {}).get("momentum_buffer", None)
                if buf is None:
                    buf = torch.clone(effective).detach()
                else:
                    buf.mul_(momentum).add_(effective)
                self.state.setdefault(p, {})["momentum_buffer"] = buf

                p.add_(buf, alpha=-lr)

        return loss


def split_params_for_lars(model, weight_decay: float, predictor_lr_mult: float = 1.0):
    """
    LARS용 param group 분리.

    - Bias / BatchNorm / LayerNorm parameter는 'exclude_from_layer_adapt=True' + wd=0
    - 일반 conv/linear weight는 LARS 적용 + wd 활성
    - model.predictor가 있고 predictor_lr_mult != 1이면 predictor group을 분리 (BYOL용).
      VICReg/MoCo-mc에는 predictor 없음 → 자동 무시.
    """
    bn_bias_params = []
    regular_params = []
    predictor_params = []

    predictor_param_ids = set()
    if predictor_lr_mult != 1.0 and hasattr(model, "predictor"):
        predictor_params_list = list(model.predictor.parameters())
        predictor_param_ids = {id(p) for p in predictor_params_list}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in predictor_param_ids:
            predictor_params.append(p)
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            bn_bias_params.append(p)
        else:
            regular_params.append(p)

    groups = [
        {
            "params": regular_params,
            "weight_decay": weight_decay,
            "exclude_from_layer_adapt": False,
        },
        {
            "params": bn_bias_params,
            "weight_decay": 0.0,
            "exclude_from_layer_adapt": True,
        },
    ]
    if predictor_params:
        groups.append({
            "params": predictor_params,
            "weight_decay": weight_decay,
            "exclude_from_layer_adapt": False,
            "lr_mult": predictor_lr_mult,
        })
    return groups
