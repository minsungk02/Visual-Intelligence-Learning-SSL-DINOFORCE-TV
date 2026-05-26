"""
VICReg — Variance-Invariance-Covariance Regularization.

Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning", ICLR 2022. arXiv:2105.04906.

핵심 메커니즘 (negative pair, momentum encoder, predictor 모두 없음):
- Invariance : MSE(z1, z2)         — 두 view의 projection이 같아야
- Variance   : per-dim std hinge   — 각 출력 차원의 std가 γ 이상 유지 (collapse 방지)
- Covariance : off-diagonal² / D   — 출력 차원 간 상관 0으로 (정보 중복 제거)

설계:
- EMA / queue / predictor / centering 등 stateful 요소 없음 → seed-robust.
- 표준 hyperparam: sim_coef=25, var_coef=25, cov_coef=1, γ=1.
- Projection MLP는 크게 (예: 8192) — variance/covariance가 잘 작동하려면 출력 차원 큰 것이 유리.

직접 작성 가능한 핵심 식 (페이퍼 Eq. 1-4) — 코드 이해 요건 통과.
"""
from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .heads import ProjectionHead


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """정사각 행렬 x의 off-diagonal 원소 flatten."""
    n, m = x.shape
    assert n == m, "off_diagonal: matrix must be square"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coef: float = 25.0,
    var_coef: float = 25.0,
    cov_coef: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, dict]:
    """
    VICReg loss.

    Args:
        z1, z2: (B, D) projection 출력. **L2 정규화하지 않은** raw projection.
        sim_coef, var_coef, cov_coef: 세 항의 가중치.
        gamma: variance hinge target (보통 1.0).
        eps: variance sqrt 안정성.

    Returns:
        (loss, log_dict) — log_dict에는 sim/var/cov 분리값 포함.
    """
    B, D = z1.shape

    # ---- invariance ----
    sim = F.mse_loss(z1, z2)

    # ---- variance hinge ----
    # batch dim에 대한 std가 gamma 미만이면 페널티
    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + eps)
    var = 0.5 * (F.relu(gamma - std_z1).mean() + F.relu(gamma - std_z2).mean())

    # ---- covariance off-diagonal ----
    z1_c = z1 - z1.mean(dim=0, keepdim=True)
    z2_c = z2 - z2.mean(dim=0, keepdim=True)
    cov_z1 = (z1_c.T @ z1_c) / max(B - 1, 1)
    cov_z2 = (z2_c.T @ z2_c) / max(B - 1, 1)
    cov = (_off_diagonal(cov_z1).pow(2).sum() / D
           + _off_diagonal(cov_z2).pow(2).sum() / D)

    loss = sim_coef * sim + var_coef * var + cov_coef * cov

    log = {
        "loss": loss.item(),
        "sim": sim.item(),
        "var": var.item(),
        "cov": cov.item(),
        "z_std": std_z1.mean().item(),  # collapse detector
    }
    return loss, log


class VICReg(nn.Module):
    """
    VICReg SSL 모델.

    Forward:
        Inputs:
            - 2-view 모드: (view1, view2)  — 기존 baseline과 호환
            - multi-crop 모드: List[Tensor] — global+local 합산 pair-wise loss
        Returns: (loss, log_dict)

    학습 후 LP 평가에는 self.backbone만 사용 (projector는 버림).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        vc = cfg["vicreg"]
        pc = cfg["projection"]

        self.sim_coef = vc.get("sim_coef", 25.0)
        self.var_coef = vc.get("var_coef", 25.0)
        self.cov_coef = vc.get("cov_coef", 1.0)
        self.gamma = vc.get("gamma", 1.0)

        bb = build_backbone(cfg)
        proj = ProjectionHead(
            input_dim=bb.feature_dim,
            hidden_dim=pc["hidden_dim"],
            output_dim=pc["output_dim"],
            num_layers=pc.get("num_layers", 3),
            last_bn=False,  # VICReg 페이퍼: 마지막 BN 없음
        )
        self._backbone = bb
        self.projector = proj

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self._backbone(x))

    def forward(
        self,
        *args,
    ) -> Tuple[torch.Tensor, dict]:
        # 두 가지 호출 형태 지원:
        #  forward(view1, view2)        — 2-view 모드
        #  forward([crop1, crop2, ...]) — multi-crop 모드
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            crops: List[torch.Tensor] = list(args[0])
        elif len(args) == 2:
            crops = [args[0], args[1]]
        else:
            raise ValueError(
                f"VICReg.forward: expected (v1, v2) or ([crops]); got {len(args)} args"
            )

        # ── Batch-concat forward (해상도별 grouping) ──
        # 같은 해상도의 crop들을 batch concat해서 1번 forward.
        # 2-view 모드면 group 1개 (96×96 × 2장) → 2 forward → 1 forward.
        # multi-crop 모드면 group 2개 (global 96, local 32) → 6 forward → 2 forward.
        from collections import defaultdict
        groups = defaultdict(list)  # (H, W) -> [(orig_idx, tensor), ...]
        for i, c in enumerate(crops):
            groups[(c.shape[-2], c.shape[-1])].append((i, c))

        zs = [None] * len(crops)
        for items in groups.values():
            idxs = [i for i, _ in items]
            batch = torch.cat([t for _, t in items], dim=0)
            z_batch = self._encode(batch)
            for idx, z in zip(idxs, z_batch.chunk(len(items), dim=0)):
                zs[idx] = z

        # Pair-wise VICReg loss — 모든 (i<j) 조합 평균
        n = len(zs)
        total_loss = 0.0
        agg = {"sim": 0.0, "var": 0.0, "cov": 0.0, "z_std": 0.0}
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                # 두 crop의 batch dim은 동일 (같은 source 이미지).
                # 해상도가 다를 수 있지만 projector는 (B, D)로 똑같이 매핑되므로 OK.
                pair_loss, pair_log = vicreg_loss(
                    zs[i], zs[j],
                    sim_coef=self.sim_coef,
                    var_coef=self.var_coef,
                    cov_coef=self.cov_coef,
                    gamma=self.gamma,
                )
                total_loss = total_loss + pair_loss
                for k in agg:
                    agg[k] += pair_log[k]
                n_pairs += 1

        loss = total_loss / n_pairs
        log = {"loss": loss.item(),
               "feature_std": agg["z_std"] / n_pairs,
               "sim": agg["sim"] / n_pairs,
               "var": agg["var"] / n_pairs,
               "cov": agg["cov"] / n_pairs}
        return loss, log
