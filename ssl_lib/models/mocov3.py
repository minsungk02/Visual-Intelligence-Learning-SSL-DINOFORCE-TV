"""
MoCo v3 — Momentum Contrast (v3).

Chen et al., "An Empirical Study of Training Self-Supervised Vision Transformers",
ICCV 2021. arXiv:2104.02057.

v2와의 핵심 차이:
1. 큐(queue) 제거 — negative를 같은 배치 안에서만 사용 (in-batch).
   → batch가 곧 negative pool. ViT-S면 batch 1024 권장 (≈1023 negatives).
2. Prediction head 추가 (BYOL/SimSiam식) — query 쪽에만 붙는 비대칭 구조.
3. Symmetric InfoNCE: loss = ctr(q1,k2) + ctr(q2,k1).
4. Momentum m = 0.99 (v2의 0.999보다 작음), cosine으로 1.0까지 증가.
5. Temperature τ = 0.2 (v2의 0.07/0.15보다 큼).

구조적으로 momentum encoder + projector + predictor 골격이되 loss는 InfoNCE
(in-batch contrastive). train_loop의 momentum_ema 경로(set_ema_tau +
optimizer.step() 후 _update_target)를 사용한다.

ViT 백본 사용 시 backbone의 frozen patch embed(stop-grad)와 함께 쓰는 게 표준.

학습 후 LP 평가에는 self.backbone (= base_encoder의 backbone)만 사용.
"""
import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .heads import ProjectionHead, Predictor


class MoCoV3(nn.Module):
    """
    MoCo v3 SSL 모델.

    Forward 동작:
        Inputs: (view1, view2) — 같은 이미지의 두 augmented view
        Returns: (loss, log_dict)

    EMA(momentum) 업데이트는 학습 루프에서 처리:
        - set_ema_tau(step, total) : cosine momentum schedule
        - _update_target()         : optimizer.step() 이후 호출
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: YAML config dict. 사용되는 키:
                cfg["backbone"]
                cfg["projection"]["hidden_dim"]   : 4096 (MoCo v3)
                cfg["projection"]["output_dim"]   : 256
                cfg["projection"]["num_layers"]   : 3  (MoCo v3 projector는 3-layer)
                cfg["predictor"]["input_dim"]     : 256 (= projection output_dim)
                cfg["predictor"]["hidden_dim"]    : 4096
                cfg["predictor"]["output_dim"]    : 256
                cfg["mocov3"]["ema_tau_base"]     : 0.99 (cosine로 1.0까지 증가)
                cfg["mocov3"]["ema_schedule"]     : "cosine" | "constant"
                cfg["mocov3"]["temperature"]      : 0.2
                cfg["mocov3"]["symmetric_loss"]   : True 권장
        """
        super().__init__()

        moco_cfg = cfg["mocov3"]
        proj_cfg = cfg["projection"]
        pred_cfg = cfg["predictor"]

        self.ema_tau_base = moco_cfg["ema_tau_base"]
        self.ema_schedule = moco_cfg.get("ema_schedule", "cosine")
        self.temperature = moco_cfg["temperature"]
        self.symmetric = moco_cfg.get("symmetric_loss", True)

        # Base(query) encoder = backbone + projector
        backbone = build_backbone(cfg)
        projector = ProjectionHead(
            input_dim=backbone.feature_dim,
            hidden_dim=proj_cfg["hidden_dim"],
            output_dim=proj_cfg["output_dim"],
            num_layers=proj_cfg.get("num_layers", 3),  # MoCo v3 표준 3-layer
            last_bn=True,
            last_bn_affine=False,  # MoCo v3 projector 마지막 BN은 affine 없음
        )
        self.base_encoder = nn.Sequential(backbone, projector)

        # Prediction head (query 쪽에만)
        self.predictor = Predictor(
            input_dim=pred_cfg["input_dim"],
            hidden_dim=pred_cfg["hidden_dim"],
            output_dim=pred_cfg["output_dim"],
        )

        # Momentum encoder = base_encoder의 deep copy (predictor 제외)
        self.momentum_encoder = copy.deepcopy(self.base_encoder)
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False

        # 현재 EMA momentum (학습 진행에 따라 cosine으로 업데이트)
        self._current_tau = self.ema_tau_base

    @property
    def backbone(self) -> nn.Module:
        """LP 평가용 backbone 접근자."""
        return self.base_encoder[0]

    def set_ema_tau(self, current_step: float, total_steps: float) -> None:
        """
        EMA momentum schedule. MoCo v3 표준: cosine으로 base → 1.0.

        Args:
            current_step: 현재까지 진행된 step(또는 epoch, float 허용)
            total_steps:  전체 학습 step(또는 epoch)
        """
        if self.ema_schedule == "constant":
            self._current_tau = self.ema_tau_base
        elif self.ema_schedule == "cosine":
            # τ = 1 - (1 - τ_base) * (cos(π · k/K) + 1) / 2
            progress = current_step / max(total_steps, 1)
            self._current_tau = 1 - (1 - self.ema_tau_base) * (
                math.cos(math.pi * progress) + 1
            ) / 2
        else:
            raise ValueError(f"Unknown ema_schedule: {self.ema_schedule}")

    @torch.no_grad()
    def _update_target(self) -> None:
        """Momentum encoder를 base encoder의 EMA로 업데이트."""
        for pb, pm in zip(
            self.base_encoder.parameters(),
            self.momentum_encoder.parameters(),
        ):
            pm.data.mul_(self._current_tau).add_(pb.data, alpha=1 - self._current_tau)

    def _contrastive_loss(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """
        In-batch InfoNCE. queue 없이 같은 배치의 다른 샘플을 negative로 사용.

        Args:
            q: (B, D) predictor 출력 (query)
            k: (B, D) momentum projector 출력 (key, stop-grad 가정)

        Returns:
            scalar loss. MoCo v3 공식 구현과 동일하게 2*τ로 스케일
            (τ에 무관하게 loss 크기를 안정적으로 유지).
        """
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        # (B, B) 유사도 — 대각이 positive, 나머지가 negative
        logits = torch.einsum("nd,md->nm", q, k) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels) * (2 * self.temperature)

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        한 step의 학습 forward.

        Args:
            view1, view2: (B, 3, H, W)

        Returns:
            (loss, log_dict)

        Note:
            momentum encoder 업데이트는 여기서 하지 않음.
            학습 루프가 optimizer.step() 이후 _update_target()을 호출.
        """
        # Query: base_encoder -> predictor (gradient O)
        q1 = self.predictor(self.base_encoder(view1))

        # Key: momentum_encoder (gradient X)
        with torch.no_grad():
            k2 = self.momentum_encoder(view2)

        if self.symmetric:
            q2 = self.predictor(self.base_encoder(view2))
            with torch.no_grad():
                k1 = self.momentum_encoder(view1)
            loss = self._contrastive_loss(q1, k2) + self._contrastive_loss(q2, k1)
            feature_std = F.normalize(q1, dim=1).std(dim=0).mean().item()
        else:
            loss = self._contrastive_loss(q1, k2)
            feature_std = F.normalize(q1, dim=1).std(dim=0).mean().item()

        log_dict = {
            "loss": loss.item(),
            "feature_std": feature_std,
            "ema_tau": self._current_tau,
        }
        return loss, log_dict
