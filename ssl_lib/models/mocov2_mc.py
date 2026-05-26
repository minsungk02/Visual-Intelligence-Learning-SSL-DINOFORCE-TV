"""
MoCo v2 + Multi-Crop — Cross-Resolution용 강화 baseline.

기존 MoCo v2 (ssl_lib/models/mocov2.py)를 multi-crop으로 확장:
- Query encoder: 모든 crops(2 global + N local) 처리
- Key encoder:   글로벌 2개만 처리 (DINO 관행 — local은 teacher에 안 줌)
- InfoNCE loss: 가능한 (q_i, k_j) pair (i != j) 모두 합산

설계 결정:
- queue에 enqueue되는 key는 글로벌 2개만 (해상도 일관성 + queue 의미 보존).
- gradient_checkpoint 옵션 그대로 사용 가능 (locals은 작아서 ckpt 없어도 메모리 여유).
- temperature, momentum, queue_size는 기존 MoCo v2 config 그대로 (0.15 등 미세조정 가능).

이 파일은 기존 mocov2.py와 분리되어 있어 baseline 비교 시 동일 코드 보장.
"""
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .heads import ProjectionHead


class MoCoV2MC(nn.Module):
    """
    MoCo v2 + multi-crop SSL 모델.

    Forward:
        Inputs:
            crops: List[Tensor] — 길이 ≥ 2. crops[0], crops[1]은 global,
                                  crops[2:]은 local crops.
            (구버전 호환을 위해 (v1, v2) 두 인자도 받음 — 그땐 [v1, v2]로 처리.)
        Returns: (loss, log_dict)

    학습 후 LP 평가에는 self.encoder_q[0] (query backbone)만 사용.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mc = cfg["mocov2"]
        pc = cfg["projection"]

        self.queue_size = mc["queue_size"]
        self.momentum = mc["momentum"]
        self.temperature = mc["temperature"]
        self.feature_dim = pc["output_dim"]
        self.symmetric = mc.get("symmetric_loss", True)

        # Query encoder
        bb_q = build_backbone(cfg)
        proj_q = ProjectionHead(
            input_dim=bb_q.feature_dim,
            hidden_dim=pc["hidden_dim"],
            output_dim=pc["output_dim"],
            num_layers=pc.get("num_layers", 2),
            last_bn=False,
        )
        self.encoder_q = nn.Sequential(bb_q, proj_q)

        # Key encoder = deepcopy of query
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        # Queue
        self.register_buffer(
            "queue",
            F.normalize(torch.randn(self.feature_dim, self.queue_size), dim=0),
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @property
    def backbone(self) -> nn.Module:
        return self.encoder_q[0]

    @torch.no_grad()
    def _momentum_update(self) -> None:
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.mul_(self.momentum).add_(pq.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """keys: (B, D), 정규화된 key. queue_size는 B의 배수여야."""
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        assert self.queue_size % B == 0, (
            f"queue_size ({self.queue_size}) must be divisible by batch ({B})"
        )
        self.queue[:, ptr:ptr + B] = keys.T
        ptr = (ptr + B) % self.queue_size
        self.queue_ptr[0] = ptr

    def _info_nce(self, q: torch.Tensor, k: torch.Tensor,
                  queue_snapshot: torch.Tensor) -> torch.Tensor:
        # q, k: (B, D), queue_snapshot: (D, K)
        l_pos = torch.einsum("nd,nd->n", q, k).unsqueeze(-1)  # (B, 1)
        l_neg = torch.einsum("nd,dk->nk", q, queue_snapshot)  # (B, K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(self, *args) -> Tuple[torch.Tensor, dict]:
        # Inputs normalize
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            crops: List[torch.Tensor] = list(args[0])
        elif len(args) == 2:
            crops = [args[0], args[1]]
        else:
            raise ValueError(
                f"MoCoV2MC.forward: expected (v1, v2) or ([crops]); got {len(args)} args"
            )

        assert len(crops) >= 2, "Need at least 2 global crops"
        global_crops = crops[:2]
        local_crops = crops[2:]
        n_g = len(global_crops)
        n_l = len(local_crops)

        # Momentum update
        self._momentum_update()

        # ── 절충 batch-concat forward ──
        # Globals(96×96)를 concat하면 단일 conv 텐서가 2.4GB → backward에서 OOM.
        # 그래서 globals는 per-crop forward, locals(32×32, 작음)만 concat.

        # Globals: per-crop forward (메모리 안전)
        q_globals = [F.normalize(self.encoder_q(g), dim=1) for g in global_crops]

        # Locals: (n_l * B, 3, 32, 32) 한 번에 처리 — 32×32라 batch 커도 안전
        if n_l > 0:
            local_batch = torch.cat(local_crops, dim=0)
            q_local_all = F.normalize(self.encoder_q(local_batch), dim=1)
            q_locals = list(q_local_all.chunk(n_l, dim=0))
        else:
            q_locals = []

        # Key encoder — globals per-crop (no grad)
        with torch.no_grad():
            k_globals = [F.normalize(self.encoder_k(g), dim=1) for g in global_crops]

        # Queue snapshot — backward graph 안전성
        queue_snapshot = self.queue.detach().clone()

        # Pair-wise InfoNCE
        # Global queries × global keys (i != j) — 항상 양방향 sym
        loss_terms = []
        loss_terms.append(self._info_nce(q_globals[0], k_globals[1], queue_snapshot))
        if self.symmetric:
            loss_terms.append(self._info_nce(q_globals[1], k_globals[0], queue_snapshot))

        # Local queries × global keys (모든 cross pair)
        for q_l in q_locals:
            for k_g in k_globals:
                loss_terms.append(self._info_nce(q_l, k_g, queue_snapshot))

        loss = torch.stack(loss_terms).mean()

        # Queue enqueue (global keys only)
        for k_g in k_globals:
            self._dequeue_and_enqueue(k_g.detach())

        # 모니터링
        feature_std = q_globals[0].std(dim=0).mean().item()
        log = {
            "loss": loss.item(),
            "feature_std": feature_std,
            "n_pairs": float(len(loss_terms)),
        }
        return loss, log
