# SSL 챌린지 — 결과 추적 (Reproducibility 노트)

> 모든 LP 점수는 **`evaluate.py` 고정 recipe** (SGD lr=0.1, mom=0.9, wd=0, cosine 100ep, batch 128)로 측정.
> 모든 학습은 **STL10 unlabeled 100k만** 사용 (외부 데이터 / pretrained weight 없음, from scratch).

## 환경

| 항목 | 값 |
|---|---|
| GPU | L4 22GB (Colab Pro+) / Ampere 24GB 동급 |
| PyTorch | 2.x |
| AMP | bfloat16 (GradScaler 없음) |
| Determinism | `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)` |
| Seed | 42 (메인), 123 (재현성 cross-check) |

## 실험 결과

| # | Method        | Backbone     | Epochs | Aug          | Optim | STL10 LP | CIFAR10 LP | 시간 (h) | Commit | 비고 |
|---|---------------|--------------|--------|--------------|-------|----------|------------|---------|--------|------|
| 1 | MoCo v2       | R50 small    | 200    | 2-view       | SGD   |  76.92   |  78.51     | ~60     | -      | baseline (기존) |
| 2 | MoCo v3       | ViT-S/16 @96 | 300    | 2-view       | AdamW |   ?      |    ?       | ?       | -      | **main** (이전 BYOL 폐기 후 전환) |
| 3 | MoCo v2-mc    | R50 small    | 200    | 2g96+4l32    | SGD   |   ?      |    ?       | ?       | -      | multi-crop, bf16, compile |
| 4 | VICReg        | R50 small    | 200    | 2-view       | LARS  |   ?      |    ?       | ?       | -      | EMA 없음, seed-robust |
| 5 | (stretch) VICReg-mc | R50 small | 150 | 2g96+4l32  | LARS  |   ?      |    ?       | ?       | -      | 시간 여유 시 |

(`?`는 학습 후 채워 넣을 것)

## 재현성 cross-check (seed sweep)

선정된 메인 method 1개를 두 seed로 short run하여 ±1% 안에 들어오는지 확인.

| Method | Seed | Epochs | STL10 LP | CIFAR10 LP |
|---|---|---|---|---|
| (선정 method) | 42  | 50 | ? | ? |
| (선정 method) | 123 | 50 | ? | ? |

## 학습 sanity (20 epoch 단계 통과 기록)

| Method | feature_std (ep20) | avg_loss (ep20) | epoch_time (s) | GPU mem (GB) | 비고 |
|---|---|---|---|---|---|
| MoCo v2-mc | ? | ? | ? | ? | feature_std > 0.1 유지 OK? |
| VICReg     | ? | ? | ? | ? | var/cov 폭주 없음 OK? |

## 사용한 config / commit

각 학습마다 아래를 기록:

```
- run: mocov2_mc_seed42
  config: configs/mocov2_mc_r50.yaml
  commit: <git rev-parse HEAD>
  start: 2026-MM-DD HH:MM (KST)
  end:   2026-MM-DD HH:MM (KST)
  feature_path: features/mocov2_mc_ep200/
```

## 최종 제출 점수

| Dataset | Top-1 Acc (%) |
|---|---|
| STL10   | (최종 선정 method) |
| CIFAR10 | (최종 선정 method) |

**제출 method**: (이름) + (config 파일) + (사용한 ckpt 경로)
