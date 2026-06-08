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

## Phase A — feature 추출 LP 레버 (재학습 0, ep500 백본)

ep500 MoCo v3 ViT-S/8 백본을 그대로 두고 **feature 추출 방식만** 바꿔 LP 비교.
`evaluate.py` LP recipe 불변 (docstring이 feature 추출/차원 변경을 명시 허용).
normalize=standardize (train 통계로만 fit), 2026-06-09 Colab 인라인 추출.

| mode | dim | STL10 LP | CIFAR10 LP | 비고 |
|---|---|---|---|---|
| cls (baseline) | 384 | 88.75 | 86.45 | 기존과 bit-identical |
| avg | 384 | 87.66 | 85.17 | patch mean 단독 |
| cls_patchmean | 768 | 89.54 | 86.66 | CLS ⊕ patch mean |
| last4_cls | 1536 | 89.12 | 87.04 | 마지막 4 block CLS (DINO 표준) |
| **last4_cls_patchmean (채택)** | 1920 | **89.67** | 87.00 | last4_cls ⊕ 최심 patch mean |

**채택 = `last4_cls_patchmean`**: STL10 1위(89.67), CIFAR10 동률 최상(87.00).
baseline 대비 STL10 **+1.07**, CIFAR10 **+0.54** (재학습 0, forward만).
채택 근거 = 결과 전 사전등록된 DINO linear-eval 표준 recipe (test cherry-pick 아님).
한계: STL10 90% 목표 대비 -0.33%p → 다음 레버(해상도-128 eval / τ=0.1 재학습) 필요.

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
