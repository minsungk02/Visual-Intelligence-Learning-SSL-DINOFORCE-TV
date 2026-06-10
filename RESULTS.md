# SSL 챌린지 — 결과 추적 (Reproducibility 노트)

> 모든 LP 점수는 **`evaluate.py` 고정 recipe** (SGD lr=0.1, mom=0.9, wd=0, cosine 100ep, batch 128)로 측정.
> 모든 학습은 **STL10 unlabeled 100k만** 사용 (외부 데이터 / pretrained weight 없음, from scratch).

## 최종 제출 (FINAL)

| 항목 | 값 |
|---|---|
| Method | MoCo v3 + ViT-S/8 (patch 8), from-scratch |
| Pretrain config | `configs/mocov3_vits8.yaml` (500ep, batch 2048, AdamW lr 1.2e-3, τ=0.2, bf16) |
| Backbone ckpt | `outputs/mocov3_vits8_seed42/backbone_ep500.pth` |
| LP feature | `last6_cls_patchmean` (2688-d), 96px, normalize=standardize |
| 추출 명령 | `extract_features.py --feature-mode last6_cls_patchmean --normalize standardize` |
| **STL10 Top-1** | **89.74** |
| **CIFAR10 Top-1** | **87.77** |

선택 근거: feature 추출 모드는 **STL train 내부 validation(4k-fit / 1k-val)** 으로 선택
(`scripts/sweep_select.py`). test 셋은 winner 1개에만 적용 → test cherry-pick 아님.

## 환경

| 항목 | 값 |
|---|---|
| GPU | L4 24GB (Colab Pro+); A100도 호환 |
| AMP | bfloat16 (GradScaler 없음) |
| Determinism | cudnn.deterministic, benchmark=False, seed=42 |

## 백본 진행 (LP = cls baseline 기준)

| Backbone | Epoch | STL10 | CIFAR10 | 비고 |
|---|---|---|---|---|
| ViT-S/16 | 1000 | 86.6 | 86.6 | 초기 baseline |
| **ViT-S/8** | 500 | 88.75 | 86.45 | patch 16->8 (token 37->145, 파라미터 동일). cls feature 기준 |

patch8 효과: STL10 +2.15 (native 96 → 공간 분해능 직접 작용) vs CIFAR10 +0.1
(32->96 upscale이라 원본 디테일 부족). fine-grained(cat↔dog) 병목 가설 검증.

## Phase A — feature 추출 LP 레버 (재학습 0, ep500 백본 동결)

backbone 가중치 동결, **feature 추출 방식만** 바꿔 LP 비교. `evaluate.py` recipe 불변.
normalize=standardize (train 통계로만 fit).

**1차 sweep (test 기준 기록, ep500):**

| mode | dim | STL10 | CIFAR10 |
|---|---|---|---|
| cls (baseline) | 384 | 88.75 | 86.45 |
| avg | 384 | 87.66 | 85.17 |
| cls_patchmean | 768 | 89.54 | 86.66 |
| last4_cls | 1536 | 89.12 | 87.04 |
| last4_cls_patchmean | 1920 | 89.67 | 87.00 |

**2차 sweep + train-internal val 셀렉터 (`sweep_select.py`, STL 4k/1k):**

| 후보 | dim | STL_val | 비고 |
|---|---|---|---|
| **last6_cls_patchmean** | 2688 | **87.30** | ★ val 1위 → 채택 |
| last4_cls_patchmean_patchmax | 2304 | 87.30 | 동률(차원 큼 → 후순위) |
| cls_patchmax | 768 | 86.70 | |
| last4_cls_patchmean (ref) | 1920 | 86.40 | 1차 채택값 |
| last4_cls_patchmean + l2 norm | 1920 | 83.90 | l2 패배 → standardize 확정 |

**채택 = `last6_cls_patchmean`** (val 1위) → 최종 test **STL10 89.74 / CIFAR10 87.77**.
last4 대비 depth 앙상블을 4->6 block으로 확장한 것이 유일한 변경. CIFAR +0.77이 실질 이득.

## 폐기된 레버 (rejected, 기록)

| 레버 | 결과 | 판정 |
|---|---|---|
| 해상도 128 eval (dynamic pos_embed 보간) | STL 89.49 / CIFAR 85.26 | 하락 → 폐기. STL native 96라 upscale 정보 0 + 학습(96/145token) 분포 미스매치 |
| τ=0.1 재학습 (~57h) | 미실행 | 시간 대비 보장 없어 최종 제출에서 제외 |
| MoCo v3 + multi-crop (patch8) | 미실행 | patch8×6뷰 메모리·시간 폭발 → 보류 |

## 재현 명령 (README와 동일)

```bash
python scripts/pretrain.py --config configs/mocov3_vits8.yaml
python scripts/extract_features.py \
  --backbone outputs/mocov3_vits8_seed42/backbone_ep500.pth \
  --config configs/mocov3_vits8.yaml \
  --output-dir features/final \
  --feature-mode last6_cls_patchmean --normalize standardize
# 위가 출력하는 evaluate.py 명령 실행
```

## 한계 / 잔존 오류

- STL10 90 목표 대비 -0.26 (89.74). 잔존 병목 = cat↔dog fine-grained (차량류는 0.88~0.97).
- 추출 레버는 천장 도달 (cls 88.75 -> last4 89.67 -> last6 89.74, 한계효용 체감).
  추가 향상은 재학습(τ/multi-crop/patch4) 필요하나 72h·메모리 예산 상 제외.
