# SSL Challenge — STL10 unlabeled SSL pretraining

STL10 unlabeled 100k로 **from-scratch** SSL pretraining 후, **frozen encoder**에 대해
STL10 / CIFAR10 Linear Probing(LP) 성능을 평가한다.

## 최종 제출 (Final submission)

| 단계 | 내용 |
|---|---|
| SSL 방법 | **MoCo v3** (momentum encoder + projector + predictor, queue 없는 in-batch symmetric InfoNCE, frozen patch embed) |
| Backbone | **ViT-S/8** (`vit_small_patch8_224`, img 96 → 145 token), from-scratch, 21.4M |
| 학습 | STL10 unlabeled 100k, 500 epoch, batch 2048, AdamW lr 1.2e-3, τ=0.2, bf16 |
| LP feature | **`last6_cls_patchmean`** (마지막 6 block CLS concat ⊕ 최심 block patch mean = **2688-d**), 96px, per-dim standardize |
| LP recipe | `evaluate.py` 고정 (SGD lr=0.1, mom=0.9, wd=0, cosine 100ep, batch 128) — **불변** |
| **결과** | **STL10 89.74 / CIFAR10 87.77** (Top-1) |

> feature 추출 모드는 **STL train 내부 validation(4k-fit/1k-val)** 으로 선택 (test 누수 없음). `scripts/sweep_select.py` 참조. 자세한 실험·근거는 `RESULTS.md`.

## 디렉토리 구조

```
.
├── ssl_lib/                # 공유 라이브러리 (editable install)
│   ├── data/               # STL10 unlabeled, two-view / multi-crop augmentation
│   ├── models/             # backbone (ResNet/ViT/Swin), heads, MoCoV2/V3, BYOL, VICReg
│   ├── utils/              # seed, schedulers, checkpoint, logging, LARS
│   └── train_loop.py       # 학습 루프 (method dispatch)
├── configs/                # YAML configs (mocov3_vits8 = 최종)
├── scripts/                # CLI 진입점 (pretrain / extract_features / sweep_select ...)
├── notebooks/              # Colab 학습·평가 노트북
├── evaluate.py             # 고정 LP 평가기 (수정 금지)
├── RESULTS.md              # 실험 기록 + 최종 결과 + 재현성 노트
└── outputs/ logs/ data/ features/   # (gitignore — 산출물)
```

## 환경 설정 (Environment)

- **Python** 3.10+
- **GPU** CUDA GPU 권장 (개발 환경: Colab L4 24GB / 서버용 RTX 5080).
- **의존성** `requirements.txt` (torch, torchvision, timm, numpy, pyyaml, tqdm, matplotlib …)

```bash
# 1) 가상환경(선택)
python -m venv .venv && source .venv/bin/activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 내부 라이브러리 editable 설치 (ssl_lib을 import 가능하게)
pip install -e .
```

## 최종 결과 재현 (3 단계)

```bash
# 1) SSL pretraining — MoCo v3 + ViT-S/8 (500 epoch)
#    산출물: outputs/mocov3_vits8_seed42/backbone_ep500.pth
python scripts/pretrain.py --config configs/mocov3_vits8.yaml

# 2) feature 추출 — 최종 모드 last6_cls_patchmean, 96px, standardize
python scripts/extract_features.py \
  --backbone outputs/mocov3_vits8_seed42/backbone_ep500.pth \
  --config   configs/mocov3_vits8.yaml \
  --output-dir features/final \
  --feature-mode last6_cls_patchmean \
  --normalize standardize

# 3) LP 평가 — (2)가 출력하는 evaluate.py 명령 그대로 실행
python evaluate.py \
  --stl10-train-features features/final/stl10_train_features.npy \
  --stl10-train-labels   features/final/stl10_train_labels.npy \
  --stl10-test-features  features/final/stl10_test_features.npy \
  --stl10-test-labels    features/final/stl10_test_labels.npy \
  --cifar10-train-features features/final/cifar10_train_features.npy \
  --cifar10-train-labels   features/final/cifar10_train_labels.npy \
  --cifar10-test-features  features/final/cifar10_test_features.npy \
  --cifar10-test-labels    features/final/cifar10_test_labels.npy
```

### (선택) feature 추출 모드 선택 재현
test 누수 없이 추출 모드를 STL train 내부 val로 고른 과정:
```bash
python scripts/sweep_select.py \
  --config configs/mocov3_vits8.yaml \
  --ckpt-ep500 outputs/mocov3_vits8_seed42/backbone_ep500.pth \
  --final-output-dir features/sweep_winner
```

## Colab (Pro+) 실행

세션마다 `git clone` -> `pip install -e .` -> Drive 마운트 후 `data/ outputs/ logs/`를
Drive에 심링크(체크포인트 영구 보존, 세션 끊겨도 resume). 학습 노트북:
`notebooks/colab_train_mocov3.ipynb`. Drive 경로: `MyDrive/ssl_project/{data,outputs,logs}`.

Resume:
```bash
python scripts/pretrain.py --config configs/mocov3_vits8.yaml \
  --resume outputs/mocov3_vits8_seed42/ckpt_ep<N>.pth
```

## 산출물 형식

```
outputs/mocov3_vits8_seed42/
├── ckpt_ep{N}.pth       # 전체 state (resume용)
└── backbone_ep{N}.pth   # backbone 가중치만 (LP feature 추출용) — 최근 N개만 유지
```
LP 평가는 `backbone_ep*.pth`만 필요. `extract_features.py`가 feature `(N,D)`를 뽑아
`evaluate.py`(고정 recipe)에 투입한다. `evaluate.py`는 이미지가 아니라 추출된 feature만 받으므로
backbone·feature 차원·해상도는 학생이 통제한다(docstring 명시).

## 핵심 설계 결정

- **MoCo v3 + ViT 선택**: 평가가 frozen **linear-probe-only**라, fine-tune 특화(MAE류)가 아닌
  contrastive/momentum 계열이 적합. ViT frozen patch embed로 학습 안정화(Chen et al. 2021).
- **ViT-S/8 (patch 16->8)**: cat<->dog 등 fine-grained 병목 = 공간 분해능 부족 가설.
  token 37->145로 해상도 4배, **파라미터는 동일(21.4M)** -> 작은 데이터(100k) 과적합 위험 없이
  locality 강화 (cf. DINO ViT-S/8). STL10 LP 86.6 -> 88.75.
- **feature 추출 다각화 (재학습 0)**: backbone 동결, "읽는 법"만 바꿔 LP 향상.
  최종 `last6_cls_patchmean` = depth 앙상블(마지막 6 block CLS) + locality(최심 patch mean).
  선택은 **STL train 내부 val** 로만 -> test cherry-pick 아님.
- **공유 라이브러리 단일 진실 소스**: backbone/data/aug 공유 -> 방법 간 공정 비교.
- **고정 seed + 결정론적 설정**: 재현성(peer-review 2-seed) 보장.

## Baselines (탐색 기록)

비교용으로 MoCo v2 (ResNet-50), MoCo v2 + multi-crop, BYOL, VICReg 구현을 유지한다
(`configs/`, `ssl_lib/models/`). 최종 제출 경로는 위 MoCo v3 + ViT-S/8.
