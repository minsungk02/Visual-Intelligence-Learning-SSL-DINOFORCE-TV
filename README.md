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
├── notebooks/              # Jupyter 서버용 노트북 (server_*: 학습·추출·평가·분석)
├── evaluate.py             # 고정 LP 평가기 (수정 금지)
├── RESULTS.md              # 실험 기록 + 최종 결과 + 재현성 노트
└── data/ outputs/ logs/ features/   # 실행 시 레포 루트에 자동 생성 (gitignore — 산출물)
```

모든 경로는 레포 루트 기준 상대경로(`./data`, `./outputs`, `./logs`)다. 서버 디스크는 영구적이므로
Colab과 달리 **Drive 마운트·심링크가 전혀 필요 없고**, STL-10(~2.6GB)은 첫 실행 시 `./data`에
자동 다운로드되며 체크포인트·로그·feature는 레포 루트 하위에 그대로 보존된다.

## 환경 설정 (Environment) — Jupyter 서버

- **Python** 3.10+
- **GPU** **NVIDIA RTX A5000 24GB** (Ampere, bf16 지원 — 기준 환경. 24GB+ CUDA GPU 호환)
- **의존성** `requirements.txt` (torch, torchvision, timm, numpy, pyyaml, tqdm, matplotlib …)

JupyterLab **Terminal** 에서 최초 1회:

```bash
# 1) 클론 (jupyter 서버용 브랜치)
git clone -b final-jupyter https://github.com/minsungk02/Visual-Intelligence-Learning-SSL-DINOFORCE-TV.git
cd Visual-Intelligence-Learning-SSL-DINOFORCE-TV

# 2) 가상환경
python -m venv .venv && source .venv/bin/activate

# 3) 의존성 설치
pip install -r requirements.txt

# 4) 내부 라이브러리 editable 설치 (ssl_lib을 import 가능하게)
pip install -e .

# 5) 노트북에서 이 venv를 쓰려면 커널 등록 (notebooks/server_*.ipynb 용)
python -m ipykernel install --user --name ssl --display-name "Python (ssl)"
```

### A5000 기준 학습 설정 (`configs/mocov3_vits8.yaml`)

- **`training.batch_size: 2048` / `optimizer.lr: 1.2e-3` — config 기본값 그대로** 사용한다.
  batch 2048의 VRAM 사용량은 ~17GB로 A5000 24GB에 안전하게 들어간다 (gradient checkpointing 활성 기준).
- `data.num_workers: 4` 도 기본값 유지 — worker 수는 per-worker augmentation RNG 스트림에 영향을
  주므로, 기록된 결과(89.74/87.77)의 결정론 경로를 보존하려면 바꾸지 않는다.
- batch를 바꾸면 **lr을 반드시 함께** 재계산한다: lr = 1.5e-4 × batch/256 (예: 16GB GPU에서
  재실행해야 한다면 batch 1024 + lr 6.0e-4). cosine schedule은 시작 시 전체가 고정되므로
  **학습 도중 epoch/batch 변경 금지**.

## 최종 결과 재현 (3 단계)

```bash
# 1) SSL pretraining — MoCo v3 + ViT-S/8 (500 epoch)
#    A5000(24GB): config 무수정 그대로 (batch 2048 / lr 1.2e-3)
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

## Jupyter 서버 실행 (학습)

500 epoch 학습은 수십 시간이 걸리므로 **노트북 셀이 아니라 JupyterLab Terminal에서
`nohup`(또는 `tmux`)으로 실행**한다. Jupyter 커널은 브라우저가 끊겨도 살아 있지만
셀 출력 스트림이 유실되고 커널 재시작 한 번에 학습이 중단된다. `nohup`이면 브라우저/SSH와
무관하게 진행되고 로그가 파일로 영구히 남는다.

```bash
cd Visual-Intelligence-Learning-SSL-DINOFORCE-TV
source .venv/bin/activate
mkdir -p logs

# 백그라운드 학습 시작
nohup python -u scripts/pretrain.py --config configs/mocov3_vits8.yaml \
  > logs/pretrain_mocov3_vits8.out 2>&1 &

# 진행 모니터링
tail -f logs/pretrain_mocov3_vits8.out   # 실시간 로그
nvidia-smi                               # GPU 사용률 / VRAM
```

서버 재부팅·프로세스 중단 시 resume (체크포인트는 `outputs/`에 영구 보존):
```bash
nohup python -u scripts/pretrain.py --config configs/mocov3_vits8.yaml \
  --resume outputs/mocov3_vits8_seed42/ckpt_ep<N>.pth \
  > logs/pretrain_resume.out 2>&1 &
```

노트북으로 작업하려면 `notebooks/server_*.ipynb` 를 사용한다 (커널: `Python (ssl)`):

| 노트북 | 용도 |
|---|---|
| `server_train_universal.ipynb` | **메인** — CONFIG 한 줄로 모든 method 학습 + 추출 + LP 평가 + 시각화 |
| `server_train_mocov3.ipynb` / `server_train_mocov2.ipynb` | 단일 method 학습 (자동 resume) |
| `server_evaluate.ipynb` | feature 추출 + evaluate.py LP 평가 |
| `server_analyze_training.ipynb` | 학습 로그 파싱·시각화 (loss / feat_std / lr) |

feature 추출(2단계)과 LP 평가(3단계)는 수 분 내로 끝나므로 Terminal이나 노트북 셀
어느 쪽에서 실행해도 무방하다. 단, **500 epoch 본학습만큼은 Terminal + nohup을 권장**한다.

> 이 브랜치(`final-jupyter`)는 Jupyter 서버 전용이다. 레거시 Colab 노트북(Drive
> 마운트·심링크 워크플로)은 `main` 브랜치의 `notebooks/colab_*.ipynb` 에 보존되어 있다.

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
