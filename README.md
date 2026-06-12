# SSL Challenge — STL-10 unlabeled SSL pretraining

STL-10 unlabeled 100k 이미지로 from-scratch SSL pretraining을 한 뒤, encoder를 freeze한 상태에서
STL-10 / CIFAR-10에 대한 Linear Probing(LP) 성능을 측정한다. 평가는 학습된 backbone에서 뽑은
feature만 사용하며, 채점에 쓰는 `evaluate.py`는 수정하지 않는다.

## 최종 제출

| 항목 | 내용 |
|---|---|
| SSL 방법 | MoCo v3 (momentum encoder + projector + predictor, queue 없는 in-batch symmetric InfoNCE, frozen patch embed) |
| Backbone | ViT-S/8 (`vit_small_patch8_224`, 96px 입력에서 145 token), from-scratch, 21.4M |
| 학습 | STL-10 unlabeled 100k, 500 epoch, batch 2048, AdamW lr 1.2e-3, τ=0.2, bf16 |
| LP feature | `last6_cls_patchmean` — 마지막 6개 block의 CLS를 concat하고 최심 block의 patch mean을 덧붙인 2688-d. 96px, per-dim standardize |
| LP recipe | `evaluate.py` 고정값 (SGD lr=0.1, momentum=0.9, wd=0, cosine 100 epoch, batch 128) |
| 결과 | STL-10 89.74 / CIFAR-10 87.77 (Top-1) |

feature 추출 모드는 STL train 내부 validation(4k-fit / 1k-val)으로 골랐고 test에는 winner 하나만
적용했다. test cherry-pick이 아니라는 뜻이며, 선택 과정은 `scripts/sweep_select.py`에 있다.
실험 기록과 근거는 `RESULTS.md`를 참고한다.

## 디렉토리 구조

```
.
├── ssl_lib/                공유 라이브러리 (editable install)
│   ├── data/               STL-10 unlabeled, two-view / multi-crop augmentation
│   ├── models/             backbone (ResNet/ViT/Swin), heads, MoCo v2/v3, BYOL, VICReg
│   ├── utils/              seed, schedulers, checkpoint, logging, LARS
│   └── train_loop.py       학습 루프 (method dispatch)
├── configs/                YAML configs (mocov3_vits8 = 최종)
├── scripts/                CLI 진입점 (pretrain / extract_features / sweep_select 등)
├── notebooks/              Jupyter 서버용 노트북 (server_*: 학습·추출·평가·분석)
├── evaluate.py             고정 LP 평가기 (수정 금지)
├── RESULTS.md              실험 기록 + 최종 결과 + 재현성 노트
└── data/ outputs/ logs/ features/   실행 시 자동 생성 (gitignore 처리된 산출물)
```

모든 경로는 레포 루트 기준 상대경로(`./data`, `./outputs`, `./logs`)다. 서버 디스크가 영구적이라
Colab 같은 Drive 마운트나 심링크가 필요 없다. STL-10(약 2.6GB)은 첫 실행 때 `./data`로 자동
다운로드되고, 체크포인트·로그·feature는 레포 루트 아래에 그대로 쌓인다.

## 환경

- Python 3.10 이상
- GPU: NVIDIA RTX A5000 24GB 기준 (Ampere, bf16 지원). 24GB 이상 CUDA GPU면 호환된다.
- 의존성: `requirements.txt` (torch, torchvision, timm, numpy, pyyaml, tqdm, matplotlib 등)

JupyterLab Terminal에서 최초 한 번 세팅한다.

```bash
git clone https://github.com/minsungk02/Visual-Intelligence-Learning-SSL-DINOFORCE-TV.git
cd Visual-Intelligence-Learning-SSL-DINOFORCE-TV

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # ssl_lib을 import 가능하게

# 노트북에서 이 venv를 쓰려면 커널 등록
python -m ipykernel install --user --name ssl --display-name "Python (ssl)"
```

학습 설정(`configs/mocov3_vits8.yaml`)은 A5000 기준으로 그대로 쓰면 된다. batch 2048에서 VRAM은
약 17GB로 gradient checkpointing을 켜면 24GB에 안전하게 들어간다. `num_workers`는 4를 유지하는데,
worker 수가 per-worker augmentation의 RNG 스트림을 바꿔서 기록된 결과(89.74 / 87.77)의 결정론
경로가 달라질 수 있기 때문이다.

batch를 바꾸려면 lr도 함께 조정해야 한다. 규칙은 lr = 1.5e-4 × batch/256이다. 예를 들어 16GB GPU에서
재실행한다면 batch 1024 + lr 6.0e-4가 된다. cosine schedule은 시작 시점에 전체 길이가 고정되므로
학습 도중에 epoch 수나 batch를 바꾸지 않는다.

## 재현 파이프라인

학습 → feature 추출 → LP 평가의 세 단계로 끝난다.

```bash
# 1) SSL pretraining (MoCo v3 + ViT-S/8, 500 epoch)
#    A5000은 config 무수정 그대로 (batch 2048 / lr 1.2e-3)
#    결과물: outputs/mocov3_vits8_seed42/backbone_ep500.pth
python scripts/pretrain.py --config configs/mocov3_vits8.yaml

# 2) feature 추출 (last6_cls_patchmean, 96px, per-dim standardize)
python scripts/extract_features.py \
  --backbone outputs/mocov3_vits8_seed42/backbone_ep500.pth \
  --config   configs/mocov3_vits8.yaml \
  --output-dir features/final \
  --feature-mode last6_cls_patchmean \
  --normalize standardize

# 3) LP 평가 (2단계가 출력하는 evaluate.py 명령을 그대로 실행)
python evaluate.py \
  --stl10-train-features   features/final/stl10_train_features.npy \
  --stl10-train-labels     features/final/stl10_train_labels.npy \
  --stl10-test-features    features/final/stl10_test_features.npy \
  --stl10-test-labels      features/final/stl10_test_labels.npy \
  --cifar10-train-features features/final/cifar10_train_features.npy \
  --cifar10-train-labels   features/final/cifar10_train_labels.npy \
  --cifar10-test-features  features/final/cifar10_test_features.npy \
  --cifar10-test-labels    features/final/cifar10_test_labels.npy
```

추출 모드를 어떻게 골랐는지(test 누수 없이 STL train 내부 val로 선택)까지 재현하려면 아래를 실행한다.

```bash
python scripts/sweep_select.py \
  --config configs/mocov3_vits8.yaml \
  --ckpt-ep500 outputs/mocov3_vits8_seed42/backbone_ep500.pth \
  --final-output-dir features/sweep_winner
```

## 서버에서 학습 돌리기

500 epoch 학습은 수십 시간이 걸린다. 노트북 셀에서 돌리면 커널이 한 번 재시작되는 순간 학습이
끊기고 출력도 사라지므로, 본학습은 Terminal에서 `nohup`(또는 `tmux`)으로 백그라운드 실행하는
편이 낫다. 이렇게 하면 브라우저나 SSH 연결과 무관하게 진행되고 로그도 파일로 남는다.

```bash
cd Visual-Intelligence-Learning-SSL-DINOFORCE-TV
source .venv/bin/activate
mkdir -p logs

nohup python -u scripts/pretrain.py --config configs/mocov3_vits8.yaml \
  > logs/pretrain_mocov3_vits8.out 2>&1 &

tail -f logs/pretrain_mocov3_vits8.out   # 진행 로그
nvidia-smi                               # GPU 사용률 / VRAM
```

프로세스가 중단되거나 서버가 재부팅돼도 체크포인트가 `outputs/`에 남아 있어 이어서 돌릴 수 있다.

```bash
nohup python -u scripts/pretrain.py --config configs/mocov3_vits8.yaml \
  --resume outputs/mocov3_vits8_seed42/ckpt_ep<N>.pth \
  > logs/pretrain_resume.out 2>&1 &
```

노트북으로 작업하려면 `notebooks/server_*.ipynb`를 쓴다 (커널은 `Python (ssl)`).

| 노트북 | 용도 |
|---|---|
| `server_train_universal.ipynb` | CONFIG 한 줄만 바꿔 모든 method 학습 + 추출 + LP 평가 + 시각화 (메인) |
| `server_train_mocov3.ipynb`, `server_train_mocov2.ipynb` | 단일 method 학습 (자동 resume) |
| `server_evaluate.ipynb` | feature 추출 + evaluate.py LP 평가 |
| `server_analyze_training.ipynb` | 학습 로그 파싱·시각화 (loss / feat_std / lr) |

feature 추출과 LP 평가는 몇 분이면 끝나서 Terminal이든 노트북이든 상관없다. 다만 500 epoch
본학습만큼은 Terminal + nohup을 권한다.

## 산출물 형식

```
outputs/mocov3_vits8_seed42/
├── ckpt_ep{N}.pth       전체 state (resume용)
└── backbone_ep{N}.pth   backbone 가중치만 (LP feature 추출용, 최근 N개만 유지)
```

LP 평가에는 `backbone_ep*.pth`만 있으면 된다. `extract_features.py`가 여기서 `(N, D)` feature를
뽑아 고정 recipe인 `evaluate.py`에 넘긴다. `evaluate.py`는 이미지가 아니라 추출된 feature만 받기
때문에 backbone 종류, feature 차원, 추출 해상도는 모두 학습하는 쪽에서 통제할 수 있다.

## 설계 결정

평가가 frozen 상태의 linear probing만으로 이뤄지기 때문에, fine-tuning에 강한 MAE 계열보다
contrastive/momentum 계열이 이 과제에 맞는다고 봤다. 그래서 MoCo v3를 골랐고, ViT의 patch
embedding을 freeze해 학습을 안정화했다(Chen et al. 2021).

backbone은 ViT-S/16에서 ViT-S/8로 바꿨다. cat과 dog처럼 fine-grained한 구분에서 막히는 원인을
공간 분해능 부족으로 보고, patch를 16에서 8로 줄여 token 수를 37에서 145로 늘렸다. 파라미터는
21.4M로 거의 동일해서 100k라는 작은 데이터에서도 과적합 위험 없이 해상도만 4배로 키운 셈이다
(DINO ViT-S/8과 같은 방향). 이 변경으로 STL-10 LP가 86.6에서 88.75로 올랐다.

feature 추출 방식도 하나의 레버로 썼다. backbone은 그대로 두고 "읽는 법"만 바꿔서 LP를 끌어올린
것인데, 최종 채택한 `last6_cls_patchmean`은 마지막 6개 block의 CLS를 모은 depth ensemble과 최심
block patch mean의 locality를 합친 것이다. 어떤 모드를 쓸지는 STL train 내부 validation으로만
정해 test를 미리 보지 않았다.

이외에 backbone, data, augmentation을 공유 라이브러리 하나로 묶어 method 간 비교가 공정하도록
했고, seed를 고정하고 결정론 설정을 켜서 peer-review의 2-seed 재현에 대비했다.

## Baselines

비교용으로 MoCo v2 (ResNet-50), MoCo v2 + multi-crop, BYOL, VICReg 구현을 함께 유지한다
(`configs/`, `ssl_lib/models/`). 최종 제출 경로는 위의 MoCo v3 + ViT-S/8이다.
