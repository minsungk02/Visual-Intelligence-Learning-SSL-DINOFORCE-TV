# SSL Project — STL10 unlabeled SSL pretraining

STL10 unlabeled 100k로 SSL pretraining 후 STL10/CIFAR10 Linear Probing 성능 평가.
주 베이스라인은 **MoCo v3 + ViT-S/16**, 비교 baseline으로 MoCo v2 (ResNet-50)도 유지.

## 디렉토리 구조

```
ssl_project/
├── ssl_lib/             # 공유 라이브러리 (editable install)
│   ├── data/            # STL10 unlabeled, two-view augmentation
│   ├── models/          # backbone (ResNet/ViT/Swin), heads, MoCoV2, MoCoV3, ...
│   ├── utils/           # seed, schedulers, checkpoint, logging
│   └── train_loop.py    # 학습 루프
├── configs/             # YAML configs
├── notebooks/           # 디버깅 + 학습 노트북
├── scripts/             # CLI 진입점 + bash 실행 스크립트
├── outputs/             # 학습 산출물 (체크포인트)
└── logs/                # 학습 로그
```

## 1. 셋업 (한 번만)

```bash
cd ssl_project
pip install -r requirements.txt
pip install -e .          # ssl_lib을 editable mode로 설치
```

## 2. 디버깅 (CPU/GPU 어디서든)

순서대로 실행해서 환경 검증:

1. `notebooks/00_data_check.ipynb` — STL10 다운로드 + augmentation 확인
2. `notebooks/01_backbone_test.ipynb` — backbone forward + feature dim 검증

## 3. 학습 — 두 가지 방법

### 방법 A: CLI 직접 실행 (단일 GPU)

```bash
# MoCo v3 + ViT-S (현재 메인 베이스라인)
CUDA_VISIBLE_DEVICES=0 python scripts/train_mocov3.py

# MoCo v2 + ResNet-50 (비교용 baseline)
CUDA_VISIBLE_DEVICES=0 python scripts/train_mocov2.py
```

### 방법 B: Colab L4 노트북

- `notebooks/colab_train_mocov3.ipynb` (메인)
- `notebooks/colab_train_mocov2.ipynb` (비교)

진행 상황 확인:
```bash
tail -f logs/mocov3_vits_seed42.log
tail -f logs/mocov2_seed42.log
nvidia-smi
```

학습 중단:
```bash
# PID는 run_*.sh 실행 시 출력됨
kill <PID>
```

## 4. Google Colab (L4)에서 학습

코드 수정 없이 Colab에서 바로 실행 가능. 수동으로 파일을 Drive에 올릴 필요 없음.

### 파일 관리 원칙

| 항목 | 저장 위치 | 설명 |
|------|-----------|------|
| 코드 | GitHub | 세션마다 `git pull`로 최신화 |
| STL10 데이터 | Google Drive | 첫 실행 시 자동 다운로드, 이후 재사용 |
| 체크포인트 | Google Drive | 세션 끊겨도 유지 |
| 로그 | Google Drive | 세션 끊겨도 유지 |

Drive 경로: `내 드라이브/ssl_project/{data, outputs, logs}`

### 실행 순서

**처음 실행:**
1. `notebooks/colab_train_mocov3.ipynb` (또는 `colab_train_mocov2.ipynb`) 열기
2. Cell 1 — GPU 확인 (L4 선택 필요: 런타임 → 런타임 유형 변경)
3. Cell 2 — Google Drive 마운트
4. Cell 3 — 레포 클론 + 패키지 설치 + Drive 심링크 연결
5. Cell 4 — 학습 시작 (처음부터 자동 시작)

**세션 재시작 후 (중간에 끊긴 경우):**
- Cell 1~3만 다시 실행
- Cell 4 실행 → Drive에 저장된 마지막 체크포인트에서 자동으로 이어서 시작

### Colab 전용 설정

- `--num-workers 2`: Colab에서 8이면 DataLoader가 불안정
- `--save-every 5`: 5 epoch마다 저장 (세션 끊겨도 최대 ~90분 손실)
- 로컬 서버와 달리 `CUDA_VISIBLE_DEVICES` 설정 불필요 (GPU 1개 환경)

---

## 5. Resume

학습이 중단되면:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py \
    --config configs/mocov2_r50.yaml \
    --resume outputs/mocov2_r50_seed42/ckpt_ep200.pth
```

## 6. 산출물

```
outputs/mocov2_r50_seed42/
├── ckpt_ep{10,20,...,400}.pth        # 전체 학습 state (resume용)
├── backbone_ep{10,20,...,400}.pth    # backbone 가중치만 (LP 평가용)
└── (최근 3개만 유지, 오래된 건 자동 정리)
```

LP 평가에는 `backbone_ep*.pth`만 있으면 됨. evaluate.py가 이 파일의 `backbone_state_dict`를 로드해서 사용.

## 주요 설계 결정

- **공유 라이브러리 + 별도 노트북**: backbone/data/aug 코드가 단일 진실 소스 → 공정 비교 보장.
- **CUDA_VISIBLE_DEVICES로 GPU 격리**: 코드 안에서 device 인자 안 받음. 실수 방지.
- **AMP 필수**: 24GB GPU + 두 인코더 동시 메모리 → mixed precision 없으면 batch size 반토막.
  - MoCo v2: fp16 + GradScaler. MoCo v3 (L4): bf16, GradScaler 불필요.
- **MoCo v2 queue 16384**: STL10 100k에 65536은 과함. 16384가 적절.
- **MoCo v3 in-batch negative**: queue 제거, batch_size=1024로 ~1023 negatives.
- **ViT frozen patch embed**: MoCo v3 ViT 핵심 안정화 (Chen et al. 2021).
- **고정 seed**: 같은 batch order, 같은 augmentation 시퀀스로 재현성 + 공정 비교.
- **checkpoint 매 epoch 저장 + 최근 N개만 유지**: 디스크 절약 + 중단 시 복구 가능.

## 다음 단계

evaluate.py를 받으면:
1. `forward()` 인터페이스 확인 → backbone wrapper 조정 필요시 수정
2. 입력 해상도 확인 → augmentation crop size 조정
3. LP 점수 확인 → 어느 모델/설정이 좋았는지 결정
