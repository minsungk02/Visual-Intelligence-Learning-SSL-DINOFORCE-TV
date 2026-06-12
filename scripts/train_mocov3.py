"""
MoCo v3 (ViT-S/16) 학습 스크립트 (GPU 0 전용).

프로젝트 루트에서 실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_mocov3.py
    CUDA_VISIBLE_DEVICES=0 python scripts/train_mocov3.py --epochs 3     # sanity check
    CUDA_VISIBLE_DEVICES=0 python scripts/train_mocov3.py --resume outputs/mocov3_vits_seed42/ckpt_ep200.pth

Jupyter 서버 실행 (장시간 학습은 Terminal + nohup 권장):
    nohup python -u scripts/train_mocov3.py > logs/train_mocov3.out 2>&1 &
    노트북: notebooks/server_train_mocov3.ipynb

config: configs/mocov3_vits.yaml
체크포인트: outputs/mocov3_vits_seed42/
로그: logs/mocov3_vits_seed42.log

★ 첫 실행 시 1 epoch 소요시간을 확인하고 72h 예산 역산:
    예) 1 epoch = T초 → 300 epoch = 300·T/3600 시간.
    72h 초과 시 --epochs 200 (warmup도 config에서 27로) 으로 조정.
"""
import os
import sys
from pathlib import Path

# CUDA 메모리 단편화 방지 — batch 1024 ViT-S에서 OOM 여유 확보.
# torch import 전에 설정해야 효과 있음.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
import yaml

from ssl_lib.train_loop import pretrain


def main():
    parser = argparse.ArgumentParser(description='MoCo v3 (ViT-S) pretraining on STL10')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume할 checkpoint 경로')
    parser.add_argument('--epochs', type=int, default=None,
                        help='학습 epoch 수 override (미지정 시 config 값 사용)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size override (OOM 시 512로 줄임). '
                             '※ 바꾸면 lr도 1.5e-4×batch/256으로 재계산 필요.')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader worker 수 override (미지정 시 config 값 — 서버는 config 값 권장)')
    parser.add_argument('--save-every', type=int, default=None,
                        help='checkpoint 저장 주기 override (미지정 시 config 값)')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'GPU 사용 불가! CUDA 환경을 확인하세요.'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    with open('configs/mocov3_vits.yaml') as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg['schedule']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        cfg['data']['num_workers'] = args.num_workers
        cfg['data']['persistent_workers'] = args.num_workers > 0
    if args.save_every is not None:
        cfg['training']['save_every'] = args.save_every

    pretrain(cfg, resume_from=args.resume)


if __name__ == '__main__':
    main()
