"""
MoCo v2 학습 스크립트 (GPU 0 전용).

프로젝트 루트에서 실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_mocov2.py
    CUDA_VISIBLE_DEVICES=0 python scripts/train_mocov2.py --epochs 5    # sanity check
    CUDA_VISIBLE_DEVICES=0 python scripts/train_mocov2.py --resume outputs/mocov2_r50_seed42/ckpt_ep200.pth

백그라운드 실행 (nohup):
    bash scripts/run_mocov2.sh

예상 소요 시간: 400 epoch ≈ 20~24시간
로그 파일: logs/mocov2_seed42.log
체크포인트: outputs/mocov2_r50_seed42/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
import yaml

from ssl_lib.train_loop import pretrain


def main():
    parser = argparse.ArgumentParser(description='MoCo v2 pretraining on STL10')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume할 checkpoint 경로')
    parser.add_argument('--epochs', type=int, default=None,
                        help='학습 epoch 수 override (미지정 시 config 값 사용)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size override (OOM 시 256으로 줄임)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader worker 수 override (미지정 시 config 값 — 서버는 config 값 권장)')
    parser.add_argument('--save-every', type=int, default=None,
                        help='checkpoint 저장 주기 override (미지정 시 config 값)')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'GPU 사용 불가! CUDA 환경을 확인하세요.'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    with open('configs/mocov2_r50.yaml') as f:
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
