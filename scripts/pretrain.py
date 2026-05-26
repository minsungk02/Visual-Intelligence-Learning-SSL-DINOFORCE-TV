"""
백그라운드 학습용 CLI 스크립트.

사용 예 (GPU 0에서 MoCo v2):
    CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py \
        --config configs/mocov2_r50.yaml \
        --seed 42

노트북에서 학습 시작하고, 안정화된 다음 .py로 옮겨 nohup 백그라운드 돌리는 용도.
"""
import os
import sys
from pathlib import Path

# CUDA 메모리 단편화 방지 — multi-crop 학습 시 OOM 줄임.
# torch import 전에 설정해야 효과 있음.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import yaml

from ssl_lib.train_loop import pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config 경로",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Config의 seed override (없으면 config 값 사용)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume할 checkpoint 경로",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Config의 epochs override",
    )
    args = parser.parse_args()
    
    # Config 로드
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # CLI override
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.epochs is not None:
        cfg["schedule"]["epochs"] = args.epochs
    
    pretrain(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
