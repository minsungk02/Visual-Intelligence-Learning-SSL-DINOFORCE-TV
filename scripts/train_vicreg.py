"""
VICReg 학습 스크립트 (thin wrapper, train_mocov2.py 패턴).

프로젝트 루트에서 실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_vicreg.py
    CUDA_VISIBLE_DEVICES=0 python scripts/train_vicreg.py --epochs 5    # sanity check
    python scripts/train_vicreg.py --resume outputs/vicreg_r50_seed42/ckpt_ep50.pth

체크포인트: outputs/vicreg_r50_seed42/
로그 파일: logs/vicreg_r50_seed42.log
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
import yaml

from ssl_lib.train_loop import pretrain


def main():
    parser = argparse.ArgumentParser(description="VICReg pretraining on STL10")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU 사용 불가! CUDA 환경 확인."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"BF16 OK: {torch.cuda.is_bf16_supported()}")

    with open("configs/vicreg_r50.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["schedule"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
        cfg["data"]["persistent_workers"] = args.num_workers > 0
    if args.save_every is not None:
        cfg["training"]["save_every"] = args.save_every
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    pretrain(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
