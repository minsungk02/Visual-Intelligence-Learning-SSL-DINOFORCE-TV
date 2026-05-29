"""
SSL pretrain 완료 후 STL10 / CIFAR10 feature 추출 스크립트.

ResNet / ViT / Swin backbone 모두 지원 (build_backbone 사용).

사용법:
    python scripts/extract_features.py \
        --backbone outputs/mocov3_vits_seed42/backbone_ep300.pth \
        --config   configs/mocov3_vits.yaml \
        --output-dir features/mocov3_ep300 \
        --normalize standardize

결과물:
    features/<out>/
    ├── stl10_train_features.npy   (5000,  D)
    ├── stl10_train_labels.npy     (5000,)
    ├── stl10_test_features.npy    (8000,  D)
    ├── stl10_test_labels.npy      (8000,)
    ├── cifar10_train_features.npy (50000, D)
    ├── cifar10_train_labels.npy   (50000,)
    ├── cifar10_test_features.npy  (10000, D)
    └── cifar10_test_labels.npy    (10000,)

추출 완료 후 evaluate.py 실행 명령이 자동으로 출력됩니다.

★ --normalize (LP 점수 레버):
  evaluate.py의 LP recipe는 lr=0.1 고정 + weight_decay=0 + feature 정규화 없음.
  따라서 feature 스케일이 점수에 직접 영향. 추출 단계 정규화는 "평가 변경"이 아니라
  "feature 추출"이라 합법이며, 재현 검증 때도 동일 코드로 재현됨.
    none        : 원본 feature 그대로 (기존 ResNet 동작, 기본값)
    l2          : 샘플별 L2 정규화 (contrastive feature에 흔함)
    standardize : 차원별 z-score. train 통계로 mean/std 계산해 train+test에 적용.
                  (ViT + lr=0.1 LP에 권장)
  ※ 어떤 방식을 쓸지는 결과 보기 전에 미리 확정하고 레포에 박아둘 것.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ssl_lib.models.backbone import build_backbone, ResNetBackbone

# pretraining과 동일한 normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats_list, labels_list = [], []
    total = len(loader.dataset)
    done = 0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = backbone(imgs).cpu()
        feats_list.append(feats)
        labels_list.append(lbls)
        done += lbls.size(0)
        print(f'  {done}/{total}', end='\r', flush=True)
    print()
    return torch.cat(feats_list).numpy(), torch.cat(labels_list).numpy()


def apply_normalize(train_x, test_x, mode):
    """train 통계 기준으로 train/test feature 정규화."""
    if mode == 'none':
        return train_x, test_x
    if mode == 'l2':
        def _l2(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            return x / np.clip(n, 1e-8, None)
        return _l2(train_x), _l2(test_x)
    if mode == 'standardize':
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True)
        std = np.clip(std, 1e-6, None)
        return (train_x - mean) / std, (test_x - mean) / std
    raise ValueError(f'Unknown normalize mode: {mode}')


def main():
    parser = argparse.ArgumentParser(description='SSL backbone feature 추출')
    parser.add_argument('--backbone', required=True, help='backbone_ep*.pth 경로')
    parser.add_argument('--config', default=None,
                        help='학습 config YAML (backbone 구조 확인용). '
                             '없으면 resnet50 + small_image=True 기본값 사용.')
    parser.add_argument('--output-dir', default='features',
                        help='feature 저장 디렉토리 (default: features/)')
    parser.add_argument('--data-dir', default='./data',
                        help='STL10 / CIFAR10 데이터 루트 (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=96,
                        help='추출 해상도. backbone 학습 해상도와 일치시킬 것 (default 96).')
    parser.add_argument('--normalize', choices=['none', 'l2', 'standardize'],
                        default='none',
                        help='feature 정규화 방식 (LP 점수 레버). 기본 none.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # backbone 구조 결정 — ResNet / ViT / Swin 모두 build_backbone로 처리
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault('backbone', {})
        cfg['backbone']['gradient_checkpoint'] = False  # 추론 시 불필요 (ViT는 무시)
        backbone = build_backbone(cfg)
    else:
        backbone = ResNetBackbone(name='resnet50', small_image=True, gradient_checkpoint=False)

    ckpt = torch.load(args.backbone, map_location='cpu')
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    backbone = backbone.to(device)
    print(f'backbone loaded  (epoch={ckpt.get("epoch", "?")}, '
          f'feature_dim={backbone.feature_dim}, normalize={args.normalize})')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf = build_transform(args.image_size)

    dataset_builders = [
        ('stl10_train',   lambda: datasets.STL10(args.data_dir,  split='train',  transform=tf, download=True)),
        ('stl10_test',    lambda: datasets.STL10(args.data_dir,  split='test',   transform=tf, download=True)),
        ('cifar10_train', lambda: datasets.CIFAR10(args.data_dir, train=True,     transform=tf, download=True)),
        ('cifar10_test',  lambda: datasets.CIFAR10(args.data_dir, train=False,    transform=tf, download=True)),
    ]

    # 1) 추출
    store = {}
    for name, build_ds in dataset_builders:
        ds = build_ds()
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'), drop_last=False,
        )
        print(f'\n[{name}] {len(ds):,} samples...')
        feats, labels = extract_features(backbone, loader, device)
        store[name] = (feats, labels)
        print(f'  extracted: {feats.shape}')

    # 2) 정규화 (train 통계 기준) + 저장
    for prefix in ('stl10', 'cifar10'):
        tr_x, tr_y = store[f'{prefix}_train']
        te_x, te_y = store[f'{prefix}_test']
        tr_x, te_x = apply_normalize(tr_x, te_x, args.normalize)
        for split, x, y in [('train', tr_x, tr_y), ('test', te_x, te_y)]:
            np.save(output_dir / f'{prefix}_{split}_features.npy', x.astype(np.float32))
            np.save(output_dir / f'{prefix}_{split}_labels.npy', y)
        print(f'[{prefix}] saved train{tr_x.shape} / test{te_x.shape}  →  {output_dir}')

    d = output_dir
    print('\n=== 추출 완료. evaluate.py 실행 명령: ===')
    print(f'python evaluate.py \\')
    print(f'  --stl10-train-features   {d}/stl10_train_features.npy \\')
    print(f'  --stl10-train-labels     {d}/stl10_train_labels.npy \\')
    print(f'  --stl10-test-features    {d}/stl10_test_features.npy \\')
    print(f'  --stl10-test-labels      {d}/stl10_test_labels.npy \\')
    print(f'  --cifar10-train-features {d}/cifar10_train_features.npy \\')
    print(f'  --cifar10-train-labels   {d}/cifar10_train_labels.npy \\')
    print(f'  --cifar10-test-features  {d}/cifar10_test_features.npy \\')
    print(f'  --cifar10-test-labels    {d}/cifar10_test_labels.npy')


if __name__ == '__main__':
    main()
