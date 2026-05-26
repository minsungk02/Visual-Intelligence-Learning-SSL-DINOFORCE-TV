"""
Multi-crop augmentation (DINO/SwAV 스타일, MoCo v2-mc/VICReg-mc 용).

핵심 아이디어:
- 2 global crops (96x96) : 표준 SSL view
- N local crops (32x32)  : CIFAR10 해상도와 동일 → cross-resolution invariance 학습
- 글로벌 view 사이엔 비대칭 blur/solarize (DINO 관행)
- 로컬 view는 약한 blur, solarize 없음

각 crop은 같은 source 이미지에서 독립적으로 sampling.
모델 입장에서 returned list의 [0,1]은 global, [2:]는 local로 가정.

이 모듈은 기존 transforms.py의 TwoViewTransform과 공존 — 기존 baseline은 보존.
"""
from typing import List

import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from .transforms import IMAGENET_MEAN, IMAGENET_STD, GaussianBlur


class _Solarize:
    """일정 확률로 solarize 적용."""

    def __init__(self, p: float = 0.0, threshold: int = 128):
        self.p = p
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p > 0 and torch.rand(1).item() < self.p:
            return ImageOps.solarize(img, threshold=self.threshold)
        return img


def _build_single_crop_pipeline(
    crop_size: int,
    crop_scale,
    color_jitter,
    color_jitter_p: float,
    grayscale_p: float,
    blur_p: float,
    solarize_p: float,
) -> transforms.Compose:
    """한 crop 종류의 augmentation pipeline (Compose) 생성."""
    b, c, s, h = color_jitter
    return transforms.Compose([
        transforms.RandomResizedCrop(size=crop_size, scale=tuple(crop_scale)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(b, c, s, h)],
            p=color_jitter_p,
        ),
        transforms.RandomGrayscale(p=grayscale_p),
        transforms.RandomApply(
            [GaussianBlur(kernel_size=9)],
            p=blur_p,
        ),
        _Solarize(p=solarize_p),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiCropTransform:
    """
    한 source 이미지에서 N개의 crop을 만들어 list로 반환.

    Returns:
        crops: List[Tensor] — 길이 = 2 + n_local_crops
            crops[0], crops[1] : (C, global_size, global_size) — global views
            crops[2:]          : (C, local_size,  local_size ) — local views
    """

    def __init__(
        self,
        global_pipeline_1: transforms.Compose,
        global_pipeline_2: transforms.Compose,
        local_pipeline: transforms.Compose,
        n_local_crops: int,
    ):
        self.global1 = global_pipeline_1
        self.global2 = global_pipeline_2
        self.local = local_pipeline
        self.n_local = n_local_crops

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        crops: List[torch.Tensor] = [self.global1(img), self.global2(img)]
        for _ in range(self.n_local):
            crops.append(self.local(img))
        return crops


def build_multicrop_transform(cfg: dict) -> MultiCropTransform:
    """
    Config로부터 multi-crop transform 생성.

    필수 cfg keys (augmentation 블록):
        global_crop_size      : int  (예: 96)
        local_crop_size       : int  (예: 32)
        n_local_crops         : int  (예: 4)
        global_crop_scale     : [min, max]  (예: [0.4, 1.0])
        local_crop_scale      : [min, max]  (예: [0.08, 0.4])
        color_jitter          : [b, c, s, h]
        color_jitter_p        : float
        grayscale_p           : float
        gaussian_blur_p_global: [p_global1, p_global2]  (예: [1.0, 0.1])
        gaussian_blur_p_local : float
        solarize_p_global2    : float (예: 0.2; global2 only)
    """
    aug = cfg["augmentation"]

    global_size = aug["global_crop_size"]
    local_size = aug["local_crop_size"]
    n_local = aug["n_local_crops"]

    cj = aug["color_jitter"]
    cj_p = aug["color_jitter_p"]
    gs_p = aug["grayscale_p"]
    blur_g = aug["gaussian_blur_p_global"]  # [g1, g2]
    blur_l = aug["gaussian_blur_p_local"]
    sol_g2 = aug.get("solarize_p_global2", 0.0)

    global1 = _build_single_crop_pipeline(
        crop_size=global_size,
        crop_scale=aug["global_crop_scale"],
        color_jitter=cj,
        color_jitter_p=cj_p,
        grayscale_p=gs_p,
        blur_p=blur_g[0],
        solarize_p=0.0,
    )
    global2 = _build_single_crop_pipeline(
        crop_size=global_size,
        crop_scale=aug["global_crop_scale"],
        color_jitter=cj,
        color_jitter_p=cj_p,
        grayscale_p=gs_p,
        blur_p=blur_g[1],
        solarize_p=sol_g2,
    )
    local = _build_single_crop_pipeline(
        crop_size=local_size,
        crop_scale=aug["local_crop_scale"],
        color_jitter=cj,
        color_jitter_p=cj_p,
        grayscale_p=gs_p,
        blur_p=blur_l,
        solarize_p=0.0,
    )

    return MultiCropTransform(global1, global2, local, n_local_crops=n_local)
