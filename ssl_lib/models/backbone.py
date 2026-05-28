"""
Backbone (encoder) 모듈.

torchvision의 ResNet을 wrap해서 SSL용으로 만든다.
- 마지막 fc layer 제거 (avgpool 출력만 반환)
- small_image=True일 때 첫 conv를 3x3/stride1, maxpool 제거
  (CIFAR-100 32×32 LP eval까지 고려해 stride=1 유지)
- gradient_checkpoint=True일 때 layer1/layer2에 gradient checkpointing 적용
  → backprop activation을 재계산으로 대체해 메모리 ~6-7 GB 절약

해상도별 layer4 직전 spatial size:
  stride=1, no maxpool | STL10 96×96 → 12×12 | CIFAR-100 32×32 → 4×4  ✓
  stride=2, no maxpool | STL10 96×96 →  6×6  | CIFAR-100 32×32 → 2×2  △

MoCo v2와 BYOL이 모두 같은 backbone 클래스를 import해서 공정 비교.
"""
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18, resnet34, resnet50


BackboneName = Literal["resnet18", "resnet34", "resnet50"]


class ResNetBackbone(nn.Module):
    """
    ResNet backbone wrapper.

    Attributes:
        feature_dim: avgpool 출력 차원 (resnet18/34 → 512, resnet50 → 2048).
    """

    def __init__(
        self,
        name: BackboneName = "resnet50",
        small_image: bool = True,
        zero_init_residual: bool = True,
        gradient_checkpoint: bool = False,
        gc_mode: str = "all",
    ):
        """
        Args:
            name: "resnet18" | "resnet34" | "resnet50"
            small_image: True면 첫 conv를 3x3/stride1, maxpool 제거.
                CIFAR-100 32×32 LP eval 시 4×4 spatial 보존 (stride=2면 2×2로 줄어듦).
                ⚠️ evaluate.py가 원본 ResNet 구조를 가정한다면 False로 둬야 함.
            zero_init_residual: ResNet의 마지막 BN을 0으로 초기화 (학습 안정성).
            gradient_checkpoint: True면 일부 layer에 gradient checkpointing 적용.
                96×96 multi-crop 학습 시 OOM 방지용.
            gc_mode: "all"    — layer1~layer4 모두 ckpt (메모리 최저, ~30% 속도 손실)
                    "layer3" — layer3만 ckpt (균형, ~10% 손실) — multi-crop 권장
                    "off"    — gradient_checkpoint=False와 동일
        """
        super().__init__()

        builders = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
        }
        if name not in builders:
            raise ValueError(f"Unknown backbone: {name}")

        # weights=None 명시 — pretrained weight 절대 금지 (챌린지 규칙)
        net = builders[name](
            weights=None,
            zero_init_residual=zero_init_residual,
        )

        # 작은 이미지 대응: 첫 conv를 3x3/stride1로 교체 + maxpool 제거
        # stride=1: 96×96 그대로 layer1 진입 → CIFAR-100 32×32 LP eval 시 4×4 보존
        if small_image:
            net.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            net.maxpool = nn.Identity()

        # 마지막 fc 제거 — feature dim 보존
        self.feature_dim = net.fc.in_features
        net.fc = nn.Identity()

        self.net = net
        self.use_gc = gradient_checkpoint
        self.gc_mode = gc_mode if gradient_checkpoint else "off"
        self.small_image = small_image

    def _ckpt(self, layer, x, layer_name: str):
        """현재 gc_mode에 따라 해당 layer에만 checkpointing 적용."""
        if not self.use_gc or not x.requires_grad:
            return layer(x)
        if self.gc_mode == "all":
            return checkpoint(layer, x, use_reentrant=False)
        if self.gc_mode == "layer3" and layer_name == "layer3":
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.small_image:
            # small_image 모드: 수동 forward로 layer별 checkpointing 제어.
            # gc_mode="layer3": layer3만 ckpt — 메모리/속도 균형 (multi-crop 권장).
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)      # Identity
            x = self._ckpt(self.net.layer1, x, "layer1")
            x = self._ckpt(self.net.layer2, x, "layer2")
            x = self._ckpt(self.net.layer3, x, "layer3")
            x = self._ckpt(self.net.layer4, x, "layer4")
            x = self.net.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        return self.net(x)


def build_backbone(cfg: dict) -> nn.Module:
    """
    Config dict로부터 backbone 빌드. ResNet 계열 + Swin Transformer 지원.

    필수 cfg keys:
        cfg["backbone"]["name"]:
            "resnet18" | "resnet34" | "resnet50"  → ResNetBackbone
            "swin_*"  (예: swin_tiny)             → SwinBackbone (timm 사용)
        cfg["backbone"]["small_image"]          : bool (ResNet 전용)
        cfg["backbone"]["gradient_checkpoint"]  : bool (ResNet 전용, optional)
        cfg["backbone"]["timm_name"]            : timm 모델명 (Swin 전용, optional)
        cfg["backbone"]["img_size"]             : int (Swin 전용, default 96)
        cfg["backbone"]["window_size"]          : int (Swin 전용, default 3)
    """
    bb_cfg = cfg["backbone"]
    name = bb_cfg["name"].lower()

    if name.startswith("swin") or name.startswith("vit"):
        from .swin_backbone import SwinBackbone
        return SwinBackbone(
            name=bb_cfg.get("timm_name", "swin_tiny_patch4_window7_224"),
            img_size=bb_cfg.get("img_size", 96),
            window_size=bb_cfg.get("window_size", 3),
        )

    # ResNet 경로
    return ResNetBackbone(
        name=bb_cfg["name"],
        small_image=bb_cfg.get("small_image", True),
        zero_init_residual=bb_cfg.get("zero_init_residual", True),
        gradient_checkpoint=bb_cfg.get("gradient_checkpoint", False),
        gc_mode=bb_cfg.get("gc_mode", "all"),
    )
