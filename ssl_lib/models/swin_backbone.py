"""
Swin Transformer Tiny backbone (timm 기반).

설계 결정:
- timm `swin_tiny_patch4_window7_224` 변형으로 img_size, window_size를 96/3으로 override.
- 96/4=24 patch grid → stage별 spatial 24/12/6/3.
  window_size=3은 모든 stage 나눔 보장 → assertion error 없음.
- num_classes=0 + global_pool='avg' → forward 결과가 (B, 768) feature vector.
- timm은 from-scratch (pretrained=False) 로 random init.

ResNetBackbone과 인터페이스 호환:
- `forward(x)` → (B, feature_dim) tensor
- `feature_dim` 속성 → projection head input_dim으로 사용
"""
from typing import Optional

import torch
import torch.nn as nn


class SwinBackbone(nn.Module):
    """
    Swin Transformer Tiny wrapper (timm).

    Attributes:
        feature_dim: 768 (Swin-T 표준 last-stage embedding × global pool).
    """

    def __init__(
        self,
        name: str = "swin_tiny_patch4_window7_224",
        img_size: int = 96,
        window_size: int = 3,
    ):
        """
        Args:
            name: timm 모델명. Swin-T 기본은 'swin_tiny_patch4_window7_224'.
            img_size: 입력 해상도. STL10 96.
            window_size: window attention 크기.
                96/4=24 patch grid에서 24의 약수(1,2,3,4,6,8,12,24) 중 모든 stage 호환 = 3.
        """
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "Swin backbone requires `timm`. Install with `pip install timm`."
            ) from e

        # weights/pretrained 모두 False — 챌린지 룰: from scratch
        self.net = timm.create_model(
            name,
            pretrained=False,
            img_size=img_size,
            window_size=window_size,
            num_classes=0,        # classifier 제거
            global_pool="avg",    # forward 결과 (B, num_features) flat
        )

        # timm은 num_features 속성에 feature dim 노출
        self.feature_dim: int = int(self.net.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
