"""
Vision Transformer backbone (timm 기반) — MoCo v3 ViT 전용.

설계 결정:
- timm `vit_small_patch16_224` 변형으로 img_size를 96으로 override.
  96/16 = 6 → 6×6 = 36 patch token + 1 CLS = 37 token (가벼움).
- num_classes=0 → forward 결과가 (B, embed_dim) feature vector.
  - global_pool="token" : CLS token 사용 (MoCo v3 표준, faithful).
  - global_pool="avg"   : patch token 평균 (해상도 변화에 더 강함, 옵션).
- timm은 pretrained=False로 random init (챌린지 룰: from scratch).

★ frozen patch embed (MoCo v3 핵심 안정화 기법):
  patch_embed.proj(Conv2d)를 학습하지 않고 random init 그대로 고정.
  Chen et al. 2021: 학습 곡선이 부드러워지고 ViT-S 정확도 +1.7%.
  재현성(2-seed 채점)에도 직접적으로 유리 — 불안정한 spike 제거.
  → state_dict에는 그대로 저장되므로 추출 시 동일한 projection 복원됨.

ResNetBackbone / SwinBackbone과 인터페이스 호환:
- forward(x) -> (B, feature_dim) tensor
- feature_dim 속성 -> projection head input_dim으로 사용
"""
import torch
import torch.nn as nn


class ViTBackbone(nn.Module):
    """
    Vision Transformer wrapper (timm).

    Attributes:
        feature_dim: embedding dim (vit_small -> 384, vit_base -> 768).
    """

    def __init__(
        self,
        name: str = "vit_small_patch16_224",
        img_size: int = 96,
        global_pool: str = "token",
        freeze_patch_embed: bool = True,
    ):
        """
        Args:
            name: timm 모델명. ViT-S 기본은 'vit_small_patch16_224'.
            img_size: 입력 해상도. STL10은 96. 96/16=6 → 36 patch token.
            global_pool: "token"(CLS, MoCo v3 표준) | "avg"(patch token 평균).
            freeze_patch_embed: True면 patch_embed.proj를 random init 그대로 고정.
                MoCo v3 ViT 학습 안정화의 핵심. 강력 권장.
        """
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "ViT backbone requires `timm`. Install with `pip install timm`."
            ) from e

        # pretrained=False — 챌린지 룰: from scratch
        self.net = timm.create_model(
            name,
            pretrained=False,
            img_size=img_size,
            num_classes=0,           # classifier 제거
            global_pool=global_pool, # "token" -> CLS, "avg" -> patch mean
        )

        # timm은 num_features 속성에 embedding dim 노출
        self.feature_dim: int = int(self.net.num_features)
        self.global_pool = global_pool

        # ★ frozen random patch projection (MoCo v3)
        self.freeze_patch_embed = freeze_patch_embed
        if freeze_patch_embed:
            for p in self.net.patch_embed.proj.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
