"""
Vision Transformer backbone (timm 기반) — MoCo v3 ViT 전용.

설계 결정:
- timm `vit_small_patch16_224` 변형으로 img_size를 96으로 override.
  96/16 = 6 → 6×6 = 36 patch token + 1 CLS = 37 token (가벼움).
- num_classes=0 → forward 결과가 (B, embed_dim) feature vector.
  - global_pool="token" : CLS token 사용 (MoCo v3 표준, faithful).
  - global_pool="avg"   : patch token 평균 (해상도 변화에 더 강함, 옵션).
- timm은 pretrained=False로 random init (챌린지 룰: from scratch).

★ feature_mode (Phase A — 재학습 0, LP 점수 레버):
  학습된 백본 가중치는 그대로 두고 **feature 읽는 법**만 바꾸는 추출-time 옵션.
  evaluate.py의 LP recipe는 불변이므로 이는 "평가 변경"이 아니라 "feature 추출"이라
  합법 (extract_features.py docstring 명시). 학습 시에는 기본 "cls"만 쓰여 동작 불변.
    cls                 : CLS token (D).  기존 동작 = self.net(x)와 bit-identical.
    avg                 : patch token 평균 (D).  forward_features 후 patch mean.
    cls_patchmean       : CLS ⊕ patch mean (2D).
    last4_cls           : 마지막 4 block의 CLS concat (4D).  DINO linear-eval 표준.
    last4_cls_patchmean : last4_cls ⊕ 최심 block patch mean (5D).
  ※ last4_* 는 .blocks[-4:] forward hook + self.norm 으로 구현 (timm 버전 무관,
    get_intermediate_layers(norm=True)와 결과 bit-identical 검증 완료).

★ frozen patch embed (MoCo v3 핵심 안정화 기법):
  patch_embed.proj(Conv2d)를 학습하지 않고 random init 그대로 고정.
  Chen et al. 2021: 학습 곡선이 부드러워지고 ViT-S 정확도 +1.7%.
  재현성(2-seed 채점)에도 직접적으로 유리 — 불안정한 spike 제거.
  → state_dict에는 그대로 저장되므로 추출 시 동일한 projection 복원됨.

★ gradient checkpointing (24GB급 GPU OOM 방어 — A5000 기준 batch 2048 ~17GB):
  transformer block의 activation을 backward 시 재계산해 메모리를 크게 절약.
  ViT-S/16 @96px symmetric MoCo v3 batch 1024는 단일 L4 24GB에서 빡빡한데,
  grad_checkpoint=True면 activation peak이 depth(12)배 가까이 줄어 여유 확보.
  속도는 ~20-30% 손실(forward 1회 추가 재계산). STL10 100k는 epoch이 짧아
  시간 예산 영향 작음. timm의 set_grad_checkpointing(True) 사용.

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
        feature_dim: 추출 feature dim. feature_mode에 따라 base embed dim(vit_small=384)의
            1/1/2/4/5 배 (cls/avg / cls_patchmean / last4_cls / last4_cls_patchmean).
    """

    # feature_mode → base embed dim 대비 배수 (feature_dim 계산용)
    _FEATURE_MODE_MULT = {
        "cls": 1,
        "avg": 1,
        "cls_patchmean": 2,
        "cls_patchmax": 2,
        "last4_cls": 4,
        "last4_cls_patchmean": 5,
        "last4_cls_patchmean_patchmax": 6,
        "last6_cls_patchmean": 7,
    }

    def __init__(
        self,
        name: str = "vit_small_patch16_224",
        img_size: int = 96,
        global_pool: str = "token",
        freeze_patch_embed: bool = True,
        grad_checkpoint: bool = False,
        feature_mode: str = "cls",
        dynamic_img_size: bool = False,
    ):
        """
        Args:
            name: timm 모델명. ViT-S 기본은 'vit_small_patch16_224'.
            img_size: 입력 해상도. STL10은 96. 96/16=6 → 36 patch token.
            global_pool: "token"(CLS, MoCo v3 표준) | "avg"(patch token 평균).
            freeze_patch_embed: True면 patch_embed.proj를 random init 그대로 고정.
                MoCo v3 ViT 학습 안정화의 핵심. 강력 권장.
            grad_checkpoint: True면 transformer block에 gradient checkpointing 적용.
                24GB급 GPU(A5000 등)에서 대배치 OOM 방어용. 추론(LP 추출) 시에는 무시됨.
            feature_mode: feature 추출 방식 (Phase A LP 레버, 추출-time 전용).
                "cls"(기본)는 학습 동작과 동일 — 학습 시에는 항상 cls만 쓰임.
                나머지는 extract_features.py --feature-mode 로 추출 때만 지정.
                {cls, avg, cls_patchmean, last4_cls, last4_cls_patchmean}.
            dynamic_img_size: True면 timm이 pos_embed를 입력 해상도에 맞게 보간(interpolate).
                img_size(학습=96)와 다른 해상도로 추출할 때만 필요 (Phase A 다음 레버: 128px).
                False(기본)면 timm 기본값과 동일 → 96px 경로는 bit-identical 보존.
                pos_embed 파라미터 자체 shape은 img_size 기준 그대로라 학습 ckpt strict 로드 OK
                (보간은 forward 시점에만 일어남). 추출-time 전용 옵션.
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
            dynamic_img_size=dynamic_img_size,  # ★ True면 pos_embed 보간 → 96px 외 해상도 평가 허용
        )
        self.dynamic_img_size = dynamic_img_size

        # feature_mode 검증 + feature_dim 계산
        if feature_mode not in self._FEATURE_MODE_MULT:
            raise ValueError(
                f"Unknown feature_mode: {feature_mode!r}. "
                f"choices={list(self._FEATURE_MODE_MULT)}"
            )
        self.feature_mode = feature_mode

        # timm은 num_features 속성에 base embedding dim 노출 (vit_small=384)
        self._base_dim: int = int(self.net.num_features)
        self.feature_dim: int = self._base_dim * self._FEATURE_MODE_MULT[feature_mode]
        # CLS/register 등 prefix token 수 (vit_small=1). patch slicing 기준.
        self.num_prefix: int = int(getattr(self.net, "num_prefix_tokens", 1))
        self.global_pool = global_pool

        # ★ frozen random patch projection (MoCo v3)
        self.freeze_patch_embed = freeze_patch_embed
        if freeze_patch_embed:
            for p in self.net.patch_embed.proj.parameters():
                p.requires_grad_(False)

        # ★ gradient checkpointing (OOM 방어). timm 표준 API.
        self.grad_checkpoint = grad_checkpoint
        if grad_checkpoint:
            self.net.set_grad_checkpointing(True)

    def _intermediate_tokens(self, x: torch.Tensor, n: int = 4) -> list:
        """마지막 n개 transformer block 출력에 final norm(self.net.norm)을 적용해 반환.

        .blocks(Sequential) + .norm 에만 의존 → timm 버전 무관(>=0.9) 안정적.
        DINO linear-eval 표준(각 block에 self.norm 적용)과 동일하며,
        get_intermediate_layers(n, return_prefix_tokens=True, norm=True)와
        bit-identical 임을 검증함.

        Returns:
            길이 n 리스트. ascending block 순서(…, 최심 block이 마지막).
            각 원소 shape: (B, 1+P, D) — norm 적용된 전체 token.
        """
        captured: list = []
        handles = [
            blk.register_forward_hook(lambda _m, _i, o: captured.append(o))
            for blk in self.net.blocks[-n:]
        ]
        try:
            self.net.forward_features(x)  # 반환값은 버리고 hook으로 중간 출력만 수집
        finally:
            for h in handles:
                h.remove()
        return [self.net.norm(t) for t in captured]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.feature_mode

        # cls: 학습 동작과 bit-identical (self.net(x) == forward_features(x)[:,0])
        if mode == "cls":
            return self.net(x)

        # forward_features 기반 (이미 self.norm 적용된 token 반환): avg / cls_patchmean / cls_patchmax
        if mode in ("avg", "cls_patchmean", "cls_patchmax"):
            tokens = self.net.forward_features(x)              # (B, 1+P, D)
            cls = tokens[:, 0]
            patch = tokens[:, self.num_prefix:]               # (B, P, D)
            if mode == "avg":
                return patch.mean(dim=1)
            if mode == "cls_patchmean":
                return torch.cat([cls, patch.mean(dim=1)], dim=-1)        # (B, 2D)
            # cls_patchmax: 국소 최대 활성(귀·주둥이 등 fine-grained 단서에 민감)
            return torch.cat([cls, patch.max(dim=1).values], dim=-1)      # (B, 2D)

        # 마지막 n block CLS concat (DINO 표준) 기반:
        #   last4_cls / last4_cls_patchmean / last4_cls_patchmean_patchmax / last6_cls_patchmean
        if mode in ("last4_cls", "last4_cls_patchmean",
                    "last4_cls_patchmean_patchmax", "last6_cls_patchmean"):
            n = 6 if mode == "last6_cls_patchmean" else 4
            toks = self._intermediate_tokens(x, n=n)
            cls_cat = torch.cat([t[:, 0] for t in toks], dim=-1)          # (B, nD)
            if mode == "last4_cls":
                return cls_cat
            deepest_patch = toks[-1][:, self.num_prefix:]                 # 최심 block patch tokens
            if mode in ("last4_cls_patchmean", "last6_cls_patchmean"):
                return torch.cat([cls_cat, deepest_patch.mean(dim=1)], dim=-1)   # (B, (n+1)D)
            # last4_cls_patchmean_patchmax: depth 앙상블 + mean(전체) + max(국소) 상보 결합
            return torch.cat(
                [cls_cat, deepest_patch.mean(dim=1), deepest_patch.max(dim=1).values],
                dim=-1,
            )                                                             # (B, 6D)

        raise ValueError(f"Unknown feature_mode: {mode!r}")
