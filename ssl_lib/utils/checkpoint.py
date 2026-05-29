"""
Checkpoint 저장/로드.

설계:
- backbone state_dict는 항상 별도 저장 (evaluate.py에서 쉽게 로드 가능하도록).
- 전체 학습 state도 저장 (resume용).
"""
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    output_dir: str,
    extra: Optional[dict] = None,
    save_backbone_only: bool = False,
) -> str:
    """
    체크포인트 저장.
    
    두 종류 파일 생성:
        - ckpt_ep{epoch}.pth          : 전체 학습 state (resume용)
        - backbone_ep{epoch}.pth      : backbone state_dict만 (LP 평가용)
    
    Args:
        model: SSL 모델 (MoCoV2 / MoCoV3 등). model.backbone 속성이 있어야 함.
        optimizer: optimizer (resume용). None이면 skip.
        epoch: 현재 epoch.
        output_dir: 저장 경로.
        extra: 추가로 저장할 dict (예: {"lr_scheduler_state": ..., "config": ...})
        save_backbone_only: True면 backbone만 저장 (디스크 절약).
    
    Returns:
        저장된 backbone 파일 경로.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Backbone만 저장 — LP 평가용
    backbone_path = output_dir / f"backbone_ep{epoch}.pth"
    torch.save(
        {
            "epoch": epoch,
            "backbone_state_dict": model.backbone.state_dict(),
        },
        backbone_path,
    )
    
    if not save_backbone_only:
        # 전체 학습 state — resume용
        full_path = output_dir / f"ckpt_ep{epoch}.pth"
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if extra is not None:
            state.update(extra)
        torch.save(state, full_path)
    
    return str(backbone_path)


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """
    체크포인트 로드. Resume용.
    
    Returns:
        다음 시작 epoch (저장된 epoch + 1).
    """
    # weights_only=False — ckpt에 config dict(extra)가 들어있어 PyTorch 2.6+의
    # weights_only=True 기본값에서 로드가 막힐 수 있음. 자체 생성 체크포인트라 안전.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "backbone_state_dict" in state:
        # backbone만 저장된 경우
        model.backbone.load_state_dict(state["backbone_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    
    return state.get("epoch", 0) + 1


def cleanup_old_checkpoints(output_dir: str, keep_last_n: int = 3) -> None:
    """
    오래된 체크포인트 정리. 디스크 절약용.
    
    Args:
        output_dir: 체크포인트 디렉토리.
        keep_last_n: 유지할 최근 체크포인트 개수.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    
    # ckpt_ep{N}.pth 패턴 파일 모음
    ckpts = sorted(
        output_dir.glob("ckpt_ep*.pth"),
        key=lambda p: int(p.stem.replace("ckpt_ep", "")),
    )
    if len(ckpts) > keep_last_n:
        for old in ckpts[:-keep_last_n]:
            old.unlink()
    
    # backbone도 동일하게
    backbones = sorted(
        output_dir.glob("backbone_ep*.pth"),
        key=lambda p: int(p.stem.replace("backbone_ep", "")),
    )
    if len(backbones) > keep_last_n:
        for old in backbones[:-keep_last_n]:
            old.unlink()
