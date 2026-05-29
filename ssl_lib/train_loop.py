"""
학습 루프 — 노트북과 CLI 양쪽에서 공유.

설계:
- train_one_epoch: 한 epoch 학습. 노트북에서도 직접 호출 가능.
- pretrain: 전체 학습 entry point. config 받아서 모든 것 빌드.
"""
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.mocov2 import MoCoV2
from .data.stl10_unlabeled import build_stl10_loader
from .utils.seed import set_seed
from .utils.schedulers import build_optimizer, CosineLRScheduler
from .utils.checkpoint import save_checkpoint, cleanup_old_checkpoints
from .utils.logging import setup_logger


def build_model(cfg: dict) -> nn.Module:
    """Config method 키에 따라 SSL 모델 빌드."""
    method = cfg["method"].lower()
    if method == "mocov2":
        return MoCoV2(cfg)
    elif method == "mocov3":
        from .models.mocov3 import MoCoV3
        return MoCoV3(cfg)
    elif method == "mocov2_mc":
        from .models.mocov2_mc import MoCoV2MC
        return MoCoV2MC(cfg)
    elif method == "vicreg":
        from .models.vicreg import VICReg
        return VICReg(cfg)
    else:
        raise ValueError(f"Unknown method: {method}")


def _move_batch_to_device(batch, device, channels_last: bool):
    """배치(tuple of 2 tensors 또는 list of N tensors)를 device로 이동."""
    out = []
    for x in batch:
        x = x.to(device, non_blocking=True)
        if channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        out.append(x)
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: CosineLRScheduler,
    epoch: int,
    total_epochs: int,
    device: torch.device,
    scaler=None,  # torch.amp.GradScaler 또는 None (bf16에선 None)
    log_every: int = 50,
    logger=None,
    momentum_ema: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    grad_clip=None,
) -> dict:
    """
    한 epoch 학습.

    Returns:
        epoch 통계 dict (avg_loss, avg_feature_std 등)
    """
    model.train()
    use_amp = (scaler is not None) or (amp_dtype == torch.bfloat16)

    n_steps = len(loader)
    loss_sum, std_sum = 0.0, 0.0
    t_epoch = time.time()
    current_lr = lr_scheduler.get_lr()

    for step, batch in enumerate(loader):
        crops = _move_batch_to_device(batch, device, channels_last)

        # MoCo v3 등 momentum encoder 모델은 epoch-based EMA(momentum) schedule
        if momentum_ema:
            global_progress = epoch + step / n_steps
            model.set_ema_tau(global_progress, total_epochs)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
            if len(crops) == 2:
                loss, log_dict = model(crops[0], crops[1])
            else:
                loss, log_dict = model(crops)

        # Backward — bf16 path는 GradScaler 불필요
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # MoCo v3: optimizer.step() 후 momentum encoder EMA 업데이트
        if momentum_ema:
            model._update_target()

        # LR schedule step
        current_lr = lr_scheduler.step()

        # Statistics
        loss_sum += log_dict["loss"]
        std_sum += log_dict.get("feature_std", 0.0)

        if logger is not None and (step + 1) % log_every == 0:
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                f"Ep {epoch:3d} | step {step+1:4d}/{n_steps} | "
                f"loss {log_dict['loss']:.4f} | "
                f"feat_std {log_dict.get('feature_std', 0.0):.4f} | "
                f"lr {current_lr:.5f} | "
                f"gpu_mem {gpu_mem:.2f}GB"
            )
    
    elapsed = time.time() - t_epoch
    stats = {
        "epoch": epoch,
        "avg_loss": loss_sum / n_steps,
        "avg_feature_std": std_sum / n_steps,
        "elapsed_sec": elapsed,
        "final_lr": current_lr,
    }
    if logger is not None:
        logger.info(
            f"[Ep {epoch} done] avg_loss={stats['avg_loss']:.4f} | "
            f"avg_feat_std={stats['avg_feature_std']:.4f} | "
            f"time={elapsed:.1f}s"
        )
    return stats


def pretrain(cfg: dict, resume_from: Optional[str] = None) -> None:
    """
    Full pretraining entry point.
    
    노트북에서 호출 예:
        import yaml
        with open("configs/mocov2_r50.yaml") as f:
            cfg = yaml.safe_load(f)
        from ssl_lib.train_loop import pretrain
        pretrain(cfg)
    """
    # 시드 고정
    set_seed(cfg["training"]["seed"])
    
    # 출력 디렉토리
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg["output"].get("log_file", None)
    
    # Logger
    logger = setup_logger(name=cfg["method"], log_file=log_file)
    logger.info(f"=== {cfg['method']} pretraining start ===")
    logger.info(f"Output dir: {output_dir}")
    
    # Device — CUDA_VISIBLE_DEVICES로 격리된 GPU만 보임
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data
    loader = build_stl10_loader(cfg)
    logger.info(f"Dataset: {len(loader.dataset)} images, batch={cfg['training']['batch_size']}, "
                f"steps/epoch={len(loader)}")
    
    # Model
    model = build_model(cfg).to(device)

    # channels_last memory format (R50 conv 10-15% 가속)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        logger.info("channels_last memory format enabled")

    # torch.compile (PyTorch 2.0+)
    if cfg["training"].get("compile", False):
        try:
            model = torch.compile(model, mode=cfg["training"].get("compile_mode", "default"))
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without: {e}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg['method']}, trainable params = {n_params/1e6:.1f}M")

    # Optimizer + LR scheduler (step-based)
    optimizer = build_optimizer(model, cfg)
    total_steps = len(loader) * cfg["schedule"]["epochs"]
    warmup_steps = len(loader) * cfg["schedule"]["warmup_epochs"]
    lr_scheduler = CosineLRScheduler(
        optimizer=optimizer,
        base_lr=cfg["optimizer"]["lr"],
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # AMP scaler — bf16일 경우 GradScaler 불필요.
    amp_dtype_name = str(cfg["training"].get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ("bf16", "bfloat16") else torch.float16
    if cfg["training"]["amp"] and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None
    logger.info(f"AMP dtype: {amp_dtype}, scaler: {scaler is not None}")
    
    # Resume
    start_epoch = 0
    if resume_from is not None:
        from .utils.checkpoint import load_checkpoint
        start_epoch = load_checkpoint(model, resume_from, optimizer)
        # scheduler step_count를 resume 지점으로 복원 (0으로 리셋되면 warmup 재실행됨)
        lr_scheduler.step_count = start_epoch * len(loader)
        correct_lr = lr_scheduler._compute_lr(lr_scheduler.step_count)
        for g, ratio in zip(optimizer.param_groups, lr_scheduler.lr_ratios):
            g["lr"] = correct_lr * ratio
        logger.info(f"Resumed from {resume_from}, starting at epoch {start_epoch}")
        logger.info(f"Scheduler restored to step {lr_scheduler.step_count}, lr={correct_lr:.5f}")
    
    momentum_ema = cfg["method"].lower() == "mocov3"
    grad_clip = cfg["training"].get("grad_clip", None)

    # Training loop
    for epoch in range(start_epoch, cfg["schedule"]["epochs"]):
        stats = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            total_epochs=cfg["schedule"]["epochs"],
            device=device,
            scaler=scaler,
            log_every=cfg["training"]["log_every"],
            logger=logger,
            momentum_ema=momentum_ema,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            grad_clip=grad_clip,
        )
        
        # Checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0 \
                or epoch + 1 == cfg["schedule"]["epochs"]:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                output_dir=str(output_dir),
                extra={"config": cfg, "stats": stats},
            )
            cleanup_old_checkpoints(
                str(output_dir),
                keep_last_n=cfg["training"]["keep_last_n_ckpts"],
            )
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    logger.info(f"=== {cfg['method']} pretraining done ===")
