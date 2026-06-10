"""
Phase-A 추출 스윕 + TRAIN-INTERNAL VAL 셀렉터 (test 누수 0).

목적:
  여러 추출 후보(checkpoint × feature_mode × normalize)를 비교하되,
  **선택은 오직 STL10 train 내부 val(4k-fit / 1k-val)** 로만 하고,
  **test(STL test 8k / CIFAR test 10k)는 val 1위 후보 1개에만** 적용한다.
  → "5개 중 test 최고 줍줍"(test-set selection) 회피. 채점/재현 방어 가능.

LP recipe:
  evaluate.py의 run_feature_linear_probe()를 그대로 import → val proxy가 실제 채점
  recipe(SGD lr=0.1, mom=0.9, wd=0, cosine 100ep, batch 128)와 100% 동일.

normalize 누수 방지:
  standardize는 항상 "fit 셋 통계"로만 mean/std를 구해 fit+eval에 적용.
  (val 단계: fit=train4k 통계 → val 적용 / 최종: fit=full train 통계 → test 적용)

사용 예 (Colab):
  !python scripts/sweep_select.py \
    --config configs/mocov3_vits8.yaml \
    --ckpt-ep500 outputs/mocov3_vits8_seed42/backbone_ep500.pth \
    --ckpt-ep480 outputs/mocov3_vits8_seed42/backbone_ep480.pth \
    --final-output-dir features/sweep_winner

출력:
  - 후보별 STL val 정확도 표
  - val 1위(winner) + 그 winner의 최종 STL test / CIFAR test
  - winner feature(.npy) 저장 + 공식 evaluate.py 재현 명령
"""
import argparse
import contextlib
import copy
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ssl_lib.models.backbone import build_backbone
# evaluate.py의 고정 LP recipe를 그대로 재사용 (val proxy == 실제 채점)
from evaluate import run_feature_linear_probe, feature_set_seed, EPOCHS, BATCH_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_backbone(cfg_path, ckpt_path, feature_mode, device):
    """config 기반으로 ViT backbone 빌드 + 지정 feature_mode + ckpt 로드 (96px, dynamic off)."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("backbone", {})
    cfg["backbone"]["gradient_checkpoint"] = False
    cfg["backbone"]["dynamic_img_size"] = False
    cfg["backbone"]["feature_mode"] = feature_mode
    bb = build_backbone(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    bb.load_state_dict(ckpt["backbone_state_dict"])
    return bb.to(device).eval()


@torch.no_grad()
def extract(backbone, loader, device):
    feats, labels = [], []
    for imgs, lbls in loader:
        feats.append(backbone(imgs.to(device, non_blocking=True)).cpu())
        labels.append(lbls)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy().reshape(-1)


def normalize_fit_eval(fit_x, eval_x, mode):
    """fit 셋 통계로만 정규화 (val/test 누수 방지)."""
    if mode == "none":
        return fit_x, eval_x
    if mode == "l2":
        def _l2(x):
            return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
        return _l2(fit_x), _l2(eval_x)
    if mode == "standardize":
        m = fit_x.mean(axis=0, keepdims=True)
        s = np.clip(fit_x.std(axis=0, keepdims=True), 1e-6, None)
        return (fit_x - m) / s, (eval_x - m) / s
    raise ValueError(f"Unknown normalize: {mode}")


def lp_accuracy(fit_x, fit_y, eval_x, eval_y, device, seed, quiet=True):
    """evaluate.py 고정 recipe로 LP 학습 후 eval 정확도 반환 (per-epoch 출력 억제)."""
    feature_set_seed(seed)  # Linear head init 재현
    tx = torch.from_numpy(fit_x.astype(np.float32))
    ty = torch.from_numpy(fit_y.astype(np.int64))
    ex = torch.from_numpy(eval_x.astype(np.float32))
    ey = torch.from_numpy(eval_y.astype(np.int64))
    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        acc = run_feature_linear_probe(
            "lp", tx, ty, ex, ey, device, EPOCHS, BATCH_SIZE, num_workers=0, seed=seed,
        )
    return acc


def main():
    ap = argparse.ArgumentParser(description="추출 스윕 + train-internal val 셀렉터")
    ap.add_argument("--config", default="configs/mocov3_vits8.yaml")
    ap.add_argument("--ckpt-ep500", required=True, help="주 백본 (ep500) 경로")
    ap.add_argument("--ckpt-ep480", default=None, help="(선택) ep480 백본 — 있으면 후보에 추가")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--val-size", type=int, default=1000, help="STL train 5k 중 val로 뗄 개수")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--lp-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256, help="추출 forward batch")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--final-output-dir", default="features/sweep_winner")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- 후보 정의: (name, ckpt, feature_mode, normalize) ----
    # row 0 = 현재 챔피언(참조). 나머지 = 대안 5개.
    cands = [
        ("ref_ep500_last4cpm_std",       args.ckpt_ep500, "last4_cls_patchmean",          "standardize"),
        ("ep500_last6cpm_std",           args.ckpt_ep500, "last6_cls_patchmean",          "standardize"),
        ("ep500_last4cpm_l2",            args.ckpt_ep500, "last4_cls_patchmean",          "l2"),
        ("ep500_cls_patchmax_std",       args.ckpt_ep500, "cls_patchmax",                 "standardize"),
        ("ep500_last4cpm_patchmax_std",  args.ckpt_ep500, "last4_cls_patchmean_patchmax", "standardize"),
    ]
    if args.ckpt_ep480:
        cands.append(("ep480_last4cpm_std", args.ckpt_ep480, "last4_cls_patchmean", "standardize"))

    tf = build_transform(96)
    stl_train = datasets.STL10(args.data_dir, split="train", transform=tf, download=True)
    train_loader = DataLoader(stl_train, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # 고정 4k/1k split (모든 후보 동일 인덱스 → 공정 비교)
    n = len(stl_train)
    perm = np.random.default_rng(args.split_seed).permutation(n)
    val_idx, fit_idx = perm[:args.val_size], perm[args.val_size:]
    print(f"STL train split: fit={len(fit_idx)} / val={len(val_idx)} (seed={args.split_seed})\n")

    # ---- Phase 1: 후보별 STL val 정확도 ----
    print("=" * 60)
    print("PHASE 1 — STL train-internal validation (선택은 여기서만)")
    print("=" * 60)
    results = []
    for name, ckpt, mode, norm in cands:
        try:
            bb = make_backbone(args.config, ckpt, mode, device)
            feats, labels = extract(bb, train_loader, device)
            del bb
            if device.type == "cuda":
                torch.cuda.empty_cache()
            fit_x, val_x = feats[fit_idx], feats[val_idx]
            fit_y, val_y = labels[fit_idx], labels[val_idx]
            fit_x, val_x = normalize_fit_eval(fit_x, val_x, norm)
            acc = lp_accuracy(fit_x, fit_y, val_x, val_y, device, args.lp_seed)
            results.append((name, ckpt, mode, norm, feats.shape[1], acc))
            print(f"  {name:32s} dim={feats.shape[1]:5d}  STL_val={acc:.2f}")
        except Exception as e:
            print(f"  {name:32s} SKIPPED ({type(e).__name__}: {str(e)[:60]})")

    if not results:
        print("후보 추출 전부 실패. 경로/환경 확인 필요."); return

    # ---- 선택: STL val 1위 (사전등록 규칙) ----
    results.sort(key=lambda r: r[-1], reverse=True)
    print("\n" + "-" * 60)
    print("VAL 순위 (STL val 기준):")
    for r in results:
        print(f"  {r[-1]:.2f}  {r[0]:32s} (dim {r[4]}, {r[3]})")
    win_name, win_ckpt, win_mode, win_norm, win_dim, win_val = results[0]
    print(f"\n★ WINNER (val 1위) = {win_name}  | mode={win_mode} norm={win_norm} dim={win_dim} | STL_val={win_val:.2f}")
    print("-" * 60)

    # ---- Phase 2: winner만 최종 test (STL test 8k / CIFAR test 10k) ----
    print("\n" + "=" * 60)
    print("PHASE 2 — WINNER만 최종 test 평가 (test 누수 0)")
    print("=" * 60)
    out = Path(args.final_output_dir); out.mkdir(parents=True, exist_ok=True)
    bb = make_backbone(args.config, win_ckpt, win_mode, device)

    sets = {
        "stl10_train":   datasets.STL10(args.data_dir, split="train", transform=tf, download=True),
        "stl10_test":    datasets.STL10(args.data_dir, split="test",  transform=tf, download=True),
        "cifar10_train": datasets.CIFAR10(args.data_dir, train=True,  transform=tf, download=True),
        "cifar10_test":  datasets.CIFAR10(args.data_dir, train=False, transform=tf, download=True),
    }
    store = {}
    for nm, ds in sets.items():
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        store[nm] = extract(bb, ld, device)
        print(f"  extracted {nm}: {store[nm][0].shape}")

    final = {}
    for prefix in ("stl10", "cifar10"):
        tr_x, tr_y = store[f"{prefix}_train"]
        te_x, te_y = store[f"{prefix}_test"]
        tr_x, te_x = normalize_fit_eval(tr_x, te_x, win_norm)  # full train 통계로 normalize
        # 저장 (공식 evaluate.py 재현용)
        np.save(out / f"{prefix}_train_features.npy", tr_x.astype(np.float32))
        np.save(out / f"{prefix}_train_labels.npy", tr_y)
        np.save(out / f"{prefix}_test_features.npy", te_x.astype(np.float32))
        np.save(out / f"{prefix}_test_labels.npy", te_y)
        final[prefix] = lp_accuracy(tr_x, tr_y, te_x, te_y, device, args.lp_seed)

    print("\n" + "=" * 60)
    print("최종 결과 (WINNER, test)")
    print("=" * 60)
    print(f"  winner       : {win_name} (mode={win_mode}, norm={win_norm}, dim={win_dim})")
    print(f"  STL_val      : {win_val:.2f}   (선택 근거)")
    print(f"  STL10  test  : {final['stl10']:.2f}")
    print(f"  CIFAR10 test : {final['cifar10']:.2f}")
    print(f"\n  (참조 baseline: STL10 89.67 / CIFAR10 87.00 @ last4_cls_patchmean 96px)")
    print(f"\n  winner feature 저장 → {out}")
    print(f"  공식 evaluate.py 재현 명령:")
    print(f"    python evaluate.py \\")
    for p in ("stl10", "cifar10"):
        for s in ("train", "test"):
            print(f"      --{p}-{s}-features {out}/{p}_{s}_features.npy \\")
            print(f"      --{p}-{s}-labels   {out}/{p}_{s}_labels.npy \\")


if __name__ == "__main__":
    main()
