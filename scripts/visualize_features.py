"""
SSL feature 시각화 — Linear Probing 결과를 눈으로 검증.

추출된 feature(.npy)만으로 (backbone 불필요):
  1. t-SNE 2D 산점도 (클래스별 색) — 표현 품질 한눈에. 잘 뭉치면 좋은 표현.
  2. confusion matrix — 어떤 클래스를 서로 헷갈리는지.
  3. 클래스별 정확도 막대.
  4. 샘플 예측 그리드 (실제 이미지 + 정답/예측) — --data-dir 줬을 때만.

LP head는 evaluate.py와 동일 recipe(SGD lr0.1 mom0.9 wd0, cosine 100ep, bs128)로
재학습해 예측을 뽑으므로, evaluate.py의 Top-1과 거의 동일한 값이 재현됩니다.

사용:
  python scripts/visualize_features.py \
    --features-dir features/mocov3_vits_ep1000 \
    --dataset stl10 \
    --data-dir ./data \
    --output-dir viz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # 파일 저장 백엔드 (headless 서버용 — png 저장 후 노트북에서 display)
import matplotlib.pyplot as plt

# torchvision STL10 / CIFAR10 클래스 순서 (라벨 인덱스 = 이 순서)
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load_split(feat_dir, prefix, split):
    x = np.load(f"{feat_dir}/{prefix}_{split}_features.npy")
    y = np.load(f"{feat_dir}/{prefix}_{split}_labels.npy").reshape(-1)
    return x.astype(np.float32), y.astype(np.int64)


def train_lp(tr_x, tr_y, te_x, device, epochs=100, bs=128, seed=42):
    """evaluate.py와 동일 recipe로 LP head 학습 후 test 예측 반환."""
    torch.manual_seed(seed)
    D = tr_x.shape[1]
    C = int(tr_y.max()) + 1
    head = nn.Linear(D, C).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    Xtr = torch.tensor(tr_x, device=device)
    Ytr = torch.tensor(tr_y, device=device)
    n = Xtr.shape[0]
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(head(Xtr[idx]), Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()

    head.eval()
    with torch.no_grad():
        pred = head(torch.tensor(te_x, device=device)).argmax(1).cpu().numpy()
    return pred


def plot_tsne(te_x, te_y, classes, out_png, n=3000, title=""):
    from sklearn.manifold import TSNE
    n = min(n, te_x.shape[0])
    rng = np.random.RandomState(42)
    sel = rng.choice(te_x.shape[0], n, replace=False)
    print(f"  t-SNE on {n} test samples...")
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=42).fit_transform(te_x[sel])
    plt.figure(figsize=(9, 8))
    for c in range(len(classes)):
        m = te_y[sel] == c
        plt.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, label=classes[c])
    plt.legend(markerscale=2, fontsize=9, loc="best")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"  saved {out_png.name}")


def plot_confusion(te_y, pred, classes, out_png, acc):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(te_y, pred)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    plt.figure(figsize=(8, 7))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cmn[i, j]
            if v > 0.01:
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.5 else "black", fontsize=7)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.title(f"Confusion matrix (Top-1 {acc:.1f}%)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"  saved {out_png.name}")


def plot_perclass(te_y, pred, classes, out_png, acc):
    pc = [(pred[te_y == c] == c).mean() * 100 for c in range(len(classes))]
    plt.figure(figsize=(9, 4))
    bars = plt.bar(range(len(classes)), pc, color="steelblue")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.axhline(acc, color="red", ls="--", label=f"overall {acc:.1f}%")
    for b, v in zip(bars, pc):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                 ha="center", fontsize=8)
    plt.legend()
    plt.title("Per-class accuracy")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"  saved {out_png.name}")


def plot_samples(dataset, data_dir, te_y, pred, classes, out_png):
    """실제 이미지 + 정답/예측. 위 2줄=맞은 것, 아래 2줄=틀린 것."""
    from torchvision import datasets
    if dataset == "stl10":
        ds = datasets.STL10(data_dir, split="test", download=True)
    else:
        ds = datasets.CIFAR10(data_dir, train=False, download=True)
    # extract_features와 동일하게 shuffle=False였으므로 feature[i] ↔ ds[i] 매칭.
    correct = np.where(pred == te_y)[0]
    wrong = np.where(pred != te_y)[0]
    rng = np.random.RandomState(0)
    cs = rng.choice(correct, min(8, len(correct)), replace=False)
    ws = rng.choice(wrong, min(8, len(wrong)), replace=False) if len(wrong) else []
    sel = list(cs) + list(ws)

    fig, axes = plt.subplots(4, 4, figsize=(11, 12))
    for ax, i in zip(axes.flat, sel):
        img, _ = ds[int(i)]  # PIL.Image
        ax.imshow(img)
        ax.axis("off")
        ok = pred[i] == te_y[i]
        ax.set_title(f"T: {classes[te_y[i]]}\nP: {classes[pred[i]]}",
                     color="green" if ok else "red", fontsize=9)
    # 남는 칸 끄기
    for ax in axes.flat[len(sel):]:
        ax.axis("off")
    plt.suptitle("Sample predictions (top 2 rows: correct / bottom 2: wrong)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"  saved {out_png.name}")


def main():
    ap = argparse.ArgumentParser(description="SSL feature 시각화")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--dataset", choices=["stl10", "cifar10"], required=True)
    ap.add_argument("--data-dir", default=None,
                    help="raw 이미지 경로 (샘플 그리드용). 없으면 그리드 스킵.")
    ap.add_argument("--output-dir", default="viz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tsne-n", type=int, default=3000,
                    help="t-SNE 샘플 수 (속도용, default 3000)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    classes = STL10_CLASSES if args.dataset == "stl10" else CIFAR10_CLASSES
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}, dataset: {args.dataset}")

    tr_x, tr_y = load_split(args.features_dir, args.dataset, "train")
    te_x, te_y = load_split(args.features_dir, args.dataset, "test")
    print(f"train {tr_x.shape}, test {te_x.shape}")

    # 1) t-SNE (표현 품질)
    plot_tsne(te_x, te_y, classes, out / f"{args.dataset}_tsne.png",
              n=args.tsne_n,
              title=f"{args.dataset} test features — t-SNE (MoCo v3 ViT-S)")

    # 2) LP 재현 → 예측
    print("LP head 학습 (evaluate.py 동일 recipe)...")
    pred = train_lp(tr_x, tr_y, te_x, device)
    acc = float((pred == te_y).mean() * 100)
    print(f"  재현 Top-1: {acc:.2f}%")

    # 3) confusion matrix
    plot_confusion(te_y, pred, classes, out / f"{args.dataset}_confusion.png", acc)

    # 4) per-class accuracy
    plot_perclass(te_y, pred, classes, out / f"{args.dataset}_perclass.png", acc)

    # 5) 샘플 예측 그리드 (이미지 있을 때만)
    if args.data_dir:
        try:
            plot_samples(args.dataset, args.data_dir, te_y, pred, classes,
                         out / f"{args.dataset}_samples.png")
        except Exception as e:
            print(f"  [경고] 샘플 그리드 스킵: {e}")
    else:
        print("  (--data-dir 미지정 → 샘플 그리드 스킵)")

    print(f"\n완료. '{out}/' 에 png 저장됨. 재현 Top-1 = {acc:.2f}%")


if __name__ == "__main__":
    main()
