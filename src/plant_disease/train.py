"""Fine-tune a pretrained CNN on leaf images.

Training runs in two phases:
  1. backbone frozen, only the new head is trained (fast, avoids wrecking pretrained features)
  2. everything unfrozen with a lower learning rate and cosine decay

    python -m plant_disease.train --data data/plantvillage --epochs 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .data import make_loaders
from .model import BACKBONES, build_model, head_parameters, save_checkpoint, set_backbone_trainable


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch(net, loader, loss_fn, device, optimizer=None, scheduler=None) -> tuple[float, float]:
    training = optimizer is not None
    net.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = net(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def confusion(net, loader, num_classes, device) -> torch.Tensor:
    net.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, labels in loader:
        preds = net(images.to(device)).argmax(1).cpu()
        for t, p in zip(labels, preds):
            matrix[t, p] += 1
    return matrix


def per_class_f1(matrix: torch.Tensor) -> list[float]:
    tp = matrix.diag().float()
    precision = tp / matrix.sum(0).clamp(min=1)
    recall = tp / matrix.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    return [round(v, 4) for v in f1.tolist()]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--backbone", choices=BACKBONES, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=8, help="total epochs (first 2 train only the head)")
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true", help="random init (used in tests / offline)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = pick_device()
    loaders = make_loaders(args.data, args.image_size, args.batch_size, seed=args.seed, workers=args.workers)
    n_classes = len(loaders.classes)
    print(f"device={device}  classes={n_classes}  train={len(loaders.train.dataset)}  val={len(loaders.val.dataset)}")

    net = build_model(n_classes, args.backbone, pretrained=not args.no_pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=loaders.class_weights.to(device), label_smoothing=0.05)
    args.out.mkdir(parents=True, exist_ok=True)
    best_path = args.out / "plant_disease.pt"

    best_acc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        if epoch == 1:
            set_backbone_trainable(net, False)
            optimizer = torch.optim.AdamW(head_parameters(net), lr=args.lr)
            scheduler = None
        if epoch == args.head_epochs + 1:
            set_backbone_trainable(net, True)
            optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr / 10, weight_decay=1e-4)
            steps = (args.epochs - args.head_epochs) * len(loaders.train)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))

        start = time.time()
        train_loss, train_acc = run_epoch(net, loaders.train, loss_fn, device, optimizer, scheduler)
        val_loss, val_acc = run_epoch(net, loaders.val, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        print(
            f"epoch {epoch:>2}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}  ({time.time() - start:.0f}s)"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_path, net, loaders.classes, args.backbone, args.image_size, {"val_acc": val_acc})

    # Final evaluation on the untouched test split, using the best checkpoint.
    best = torch.load(best_path, map_location=device, weights_only=False)
    net.load_state_dict(best["state_dict"])
    matrix = confusion(net, loaders.test, n_classes, device)
    test_acc = (matrix.diag().sum() / matrix.sum().clamp(min=1)).item()
    report = {
        "backbone": args.backbone,
        "best_val_acc": round(best_acc, 4),
        "test_acc": round(test_acc, 4),
        "per_class_f1": dict(zip(loaders.classes, per_class_f1(matrix))),
        "history": history,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"test accuracy {test_acc:.4f} – checkpoint saved to {best_path}")
    return report


if __name__ == "__main__":
    main()
