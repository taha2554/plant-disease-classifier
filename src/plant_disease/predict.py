"""Predict the disease shown on a leaf photo.

    python -m plant_disease.predict leaf.jpg another.jpg --top 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .data import eval_transform, pretty_label
from .model import load_checkpoint


class LeafClassifier:
    def __init__(self, checkpoint: Path | str = "models/plant_disease.pt", device: str = "cpu"):
        self.device = device
        self.net, ckpt = load_checkpoint(checkpoint, device)
        self.classes: list[str] = ckpt["classes"]
        self.transform = eval_transform(ckpt["image_size"])

    @torch.no_grad()
    def predict(self, image: Image.Image, top: int = 3) -> list[tuple[str, float]]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.net(x), dim=1)[0]
        values, indices = probs.topk(min(top, len(self.classes)))
        return [(self.classes[i], round(v.item(), 4)) for v, i in zip(values, indices)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/plant_disease.pt"))
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args(argv)

    clf = LeafClassifier(args.checkpoint)
    for path in args.images:
        print(path)
        for label, prob in clf.predict(Image.open(path), args.top):
            print(f"  {prob:6.1%}  {pretty_label(label)}")


if __name__ == "__main__":
    main()
