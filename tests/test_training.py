"""Smoke tests on a tiny synthetic dataset (no downloads, runs on CPU in seconds)."""

import json
import random

import pytest
import torch
from PIL import Image, ImageDraw

from plant_disease.data import make_loaders, pretty_label, split_indices
from plant_disease.model import build_model, head_parameters, set_backbone_trainable
from plant_disease.predict import LeafClassifier
from plant_disease.train import main as train_main


@pytest.fixture(scope="module")
def fake_dataset(tmp_path_factory):
    """Green 'healthy' leaves vs leaves with brown spots."""
    root = tmp_path_factory.mktemp("leaves")
    rng = random.Random(0)
    for cls in ["Tomato___healthy", "Tomato___Leaf_spot"]:
        (root / cls).mkdir()
        for i in range(12):
            img = Image.new("RGB", (64, 64), (240, 240, 235))
            d = ImageDraw.Draw(img)
            d.ellipse((8, 8, 56, 56), fill=(40, 140 + rng.randint(-20, 20), 50))
            if cls.endswith("spot"):
                for _ in range(6):
                    x, y = rng.randint(14, 46), rng.randint(14, 46)
                    d.ellipse((x, y, x + 6, y + 6), fill=(110, 70, 30))
            img.save(root / cls / f"{i}.png")
    return root


def test_split_is_stratified_and_disjoint():
    targets = [0] * 50 + [1] * 10
    train, val, test = split_indices(targets, val=0.1, test=0.2, seed=1)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    assert len(train) + len(val) + len(test) == 60
    assert sum(targets[i] for i in test) == 2  # 20% of the minority class


def test_pretty_label():
    assert pretty_label("Tomato___Early_blight") == "Tomato – Early blight"
    assert pretty_label("Pepper,_bell___healthy") == "Pepper bell – healthy"


@pytest.mark.parametrize("backbone", ["mobilenet_v3_small", "resnet18"])
def test_model_output_shape_and_freezing(backbone):
    net = build_model(5, backbone, pretrained=False)
    assert net(torch.randn(2, 3, 64, 64)).shape == (2, 5)
    set_backbone_trainable(net, False)
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert len(trainable) == len(head_parameters(net)) == 2  # weight + bias


def test_loaders_and_class_weights(fake_dataset):
    loaders = make_loaders(fake_dataset, image_size=64, batch_size=4, workers=0)
    assert loaders.classes == ["Tomato___Leaf_spot", "Tomato___healthy"]
    images, labels = next(iter(loaders.train))
    assert images.shape[1:] == (3, 64, 64)
    assert torch.allclose(loaders.class_weights, torch.ones(2))


def test_end_to_end_train_and_predict(fake_dataset, tmp_path):
    report = train_main(
        ["--data", str(fake_dataset), "--out", str(tmp_path), "--epochs", "2", "--head-epochs", "1",
         "--image-size", "64", "--batch-size", "4", "--workers", "0", "--no-pretrained", "--backbone", "resnet18"]
    )
    assert (tmp_path / "plant_disease.pt").exists()
    assert 0 <= report["test_acc"] <= 1
    assert json.loads((tmp_path / "report.json").read_text())["backbone"] == "resnet18"

    clf = LeafClassifier(tmp_path / "plant_disease.pt")
    preds = clf.predict(Image.open(next((fake_dataset / "Tomato___healthy").iterdir())), top=2)
    assert len(preds) == 2
    assert abs(sum(p for _, p in preds) - 1) < 1e-3
