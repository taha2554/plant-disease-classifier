"""Transfer-learning model: an ImageNet-pretrained backbone with a new classification head."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

BACKBONES = ("mobilenet_v3_small", "resnet18", "efficientnet_b0")


def build_model(num_classes: int, backbone: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    elif backbone == "resnet18":
        net = models.resnet18(weights="DEFAULT" if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {backbone!r}, choose from {BACKBONES}")
    return net


def head_parameters(net: nn.Module) -> list[nn.Parameter]:
    """Parameters of the newly added layer (trained first while the backbone is frozen)."""
    head = getattr(net, "fc", None) or net.classifier[-1]
    return list(head.parameters())


def set_backbone_trainable(net: nn.Module, trainable: bool) -> None:
    head_ids = {id(p) for p in head_parameters(net)}
    for p in net.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def save_checkpoint(path, net: nn.Module, classes: list[str], backbone: str, image_size: int, metrics: dict) -> None:
    torch.save(
        {"state_dict": net.state_dict(), "classes": classes, "backbone": backbone, "image_size": image_size, "metrics": metrics},
        path,
    )


def load_checkpoint(path, device: str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = build_model(len(ckpt["classes"]), ckpt["backbone"], pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net.to(device), ckpt
