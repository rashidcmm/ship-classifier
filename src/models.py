"""The network: a backbone that turns a spectrogram into a feature vector, plus three small output heads.

    spectrogram (1 x 128 x frames) -> backbone -> features -> dropout -> class logits
                                                                      -> distance (km)
                                                                      -> speed (knots / 10)
"""
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless Windows cache warning
import timm
import torch
from torch import nn

BACKBONES = ("efficientnet_b0", "mobilenetv3_large_100", "resnet18", "tiny_cnn")


class PretrainedDownloadError(RuntimeError):
    pass


class TinyCNN(nn.Module):
    """A small CNN written out by hand (lesson 04): 4 x [conv 3x3 -> batch norm -> ReLU -> max pool]."""

    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = c_in

    def forward(self, x):
        return self.pool(self.body(x)).flatten(1)


class ShipNet(nn.Module):
    def __init__(self, backbone: str, n_classes: int, pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; choose one of {BACKBONES}")
        if backbone == "tiny_cnn":
            self.backbone = TinyCNN()
        else:
            try:
                self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=0)
            except Exception as e:
                if not pretrained:
                    raise
                raise PretrainedDownloadError(
                    f"Could not download pretrained weights for {backbone} ({e}).\n"
                    f"Either connect to the internet once, copy the timm cache "
                    f"(~/.cache/huggingface) from a machine that has it, or train with --no-pretrained "
                    f"(slower to learn, usually lower accuracy).") from None
        # timm's head_hidden_size is what forward() returns; num_features is smaller for MobileNetV3 (960 vs 1280)
        dim = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.class_head = nn.Linear(dim, n_classes)
        self.distance_head = nn.Linear(dim, 1)
        self.speed_head = nn.Linear(dim, 1)

    def forward(self, x):
        f = self.backbone(x)
        h = self.dropout(f)
        return {"logits": self.class_head(h), "distance": self.distance_head(h).squeeze(-1),
                "speed": self.speed_head(h).squeeze(-1), "features": f}


def save_checkpoint(path: Path, **state) -> None:
    """Write via a temporary file and rename, so a crash mid-write never leaves a broken checkpoint."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_model(run_dir_or_ckpt: Path, map_location="cpu", which: str = "best") -> tuple[ShipNet, dict]:
    path = Path(run_dir_or_ckpt)
    if path.is_dir():
        path = path / f"{which}.pt"
    # checkpoints hold only tensors and plain values (dicts, lists, numbers, strings), so the safe
    # loader is enough and a tampered file cannot run code
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = ckpt["config"]
    model = ShipNet(cfg["backbone"], len(ckpt["known"]), pretrained=False, drop_rate=cfg.get("drop_rate", 0.3))
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = (np.asarray(ckpt["norm"]["mean"], np.float32), np.asarray(ckpt["norm"]["std"], np.float32))
    return model, {"config": cfg, "known": list(ckpt["known"]), "norm": norm, "checkpoint": ckpt}
