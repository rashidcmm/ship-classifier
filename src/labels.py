"""Class names, open-set ("unknown") groups and default settings."""
from pathlib import Path

import yaml

from paths import CONFIG_DIR

# Fixed order: index i in every model output means CLASSES[i] (or known_classes(...)[i] for holdout runs).
CLASSES = ["background", "cargo", "dredger", "fishing", "other", "passengership",
           "pilotvessel", "pleasurecraft", "rescue", "sailing", "tanker", "tug"]


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_groups(path: Path | None = None) -> dict[str, list[str]]:
    with open(path or CONFIG_DIR / "open_set.yaml", encoding="utf-8") as f:
        groups = yaml.safe_load(f)["groups"]
    for name, members in groups.items():
        unknown = set(members) - set(CLASSES)
        if unknown:
            raise ValueError(f"open_set.yaml group {name}: not a class name: {sorted(unknown)}")
    return groups


def known_classes(holdout: str | None) -> list[str]:
    """Classes the model is trained on: all 12, or all minus one held-out group."""
    if holdout is None:
        return list(CLASSES)
    held = set(load_groups()[holdout])
    return [c for c in CLASSES if c not in held]
