import shutil
from collections.abc import Iterator
from enum import Enum
from pathlib import Path

import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DatasetSplit(Enum):
    TRAIN = "train"
    VALID = "valid"


def _copy_label(src: Path, dst: Path, class_id: int | None) -> None:
    """Copy a YOLO label file, optionally rewriting every class id to `class_id`.

    A missing or empty source yields an empty label file (a negative sample).
    """
    lines: list[str] = []
    if src.exists():
        for line in src.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            if class_id is not None:
                parts[0] = str(class_id)
            lines.append(" ".join(parts))
    dst.write_text("".join(f"{line}\n" for line in lines))


def _iter_roboflow(src: Path) -> Iterator[tuple[str, Path, Path]]:
    """Walk a Roboflow/Ultralytics layout: <split>/images + <split>/labels."""
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            yield split, img, src / split / "labels" / f"{img.stem}.txt"


def _iter_deepfish(src: Path) -> Iterator[tuple[str, Path, Path]]:
    """Walk a DeepFish layout: <scene>/{train,valid}/*.jpg with sibling *.txt."""
    for scene in sorted(src.iterdir()):
        if not scene.is_dir() or scene.name == "Nagative_samples":
            continue
        for split in ("train", "valid"):
            split_dir = scene / split
            if not split_dir.is_dir():
                continue
            for img in sorted(split_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                yield split, img, img.with_suffix(".txt")


_WALKERS = {"deepfish": _iter_deepfish, "roboflow": _iter_roboflow}


def _read_names(src: Path) -> list[str]:
    """Best-effort class names from a source data.yaml."""
    data_yaml = src / "data.yaml"
    if data_yaml.exists():
        loaded = yaml.safe_load(data_yaml.read_text()) or {}
        names = loaded.get("names")
        if isinstance(names, dict):
            names = [names[key] for key in sorted(names)]
        if names:
            return list(names)
    return ["fish"]


def _write_data_yaml(dst: Path, names: list[str], splits: set[str]) -> None:
    """Write a data.yaml: dataset path + per-split image dirs + class names."""
    lines = [f"path: {dst.resolve()}"]
    for key, split in (("train", "train"), ("val", "valid"), ("test", "test")):
        if split in splits:
            lines.append(f"{key}: images/{split}")
    lines.append(f"nc: {len(names)}")
    lines.append("names:")
    lines.extend(f"  {i}: {name}" for i, name in enumerate(names))
    (dst / "data.yaml").write_text("\n".join(lines) + "\n")


def convert_dataset(
    src: Path,
    dst: Path,
    fmt: str,
    single_class: bool = True,
) -> None:
    """Normalize a dataset into the Ultralytics layout + data.yaml.

    Produces images/<split>/ + labels/<split>/ under `dst` and a data.yaml.
    `fmt` is one of "deepfish" or "roboflow". With `single_class` (default),
    every box is relabeled to class 0 ("fish").
    otherwise the source class ids and names are kept.
    """
    if fmt not in _WALKERS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {sorted(_WALKERS)}")

    class_id = 0 if single_class else None
    splits: set[str] = set()
    for split, img, label in _WALKERS[fmt](src):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dst / "images" / split / img.name)
        _copy_label(label, dst / "labels" / split / f"{img.stem}.txt", class_id)
        splits.add(split)

    names = ["fish"] if single_class else _read_names(src)
    _write_data_yaml(dst, names, splits)


def _parse_yolo_label(path: Path) -> torch.Tensor:
    """Parse a YOLO label file into a (N, 5) tensor of [class, cx, cy, w, h]."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                rows.append([float(x) for x in parts])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


class DeepFishDataset(Dataset):
    def __init__(
        self,
        split: DatasetSplit,
        img_dir: Path,
        label_dir: Path,
        transform=None,
    ):
        super().__init__()
        self.img_dir = Path(img_dir) / split.value
        self.label_dir = Path(label_dir) / split.value
        self.transform = transform

        self.stems = sorted(p.stem for p in self.img_dir.iterdir() if p.suffix == ".jpg")

        missing = [s for s in self.stems if not (self.label_dir / f"{s}.txt").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image(s) have no matching label file")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        stem = self.stems[index]
        img = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        label = _parse_yolo_label(self.label_dir / f"{stem}.txt")
        if self.transform:
            img = self.transform(img)
        else:
            img = F.to_tensor(img)
        return img, label
