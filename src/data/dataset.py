import shutil
from enum import Enum
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class DatasetSplit(Enum):
    TRAIN = "train"
    VALID = "valid"


def create_my_deep_fish(src: Path, dst: Path) -> None:
    """Reorganize DeepFish into flat images/{train,valid}/ + labels/{train,valid}/."""
    for split in ("train", "valid"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    for species_dir in src.iterdir():
        if not species_dir.is_dir() or species_dir.name == "Nagative_samples":
            continue
        for split in ("train", "valid"):
            split_dir = species_dir / split
            if not split_dir.exists():
                continue
            for file in split_dir.iterdir():
                if file.suffix == ".jpg":
                    shutil.copy(file, dst / "images" / split / file.name)
                elif file.suffix == ".txt":
                    shutil.copy(file, dst / "labels" / split / file.name)


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
