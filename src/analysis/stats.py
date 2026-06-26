from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import _parse_yolo_label


@dataclass
class ImageRecord:
    """One image and its parsed YOLO labels, the raw input every metric reads."""

    path: Path
    width: int
    height: int
    boxes: torch.Tensor  # (N, 5) of [class, cx, cy, w, h], normalized


def load_records(data_dir: Path | str, split: str = "all") -> list[ImageRecord]:
    """Walk a YOLO-format dataset once and collect one ImageRecord per image.

    Expects the my_deep_fish layout: images/{train,valid}/ + labels/{train,valid}/.
    An image with no matching label file is discarded (unlabeled). A negative
    sample is an image whose label file exists but is empty (0 boxes).
    """
    data_dir = Path(data_dir)
    splits = ["train", "valid"] if split == "all" else [split]

    records: list[ImageRecord] = []
    for sp in splits:
        img_dir = data_dir / "images" / sp
        label_dir = data_dir / "labels" / sp
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue  # unlabeled image — discard
            with Image.open(img_path) as im:
                width, height = im.size
            boxes = _parse_yolo_label(label_path)
            records.append(
                ImageRecord(path=img_path, width=width, height=height, boxes=boxes)
            )
    return records


def relative_box_areas(records: list[ImageRecord]) -> list[float]:
    """Relative box area (w * h) for every box, in [0, 1].

    The core "far vs close" signal: small values dominate when a dataset is
    skewed toward distant fish (the root cause of the DeepFish failure).
    """
    areas: list[float] = []
    for record in records:
        for _, _, _, w, h in record.boxes.tolist():
            areas.append(w * h)
    return areas
