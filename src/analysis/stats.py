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


def aspect_ratios(records: list[ImageRecord]) -> list[float]:
    """Box aspect ratio (pixel width / pixel height) for every box.

    Normalized w/h is taken back to pixel space via the image resolution, so the
    value reflects the box's true on-screen shape: >1 is a wide (horizontal) box,
    <1 a tall (vertical) one. The spread of this distribution measures orientation
    diversity and informs anchor tuning.
    """
    ratios: list[float] = []
    for record in records:
        for _, _, _, w, h in record.boxes.tolist():
            ratios.append((w * record.width) / (h * record.height))
    return ratios


def box_centers(records: list[ImageRecord]) -> list[tuple[float, float]]:
    """Normalized (cx, cy) center of every box, each coordinate in [0, 1].

    The raw input for the center heatmap: clustering near (0.5, 0.5) means fish
    are always framed dead-center (a spatial bias the detector can overfit to),
    while a spread-out cloud means varied, more realistic placement.
    """
    centers: list[tuple[float, float]] = []
    for record in records:
        for _, cx, cy, _, _ in record.boxes.tolist():
            centers.append((cx, cy))
    return centers


def boxes_per_image(records: list[ImageRecord]) -> list[int]:
    """Number of boxes in each image — one count per record, including 0.

    Measures crowding and occlusion: a long right tail means densely packed
    scenes (more overlap, harder detection). Negatives contribute a 0, so this
    also captures how many images are empty.
    """
    return [record.boxes.shape[0] for record in records]


def negative_rate(records: list[ImageRecord]) -> float:
    """Fraction of images with no boxes, in [0, 1]; 0.0 for an empty dataset.

    High values mean many empty frames, which shifts the training balance toward
    background and can suppress recall if left unmanaged.
    """
    if not records:
        return 0.0
    negatives = sum(1 for record in records if record.boxes.shape[0] == 0)
    return negatives / len(records)


def resolutions(records: list[ImageRecord]) -> list[tuple[int, int]]:
    """(width, height) in pixels for every image — one pair per record.

    The spread of image sizes informs the training `imgsz`: small fish need a
    high enough resolution to survive downsampling, so a dataset of small images
    caps how well distant fish can be detected.
    """
    return [(record.width, record.height) for record in records]


@dataclass
class DatasetStats:
    """All six metrics for one dataset split, bundled for reporting/plotting."""

    num_images: int
    num_boxes: int
    relative_box_areas: list[float]
    aspect_ratios: list[float]
    box_centers: list[tuple[float, float]]
    boxes_per_image: list[int]
    negative_rate: float
    resolutions: list[tuple[int, int]]


def compute_stats(records: list[ImageRecord]) -> DatasetStats:
    """Run every metric over `records` once and bundle them into a DatasetStats."""
    return DatasetStats(
        num_images=len(records),
        num_boxes=sum(r.boxes.shape[0] for r in records),
        relative_box_areas=relative_box_areas(records),
        aspect_ratios=aspect_ratios(records),
        box_centers=box_centers(records),
        boxes_per_image=boxes_per_image(records),
        negative_rate=negative_rate(records),
        resolutions=resolutions(records),
    )
