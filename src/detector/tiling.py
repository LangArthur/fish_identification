"""Geometry and merge helpers for tiled (sliced) inference.

Full-image inference resizes the whole frame to ``imgsz``, so fish that are
small *relative to the frame* land below the pixel scale the detector was
trained on. Tiling runs the detector on overlapping crops instead: each tile is
rendered at ``imgsz``, magnifying objects by ``imgsz / tile`` without ever
holding a full upscaled frame in GPU memory.
"""

import torch
from torchvision.ops import nms

# (x0, y0, x1, y1) crop box in pixel coords
TileBox = tuple[int, int, int, int]


def _axis_offsets(length: int, tile: int, stride: int) -> list[int]:
    """Tile start positions along one axis, always covering the far edge.

    The last offset is snapped back to ``length - tile`` rather than padding a
    partial tile, so every tile is full size and the edge is never cropped off.
    """
    if length <= tile:
        return [0]
    offsets = list(range(0, length - tile + 1, stride))
    if offsets[-1] != length - tile:
        offsets.append(length - tile)
    return offsets


def tile_grid(
    width: int,
    height: int,
    tile: int = 640,
    overlap: float = 0.25,
) -> list[TileBox]:
    """Overlapping tile boxes covering a ``width`` x ``height`` image.

    ``overlap`` is the fraction of a tile shared with its neighbour. A fish
    fully inside the overlap band is whole in at least one tile, so keep
    ``tile * overlap`` comfortably above the largest fish you expect.
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    stride = max(1, round(tile * (1.0 - overlap)))
    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in _axis_offsets(height, tile, stride)
        for x in _axis_offsets(width, tile, stride)
    ]


def _suppress_contained(
    boxes: torch.Tensor, order: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Drop boxes mostly swallowed by a larger one, by IoS.

    Two things produce a part-of-a-fish box: a tile that clips a fish at its
    edge, and over-magnification, where a fish rendered past the detector's
    learned scale makes the heads fire on its tail or body alone. Either way
    the part sits below the IoU threshold against the whole fish and survives
    plain NMS, so compare by intersection-over-smaller instead, which is ~1.0
    for that pair.

    Processed largest-first rather than highest-scoring-first: the whole fish
    is the box worth keeping, and a part detection routinely outscores it.
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = order[areas[order].argsort(descending=True)]

    kept: list[int] = []
    for idx in order.tolist():
        box = boxes[idx]
        if kept:
            others = boxes[kept]
            lt = torch.max(others[:, :2], box[:2])
            rb = torch.min(others[:, 2:], box[2:])
            inter = (rb - lt).clamp(min=0).prod(dim=1)
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_others = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
            ios = inter / torch.minimum(area_others, area_box).clamp(min=1e-6)
            if bool((ios > threshold).any()):
                continue
        kept.append(idx)
    return torch.tensor(kept, dtype=torch.long)


def merge_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou: float = 0.5,
    max_det: int = 300,
    containment: float | None = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NMS over boxes pooled from every tile, to drop overlap duplicates.

    ``containment`` adds a second pass that also removes boxes contained in a
    surviving one (see :func:`_suppress_contained`). Set it to ``None`` to use
    plain IoU NMS -- worth doing on heavily occluded scenes, where a small fish
    genuinely in front of a large one would otherwise be suppressed.
    """
    if boxes.numel() == 0:
        return boxes.reshape(0, 4), scores.reshape(0)
    keep = nms(boxes, scores, iou)
    if containment is not None:
        keep = _suppress_contained(boxes, keep, containment)
        keep = keep[scores[keep].argsort(descending=True)]
    keep = keep[:max_det]
    return boxes[keep], scores[keep]
