from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from src.detector.tiling import merge_boxes, tile_grid


#: Magnification for the coarse whole-frame pass, relative to the image's long
#: side. Slightly above native: at 1.0x the detector was erratic on large fish
#: in reef.jpg, while 1.1x reported them whole. Empirical, not derived.
#: FIXME: this is a temporary fix and should be solved inside in the model instead
COARSE_MAGNIFICATION = 1.1


def _coarse_imgsz(image: Image.Image) -> int:
    """Whole-frame inference size, rounded to the stride-32 grid YOLO expects."""
    target = max(image.width, image.height) * COARSE_MAGNIFICATION
    return max(32, round(target / 32) * 32)


@dataclass
class Detection:
    boxes: torch.Tensor        # (N, 4) in xyxy pixel coords
    scores: torch.Tensor       # (N,)
    crops: list[Image.Image] = field(default_factory=list)


class FishDetector:
    def __init__(self, weights: Path | str = "yolo11n.pt"):
        self.model = YOLO(str(weights))

    @property
    def train_imgsz(self) -> int | None:
        """The ``imgsz`` these weights were trained at, if recorded.

        A detector only performs at the object scale it was trained on, so
        callers should default their inference ``imgsz`` to this rather than to
        a hardcoded constant.
        """
        ckpt = getattr(self.model, "ckpt", None) or {}
        return ckpt.get("train_args", {}).get("imgsz")

    def train(
        self,
        data: Path | str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "0",
        **kwargs,
    ) -> None:
        self.model.train(
            data=str(data),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            **kwargs,
        )

    def detect(
        self,
        image: Image.Image,
        conf: float = 0.25,
        imgsz: int = 640,
        max_det: int = 300,
        iou: float = 0.7,
    ) -> Detection:
        results = self.model(
            image, conf=conf, imgsz=imgsz, max_det=max_det, iou=iou, verbose=False
        )[0]
        boxes = results.boxes.xyxy.cpu()
        scores = results.boxes.conf.cpu()
        return self._to_detection(image, boxes, scores)

    def detect_tiled(
        self,
        image: Image.Image,
        tile: int = 512,
        overlap: float = 0.25,
        imgsz: int | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        merge_iou: float = 0.5,
        containment: float | None = 0.8,
        max_det: int = 300,
        include_full_image: bool = True,
        full_image_imgsz: int | None = None,
        batch: int = 4,
    ) -> Detection:
        """Detect on overlapping tiles, then merge back to image coords.

        Each tile is cropped at ``tile`` pixels and rendered at ``imgsz``, so
        objects are magnified by ``imgsz / tile``. That ratio is the knob that
        matters: it lifts small fish into the pixel scale the detector was
        trained on, and its memory cost is set by ``imgsz`` alone rather than by
        the size of the full frame.

        ``include_full_image`` adds one whole-frame pass so fish too large to
        fit in a tile are still found.
        """
        imgsz = tile if imgsz is None else imgsz
        tiles = tile_grid(image.width, image.height, tile=tile, overlap=overlap)

        collected_boxes: list[torch.Tensor] = []
        collected_scores: list[torch.Tensor] = []

        def run(windows, offsets, at_imgsz):
            for start in range(0, len(windows), batch):
                results = self.model(
                    windows[start : start + batch],
                    conf=conf,
                    imgsz=at_imgsz,
                    max_det=max_det,
                    iou=iou,
                    verbose=False,
                )
                for (x0, y0), result in zip(offsets[start : start + batch], results):
                    boxes = result.boxes.xyxy.cpu()
                    if boxes.numel() == 0:
                        continue
                    shift = torch.tensor([x0, y0, x0, y0], dtype=boxes.dtype)
                    collected_boxes.append(boxes + shift)
                    collected_scores.append(result.boxes.conf.cpu())

        run([image.crop(box) for box in tiles], [(t[0], t[1]) for t in tiles], imgsz)

        if include_full_image:
            # The tiles magnify by imgsz/tile, which can push a large fish past
            # the scale the detector learned -- it then reports the tail and the
            # body as separate boxes. This coarse pass sees such a fish whole, so
            # the containment merge can absorb those parts into one box.
            run([image], [(0, 0)], full_image_imgsz or _coarse_imgsz(image))

        if not collected_boxes:
            return Detection(boxes=torch.zeros((0, 4)), scores=torch.zeros((0,)))

        boxes = torch.cat(collected_boxes)
        boxes[:, 0::2].clamp_(0, image.width)
        boxes[:, 1::2].clamp_(0, image.height)
        boxes, scores = merge_boxes(
            boxes,
            torch.cat(collected_scores),
            iou=merge_iou,
            max_det=max_det,
            containment=containment,
        )
        return self._to_detection(image, boxes, scores)

    @staticmethod
    def _to_detection(
        image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor
    ) -> Detection:
        crops = [image.crop(box.tolist()) for box in boxes]
        return Detection(boxes=boxes, scores=scores, crops=crops)
