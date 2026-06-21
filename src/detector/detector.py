from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    boxes: torch.Tensor        # (N, 4) in xyxy pixel coords
    scores: torch.Tensor       # (N,)
    crops: list[Image.Image] = field(default_factory=list)


class FishDetector:
    def __init__(self, weights: Path | str = "yolo11n.pt"):
        self.model = YOLO(str(weights))

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

    def detect(self, image: Image.Image, conf: float = 0.25) -> Detection:
        results = self.model(image, conf=conf)[0]
        boxes = results.boxes.xyxy.cpu()
        scores = results.boxes.conf.cpu()
        crops = [image.crop(box.tolist()) for box in boxes]
        return Detection(boxes=boxes, scores=scores, crops=crops)
