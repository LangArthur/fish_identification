from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.classifier.classifier import FishClassifier
from src.detector.detector import Detection, FishDetector


@dataclass
class Prediction:
    detection: Detection
    # parallel to detection.crops: (class_idx, confidence) per crop
    classifications: list[tuple[int, float]]


class Pipeline:
    def __init__(
        self,
        detector: FishDetector,
        classifier: FishClassifier | None = None,
    ):
        self.detector = detector
        self.classifier = classifier

    @classmethod
    def from_weights(
        cls,
        detector_weights: Path | str,
        classifier_weights: Path | None = None,
        num_classes: int | None = None,
        backbone: str = "efficientnet_b0",
    ) -> "Pipeline":
        detector = FishDetector(weights=detector_weights)
        classifier = None
        if classifier_weights is not None:
            if num_classes is None:
                raise ValueError("num_classes is required when loading a classifier")
            classifier = FishClassifier.load(
                path=classifier_weights,
                num_classes=num_classes,
                backbone=backbone,
            )
        return cls(detector=detector, classifier=classifier)

    def run(
        self,
        image: Image.Image,
        conf: float = 0.25,
        imgsz: int = 640,
        max_det: int = 300,
        iou: float = 0.7,
        tile: int | None = None,
        overlap: float = 0.25,
    ) -> Detection | Prediction:
        if tile is not None:
            detection = self.detector.detect_tiled(
                image,
                tile=tile,
                overlap=overlap,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
            )
        else:
            detection = self.detector.detect(
                image, conf=conf, imgsz=imgsz, max_det=max_det, iou=iou
            )
        if self.classifier is None:
            return detection
        classifications = [self.classifier.predict(crop) for crop in detection.crops]
        return Prediction(detection=detection, classifications=classifications)
