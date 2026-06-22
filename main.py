#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.classifier.classifier import FishClassifier
from src.detector.detector import FishDetector, Detection
from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the detection & classification pipeline"
    )
    parser.add_argument("input", type=Path, help="Path to the input image")
    parser.add_argument("--id", type=bool, default=False)
    return parser.parse_args()


def draw_detections(img: Image.Image, detection: Detection) -> np.ndarray:
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box, score in zip(detection.boxes, detection.scores):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{score:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


def main():
    args = parse_args()

    img = Image.open(args.input).convert("RGB")

    pipeline = Pipeline.from_weights(
        detector_weights="runs/detect/train-2/weights/best.pt"
    )
    detection = pipeline.run(img)

    if isinstance(detection, Detection):
        print("Detected: {} fishes".format(len(detection.boxes)))
        frame = draw_detections(img, detection)
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
