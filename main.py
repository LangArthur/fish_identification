#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.detector.detector import Detection
from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the detection & classification pipeline"
    )
    parser.add_argument("input", type=Path, help="Path to the input image")
    parser.add_argument("--id", action="store_true", help="Enable fish identification")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/train-5/weights/best.pt"),
        help="Detector checkpoint to run",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference resolution; defaults to the imgsz the weights were "
        "trained at, which is where the detector performs best",
    )
    parser.add_argument(
        "--max-det", type=int, default=300, help="Cap on detections per image"
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help="Tile size in px for sliced inference; each tile is rendered at "
        "--imgsz, so fish are magnified by imgsz/tile. Off by default",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Fraction of a tile shared with its neighbour (needs --tile)",
    )
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

    pipeline = Pipeline.from_weights(detector_weights=args.weights)

    trained_at = pipeline.detector.train_imgsz
    imgsz = args.imgsz if args.imgsz is not None else (trained_at or 640)
    scale = f"{imgsz / args.tile:.2f}x magnification" if args.tile else "no tiling"
    print(f"Weights: {args.weights} (trained at imgsz={trained_at})")
    print(f"Inference: imgsz={imgsz}, tile={args.tile} -> {scale}")

    detection = pipeline.run(
        img,
        conf=args.conf,
        imgsz=imgsz,
        max_det=args.max_det,
        iou=args.iou,
        tile=args.tile,
        overlap=args.overlap,
    )

    if isinstance(detection, Detection):
        print(f"Detected: {len(detection.boxes)} fishes")
        frame = draw_detections(img, detection)
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
