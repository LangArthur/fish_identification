#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.detector.detector import FishDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fish detector (Stage 1)")
    parser.add_argument("--data", type=Path, default=Path("detector_data.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = FishDetector(weights=args.weights)
    detector.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
