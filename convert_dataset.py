"""CLI to normalize a fish dataset into the standard Ultralytics YOLO layout.

Produces  images/{train,valid[,test]}/  +  labels/{train,valid[,test]}/  +
data.yaml  under the destination, so any source dataset trains the same way.

Usage:
    # DeepFish (scene-grouped) -> single-class detector dataset
    uv run convert_dataset.py --src dataset/Deepfish --dst dataset/my_deep_fish --format deepfish

    # Roboflow export (already split/images + split/labels)
    uv run convert_dataset.py --src dataset/Fish_Detection_v5 --dst dataset/my_fish_detection --format roboflow

By default every box is collapsed to a single "fish" class (class 0) for
Stage-1 detector training. Pass --keep-classes to preserve the original
species classes and names (read from the source data.yaml).
"""

import argparse
from pathlib import Path

from src.data.dataset import convert_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, required=True, help="Path to the source dataset root"
    )
    parser.add_argument(
        "--dst", type=Path, required=True, help="Destination for the normalized dataset"
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["deepfish", "roboflow"],
        help="Source layout: 'deepfish' (scene-grouped) or 'roboflow' (split/images+labels)",
    )
    parser.add_argument(
        "--keep-classes",
        action="store_true",
        help="Preserve original species classes (default collapses all to a single 'fish' class)",
    )
    args = parser.parse_args()

    convert_dataset(
        args.src, args.dst, args.format, single_class=not args.keep_classes
    )
    print(f"Converted {args.src} [{args.format}] -> {args.dst}")
    print(f"  wrote {args.dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
