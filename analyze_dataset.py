"""CLI to analyze a YOLO-format dataset and report distribution metrics.

Usage:
    uv run analyze_dataset.py --data dataset/my_deep_fish --split all

Prints a text summary for each metric. Plots (via src/analysis/visualize.py)
will be added once that module exists; metrics are wired in here as they land.
"""

import argparse
from pathlib import Path

import numpy as np

from collections import Counter

from src.analysis.stats import compute_stats, load_records


def _summarize(name: str, values: list[float]) -> None:
    """Print count + key percentiles for one metric's distribution."""
    print(f"\n{name}")
    print("-" * len(name))
    if not values:
        print("  (no values)")
        return
    arr = np.asarray(values, dtype=float)
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    print(f"  count : {arr.size}")
    print(f"  min   : {arr.min():.4f}")
    print(f"  p10   : {p10:.4f}")
    print(f"  median: {p50:.4f}")
    print(f"  p90   : {p90:.4f}")
    print(f"  max   : {arr.max():.4f}")
    print(f"  mean  : {arr.mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to YOLO-format dataset root"
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "valid", "all"],
        help="Which split to analyze",
    )
    args = parser.parse_args()

    records = load_records(args.data, split=args.split)
    stats = compute_stats(records)
    print(
        f"Loaded {stats.num_images} images ({stats.num_boxes} boxes) "
        f"from {args.data} [{args.split}]"
    )

    _summarize("Relative box area (w * h)", stats.relative_box_areas)
    _summarize("Box aspect ratio (px w / px h)", stats.aspect_ratios)
    _summarize("Boxes per image", [float(n) for n in stats.boxes_per_image])

    print("\nNegative sample rate")
    print("--------------------")
    print(f"  {stats.negative_rate:.4f} ({stats.negative_rate * 100:.1f}% empty images)")

    print("\nBox center (cx, cy)")
    print("-------------------")
    if stats.box_centers:
        mean_cx = sum(cx for cx, _ in stats.box_centers) / len(stats.box_centers)
        mean_cy = sum(cy for _, cy in stats.box_centers) / len(stats.box_centers)
        print(f"  mean: ({mean_cx:.4f}, {mean_cy:.4f})")
    else:
        print("  (no boxes)")

    print("\nImage resolution (w x h)")
    print("------------------------")
    if stats.resolutions:
        for (w, h), count in Counter(stats.resolutions).most_common():
            print(f"  {w} x {h}: {count} images")
    else:
        print("  (no images)")


if __name__ == "__main__":
    main()
