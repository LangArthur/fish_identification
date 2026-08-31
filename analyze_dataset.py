"""CLI to analyze a YOLO-format dataset and report distribution metrics.

Usage:
    uv run analyze_dataset.py --data dataset/my_deep_fish --split all

Prints a text summary for each metric, then opens every distribution plot as a
 dashboard window. With --output <dir>, also saves that dashboard as dashboard.png in that directory.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from collections import Counter

from src.analysis.stats import DatasetStats, compute_stats, load_records
from src.analysis.visualize import plot_dashboard


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


def _print_report(stats: DatasetStats) -> None:
    """Print the full text summary for every metric in `stats`."""
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


def _save_figure(fig: Figure, output_dir: Path) -> None:
    """Save the dashboard as a single PNG into `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dashboard.png"
    fig.savefig(path, dpi=120)
    print(f"  saved {path}")


def _show_figure(fig: Figure) -> None:
    """Open the dashboard in one interactive window and block until closed.

    The plots are built with the bare Figure() constructor (no pyplot state), so
    the figure is re-homed onto a fresh pyplot-managed window before plt.show().
    """
    import matplotlib.pyplot as plt

    manager = plt.figure().canvas.manager
    manager.canvas.figure = fig
    fig.set_canvas(manager.canvas)
    plt.show()


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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to also save distribution plots as PNGs (skipped if omitted)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot windows (e.g. headless runs); pair with --output",
    )
    args = parser.parse_args()

    records = load_records(args.data, split=args.split)
    stats = compute_stats(records)
    print(
        f"Loaded {stats.num_images} images ({stats.num_boxes} boxes) "
        f"from {args.data} [{args.split}]"
    )

    _print_report(stats)

    fig = plot_dashboard(stats, records)

    if args.output is not None:
        print(f"\nSaving plots to {args.output}")
        print("-" * (16 + len(str(args.output))))
        _save_figure(fig, args.output)

    if not args.no_show:
        _show_figure(fig)


if __name__ == "__main__":
    main()
