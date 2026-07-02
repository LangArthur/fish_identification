import math
import random

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from src.analysis.stats import DatasetStats, ImageRecord


def plot_box_size_distribution(stats: DatasetStats) -> Figure:
    """Histogram of relative box areas with p50/p90 percentile markers.

    The headline "far vs close" plot: a distribution crammed against 0 means the
    dataset is dominated by small, distant fish (the root cause of the DeepFish
    failure). The percentile vlines make that skew readable at a glance.
    """
    fig = Figure()
    ax = fig.subplots()

    areas = stats.relative_box_areas
    ax.hist(areas, bins=50, range=(0, 1), color="steelblue", edgecolor="none")

    if areas:
        p50, p90 = np.percentile(areas, [50, 90])
        ax.axvline(p50, color="orange", linestyle="--", label=f"p50 = {p50:.3f}")
        ax.axvline(p90, color="red", linestyle="--", label=f"p90 = {p90:.3f}")
        ax.legend()

    ax.set_xlabel("Relative box area (w x h, fraction of image)")
    ax.set_ylabel("Number of boxes")
    ax.set_title("Box size distribution")
    return fig


def plot_aspect_ratio_distribution(stats: DatasetStats) -> Figure:
    """Histogram of box aspect ratios (pixel w/h) with a reference line at 1.0.

    Ratios cluster at 1.0 for square boxes; the vline makes the wide (>1) vs tall
    (<1) split readable at a glance, so a distribution skewed to one side flags a
    dominant fish orientation worth accounting for in anchor tuning.
    """
    fig = Figure()
    ax = fig.subplots()

    ratios = stats.aspect_ratios
    ax.hist(ratios, bins=50, color="steelblue", edgecolor="none")
    ax.axvline(1.0, color="orange", linestyle="--", label="square (w/h = 1)")
    ax.legend()

    ax.set_xlabel("Aspect ratio (pixel width / pixel height)")
    ax.set_ylabel("Number of boxes")
    ax.set_title("Aspect ratio distribution")
    return fig


def plot_center_heatmap(stats: DatasetStats) -> Figure:
    """2D density map of box centers over the [0, 1] x [0, 1] image frame.

    A hot blob at the middle means fish are always framed dead-center (a spatial
    bias the detector can overfit to); an even spread means varied placement. The
    y-axis is inverted so (0, 0) is top-left, matching image coordinates.
    """
    fig = Figure()
    ax = fig.subplots()

    centers = stats.box_centers
    xs = [cx for cx, _ in centers]
    ys = [cy for _, cy in centers]
    ax.hist2d(xs, ys, bins=50, range=[[0, 1], [0, 1]], cmap="inferno")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # invert y so origin is top-left, like an image
    ax.set_xlabel("Box center x (normalized)")
    ax.set_ylabel("Box center y (normalized)")
    ax.set_title("Box center heatmap")
    return fig


def plot_boxes_per_image_distribution(stats: DatasetStats) -> Figure:
    """Histogram of box counts per image, with 0 (negatives) as its own bin.

    A long right tail means densely packed scenes (more overlap, harder
    detection); a tall bar at 0 means many empty frames. Bins are integer-aligned
    so each count lands on its own bar and the negative bin is always visible.
    """
    fig = Figure()
    ax = fig.subplots()

    counts = stats.boxes_per_image
    hi = max(counts) if counts else 0
    # one bin per integer count from 0..hi (edges at -0.5, 0.5, ... center bars)
    bins = np.arange(-0.5, hi + 1.5, 1.0)
    ax.hist(counts, bins=bins, color="steelblue", edgecolor="white")

    ax.set_xlabel("Boxes per image")
    ax.set_ylabel("Number of images")
    ax.set_title("Boxes per image distribution")
    return fig


def plot_sample_grid(records: list[ImageRecord], n: int = 16, seed: int = 0) -> Figure:
    """Grid of up to `n` random images with their YOLO boxes drawn in pixels.

    The eyeball check on top of the numbers: it shows whether boxes actually land
    on fish and how the far-vs-close skew looks in real frames. Normalized
    (cx, cy, w, h) labels are denormalized to the image size to draw each box.
    Fewer than `n` records fills a smaller grid; a negative (0-box) image shows
    plainly with no boxes.
    """
    k = min(n, len(records))
    fig = Figure(figsize=(12, 12))
    if k == 0:
        return fig  # nothing to show; caller gets an empty canvas

    sample = random.Random(seed).sample(records, k)
    ncols = math.ceil(math.sqrt(k))
    nrows = math.ceil(k / ncols)
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()

    for ax, record in zip(axes, sample):
        with Image.open(record.path) as im:
            ax.imshow(im)
        for _, cx, cy, w, h in record.boxes.tolist():
            px, py = cx * record.width, cy * record.height
            pw, ph = w * record.width, h * record.height
            ax.add_patch(
                Rectangle(
                    (px - pw / 2, py - ph / 2),
                    pw,
                    ph,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.5,
                )
            )
        ax.set_axis_off()

    # blank any leftover cells so partial grids stay clean
    for ax in axes[k:]:
        ax.set_axis_off()

    fig.suptitle("Sample images with boxes")
    return fig
