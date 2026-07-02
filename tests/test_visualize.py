from matplotlib.figure import Figure

from src.analysis.stats import DatasetStats, load_records
from src.analysis.visualize import (
    plot_aspect_ratio_distribution,
    plot_box_size_distribution,
    plot_boxes_per_image_distribution,
    plot_center_heatmap,
    plot_sample_grid,
)


def _stats(**overrides) -> DatasetStats:
    """A DatasetStats with harmless defaults; override only what a test needs."""
    base = dict(
        num_images=0,
        num_boxes=0,
        relative_box_areas=[],
        aspect_ratios=[],
        box_centers=[],
        boxes_per_image=[],
        negative_rate=0.0,
        resolutions=[],
    )
    base.update(overrides)
    return DatasetStats(**base)


def test_plot_box_size_distribution_returns_figure():
    fig = plot_box_size_distribution(_stats(relative_box_areas=[0.01, 0.05, 0.2, 0.5]))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_box_size_distribution_draws_percentile_lines():
    fig = plot_box_size_distribution(_stats(relative_box_areas=[0.01, 0.05, 0.2, 0.5]))
    ax = fig.axes[0]
    # two axvlines (p50, p90) -> two Line2D artists carrying legend labels
    labels = [line.get_label() for line in ax.get_lines()]
    assert sum(lbl.startswith("p50") for lbl in labels) == 1
    assert sum(lbl.startswith("p90") for lbl in labels) == 1


def test_plot_box_size_distribution_handles_empty():
    fig = plot_box_size_distribution(_stats(relative_box_areas=[]))
    assert isinstance(fig, Figure)
    # no percentile lines when there is nothing to summarize
    assert fig.axes[0].get_lines() == []


def test_plot_aspect_ratio_distribution_returns_figure():
    fig = plot_aspect_ratio_distribution(_stats(aspect_ratios=[0.5, 1.0, 1.5, 2.0]))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_aspect_ratio_distribution_draws_reference_line():
    fig = plot_aspect_ratio_distribution(_stats(aspect_ratios=[0.5, 1.0, 1.5, 2.0]))
    ax = fig.axes[0]
    # reference vline sits at x = 1.0 (square box)
    xs = [line.get_xdata()[0] for line in ax.get_lines()]
    assert 1.0 in xs


def test_plot_aspect_ratio_distribution_handles_empty():
    fig = plot_aspect_ratio_distribution(_stats(aspect_ratios=[]))
    assert isinstance(fig, Figure)
    # reference line is drawn regardless, so the plot is still meaningful when empty
    assert len(fig.axes[0].get_lines()) == 1


def test_plot_center_heatmap_returns_figure():
    fig = plot_center_heatmap(_stats(box_centers=[(0.5, 0.5), (0.2, 0.8), (0.9, 0.1)]))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_center_heatmap_handles_empty():
    fig = plot_center_heatmap(_stats(box_centers=[]))
    assert isinstance(fig, Figure)


def test_plot_boxes_per_image_distribution_returns_figure():
    fig = plot_boxes_per_image_distribution(_stats(boxes_per_image=[0, 1, 1, 3, 5]))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_boxes_per_image_distribution_covers_negatives():
    # the leftmost bin edge reaches below 0 so the negative (0-box) count is shown
    fig = plot_boxes_per_image_distribution(_stats(boxes_per_image=[0, 0, 2]))
    ax = fig.axes[0]
    left_edges = [patch.get_x() for patch in ax.patches]
    assert min(left_edges) <= 0.0


def test_plot_boxes_per_image_distribution_handles_empty():
    fig = plot_boxes_per_image_distribution(_stats(boxes_per_image=[]))
    assert isinstance(fig, Figure)


def test_plot_sample_grid_returns_figure(tmp_dataset):
    records = load_records(tmp_dataset, split="all")  # 4 images, 1 box each
    fig = plot_sample_grid(records, n=4)
    assert isinstance(fig, Figure)
    # each sampled image draws its single box as a Rectangle patch
    total_patches = sum(len(ax.patches) for ax in fig.axes)
    assert total_patches == 4


def test_plot_sample_grid_fewer_than_n(tmp_dataset):
    records = load_records(tmp_dataset, split="train")  # only 2 images
    fig = plot_sample_grid(records, n=16)
    assert isinstance(fig, Figure)
    # never draws more image cells than there are records
    assert sum(ax.has_data() for ax in fig.axes) == 2


def test_plot_sample_grid_handles_negatives(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    records[0].boxes = records[0].boxes[:0]  # make one image a negative (0 boxes)
    fig = plot_sample_grid(records, n=2)
    assert isinstance(fig, Figure)
    # negative image contributes no Rectangle; the other still has its one box
    assert sum(len(ax.patches) for ax in fig.axes) == 1


def test_plot_sample_grid_handles_empty():
    fig = plot_sample_grid([], n=16)
    assert isinstance(fig, Figure)
