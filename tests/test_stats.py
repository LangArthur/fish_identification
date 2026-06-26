import pytest
import torch

from src.analysis.stats import (
    DatasetStats,
    ImageRecord,
    aspect_ratios,
    box_centers,
    boxes_per_image,
    compute_stats,
    load_records,
    negative_rate,
    relative_box_areas,
    resolutions,
)


def test_load_records_count_single_split(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    assert len(records) == 2


def test_load_records_all_splits(tmp_dataset):
    records = load_records(tmp_dataset, split="all")
    assert len(records) == 4  # 2 train + 2 valid


def test_load_records_reads_resolution(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    assert records[0].width == 640
    assert records[0].height == 480


def test_load_records_parses_boxes(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    boxes = records[0].boxes
    assert isinstance(boxes, torch.Tensor)
    assert boxes.shape == (1, 5)  # 1 object, [class, cx, cy, w, h]


def test_load_records_missing_label_is_discarded(tmp_dataset):
    (tmp_dataset / "labels" / "train" / "fish_000.txt").unlink()
    records = load_records(tmp_dataset, split="train")
    assert len(records) == 1  # unlabeled image dropped


def test_load_records_empty_label_is_negative(tmp_dataset):
    (tmp_dataset / "labels" / "train" / "fish_000.txt").write_text("")
    records = load_records(tmp_dataset, split="train")
    assert len(records) == 2  # kept
    empties = [r for r in records if r.boxes.shape[0] == 0]
    assert len(empties) == 1  # the emptied label is a negative sample


def test_load_records_missing_split_returns_empty(tmp_dataset):
    records = load_records(tmp_dataset, split="test")  # no such split
    assert records == []


def test_relative_box_areas_one_per_box(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    areas = relative_box_areas(records)
    assert len(areas) == 2  # 2 images, 1 box each
    assert all(a == pytest.approx(0.2 * 0.3) for a in areas)


def test_relative_box_areas_skips_negatives():
    record = ImageRecord(path=None, width=640, height=480, boxes=torch.zeros((0, 5)))
    assert relative_box_areas([record]) == []


def test_aspect_ratios_uses_pixel_dimensions(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    ratios = aspect_ratios(records)
    assert len(ratios) == 2  # 2 images, 1 box each
    # box 0.2 x 0.3 on a 640 x 480 image -> (0.2*640)/(0.3*480)
    assert all(r == pytest.approx((0.2 * 640) / (0.3 * 480)) for r in ratios)


def test_aspect_ratios_skips_negatives():
    record = ImageRecord(path=None, width=640, height=480, boxes=torch.zeros((0, 5)))
    assert aspect_ratios([record]) == []


def test_box_centers_extracts_cx_cy(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    centers = box_centers(records)
    assert len(centers) == 2  # 2 images, 1 box each
    assert all(c == pytest.approx((0.5, 0.5)) for c in centers)


def test_box_centers_skips_negatives():
    record = ImageRecord(path=None, width=640, height=480, boxes=torch.zeros((0, 5)))
    assert box_centers([record]) == []


def test_boxes_per_image_counts_each_record(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    assert boxes_per_image(records) == [1, 1]  # 2 images, 1 box each


def test_boxes_per_image_counts_negatives_as_zero():
    record = ImageRecord(path=None, width=640, height=480, boxes=torch.zeros((0, 5)))
    assert boxes_per_image([record]) == [0]


def test_negative_rate_all_positive(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    assert negative_rate(records) == pytest.approx(0.0)


def test_negative_rate_mixed():
    positive = ImageRecord(path=None, width=640, height=480, boxes=torch.zeros((1, 5)))
    negative = ImageRecord(path=None, width=640, height=480, boxes=torch.zeros((0, 5)))
    assert negative_rate([positive, negative]) == pytest.approx(0.5)


def test_negative_rate_empty_dataset():
    assert negative_rate([]) == pytest.approx(0.0)


def test_resolutions_one_pair_per_image(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    assert resolutions(records) == [(640, 480), (640, 480)]


def test_resolutions_includes_negatives():
    record = ImageRecord(path=None, width=1920, height=1080, boxes=torch.zeros((0, 5)))
    assert resolutions([record]) == [(1920, 1080)]


def test_compute_stats_bundles_all_metrics(tmp_dataset):
    records = load_records(tmp_dataset, split="train")
    stats = compute_stats(records)
    assert isinstance(stats, DatasetStats)
    assert stats.num_images == 2
    assert stats.num_boxes == 2
    assert stats.relative_box_areas == relative_box_areas(records)
    assert stats.aspect_ratios == aspect_ratios(records)
    assert stats.box_centers == box_centers(records)
    assert stats.boxes_per_image == boxes_per_image(records)
    assert stats.negative_rate == negative_rate(records)
    assert stats.resolutions == resolutions(records)


def test_compute_stats_empty_records():
    stats = compute_stats([])
    assert stats.num_images == 0
    assert stats.num_boxes == 0
    assert stats.relative_box_areas == []
    assert stats.negative_rate == pytest.approx(0.0)
