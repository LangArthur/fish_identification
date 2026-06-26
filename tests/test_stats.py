import torch

from src.analysis.stats import ImageRecord, load_records


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
