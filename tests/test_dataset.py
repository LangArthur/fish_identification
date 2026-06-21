import pytest
import torch
from pathlib import Path

from src.data.dataset import DeepFishDataset, DatasetSplit


def test_dataset_length(tmp_dataset):
    ds = DeepFishDataset(DatasetSplit.TRAIN, tmp_dataset / "images", tmp_dataset / "labels")
    assert len(ds) == 2


def test_dataset_image_is_rgb_tensor(tmp_dataset):
    ds = DeepFishDataset(DatasetSplit.TRAIN, tmp_dataset / "images", tmp_dataset / "labels")
    img, _ = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3  # RGB channels


def test_dataset_label_shape(tmp_dataset):
    ds = DeepFishDataset(DatasetSplit.TRAIN, tmp_dataset / "images", tmp_dataset / "labels")
    _, label = ds[0]
    assert isinstance(label, torch.Tensor)
    assert label.shape == (1, 5)  # 1 object, [class, cx, cy, w, h]


def test_dataset_label_values_are_normalized(tmp_dataset):
    ds = DeepFishDataset(DatasetSplit.TRAIN, tmp_dataset / "images", tmp_dataset / "labels")
    _, label = ds[0]
    coords = label[:, 1:]  # cx, cy, w, h — all should be in [0, 1]
    assert (coords >= 0).all() and (coords <= 1).all()


def test_dataset_valid_split(tmp_dataset):
    ds = DeepFishDataset(DatasetSplit.VALID, tmp_dataset / "images", tmp_dataset / "labels")
    assert len(ds) == 2


def test_dataset_missing_label_raises(tmp_dataset):
    (tmp_dataset / "labels" / "train" / "fish_000.txt").unlink()
    with pytest.raises(FileNotFoundError):
        DeepFishDataset(DatasetSplit.TRAIN, tmp_dataset / "images", tmp_dataset / "labels")
