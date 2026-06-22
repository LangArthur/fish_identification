import torch
from unittest.mock import MagicMock, patch
from PIL import Image

from src.detector.detector import Detection
from src.pipeline import Pipeline, Prediction


def _make_detector_mock(boxes, scores):
    mock = MagicMock()
    crops = [Image.new("RGB", (50, 50)) for _ in range(len(boxes))]
    mock.detect.return_value = Detection(
        boxes=torch.tensor(boxes, dtype=torch.float32),
        scores=torch.tensor(scores, dtype=torch.float32),
        crops=crops,
    )
    return mock


def test_detect_only_returns_detection(sample_image):
    detector = _make_detector_mock([[10.0, 20.0, 100.0, 150.0]], [0.9])
    pipeline = Pipeline(detector=detector)

    result = pipeline.run(sample_image)

    assert isinstance(result, Detection)
    assert result.boxes.shape == (1, 4)
    assert len(result.crops) == 1


def test_detect_only_no_fish(sample_image):
    detector = _make_detector_mock([], [])
    pipeline = Pipeline(detector=detector)

    result = pipeline.run(sample_image)

    assert isinstance(result, Detection)
    assert result.boxes.shape == (0,)
    assert len(result.crops) == 0


def test_with_classifier_returns_prediction(sample_image):
    detector = _make_detector_mock([[10.0, 20.0, 100.0, 150.0]], [0.9])
    classifier = MagicMock()
    classifier.predict.return_value = (3, 0.87)
    pipeline = Pipeline(detector=detector, classifier=classifier)

    result = pipeline.run(sample_image)

    assert isinstance(result, Prediction)
    assert len(result.classifications) == 1
    assert result.classifications[0] == (3, 0.87)


def test_classifier_called_once_per_crop(sample_image):
    detector = _make_detector_mock(
        [[10.0, 20.0, 100.0, 150.0], [200.0, 50.0, 300.0, 180.0]],
        [0.9, 0.7],
    )
    classifier = MagicMock()
    classifier.predict.return_value = (1, 0.5)
    pipeline = Pipeline(detector=detector, classifier=classifier)

    pipeline.run(sample_image)

    assert classifier.predict.call_count == 2


@patch("src.pipeline.FishDetector")
def test_from_weights_detect_only(mock_detector_cls, sample_image):
    mock_detector = _make_detector_mock([[10.0, 20.0, 100.0, 150.0]], [0.9])
    mock_detector_cls.return_value = mock_detector

    pipeline = Pipeline.from_weights(detector_weights="yolo11n.pt")

    assert pipeline.classifier is None
    result = pipeline.run(sample_image)
    assert isinstance(result, Detection)


@patch("src.pipeline.FishDetector")
def test_from_weights_requires_num_classes_for_classifier(mock_detector_cls):
    mock_detector_cls.return_value = MagicMock()

    import pytest
    with pytest.raises(ValueError, match="num_classes"):
        Pipeline.from_weights(
            detector_weights="yolo11n.pt",
            classifier_weights="classifier.pt",
        )
