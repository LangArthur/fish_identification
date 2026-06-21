import torch
from unittest.mock import MagicMock, patch
from PIL import Image

from src.detector.detector import Detection, FishDetector


@patch("src.detector.detector.YOLO")
def test_detector_instantiation(mock_yolo):
    FishDetector("yolo11n.pt")
    mock_yolo.assert_called_once_with("yolo11n.pt")


@patch("src.detector.detector.YOLO")
def test_detect_returns_detection(mock_yolo, sample_image):
    mock_result = MagicMock()
    mock_result.boxes.xyxy = torch.tensor([[10.0, 20.0, 100.0, 150.0]])
    mock_result.boxes.conf = torch.tensor([0.9])
    mock_yolo.return_value.return_value = [mock_result]

    detector = FishDetector("yolo11n.pt")
    detection = detector.detect(sample_image)

    assert isinstance(detection, Detection)
    assert detection.boxes.shape == (1, 4)
    assert detection.scores.shape == (1,)
    assert len(detection.crops) == 1


@patch("src.detector.detector.YOLO")
def test_detect_crops_are_pil_images(mock_yolo, sample_image):
    mock_result = MagicMock()
    mock_result.boxes.xyxy = torch.tensor([[10.0, 20.0, 100.0, 150.0]])
    mock_result.boxes.conf = torch.tensor([0.9])
    mock_yolo.return_value.return_value = [mock_result]

    detector = FishDetector("yolo11n.pt")
    detection = detector.detect(sample_image)

    assert isinstance(detection.crops[0], Image.Image)


@patch("src.detector.detector.YOLO")
def test_detect_no_fish_returns_empty(mock_yolo, sample_image):
    mock_result = MagicMock()
    mock_result.boxes.xyxy = torch.zeros((0, 4))
    mock_result.boxes.conf = torch.zeros((0,))
    mock_yolo.return_value.return_value = [mock_result]

    detector = FishDetector("yolo11n.pt")
    detection = detector.detect(sample_image)

    assert detection.boxes.shape == (0, 4)
    assert len(detection.crops) == 0
