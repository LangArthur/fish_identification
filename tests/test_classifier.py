import torch
import pytest
from PIL import Image

from src.classifier.classifier import FishClassifier


@pytest.fixture
def model():
    return FishClassifier(num_classes=10, backbone="efficientnet_b0", pretrained=False)


def test_classifier_instantiation(model):
    assert model is not None


def test_classifier_forward(model):
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 10)


def test_classifier_predict_returns_valid_types(model, sample_image):
    class_idx, confidence = model.predict(sample_image)
    assert isinstance(class_idx, int)
    assert isinstance(confidence, float)


def test_classifier_predict_confidence_in_range(model, sample_image):
    _, confidence = model.predict(sample_image)
    assert 0.0 <= confidence <= 1.0


def test_classifier_predict_class_in_range(model, sample_image):
    class_idx, _ = model.predict(sample_image)
    assert 0 <= class_idx < 10


def test_classifier_save_load(model, tmp_path):
    path = tmp_path / "classifier.pt"
    model.save(path)
    loaded = FishClassifier.load(path, num_classes=10, backbone="efficientnet_b0")
    for p1, p2 in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(p1, p2)
