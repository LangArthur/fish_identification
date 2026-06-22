import pytest
from PIL import Image


@pytest.fixture
def tmp_dataset(tmp_path):
    """Minimal fake dataset: 2 images + labels per split."""
    for split in ("train", "valid"):
        img_dir = tmp_path / "images" / split
        lbl_dir = tmp_path / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (640, 480), color=(i * 80, 120, 200)).save(
                img_dir / f"fish_{i:03d}.jpg"
            )
            (lbl_dir / f"fish_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    return tmp_path


@pytest.fixture
def sample_image():
    return Image.new("RGB", (640, 480), color=(100, 149, 200))
