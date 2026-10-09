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


def _fake_result(boxes, scores):
    result = MagicMock()
    result.boxes.xyxy = torch.tensor(boxes).reshape(-1, 4)
    result.boxes.conf = torch.tensor(scores).reshape(-1)
    return result


@patch("src.detector.detector.YOLO")
def test_detect_tiled_offsets_boxes_into_image_coords(mock_yolo):
    image = Image.new("RGB", (1024, 512))
    # 2x1 grid of 512px tiles at overlap 0; each tile reports one box at (10,10)
    mock_yolo.return_value.side_effect = lambda imgs, **kw: [
        _fake_result([[10.0, 10.0, 40.0, 40.0]], [0.9]) for _ in imgs
    ]

    detector = FishDetector("yolo11n.pt")
    detection = detector.detect_tiled(
        image, tile=512, overlap=0.0, include_full_image=False, batch=8
    )

    xs = sorted(box[0].item() for box in detection.boxes)
    assert xs == [10.0, 522.0]  # second tile's box shifted by its x offset


@patch("src.detector.detector.YOLO")
def test_detect_tiled_merges_duplicates_across_overlap(mock_yolo):
    image = Image.new("RGB", (1024, 512))
    # Every tile reports a box at the same global spot -> one fish, not many
    def per_tile(imgs, **kw):
        return [_fake_result([[0.0, 0.0, 30.0, 30.0]], [0.9]) for _ in imgs]

    mock_yolo.return_value.side_effect = per_tile
    detector = FishDetector("yolo11n.pt")
    detection = detector.detect_tiled(
        image, tile=512, overlap=0.5, include_full_image=False, batch=8
    )

    # tiles all report (0,0,30,30) locally; only the tile at x=0 lands on the
    # same global box, so duplicates collapse but distinct offsets survive
    assert len(detection.boxes) == len({tuple(b.tolist()) for b in detection.boxes})


@patch("src.detector.detector.YOLO")
def test_detect_tiled_clamps_boxes_to_image_bounds(mock_yolo):
    image = Image.new("RGB", (1024, 512))
    mock_yolo.return_value.side_effect = lambda imgs, **kw: [
        _fake_result([[500.0, 500.0, 900.0, 900.0]], [0.9]) for _ in imgs
    ]

    detector = FishDetector("yolo11n.pt")
    detection = detector.detect_tiled(
        image, tile=512, overlap=0.0, include_full_image=False, batch=8
    )

    assert detection.boxes[:, 0::2].max().item() <= 1024
    assert detection.boxes[:, 1::2].max().item() <= 512


@patch("src.detector.detector.YOLO")
def test_detect_tiled_no_fish_returns_empty(mock_yolo):
    image = Image.new("RGB", (1024, 512))
    mock_yolo.return_value.side_effect = lambda imgs, **kw: [
        _fake_result([], []) for _ in imgs
    ]

    detector = FishDetector("yolo11n.pt")
    detection = detector.detect_tiled(image, tile=512, include_full_image=False)

    assert detection.boxes.shape == (0, 4)
    assert len(detection.crops) == 0


@patch("src.detector.detector.YOLO")
def test_detect_tiled_renders_tiles_at_imgsz(mock_yolo):
    image = Image.new("RGB", (1024, 512))
    seen = {}

    def capture(imgs, **kw):
        seen["imgsz"] = kw["imgsz"]
        seen["tile_size"] = imgs[0].size
        return [_fake_result([], []) for _ in imgs]

    mock_yolo.return_value.side_effect = capture
    detector = FishDetector("yolo11n.pt")
    detector.detect_tiled(
        image, tile=256, imgsz=1024, overlap=0.0, include_full_image=False, batch=8
    )

    assert seen["tile_size"] == (256, 256)  # cropped at tile size
    assert seen["imgsz"] == 1024           # but rendered 4x magnified


@patch("src.detector.detector.YOLO")
def test_detect_tiled_includes_full_image_pass(mock_yolo):
    image = Image.new("RGB", (1024, 512))
    sizes = []

    def capture(imgs, **kw):
        sizes.extend(img.size for img in imgs)
        return [_fake_result([], []) for _ in imgs]

    mock_yolo.return_value.side_effect = capture
    detector = FishDetector("yolo11n.pt")
    detector.detect_tiled(
        image, tile=512, overlap=0.0, include_full_image=True, batch=8
    )

    assert (1024, 512) in sizes  # the whole frame was passed too


@patch("src.detector.detector.YOLO")
def test_detect_tiled_batches_respect_batch_size(mock_yolo):
    image = Image.new("RGB", (2048, 512))
    calls = []

    def capture(imgs, **kw):
        calls.append(len(imgs))
        return [_fake_result([], []) for _ in imgs]

    mock_yolo.return_value.side_effect = capture
    detector = FishDetector("yolo11n.pt")
    detector.detect_tiled(
        image, tile=512, overlap=0.0, include_full_image=False, batch=2
    )

    assert calls and max(calls) <= 2


@patch("src.detector.detector.YOLO")
def test_train_imgsz_reads_checkpoint(mock_yolo):
    mock_yolo.return_value.ckpt = {"train_args": {"imgsz": 1024}}
    assert FishDetector("best.pt").train_imgsz == 1024


@patch("src.detector.detector.YOLO")
def test_train_imgsz_none_when_unrecorded(mock_yolo):
    mock_yolo.return_value.ckpt = None
    assert FishDetector("yolo11n.pt").train_imgsz is None


def test_coarse_imgsz_is_slightly_above_native_on_the_stride_grid():
    from src.detector.detector import _coarse_imgsz

    assert _coarse_imgsz(Image.new("RGB", (1280, 720))) == 1408  # 1.1x, /32
    assert _coarse_imgsz(Image.new("RGB", (720, 1280))) == 1408  # long side wins
    assert _coarse_imgsz(Image.new("RGB", (1920, 1080))) % 32 == 0


@patch("src.detector.detector.YOLO")
def test_detect_tiled_runs_full_image_pass_at_coarse_imgsz(mock_yolo):
    image = Image.new("RGB", (1280, 720))
    seen = []

    def capture(imgs, **kw):
        seen.extend((img.size, kw["imgsz"]) for img in imgs)
        return [_fake_result([], []) for _ in imgs]

    mock_yolo.return_value.side_effect = capture
    FishDetector("yolo11n.pt").detect_tiled(
        image, tile=640, imgsz=1280, include_full_image=True, batch=8
    )

    tiles = [s for s in seen if s[0] == (640, 640)]
    full = [s for s in seen if s[0] == (1280, 720)]
    assert tiles and all(imgsz == 1280 for _, imgsz in tiles)
    assert full == [((1280, 720), 1408)]  # coarse pass, not the tile imgsz


@patch("src.detector.detector.YOLO")
def test_detect_tiled_full_image_imgsz_overrides_the_default(mock_yolo):
    image = Image.new("RGB", (1280, 720))
    seen = []
    mock_yolo.return_value.side_effect = lambda imgs, **kw: (
        seen.extend((i.size, kw["imgsz"]) for i in imgs)
        or [_fake_result([], []) for _ in imgs]
    )

    FishDetector("yolo11n.pt").detect_tiled(
        image, tile=640, imgsz=1280, full_image_imgsz=1920, batch=8
    )

    assert ((1280, 720), 1920) in seen
