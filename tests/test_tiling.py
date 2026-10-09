import pytest
import torch

from src.detector.tiling import merge_boxes, tile_grid


def test_single_tile_when_image_fits():
    assert tile_grid(640, 480, tile=640, overlap=0.25) == [(0, 0, 640, 480)]


def test_tiles_cover_every_pixel():
    width, height, tile = 1280, 720, 512
    tiles = tile_grid(width, height, tile=tile, overlap=0.25)

    covered = torch.zeros((height, width), dtype=torch.bool)
    for x0, y0, x1, y1 in tiles:
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_tiles_stay_inside_the_image():
    for x0, y0, x1, y1 in tile_grid(1280, 720, tile=512, overlap=0.25):
        assert 0 <= x0 < x1 <= 1280
        assert 0 <= y0 < y1 <= 720


def test_last_offset_snaps_back_to_the_edge():
    # 1280 wide, tile 512, stride 384 -> 0, 384, then snapped to 768 (not 1152)
    xs = sorted({t[0] for t in tile_grid(1280, 512, tile=512, overlap=0.25)})
    assert xs == [0, 384, 768]


def test_overlap_zero_gives_contiguous_tiles():
    xs = sorted({t[0] for t in tile_grid(1280, 512, tile=512, overlap=0.0)})
    assert xs == [0, 512, 768]  # 768 is the snapped final tile


def test_more_overlap_yields_more_tiles():
    few = tile_grid(1280, 720, tile=512, overlap=0.1)
    many = tile_grid(1280, 720, tile=512, overlap=0.5)
    assert len(many) > len(few)


@pytest.mark.parametrize("tile", [0, -1])
def test_invalid_tile_rejected(tile):
    with pytest.raises(ValueError, match="tile must be positive"):
        tile_grid(640, 480, tile=tile)


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])
def test_invalid_overlap_rejected(overlap):
    with pytest.raises(ValueError, match=r"overlap must be in \[0, 1\)"):
        tile_grid(640, 480, overlap=overlap)


def test_merge_drops_duplicate_boxes():
    boxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],
            [11.0, 11.0, 51.0, 51.0],  # near-duplicate from the neighbouring tile
            [200.0, 200.0, 260.0, 260.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept_boxes, kept_scores = merge_boxes(boxes, scores, iou=0.5)

    assert kept_boxes.shape == (2, 4)
    assert torch.allclose(kept_boxes[0], boxes[0])  # highest score survives
    assert torch.allclose(kept_scores, torch.tensor([0.9, 0.7]))


def test_merge_keeps_distinct_neighbours():
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [25.0, 25.0, 45.0, 45.0]])
    scores = torch.tensor([0.9, 0.8])
    kept_boxes, _ = merge_boxes(boxes, scores, iou=0.5)
    assert kept_boxes.shape == (2, 4)


def test_merge_respects_max_det():
    boxes = torch.tensor([[float(i) * 100, 0.0, float(i) * 100 + 50, 50.0] for i in range(10)])
    scores = torch.linspace(0.9, 0.1, 10)
    kept_boxes, kept_scores = merge_boxes(boxes, scores, iou=0.5, max_det=3)
    assert kept_boxes.shape == (3, 4)
    assert kept_scores.shape == (3,)


def test_merge_handles_empty_input():
    kept_boxes, kept_scores = merge_boxes(torch.zeros((0, 4)), torch.zeros((0,)))
    assert kept_boxes.shape == (0, 4)
    assert kept_scores.shape == (0,)


def test_containment_drops_a_seam_fragment_inside_a_whole_box():
    # Real pair from reef.jpg at the x=640 seam: the fragment is fully inside
    # the whole fish box, but their IoU (0.48) sits under the NMS threshold.
    whole = [617.0, 251.0, 666.0, 283.0]
    fragment = [640.0, 251.0, 664.0, 282.0]
    boxes = torch.tensor([whole, fragment])
    scores = torch.tensor([0.9, 0.85])

    survives_iou_only, _ = merge_boxes(boxes, scores, iou=0.5, containment=None)
    assert len(survives_iou_only) == 2  # the bug

    merged, _ = merge_boxes(boxes, scores, iou=0.5, containment=0.8)
    assert len(merged) == 1
    assert torch.allclose(merged[0], torch.tensor(whole))


def test_containment_keeps_merely_adjacent_boxes():
    boxes = torch.tensor([[0.0, 0.0, 40.0, 40.0], [38.0, 0.0, 78.0, 40.0]])
    scores = torch.tensor([0.9, 0.8])
    merged, _ = merge_boxes(boxes, scores, iou=0.5, containment=0.8)
    assert len(merged) == 2


def test_containment_threshold_is_respected():
    # small box overlaps half of its own area with the big one -> IoS 0.5
    boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [50.0, 0.0, 150.0, 100.0]])
    scores = torch.tensor([0.9, 0.8])
    assert len(merge_boxes(boxes, scores, iou=0.9, containment=0.8)[0]) == 2
    assert len(merge_boxes(boxes, scores, iou=0.9, containment=0.4)[0]) == 1


def test_containment_disabled_matches_plain_nms():
    boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [10.0, 10.0, 40.0, 40.0]])
    scores = torch.tensor([0.9, 0.8])
    assert len(merge_boxes(boxes, scores, iou=0.5, containment=None)[0]) == 2
    assert len(merge_boxes(boxes, scores, iou=0.5, containment=0.8)[0]) == 1


def test_containment_still_honours_max_det():
    boxes = torch.tensor([[float(i) * 100, 0.0, float(i) * 100 + 50, 50.0] for i in range(10)])
    scores = torch.linspace(0.9, 0.1, 10)
    merged, _ = merge_boxes(boxes, scores, iou=0.5, max_det=3, containment=0.8)
    assert len(merged) == 3


def test_containment_keeps_the_larger_box_not_the_higher_scoring_part():
    # Real case from reef.jpg at 2.0x magnification: the body fragment
    # outscores the whole fish, but the whole fish is the box worth keeping.
    whole = [885.0, 477.0, 1004.0, 536.0]
    body = [908.0, 488.0, 994.0, 526.0]
    boxes = torch.tensor([whole, body])
    scores = torch.tensor([0.43, 0.51])  # the part scores higher

    merged, merged_scores = merge_boxes(boxes, scores, iou=0.5, containment=0.8)

    assert len(merged) == 1
    assert torch.allclose(merged[0], torch.tensor(whole))
    assert float(merged_scores[0]) == pytest.approx(0.43)


def test_containment_absorbs_a_tail_box_into_the_whole_fish():
    whole = [885.0, 477.0, 1004.0, 536.0]
    tail = [885.0, 475.0, 924.0, 514.0]   # IoS 0.95 against whole, IoU 0.17
    boxes = torch.tensor([whole, tail])
    scores = torch.tensor([0.43, 0.36])

    assert len(merge_boxes(boxes, scores, iou=0.5, containment=None)[0]) == 2
    assert len(merge_boxes(boxes, scores, iou=0.5, containment=0.8)[0]) == 1


def test_tail_and_body_do_not_merge_without_the_whole_fish_box():
    # Neither IoU (0.10) nor IoS (0.27) can relate these two parts; only the
    # coarse whole-frame pass can supply the box that absorbs them.
    tail = [885.0, 475.0, 924.0, 514.0]
    body = [908.0, 488.0, 994.0, 526.0]
    boxes = torch.tensor([tail, body])
    scores = torch.tensor([0.36, 0.51])
    assert len(merge_boxes(boxes, scores, iou=0.5, containment=0.8)[0]) == 2


def test_merged_output_stays_sorted_by_score():
    boxes = torch.tensor([[float(i) * 200, 0.0, float(i) * 200 + 50, 50.0] for i in range(5)])
    scores = torch.tensor([0.3, 0.9, 0.5, 0.7, 0.1])
    _, merged_scores = merge_boxes(boxes, scores, iou=0.5, containment=0.8)
    assert merged_scores.tolist() == sorted(merged_scores.tolist(), reverse=True)
