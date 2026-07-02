# Fish Identification

## Project

This project is under development. A better readme will be upload later.

## Datasets

- <https://public.roboflow.com/object-detection/fish/1/download>
- Australian Institute of Marine Science (AIMS), University of Western Australia (UWA) and Curtin University. (2019), OzFish Dataset - Machine learning dataset for Baited Remote Underwater Video Stations, <https://doi.org/10.25845/5e28f062c5097> <https://github.com/open-AIMS/ozfish>
- <https://alzayats.github.io/DeepFish>
- <https://www.kaggle.com/datasets/zehraatlgan/fish-detection/data>


<https://www.youtube.com/watch?v=Egz4bXMlmDM>

## Tests

run test with

```
uv run pytest tests/
```

## Usage

### Train the detector (Stage 1)

```bash
# defaults: yolo11n.pt weights, 50 epochs, imgsz=640, batch=16, GPU 0
uv run python train_detector.py

# custom run
uv run python train_detector.py --weights yolo11n.pt --epochs 100 --imgsz 1280 --batch 8 --device cpu
```

Results are saved to `runs/detect/train*/` (weights, metrics, TensorBoard logs).

Interpret the training output:

Training row:

- box_loss — how far off the predicted bounding boxes are from the ground truth.
- cls_loss — how wrong the class predictions are.
- dfl_loss — Distribution Focal Loss: measures how precisely the box edges are placed (not just the center).

Validation row:

- P (Precision) — of every box the model predicted, how much actually contained a fish.
- R (Recall) — of every real fish in the images, how much the model found them.
- mAP50 — Measures detection accuracy at IoU ≥ 0.5 (a predicted box counts as correct if it overlaps the ground truth by at least 50%).
- mAP50-95 — Averaged over IoU thresholds from 0.5 to 0.95. Penalizes sloppy boxes.
