# Fish Identification — Project Guide

## Overview

Fish detection and identification system using YOLO-based object detection on underwater imagery.

## Tech Stack

- **Python 3.13**, managed with `uv`
- **torch 2.12.1** + **torchvision 0.27.1** — core deep learning framework
- **ultralytics 8.4.72** — YOLO11 for Stage 1 detection
- **timm 1.0.27** — model zoo for Stage 2 classifier (EfficientNet, ConvNeXt, ViT, etc.)
- **albumentations 2.0.8** — augmentation pipeline
- **opencv-python-headless 4.13.0.92** — image processing
- **scikit-learn 1.9.0** — evaluation metrics (confusion matrix, classification report)
- **tqdm 4.68.3** — training progress bars
- **tensorboard 2.20.0** — experiment tracking
- **matplotlib 3.11.0** + **Pillow 12.2.0** — visualization and image I/O
- `src/AModel.py` (TensorFlow/Keras) is legacy and not actively used

## Project Structure

```bash
fish_identification/
├── src/
│   ├── data/
│   │   ├── dataset.py        # DeepFishDataset (PyTorch Dataset), create_my_deep_fish()
│   │   └── transforms.py     # Albumentations augmentation pipelines
│   ├── detector/
│   │   └── detector.py       # Stage 1: YOLO11 fish detector wrapper
│   ├── classifier/
│   │   └── classifier.py     # Stage 2: timm-based species classifier
│   ├── analysis/
│   │   ├── stats.py          # DatasetStats dataclass + compute_stats()
│   │   └── visualize.py      # plot functions (box size, aspect ratio, heatmap, sample grid)
│   └── pipeline.py           # End-to-end: detector → classifier
├── dataset/
│   ├── Deepfish/             # Original DeepFish dataset (YOLO format)
│   └── my_deep_fish/         # Reorganized: images/{train,valid}/ + labels/{train,valid}/
├── runs/detect/              # YOLO training run outputs
├── analyze_dataset.py        # CLI: analyze any YOLO-format dataset
├── main.py                   # Temporary entry point (to be replaced)
├── yolov8n.pt                # Pretrained YOLOv8 Nano weights
├── yolo11n.pt                # Pretrained YOLO11 Nano weights
└── pyproject.toml
```

## Dataset

- **Source:** DeepFish — underwater fish images with YOLO-format labels (`.jpg` + `.txt`)
- **Label format:** `class cx cy w h` (normalized floats, one object per line)
- **Reorganization:** `create_my_deep_fish()` in `src/dataset.py` copies files from `dataset/Deepfish/` into the flat `dataset/my_deep_fish/` structure
- Other datasets referenced in README: Roboflow fish, OzFish (AIMS/UWA)

## Training History

Past runs in `runs/detect/train*/` used:

- Model: YOLOv8n, epochs: 5, batch: 16, imgsz: 1920, device: GPU 0
- Data config: `custom_dataset.yaml`

## Architecture

### Pipeline

```
image → [Stage 1: YOLO detector] → fish crop(s) → [Stage 2: species classifier] → species + confidence
```

### Stage 1 — Fish Detector

- Model: YOLO11n (generic fish finder)
- Task: binary detection — locate fish and produce bounding box crops
- No species knowledge at this stage
- Train once; reuse across all species/datasets

### Stage 2 — Species Classifier

- Model: fine-grained classifier (EfficientNet, ConvNeXt, or ViT)
- Input: cropped fish image from Stage 1
- Output: species label + confidence score
- Flat classification (all species in one model) to start

### Hierarchical classification (family → species)

Considered and deferred. The idea of one model per fish family is taxonomically sound but adds cascading errors (wrong family = guaranteed wrong species) and significant operational complexity. The preferred approach is to bake taxonomy into training via **hierarchical auxiliary labels** on the single classifier if fine-grained confusion becomes a measured problem. Revisit if the flat classifier consistently confuses species within the same family.

---

## Dataset Analysis Tooling

Built to evaluate candidate datasets before committing to training. All tools operate on YOLO-format datasets (images + `.txt` label files). Entry point: `analyze_dataset.py --data <path> --split <train|valid|all> --output <dir>`.

### Metrics computed (`src/analysis/stats.py` → `DatasetStats`)

| Metric | What it reveals |
|---|---|
| Relative box area (`w × h`) | Core "far vs close" fish bias — the root cause of DeepFish failure |
| Box aspect ratio (`w / h`) | Orientation diversity; informs anchor tuning |
| Box center heatmap (`cx, cy`) | Spatial bias (fish always centered / always at edges?) |
| Boxes per image | Crowding and occlusion level |
| Negative sample rate | Fraction of images with no fish; affects training balance |
| Image resolution distribution | Informs `imgsz` training parameter |

### Visualizations (`src/analysis/visualize.py`)

- `plot_box_size_distribution()` — histogram of relative box areas with percentile markers
- `plot_aspect_ratio_distribution()` — histogram of box w/h ratios
- `plot_center_heatmap()` — 2D density map of box center positions
- `plot_sample_grid()` — random sample of N images with YOLO boxes overlaid

---

## Session Log

### 2026-06-21

- Initial session: explored project structure, created memory files and this CLAUDE.md
- Project is early-stage; dataset loading is working, YOLO training pipeline partially set up but not complete
- Defined two-stage architecture: YOLO detector → species classifier
- Decided against per-family specialist models for now; will use hierarchical auxiliary labels if needed
- Updated all dependencies to latest versions (torch 2.12.1, ultralytics 8.4.72, timm, albumentations, opencv-headless, scikit-learn, tqdm, tensorboard)
- Deleted legacy files: `src/AModel.py` (TF/Keras) and root `dataset.py`
- Set up new `src/` structure: `data/`, `detector/`, `classifier/` submodules + `pipeline.py`
- Refactored `src/data/dataset.py`: pathlib throughout, fixed mismatch check bug, `DatasetSplit` enum with string values, `_parse_yolo_label` returns (N, 5) tensor, `create_my_deep_fish()` accepts Path args
- Built `src/detector/detector.py`: `FishDetector` wraps YOLO11, exposes `train()` and `detect()` → returns `Detection` (boxes, scores, crops)
- Created `detector_data.yaml`: single class (`fish`, id=0), points to `dataset/my_deep_fish/`
- Built `src/classifier/classifier.py`: `FishClassifier(nn.Module)` wraps any timm backbone, exposes `predict()` (PIL Image → class + confidence), `save()`/`load()` via state_dict
- Full project review passed: structure, imports, dataset, detector, classifier, and YAML all verified correct
- Created `train_detector.py`: CLI wrapper around `FishDetector.train()`, fine-tunes from `yolo11n.pt` by default

### 2026-06-25

- Diagnosed DeepFish training failure: dataset is skewed toward small, distant fish — model learned to ignore large foreground fish
- Decided to build dataset analysis tooling (`src/analysis/`) to evaluate candidate replacement datasets
- Defined 6 key metrics: relative box area, aspect ratio, box center heatmap, boxes per image, negative sample rate, resolution distribution
- Architecture: `stats.py` (compute) + `visualize.py` (plot) + `analyze_dataset.py` (CLI); no report.py
- No `report.py` — stats and plots are used directly via the CLI or imported
