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
│   └── pipeline.py           # End-to-end: detector → classifier
├── dataset/
│   ├── Deepfish/             # Original DeepFish dataset (YOLO format)
│   └── my_deep_fish/         # Reorganized: images/{train,valid}/ + labels/{train,valid}/
├── runs/detect/              # YOLO training run outputs
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
