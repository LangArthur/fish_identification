#!/usr/bin/env python3
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches

from ultralytics import YOLO
from src.dataset import DeepFishDataset, DatasetType
from torch.utils.data import DataLoader


def plot_img(ds, indexes):
    for i in indexes:
        img, labels = ds[0]
        fig, a = plt.subplots(1, 1)
        a.imshow(torch.permute(img, (1, 2, 0)))
        for label in labels:
            x, y, width, height = label[1], label[2], label[3], label[4]
            print(x, y)
            rect = patches.Rectangle((x, y), width, height)
            a.add_patch(rect)
        plt.show()


def main():
    # print(torch.cuda.is_available())

    print(plt.get_backend())
    train_ds = DeepFishDataset(
        DatasetType.TRAIN, "dataset/my_deep_fish/labels", "dataset/my_deep_fish/images"
    )
    test_ds = DeepFishDataset(
        DatasetType.VALID, "dataset/my_deep_fish/labels", "dataset/my_deep_fish/images"
    )

    print("dataset size: {}".format(len(train_ds)))
    img, _ = train_ds[0]
    print(img.shape)

    train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=4, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    # model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano pretrained weights
    # model.train(
    #     data="yolo.yaml",  # Path to YAML config
    #     epochs=5,  # Number of epochs
    #     imgsz=1920,  # Image size
    #     batch=16,  # Batch size
    #     device=0,
    # )  # GPU device index


if __name__ == "__main__":
    main()
