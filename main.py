#!/usr/bin/env python3
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches

# from ultralytics import YOLO
from src.dataset import DeepFishDataset


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
    ds = DeepFishDataset("dataset/my_deep_fish/labels", "dataset/my_deep_fish/images")

    print("dataset size: {}".format(len(ds)))
    img, _ = ds[0]
    print(img.shape)

    plot_img(ds, [0])

    # model = YOLO('yolov8n.pt')  # Load YOLOv8 Nano pretrained weights
    # model.train(data='yolo.yaml',  # Path to YAML config
    #         epochs=5,                  # Number of epochs
    #         imgsz=1920,                  # Image size
    #         batch=16,                   # Batch size
    #         device=0)                   # GPU device index


if __name__ == "__main__":
    main()
