# import torch
from ultralytics import YOLO

from src.dataset import DeepFishDataset

def main():
    # print(torch.cuda.is_available())

    ds = DeepFishDataset("dataset/my_deep_fish/labels", "dataset/my_deep_fish/images")

    print(len(ds))
    print(ds[0])

    # model = YOLO('yolov8n.pt')  # Load YOLOv8 Nano pretrained weights
    # model.train(data='yolo.yaml',  # Path to YAML config
    #         epochs=5,                  # Number of epochs
    #         imgsz=1920,                  # Image size
    #         batch=16,                   # Batch size
    #         device=0)                   # GPU device index


if __name__ == "__main__":
    main()
