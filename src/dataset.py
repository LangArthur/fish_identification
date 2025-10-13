import os
import shutil
import torch
from enum import Enum
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


# Generate my_deep_fish dataset from the deepfish one.
# My_deep_fish is only a reorganisation of the deepfish one.
def create_my_deep_fish():
    original_path = "dataset/Deepfish"
    dest_path = "dataset/my_deep_fish"
    for entry in os.scandir(original_path):
        if entry.is_dir() and entry.name != "Nagative_samples":
            for folder in ["train", "valid"]:
                path = original_path + "/" + entry.name + "/" + folder
                print(path)
                for file in os.scandir(path):
                    _, ext = os.path.splitext(file.path)
                    if ext == ".txt":
                        shutil.copy(
                            path + "/" + file.name,
                            dest_path + "/labels/" + folder + "/" + file.name,
                        )
                    elif ext == ".jpg":
                        shutil.copy(
                            path + "/" + file.name,
                            dest_path + "/images/" + folder + "/" + file.name,
                        )
                    else:
                        print("Unexpected extension " + ext + ": ignoring file")


def decode_detection(path):
    labels = []
    with open(path) as fd:
        for line in fd:
            elem = line.split(" ")
            labels.append(
                torch.tensor(
                    [
                        float(elem[0]),
                        float(elem[1]),
                        float(elem[2]),
                        float(elem[3]),
                        float(elem[4]),
                    ],
                    dtype=torch.float32,
                )
            )
    return labels


class InvalidArgument(ValueError):
    def __init__(self, msg, *args: object) -> None:
        super().__init__(*args)
        self.msg = msg

    def __str__(self) -> str:
        return self.msg + super().__str__()


class DatasetType(Enum):
    TRAIN = 0
    VALID = 1


class DeepFishDataset(Dataset):
    def __init__(self, type, label_dir, img_dir, transform=None, target_transform=None):
        super().__init__()

        self.type_folder = "train" if type == DatasetType.TRAIN else "valid"
        self.filenames = [
            img.name.removesuffix(".jpg")
            for img in os.scandir(os.path.join(img_dir, self.type_folder))
        ]
        ds_len = len(self.filenames)
        if ds_len != len(
            [_ for _ in os.scandir(os.path.join(img_dir, self.type_folder))]
        ):
            raise InvalidArgument("Mismatch of label and images")
        self.len = ds_len
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_path = os.path.join(
            self.img_dir, self.type_folder, self.filenames[index] + ".jpg"
        )
        raw_img = Image.open(img_path)
        transform = transforms.Compose([transforms.PILToTensor()])
        img = transform(raw_img)
        label_path = os.path.join(
            self.label_dir, self.type_folder, self.filenames[index] + ".txt"
        )
        label = decode_detection(label_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
