from pathlib import Path

import timm
import timm.data
import torch
import torch.nn as nn
from PIL import Image


class FishClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, image: Image.Image) -> tuple[int, float]:
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            tensor = self.transform(image).unsqueeze(0).to(device)
            probs = torch.softmax(self(tensor), dim=1)
            confidence, class_idx = probs.max(dim=1)
        return class_idx.item(), confidence.item()

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(
        cls,
        path: Path,
        num_classes: int,
        backbone: str = "efficientnet_b0",
    ) -> "FishClassifier":
        model = cls(num_classes=num_classes, backbone=backbone, pretrained=False)
        model.load_state_dict(torch.load(Path(path), weights_only=True))
        return model
