import torch.nn as nn
from torchvision import models


class MobileNetV3SmallBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)