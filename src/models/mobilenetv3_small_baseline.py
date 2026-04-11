import torch.nn as nn
from torchvision import models


class MobileNetV3SmallBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)