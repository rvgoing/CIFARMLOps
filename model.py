import torch.nn as nn
from torchvision import models


try:
    from torchvision.models import ResNet18_Weights
except ImportError:
    ResNet18_Weights = None


def get_model(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    weights = None
    if pretrained:
        weights = ResNet18_Weights.DEFAULT if ResNet18_Weights is not None else None

    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
