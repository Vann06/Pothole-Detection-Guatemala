"""ResNet18 with one logit per target: D40 first, optional auxiliary damage classes after it."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(pretrained=False, outputs=1):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, outputs)
    return model
