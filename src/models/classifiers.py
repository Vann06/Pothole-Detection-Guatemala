"""Supervised image classifiers with D40 at logit 0; no implicit pretrained downloads at inference."""
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.models.resnet import build_resnet18

CLASSIFIER_ARCHITECTURES = {"resnet": "resnet18", "efficientnet": "efficientnet_b0"}


def build_classifier(family="resnet", pretrained=False, outputs=1):
    if family == "resnet":
        return build_resnet18(pretrained=pretrained, outputs=outputs)
    if family == "efficientnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        # Preserve B0's dropout. BCEWithLogitsLoss consumes logits, not sigmoid probabilities.
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, outputs)
        return model
    raise ValueError(f"Unknown classifier family: {family}")
