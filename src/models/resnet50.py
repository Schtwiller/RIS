"""
src/models/resnet50.py
-----------------------

Defines the ResNet-50 constructor and a feature-extractor wrapper.
No evaluation, plotting, or external deps beyond torch/torchvision.
"""

import torch.nn as nn
from torchvision import models

def create_resnet50_model(
    num_classes: int,
    *,
    use_pretrained: bool = True,
    feature_extract: bool = False,
) -> nn.Module:
    """
    Build a ResNet-50, optionally loading ImageNet weights and freezing
    the conv backbone if feature_extract=True.
    """
    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None
    )
    if feature_extract:
        for p in model.parameters():
            p.requires_grad_(False)

    in_feats = model.fc.in_features  # 2048
    model.fc = nn.Linear(in_feats, num_classes)
    return model


class FeatureExtractor(nn.Module):
    """
    Wraps a trained ResNet-50 so that forward(x) returns the penultimate
    feature vector instead of class logits.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        # everything except the final FC
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        x = self.features(x)            # [B, 2048, 1, 1]
        return x.view(x.size(0), -1)    # [B, 2048]
