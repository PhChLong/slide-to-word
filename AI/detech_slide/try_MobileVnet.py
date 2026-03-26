from __future__ import annotations

import torch
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def load_model(variant: str, pretrained: bool) -> torch.nn.Module:
    if variant == "large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return mobilenet_v3_large(weights=weights)

    if variant != "small":
        raise ValueError("variant must be 'large' or 'small'")

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    return mobilenet_v3_small(weights=weights)


def print_architecture(model: torch.nn.Module) -> None:
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nModel structure:")
    print(model)


if __name__ == "__main__":
    variant = "large"
    pretrained = False
    model = load_model(variant=variant, pretrained=pretrained)
    print(f"MobileNetV3-{variant}")
    print(f"Pretrained weights: {pretrained}")
    print_architecture(model)
