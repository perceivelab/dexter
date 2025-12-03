"""Factory helpers to load classifiers and their paired preprocessing transforms."""

import os
from typing import Optional, Tuple

import dill
import gdown
import torch
from torchvision import models, transforms as tfms
from torchvision.models import (
    AlexNet_Weights,
    DenseNet121_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    Swin_B_Weights,
    VGG16_Weights,
    ViT_B_16_Weights,
)
from torchvision.transforms import InterpolationMode

from dexter.models.resnet import resnet50


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

_TRANSFORM_LOOKUP = {
    "robust50": (256, DEFAULT_MEAN, DEFAULT_STD),
    "resnet18": (256, DEFAULT_MEAN, DEFAULT_STD),
    "resnet50": (256, DEFAULT_MEAN, DEFAULT_STD),
    "alexnet": (256, DEFAULT_MEAN, DEFAULT_STD),
    "vit_b_16": (224, DEFAULT_MEAN, DEFAULT_STD),
    "vgg": (256, DEFAULT_MEAN, DEFAULT_STD),
    "densenet": (256, DEFAULT_MEAN, DEFAULT_STD),
    "swin_b": (224, DEFAULT_MEAN, DEFAULT_STD),
    "fairfaces_resnet50": (256, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "fairfaces_resnet50_nobias": (256, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "waterbirds_resnet50": (256, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "waterbirds_balanced_resnet50": (256, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "waterbirds_erm": (256, DEFAULT_MEAN, DEFAULT_STD),
}


def _load_robust_resnet(device: str) -> torch.nn.Module:
    """Load the robust ResNet-50 checkpoint from disk.

    Args:
        device: Device string accepted by `torch.load` (e.g., "cpu", "cuda").

    Returns:
        A ResNet-50 model with weights loaded and moved to `device`.
    """
    try:
        model = resnet50(pretrained=False, num_classes=1000)
        checkpoint = torch.load(
            "checkpoints/robust_resnet50.pth", pickle_module=dill, map_location=device
        )
        state_dict = checkpoint["model"]
        state_dict = {k: v for k, v in state_dict.items() if "module.attacker" not in k}
        state_dict = {k: v for k, v in state_dict.items() if "module.normalizer" not in k}
        state_dict = {k[len("module.model.") :]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    except:
        raise RuntimeError("Failed to load robust ResNet-50 checkpoint. Please follow the instructions in the README to download the checkpoint.")
    return model


def _load_fairfaces_model(classifier: str) -> torch.nn.Module:
    """Load FairFace fine-tuned ResNet-50 variants.

    Args:
        classifier: Either "fairfaces_resnet50" or "fairfaces_resnet50_nobias".
    Returns:
        A ResNet-50 configured for 7-class FairFace prediction.
    """
    model = models.resnet50(pretrained=False)
    checkpoint_path = (
        "fairfaces/models_balanced/fairface_95.pth"
        if classifier == "fairfaces_resnet50"
        else "fairfaces/models_balanced/fairface_nobias_40.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.fc = torch.nn.Linear(2048, 7)
    model.load_state_dict(checkpoint, strict=True)
    return model


def _load_waterbirds_model(classifier: str) -> torch.nn.Module:
    """Load Waterbirds ResNet-50 checkpoints with the correct output head.

    Args:
        classifier: One of "waterbirds_resnet50" or "waterbirds_balanced_resnet50".
    Returns:
        A ResNet-50 configured for 2-class Waterbirds prediction.
    """
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2)
    checkpoint_path = (
        "waterbirds/models_balanced/waterbirds_90.pth"
        if classifier == "waterbirds_resnet50"
        else "waterbirds/models_balanced_test/waterbirds_90.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    return model


def _load_waterbirds_erm_model() -> torch.nn.Module:
    """Load the ERM-trained Waterbirds model persisted alongside results.

    Returns:
        A torch module restored from `best_model.pth`.
    """
    model = torch.load("best_model.pth", map_location="cpu")
    return model


def build_classifier(classifier: str, device: str) -> torch.nn.Module:
    """Instantiate a classifier by name and move it to the requested device.

    Args:
        classifier: Identifier matching a supported backbone or fine-tuned variant.
        device: Device string to move the model to (e.g., "cpu", "cuda").

    Returns:
        The initialized classifier set to eval mode and half precision.
    """
    if classifier == "robust50":
        model = _load_robust_resnet(device)
    elif classifier == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif classifier == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif classifier == "alexnet":
        model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    elif classifier == "vit_b_16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    elif classifier == "vgg":
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif classifier == "densenet":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    elif classifier == "swin_b":
        model = models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
    elif classifier in ("fairfaces_resnet50", "fairfaces_resnet50_nobias"):
        model = _load_fairfaces_model(classifier)
    elif classifier in ("waterbirds_resnet50", "waterbirds_balanced_resnet50"):
        model = _load_waterbirds_model(classifier)
    elif classifier == "waterbirds_erm":
        model = _load_waterbirds_erm_model()
    else:
        raise ValueError(
            "classifier must be one of [robust50, resnet18, resnet50, alexnet, vit_b_16, "
            "vgg, densenet, swin_b, fairfaces_resnet50, fairfaces_resnet50_nobias, "
            "waterbirds_resnet50, waterbirds_balanced_resnet50, waterbirds_erm]"
        )

    model.eval()
    return model.to(device).half()


def build_transforms(classifier: str, use_transforms: bool) -> Optional[tfms.Compose]:
    """Return the resize/normalize pipeline that matches the chosen classifier.

    Args:
        classifier: Identifier matching a supported backbone or fine-tuned variant.
        use_transforms: If False, skip preprocessing and return None.

    Returns:
        A torchvision `Compose` with resize + normalization, or None when disabled.
    """
    if not use_transforms:
        return None

    if classifier not in _TRANSFORM_LOOKUP:
        raise ValueError(f"Unsupported classifier '{classifier}' for transforms")

    size, mean, std = _TRANSFORM_LOOKUP[classifier]
    return tfms.Compose(
        [
            tfms.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=None),
            tfms.Normalize(mean=mean, std=std),
        ]
    )


def build_classifier_and_transforms(
    clf: str, device: str, use_tfms: bool
) -> Tuple[torch.nn.Module, Optional[tfms.Compose]]:
    """Convenience wrapper to fetch both a classifier and its transforms.

    Args:
        clf: Classifier identifier, forwarded to `build_classifier`.
        device: Device string to host the model.
        use_tfms: Whether to construct paired preprocessing transforms.

    Returns:
        (classifier, transforms) where transforms may be None if disabled.
    """
    classifier = build_classifier(clf, device)
    transforms = build_transforms(clf, use_tfms)
    return classifier, transforms
