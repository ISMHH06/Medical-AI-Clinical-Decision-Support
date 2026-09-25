"""Grad-CAM utilities for the Phase 7 E08 early-fusion model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.image_dataset import IMAGENET_MEAN, IMAGENET_STD, build_transform
from src.models.fusion_model import FusionModel
from src.models.vision_baseline import build_vision_baseline_frozen


REPO_ROOT = Path(__file__).resolve().parents[2]
VISION_CHECKPOINT_PATH = REPO_ROOT / "models" / "vision_baseline_champion.pt"
FUSION_CHECKPOINT_PATH = REPO_ROOT / "models" / "fusion_e08_best.pt"


class MultiLabelOutputTarget:
    """Select one sigmoid logit from a multi-label model output tensor."""

    def __init__(self, label_idx: int) -> None:
        self.label_idx = label_idx

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if len(model_output.shape) == 1:
            return model_output[self.label_idx]
        return model_output[:, self.label_idx]


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Accept either a wrapped checkpoint or a raw state_dict."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint  # raw OrderedDict/state_dict checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


class E04Explainer(nn.Module):
    """Vision-only E04 explainer that keeps the trained 5-output classifier intact."""

    def __init__(self) -> None:
        super().__init__()
        if not VISION_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Vision checkpoint not found: {VISION_CHECKPOINT_PATH}")

        self.densenet = build_vision_baseline_frozen(unfreeze_from_block="denseblock3")
        vision_state = torch.load(VISION_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self.densenet.load_state_dict(_extract_state_dict(vision_state))
        for parameter in self.densenet.parameters():
            parameter.requires_grad_(False)
        self.densenet.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return the 5-class vision-only logits for a raw image."""
        return self.densenet(image)


class E08Explainer(nn.Module):
    """Live image-to-logit wrapper for Grad-CAM on the E08 fusion model."""

    def __init__(self) -> None:
        super().__init__()
        if not VISION_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Vision checkpoint not found: {VISION_CHECKPOINT_PATH}")
        if not FUSION_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Fusion checkpoint not found: {FUSION_CHECKPOINT_PATH}")

        self.densenet = build_vision_baseline_frozen(unfreeze_from_block="denseblock3")
        vision_state = torch.load(VISION_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self.densenet.load_state_dict(_extract_state_dict(vision_state))
        self.densenet.classifier = nn.Identity()
        for parameter in self.densenet.parameters():
            parameter.requires_grad_(False)
        self.densenet.eval()

        self.fusion_model = FusionModel()
        fusion_state = torch.load(FUSION_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self.fusion_model.load_state_dict(_extract_state_dict(fusion_state))
        for parameter in self.fusion_model.parameters():
            parameter.requires_grad_(False)
        self.fusion_model.eval()

    def forward(self, image: torch.Tensor, clinical_feat: torch.Tensor) -> torch.Tensor:
        """Map a raw image and clinical features to 5 fusion logits."""
        emb = self.densenet(image)
        logits = self.fusion_model(emb, clinical_feat)
        return logits


def _load_image_for_cam(image_path: str) -> tuple[torch.Tensor, np.ndarray]:
    """Load the val-transformed image tensor and a matching RGB float image."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        rgb_image = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
        transformed = build_transform("val")(image)
    return transformed, rgb_image


def generate_gradcam(
    explainer: E08Explainer,
    image_path: str,
    clinical_feat: torch.Tensor,
    label_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Grad-CAM heatmap and overlay for one E08 prediction."""
    if label_idx < 0 or label_idx >= 5:
        raise ValueError(f"label_idx must be between 0 and 4, got {label_idx}.")

    explainer = explainer.to(device)
    explainer.eval()

    image_tensor, _ = _load_image_for_cam(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    if clinical_feat.dim() == 1:
        clinical_feat = clinical_feat.unsqueeze(0)
    clinical_feat = clinical_feat.to(device)

    class _ImageOnlyWrapper(nn.Module):
        def __init__(self, wrapped_explainer: E08Explainer, fixed_clinical_feat: torch.Tensor) -> None:
            super().__init__()
            self.explainer = wrapped_explainer
            self.register_buffer("fixed_clinical_feat", fixed_clinical_feat)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.explainer(image, self.fixed_clinical_feat)

    wrapped_model = _ImageOnlyWrapper(explainer, clinical_feat)
    target_layers = [wrapped_model.explainer.densenet.features.norm5]
    targets = [MultiLabelOutputTarget(label_idx)]

    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    heatmap = grayscale_cam[0] if grayscale_cam.ndim == 3 else grayscale_cam

    if image_tensor.ndim != 4:
        raise RuntimeError(f"Expected a batched image tensor, got shape {tuple(image_tensor.shape)}")
    normalized = image_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    unnormalized = normalized * np.asarray(IMAGENET_STD, dtype=np.float32) + np.asarray(
        IMAGENET_MEAN, dtype=np.float32
    )
    unnormalized = np.clip(unnormalized, 0.0, 1.0)
    overlay_image = show_cam_on_image(unnormalized, heatmap, use_rgb=True)

    return heatmap, overlay_image


def generate_gradcam_e04(
    explainer: E04Explainer,
    image_path: str,
    label_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Grad-CAM heatmap and overlay for one E04 vision-only prediction."""
    if label_idx < 0 or label_idx >= 5:
        raise ValueError(f"label_idx must be between 0 and 4, got {label_idx}.")

    explainer = explainer.to(device)
    explainer.eval()

    image_tensor, _ = _load_image_for_cam(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    target_layers = [explainer.densenet.features.norm5]
    targets = [MultiLabelOutputTarget(label_idx)]

    cam = GradCAM(model=explainer, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    heatmap = grayscale_cam[0] if grayscale_cam.ndim == 3 else grayscale_cam

    if image_tensor.ndim != 4:
        raise RuntimeError(f"Expected a batched image tensor, got shape {tuple(image_tensor.shape)}")
    normalized = image_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    unnormalized = normalized * np.asarray(IMAGENET_STD, dtype=np.float32) + np.asarray(
        IMAGENET_MEAN, dtype=np.float32
    )
    unnormalized = np.clip(unnormalized, 0.0, 1.0)
    overlay_image = show_cam_on_image(unnormalized, heatmap, use_rgb=True)

    return heatmap, overlay_image