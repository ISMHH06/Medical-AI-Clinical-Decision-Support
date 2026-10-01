"""Read-only inspection of E08 FusionModel weights.

This script prints the trained model structure and simple weight statistics
to check whether the clinical branch appears down-weighted relative to the
vision embedding path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.fusion_model import FusionModel

FUSION_CHECKPOINT_PATH = REPO_ROOT / "models" / "fusion_e08_best.pt"


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def load_e08_model() -> FusionModel:
    if not FUSION_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Fusion checkpoint not found: {FUSION_CHECKPOINT_PATH}")

    model = FusionModel()
    checkpoint = torch.load(FUSION_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.eval()
    return model


def mean_abs_and_normalized_l2(weight: torch.Tensor) -> tuple[float, float]:
    weight_np = weight.detach().cpu().numpy().astype(np.float64)
    mean_abs = float(np.abs(weight_np).mean())
    normalized_l2 = float(np.linalg.norm(weight_np) / weight_np.size)
    return mean_abs, normalized_l2


def main() -> None:
    model = load_e08_model()

    print("E08 FusionModel structure:")
    print(model)

    print("\nNamed modules:")
    for name, module in model.named_modules():
        if name == "":
            continue
        print(f"{name}: {module.__class__.__name__}")

    first_fusion_layer = model.fusion_head[0]
    if not isinstance(first_fusion_layer, torch.nn.Linear):
        raise TypeError(f"Expected fusion_head[0] to be Linear, got {type(first_fusion_layer)!r}")

    weight = first_fusion_layer.weight.detach().cpu()
    vision_slice = weight[:, :1024]
    clinical_slice = weight[:, 1024:]

    vision_mean_abs, vision_norm = mean_abs_and_normalized_l2(vision_slice)
    clinical_mean_abs, clinical_norm = mean_abs_and_normalized_l2(clinical_slice)

    print("\nFirst fusion layer:")
    print(f"name: fusion_head[0]")
    print(f"weight shape: {tuple(weight.shape)}")
    print(f"vision slice shape: {tuple(vision_slice.shape)}")
    print(f"clinical slice shape: {tuple(clinical_slice.shape)}")
    print(f"vision slice mean_abs_weight: {vision_mean_abs:.8f}")
    print(f"vision slice normalized_l2: {vision_norm:.8f}")
    print(f"clinical slice mean_abs_weight: {clinical_mean_abs:.8f}")
    print(f"clinical slice normalized_l2: {clinical_norm:.8f}")
    print(f"vision/clinical mean_abs ratio: {vision_mean_abs / clinical_mean_abs:.8f}")

    print("\nClinical encoder layer weight stats:")
    for name, module in model.clinical_encoder.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        layer_weight = module.weight.detach().cpu()
        mean_abs, normalized_l2 = mean_abs_and_normalized_l2(layer_weight)
        print(f"{name}: shape={tuple(layer_weight.shape)} mean_abs_weight={mean_abs:.8f} normalized_l2={normalized_l2:.8f}")


if __name__ == "__main__":
    main()