import sys
sys.path.insert(0, ".")
from src.models.fusion_model_e09 import FusionModelE09

model = FusionModelE09()

total = sum(p.numel() for p in model.vision.parameters())
trainable = sum(p.numel() for p in model.vision.parameters() if p.requires_grad)
print(f"Vision-only trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

for name, param in model.vision.features.named_parameters():
    top_level = name.split(".")[0]
    if top_level == "denseblock3":
        print(f"denseblock3 param '{name}': requires_grad={param.requires_grad}")
        break  # just check the first one is enough to confirm