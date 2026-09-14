# Experiment Log

Track every modeling experiment here. One row per experiment.

| ID | Date | Phase | Model | Changes | AUROC | AUPRC | Notes |
| -- | ---- | ----- | ----- | ------- | ----- | ----- | ----- |
|    |      |       |       |         |       |       |       |
| E01 | 2026-09-11 | 4 | DenseNet-121 (ImageNet pretrained) | baseline, no augmentation beyond h-flip, no class weighting | 0.7281 | 0.5028 | Best: Pleural Effusion (0.7854), Worst: Atelectasis (0.6496). |
| E02 | 2026-09-12 | 4 | DenseNet-121 (ImageNet pretrained) | class-imbalance weighting via per-label pos_weight (auto-computed from train split), otherwise identical to E01 | 0.7203 | 0.4981 | Best: Pleural Effusion (0.7871), Worst: Atelectasis (0.6382). |
| E03 | 2026-09-12 | 4 | DenseNet-121 (ImageNet pretrained) | frozen early DenseNet blocks (only denseblock3, denseblock4, classifier trainable) + added rotation/brightness/contrast augmentation, keeping E02's pos_weight class weighting | 0.7240 | 0.5001 | Best: Pleural Effusion (0.7819), Worst: Atelectasis (0.6295). |
| E04 | 2026-09-12 | 4 | DenseNet-121 (ImageNet pretrained) | added ReduceLROnPlateau LR scheduler (factor=0.5, patience=2, monitoring val AUROC), max_epochs increased to 20, otherwise identical to E03 | 0.7267 | 0.5031 | Best: Pleural Effusion (0.7829), Worst: Atelectasis (0.6218). |
| E05 | 2026-09-14 | 4 | DenseNet-121 (ImageNet pretrained) | stronger freeze (only denseblock4, norm5, classifier trainable -- froze denseblock3 in addition to E04's frozen blocks), otherwise identical to E04 | 0.6999 | 0.4726 | Best: Pleural Effusion (0.7421), Worst: Atelectasis (0.6298). |
