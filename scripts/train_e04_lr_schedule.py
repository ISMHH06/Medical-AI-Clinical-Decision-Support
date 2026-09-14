"""Command-line entry point for the Phase 5 (E04) LR-scheduled vision training."""

from pathlib import Path
import sys


# Permit direct execution from the repository without requiring package install.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.training.train_vision_e04_lr_schedule import main  # noqa: E402


if __name__ == "__main__":
    main()
