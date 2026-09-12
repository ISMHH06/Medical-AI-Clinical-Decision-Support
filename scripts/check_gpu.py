"""Diagnostic script to check PyTorch and CUDA availability.

Reports what is currently installed without modifying dependencies.
Run with: python scripts/check_gpu.py
"""

import torch


def main() -> None:
    """Print PyTorch version, CUDA availability, and device details."""
    print(f"torch.__version__: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    if cuda_available:
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        print(f"torch.version.cuda (CUDA build version): {torch.version.cuda}")
    else:
        print(
            "CUDA is not available. This usually means the installed PyTorch "
            "build is CPU-only. To use the GPU, reinstall PyTorch with CUDA "
            "support (not the CPU-only build). Use the install selector at "
            "https://pytorch.org/ to get the correct pip command matching "
            "your installed CUDA driver version."
        )


if __name__ == "__main__":
    main()
