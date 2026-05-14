"""Shared torch device picker for runner scripts.

Preference order: CUDA > MPS > CPU. When picking MPS we also set
PYTORCH_ENABLE_MPS_FALLBACK so individual ops that aren't implemented
for MPS fall back to CPU instead of crashing.
"""
from __future__ import annotations

import os
import sys

import torch

# macOS only: PyTorch, numpy, and matplotlib each ship their own libomp,
# which OpenMP refuses to load twice into the same process (OMP Error #15).
# The escape hatch is documented and safe for our inference-only workloads.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"
