"""Centralized seeding for reproducibility."""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
