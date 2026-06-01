"""I/O utilities: save/load JSON results and run directories."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create the directory if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Save *data* as a JSON file at *path*.

    Args:
        data: JSON-serializable object.
        path: Target file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load and return the JSON object at *path*."""
    with Path(path).open() as fh:
        return json.load(fh)
