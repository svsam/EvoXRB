"""Small, dependency-light persistence and provenance helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import SYNTHETIC_LABEL, __version__


def ensure_directories(*paths: str | Path) -> tuple[Path, ...]:
    resolved = tuple(Path(path) for path in paths)
    for path in resolved:
        path.mkdir(parents=True, exist_ok=True)
    return resolved


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def write_csv(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(list(records))
    frame.to_csv(destination, index=False)
    return destination


def save_npz(path: str | Path, **arrays: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def package_versions(names: Iterable[str] = ()) -> dict[str, str]:
    packages = ["numpy", "scipy", "pandas", "matplotlib", "PyYAML", *names]
    result: dict[str, str] = {}
    for name in dict.fromkeys(packages):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def provenance(config_hash: str, master_seed: int) -> dict[str, Any]:
    return {
        "label": SYNTHETIC_LABEL,
        "evoxrb_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_hash": config_hash,
        "master_seed": int(master_seed),
        "packages": package_versions(["emcee", "dynesty"]),
    }

