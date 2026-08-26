"""Configuration loading and validation for the synthetic case study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG_PATH = Path("config/case_study.yaml")


@dataclass(frozen=True, slots=True)
class CaseStudyConfig:
    """Validated top-level configuration.

    Section mappings remain dictionaries so the command-line stages can pass
    settings to independent numerical subsystems without a large dependency
    graph.  Required keys are checked when loading.
    """

    project: dict[str, Any]
    instrument: dict[str, Any]
    search: dict[str, Any]
    ga: dict[str, Any]
    scipy: dict[str, Any]
    posterior: dict[str, Any]
    recovery: dict[str, Any]
    timing: dict[str, Any]
    epochs: tuple[dict[str, Any], ...]
    source_path: Path
    digest: str

    @property
    def master_seed(self) -> int:
        return int(self.project["master_seed"])

    @property
    def output_dir(self) -> Path:
        return Path(self.project["output_dir"])

    @property
    def figure_dir(self) -> Path:
        return Path(self.project["figure_dir"])

    @property
    def report_dir(self) -> Path:
        return Path(self.project["report_dir"])


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> CaseStudyConfig:
    """Load a YAML file and validate the stable public configuration shape."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("case-study configuration must be a mapping")

    required = {
        "project",
        "instrument",
        "search",
        "ga",
        "scipy",
        "posterior",
        "recovery",
        "timing",
        "epochs",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"configuration is missing sections: {', '.join(missing)}")
    epochs = payload["epochs"]
    if not isinstance(epochs, list) or len(epochs) != 12:
        raise ValueError("the fixed synthetic outburst must contain exactly 12 epochs")
    epoch_ids = [str(item.get("epoch_id", "")) for item in epochs]
    if len(set(epoch_ids)) != len(epoch_ids) or any(not item for item in epoch_ids):
        raise ValueError("epoch_id values must be non-empty and unique")

    return CaseStudyConfig(
        project=dict(payload["project"]),
        instrument=dict(payload["instrument"]),
        search=dict(payload["search"]),
        ga=dict(payload["ga"]),
        scipy=dict(payload["scipy"]),
        posterior=dict(payload["posterior"]),
        recovery=dict(payload["recovery"]),
        timing=dict(payload["timing"]),
        epochs=tuple(dict(item) for item in epochs),
        source_path=source,
        digest=_canonical_digest(payload),
    )


def smoke_overrides(config: CaseStudyConfig) -> dict[str, dict[str, Any]]:
    """Return bounded, fast settings used by CI and local smoke checks."""

    return {
        "ga": {
            **config.ga,
            "population_size": 32,
            "max_generations": 24,
            "min_generations": 8,
            "stop_window": 8,
            "stagnation_generations": 6,
            "seeds": 1,
        },
        "scipy": {**config.scipy, "starts": 4},
        "posterior": {
            **config.posterior,
            "batch_steps": 50,
            "max_steps": 200,
            "dynesty_live_points": 50,
        },
        "recovery": {
            **config.recovery,
            "high_signal_exposure_s": 4096.0,
            "exposure_levels_s": [512.0],
            "realizations": 2,
        },
        "timing": {
            **config.timing,
            "dt_s": 1.0 / 64.0,
            "segment_s": 32.0,
            "bootstrap_samples": 8,
        },
    }

