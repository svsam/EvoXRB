"""Acceptance checks for generated synthetic case-study artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import SYNTHETIC_LABEL
from .config import CaseStudyConfig


EXPECTED_FIGURES = (
    "response_area.png",
    "folded_spectrum.png",
    "ga_convergence.png",
    "ga_population.png",
    "recovery.png",
    "optimizer_comparison.png",
    "posterior.png",
    "hardness_intensity.png",
    "parameter_qpo_evolution.png",
    "qpo_gamma.png",
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    checked_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": SYNTHETIC_LABEL,
            "valid": self.valid,
            "errors": list(self.errors),
            "checked_files": list(self.checked_files),
        }


def validate_case_study(
    config: CaseStudyConfig,
    *,
    profile: str,
    results_dir: str | Path,
    figure_dir: str | Path,
    report_dir: str | Path,
) -> ValidationResult:
    """Check completeness, synthetic labels, response integrity, and provenance."""

    results = Path(results_dir)
    figures = Path(figure_dir)
    reports = Path(report_dir)
    errors: list[str] = []
    checked: list[str] = []

    required = [
        results / "truth.csv",
        results / "fit_results.csv",
        results / "recovery_results.csv",
        results / "recovery_summary.csv",
        results / "posterior_summary.csv",
        results / "timing_results.csv",
        results / "provenance.json",
        results / "run_summary.json",
        results / "instrument_response.npz",
        reports / "case_study.md",
        reports / "README.md",
        *(figures / name for name in EXPECTED_FIGURES),
        *(results / "spectra" / f"E{index:02d}.npz" for index in range(1, 13)),
    ]
    posterior_epochs = ("E08",) if profile == "smoke" else ("E01", "E08", "E10")
    required.extend(
        results / "posterior" / profile / f"{epoch}_powerlaw.npz"
        for epoch in posterior_epochs
    )
    for path in required:
        checked.append(str(path))
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {path}")

    table_requirements = {
        "truth.csv": 12,
        "fit_results.csv": 1,
        "recovery_results.csv": 1,
        "posterior_summary.csv": 4,
        "timing_results.csv": 3 if profile == "smoke" else 12,
    }
    for filename, minimum_rows in table_requirements.items():
        path = results / filename
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as error:
            errors.append(f"cannot read {filename}: {error}")
            continue
        if "profile" in frame.columns:
            frame = frame[frame["profile"].astype(str) == profile]
        if len(frame) < minimum_rows:
            errors.append(f"{filename} has {len(frame)} rows; expected at least {minimum_rows}")
        if "label" not in frame.columns or not frame["label"].astype(str).str.startswith(SYNTHETIC_LABEL).all():
            errors.append(f"{filename} has missing or invalid synthetic labels")

    truth_path = results / "truth.csv"
    if truth_path.exists():
        truth = pd.read_csv(truth_path)
        expected_ids = [f"E{index:02d}" for index in range(1, 13)]
        if truth.get("epoch_id", pd.Series(dtype=str)).astype(str).tolist() != expected_ids:
            errors.append("truth.csv does not contain the fixed E01--E12 sequence")

    fit_path = results / "fit_results.csv"
    if fit_path.exists():
        fits = pd.read_csv(fit_path)
        if "profile" in fits.columns:
            fits = fits[fits["profile"].astype(str) == profile]
        methods = set(fits.get("method", pd.Series(dtype=str)).astype(str))
        if not {"scipy", "ga", "ga+scipy"}.issubset(methods):
            errors.append("fit_results.csv lacks separate SciPy, raw GA, or polished GA rows")
        if not (fits.get("fit_model", pd.Series(dtype=str)) == "educational_powerlaw_free_nh_sensitivity").any():
            errors.append("fit_results.csv lacks the free-NH sensitivity run")

    recovery_path = results / "recovery_results.csv"
    if recovery_path.exists():
        recovery = pd.read_csv(recovery_path)
        if "profile" in recovery.columns:
            recovery = recovery[recovery["profile"].astype(str) == profile]
        scenarios = set(recovery.get("scenario", pd.Series(dtype=str)).astype(str))
        if not {"correct_model", "cutoff_truth_powerlaw_fit", "high_signal_correct"}.issubset(scenarios):
            errors.append("recovery_results.csv lacks a required recovery scenario")

    response_path = results / "instrument_response.npz"
    if response_path.exists():
        try:
            with np.load(response_path, allow_pickle=False) as response:
                redistribution = np.asarray(response["redistribution"], dtype=float)
            if redistribution.shape != (236, 600):
                errors.append(f"response has shape {redistribution.shape}, expected (236, 600)")
            if not np.allclose(redistribution.sum(axis=0), 1.0, atol=2e-13, rtol=2e-13):
                errors.append("response redistribution columns are not normalized")
        except Exception as error:
            errors.append(f"cannot validate instrument response: {error}")

    provenance_path = results / "provenance.json"
    if provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            for key in ("config_hash", "master_seed", "python", "packages", "runtime_seconds"):
                if key not in provenance:
                    errors.append(f"provenance.json lacks {key}")
            if provenance.get("config_hash") != config.digest:
                errors.append("provenance configuration hash does not match")
            if int(provenance.get("master_seed", -1)) != config.master_seed:
                errors.append("provenance master seed does not match")
        except Exception as error:
            errors.append(f"cannot validate provenance.json: {error}")

    report_path = reports / "case_study.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        if SYNTHETIC_LABEL not in text:
            errors.append("case-study report lacks the synthetic label")
        if "contains no real NICER observations" not in text:
            errors.append("case-study report lacks the real-data disclaimer")

    return ValidationResult(not errors, tuple(errors), tuple(checked))
