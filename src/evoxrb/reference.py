"""User-supplied reference spectra for visual comparison only.

EvoXRB does not bundle or reduce observational data.  This module provides a
small, response-agnostic container for plotting a user-supplied spectrum next
to an EvoXRB result.  A :class:`ReferenceSpectrum` deliberately is not a
``SyntheticSpectrum`` and is not accepted by the Poisson C-statistic objective:
meaningful fitting of observed counts requires the matching response,
background, exposure, and calibration products.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd


DEFAULT_REFERENCE_LABEL = "User-supplied reference spectrum"
VISUAL_COMPARISON_NOTICE = (
    "Visual comparison only; this reference spectrum is not used for C-stat "
    "fitting without its matching instrument response and calibration products."
)


def _finite_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


@dataclass(frozen=True, slots=True)
class ReferenceSpectrum:
    """A validated spectrum supplied by the user for plotting.

    The energy grid is kept at its native sampling and sorted into increasing
    order.  No interpolation, response folding, or statistical comparison is
    performed.  ``count_rate_density`` and its optional uncertainty are in
    counts per second per keV.
    """

    energy_keV: NDArray[np.float64]
    count_rate_density: NDArray[np.float64]
    count_rate_error: NDArray[np.float64] | None = None
    label: str = DEFAULT_REFERENCE_LABEL
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energy = _finite_vector("energy_keV", self.energy_keV)
        rate = _finite_vector("count_rate_density", self.count_rate_density)
        if rate.shape != energy.shape:
            raise ValueError("count_rate_density must have one value per energy bin")
        if np.any(energy <= 0.0):
            raise ValueError("energy_keV must be strictly positive")
        if np.any(rate < 0.0):
            raise ValueError("count_rate_density cannot be negative")

        error: NDArray[np.float64] | None = None
        if self.count_rate_error is not None:
            error = _finite_vector("count_rate_error", self.count_rate_error)
            if error.shape != energy.shape:
                raise ValueError("count_rate_error must have one value per energy bin")
            if np.any(error < 0.0):
                raise ValueError("count_rate_error cannot be negative")

        order = np.argsort(energy, kind="stable")
        energy = energy[order]
        rate = rate[order]
        if error is not None:
            error = error[order]
        if np.any(np.diff(energy) <= 0.0):
            raise ValueError("energy_keV values must be unique")

        label = str(self.label).strip()
        if not label:
            raise ValueError("label must not be empty")
        source = None if self.source is None else str(self.source).strip()
        if source == "":
            raise ValueError("source must not be empty when supplied")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        metadata: dict[str, Any] = {}
        for key, value in self.metadata.items():
            name = str(key).strip()
            if not name:
                raise ValueError("metadata keys must not be empty")
            metadata[name] = value.item() if isinstance(value, np.generic) else value

        # The frozen, read-only arrays make a ReferenceSpectrum safe to share
        # with live animation callbacks.  Plot helpers return writable copies.
        energy.setflags(write=False)
        rate.setflags(write=False)
        if error is not None:
            error.setflags(write=False)
        object.__setattr__(self, "energy_keV", energy)
        object.__setattr__(self, "count_rate_density", rate)
        object.__setattr__(self, "count_rate_error", error)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def comparison_notice(self) -> str:
        """State the intentionally limited scientific scope of this object."""

        return VISUAL_COMPARISON_NOTICE

    @property
    def size(self) -> int:
        return int(self.energy_keV.size)

    @property
    def has_errors(self) -> bool:
        return self.count_rate_error is not None

    @property
    def energy(self) -> NDArray[np.float64]:
        """Read-only alias for the native energy grid."""

        return self.energy_keV

    def as_plot_data(self) -> dict[str, Any]:
        """Return copies ready to unpack into ``matplotlib.Axes.errorbar``."""

        return {
            "x": self.energy_keV.copy(),
            "y": self.count_rate_density.copy(),
            "yerr": None
            if self.count_rate_error is None
            else self.count_rate_error.copy(),
            "label": self.label,
        }

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        label: str | None = None,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReferenceSpectrum:
        """Load :class:`ReferenceSpectrum` from a documented CSV schema."""

        return load_reference_spectrum_csv(
            path, label=label, source=source, metadata=metadata
        )


def _normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    columns: dict[str, str] = {}
    for raw_name in frame.columns:
        name = str(raw_name).strip()
        normalized = name.casefold()
        if not normalized:
            raise ValueError("CSV column names must not be empty")
        if normalized in columns:
            raise ValueError(f"duplicate CSV column after normalization: {name!r}")
        columns[normalized] = str(raw_name)
    return columns


def _numeric_column(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"CSV column {column!r} must contain only finite numbers")
    return values


def _constant_text_column(frame: pd.DataFrame, column: str | None) -> str | None:
    if column is None:
        return None
    values = {
        str(value).strip()
        for value in frame[column].tolist()
        if not pd.isna(value) and str(value).strip()
    }
    if len(values) > 1:
        raise ValueError(f"CSV column {column!r} must contain one constant value")
    return next(iter(values), None)


def _error_column(columns: Mapping[str, str]) -> str | None:
    aliases = ("count_rate_error", "count_rate_density_error")
    matches = [columns[name] for name in aliases if name in columns]
    if len(matches) > 1:
        raise ValueError(
            "CSV must not contain both count_rate_error and "
            "count_rate_density_error"
        )
    return matches[0] if matches else None


def load_reference_spectrum_csv(
    path: str | Path,
    *,
    label: str | None = None,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ReferenceSpectrum:
    """Load a user-supplied spectrum for visual comparison.

    The preferred schema contains ``energy_keV`` and
    ``count_rate_density``.  An optional uncertainty column may be named
    ``count_rate_error`` or ``count_rate_density_error``.

    A counts-based CSV is also accepted when it contains ``counts``,
    ``exposure_s``, and either ``bin_width_keV`` or both ``energy_low_keV`` and
    ``energy_high_keV``. Counts and conservative square-root display
    uncertainties ``sqrt(max(counts, 1))`` are converted to rate density.
    Optional ``counts_error`` overrides that uncertainty. In either schema,
    ``label`` and ``source`` may be constant CSV columns or explicit keyword
    arguments.

    This loader does not establish that the input is calibrated observational
    data and never makes it eligible for EvoXRB's C-statistic objective.
    """

    destination = Path(path)
    if not destination.exists():
        raise FileNotFoundError(destination)
    try:
        frame = pd.read_csv(destination)
    except pd.errors.EmptyDataError as error:
        raise ValueError("reference spectrum CSV is empty") from error
    if frame.empty:
        raise ValueError("reference spectrum CSV has no data rows")
    columns = _normalized_columns(frame)
    if "energy_kev" not in columns:
        raise ValueError("reference spectrum CSV requires an energy_keV column")

    energy = _numeric_column(frame, columns["energy_kev"])
    direct_error = _error_column(columns)
    error = (
        _numeric_column(frame, direct_error) if direct_error is not None else None
    )

    if "count_rate_density" in columns:
        rate = _numeric_column(frame, columns["count_rate_density"])
        representation = "count_rate_density"
    elif "counts" in columns:
        required = [name for name in ("exposure_s",) if name not in columns]
        if required:
            raise ValueError(
                "counts-based CSV requires exposure_s and bin-width information"
            )
        counts = _numeric_column(frame, columns["counts"])
        if np.any(counts < 0.0):
            raise ValueError("CSV counts cannot be negative")
        exposure = _numeric_column(frame, columns["exposure_s"])
        if "bin_width_kev" in columns:
            width = _numeric_column(frame, columns["bin_width_kev"])
        elif "energy_low_kev" in columns and "energy_high_kev" in columns:
            lower = _numeric_column(frame, columns["energy_low_kev"])
            upper = _numeric_column(frame, columns["energy_high_kev"])
            width = upper - lower
            if np.any((energy < lower) | (energy > upper)):
                raise ValueError("energy_keV must lie within each supplied bin")
        else:
            raise ValueError(
                "counts-based CSV requires bin_width_keV or energy_low_keV/"
                "energy_high_keV"
            )
        if np.any(exposure <= 0.0) or np.any(width <= 0.0):
            raise ValueError("exposure_s and bin widths must be strictly positive")
        scale = exposure * width
        rate = counts / scale
        if error is None:
            if "counts_error" in columns:
                counts_error = _numeric_column(frame, columns["counts_error"])
                if np.any(counts_error < 0.0):
                    raise ValueError("CSV counts_error cannot be negative")
                error = counts_error / scale
            else:
                error = np.sqrt(np.maximum(counts, 1.0)) / scale
        representation = "counts"
    else:
        raise ValueError(
            "reference spectrum CSV requires count_rate_density, or counts with "
            "exposure_s and bin-width information"
        )

    csv_label = _constant_text_column(frame, columns.get("label"))
    csv_source = _constant_text_column(frame, columns.get("source"))
    resolved_label = label if label is not None else csv_label
    resolved_source = source if source is not None else csv_source
    if resolved_source is None:
        resolved_source = str(destination.resolve())
    details = dict(metadata or {})
    details["input_path"] = str(destination.resolve())
    details["input_representation"] = representation
    details["visual_comparison_only"] = True

    return ReferenceSpectrum(
        energy_keV=energy,
        count_rate_density=rate,
        count_rate_error=error,
        label=DEFAULT_REFERENCE_LABEL if resolved_label is None else resolved_label,
        source=resolved_source,
        metadata=details,
    )


# Short, discoverable alias for callers that already know the input is a CSV.
load_reference_csv = load_reference_spectrum_csv


__all__ = [
    "DEFAULT_REFERENCE_LABEL",
    "ReferenceSpectrum",
    "VISUAL_COMPARISON_NOTICE",
    "load_reference_csv",
    "load_reference_spectrum_csv",
]
