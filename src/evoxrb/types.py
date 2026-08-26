"""Typed data containers used by the synthetic EvoXRB case study.

The classes in this module deliberately contain only NumPy-friendly data.  In
particular, their :meth:`to_npz_dict` methods avoid object arrays, so archives
can be read with ``allow_pickle=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


SYNTHETIC_LABEL = "Synthetic / NICER-inspired"


def _scalar(value: Any) -> Any:
    """Return a Python scalar from an NPZ scalar or one-element array."""

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar value")
    return array.reshape(()).item()


@dataclass(frozen=True, slots=True)
class EpochTruth:
    """Injected parameters for one epoch of the fixed synthetic outburst.

    The dates are reference anchors only.  They are not observations, and the
    parameter values are not measurements of a real X-ray binary.
    """

    epoch_id: str
    phase: str
    reference_mjd: float
    tin: float
    ndisk: float
    gamma: float
    powerlaw_norm: float
    qpo_hz: float | None
    exposure_s: float = 2048.0
    nh: float = 0.15
    label: str = SYNTHETIC_LABEL

    def __post_init__(self) -> None:
        if not self.epoch_id:
            raise ValueError("epoch_id must not be empty")
        if not self.phase:
            raise ValueError("phase must not be empty")
        for name in (
            "reference_mjd",
            "tin",
            "ndisk",
            "gamma",
            "powerlaw_norm",
            "exposure_s",
            "nh",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.tin <= 0.0 or self.ndisk <= 0.0 or self.powerlaw_norm <= 0.0:
            raise ValueError("spectral temperatures and normalizations must be positive")
        if self.exposure_s <= 0.0 or self.nh < 0.0:
            raise ValueError("exposure_s must be positive and nh must be non-negative")
        if self.qpo_hz is not None and (
            not np.isfinite(self.qpo_hz) or self.qpo_hz <= 0.0
        ):
            raise ValueError("qpo_hz must be positive when supplied")
        if not self.label.startswith(SYNTHETIC_LABEL):
            raise ValueError(f"label must start with {SYNTHETIC_LABEL!r}")

    @property
    def parameters(self) -> dict[str, float]:
        """Canonical parameter mapping accepted by :class:`SpectrumModel`."""

        return {
            "Tin": float(self.tin),
            "Ndisk": float(self.ndisk),
            "Gamma": float(self.gamma),
            "K": float(self.powerlaw_norm),
            "NH": float(self.nh),
        }

    # Read-only aliases make table-style and Python-style access equally easy.
    @property
    def epoch(self) -> str:
        return self.epoch_id

    @property
    def Tin(self) -> float:  # noqa: N802 - matches the case-study notation
        return self.tin

    @property
    def Ndisk(self) -> float:  # noqa: N802 - matches the case-study notation
        return self.ndisk

    @property
    def K(self) -> float:  # noqa: N802 - matches the case-study notation
        return self.powerlaw_norm

    def to_record(self) -> dict[str, str | float | None]:
        """Return a flat record suitable for CSV or JSON output."""

        return {
            "label": self.label,
            "epoch": self.epoch_id,
            "phase": self.phase,
            "reference_mjd": self.reference_mjd,
            "Tin": self.tin,
            "Ndisk": self.ndisk,
            "Gamma": self.gamma,
            "K": self.powerlaw_norm,
            "NH": self.nh,
            "qpo_hz": self.qpo_hz,
            "exposure_s": self.exposure_s,
        }


@dataclass(slots=True)
class SyntheticSpectrum:
    """One response-folded, Poisson-sampled synthetic count spectrum."""

    detector_energy: NDArray[np.float64]
    detector_edges: NDArray[np.float64]
    counts: NDArray[np.int64]
    expected_counts: NDArray[np.float64]
    source_expected_counts: NDArray[np.float64]
    background_expected_counts: NDArray[np.float64]
    fit_mask: NDArray[np.bool_]
    exposure_s: float
    truth_parameters: Mapping[str, float]
    seed: int
    epoch_id: str = "custom"
    phase: str = "synthetic"
    reference_mjd: float | None = None
    truth_model: str = "educational absorbed disk-like + power law"
    label: str = SYNTHETIC_LABEL

    def __post_init__(self) -> None:
        self.detector_energy = np.asarray(self.detector_energy, dtype=np.float64)
        self.detector_edges = np.asarray(self.detector_edges, dtype=np.float64)
        self.counts = np.asarray(self.counts, dtype=np.int64)
        self.expected_counts = np.asarray(self.expected_counts, dtype=np.float64)
        self.source_expected_counts = np.asarray(
            self.source_expected_counts, dtype=np.float64
        )
        self.background_expected_counts = np.asarray(
            self.background_expected_counts, dtype=np.float64
        )
        self.fit_mask = np.asarray(self.fit_mask, dtype=np.bool_)
        self.truth_parameters = {
            str(name): float(value) for name, value in self.truth_parameters.items()
        }

        n_channels = self.detector_energy.size
        if self.detector_energy.ndim != 1 or n_channels == 0:
            raise ValueError("detector_energy must be a non-empty one-dimensional array")
        if self.detector_edges.shape != (n_channels + 1,):
            raise ValueError("detector_edges must have one more entry than detector_energy")
        for name in (
            "counts",
            "expected_counts",
            "source_expected_counts",
            "background_expected_counts",
            "fit_mask",
        ):
            if getattr(self, name).shape != (n_channels,):
                raise ValueError(f"{name} must have one value per detector channel")
        if np.any(self.counts < 0):
            raise ValueError("counts cannot be negative")
        if not np.all(np.isfinite(self.expected_counts)) or np.any(
            self.expected_counts < 0.0
        ):
            raise ValueError("expected_counts must be finite and non-negative")
        if not np.allclose(
            self.expected_counts,
            self.source_expected_counts + self.background_expected_counts,
            rtol=2e-12,
            atol=1e-12,
        ):
            raise ValueError("expected_counts must equal source plus background counts")
        if not np.isfinite(self.exposure_s) or self.exposure_s <= 0.0:
            raise ValueError("exposure_s must be positive")
        if not self.label.startswith(SYNTHETIC_LABEL):
            raise ValueError(f"label must start with {SYNTHETIC_LABEL!r}")

    @property
    def energy(self) -> NDArray[np.float64]:
        """Alias for detector-channel centres."""

        return self.detector_energy

    @property
    def fitted_counts(self) -> NDArray[np.int64]:
        return self.counts[self.fit_mask]

    def to_npz_dict(self) -> dict[str, NDArray[Any]]:
        """Return a pickle-free mapping accepted by ``numpy.savez``."""

        parameter_names = tuple(sorted(self.truth_parameters))
        parameter_values = [self.truth_parameters[name] for name in parameter_names]
        return {
            "detector_energy": self.detector_energy,
            "detector_edges": self.detector_edges,
            "counts": self.counts,
            "expected_counts": self.expected_counts,
            "source_expected_counts": self.source_expected_counts,
            "background_expected_counts": self.background_expected_counts,
            "fit_mask": self.fit_mask,
            "exposure_s": np.asarray(self.exposure_s, dtype=np.float64),
            "truth_parameter_names": np.asarray(parameter_names, dtype=np.str_),
            "truth_parameter_values": np.asarray(parameter_values, dtype=np.float64),
            "seed": np.asarray(self.seed, dtype=np.uint64),
            "epoch_id": np.asarray(self.epoch_id, dtype=np.str_),
            "phase": np.asarray(self.phase, dtype=np.str_),
            "reference_mjd": np.asarray(
                np.nan if self.reference_mjd is None else self.reference_mjd,
                dtype=np.float64,
            ),
            "truth_model": np.asarray(self.truth_model, dtype=np.str_),
            "label": np.asarray(self.label, dtype=np.str_),
        }

    @classmethod
    def from_npz_dict(cls, archive: Mapping[str, Any]) -> SyntheticSpectrum:
        """Reconstruct a spectrum from an NPZ-like mapping."""

        names = np.asarray(archive["truth_parameter_names"], dtype=np.str_)
        values = np.asarray(archive["truth_parameter_values"], dtype=np.float64)
        if names.shape != values.shape:
            raise ValueError("truth parameter name/value arrays must have equal shapes")
        reference_mjd = float(_scalar(archive["reference_mjd"]))
        return cls(
            detector_energy=np.asarray(archive["detector_energy"], dtype=np.float64),
            detector_edges=np.asarray(archive["detector_edges"], dtype=np.float64),
            counts=np.asarray(archive["counts"], dtype=np.int64),
            expected_counts=np.asarray(archive["expected_counts"], dtype=np.float64),
            source_expected_counts=np.asarray(
                archive["source_expected_counts"], dtype=np.float64
            ),
            background_expected_counts=np.asarray(
                archive["background_expected_counts"], dtype=np.float64
            ),
            fit_mask=np.asarray(archive["fit_mask"], dtype=np.bool_),
            exposure_s=float(_scalar(archive["exposure_s"])),
            truth_parameters=dict(zip(names.tolist(), values.tolist(), strict=True)),
            seed=int(_scalar(archive["seed"])),
            epoch_id=str(_scalar(archive["epoch_id"])),
            phase=str(_scalar(archive["phase"])),
            reference_mjd=None if np.isnan(reference_mjd) else reference_mjd,
            truth_model=str(_scalar(archive["truth_model"])),
            label=str(_scalar(archive["label"])),
        )


@dataclass(slots=True)
class PosteriorResult:
    """Sampler-independent posterior samples and diagnostics."""

    parameter_names: tuple[str, ...]
    samples: NDArray[np.float64]
    log_probability: NDArray[np.float64]
    sampler: str
    converged: bool
    weights: NDArray[np.float64] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    label: str = SYNTHETIC_LABEL

    def __post_init__(self) -> None:
        self.parameter_names = tuple(str(name) for name in self.parameter_names)
        self.samples = np.asarray(self.samples, dtype=np.float64)
        self.log_probability = np.asarray(self.log_probability, dtype=np.float64)
        if self.samples.ndim != 2:
            raise ValueError("samples must have shape (draws, parameters)")
        if self.samples.shape[1] != len(self.parameter_names):
            raise ValueError("parameter_names must match the sample columns")
        if self.log_probability.shape != (self.samples.shape[0],):
            raise ValueError("log_probability must have one value per posterior draw")
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=np.float64)
            if self.weights.shape != (self.samples.shape[0],):
                raise ValueError("weights must have one value per posterior draw")
            if np.any(self.weights < 0.0) or not np.all(np.isfinite(self.weights)):
                raise ValueError("weights must be finite and non-negative")
        self.diagnostics = dict(self.diagnostics)
        if not self.label.startswith(SYNTHETIC_LABEL):
            raise ValueError(f"label must start with {SYNTHETIC_LABEL!r}")

    def quantiles(
        self, probabilities: tuple[float, ...] = (0.16, 0.5, 0.84)
    ) -> dict[str, NDArray[np.float64]]:
        """Return per-parameter (optionally weighted) posterior quantiles."""

        q = np.asarray(probabilities, dtype=np.float64)
        if np.any((q < 0.0) | (q > 1.0)):
            raise ValueError("quantile probabilities must lie in [0, 1]")
        if self.weights is None:
            values = np.quantile(self.samples, q, axis=0)
        else:
            values = np.empty((q.size, self.samples.shape[1]), dtype=np.float64)
            for column in range(self.samples.shape[1]):
                order = np.argsort(self.samples[:, column])
                sorted_values = self.samples[order, column]
                sorted_weights = self.weights[order]
                cumulative = np.cumsum(sorted_weights)
                if cumulative[-1] <= 0.0:
                    raise ValueError("posterior weights must have a positive sum")
                cumulative /= cumulative[-1]
                values[:, column] = np.interp(q, cumulative, sorted_values)
        return {
            name: values[:, index].copy()
            for index, name in enumerate(self.parameter_names)
        }

    def to_npz_dict(self) -> dict[str, NDArray[Any]]:
        """Return a pickle-free mapping accepted by ``numpy.savez``."""

        try:
            diagnostics_json = json.dumps(self.diagnostics, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("diagnostics must be JSON serializable") from exc
        return {
            "parameter_names": np.asarray(self.parameter_names, dtype=np.str_),
            "samples": self.samples,
            "log_probability": self.log_probability,
            "weights": np.asarray([], dtype=np.float64)
            if self.weights is None
            else self.weights,
            "has_weights": np.asarray(self.weights is not None, dtype=np.bool_),
            "sampler": np.asarray(self.sampler, dtype=np.str_),
            "converged": np.asarray(self.converged, dtype=np.bool_),
            "diagnostics_json": np.asarray(diagnostics_json, dtype=np.str_),
            "label": np.asarray(self.label, dtype=np.str_),
        }

    @classmethod
    def from_npz_dict(cls, archive: Mapping[str, Any]) -> PosteriorResult:
        has_weights = bool(_scalar(archive["has_weights"]))
        return cls(
            parameter_names=tuple(
                np.asarray(archive["parameter_names"], dtype=np.str_).tolist()
            ),
            samples=np.asarray(archive["samples"], dtype=np.float64),
            log_probability=np.asarray(archive["log_probability"], dtype=np.float64),
            weights=np.asarray(archive["weights"], dtype=np.float64)
            if has_weights
            else None,
            sampler=str(_scalar(archive["sampler"])),
            converged=bool(_scalar(archive["converged"])),
            diagnostics=json.loads(str(_scalar(archive["diagnostics_json"]))),
            label=str(_scalar(archive["label"])),
        )
