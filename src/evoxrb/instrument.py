"""Synthetic, NICER-inspired response and background construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from .types import SYNTHETIC_LABEL


AREA_KNOT_ENERGY_KEV = np.asarray(
    [0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    dtype=np.float64,
)
AREA_KNOT_CM2 = np.asarray(
    [0.0, 200.0, 900.0, 1700.0, 1900.0, 1800.0, 1300.0, 800.0, 400.0, 150.0, 0.0],
    dtype=np.float64,
)


def nicer_inspired_effective_area(energy: ArrayLike) -> NDArray[np.float64]:
    """Shape-preserving interpolation of the fixed synthetic area anchors."""

    values = np.asarray(energy, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("energy values must be finite")
    interpolation = PchipInterpolator(
        AREA_KNOT_ENERGY_KEV, AREA_KNOT_CM2, extrapolate=False
    )
    area = np.asarray(interpolation(values), dtype=np.float64)
    in_range = (values >= AREA_KNOT_ENERGY_KEV[0]) & (
        values <= AREA_KNOT_ENERGY_KEV[-1]
    )
    area = np.where(in_range, area, 0.0)
    return np.clip(area, 0.0, None)


def nicer_inspired_background_density(energy: ArrayLike) -> NDArray[np.float64]:
    """Background density in counts s^-1 keV^-1 at detector energy."""

    values = np.asarray(energy, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("energy values must be finite")
    return 0.02 + 0.005 * values + 0.03 * np.exp(
        -np.square(values - 8.0) / (2.0 * 0.5**2)
    )


def gaussian_redistribution(
    true_energy: ArrayLike,
    detector_energy: ArrayLike,
    detector_width: ArrayLike,
) -> NDArray[np.float64]:
    """Build a column-normalized Gaussian redistribution matrix.

    The fixed resolution is ``FWHM(E) = 0.085 sqrt(E)`` keV.  Midpoint
    quadrature over each detector channel is sufficient for this educational
    response.  Subtracting the largest log-weight in each column keeps even
    out-of-band tails numerically normalizable.
    """

    true_e = np.asarray(true_energy, dtype=np.float64)
    detected_e = np.asarray(detector_energy, dtype=np.float64)
    widths = np.asarray(detector_width, dtype=np.float64)
    if true_e.ndim != 1 or detected_e.ndim != 1 or widths.ndim != 1:
        raise ValueError("energy grids and detector_width must be one-dimensional")
    if detected_e.shape != widths.shape:
        raise ValueError("detector energy and width arrays must have equal shapes")
    if (
        np.any(true_e <= 0.0)
        or np.any(widths <= 0.0)
        or not np.all(np.isfinite(true_e))
        or not np.all(np.isfinite(detected_e))
        or not np.all(np.isfinite(widths))
    ):
        raise ValueError("response grids must contain valid positive widths/energies")

    sigma = 0.085 * np.sqrt(true_e) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    residual = (detected_e[:, None] - true_e[None, :]) / sigma[None, :]
    log_weight = -0.5 * np.square(residual) + np.log(widths[:, None])
    log_weight -= np.max(log_weight, axis=0, keepdims=True)
    redistribution = np.exp(log_weight)
    redistribution /= np.sum(redistribution, axis=0, keepdims=True)
    return redistribution


@dataclass(slots=True)
class InstrumentResponse:
    """Response operator for the synthetic, NICER-inspired experiment."""

    true_edges: NDArray[np.float64]
    true_energy: NDArray[np.float64]
    detector_edges: NDArray[np.float64]
    detector_energy: NDArray[np.float64]
    effective_area: NDArray[np.float64]
    redistribution: NDArray[np.float64]
    background_rate_density: NDArray[np.float64]
    label: str = SYNTHETIC_LABEL

    def __post_init__(self) -> None:
        for name in (
            "true_edges",
            "true_energy",
            "detector_edges",
            "detector_energy",
            "effective_area",
            "redistribution",
            "background_rate_density",
        ):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        n_true = self.true_energy.size
        n_detector = self.detector_energy.size
        if self.true_energy.ndim != 1 or self.detector_energy.ndim != 1:
            raise ValueError("energy centres must be one-dimensional")
        if self.true_edges.shape != (n_true + 1,):
            raise ValueError("true_edges must have one more element than true_energy")
        if self.detector_edges.shape != (n_detector + 1,):
            raise ValueError(
                "detector_edges must have one more element than detector_energy"
            )
        if self.effective_area.shape != (n_true,):
            raise ValueError("effective_area must have one value per true-energy bin")
        if self.background_rate_density.shape != (n_detector,):
            raise ValueError(
                "background_rate_density must have one value per detector channel"
            )
        if self.redistribution.shape != (n_detector, n_true):
            raise ValueError("redistribution shape must be (detector, true energy)")
        if np.any(np.diff(self.true_edges) <= 0.0) or np.any(
            np.diff(self.detector_edges) <= 0.0
        ):
            raise ValueError("response edges must be strictly increasing")
        if np.any(self.effective_area < 0.0) or np.any(
            self.background_rate_density < 0.0
        ):
            raise ValueError("area and background values cannot be negative")
        if not np.all(np.isfinite(self.redistribution)) or np.any(
            self.redistribution < 0.0
        ):
            raise ValueError("redistribution must be finite and non-negative")
        if not np.allclose(
            self.redistribution.sum(axis=0), 1.0, rtol=2e-13, atol=2e-13
        ):
            raise ValueError("every redistribution column must sum to one")
        if not self.label.startswith(SYNTHETIC_LABEL):
            raise ValueError(f"label must start with {SYNTHETIC_LABEL!r}")

    @property
    def true_width(self) -> NDArray[np.float64]:
        return np.diff(self.true_edges)

    @property
    def detector_width(self) -> NDArray[np.float64]:
        return np.diff(self.detector_edges)

    @property
    def background_rate(self) -> NDArray[np.float64]:
        """Background rate integrated within each detector channel."""

        return self.background_rate_density * self.detector_width

    @property
    def fit_mask(self) -> NDArray[np.bool_]:
        """Mask selecting detector centres in the fixed 0.5--10 keV fit band."""

        return (self.detector_energy >= 0.5) & (self.detector_energy <= 10.0)

    @classmethod
    def default(cls) -> InstrumentResponse:
        """Construct the fixed 600-bin by 236-channel educational response."""

        return default_nicer_inspired_response()

    def fold_components(
        self, photon_flux: ArrayLike, exposure: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return expected source and background counts separately."""

        flux = np.asarray(photon_flux, dtype=np.float64)
        exposure_s = float(exposure)
        if flux.shape != self.true_energy.shape:
            raise ValueError("photon_flux must have one value per true-energy bin")
        if not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
            raise ValueError("photon_flux must be finite and non-negative")
        if not np.isfinite(exposure_s) or exposure_s <= 0.0:
            raise ValueError("exposure must be finite and positive")

        incident_rate = self.effective_area * flux * self.true_width
        source_counts = exposure_s * (self.redistribution @ incident_rate)
        background_counts = exposure_s * self.background_rate
        return source_counts, background_counts

    def fold(self, photon_flux: ArrayLike, exposure: float) -> NDArray[np.float64]:
        """Fold photon flux to expected detector counts, including background."""

        source_counts, background_counts = self.fold_components(photon_flux, exposure)
        return source_counts + background_counts

    def to_npz_dict(self) -> dict[str, NDArray[Any]]:
        """Return a pickle-free mapping accepted by ``numpy.savez``."""

        return {
            "true_edges": self.true_edges,
            "true_energy": self.true_energy,
            "detector_edges": self.detector_edges,
            "detector_energy": self.detector_energy,
            "effective_area": self.effective_area,
            "redistribution": self.redistribution,
            "background_rate_density": self.background_rate_density,
            "label": np.asarray(self.label, dtype=np.str_),
        }

    @classmethod
    def from_npz_dict(cls, archive: Mapping[str, Any]) -> InstrumentResponse:
        label_value = np.asarray(archive["label"])
        if label_value.size != 1:
            raise ValueError("label must be scalar")
        return cls(
            true_edges=np.asarray(archive["true_edges"], dtype=np.float64),
            true_energy=np.asarray(archive["true_energy"], dtype=np.float64),
            detector_edges=np.asarray(archive["detector_edges"], dtype=np.float64),
            detector_energy=np.asarray(archive["detector_energy"], dtype=np.float64),
            effective_area=np.asarray(archive["effective_area"], dtype=np.float64),
            redistribution=np.asarray(archive["redistribution"], dtype=np.float64),
            background_rate_density=np.asarray(
                archive["background_rate_density"], dtype=np.float64
            ),
            label=str(label_value.reshape(()).item()),
        )


def default_nicer_inspired_response() -> InstrumentResponse:
    """Construct the case study's fixed synthetic response."""

    true_edges = np.linspace(0.1, 12.0, 601, dtype=np.float64)
    detector_edges = np.linspace(0.2, 12.0, 237, dtype=np.float64)
    true_energy = 0.5 * (true_edges[:-1] + true_edges[1:])
    detector_energy = 0.5 * (detector_edges[:-1] + detector_edges[1:])
    detector_width = np.diff(detector_edges)
    redistribution = gaussian_redistribution(
        true_energy, detector_energy, detector_width
    )
    return InstrumentResponse(
        true_edges=true_edges,
        true_energy=true_energy,
        detector_edges=detector_edges,
        detector_energy=detector_energy,
        effective_area=nicer_inspired_effective_area(true_energy),
        redistribution=redistribution,
        background_rate_density=nicer_inspired_background_density(detector_energy),
        label=f"{SYNTHETIC_LABEL} educational response",
    )


build_nicer_inspired_response = default_nicer_inspired_response
