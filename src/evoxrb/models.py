"""Educational photon-spectrum approximations for the synthetic case study.

These compact formulae are intentionally transparent teaching models.  They
are not drop-in implementations of any calibrated astronomy-package model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import SYNTHETIC_LABEL


ContinuumKind = Literal["powerlaw", "cutoff"]


def _energy_array(energy: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(energy, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("energy must be a non-empty one-dimensional array")
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("energy values must be finite and strictly positive")
    return values


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not np.isfinite(result) or not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def educational_absorption(
    energy: ArrayLike, nh: float
) -> NDArray[np.float64]:
    """Return ``exp(-3 NH E^-3)`` for positive energy in keV."""

    e = _energy_array(energy)
    column = _positive_finite("NH", nh, allow_zero=True)
    return np.exp(-3.0 * column * np.power(e, -3.0))


def educational_disk_like(
    energy: ArrayLike, tin: float, ndisk: float
) -> NDArray[np.float64]:
    """Return ``1e-4 Ndisk E^(-2/3) exp(-E/Tin)``."""

    e = _energy_array(energy)
    temperature = _positive_finite("Tin", tin)
    normalization = _positive_finite("Ndisk", ndisk)
    return (
        1.0e-4
        * normalization
        * np.power(e, -2.0 / 3.0)
        * np.exp(-e / temperature)
    )


def educational_power_law(
    energy: ArrayLike, gamma: float, normalization: float
) -> NDArray[np.float64]:
    """Return the educational power law ``K E^-Gamma``."""

    e = _energy_array(energy)
    index = _positive_finite("Gamma", gamma)
    norm = _positive_finite("K", normalization)
    return norm * np.power(e, -index)


def educational_cutoff_surrogate(
    energy: ArrayLike,
    gamma: float,
    normalization: float,
    cutoff_energy_keV: float = 20.0,
) -> NDArray[np.float64]:
    """Return ``K E^-Gamma exp(-E/Ecut)`` as a teaching surrogate."""

    e = _energy_array(energy)
    cutoff = _positive_finite("cutoff_energy_keV", cutoff_energy_keV)
    return educational_power_law(e, gamma, normalization) * np.exp(-e / cutoff)


def _lookup(parameters: Mapping[str, float], *aliases: str) -> float:
    lower = {str(name).casefold(): value for name, value in parameters.items()}
    for alias in aliases:
        if alias.casefold() in lower:
            return float(lower[alias.casefold()])
    choices = ", ".join(aliases)
    raise KeyError(f"missing model parameter; expected one of: {choices}")


@dataclass(frozen=True, slots=True)
class SpectrumModel:
    """Absorbed disk-like plus power-law/cutoff educational spectrum.

    Set ``fixed_nh=None`` for the documented free-column sensitivity fit.
    Parameter mappings may use the case-study names ``Tin``, ``Ndisk``,
    ``Gamma``, ``K``, and ``NH`` or their lower-case Python spellings.
    """

    continuum: ContinuumKind = "powerlaw"
    fixed_nh: float | None = 0.15
    cutoff_energy_keV: float = 20.0
    label: str = SYNTHETIC_LABEL

    def __post_init__(self) -> None:
        if self.continuum not in ("powerlaw", "cutoff"):
            raise ValueError("continuum must be 'powerlaw' or 'cutoff'")
        if self.fixed_nh is not None:
            _positive_finite("fixed_nh", self.fixed_nh, allow_zero=True)
        _positive_finite("cutoff_energy_keV", self.cutoff_energy_keV)
        if not self.label.startswith(SYNTHETIC_LABEL):
            raise ValueError(f"label must start with {SYNTHETIC_LABEL!r}")

    @property
    def name(self) -> str:
        continuum_name = (
            "power law"
            if self.continuum == "powerlaw"
            else "20-keV Comptonization cutoff surrogate"
        )
        return f"{self.label} educational absorption × (disk-like + {continuum_name})"

    def components(
        self, energy: ArrayLike, parameters: Mapping[str, float]
    ) -> dict[str, NDArray[np.float64]]:
        """Evaluate absorption and the two unabsorbed additive components."""

        e = _energy_array(energy)
        tin = _lookup(parameters, "Tin", "tin")
        ndisk = _lookup(parameters, "Ndisk", "ndisk", "disk_norm")
        gamma = _lookup(parameters, "Gamma", "gamma")
        normalization = _lookup(
            parameters, "K", "k", "norm", "powerlaw_norm", "normalization"
        )
        nh = (
            float(self.fixed_nh)
            if self.fixed_nh is not None
            else _lookup(parameters, "NH", "nh")
        )

        absorption = educational_absorption(e, nh)
        disk = educational_disk_like(e, tin, ndisk)
        if self.continuum == "powerlaw":
            continuum = educational_power_law(e, gamma, normalization)
        else:
            continuum = educational_cutoff_surrogate(
                e, gamma, normalization, self.cutoff_energy_keV
            )
        return {"absorption": absorption, "disk_like": disk, "continuum": continuum}

    def evaluate(
        self, energy: ArrayLike, parameters: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """Evaluate the absorbed photon flux on an energy-centre grid."""

        parts = self.components(energy, parameters)
        flux = parts["absorption"] * (parts["disk_like"] + parts["continuum"])
        if not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
            raise ValueError("model evaluation produced invalid photon flux")
        return flux


# Short aliases are convenient in examples while retaining explicit educational
# names in documentation and plot labels.
absorption = educational_absorption
disk_like = educational_disk_like
power_law = educational_power_law
cutoff_surrogate = educational_cutoff_surrogate
educational_comptonization_surrogate = educational_cutoff_surrogate
