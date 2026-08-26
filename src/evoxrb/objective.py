"""Response-folded Poisson objectives used by all spectral fitters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import CaseStudyConfig
from .instrument import InstrumentResponse
from .models import SpectrumModel
from .parameters import ParameterSpec, SearchSpace
from .statistics import poisson_cstat
from .types import SyntheticSpectrum


@dataclass(slots=True)
class Objective:
    """Poisson C-statistic for one synthetic spectrum and fit model.

    The model is evaluated on the response's true-energy grid and folded into
    detector space before the fixed 0.5--10 keV mask is applied. Invalid model
    values return ``+inf`` so optimizers can safely reject them.
    """

    spectrum: SyntheticSpectrum
    response: InstrumentResponse
    model: SpectrumModel

    def __post_init__(self) -> None:
        if not np.array_equal(self.spectrum.detector_energy, self.response.detector_energy):
            raise ValueError("spectrum and response detector grids do not match")

    def expected_counts(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        """Return source-plus-background detector counts for ``parameters``."""

        flux = self.model.evaluate(self.response.true_energy, parameters)
        return self.response.fold(flux, self.spectrum.exposure_s)

    def evaluate(self, parameters: Mapping[str, float]) -> float:
        """Evaluate the Poisson C-statistic in the configured fit band."""

        try:
            expected = self.expected_counts(parameters)
            mask = self.spectrum.fit_mask
            return poisson_cstat(self.spectrum.counts[mask], expected[mask])
        except (ArithmeticError, KeyError, TypeError, ValueError, FloatingPointError):
            return float("inf")

    __call__ = evaluate


SpectralObjective = Objective


def search_space_from_config(
    config: CaseStudyConfig | Mapping[str, Any], *, free_nh: bool = False
) -> SearchSpace:
    """Build the normalized fit space from a loaded case-study configuration."""

    search = config.search if isinstance(config, CaseStudyConfig) else dict(config)
    items = list(search["parameters"])
    if free_nh:
        items.append(search["nh_sensitivity"])
    specs = [
        ParameterSpec(
            name=str(item["name"]),
            lower=float(item["lower"]),
            upper=float(item["upper"]),
            scale=str(item.get("transform", "linear")),
        )
        for item in items
    ]
    return SearchSpace(specs)


def model_from_config(
    config: CaseStudyConfig,
    *,
    continuum: str = "powerlaw",
    free_nh: bool = False,
) -> SpectrumModel:
    """Construct an explicitly educational spectral model for fitting."""

    return SpectrumModel(
        continuum=continuum,  # type: ignore[arg-type]
        fixed_nh=None if free_nh else float(config.search["nh_fixed"]),
    )
