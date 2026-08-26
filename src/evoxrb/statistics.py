"""Likelihood statistics used by the synthetic spectroscopy case study."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cstat_contributions(observed: ArrayLike, expected: ArrayLike) -> NDArray[np.float64]:
    """Return per-bin Poisson C-statistic contributions.

    The data-dependent constant is retained, so the statistic is the Poisson
    deviance ``2 * (m - d + d*log(d/m))``.  For a zero-count bin the limiting
    contribution is evaluated analytically as ``2*m``.  A non-positive or
    non-finite model is invalid and produces all-``inf`` contributions.

    Raises
    ------
    ValueError
        If shapes differ or the observed counts are negative/non-finite.
    """

    data = np.asarray(observed, dtype=np.float64)
    model = np.asarray(expected, dtype=np.float64)
    if data.shape != model.shape:
        raise ValueError("observed and expected arrays must have identical shapes")
    if np.any(~np.isfinite(data)) or np.any(data < 0.0):
        raise ValueError("observed counts must be finite and non-negative")
    if np.any(~np.isfinite(model)) or np.any(model <= 0.0):
        return np.full(data.shape, np.inf, dtype=np.float64)

    result = np.empty_like(model, dtype=np.float64)
    zero = data == 0.0
    result[zero] = 2.0 * model[zero]
    positive = ~zero
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result[positive] = 2.0 * (
            model[positive]
            - data[positive]
            + data[positive] * np.log(data[positive] / model[positive])
        )
    # Tiny negative values can appear through cancellation at d == m.
    result[np.logical_and(result < 0.0, result > -1.0e-10)] = 0.0
    return result


def poisson_cstat(observed: ArrayLike, expected: ArrayLike) -> float:
    """Compute the robust Poisson C-statistic for count data."""

    contributions = cstat_contributions(observed, expected)
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.sum(contributions, dtype=np.float64))
    return value if np.isfinite(value) else float("inf")


# Descriptive aliases used by reports and tests.
cash_statistic = poisson_cstat
c_statistic = poisson_cstat


def reduced_deviance(
    observed: ArrayLike,
    expected: ArrayLike,
    fitted_parameters: int,
    *,
    mask: Sequence[bool] | np.ndarray | None = None,
) -> float:
    """Return C-stat divided by the nominal degrees of freedom."""

    data = np.asarray(observed)
    model = np.asarray(expected)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != data.shape:
            raise ValueError("mask must have the same shape as the count arrays")
        data = data[selected]
        model = model[selected]
    dof = int(data.size) - int(fitted_parameters)
    if dof <= 0:
        raise ValueError("degrees of freedom must be positive")
    return poisson_cstat(data, model) / dof

