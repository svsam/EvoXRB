from __future__ import annotations

import numpy as np
import pytest

from evoxrb.parameters import ParameterSpec, SearchSpace
from evoxrb.statistics import cstat_contributions, poisson_cstat


def test_linear_and_log10_parameter_round_trips() -> None:
    linear = ParameterSpec("temperature", 0.1, 1.1)
    logarithmic = ParameterSpec("normalization", 1.0e-3, 1.0e3, "log10")

    assert linear.decode(0.25) == pytest.approx(0.35)
    assert linear.encode(0.35) == pytest.approx(0.25)
    assert logarithmic.decode(0.5) == pytest.approx(1.0)
    assert logarithmic.encode(1.0) == pytest.approx(0.5)

    space = SearchSpace(linear, logarithmic)
    physical = {"temperature": 0.65, "normalization": 100.0}
    assert space.decode(space.encode(physical)) == pytest.approx(physical)


def test_parameter_validation_bounds_and_clipping() -> None:
    with pytest.raises(ValueError, match="positive bounds"):
        ParameterSpec("bad_log", 0.0, 10.0, "log10")
    with pytest.raises(ValueError, match="lower bound"):
        ParameterSpec("reversed", 2.0, 1.0)

    parameter = ParameterSpec("x", -2.0, 2.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parameter.decode(1.01)
    with pytest.raises(ValueError, match="must lie"):
        parameter.encode(3.0)
    assert parameter.decode(2.0, clip=True) == 2.0
    assert parameter.encode(3.0, clip=True) == 1.0


def test_cstat_handles_zero_counts_analytically() -> None:
    observed = np.array([0.0, 2.0, 10.0])
    expected = np.array([1.5, 2.0, 10.0])

    contributions = cstat_contributions(observed, expected)
    assert contributions == pytest.approx([3.0, 0.0, 0.0], abs=1.0e-14)
    assert poisson_cstat(observed, expected) == pytest.approx(3.0)


@pytest.mark.parametrize(
    "expected",
    [
        np.array([1.0, 0.0]),
        np.array([1.0, -0.1]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_cstat_invalid_models_return_infinity(expected: np.ndarray) -> None:
    assert np.isinf(poisson_cstat(np.array([0.0, 1.0]), expected))


def test_cstat_rejects_invalid_observations_and_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        poisson_cstat([0.0, -1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="identical shapes"):
        poisson_cstat([1.0], [1.0, 2.0])

