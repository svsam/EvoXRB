from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from evoxrb.genetic import GAConfig, GeneticOptimizer
from evoxrb.parameters import ParameterSpec, SearchSpace


Benchmark = Callable[[dict[str, float]], float]


def sphere(parameters: dict[str, float]) -> float:
    return parameters["x"] ** 2 + parameters["y"] ** 2


def rosenbrock(parameters: dict[str, float]) -> float:
    x, y = parameters["x"], parameters["y"]
    return 100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2


def rastrigin(parameters: dict[str, float]) -> float:
    x, y = parameters["x"], parameters["y"]
    return 20.0 + x**2 - 10.0 * np.cos(2.0 * np.pi * x) + y**2 - 10.0 * np.cos(
        2.0 * np.pi * y
    )


@pytest.mark.parametrize(
    ("objective", "bounds", "maximum_score"),
    [
        (sphere, (-5.12, 5.12), 1.0e-3),
        (rosenbrock, (-2.0, 2.0), 1.0e-2),
        (rastrigin, (-5.12, 5.12), 1.0e-2),
    ],
    ids=["sphere", "rosenbrock", "rastrigin"],
)
def test_ga_minimizes_standard_benchmarks(
    objective: Benchmark,
    bounds: tuple[float, float],
    maximum_score: float,
) -> None:
    space = SearchSpace(
        ParameterSpec("x", *bounds),
        ParameterSpec("y", *bounds),
    )
    config = GAConfig(
        population_size=48,
        max_generations=60,
        tournament_size=3,
        min_generations=61,
        mutation_anneal_generations=60,
        immigrant_stagnation_generations=12,
        stop_stagnation_generations=20,
    )

    result = GeneticOptimizer(space, config).optimize(objective, seed=1234)

    assert result.best_score < maximum_score
    assert result.generations == config.max_generations
    assert np.all((result.best_genes >= 0.0) & (result.best_genes <= 1.0))

