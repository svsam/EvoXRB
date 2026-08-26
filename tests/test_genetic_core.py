from __future__ import annotations

import numpy as np
import pytest

from evoxrb.genetic import (
    GAConfig,
    GeneticOptimizer,
    bounded_polynomial_mutation,
    simulated_binary_crossover,
    tournament_select,
)
from evoxrb.optimization import latin_hypercube_starts, local_polish
from evoxrb.parameters import ParameterSpec, SearchSpace


def _quadratic(parameters: dict[str, float]) -> float:
    return (parameters["x"] - 0.4) ** 2 + (parameters["y"] + 0.7) ** 2


def _small_config(max_generations: int = 8) -> GAConfig:
    return GAConfig(
        population_size=24,
        max_generations=max_generations,
        tournament_size=3,
        min_generations=max_generations + 1,
        mutation_anneal_generations=max_generations,
        immigrant_stagnation_generations=4,
        stop_stagnation_generations=4,
    )


def test_genetic_operators_are_seeded_and_bounded() -> None:
    population = np.array([[0.1, 0.2], [0.8, 0.7], [0.4, 0.5]])
    scores = np.array([3.0, 1.0, 2.0])
    rng = np.random.default_rng(17)

    selected = tournament_select(population, scores, 3, rng)
    assert selected == pytest.approx(population[1])

    first, second = simulated_binary_crossover(
        population[0], population[1], rng, probability=1.0, eta=15.0
    )
    mutated = bounded_polynomial_mutation(
        first, rng, probability=1.0, eta=10.0
    )
    for child in (first, second, mutated):
        assert np.all(np.isfinite(child))
        assert np.all((0.0 <= child) & (child <= 1.0))
    # Operators must not mutate caller-owned parents in place.
    assert np.allclose(population, [[0.1, 0.2], [0.8, 0.7], [0.4, 0.5]])


def test_invalid_objective_values_become_infinite_scores() -> None:
    space = SearchSpace(ParameterSpec("x", -1.0, 1.0))
    optimizer = GeneticOptimizer(space, _small_config(max_generations=2))

    result = optimizer.optimize(lambda _: float("nan"), seed=5)

    assert np.isinf(result.best_score)
    assert np.all(np.isinf(result.score_history))
    assert result.evaluations == 3 * optimizer.config.population_size
    assert not result.converged


@pytest.mark.parametrize("suffix", [".npz", ".pkl"])
def test_checkpoint_resume_exactly_matches_uninterrupted_run(tmp_path, suffix: str) -> None:
    space = SearchSpace(
        ParameterSpec("x", -2.0, 2.0),
        ParameterSpec("y", -2.0, 2.0),
    )
    config = _small_config(max_generations=8)
    uninterrupted = GeneticOptimizer(space, config).optimize(_quadratic, seed=1820070)

    checkpoint = tmp_path / f"ga_checkpoint{suffix}"
    optimizer = GeneticOptimizer(space, config)
    paused = optimizer.optimize(
        _quadratic,
        seed=1820070,
        checkpoint_path=checkpoint,
        generation_limit=3,
    )
    resumed = optimizer.optimize(
        _quadratic,
        seed=999,  # Checkpoint RNG and seed must take precedence.
        checkpoint_path=checkpoint,
        resume=True,
    )

    assert paused.stop_reason == "paused"
    assert paused.generations == 3
    assert resumed.seed == uninterrupted.seed == 1820070
    assert resumed.evaluations == uninterrupted.evaluations
    assert np.array_equal(resumed.population_history, uninterrupted.population_history)
    assert np.array_equal(resumed.score_history, uninterrupted.score_history)
    assert np.array_equal(resumed.best_gene_history, uninterrupted.best_gene_history)
    assert resumed.best_score == uninterrupted.best_score


def test_latin_hypercube_and_scipy_polish() -> None:
    starts_a = latin_hypercube_starts(6, 2, seed=29)
    starts_b = latin_hypercube_starts(6, 2, seed=29)
    assert np.array_equal(starts_a, starts_b)
    assert np.all((starts_a >= 0.0) & (starts_a <= 1.0))
    for dimension in range(2):
        assert sorted(np.floor(starts_a[:, dimension] * 6).astype(int)) == list(range(6))

    space = SearchSpace(
        ParameterSpec("x", -2.0, 2.0),
        ParameterSpec("y", -2.0, 2.0),
    )
    result = local_polish(_quadratic, space, np.array([0.1, 0.9]))
    assert result.score < 1.0e-10
    assert result.parameters == pytest.approx({"x": 0.4, "y": -0.7}, abs=1.0e-5)
