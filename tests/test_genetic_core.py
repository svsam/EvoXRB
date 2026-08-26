from __future__ import annotations

import pickle

import numpy as np
import pytest

from evoxrb.genetic import (
    GAConfig,
    GeneticOptimizer,
    bounded_polynomial_mutation,
    load_ga_checkpoint_history,
    simulated_binary_crossover,
    tournament_select,
)
from evoxrb.optimization import latin_hypercube_starts, local_polish
from evoxrb.parameters import ParameterSpec, SearchSpace
from evoxrb.plotting import _ga_history, plot_ga_convergence, plot_population_evolution


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

    history = load_ga_checkpoint_history(checkpoint)
    assert np.array_equal(history["population_history"], resumed.population_history)
    assert np.array_equal(history["best_score_history"], resumed.best_score_history)
    assert history["parameter_names"].tolist() == ["x", "y"]
    if suffix == ".npz":
        with np.load(checkpoint, allow_pickle=False) as archive:
            assert "population_history" in archive.files
            assert "best_score_history" in archive.files


def test_plotting_recovers_history_from_legacy_payload_checkpoint(tmp_path) -> None:
    generations, population_size, dimensions = 4, 6, 2
    population = np.linspace(
        0.05, 0.95, generations * population_size * dimensions
    ).reshape(generations, population_size, dimensions)
    best_scores = np.array([100.0, 25.0, 9.0, 4.0])
    state = {
        "checkpoint_version": 1,
        "space_signature": (
            ("tin", 0.05, 2.0, "linear"),
            ("ndisk", 0.1, 1.0e6, "log10"),
        ),
        "histories": {
            "population": population,
            "scores": np.tile(best_scores[:, None], (1, population_size)),
            "best_gene": population[:, 0, :],
            "best_score": best_scores,
            "median_score": best_scores * 1.5,
            "spread": np.std(population, axis=1),
            "boundary_hits": np.zeros((generations, dimensions), dtype=int),
        },
        "immigrant_generations": (2,),
        "generation": generations - 1,
        "evaluations": generations * population_size,
    }
    checkpoint = tmp_path / "legacy_checkpoint.npz"
    payload = np.frombuffer(pickle.dumps(state, protocol=5), dtype=np.uint8)
    np.savez_compressed(checkpoint, payload=payload)

    history = _ga_history(checkpoint)
    assert np.array_equal(history["population"], population)
    assert np.array_equal(history["best"], best_scores)
    assert history["parameter_names"].tolist() == ["tin", "ndisk"]
    assert history["parameter_scales"].tolist() == ["linear", "log10"]

    convergence = plot_ga_convergence(checkpoint, tmp_path / "convergence.png")
    evolution = plot_population_evolution(checkpoint, tmp_path / "population.png")
    assert convergence.stat().st_size > 1_000
    assert evolution.stat().st_size > 1_000


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
