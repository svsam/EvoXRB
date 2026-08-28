from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from evoxrb.genetic import GAConfig, GenerationSnapshot, GeneticOptimizer
from evoxrb.parameters import ParameterSpec, SearchSpace


def _objective(parameters: dict[str, float]) -> float:
    return (parameters["x"] - 0.35) ** 2 + (parameters["y"] + 0.6) ** 2


def _optimizer(max_generations: int = 4) -> GeneticOptimizer:
    space = SearchSpace(
        ParameterSpec("x", -2.0, 2.0),
        ParameterSpec("y", -2.0, 2.0),
    )
    config = GAConfig(
        population_size=12,
        max_generations=max_generations,
        tournament_size=3,
        min_generations=max_generations + 1,
        mutation_anneal_generations=max_generations,
        immigrant_stagnation_generations=2,
        stop_stagnation_generations=2,
    )
    return GeneticOptimizer(space, config)


def test_generation_callback_is_read_only_and_deterministic() -> None:
    control = _optimizer().optimize(_objective, seed=1729)
    snapshots: list[GenerationSnapshot] = []
    optimizer = _optimizer()

    observed = optimizer.optimize(
        _objective,
        seed=1729,
        on_generation=snapshots.append,
    )

    assert [item.generation for item in snapshots] == list(range(5))
    assert [item.evaluations for item in snapshots] == [12, 24, 36, 48, 60]
    assert all(item.seed == 1729 for item in snapshots)
    assert snapshots[0].stop_reason is None
    assert snapshots[-1].stop_reason == "max_generations"

    np.testing.assert_array_equal(observed.population_history, control.population_history)
    np.testing.assert_array_equal(observed.score_history, control.score_history)
    np.testing.assert_array_equal(observed.best_gene_history, control.best_gene_history)
    np.testing.assert_array_equal(observed.best_score_history, control.best_score_history)
    assert observed.best_parameters == control.best_parameters
    assert observed.best_score == control.best_score

    for generation, snapshot in enumerate(snapshots):
        np.testing.assert_array_equal(
            snapshot.population, observed.population_history[generation]
        )
        np.testing.assert_array_equal(snapshot.scores, observed.score_history[generation])
        np.testing.assert_array_equal(
            snapshot.best_genes, observed.best_gene_history[generation]
        )
        assert snapshot.best_score == observed.best_score_history[generation]
        assert dict(snapshot.best_parameters) == pytest.approx(
            optimizer.search_space.decode(observed.best_gene_history[generation])
        )
        assert not np.shares_memory(
            snapshot.population, observed.population_history[generation]
        )
        for values in (
            snapshot.population,
            snapshot.scores,
            snapshot.best_genes,
            snapshot.gene_spread,
            snapshot.boundary_hits,
        ):
            assert values.flags.writeable is False

    with pytest.raises(ValueError):
        snapshots[0].population[0, 0] = 1.0
    with pytest.raises(TypeError):
        snapshots[0].best_parameters["x"] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshots[0].generation = 99  # type: ignore[misc]


def test_resume_callback_starts_with_current_checkpoint_state(tmp_path) -> None:
    checkpoint = tmp_path / "callback_resume.npz"
    optimizer = _optimizer()
    paused_events: list[GenerationSnapshot] = []
    paused = optimizer.optimize(
        _objective,
        seed=2468,
        checkpoint_path=checkpoint,
        generation_limit=2,
        on_generation=paused_events.append,
    )

    assert paused.stop_reason == "paused"
    assert [item.generation for item in paused_events] == [0, 1, 2]
    assert paused_events[-1].stop_reason == "paused"

    resumed_events: list[GenerationSnapshot] = []
    resumed = optimizer.optimize(
        _objective,
        seed=999,
        checkpoint_path=checkpoint,
        resume=True,
        on_generation=resumed_events.append,
    )
    uninterrupted = _optimizer().optimize(_objective, seed=2468)

    assert [item.generation for item in resumed_events] == [2, 3, 4]
    assert resumed_events[0].stop_reason is None
    assert resumed_events[0].seed == 2468
    assert resumed_events[-1].stop_reason == "max_generations"
    np.testing.assert_array_equal(
        resumed.population_history, uninterrupted.population_history
    )
    np.testing.assert_array_equal(resumed.score_history, uninterrupted.score_history)

    terminal_events: list[GenerationSnapshot] = []
    terminal = optimizer.optimize(
        _objective,
        seed=123,
        checkpoint_path=checkpoint,
        resume=True,
        on_generation=terminal_events.append,
    )
    assert terminal.generations == 4
    assert [item.generation for item in terminal_events] == [4]
    assert terminal_events[0].stop_reason == "max_generations"


def test_optimize_many_forwards_one_callback_across_seeds() -> None:
    events: list[tuple[int, int]] = []
    optimizer = _optimizer(max_generations=1)

    optimizer.optimize_many(
        _objective,
        (11, 22),
        on_generation=lambda snapshot: events.append(
            (snapshot.seed, snapshot.generation)
        ),
    )

    assert events == [(11, 0), (11, 1), (22, 0), (22, 1)]


def test_checkpoint_objective_signature_rejects_different_data(tmp_path) -> None:
    checkpoint = tmp_path / "objective_bound.npz"
    optimizer = _optimizer(max_generations=2)
    optimizer.optimize(
        _objective,
        seed=1234,
        checkpoint_path=checkpoint,
        generation_limit=1,
        objective_signature="dataset-and-model-a",
    )

    with pytest.raises(ValueError, match="objective/data"):
        optimizer.optimize(
            _objective,
            seed=1234,
            checkpoint_path=checkpoint,
            resume=True,
            objective_signature="dataset-and-model-b",
        )

    resumed = optimizer.optimize(
        _objective,
        seed=999,
        checkpoint_path=checkpoint,
        resume=True,
        objective_signature="dataset-and-model-a",
    )
    assert resumed.generations == 2


def test_signed_checkpoint_requires_signature_on_resume(tmp_path) -> None:
    checkpoint = tmp_path / "signed.npz"
    optimizer = _optimizer(max_generations=2)
    optimizer.optimize(
        _objective,
        seed=17,
        checkpoint_path=checkpoint,
        generation_limit=1,
        objective_signature="bound-objective",
    )

    with pytest.raises(ValueError, match="requires its objective signature"):
        optimizer.optimize(
            _objective,
            seed=17,
            checkpoint_path=checkpoint,
            resume=True,
        )
