"""A reproducible real-valued genetic optimizer for bounded fitting problems."""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .parameters import SearchSpace


DecodedObjective: TypeAlias = Callable[[dict[str, float]], float]


@dataclass(frozen=True, slots=True)
class GenerationSnapshot:
    """Read-only state emitted after a GA generation has been evaluated.

    Population and gene arrays use normalized coordinates in ``[0, 1]``.
    Every array is an independent, non-writeable copy, and the decoded best
    parameters are exposed through an immutable mapping.  A callback can
    therefore retain or inspect a snapshot without mutating optimizer state.
    """

    generation: int
    evaluations: int
    seed: int
    population: NDArray[np.float64]
    scores: NDArray[np.float64]
    best_genes: NDArray[np.float64]
    best_parameters: Mapping[str, float]
    best_score: float
    median_score: float
    gene_spread: NDArray[np.float64]
    boundary_hits: NDArray[np.int64]
    immigrant_generations: tuple[int, ...] = field(default_factory=tuple)
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        for name, dtype in (
            ("population", np.float64),
            ("scores", np.float64),
            ("best_genes", np.float64),
            ("gene_spread", np.float64),
            ("boundary_hits", np.int64),
        ):
            copied = np.array(getattr(self, name), dtype=dtype, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(
            self,
            "best_parameters",
            MappingProxyType(
                {
                    str(name): float(value)
                    for name, value in self.best_parameters.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "immigrant_generations",
            tuple(int(item) for item in self.immigrant_generations),
        )


GenerationCallback: TypeAlias = Callable[[GenerationSnapshot], None]


# Human-readable arrays stored alongside the resumable checkpoint payload.  The
# names intentionally match ``GARunResult`` so NPZ inspection and plotting do
# not require unpickling the optimizer state.
_CHECKPOINT_HISTORY_NAMES = {
    "population": "population_history",
    "scores": "score_history",
    "best_gene": "best_gene_history",
    "best_score": "best_score_history",
    "median_score": "median_score_history",
    "spread": "gene_spread_history",
    "boundary_hits": "boundary_hit_history",
}


@dataclass(frozen=True, slots=True)
class GAConfig:
    """Configuration for :class:`GeneticOptimizer`.

    Defaults reproduce the full case-study design.  Smaller values can be used
    by the smoke profile and unit tests without changing the operators.
    """

    population_size: int = 192
    max_generations: int = 300
    tournament_size: int = 3
    crossover_probability: float = 0.9
    crossover_eta: float = 15.0
    mutation_probability: float | None = None
    mutation_eta_start: float = 10.0
    mutation_eta_end: float = 50.0
    mutation_anneal_generations: int = 300
    elite_fraction: float = 0.03
    immigrant_fraction: float = 0.05
    immigrant_stagnation_generations: int = 20
    immigrant_spread_threshold: float = 0.02
    min_generations: int = 75
    stop_stagnation_generations: int = 40
    improvement_tolerance: float = 0.1
    stop_spread_threshold: float = 0.01
    boundary_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least four")
        if self.max_generations < 1:
            raise ValueError("max_generations must be positive")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be between two and population_size")
        for name in ("crossover_probability", "elite_fraction", "immigrant_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.mutation_probability is not None and not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must lie in [0, 1]")
        if self.crossover_eta <= 0.0:
            raise ValueError("crossover_eta must be positive")
        if self.mutation_eta_start <= 0.0 or self.mutation_eta_end <= 0.0:
            raise ValueError("mutation distribution indices must be positive")
        if self.mutation_anneal_generations < 1:
            raise ValueError("mutation_anneal_generations must be positive")
        if self.immigrant_stagnation_generations < 1:
            raise ValueError("immigrant_stagnation_generations must be positive")
        if self.stop_stagnation_generations < 1:
            raise ValueError("stop_stagnation_generations must be positive")
        if self.min_generations < 0:
            raise ValueError("min_generations cannot be negative")
        if self.improvement_tolerance < 0.0:
            raise ValueError("improvement_tolerance cannot be negative")
        if self.immigrant_spread_threshold < 0.0 or self.stop_spread_threshold < 0.0:
            raise ValueError("spread thresholds cannot be negative")
        if self.boundary_tolerance < 0.0 or self.boundary_tolerance > 0.5:
            raise ValueError("boundary_tolerance must lie in [0, 0.5]")

    def mutation_probability_for(self, dimensions: int) -> float:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if self.mutation_probability is None:
            return 1.0 / dimensions
        return self.mutation_probability

    def mutation_eta_at(self, generation: int) -> float:
        fraction = np.clip(generation / self.mutation_anneal_generations, 0.0, 1.0)
        return float(
            self.mutation_eta_start
            + fraction * (self.mutation_eta_end - self.mutation_eta_start)
        )

    @property
    def elite_count(self) -> int:
        return max(1, int(math.ceil(self.elite_fraction * self.population_size)))

    @property
    def immigrant_count(self) -> int:
        return int(math.ceil(self.immigrant_fraction * self.population_size))


@dataclass(slots=True)
class GARunResult:
    """Complete result and diagnostic history for one GA seed."""

    best_parameters: dict[str, float]
    best_genes: NDArray[np.float64]
    best_score: float
    generations: int
    evaluations: int
    converged: bool
    stop_reason: str
    seed: int
    runtime_seconds: float
    population_history: NDArray[np.float64]
    score_history: NDArray[np.float64]
    best_gene_history: NDArray[np.float64]
    best_score_history: NDArray[np.float64]
    median_score_history: NDArray[np.float64]
    gene_spread_history: NDArray[np.float64]
    boundary_hit_history: NDArray[np.int64]
    immigrant_generations: tuple[int, ...] = field(default_factory=tuple)
    checkpoint_path: str | None = None

    @property
    def final_population(self) -> NDArray[np.float64]:
        return self.population_history[-1]

    @property
    def final_scores(self) -> NDArray[np.float64]:
        return self.score_history[-1]

    @property
    def median_gene_spread_history(self) -> NDArray[np.float64]:
        return np.median(self.gene_spread_history, axis=1)

    @property
    def boundary_hits(self) -> int:
        tolerance = 1.0e-8
        return int(np.count_nonzero((self.best_genes <= tolerance) | (self.best_genes >= 1.0 - tolerance)))

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable scalar summary."""

        return {
            "best_parameters": dict(self.best_parameters),
            "best_genes": self.best_genes.tolist(),
            "best_score": self.best_score,
            "generations": self.generations,
            "evaluations": self.evaluations,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "seed": self.seed,
            "runtime_seconds": self.runtime_seconds,
            "boundary_hits": self.boundary_hits,
            "immigrant_generations": list(self.immigrant_generations),
            "checkpoint_path": self.checkpoint_path,
        }


def tournament_select(
    population: NDArray[np.float64],
    scores: NDArray[np.float64],
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Select and copy the best member of a random tournament."""

    if population.ndim != 2 or scores.shape != (population.shape[0],):
        raise ValueError("population and score shapes are incompatible")
    if not 1 <= size <= population.shape[0]:
        raise ValueError("invalid tournament size")
    contestants = rng.choice(population.shape[0], size=size, replace=False)
    winner = contestants[int(np.argmin(scores[contestants]))]
    return population[winner].copy()


def simulated_binary_crossover(
    parent_a: Sequence[float] | NDArray[np.float64],
    parent_b: Sequence[float] | NDArray[np.float64],
    rng: np.random.Generator,
    *,
    probability: float = 0.9,
    eta: float = 15.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Deb's bounded simulated-binary crossover on ``[0, 1]``."""

    first = np.asarray(parent_a, dtype=float).copy()
    second = np.asarray(parent_b, dtype=float).copy()
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("parents must be equally-sized one-dimensional vectors")
    if np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
        raise ValueError("parents must be finite")
    if np.any((first < 0.0) | (first > 1.0)) or np.any((second < 0.0) | (second > 1.0)):
        raise ValueError("parents must lie in the unit hypercube")
    if not 0.0 <= probability <= 1.0 or eta <= 0.0:
        raise ValueError("invalid crossover settings")
    if rng.random() > probability:
        return first, second

    for index, (x1, x2) in enumerate(zip(first.copy(), second.copy(), strict=True)):
        if rng.random() > 0.5 or abs(x1 - x2) <= 1.0e-14:
            continue
        lower_parent, upper_parent = sorted((float(x1), float(x2)))
        random_value = rng.random()

        beta = 1.0 + 2.0 * lower_parent / (upper_parent - lower_parent)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if random_value <= 1.0 / alpha:
            beta_q = (random_value * alpha) ** (1.0 / (eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - random_value * alpha)) ** (1.0 / (eta + 1.0))
        child_low = 0.5 * (
            lower_parent + upper_parent - beta_q * (upper_parent - lower_parent)
        )

        beta = 1.0 + 2.0 * (1.0 - upper_parent) / (upper_parent - lower_parent)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if random_value <= 1.0 / alpha:
            beta_q = (random_value * alpha) ** (1.0 / (eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - random_value * alpha)) ** (1.0 / (eta + 1.0))
        child_high = 0.5 * (
            lower_parent + upper_parent + beta_q * (upper_parent - lower_parent)
        )

        if rng.random() <= 0.5:
            first[index], second[index] = child_high, child_low
        else:
            first[index], second[index] = child_low, child_high

    return np.clip(first, 0.0, 1.0), np.clip(second, 0.0, 1.0)


def bounded_polynomial_mutation(
    genes: Sequence[float] | NDArray[np.float64],
    rng: np.random.Generator,
    *,
    probability: float,
    eta: float,
) -> NDArray[np.float64]:
    """Apply bounded polynomial mutation independently to each unit gene."""

    child = np.asarray(genes, dtype=float).copy()
    if child.ndim != 1 or np.any(~np.isfinite(child)):
        raise ValueError("genes must be a finite one-dimensional vector")
    if np.any((child < 0.0) | (child > 1.0)):
        raise ValueError("genes must lie in the unit hypercube")
    if not 0.0 <= probability <= 1.0 or eta <= 0.0:
        raise ValueError("invalid mutation settings")

    exponent = 1.0 / (eta + 1.0)
    for index, value in enumerate(child.copy()):
        if rng.random() > probability:
            continue
        random_value = rng.random()
        if random_value < 0.5:
            xy = 1.0 - value
            val = 2.0 * random_value + (1.0 - 2.0 * random_value) * xy ** (eta + 1.0)
            delta = val**exponent - 1.0
        else:
            xy = value
            val = 2.0 * (1.0 - random_value) + 2.0 * (random_value - 0.5) * xy ** (eta + 1.0)
            delta = 1.0 - val**exponent
        child[index] = value + delta
    return np.clip(child, 0.0, 1.0)


class GeneticOptimizer:
    """Minimize a decoded-dictionary objective with a real-valued GA."""

    CHECKPOINT_VERSION = 1

    def __init__(self, search_space: SearchSpace, config: GAConfig | None = None) -> None:
        self.search_space = search_space
        self.config = config or GAConfig()

    def optimize(
        self,
        objective: DecodedObjective,
        *,
        seed: int,
        checkpoint_path: str | os.PathLike[str] | None = None,
        resume: bool = False,
        initial_population: Sequence[Sequence[float]] | NDArray[np.float64] | None = None,
        generation_limit: int | None = None,
        on_generation: GenerationCallback | None = None,
        objective_signature: str | None = None,
    ) -> GARunResult:
        """Run or resume one optimization.

        ``generation_limit`` is an absolute, invocation-only pause point useful
        for schedulers and deterministic resume tests.  It does not alter the
        configured mutation schedule or final generation limit.

        ``on_generation`` receives an immutable snapshot for the initial or
        resumed state and after every newly evaluated generation.  Callback
        return values are ignored and snapshots never share writable memory
        with the optimizer.

        ``objective_signature`` binds a checkpoint to caller-defined data and
        model identity. Generic objectives may omit it, but domain workflows
        should supply a stable digest before enabling resume.
        """

        if generation_limit is not None:
            if generation_limit < 0:
                raise ValueError("generation_limit cannot be negative")
            generation_limit = min(generation_limit, self.config.max_generations)
        if objective_signature is not None:
            objective_signature = str(objective_signature).strip()
            if not objective_signature:
                raise ValueError("objective_signature must not be empty")
        path = Path(checkpoint_path) if checkpoint_path is not None else None
        started = time.perf_counter()

        if resume:
            if path is None:
                raise ValueError("resume=True requires checkpoint_path")
            state = self._load_checkpoint(path)
            self._validate_checkpoint(state, objective_signature)
            population = np.asarray(state["population"], dtype=float)
            scores = np.asarray(state["scores"], dtype=float)
            generation = int(state["generation"])
            evaluations = int(state["evaluations"])
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng_state"]
            seed = int(state["seed"])
            histories = self._restore_histories(state)
            immigrant_generations = list(state.get("immigrant_generations", ()))
            prior_runtime = float(state.get("runtime_seconds", 0.0))
            old_reason = state.get("stop_reason")
            resumed_reason = (
                str(old_reason)
                if old_reason == "converged" or generation >= self.config.max_generations
                else None
            )
            self._notify_generation(
                on_generation,
                population,
                scores,
                generation,
                evaluations,
                seed,
                immigrant_generations,
                resumed_reason,
            )
            if old_reason == "converged":
                return self._make_result(
                    population,
                    scores,
                    generation,
                    evaluations,
                    seed,
                    prior_runtime,
                    histories,
                    immigrant_generations,
                    "converged",
                    path,
                )
            if generation >= self.config.max_generations:
                return self._make_result(
                    population,
                    scores,
                    generation,
                    evaluations,
                    seed,
                    prior_runtime,
                    histories,
                    immigrant_generations,
                    "max_generations",
                    path,
                )
        else:
            rng = np.random.default_rng(seed)
            population = self._initialize_population(rng, initial_population)
            scores = self._evaluate_population(population, objective)
            evaluations = self.config.population_size
            generation = 0
            histories = self._new_histories(population, scores)
            immigrant_generations: list[int] = []
            prior_runtime = 0.0
            initial_reason = "paused" if generation_limit == 0 else None
            if path is not None:
                self._save_checkpoint(
                    path,
                    self._checkpoint_state(
                        population,
                        scores,
                        generation,
                        evaluations,
                        seed,
                        rng,
                        histories,
                        immigrant_generations,
                        prior_runtime + time.perf_counter() - started,
                        initial_reason,
                        objective_signature,
                    ),
                )
            self._notify_generation(
                on_generation,
                population,
                scores,
                generation,
                evaluations,
                seed,
                immigrant_generations,
                initial_reason,
            )

        stop_reason: str | None = None
        if generation_limit == 0:
            stop_reason = "paused"

        invocation_end = (
            self.config.max_generations
            if generation_limit is None
            else min(generation_limit, self.config.max_generations)
        )
        for next_generation in range(generation + 1, invocation_end + 1):
            population = self._breed_generation(population, scores, rng, next_generation)
            if self._should_inject_immigrants(histories):
                count = min(
                    self.config.immigrant_count,
                    self.config.population_size - self.config.elite_count,
                )
                if count > 0:
                    population[-count:] = rng.random((count, self.search_space.ndim))
                    immigrant_generations.append(next_generation)
            scores = self._evaluate_population(population, objective)
            evaluations += self.config.population_size
            generation = next_generation
            self._append_histories(histories, population, scores)

            if self._has_converged(generation, histories):
                stop_reason = "converged"
            elif generation >= self.config.max_generations:
                stop_reason = "max_generations"
            elif generation_limit is not None and generation >= generation_limit:
                stop_reason = "paused"

            elapsed = prior_runtime + time.perf_counter() - started
            if path is not None:
                self._save_checkpoint(
                    path,
                    self._checkpoint_state(
                        population,
                        scores,
                        generation,
                        evaluations,
                        seed,
                        rng,
                        histories,
                        immigrant_generations,
                        elapsed,
                        stop_reason,
                        objective_signature,
                    ),
                )
            self._notify_generation(
                on_generation,
                population,
                scores,
                generation,
                evaluations,
                seed,
                immigrant_generations,
                stop_reason,
            )
            if stop_reason is not None:
                break

        if stop_reason is None:
            if generation >= self.config.max_generations:
                stop_reason = "max_generations"
            elif generation_limit is not None and generation >= generation_limit:
                stop_reason = "paused"
            else:  # Defensive: the configured loop always establishes a reason.
                stop_reason = "paused"

        runtime = prior_runtime + time.perf_counter() - started
        return self._make_result(
            population,
            scores,
            generation,
            evaluations,
            seed,
            runtime,
            histories,
            immigrant_generations,
            stop_reason,
            path,
        )

    def optimize_many(
        self,
        objective: DecodedObjective,
        seeds: Sequence[int],
        *,
        checkpoint_directory: str | os.PathLike[str] | None = None,
        resume: bool = False,
        on_generation: GenerationCallback | None = None,
        objective_signature: str | None = None,
    ) -> list[GARunResult]:
        """Run independent GA seeds in deterministic sequence.

        When supplied, ``on_generation`` observes every seed; snapshots carry
        the active seed so callers can keep their progress streams separate.
        """

        directory = Path(checkpoint_directory) if checkpoint_directory is not None else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        results = []
        for seed in seeds:
            path = directory / f"ga_seed_{int(seed)}.npz" if directory is not None else None
            results.append(
                self.optimize(
                    objective,
                    seed=int(seed),
                    checkpoint_path=path,
                    resume=resume and path is not None and path.exists(),
                    on_generation=on_generation,
                    objective_signature=objective_signature,
                )
            )
        return results

    def _initialize_population(
        self,
        rng: np.random.Generator,
        initial_population: Sequence[Sequence[float]] | NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        population = rng.random((self.config.population_size, self.search_space.ndim))
        if initial_population is None:
            return population
        initial = np.asarray(initial_population, dtype=float)
        if initial.ndim == 1:
            initial = initial[np.newaxis, :]
        if initial.ndim != 2 or initial.shape[1] != self.search_space.ndim:
            raise ValueError(
                f"initial_population must have shape (n, {self.search_space.ndim})"
            )
        if initial.shape[0] > self.config.population_size:
            raise ValueError("initial_population is larger than configured population")
        if np.any(~np.isfinite(initial)) or np.any((initial < 0.0) | (initial > 1.0)):
            raise ValueError("initial population must be finite and lie in [0, 1]")
        population[: initial.shape[0]] = initial
        return population

    def _safe_score(self, genes: NDArray[np.float64], objective: DecodedObjective) -> float:
        try:
            value = float(objective(self.search_space.decode(genes)))
        except Exception:
            return float("inf")
        return value if np.isfinite(value) else float("inf")

    def _evaluate_population(
        self,
        population: NDArray[np.float64],
        objective: DecodedObjective,
    ) -> NDArray[np.float64]:
        return np.asarray(
            [self._safe_score(individual, objective) for individual in population],
            dtype=float,
        )

    def _breed_generation(
        self,
        population: NDArray[np.float64],
        scores: NDArray[np.float64],
        rng: np.random.Generator,
        generation: int,
    ) -> NDArray[np.float64]:
        order = np.argsort(scores, kind="stable")
        elite_count = self.config.elite_count
        next_population: list[NDArray[np.float64]] = [
            population[index].copy() for index in order[:elite_count]
        ]
        mutation_probability = self.config.mutation_probability_for(self.search_space.ndim)
        mutation_eta = self.config.mutation_eta_at(generation)

        while len(next_population) < self.config.population_size:
            parent_a = tournament_select(
                population, scores, self.config.tournament_size, rng
            )
            parent_b = tournament_select(
                population, scores, self.config.tournament_size, rng
            )
            child_a, child_b = simulated_binary_crossover(
                parent_a,
                parent_b,
                rng,
                probability=self.config.crossover_probability,
                eta=self.config.crossover_eta,
            )
            child_a = bounded_polynomial_mutation(
                child_a,
                rng,
                probability=mutation_probability,
                eta=mutation_eta,
            )
            child_b = bounded_polynomial_mutation(
                child_b,
                rng,
                probability=mutation_probability,
                eta=mutation_eta,
            )
            next_population.append(child_a)
            if len(next_population) < self.config.population_size:
                next_population.append(child_b)
        return np.asarray(next_population, dtype=float)

    def _snapshot(
        self,
        population: NDArray[np.float64],
        scores: NDArray[np.float64],
    ) -> dict[str, Any]:
        best_index = int(np.argmin(scores))
        spread = np.std(population, axis=0)
        tolerance = self.config.boundary_tolerance
        boundary_hits = np.count_nonzero(
            (population <= tolerance) | (population >= 1.0 - tolerance), axis=0
        )
        return {
            "population": population.copy(),
            "scores": scores.copy(),
            "best_gene": population[best_index].copy(),
            "best_score": float(scores[best_index]),
            "median_score": float(np.median(scores)),
            "spread": spread,
            "boundary_hits": boundary_hits.astype(np.int64),
        }

    def _notify_generation(
        self,
        callback: GenerationCallback | None,
        population: NDArray[np.float64],
        scores: NDArray[np.float64],
        generation: int,
        evaluations: int,
        seed: int,
        immigrant_generations: Sequence[int],
        stop_reason: str | None,
    ) -> None:
        if callback is None:
            return
        state = self._snapshot(population, scores)
        best_genes = np.asarray(state["best_gene"], dtype=np.float64)
        callback(
            GenerationSnapshot(
                generation=int(generation),
                evaluations=int(evaluations),
                seed=int(seed),
                population=np.asarray(state["population"], dtype=np.float64),
                scores=np.asarray(state["scores"], dtype=np.float64),
                best_genes=best_genes,
                best_parameters=self.search_space.decode(best_genes),
                best_score=float(state["best_score"]),
                median_score=float(state["median_score"]),
                gene_spread=np.asarray(state["spread"], dtype=np.float64),
                boundary_hits=np.asarray(state["boundary_hits"], dtype=np.int64),
                immigrant_generations=tuple(immigrant_generations),
                stop_reason=stop_reason,
            )
        )

    def _new_histories(
        self,
        population: NDArray[np.float64],
        scores: NDArray[np.float64],
    ) -> dict[str, list[Any]]:
        item = self._snapshot(population, scores)
        return {name: [value] for name, value in item.items()}

    def _append_histories(
        self,
        histories: dict[str, list[Any]],
        population: NDArray[np.float64],
        scores: NDArray[np.float64],
    ) -> None:
        for name, value in self._snapshot(population, scores).items():
            histories[name].append(value)

    @staticmethod
    def _window_improvement(scores: Sequence[float], window: int) -> float:
        if len(scores) <= window:
            return float("inf")
        older = float(scores[-window - 1])
        current = float(scores[-1])
        if np.isinf(older) and np.isinf(current):
            return 0.0
        return older - current

    def _should_inject_immigrants(self, histories: dict[str, list[Any]]) -> bool:
        window = self.config.immigrant_stagnation_generations
        improvement = self._window_improvement(histories["best_score"], window)
        median_spread = float(np.median(histories["spread"][-1]))
        return (
            improvement < self.config.improvement_tolerance
            and median_spread < self.config.immigrant_spread_threshold
        )

    def _has_converged(self, generation: int, histories: dict[str, list[Any]]) -> bool:
        if generation < self.config.min_generations:
            return False
        current_best = float(histories["best_score"][-1])
        if not np.isfinite(current_best):
            return False
        improvement = self._window_improvement(
            histories["best_score"], self.config.stop_stagnation_generations
        )
        median_spread = float(np.median(histories["spread"][-1]))
        return (
            improvement < self.config.improvement_tolerance
            and median_spread < self.config.stop_spread_threshold
        )

    def _checkpoint_state(
        self,
        population: NDArray[np.float64],
        scores: NDArray[np.float64],
        generation: int,
        evaluations: int,
        seed: int,
        rng: np.random.Generator,
        histories: dict[str, list[Any]],
        immigrant_generations: Sequence[int],
        runtime_seconds: float,
        stop_reason: str | None,
        objective_signature: str | None,
    ) -> dict[str, Any]:
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "space_signature": self.search_space.signature,
            "config_signature": self._config_signature(),
            "objective_signature": objective_signature,
            "population": population,
            "scores": scores,
            "generation": generation,
            "evaluations": evaluations,
            "seed": int(seed),
            "rng_state": rng.bit_generator.state,
            "histories": histories,
            "immigrant_generations": tuple(immigrant_generations),
            "runtime_seconds": float(runtime_seconds),
            "stop_reason": stop_reason,
        }

    def _config_signature(self) -> dict[str, Any]:
        signature = asdict(self.config)
        # A checkpoint may be deliberately extended to a larger maximum.
        signature.pop("max_generations")
        return signature

    def _validate_checkpoint(
        self,
        state: Mapping[str, Any],
        objective_signature: str | None,
    ) -> None:
        if state.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported GA checkpoint version")
        if tuple(tuple(item) for item in state.get("space_signature", ())) != self.search_space.signature:
            raise ValueError("checkpoint search space does not match optimizer")
        if state.get("config_signature") != self._config_signature():
            raise ValueError("checkpoint GA configuration does not match optimizer")
        saved_signature = state.get("objective_signature")
        if saved_signature is not None:
            if objective_signature is None:
                raise ValueError(
                    "checkpoint requires its objective signature when resuming"
                )
            if saved_signature != objective_signature:
                raise ValueError("checkpoint objective/data does not match current run")
        elif objective_signature is not None:
            raise ValueError(
                "checkpoint has no objective signature; start a fresh run"
            )
        population = np.asarray(state.get("population"), dtype=float)
        expected = (self.config.population_size, self.search_space.ndim)
        if population.shape != expected:
            raise ValueError("checkpoint population shape is incompatible")

    @staticmethod
    def _restore_histories(state: Mapping[str, Any]) -> dict[str, list[Any]]:
        raw = state.get("histories")
        if not isinstance(raw, Mapping):
            raise ValueError("checkpoint has no valid histories")
        return {name: list(values) for name, values in raw.items()}

    @staticmethod
    def _save_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        if path.suffix.lower() == ".npz":
            checkpoint_state = dict(state)
            raw_histories = checkpoint_state.pop("histories", None)
            payload = np.frombuffer(
                pickle.dumps(checkpoint_state, protocol=5), dtype=np.uint8
            )
            archive: dict[str, Any] = {"payload": payload}
            if isinstance(raw_histories, Mapping):
                for internal_name, archive_name in _CHECKPOINT_HISTORY_NAMES.items():
                    if internal_name in raw_histories:
                        archive[archive_name] = np.asarray(raw_histories[internal_name])
            signature = state.get("space_signature", ())
            archive["parameter_names"] = np.asarray(
                [str(item[0]) for item in signature], dtype=str
            )
            archive["parameter_scales"] = np.asarray(
                [str(item[3]) for item in signature], dtype=str
            )
            archive["immigrant_generations"] = np.asarray(
                state.get("immigrant_generations", ()), dtype=np.int64
            )
            archive["generation"] = np.asarray(state.get("generation", -1), dtype=np.int64)
            archive["evaluations"] = np.asarray(state.get("evaluations", 0), dtype=np.int64)
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **archive)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with temporary.open("wb") as handle:
                pickle.dump(dict(state), handle, protocol=5)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(path)

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                payload = np.asarray(archive["payload"], dtype=np.uint8).tobytes()
                state = pickle.loads(payload)
                if isinstance(state, dict) and not isinstance(
                    state.get("histories"), Mapping
                ):
                    histories = {
                        internal_name: np.asarray(archive[archive_name]).copy()
                        for internal_name, archive_name in _CHECKPOINT_HISTORY_NAMES.items()
                        if archive_name in archive.files
                    }
                    if histories:
                        state["histories"] = histories
        else:
            with path.open("rb") as handle:
                state = pickle.load(handle)
        if not isinstance(state, dict):
            raise ValueError("invalid GA checkpoint payload")
        return state

    def _make_result(
        self,
        population: NDArray[np.float64],
        scores: NDArray[np.float64],
        generation: int,
        evaluations: int,
        seed: int,
        runtime_seconds: float,
        histories: dict[str, list[Any]],
        immigrant_generations: Sequence[int],
        stop_reason: str,
        checkpoint_path: Path | None,
    ) -> GARunResult:
        all_best_scores = np.asarray(histories["best_score"], dtype=float)
        best_generation = int(np.argmin(all_best_scores))
        best_genes = np.asarray(histories["best_gene"][best_generation], dtype=float).copy()
        return GARunResult(
            best_parameters=self.search_space.decode(best_genes),
            best_genes=best_genes,
            best_score=float(all_best_scores[best_generation]),
            generations=generation,
            evaluations=evaluations,
            converged=stop_reason == "converged",
            stop_reason=stop_reason,
            seed=int(seed),
            runtime_seconds=float(runtime_seconds),
            population_history=np.asarray(histories["population"], dtype=float),
            score_history=np.asarray(histories["scores"], dtype=float),
            best_gene_history=np.asarray(histories["best_gene"], dtype=float),
            best_score_history=all_best_scores,
            median_score_history=np.asarray(histories["median_score"], dtype=float),
            gene_spread_history=np.asarray(histories["spread"], dtype=float),
            boundary_hit_history=np.asarray(histories["boundary_hits"], dtype=np.int64),
            immigrant_generations=tuple(int(item) for item in immigrant_generations),
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        )


def load_ga_checkpoint_history(path: str | Path) -> dict[str, NDArray[Any]]:
    """Load plotting-ready diagnostic arrays from an EvoXRB GA checkpoint.

    Checkpoints are trusted local artifacts because resumable optimizer state
    includes a pickled NumPy RNG state.  New NPZ checkpoints also expose these
    history arrays directly so they can be inspected with ``numpy.load``;
    this loader retains support for version-1 checkpoints that kept them only
    inside the serialized payload.
    """

    state = GeneticOptimizer._load_checkpoint(Path(path))
    raw_histories = state.get("histories")
    if not isinstance(raw_histories, Mapping):
        raise ValueError("checkpoint has no valid histories")

    result: dict[str, NDArray[Any]] = {}
    for internal_name, public_name in _CHECKPOINT_HISTORY_NAMES.items():
        if internal_name in raw_histories:
            result[public_name] = np.asarray(raw_histories[internal_name])

    signature = state.get("space_signature", ())
    result["parameter_names"] = np.asarray(
        [str(item[0]) for item in signature], dtype=str
    )
    result["parameter_scales"] = np.asarray(
        [str(item[3]) for item in signature], dtype=str
    )
    result["immigrant_generations"] = np.asarray(
        state.get("immigrant_generations", ()), dtype=np.int64
    )
    result["generation"] = np.asarray(state.get("generation", -1), dtype=np.int64)
    result["evaluations"] = np.asarray(state.get("evaluations", 0), dtype=np.int64)
    return result
