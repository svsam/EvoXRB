"""Deterministic SciPy controls and local polishing for GA solutions."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .genetic import GARunResult
from .parameters import SearchSpace


DecodedObjective: TypeAlias = Callable[[dict[str, float]], float]


@dataclass(slots=True)
class SciPyRunResult:
    """One bounded SciPy minimization in normalized gene coordinates."""

    parameters: dict[str, float]
    genes: NDArray[np.float64]
    score: float
    start_genes: NDArray[np.float64]
    start_score: float
    success: bool
    message: str
    method: str
    evaluations: int
    iterations: int
    runtime_seconds: float

    @property
    def best_parameters(self) -> dict[str, float]:
        return self.parameters

    @property
    def best_genes(self) -> NDArray[np.float64]:
        return self.genes

    @property
    def best_score(self) -> float:
        return self.score

    def summary(self) -> dict[str, Any]:
        return {
            "parameters": dict(self.parameters),
            "genes": self.genes.tolist(),
            "score": self.score,
            "start_genes": self.start_genes.tolist(),
            "start_score": self.start_score,
            "success": self.success,
            "message": self.message,
            "method": self.method,
            "evaluations": self.evaluations,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(slots=True)
class SciPyMultiStartResult:
    """All deterministic starts plus direct access to the best run."""

    runs: tuple[SciPyRunResult, ...]
    starts: NDArray[np.float64]
    seed: int
    runtime_seconds: float

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError("a multi-start result requires at least one run")

    @property
    def best(self) -> SciPyRunResult:
        return min(self.runs, key=lambda run: run.score)

    @property
    def best_parameters(self) -> dict[str, float]:
        return self.best.parameters

    @property
    def best_genes(self) -> NDArray[np.float64]:
        return self.best.genes

    @property
    def best_score(self) -> float:
        return self.best.score

    @property
    def evaluations(self) -> int:
        return sum(run.evaluations for run in self.runs)


# Compatibility spelling for callers that prefer conventional camel casing.
ScipyRunResult = SciPyRunResult
ScipyMultiStartResult = SciPyMultiStartResult


def latin_hypercube_starts(
    n_starts: int,
    dimensions: int,
    *,
    seed: int,
) -> NDArray[np.float64]:
    """Construct a deterministic randomized Latin hypercube on ``[0, 1]``."""

    if n_starts < 1 or dimensions < 1:
        raise ValueError("n_starts and dimensions must be positive")
    rng = np.random.default_rng(seed)
    starts = np.empty((n_starts, dimensions), dtype=float)
    for dimension in range(dimensions):
        strata = rng.permutation(n_starts)
        starts[:, dimension] = (strata + rng.random(n_starts)) / n_starts
    return starts


def _safe_gene_objective(
    genes: NDArray[np.float64],
    objective: DecodedObjective,
    search_space: SearchSpace,
) -> float:
    vector = np.asarray(genes, dtype=float)
    if vector.shape != (search_space.ndim,) or np.any(~np.isfinite(vector)):
        return float("inf")
    if np.any((vector < 0.0) | (vector > 1.0)):
        return float("inf")
    try:
        value = float(objective(search_space.decode(vector)))
    except Exception:
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def local_polish(
    objective: DecodedObjective,
    search_space: SearchSpace,
    initial: Mapping[str, float] | Sequence[float] | NDArray[np.float64] | GARunResult,
    *,
    method: str = "L-BFGS-B",
    options: Mapping[str, Any] | None = None,
) -> SciPyRunResult:
    """Refine one GA/dictionary/gene solution with bounded SciPy minimization."""

    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("SciPy is required for local optimization") from error

    if isinstance(initial, GARunResult):
        start = np.asarray(initial.best_genes, dtype=float).copy()
    elif isinstance(initial, Mapping):
        start = search_space.encode(initial)
    else:
        start = np.asarray(initial, dtype=float).copy()
        if start.shape != (search_space.ndim,):
            raise ValueError(f"initial genes must have shape ({search_space.ndim},)")
        if np.any(~np.isfinite(start)) or np.any((start < 0.0) | (start > 1.0)):
            raise ValueError("initial genes must be finite and lie in [0, 1]")

    start_score = _safe_gene_objective(start, objective, search_space)
    started = time.perf_counter()
    result = minimize(
        _safe_gene_objective,
        start,
        args=(objective, search_space),
        method=method,
        bounds=search_space.unit_bounds,
        options=dict(options) if options is not None else None,
    )
    runtime = time.perf_counter() - started
    candidate = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    candidate_score = float(result.fun) if np.isfinite(result.fun) else float("inf")

    # A failed line search must never make a supplied GA solution worse.
    if start_score < candidate_score:
        genes = start.copy()
        score = start_score
        success = False
        message = f"{result.message}; retained lower-scoring start"
    else:
        genes = candidate
        score = candidate_score
        success = bool(result.success and np.isfinite(score))
        message = str(result.message)

    return SciPyRunResult(
        parameters=search_space.decode(genes),
        genes=genes,
        score=float(score),
        start_genes=start,
        start_score=float(start_score),
        success=success,
        message=message,
        method=method,
        evaluations=int(getattr(result, "nfev", 0)),
        iterations=int(getattr(result, "nit", 0)),
        runtime_seconds=float(runtime),
    )


def multistart_scipy(
    objective: DecodedObjective,
    search_space: SearchSpace,
    *,
    n_starts: int = 20,
    seed: int = 1_820_070,
    method: str = "L-BFGS-B",
    options: Mapping[str, Any] | None = None,
) -> SciPyMultiStartResult:
    """Run bounded SciPy minimization from deterministic Latin-hypercube starts."""

    starts = latin_hypercube_starts(n_starts, search_space.ndim, seed=seed)
    started = time.perf_counter()
    runs = tuple(
        local_polish(
            objective,
            search_space,
            start,
            method=method,
            options=options,
        )
        for start in starts
    )
    return SciPyMultiStartResult(
        runs=runs,
        starts=starts,
        seed=int(seed),
        runtime_seconds=float(time.perf_counter() - started),
    )


# Readable aliases for CLI and downstream notebooks/scripts.
multi_start_minimize = multistart_scipy
scipy_multistart = multistart_scipy

