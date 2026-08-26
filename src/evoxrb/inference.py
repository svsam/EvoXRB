"""GA-seeded posterior sampling with emcee and a dynesty fallback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .genetic import GARunResult
from .parameters import SearchSpace
from .types import PosteriorResult


DecodedObjective = Callable[[dict[str, float]], float]


@dataclass(slots=True)
class _DynestyLogLikelihood:
    """Pickle-friendly likelihood wrapper for dynesty checkpoints."""

    objective: DecodedObjective
    search_space: SearchSpace

    def __call__(self, unit: NDArray[np.float64]) -> float:
        return _safe_log_probability(
            np.asarray(unit, dtype=float), self.objective, self.search_space
        )


@dataclass(frozen=True, slots=True)
class _IdentityPriorTransform:
    """Keep nested-sampling coordinates in the normalized gene cube."""

    def __call__(self, unit: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(unit, dtype=float)


@dataclass(frozen=True, slots=True)
class PosteriorConfig:
    """Configuration shared by the ensemble and nested samplers."""

    batch_steps: int = 500
    max_steps: int = 20_000
    min_autocorrelation_times: float = 50.0
    autocorrelation_change: float = 0.01
    acceptance_min: float = 0.15
    acceptance_max: float = 0.60
    dynesty_live_points: int = 400
    dynesty_dlogz: float = 0.1
    boundary_mass_trigger: float = 0.05
    mode_delta_c: float = 2.0
    mode_separation: float = 0.1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> PosteriorConfig:
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{name: values[name] for name in known if name in values})


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _safe_log_probability(
    genes: NDArray[np.float64],
    objective: DecodedObjective,
    search_space: SearchSpace,
) -> float:
    vector = np.asarray(genes, dtype=float)
    if vector.shape != (search_space.ndim,) or np.any(~np.isfinite(vector)):
        return -np.inf
    # A uniform prior in unit genes is linear-uniform for linear parameters and
    # log-uniform for normalization/NH parameters by construction.
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        return -np.inf
    try:
        statistic = float(objective(search_space.decode(vector)))
    except Exception:
        return -np.inf
    return -0.5 * statistic if np.isfinite(statistic) else -np.inf


def _initial_walkers(
    seed_genes: NDArray[np.float64], nwalkers: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    dimensions = seed_genes.size
    walkers = seed_genes + rng.normal(0.0, 0.02, size=(nwalkers, dimensions))
    walkers = np.clip(walkers, 1.0e-7, 1.0 - 1.0e-7)
    # A few diffuse walkers make the ensemble less dependent on one GA basin.
    diffuse = max(2, nwalkers // 8)
    walkers[-diffuse:] = rng.uniform(0.02, 0.98, size=(diffuse, dimensions))
    return walkers


def _physical_samples(unit_samples: NDArray[np.float64], space: SearchSpace) -> NDArray[np.float64]:
    return np.asarray(
        [[space.decode(row)[name] for name in space.names] for row in unit_samples],
        dtype=float,
    )


def boundary_mass_fraction(unit_samples: NDArray[np.float64], width: float = 0.01) -> float:
    """Fraction of draws touching any edge of the normalized prior volume."""

    samples = np.asarray(unit_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return 1.0
    at_edge = np.any((samples <= width) | (samples >= 1.0 - width), axis=1)
    return float(np.mean(at_edge))


def separated_ga_modes(
    runs: Sequence[GARunResult], *, delta_c: float = 2.0, separation: float = 0.1
) -> bool:
    """Detect competitive GA solutions separated in normalized gene space."""

    if len(runs) < 2:
        return False
    best = min(run.best_score for run in runs)
    competitive = [run for run in runs if run.best_score <= best + delta_c]
    return any(
        np.linalg.norm(first.best_genes - second.best_genes) >= separation
        for index, first in enumerate(competitive)
        for second in competitive[index + 1 :]
    )


def run_emcee(
    objective: DecodedObjective,
    search_space: SearchSpace,
    seed_parameters: Mapping[str, float] | Sequence[float] | NDArray[np.float64],
    *,
    seed: int,
    config: PosteriorConfig | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> PosteriorResult:
    """Run a checkpointed ensemble chain in normalized coordinates."""

    try:
        import emcee
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("emcee is required for ensemble posterior sampling") from error

    settings = config or PosteriorConfig()
    if isinstance(seed_parameters, Mapping):
        seed_genes = search_space.encode(seed_parameters, clip=True)
    else:
        seed_genes = np.asarray(seed_parameters, dtype=float)
    if seed_genes.shape != (search_space.ndim,):
        raise ValueError("seed must contain one value per search dimension")

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    rng = np.random.default_rng(seed)
    nwalkers = max(32, 8 * search_space.ndim)
    old_tau: NDArray[np.float64] | None = None
    completed = 0
    chain_parts: list[NDArray[np.float64]] = []
    log_parts: list[NDArray[np.float64]] = []
    acceptance_parts: list[float] = []
    state: Any = _initial_walkers(seed_genes, nwalkers, rng)
    checkpoint_converged = False

    if resume and checkpoint is not None and checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        if tuple(saved["space_signature"]) != search_space.signature:
            raise ValueError("posterior checkpoint search space does not match")
        completed = int(saved["completed"])
        state = saved["state"]
        chain_parts = [np.asarray(saved["chain"], dtype=float)]
        log_parts = [np.asarray(saved["log_probability"], dtype=float)]
        saved_tau = saved.get("old_tau")
        old_tau = None if saved_tau is None else np.asarray(saved_tau, dtype=float)
        acceptance_parts = [float(value) for value in saved.get("acceptance_parts", ())]
        checkpoint_converged = bool(saved.get("converged", False))

    sampler = emcee.EnsembleSampler(
        nwalkers,
        search_space.ndim,
        _safe_log_probability,
        args=(objective, search_space),
        moves=emcee.moves.StretchMove(a=3.0),
    )
    if completed == 0:
        sampler.random_state = np.random.RandomState(seed % (2**32)).get_state()
    converged = checkpoint_converged
    latest_tau = (
        old_tau.copy()
        if old_tau is not None
        else np.full(search_space.ndim, np.nan)
    )
    while completed < settings.max_steps and not converged:
        steps = min(settings.batch_steps, settings.max_steps - completed)
        state = sampler.run_mcmc(state, steps, progress=False)
        completed += steps
        chain_parts.append(sampler.get_chain().copy())
        log_parts.append(sampler.get_log_prob().copy())
        acceptance_parts.append(float(np.mean(sampler.acceptance_fraction)))
        combined_chain = np.concatenate(chain_parts, axis=0)
        combined_log = np.concatenate(log_parts, axis=0)

        try:
            latest_tau = np.asarray(
                emcee.autocorr.integrated_time(
                    combined_chain, tol=0, quiet=True, has_walkers=True
                ),
                dtype=float,
            )
            stable = old_tau is not None and np.all(
                np.abs(latest_tau - old_tau) / np.maximum(latest_tau, 1.0e-12)
                < settings.autocorrelation_change
            )
            long_enough = completed > settings.min_autocorrelation_times * np.max(latest_tau)
            converged = bool(stable and long_enough)
            old_tau = latest_tau
        except Exception:
            converged = False

        if checkpoint is not None:
            _atomic_pickle(
                checkpoint,
                {
                    "version": 1,
                    "space_signature": search_space.signature,
                    "completed": completed,
                    "state": state,
                    "chain": combined_chain,
                    "log_probability": combined_log,
                    "old_tau": old_tau,
                    "acceptance_parts": acceptance_parts,
                    "converged": converged,
                },
            )
            chain_parts = [combined_chain]
            log_parts = [combined_log]
        sampler.reset()
        if converged:
            break

    chain = np.concatenate(chain_parts, axis=0)
    log_probability = np.concatenate(log_parts, axis=0)
    burn = min(chain.shape[0] // 2, int(2.0 * np.nanmax(latest_tau))) if np.any(np.isfinite(latest_tau)) else chain.shape[0] // 2
    thin = max(1, int(0.5 * np.nanmin(latest_tau))) if np.any(np.isfinite(latest_tau)) else 1
    unit_samples = chain[burn::thin].reshape(-1, search_space.ndim)
    flat_log_probability = log_probability[burn::thin].reshape(-1)
    finite = np.isfinite(flat_log_probability)
    unit_samples = unit_samples[finite]
    flat_log_probability = flat_log_probability[finite]
    mean_acceptance = (
        float(np.mean(acceptance_parts)) if acceptance_parts else float("nan")
    )
    diagnostics = {
        "steps": int(completed),
        "walkers": int(nwalkers),
        "autocorrelation_time": latest_tau.tolist(),
        "acceptance_fraction": mean_acceptance,
        "acceptance_ok": bool(
            settings.acceptance_min <= mean_acceptance <= settings.acceptance_max
        ),
        "boundary_mass_fraction": boundary_mass_fraction(unit_samples),
        "burn_in": int(burn),
        "thin": int(thin),
        "seed": int(seed),
    }
    return PosteriorResult(
        parameter_names=search_space.names,
        samples=_physical_samples(unit_samples, search_space),
        log_probability=flat_log_probability,
        sampler="emcee",
        converged=converged,
        diagnostics=diagnostics,
    )


def run_dynesty(
    objective: DecodedObjective,
    search_space: SearchSpace,
    *,
    seed: int,
    config: PosteriorConfig | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> PosteriorResult:
    """Run static nested sampling as a multimodality/convergence fallback."""

    try:
        import dynesty
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("dynesty is required for nested posterior sampling") from error

    settings = config or PosteriorConfig()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if resume and checkpoint is not None and checkpoint.exists():
        sampler = dynesty.NestedSampler.restore(str(checkpoint))
    else:
        sampler = dynesty.NestedSampler(
            _DynestyLogLikelihood(objective, search_space),
            _IdentityPriorTransform(),
            search_space.ndim,
            nlive=settings.dynesty_live_points,
            rstate=np.random.default_rng(seed),
            bound="multi",
            sample="rwalk",
        )
    kwargs: dict[str, Any] = {
        "dlogz": settings.dynesty_dlogz,
        "print_progress": False,
    }
    if checkpoint is not None:
        kwargs["checkpoint_file"] = str(checkpoint)
    sampler.run_nested(**kwargs)
    result = sampler.results
    unit_samples = np.asarray(result.samples, dtype=float)
    weights = np.exp(np.asarray(result.logwt) - float(result.logz[-1]))
    weights /= np.sum(weights)
    diagnostics = {
        "live_points": int(settings.dynesty_live_points),
        "dlogz": float(settings.dynesty_dlogz),
        "log_evidence": float(result.logz[-1]),
        "log_evidence_error": float(result.logzerr[-1]),
        "evaluations": int(np.sum(result.ncall)),
        "boundary_mass_fraction": boundary_mass_fraction(unit_samples),
        "seed": int(seed),
    }
    return PosteriorResult(
        parameter_names=search_space.names,
        samples=_physical_samples(unit_samples, search_space),
        log_probability=np.asarray(result.logl, dtype=float),
        weights=weights,
        sampler="dynesty",
        converged=True,
        diagnostics=diagnostics,
    )


def run_posterior(
    objective: DecodedObjective,
    search_space: SearchSpace,
    seed_parameters: Mapping[str, float] | Sequence[float] | NDArray[np.float64],
    *,
    seed: int,
    config: PosteriorConfig | None = None,
    ga_runs: Sequence[GARunResult] = (),
    checkpoint_directory: str | Path | None = None,
    resume: bool = False,
) -> PosteriorResult:
    """Run emcee, switching to dynesty when the documented triggers fire."""

    settings = config or PosteriorConfig()
    directory = Path(checkpoint_directory) if checkpoint_directory is not None else None
    emcee_checkpoint = directory / "emcee.pkl" if directory is not None else None
    ensemble = run_emcee(
        objective,
        search_space,
        seed_parameters,
        seed=seed,
        config=settings,
        checkpoint_path=emcee_checkpoint,
        resume=resume,
    )
    use_nested = (
        separated_ga_modes(
            ga_runs,
            delta_c=settings.mode_delta_c,
            separation=settings.mode_separation,
        )
        or not ensemble.converged
        or float(ensemble.diagnostics.get("boundary_mass_fraction", 0.0))
        > settings.boundary_mass_trigger
    )
    if not use_nested:
        return ensemble
    nested_checkpoint = directory / "dynesty.save" if directory is not None else None
    child = np.random.SeedSequence(int(seed), spawn_key=(1,)).generate_state(
        2, dtype=np.uint32
    )
    nested_seed = int(child[0]) | (int(child[1]) << 32)
    nested = run_dynesty(
        objective,
        search_space,
        seed=nested_seed,
        config=settings,
        checkpoint_path=nested_checkpoint,
        resume=resume,
    )
    nested.diagnostics["fallback_reasons"] = {
        "separated_ga_modes": separated_ga_modes(
            ga_runs,
            delta_c=settings.mode_delta_c,
            separation=settings.mode_separation,
        ),
        "emcee_unconverged": not ensemble.converged,
        "emcee_boundary_mass": float(
            ensemble.diagnostics.get("boundary_mass_fraction", 0.0)
        ),
    }
    return nested


def posterior_summary(result: PosteriorResult) -> list[dict[str, Any]]:
    """Return tidy 16th/50th/84th-percentile summary records."""

    quantiles = result.quantiles()
    return [
        {
            "label": result.label,
            "sampler": result.sampler,
            "converged": result.converged,
            "parameter": name,
            "q16": float(values[0]),
            "median": float(values[1]),
            "q84": float(values[2]),
            "minus": float(values[1] - values[0]),
            "plus": float(values[2] - values[1]),
        }
        for name, values in quantiles.items()
    ]
