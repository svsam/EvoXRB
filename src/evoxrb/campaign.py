"""Reproducible stage orchestration for the synthetic EvoXRB case study."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from . import SYNTHETIC_LABEL
from .config import CaseStudyConfig, load_config, smoke_overrides
from .genetic import GAConfig, GARunResult, GeneticOptimizer
from .inference import PosteriorConfig, posterior_summary, run_posterior
from .instrument import InstrumentResponse, default_nicer_inspired_response
from .io import atomic_write_json, ensure_directories, provenance, save_npz, write_csv
from .models import SpectrumModel
from .objective import Objective, search_space_from_config
from .optimization import local_polish, multistart_scipy
from .reporting import build_report
from .simulation import derive_seed, load_spectrum, save_spectrum, simulate_spectrum
from .timing import simulate_timing_epoch
from .types import EpochTruth, PosteriorResult, SyntheticSpectrum


Profile = Literal["smoke", "full"]
FitMethod = Literal["scipy", "ga", "both"]


@dataclass(frozen=True, slots=True)
class CampaignPaths:
    """Resolved locations for generated outputs and checkpoints."""

    root: Path
    results: Path
    figures: Path
    reports: Path
    spectra: Path
    checkpoints: Path
    posterior: Path
    timing: Path

    @classmethod
    def from_config(cls, config: CaseStudyConfig) -> CampaignPaths:
        root = config.source_path.parent.parent

        def resolve(value: str | Path) -> Path:
            path = Path(value)
            return path if path.is_absolute() else root / path

        results = resolve(config.project["output_dir"])
        return cls(
            root=root,
            results=results,
            figures=resolve(config.project["figure_dir"]),
            reports=resolve(config.project["report_dir"]),
            spectra=results / "spectra",
            checkpoints=results / "checkpoints",
            posterior=results / "posterior",
            timing=results / "timing",
        )

    def create(self) -> CampaignPaths:
        ensure_directories(
            self.results,
            self.figures,
            self.reports,
            self.spectra,
            self.checkpoints,
            self.posterior,
            self.timing,
        )
        return self


def _profile_sections(config: CaseStudyConfig, profile: Profile) -> dict[str, dict[str, Any]]:
    if profile == "smoke":
        return smoke_overrides(config)
    if profile != "full":
        raise ValueError("profile must be 'smoke' or 'full'")
    return {
        "ga": dict(config.ga),
        "scipy": dict(config.scipy),
        "posterior": dict(config.posterior),
        "recovery": dict(config.recovery),
        "timing": dict(config.timing),
    }


def _ga_config(values: Mapping[str, Any]) -> GAConfig:
    maximum = int(values["max_generations"])
    return GAConfig(
        population_size=int(values["population_size"]),
        max_generations=maximum,
        tournament_size=int(values["tournament_size"]),
        crossover_probability=float(values["crossover_probability"]),
        crossover_eta=float(values["crossover_eta"]),
        mutation_eta_start=float(values["mutation_eta_start"]),
        mutation_eta_end=float(values["mutation_eta_end"]),
        mutation_anneal_generations=maximum,
        elite_fraction=float(values["elite_fraction"]),
        immigrant_fraction=float(values["immigrant_fraction"]),
        immigrant_stagnation_generations=int(values["stagnation_generations"]),
        immigrant_spread_threshold=0.02,
        min_generations=int(values["min_generations"]),
        stop_stagnation_generations=int(values["stop_window"]),
        improvement_tolerance=float(values["stop_improvement"]),
        stop_spread_threshold=float(values["stop_gene_spread"]),
    )


def epoch_truths(config: CaseStudyConfig) -> tuple[EpochTruth, ...]:
    """Return the exact twelve injected epochs encoded by configuration."""

    return tuple(
        EpochTruth(
            epoch_id=str(row["epoch_id"]),
            phase=str(row["phase"]),
            reference_mjd=float(row["reference_mjd"]),
            tin=float(row["tin"]),
            ndisk=float(row["ndisk"]),
            gamma=float(row["gamma"]),
            powerlaw_norm=float(row["norm"]),
            qpo_hz=None if row["qpo_hz"] is None else float(row["qpo_hz"]),
            exposure_s=2048.0,
            nh=float(config.search["nh_fixed"]),
        )
        for row in config.epochs
    )


def _truth_parameters(epoch: EpochTruth) -> dict[str, float]:
    return {
        "tin": epoch.tin,
        "ndisk": epoch.ndisk,
        "gamma": epoch.gamma,
        "norm": epoch.powerlaw_norm,
        "nh": epoch.nh,
    }


def _band_metrics(spectrum: SyntheticSpectrum) -> tuple[float, float]:
    energy = spectrum.detector_energy
    soft = float(np.sum(spectrum.expected_counts[(energy >= 0.5) & (energy < 2.0)]))
    hard = float(np.sum(spectrum.expected_counts[(energy >= 2.0) & (energy <= 10.0)]))
    return hard / max(soft, np.finfo(float).tiny), (soft + hard) / spectrum.exposure_s


def _mark_stage(paths: CampaignPaths, stage: str, config: CaseStudyConfig, profile: Profile) -> None:
    state_path = paths.results / "stage_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    else:
        state = {}
    state.update(
        {
            "label": SYNTHETIC_LABEL,
            "config_hash": config.digest,
            "profile": profile,
        }
    )
    state.setdefault("completed", {})[stage] = True
    atomic_write_json(state_path, state)


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def _recalculate_fit_deltas(records: list[dict[str, Any]]) -> None:
    """Express all fit statistics relative to the matching best SciPy control."""

    references: dict[tuple[Any, ...], float] = {}
    for row in records:
        if row.get("method") != "scipy":
            continue
        key = (
            row.get("config_hash"),
            row.get("profile"),
            row.get("epoch_id"),
            row.get("fit_model"),
        )
        score = float(row["statistic"])
        references[key] = min(score, references.get(key, float("inf")))
    for row in records:
        key = (
            row.get("config_hash"),
            row.get("profile"),
            row.get("epoch_id"),
            row.get("fit_model"),
        )
        if key in references:
            row["delta_c"] = float(row["statistic"]) - references[key]


def simulate_stage(
    config: CaseStudyConfig,
    *,
    profile: Profile = "full",
    resume: bool = False,
) -> tuple[InstrumentResponse, list[SyntheticSpectrum]]:
    """Generate the response and all twelve response-folded Poisson spectra."""

    paths = CampaignPaths.from_config(config).create()
    response_path = paths.results / "instrument_response.npz"
    response = default_nicer_inspired_response()
    if not (resume and response_path.exists()):
        save_npz(response_path, **response.to_npz_dict())

    spectra: list[SyntheticSpectrum] = []
    truth_records: list[dict[str, Any]] = []
    for index, epoch in enumerate(epoch_truths(config)):
        destination = paths.spectra / f"{epoch.epoch_id}.npz"
        if resume and destination.exists():
            spectrum = load_spectrum(destination)
        else:
            spectrum = simulate_spectrum(
                response,
                SpectrumModel("powerlaw", fixed_nh=epoch.nh),
                _truth_parameters(epoch),
                epoch.exposure_s,
                derive_seed(config.master_seed, 10, index),
                epoch_id=epoch.epoch_id,
                phase=epoch.phase,
                reference_mjd=epoch.reference_mjd,
            )
            save_spectrum(destination, spectrum)
        spectra.append(spectrum)
        hardness, intensity = _band_metrics(spectrum)
        configured = config.epochs[index]
        truth_records.append(
            {
                "label": SYNTHETIC_LABEL,
                "epoch_id": epoch.epoch_id,
                "phase": epoch.phase,
                "reference_mjd": epoch.reference_mjd,
                "tin": epoch.tin,
                "ndisk": epoch.ndisk,
                "gamma": epoch.gamma,
                "norm": epoch.powerlaw_norm,
                "nh": epoch.nh,
                "qpo_hz": epoch.qpo_hz,
                "fractional_rms": float(configured["fractional_rms"]),
                "exposure_s": epoch.exposure_s,
                "hardness_ratio": hardness,
                "intensity_cps": intensity,
                "seed": spectrum.seed,
                "truth_model": "educational absorbed disk-like + power law",
                "context": "reference date/phase only; not an analysed observation",
            }
        )
    write_csv(paths.results / "truth.csv", truth_records)
    atomic_write_json(
        paths.results / "provenance.json",
        {
            **provenance(config.digest, config.master_seed),
            "profile": profile,
            "response_shape": list(response.redistribution.shape),
            "fit_channels": int(np.sum(response.fit_mask)),
        },
    )
    _mark_stage(paths, "simulate", config, profile)
    return response, spectra


def _load_response(path: Path) -> InstrumentResponse:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    return InstrumentResponse.from_npz_dict(payload)


def load_simulation(config: CaseStudyConfig) -> tuple[InstrumentResponse, list[SyntheticSpectrum]]:
    paths = CampaignPaths.from_config(config)
    response_path = paths.results / "instrument_response.npz"
    if not response_path.exists():
        return simulate_stage(config)
    response = _load_response(response_path)
    spectra = [load_spectrum(paths.spectra / f"{epoch.epoch_id}.npz") for epoch in epoch_truths(config)]
    return response, spectra


def _base_fit_record(
    config: CaseStudyConfig,
    profile: Profile,
    spectrum: SyntheticSpectrum,
    fit_model: str,
    method: str,
    stage: str,
    seed: int,
    parameters: Mapping[str, float],
    statistic: float,
    evaluations: int,
    runtime_s: float,
    *,
    success: bool,
    boundary_hits: int,
    generations: int | None = None,
    stop_reason: str | None = None,
    genes: Sequence[float] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": SYNTHETIC_LABEL,
        "config_hash": config.digest,
        "profile": profile,
        "epoch_id": spectrum.epoch_id,
        "phase": spectrum.phase,
        "reference_mjd": spectrum.reference_mjd,
        "truth_model": spectrum.truth_model,
        "fit_model": fit_model,
        "method": method,
        "stage": stage,
        "seed": int(seed),
        "statistic": float(statistic),
        "evaluations": int(evaluations),
        "wall_time_s": float(runtime_s),
        "success": bool(success),
        "boundary_hits": int(boundary_hits),
        "generations": generations,
        "stop_reason": stop_reason,
        **{name: float(value) for name, value in parameters.items()},
    }
    if genes is not None:
        for name, value in zip(parameters, genes, strict=True):
            record[f"gene_{name}"] = float(value)
    return record


def _optimizer_records(
    objective: Objective,
    config: CaseStudyConfig,
    profile: Profile,
    fit_model: str,
    search_space: Any,
    sections: Mapping[str, Mapping[str, Any]],
    *,
    epoch_index: int,
    model_index: int,
    method: FitMethod,
    checkpoint_directory: Path,
    resume: bool,
    ga_seed_count: int | None = None,
) -> tuple[list[dict[str, Any]], list[GARunResult]]:
    spectrum = objective.spectrum
    rows: list[dict[str, Any]] = []
    ga_runs: list[GARunResult] = []
    scipy_best = None

    if method in ("scipy", "both"):
        seed = derive_seed(config.master_seed, 20, epoch_index, model_index)
        scipy_result = multistart_scipy(
            objective.evaluate,
            search_space,
            n_starts=int(sections["scipy"]["starts"]),
            seed=seed,
            method=str(sections["scipy"].get("method", "L-BFGS-B")),
        )
        scipy_best = scipy_result.best
        rows.append(
            _base_fit_record(
                config,
                profile,
                spectrum,
                fit_model,
                "scipy",
                "best_multistart",
                seed,
                scipy_best.parameters,
                scipy_best.score,
                scipy_result.evaluations,
                scipy_result.runtime_seconds,
                success=scipy_best.success,
                boundary_hits=int(np.count_nonzero((scipy_best.genes <= 1e-8) | (scipy_best.genes >= 1 - 1e-8))),
                genes=scipy_best.genes,
            )
        )

    if method in ("ga", "both"):
        ga_values = sections["ga"]
        optimizer = GeneticOptimizer(search_space, _ga_config(ga_values))
        count = int(ga_values["seeds"] if ga_seed_count is None else ga_seed_count)
        for run_index in range(count):
            seed = derive_seed(config.master_seed, 30, epoch_index, model_index, run_index)
            checkpoint = checkpoint_directory / f"seed_{run_index + 1:02d}.npz"
            result = optimizer.optimize(
                objective.evaluate,
                seed=seed,
                checkpoint_path=checkpoint,
                resume=resume and checkpoint.exists(),
            )
            ga_runs.append(result)
            rows.append(
                _base_fit_record(
                    config,
                    profile,
                    spectrum,
                    fit_model,
                    "ga",
                    "raw",
                    seed,
                    result.best_parameters,
                    result.best_score,
                    result.evaluations,
                    result.runtime_seconds,
                    success=np.isfinite(result.best_score),
                    boundary_hits=result.boundary_hits,
                    generations=result.generations,
                    stop_reason=result.stop_reason,
                    genes=result.best_genes,
                )
            )
            polished = local_polish(objective.evaluate, search_space, result)
            rows.append(
                _base_fit_record(
                    config,
                    profile,
                    spectrum,
                    fit_model,
                    "ga+scipy",
                    "polished",
                    seed,
                    polished.parameters,
                    polished.score,
                    result.evaluations + polished.evaluations,
                    result.runtime_seconds + polished.runtime_seconds,
                    success=polished.success or np.isfinite(polished.score),
                    boundary_hits=int(np.count_nonzero((polished.genes <= 1e-8) | (polished.genes >= 1 - 1e-8))),
                    generations=result.generations,
                    stop_reason=result.stop_reason,
                    genes=polished.genes,
                )
            )
            atomic_write_json(checkpoint.with_suffix(".summary.json"), result.summary())

    reference = scipy_best.score if scipy_best is not None else min(row["statistic"] for row in rows)
    for row in rows:
        row["delta_c"] = float(row["statistic"] - reference)
    return rows, ga_runs


def fit_stage(
    config: CaseStudyConfig,
    *,
    profile: Profile = "full",
    method: FitMethod = "both",
    resume: bool = False,
) -> Path:
    """Fit the synthetic epochs with raw GA, SciPy, and GA-polish outputs."""

    paths = CampaignPaths.from_config(config).create()
    response, spectra = simulate_stage(config, profile=profile, resume=True)
    sections = _profile_sections(config, profile)
    selected = spectra if profile == "full" else [spectra[index] for index in (0, 7, 9)]
    output = paths.results / "fit_results.csv"
    existing = _read_csv_records(output)
    if not resume:
        replaced_methods = (
            {"scipy"}
            if method == "scipy"
            else ({"ga", "ga+scipy"} if method == "ga" else {"scipy", "ga", "ga+scipy"})
        )
        existing = [
            row
            for row in existing
            if row.get("config_hash") != config.digest
            or row.get("profile") != profile
            or row.get("method") not in replaced_methods
        ]
    records = list(existing)

    for spectrum in selected:
        epoch_index = next(index for index, epoch in enumerate(epoch_truths(config)) if epoch.epoch_id == spectrum.epoch_id)
        for model_index, continuum in enumerate(("powerlaw", "cutoff")):
            fit_name = "educational_powerlaw" if continuum == "powerlaw" else "educational_cutoff_surrogate"
            expected_methods = {"scipy"} if method == "scipy" else ({"ga", "ga+scipy"} if method == "ga" else {"scipy", "ga", "ga+scipy"})
            if resume and expected_methods.issubset(
                {
                    str(row.get("method"))
                    for row in records
                    if row.get("config_hash") == config.digest
                    and row.get("profile") == profile
                    and row.get("epoch_id") == spectrum.epoch_id
                    and row.get("fit_model") == fit_name
                }
            ):
                continue
            objective = Objective(
                spectrum,
                response,
                SpectrumModel(continuum, fixed_nh=float(config.search["nh_fixed"])),
            )
            new_rows, _ = _optimizer_records(
                objective,
                config,
                profile,
                fit_name,
                search_space_from_config(config),
                sections,
                epoch_index=epoch_index,
                model_index=model_index,
                method=method,
                checkpoint_directory=paths.checkpoints / "fits" / profile / spectrum.epoch_id / continuum,
                resume=resume,
            )
            records.extend(new_rows)
            _recalculate_fit_deltas(records)
            write_csv(output, records)

    # One explicit free-column sensitivity run documents the NH degeneracy.
    sensitivity = spectra[7]
    sensitivity_name = "educational_powerlaw_free_nh_sensitivity"
    expected_methods = {"scipy"} if method == "scipy" else ({"ga", "ga+scipy"} if method == "ga" else {"scipy", "ga", "ga+scipy"})
    existing_sensitivity_methods = {
        str(row.get("method"))
        for row in records
        if row.get("config_hash") == config.digest
        and row.get("profile") == profile
        and row.get("epoch_id") == "E08"
        and row.get("fit_model") == sensitivity_name
    }
    if not (resume and expected_methods.issubset(existing_sensitivity_methods)):
        objective = Objective(sensitivity, response, SpectrumModel("powerlaw", fixed_nh=None))
        new_rows, _ = _optimizer_records(
            objective,
            config,
            profile,
            sensitivity_name,
            search_space_from_config(config, free_nh=True),
            sections,
            epoch_index=7,
            model_index=2,
            method=method,
            checkpoint_directory=paths.checkpoints / "fits" / profile / "E08" / "free_nh",
            resume=resume,
        )
        records.extend(new_rows)
    _recalculate_fit_deltas(records)
    write_csv(output, records)
    _mark_stage(paths, "fit", config, profile)
    return output


def _recovery_truth() -> dict[str, float]:
    return {"tin": 0.70, "ndisk": 8_000.0, "gamma": 2.10, "norm": 0.25, "nh": 0.15}


def recovery_stage(
    config: CaseStudyConfig,
    *,
    profile: Profile = "full",
    resume: bool = False,
) -> Path:
    """Run the exposure/noise/model-misspecification recovery campaign."""

    paths = CampaignPaths.from_config(config).create()
    sections = _profile_sections(config, profile)
    response = default_nicer_inspired_response()
    space = search_space_from_config(config)
    truth = _recovery_truth()
    recovery = sections["recovery"]
    datasets: list[tuple[str, float, int, str, str, int]] = []
    high_exposure = float(recovery["high_signal_exposure_s"])
    datasets.append(("high_signal_correct", high_exposure, 0, "powerlaw", "powerlaw", 5 if profile == "full" else 1))
    for exposure in recovery["exposure_levels_s"]:
        for realization in range(int(recovery["realizations"])):
            seed_count = 5 if profile == "full" else 1
            datasets.append(("correct_model", float(exposure), realization, "powerlaw", "powerlaw", seed_count))
            datasets.append(("cutoff_truth_powerlaw_fit", float(exposure), realization, "cutoff", "powerlaw", seed_count))

    output = paths.results / "recovery_results.csv"
    optimizer_output = paths.results / "recovery_optimizers.csv"
    parameter_rows = _read_csv_records(output) if resume else []
    optimizer_rows = _read_csv_records(optimizer_output) if resume else []
    for dataset_index, (scenario, exposure, realization, truth_model, fit_model, ga_seeds) in enumerate(datasets):
        matching = [
            row
            for row in optimizer_rows
            if row.get("config_hash") == config.digest
            and row.get("profile") == profile
            and row.get("scenario") == scenario
            and float(row.get("exposure_s", -1.0)) == exposure
            and int(row.get("realization", -1)) == realization
        ]
        counts = {
            name: sum(row.get("method") == name for row in matching)
            for name in ("scipy", "ga", "ga+scipy")
        }
        if resume and counts["scipy"] >= 1 and counts["ga"] >= ga_seeds and counts["ga+scipy"] >= ga_seeds:
            continue
        if matching:
            optimizer_rows = [row for row in optimizer_rows if row not in matching]
            parameter_rows = [
                row
                for row in parameter_rows
                if not (
                    row.get("config_hash") == config.digest
                    and row.get("profile") == profile
                    and row.get("scenario") == scenario
                    and float(row.get("exposure_s", -1.0)) == exposure
                    and int(row.get("realization", -1)) == realization
                )
            ]
        seed = derive_seed(config.master_seed, 40, dataset_index, realization)
        spectrum = simulate_spectrum(
            response,
            SpectrumModel(truth_model, fixed_nh=truth["nh"]),
            truth,
            exposure,
            seed,
            epoch_id=f"R{dataset_index:03d}",
            phase=scenario,
        )
        objective = Objective(spectrum, response, SpectrumModel(fit_model, fixed_nh=truth["nh"]))
        fit_rows, _ = _optimizer_records(
            objective,
            config,
            profile,
            f"educational_{fit_model}",
            space,
            sections,
            epoch_index=1000 + dataset_index,
            model_index=0,
            method="both",
            checkpoint_directory=paths.checkpoints / "recovery" / profile / f"dataset_{dataset_index:03d}",
            resume=resume,
            ga_seed_count=ga_seeds,
        )
        scipy_reference = min(row["statistic"] for row in fit_rows if row["method"] == "scipy")
        for row in fit_rows:
            row.update(
                {
                    "scenario": scenario,
                    "exposure_s": exposure,
                    "realization": realization,
                    "injected_model": truth_model,
                    "delta_c": float(row["statistic"] - scipy_reference),
                }
            )
            optimizer_rows.append(row)
            for parameter in space.names:
                estimate = float(row[parameter])
                parameter_rows.append(
                    {
                        "label": SYNTHETIC_LABEL,
                        "config_hash": config.digest,
                        "profile": profile,
                        "scenario": scenario,
                        "exposure_s": exposure,
                        "realization": realization,
                        "injected_model": truth_model,
                        "fit_model": fit_model,
                        "method": row["method"],
                        "stage": row["stage"],
                        "seed": row["seed"],
                        "parameter": parameter,
                        "truth": truth[parameter],
                        "estimate": estimate,
                        "error": estimate - truth[parameter],
                        "relative_error": (estimate - truth[parameter]) / truth[parameter],
                        "delta_c": row["delta_c"],
                        "evaluations": row["evaluations"],
                        "wall_time_s": row["wall_time_s"],
                        "boundary_hits": row["boundary_hits"],
                        "failure": (
                            not bool(row["success"])
                            or not np.isfinite(row["statistic"])
                            or int(row["boundary_hits"]) > 0
                            or float(row["delta_c"]) > 1.0
                        ),
                    }
                )
        write_csv(output, parameter_rows)
        write_csv(optimizer_output, optimizer_rows)

    frame = pd.DataFrame(parameter_rows)
    if not frame.empty:
        grouped = frame.groupby(["scenario", "exposure_s", "method", "stage", "parameter"], dropna=False)
        summary = grouped.agg(
            bias=("error", "mean"),
            rmse=("error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            failure_rate=("failure", "mean"),
            boundary_hits=("boundary_hits", "sum"),
            mean_delta_c=("delta_c", "mean"),
            mean_evaluations=("evaluations", "mean"),
            mean_runtime_s=("wall_time_s", "mean"),
            samples=("estimate", "size"),
        ).reset_index()
        summary.insert(0, "label", SYNTHETIC_LABEL)
        summary.to_csv(paths.results / "recovery_summary.csv", index=False)
    _mark_stage(paths, "recovery", config, profile)
    return output


def _load_fit_seed(config: CaseStudyConfig, profile: Profile, epoch_id: str) -> tuple[dict[str, float], list[Any]]:
    frame = pd.read_csv(CampaignPaths.from_config(config).results / "fit_results.csv")
    candidates = frame[
        (frame["config_hash"] == config.digest)
        & (frame["profile"] == profile)
        & (frame["epoch_id"] == epoch_id)
        & (frame["fit_model"] == "educational_powerlaw")
        & (frame["stage"].isin(["polished", "best_multistart"]))
    ].sort_values("statistic")
    if candidates.empty:
        raise RuntimeError(f"no completed power-law fit is available for {epoch_id}")
    row = candidates.iloc[0]
    parameters = {name: float(row[name]) for name in ("tin", "ndisk", "gamma", "norm")}

    @dataclass(slots=True)
    class StoredMode:
        best_score: float
        best_genes: np.ndarray

    raw = frame[
        (frame["config_hash"] == config.digest)
        & (frame["profile"] == profile)
        & (frame["epoch_id"] == epoch_id)
        & (frame["fit_model"] == "educational_powerlaw")
        & (frame["method"] == "ga")
        & (frame["stage"] == "raw")
    ]
    modes = [
        StoredMode(
            float(item["statistic"]),
            np.asarray([item[f"gene_{name}"] for name in ("tin", "ndisk", "gamma", "norm")], dtype=float),
        )
        for _, item in raw.iterrows()
    ]
    return parameters, modes


def _save_posterior(path: Path, result: PosteriorResult) -> None:
    save_npz(path, **result.to_npz_dict())


def _load_posterior(path: Path) -> PosteriorResult:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    return PosteriorResult.from_npz_dict(payload)


def inference_stage(
    config: CaseStudyConfig,
    *,
    profile: Profile = "full",
    resume: bool = False,
) -> Path:
    """Run GA-seeded posterior inference for representative outburst states."""

    paths = CampaignPaths.from_config(config).create()
    fit_path = paths.results / "fit_results.csv"
    if not fit_path.exists():
        fit_stage(config, profile=profile, method="both", resume=resume)
    response, spectra = load_simulation(config)
    by_epoch = {spectrum.epoch_id: spectrum for spectrum in spectra}
    selected = ("E08",) if profile == "smoke" else ("E01", "E08", "E10")
    sections = _profile_sections(config, profile)
    posterior_values = dict(sections["posterior"])
    if profile == "smoke":
        posterior_values["min_autocorrelation_times"] = 3.0
        posterior_values["autocorrelation_change"] = 0.50
        posterior_values["dynesty_dlogz"] = 0.5
    settings = PosteriorConfig.from_mapping(posterior_values)
    space = search_space_from_config(config)
    summaries: list[dict[str, Any]] = []
    for epoch_index, epoch_id in enumerate(selected):
        destination = paths.posterior / profile / f"{epoch_id}_powerlaw.npz"
        seed_parameters, ga_modes = _load_fit_seed(config, profile, epoch_id)
        spectrum = by_epoch[epoch_id]
        objective = Objective(spectrum, response, SpectrumModel("powerlaw", fixed_nh=float(config.search["nh_fixed"])))
        if resume and destination.exists():
            result = _load_posterior(destination)
        else:
            result = run_posterior(
                objective.evaluate,
                space,
                seed_parameters,
                seed=derive_seed(config.master_seed, 50, epoch_index),
                config=settings,
                ga_runs=ga_modes,  # type: ignore[arg-type]
                checkpoint_directory=paths.checkpoints / "posterior" / profile / epoch_id,
                resume=resume,
            )
            _save_posterior(destination, result)
        for row in posterior_summary(result):
            row.update(
                {
                    "config_hash": config.digest,
                    "profile": profile,
                    "epoch_id": epoch_id,
                    "fit_model": "educational_powerlaw",
                    "error_minus": row.pop("minus"),
                    "error_plus": row.pop("plus"),
                }
            )
            summaries.append(row)
    output = paths.results / "posterior_summary.csv"
    write_csv(output, summaries)
    _mark_stage(paths, "infer", config, profile)
    return output


def timing_stage(
    config: CaseStudyConfig,
    *,
    profile: Profile = "full",
    resume: bool = False,
) -> Path:
    """Simulate and fit the synthetic broadband noise and type-C-like QPOs."""

    paths = CampaignPaths.from_config(config).create()
    epochs = epoch_truths(config)
    selected = epochs if profile == "full" else tuple(epochs[index] for index in (0, 7, 9))
    configured = {str(row["epoch_id"]): row for row in config.epochs}
    output = paths.results / "timing_results.csv"
    if resume and output.exists():
        completed = pd.read_csv(output)
        if (
            "epoch_id" in completed
            and set(completed["epoch_id"].astype(str)) >= {epoch.epoch_id for epoch in selected}
            and "config_hash" in completed
            and set(completed["config_hash"].astype(str)) == {config.digest}
        ):
            return output
    records: list[dict[str, Any]] = []
    for index, epoch in enumerate(selected):
        seed = derive_seed(config.master_seed, 60, index)
        result = simulate_timing_epoch(
            epoch,
            profile=profile,
            fractional_rms=float(configured[epoch.epoch_id]["fractional_rms"]),
            seed=seed,
        )
        save_npz(
            paths.timing / f"{epoch.epoch_id}.npz",
            frequencies_hz=result.frequencies_hz,
            power=result.power,
            power_error=result.power_error,
            model_power=result.model_power,
        )
        row = result.to_record()
        row.update(
            {
                "label": SYNTHETIC_LABEL,
                "config_hash": config.digest,
                "profile": profile,
                "epoch_id": epoch.epoch_id,
                "phase": epoch.phase,
                "reference_mjd": epoch.reference_mjd,
                "hardness_ratio": float(result.metadata.get("hardness", np.nan)),
                "intensity_cps": float(result.metadata.get("mean_rate_hz", np.nan)),
                "qpo_injected_hz": epoch.qpo_hz,
                "qpo_centroid_hz": result.centroid_hz,
                "significance": result.amplitude_significance,
                "seed": seed,
            }
        )
        records.append(row)
    write_csv(output, records)
    _mark_stage(paths, "timing", config, profile)
    return output


def figures_stage(config: CaseStudyConfig, *, profile: Profile = "full") -> list[Path]:
    """Generate the full selected figure set from machine-readable artifacts."""

    from .plotting import generate_selected_figures

    paths = CampaignPaths.from_config(config).create()
    ga_candidates = sorted(
        (paths.checkpoints / "fits" / profile / "E08" / "powerlaw").glob("seed_*.npz")
    )
    posterior_path = paths.posterior / profile / "E08_powerlaw.npz"

    def profile_table(filename: str) -> pd.DataFrame:
        path = paths.results / filename
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        if "profile" in frame.columns:
            frame = frame[frame["profile"].astype(str) == profile]
        return frame

    products = generate_selected_figures(
        paths.figures,
        response=paths.results / "instrument_response.npz",
        spectrum=paths.spectra / "E08.npz",
        ga_result=ga_candidates[0] if ga_candidates else None,
        recovery_rows=profile_table("recovery_results.csv"),
        fit_rows=profile_table("fit_results.csv"),
        posterior=posterior_path if posterior_path.exists() else paths.results / "posterior_summary.csv",
        timing_rows=profile_table("timing_results.csv"),
        spectral_rows=paths.results / "truth.csv",
        hid_rows=paths.results / "truth.csv",
    )
    _mark_stage(paths, "figures", config, profile)
    return list(products.values())


def report_stage(config: CaseStudyConfig, *, profile: Profile = "full") -> Path:
    paths = CampaignPaths.from_config(config).create()
    report = build_report(
        paths.results,
        paths.figures,
        paths.reports,
        config_hash=config.digest,
        profile=profile,
    )
    readme = paths.reports / "README.md"
    shutil.copyfile(report, readme)
    _mark_stage(paths, "report", config, profile)
    return report


def run_case_study(
    config: CaseStudyConfig | str | Path = "config/case_study.yaml",
    *,
    profile: Profile = "smoke",
    resume: bool = False,
) -> dict[str, Any]:
    """Run every case-study stage and return its generated entry points."""

    loaded = load_config(config) if not isinstance(config, CaseStudyConfig) else config
    started = time.perf_counter()
    paths = CampaignPaths.from_config(loaded).create()
    simulate_stage(loaded, profile=profile, resume=resume)
    fit_path = fit_stage(loaded, profile=profile, method="both", resume=resume)
    recovery_path = recovery_stage(loaded, profile=profile, resume=resume)
    posterior_path = inference_stage(loaded, profile=profile, resume=resume)
    timing_path = timing_stage(loaded, profile=profile, resume=resume)
    figures = figures_stage(loaded, profile=profile)
    report = report_stage(loaded, profile=profile)
    runtime = time.perf_counter() - started
    run_summary = {
        "label": SYNTHETIC_LABEL,
        "config_hash": loaded.digest,
        "master_seed": loaded.master_seed,
        "profile": profile,
        "resume": bool(resume),
        "runtime_seconds": runtime,
        "fit_results": str(fit_path),
        "recovery_results": str(recovery_path),
        "posterior_summary": str(posterior_path),
        "timing_results": str(timing_path),
        "figures": [str(path) for path in figures],
        "report": str(report),
    }
    atomic_write_json(paths.results / "run_summary.json", run_summary)
    provenance_path = paths.results / "provenance.json"
    provenance_record = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_record["runtime_seconds"] = runtime
    atomic_write_json(provenance_path, provenance_record)

    from .validation import validate_case_study

    acceptance = validate_case_study(
        loaded,
        profile=profile,
        results_dir=paths.results,
        figure_dir=paths.figures,
        report_dir=paths.reports,
    )
    atomic_write_json(paths.results / "acceptance.json", acceptance.to_dict())
    # Re-render once provenance and acceptance are final so the portfolio
    # report records the same completed state as the machine-readable summary.
    report = report_stage(loaded, profile=profile)
    run_summary["report"] = str(report)
    run_summary["acceptance_valid"] = acceptance.valid
    run_summary["acceptance_errors"] = list(acceptance.errors)
    atomic_write_json(paths.results / "run_summary.json", run_summary)
    if not acceptance.valid:
        raise RuntimeError("artifact acceptance failed: " + "; ".join(acceptance.errors))
    return run_summary
