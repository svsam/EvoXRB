"""Headless plotting helpers for the synthetic EvoXRB case study.

Every saved figure carries the visible label ``Synthetic / NICER-inspired``.
The public helpers accept live result objects as well as CSV/NPZ artifacts and
produce an informative placeholder instead of failing when a stage has no
completed rows yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

# This module is used by the Windows CLI and CI, neither of which should need a
# display server.  The backend must be selected before pyplot is imported.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from .genetic import load_ga_checkpoint_history
from .instrument import InstrumentResponse, default_nicer_inspired_response
from .types import PosteriorResult, SYNTHETIC_LABEL, SyntheticSpectrum


PathLike = str | Path
TableLike = pd.DataFrame | Mapping[str, Any] | Sequence[Any] | PathLike | None

_BLUE = "#2864a8"
_ORANGE = "#dd7831"
_GREEN = "#3c8d5a"
_PURPLE = "#7851a9"
_RED = "#b5413e"
_GREY = "#65707b"
_PARAMETER_LABELS = {
    "tin": r"$T_{\rm in}$ (keV)",
    "ndisk": r"$N_{\rm disk}$",
    "gamma": r"$\Gamma$",
    "k": r"$K$",
    "norm": r"$K$",
    "powerlaw_norm": r"$K$",
    "nh": r"$N_{\rm H}$",
}


def _new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] = (8.0, 5.0),
    sharex: bool = False,
) -> tuple[Figure, NDArray[Any]]:
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=sharex,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    return fig, np.asarray(axes, dtype=object)


def _finish(fig: Figure, output_path: PathLike) -> Path:
    """Apply the mandatory visible label and atomically-ish save a figure."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.text(
        0.5,
        0.995,
        SYNTHETIC_LABEL,
        ha="center",
        va="top",
        color=_RED,
        fontsize=10,
        fontweight="bold",
    )
    try:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    except ValueError:
        # Some third-party Matplotlib layouts cannot be tightened; the label
        # and explicit figure sizes still make the result usable.
        pass
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    fig.savefig(temporary, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    temporary.replace(destination)
    return destination


def _empty(ax: Axes, message: str = "No completed rows available") -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=_GREY,
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d5d9dc")


def _load_npz(path: PathLike) -> dict[str, NDArray[Any]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _records_from_sequence(source: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in source:
        if isinstance(item, Mapping):
            records.append(dict(item))
        elif hasattr(item, "to_record"):
            records.append(dict(item.to_record()))
        elif hasattr(item, "summary"):
            records.append(dict(item.summary()))
        elif is_dataclass(item):
            records.append(asdict(item))
    return records


def _npz_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert compatible one-dimensional NPZ arrays to a tidy frame."""

    lengths = [
        np.asarray(value).size
        for value in payload.values()
        if np.asarray(value).ndim == 1 and np.asarray(value).size > 1
    ]
    if not lengths:
        scalar = {
            name: np.asarray(value).reshape(()).item()
            for name, value in payload.items()
            if np.asarray(value).size == 1
        }
        return pd.DataFrame([scalar]) if scalar else pd.DataFrame()
    row_count = max(set(lengths), key=lengths.count)
    columns: dict[str, Any] = {}
    for name, value in payload.items():
        array = np.asarray(value)
        if array.ndim == 1 and array.size == row_count:
            columns[name] = array
        elif array.size == 1:
            columns[name] = np.repeat(array.reshape(()).item(), row_count)
    return pd.DataFrame(columns)


def _as_frame(source: TableLike) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame()
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return pd.DataFrame()
        if path.suffix.casefold() == ".csv":
            try:
                return pd.read_csv(path)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                return pd.DataFrame()
        if path.suffix.casefold() in (".json", ".jsonl"):
            try:
                return pd.read_json(path, lines=path.suffix.casefold() == ".jsonl")
            except (OSError, ValueError):
                return pd.DataFrame()
        if path.suffix.casefold() == ".npz":
            try:
                return _npz_frame(_load_npz(path))
            except (OSError, KeyError, ValueError):
                return pd.DataFrame()
        return pd.DataFrame()
    if isinstance(source, Mapping):
        try:
            return _npz_frame(source)
        except (TypeError, ValueError):
            return pd.DataFrame([dict(source)])
    if isinstance(source, Sequence):
        return pd.DataFrame.from_records(_records_from_sequence(source))
    if hasattr(source, "to_record"):
        return pd.DataFrame.from_records([source.to_record()])
    if hasattr(source, "summary"):
        return pd.DataFrame.from_records([source.summary()])
    if is_dataclass(source):
        return pd.DataFrame.from_records([asdict(source)])
    return pd.DataFrame()


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lower = {str(column).casefold(): str(column) for column in frame.columns}
    for alias in aliases:
        match = lower.get(alias.casefold())
        if match is not None:
            return match
    return None


def _numeric(frame: pd.DataFrame, column: str | None) -> NDArray[np.float64]:
    if column is None or column not in frame:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _accepted_qpo_mask(frame: pd.DataFrame) -> NDArray[np.bool_]:
    """Select rows explicitly accepted by the type-C-like detection rule."""

    detected = _find_column(frame, ("detected", "qpo_detected"))
    if detected is not None:
        values = frame[detected]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).to_numpy(dtype=bool)
        return values.astype(str).str.casefold().isin(("true", "1", "yes")).to_numpy()
    classification = _find_column(frame, ("classification", "qpo_classification"))
    if classification is not None:
        return (
            frame[classification]
            .astype(str)
            .str.casefold()
            .eq("type-c-like")
            .to_numpy()
        )
    return np.ones(len(frame), dtype=bool)


def _mapping_from_source(source: Any) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if isinstance(source, (str, Path)) and Path(source).suffix.casefold() == ".npz":
        try:
            return _load_npz(source)
        except (OSError, ValueError):
            return {}
    return {}


def _get_array(source: Any, *aliases: str) -> NDArray[np.float64]:
    for alias in aliases:
        if hasattr(source, alias):
            try:
                return np.asarray(getattr(source, alias), dtype=np.float64)
            except (TypeError, ValueError):
                continue
    mapping = _mapping_from_source(source)
    lower = {str(name).casefold(): name for name in mapping}
    for alias in aliases:
        key = lower.get(alias.casefold())
        if key is not None:
            try:
                return np.asarray(mapping[key], dtype=np.float64)
            except (TypeError, ValueError):
                continue
    return np.asarray([], dtype=np.float64)


def _instrument(source: InstrumentResponse | Mapping[str, Any] | PathLike | None) -> InstrumentResponse:
    if isinstance(source, InstrumentResponse):
        return source
    if source is None:
        return default_nicer_inspired_response()
    payload = _mapping_from_source(source)
    try:
        return InstrumentResponse.from_npz_dict(payload)
    except (KeyError, TypeError, ValueError):
        return default_nicer_inspired_response()


def _spectrum(source: SyntheticSpectrum | Mapping[str, Any] | PathLike | None) -> SyntheticSpectrum | None:
    if isinstance(source, SyntheticSpectrum):
        return source
    if source is None:
        return None
    payload = _mapping_from_source(source)
    try:
        return SyntheticSpectrum.from_npz_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def plot_response_area(
    response: InstrumentResponse | Mapping[str, Any] | PathLike | None,
    output_path: PathLike,
) -> Path:
    """Plot effective area/background and the Gaussian redistribution matrix."""

    instrument = _instrument(response)
    fig, axes = _new_figure(1, 2, figsize=(11.0, 4.4))
    area_ax, matrix_ax = axes[0]

    area_ax.plot(instrument.true_energy, instrument.effective_area, color=_BLUE, lw=2)
    area_ax.set(xlabel="True energy (keV)", ylabel=r"Effective area (cm$^2$)")
    area_ax.set_title("Educational effective area")
    area_ax.grid(alpha=0.22)
    background_ax = area_ax.twinx()
    background_ax.plot(
        instrument.detector_energy,
        instrument.background_rate_density,
        color=_ORANGE,
        lw=1.4,
        alpha=0.85,
    )
    background_ax.set_ylabel(r"Background density (count s$^{-1}$ keV$^{-1}$)")
    background_ax.tick_params(axis="y", colors=_ORANGE)

    log_matrix = np.log10(np.maximum(instrument.redistribution, 1.0e-12))
    image = matrix_ax.imshow(
        log_matrix,
        origin="lower",
        aspect="auto",
        extent=(
            instrument.true_edges[0],
            instrument.true_edges[-1],
            instrument.detector_edges[0],
            instrument.detector_edges[-1],
        ),
        cmap="magma",
        vmin=-6.0,
        vmax=0.0,
    )
    matrix_ax.plot([0.2, 12.0], [0.2, 12.0], color="white", lw=0.7, alpha=0.6)
    matrix_ax.set(
        xlabel="True energy (keV)",
        ylabel="Detector energy (keV)",
        title="Gaussian redistribution",
    )
    colorbar = fig.colorbar(image, ax=matrix_ax, pad=0.02)
    colorbar.set_label(r"$\log_{10}$ redistribution probability")
    return _finish(fig, output_path)


def plot_folded_spectrum(
    spectrum: SyntheticSpectrum | Mapping[str, Any] | PathLike | None,
    output_path: PathLike,
    *,
    objective: Any | None = None,
    parameters: Mapping[str, float] | None = None,
    expected_counts: ArrayLike | None = None,
) -> Path:
    """Plot a folded count spectrum and Poisson standardized residuals."""

    data = _spectrum(spectrum)
    fig, axes = _new_figure(2, 1, figsize=(8.5, 6.8), sharex=True)
    spectrum_ax, residual_ax = axes[:, 0]
    if data is None:
        _empty(spectrum_ax, "No synthetic spectrum artifact available")
        _empty(residual_ax, "Residuals unavailable")
        return _finish(fig, output_path)

    model_counts: NDArray[np.float64]
    if expected_counts is not None:
        model_counts = np.asarray(expected_counts, dtype=np.float64)
    elif objective is not None and parameters is not None:
        try:
            model_counts = np.asarray(objective.expected_counts(parameters), dtype=np.float64)
        except (AttributeError, ArithmeticError, KeyError, TypeError, ValueError):
            model_counts = data.expected_counts
    else:
        model_counts = data.expected_counts
    if model_counts.shape != data.counts.shape or np.any(~np.isfinite(model_counts)):
        model_counts = data.expected_counts

    width = np.diff(data.detector_edges)
    scale = np.maximum(data.exposure_s * width, np.finfo(float).tiny)
    observed_rate = data.counts / scale
    model_rate = model_counts / scale
    uncertainty = np.sqrt(np.maximum(data.counts, 1.0)) / scale
    spectrum_ax.errorbar(
        data.detector_energy,
        observed_rate,
        yerr=uncertainty,
        fmt=".",
        ms=2.8,
        lw=0.5,
        color="black",
        alpha=0.75,
        label="Poisson realization",
    )
    spectrum_ax.plot(
        data.detector_energy,
        model_rate,
        color=_ORANGE,
        lw=1.7,
        label="Response-folded expectation",
    )
    positive = np.concatenate((observed_rate[observed_rate > 0], model_rate[model_rate > 0]))
    if positive.size:
        spectrum_ax.set_yscale("log")
        spectrum_ax.set_ylim(max(float(np.min(positive)) * 0.5, 1.0e-4), float(np.max(positive)) * 2.0)
    spectrum_ax.set_ylabel(r"Count-rate density (s$^{-1}$ keV$^{-1}$)")
    spectrum_ax.set_title(f"{data.epoch_id}: folded synthetic spectrum")
    spectrum_ax.legend(frameon=False, fontsize=8)
    spectrum_ax.grid(alpha=0.18, which="both")

    residual = (data.counts - model_counts) / np.sqrt(np.maximum(model_counts, 1.0))
    residual_ax.axhline(0.0, color="black", lw=0.8)
    residual_ax.axhline(3.0, color=_GREY, lw=0.6, ls="--")
    residual_ax.axhline(-3.0, color=_GREY, lw=0.6, ls="--")
    residual_ax.plot(data.detector_energy, residual, ".", ms=3.0, color=_BLUE)
    excluded = ~data.fit_mask
    if np.any(excluded):
        residual_ax.fill_between(
            data.detector_energy,
            -4.5,
            4.5,
            where=excluded,
            color="#d8dde2",
            alpha=0.55,
            label="Outside 0.5–10 keV fit band",
        )
    residual_ax.set(
        xlabel="Detector energy (keV)",
        ylabel="Poisson residual",
        ylim=(-5.0, 5.0),
    )
    residual_ax.grid(alpha=0.18)
    return _finish(fig, output_path)


def _ga_history(source: Any) -> dict[str, NDArray[Any]]:
    """Normalize live results, flat NPZ files, and resumable checkpoints.

    Early version-1 checkpoints stored histories inside their serialized
    ``payload`` member. New checkpoints expose flat arrays, while the fallback
    loader keeps already-generated campaign products useful.
    """

    candidate = source
    mapping = _mapping_from_source(source)
    if mapping:
        candidate = mapping
        flat_history_names = {
            "population_history",
            "best_score_history",
            "median_score_history",
            "gene_spread_history",
        }
        if (
            "payload" in mapping
            and not flat_history_names.intersection(mapping)
            and isinstance(source, (str, Path))
        ):
            try:
                candidate = load_ga_checkpoint_history(source)
            except Exception:
                # Reporting accepts absent/corrupt optional artifacts and emits
                # the documented placeholder rather than aborting the report.
                candidate = mapping

    def strings(name: str) -> NDArray[np.str_]:
        raw: Any = None
        if isinstance(candidate, Mapping):
            raw = candidate.get(name)
        elif hasattr(candidate, name):
            raw = getattr(candidate, name)
        if raw is None:
            return np.asarray([], dtype=str)
        try:
            return np.asarray(raw, dtype=str).ravel()
        except (TypeError, ValueError):
            return np.asarray([], dtype=str)

    return {
        "best": _get_array(candidate, "best_score_history", "best_scores", "best_score"),
        "median": _get_array(candidate, "median_score_history", "median_scores"),
        "spread": _get_array(candidate, "gene_spread_history", "spread_history", "spread"),
        "population": _get_array(candidate, "population_history", "populations", "population"),
        "best_genes": _get_array(candidate, "best_gene_history", "best_genes_history"),
        "boundary": _get_array(candidate, "boundary_hit_history", "boundary_hits"),
        "immigrants": _get_array(candidate, "immigrant_generations"),
        "evaluations": _get_array(candidate, "evaluations"),
        "parameter_names": strings("parameter_names"),
        "parameter_scales": strings("parameter_scales"),
    }


def plot_ga_convergence(result: Any, output_path: PathLike) -> Path:
    """Plot best/median objective convergence and population diagnostics."""

    history = _ga_history(result)
    fig, axes = _new_figure(2, 1, figsize=(8.5, 6.5), sharex=True)
    score_ax, spread_ax = axes[:, 0]
    best = np.ravel(history["best"])
    median = np.ravel(history["median"])
    if best.size:
        generations = np.arange(best.size)
        score_ax.plot(generations, best, color=_BLUE, lw=1.8, label="Best C-stat")
        if median.size == best.size:
            score_ax.plot(generations, median, color=_ORANGE, lw=1.2, label="Median C-stat")
            finite_scores = np.concatenate(
                (best[np.isfinite(best)], median[np.isfinite(median)])
            )
            if (
                finite_scores.size
                and np.all(finite_scores > 0.0)
                and np.nanmax(finite_scores) / np.nanmin(finite_scores) > 100.0
            ):
                score_ax.set_yscale("log")
        score_ax.set_ylabel("C-statistic")
        score_ax.set_title("Genetic-algorithm convergence")
        evaluations = np.ravel(history["evaluations"])
        diagnostic = f"{best.size} saved generations"
        if evaluations.size:
            diagnostic += f" | {int(evaluations[-1]):,} evaluations"
        score_ax.text(
            0.98,
            0.96,
            diagnostic,
            transform=score_ax.transAxes,
            ha="right",
            va="top",
            color=_GREY,
            fontsize=8,
        )
        score_ax.legend(frameon=False)
        score_ax.grid(alpha=0.2)
    else:
        _empty(score_ax, "No GA convergence history available")

    spread = history["spread"]
    if spread.size:
        if spread.ndim > 1:
            median_spread = np.nanmedian(spread, axis=tuple(range(1, spread.ndim)))
        else:
            median_spread = spread
        spread_ax.plot(np.arange(median_spread.size), median_spread, color=_GREEN, lw=1.6)
        spread_ax.set(xlabel="Generation", ylabel="Median gene spread")
        spread_ax.grid(alpha=0.2)
        boundary = history["boundary"]
        if boundary.ndim > 1:
            boundary = np.nansum(boundary, axis=tuple(range(1, boundary.ndim)))
        boundary = np.ravel(boundary)
        if boundary.size == median_spread.size:
            boundary_ax = spread_ax.twinx()
            boundary_ax.plot(boundary, color=_RED, alpha=0.5, lw=1.0)
            boundary_ax.set_ylabel("Boundary hits", color=_RED)
            finite_boundary = boundary[np.isfinite(boundary)]
            upper = float(np.max(finite_boundary)) if finite_boundary.size else 0.0
            boundary_ax.set_ylim(0.0, max(1.0, upper * 1.1))
        immigrants = np.ravel(history["immigrants"])
        for index, generation in enumerate(immigrants):
            label = "Random immigrants" if index == 0 else None
            score_ax.axvline(
                generation,
                color=_PURPLE,
                ls="--",
                lw=0.9,
                alpha=0.6,
                label=label,
            )
            spread_ax.axvline(
                generation, color=_PURPLE, ls="--", lw=0.9, alpha=0.6
            )
        if immigrants.size:
            score_ax.legend(frameon=False)
    else:
        _empty(spread_ax, "No population-spread history available")
    return _finish(fig, output_path)


def plot_population_evolution(
    result: Any,
    output_path: PathLike,
    *,
    parameter_names: Sequence[str] | None = None,
) -> Path:
    """Plot population median and central 80% interval in normalized genes."""

    history = _ga_history(result)
    population = history["population"]
    fig, axes = _new_figure(1, 1, figsize=(8.5, 5.0))
    ax = axes[0, 0]
    if population.ndim == 2:
        population = population[None, :, :]
    if population.ndim != 3 or population.shape[0] == 0:
        _empty(ax, "No GA population history available")
        ax.set_title("Population evolution")
        return _finish(fig, output_path)

    generations = np.arange(population.shape[0])
    dimensions = population.shape[2]
    saved_names = [str(name) for name in history["parameter_names"]]
    saved_scales = [str(scale) for scale in history["parameter_scales"]]

    def display_name(index: int) -> str:
        if index >= len(saved_names):
            return f"gene {index + 1}"
        name = saved_names[index].casefold()
        scale = (
            saved_scales[index].casefold()
            if index < len(saved_scales)
            else "linear"
        )
        if name == "tin":
            return r"$T_{\rm in}$"
        if name == "gamma":
            return r"$\Gamma$"
        if name == "ndisk":
            return (
                r"$\log_{10}(N_{\rm disk})$"
                if scale == "log10"
                else r"$N_{\rm disk}$"
            )
        if name in {"k", "norm", "powerlaw_norm"}:
            return r"$\log_{10}(K)$" if scale == "log10" else r"$K$"
        if name == "nh":
            return r"$N_{\rm H}$"
        return saved_names[index]

    names = list(parameter_names or [display_name(index) for index in range(dimensions)])
    names.extend(f"gene {index + 1}" for index in range(len(names), dimensions))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(1, dimensions)))
    best_genes = history["best_genes"]
    for index in range(dimensions):
        lower, median, upper = np.nanquantile(
            population[:, :, index], [0.1, 0.5, 0.9], axis=1
        )
        ax.fill_between(generations, lower, upper, color=colors[index], alpha=0.12)
        ax.plot(generations, median, color=colors[index], lw=1.4, label=names[index])
        if best_genes.ndim == 2 and best_genes.shape == (population.shape[0], dimensions):
            ax.plot(generations, best_genes[:, index], color=colors[index], ls=":", lw=0.8)
    ax.set(
        xlabel="Generation",
        ylabel="Normalized gene value",
        ylim=(-0.03, 1.03),
        title="GA population evolution (band: 10th–90th; dotted: best)",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=min(4, dimensions), fontsize=8)
    return _finish(fig, output_path)


def _recovery_long(source: TableLike) -> pd.DataFrame:
    frame = _as_frame(source)
    if frame.empty:
        return frame
    parameter_column = _find_column(frame, ("parameter", "parameter_name"))
    truth_column = _find_column(frame, ("truth", "injected", "true_value"))
    estimate_column = _find_column(frame, ("estimate", "recovered", "fit_value"))
    if parameter_column and truth_column and estimate_column:
        result = frame.copy()
        result["parameter"] = result[parameter_column].astype(str)
        result["truth"] = pd.to_numeric(result[truth_column], errors="coerce")
        result["estimate"] = pd.to_numeric(result[estimate_column], errors="coerce")
        return result

    records: list[pd.DataFrame] = []
    for name in ("tin", "ndisk", "gamma", "k", "norm", "powerlaw_norm", "nh"):
        injected = _find_column(frame, (f"truth_{name}", f"injected_{name}", f"true_{name}"))
        estimate = _find_column(frame, (f"estimate_{name}", f"recovered_{name}", name))
        if injected and estimate:
            part = frame.copy()
            part["parameter"] = name
            part["truth"] = pd.to_numeric(part[injected], errors="coerce")
            part["estimate"] = pd.to_numeric(part[estimate], errors="coerce")
            records.append(part)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def plot_recovery_summary(results: TableLike, output_path: PathLike) -> Path:
    """Plot recovered-versus-injected values plus bias and RMSE curves."""

    frame = _recovery_long(results)
    fig, axes = _new_figure(1, 3, figsize=(13.0, 4.3))
    recovered_ax, bias_ax, rmse_ax = axes[0]
    finite = (
        np.isfinite(_numeric(frame, "truth")) & np.isfinite(_numeric(frame, "estimate"))
        if not frame.empty
        else np.asarray([], dtype=bool)
    )
    frame = frame.loc[finite].copy() if finite.size else pd.DataFrame()
    if frame.empty:
        for ax in axes[0]:
            _empty(ax, "No recovery rows available")
        return _finish(fig, output_path)

    parameters = list(dict.fromkeys(frame["parameter"].astype(str)))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(1, len(parameters))))
    for color, parameter in zip(colors, parameters, strict=True):
        subset = frame.loc[frame["parameter"].astype(str) == parameter]
        recovered_ax.scatter(
            subset["truth"], subset["estimate"], s=18, alpha=0.65, color=color, label=parameter
        )
    limits = np.asarray([frame["truth"].min(), frame["truth"].max(), frame["estimate"].min(), frame["estimate"].max()])
    low, high = float(np.nanmin(limits)), float(np.nanmax(limits))
    padding = 0.05 * (high - low) if high > low else max(abs(low) * 0.05, 0.05)
    recovered_ax.plot([low - padding, high + padding], [low - padding, high + padding], "--", color=_GREY, lw=1)
    recovered_ax.set(
        xlabel="Injected value", ylabel="Recovered value", title="Recovered vs injected"
    )
    recovered_ax.legend(frameon=False, fontsize=8)
    recovered_ax.grid(alpha=0.18)

    exposure_column = _find_column(frame, ("exposure_s", "exposure", "duration_s"))
    frame["exposure"] = _numeric(frame, exposure_column) if exposure_column else 1.0
    frame["error_value"] = frame["estimate"] - frame["truth"]
    grouped = (
        frame.groupby(["parameter", "exposure"], dropna=True)["error_value"]
        .agg(
            bias="mean",
            rmse=lambda values: float(np.sqrt(np.mean(np.square(values)))),
        )
        .reset_index()
    )
    for color, parameter in zip(colors, parameters, strict=True):
        subset = grouped.loc[grouped["parameter"].astype(str) == parameter].sort_values("exposure")
        bias_ax.plot(subset["exposure"], subset["bias"], "o-", color=color, ms=4, label=parameter)
        rmse_ax.plot(subset["exposure"], subset["rmse"], "o-", color=color, ms=4, label=parameter)
    bias_ax.axhline(0.0, color="black", lw=0.8)
    bias_ax.set(xlabel="Exposure (s)", ylabel="Mean estimate − truth", title="Bias")
    rmse_ax.set(xlabel="Exposure (s)", ylabel="RMSE", title="Recovery RMSE")
    if np.all(grouped["exposure"] > 0) and grouped["exposure"].nunique() > 1:
        bias_ax.set_xscale("log")
        rmse_ax.set_xscale("log")
    for ax in (bias_ax, rmse_ax):
        ax.grid(alpha=0.18)
    return _finish(fig, output_path)


def plot_optimizer_comparison(results: TableLike, output_path: PathLike) -> Path:
    """Compare GA/SciPy accuracy and runtime using tidy fit-result rows."""

    frame = _as_frame(results)
    fig, axes = _new_figure(1, 2, figsize=(10.0, 4.3))
    accuracy_ax, runtime_ax = axes[0]
    method_column = _find_column(frame, ("method", "optimizer", "algorithm"))
    statistic_column = _find_column(frame, ("delta_c", "statistic", "cstat", "score", "best_score"))
    runtime_column = _find_column(frame, ("wall_time_s", "runtime_seconds", "runtime_s", "runtime"))
    if frame.empty or method_column is None:
        _empty(accuracy_ax, "No optimizer comparison rows available")
        _empty(runtime_ax, "No runtime rows available")
        return _finish(fig, output_path)

    methods = frame[method_column].fillna("unknown").astype(str)
    method_order = list(dict.fromkeys(methods))
    colors = plt.get_cmap("Set2")(np.linspace(0.0, 0.85, max(1, len(method_order))))
    statistic = _numeric(frame, statistic_column)
    if statistic_column and statistic_column.casefold() != "delta_c":
        group_columns = [
            column
            for column in (
                _find_column(frame, ("scenario",)),
                _find_column(frame, ("epoch_id", "epoch")),
                _find_column(frame, ("realization", "replicate")),
            )
            if column is not None
        ]
        if group_columns:
            numeric_statistic = pd.Series(statistic, index=frame.index)
            statistic = (
                numeric_statistic
                - numeric_statistic.groupby([frame[column] for column in group_columns]).transform("min")
            ).to_numpy(dtype=float)
        elif np.any(np.isfinite(statistic)):
            statistic = statistic - np.nanmin(statistic)
    positions = np.arange(len(method_order), dtype=float)
    for position, color, method in zip(positions, colors, method_order, strict=True):
        selected = (methods == method).to_numpy()
        values = statistic[selected]
        values = values[np.isfinite(values)]
        if values.size:
            jitter = np.linspace(-0.10, 0.10, values.size) if values.size > 1 else np.zeros(1)
            accuracy_ax.scatter(position + jitter, values, color=color, s=24, alpha=0.75)
            accuracy_ax.plot(position, np.median(values), marker="_", color="black", ms=16, mew=2)
    accuracy_ax.axhline(1.0, color=_GREY, ls="--", lw=0.8, label=r"$\Delta C=1$")
    accuracy_ax.set(
        xticks=positions,
        xticklabels=method_order,
        ylabel=r"$\Delta C$ from best",
        title="Optimizer accuracy",
    )
    accuracy_ax.grid(alpha=0.18, axis="y")

    runtime = _numeric(frame, runtime_column)
    for position, color, method in zip(positions, colors, method_order, strict=True):
        selected = (methods == method).to_numpy()
        values = runtime[selected]
        values = values[np.isfinite(values) & (values >= 0.0)]
        if values.size:
            jitter = np.linspace(-0.10, 0.10, values.size) if values.size > 1 else np.zeros(1)
            runtime_ax.scatter(position + jitter, values, color=color, s=24, alpha=0.75)
            runtime_ax.plot(position, np.median(values), marker="_", color="black", ms=16, mew=2)
    if np.any(np.isfinite(runtime) & (runtime > 0.0)):
        runtime_ax.set_yscale("log")
    runtime_ax.set(
        xticks=positions,
        xticklabels=method_order,
        ylabel="Runtime (s)",
        title="Optimizer runtime",
    )
    runtime_ax.grid(alpha=0.18, axis="y", which="both")
    return _finish(fig, output_path)


def _posterior_payload(
    source: PosteriorResult | Mapping[str, Any] | pd.DataFrame | PathLike | None,
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64] | None, pd.DataFrame]:
    if isinstance(source, PosteriorResult):
        return source.parameter_names, source.samples, source.weights, pd.DataFrame()
    if isinstance(source, (str, Path)) and Path(source).suffix.casefold() == ".csv":
        return (), np.empty((0, 0)), None, _as_frame(source)
    payload = _mapping_from_source(source)
    if payload and "samples" in payload:
        samples = np.asarray(payload["samples"], dtype=np.float64)
        raw_names = payload.get("parameter_names", [f"p{index + 1}" for index in range(samples.shape[1] if samples.ndim == 2 else 0)])
        names = tuple(str(item) for item in np.asarray(raw_names).tolist())
        raw_weights = payload.get("weights")
        weights = None if raw_weights is None or np.asarray(raw_weights).size == 0 else np.asarray(raw_weights, dtype=np.float64)
        return names, samples, weights, pd.DataFrame()
    frame = _as_frame(source)
    return (), np.empty((0, 0)), None, frame


def _plot_posterior_summary(frame: pd.DataFrame, output_path: PathLike) -> Path:
    fig, axes = _new_figure(1, 1, figsize=(7.0, 4.5))
    ax = axes[0, 0]
    parameter_column = _find_column(frame, ("parameter", "parameter_name"))
    median_column = _find_column(frame, ("median", "q50", "estimate"))
    if frame.empty or parameter_column is None or median_column is None:
        _empty(ax, "No posterior samples or summary rows available")
        ax.set_title("Posterior summary")
        return _finish(fig, output_path)
    median = _numeric(frame, median_column)
    q16_column = _find_column(frame, ("q16", "lower"))
    q84_column = _find_column(frame, ("q84", "upper"))
    minus_column = _find_column(frame, ("minus", "error_minus"))
    plus_column = _find_column(frame, ("plus", "error_plus"))
    lower = median - _numeric(frame, q16_column) if q16_column else _numeric(frame, minus_column)
    upper = _numeric(frame, q84_column) - median if q84_column else _numeric(frame, plus_column)
    lower = np.where(np.isfinite(lower), np.maximum(lower, 0.0), 0.0)
    upper = np.where(np.isfinite(upper), np.maximum(upper, 0.0), 0.0)
    y = np.arange(len(frame))
    ax.errorbar(median, y, xerr=np.vstack((lower, upper)), fmt="o", color=_BLUE, capsize=3)
    ax.set(yticks=y, yticklabels=frame[parameter_column].astype(str), xlabel="Posterior value", title="Posterior credible intervals")
    ax.grid(alpha=0.18, axis="x")
    return _finish(fig, output_path)


def plot_posterior_matrix(
    posterior: PosteriorResult | Mapping[str, Any] | pd.DataFrame | PathLike | None,
    output_path: PathLike,
    *,
    truth: Mapping[str, float] | None = None,
    max_draws: int = 5_000,
) -> Path:
    """Draw a dependency-free posterior corner matrix or summary fallback."""

    names, samples, weights, summary = _posterior_payload(posterior)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        return _plot_posterior_summary(summary, output_path)
    finite = np.all(np.isfinite(samples), axis=1)
    samples = samples[finite]
    if weights is not None and weights.shape == finite.shape:
        weights = weights[finite]
    elif weights is not None and weights.shape != (samples.shape[0],):
        weights = None
    if samples.shape[0] == 0:
        return _plot_posterior_summary(pd.DataFrame(), output_path)
    if samples.shape[0] > max_draws:
        selected = np.linspace(0, samples.shape[0] - 1, max_draws, dtype=int)
        samples = samples[selected]
        weights = weights[selected] if weights is not None else None

    dimensions = min(samples.shape[1], 6)
    samples = samples[:, :dimensions]
    names = tuple(names[:dimensions]) if names else tuple(f"p{index + 1}" for index in range(dimensions))
    fig, axes = _new_figure(dimensions, dimensions, figsize=(2.25 * dimensions, 2.25 * dimensions))
    for row in range(dimensions):
        for column in range(dimensions):
            ax = axes[row, column]
            if row == column:
                ax.hist(samples[:, column], bins=35, weights=weights, color=_BLUE, alpha=0.75, density=True)
                quantiles = np.quantile(samples[:, column], [0.16, 0.5, 0.84])
                for value, style in zip(quantiles, (":", "-", ":"), strict=True):
                    ax.axvline(value, color=_ORANGE, lw=1.0, ls=style)
                if truth and names[column] in truth:
                    ax.axvline(float(truth[names[column]]), color=_RED, lw=1.2, ls="--")
            elif row > column:
                ax.hist2d(samples[:, column], samples[:, row], bins=35, weights=weights, cmap="Blues")
                if truth and names[column] in truth and names[row] in truth:
                    ax.plot(float(truth[names[column]]), float(truth[names[row]]), "+", color=_RED, ms=8, mew=1.5)
            else:
                correlation = np.corrcoef(samples[:, column], samples[:, row])[0, 1]
                ax.text(0.5, 0.5, f"r = {correlation:.2f}", transform=ax.transAxes, ha="center", va="center", color=_GREY)
                ax.set_xticks([])
                ax.set_yticks([])
            if row == dimensions - 1:
                ax.set_xlabel(_PARAMETER_LABELS.get(names[column].casefold(), names[column]), fontsize=8)
            else:
                ax.set_xticklabels([])
            if column == 0 and row > 0:
                ax.set_ylabel(_PARAMETER_LABELS.get(names[row].casefold(), names[row]), fontsize=8)
            elif column > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle("Posterior matrix", y=0.972, fontsize=12)
    return _finish(fig, output_path)


def _spectra_hid(source: Any) -> pd.DataFrame:
    if isinstance(source, SyntheticSpectrum):
        items = [source]
    elif isinstance(source, Sequence) and source and all(isinstance(item, SyntheticSpectrum) for item in source):
        items = list(source)
    else:
        return pd.DataFrame()
    records = []
    for index, spectrum in enumerate(items):
        soft = (spectrum.detector_energy >= 0.5) & (spectrum.detector_energy < 2.0)
        hard = (spectrum.detector_energy >= 2.0) & (spectrum.detector_energy <= 10.0)
        soft_rate = float(np.sum(spectrum.counts[soft]) / spectrum.exposure_s)
        hard_rate = float(np.sum(spectrum.counts[hard]) / spectrum.exposure_s)
        records.append(
            {
                "epoch_id": spectrum.epoch_id,
                "order": index,
                "hardness": hard_rate / soft_rate if soft_rate > 0.0 else np.nan,
                "intensity": hard_rate + soft_rate,
            }
        )
    return pd.DataFrame.from_records(records)


def _hid_frame(source: TableLike | Sequence[SyntheticSpectrum]) -> pd.DataFrame:
    spectral = _spectra_hid(source)
    if not spectral.empty:
        return spectral
    frame = _as_frame(source)
    if frame.empty:
        return frame
    hardness_column = _find_column(frame, ("hardness_ratio", "hardness", "hard_soft_ratio"))
    intensity_column = _find_column(frame, ("intensity_cps", "intensity", "total_rate_hz", "count_rate"))
    if hardness_column is None:
        hard_column = _find_column(frame, ("hard_rate_hz", "hard_rate", "hard_cps"))
        soft_column = _find_column(frame, ("soft_rate_hz", "soft_rate", "soft_cps"))
        hard = _numeric(frame, hard_column)
        soft = _numeric(frame, soft_column)
        frame["hardness"] = np.divide(hard, soft, out=np.full_like(hard, np.nan), where=soft > 0)
    else:
        frame["hardness"] = _numeric(frame, hardness_column)
    if intensity_column is None:
        hard_column = _find_column(frame, ("hard_rate_hz", "hard_rate", "hard_cps"))
        soft_column = _find_column(frame, ("soft_rate_hz", "soft_rate", "soft_cps"))
        frame["intensity"] = _numeric(frame, hard_column) + _numeric(frame, soft_column)
    else:
        frame["intensity"] = _numeric(frame, intensity_column)
    return frame


def plot_hardness_intensity(source: TableLike | Sequence[SyntheticSpectrum], output_path: PathLike) -> Path:
    """Plot the synthetic hard/soft hardness–intensity track."""

    frame = _hid_frame(source)
    fig, axes = _new_figure(1, 1, figsize=(6.5, 5.2))
    ax = axes[0, 0]
    if frame.empty:
        _empty(ax, "No hardness/intensity rows available")
        ax.set_title("Synthetic hardness–intensity diagram")
        return _finish(fig, output_path)
    hardness = _numeric(frame, "hardness")
    intensity = _numeric(frame, "intensity")
    finite = np.isfinite(hardness) & np.isfinite(intensity) & (intensity >= 0.0)
    hardness, intensity = hardness[finite], intensity[finite]
    selected = frame.loc[finite].reset_index(drop=True)
    if hardness.size == 0:
        _empty(ax, "No finite hardness/intensity rows available")
        return _finish(fig, output_path)
    colors = np.arange(hardness.size)
    ax.plot(hardness, intensity, color="#aeb6bd", lw=1.0, zorder=1)
    scatter = ax.scatter(hardness, intensity, c=colors, cmap="viridis", s=45, edgecolor="white", linewidth=0.5, zorder=2)
    for index in range(hardness.size - 1):
        ax.annotate("", xy=(hardness[index + 1], intensity[index + 1]), xytext=(hardness[index], intensity[index]), arrowprops={"arrowstyle": "->", "color": "#727b84", "lw": 0.7})
    epoch_column = _find_column(selected, ("epoch_id", "epoch"))
    if epoch_column:
        for x_value, y_value, epoch in zip(hardness, intensity, selected[epoch_column], strict=True):
            ax.annotate(str(epoch), (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=7)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Outburst sequence")
    ax.set(xlabel="Hard / soft count ratio", ylabel=r"0.5–10 keV-inspired intensity (count s$^{-1}$)", title="Synthetic hardness–intensity diagram")
    ax.grid(alpha=0.18)
    return _finish(fig, output_path)


def _wide_spectral_frame(source: TableLike) -> pd.DataFrame:
    frame = _as_frame(source)
    parameter_column = _find_column(frame, ("parameter", "parameter_name"))
    estimate_column = _find_column(frame, ("estimate", "median", "value"))
    epoch_column = _find_column(frame, ("epoch_id", "epoch"))
    if parameter_column and estimate_column and epoch_column:
        metadata = [column for column in (epoch_column, _find_column(frame, ("reference_mjd", "mjd")), _find_column(frame, ("phase",))) if column]
        try:
            wide = frame.pivot_table(index=metadata, columns=parameter_column, values=estimate_column, aggfunc="first").reset_index()
            wide.columns = [str(column) for column in wide.columns]
            return wide
        except (KeyError, TypeError, ValueError):
            return frame
    return frame


def _sequence_x(frame: pd.DataFrame) -> tuple[NDArray[np.float64], str]:
    mjd_column = _find_column(frame, ("reference_mjd", "mjd"))
    if mjd_column:
        mjd = _numeric(frame, mjd_column)
        if np.any(np.isfinite(mjd)):
            return mjd, "Reference MJD"
    order_column = _find_column(frame, ("order", "epoch_index"))
    if order_column:
        order = _numeric(frame, order_column)
        if np.any(np.isfinite(order)):
            return order, "Synthetic epoch sequence"
    return np.arange(1, len(frame) + 1, dtype=float), "Synthetic epoch sequence"


def plot_parameter_qpo_evolution(
    spectral_rows: TableLike,
    timing_rows: TableLike | PathLike,
    output_path: PathLike | None = None,
) -> Path:
    """Plot spectral-parameter and type-C-like QPO evolution together."""

    if output_path is None:
        # Convenient two-argument form for a combined artifact table.
        output_path = timing_rows  # type: ignore[assignment]
        timing_rows = spectral_rows
    spectral = _wide_spectral_frame(spectral_rows)
    timing = _as_frame(timing_rows)
    fig, axes = _new_figure(3, 2, figsize=(10.5, 9.0))
    panels = list(axes.ravel())
    parameter_specs = [
        (("tin", "Tin"), r"$T_{\rm in}$ (keV)"),
        (("gamma", "Gamma"), r"Photon index $\Gamma$"),
        (("ndisk", "Ndisk", "disk_norm"), r"Disk-like normalization"),
        (("k", "K", "norm", "powerlaw_norm"), r"Continuum normalization $K$"),
    ]
    x, x_label = _sequence_x(spectral)
    epoch_column = _find_column(spectral, ("epoch_id", "epoch"))
    for panel, (aliases, ylabel) in zip(panels, parameter_specs, strict=False):
        column = _find_column(spectral, aliases)
        values = _numeric(spectral, column)
        finite = np.isfinite(x) & np.isfinite(values)
        if np.any(finite):
            panel.plot(x[finite], values[finite], "o-", color=_BLUE, ms=4)
            if epoch_column:
                for x_value, y_value, epoch in zip(x[finite], values[finite], spectral.loc[finite, epoch_column], strict=True):
                    panel.annotate(str(epoch), (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=6)
            panel.set_ylabel(ylabel)
            panel.grid(alpha=0.18)
        else:
            _empty(panel, f"No {aliases[0]} evolution rows")

    qpo_ax = panels[4]
    qpo_column = _find_column(timing, ("qpo_centroid_hz", "centroid_hz", "qpo_frequency_hz", "qpo_hz"))
    injected_column = _find_column(timing, ("qpo_injected_hz", "injected_qpo_hz"))
    timing_epoch = _find_column(timing, ("epoch_id", "epoch"))
    if not timing.empty:
        if timing_epoch and epoch_column:
            x_by_epoch = {str(epoch): value for epoch, value in zip(spectral[epoch_column], x, strict=True)}
            qpo_x = np.asarray([x_by_epoch.get(str(epoch), np.nan) for epoch in timing[timing_epoch]], dtype=float)
        else:
            qpo_x, _ = _sequence_x(timing)
        plotted = False
        for column, label, color, marker in (
            (injected_column, "Injected QPO", _GREY, "x"),
            (qpo_column, "Recovered type-C-like", _RED, "o"),
        ):
            values = _numeric(timing, column)
            finite = np.isfinite(qpo_x) & np.isfinite(values) & (values > 0.0)
            if label.startswith("Recovered"):
                finite &= _accepted_qpo_mask(timing)
            if np.any(finite):
                qpo_ax.plot(qpo_x[finite], values[finite], marker=marker, ls="-", color=color, ms=5, label=label)
                plotted = True
        if plotted:
            qpo_ax.set_yscale("log")
            qpo_ax.set_ylabel("QPO frequency (Hz)")
            qpo_ax.legend(frameon=False, fontsize=8)
            qpo_ax.grid(alpha=0.18, which="both")
        else:
            _empty(qpo_ax, "No accepted type-C-like QPO rows")
    else:
        _empty(qpo_ax, "No timing rows available")
    panels[5].axis("off")
    panels[5].text(0.5, 0.58, "Educational outburst evolution", transform=panels[5].transAxes, ha="center", va="center", fontsize=11, color=_GREY)
    panels[5].text(0.5, 0.42, "Reference dates are contextual anchors only", transform=panels[5].transAxes, ha="center", va="center", fontsize=8, color=_GREY)
    for panel in panels[:5]:
        if panel.axison:
            panel.set_xlabel(x_label)
    fig.suptitle("Spectral parameter and QPO evolution", y=0.974)
    return _finish(fig, output_path)


def plot_qpo_gamma_correlation(
    spectral_rows: TableLike,
    timing_rows: TableLike | PathLike | None,
    output_path: PathLike | None = None,
) -> Path:
    """Plot recovered type-C-like QPO frequency against photon index."""

    if output_path is None:
        # Convenient two-argument form for a combined artifact table.
        output_path = timing_rows  # type: ignore[assignment]
        timing_rows = None
    if output_path is None:
        raise ValueError("output_path is required")
    spectral = _wide_spectral_frame(spectral_rows)
    timing = spectral.copy() if timing_rows is None else _as_frame(timing_rows)
    if timing_rows is not None and not timing.empty:
        timing = timing.loc[_accepted_qpo_mask(timing)].reset_index(drop=True)
    gamma_column = _find_column(spectral, ("gamma", "Gamma"))
    qpo_column = _find_column(timing, ("qpo_centroid_hz", "centroid_hz", "qpo_frequency_hz", "qpo_hz", "injected_qpo_hz", "qpo_injected_hz"))
    spectral_epoch = _find_column(spectral, ("epoch_id", "epoch"))
    timing_epoch = _find_column(timing, ("epoch_id", "epoch"))
    if timing_rows is not None and spectral_epoch and timing_epoch and gamma_column:
        left = spectral[[spectral_epoch, gamma_column]].copy()
        left.columns = ["_epoch", "_gamma"]
        right_columns = [timing_epoch] + ([qpo_column] if qpo_column else [])
        right = timing[right_columns].copy()
        right.columns = ["_epoch"] + (["_qpo"] if qpo_column else [])
        merged = left.merge(right, on="_epoch", how="inner")
        gamma = _numeric(merged, "_gamma")
        qpo = _numeric(merged, "_qpo" if "_qpo" in merged else None)
    else:
        gamma = _numeric(spectral, gamma_column)
        qpo = _numeric(timing, qpo_column)
    if gamma.size != qpo.size:
        common_size = min(gamma.size, qpo.size)
        gamma, qpo = gamma[:common_size], qpo[:common_size]
    finite = np.isfinite(gamma) & np.isfinite(qpo) & (qpo > 0.0)
    gamma, qpo = gamma[finite], qpo[finite]

    fig, axes = _new_figure(1, 1, figsize=(6.4, 5.0))
    ax = axes[0, 0]
    if gamma.size == 0:
        _empty(ax, "No matched Gamma/type-C-like QPO rows available")
        ax.set_title("QPO frequency versus photon index")
        return _finish(fig, output_path)
    ax.scatter(gamma, qpo, s=45, color=_PURPLE, edgecolor="white", linewidth=0.6, label="Synthetic epochs")
    if gamma.size >= 2 and np.ptp(gamma) > 0.0:
        coefficients = np.polyfit(gamma, qpo, 1)
        line_x = np.linspace(float(np.min(gamma)), float(np.max(gamma)), 100)
        ax.plot(line_x, np.polyval(coefficients, line_x), color=_ORANGE, lw=1.4, label="Linear guide")
        correlation = float(np.corrcoef(gamma, qpo)[0, 1])
        ax.text(0.04, 0.95, f"Pearson r = {correlation:.2f}", transform=ax.transAxes, ha="left", va="top")
    if np.all(qpo > 0.0) and np.nanmax(qpo) / np.nanmin(qpo) > 10.0:
        ax.set_yscale("log")
    ax.set(xlabel=r"Photon index $\Gamma$", ylabel="Type-C-like QPO frequency (Hz)", title=r"QPO frequency versus $\Gamma$")
    ax.grid(alpha=0.18, which="both")
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, output_path)


def generate_selected_figures(
    output_directory: PathLike,
    *,
    response: InstrumentResponse | Mapping[str, Any] | PathLike | None = None,
    spectrum: SyntheticSpectrum | Mapping[str, Any] | PathLike | None = None,
    objective: Any | None = None,
    parameters: Mapping[str, float] | None = None,
    ga_result: Any | None = None,
    recovery_rows: TableLike = None,
    fit_rows: TableLike = None,
    posterior: PosteriorResult | Mapping[str, Any] | pd.DataFrame | PathLike | None = None,
    timing_rows: TableLike = None,
    spectral_rows: TableLike = None,
    hid_rows: TableLike = None,
) -> dict[str, Path]:
    """Generate the selected portfolio figures, including empty-stage cards."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    products = {
        "response_area": plot_response_area(response, directory / "response_area.png"),
        "folded_spectrum": plot_folded_spectrum(
            spectrum,
            directory / "folded_spectrum.png",
            objective=objective,
            parameters=parameters,
        ),
        "ga_convergence": plot_ga_convergence(ga_result, directory / "ga_convergence.png"),
        "ga_population": plot_population_evolution(ga_result, directory / "ga_population.png"),
        "recovery": plot_recovery_summary(recovery_rows, directory / "recovery.png"),
        "optimizer_comparison": plot_optimizer_comparison(fit_rows, directory / "optimizer_comparison.png"),
        "posterior": plot_posterior_matrix(posterior, directory / "posterior.png"),
        "hardness_intensity": plot_hardness_intensity(
            timing_rows if hid_rows is None else hid_rows,
            directory / "hardness_intensity.png",
        ),
        "parameter_qpo_evolution": plot_parameter_qpo_evolution(
            spectral_rows, timing_rows, directory / "parameter_qpo_evolution.png"
        ),
        "qpo_gamma": plot_qpo_gamma_correlation(
            spectral_rows, timing_rows, directory / "qpo_gamma.png"
        ),
    }
    return products


def generate_artifact_figures(results_directory: PathLike, output_directory: PathLike) -> dict[str, Path]:
    """Discover conventional case-study artifacts and render selected plots."""

    results = Path(results_directory)

    def first_existing(*relative_paths: str) -> Path | None:
        for relative in relative_paths:
            candidate = results / relative
            if candidate.exists():
                return candidate
        return None

    response = first_existing("response.npz", "instrument_response.npz")
    spectrum = first_existing("spectrum.npz", "synthetic_spectrum.npz")
    if spectrum is None:
        candidates = sorted(results.glob("**/*spectrum*.npz")) if results.exists() else []
        spectrum = candidates[0] if candidates else None
    ga_result = first_existing("ga_result.npz", "ga_history.npz")
    posterior = first_existing("posterior.npz", "posterior_samples.npz", "posterior_summary.csv")
    return generate_selected_figures(
        output_directory,
        response=response,
        spectrum=spectrum,
        ga_result=ga_result,
        recovery_rows=first_existing("recovery_results.csv"),
        fit_rows=first_existing("fit_results.csv"),
        posterior=posterior,
        timing_rows=first_existing("timing_results.csv"),
        spectral_rows=first_existing("truth.csv", "fit_results.csv"),
    )


# Concise aliases for callers and case-study prose.
plot_response = plot_response_area
plot_spectrum_fit = plot_folded_spectrum
plot_recovery = plot_recovery_summary
plot_ga_vs_scipy = plot_optimizer_comparison
plot_corner = plot_posterior_matrix
plot_hid = plot_hardness_intensity
plot_evolution = plot_parameter_qpo_evolution
plot_qpo_gamma = plot_qpo_gamma_correlation


__all__ = [
    "generate_artifact_figures",
    "generate_selected_figures",
    "plot_corner",
    "plot_evolution",
    "plot_folded_spectrum",
    "plot_ga_convergence",
    "plot_ga_vs_scipy",
    "plot_hardness_intensity",
    "plot_hid",
    "plot_optimizer_comparison",
    "plot_parameter_qpo_evolution",
    "plot_population_evolution",
    "plot_posterior_matrix",
    "plot_qpo_gamma",
    "plot_qpo_gamma_correlation",
    "plot_recovery",
    "plot_recovery_summary",
    "plot_response",
    "plot_response_area",
    "plot_spectrum_fit",
]
