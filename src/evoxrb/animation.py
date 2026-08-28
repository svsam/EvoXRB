"""Replay and live views of genetic-algorithm spectral evolution.

The saved replay is the primary interface.  It reconstructs the response-folded
model associated with the best member of every recorded GA generation and
compares it with the synthetic spectrum used by :class:`~evoxrb.objective.Objective`.
A user-supplied :class:`~evoxrb.reference.ReferenceSpectrum` may be overlaid for
visual context, but is deliberately excluded from the C-statistic and residuals.

This module intentionally does not import :mod:`evoxrb.plotting`: that module
selects a headless backend and stamps every figure as wholly synthetic, neither
of which is appropriate for an optional user-supplied reference overlay.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
from typing import Any, Literal, TypeAlias

import matplotlib
from matplotlib import animation as mpl_animation
from matplotlib.animation import FFMpegWriter, FuncAnimation, HTMLWriter, PillowWriter
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .genetic import GARunResult, GenerationSnapshot
from .objective import Objective
from .parameters import SearchSpace
from .reference import ReferenceSpectrum


PathLike: TypeAlias = str | Path
PopulationDisplay: TypeAlias = Literal["none", "curves", "envelope"]

_BLUE = "#2864a8"
_ORANGE = "#dd7831"
_GREEN = "#3c8d5a"
_PURPLE = "#7851a9"
_RED = "#b5413e"
_GREY = "#65707b"


@dataclass(frozen=True, slots=True)
class _TargetPlotData:
    energy: NDArray[np.float64]
    counts: NDArray[np.float64]
    scale: NDArray[np.float64]
    rate: NDArray[np.float64]
    error: NDArray[np.float64]
    fit_mask: NDArray[np.bool_]
    label: str


@dataclass(frozen=True, slots=True)
class _ReplayData:
    parameters: tuple[Mapping[str, float], ...]
    model_counts: NDArray[np.float64]
    model_rates: NDArray[np.float64]
    residuals: NDArray[np.float64]
    best_scores: NDArray[np.float64]
    median_scores: NDArray[np.float64]

    @property
    def frame_count(self) -> int:
        return len(self.parameters)


def _target_plot_data(objective: Objective) -> _TargetPlotData:
    """Validate and normalize the synthetic target used by an Objective."""

    spectrum = objective.spectrum
    energy = np.asarray(spectrum.detector_energy, dtype=np.float64)
    edges = np.asarray(spectrum.detector_edges, dtype=np.float64)
    counts = np.asarray(spectrum.counts, dtype=np.float64)
    exposure = float(spectrum.exposure_s)
    fit_mask = np.asarray(spectrum.fit_mask, dtype=np.bool_)

    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("objective spectrum must have a non-empty detector grid")
    if edges.shape != (energy.size + 1,):
        raise ValueError("objective spectrum detector edges are incompatible")
    if counts.shape != energy.shape or fit_mask.shape != energy.shape:
        raise ValueError("objective spectrum arrays must match its detector grid")
    if np.any(~np.isfinite(energy)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("objective spectrum detector grid must be finite and ordered")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("objective spectrum counts must be finite and non-negative")
    if not np.isfinite(exposure) or exposure <= 0.0:
        raise ValueError("objective spectrum exposure must be finite and positive")

    scale = exposure * np.diff(edges)
    rate = counts / scale
    error = np.sqrt(np.maximum(counts, 1.0)) / scale
    label = str(getattr(spectrum, "label", "Synthetic fit target"))
    return _TargetPlotData(
        energy=energy.copy(),
        counts=counts.copy(),
        scale=scale,
        rate=rate,
        error=error,
        fit_mask=fit_mask.copy(),
        label=label,
    )


def _reference_plot_data(
    reference: ReferenceSpectrum | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None, str]:
    if reference is None:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            None,
            "",
        )
    payload = reference.as_plot_data()
    x = np.asarray(payload["x"], dtype=np.float64)
    y = np.asarray(payload["y"], dtype=np.float64)
    raw_error = payload.get("yerr")
    error = None if raw_error is None else np.asarray(raw_error, dtype=np.float64)
    return x, y, error, str(payload["label"])


def _expected_counts(
    objective: Objective,
    parameters: Mapping[str, float],
    channels: int,
    *,
    context: str,
) -> NDArray[np.float64]:
    try:
        counts = np.asarray(objective.expected_counts(parameters), dtype=np.float64)
    except Exception as error:
        raise ValueError(f"could not evaluate folded model for {context}") from error
    if counts.shape != (channels,):
        raise ValueError(f"folded model for {context} has an incompatible shape")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError(f"folded model for {context} is not finite and non-negative")
    return counts


def _prepare_replay(
    result: GARunResult,
    objective: Objective,
    search_space: SearchSpace,
    target: _TargetPlotData,
) -> _ReplayData:
    best_genes = np.asarray(result.best_gene_history, dtype=np.float64)
    if best_genes.ndim != 2 or best_genes.shape[0] == 0:
        raise ValueError("GA result has no sampled best-gene history")
    if best_genes.shape[1] != search_space.ndim:
        raise ValueError("GA best-gene history does not match the search space")

    frame_count = best_genes.shape[0]
    best_scores = np.asarray(result.best_score_history, dtype=np.float64).reshape(-1)
    median_scores = np.asarray(result.median_score_history, dtype=np.float64).reshape(-1)
    if best_scores.shape != (frame_count,) or median_scores.shape != (frame_count,):
        raise ValueError("GA score histories must have one value per sampled generation")

    parameters: list[Mapping[str, float]] = []
    model_counts = np.empty((frame_count, target.energy.size), dtype=np.float64)
    for frame, genes in enumerate(best_genes):
        decoded = search_space.decode(genes)
        parameters.append(decoded)
        model_counts[frame] = _expected_counts(
            objective,
            decoded,
            target.energy.size,
            context=f"generation {frame}",
        )
    model_rates = model_counts / target.scale[None, :]
    residuals = (target.counts[None, :] - model_counts) / np.sqrt(
        np.maximum(model_counts, 1.0)
    )
    return _ReplayData(
        parameters=tuple(parameters),
        model_counts=model_counts,
        model_rates=model_rates,
        residuals=residuals,
        best_scores=best_scores.copy(),
        median_scores=median_scores.copy(),
    )


def _positive_limits(*arrays: NDArray[np.float64]) -> tuple[float, float]:
    finite_positive = [
        np.asarray(array, dtype=np.float64).reshape(-1)
        for array in arrays
        if np.asarray(array).size
    ]
    if not finite_positive:
        return 1.0e-4, 1.0
    values = np.concatenate(finite_positive)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0e-4, 1.0
    lower = max(float(np.min(values)) * 0.45, np.finfo(float).tiny)
    upper = float(np.max(values)) * 2.1
    if upper <= lower:
        upper = lower * 10.0
    return lower, upper


def _linear_limits(
    values: NDArray[np.float64], *, minimum_span: float = 1.0
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, minimum_span
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = max(upper - lower, minimum_span)
    return lower - 0.08 * span, upper + 0.08 * span


def _residual_limit(residuals: NDArray[np.float64]) -> float:
    finite = np.abs(np.asarray(residuals, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 5.0
    return float(max(np.quantile(finite, 0.985) * 1.1, 5.0))


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "inf" if np.isposinf(value) else "-inf" if np.isneginf(value) else "nan"
    return f"{value:.6g}"


def _parameter_text(
    parameters: Mapping[str, float],
    *,
    frame: int,
    frame_count: int,
    best_score: float,
    median_score: float,
    population_size: int,
    reference: ReferenceSpectrum | None,
) -> str:
    lines = [
        f"Generation {frame} / {frame_count - 1}",
        f"Evaluations through frame: {(frame + 1) * population_size:,}",
        f"Best C-stat: {_format_number(best_score)}",
        f"Median C-stat: {_format_number(median_score)}",
        "",
        "Decoded best parameters",
    ]
    lines.extend(f"{name}: {_format_number(float(value))}" for name, value in parameters.items())
    if reference is not None:
        lines.extend(("", f"Reference: {reference.label}"))
        if reference.source is not None:
            lines.append(f"Source: {reference.source}")
    return "\n".join(lines)


def _plot_target_and_reference(
    spectrum_ax: Axes,
    target: _TargetPlotData,
    reference: ReferenceSpectrum | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    spectrum_ax.errorbar(
        target.energy,
        target.rate,
        yerr=target.error,
        fmt=".",
        ms=3.0,
        lw=0.55,
        color="black",
        alpha=0.72,
        label=target.label,
        zorder=4,
    )
    reference_x, reference_y, reference_error, reference_label = _reference_plot_data(reference)
    if reference is not None:
        spectrum_ax.errorbar(
            reference_x,
            reference_y,
            yerr=reference_error,
            fmt="o",
            mfc="none",
            mec=_PURPLE,
            ecolor=_PURPLE,
            ms=3.5,
            lw=0.7,
            alpha=0.72,
            label=f"{reference_label} (visual overlay only)",
            zorder=3,
        )
    return reference_x, reference_y, reference_error


def _population_indices(
    scores: NDArray[np.float64],
    limit: int,
    *,
    stratified: bool,
) -> NDArray[np.int64]:
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")
    count = min(int(limit), order.size)
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    if not stratified or count == order.size:
        return np.asarray(order[:count], dtype=np.int64)
    locations = np.linspace(0, order.size - 1, count).round().astype(np.int64)
    return np.asarray(order[locations], dtype=np.int64)


def _population_model_rates(
    frame: int,
    result: GARunResult,
    objective: Objective,
    search_space: SearchSpace,
    target: _TargetPlotData,
    *,
    limit: int,
    stratified: bool,
) -> NDArray[np.float64]:
    populations = np.asarray(result.population_history, dtype=np.float64)
    scores = np.asarray(result.score_history, dtype=np.float64)
    if populations.ndim != 3 or populations.shape[0] <= frame:
        raise ValueError("GA population history is unavailable for population replay")
    if populations.shape[2] != search_space.ndim:
        raise ValueError("GA population history does not match the search space")
    if scores.shape != populations.shape[:2]:
        raise ValueError("GA score and population histories are incompatible")

    selected = _population_indices(scores[frame], limit, stratified=stratified)
    curves: list[NDArray[np.float64]] = []
    for index in selected:
        try:
            parameters = search_space.decode(populations[frame, index])
            counts = _expected_counts(
                objective,
                parameters,
                target.energy.size,
                context=f"generation {frame}, population member {int(index)}",
            )
        except ValueError:
            continue
        curves.append(counts / target.scale)
    if not curves:
        return np.empty((0, target.energy.size), dtype=np.float64)
    return np.asarray(curves, dtype=np.float64)


def _set_envelope_vertices(
    artist: PolyCollection,
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> None:
    vertices = np.concatenate(
        (
            np.column_stack((x, lower)),
            np.column_stack((x[::-1], upper[::-1])),
        ),
        axis=0,
    )
    artist.set_verts([vertices])


def create_spectral_animation(
    result: GARunResult,
    objective: Objective,
    search_space: SearchSpace,
    *,
    reference: ReferenceSpectrum | None = None,
    population_display: PopulationDisplay = "none",
    max_population_curves: int = 20,
    interval_ms: int = 120,
    repeat: bool = True,
) -> FuncAnimation:
    """Create a replay containing every generation sampled by ``result``.

    ``population_display="curves"`` shows the best finite population members.
    ``"envelope"`` draws a 10th--90th percentile band from a deterministic,
    score-stratified sample.  The cap keeps a full 300-generation replay
    responsive without implying that the reference spectrum was fitted.
    """

    if population_display not in ("none", "curves", "envelope"):
        raise ValueError("population_display must be 'none', 'curves', or 'envelope'")
    if int(max_population_curves) < 1:
        raise ValueError("max_population_curves must be positive")
    if int(interval_ms) < 1:
        raise ValueError("interval_ms must be positive")

    target = _target_plot_data(objective)
    replay = _prepare_replay(result, objective, search_space, target)
    frame_numbers = np.arange(replay.frame_count, dtype=np.int64)
    populations = np.asarray(result.population_history)
    population_size = int(populations.shape[1]) if populations.ndim == 3 else 0

    # Bind saved replay to an off-screen canvas without changing the user's
    # global Matplotlib backend.  LiveSpectrumViewer remains the explicit GUI
    # path below.
    figure = Figure(figsize=(12.0, 7.5), facecolor="white")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.55, 1.0),
        height_ratios=(2.0, 1.0),
        hspace=0.08,
        wspace=0.28,
    )
    spectrum_ax = figure.add_subplot(grid[0, 0])
    residual_ax = figure.add_subplot(grid[1, 0], sharex=spectrum_ax)
    score_ax = figure.add_subplot(grid[0, 1])
    parameter_ax = figure.add_subplot(grid[1, 1])
    parameter_ax.axis("off")

    reference_x, reference_y, reference_error = _plot_target_and_reference(
        spectrum_ax, target, reference
    )
    population_cache: dict[int, NDArray[np.float64]] = {}

    def population_rates(frame: int) -> NDArray[np.float64]:
        if population_display == "none":
            return np.empty((0, target.energy.size), dtype=np.float64)
        if frame not in population_cache:
            population_cache[frame] = _population_model_rates(
                frame,
                result,
                objective,
                search_space,
                target,
                limit=int(max_population_curves),
                stratified=population_display == "envelope",
            )
        return population_cache[frame]

    population_collection: LineCollection | None = None
    envelope_artist: PolyCollection | None = None
    initial_population = population_rates(0)
    if population_display == "curves":
        population_collection = LineCollection(
            [np.column_stack((target.energy, curve)) for curve in initial_population],
            colors=_GREEN,
            linewidths=0.55,
            alpha=0.16,
            label=f"Best {min(max_population_curves, population_size)} population curves",
            zorder=1,
        )
        spectrum_ax.add_collection(population_collection)
    elif population_display == "envelope":
        if initial_population.size:
            lower, upper = np.quantile(initial_population, (0.1, 0.9), axis=0)
        else:
            lower = upper = replay.model_rates[0]
        envelope_artist = spectrum_ax.fill_between(
            target.energy,
            lower,
            upper,
            color=_GREEN,
            alpha=0.16,
            label=(
                f"Sampled population 10th--90th percentile "
                f"(up to {max_population_curves})"
            ),
            zorder=1,
        )

    model_line, = spectrum_ax.plot(
        target.energy,
        replay.model_rates[0],
        color=_ORANGE,
        lw=1.9,
        label="Best evolved response-folded model",
        zorder=5,
    )
    spectrum_ax.set_yscale("log")
    y_arrays = [target.rate, replay.model_rates]
    if population_display != "none":
        # Precompute the same deterministic population samples used by the
        # frames so global limits include the complete evolving population.
        for frame in range(replay.frame_count):
            curves = population_rates(frame)
            if curves.size:
                y_arrays.append(
                    np.quantile(curves, (0.1, 0.9), axis=0)
                    if population_display == "envelope"
                    else curves
                )
    if reference is not None:
        y_arrays.append(reference_y)
        if reference_error is not None:
            y_arrays.append(reference_y + reference_error)
    spectrum_ax.set_ylim(*_positive_limits(*y_arrays))
    x_values = [target.energy]
    if reference_x.size:
        x_values.append(reference_x)
    all_x = np.concatenate(x_values)
    x_span = max(float(np.max(all_x) - np.min(all_x)), 1.0)
    spectrum_ax.set_xlim(
        float(np.min(all_x)) - 0.02 * x_span,
        float(np.max(all_x)) + 0.02 * x_span,
    )
    spectrum_ax.set_ylabel(r"Count-rate density (s$^{-1}$ keV$^{-1}$)")
    spectrum_ax.grid(alpha=0.18, which="both")
    spectrum_ax.tick_params(labelbottom=False)
    spectrum_ax.legend(frameon=False, fontsize=7.5, loc="best")
    title_artist = spectrum_ax.set_title("GA spectral evolution")

    residual_ax.axhline(0.0, color="black", lw=0.8)
    residual_ax.axhline(3.0, color=_GREY, lw=0.65, ls="--")
    residual_ax.axhline(-3.0, color=_GREY, lw=0.65, ls="--")
    residual_line, = residual_ax.plot(
        target.energy,
        replay.residuals[0],
        ".",
        color=_BLUE,
        ms=3.0,
    )
    excluded = ~target.fit_mask
    if np.any(excluded):
        limit = _residual_limit(replay.residuals[:, target.fit_mask])
        residual_ax.fill_between(
            target.energy,
            0.0,
            1.0,
            where=excluded,
            transform=residual_ax.get_xaxis_transform(),
            color="#d8dde2",
            alpha=0.55,
            label="Outside objective fit band",
        )
    else:
        limit = _residual_limit(replay.residuals)
    residual_ax.set_ylim(-limit, limit)
    residual_ax.set_xlabel("Detector energy (keV)")
    residual_ax.set_ylabel("Pearson residual")
    residual_ax.grid(alpha=0.18)
    if np.any(excluded):
        residual_ax.legend(frameon=False, fontsize=7.5, loc="best")

    best_line, = score_ax.plot([], [], color=_BLUE, lw=1.8, label="Best C-stat")
    median_line, = score_ax.plot([], [], color=_ORANGE, lw=1.3, label="Median C-stat")
    score_cursor = score_ax.axvline(0, color=_PURPLE, lw=0.9, ls="--", alpha=0.65)
    score_ax.set_xlim(0.0, max(1.0, float(replay.frame_count - 1)))
    finite_scores = np.concatenate(
        (
            replay.best_scores[np.isfinite(replay.best_scores)],
            replay.median_scores[np.isfinite(replay.median_scores)],
        )
    )
    if (
        finite_scores.size
        and np.all(finite_scores > 0.0)
        and float(np.max(finite_scores)) / float(np.min(finite_scores)) > 100.0
    ):
        score_ax.set_yscale("log")
        score_ax.set_ylim(*_positive_limits(finite_scores))
    else:
        score_ax.set_ylim(*_linear_limits(finite_scores))
    score_ax.set_xlabel("Generation")
    score_ax.set_ylabel("C-statistic")
    score_ax.set_title("Population convergence")
    score_ax.grid(alpha=0.2, which="both")
    score_ax.legend(frameon=False, fontsize=8)

    parameter_text = parameter_ax.text(
        0.02,
        0.98,
        "",
        ha="left",
        va="top",
        transform=parameter_ax.transAxes,
        family="monospace",
        fontsize=8.5,
        color="#29323a",
    )

    notice = (
        f"Fit target and residuals: {target.label}. "
        "Pearson residual denominator: sqrt(max(model counts, 1))."
    )
    if reference is not None:
        notice += " " + reference.comparison_notice
    figure.text(
        0.5,
        0.995,
        textwrap.fill(notice, width=145),
        ha="center",
        va="top",
        color=_RED if reference is not None else _GREY,
        fontsize=8.5,
        fontweight="bold" if reference is not None else "normal",
    )
    figure.subplots_adjust(top=0.91, bottom=0.09, left=0.08, right=0.98)

    def update(frame_value: int) -> tuple[Any, ...]:
        frame = int(frame_value)
        model_line.set_ydata(replay.model_rates[frame])
        residual_line.set_ydata(replay.residuals[frame])
        prefix = frame_numbers[: frame + 1]
        best_line.set_data(prefix, replay.best_scores[: frame + 1])
        median_line.set_data(prefix, replay.median_scores[: frame + 1])
        score_cursor.set_xdata([frame, frame])
        title_artist.set_text(
            f"GA spectral evolution -- generation {frame} of {replay.frame_count - 1}"
        )
        parameter_text.set_text(
            _parameter_text(
                replay.parameters[frame],
                frame=frame,
                frame_count=replay.frame_count,
                best_score=float(replay.best_scores[frame]),
                median_score=float(replay.median_scores[frame]),
                population_size=population_size,
                reference=reference,
            )
        )

        changing: list[Any] = [
            model_line,
            residual_line,
            best_line,
            median_line,
            score_cursor,
            title_artist,
            parameter_text,
        ]
        curves = population_rates(frame)
        if population_collection is not None:
            population_collection.set_segments(
                [np.column_stack((target.energy, curve)) for curve in curves]
            )
            changing.append(population_collection)
        elif envelope_artist is not None:
            if curves.size:
                lower, upper = np.quantile(curves, (0.1, 0.9), axis=0)
            else:
                lower = upper = replay.model_rates[frame]
            _set_envelope_vertices(envelope_artist, target.energy, lower, upper)
            changing.append(envelope_artist)
        return tuple(changing)

    animation = FuncAnimation(
        figure,
        update,
        frames=range(replay.frame_count),
        init_func=lambda: update(0),
        interval=int(interval_ms),
        repeat=bool(repeat),
        blit=False,
        cache_frame_data=False,
    )
    # Small discoverability hooks are intentionally namespaced and do not alter
    # Matplotlib's public animation contract.
    animation._evoxrb_frame_count = replay.frame_count  # type: ignore[attr-defined]
    animation._evoxrb_frame_parameters = replay.parameters  # type: ignore[attr-defined]
    animation._evoxrb_update = update  # type: ignore[attr-defined]
    return animation


# A concise verb reads naturally at call sites while preserving the more
# explicit constructor name above.
build_spectral_animation = create_spectral_animation


def _animation_fps(animation: FuncAnimation, fps: float | None) -> float:
    if fps is not None:
        value = float(fps)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("fps must be finite and positive")
        return value
    event_source = getattr(animation, "event_source", None)
    interval = float(getattr(event_source, "interval", 120.0))
    return 1000.0 / max(interval, 1.0)


def _temporary_output(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.tmp{destination.suffix}")


def _make_html_self_contained(path: Path) -> None:
    """Remove Matplotlib's optional Font Awesome network dependency."""

    html = path.read_text(encoding="utf-8")
    html = re.sub(
        r"<link\s+rel=[\"']stylesheet[\"']\s+"
        r"href=[\"'][^\"']*font-awesome[^\"']*[\"']\s*/?>",
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    icons = {
        '<i class="fa fa-minus"></i>': "<span aria-hidden=\"true\">-</span>",
        '<i class="fa fa-fast-backward"></i>': "<span aria-hidden=\"true\">|&lt;</span>",
        '<i class="fa fa-step-backward"></i>': "<span aria-hidden=\"true\">&lt;</span>",
        '<i class="fa fa-play fa-flip-horizontal"></i>': "<span aria-hidden=\"true\">&#9664;</span>",
        '<i class="fa fa-pause"></i>': "<span aria-hidden=\"true\">||</span>",
        '<i class="fa fa-play"></i>': "<span aria-hidden=\"true\">&#9654;</span>",
        '<i class="fa fa-step-forward"></i>': "<span aria-hidden=\"true\">&gt;</span>",
        '<i class="fa fa-fast-forward"></i>': "<span aria-hidden=\"true\">&gt;|</span>",
        '<i class="fa fa-plus"></i>': "<span aria-hidden=\"true\">+</span>",
    }
    for original, replacement in icons.items():
        html = html.replace(original, replacement)
    path.write_text(html, encoding="utf-8")


def _validate_embedded_html(animation: FuncAnimation, path: Path) -> None:
    """Reject HTMLWriter output that silently dropped embedded frames."""

    html = path.read_text(encoding="utf-8")
    embedded = len(
        re.findall(r"^\s*frames\[\d+\]\s*=", html, flags=re.MULTILINE)
    )
    expected = getattr(animation, "_evoxrb_frame_count", None)
    if embedded == 0 or (expected is not None and embedded != int(expected)):
        expectation = "all frames" if expected is None else f"{int(expected)} frames"
        raise RuntimeError(
            "HTML export embedded "
            f"{embedded} frames instead of {expectation}; increase "
            "html_embed_limit_mb"
        )


def save_spectral_animation(
    animation: FuncAnimation,
    output_path: PathLike | None = None,
    *,
    fps: float | None = None,
    dpi: int = 110,
    html_embed_limit_mb: float = 512.0,
) -> Path:
    """Atomically save a replay as HTML, GIF, or MP4.

    HTML is the default and embeds all rendered frames plus playback controls.
    GIF uses Matplotlib's Pillow writer.  MP4 is accepted only when Matplotlib
    can locate an ffmpeg executable; no silent format fallback is performed.
    """

    destination = Path("spectral_evolution.html") if output_path is None else Path(output_path)
    if not destination.suffix:
        destination = destination.with_suffix(".html")
    suffix = destination.suffix.casefold()
    if suffix not in (".html", ".htm", ".gif", ".mp4"):
        raise ValueError("animation output must use .html, .htm, .gif, or .mp4")
    if int(dpi) < 1:
        raise ValueError("dpi must be positive")
    if not np.isfinite(html_embed_limit_mb) or html_embed_limit_mb <= 0.0:
        raise ValueError("html_embed_limit_mb must be finite and positive")

    resolved_fps = _animation_fps(animation, fps)
    metadata = {
        "title": "EvoXRB genetic-algorithm spectral evolution",
        "artist": "EvoXRB",
        "comment": "Reference overlays, when present, are visual-only and are not fitted.",
    }
    if suffix in (".html", ".htm"):
        writer: mpl_animation.AbstractMovieWriter = HTMLWriter(
            fps=resolved_fps,
            metadata=metadata,
            embed_frames=True,
            default_mode="loop",
            embed_limit=float(html_embed_limit_mb),
        )
    elif suffix == ".gif":
        if not mpl_animation.writers.is_available("pillow"):
            raise RuntimeError(
                "GIF export requires Matplotlib's Pillow writer; install Pillow first"
            )
        writer = PillowWriter(fps=resolved_fps, metadata=metadata)
    else:
        if not mpl_animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "MP4 export requires an ffmpeg executable visible to Matplotlib"
            )
        writer = FFMpegWriter(
            fps=resolved_fps,
            codec="h264",
            metadata=metadata,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_output(destination)
    try:
        animation.save(str(temporary), writer=writer, dpi=int(dpi))
        if suffix in (".html", ".htm"):
            _validate_embedded_html(animation, temporary)
            _make_html_self_contained(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def plot_best_fit_comparison(
    result: GARunResult,
    objective: Objective,
    search_space: SearchSpace,
    output_path: PathLike,
    *,
    reference: ReferenceSpectrum | None = None,
    polished_parameters: Mapping[str, float] | None = None,
    polished_score: float | None = None,
    polished_label: str = "GA + SciPy L-BFGS-B",
    dpi: int = 170,
) -> Path:
    """Save a static best-fit comparison with scientifically explicit labels.

    When ``polished_parameters`` are supplied, a distinct curve records the
    local SciPy L-BFGS-B refinement of the retained GA solution. Its
    C-statistic is always evaluated with the same objective used for the
    plotted synthetic target; a supplied score is checked for consistency.
    """

    destination = Path(output_path)
    if not destination.suffix:
        destination = destination.with_suffix(".png")
    if destination.suffix.casefold() != ".png":
        raise ValueError("best-fit comparison output must use .png")
    if int(dpi) < 1:
        raise ValueError("dpi must be positive")
    if polished_score is not None and polished_parameters is None:
        raise ValueError("polished_score requires polished_parameters")
    polished_label = str(polished_label).strip()
    if polished_parameters is not None and not polished_label:
        raise ValueError("polished_label must not be empty")

    target = _target_plot_data(objective)
    genes = np.asarray(result.best_genes, dtype=np.float64)
    if genes.shape != (search_space.ndim,):
        raise ValueError("GA best genes do not match the search space")
    parameters = search_space.decode(genes)
    model_counts = _expected_counts(
        objective,
        parameters,
        target.energy.size,
        context="the retained best GA solution",
    )
    model_rate = model_counts / target.scale
    residual = (target.counts - model_counts) / np.sqrt(np.maximum(model_counts, 1.0))
    polished_rate: NDArray[np.float64] | None = None
    polished_residual: NDArray[np.float64] | None = None
    resolved_polished_score: float | None = None
    if polished_parameters is not None:
        polished = {
            str(name): float(value) for name, value in polished_parameters.items()
        }
        polished_counts = _expected_counts(
            objective,
            polished,
            target.energy.size,
            context="the GA + SciPy L-BFGS-B solution",
        )
        polished_rate = polished_counts / target.scale
        polished_residual = (target.counts - polished_counts) / np.sqrt(
            np.maximum(polished_counts, 1.0)
        )
        evaluated_score = float(objective.evaluate(polished))
        if not np.isfinite(evaluated_score):
            raise ValueError("polished parameters do not have a finite C-statistic")
        if polished_score is not None and not np.isclose(
            evaluated_score,
            float(polished_score),
            rtol=1.0e-8,
            atol=1.0e-8,
        ):
            raise ValueError(
                "polished_score does not match the plotted polished parameters"
            )
        resolved_polished_score = evaluated_score

    figure = Figure(figsize=(9.0, 7.1), facecolor="white")
    FigureCanvasAgg(figure)
    spectrum_ax, residual_ax = figure.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    reference_x, reference_y, reference_error = _plot_target_and_reference(
        spectrum_ax, target, reference
    )
    spectrum_ax.plot(
        target.energy,
        model_rate,
        color=_ORANGE,
        lw=1.9,
        label="Raw GA best response-folded model",
        zorder=5,
    )
    if polished_rate is not None:
        spectrum_ax.plot(
            target.energy,
            polished_rate,
            color=_GREEN,
            lw=1.7,
            ls="--",
            label=polished_label,
            zorder=6,
        )
    spectrum_ax.set_yscale("log")
    y_arrays = [target.rate, model_rate]
    if polished_rate is not None:
        y_arrays.append(polished_rate)
    if reference is not None:
        y_arrays.append(reference_y)
        if reference_error is not None:
            y_arrays.append(reference_y + reference_error)
    spectrum_ax.set_ylim(*_positive_limits(*y_arrays))
    spectrum_ax.set_ylabel(r"Count-rate density (s$^{-1}$ keV$^{-1}$)")
    spectrum_ax.set_title("Best GA spectrum and optional user reference")
    spectrum_ax.grid(alpha=0.18, which="both")
    spectrum_ax.legend(frameon=False, fontsize=8, loc="best")

    residual_ax.axhline(0.0, color="black", lw=0.8)
    residual_ax.axhline(3.0, color=_GREY, lw=0.65, ls="--")
    residual_ax.axhline(-3.0, color=_GREY, lw=0.65, ls="--")
    residual_ax.plot(
        target.energy,
        residual,
        ".",
        color=_BLUE,
        ms=3.0,
        label="Raw GA",
    )
    residual_arrays = [residual[target.fit_mask]]
    if polished_residual is not None:
        residual_ax.plot(
            target.energy,
            polished_residual,
            "o",
            mfc="none",
            mec=_GREEN,
            ms=3.0,
            label=polished_label,
        )
        residual_arrays.append(polished_residual[target.fit_mask])
    limit = _residual_limit(np.concatenate(residual_arrays))
    excluded = ~target.fit_mask
    if np.any(excluded):
        residual_ax.fill_between(
            target.energy,
            0.0,
            1.0,
            where=excluded,
            transform=residual_ax.get_xaxis_transform(),
            color="#d8dde2",
            alpha=0.55,
            label="Outside objective fit band",
        )
    residual_ax.set(
        xlabel="Detector energy (keV)",
        ylabel="Pearson residual",
        ylim=(-limit, limit),
    )
    residual_ax.grid(alpha=0.18)
    if polished_residual is not None or np.any(excluded):
        residual_ax.legend(frameon=False, fontsize=8, loc="best")

    parameter_lines = [
        f"Raw GA C-stat = {_format_number(float(result.best_score))}",
        *(f"{name} = {_format_number(float(value))}" for name, value in parameters.items()),
    ]
    if resolved_polished_score is not None:
        parameter_lines.insert(
            1,
            f"{polished_label} C-stat = "
            + _format_number(resolved_polished_score),
        )
    spectrum_ax.text(
        0.985,
        0.03,
        "\n".join(parameter_lines),
        transform=spectrum_ax.transAxes,
        ha="right",
        va="bottom",
        family="monospace",
        fontsize=8,
        color="#29323a",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d5d9dc", "alpha": 0.85},
    )

    notice = (
        f"C-stat and residuals use only: {target.label}. "
        "Pearson residual denominator: sqrt(max(model counts, 1))."
    )
    if reference is not None:
        notice += " " + reference.comparison_notice
        if reference.source is not None:
            notice += f" Reference source: {reference.source}."
    figure.text(
        0.5,
        0.995,
        textwrap.fill(notice, width=132),
        ha="center",
        va="top",
        color=_RED if reference is not None else _GREY,
        fontsize=8.5,
        fontweight="bold" if reference is not None else "normal",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_output(destination)
    try:
        figure.savefig(
            temporary,
            dpi=int(dpi),
            bbox_inches="tight",
            facecolor="white",
        )
        temporary.replace(destination)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()
    return destination


def _backend_is_interactive(backend: str | None = None) -> bool:
    name = str(matplotlib.get_backend() if backend is None else backend).casefold()
    if "inline" in name:
        return False
    terminal = name.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
    return terminal not in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


class LiveSpectrumViewer:
    """A lightweight callable suitable for ``GeneticOptimizer.on_generation``.

    Construction is an explicit request for a GUI.  A clear error is therefore
    raised when Matplotlib is using a non-interactive backend; saved HTML replay
    remains available in every environment.
    """

    def __init__(
        self,
        objective: Objective,
        search_space: SearchSpace,
        *,
        reference: ReferenceSpectrum | None = None,
    ) -> None:
        if not _backend_is_interactive():
            raise RuntimeError(
                f"LiveSpectrumViewer requires an interactive Matplotlib backend; "
                f"current backend is {matplotlib.get_backend()!r}. Use the saved "
                "HTML replay in headless environments."
            )
        self.objective = objective
        self.search_space = search_space
        self.reference = reference
        self._target = _target_plot_data(objective)
        self._closed = False

        plt.ion()
        self.figure, (self.spectrum_ax, self.residual_ax) = plt.subplots(
            2,
            1,
            figsize=(8.6, 6.8),
            sharex=True,
            gridspec_kw={"height_ratios": (2.0, 1.0)},
        )
        _, reference_y, reference_error = _plot_target_and_reference(
            self.spectrum_ax, self._target, reference
        )
        self._reference_y = reference_y
        self._reference_error = reference_error
        self.model_line, = self.spectrum_ax.plot(
            self._target.energy,
            np.full_like(self._target.energy, np.nan),
            color=_ORANGE,
            lw=1.8,
            label="Current best folded model",
        )
        self.spectrum_ax.set_yscale("log")
        y_arrays = [self._target.rate]
        if reference is not None:
            y_arrays.append(reference_y)
            if reference_error is not None:
                y_arrays.append(reference_y + reference_error)
        self.spectrum_ax.set_ylim(*_positive_limits(*y_arrays))
        self.spectrum_ax.set_ylabel(r"Count-rate density (s$^{-1}$ keV$^{-1}$)")
        self.spectrum_ax.grid(alpha=0.18, which="both")
        self.spectrum_ax.legend(frameon=False, fontsize=8)
        self.status_text = self.spectrum_ax.text(
            0.985,
            0.03,
            "Waiting for generation 0",
            transform=self.spectrum_ax.transAxes,
            ha="right",
            va="bottom",
            family="monospace",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d5d9dc", "alpha": 0.85},
        )
        self.residual_ax.axhline(0.0, color="black", lw=0.8)
        self.residual_ax.axhline(3.0, color=_GREY, lw=0.65, ls="--")
        self.residual_ax.axhline(-3.0, color=_GREY, lw=0.65, ls="--")
        self.residual_line, = self.residual_ax.plot(
            self._target.energy,
            np.full_like(self._target.energy, np.nan),
            ".",
            color=_BLUE,
            ms=3.0,
        )
        self.residual_ax.set(
            xlabel="Detector energy (keV)",
            ylabel="Pearson residual",
            ylim=(-10.0, 10.0),
        )
        excluded = ~self._target.fit_mask
        if np.any(excluded):
            self.residual_ax.fill_between(
                self._target.energy,
                0.0,
                1.0,
                where=excluded,
                transform=self.residual_ax.get_xaxis_transform(),
                color="#d8dde2",
                alpha=0.55,
                label="Outside objective fit band",
            )
            self.residual_ax.legend(frameon=False, fontsize=7.5, loc="best")
        self.residual_ax.grid(alpha=0.18)
        self.figure.tight_layout()
        self.figure.show()

    def __call__(self, snapshot: GenerationSnapshot) -> None:
        if self._closed:
            return
        genes = np.asarray(snapshot.best_genes, dtype=np.float64)
        if genes.shape != (self.search_space.ndim,):
            raise ValueError("generation snapshot does not match the viewer search space")
        parameters = self.search_space.decode(genes)
        model_counts = _expected_counts(
            self.objective,
            parameters,
            self._target.energy.size,
            context=f"live generation {snapshot.generation}",
        )
        model_rate = model_counts / self._target.scale
        residual = (self._target.counts - model_counts) / np.sqrt(
            np.maximum(model_counts, 1.0)
        )
        self.model_line.set_ydata(model_rate)
        self.residual_line.set_ydata(residual)
        y_arrays = [self._target.rate, model_rate, self._reference_y]
        if self._reference_error is not None:
            y_arrays.append(self._reference_y + self._reference_error)
        self.spectrum_ax.set_ylim(*_positive_limits(*y_arrays))
        limit = _residual_limit(residual[self._target.fit_mask])
        self.residual_ax.set_ylim(-limit, limit)
        self.spectrum_ax.set_title(
            f"Live GA spectrum -- generation {snapshot.generation}"
        )
        lines = [
            f"Evaluations: {snapshot.evaluations:,}",
            f"Best C-stat: {_format_number(snapshot.best_score)}",
            f"Median C-stat: {_format_number(snapshot.median_score)}",
            *(f"{name}: {_format_number(value)}" for name, value in parameters.items()),
        ]
        self.status_text.set_text("\n".join(lines))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def show(self, *, block: bool = True) -> None:
        """Show the current figure, optionally blocking until it is closed."""

        if not self._closed:
            plt.show(block=block)

    def close(self) -> None:
        """Close the viewer; later callback notifications become no-ops."""

        if not self._closed:
            plt.close(self.figure)
            self._closed = True


__all__ = [
    "LiveSpectrumViewer",
    "PopulationDisplay",
    "build_spectral_animation",
    "create_spectral_animation",
    "plot_best_fit_comparison",
    "save_spectral_animation",
]
