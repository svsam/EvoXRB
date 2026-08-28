from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pytest

import evoxrb.animation as spectral_animation
from evoxrb.animation import (
    LiveSpectrumViewer,
    create_spectral_animation,
    plot_best_fit_comparison,
    save_spectral_animation,
)
from evoxrb.genetic import GAConfig, GARunResult, GeneticOptimizer
from evoxrb.instrument import InstrumentResponse
from evoxrb.models import SpectrumModel
from evoxrb.objective import Objective
from evoxrb.parameters import ParameterSpec, SearchSpace
from evoxrb.reference import ReferenceSpectrum
from evoxrb.simulation import simulate_spectrum


@dataclass(frozen=True)
class _AnimationCase:
    result: GARunResult
    objective: Objective
    search_space: SearchSpace
    reference: ReferenceSpectrum


@pytest.fixture(scope="module")
def animation_case() -> _AnimationCase:
    edges = np.linspace(0.4, 5.2, 13)
    energy = 0.5 * (edges[:-1] + edges[1:])
    response = InstrumentResponse(
        true_edges=edges,
        true_energy=energy,
        detector_edges=edges.copy(),
        detector_energy=energy.copy(),
        effective_area=np.linspace(90.0, 125.0, energy.size),
        redistribution=np.eye(energy.size),
        background_rate_density=np.full(energy.size, 0.015),
        label="Synthetic / NICER-inspired compact test response",
    )
    model = SpectrumModel("powerlaw", fixed_nh=0.15)
    truth = {"tin": 0.75, "ndisk": 650.0, "gamma": 2.05, "norm": 0.22}
    spectrum = simulate_spectrum(
        response,
        model,
        truth,
        exposure_s=24.0,
        seed=71,
        epoch_id="animation-test",
    )
    objective = Objective(spectrum, response, model)
    search_space = SearchSpace(
        ParameterSpec("tin", 0.3, 1.3),
        ParameterSpec("ndisk", 40.0, 4_000.0, "log10"),
        ParameterSpec("gamma", 1.3, 3.0),
        ParameterSpec("norm", 0.01, 2.0, "log10"),
    )
    optimizer = GeneticOptimizer(
        search_space,
        GAConfig(
            population_size=8,
            max_generations=2,
            min_generations=9,
            immigrant_stagnation_generations=4,
            stop_stagnation_generations=4,
        ),
    )
    result = optimizer.optimize(objective.evaluate, seed=1_820_070)

    scale = spectrum.exposure_s * np.diff(spectrum.detector_edges)
    reference = ReferenceSpectrum(
        energy_keV=spectrum.detector_energy[::-1],
        count_rate_density=(spectrum.counts / scale * 1.03)[::-1],
        count_rate_error=(np.sqrt(np.maximum(spectrum.counts, 1.0)) / scale)[::-1],
        label="User-supplied comparison data",
        source="animation-reference.csv",
    )
    return _AnimationCase(result, objective, search_space, reference)


def _close_animation(animation: FuncAnimation) -> None:
    # Matplotlib warns when an animation is intentionally inspected without
    # being rendered; focused unit tests explicitly account for that lifecycle.
    animation._draw_was_started = True
    plt.close(animation._fig)


def test_replay_contains_and_updates_every_sampled_generation(
    animation_case: _AnimationCase,
) -> None:
    case = animation_case
    animation = create_spectral_animation(
        case.result,
        case.objective,
        case.search_space,
        reference=case.reference,
        population_display="envelope",
        max_population_curves=4,
        interval_ms=40,
    )

    expected_frames = case.result.best_gene_history.shape[0]
    assert isinstance(animation, FuncAnimation)
    assert animation._evoxrb_frame_count == expected_frames
    assert expected_frames == case.result.generations + 1
    assert len(animation._evoxrb_frame_parameters) == expected_frames

    final_frame = expected_frames - 1
    changed = animation._evoxrb_update(final_frame)
    texts = [artist.get_text() for artist in changed if hasattr(artist, "get_text")]
    assert any(f"generation {final_frame}" in text.casefold() for text in texts)
    assert any("Best C-stat" in text and "Decoded best parameters" in text for text in texts)
    target = spectral_animation._target_plot_data(case.objective)
    envelope_bounds = []
    for frame in range(expected_frames):
        curves = spectral_animation._population_model_rates(
            frame,
            case.result,
            case.objective,
            case.search_space,
            target,
            limit=4,
            stratified=True,
        )
        envelope_bounds.append(np.quantile(curves, (0.1, 0.9), axis=0))
    y_min, y_max = animation._fig.axes[0].get_ylim()
    positive_bounds = np.asarray(envelope_bounds)[np.asarray(envelope_bounds) > 0.0]
    assert y_min <= float(np.min(positive_bounds))
    assert y_max >= float(np.max(positive_bounds))
    _close_animation(animation)


def test_default_html_export_embeds_frames_and_playback_controls(
    animation_case: _AnimationCase,
    tmp_path: Path,
) -> None:
    case = animation_case
    animation = create_spectral_animation(
        case.result,
        case.objective,
        case.search_space,
        reference=case.reference,
        interval_ms=80,
    )

    output = save_spectral_animation(
        animation,
        tmp_path / "spectral_replay",
        fps=5,
        dpi=55,
    )

    assert output == tmp_path / "spectral_replay.html"
    html = output.read_text(encoding="utf-8")
    assert "data:image/png;base64" in html
    assert "button" in html.casefold()
    assert "font-awesome" not in html.casefold()
    assert "https://" not in html.casefold()
    assert output.stat().st_size > 10_000
    assert not list(tmp_path.glob("spectral_replay_frames*"))
    _close_animation(animation)


def test_gif_export_uses_pillow_writer(
    animation_case: _AnimationCase,
    tmp_path: Path,
) -> None:
    if not spectral_animation.mpl_animation.writers.is_available("pillow"):
        pytest.skip("Matplotlib Pillow writer is unavailable")
    case = animation_case
    animation = create_spectral_animation(
        case.result,
        case.objective,
        case.search_space,
        interval_ms=100,
    )

    output = save_spectral_animation(
        animation,
        tmp_path / "spectral_replay.gif",
        fps=4,
        dpi=45,
    )

    assert output.read_bytes()[:4] == b"GIF8"
    assert output.stat().st_size > 1_000
    _close_animation(animation)


def test_html_embed_limit_failure_preserves_existing_output(
    animation_case: _AnimationCase,
    tmp_path: Path,
) -> None:
    case = animation_case
    animation = create_spectral_animation(
        case.result,
        case.objective,
        case.search_space,
    )
    output = tmp_path / "existing_replay.html"
    output.write_text("existing-good-output", encoding="utf-8")

    with pytest.raises(RuntimeError, match="increase html_embed_limit_mb"):
        save_spectral_animation(
            animation,
            output,
            dpi=45,
            html_embed_limit_mb=1.0e-6,
        )

    assert output.read_text(encoding="utf-8") == "existing-good-output"
    assert not list(tmp_path.glob(".existing_replay.tmp*"))
    _close_animation(animation)


def test_mp4_fails_clearly_when_ffmpeg_is_unavailable(
    animation_case: _AnimationCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = animation_case
    animation = create_spectral_animation(
        case.result,
        case.objective,
        case.search_space,
    )
    registry_type = type(spectral_animation.mpl_animation.writers)
    original = registry_type.is_available

    def unavailable(registry: object, name: str) -> bool:
        return False if name == "ffmpeg" else original(registry, name)

    monkeypatch.setattr(registry_type, "is_available", unavailable)
    with pytest.raises(RuntimeError, match="ffmpeg executable"):
        save_spectral_animation(animation, tmp_path / "spectral_replay.mp4")
    assert not (tmp_path / "spectral_replay.mp4").exists()
    _close_animation(animation)


def test_mp4_export_when_ffmpeg_is_available(
    animation_case: _AnimationCase,
    tmp_path: Path,
) -> None:
    if not spectral_animation.mpl_animation.writers.is_available("ffmpeg"):
        pytest.skip("ffmpeg executable is unavailable")
    case = animation_case
    animation = create_spectral_animation(
        case.result,
        case.objective,
        case.search_space,
    )

    output = save_spectral_animation(
        animation,
        tmp_path / "spectral_replay.mp4",
        fps=4,
        dpi=45,
    )

    assert b"ftyp" in output.read_bytes()[:32]
    assert output.stat().st_size > 1_000
    _close_animation(animation)


def test_static_comparison_is_a_nonempty_png(
    animation_case: _AnimationCase,
    tmp_path: Path,
) -> None:
    case = animation_case
    output = plot_best_fit_comparison(
        case.result,
        case.objective,
        case.search_space,
        tmp_path / "best_reference_comparison",
        reference=case.reference,
        polished_parameters=case.result.best_parameters,
        polished_score=case.result.best_score,
        dpi=75,
    )

    assert output == tmp_path / "best_reference_comparison.png"
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert output.stat().st_size > 5_000

    with pytest.raises(ValueError, match="polished_score requires"):
        plot_best_fit_comparison(
            case.result,
            case.objective,
            case.search_space,
            tmp_path / "invalid_polish.png",
            polished_score=case.result.best_score,
        )


def test_export_validation_and_headless_live_viewer_are_explicit(
    animation_case: _AnimationCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = animation_case
    animation = create_spectral_animation(
        case.result,
        case.objective,
        case.search_space,
    )
    with pytest.raises(ValueError, match="html.*gif.*mp4"):
        save_spectral_animation(animation, tmp_path / "replay.avi")
    _close_animation(animation)

    monkeypatch.setattr(spectral_animation.matplotlib, "get_backend", lambda: "Agg")
    with pytest.raises(RuntimeError, match="interactive Matplotlib backend.*Agg"):
        LiveSpectrumViewer(case.objective, case.search_space)
    assert spectral_animation._backend_is_interactive("QtAgg")
    assert not spectral_animation._backend_is_interactive(
        "module://matplotlib_inline.backend_inline"
    )
