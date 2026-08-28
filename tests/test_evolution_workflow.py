from __future__ import annotations

from dataclasses import replace
import json

import pytest

from evoxrb.cli import build_parser
from evoxrb.config import load_config
from evoxrb.evolution import (
    _build_synthetic_target,
    _objective_signature,
    run_animated_fit,
)


def test_animated_fit_runs_ga_polishes_and_saves_artifacts(tmp_path) -> None:
    config = load_config()
    isolated = replace(
        config,
        project={
            **config.project,
            "output_dir": str(tmp_path / "results"),
            "figure_dir": str(tmp_path / "figures"),
            "report_dir": str(tmp_path / "reports"),
        },
    )
    generations: list[int] = []
    artifacts = run_animated_fit(
        isolated,
        profile="smoke",
        epoch_id="E08",
        generations=2,
        population_size=8,
        output_path=tmp_path / "evolution.html",
        comparison_path=tmp_path / "comparison.png",
        population_display="none",
        fps=4.0,
        on_generation=lambda snapshot: generations.append(snapshot.generation),
    )

    assert generations == [0, 1, 2]
    assert artifacts.ga_result.generations == 2
    assert artifacts.scipy_result.score <= artifacts.ga_result.best_score
    assert artifacts.animation_path.stat().st_size > 10_000
    assert artifacts.comparison_path.stat().st_size > 1_000
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["epoch_id"] == "E08"
    assert summary["ga"]["generations"] == 2
    assert summary["scipy_polish"]["score"] <= summary["ga"]["best_score"]


def test_animate_cli_exposes_runtime_and_export_controls() -> None:
    args = build_parser().parse_args(
        [
            "animate",
            "--epoch",
            "E10",
            "--generations",
            "3",
            "--population-size",
            "8",
            "--output",
            "movie.gif",
            "--population-display",
            "curves",
            "--quiet",
        ]
    )

    assert args.command == "animate"
    assert args.epoch == "E10"
    assert args.generations == 3
    assert args.population_size == 8
    assert args.output.name == "movie.gif"
    assert args.population_display == "curves"
    assert args.quiet is True


def test_animated_fit_checkpoint_can_extend_generation_limit(tmp_path) -> None:
    config = load_config()
    isolated = replace(
        config,
        project={
            **config.project,
            "output_dir": str(tmp_path / "results"),
            "figure_dir": str(tmp_path / "figures"),
            "report_dir": str(tmp_path / "reports"),
        },
    )
    checkpoint = tmp_path / "extendable.npz"
    first = run_animated_fit(
        isolated,
        generations=1,
        population_size=8,
        checkpoint_path=checkpoint,
        output_path=tmp_path / "first.html",
        comparison_path=tmp_path / "first.png",
        population_display="none",
    )
    events = []
    resumed = run_animated_fit(
        isolated,
        generations=2,
        population_size=8,
        checkpoint_path=checkpoint,
        output_path=tmp_path / "resumed.html",
        comparison_path=tmp_path / "resumed.png",
        population_display="none",
        resume=True,
        on_generation=events.append,
    )

    assert first.ga_result.generations == 1
    assert resumed.ga_result.generations == 2
    assert [event.generation for event in events] == [1, 2]
    assert events[0].stop_reason is None
    assert events[-1].stop_reason == "max_generations"

    changed_objective = replace(
        isolated,
        search={**isolated.search, "nh_fixed": 0.2},
    )
    with pytest.raises(ValueError, match="objective/data"):
        run_animated_fit(
            changed_objective,
            generations=3,
            population_size=8,
            checkpoint_path=checkpoint,
            output_path=tmp_path / "mismatch.html",
            comparison_path=tmp_path / "mismatch.png",
            population_display="none",
            resume=True,
        )


def test_objective_signature_includes_response_bin_edges() -> None:
    config = load_config()
    response, spectrum, _ = _build_synthetic_target(config, "E08")
    original = _objective_signature(config, response, spectrum, "powerlaw")

    true_edges = response.true_edges.copy()
    true_edges[1] += 0.1 * (true_edges[2] - true_edges[1])
    changed_true = replace(response, true_edges=true_edges)
    assert _objective_signature(config, changed_true, spectrum, "powerlaw") != original

    detector_edges = response.detector_edges.copy()
    detector_edges[1] += 0.1 * (detector_edges[2] - detector_edges[1])
    changed_detector = replace(response, detector_edges=detector_edges)
    assert (
        _objective_signature(config, changed_detector, spectrum, "powerlaw")
        != original
    )
