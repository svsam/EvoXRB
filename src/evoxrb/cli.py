"""Command-line interface for the pure-Python synthetic case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from . import SYNTHETIC_LABEL, __version__
from .campaign import (
    CampaignPaths,
    figures_stage,
    fit_stage,
    inference_stage,
    report_stage,
    run_case_study,
    simulate_stage,
    timing_stage,
)
from .config import load_config
from .genetic import GenerationSnapshot


def _add_common(parser: argparse.ArgumentParser, *, resume: bool = False) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/case_study.yaml"),
        help="YAML case-study configuration (default: config/case_study.yaml)",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="fast deterministic CI run or the complete acceptance campaign",
    )
    if resume:
        parser.add_argument(
            "--resume",
            action="store_true",
            help="resume compatible GA/posterior checkpoints and completed artifacts",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoxrb",
        description=(
            f"{SYNTHETIC_LABEL} educational X-ray optimization and timing case study"
        ),
    )
    parser.add_argument("--version", action="version", version=f"evoxrb {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="simulate response-folded Poisson spectra")
    _add_common(simulate, resume=True)

    fit = commands.add_parser("fit", help="fit simulated spectra")
    _add_common(fit, resume=True)
    fit.add_argument("--method", required=True, choices=("scipy", "ga"))

    infer = commands.add_parser("infer", help="run GA-seeded posterior inference")
    _add_common(infer, resume=True)

    timing = commands.add_parser("timing", help="simulate and fit synthetic power spectra")
    _add_common(timing, resume=True)

    animate = commands.add_parser(
        "animate",
        help="run one GA fit and save a playable generation-by-generation spectrum",
    )
    _add_common(animate, resume=True)
    animate.add_argument("--epoch", default="E08", help="synthetic epoch ID (default: E08)")
    animate.add_argument(
        "--model",
        choices=("powerlaw", "cutoff"),
        default="powerlaw",
        help="educational continuum fitted by the GA",
    )
    animate.add_argument("--seed", type=int, help="override the deterministic GA seed")
    animate.add_argument("--generations", type=int, help="override the profile generation limit")
    animate.add_argument(
        "--population-size",
        type=int,
        help="override the profile population size (minimum four)",
    )
    animate.add_argument(
        "--output",
        type=Path,
        help="saved .html, .gif, or .mp4 replay (default: results/animations/...html)",
    )
    animate.add_argument(
        "--comparison-output",
        type=Path,
        help="saved static .png comparison (default: results/animations/...png)",
    )
    animate.add_argument(
        "--checkpoint",
        type=Path,
        help="override the resumable GA checkpoint path",
    )
    animate.add_argument(
        "--reference-csv",
        type=Path,
        help="user-supplied real/reference spectrum for visual comparison only",
    )
    animate.add_argument(
        "--population-display",
        choices=("none", "curves", "envelope"),
        default="envelope",
        help="population spectra drawn behind the best member (default: envelope)",
    )
    animate.add_argument(
        "--max-population-curves",
        type=int,
        default=16,
        help="maximum population spectra sampled per frame (default: 16)",
    )
    animate.add_argument("--fps", type=float, default=8.0, help="saved replay speed")
    animate.add_argument(
        "--live",
        action="store_true",
        help="also update an interactive Matplotlib window during calculation",
    )
    animate.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-generation terminal progress",
    )

    report = commands.add_parser("report", help="render figures and Markdown report")
    _add_common(report)

    complete = commands.add_parser(
        "run-case-study", help="run simulation, fits, recovery, inference, timing, and report"
    )
    _add_common(complete, resume=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    profile = args.profile
    resume = bool(getattr(args, "resume", False))

    if args.command == "simulate":
        _, spectra = simulate_stage(config, profile=profile, resume=resume)
        paths = CampaignPaths.from_config(config)
        result = {"spectra": len(spectra), "results": str(paths.results)}
    elif args.command == "fit":
        result = {
            "fit_results": str(
                fit_stage(
                    config,
                    profile=profile,
                    method=args.method,
                    resume=resume,
                )
            )
        }
    elif args.command == "infer":
        result = {
            "posterior_summary": str(
                inference_stage(config, profile=profile, resume=resume)
            )
        }
    elif args.command == "timing":
        result = {
            "timing_results": str(
                timing_stage(config, profile=profile, resume=resume)
            )
        }
    elif args.command == "animate":
        from .evolution import run_animated_fit

        def show_progress(snapshot: GenerationSnapshot) -> None:
            generation = snapshot.generation
            evaluations = snapshot.evaluations
            best_score = snapshot.best_score
            median_score = snapshot.median_score
            stop_reason = snapshot.stop_reason
            suffix = "" if stop_reason is None else f" | {stop_reason}"
            print(
                f"generation {generation:4d} | evaluations {evaluations:7d} | "
                f"best C-stat {best_score:12.6g} | median {median_score:12.6g}{suffix}",
                file=sys.stderr,
                flush=True,
            )

        artifacts = run_animated_fit(
            config,
            profile=profile,
            epoch_id=args.epoch,
            continuum=args.model,
            seed=args.seed,
            generations=args.generations,
            population_size=args.population_size,
            output_path=args.output,
            comparison_path=args.comparison_output,
            checkpoint_path=args.checkpoint,
            reference_csv=args.reference_csv,
            population_display=args.population_display,
            max_population_curves=args.max_population_curves,
            fps=args.fps,
            live=args.live,
            resume=resume,
            on_generation=None if args.quiet else show_progress,
        )
        result = artifacts.summary()
        if not args.quiet:
            polish_status = (
                "converged"
                if artifacts.scipy_result.success
                else f"not converged: {artifacts.scipy_result.message}"
            )
            print(
                f"SciPy polish C-stat: {artifacts.scipy_result.score:.6g} "
                f"({polish_status})",
                file=sys.stderr,
                flush=True,
            )
    elif args.command == "report":
        figures = figures_stage(config, profile=profile)
        report_path = report_stage(config, profile=profile)
        result = {"figures": [str(path) for path in figures], "report": str(report_path)}
    else:
        result = run_case_study(config, profile=profile, resume=resume)

    print(json.dumps({"label": SYNTHETIC_LABEL, **result}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
