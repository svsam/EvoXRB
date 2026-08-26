"""Command-line interface for the pure-Python synthetic case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
