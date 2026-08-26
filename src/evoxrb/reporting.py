"""Generate the portfolio-facing Markdown case-study report."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import SYNTHETIC_LABEL
from .io import ensure_directories


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str], limit: int = 12) -> str:
    available = [name for name in columns if name in frame.columns]
    if frame.empty or not available:
        return "_No completed rows are available for this stage._"
    view = frame.loc[:, available].head(limit).copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda value: f"{value:.5g}")
    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join("---" for _ in available) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return f"**{SYNTHETIC_LABEL} table.**\n\n" + "\n".join([header, separator, *rows])


def build_report(
    results_dir: str | Path,
    figure_dir: str | Path,
    report_dir: str | Path,
    *,
    config_hash: str,
    profile: str,
) -> Path:
    """Build a self-contained Markdown summary from machine-readable outputs."""

    results = Path(results_dir)
    figures = Path(figure_dir)
    reports = ensure_directories(report_dir)[0]

    truth = _read_csv(results / "truth.csv")
    fits = _read_csv(results / "fit_results.csv")
    recovery = _read_csv(results / "recovery_results.csv")
    recovery_summary = _read_csv(results / "recovery_summary.csv")
    posterior = _read_csv(results / "posterior_summary.csv")
    timing = _read_csv(results / "timing_results.csv")
    provenance = _read_json(results / "provenance.json")
    acceptance = _read_json(results / "acceptance.json")
    for frame in (fits, recovery, recovery_summary, posterior, timing):
        if not frame.empty and "profile" in frame.columns:
            frame.drop(
                frame.index[frame["profile"].astype(str) != profile], inplace=True
            )

    relative_figures: list[str] = []
    if figures.exists():
        for path in sorted(figures.glob("*.png")):
            relative_figures.append(Path(os.path.relpath(path, reports)).as_posix())
    figure_markdown = "\n\n".join(
        f"![{SYNTHETIC_LABEL}: {Path(path).stem}]({path})" for path in relative_figures
    ) or "_Figures have not been generated for this run._"

    fit_summary = ""
    if not fits.empty and "statistic" in fits:
        fixed_primary = fits[
            (fits.get("fit_model") == "educational_powerlaw")
            & (fits.get("stage").isin(["polished", "best_multistart"]))
        ]
        best = (fixed_primary if not fixed_primary.empty else fits).sort_values("statistic").iloc[0]
        fit_summary = (
            f"The lowest completed fixed-column primary-model C-statistic is "
            f"**{best['statistic']:.3f}** "
            f"for `{best.get('epoch_id', 'unknown')}` using "
            f"`{best.get('method', 'unknown')}` / `{best.get('fit_model', 'unknown')}`."
        )
    else:
        fit_summary = "No optimisation results were completed in this profile."

    package_text = ", ".join(
        f"{name} {version}" for name, version in provenance.get("packages", {}).items()
    ) or "package versions unavailable"
    acceptance_text = (
        "passed" if acceptance.get("valid") is True else "not yet run or incomplete"
    )
    fit_table = (
        fits.sort_values(["epoch_id", "fit_model", "method", "stage"])
        if not fits.empty
        and {"epoch_id", "fit_model", "method", "stage"}.issubset(fits.columns)
        else fits
    )

    content = f"""# EvoXRB: Pure-Python Synthetic Case Study

> **{SYNTHETIC_LABEL}.** This report contains no real NICER observations and
> makes no measurement of MAXI J1820+070. The spectral components, detector,
> background and timing signals are educational approximations with known
> injected truth.

## Run provenance

- Profile: `{profile}`
- Configuration hash: `{config_hash}`
- Master seed: `{provenance.get('master_seed', 'unavailable')}`
- Python: `{provenance.get('python', 'unavailable')}`
- Packages: {package_text}
- Recorded end-to-end runtime: `{provenance.get('runtime_seconds', 'unavailable')}` seconds
- Artifact acceptance: **{acceptance_text}**
- Truth epochs generated: `{len(truth)}`
- Optimisation rows completed: `{len(fits)}`
- Recovery rows completed: `{len(recovery)}`
- Posterior rows completed: `{len(posterior)}`
- Timing rows completed: `{len(timing)}`

## Question and scope

The case study asks when a population-based global optimiser is useful for a
response-folded, Poisson X-ray fitting problem. A custom real-valued genetic
algorithm is compared with deterministic multi-start SciPy optimisation. The
best global solution seeds posterior inference; it is not itself treated as an
uncertainty distribution.

Expected detector counts are generated only after the photon model has been
multiplied by an effective-area curve and redistributed between detector
channels. The primary fit statistic is the Poisson C-statistic. A second,
cutoff continuum supplies a controlled model-misspecification experiment.

The phase names and reference dates are contextual anchors inspired by the
published evolution; they are not analyzed observations. See
[Li et al.](https://arxiv.org/html/2407.08421v2). The soft/hard timing-band
design is publication-inspired; see
[Stiele & Kong](https://arxiv.org/abs/1912.07625).

## Educational forward model

The photon model is explicitly approximate:

- absorption: `exp(-3 NH E^-3)`;
- disk-like component: `1e-4 Ndisk E^(-2/3) exp(-E/Tin)`;
- power law: `K E^(-Gamma)`;
- Comptonization cutoff surrogate: `K E^(-Gamma) exp(-E/20 keV)`.

These are **not** exact `TBabs`, `diskbb`, or `nthcomp` implementations. The
primary spectrum is absorption times (disk-like + power law); the controlled
alternative replaces the power law with the cutoff surrogate.

The response uses 600 true-energy bins over 0.1–12 keV, 236 detector channels
over 0.2–12 keV, and a 0.5–10 keV fit mask. Its effective area is a
shape-preserving interpolation through the declared NICER-inspired anchors.
Gaussian redistribution uses `FWHM(E)=0.085 sqrt(E)` keV and every response
column is normalized. Expected counts include the fixed synthetic background
and are computed as `exposure * R @ (area * flux * dE) + exposure * background`
before Poisson sampling.

## Optimization and inference design

The four normalized GA genes represent `Tin`, `log10(Ndisk)`, `Gamma`, and
`log10(K)`. The full profile uses 192 individuals, five independent seeds,
tournament size 3, 0.9 simulated-binary crossover, annealed bounded polynomial
mutation, 3% elitism, stagnation-triggered 5% immigrants, and at most 300
generations. Twenty deterministic Latin-hypercube SciPy starts form the
control. Raw GA, best SciPy, and GA-followed-by-local-polish rows remain
separate. A free-`NH` 0.01–1.0 sensitivity run is also retained separately.
The sub-unit high-signal `Delta C` acceptance check applies to the explicitly
defined GA-seeded local-polish stage; raw GA scores remain visible as a
basin-finding diagnostic rather than being silently replaced.

The GA solution seeds `emcee`. Linear parameters receive uniform priors and
normalizations/optional `NH` receive log-uniform priors through the normalized
gene transform. `dynesty` is triggered for competitive separated GA modes,
unconverged ensemble chains, or excessive posterior boundary mass.

## Injected outburst

{_markdown_table(truth, ['epoch_id', 'phase', 'reference_mjd', 'tin', 'ndisk', 'gamma', 'norm', 'qpo_hz'])}

## Optimisation comparison

{fit_summary}

{_markdown_table(fit_table, ['epoch_id', 'fit_model', 'method', 'stage', 'seed', 'statistic', 'delta_c', 'tin', 'ndisk', 'gamma', 'norm', 'wall_time_s'], limit=30)}

## Recovery and uncertainty

{_markdown_table(recovery, ['scenario', 'exposure_s', 'realization', 'parameter', 'truth', 'estimate', 'error', 'delta_c'])}

### Aggregate recovery metrics

{_markdown_table(recovery_summary, ['scenario', 'exposure_s', 'method', 'stage', 'parameter', 'bias', 'rmse', 'failure_rate', 'boundary_hits', 'mean_delta_c', 'mean_evaluations', 'mean_runtime_s'], limit=30)}

{_markdown_table(posterior, ['epoch_id', 'fit_model', 'parameter', 'median', 'error_minus', 'error_plus', 'sampler', 'converged'])}

## Synthetic timing

The timing series use 0.5–2 and 2–10 keV-inspired bands. Detections are labelled
only as **type-C-like**, because the signals were injected into synthetic light
curves rather than classified from telescope data. Full runs use 1/512-s bins,
512-s segments, Timmer–Koenig red noise, Lorentzian QPOs, averaged periodograms,
SciPy Lorentzian fits, and 200 bootstrap realizations. Acceptance requires
`Q >= 2` and QPO amplitude/error `>= 3`.

{_markdown_table(timing, ['epoch_id', 'phase', 'hardness_ratio', 'intensity_cps', 'qpo_injected_hz', 'qpo_centroid_hz', 'q_factor', 'significance', 'classification'])}

## Figures

{figure_markdown}

## Interpretation boundary

Successful recovery demonstrates that the optimiser and inference machinery
work for this declared synthetic forward model. Failure under the cutoff-truth
/ power-law-fit experiment demonstrates model misspecification: a numerically
excellent optimum does not make an approximate model physically correct.

The smoke profile proves integration and reproducibility with reduced numerical
budgets. Scientific acceptance of the stated campaign metrics belongs to the
opt-in `full` profile; smoke-run precision is not presented as a replacement.
"""
    destination = reports / "case_study.md"
    destination.write_text(content, encoding="utf-8")
    return destination
