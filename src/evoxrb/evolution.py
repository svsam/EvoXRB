"""Run, watch, replay, and compare one response-folded spectral GA fit."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
import numpy as np
from numpy.typing import ArrayLike

from . import SYNTHETIC_LABEL
from .animation import (
    LiveSpectrumViewer,
    PopulationDisplay,
    create_spectral_animation,
    plot_best_fit_comparison,
    save_spectral_animation,
)
from .campaign import CampaignPaths, _ga_config, _profile_sections, epoch_truths
from .config import CaseStudyConfig
from .genetic import (
    GARunResult,
    GenerationCallback,
    GenerationSnapshot,
    GeneticOptimizer,
)
from .io import atomic_write_json
from .instrument import InstrumentResponse, default_nicer_inspired_response
from .models import SpectrumModel
from .objective import Objective, search_space_from_config
from .optimization import SciPyRunResult, local_polish
from .reference import ReferenceSpectrum, load_reference_spectrum_csv
from .simulation import derive_seed, simulate_spectrum
from .types import SyntheticSpectrum


Continuum = Literal["powerlaw", "cutoff"]


@dataclass(frozen=True, slots=True)
class EvolutionArtifacts:
    """Generated files and numerical results from one animated fit."""

    animation_path: Path
    comparison_path: Path
    summary_path: Path
    checkpoint_path: Path
    ga_result: GARunResult
    scipy_result: SciPyRunResult
    epoch_id: str
    continuum: Continuum
    profile: str
    config_digest: str
    objective_signature: str
    reference: ReferenceSpectrum | None = None

    def summary(self) -> dict[str, Any]:
        reference_summary = None
        if self.reference is not None:
            reference_summary = {
                "label": self.reference.label,
                "source": self.reference.source,
                "points": self.reference.size,
                "notice": self.reference.comparison_notice,
                "input_path": self.reference.metadata.get("input_path"),
                "input_representation": self.reference.metadata.get(
                    "input_representation"
                ),
            }
        return {
            "label": SYNTHETIC_LABEL,
            "profile": self.profile,
            "config_digest": self.config_digest,
            "objective_signature": self.objective_signature,
            "epoch_id": self.epoch_id,
            "continuum": self.continuum,
            "animation": str(self.animation_path),
            "comparison": str(self.comparison_path),
            "summary": str(self.summary_path),
            "checkpoint": str(self.checkpoint_path),
            "ga": self.ga_result.summary(),
            "scipy_polish": self.scipy_result.summary(),
            "reference": reference_summary,
        }


def _digest_array(digest: Any, name: str, values: ArrayLike) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _objective_signature(
    config: CaseStudyConfig,
    response: InstrumentResponse,
    spectrum: SyntheticSpectrum,
    continuum: Continuum,
) -> str:
    """Bind resumable scores to the exact model, response, and count data."""

    digest = hashlib.sha256()
    digest.update(b"evoxrb-spectral-objective-v1")
    digest.update(config.digest.encode("ascii"))
    digest.update(continuum.encode("ascii"))
    digest.update(spectrum.epoch_id.encode("utf-8"))
    digest.update(np.asarray(spectrum.exposure_s, dtype=np.float64).tobytes())
    for name, values in (
        ("counts", spectrum.counts),
        ("fit_mask", spectrum.fit_mask),
        ("detector_edges", spectrum.detector_edges),
        ("response_true_edges", response.true_edges),
        ("true_energy", response.true_energy),
        ("response_detector_edges", response.detector_edges),
        ("effective_area", response.effective_area),
        ("redistribution", response.redistribution),
        ("background", response.background_rate_density),
    ):
        _digest_array(digest, name, values)
    return digest.hexdigest()


def _build_synthetic_target(
    config: CaseStudyConfig,
    epoch_id: str,
) -> tuple[InstrumentResponse, SyntheticSpectrum, int]:
    """Build the requested target in memory from the current configuration."""

    epochs = epoch_truths(config)
    available = {epoch.epoch_id: (index, epoch) for index, epoch in enumerate(epochs)}
    if epoch_id not in available:
        choices = ", ".join(sorted(available))
        raise ValueError(f"unknown epoch {epoch_id!r}; choose one of: {choices}")
    epoch_index, epoch = available[epoch_id]
    response = default_nicer_inspired_response()
    truth = {
        "tin": epoch.tin,
        "ndisk": epoch.ndisk,
        "gamma": epoch.gamma,
        "norm": epoch.powerlaw_norm,
        "nh": epoch.nh,
    }
    spectrum = simulate_spectrum(
        response,
        SpectrumModel("powerlaw", fixed_nh=epoch.nh),
        truth,
        epoch.exposure_s,
        derive_seed(config.master_seed, 10, epoch_index),
        epoch_id=epoch.epoch_id,
        phase=epoch.phase,
        reference_mjd=epoch.reference_mjd,
    )
    return response, spectrum, epoch_index


def _resolve_output(root: Path, value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def run_animated_fit(
    config: CaseStudyConfig,
    *,
    profile: Literal["smoke", "full"] = "smoke",
    epoch_id: str = "E08",
    continuum: Continuum = "powerlaw",
    seed: int | None = None,
    generations: int | None = None,
    population_size: int | None = None,
    output_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    reference_csv: str | Path | None = None,
    population_display: PopulationDisplay = "envelope",
    max_population_curves: int = 16,
    fps: float = 8.0,
    live: bool = False,
    resume: bool = False,
    on_generation: GenerationCallback | None = None,
) -> EvolutionArtifacts:
    """Run one real-valued GA, SciPy-polish it, and save its spectral replay.

    The optional reference CSV is a visual overlay only. It is never passed to
    the objective, because fitting real observations requires their matched
    calibrated response and background products.
    """

    if profile not in ("smoke", "full"):
        raise ValueError("profile must be 'smoke' or 'full'")
    if continuum not in ("powerlaw", "cutoff"):
        raise ValueError("continuum must be 'powerlaw' or 'cutoff'")
    if generations is not None and int(generations) < 1:
        raise ValueError("generations must be positive")
    if population_size is not None and int(population_size) < 4:
        raise ValueError("population_size must be at least four")
    if not math.isfinite(float(fps)) or not 0.0 < float(fps):
        raise ValueError("fps must be finite and positive")
    if population_display not in ("none", "curves", "envelope"):
        raise ValueError("population_display must be 'none', 'curves', or 'envelope'")
    if int(max_population_curves) < 1:
        raise ValueError("max_population_curves must be positive")

    paths = CampaignPaths.from_config(config)
    response, spectrum, epoch_index = _build_synthetic_target(config, epoch_id)

    search_space = search_space_from_config(config)
    objective = Objective(
        spectrum,
        response,
        SpectrumModel(continuum, fixed_nh=float(config.search["nh_fixed"])),
    )
    ga_config = _ga_config(_profile_sections(config, profile)["ga"])
    replacements: dict[str, Any] = {}
    if generations is not None:
        replacements["max_generations"] = int(generations)
    if population_size is not None:
        replacements["population_size"] = int(population_size)
    if replacements:
        ga_config = replace(ga_config, **replacements)

    resolved_seed = (
        int(seed)
        if seed is not None
        else derive_seed(config.master_seed, 70, epoch_index, 0 if continuum == "powerlaw" else 1)
    )
    animation_directory = paths.results / "animations"
    animation_directory.mkdir(parents=True, exist_ok=True)
    requested_animation = _resolve_output(
        paths.root,
        output_path,
        animation_directory
        / f"{epoch_id}_{continuum}_{profile}_seed_{resolved_seed}_ga.html",
    )
    if not requested_animation.suffix:
        requested_animation = requested_animation.with_suffix(".html")
    if requested_animation.suffix.casefold() not in (".html", ".htm", ".gif", ".mp4"):
        raise ValueError("animation output must use .html, .htm, .gif, or .mp4")
    if (
        requested_animation.suffix.casefold() == ".gif"
        and not mpl_animation.writers.is_available("pillow")
    ):
        raise RuntimeError("GIF export requires Matplotlib's Pillow writer")
    if (
        requested_animation.suffix.casefold() == ".mp4"
        and not mpl_animation.writers.is_available("ffmpeg")
    ):
        raise RuntimeError("MP4 export requires an ffmpeg executable")
    requested_comparison = _resolve_output(
        paths.root,
        comparison_path,
        animation_directory
        / f"{epoch_id}_{continuum}_{profile}_seed_{resolved_seed}_comparison.png",
    )
    if not requested_comparison.suffix:
        requested_comparison = requested_comparison.with_suffix(".png")
    if requested_comparison.suffix.casefold() != ".png":
        raise ValueError("best-fit comparison output must use .png")
    resolved_checkpoint = _resolve_output(
        paths.root,
        checkpoint_path,
        paths.checkpoints
        / "animations"
        / profile
        / epoch_id
        / continuum
        / f"seed_{resolved_seed}.npz",
    )
    resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    reference = (
        None
        if reference_csv is None
        else load_reference_spectrum_csv(reference_csv)
    )
    objective_signature = _objective_signature(
        config,
        response,
        spectrum,
        continuum,
    )

    viewer = (
        LiveSpectrumViewer(objective, search_space, reference=reference)
        if live
        else None
    )

    def notify(snapshot: GenerationSnapshot) -> None:
        if on_generation is not None:
            on_generation(snapshot)
        if viewer is not None:
            viewer(snapshot)

    replay = None
    try:
        result = GeneticOptimizer(search_space, ga_config).optimize(
            objective.evaluate,
            seed=resolved_seed,
            checkpoint_path=resolved_checkpoint,
            resume=bool(resume and resolved_checkpoint.exists()),
            on_generation=notify if on_generation is not None or viewer is not None else None,
            objective_signature=objective_signature,
        )
        polished = local_polish(objective.evaluate, search_space, result)
        interval_ms = max(1, int(round(1000.0 / float(fps))))
        replay = create_spectral_animation(
            result,
            objective,
            search_space,
            reference=reference,
            population_display=population_display,
            max_population_curves=int(max_population_curves),
            interval_ms=interval_ms,
        )
        saved_animation = save_spectral_animation(
            replay,
            requested_animation,
            fps=float(fps),
        )
        if polished.success:
            polished_label = "GA + SciPy L-BFGS-B"
        elif math.isclose(polished.score, polished.start_score):
            polished_label = "Raw GA retained after SciPy attempt"
        else:
            polished_label = "GA + SciPy best attempt (not converged)"
        saved_comparison = plot_best_fit_comparison(
            result,
            objective,
            search_space,
            requested_comparison,
            reference=reference,
            polished_parameters=polished.parameters,
            polished_score=polished.score,
            polished_label=polished_label,
        )
        summary_path = saved_animation.with_name(f"{saved_animation.stem}.summary.json")
        artifacts = EvolutionArtifacts(
            animation_path=saved_animation,
            comparison_path=saved_comparison,
            summary_path=summary_path,
            checkpoint_path=resolved_checkpoint,
            ga_result=result,
            scipy_result=polished,
            epoch_id=epoch_id,
            continuum=continuum,
            profile=profile,
            config_digest=config.digest,
            objective_signature=objective_signature,
            reference=reference,
        )
        atomic_write_json(summary_path, artifacts.summary())
        if viewer is not None:
            viewer.show(block=True)
        return artifacts
    finally:
        if replay is not None:
            plt.close(replay._fig)  # type: ignore[attr-defined]
        if viewer is not None:
            viewer.close()


# Stage-style spelling matches the other CLI orchestration helpers.
animate_stage = run_animated_fit


__all__ = ["EvolutionArtifacts", "animate_stage", "run_animated_fit"]
