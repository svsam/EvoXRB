"""Deterministic simulation utilities and the fixed synthetic outburst."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .instrument import InstrumentResponse, default_nicer_inspired_response
from .models import ContinuumKind, SpectrumModel
from .types import EpochTruth, SYNTHETIC_LABEL, SyntheticSpectrum


MASTER_SEED = 1_820_070


OUTBURST_EPOCHS: tuple[EpochTruth, ...] = (
    EpochTruth("E01", "hard rise", 58193.2, 0.20, 15000, 1.55, 0.35, 0.05),
    EpochTruth("E02", "hard plateau", 58210.0, 0.24, 14000, 1.60, 0.45, 0.10),
    EpochTruth("E03", "hard plateau", 58235.3, 0.28, 13000, 1.65, 0.50, 0.20),
    EpochTruth("E04", "hard plateau", 58259.1, 0.32, 12000, 1.70, 0.48, 0.40),
    EpochTruth("E05", "hard decline", 58275.4, 0.38, 11000, 1.68, 0.40, 0.80),
    EpochTruth("E06", "hard decline", 58289.1, 0.45, 10000, 1.75, 0.34, 1.50),
    EpochTruth("E07", "intermediate", 58297.2, 0.55, 9000, 1.95, 0.32, 3.00),
    EpochTruth("E08", "intermediate", 58302.1, 0.65, 8000, 2.15, 0.27, 5.00),
    EpochTruth("E09", "intermediate", 58304.3, 0.75, 7000, 2.35, 0.20, 8.00),
    EpochTruth("E10", "soft", 58330.1, 0.70, 7500, 2.40, 0.06, None),
    EpochTruth(
        "E11", "decay intermediate", 58390.0, 0.45, 9000, 2.00, 0.12, 0.50
    ),
    EpochTruth("E12", "return hard", 58403.1, 0.30, 11000, 1.70, 0.20, 0.20),
)

EPOCH_BY_ID: dict[str, EpochTruth] = {
    epoch.epoch_id: epoch for epoch in OUTBURST_EPOCHS
}


def derive_seed(master_seed: int, *spawn_key: int) -> int:
    """Derive a stable uint64 seed without relying on process-randomized hashes."""

    if int(master_seed) < 0 or any(int(item) < 0 for item in spawn_key):
        raise ValueError("seed and spawn-key values must be non-negative integers")
    sequence = np.random.SeedSequence(
        int(master_seed), spawn_key=tuple(int(item) for item in spawn_key)
    )
    words = sequence.generate_state(2, dtype=np.uint32)
    return int(words[0]) | (int(words[1]) << 32)


def _rng_and_seed(
    seed: int | np.random.SeedSequence | np.random.Generator,
) -> tuple[np.random.Generator, int]:
    if isinstance(seed, np.random.Generator):
        # A generator's state need not expose a portable original seed.  Draw a
        # child seed and record it, making the simulated spectrum replayable.
        recorded_seed = int(seed.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        return np.random.default_rng(recorded_seed), recorded_seed
    if isinstance(seed, np.random.SeedSequence):
        words = seed.generate_state(2, dtype=np.uint32)
        recorded_seed = int(words[0]) | (int(words[1]) << 32)
        return np.random.default_rng(recorded_seed), recorded_seed
    recorded_seed = int(seed)
    if recorded_seed < 0:
        raise ValueError("seed must be non-negative")
    return np.random.default_rng(recorded_seed), recorded_seed


def simulate_spectrum(
    response: InstrumentResponse,
    model: SpectrumModel,
    parameters: Mapping[str, float],
    exposure_s: float,
    seed: int | np.random.SeedSequence | np.random.Generator,
    *,
    epoch_id: str = "custom",
    phase: str = "synthetic",
    reference_mjd: float | None = None,
) -> SyntheticSpectrum:
    """Fold a photon model and draw reproducible Poisson detector counts."""

    flux = model.evaluate(response.true_energy, parameters)
    source_counts, background_counts = response.fold_components(flux, exposure_s)
    expected_counts = source_counts + background_counts
    rng, recorded_seed = _rng_and_seed(seed)
    counts = rng.poisson(expected_counts).astype(np.int64, copy=False)
    return SyntheticSpectrum(
        detector_energy=response.detector_energy.copy(),
        detector_edges=response.detector_edges.copy(),
        counts=counts,
        expected_counts=expected_counts,
        source_expected_counts=source_counts,
        background_expected_counts=background_counts,
        fit_mask=response.fit_mask,
        exposure_s=float(exposure_s),
        truth_parameters=dict(parameters),
        seed=recorded_seed,
        epoch_id=str(epoch_id),
        phase=str(phase),
        reference_mjd=reference_mjd,
        truth_model=model.name,
        label=f"{SYNTHETIC_LABEL} Poisson spectrum",
    )


def simulate_epoch(
    epoch: EpochTruth | str,
    response: InstrumentResponse | None = None,
    *,
    continuum: ContinuumKind = "powerlaw",
    seed: int | np.random.SeedSequence | np.random.Generator | None = None,
    master_seed: int = MASTER_SEED,
    stream: int = 0,
) -> SyntheticSpectrum:
    """Simulate one of the twelve fixed epochs.

    Passing an epoch ID uses :data:`OUTBURST_EPOCHS`.  With no explicit seed,
    the seed is derived from the master seed, epoch index, and stream number.
    """

    truth = EPOCH_BY_ID[epoch] if isinstance(epoch, str) else epoch
    instrument = response if response is not None else default_nicer_inspired_response()
    if seed is None:
        try:
            epoch_index = OUTBURST_EPOCHS.index(truth)
        except ValueError:
            digits = "".join(character for character in truth.epoch_id if character.isdigit())
            epoch_index = int(digits) if digits else 0
        seed = derive_seed(master_seed, epoch_index, int(stream))
    model = SpectrumModel(continuum=continuum, fixed_nh=truth.nh)
    return simulate_spectrum(
        instrument,
        model,
        truth.parameters,
        truth.exposure_s,
        seed,
        epoch_id=truth.epoch_id,
        phase=truth.phase,
        reference_mjd=truth.reference_mjd,
    )


def simulate_outburst(
    response: InstrumentResponse | None = None,
    *,
    continuum: ContinuumKind = "powerlaw",
    master_seed: int = MASTER_SEED,
    stream: int = 0,
    epochs: Iterable[EpochTruth] = OUTBURST_EPOCHS,
) -> list[SyntheticSpectrum]:
    """Simulate a deterministic collection of synthetic outburst epochs."""

    instrument = response if response is not None else default_nicer_inspired_response()
    return [
        simulate_epoch(
            epoch,
            instrument,
            continuum=continuum,
            seed=derive_seed(master_seed, index, int(stream)),
        )
        for index, epoch in enumerate(epochs)
    ]


def save_spectrum(path: str | Path, spectrum: SyntheticSpectrum) -> Path:
    """Save a spectrum as a compressed, pickle-free NPZ archive."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **spectrum.to_npz_dict())
    return destination


def load_spectrum(path: str | Path) -> SyntheticSpectrum:
    """Load a spectrum written by :func:`save_spectrum`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        payload: dict[str, Any] = {name: archive[name] for name in archive.files}
    return SyntheticSpectrum.from_npz_dict(payload)
