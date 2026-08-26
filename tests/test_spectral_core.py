"""Focused tests for the synthetic spectral and response core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evoxrb.instrument import (
    AREA_KNOT_CM2,
    AREA_KNOT_ENERGY_KEV,
    InstrumentResponse,
    nicer_inspired_effective_area,
)
from evoxrb.models import SpectrumModel
from evoxrb.simulation import OUTBURST_EPOCHS, load_spectrum, save_spectrum, simulate_epoch
from evoxrb.types import SYNTHETIC_LABEL


@pytest.fixture(scope="module")
def response() -> InstrumentResponse:
    return InstrumentResponse.default()


def test_educational_models_are_finite_and_positive() -> None:
    energy = np.geomspace(0.2, 12.0, 300)
    parameters = {"Tin": 0.55, "Ndisk": 9_000.0, "Gamma": 1.95, "K": 0.32}

    primary = SpectrumModel(continuum="powerlaw", fixed_nh=0.15)
    cutoff = SpectrumModel(continuum="cutoff", fixed_nh=0.15)
    primary_flux = primary.evaluate(energy, parameters)
    cutoff_flux = cutoff.evaluate(energy, parameters)

    assert np.all(np.isfinite(primary_flux))
    assert np.all(primary_flux > 0.0)
    assert np.all(np.isfinite(cutoff_flux))
    assert np.all(cutoff_flux > 0.0)
    assert np.all(cutoff_flux < primary_flux)
    assert primary.name.startswith(SYNTHETIC_LABEL)

    free_nh_flux = SpectrumModel(fixed_nh=None).evaluate(
        energy, {**parameters, "NH": 0.25}
    )
    assert np.all(np.isfinite(free_nh_flux))
    assert np.all(free_nh_flux > 0.0)


def test_default_response_normalization_and_anchors(
    response: InstrumentResponse,
) -> None:
    assert response.true_energy.shape == (600,)
    assert response.detector_energy.shape == (236,)
    assert response.redistribution.shape == (236, 600)
    np.testing.assert_allclose(response.redistribution.sum(axis=0), 1.0, atol=2e-13)
    np.testing.assert_allclose(
        nicer_inspired_effective_area(AREA_KNOT_ENERGY_KEV), AREA_KNOT_CM2
    )
    assert response.fit_mask.sum() == 190
    assert response.label.startswith(SYNTHETIC_LABEL)
    energy = response.detector_energy
    expected_background = 0.02 + 0.005 * energy + 0.03 * np.exp(
        -np.square(energy - 8.0) / (2.0 * 0.5**2)
    )
    np.testing.assert_allclose(response.background_rate_density, expected_background)


def test_response_folding_matches_explicit_matrix_expression(
    response: InstrumentResponse,
) -> None:
    exposure_s = 37.5
    photon_flux = np.linspace(0.01, 0.03, response.true_energy.size)
    source, background = response.fold_components(photon_flux, exposure_s)
    manual_source = exposure_s * (
        response.redistribution
        @ (response.effective_area * photon_flux * np.diff(response.true_edges))
    )
    manual_background = (
        exposure_s
        * response.background_rate_density
        * np.diff(response.detector_edges)
    )

    np.testing.assert_allclose(source, manual_source, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(background, manual_background, rtol=1e-14)
    np.testing.assert_allclose(response.fold(photon_flux, exposure_s), source + background)
    assert np.all(source >= 0.0)
    assert np.all(background > 0.0)


def test_seeded_poisson_simulation_is_deterministic(
    response: InstrumentResponse,
) -> None:
    first = simulate_epoch("E08", response, seed=1729)
    repeat = simulate_epoch("E08", response, seed=1729)
    different = simulate_epoch("E08", response, seed=1730)

    np.testing.assert_array_equal(first.counts, repeat.counts)
    np.testing.assert_array_equal(first.expected_counts, repeat.expected_counts)
    assert not np.array_equal(first.counts, different.counts)
    assert first.seed == repeat.seed == 1729
    assert first.epoch_id == "E08"
    assert first.label.startswith(SYNTHETIC_LABEL)


def test_spectrum_npz_round_trip(
    response: InstrumentResponse, tmp_path: Path
) -> None:
    original = simulate_epoch("E10", response, seed=991)
    destination = save_spectrum(tmp_path / "synthetic_e10.npz", original)
    restored = load_spectrum(destination)

    assert restored.epoch_id == original.epoch_id
    assert restored.phase == original.phase
    assert restored.reference_mjd == original.reference_mjd
    assert restored.exposure_s == original.exposure_s
    assert restored.seed == original.seed
    assert restored.truth_parameters == original.truth_parameters
    assert restored.truth_model == original.truth_model
    assert restored.label == original.label
    for name in (
        "detector_energy",
        "detector_edges",
        "counts",
        "expected_counts",
        "source_expected_counts",
        "background_expected_counts",
        "fit_mask",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))


def test_fixed_outburst_table_is_exact() -> None:
    expected = (
        ("E01", "hard rise", 58193.2, 0.20, 15000, 1.55, 0.35, 0.05),
        ("E02", "hard plateau", 58210.0, 0.24, 14000, 1.60, 0.45, 0.10),
        ("E03", "hard plateau", 58235.3, 0.28, 13000, 1.65, 0.50, 0.20),
        ("E04", "hard plateau", 58259.1, 0.32, 12000, 1.70, 0.48, 0.40),
        ("E05", "hard decline", 58275.4, 0.38, 11000, 1.68, 0.40, 0.80),
        ("E06", "hard decline", 58289.1, 0.45, 10000, 1.75, 0.34, 1.50),
        ("E07", "intermediate", 58297.2, 0.55, 9000, 1.95, 0.32, 3.00),
        ("E08", "intermediate", 58302.1, 0.65, 8000, 2.15, 0.27, 5.00),
        ("E09", "intermediate", 58304.3, 0.75, 7000, 2.35, 0.20, 8.00),
        ("E10", "soft", 58330.1, 0.70, 7500, 2.40, 0.06, None),
        ("E11", "decay intermediate", 58390.0, 0.45, 9000, 2.00, 0.12, 0.50),
        ("E12", "return hard", 58403.1, 0.30, 11000, 1.70, 0.20, 0.20),
    )
    actual = tuple(
        (
            item.epoch_id,
            item.phase,
            item.reference_mjd,
            item.tin,
            item.ndisk,
            item.gamma,
            item.powerlaw_norm,
            item.qpo_hz,
        )
        for item in OUTBURST_EPOCHS
    )

    assert actual == expected
    assert all(item.exposure_s == 2048.0 for item in OUTBURST_EPOCHS)
    assert all(item.nh == 0.15 for item in OUTBURST_EPOCHS)
    assert all(item.label.startswith(SYNTHETIC_LABEL) for item in OUTBURST_EPOCHS)
