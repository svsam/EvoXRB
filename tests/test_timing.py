"""Fast tests for the pure-NumPy/SciPy synthetic timing subsystem."""

from __future__ import annotations

import numpy as np

from evoxrb.simulation import EPOCH_BY_ID
from evoxrb.timing import (
    DEFAULT_DT_S,
    TimingSettings,
    averaged_periodogram,
    epoch_fractional_rms,
    fit_power_spectrum,
    simulate_light_curves,
    simulate_timing_epoch,
)


def test_fixed_fractional_rms_evolution_is_exact() -> None:
    expected = (0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.15, 0.10, 0.03, 0.12, 0.20)
    actual = tuple(epoch_fractional_rms(f"E{index:02d}") for index in range(1, 13))
    assert actual == expected


def test_seeded_light_curves_are_deterministic_and_consistent() -> None:
    first = simulate_light_curves(profile="smoke", qpo_frequency_hz=0.5, seed=1729)
    repeat = simulate_light_curves(profile="smoke", qpo_frequency_hz=0.5, seed=1729)
    different = simulate_light_curves(
        profile="smoke", qpo_frequency_hz=0.5, seed=1730
    )

    np.testing.assert_array_equal(first.soft_counts, repeat.soft_counts)
    np.testing.assert_array_equal(first.hard_counts, repeat.hard_counts)
    np.testing.assert_array_equal(first.total_counts, repeat.total_counts)
    assert not np.array_equal(first.total_counts, different.total_counts)
    np.testing.assert_array_equal(
        first.total_counts, first.soft_counts + first.hard_counts
    )
    assert np.all(first.total_counts >= 0)
    assert first.dt_s == DEFAULT_DT_S
    assert first.segment_s == TimingSettings.for_profile("smoke").segment_s
    assert first.metadata["minimum_model_rate_hz"] > 0.0


def test_injected_qpo_centroid_is_recovered_within_one_fwhm() -> None:
    injected_hz = 0.5
    light_curves = simulate_light_curves(
        profile="smoke",
        fractional_rms=0.24,
        qpo_frequency_hz=injected_hz,
        qpo_fractional_rms=0.12,
        seed=123,
    )
    periodogram = averaged_periodogram(
        light_curves.total_counts,
        dt_s=light_curves.dt_s,
        segment_s=light_curves.segment_s,
        normalization="fractional_rms",
        max_frequency_hz=32.0,
        frequency_bins=256,
    )
    result = fit_power_spectrum(
        periodogram,
        expected_qpo_hz=injected_hz,
        n_bootstrap=8,
        seed=456,
    )

    injected_fwhm = float(light_curves.metadata["qpo_fwhm_hz"])
    assert result.fit_success
    assert result.centroid_hz is not None
    assert abs(result.centroid_hz - injected_hz) <= injected_fwhm
    assert result.detected
    assert result.classification == "type-C-like"
    assert result.q_factor is not None and result.q_factor >= 2.0
    assert result.amplitude_significance >= 3.0


def test_soft_state_e10_has_no_qpo_detection() -> None:
    result = simulate_timing_epoch(EPOCH_BY_ID["E10"], profile="smoke", seed=1820070)

    assert result.epoch_id == "E10"
    assert result.injected_qpo_hz is None
    assert result.fit_success
    assert not result.detected
    assert result.classification is None
    assert result.centroid_hz is None
    assert result.q_factor is None
    assert result.qpo_amplitude == 0.0


def test_periodogram_normalizations_have_expected_poisson_scaling() -> None:
    settings = TimingSettings.for_profile("smoke")
    samples_per_segment = round(settings.segment_s / settings.dt_s)
    generator = np.random.default_rng(909)
    one_segment = generator.poisson(4.0, size=samples_per_segment)
    # Tiling makes every segment mean identical, so the algebraic conversion
    # between Leahy and fractional-rms normalization is exact.
    counts = np.tile(one_segment, 4)

    leahy = averaged_periodogram(
        counts,
        dt_s=settings.dt_s,
        segment_s=settings.segment_s,
        normalization="leahy",
        max_frequency_hz=32.0,
        frequency_bins=192,
    )
    fractional = averaged_periodogram(
        counts,
        dt_s=settings.dt_s,
        segment_s=settings.segment_s,
        normalization="fractional_rms",
        max_frequency_hz=32.0,
        frequency_bins=192,
    )

    np.testing.assert_array_equal(leahy.frequencies_hz, fractional.frequencies_hz)
    conversion = settings.dt_s / float(np.mean(one_segment))
    np.testing.assert_allclose(
        fractional.power, leahy.power * conversion, rtol=2e-14, atol=1e-15
    )
    # Weighting by the number of raw Fourier ordinates reverses logarithmic
    # rebinning; pure Poisson noise has expectation two in Leahy normalization.
    weighted_white_level = float(np.average(leahy.power, weights=leahy.group_sizes))
    assert 1.9 < weighted_white_level < 2.1
    assert leahy.n_segments == fractional.n_segments == 4
    assert np.all(leahy.power_error > 0.0)
    assert np.all(fractional.power_error > 0.0)
