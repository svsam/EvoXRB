"""Pure-Python synthetic timing analysis for the EvoXRB case study.

This module intentionally depends only on NumPy and SciPy.  It creates
NICER-inspired *synthetic* light curves; it does not read mission products and
must not be interpreted as a calibration or an analysis of real observations.

The simulation uses Timmer--Koenig-style Fourier-domain Gaussian processes for
the broad-band noise and (when requested) a Lorentzian quasi-periodic
oscillation (QPO).  The latent Gaussian variability is mapped to a positive
rate with a log-normal transform before Poisson counts are drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Normalization = Literal["fractional_rms", "leahy"]

DEFAULT_DT_S = 1.0 / 512.0
DEFAULT_SEGMENT_S = 512.0
DEFAULT_DURATION_S = 2048.0
DEFAULT_BOOTSTRAPS = 200

# Fixed, deterministic variability evolution used by the twelve-epoch case
# study.  The soft epoch is deliberately quiet and has no injected QPO.
OUTBURST_FRACTIONAL_RMS: Mapping[str, float] = {
    "E01": 0.30,
    "E02": 0.28,
    "E03": 0.26,
    "E04": 0.24,
    "E05": 0.22,
    "E06": 0.20,
    "E07": 0.18,
    "E08": 0.15,
    "E09": 0.10,
    "E10": 0.03,
    "E11": 0.12,
    "E12": 0.20,
}

# These count rates control only the scale of the timing demonstration.  They
# are not measured NICER rates.  Spectral simulations may pass their own rates.
OUTBURST_TOTAL_RATE_HZ: Mapping[str, float] = {
    "E01": 1_400.0,
    "E02": 1_750.0,
    "E03": 2_050.0,
    "E04": 2_300.0,
    "E05": 2_200.0,
    "E06": 2_100.0,
    "E07": 2_450.0,
    "E08": 2_750.0,
    "E09": 2_950.0,
    "E10": 2_700.0,
    "E11": 1_450.0,
    "E12": 950.0,
}


@dataclass(slots=True, frozen=True)
class TimingSettings:
    """Numerical settings for a timing run.

    The full profile follows the case-study design exactly.  The smoke profile
    retains the 1/512-s sampling but uses 128-s segments, four segments, fewer
    frequency bins, and fewer bootstrap fits so CI remains quick.
    """

    dt_s: float = DEFAULT_DT_S
    segment_s: float = DEFAULT_SEGMENT_S
    duration_s: float = DEFAULT_DURATION_S
    max_frequency_hz: float = 64.0
    frequency_bins: int = 512
    bootstraps: int = DEFAULT_BOOTSTRAPS
    profile: str = "full"

    @classmethod
    def for_profile(cls, profile: str = "full") -> "TimingSettings":
        profile = profile.lower()
        if profile == "full":
            return cls()
        if profile == "smoke":
            return cls(
                segment_s=128.0,
                duration_s=512.0,
                max_frequency_hz=32.0,
                frequency_bins=256,
                bootstraps=24,
                profile="smoke",
            )
        raise ValueError("profile must be 'smoke' or 'full'")

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if not np.isfinite(self.segment_s) or self.segment_s <= self.dt_s:
            raise ValueError("segment_s must be larger than dt_s")
        if not np.isfinite(self.duration_s) or self.duration_s < self.segment_s:
            raise ValueError("duration_s must contain at least one segment")
        if self.max_frequency_hz <= 0.0:
            raise ValueError("max_frequency_hz must be positive")
        if self.frequency_bins < 16:
            raise ValueError("frequency_bins must be at least 16")
        if self.bootstraps < 0:
            raise ValueError("bootstraps cannot be negative")


@dataclass(slots=True)
class LightCurveBands:
    """Poisson light curves in synthetic soft, hard, and total bands."""

    time_s: FloatArray
    soft_counts: IntArray
    hard_counts: IntArray
    total_counts: IntArray
    dt_s: float
    segment_s: float
    soft_rate_hz: float
    hard_rate_hz: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        size = self.time_s.size
        if any(
            values.ndim != 1 or values.size != size
            for values in (self.soft_counts, self.hard_counts, self.total_counts)
        ):
            raise ValueError("all light-curve arrays must be one-dimensional and equal length")
        if np.any(self.soft_counts < 0) or np.any(self.hard_counts < 0):
            raise ValueError("Poisson counts cannot be negative")
        if not np.array_equal(self.total_counts, self.soft_counts + self.hard_counts):
            raise ValueError("total_counts must equal soft_counts + hard_counts")

    def counts_for(self, band: Literal["soft", "hard", "total"] = "total") -> IntArray:
        """Return counts for one named energy band."""

        if band == "soft":
            return self.soft_counts
        if band == "hard":
            return self.hard_counts
        if band == "total":
            return self.total_counts
        raise ValueError("band must be 'soft', 'hard', or 'total'")

    @property
    def hardness(self) -> float:
        """Hard/soft count ratio, with a finite result for an empty soft band."""

        soft = float(np.sum(self.soft_counts, dtype=np.float64))
        hard = float(np.sum(self.hard_counts, dtype=np.float64))
        return hard / soft if soft > 0.0 else float("nan")


@dataclass(slots=True)
class AveragedPeriodogram:
    """A rebinned average of equal-duration segment periodograms."""

    frequencies_hz: FloatArray
    power: FloatArray
    power_error: FloatArray
    segment_powers: FloatArray
    group_sizes: IntArray
    normalization: Normalization
    n_segments: int
    segment_s: float
    dt_s: float
    mean_rate_hz: float

    @property
    def frequency_hz(self) -> FloatArray:
        """Singular-name compatibility alias."""

        return self.frequencies_hz


@dataclass(slots=True)
class TimingResult:
    """Result of a synthetic QPO search and Lorentzian fit."""

    frequencies_hz: FloatArray
    power: FloatArray
    power_error: FloatArray
    model_power: FloatArray
    normalization: Normalization
    band: str
    n_segments: int
    centroid_hz: float | None
    centroid_error_hz: float | None
    fwhm_hz: float | None
    fwhm_error_hz: float | None
    q_factor: float | None
    qpo_amplitude: float
    qpo_amplitude_error: float | None
    amplitude_significance: float
    detected: bool
    classification: str | None
    fit_success: bool
    bootstrap_successes: int
    fit_parameters: dict[str, float]
    epoch_id: str | None = None
    injected_qpo_hz: float | None = None
    injected_fractional_rms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequency_hz(self) -> FloatArray:
        """Singular-name compatibility alias."""

        return self.frequencies_hz

    @property
    def qpo_frequency_hz(self) -> float | None:
        """Alias used by reporting code."""

        return self.centroid_hz

    @property
    def significance(self) -> float:
        """QPO integrated-amplitude/bootstrap-error ratio."""

        return self.amplitude_significance

    def to_record(self) -> dict[str, Any]:
        """Return a flat, JSON/CSV-friendly summary (arrays are omitted)."""

        return {
            "epoch_id": self.epoch_id,
            "band": self.band,
            "normalization": self.normalization,
            "n_segments": self.n_segments,
            "centroid_hz": self.centroid_hz,
            "centroid_error_hz": self.centroid_error_hz,
            "fwhm_hz": self.fwhm_hz,
            "fwhm_error_hz": self.fwhm_error_hz,
            "q_factor": self.q_factor,
            "qpo_amplitude": self.qpo_amplitude,
            "qpo_amplitude_error": self.qpo_amplitude_error,
            "amplitude_significance": self.amplitude_significance,
            "detected": self.detected,
            "classification": self.classification,
            "fit_success": self.fit_success,
            "bootstrap_successes": self.bootstrap_successes,
            "injected_qpo_hz": self.injected_qpo_hz,
            "injected_fractional_rms": self.injected_fractional_rms,
            **self.fit_parameters,
        }


def _coerce_rng(
    rng: np.random.Generator | None = None,
    seed: int | np.random.SeedSequence | None = None,
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    return rng if rng is not None else np.random.default_rng(seed)


def _unit_lorentzian(
    frequencies_hz: FloatArray,
    centroid_hz: float,
    fwhm_hz: float,
) -> FloatArray:
    """One-sided Lorentzian normalized to unit integral on [0, infinity)."""

    half_width = max(0.5 * float(fwhm_hz), np.finfo(float).tiny)
    centroid = max(float(centroid_hz), 0.0)
    normalization = np.pi / 2.0 + np.arctan(centroid / half_width)
    return half_width / (
        normalization * ((frequencies_hz - centroid) ** 2 + half_width**2)
    )


def lorentzian_power(
    frequencies_hz: ArrayLike,
    amplitude: float,
    centroid_hz: float,
    fwhm_hz: float,
) -> FloatArray:
    """Return a one-sided Lorentzian with integrated ``amplitude``.

    In fractional-rms normalization, ``amplitude`` is the component's squared
    fractional rms.  It is merely an integrated power in Leahy normalization.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    return max(float(amplitude), 0.0) * _unit_lorentzian(
        frequencies, centroid_hz, fwhm_hz
    )


def _timmer_koenig_process(
    n_samples: int,
    dt_s: float,
    psd_shape: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw a zero-mean, unit-variance Fourier-domain Gaussian process."""

    expected = n_samples // 2 + 1
    if psd_shape.shape != (expected,):
        raise ValueError("psd_shape does not match the real-FFT frequency grid")
    shape = np.maximum(np.asarray(psd_shape, dtype=np.float64), 0.0)
    coefficients = (
        rng.normal(size=expected) + 1j * rng.normal(size=expected)
    ) * np.sqrt(shape / 2.0)
    coefficients[0] = 0.0
    if n_samples % 2 == 0:
        coefficients[-1] = rng.normal() * np.sqrt(shape[-1])
    process = np.fft.irfft(coefficients, n=n_samples)
    process -= float(np.mean(process))
    scale = float(np.std(process))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("the requested PSD produced zero Fourier variance")
    return np.asarray(process / scale, dtype=np.float64)


def simulate_light_curves(
    *,
    soft_rate_hz: float = 800.0,
    hard_rate_hz: float = 1_200.0,
    fractional_rms: float = 0.24,
    qpo_frequency_hz: float | None = 0.5,
    qpo_fractional_rms: float | None = None,
    qpo_quality: float = 8.0,
    broadband_fwhm_hz: float = 0.35,
    dt_s: float | None = None,
    segment_s: float | None = None,
    duration_s: float | None = None,
    profile: str = "full",
    soft_rms_scale: float = 0.85,
    hard_rms_scale: float = 1.10,
    rng: np.random.Generator | None = None,
    seed: int | np.random.SeedSequence | None = None,
) -> LightCurveBands:
    """Simulate positive-rate, Poisson soft/hard/total light curves.

    Parameters use count rates rather than a detector response so this routine
    can also be tested independently.  ``qpo_frequency_hz=None`` produces a
    broad-band-only soft-state realization.  A duration is truncated to an
    integer number of complete segments.
    """

    settings = TimingSettings.for_profile(profile)
    dt = settings.dt_s if dt_s is None else float(dt_s)
    segment = settings.segment_s if segment_s is None else float(segment_s)
    duration = settings.duration_s if duration_s is None else float(duration_s)
    values = np.asarray(
        [soft_rate_hz, hard_rate_hz, fractional_rms, broadband_fwhm_hz, dt, segment, duration],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("simulation inputs must be finite")
    if soft_rate_hz <= 0.0 or hard_rate_hz <= 0.0:
        raise ValueError("soft and hard count rates must be positive")
    if not 0.0 <= fractional_rms < 2.0:
        raise ValueError("fractional_rms must lie in [0, 2)")
    if dt <= 0.0 or segment <= dt or duration < segment:
        raise ValueError("require 0 < dt_s < segment_s <= duration_s")
    if broadband_fwhm_hz <= 0.0:
        raise ValueError("broadband_fwhm_hz must be positive")
    if soft_rms_scale < 0.0 or hard_rms_scale < 0.0:
        raise ValueError("band rms scales cannot be negative")

    samples_per_segment = int(round(segment / dt))
    if samples_per_segment < 16:
        raise ValueError("a segment must contain at least 16 samples")
    actual_segment = samples_per_segment * dt
    n_segments = int(np.floor(duration / actual_segment + 1e-12))
    if n_segments < 1:
        raise ValueError("duration_s does not contain a complete segment")
    n_samples = n_segments * samples_per_segment
    actual_duration = n_samples * dt
    nyquist = 0.5 / dt

    qpo_frequency: float | None
    if qpo_frequency_hz is None:
        qpo_frequency = None
    else:
        qpo_frequency = float(qpo_frequency_hz)
        if not np.isfinite(qpo_frequency) or not 0.0 < qpo_frequency < nyquist:
            raise ValueError("qpo_frequency_hz must lie between zero and Nyquist")
        if not np.isfinite(qpo_quality) or qpo_quality <= 0.0:
            raise ValueError("qpo_quality must be positive")

    if qpo_fractional_rms is None:
        qpo_rms = (
            min(fractional_rms, 0.10, max(0.025, 0.38 * fractional_rms))
            if qpo_frequency is not None and fractional_rms > 0.0
            else 0.0
        )
    else:
        qpo_rms = float(qpo_fractional_rms)
    if not np.isfinite(qpo_rms) or qpo_rms < 0.0:
        raise ValueError("qpo_fractional_rms must be non-negative")
    if qpo_frequency is None and qpo_rms > 0.0:
        raise ValueError("qpo_fractional_rms requires qpo_frequency_hz")
    if qpo_rms > fractional_rms + 1e-15:
        raise ValueError("qpo_fractional_rms cannot exceed total fractional_rms")

    generator = _coerce_rng(rng, seed)
    frequencies = np.fft.rfftfreq(n_samples, d=dt)
    broad_shape = _unit_lorentzian(frequencies, 0.0, broadband_fwhm_hz)
    broad_process = _timmer_koenig_process(
        n_samples, dt, broad_shape, generator
    )

    qpo_process = np.zeros(n_samples, dtype=np.float64)
    qpo_fwhm: float | None = None
    if qpo_frequency is not None and qpo_rms > 0.0:
        # The resolution floor prevents an unresolved line, especially in the
        # intentionally shortened smoke profile.
        qpo_fwhm = max(qpo_frequency / qpo_quality, 2.0 / actual_segment)
        qpo_shape = _unit_lorentzian(frequencies, qpo_frequency, qpo_fwhm)
        qpo_process = _timmer_koenig_process(
            n_samples, dt, qpo_shape, generator
        )

    if fractional_rms == 0.0:
        latent = np.zeros(n_samples, dtype=np.float64)
        broad_rms = 0.0
        log_variance = 0.0
    else:
        broad_rms = float(np.sqrt(max(fractional_rms**2 - qpo_rms**2, 0.0)))
        log_variance = float(np.log1p(fractional_rms**2))
        broad_fraction = broad_rms**2 / fractional_rms**2
        qpo_fraction = qpo_rms**2 / fractional_rms**2
        latent = (
            np.sqrt(log_variance * broad_fraction) * broad_process
            + np.sqrt(log_variance * qpo_fraction) * qpo_process
        )

    def positive_rate(mean_rate: float, scale: float) -> FloatArray:
        band_log = scale * latent
        # Subtracting half the sample variance approximately preserves the
        # requested mean; the final normalization makes it exact for this draw.
        rate = np.exp(np.clip(band_log - 0.5 * np.var(band_log), -40.0, 40.0))
        rate *= mean_rate / float(np.mean(rate))
        return np.maximum(rate, np.finfo(float).tiny)

    soft_model_rate = positive_rate(float(soft_rate_hz), float(soft_rms_scale))
    hard_model_rate = positive_rate(float(hard_rate_hz), float(hard_rms_scale))
    soft_counts = generator.poisson(soft_model_rate * dt).astype(np.int64, copy=False)
    hard_counts = generator.poisson(hard_model_rate * dt).astype(np.int64, copy=False)
    total_counts = soft_counts + hard_counts
    time_s = np.arange(n_samples, dtype=np.float64) * dt

    return LightCurveBands(
        time_s=time_s,
        soft_counts=soft_counts,
        hard_counts=hard_counts,
        total_counts=total_counts,
        dt_s=dt,
        segment_s=actual_segment,
        soft_rate_hz=float(soft_rate_hz),
        hard_rate_hz=float(hard_rate_hz),
        metadata={
            "label": "Synthetic / NICER-inspired",
            "profile": profile,
            "duration_s": actual_duration,
            "n_segments": n_segments,
            "fractional_rms": float(fractional_rms),
            "broadband_fractional_rms": broad_rms,
            "qpo_fractional_rms": qpo_rms,
            "qpo_frequency_hz": qpo_frequency,
            "qpo_fwhm_hz": qpo_fwhm,
            "minimum_model_rate_hz": float(
                min(np.min(soft_model_rate), np.min(hard_model_rate))
            ),
        },
    )


def _geometric_groups(size: int, requested_bins: int) -> list[tuple[int, int]]:
    if size < 1:
        return []
    if requested_bins >= size:
        return [(index, index + 1) for index in range(size)]
    # Integerized geometric edges keep the lowest Fourier frequencies at their
    # native resolution while progressively averaging the noisy high-frequency
    # tail.  Duplicate integer edges are removed.
    edges = np.rint(np.geomspace(1.0, float(size + 1), requested_bins + 1) - 1.0)
    edges = np.unique(np.clip(edges.astype(np.int64), 0, size))
    if edges[0] != 0:
        edges = np.insert(edges, 0, 0)
    if edges[-1] != size:
        edges = np.append(edges, size)
    return [
        (int(left), int(right))
        for left, right in zip(edges[:-1], edges[1:], strict=True)
        if right > left
    ]


def averaged_periodogram(
    counts: ArrayLike,
    *,
    dt_s: float = DEFAULT_DT_S,
    segment_s: float = DEFAULT_SEGMENT_S,
    normalization: Normalization = "fractional_rms",
    max_frequency_hz: float | None = 64.0,
    frequency_bins: int | None = 512,
) -> AveragedPeriodogram:
    """Compute an averaged periodogram from equal-duration count segments.

    ``fractional_rms`` uses the one-sided ``rms^2 / Hz`` convention, retaining
    (and fitting) the Poisson white-noise level.  ``leahy`` has an expected
    Poisson level of two.  The final incomplete segment is ignored.
    """

    data = np.asarray(counts)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(data)) or np.any(data < 0.0):
        raise ValueError("counts must be finite and non-negative")
    if normalization not in ("fractional_rms", "leahy"):
        raise ValueError("normalization must be 'fractional_rms' or 'leahy'")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    if not np.isfinite(segment_s) or segment_s <= dt_s:
        raise ValueError("segment_s must be larger than dt_s")

    samples_per_segment = int(round(segment_s / dt_s))
    n_segments = data.size // samples_per_segment
    if n_segments < 1:
        raise ValueError("light curve does not contain one complete segment")
    used = np.asarray(
        data[: n_segments * samples_per_segment], dtype=np.float64
    ).reshape(n_segments, samples_per_segment)
    means = np.mean(used, axis=1)
    if np.any(means <= 0.0):
        raise ValueError("each segment must contain at least one count")
    centered = used - means[:, None]
    transforms = np.fft.rfft(centered, axis=1)
    squared = np.abs(transforms) ** 2

    if normalization == "fractional_rms":
        factors = 2.0 * dt_s / (samples_per_segment * means**2)
    else:
        factors = 2.0 / (samples_per_segment * means)
    segment_power = squared * factors[:, None]
    # The Nyquist coefficient has no negative-frequency partner.
    if samples_per_segment % 2 == 0:
        segment_power[:, -1] *= 0.5
    frequencies = np.fft.rfftfreq(samples_per_segment, d=dt_s)

    keep = frequencies > 0.0
    if max_frequency_hz is not None:
        if not np.isfinite(max_frequency_hz) or max_frequency_hz <= 0.0:
            raise ValueError("max_frequency_hz must be positive when supplied")
        keep &= frequencies <= min(float(max_frequency_hz), 0.5 / dt_s)
    frequencies = frequencies[keep]
    segment_power = segment_power[:, keep]
    if frequencies.size == 0:
        raise ValueError("frequency selection is empty")

    requested = frequencies.size if frequency_bins is None else int(frequency_bins)
    if requested < 1:
        raise ValueError("frequency_bins must be positive or None")
    groups = _geometric_groups(frequencies.size, requested)
    rebinned_frequency = np.empty(len(groups), dtype=np.float64)
    rebinned_segments = np.empty((n_segments, len(groups)), dtype=np.float64)
    group_sizes = np.empty(len(groups), dtype=np.int64)
    for group_index, (left, right) in enumerate(groups):
        rebinned_frequency[group_index] = float(np.mean(frequencies[left:right]))
        rebinned_segments[:, group_index] = np.mean(
            segment_power[:, left:right], axis=1
        )
        group_sizes[group_index] = right - left

    power = np.mean(rebinned_segments, axis=0)
    # Averaging M independent segment powers and K adjacent Fourier powers gives
    # approximately P/sqrt(MK), including the well-defined one-segment case.
    power_error = power / np.sqrt(n_segments * group_sizes)
    floor = np.finfo(float).eps * max(float(np.max(power)), 1.0)
    power_error = np.maximum(power_error, floor)

    return AveragedPeriodogram(
        frequencies_hz=rebinned_frequency,
        power=np.asarray(power, dtype=np.float64),
        power_error=np.asarray(power_error, dtype=np.float64),
        segment_powers=rebinned_segments,
        group_sizes=group_sizes,
        normalization=normalization,
        n_segments=n_segments,
        segment_s=samples_per_segment * dt_s,
        dt_s=float(dt_s),
        mean_rate_hz=float(np.mean(used) / dt_s),
    )


# Public spelling requested by the case-study design.
compute_averaged_periodogram = averaged_periodogram


def _power_model(
    frequencies_hz: FloatArray,
    parameters: FloatArray,
    include_qpo: bool,
) -> FloatArray:
    broad_amplitude, broad_fwhm = parameters[0], parameters[1]
    if include_qpo:
        qpo_amplitude, centroid, qpo_fwhm, white_noise = parameters[2:6]
        qpo = lorentzian_power(
            frequencies_hz, qpo_amplitude, centroid, qpo_fwhm
        )
    else:
        white_noise = parameters[2]
        qpo = 0.0
    broad = lorentzian_power(frequencies_hz, broad_amplitude, 0.0, broad_fwhm)
    return np.asarray(white_noise + broad + qpo, dtype=np.float64)


@dataclass(slots=True)
class _Fit:
    parameters: FloatArray
    success: bool
    cost: float
    parameter_errors: FloatArray


def _least_squares_fit(
    frequencies: FloatArray,
    power: FloatArray,
    error: FloatArray,
    *,
    expected_qpo_hz: float | None,
    include_qpo: bool,
    initial_parameters: FloatArray | None = None,
    max_nfev: int = 2_000,
) -> _Fit:
    resolution = max(
        float(np.min(np.diff(frequencies))) if frequencies.size > 1 else frequencies[0],
        np.finfo(float).eps,
    )
    frequency_max = float(frequencies[-1])
    white_initial = max(float(np.median(power[-max(8, power.size // 5) :])), 1e-14)
    widths = np.gradient(frequencies)
    excess = np.maximum(power - white_initial, 0.0)
    integrated = max(float(np.sum(excess * widths)), 1e-10)
    amplitude_upper = max(
        100.0 * integrated,
        20.0 * float(np.max(power)) * frequency_max,
        1e-6,
    )
    white_upper = max(20.0 * float(np.max(power)), 10.0 * white_initial, 1e-8)
    broad_width_upper = max(min(frequency_max, 64.0), 4.0 * resolution)

    null_lower = np.asarray([0.0, resolution, 0.0], dtype=np.float64)
    null_upper = np.asarray(
        [amplitude_upper, broad_width_upper, white_upper], dtype=np.float64
    )
    null_initial = np.asarray(
        [min(integrated, 0.25 * amplitude_upper), min(0.5, broad_width_upper), white_initial],
        dtype=np.float64,
    )

    safe_power = np.maximum(power, np.finfo(float).tiny)
    relative_error = np.clip(error / safe_power, 0.02, 2.0)

    def residual(parameters: FloatArray, qpo: bool) -> FloatArray:
        model = np.maximum(
            _power_model(frequencies, parameters, qpo), np.finfo(float).tiny
        )
        return (np.log(model) - np.log(safe_power)) / relative_error

    null_fit = least_squares(
        residual,
        null_initial,
        args=(False,),
        bounds=(null_lower, null_upper),
        x_scale="jac",
        loss="soft_l1",
        max_nfev=max_nfev,
    )
    if not include_qpo:
        errors = _covariance_errors(null_fit)
        return _Fit(
            np.asarray(null_fit.x, dtype=np.float64),
            bool(null_fit.success and np.all(np.isfinite(null_fit.x))),
            float(null_fit.cost),
            errors,
        )

    null_model = _power_model(frequencies, null_fit.x, False)
    score = (power - null_model) / np.maximum(error, np.finfo(float).tiny)
    search = (frequencies >= max(2.0 * resolution, 0.01)) & (
        frequencies <= min(frequency_max, 32.0)
    )
    if expected_qpo_hz is None:
        if np.any(search):
            candidates = np.flatnonzero(search)
            # A short smoothing kernel favors a coherent cluster rather than a
            # single exponential periodogram outlier.
            smoothed = np.convolve(score, np.ones(3) / 3.0, mode="same")
            centroid_initial = float(frequencies[candidates[np.argmax(smoothed[candidates])]])
        else:
            centroid_initial = float(frequencies[np.argmax(score)])
        centroid_lower = max(float(frequencies[0]), 2.0 * resolution)
        centroid_upper = min(frequency_max, 32.0)
    else:
        expected = float(expected_qpo_hz)
        if not np.isfinite(expected) or expected <= 0.0:
            raise ValueError("expected_qpo_hz must be positive when supplied")
        span = max(0.75 * expected, 8.0 * resolution)
        centroid_lower = max(float(frequencies[0]), expected - span)
        centroid_upper = min(frequency_max, expected + span)
        if centroid_upper <= centroid_lower:
            raise ValueError("expected_qpo_hz is outside the fitted frequency range")
        centroid_initial = float(np.clip(expected, centroid_lower, centroid_upper))

    qpo_fwhm_lower = max(resolution, centroid_initial / 100.0)
    qpo_fwhm_upper = min(
        frequency_max,
        max(4.0 * centroid_initial, 8.0 * resolution),
    )
    qpo_fwhm_upper = max(qpo_fwhm_upper, qpo_fwhm_lower * 1.01)
    qpo_fwhm_initial = float(
        np.clip(max(3.0 * resolution, centroid_initial / 8.0), qpo_fwhm_lower, qpo_fwhm_upper)
    )
    local = np.abs(frequencies - centroid_initial) <= max(
        2.0 * qpo_fwhm_initial, 3.0 * resolution
    )
    qpo_initial = max(
        float(np.sum(np.maximum(power[local] - null_model[local], 0.0) * widths[local])),
        integrated * 0.01,
        1e-12,
    )
    lower = np.asarray(
        [0.0, resolution, 0.0, centroid_lower, qpo_fwhm_lower, 0.0],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            amplitude_upper,
            broad_width_upper,
            amplitude_upper,
            centroid_upper,
            qpo_fwhm_upper,
            white_upper,
        ],
        dtype=np.float64,
    )
    if initial_parameters is None or initial_parameters.shape != (6,):
        initial = np.asarray(
            [
                null_fit.x[0],
                null_fit.x[1],
                min(qpo_initial, 0.5 * amplitude_upper),
                centroid_initial,
                qpo_fwhm_initial,
                null_fit.x[2],
            ],
            dtype=np.float64,
        )
    else:
        initial = np.asarray(initial_parameters, dtype=np.float64).copy()
    initial = np.clip(initial, lower + 1e-12 * (upper - lower), upper - 1e-12 * (upper - lower))

    # Base fits use a few centroid starts to reduce local-mode sensitivity.
    starts = [initial]
    if initial_parameters is None:
        for multiplier in (0.8, 1.2):
            alternate = initial.copy()
            alternate[3] = np.clip(
                centroid_initial * multiplier, centroid_lower, centroid_upper
            )
            starts.append(alternate)
    best = None
    for start in starts:
        candidate = least_squares(
            residual,
            start,
            args=(True,),
            bounds=(lower, upper),
            x_scale="jac",
            loss="soft_l1",
            max_nfev=max_nfev,
        )
        if best is None or candidate.cost < best.cost:
            best = candidate
    assert best is not None
    return _Fit(
        np.asarray(best.x, dtype=np.float64),
        bool(best.success and np.all(np.isfinite(best.x))),
        float(best.cost),
        _covariance_errors(best),
    )


def _covariance_errors(fit: Any) -> FloatArray:
    parameters = np.asarray(fit.x, dtype=np.float64)
    fallback = np.full(parameters.size, np.nan, dtype=np.float64)
    jacobian = np.asarray(fit.jac, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] <= jacobian.shape[1]:
        return fallback
    try:
        covariance = np.linalg.pinv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        return fallback
    degrees = max(jacobian.shape[0] - jacobian.shape[1], 1)
    scale = 2.0 * float(fit.cost) / degrees
    diagonal = np.diag(covariance) * scale
    return np.sqrt(np.maximum(diagonal, 0.0))


def fit_power_spectrum(
    periodogram: AveragedPeriodogram,
    *,
    expected_qpo_hz: float | None = None,
    allow_qpo: bool = True,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    band: str = "total",
    rng: np.random.Generator | None = None,
    seed: int | np.random.SeedSequence | None = None,
) -> TimingResult:
    """Fit broad-band and optional QPO Lorentzians plus white noise.

    Bootstrap realizations resample independent segment periodograms.  If only
    one segment is available, Gamma draws using the number of averaged Fourier
    ordinates provide a parametric fallback.  A detection is labelled only when
    both specified case-study rules hold: ``Q >= 2`` and integrated QPO
    amplitude divided by its bootstrap error is at least three.
    """

    if not isinstance(periodogram, AveragedPeriodogram):
        raise TypeError("periodogram must be an AveragedPeriodogram")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap cannot be negative")
    frequencies = np.asarray(periodogram.frequencies_hz, dtype=np.float64)
    power = np.asarray(periodogram.power, dtype=np.float64)
    error = np.asarray(periodogram.power_error, dtype=np.float64)
    if frequencies.size < 12 or frequencies.shape != power.shape or power.shape != error.shape:
        raise ValueError("periodogram needs at least 12 aligned frequency bins")
    if np.any(~np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError("periodogram powers must be positive and finite")

    base = _least_squares_fit(
        frequencies,
        power,
        error,
        expected_qpo_hz=expected_qpo_hz,
        include_qpo=allow_qpo,
    )
    generator = _coerce_rng(rng, seed)

    if not allow_qpo:
        parameters = base.parameters
        model = _power_model(frequencies, parameters, False)
        return TimingResult(
            frequencies_hz=frequencies.copy(),
            power=power.copy(),
            power_error=error.copy(),
            model_power=model,
            normalization=periodogram.normalization,
            band=band,
            n_segments=periodogram.n_segments,
            centroid_hz=None,
            centroid_error_hz=None,
            fwhm_hz=None,
            fwhm_error_hz=None,
            q_factor=None,
            qpo_amplitude=0.0,
            qpo_amplitude_error=None,
            amplitude_significance=0.0,
            detected=False,
            classification=None,
            fit_success=base.success,
            bootstrap_successes=0,
            fit_parameters={
                "broadband_amplitude": float(parameters[0]),
                "broadband_fwhm_hz": float(parameters[1]),
                "white_noise": float(parameters[2]),
            },
            metadata={"label": "Synthetic / NICER-inspired", "qpo_fit": False},
        )

    bootstrap_parameters: list[FloatArray] = []
    segments = np.asarray(periodogram.segment_powers, dtype=np.float64)
    for _ in range(int(n_bootstrap)):
        if segments.ndim == 2 and segments.shape[0] >= 2:
            indices = generator.integers(0, segments.shape[0], size=segments.shape[0])
            trial_power = np.mean(segments[indices], axis=0)
        else:
            effective_shape = np.maximum(
                (power / np.maximum(error, np.finfo(float).tiny)) ** 2, 1.0
            )
            trial_power = generator.gamma(effective_shape, power / effective_shape)
        try:
            trial = _least_squares_fit(
                frequencies,
                np.maximum(trial_power, np.finfo(float).tiny),
                error,
                expected_qpo_hz=expected_qpo_hz,
                include_qpo=True,
                initial_parameters=base.parameters,
                max_nfev=500,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if trial.success and np.all(np.isfinite(trial.parameters)):
            bootstrap_parameters.append(trial.parameters)

    if len(bootstrap_parameters) >= 2:
        samples = np.asarray(bootstrap_parameters, dtype=np.float64)
        bootstrap_errors = np.std(samples, axis=0, ddof=1)
    else:
        bootstrap_errors = np.full(6, np.nan, dtype=np.float64)
    # A singular resample distribution can occur with very few segments.  The
    # local covariance is a conservative finite fallback, while successful
    # bootstrap estimates remain primary.
    uncertainties = bootstrap_errors.copy()
    invalid = ~np.isfinite(uncertainties) | (uncertainties <= 0.0)
    uncertainties[invalid] = base.parameter_errors[invalid]

    parameters = base.parameters
    amplitude_error = float(uncertainties[2]) if np.isfinite(uncertainties[2]) else None
    centroid_error = float(uncertainties[3]) if np.isfinite(uncertainties[3]) else None
    fwhm_error = float(uncertainties[4]) if np.isfinite(uncertainties[4]) else None
    centroid = float(parameters[3])
    fwhm = float(parameters[4])
    amplitude = max(float(parameters[2]), 0.0)
    q_factor = centroid / fwhm if fwhm > 0.0 else float("inf")
    significance = (
        amplitude / amplitude_error
        if amplitude_error is not None and amplitude_error > 0.0
        else 0.0
    )
    detected = bool(
        base.success
        and np.isfinite(q_factor)
        and q_factor >= 2.0
        and np.isfinite(significance)
        and significance >= 3.0
    )
    model = _power_model(frequencies, parameters, True)
    return TimingResult(
        frequencies_hz=frequencies.copy(),
        power=power.copy(),
        power_error=error.copy(),
        model_power=model,
        normalization=periodogram.normalization,
        band=band,
        n_segments=periodogram.n_segments,
        centroid_hz=centroid,
        centroid_error_hz=centroid_error,
        fwhm_hz=fwhm,
        fwhm_error_hz=fwhm_error,
        q_factor=float(q_factor),
        qpo_amplitude=amplitude,
        qpo_amplitude_error=amplitude_error,
        amplitude_significance=float(significance),
        detected=detected,
        classification="type-C-like" if detected else None,
        fit_success=base.success,
        bootstrap_successes=len(bootstrap_parameters),
        fit_parameters={
            "broadband_amplitude": float(parameters[0]),
            "broadband_fwhm_hz": float(parameters[1]),
            "qpo_amplitude": amplitude,
            "qpo_centroid_hz": centroid,
            "qpo_fwhm_hz": fwhm,
            "white_noise": float(parameters[5]),
        },
        metadata={
            "label": "Synthetic / NICER-inspired",
            "qpo_fit": True,
            "detection_rule": "Q >= 2 and amplitude/bootstrap_error >= 3",
        },
    )


def analyze_light_curves(
    light_curves: LightCurveBands,
    *,
    band: Literal["soft", "hard", "total"] = "total",
    normalization: Normalization = "fractional_rms",
    expected_qpo_hz: float | None = None,
    allow_qpo: bool = True,
    max_frequency_hz: float = 64.0,
    frequency_bins: int = 512,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    rng: np.random.Generator | None = None,
    seed: int | np.random.SeedSequence | None = None,
) -> TimingResult:
    """Compute and fit the averaged periodogram of one light-curve band."""

    periodogram = averaged_periodogram(
        light_curves.counts_for(band),
        dt_s=light_curves.dt_s,
        segment_s=light_curves.segment_s,
        normalization=normalization,
        max_frequency_hz=max_frequency_hz,
        frequency_bins=frequency_bins,
    )
    result = fit_power_spectrum(
        periodogram,
        expected_qpo_hz=expected_qpo_hz,
        allow_qpo=allow_qpo,
        n_bootstrap=n_bootstrap,
        band=band,
        rng=rng,
        seed=seed,
    )
    result.injected_qpo_hz = light_curves.metadata.get("qpo_frequency_hz")
    result.injected_fractional_rms = light_curves.metadata.get("fractional_rms")
    result.metadata.update(
        {
            "hardness": light_curves.hardness,
            "mean_rate_hz": periodogram.mean_rate_hz,
            "duration_s": light_curves.metadata.get("duration_s"),
            "profile": light_curves.metadata.get("profile"),
        }
    )
    return result


def epoch_fractional_rms(epoch_id: str) -> float:
    """Return the fixed injected rms for a case-study epoch."""

    try:
        return OUTBURST_FRACTIONAL_RMS[str(epoch_id).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown case-study epoch: {epoch_id!r}") from exc


def simulate_timing_epoch(
    epoch: Any,
    *,
    profile: str = "full",
    soft_rate_hz: float | None = None,
    hard_rate_hz: float | None = None,
    fractional_rms: float | None = None,
    normalization: Normalization = "fractional_rms",
    band: Literal["soft", "hard", "total"] = "total",
    rng: np.random.Generator | None = None,
    seed: int | np.random.SeedSequence | None = None,
) -> TimingResult:
    """Simulate and analyse one epoch-like object from the fixed outburst.

    The object may expose ``epoch_id`` (or ``epoch``), ``qpo_hz``, and
    optionally ``gamma``.  This duck-typed boundary avoids coupling the timing
    implementation to the spectral dataclasses.
    """

    epoch_id = str(getattr(epoch, "epoch_id", getattr(epoch, "epoch", epoch)))
    qpo_value = getattr(epoch, "qpo_hz", None)
    qpo_hz = None if qpo_value is None else float(qpo_value)
    rms = epoch_fractional_rms(epoch_id) if fractional_rms is None else float(fractional_rms)
    total_rate = OUTBURST_TOTAL_RATE_HZ.get(epoch_id.upper(), 2_000.0)

    if soft_rate_hz is None or hard_rate_hz is None:
        gamma = float(getattr(epoch, "gamma", 1.8))
        # A bounded, monotonic surrogate gives softer epochs a lower synthetic
        # hardness without claiming a detector-calibrated band conversion.
        hard_fraction = float(np.clip(0.72 - 0.20 * (gamma - 1.5), 0.28, 0.72))
        inferred_hard = total_rate * hard_fraction
        inferred_soft = total_rate - inferred_hard
        soft = inferred_soft if soft_rate_hz is None else float(soft_rate_hz)
        hard = inferred_hard if hard_rate_hz is None else float(hard_rate_hz)
    else:
        soft, hard = float(soft_rate_hz), float(hard_rate_hz)

    # Spawn independent streams for the light curve and bootstrap even when a
    # caller supplies a single integer seed.
    if rng is None:
        sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        simulation_seed, bootstrap_seed = sequence.spawn(2)
        simulation_rng = np.random.default_rng(simulation_seed)
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
    else:
        if seed is not None:
            raise ValueError("pass either rng or seed, not both")
        simulation_rng = rng
        bootstrap_rng = rng

    settings = TimingSettings.for_profile(profile)
    light_curves = simulate_light_curves(
        soft_rate_hz=soft,
        hard_rate_hz=hard,
        fractional_rms=rms,
        qpo_frequency_hz=qpo_hz,
        profile=profile,
        rng=simulation_rng,
    )
    result = analyze_light_curves(
        light_curves,
        band=band,
        normalization=normalization,
        expected_qpo_hz=qpo_hz,
        allow_qpo=qpo_hz is not None,
        max_frequency_hz=settings.max_frequency_hz,
        frequency_bins=settings.frequency_bins,
        n_bootstrap=settings.bootstraps,
        rng=bootstrap_rng,
    )
    result.epoch_id = epoch_id
    return result


__all__ = [
    "AveragedPeriodogram",
    "DEFAULT_BOOTSTRAPS",
    "DEFAULT_DT_S",
    "DEFAULT_DURATION_S",
    "DEFAULT_SEGMENT_S",
    "LightCurveBands",
    "OUTBURST_FRACTIONAL_RMS",
    "TimingResult",
    "TimingSettings",
    "analyze_light_curves",
    "averaged_periodogram",
    "compute_averaged_periodogram",
    "epoch_fractional_rms",
    "fit_power_spectrum",
    "lorentzian_power",
    "simulate_light_curves",
    "simulate_timing_epoch",
]
