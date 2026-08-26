from __future__ import annotations

import numpy as np
import pytest

from evoxrb.inference import PosteriorConfig, run_emcee, run_posterior, separated_ga_modes
from evoxrb.instrument import default_nicer_inspired_response
from evoxrb.models import SpectrumModel
from evoxrb.objective import Objective
from evoxrb.optimization import multistart_scipy
from evoxrb.parameters import ParameterSpec, SearchSpace
from evoxrb.simulation import simulate_spectrum


def analytic_objective(parameters: dict[str, float]) -> float:
    return ((parameters["x"] - 0.2) / 0.3) ** 2


def test_response_folded_objective_handles_invalid_models() -> None:
    response = default_nicer_inspired_response()
    truth = {"tin": 0.7, "ndisk": 8_000.0, "gamma": 2.1, "norm": 0.25}
    model = SpectrumModel("powerlaw", fixed_nh=0.15)
    spectrum = simulate_spectrum(response, model, truth, 512.0, 1234)
    objective = Objective(spectrum, response, model)

    assert np.isfinite(objective.evaluate(truth))
    assert objective.evaluate({**truth, "tin": -1.0}) == np.inf
    assert objective.evaluate({"tin": 0.7}) == np.inf
    assert objective.expected_counts(truth).shape == spectrum.counts.shape


def test_high_signal_scipy_recovers_injected_parameters() -> None:
    response = default_nicer_inspired_response()
    truth = {"tin": 0.7, "ndisk": 8_000.0, "gamma": 2.1, "norm": 0.25}
    model = SpectrumModel("powerlaw", fixed_nh=0.15)
    spectrum = simulate_spectrum(response, model, truth, 100_000.0, 9981)
    objective = Objective(spectrum, response, model)
    space = SearchSpace(
        ParameterSpec("tin", 0.05, 2.0),
        ParameterSpec("ndisk", 0.1, 1_000_000.0, "log10"),
        ParameterSpec("gamma", 1.2, 3.5),
        ParameterSpec("norm", 1e-6, 100.0, "log10"),
    )

    result = multistart_scipy(objective.evaluate, space, n_starts=8, seed=82).best
    assert abs(result.parameters["tin"] / truth["tin"] - 1.0) < 0.05
    assert abs(result.parameters["gamma"] / truth["gamma"] - 1.0) < 0.05
    assert abs(result.parameters["ndisk"] / truth["ndisk"] - 1.0) < 0.10
    assert abs(result.parameters["norm"] / truth["norm"] - 1.0) < 0.10


def test_emcee_analytic_gaussian_is_finite_and_deterministic(tmp_path) -> None:
    space = SearchSpace(
        ParameterSpec("x", -3.0, 3.0),
        ParameterSpec("y", -3.0, 3.0),
    )

    def objective(parameters: dict[str, float]) -> float:
        return ((parameters["x"] - 0.4) / 0.25) ** 2 + ((parameters["y"] + 0.3) / 0.40) ** 2

    settings = PosteriorConfig(
        batch_steps=100,
        max_steps=300,
        min_autocorrelation_times=2.0,
        autocorrelation_change=1.0,
        dynesty_live_points=30,
    )
    first = run_emcee(
        objective,
        space,
        {"x": 0.4, "y": -0.3},
        seed=771,
        config=settings,
        checkpoint_path=tmp_path / "first.pkl",
    )
    second = run_emcee(
        objective,
        space,
        {"x": 0.4, "y": -0.3},
        seed=771,
        config=settings,
        checkpoint_path=tmp_path / "second.pkl",
    )
    resumed = run_emcee(
        objective,
        space,
        {"x": 0.4, "y": -0.3},
        seed=999999,
        config=settings,
        checkpoint_path=tmp_path / "first.pkl",
        resume=True,
    )
    assert np.array_equal(first.samples, second.samples)
    assert np.array_equal(first.samples, resumed.samples)
    assert np.all(np.isfinite(first.samples))
    medians = {name: values[1] for name, values in first.quantiles().items()}
    assert medians["x"] == pytest.approx(0.4, abs=0.12)
    assert medians["y"] == pytest.approx(-0.3, abs=0.16)
    assert 0.0 < float(first.diagnostics["acceptance_fraction"]) < 1.0
    assert first.diagnostics["acceptance_ok"] is True
    assert np.all(np.isfinite(first.diagnostics["autocorrelation_time"]))
    assert float(first.diagnostics["boundary_mass_fraction"]) < 0.05


def test_unconverged_emcee_triggers_checkpointed_dynesty(tmp_path) -> None:
    space = SearchSpace(ParameterSpec("x", -3.0, 3.0))
    settings = PosteriorConfig(
        batch_steps=25,
        max_steps=50,
        min_autocorrelation_times=1_000.0,
        autocorrelation_change=0.01,
        dynesty_live_points=30,
        dynesty_dlogz=3.0,
    )
    result = run_posterior(
        analytic_objective,
        space,
        {"x": 0.2},
        seed=223,
        config=settings,
        checkpoint_directory=tmp_path,
    )

    assert result.sampler == "dynesty"
    assert result.converged
    assert np.all(np.isfinite(result.samples))
    assert (tmp_path / "emcee.pkl").exists()
    assert (tmp_path / "dynesty.save").exists()
    assert result.diagnostics["fallback_reasons"]["emcee_unconverged"] is True


def test_separated_competitive_modes_are_detected() -> None:
    class Mode:
        def __init__(self, score: float, genes: list[float]) -> None:
            self.best_score = score
            self.best_genes = np.asarray(genes, dtype=float)

    assert separated_ga_modes(
        [Mode(10.0, [0.1, 0.1]), Mode(11.0, [0.8, 0.8])],  # type: ignore[list-item]
        delta_c=2.0,
        separation=0.1,
    )
    assert not separated_ga_modes(
        [Mode(10.0, [0.1, 0.1]), Mode(13.0, [0.8, 0.8])],  # type: ignore[list-item]
        delta_c=2.0,
        separation=0.1,
    )
