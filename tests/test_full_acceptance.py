from __future__ import annotations

import os

import numpy as np
import pytest

from evoxrb.genetic import GAConfig, GeneticOptimizer
from evoxrb.instrument import default_nicer_inspired_response
from evoxrb.models import SpectrumModel
from evoxrb.objective import Objective
from evoxrb.optimization import local_polish, multistart_scipy
from evoxrb.parameters import ParameterSpec, SearchSpace
from evoxrb.simulation import derive_seed, simulate_spectrum


FULL = os.environ.get("EVOXRB_RUN_FULL") == "1"


@pytest.mark.full
@pytest.mark.slow
@pytest.mark.skipif(not FULL, reason="set EVOXRB_RUN_FULL=1 for the opt-in acceptance campaign")
def test_five_full_ga_seeds_polish_to_high_signal_scipy() -> None:
    response = default_nicer_inspired_response()
    truth = {"tin": 0.7, "ndisk": 8_000.0, "gamma": 2.1, "norm": 0.25}
    model = SpectrumModel("powerlaw", fixed_nh=0.15)
    spectrum = simulate_spectrum(response, model, truth, 100_000.0, 1820070)
    objective = Objective(spectrum, response, model)
    space = SearchSpace(
        ParameterSpec("tin", 0.05, 2.0),
        ParameterSpec("ndisk", 0.1, 1_000_000.0, "log10"),
        ParameterSpec("gamma", 1.2, 3.5),
        ParameterSpec("norm", 1e-6, 100.0, "log10"),
    )
    scipy_score = multistart_scipy(objective.evaluate, space, n_starts=20, seed=1820070).best_score
    optimizer = GeneticOptimizer(space, GAConfig())
    runs = [
        optimizer.optimize(objective.evaluate, seed=derive_seed(1820070, 99, index))
        for index in range(5)
    ]
    polished = [local_polish(objective.evaluate, space, run) for run in runs]
    assert sum(run.best_score - scipy_score <= 1.0 for run in polished) >= 4
    best = min(polished, key=lambda run: run.best_score)
    assert abs(best.best_parameters["tin"] / truth["tin"] - 1.0) < 0.05
    assert abs(best.best_parameters["gamma"] / truth["gamma"] - 1.0) < 0.05
    assert abs(best.best_parameters["ndisk"] / truth["ndisk"] - 1.0) < 0.10
    assert abs(best.best_parameters["norm"] / truth["norm"] - 1.0) < 0.10
