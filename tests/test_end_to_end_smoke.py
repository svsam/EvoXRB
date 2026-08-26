from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
import pytest

from evoxrb import SYNTHETIC_LABEL
from evoxrb.campaign import run_case_study
from evoxrb.config import load_config


@pytest.mark.slow
def test_reduced_case_study_produces_complete_accepted_artifacts(tmp_path) -> None:
    config = load_config()
    isolated = replace(
        config,
        project={
            **config.project,
            "output_dir": str(tmp_path / "results"),
            "figure_dir": str(tmp_path / "figures"),
            "report_dir": str(tmp_path / "reports"),
        },
    )
    summary = run_case_study(isolated, profile="smoke", resume=False)
    assert summary["acceptance_valid"] is True

    results = tmp_path / "results"
    truth = pd.read_csv(results / "truth.csv")
    fits = pd.read_csv(results / "fit_results.csv")
    timing = pd.read_csv(results / "timing_results.csv")
    assert truth["epoch_id"].tolist() == [f"E{index:02d}" for index in range(1, 13)]
    assert {"scipy", "ga", "ga+scipy"}.issubset(set(fits["method"]))
    assert set(timing["epoch_id"]) == {"E01", "E08", "E10"}
    for frame in (truth, fits, timing):
        assert frame["label"].str.startswith(SYNTHETIC_LABEL).all()

    acceptance = json.loads((results / "acceptance.json").read_text(encoding="utf-8"))
    provenance = json.loads((results / "provenance.json").read_text(encoding="utf-8"))
    assert acceptance["valid"] is True
    assert provenance["config_hash"] == config.digest
    assert provenance["master_seed"] == 1_820_070
    assert provenance["runtime_seconds"] > 0.0
    assert len(list((tmp_path / "figures").glob("*.png"))) == 10
    report = (tmp_path / "reports" / "case_study.md").read_text(encoding="utf-8")
    assert SYNTHETIC_LABEL in report
    assert "contains no real NICER observations" in report
