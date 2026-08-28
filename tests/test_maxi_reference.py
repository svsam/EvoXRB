from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


astropy = pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_maxi_reference.py"
SPEC = importlib.util.spec_from_file_location("prepare_maxi_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare_maxi_reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_maxi_reference)


def _write_spectrum(
    path: Path,
    *,
    counts: list[int],
    exposure: float,
    backscal: float,
) -> None:
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="CHANNEL", format="J", array=np.arange(len(counts))
            ),
            fits.Column(name="COUNTS", format="J", array=np.asarray(counts)),
        ],
        name="SPECTRUM",
    )
    spectrum.header["TELESCOP"] = "MAXI"
    spectrum.header["INSTRUME"] = "GSC"
    spectrum.header["EXPOSURE"] = exposure
    spectrum.header["BACKSCAL"] = backscal
    spectrum.header["DATE-OBS"] = "2018-07-02"
    spectrum.header["DATE-END"] = "2018-07-03"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)


def _write_response(path: Path) -> None:
    bounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=np.arange(4)),
            fits.Column(
                name="E_MIN", format="E", array=np.asarray([2.0, 2.5, 3.0, 3.5])
            ),
            fits.Column(
                name="E_MAX", format="E", array=np.asarray([2.5, 3.0, 3.5, 4.0])
            ),
        ],
        name="EBOUNDS",
    )
    fits.HDUList([fits.PrimaryHDU(), bounds]).writeto(path)


def test_convert_reference_scales_background_and_propagates_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pi"
    background = tmp_path / "background.pi"
    response = tmp_path / "response.rmf"
    output = tmp_path / "reference.csv"
    provenance = tmp_path / "reference.provenance.json"
    _write_spectrum(
        source, counts=[100, 90, 64, 56], exposure=100.0, backscal=0.1
    )
    _write_spectrum(
        background, counts=[20, 20, 16, 16], exposure=200.0, backscal=0.2
    )
    _write_response(response)

    rows = prepare_maxi_reference.convert_reference(
        source_path=source,
        background_path=background,
        response_path=response,
        output_path=output,
        provenance_path=provenance,
        energy_min=2.0,
        energy_max=4.0,
        bin_width=1.0,
        label="Test MAXI/GSC spectrum",
        result_url="https://example.test/maxi-result",
        job_id="test-job",
    )

    assert len(rows) == 2
    assert rows[0]["source_counts"] == 190
    assert rows[0]["background_counts"] == 40
    assert rows[0]["count_rate_density"] == pytest.approx(1.8)
    assert rows[1]["count_rate_density"] == pytest.approx(1.12)
    assert rows[0]["count_rate_error"] == pytest.approx(np.sqrt(192.5) / 100.0)
    assert rows[1]["count_rate_error"] == pytest.approx(np.sqrt(122.0) / 100.0)

    frame = pd.read_csv(output)
    assert frame["energy_keV"].tolist() == [2.5, 3.5]
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    assert metadata["background_scale_in_source_counts"] == pytest.approx(0.25)
    assert metadata["input_files"].keys() == {
        "source.pi",
        "background.pi",
        "response.rmf",
    }
    assert str(tmp_path) not in provenance.read_text(encoding="utf-8")


def test_bundled_maxi_reference_matches_published_count_totals() -> None:
    reference = pd.read_csv(
        ROOT / "data" / "reference" / "maxi_j1820p070_mjd58302.csv"
    )

    assert len(reference) == 16
    assert reference["energy_keV"].tolist() == pytest.approx(
        np.arange(2.25, 10.0, 0.5)
    )
    assert int(reference["source_counts"].sum()) == 3004
    assert int(reference["background_counts"].sum()) == 330
    assert (reference["count_rate_density"] >= 0.0).all()
    assert (reference["count_rate_error"] > 0.0).all()
