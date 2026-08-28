from __future__ import annotations

import numpy as np
import pytest

from evoxrb.reference import (
    DEFAULT_REFERENCE_LABEL,
    ReferenceSpectrum,
    load_reference_spectrum_csv,
)


def test_reference_spectrum_sorts_native_grid_and_is_plot_ready() -> None:
    reference = ReferenceSpectrum(
        energy_keV=[3.0, 0.5, 1.5],
        count_rate_density=[30.0, 5.0, 15.0],
        count_rate_error=[3.0, 0.5, 1.5],
        label="Example observed spectrum",
        source="doi:example",
        metadata={"instrument": "example detector"},
    )

    np.testing.assert_array_equal(reference.energy_keV, [0.5, 1.5, 3.0])
    np.testing.assert_array_equal(reference.count_rate_density, [5.0, 15.0, 30.0])
    np.testing.assert_array_equal(reference.count_rate_error, [0.5, 1.5, 3.0])
    assert reference.size == 3
    assert reference.has_errors
    assert reference.source == "doi:example"
    assert reference.metadata["instrument"] == "example detector"
    assert "Visual comparison only" in reference.comparison_notice
    assert "not used for C-stat" in reference.comparison_notice
    assert not reference.energy_keV.flags.writeable

    plot_data = reference.as_plot_data()
    assert set(plot_data) == {"x", "y", "yerr", "label"}
    assert plot_data["label"] == reference.label
    plot_data["x"][0] = 99.0
    assert reference.energy_keV[0] == 0.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"energy_keV": [], "count_rate_density": []},
            "non-empty one-dimensional",
        ),
        (
            {"energy_keV": [1.0, 1.0], "count_rate_density": [2.0, 3.0]},
            "unique",
        ),
        (
            {"energy_keV": [0.0, 1.0], "count_rate_density": [2.0, 3.0]},
            "strictly positive",
        ),
        (
            {"energy_keV": [1.0, 2.0], "count_rate_density": [2.0, -3.0]},
            "cannot be negative",
        ),
        (
            {"energy_keV": [1.0, 2.0], "count_rate_density": [2.0]},
            "one value per energy bin",
        ),
        (
            {
                "energy_keV": [1.0, 2.0],
                "count_rate_density": [2.0, 3.0],
                "count_rate_error": [0.2, np.nan],
            },
            "finite",
        ),
    ],
)
def test_reference_spectrum_rejects_invalid_arrays(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ReferenceSpectrum(**kwargs)  # type: ignore[arg-type]


def test_load_direct_rate_csv_with_provenance(tmp_path) -> None:
    path = tmp_path / "reference.csv"
    path.write_text(
        "energy_keV,count_rate_density,count_rate_error,label,source\n"
        "2.0,8.0,0.8,Published comparison,doi:123\n"
        "0.5,20.0,2.0,Published comparison,doi:123\n",
        encoding="utf-8",
    )

    reference = load_reference_spectrum_csv(
        path, metadata={"retrieved": "2026-08-26"}
    )

    np.testing.assert_array_equal(reference.energy_keV, [0.5, 2.0])
    np.testing.assert_array_equal(reference.count_rate_density, [20.0, 8.0])
    np.testing.assert_array_equal(reference.count_rate_error, [2.0, 0.8])
    assert reference.label == "Published comparison"
    assert reference.source == "doi:123"
    assert reference.metadata["retrieved"] == "2026-08-26"
    assert reference.metadata["input_representation"] == "count_rate_density"
    assert reference.metadata["visual_comparison_only"] is True


def test_load_counts_csv_converts_to_rate_density_and_poisson_error(tmp_path) -> None:
    path = tmp_path / "counts.csv"
    path.write_text(
        "energy_keV,counts,exposure_s,bin_width_keV\n"
        "1.0,100,10,0.5\n"
        "2.0,36,20,0.25\n",
        encoding="utf-8",
    )

    reference = ReferenceSpectrum.from_csv(path)

    np.testing.assert_allclose(reference.count_rate_density, [20.0, 7.2])
    np.testing.assert_allclose(reference.count_rate_error, [2.0, 1.2])
    assert reference.label == DEFAULT_REFERENCE_LABEL
    assert reference.metadata["input_representation"] == "counts"


def test_load_counts_csv_accepts_bin_edges_and_explicit_count_error(tmp_path) -> None:
    path = tmp_path / "binned_counts.csv"
    path.write_text(
        "energy_keV,energy_low_keV,energy_high_keV,counts,counts_error,exposure_s\n"
        "1.5,1.0,2.0,25,10,5\n",
        encoding="utf-8",
    )

    reference = load_reference_spectrum_csv(path, label="My comparison")

    np.testing.assert_allclose(reference.count_rate_density, [5.0])
    np.testing.assert_allclose(reference.count_rate_error, [2.0])
    assert reference.label == "My comparison"


def test_zero_count_display_error_and_loader_provenance_are_conservative(tmp_path) -> None:
    path = tmp_path / "zero_counts.csv"
    path.write_text(
        "energy_keV,counts,exposure_s,bin_width_keV\n"
        "1.0,0,10,0.5\n",
        encoding="utf-8",
    )

    reference = load_reference_spectrum_csv(
        path,
        metadata={"visual_comparison_only": False, "input_path": "wrong"},
    )

    np.testing.assert_allclose(reference.count_rate_error, [0.2])
    assert reference.source == str(path.resolve())
    assert reference.metadata["input_path"] == str(path.resolve())
    assert reference.metadata["visual_comparison_only"] is True


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("energy_keV,value\n1.0,2.0\n", "requires count_rate_density"),
        ("energy_keV,counts,exposure_s\n1.0,5,10\n", "bin_width_keV"),
        (
            "energy_keV,count_rate_density,label\n"
            "1.0,2.0,first\n2.0,3.0,second\n",
            "constant value",
        ),
        (
            "energy_keV,counts,exposure_s,bin_width_keV\n1.0,5,0,0.5\n",
            "strictly positive",
        ),
    ],
)
def test_reference_csv_rejects_incomplete_or_inconsistent_input(
    tmp_path, contents: str, match: str
) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_reference_spectrum_csv(path)
