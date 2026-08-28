"""Convert an official MAXI on-demand OGIP spectrum into an EvoXRB reference CSV.

The output is a background-subtracted detector count-rate density for visual
comparison only.  It is not made eligible for EvoXRB's NICER-inspired
likelihood: MAXI/GSC and the educational response are different instruments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray


DEFAULT_JOB_ID = "20260828013000_evoxrbv020real01"
DEFAULT_RESULT_URL = (
    "https://maxi.riken.jp/mxondem/api/results/"
    f"{DEFAULT_JOB_ID}/index.html"
)
DEFAULT_LABEL = "MAXI/GSC MAXI J1820+070, MJD 58301.5-58302.5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _spectrum(path: Path) -> tuple[fits.Header, Any]:
    with fits.open(path, memmap=False) as hdus:
        header = hdus["SPECTRUM"].header.copy()
        data = hdus["SPECTRUM"].data.copy()
    required = {"CHANNEL", "COUNTS"}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError(f"{path.name} is not an OGIP counts spectrum")
    return header, data


def _energy_bounds(path: Path) -> Any:
    with fits.open(path, memmap=False) as hdus:
        data = hdus["EBOUNDS"].data.copy()
    required = {"CHANNEL", "E_MIN", "E_MAX"}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError(f"{path.name} has no usable EBOUNDS extension")
    return data


def _counts_on_energy_grid(
    spectrum: Any, bounds: Any, *, name: str
) -> NDArray[np.float64]:
    by_channel = {
        int(channel): float(counts)
        for channel, counts in zip(
            spectrum["CHANNEL"], spectrum["COUNTS"], strict=True
        )
    }
    try:
        values = np.asarray(
            [by_channel[int(channel)] for channel in bounds["CHANNEL"]],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError(f"{name} channels do not match the response EBOUNDS") from error
    if np.any(values < 0.0):
        raise ValueError(f"{name} counts cannot be negative")
    return values


def convert_reference(
    *,
    source_path: Path,
    background_path: Path,
    response_path: Path,
    output_path: Path,
    provenance_path: Path,
    energy_min: float,
    energy_max: float,
    bin_width: float,
    label: str,
    result_url: str,
    job_id: str,
) -> list[dict[str, Any]]:
    """Write background-subtracted MAXI/GSC display bins and provenance."""

    for path in (source_path, background_path, response_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if energy_min <= 0.0 or energy_max <= energy_min or bin_width <= 0.0:
        raise ValueError("energy limits and bin width must be positive and ordered")
    span = (energy_max - energy_min) / bin_width
    if not np.isclose(span, round(span), rtol=0.0, atol=1e-10):
        raise ValueError("the requested energy range must contain whole display bins")

    source_header, source = _spectrum(source_path)
    background_header, background = _spectrum(background_path)
    bounds = _energy_bounds(response_path)
    for header, name in (
        (source_header, "source"),
        (background_header, "background"),
    ):
        if str(header.get("TELESCOP", "")).strip() != "MAXI":
            raise ValueError(f"{name} spectrum is not labelled as MAXI data")
        if str(header.get("INSTRUME", "")).strip() != "GSC":
            raise ValueError(f"{name} spectrum is not labelled as GSC data")

    source_counts = _counts_on_energy_grid(source, bounds, name="source")
    background_counts = _counts_on_energy_grid(
        background, bounds, name="background"
    )
    exposure_source = float(source_header["EXPOSURE"])
    exposure_background = float(background_header["EXPOSURE"])
    backscal_source = float(source_header["BACKSCAL"])
    backscal_background = float(background_header["BACKSCAL"])
    if min(
        exposure_source,
        exposure_background,
        backscal_source,
        backscal_background,
    ) <= 0.0:
        raise ValueError("OGIP exposure and BACKSCAL values must be positive")

    # XSPEC/OGIP scaling expressed in source-spectrum counts.  Dividing the
    # result by the source exposure gives the background-subtracted rate.
    background_scale = (
        exposure_source
        / exposure_background
        * backscal_source
        / backscal_background
    )
    # EBOUNDS is stored as float32; round away sub-micro-eV representation
    # noise so a nominal 8.000 keV channel lands in the 8.0--8.5 keV bin.
    channel_energy = np.round(
        0.5
        * (
            np.asarray(bounds["E_MIN"], dtype=np.float64)
            + np.asarray(bounds["E_MAX"], dtype=np.float64)
        ),
        decimals=6,
    )
    display_edges = np.linspace(
        energy_min, energy_max, int(round(span)) + 1, dtype=np.float64
    )

    rows: list[dict[str, Any]] = []
    for low, high in zip(display_edges[:-1], display_edges[1:], strict=True):
        selected = (channel_energy >= low) & (channel_energy < high)
        if not np.any(selected):
            raise ValueError(f"no response channels fall in {low:g}-{high:g} keV")
        raw_source = float(np.sum(source_counts[selected]))
        raw_background = float(np.sum(background_counts[selected]))
        net_counts = raw_source - background_scale * raw_background
        width = float(high - low)
        rate_density = net_counts / exposure_source / width
        error_density = (
            np.sqrt(raw_source + background_scale**2 * raw_background)
            / exposure_source
            / width
        )
        if rate_density < 0.0:
            raise ValueError(
                f"background-subtracted bin {low:g}-{high:g} keV is negative; "
                "use wider display bins"
            )
        rows.append(
            {
                "energy_keV": 0.5 * (low + high),
                "count_rate_density": rate_density,
                "count_rate_error": error_density,
                "energy_low_keV": low,
                "energy_high_keV": high,
                "source_counts": int(raw_source),
                "background_counts": int(raw_background),
                "label": label,
                "source": result_url,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "schema_version": 1,
        "target": "MAXI J1820+070",
        "position_j2000_degrees": {"ra": 275.0914, "dec": 7.1853},
        "instrument": "MAXI/GSC",
        "mjd_start": 58301.5,
        "mjd_stop": 58302.5,
        "job_id": job_id,
        "result_url": result_url,
        "service": "RIKEN/JAXA MAXI on-demand process",
        "source_observation": {
            "date_obs": source_header.get("DATE-OBS"),
            "date_end": source_header.get("DATE-END"),
            "exposure_s": exposure_source,
            "backscal": backscal_source,
        },
        "background_observation": {
            "exposure_s": exposure_background,
            "backscal": backscal_background,
        },
        "background_scale_in_source_counts": background_scale,
        "display_binning": {
            "energy_min_keV": energy_min,
            "energy_max_keV": energy_max,
            "bin_width_keV": bin_width,
            "selection": (
                "EBOUNDS channel centre, rounded to 1e-6 keV, in half-open "
                "display bin"
            ),
        },
        "input_files": {
            source_path.name: _sha256(source_path),
            background_path.name: _sha256(background_path),
            response_path.name: _sha256(response_path),
        },
        "interpretation": (
            "Background-subtracted MAXI/GSC detector count-rate density. "
            "It is a real-data visual reference, not a NICER-calibrated fit "
            "target and not included in the EvoXRB C-statistic."
        ),
        "acknowledgement": (
            "This research has made use of MAXI data provided by RIKEN, "
            "JAXA and the MAXI team."
        ),
        "citation": (
            "Matsuoka et al. (2009), The MAXI Mission on the ISS, "
            "PASJ 61, 999"
        ),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--energy-min", type=float, default=2.0)
    parser.add_argument("--energy-max", type=float, default=10.0)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--result-url", default=DEFAULT_RESULT_URL)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = convert_reference(
        source_path=args.source.resolve(),
        background_path=args.background.resolve(),
        response_path=args.response.resolve(),
        output_path=args.output.resolve(),
        provenance_path=args.provenance.resolve(),
        energy_min=float(args.energy_min),
        energy_max=float(args.energy_max),
        bin_width=float(args.bin_width),
        label=str(args.label),
        result_url=str(args.result_url),
        job_id=str(args.job_id),
    )
    print(f"Wrote {len(rows)} real MAXI/GSC display bins to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
