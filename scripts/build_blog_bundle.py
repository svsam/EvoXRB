"""Compile the EvoXRB v0.2.0 blog bundle from generated GA artifacts.

The Matplotlib HTML writer intentionally produces a self-contained fragment
with base64 frames and inline JavaScript.  That is convenient for a local
scientific replay, but awkward to paste into a website with a strict content
security policy.  This compiler extracts the PNG frames, writes an accessible
external-script player, copies the comparison plot, and emits sanitized public
metadata containing no workstation paths.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "website" / "evoxrb-v0.2.0"
DEFAULT_ANIMATION = ROOT / "results" / "animations" / "E08_ga_spectra.html"
DEFAULT_COMPARISON = ROOT / "results" / "animations" / "E08_ga_comparison.png"
DEFAULT_SUMMARY = (
    ROOT / "results" / "animations" / "E08_ga_spectra.summary.json"
)
DEFAULT_REFERENCE_CSV = (
    ROOT / "data" / "reference" / "maxi_j1820p070_mjd58302.csv"
)
DEFAULT_REFERENCE_PROVENANCE = (
    ROOT
    / "data"
    / "reference"
    / "maxi_j1820p070_mjd58302.provenance.json"
)
DEFAULT_REFERENCE_REQUEST = (
    ROOT / "data" / "reference" / "maxi_j1820p070_mjd58302.request.conf"
)

_FRAME_PATTERN = re.compile(
    r"frames\[(\d+)\]\s*=\s*\"data:image/png;base64,(.*?)\"\s*;?",
    flags=re.DOTALL,
)
# A drive path starts at a token boundary.  Without the look-behind, the
# trailing ``s:/`` in an ordinary ``https://`` URL is a false positive.
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")


def _write_bytes_if_changed(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    path.write_bytes(payload)


def _write_text_if_changed(path: Path, payload: str) -> None:
    _write_bytes_if_changed(path, payload.encode("utf-8"))


def extract_png_frames(fragment: str) -> list[bytes]:
    """Decode contiguous Matplotlib HTMLWriter PNG frames."""

    matches = _FRAME_PATTERN.findall(fragment)
    if not matches:
        raise ValueError("animation HTML contains no embedded PNG frames")
    indexed: dict[int, bytes] = {}
    for raw_index, raw_payload in matches:
        index = int(raw_index)
        compact = re.sub(r"\\?\s+", "", raw_payload)
        try:
            frame = base64.b64decode(compact, validate=True)
        except ValueError as error:
            raise ValueError(f"animation frame {index} is not valid base64") from error
        if not frame.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"animation frame {index} is not a PNG")
        if index in indexed:
            raise ValueError(f"animation frame {index} is duplicated")
        indexed[index] = frame
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise ValueError("animation frame indices must be contiguous from zero")
    return [indexed[index] for index in expected]


def public_summary(summary: dict[str, Any], frame_count: int) -> dict[str, Any]:
    """Select publication-safe numerical metadata from a run summary."""

    ga = dict(summary.get("ga", {}))
    scipy = dict(summary.get("scipy_polish", {}))
    raw_seed = ga.get("seed")
    raw_reference = summary.get("reference")
    reference: dict[str, Any] | None = None
    if isinstance(raw_reference, dict):
        raw_source = raw_reference.get("source")
        public_source = (
            raw_source
            if isinstance(raw_source, str)
            and raw_source.startswith(("https://", "http://"))
            else None
        )
        reference = {
            "label": raw_reference.get("label"),
            "source": public_source,
            "points": raw_reference.get("points"),
            "notice": raw_reference.get("notice"),
            "input_representation": raw_reference.get("input_representation"),
        }
    payload = {
        "schema_version": 1,
        "project_version": "0.2.0",
        "scope": "Synthetic / NICER-inspired",
        "profile": summary.get("profile"),
        "config_digest": summary.get("config_digest"),
        "objective_signature": summary.get("objective_signature"),
        "epoch_id": summary.get("epoch_id"),
        "continuum": summary.get("continuum"),
        "frames": int(frame_count),
        "reference_used": reference is not None,
        "reference": reference,
        "ga": {
            "best_parameters": ga.get("best_parameters"),
            "best_score": ga.get("best_score"),
            "generations": ga.get("generations"),
            "evaluations": ga.get("evaluations"),
            "converged": ga.get("converged"),
            "stop_reason": ga.get("stop_reason"),
            # Seeds may use the full unsigned 64-bit range.  A decimal string
            # survives JSON.parse without JavaScript's Number precision loss.
            "seed": str(raw_seed) if raw_seed is not None else None,
        },
        "scipy_polish": {
            "parameters": scipy.get("parameters"),
            "score": scipy.get("score"),
            "success": scipy.get("success"),
            "method": scipy.get("method"),
            "iterations": scipy.get("iterations"),
            "evaluations": scipy.get("evaluations"),
        },
        "notice": (
            "The replay target is synthetic. The real MAXI/GSC reference is a "
            "visual overlay only and is not included in the C-statistic."
            if reference is not None
            else "The replay target is synthetic; no reference was supplied."
        ),
    }
    encoded = json.dumps(payload, sort_keys=True)
    if _WINDOWS_PATH.search(encoded):
        raise ValueError("public summary unexpectedly contains an absolute Windows path")
    return payload


class _LocalReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attribute = "src" if tag in {"img", "iframe", "script"} else "href"
        if tag not in {"a", "img", "iframe", "link", "script"}:
            return
        values = dict(attrs)
        value = values.get(attribute)
        if value:
            self.references.append(value)


def _validate_html(path: Path, bundle: Path) -> None:
    parser = _LocalReferenceParser()
    parser.feed(path.read_text(encoding="utf-8"))
    for reference in parser.references:
        if reference.startswith(("#", "http://", "https://", "mailto:", "data:")):
            continue
        if reference.startswith(("/", "\\")) or _WINDOWS_PATH.search(reference):
            raise ValueError(f"{path.name} contains a non-portable path: {reference}")
        destination = (path.parent / reference.split("#", 1)[0]).resolve()
        try:
            destination.relative_to(bundle.resolve())
        except ValueError as error:
            raise ValueError(f"{path.name} links outside the bundle: {reference}") from error
        if not destination.exists():
            raise FileNotFoundError(f"missing local website asset: {reference}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_bundle(
    *,
    animation_path: Path,
    comparison_path: Path,
    summary_path: Path,
    reference_csv_path: Path,
    reference_provenance_path: Path,
    reference_request_path: Path,
    bundle: Path,
) -> dict[str, Any]:
    """Compile generated scientific artifacts into the static site bundle."""

    for path in (
        animation_path,
        comparison_path,
        summary_path,
        reference_csv_path,
        reference_provenance_path,
        reference_request_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    required_templates = (
        bundle / "index.html",
        bundle / "assets" / "site.css",
        bundle / "assets" / "site.js",
        bundle / "assets" / "replay" / "index.html",
        bundle / "assets" / "replay" / "replay.css",
        bundle / "assets" / "replay" / "replay.js",
        bundle / "article" / "evoxrb-v0.2.0.tex",
    )
    missing_templates = [path for path in required_templates if not path.is_file()]
    if missing_templates:
        joined = ", ".join(str(path) for path in missing_templates)
        raise FileNotFoundError(f"website source files are missing: {joined}")

    frames = extract_png_frames(animation_path.read_text(encoding="utf-8"))
    frame_directory = bundle / "assets" / "replay" / "frames"
    expected_names = {f"frame-{index:03d}.png" for index in range(len(frames))}
    for index, frame in enumerate(frames):
        _write_bytes_if_changed(frame_directory / f"frame-{index:03d}.png", frame)
    if frame_directory.exists():
        for stale in frame_directory.glob("frame-*.png"):
            if stale.name not in expected_names:
                stale.unlink()

    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    safe_summary = public_summary(raw_summary, len(frames))

    replay_manifest = {
        "frames": [f"frames/frame-{index:03d}.png" for index in range(len(frames))],
        "generation_max": len(frames) - 1,
        "default_fps": 8,
        "alt_prefix": (
            "EvoXRB synthetic GA spectral fit with real MAXI/GSC reference at "
            "generation"
            if safe_summary["reference_used"]
            else "EvoXRB synthetic GA spectral fit at generation"
        ),
    }
    _write_text_if_changed(
        bundle / "assets" / "replay" / "manifest.js",
        "window.EVOXRB_REPLAY = Object.freeze("
        + json.dumps(replay_manifest, separators=(",", ":"))
        + ");\n",
    )

    comparison_destination = bundle / "assets" / "e08-ga-comparison.png"
    comparison_destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        not comparison_destination.exists()
        or _sha256(comparison_destination) != _sha256(comparison_path)
    ):
        shutil.copy2(comparison_path, comparison_destination)

    publication_assets = (
        (reference_csv_path, bundle / "assets" / "maxi-j1820-reference.csv"),
        (
            reference_provenance_path,
            bundle / "assets" / "maxi-j1820-reference.provenance.json",
        ),
        (
            reference_request_path,
            bundle / "assets" / "maxi-j1820-reference.request.conf",
        ),
    )
    for source, destination in publication_assets:
        payload = source.read_bytes()
        if _WINDOWS_PATH.search(payload.decode("utf-8")):
            raise ValueError(f"publication asset contains a Windows path: {source}")
        _write_bytes_if_changed(destination, payload)

    _write_text_if_changed(
        bundle / "assets" / "e08-run-summary.json",
        json.dumps(safe_summary, indent=2, sort_keys=True) + "\n",
    )

    _validate_html(bundle / "index.html", bundle)
    _validate_html(bundle / "assets" / "replay" / "index.html", bundle)

    manifest_files = sorted(
        path
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "asset-manifest.json"
    )
    asset_manifest = {
        "schema_version": 1,
        "project_version": "0.2.0",
        "files": {
            path.relative_to(bundle).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in manifest_files
        },
    }
    _write_text_if_changed(
        bundle / "asset-manifest.json",
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n",
    )
    return asset_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animation", type=Path, default=DEFAULT_ANIMATION)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument(
        "--reference-provenance", type=Path, default=DEFAULT_REFERENCE_PROVENANCE
    )
    parser.add_argument(
        "--reference-request", type=Path, default=DEFAULT_REFERENCE_REQUEST
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = compile_bundle(
        animation_path=args.animation.resolve(),
        comparison_path=args.comparison.resolve(),
        summary_path=args.summary.resolve(),
        reference_csv_path=args.reference_csv.resolve(),
        reference_provenance_path=args.reference_provenance.resolve(),
        reference_request_path=args.reference_request.resolve(),
        bundle=args.bundle.resolve(),
    )
    total_bytes = sum(item["bytes"] for item in manifest["files"].values())
    print(
        f"Compiled {len(manifest['files'])} files "
        f"({total_bytes / (1024 * 1024):.2f} MiB) into {args.bundle.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
