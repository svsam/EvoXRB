from __future__ import annotations

import base64
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_blog_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_blog_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_blog_bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_blog_bundle)


def test_extract_png_frames_requires_contiguous_png_payloads() -> None:
    first = b"\x89PNG\r\n\x1a\nfirst"
    second = b"\x89PNG\r\n\x1a\nsecond"
    fragment = (
        'frames[0] = "data:image/png;base64,'
        + base64.b64encode(first).decode("ascii")
        + '";\nframes[1] = "data:image/png;base64,'
        + base64.b64encode(second).decode("ascii")
        + '";'
    )

    assert build_blog_bundle.extract_png_frames(fragment) == [first, second]


def test_public_summary_drops_private_artifact_paths() -> None:
    raw = {
        "profile": "smoke",
        "epoch_id": "E08",
        "continuum": "powerlaw",
        "animation": r"C:\Users\person\private\replay.html",
        "checkpoint": r"C:\Users\person\private\checkpoint.npz",
        "ga": {
            "best_score": 12.0,
            "generations": 1,
            "seed": 9_554_544_384_653_782_853,
        },
        "scipy_polish": {"score": 10.0, "success": True},
        "reference": None,
    }

    public = build_blog_bundle.public_summary(raw, 2)
    encoded = json.dumps(public)
    assert "C:\\Users" not in encoded
    assert "animation" not in public
    assert "checkpoint" not in public
    assert public["reference_used"] is False
    assert public["reference"] is None
    assert public["frames"] == 2
    assert public["ga"]["seed"] == "9554544384653782853"


def test_public_summary_preserves_an_https_reference_source() -> None:
    public = build_blog_bundle.public_summary(
        {
            "reference": {
                "label": "MAXI/GSC reference",
                "source": "https://maxi.riken.jp/example",
                "points": 16,
            }
        },
        25,
    )

    assert public["reference_used"] is True
    assert public["reference"]["source"] == "https://maxi.riken.jp/example"


def test_compiled_blog_bundle_is_portable_and_complete() -> None:
    bundle = ROOT / "website" / "evoxrb-v0.2.0"
    index = (bundle / "index.html").read_text(encoding="utf-8")
    replay = (bundle / "assets" / "replay" / "index.html").read_text(
        encoding="utf-8"
    )
    bundle_readme = (bundle / "README.md").read_text(encoding="utf-8")
    summary = json.loads(
        (bundle / "assets" / "e08-run-summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (bundle / "asset-manifest.json").read_text(encoding="utf-8")
    )
    frames = sorted((bundle / "assets" / "replay" / "frames").glob("frame-*.png"))

    assert len(frames) == 25
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in frames)
    assert summary["frames"] == 25
    assert summary["reference_used"] is True
    assert summary["reference"]["points"] == 16
    assert summary["reference"]["source"].startswith("https://maxi.riken.jp/")
    assert summary["ga"]["converged"] is False
    assert summary["ga"]["seed"] == "9554544384653782853"
    assert "C:\\Users" not in json.dumps(summary)
    assert 'sandbox="allow-scripts"' in index
    assert "assets/replay/index.html" in index
    assert "Content-Security-Policy" in replay
    assert "manifest.js" in replay and "replay.js" in replay
    assert 'id="replay-status" aria-live' not in replay
    assert 'id="replay-announcement"' in replay
    assert 'aria-live="polite"' in replay
    assert "--output results/animations/E08_ga_spectra.html" in bundle_readme
    assert "--comparison-output results/animations/E08_ga_comparison.png" in (
        bundle_readme
    )
    assert "assets/replay/frames/frame-024.png" in manifest["files"]
    assert "assets/maxi-j1820-reference.csv" in manifest["files"]
    assert "assets/maxi-j1820-reference.provenance.json" in manifest["files"]
    assert "assets/maxi-j1820-reference.request.conf" in manifest["files"]


def test_latex_article_has_balanced_structure() -> None:
    article = (
        ROOT / "website" / "evoxrb-v0.2.0" / "article" / "evoxrb-v0.2.0.tex"
    ).read_text(encoding="utf-8")

    assert "\\documentclass[11pt]{article}" in article
    assert "\\begin{document}" in article
    assert article.rstrip().endswith("\\end{document}")

    environment_stack: list[str] = []
    for action, environment in re.findall(r"\\(begin|end)\{([^}]+)\}", article):
        if action == "begin":
            environment_stack.append(environment)
            continue
        assert environment_stack, f"unexpected \\end{{{environment}}}"
        opened = environment_stack.pop()
        assert opened == environment, (
            f"environment mismatch: \\begin{{{opened}}} closed by "
            f"\\end{{{environment}}}"
        )

    assert environment_stack == []

    brace_depth = 0
    for line in article.splitlines():
        content = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        for position, character in enumerate(content):
            escaped = position > 0 and content[position - 1] == "\\"
            if character == "{" and not escaped:
                brace_depth += 1
            elif character == "}" and not escaped:
                brace_depth -= 1
            assert brace_depth >= 0, "unexpected closing brace in LaTeX source"

    assert brace_depth == 0
