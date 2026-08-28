# EvoXRB v0.2.0 website bundle

This directory is the copy-ready continuation of the first EvoXRB project
post. It contains no framework, package-manager, CDN, or server-side
dependency.

## Publish

Copy this entire directory into the target website repository, for example:

```text
blog/
└── evoxrb-v0-2-0/
    ├── index.html
    ├── asset-manifest.json
    ├── assets/
    └── article/
```

The included stylesheet reproduces the visual language and `blogPost...` class
structure of the existing `svsam.com` article. If the target repository already
loads `css/dev.css` and `css/blog-post.css`, the local stylesheet can remain as
the self-contained option or be replaced with the site's normal relative links.

The navigation URLs in `index.html` are absolute public `svsam.com` URLs so the
bundle works at any directory depth. Change them if this is published under a
different site.

## Replay architecture

The main page embeds `assets/replay/index.html` in an iframe. The player:

- starts paused;
- is keyboard operable;
- uses external CSS and JavaScript;
- loads 25 ordinary PNG frames;
- has a strict local-only content-security policy;
- provides labels and a live generation announcement;
- falls back to the static comparison when JavaScript is unavailable.

The iframe is intentionally isolated from the host page. Serve the directory
over HTTP(S); some browsers restrict iframe scripts when a site is opened with
the `file://` protocol.

## Rebuild scientific assets

From the EvoXRB repository root:

```powershell
python -m evoxrb animate --profile smoke --epoch E08 --reference-csv data/reference/maxi_j1820p070_mjd58302.csv --output results/animations/E08_ga_spectra.html --comparison-output results/animations/E08_ga_comparison.png
python scripts/build_blog_bundle.py
```

The compiler extracts PNG frames from Matplotlib's self-contained replay,
copies the comparison figure, strips local paths from the public summary, and
regenerates `asset-manifest.json` with file sizes and SHA-256 hashes.

## LaTeX edition

The print source is `article/evoxrb-v0.2.0.tex`. With TeX Live, MiKTeX, or
another pdfLaTeX distribution installed:

```powershell
Set-Location website/evoxrb-v0.2.0/article
latexmk -pdf evoxrb-v0.2.0.tex
```

Alternatively, run `pdflatex evoxrb-v0.2.0.tex` twice. The source uses a
portable pdfLaTeX package set and shares the website comparison image through
`../assets/`.

## Scientific boundary

The fitted target remains synthetic / NICER-inspired. The purple reference
points are a real, background-subtracted MAXI/GSC spectrum of MAXI J1820+070
from MJD 58301.5-58302.5. They are a cross-instrument visual overlay and are
never included in the C-statistic. The derived CSV, official result URL, request
configuration, input hashes, acknowledgement and citation travel in `assets/`.
