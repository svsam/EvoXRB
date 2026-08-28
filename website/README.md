# EvoXRB publication bundles

`evoxrb-v0.2.0/` is a compiled, framework-independent blog bundle. It is
designed to be copied as one directory into an ordinary static website
repository. All runtime URLs are relative and all required CSS, JavaScript,
images, replay frames, metadata, and LaTeX source are included.

Rebuild it after generating the E08 animation:

```powershell
python -m evoxrb animate --profile smoke --epoch E08 --reference-csv data/reference/maxi_j1820p070_mjd58302.csv --output results/animations/E08_ga_spectra.html --comparison-output results/animations/E08_ga_comparison.png
python scripts/build_blog_bundle.py
```

Preview it over HTTP rather than opening `index.html` directly:

```powershell
python -m http.server 8000 --directory website/evoxrb-v0.2.0
```

Then open `http://localhost:8000/`.
