# EvoXRB 0.2.0

## [Blog post](https://svsam.com/blog/evoxrb-genetic-algorithms-x-ray-astronomy/)

## The problem

Response-folded X-ray spectra produce bounded, correlated fitting problems where
a local optimiser can settle into the wrong basin. EvoXRB asks when a
population-based genetic algorithm helps, whether a local polishing stage is
still necessary, and how an apparently good optimum can remain scientifically
wrong when the forward model is misspecified.

## The approach

The project generates a twelve-epoch synthetic outburst with known injected
truth, folds approximate disk and continuum models through a transparent
NICER-inspired response, adds Poisson counts and background, and fits the result
with both a custom real-valued GA and deterministic multi-start SciPy. The best
global solution seeds posterior inference, while a separate synthetic timing
pipeline tests recovery of injected QPO-like signals.

![Synthetic optimiser comparison across the committed smoke run](figures/generated/optimizer_comparison.png)

## What I found

The committed smoke run makes the central result quite stark. Its reduced-budget
raw GA rows often stop far from the best solution, but GA followed by bounded
SciPy polishing reaches essentially the same C-statistic as multi-start SciPy.
For the correct 512-second model, the polished and multi-start fits both report
zero failures in the committed recovery table; fitting cutoff-generated data
with the simpler power law still leaves systematic parameter bias even after the
numerical optimum is found. The injected 5 Hz signal in epoch E08 is recovered at
`5.03 Hz` and labelled type-C-like under the declared synthetic criteria.

Those are smoke-profile integration results, not measurements of MAXI J1820+070
and not a completed full numerical campaign. They support the software design
and the importance of local polishing and model checking; the opt-in full profile
is where the stated campaign-level acceptance criteria belong.

## Current scope

EvoXRB is a **Synthetic / NICER-inspired** case study in response-folded X-ray
spectral fitting, global optimization, posterior inference, and timing. It is
implemented entirely with native Windows Python 3.14 packages—there is no
HEASoft, XSPEC, PyXspec, WSL, mission download, or notebook-only logic.

This repository does **not** reduce or fit real NICER observations and does not
estimate physical parameters of MAXI J1820+070. The response, absorption, disk-like
spectrum, power law, Comptonization cutoff surrogate, background, and timing signals are
transparent educational approximations with known injected truth. They are not
exact `TBabs`, `diskbb`, or `nthcomp` implementations.

A separately labelled, background-subtracted MAXI/GSC spectrum of the real
MAXI J1820+070 is bundled only as a cross-instrument visual reference. It is
never included in the synthetic NICER-inspired likelihood.

## Quick start

```powershell
python -m pip install -e ".[test]"
python -m evoxrb run-case-study --profile smoke
python -m pytest
```

The smoke profile is a deterministic, reduced CI run. The opt-in full profile
uses the specified 192-member populations, five GA seeds, up to 300 generations,
20 SciPy starts, 20 Poisson realizations per exposure, 200 timing bootstraps,
and posterior convergence/fallback rules:

```powershell
python -m evoxrb run-case-study --profile full --resume
```

The five-seed 100 ks numerical acceptance test is also opt-in:

```powershell
$env:EVOXRB_RUN_FULL = "1"
python -m pytest tests/test_full_acceptance.py
```

Individual stages are composable:

```powershell
python -m evoxrb simulate --profile smoke
python -m evoxrb fit --method scipy --profile smoke
python -m evoxrb fit --method ga --profile smoke --resume
python -m evoxrb infer --profile smoke --resume
python -m evoxrb timing --profile smoke --resume
python -m evoxrb report --profile smoke
```

## Animated genetic-algorithm fitting

Version 0.2.0 can run one genuine real-valued GA fit while reporting every
evaluated generation, replay the evolving response-folded spectra, and save a
best-fit comparison. The default HTML output is self-contained and includes
play, pause, step, loop, and timeline controls:

```powershell
python -m evoxrb animate --profile smoke --epoch E08
```

The command writes the replay and comparison under `results/animations/`. Use
the output extension to select a writer, or request a live Matplotlib window
while the calculations run:

```powershell
python -m evoxrb animate --profile smoke --epoch E08 --output results/animations/E08.gif
python -m evoxrb animate --profile smoke --epoch E08 --live
```

HTML and GIF work without an external encoder. MP4 export is available when
FFmpeg is installed. The GA uses tournament selection, simulated-binary
crossover, bounded polynomial mutation, elitism, diversity immigrants, and
convergence checks; its final solution is also polished with bounded SciPy
L-BFGS-B and shown in the saved comparison.

### Observational reference

The repository includes 16 real MAXI/GSC display bins spanning MJD
58301.5-58302.5 and 2-10 keV. They were derived from the source, background and
response products generated by the official RIKEN/JAXA MAXI on-demand service.
The provenance, input hashes and exact request are under `data/reference/`.

Graph the bundled observation beside the evolving synthetic fit with:

```powershell
python -m evoxrb animate --reference-csv data/reference/maxi_j1820p070_mjd58302.csv
```

The preferred CSV columns are `energy_keV`, `count_rate_density`, and optional
`count_rate_error`, with optional constant `label` and `source` columns. A
counts-based table can instead provide `counts`, `exposure_s`, and either
`bin_width_keV` or `energy_low_keV` plus `energy_high_keV`. The loader preserves
the reference's native grid and provenance.

This is deliberately a visual comparison only. MAXI/GSC and the educational
NICER-inspired detector have different responses, so their count-rate densities
are not numerically interchangeable. A physical observational fit requires its
matching response, background, exposure, calibration and likelihood.

To regenerate the CSV from the official OGIP files, install the optional
`real-data` dependency and run `scripts/prepare_maxi_reference.py`. The required
MAXI acknowledgement and citation are recorded in `data/reference/README.md`.

The repository currently commits deterministic smoke-profile artifacts and a
passing artifact-validation record. The larger five-seed, 192-member campaign is
implemented but deliberately opt-in because of its computational cost.


The fixed phase/date anchors follow the published evolution only as context:
[Li et al.](https://arxiv.org/html/2407.08421v2). The selected soft/hard timing
band design is publication-inspired: [Stiele & Kong](https://arxiv.org/abs/1912.07625).
