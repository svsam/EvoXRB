# EvoXRB

EvoXRB is a **Synthetic / NICER-inspired** case study in response-folded X-ray
spectral fitting, global optimization, posterior inference, and timing. It is
implemented entirely with native Windows Python 3.14 packages—there is no
HEASoft, XSPEC, PyXspec, WSL, mission download, or notebook-only logic.

This repository does **not** reduce or analyze real NICER observations and does
not estimate parameters of MAXI J1820+070. The response, absorption, disk-like
spectrum, power law, Comptonization cutoff surrogate, background, and timing signals are
transparent educational approximations with known injected truth. They are not
exact `TBabs`, `diskbb`, or `nthcomp` implementations.

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

Machine-readable aggregate tables are written to `results/`; selected figures
to `figures/generated/`; and the portfolio report to
`reports/generated/case_study.md`. Large reproducible spectra, chains, and
checkpoints are ignored by Git. Every report and scientific plot visibly says
“Synthetic / NICER-inspired.”

The fixed phase/date anchors follow the published evolution only as context:
[Li et al.](https://arxiv.org/html/2407.08421v2). The selected soft/hard timing
band design is publication-inspired: [Stiele & Kong](https://arxiv.org/abs/1912.07625).
