"""EvoXRB: a synthetic, NICER-inspired optimisation case study.

Nothing in this package should be interpreted as a reduction or analysis of
real NICER observations.  The numerical models are deliberately transparent
educational approximations.
"""

from __future__ import annotations

__version__ = "0.2.0"
SYNTHETIC_LABEL = "Synthetic / NICER-inspired"

from .genetic import (
    GARunResult,
    GenerationSnapshot,
    GeneticOptimizer,
    load_ga_checkpoint_history,
)
from .instrument import InstrumentResponse
from .models import SpectrumModel
from .objective import Objective
from .parameters import ParameterSpec, SearchSpace
from .reference import ReferenceSpectrum, load_reference_spectrum_csv
from .timing import TimingResult
from .types import EpochTruth, PosteriorResult, SyntheticSpectrum

__all__ = [
    "EpochTruth",
    "GARunResult",
    "GenerationSnapshot",
    "GeneticOptimizer",
    "InstrumentResponse",
    "load_ga_checkpoint_history",
    "Objective",
    "ParameterSpec",
    "PosteriorResult",
    "ReferenceSpectrum",
    "SYNTHETIC_LABEL",
    "SearchSpace",
    "SpectrumModel",
    "SyntheticSpectrum",
    "TimingResult",
    "__version__",
    "load_reference_spectrum_csv",
]
