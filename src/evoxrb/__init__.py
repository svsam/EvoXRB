"""EvoXRB: a synthetic, NICER-inspired optimisation case study.

Nothing in this package should be interpreted as a reduction or analysis of
real NICER observations.  The numerical models are deliberately transparent
educational approximations.
"""

from __future__ import annotations

__version__ = "0.1.0"
SYNTHETIC_LABEL = "Synthetic / NICER-inspired"

from .genetic import GARunResult, GeneticOptimizer
from .instrument import InstrumentResponse
from .models import SpectrumModel
from .objective import Objective
from .parameters import ParameterSpec, SearchSpace
from .timing import TimingResult
from .types import EpochTruth, PosteriorResult, SyntheticSpectrum

__all__ = [
    "EpochTruth",
    "GARunResult",
    "GeneticOptimizer",
    "InstrumentResponse",
    "Objective",
    "ParameterSpec",
    "PosteriorResult",
    "SYNTHETIC_LABEL",
    "SearchSpace",
    "SpectrumModel",
    "SyntheticSpectrum",
    "TimingResult",
    "__version__",
]
