"""Parameter definitions and normalized search-space transformations.

The genetic algorithm always operates on a unit hypercube.  Physical values are
decoded only at the objective boundary, which keeps the optimizer independent of
the very different scales used by spectral temperatures and normalizations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np


Scale: TypeAlias = Literal["linear", "log10"]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One bounded physical parameter represented by a normalized gene.

    Parameters
    ----------
    name:
        Unique parameter name used in decoded dictionaries.
    lower, upper:
        Inclusive physical bounds.  Log-scaled parameters require positive
        bounds.
    scale:
        ``"linear"`` or ``"log10"``.  The latter makes a uniform gene map to
        a value that is uniform in base-10 logarithm.
    """

    name: str
    lower: float
    upper: float
    scale: Scale = "linear"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be a non-empty string")
        if self.scale not in ("linear", "log10"):
            raise ValueError("scale must be 'linear' or 'log10'")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError(f"bounds for {self.name!r} must be finite")
        if self.lower >= self.upper:
            raise ValueError(f"lower bound must be below upper bound for {self.name!r}")
        if self.scale == "log10" and self.lower <= 0.0:
            raise ValueError(f"log10 parameter {self.name!r} requires positive bounds")

    @property
    def transformed_lower(self) -> float:
        """Lower bound in the parameter's interpolation coordinate."""

        if self.scale == "log10":
            return float(np.log10(self.lower))
        return float(self.lower)

    @property
    def transformed_upper(self) -> float:
        """Upper bound in the parameter's interpolation coordinate."""

        if self.scale == "log10":
            return float(np.log10(self.upper))
        return float(self.upper)

    def decode(self, gene: float, *, clip: bool = False) -> float:
        """Decode one normalized gene into its physical value."""

        value = float(gene)
        if not np.isfinite(value):
            raise ValueError(f"gene for {self.name!r} must be finite")
        if clip:
            value = float(np.clip(value, 0.0, 1.0))
        elif not 0.0 <= value <= 1.0:
            raise ValueError(f"gene for {self.name!r} must lie in [0, 1]")

        transformed = self.transformed_lower + value * (
            self.transformed_upper - self.transformed_lower
        )
        if self.scale == "log10":
            return float(10.0**transformed)
        return float(transformed)

    def encode(self, physical_value: float, *, clip: bool = False) -> float:
        """Encode one physical value as a normalized gene."""

        value = float(physical_value)
        if not np.isfinite(value):
            raise ValueError(f"value for {self.name!r} must be finite")
        if self.scale == "log10" and value <= 0.0:
            raise ValueError(f"value for log10 parameter {self.name!r} must be positive")
        if clip:
            value = float(np.clip(value, self.lower, self.upper))
        elif not self.lower <= value <= self.upper:
            raise ValueError(
                f"value for {self.name!r} must lie in [{self.lower}, {self.upper}]"
            )

        transformed = float(np.log10(value)) if self.scale == "log10" else value
        return float(
            (transformed - self.transformed_lower)
            / (self.transformed_upper - self.transformed_lower)
        )

    def contains(self, value: float) -> bool:
        """Return whether a finite physical value lies inside the bounds."""

        return bool(np.isfinite(value) and self.lower <= value <= self.upper)


@dataclass(frozen=True, slots=True, init=False)
class SearchSpace:
    """Ordered collection of :class:`ParameterSpec` objects.

    ``SearchSpace`` accepts either ``SearchSpace(spec1, spec2)`` or
    ``SearchSpace([spec1, spec2])`` for convenient use from configuration code.
    """

    parameters: tuple[ParameterSpec, ...]

    def __init__(self, *parameters: ParameterSpec | Sequence[ParameterSpec]) -> None:
        if len(parameters) == 1 and not isinstance(parameters[0], ParameterSpec):
            values = tuple(parameters[0])
        else:
            values = tuple(parameters)  # type: ignore[arg-type]
        if not values:
            raise ValueError("a search space requires at least one parameter")
        if not all(isinstance(item, ParameterSpec) for item in values):
            raise TypeError("all search-space entries must be ParameterSpec instances")
        names = [item.name for item in values]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        object.__setattr__(self, "parameters", values)

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    def __getitem__(self, item: int | str) -> ParameterSpec:
        if isinstance(item, str):
            for parameter in self.parameters:
                if parameter.name == item:
                    return parameter
            raise KeyError(item)
        return self.parameters[item]

    @property
    def ndim(self) -> int:
        return len(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def unit_bounds(self) -> tuple[tuple[float, float], ...]:
        return ((0.0, 1.0),) * self.ndim

    @property
    def physical_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((parameter.lower, parameter.upper) for parameter in self.parameters)

    @property
    def signature(self) -> tuple[tuple[str, float, float, str], ...]:
        """Stable description used to reject incompatible checkpoints."""

        return tuple(
            (parameter.name, parameter.lower, parameter.upper, parameter.scale)
            for parameter in self.parameters
        )

    def decode(
        self,
        genes: Sequence[float] | np.ndarray,
        *,
        clip: bool = False,
    ) -> dict[str, float]:
        """Decode one gene vector to an ordered physical-parameter dictionary."""

        vector = np.asarray(genes, dtype=float)
        if vector.shape != (self.ndim,):
            raise ValueError(f"expected gene vector with shape ({self.ndim},)")
        return {
            parameter.name: parameter.decode(gene, clip=clip)
            for parameter, gene in zip(self.parameters, vector, strict=True)
        }

    def encode(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray,
        *,
        clip: bool = False,
    ) -> np.ndarray:
        """Encode a mapping or an ordered physical vector into unit genes."""

        if isinstance(values, Mapping):
            missing = set(self.names).difference(values)
            if missing:
                raise KeyError(f"missing parameters: {', '.join(sorted(missing))}")
            physical = [values[name] for name in self.names]
        else:
            physical = np.asarray(values, dtype=float)
            if physical.shape != (self.ndim,):
                raise ValueError(f"expected physical vector with shape ({self.ndim},)")
        return np.asarray(
            [
                parameter.encode(value, clip=clip)
                for parameter, value in zip(self.parameters, physical, strict=True)
            ],
            dtype=float,
        )

    def contains(self, values: Mapping[str, float] | Sequence[float] | np.ndarray) -> bool:
        """Return whether all supplied physical values lie within the space."""

        try:
            if isinstance(values, Mapping):
                physical = [values[name] for name in self.names]
            else:
                physical = np.asarray(values, dtype=float)
                if physical.shape != (self.ndim,):
                    return False
            return all(
                parameter.contains(value)
                for parameter, value in zip(self.parameters, physical, strict=True)
            )
        except (KeyError, TypeError, ValueError):
            return False

    def clip_genes(self, genes: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return a finite gene vector clipped to the unit hypercube."""

        vector = np.asarray(genes, dtype=float)
        if vector.shape != (self.ndim,):
            raise ValueError(f"expected gene vector with shape ({self.ndim},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("genes must be finite")
        return np.clip(vector, 0.0, 1.0)

