"""Unconstrained coordinates for parameters.

Fitters search in ``R^d``. Sliders live on the bounded parameter. Both come
from :class:`~volsurface.core.types.ParamSpec`.
"""

from __future__ import annotations

import numpy as np

from volsurface.core.types import ParamSpec


def to_unconstrained_value(value: float, spec: ParamSpec) -> float:
    """Map a parameter value to an unconstrained real."""
    if spec.transform == "identity":
        return float(value)
    if spec.transform == "log":
        if value <= 0.0:
            raise ValueError(f"{spec.name} must be positive for a log transform, got {value}")
        return float(np.log(value))
    if spec.transform == "logit":
        span = spec.upper - spec.lower
        if span <= 0.0:
            raise ValueError(f"{spec.name} needs upper > lower")
        unit = (float(value) - spec.lower) / span
        unit = float(np.clip(unit, 1e-12, 1.0 - 1e-12))
        return float(np.log(unit / (1.0 - unit)))
    raise ValueError(f"unknown transform {spec.transform!r} on {spec.name}")


def from_unconstrained_value(x: float, spec: ParamSpec) -> float:
    """Map an unconstrained real back to the parameter's range."""
    if spec.transform == "identity":
        return float(np.clip(x, spec.lower, spec.upper))
    if spec.transform == "log":
        return float(np.exp(x))
    if spec.transform == "logit":
        # Sigmoid, then the affine map onto [lower, upper].
        unit = 1.0 / (1.0 + np.exp(-float(x)))
        return float(spec.lower + (spec.upper - spec.lower) * unit)
    raise ValueError(f"unknown transform {spec.transform!r} on {spec.name}")


def to_unconstrained(values: dict[str, float], specs: list[ParamSpec]) -> np.ndarray:
    """Stack ``values`` in the order of ``specs``."""
    return np.array(
        [to_unconstrained_value(values[spec.name], spec) for spec in specs], dtype=float
    )


def from_unconstrained(x: np.ndarray, specs: list[ParamSpec]) -> dict[str, float]:
    """Inverse of :func:`to_unconstrained`."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != len(specs):
        raise ValueError(f"expected {len(specs)} parameters, got {x.size}")
    return {spec.name: from_unconstrained_value(float(x[i]), spec) for i, spec in enumerate(specs)}
