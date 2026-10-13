"""Implied-volatility surfaces.

The only primitive is the normalised call. ``docs/DESIGN.md`` section 5.
Linear, cubic-spline, and raw SVI interpolation are the Phase 1 surfaces.
"""

from volsurface.surface.linear import LinearSurface, linear_surface
from volsurface.surface.model import ModelSurface
from volsurface.surface.spline import SplineSurface, spline_surface
from volsurface.surface.svi import VOGT_2014, RawSVI, SVISurface, svi_surface

__all__ = [
    "LinearSurface",
    "ModelSurface",
    "RawSVI",
    "SVISurface",
    "SplineSurface",
    "VOGT_2014",
    "linear_surface",
    "spline_surface",
    "svi_surface",
]
