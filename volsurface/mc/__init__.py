"""Monte Carlo utilities: seeds, a standard error kept off the price array, and a shared-path grid.

Geometric Brownian motion lives in :mod:`volsurface.models.gbm`. The grid
pricer simulates once and prices every strike and expiry from those paths.
"""

from volsurface.mc.grid import price_grid
from volsurface.mc.pricing import price_with_error
from volsurface.mc.rng import as_seed_sequence, generator, spawn

__all__ = ["as_seed_sequence", "generator", "price_grid", "price_with_error", "spawn"]
