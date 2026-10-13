"""Monte Carlo utilities: seeds, and prices that carry a standard error.

Geometric Brownian motion itself lives in :mod:`volsurface.models.gbm`
(Phase 1c). What is here is shared by every simulation.
"""

from volsurface.mc.pricing import price_with_error
from volsurface.mc.rng import as_seed_sequence, generator, spawn

__all__ = ["as_seed_sequence", "generator", "price_with_error", "spawn"]
