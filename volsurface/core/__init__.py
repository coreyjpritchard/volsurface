"""Shared types and protocols.

The mathematical objects live in :mod:`volsurface.core.types`. The interfaces
live in :mod:`volsurface.core.protocols`. ``docs/DESIGN.md`` is the source of
truth; these modules implement sections 3 and 4 of that file.
"""

from volsurface.core.market import market_data_from_snapshot
from volsurface.core.protocols import (
    Fitter,
    GreeksEngine,
    Model,
    Parametric,
    SupportsCharFunc,
    SupportsSimulate,
    Surface,
    capabilities,
)
from volsurface.core.types import (
    ArbReport,
    ArbTol,
    Contract,
    Curves,
    Digital,
    Domain,
    FitResult,
    Forward,
    GreeksResult,
    MarketData,
    Param,
    ParamSpec,
    Paths,
    PriceResult,
    Quote,
    Spot,
    Vanilla,
)

__all__ = [
    "ArbReport",
    "ArbTol",
    "Contract",
    "Curves",
    "Digital",
    "Domain",
    "FitResult",
    "Fitter",
    "Forward",
    "GreeksEngine",
    "GreeksResult",
    "MarketData",
    "Model",
    "Param",
    "ParamSpec",
    "Parametric",
    "Paths",
    "PriceResult",
    "Quote",
    "Spot",
    "SupportsCharFunc",
    "SupportsSimulate",
    "Surface",
    "Vanilla",
    "capabilities",
    "market_data_from_snapshot",
]
