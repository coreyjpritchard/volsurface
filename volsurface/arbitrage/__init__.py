"""Static arbitrage checks.

``check_surface`` is pointwise on a grid. ``check_quotes`` is a linear program:
does a martingale measure sit inside the bid and ask? ``docs/DESIGN.md`` section 6.
"""

from volsurface.arbitrage.check import check_quotes, check_surface

__all__ = ["check_quotes", "check_surface"]
