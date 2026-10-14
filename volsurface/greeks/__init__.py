"""Five ways to compute a greek.

Analytic Black-Scholes, a central finite difference, the pathwise derivative,
the likelihood ratio, and torch autodiff of the same paths the pathwise method
uses. ``docs/DESIGN.md`` section 8. Each engine is a ``GreeksEngine``.
``supports`` is what a caller checks before ``compute``.

Torch is an optional extra. :class:`Autodiff` is imported on first use, and
the class itself imports torch only when ``compute`` runs.
"""

from volsurface.greeks.analytic import Analytic
from volsurface.greeks.finite_difference import FiniteDifference
from volsurface.greeks.likelihood import LikelihoodRatio
from volsurface.greeks.pathwise import Pathwise

__all__ = ["Analytic", "Autodiff", "FiniteDifference", "LikelihoodRatio", "Pathwise"]


def __getattr__(name: str):
    if name == "Autodiff":
        from volsurface.greeks.autodiff import Autodiff

        return Autodiff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
