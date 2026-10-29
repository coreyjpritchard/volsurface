"""The z-rule of ``docs/DESIGN.md`` section 7.

A price test on a grid fails by chance if every node must lie inside a fixed
number of standard errors. The bounds below move with the node count.

Let ``z_i = (estimate_i - reference_i) / stderr_i``. When the nodes do not
share paths the ``z_i`` are treated as i.i.d. standard normal.

- ``max |z|`` is below ``Phi^{-1}(1 - 0.001 / (2 n))``. One normal exceeds
  that level with probability ``0.001 / n``, so a union bound puts the
  probability of any exceedance at most 0.1 percent.
- The fraction of nodes with ``|z| < 2`` lies between the 0.1 percent and
  99.9 percent quantiles of ``Binomial(n, p) / n``, where
  ``p = 2 Phi(2) - 1`` (about 0.9545).
- ``|mean z| < 3.5 / sqrt(n_eff)``. For independent nodes ``n_eff = n``.
  Nodes that share paths are one draw. Pass ``mean=False`` on that grid and
  check the mean on independent replications with
  :func:`check_replication_means`.

:func:`check_z` returns ``None`` when the selected bounds hold, and a sentence
with the numbers when one fails.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom, norm

# P(|Z| < 2) for a standard normal.
_WITHIN_2 = float(2.0 * norm.cdf(2.0) - 1.0)


def max_abs_bound(n: int) -> float:
    """``Phi^{-1}(1 - 0.001 / (2 n))``."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return float(norm.ppf(1.0 - 0.001 / (2.0 * n)))


def fraction_band(n: int) -> tuple[float, float]:
    """Lower and upper fractions for the count of nodes with ``|z| < 2``."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    lo = float(binom.ppf(0.001, n, _WITHIN_2)) / n
    hi = float(binom.ppf(0.999, n, _WITHIN_2)) / n
    return lo, hi


def mean_abs_bound(n_eff: float) -> float:
    """``3.5 / sqrt(n_eff)``."""
    if n_eff <= 0.0:
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    return 3.5 / math.sqrt(n_eff)


def check_z(
    z: np.ndarray,
    *,
    n_eff: float | None = None,
    mean: bool = True,
    fraction: bool = True,
) -> str | None:
    """Return None if ``z`` meets the z-rule, else a sentence naming the failure.

    ``n_eff`` is the sample size behind the mean. It defaults to the length of
    ``z``. Set ``mean=False`` when the nodes share paths. Set ``fraction=False``
    for the same reason: the count is binomial only when the nodes are
    independent.
    """
    scores = np.asarray(z, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("z is empty")
    if not np.all(np.isfinite(scores)):
        return "z-rule: a node is not finite"
    n = int(scores.size)
    max_abs = float(np.max(np.abs(scores)))
    cap = max_abs_bound(n)
    if max_abs >= cap:
        return f"z-rule max |z| is {max_abs:.4f}, bound {cap:.4f} at n={n}"
    if mean:
        effective = float(n if n_eff is None else n_eff)
        bound = mean_abs_bound(effective)
        centre = float(np.mean(scores))
        if abs(centre) >= bound:
            return (
                f"z-rule mean z is {centre:.4f}, bound {bound:.4f} at n_eff={effective:g} (n={n})"
            )
    if fraction:
        frac = float(np.mean(np.abs(scores) < 2.0))
        lo, hi = fraction_band(n)
        if frac < lo or frac > hi:
            return f"z-rule fraction |z|<2 is {frac:.4f}, band [{lo:.4f}, {hi:.4f}] at n={n}"
    return None


def check_replication_means(means: np.ndarray) -> str | None:
    """Bias check for a grid whose nodes share paths.

    ``means`` holds one mean of ``z`` per independent replication. The bound is
    ``|mean(means)| < 3.5 * sd(means) / sqrt(R)``, with ``sd`` the sample
    standard deviation (``ddof=1``). ``R`` is the length of ``means`` and must
    be at least 2. The failure sentence states ``R`` and ``sd``.
    """
    sample = np.asarray(means, dtype=float).reshape(-1)
    if sample.size < 2:
        raise ValueError(f"need at least 2 replications, got {sample.size}")
    if not np.all(np.isfinite(sample)):
        return "replication means are not all finite"
    n_rep = int(sample.size)
    centre = float(np.mean(sample))
    sd = float(np.std(sample, ddof=1))
    if sd == 0.0:
        if centre == 0.0:
            return None
        return f"replication mean is {centre:.4g} with sample sd 0 over R={n_rep}"
    bound = 3.5 * sd / math.sqrt(n_rep)
    if abs(centre) >= bound:
        return (
            f"replication mean is {centre:.4f}, bound {bound:.4f} "
            f"(3.5 * sd/sqrt(R), sd={sd:.4f}, R={n_rep})"
        )
    return None
