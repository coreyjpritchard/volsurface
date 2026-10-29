"""One simulation, a whole strike and expiry grid.

``docs/DESIGN.md`` section 7. The paths are the unit-mean martingale ``Z``.
A call payoff is ``df * (F Z - K)+`` and a put is ``df * (K - F Z)+``. Strikes
at one expiry share the column of ``Z`` at that expiry, so their errors move
together. That is the grid an implied-volatility figure should plot: the wing
noise is one draw, not an independent disaster at every strike.

Antithetic paths are stored as the second half of the batch, which is how
:class:`~volsurface.models.gbm.GBM` writes them. The control variate is ``Z``
at the expiry being priced, mean 1. The slope is ordinary least squares of the
batch payoff on ``Z``, and the control is then applied with the known mean,
``payoff - beta (Z - 1)``.
"""

from __future__ import annotations

import time

import numpy as np

from volsurface.core.types import PriceResult


def estimate_terminal(
    Z,
    F: float,
    K,
    df: float,
    kind,
    *,
    antithetic: bool,
    control_variate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean cash price and its standard error from one column of ``Z``.

    ``Z`` has shape ``(n_stored,)``. With ``antithetic``, the second half is
    the mirror of the first and the two are averaged before the standard error
    is formed, so the divisor is the number of batches.
    """
    Z = np.asarray(Z, dtype=float)
    payoff = _cash_payoff(Z, F, K, df, kind)
    z_level = Z
    if antithetic:
        if Z.size % 2 != 0:
            raise ValueError("antithetic paths must come in pairs")
        half = Z.size // 2
        payoff = 0.5 * (payoff[:half] + payoff[half:])
        z_level = 0.5 * (Z[:half] + Z[half:])
    if control_variate:
        payoff = _control(payoff, z_level)
    n_batches = payoff.shape[0]
    mean = payoff.mean(axis=0)
    stderr = payoff.std(axis=0, ddof=1) / np.sqrt(n_batches)
    return mean, stderr


def price_grid(
    model,
    F: float,
    strikes,
    maturities,
    df=1.0,
    kind="call",
    *,
    n_paths: int,
    seed: int | np.random.SeedSequence,
) -> PriceResult:
    """Price every strike at every maturity from one ``simulate`` call.

    ``maturities`` must be strictly increasing and positive. ``strikes`` is
    one row shared by all maturities, shape ``(n_K,)``, or one row per
    maturity, shape ``(n_T, n_K)``. ``df`` is a scalar or one value per
    maturity. The model's ``scheme``, ``antithetic`` and ``control_variate``
    are used. ``n_steps`` is not: the time grid is ``0`` plus the maturities.

    Returns ``value`` and ``stderr`` of shape ``(n_T, n_K)``.
    """
    started = time.perf_counter()
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    if maturities.ndim != 1 or np.any(maturities <= 0.0) or np.any(np.diff(maturities) <= 0.0):
        raise ValueError("maturities must be strictly increasing and positive")
    strikes = np.asarray(strikes, dtype=float)
    if strikes.ndim == 1:
        strikes = np.broadcast_to(strikes, (maturities.size, strikes.shape[0])).copy()
    elif strikes.shape != (maturities.size, strikes.shape[-1]):
        raise ValueError(
            f"strikes shape {strikes.shape} does not match maturities {maturities.size}"
        )
    dfs = np.broadcast_to(np.asarray(df, dtype=float), maturities.shape).copy()
    kinds = _kind_grid(kind, strikes.shape)
    times = np.concatenate(([0.0], maturities))
    paths = model.simulate(int(n_paths), times, seed=seed)
    value = np.empty(strikes.shape, dtype=float)
    stderr = np.empty(strikes.shape, dtype=float)
    antithetic = bool(getattr(model, "antithetic", False))
    control = bool(getattr(model, "control_variate", False))
    for i in range(maturities.size):
        value[i], stderr[i] = estimate_terminal(
            paths.Z[:, i + 1],
            float(F),
            strikes[i],
            float(dfs[i]),
            kinds[i],
            antithetic=antithetic,
            control_variate=control,
        )
    return PriceResult(
        value=value,
        stderr=stderr,
        n_evals=int(paths.Z.shape[0]),
        wall_time=time.perf_counter() - started,
    )


def _cash_payoff(Z: np.ndarray, F: float, K: np.ndarray, df: float, kind) -> np.ndarray:
    """``df * (F Z - K)+`` or the put. Shape ``(n_paths, n_strikes)``."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    forward = F * Z[:, None]
    strike = K[None, :]
    call = np.maximum(forward - strike, 0.0)
    put = np.maximum(strike - forward, 0.0)
    if isinstance(kind, str):
        if kind == "call":
            return df * call
        if kind == "put":
            return df * put
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    kinds = np.asarray(kind)
    if kinds.shape != K.shape or not np.all((kinds == "call") | (kinds == "put")):
        raise ValueError(f"kind must be 'call' or 'put' with shape {K.shape}, got {kinds}")
    return df * np.where(kinds[None, :] == "call", call, put)


def _control(payoff: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Subtract ``beta (z - 1)``. ``beta`` is the least-squares slope on ``z``.

    A column of ``z`` with no variation (a one-step Euler antithetic pair
    averages to 1 exactly) is left uncontrolled.
    """
    z_center = z - z.mean()
    var = float(np.dot(z_center, z_center))
    if var == 0.0:
        return payoff
    beta = (z_center @ payoff) / var
    return payoff - (z - 1.0)[:, None] * beta[None, :]


def _kind_grid(kind, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(kind, str):
        return np.full(shape, kind)
    kinds = np.asarray(kind)
    if kinds.shape == (shape[1],):
        return np.broadcast_to(kinds, shape).copy()
    if kinds.shape != shape:
        raise ValueError(f"kind shape {kinds.shape} does not match strikes {shape}")
    return kinds
