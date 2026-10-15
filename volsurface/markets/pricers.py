"""Cash prices along a path. DESIGN section 21.

``ClosedFormPricer`` calls the model's formula. Before expiry the forward
is ``Z_t * F0(T)`` and the discount from ``t`` to ``T`` is ``df(T) / df(t)``.
At and after expiry the entry is the payoff at the first grid time at or
after the contract expiry, held constant. It is not grown at interest.

``VarianceSwapPricer`` integrates the remaining forward variance
``int_t^T xi_t(u) du``. The integral is a left Riemann sum on
``state["xi_tenor"]``. The left-point model variance from the contract's
start up to ``t``, the sum of ``v`` at the left of each step times the
step, is not added. At inception that leg is zero.

``SurrogatePricer`` stores its validation error in basis points and raises
when that error is above the bar it was given.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.special import ndtr

from volsurface.core.types import Contract, Digital, MarketPaths, Vanilla


def integrate_forward_variance(
    xi: np.ndarray, tenors: np.ndarray, t: float, maturity: float
) -> np.ndarray:
    """Left Riemann sum of a forward-variance curve from ``t`` to ``maturity``.

    ``xi`` has shape ``(m, n_tenors)``. Column ``j`` is the curve on the
    interval that starts at ``tenors[j]``.
    """
    grid = np.asarray(tenors, dtype=float).reshape(-1)
    curve = np.asarray(xi, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != grid.size:
        raise ValueError(f"xi must have shape (m, {grid.size}), got {curve.shape}")
    if maturity <= t:
        return np.zeros(curve.shape[0])
    total = np.zeros(curve.shape[0])
    for j in range(grid.size - 1):
        left = float(grid[j])
        right = float(grid[j + 1])
        start = max(left, t)
        stop = min(right, maturity)
        if stop > start:
            total += curve[:, j] * (stop - start)
    if maturity > float(grid[-1]):
        total += curve[:, -1] * (maturity - max(float(grid[-1]), t))
    return total


class ClosedFormPricer:
    """Formula prices for Black-Scholes, Heston, and local volatility."""

    def __init__(self, model):
        self.model = model

    def supports(self, model, contract) -> bool:
        if model is not self.model and type(model) is not type(self.model):
            return False
        return isinstance(contract.payoff, (Vanilla, Digital))

    def price_along(self, paths: MarketPaths, contracts: Sequence[Contract]) -> np.ndarray:
        m, n = paths.Z.shape
        out = np.empty((m, n, len(contracts)))
        for j, contract in enumerate(contracts):
            if not self.supports(self.model, contract):
                raise TypeError(f"this pricer does not price {type(contract.payoff).__name__}")
            out[:, :, j] = self._one(paths, contract)
        return out

    def _one(self, paths: MarketPaths, contract: Contract) -> np.ndarray:
        times = np.asarray(paths.times, dtype=float)
        expiry = float(contract.T)
        index = int(np.searchsorted(times, expiry, side="left"))
        index = min(index, times.size - 1)
        forward_T = float(np.asarray(paths.F0(expiry), dtype=float).reshape(-1)[0])
        payoff = _payoff(paths.Z[:, index], forward_T, contract)
        out = np.empty(paths.Z.shape)
        for i, t in enumerate(times):
            if float(t) + 1e-12 >= expiry:
                out[:, i] = payoff
                continue
            forward = paths.Z[:, i] * forward_T
            discount = _level(paths.df, expiry) / _level(paths.df, float(t))
            out[:, i] = self._formula(forward, contract, expiry - float(t), discount, paths, i)
        return out

    def _formula(self, forward, contract, time_left, discount, paths, i) -> np.ndarray:
        payoff = contract.payoff
        kind = payoff.kind
        if isinstance(payoff, Digital):
            sigma = getattr(self.model, "sigma", None)
            if sigma is None:
                raise TypeError("a digital along the path is priced for Black-Scholes")
            return _digital(forward, payoff.K, time_left, float(sigma), discount, kind)
        state = _state_slice(self.model, paths, i)
        if state is None:
            return np.asarray(
                self.model.price(forward, payoff.K, time_left, discount, kind), dtype=float
            )
        out = np.empty(forward.shape[0])
        for p in range(forward.shape[0]):
            one = {key: np.asarray(value)[p] for key, value in state.items()}
            out[p] = self.model.price(
                float(forward[p]), payoff.K, time_left, discount, kind, state=one
            )
        return out


class VarianceSwapPricer:
    """Remaining total variance ``int_t^T xi_t(u) du`` from the stored curve.

    The left-point model variance up to ``t`` is not added. That leg is the
    sum of ``v`` at the left of each step times the step. It is not the sum
    of squared log returns. A mark of a live swap is that leg, plus this
    remaining integral, minus the strike times the contractual variance period.
    """

    def supports(self, model, contract) -> bool:
        return True

    def price_along(self, paths: MarketPaths, contracts: Sequence[Contract]) -> np.ndarray:
        if "xi" not in paths.state or "xi_tenor" not in paths.state:
            raise KeyError("VarianceSwapPricer reads state['xi'] and state['xi_tenor']")
        xi = np.asarray(paths.state["xi"], dtype=float)
        tenors = np.asarray(paths.state["xi_tenor"], dtype=float)
        times = np.asarray(paths.times, dtype=float)
        m, n = paths.Z.shape
        out = np.empty((m, n, len(contracts)))
        for j, contract in enumerate(contracts):
            maturity = float(contract.T)
            for i, t in enumerate(times):
                out[:, i, j] = integrate_forward_variance(xi[:, i, :], tenors, float(t), maturity)
        return out


class SurrogatePricer:
    """A table or a network, refused when its validation error is above the bar.

    ``predict(paths, contracts)`` returns cash prices of shape ``(m, n + 1, c)``.
    ``validation_error_bp`` and ``bar_bp`` are basis points of price.
    """

    def __init__(self, predict: Callable, validation_error_bp: float, bar_bp: float):
        self.predict = predict
        self.validation_error_bp = float(validation_error_bp)
        self.bar_bp = float(bar_bp)
        if self.validation_error_bp > self.bar_bp:
            raise ValueError(
                f"validation error {self.validation_error_bp:g} bp is above the bar "
                f"{self.bar_bp:g} bp"
            )

    def supports(self, model, contract) -> bool:
        return True

    def price_along(self, paths: MarketPaths, contracts: Sequence[Contract]) -> np.ndarray:
        return np.asarray(self.predict(paths, contracts), dtype=float)


def _level(curve, t: float) -> float:
    return float(np.asarray(curve(t), dtype=float).reshape(-1)[0])


def _payoff(z_expiry: np.ndarray, forward: float, contract: Contract) -> np.ndarray:
    terminal = z_expiry * forward
    strike = float(contract.payoff.K)
    if isinstance(contract.payoff, Digital):
        if contract.payoff.kind == "call":
            return (terminal > strike).astype(float)
        return (terminal < strike).astype(float)
    if contract.payoff.kind == "call":
        return np.maximum(terminal - strike, 0.0)
    return np.maximum(strike - terminal, 0.0)


def _state_slice(model, paths: MarketPaths, i: int):
    initial = model.initial_state() if hasattr(model, "initial_state") else {}
    if not initial:
        return None
    out = {}
    for key in initial:
        if key not in paths.state:
            continue
        value = np.asarray(paths.state[key])
        out[key] = value[:, i] if value.ndim >= 2 else value
    return out or None


def _digital(forward, strike, time_left, sigma, discount, kind: str) -> np.ndarray:
    forward = np.asarray(forward, dtype=float)
    sd = sigma * np.sqrt(time_left)
    if sd == 0.0:
        if kind == "call":
            return discount * (forward > strike).astype(float)
        return discount * (forward < strike).astype(float)
    d2 = (np.log(np.maximum(forward, 1e-300) / strike) - 0.5 * sd * sd) / sd
    if kind == "call":
        return discount * ndtr(d2)
    return discount * ndtr(-d2)
