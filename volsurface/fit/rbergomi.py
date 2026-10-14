"""Rough Bergomi through the ``roughvol`` entry point. DESIGN section 20.

This module does not simulate the hybrid scheme. ``calibrate`` builds the
model with ``create("rbergomi")`` and, only if that object exposes ``H``,
``nu`` and ``rho``, matches its at-the-money level and skew. A missing
plugin raises ``KeyError`` from the registry. Nothing here is a substitute
model, and a test double is not labelled rBergomi.

``variance_swap`` is the log contract of a SANOS surface,
``VS(T) = -2 E[log Z_T]``, the integral of instantaneous forward variance.
``forward_variance`` is ``d VS / d T``. That curve is the ``xi0`` passed
into the plugin. The plugin default ``xi0 = 0.04`` is a flat 20 percent
vol, and it is not used once a surface is available. Calendar order makes
``VS`` increasing in ``T``.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
from scipy.optimize import least_squares

from volsurface.core.params import from_unconstrained_value, to_unconstrained_value
from volsurface.core.types import FitResult, MarketData
from volsurface.plugins import create
from volsurface.smile import market_term_structure, model_term_structure

# The three settings in DESIGN section 20. None means H is free.
FIXED_H = (0.167, 0.149)


def variance_swap(surface, maturities: np.ndarray) -> np.ndarray:
    """Total variance of the log contract, ``-2 E[log Z_T]``.

    At a SANOS pillar the mixture is exact: each atom of strike ``K`` and
    kernel variance ``v`` contributes ``log K - v / 2``. Between pillars the
    density is the same blend as the call, so the expectation uses that blend.
    """
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    pillars = np.empty(surface.T.size)
    for j in range(surface.T.size):
        kernel = surface.eta * float(surface.variance[j])
        log_mean = float(surface.weights[j] @ np.log(np.maximum(surface.strikes[j], 1e-300)))
        pillars[j] = -2.0 * (log_mean - 0.5 * kernel)
    out = np.empty(maturities.size)
    for i, maturity in enumerate(maturities):
        index, alpha, _ = surface._bracket(float(maturity))
        if alpha == 0.0 or index == surface.T.size - 1:
            out[i] = pillars[index]
        else:
            out[i] = (1.0 - alpha) * pillars[index] + alpha * pillars[index + 1]
    return out


def forward_variance(surface):
    """Instantaneous forward variance, ``d VS / d T``, as a callable of time.

    Between pillars the log contract is linear, so the level is constant:

        xi(t) = (VS(T_{j+1}) - VS(T_j)) / (T_{j+1} - T_j),  T_j <= t < T_{j+1}.

    On ``[0, T_0]`` the level is ``VS(T_0) / T_0``. Past the last pillar the
    last level is held. The callable returns an array of the same shape as
    its argument. A non-positive level raises: the plugin rejects it, and a
    flat 20 percent vol is not filled in instead.
    """
    times = np.asarray(surface.T, dtype=float).reshape(-1)
    total = np.asarray(variance_swap(surface, times), dtype=float).reshape(-1)
    if times.size == 1:
        if times[0] <= 0.0:
            raise ValueError("a single pillar at a non-positive time has no forward variance")
        slopes = np.array([total[0] / times[0]])
        front = float(slopes[0])
    else:
        slopes = np.diff(total) / np.diff(times)
        front = float(total[0] / times[0]) if times[0] > 0.0 else float(slopes[0])
    levels = np.concatenate([[front], slopes]) if times.size > 1 else slopes
    if np.any(levels < -1e-8):
        raise ValueError(f"forward variance fell below 0: {levels[levels < -1e-8]}")
    levels = np.maximum(levels, 0.0)
    if not np.any(levels > 0.0):
        raise ValueError("the log contract is flat, so instantaneous forward variance is 0")
    positive = np.flatnonzero(levels > 0.0)
    for index in range(levels.size):
        if levels[index] == 0.0:
            neighbour = positive[np.argmin(np.abs(positive - index))]
            levels[index] = levels[neighbour]

    def xi0(t):
        grid = np.asarray(t, dtype=float)
        scalar = grid.ndim == 0
        flat = np.atleast_1d(grid)
        out = np.empty(flat.shape, dtype=float)
        early = flat < times[0]
        late = flat >= times[-1]
        out[early] = levels[0]
        out[late] = levels[-1]
        middle = ~early & ~late
        if np.any(middle):
            segment = np.searchsorted(times, flat[middle], side="right") - 1
            segment = np.clip(segment, 0, slopes.size - 1)
            # ``levels[0]`` is the front. Segment ``j`` is ``levels[j + 1]`` when
            # there is more than one pillar, and ``levels[0]`` when there is one.
            out[middle] = levels[segment] if times.size == 1 else levels[segment + 1]
        if scalar:
            return float(out[0])
        return out.reshape(grid.shape)

    return xi0


def _check_h(H: float | None) -> None:
    if H is None:
        return
    if not any(np.isclose(H, fixed) for fixed in FIXED_H):
        raise ValueError(f"H is free, or fixed at 0.167 or 0.149, got {H}")


def _build(model, values: dict):
    """Construct a plugin instance, including ``xi0``.

    Keys the class does not accept are dropped, except ``xi0``. A plugin
    that cannot take ``xi0`` would keep the flat default, so that case raises.
    """
    if "xi0" not in values:
        raise TypeError("xi0 was not set; the plugin default is a flat 20 percent vol")
    cls = type(model)
    params = inspect.signature(cls).parameters
    var_keyword = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values())
    if not var_keyword and "xi0" not in params:
        raise TypeError(
            "the rbergomi plugin does not accept xi0, "
            "so the forward curve would stay at the default"
        )
    if var_keyword:
        payload = dict(values)
    else:
        payload = {key: val for key, val in values.items() if key in params}
    return cls(**payload)


def calibrate(
    data: MarketData,
    *,
    H: float | None = None,
    seed: int = 0,
    surface=None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    max_nfev: int | None = None,
) -> FitResult:
    """Match at-the-money level and skew. ``H`` is free, 0.167, or 0.149.

    The model comes from ``create("rbergomi")``. ``xi0`` is
    :func:`forward_variance` of ``surface``, or of one SANOS fit of ``data``
    at ``eta = 0.25`` and ``omega = 0`` when ``surface`` is omitted. The
    surface is fit once, not inside the residual. Common random numbers are
    the ``seed`` argument. ``n_paths`` and ``n_steps`` are passed when the
    plugin accepts them. They are not free parameters.
    """
    _check_h(H)
    started = time.perf_counter()
    model = create("rbergomi")
    spec = list(model.param_spec())
    names = [item.name for item in spec]
    missing = [name for name in ("nu", "rho") if name not in names]
    if H is None and "H" not in names:
        missing.append("H")
    if missing:
        raise TypeError(
            "the rbergomi plugin is missing "
            + ", ".join(missing)
            + " on param_spec; this fit does not substitute another model"
        )
    market = market_term_structure(data.quotes)
    if market.empty:
        raise ValueError("no expiry had enough quotes for an at-the-money skew")
    if surface is None:
        from volsurface.fit.sanos import SanosFitter

        surface = SanosFitter(eta=0.25, omega=0.0).fit(data).surface
    xi0 = forward_variance(surface)
    free = [item for item in spec if not (item.name == "H" and H is not None)]

    def build(x: np.ndarray):
        values = {}
        cursor = 0
        for item in spec:
            if item.name == "H" and H is not None:
                values["H"] = float(H)
                continue
            values[item.name] = from_unconstrained_value(float(x[cursor]), item)
            cursor += 1
        values["seed"] = int(seed)
        values["xi0"] = xi0
        if n_paths is not None:
            values["n_paths"] = int(n_paths)
        if n_steps is not None:
            values["n_steps"] = int(n_steps)
        return _build(model, values)

    start = []
    current = {item.name: getattr(model, item.name) for item in spec}
    if H is not None:
        current["H"] = float(H)
    for item in free:
        start.append(to_unconstrained_value(float(current[item.name]), item))

    def residual(x: np.ndarray) -> np.ndarray:
        fitted = build(x)
        panel = model_term_structure(fitted, market["T"].to_numpy(), n=9)
        level = panel[:, 0] - market["atm"].to_numpy()
        skew = panel[:, 1] - market["skew"].to_numpy()
        return np.concatenate([level, skew])

    options = {"method": "trf"}
    if max_nfev is not None:
        options["max_nfev"] = int(max_nfev)
    solved = least_squares(residual, np.asarray(start, dtype=float), **options)
    fitted = build(solved.x)
    names_out = {item.name: float(getattr(fitted, item.name)) for item in spec}
    if H is not None:
        names_out["H"] = float(H)
    label = type(fitted).__name__
    front_t = float(market["T"].iloc[0])
    return FitResult(
        surface=None,
        residuals=np.asarray(solved.fun, dtype=float),
        in_spread=float("nan"),
        wall_time=time.perf_counter() - started,
        solver_status=str(solved.message),
        slice_report=market,
        detail={
            "label": label,
            "rbergomi": label == "RoughBergomi" and type(fitted).__module__.startswith("roughvol"),
            "parameters": names_out,
            "objective": float(np.dot(solved.fun, solved.fun)),
            "H_setting": "free" if H is None else float(H),
            "seed": int(seed),
            "cost": float(solved.cost),
            "n_paths": getattr(fitted, "n_paths", None),
            "n_steps": getattr(fitted, "n_steps", None),
            "xi0_front": float(xi0(front_t)),
        },
    )
