"""SANOS linear program, section 4.2 of arXiv:2601.11209.

``docs/DESIGN.md`` section 18. The objective is Remark 4.1 item 3:
``eps = 1e-8`` and ``w = 1/(ask - bid)``. ``omega = 0`` is the default.
``omega = 1`` is solved and labelled not guaranteed. Out-of-the-money quotes
only. A call and a put at one normalised strike become one row, keeping the
tighter bid and the tighter ask.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

from volsurface import bs
from volsurface.core.types import Curves, Domain, FitResult, MarketData
from volsurface.fit._lp import ConstraintList, solve_linear
from volsurface.surface.sanos import SanosSurface, black_call

_EPS = 1e-8


def project_marginal(weights: np.ndarray, strikes: np.ndarray) -> tuple[np.ndarray, float]:
    """Clip at 0 and tilt so the mass is 1 and the mean is 1.

    The return is the projected weights and the L1 distance from the input.
    """
    raw = np.maximum(np.asarray(weights, dtype=float), 0.0)
    strikes = np.asarray(strikes, dtype=float)
    if raw.sum() <= 0.0 or not np.isfinite(raw).all():
        raw = np.ones(strikes.size)
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        tilted = raw * np.exp(np.clip(mid * (strikes - 1.0), -60.0, 60.0))
        tilted /= tilted.sum()
        if float(tilted @ strikes) > 1.0:
            hi = mid
        else:
            lo = mid
    tilted = raw * np.exp(np.clip(0.5 * (lo + hi) * (strikes - 1.0), -60.0, 60.0))
    tilted /= tilted.sum()
    correction = float(np.abs(tilted - np.asarray(weights, dtype=float)).sum())
    return tilted, correction


def _fill(strikes: np.ndarray, dx: float) -> np.ndarray:
    """Insert knots so that every gap is at most ``dx``."""
    if strikes.size == 0:
        return strikes
    out = [float(strikes[0])]
    for left, right in zip(strikes[:-1], strikes[1:], strict=True):
        gap = float(right - left)
        if gap <= dx:
            out.append(float(right))
            continue
        n = int(np.ceil(gap / dx))
        grid = np.linspace(left, right, n + 1)[1:]
        out.extend(float(x) for x in grid)
    return np.asarray(out, dtype=float)


def _wing_bounds(frames: list[pd.DataFrame]) -> tuple[float, float]:
    """Remark 2.7, footnote 9. ``K_min = min K* / 10``, ``K_max = 1.5 max K#``."""
    stars: list[float] = []
    hashes: list[float] = []
    for frame in frames:
        x = frame["x"].to_numpy(dtype=float)
        mid = 0.5 * (frame["bid"].to_numpy(dtype=float) + frame["ask"].to_numpy(dtype=float))
        order = np.argsort(x)
        x, mid = x[order], mid[order]
        if x.size < 2:
            continue
        slope = (mid[1] - mid[0]) / (x[1] - x[0])
        denom = 1.0 + slope
        if abs(denom) < 1e-10:
            stars.append(float(x[0]))
        else:
            stars.append(float((1.0 - mid[0] + slope * x[0]) / denom))
        end = (mid[-1] - mid[-2]) / (x[-1] - x[-2])
        if abs(end) < 1e-12:
            hashes.append(float(x[-1]))
        else:
            hashes.append(float(x[-1] - mid[-1] / end))
    market_min = min(float(frame["x"].min()) for frame in frames)
    market_max = max(float(frame["x"].max()) for frame in frames)
    k_min = 0.1 * min(stars) if stars else market_min / 10.0
    k_max = 1.5 * max(hashes) if hashes else market_max * 1.5
    if not (0.0 < k_min < market_min):
        k_min = market_min / 10.0
    if not (k_max > market_max):
        k_max = market_max * 1.5
    return float(k_min), float(k_max)


def _otm_frames(data: MarketData) -> list[pd.DataFrame]:
    """One frame per expiry: OTM quotes, merged to the tighter bid and ask."""
    frames = []
    for expiry, group in data.quotes.groupby("expiry", sort=True):
        rows = []
        for _, row in group.iterrows():
            forward = float(row["F"])
            discount = float(row["df"])
            strike = float(row["K"])
            x = strike / forward
            kind = row["kind"]
            otm = (kind == "put" and x < 1.0) or (kind == "call" and x >= 1.0)
            if not otm:
                continue
            bid = float(row["bid"]) if pd.notna(row["bid"]) else float("nan")
            ask = float(row["ask"]) if pd.notna(row["ask"]) else float("nan")
            if not np.isfinite(bid) or not np.isfinite(ask) or ask < bid:
                continue
            if kind == "put":
                parity = discount * (forward - strike)
                bid += parity
                ask += parity
            scale = discount * forward
            rows.append(
                {
                    "x": x,
                    "bid": bid / scale,
                    "ask": ask / scale,
                    "T": float(row["T"]),
                    "F": forward,
                    "df": discount,
                    "expiry": expiry,
                }
            )
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        # Tighter bid is the larger bid. Tighter ask is the smaller ask.
        frame = frame.groupby("x", as_index=False).agg(
            bid=("bid", "max"),
            ask=("ask", "min"),
            T=("T", "first"),
            F=("F", "first"),
            df=("df", "first"),
            expiry=("expiry", "first"),
        )
        frame = frame.sort_values("x").reset_index(drop=True)
        frames.append(frame)
    frames.sort(key=lambda frame: float(frame["T"].iloc[0]))
    return frames


def _atm_variance(frame: pd.DataFrame) -> float:
    """At-the-money bid total variance. The mid is used when the bid has no vol."""
    x = frame["x"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(np.log(np.maximum(x, 1e-300)))))
    row = frame.iloc[i]
    forward = float(row["F"])
    discount = float(row["df"])
    strike = float(row["x"]) * forward
    maturity = float(row["T"])

    def _var(norm_price: float) -> float:
        cash = norm_price * discount * forward
        iv = bs.implied_vol(cash, forward, strike, maturity, discount, "call")
        iv = float(np.asarray(iv).reshape(-1)[0])
        if not np.isfinite(iv):
            return float("nan")
        return iv * iv * maturity

    bid_var = _var(float(row["bid"]))
    if np.isfinite(bid_var) and bid_var > 0.0:
        return bid_var
    mid_var = _var(0.5 * (float(row["bid"]) + float(row["ask"])))
    if np.isfinite(mid_var) and mid_var > 0.0:
        return mid_var
    return 1e-6


def _monotone(raw: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Cumulative maximum, then a 1e-10 step where two expiries still tie."""
    used = np.maximum.accumulate(np.asarray(raw, dtype=float))
    raised: list[int] = []
    for j in range(used.size):
        if used[j] > raw[j] + 1e-15:
            raised.append(j)
    for j in range(1, used.size):
        if used[j] <= used[j - 1]:
            used[j] = used[j - 1] + 1e-10
            if j not in raised:
                raised.append(j)
    return used, raised


def quote_in_spread(quotes: pd.DataFrame, surface) -> float:
    """Fraction of two-sided quotes whose model call lies between bid and ask.

    Same rule as the raw SVI fitter: bid strictly positive, ask above bid.
    Puts are converted by parity before the comparison.
    """
    usable = quotes.dropna(subset=["bid", "ask", "F", "df", "k", "T"])
    usable = usable[np.isfinite(usable["bid"]) & np.isfinite(usable["ask"])]
    usable = usable[(usable["bid"] > 0.0) & (usable["ask"] > usable["bid"])]
    if usable.empty:
        return float("nan")
    inside = 0
    for _, row in usable.iterrows():
        model = float(surface.call([row["T"]], [row["k"]])[0, 0])
        cash = model * float(row["df"]) * float(row["F"])
        if row["kind"] == "put":
            parity = float(row["df"]) * (float(row["F"]) - float(row["K"]))
            bid = float(row["bid"]) + parity
            ask = float(row["ask"]) + parity
        else:
            bid, ask = float(row["bid"]), float(row["ask"])
        lo, hi = min(bid, ask), max(bid, ask)
        inside += int(lo - 1e-10 <= cash <= hi + 1e-10)
    return inside / len(usable)


class SanosFitter:
    """Joint SANOS linear program. ``eta = 0`` is the linear surface.

    ``dx`` is the largest gap, in normalised strike, before an interior
    strike is inserted. It is stored on ``FitResult.detail``.

    The default solver is the HiGHS interior point method with crossover
    off. On the smooth SPX book the simplex run returns HiGHS status 15
    (model status unknown, primal feasible) and no solution vector.
    """

    def __init__(
        self,
        *,
        eta: float = 0.0,
        omega: float = 0.0,
        dx: float = 0.05,
        method: str = "highs-ipm",
        time_interp: str = "linear",
    ):
        if not 0.0 <= eta < 1.0:
            raise ValueError(f"eta must lie in [0, 1), got {eta}")
        if omega not in (0.0, 1.0):
            raise ValueError(f"omega must be 0 or 1, got {omega}")
        if dx <= 0.0:
            raise ValueError(f"dx must be positive, got {dx}")
        self.eta = float(eta)
        self.omega = float(omega)
        self.dx = float(dx)
        self.method = method
        self.time_interp = time_interp

    def fit(self, data: MarketData) -> FitResult:
        started = time.perf_counter()
        frames = _otm_frames(data)
        if not frames:
            raise ValueError("SANOS fit needs at least one out-of-the-money quote")
        k_min, k_max = _wing_bounds(frames)
        raw_v = np.array([_atm_variance(frame) for frame in frames])
        variance, raised = _monotone(raw_v)
        strikes = []
        for frame in frames:
            nodes = np.unique(
                np.concatenate(
                    [
                        [k_min],
                        frame["x"].to_numpy(dtype=float),
                        [k_max],
                    ]
                )
            )
            nodes = nodes[(nodes >= k_min - 1e-12) & (nodes <= k_max + 1e-12)]
            strikes.append(_fill(nodes, self.dx))

        solved, quote_index = self._solve(frames, strikes, variance)
        if not solved.success or solved.x is None:
            raise RuntimeError(f"HiGHS did not return a SANOS solution ({solved.message})")

        raw_weights = []
        weights = []
        corrections = []
        cursor = 0
        for grid in strikes:
            raw = np.asarray(solved.x[cursor : cursor + grid.size], dtype=float)
            projected, correction = project_marginal(raw, grid)
            raw_weights.append(raw)
            weights.append(projected)
            corrections.append(correction)
            cursor += grid.size
        solver_prices = _quote_model_weights(frames, strikes, raw_weights, self.eta, variance)

        maturities = np.array([float(frame["T"].iloc[0]) for frame in frames])
        domain = Domain(
            float(maturities[0]),
            float(maturities[-1]),
            float(np.log(k_min)),
            float(np.log(k_max)),
        )
        surface = SanosSurface(
            maturities,
            strikes,
            weights,
            variance,
            self.eta,
            Curves(data.F, data.df),
            domain,
            time_interp=self.time_interp,
        )
        model_norm = _quote_model(surface, frames, strikes, weights)
        mids = []
        bids = []
        asks = []
        for frame in frames:
            bids.append(frame["bid"].to_numpy(dtype=float))
            asks.append(frame["ask"].to_numpy(dtype=float))
            mids.append(0.5 * (bids[-1] + asks[-1]))
        mid = np.concatenate(mids)
        residuals = model_norm - mid
        in_spread = quote_in_spread(data.quotes, surface)
        rows = []
        offset = 0
        for j, frame in enumerate(frames):
            n = len(frame)
            cash_scale = frame["df"].to_numpy(dtype=float) * frame["F"].to_numpy(dtype=float)
            model_cash = model_norm[offset : offset + n] * cash_scale
            bid_cash = frame["bid"].to_numpy(dtype=float) * cash_scale
            ask_cash = frame["ask"].to_numpy(dtype=float) * cash_scale
            inside = (model_cash >= bid_cash - 1e-8) & (model_cash <= ask_cash + 1e-8)
            rows.append(
                {
                    "T": float(maturities[j]),
                    "in_spread": float(inside.mean()) if n else float("nan"),
                    "n_quotes": int(n),
                    "v_atm": float(raw_v[j]),
                    "v_used": float(variance[j]),
                    "v_raised": bool(j in raised),
                    "n_strikes": int(strikes[j].size),
                    "projection_l1": float(corrections[j]),
                }
            )
            offset += n
        label = "calendar guaranteed" if self.omega == 0.0 else "not guaranteed"
        detail = {
            "objective": float(solved.fun),
            "dx": self.dx,
            "eta": self.eta,
            "omega": self.omega,
            "omega_label": label,
            "method": self.method,
            "k_min": k_min,
            "k_max": k_max,
            "projection_l1": float(sum(corrections)),
            "v_raised": raised,
            "quote_prices": model_norm,
            "solver_quote_prices": solver_prices,
            "quote_x": np.concatenate([frame["x"].to_numpy(dtype=float) for frame in frames]),
            "quote_bid": np.concatenate(bids),
            "quote_ask": np.concatenate(asks),
            "quote_T": np.concatenate(
                [np.full(len(frame), maturities[j]) for j, frame in enumerate(frames)]
            ),
            "quote_index": quote_index,
            "message": str(solved.message),
        }
        return FitResult(
            surface=surface,
            residuals=residuals,
            in_spread=float(in_spread),
            wall_time=time.perf_counter() - started,
            solver_status=f"{solved.message} objective={solved.fun:.6e} omega={label}",
            slice_report=pd.DataFrame(rows),
            detail=detail,
        )

    def _solve(self, frames, strikes, variance):
        widths = [grid.size for grid in strikes]
        offsets = np.cumsum([0, *widths[:-1]]).astype(int)
        n_q = int(sum(widths))
        n_quotes = int(sum(len(frame) for frame in frames))
        # Four slacks per quote: e+, e-, outside-ask, outside-bid.
        n_slack = 4 * n_quotes
        n_var = n_q + n_slack
        cost = np.zeros(n_var)
        equalities = ConstraintList()
        inequalities = ConstraintList()

        for j, grid in enumerate(strikes):
            equalities.add(np.arange(offsets[j], offsets[j] + grid.size), np.ones(grid.size), 1.0)
            equalities.add(np.arange(offsets[j], offsets[j] + grid.size), grid, 1.0)

        slack = n_q
        quote_index = []
        for j, frame in enumerate(frames):
            v = self.eta * float(variance[j])
            coeff = black_call(
                strikes[j][None, :],
                frame["x"].to_numpy(dtype=float)[:, None],
                v,
            )
            for i in range(len(frame)):
                row = coeff[i]
                cols = np.arange(offsets[j], offsets[j] + row.size)
                bid = float(frame["bid"].iloc[i])
                ask = float(frame["ask"].iloc[i])
                mid = 0.5 * (bid + ask)
                spread = max(ask - bid, 1e-12)
                weight = 1.0 / spread
                # e+ - e- - C = -mid
                equalities.add(
                    np.concatenate([cols, [slack, slack + 1]]),
                    np.concatenate([-row, [1.0, -1.0]]),
                    -mid,
                )
                # C - a <= ask
                inequalities.add(
                    np.concatenate([cols, [slack + 2]]),
                    np.concatenate([row, [-1.0]]),
                    ask,
                )
                # -C - b <= -bid
                inequalities.add(
                    np.concatenate([cols, [slack + 3]]),
                    np.concatenate([-row, [-1.0]]),
                    -bid,
                )
                cost[slack] = weight * _EPS
                cost[slack + 1] = weight * _EPS
                cost[slack + 2] = weight
                cost[slack + 3] = weight
                quote_index.append(j)
                slack += 4

        for j in range(1, len(strikes)):
            query = strikes[j]
            v_now = self.eta * self.omega * float(variance[j])
            v_prev = self.eta * self.omega * float(variance[j - 1])
            now = black_call(strikes[j][None, :], query[:, None], v_now)
            prev = black_call(strikes[j - 1][None, :], query[:, None], v_prev)
            for i in range(query.size):
                # prev - now <= 0
                inequalities.add(
                    np.concatenate(
                        [
                            np.arange(offsets[j - 1], offsets[j - 1] + strikes[j - 1].size),
                            np.arange(offsets[j], offsets[j] + strikes[j].size),
                        ]
                    ),
                    np.concatenate([prev[i], -now[i]]),
                    0.0,
                )

        a_ub, b_ub = inequalities.to_csr(n_var)
        a_eq, b_eq = equalities.to_csr(n_var)
        # Quote weights reach 1/spread ~ 1e5. Dividing the cost by that scale
        # leaves the minimiser unchanged and keeps the objective entries near 1.
        cost_scale = float(max(np.max(cost), 1.0))
        # run_crossover is a HiGHS option SciPy does not list. It is applied.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected",
                category=OptimizeWarning,
            )
            solved = solve_linear(
                cost / cost_scale,
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=(0.0, None),
                method=self.method,
                options={"run_crossover": "off"} if self.method == "highs-ipm" else None,
            )
        if solved.fun is not None:
            solved.fun = float(solved.fun) * cost_scale
        return solved, quote_index


def _quote_model_weights(frames, strikes, weights, eta: float, variance) -> np.ndarray:
    """Normalised calls at the quotes from a weight vector, before projection."""
    out = []
    for j, frame in enumerate(frames):
        v = float(eta) * float(variance[j])
        coeff = black_call(strikes[j][None, :], frame["x"].to_numpy(dtype=float)[:, None], v)
        out.append(coeff @ np.asarray(weights[j], dtype=float))
    return np.concatenate(out)


def _quote_model(surface, frames, strikes, weights) -> np.ndarray:
    """Model normalised calls at the quote strikes, from the projected weights."""
    out = []
    for j, frame in enumerate(frames):
        v = surface.eta * float(surface.variance[j])
        coeff = black_call(strikes[j][None, :], frame["x"].to_numpy(dtype=float)[:, None], v)
        out.append(coeff @ weights[j])
    return np.concatenate(out)
