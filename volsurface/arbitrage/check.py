"""Two arbitrage checks that answer different questions.

``check_surface`` looks at one grid of a continuous object. At each node it
tests the normalised call ``C``:

- bounds: ``(1 - exp(k))+ <= C <= 1``
- call spread: the slope ``dC/dx`` lies in ``[-1, 0]``, where ``x = exp(k)``
- butterfly: ``d²C/dx² >= 0``
- calendar: ``dC/dT >= 0`` at fixed ``k``

Passing says nothing about the gaps between the nodes. The report's ``note``
says that, every time.

``check_quotes`` asks whether any discrete martingale measure reproduces the
quotes. Probabilities ``q`` on the quoted strikes, plus a strike at 0 and one
past the last quote, satisfy ``q >= 0``, ``sum q = 1`` and ``sum q x = 1``.
The model call is ``sum q_j (x_j - x)+``. Slacks soak up bid, ask, and
calendar violations, and the solver minimises the sum of the slacks. This is
the feasibility form of the SANOS ``eta = 0`` linear program, not the fit.

Puts are turned into calls by parity before either check. On American options
that step is approximate, and a violation can be the early-exercise premium
rather than a tradable arbitrage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from volsurface.core.types import ArbReport, ArbTol, MarketData

_VIOLATION_COLUMNS = ["kind", "T", "k", "magnitude", "magnitude_in_spreads"]
# HiGHS returns constraints feasible to roughly 1e-7. Slacks below that are noise.
_SOLVER_TOL = 1e-7
_SURFACE_NOTE = (
    "Pointwise on the supplied grid. A pass says nothing about the surface between the nodes."
)


def check_surface(surface, T, k, *, tol: ArbTol | None = None) -> ArbReport:
    """Pointwise bounds, call spread, butterfly, and calendar. See the module note."""
    tol = ArbTol() if tol is None else tol
    T = np.atleast_1d(np.asarray(T, dtype=float))
    k = np.atleast_1d(np.asarray(k, dtype=float))
    call = np.asarray(surface.call(T, k), dtype=float)
    dk = np.asarray(surface.dcall_dk(T, k), dtype=float)
    dkk = np.asarray(surface.d2call_dk2(T, k), dtype=float)
    dt = np.asarray(surface.dcall_dT(T, k), dtype=float)

    x = np.exp(k)[None, :]
    lower = np.maximum(1.0 - x, 0.0)
    bounds_mag = _outside(call, lower, np.ones_like(call))
    # Non-finite prices are a bounds failure of infinite magnitude.
    bad = ~np.isfinite(call)
    bounds_mag = np.where(bad, np.inf, bounds_mag)

    slope = np.exp(-k)[None, :] * dk
    spread_mag = _outside(slope, -np.ones_like(slope), np.zeros_like(slope))

    # d²C/dx² = exp(-2k) (d²C/dk² - dC/dk). Negative means the call is not convex.
    convexity = np.exp(-2.0 * k)[None, :] * (dkk - dk)
    butterfly_mag = np.maximum(-convexity, 0.0)
    butterfly_mag = np.where(np.isfinite(convexity), butterfly_mag, np.inf)

    calendar_mag = np.maximum(-dt, 0.0)
    calendar_mag = np.where(np.isfinite(dt), calendar_mag, np.inf)

    magnitudes = {
        "bounds": bounds_mag,
        "call_spread": spread_mag,
        "butterfly": butterfly_mag,
        "calendar": calendar_mag,
    }
    masks = {name: mag > tol.price for name, mag in magnitudes.items()}
    violations = _rows(T, k, magnitudes, masks, tol)
    min_density = _min_density(surface, T, k)
    return ArbReport(
        violations=violations,
        masks=masks,
        min_density=min_density,
        ok=violations.empty,
        note=_SURFACE_NOTE,
    )


def check_quotes(
    data: MarketData,
    *,
    side: str = "bidask",
    tol: ArbTol | None = None,
) -> ArbReport:
    """Linear-program feasibility of the quotes. ``side`` is ``"bidask"`` or ``"mid"``.

    ``"mid"`` uses the working price as both bid and ask. ``"bidask"`` drops
    quotes with a missing bid or ask, and the note records how many were dropped.
    """
    if side not in ("bidask", "mid"):
        raise ValueError(f"side must be 'bidask' or 'mid', got {side!r}")
    tol = ArbTol() if tol is None else tol
    expiries = _expiry_books(data, side)
    dropped = int(expiries.pop("_dropped"))
    if not expiries:
        note = (
            f"No quotes to check (dropped {dropped} without a two-sided market). "
            "The committed SPY snapshot is in this state: last trades, empty bid and ask."
        )
        return ArbReport(
            violations=_empty(),
            masks={},
            min_density=float("nan"),
            ok=False,
            note=note,
        )
    result = _solve(expiries, tol)
    result_note = (
        f"{result.note} Dropped {dropped} quotes with no two-sided market. "
        "Puts entered through put-call parity. On American options that parity "
        "is approximate, so a slack is not by itself a tradable arbitrage."
    )
    return ArbReport(
        violations=result.violations,
        masks=result.masks,
        min_density=float("nan"),
        ok=result.ok,
        note=result_note,
    )


def _outside(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """How far ``values`` sits outside ``[lower, upper]``. Zero when inside."""
    return np.maximum(np.maximum(lower - values, values - upper), 0.0)


def _rows(T, k, magnitudes, masks, tol: ArbTol) -> pd.DataFrame:
    records = []
    for kind, mag in magnitudes.items():
        ii, jj = np.nonzero(masks[kind])
        for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
            magnitude = float(mag[i, j])
            records.append(
                {
                    "kind": kind,
                    "T": float(T[i]),
                    "k": float(k[j]),
                    "magnitude": magnitude,
                    "magnitude_in_spreads": (
                        magnitude / tol.spreads if tol.spreads else float("nan")
                    ),
                }
            )
    if not records:
        return _empty()
    return pd.DataFrame.from_records(records, columns=_VIOLATION_COLUMNS)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_VIOLATION_COLUMNS)


def _min_density(surface, T, k) -> float:
    if not hasattr(surface, "density"):
        return float("nan")
    values = np.asarray(surface.density(T, k), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def _expiry_books(data: MarketData, side: str) -> dict:
    """Per expiry: normalised strikes, bids, asks, and the year fraction.

    The returned dict also carries ``_dropped``, the number of rows a bid-ask
    check could not use.
    """
    quotes = data.quotes
    dropped = 0
    books: dict = {}
    for expiry, group in quotes.groupby("expiry", sort=True):
        rows = []
        for _, row in group.iterrows():
            forward = float(row["F"])
            discount = float(row["df"])
            strike = float(row["K"])
            scale = discount * forward
            if side == "mid":
                price = float(row["price"])
                bid = ask = price
            else:
                bid_cash = float(row["bid"]) if pd.notna(row["bid"]) else float("nan")
                ask_cash = float(row["ask"]) if pd.notna(row["ask"]) else float("nan")
                if not np.isfinite(bid_cash) or not np.isfinite(ask_cash) or ask_cash < bid_cash:
                    dropped += 1
                    continue
                bid, ask = bid_cash, ask_cash
            if row["kind"] == "put":
                parity = discount * (forward - strike)
                bid = bid + parity
                ask = ask + parity
            x = strike / forward
            rows.append((x, bid / scale, ask / scale, float(row["T"])))
        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=["x", "bid", "ask", "T"])
        frame = frame.groupby("x", as_index=False).mean()
        books[expiry] = frame
    ordered = dict(sorted(books.items(), key=lambda kv: float(kv[1]["T"].iloc[0])))
    ordered["_dropped"] = dropped
    return ordered


def _solve(books: dict, tol: ArbTol) -> ArbReport:
    """Minimise total slack. Equalities are the two martingale constraints per expiry."""
    names = list(books)
    supports = []
    for name in names:
        x = books[name]["x"].to_numpy(dtype=float)
        hi = max(3.0, float(np.max(x)) * 1.5)
        # Boundary strikes carry no quote. They let the mean equal 1.
        support = np.unique(np.concatenate([[0.0], x, [hi]]))
        supports.append(support)

    lengths = [s.size for s in supports]
    offsets = np.cumsum([0, *lengths[:-1]])
    n_q = int(sum(lengths))
    n_quotes = int(sum(len(books[name]) for name in names))
    # Calendar grid: the union of quoted strikes, so each expiry is compared where it was quoted.
    if len(names) >= 2:
        z = np.unique(np.concatenate([books[name]["x"].to_numpy(dtype=float) for name in names]))
    else:
        z = np.array([])
    n_cal = z.size * max(len(names) - 1, 0)
    n_slack = 2 * n_quotes + n_cal
    n_var = n_q + n_slack

    cost = np.zeros(n_var)
    cost[n_q:] = 1.0
    bounds = [(0.0, None)] * n_var

    a_eq = []
    b_eq = []
    for offset, support in zip(offsets, supports, strict=True):
        mass = np.zeros(n_var)
        mean = np.zeros(n_var)
        mass[offset : offset + support.size] = 1.0
        mean[offset : offset + support.size] = support
        a_eq.append(mass)
        b_eq.append(1.0)
        a_eq.append(mean)
        b_eq.append(1.0)

    a_ub = []
    b_ub = []
    # Quote slacks. Layout: [lo_0, hi_0, lo_1, hi_1, ...] then calendar slacks.
    slack_cursor = n_q
    quote_meta = []
    for name, support, offset in zip(names, supports, offsets, strict=True):
        book = books[name]
        matrix = np.maximum(support[None, :] - book["x"].to_numpy(dtype=float)[:, None], 0.0)
        for i in range(book.shape[0]):
            row_lo = np.zeros(n_var)
            row_hi = np.zeros(n_var)
            row_lo[offset : offset + support.size] = -matrix[i]
            row_hi[offset : offset + support.size] = matrix[i]
            row_lo[slack_cursor] = -1.0
            row_hi[slack_cursor + 1] = -1.0
            a_ub.append(row_lo)
            b_ub.append(-float(book["bid"].iloc[i]))
            a_ub.append(row_hi)
            b_ub.append(float(book["ask"].iloc[i]))
            quote_meta.append(
                {
                    "lo": slack_cursor,
                    "hi": slack_cursor + 1,
                    "T": float(book["T"].iloc[i]),
                    "k": float(np.log(book["x"].iloc[i])),
                }
            )
            slack_cursor += 2

    cal_meta = []
    for e in range(len(names) - 1):
        left_m = _call_on(supports[e], z)
        right_m = _call_on(supports[e + 1], z)
        for i in range(z.size):
            row = np.zeros(n_var)
            row[offsets[e] : offsets[e] + lengths[e]] = left_m[i]
            row[offsets[e + 1] : offsets[e + 1] + lengths[e + 1]] = -right_m[i]
            row[slack_cursor] = -1.0
            # C_right - C_left + slack >= 0  =>  C_left - C_right - slack <= 0.
            a_ub.append(row)
            b_ub.append(0.0)
            maturity = float(books[names[e + 1]]["T"].iloc[0])
            cal_meta.append(
                {"slack": slack_cursor, "T": maturity, "k": float(np.log(z[i]))}
            )
            slack_cursor += 1

    solved = linprog(
        cost,
        A_ub=np.vstack(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.vstack(a_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not solved.success or solved.x is None:
        return ArbReport(
            violations=_empty(),
            masks={},
            min_density=float("nan"),
            ok=False,
            note=f"HiGHS did not return a solution ({solved.message}).",
        )
    records = []
    for meta in quote_meta:
        for kind, column in (("quote_bid", meta["lo"]), ("quote_ask", meta["hi"])):
            magnitude = float(solved.x[column])
            if magnitude > max(tol.price, _SOLVER_TOL):
                records.append(_record(kind, meta["T"], meta["k"], magnitude, tol))
    for meta in cal_meta:
        magnitude = float(solved.x[meta["slack"]])
        if magnitude > max(tol.price, _SOLVER_TOL):
            records.append(_record("calendar", meta["T"], meta["k"], magnitude, tol))
    violations = (
        pd.DataFrame.from_records(records, columns=_VIOLATION_COLUMNS) if records else _empty()
    )
    return ArbReport(
        violations=violations,
        masks={},
        min_density=float("nan"),
        ok=violations.empty,
        note=f"HiGHS status {solved.message}. Objective (total slack) {solved.fun:.3e}.",
    )


def _call_on(support: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``M[i, j] = max(support[j] - z[i], 0)``."""
    return np.maximum(support[None, :] - z[:, None], 0.0)


def _record(kind: str, T: float, k: float, magnitude: float, tol: ArbTol) -> dict:
    return {
        "kind": kind,
        "T": T,
        "k": k,
        "magnitude": magnitude,
        "magnitude_in_spreads": magnitude / tol.spreads if tol.spreads else float("nan"),
    }
