"""Arbitrage checker: planted magnitudes, a martingale, the Vogt slice, the snapshot."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tests.test_surface import _Bump, _flat_market
from volsurface.arbitrage import check_quotes, check_surface
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import ArbTol, Curves, Domain, MarketData
from volsurface.surface.linear import LinearSurface
from volsurface.surface.svi import RawSVI, SVISurface


def test_planted_violations_have_the_right_node_and_magnitude():
    """One call-spread, one butterfly, one calendar, each with a known size."""
    surface = _Bump()
    T = np.array([0.3, 0.5, 1.0])
    k = np.array([-0.2, 0.0, 0.2])
    report = check_surface(surface, T, k, tol=ArbTol(price=1e-12))
    assert not report.ok
    assert "between the nodes" in report.note
    expected = {
        "call_spread": (0.5, 0.0, 0.25),
        "butterfly": (1.0, 0.2, 0.05),
        "calendar": (0.3, -0.2, 0.02),
    }
    for kind, (t, log_k, magnitude) in expected.items():
        rows = report.violations[report.violations["kind"] == kind]
        assert len(rows) == 1, kind
        row = rows.iloc[0]
        assert row["T"] == pytest.approx(t)
        assert row["k"] == pytest.approx(log_k)
        assert row["magnitude"] == pytest.approx(magnitude, abs=1e-12)
    assert report.masks["call_spread"][1, 1]
    assert not report.violations["kind"].eq("bounds").any()


def test_flat_spline_passes_and_a_crossed_calendar_is_flagged():
    from volsurface.surface.spline import SplineSurface

    k = np.linspace(-0.3, 0.3, 7)
    market = _flat_market()
    flat = SplineSurface(
        [
            {"T": 0.25, "k": k, "iv": np.full_like(k, 0.2)},
            {"T": 1.0, "k": k, "iv": np.full_like(k, 0.2)},
        ],
        market,
        Domain(0.25, 1.0, k[0], k[-1]),
    )
    grid_k = np.linspace(k[0], k[-1], 9)
    # Avoid the exact pillars' edge knots, where the natural cubic's second
    # derivative is constrained and a one-sided blend is not the question.
    report = check_surface(flat, np.array([0.25, 0.6, 1.0]), grid_k, tol=ArbTol(price=1e-8))
    assert report.ok, report.violations

    crossed = SplineSurface(
        [
            {"T": 0.25, "k": k, "iv": np.full_like(k, 0.4)},
            {"T": 1.0, "k": k, "iv": np.full_like(k, 0.05)},
        ],
        market,
        Domain(0.25, 1.0, k[0], k[-1]),
    )
    # w = iv^2 T falls from 0.04 to 0.0025, so the normalised call falls in T.
    bad = check_surface(crossed, np.array([0.25, 1.0]), np.array([0.0]), tol=ArbTol(price=1e-6))
    calendar = bad.violations[bad.violations["kind"] == "calendar"]
    assert not calendar.empty


def test_vogt_surface_butterfly_matches_g():
    vogt = RawSVI.vogt()
    surface = SVISurface([(1.0, vogt)], _flat_market(), Domain(1.0, 1.0, -1.5, 1.5))
    k = np.linspace(-1.2, 1.2, 81)
    report = check_surface(surface, np.array([1.0]), k, tol=ArbTol(price=1e-10))
    flies = report.violations[report.violations["kind"] == "butterfly"]
    assert not flies.empty
    # Magnitude is the negative part of d²C/dx², which is -exp(-k) times the density.
    from volsurface.surface.black import log_density

    phi = log_density(k, vogt.w(k), vogt.w_k(k), vogt.w_kk(k))
    worst = int(np.argmin(phi))
    row = flies.iloc[(flies["k"] - k[worst]).abs().argmin()]
    magnitude = -np.exp(-k[worst]) * phi[worst]
    assert magnitude > 0.0
    assert row["magnitude"] == pytest.approx(magnitude, rel=1e-6, abs=1e-8)


def _random_marginal(x, rng):
    """A probability on ``x`` with mean 1. ``x`` must sit on both sides of 1."""
    weight = rng.random(x.size) + 0.05
    lo, hi = -30.0, 30.0
    for _ in range(80):
        lam = 0.5 * (lo + hi)
        tilted = weight * np.exp(lam * x)
        tilted /= tilted.sum()
        mean = float(tilted @ x)
        if mean > 1.0:
            hi = lam
        else:
            lo = lam
    tilted = weight * np.exp(0.5 * (lo + hi) * x)
    tilted /= tilted.sum()
    return tilted


def _calls_from_q(x, q, query):
    return np.array([np.sum(q * np.maximum(x - xq, 0.0)) for xq in query])


def test_random_marginal_passes_and_a_dent_is_flagged():
    rng = np.random.default_rng(7)
    x = np.linspace(0.4, 1.8, 15)
    q = _random_marginal(x, rng)
    assert q.sum() == pytest.approx(1.0)
    assert q @ x == pytest.approx(1.0, abs=1e-8)
    calls = _calls_from_q(x, q, x)
    k = np.log(x)
    pillar = {"T": 0.5, "k": k, "x": x, "c": calls}
    surface = LinearSurface([pillar], _flat_market(), Domain(0.5, 0.5, k[0], k[-1]))
    # Interior nodes: the discrete butterfly lives on the slope jumps there.
    report = check_surface(surface, np.array([0.5]), k[1:-1], tol=ArbTol(price=1e-8))
    assert report.ok, report.violations

    # Raising an interior call price makes the slope fall across that node.
    dented = calls.copy()
    dented[7] += 0.02
    slopes = np.diff(dented) / np.diff(x)
    jump = slopes[7] - slopes[6]  # right slope at node 7 minus left slope
    bad_pillar = {"T": 0.5, "k": k, "x": x, "c": dented}
    bad = LinearSurface([bad_pillar], _flat_market(), Domain(0.5, 0.5, k[0], k[-1]))
    flagged = check_surface(bad, np.array([0.5]), k, tol=ArbTol(price=1e-10))
    flies = flagged.violations[flagged.violations["kind"] == "butterfly"]
    at = flies[np.isclose(flies["k"], k[7])]
    assert len(at) == 1
    # The reported convexity is the slope jump, so the magnitude is how far it falls.
    assert at.iloc[0]["magnitude"] == pytest.approx(-jump, rel=1e-6, abs=1e-8)


def _market(rows: list[dict]) -> MarketData:
    frame = pd.DataFrame(rows)
    def one(t):
        return np.ones_like(np.asarray(t, dtype=float), dtype=float)

    return MarketData(
        asof=dt.datetime(2026, 10, 7, tzinfo=dt.UTC),
        quotes=frame,
        F=one,
        df=one,
        F_err=None,
        day_count="ACT/365",
    )


def _quote_row(T, x, bid, ask, expiry="e1"):
    return {
        "expiry": expiry,
        "T": T,
        "K": x,  # F = 1, so the strike is the normalised strike
        "kind": "call",
        "bid": bid,
        "ask": ask,
        "price": 0.5 * (bid + ask),
        "F": 1.0,
        "df": 1.0,
        "k": float(np.log(x)),
        "exercise": "european",
        "settlement": "cash",
        "volume": 1.0,
    }


def test_quote_lp_accepts_a_martingale_and_prices_a_bid_above_one():
    x = np.array([0.5, 1.0, 1.5])
    q = np.array([0.25, 0.5, 0.25])
    calls = _calls_from_q(x, q, x)
    rows = []
    for strike, price in zip(x, calls, strict=True):
        rows.append(_quote_row(0.5, strike, price - 0.02, price + 0.02))
    ok = check_quotes(_market(rows), side="bidask", tol=ArbTol(price=1e-8))
    assert ok.ok, ok.violations

    # One quoted strike, x = 1, bid 1. The support used by the LP is {0, 1, 3},
    # and the largest call that support can make at x = 1 is (3 - 1) / 3 = 2/3.
    # The slack is 1 - 2/3.
    lone = [_quote_row(0.5, 1.0, 1.0, 1.1)]
    bad = check_quotes(_market(lone), side="bidask", tol=ArbTol(price=1e-8))
    assert not bad.ok
    hit = bad.violations[bad.violations["kind"] == "quote_bid"]
    assert len(hit) == 1
    assert hit["magnitude"].iloc[0] == pytest.approx(1.0 - 2.0 / 3.0, abs=1e-5)


def test_quote_lp_calendar_slack_equals_the_price_gap():
    x = np.array([0.5, 1.0, 1.5])
    calm = np.array([0.0, 1.0, 0.0])  # point mass at 1, calls are intrinsic
    lively = np.array([0.5, 0.0, 0.5])  # mass at 0.5 and 1.5
    # The lively book is the longer expiry in calendar order only if we pass it second.
    # Here the short expiry is lively and the long expiry is the point mass, so calls fall.
    # Both books are pinned to their own calls. The short book is worth 0.25 at
    # x = 1 and the long book is worth 0. Equal weights on quote slacks and
    # calendar slacks mean the solver may book the gap on either; the size of
    # the gap is what is identified.
    rows = []
    for strike, price in zip(x, _calls_from_q(x, lively, x), strict=True):
        rows.append(_quote_row(0.4, strike, price, price, expiry="short"))
    for strike, price in zip(x, _calls_from_q(x, calm, x), strict=True):
        rows.append(_quote_row(1.0, strike, price, price, expiry="long"))
    report = check_quotes(_market(rows), side="bidask", tol=ArbTol(price=1e-8))
    assert not report.ok
    at_one = report.violations[np.isclose(report.violations["k"], 0.0)]
    assert at_one["magnitude"].sum() == pytest.approx(0.25, abs=1e-4)


def test_committed_snapshot_runs_the_checker():
    """The SPY file has last trades and no bid or ask. The check still runs."""
    from volsurface import chains
    from volsurface.surface.linear import linear_surface
    from volsurface.surface.spline import spline_surface

    chain = chains.load_snapshot(chains.latest_snapshot("SPY"))
    data = market_data_from_snapshot(chain)
    quotes = check_quotes(data, side="bidask")
    assert not quotes.ok
    assert "Dropped" in quotes.note or "dropped" in quotes.note
    assert quotes.violations.empty

    # Two expiries are enough to exercise calendar and the real strike grid.
    expiries = list(data.quotes.groupby("expiry", sort=True).groups)[:2]
    subset = data.quotes[data.quotes["expiry"].isin(expiries)].copy()
    linear = linear_surface(subset, Curves(data.F, data.df))
    spline = spline_surface(subset, Curves(data.F, data.df))
    k = np.linspace(subset["k"].quantile(0.1), subset["k"].quantile(0.9), 11)
    T = subset.groupby("expiry")["T"].first().to_numpy()
    for surface in (linear, spline):
        report = check_surface(surface, T, k)
        assert list(report.violations.columns) == [
            "kind", "T", "k", "magnitude", "magnitude_in_spreads",
        ]
        assert "between the nodes" in report.note
        assert np.isfinite(report.min_density) or report.min_density != report.min_density
