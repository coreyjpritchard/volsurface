"""SANOS at eta = 0. DESIGN section 18, the linear program (MDL)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from volsurface.arbitrage import check_surface
from volsurface.core.types import Curves, Domain, MarketData
from volsurface.fit.sanos import SanosFitter, project_marginal
from volsurface.models.heston import Heston
from volsurface.surface.sanos import SanosSurface, black_call


def _curves(forward: float = 100.0):
    def flat(level):
        def curve(t):
            return np.full(np.shape(t), level, dtype=float)

        return curve

    return flat(forward), flat(1.0)


def _heston_book(half_spread: float) -> MarketData:
    """Three expiries, one common strike grid, Heston mids with a cash half-spread."""
    model = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.5)
    forward = 100.0
    rows = []
    for maturity in (0.25, 0.5, 1.0):
        log_k = np.linspace(-0.3, 0.3, 9)
        strike = forward * np.exp(log_k)
        kind = np.where(log_k < 0.0, "put", "call")
        price = np.asarray(model.price(forward, strike, maturity, 1.0, kind), dtype=float)
        for i in range(strike.size):
            mid = float(price[i])
            rows.append(
                {
                    "T": maturity,
                    "K": float(strike[i]),
                    "kind": str(kind[i]),
                    "bid": mid - half_spread,
                    "ask": mid + half_spread,
                    "exercise": "european",
                    "settlement": "cash",
                    "volume": 1.0,
                    "price": mid,
                    "k": float(log_k[i]),
                    "F": forward,
                    "df": 1.0,
                    "expiry": f"T{maturity}",
                }
            )
    spot, discount = _curves(forward)
    return MarketData(
        asof=dt.datetime(2026, 10, 14, tzinfo=dt.UTC),
        quotes=pd.DataFrame(rows),
        F=spot,
        df=discount,
        F_err=None,
        day_count="ACT/365",
    )


def _known_surface() -> SanosSurface:
    """Two pillars, atoms at 0.5, 1 and 1.5, mass 1 and mean 1."""
    weights = np.array([0.25, 0.5, 0.25])
    strikes = np.array([0.5, 1.0, 1.5])
    spot, discount = _curves(1.0)
    return SanosSurface(
        np.array([0.5, 1.0]),
        [strikes, strikes],
        [weights, weights],
        np.array([0.04, 0.08]),
        0.0,
        Curves(spot, discount),
        Domain(0.5, 1.0, float(np.log(0.5)), float(np.log(1.5))),
    )


def test_linear_limit_is_piecewise_linear_in_the_strike():
    """At a pillar, eta = 0 is sum q (K - x)+ to 1e-12."""
    surface = _known_surface()
    weights = surface.weights[0]
    strikes = surface.strikes[0]
    log_k = np.linspace(-0.7, 0.5, 13)
    x = np.exp(log_k)
    expected = weights @ np.maximum(strikes[:, None] - x[None, :], 0.0)
    got = surface.call(np.array([0.5]), log_k)[0]
    assert got == pytest.approx(expected, abs=1e-12)
    # The same formula is black_call at zero variance.
    kernel = black_call(strikes[:, None], x[None, :], 0.0)
    assert weights @ kernel == pytest.approx(expected, abs=1e-12)


def test_projection_restores_mass_and_mean():
    raw = np.array([-0.2, 0.4, 0.9, 0.1])
    strikes = np.array([0.4, 0.8, 1.2, 2.0])
    projected, correction = project_marginal(raw, strikes)
    assert np.all(projected >= 0.0)
    assert projected.sum() == pytest.approx(1.0, abs=1e-12)
    assert float(projected @ strikes) == pytest.approx(1.0, abs=1e-12)
    assert correction > 0.0


def test_heston_book_is_recovered_inside_the_spread():
    """Objective near 0, every two-sided quote inside the spread."""
    book = _heston_book(half_spread=0.15)
    result = SanosFitter(eta=0.0, omega=0.0, dx=0.05).fit(book)
    assert result.detail["objective"] == pytest.approx(0.0, abs=1e-6)
    assert result.in_spread == pytest.approx(1.0)
    assert result.detail["omega_label"] == "calendar guaranteed"
    surface = result.surface
    for weights, strikes in zip(surface.weights, surface.strikes, strict=True):
        assert np.all(weights >= -1e-15)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert float(weights @ strikes) == pytest.approx(1.0, abs=1e-12)


def test_calendar_holds_off_the_quoted_expiries():
    """omega = 0. Dense times that are not pillars have no calendar violation."""
    book = _heston_book(half_spread=0.15)
    result = SanosFitter(eta=0.0, omega=0.0, dx=0.05).fit(book)
    times = np.array([0.30, 0.37, 0.44, 0.63, 0.71, 0.88, 0.97])
    log_k = np.linspace(-0.25, 0.25, 11)
    report = check_surface(result.surface, times, log_k)
    calendar = report.violations[report.violations["kind"] == "calendar"]
    assert calendar.empty
    assert report.n_checked == times.size * log_k.size


def test_objective_and_quote_prices_match_across_highs_methods():
    """Two runs, and highs against highs-ds. q itself is not compared."""
    book = _heston_book(half_spread=0.15)
    first = SanosFitter(method="highs").fit(book)
    second = SanosFitter(method="highs").fit(book)
    dual = SanosFitter(method="highs-ds").fit(book)
    assert first.detail["objective"] == pytest.approx(second.detail["objective"], abs=1e-9)
    assert first.detail["solver_quote_prices"] == pytest.approx(
        second.detail["solver_quote_prices"], abs=1e-9
    )
    assert first.detail["objective"] == pytest.approx(dual.detail["objective"], abs=1e-9)
    assert first.detail["solver_quote_prices"] == pytest.approx(
        dual.detail["solver_quote_prices"], abs=1e-9
    )
