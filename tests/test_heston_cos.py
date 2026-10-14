"""Heston cosine prices, Euler full truncation, and calibration.

The Fang and Oosterlee reference is 5.785155450 at T = 1. Lewis and the
cosine expansion are both required to 1e-7 there. The cosine interval uses
L = 10 and N = 256, the defaults on ``cos_price``.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from volsurface import bs
from volsurface.arbitrage import check_surface
from volsurface.core.protocols import SupportsSimulate, capabilities
from volsurface.core.types import Curves, Domain, MarketData
from volsurface.fit.heston import HestonFitter
from volsurface.mc.rng import generator, spawn
from volsurface.mc.zrule import check_z, max_abs_bound
from volsurface.models.bs import BlackScholes
from volsurface.models.heston import Heston, euler_price
from volsurface.surface.model import ModelSurface

# Fang and Oosterlee (2008), section 5.2. Feller ratio is about 0.38.
FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
FO_CALL = 5.785155450


def test_fang_oosterlee_reference_by_lewis_and_cos():
    model = Heston(**FO)
    lewis = model.price(100.0, np.array([100.0]), 1.0)[0]
    cos = model.cos_price(100.0, np.array([100.0]), 1.0)[0]
    assert lewis == pytest.approx(FO_CALL, abs=1e-7)
    assert cos == pytest.approx(FO_CALL, abs=1e-7)
    assert model.feller_ratio() < 1.0


def test_lewis_and_cos_agree_on_a_grid_including_the_wings_and_one_week():
    """FO parameters, which violate Feller, on one week through two years.

    Strikes run from k = -1 to k = 1. N = 256, L = 10. The tolerance is
    1e-6 in cash price, wider than the T = 1 at-the-money gap (about 8e-8)
    and tighter than a wrong characteristic function.
    """
    model = Heston(**FO)
    assert model.feller_ratio() == pytest.approx(0.3794922785422908, rel=1e-12)
    maturities = np.array([7.0 / 365.0, 0.25, 1.0, 2.0])
    log_k = np.linspace(-1.0, 1.0, 9)
    worst = 0.0
    for maturity in maturities:
        strike = 100.0 * np.exp(log_k)
        lewis = model.price(100.0, strike, float(maturity))
        cos = model.cos_price(100.0, strike, float(maturity))
        worst = max(worst, float(np.max(np.abs(lewis - cos))))
    assert worst < 1e-6, worst


def test_cos_matches_lewis_on_a_second_feller_violation():
    """kappa = 1, theta = v0 = 0.04, xi = 0.5, rho = -0.7. Feller ratio 0.32."""
    model = Heston(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.7)
    assert model.feller_ratio() == pytest.approx(0.32)
    strike = 100.0 * np.exp(np.linspace(-0.8, 0.8, 7))
    for maturity in (7.0 / 365.0, 1.0):
        gap = np.max(np.abs(
            model.price(100.0, strike, maturity) - model.cos_price(100.0, strike, maturity)
        ))
        assert gap < 1e-6, (maturity, gap)


def test_cos_converges_to_black_scholes_as_xi_vanishes():
    """v0 = theta and xi = 1e-4. The cumulant stencil's c4 is negative and replaced by 0.

    The remaining gap is the vol-of-vol, the same size for Lewis and for COS.
    A smaller xi makes the degree-8 stencil on log phi unusable (c4 of order
    100), which is why this test stops at 1e-4.
    """
    theta = 0.04
    model = Heston(v0=theta, kappa=1.5, theta=theta, xi=1e-4, rho=-0.5)
    black = BlackScholes(sigma=np.sqrt(theta))
    strike = np.array([70.0, 85.0, 100.0, 115.0, 140.0])
    for maturity in (0.25, 1.0):
        cos = model.cos_price(100.0, strike, maturity)
        lewis = model.price(100.0, strike, maturity)
        reference = black.price(100.0, strike, maturity)
        # The largest gap on this grid is about 2.1e-4, at K = 115, T = 1.
        np.testing.assert_allclose(cos, reference, atol=5e-4)
        np.testing.assert_allclose(lewis, reference, atol=5e-4)
        np.testing.assert_allclose(cos, lewis, atol=1e-6)


def test_heston_simulates_and_the_draw_order_is_forward_then_variance():
    model = Heston(**FO, n_paths=4, n_steps=1, seed=7)
    assert isinstance(model, SupportsSimulate)
    assert "simulate" in capabilities(model)
    times = np.array([0.0, 0.25])
    paths = model.simulate(4, times, seed=7)
    assert paths.Z.shape == (4, 2)
    assert paths.state["v"].shape == (4, 2)
    assert np.all(paths.Z[:, 0] == 1.0)
    rng = generator(spawn(7, 1)[0])
    shock_s = rng.standard_normal(4)
    shock_v = rng.standard_normal(4)
    dt = 0.25
    mixed = model.rho * shock_s + np.sqrt(1.0 - model.rho**2) * shock_v
    floor = np.maximum(paths.state["v"][:, 0], 0.0)
    log_z = -0.5 * floor * dt + np.sqrt(floor) * np.sqrt(dt) * shock_s
    variance = paths.state["v"][:, 0] + model.kappa * (model.theta - floor) * dt
    variance = variance + model.xi * np.sqrt(floor) * np.sqrt(dt) * mixed
    np.testing.assert_allclose(paths.Z[:, 1], np.exp(log_z))
    np.testing.assert_allclose(paths.state["v"][:, 1], variance)


def test_euler_matches_lewis_by_the_z_rule():
    """24 independent prices. 8,000 paths, 64 steps, seed per node.

    Strikes are k in [-0.15, 0.15] at T = 1. Parameters are inside the Feller
    region (ratio 4.8) so the Euler bias stays inside the standard error.
    Nodes do not share paths, so the fraction band applies. The max bound is
    the z-rule value at n = 24.
    """
    model = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.25, rho=-0.4)
    assert model.feller_ratio() > 1.0
    log_k = np.linspace(-0.15, 0.15, 6)
    maturities = np.array([0.5, 1.0, 1.5, 2.0])
    estimates = []
    errors = []
    references = []
    node = 0
    for maturity in maturities:
        strike = 100.0 * np.exp(log_k)
        reference = model.price(100.0, strike, float(maturity))
        for j, one in enumerate(strike):
            result = euler_price(
                model, 100.0, np.array([one]), float(maturity),
                n_paths=8_000, n_steps=64, seed=100 + node,
            )
            estimates.append(result.value[0])
            errors.append(result.stderr[0])
            references.append(reference[j])
            node += 1
    z = (np.array(estimates) - np.array(references)) / np.array(errors)
    n = z.size
    assert n == 24
    assert max_abs_bound(n) < 4.5
    failure = check_z(z)
    assert failure is None, failure


def test_put_call_parity_on_the_cosine_price():
    model = Heston(**FO)
    strike = np.array([80.0, 100.0, 130.0])
    call = model.cos_price(100.0, strike, 1.0, df=0.97, kind="call")
    put = model.cos_price(100.0, strike, 1.0, df=0.97, kind="put")
    np.testing.assert_allclose(call - put, 0.97 * (100.0 - strike), atol=1e-10)


def test_synthetic_calibration_recovers_the_parameters_and_the_prices():
    """Known parameters, exact prices, vega-weighted least squares, four starts.

    The truth is not the fitter's default start. Prices come back to 1e-8
    (the run sits near 1e-11). Parameter tolerances are absolute and wider
    than that gap; the test fails if a start is dropped and the fit settles
    elsewhere.
    """
    truth = Heston(v0=0.03, kappa=1.2, theta=0.05, xi=0.4, rho=-0.6)
    data = _quotes_from(truth, half_spread=None)
    fitted = HestonFitter(n_starts=4, loss="vega", seed=1).fit(data)
    row = fitted.slice_report.iloc[0]
    assert row["v0"] == pytest.approx(truth.v0, abs=1e-8)
    assert row["kappa"] == pytest.approx(truth.kappa, abs=1e-6)
    assert row["theta"] == pytest.approx(truth.theta, abs=1e-8)
    assert row["xi"] == pytest.approx(truth.xi, abs=1e-6)
    assert row["rho"] == pytest.approx(truth.rho, abs=1e-6)
    assert row["max_abs_price"] < 1e-8
    assert row["feller"] == pytest.approx(truth.feller_ratio(), rel=1e-6)
    assert fitted.wall_time > 0.0
    assert "cost=" in fitted.solver_status
    # No bid or ask on this book, so in_spread is NaN rather than a pass.
    assert np.isnan(fitted.in_spread)


def test_in_spread_loss_lands_inside_the_quotes():
    truth = Heston(v0=0.03, kappa=1.2, theta=0.05, xi=0.4, rho=-0.6)
    data = _quotes_from(truth, half_spread=0.1)
    fitted = HestonFitter(n_starts=3, loss="in_spread", seed=2).fit(data)
    assert fitted.in_spread == pytest.approx(1.0)


def test_heston_surface_has_no_arbitrage_on_the_grid():
    """A Heston ModelSurface on a grid inside the quoted rectangle.

    Derivatives are finite differences, so the butterfly tolerance is the
    checker's 1e-8 floor. The grid is five expiries by nine strikes.
    """
    model = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.5)
    market = Curves(
        F=lambda t: np.full(np.shape(t), 100.0, dtype=float),
        df=lambda t: np.ones(np.shape(t), dtype=float),
    )
    domain = Domain(t_min=0.1, t_max=1.0, k_min=-0.3, k_max=0.3)
    surface = ModelSurface(model, market, domain)
    maturity = np.linspace(0.2, 0.8, 5)
    log_k = np.linspace(-0.2, 0.2, 9)
    report = check_surface(surface, maturity, log_k)
    assert report.ok, report.violations
    assert report.violations.empty
    assert report.n_checked == maturity.size * log_k.size


def _quotes_from(model: Heston, half_spread: float | None) -> MarketData:
    """European prices of ``model`` on three expiries and seven strikes."""
    forward = 100.0
    rows = []
    for maturity in (0.25, 1.0, 2.0):
        log_k = np.linspace(-0.3, 0.3, 7)
        strike = forward * np.exp(log_k)
        kind = np.where(log_k < 0.0, "put", "call")
        price = model.price(forward, strike, maturity, 1.0, kind)
        iv = bs.implied_vol(price, forward, strike, maturity, 1.0, kind)
        for i in range(strike.size):
            mid = float(price[i])
            if half_spread is None:
                bid = ask = np.nan
            else:
                bid = mid - half_spread
                ask = mid + half_spread
            rows.append(
                {
                    "T": maturity,
                    "K": float(strike[i]),
                    "kind": str(kind[i]),
                    "bid": bid,
                    "ask": ask,
                    "exercise": "european",
                    "settlement": "cash",
                    "volume": 1.0,
                    "price": mid,
                    "iv": float(iv[i]),
                    "k": float(log_k[i]),
                    "F": forward,
                    "df": 1.0,
                    "expiry": f"T{maturity}",
                }
            )
    frame = pd.DataFrame(rows)

    def flat(level):
        def curve(t):
            return np.full(np.shape(t), level, dtype=float)

        return curve

    return MarketData(
        asof=dt.datetime(2026, 10, 7, tzinfo=dt.UTC),
        quotes=frame,
        F=flat(forward),
        df=flat(1.0),
        F_err=None,
        day_count="ACT/365",
    )
