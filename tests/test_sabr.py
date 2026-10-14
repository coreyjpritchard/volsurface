"""Hagan's SABR formula, log-Euler, and calibration with beta held fixed.

beta = 1 and nu = 0 is Black at volatility alpha. The Monte Carlo comparison
is a z-rule on a grid where nu^2 T is a few hundredths. A long maturity and
a large nu is a negative density, checked by ``check_surface``, not a claim
that the formula and the simulation agree there.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from volsurface import bs
from volsurface.arbitrage import check_surface
from volsurface.core.protocols import SupportsSimulate, capabilities
from volsurface.core.types import Curves, Domain, MarketData
from volsurface.fit.sabr import SABRFitter
from volsurface.mc.rng import generator, spawn
from volsurface.mc.zrule import check_z, max_abs_bound
from volsurface.models.bs import BlackScholes
from volsurface.models.sabr import SABR, euler_price
from volsurface.plugins import create
from volsurface.surface.model import ModelSurface


def test_registry_builds_sabr():
    built = create("sabr", alpha=0.2, beta=1.0, rho=-0.2, nu=0.1)
    assert isinstance(built, SABR)
    assert built.initial_state()["alpha"] == pytest.approx(0.2)


def test_beta_one_and_nu_zero_is_a_flat_smile_at_alpha():
    """The Hagan factor z/x(z) is 1 at z = 0, and every maturity correction vanishes."""
    alpha = 0.2
    model = SABR(alpha=alpha, beta=1.0, rho=-0.4, nu=0.0)
    strike = 100.0 * np.exp(np.linspace(-1.0, 1.0, 21))
    for maturity in (0.25, 1.0, 3.0):
        iv = model.implied_vol(100.0, strike, maturity)
        assert np.max(np.abs(iv - alpha)) < 1e-12
        black = BlackScholes(alpha).price(100.0, strike, maturity)
        np.testing.assert_allclose(model.price(100.0, strike, maturity), black, atol=1e-12)


def test_at_the_money_limit_is_the_closed_form():
    """beta = 1, so the only maturity terms are the rho beta nu piece and nu^2.

    alpha = 0.2, rho = -0.3, nu = 0.4, T = 1, F = 100.
    correction = 0.25 * rho * nu * alpha + (2 - 3 rho^2) / 24 * nu^2
    = -0.006 + 0.0115333... = 0.0055333..., and the vol is 0.2011066...
    A strike one ulp of log-moneyness off the forward uses the same value.
    """
    model = SABR(alpha=0.2, beta=1.0, rho=-0.3, nu=0.4)
    rho, nu, alpha = model.rho, model.nu, model.alpha
    correction = 0.25 * rho * nu * alpha + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    expected = alpha * (1.0 + correction)
    got = float(model.implied_vol(100.0, np.array([100.0]), 1.0)[0])
    assert got == pytest.approx(expected, rel=0, abs=1e-15)
    assert got == pytest.approx(0.20110666666666668, abs=1e-15)
    neighbour = float(model.implied_vol(100.0, np.array([100.0 * np.exp(1e-8)]), 1.0)[0])
    assert neighbour == pytest.approx(got, abs=1e-8)


def test_draw_order_is_forward_then_volatility():
    model = SABR(alpha=0.2, beta=1.0, rho=-0.4, nu=0.3)
    assert isinstance(model, SupportsSimulate)
    assert "simulate" in capabilities(model)
    times = np.array([0.0, 0.25])
    paths = model.simulate(4, times, seed=7)
    assert paths.Z.shape == (4, 2)
    assert paths.state["alpha"].shape == (4, 2)
    assert np.all(paths.Z[:, 0] == 1.0)
    rng = generator(spawn(7, 1)[0])
    shock_f = rng.standard_normal(4)
    shock_v = rng.standard_normal(4)
    dt = 0.25
    mixed = model.rho * shock_f + np.sqrt(1.0 - model.rho**2) * shock_v
    local = model.alpha
    log_z = -0.5 * local**2 * dt + local * np.sqrt(dt) * shock_f
    log_a = np.log(model.alpha) - 0.5 * model.nu**2 * dt + model.nu * np.sqrt(dt) * mixed
    np.testing.assert_allclose(paths.Z[:, 1], np.exp(log_z))
    np.testing.assert_allclose(paths.state["alpha"][:, 1], np.exp(log_a))


def test_hagan_matches_log_euler_when_nu_squared_t_is_small():
    """nu = 0.15, beta = 1, rho = -0.2. At T = 1, nu^2 T = 0.0225.

    Twelve independent nodes, 8,000 paths, 32 steps. The z-rule at n = 12
    allows max |z| below 3.93. The Euler bias is inside that band on this
    grid. A larger nu^2 T is the figure, not this assertion.
    """
    model = SABR(alpha=0.2, beta=1.0, rho=-0.2, nu=0.15)
    log_k = np.linspace(-0.12, 0.12, 6)
    maturities = np.array([0.5, 1.0])
    estimates = []
    errors = []
    references = []
    node = 0
    for maturity in maturities:
        strike = 100.0 * np.exp(log_k)
        reference = model.price(100.0, strike, float(maturity))
        for j, one in enumerate(strike):
            result = euler_price(
                model,
                100.0,
                np.array([one]),
                float(maturity),
                n_paths=8_000,
                n_steps=32,
                seed=50 + node,
            )
            estimates.append(result.value[0])
            errors.append(result.stderr[0])
            references.append(reference[j])
            node += 1
    z = (np.array(estimates) - np.array(references)) / np.array(errors)
    assert z.size == 12
    assert max_abs_bound(z.size) == pytest.approx(3.9346058615604647, rel=1e-12)
    failure = check_z(z)
    assert failure is None, failure


def test_beta_below_one_absorbs_at_the_bessel_rate():
    """Unit forward, alpha = 0.8, beta = 0.5, nu = 0, 64 steps, seed 2.

    With nu = 0 the hitting probability is exp(-2 / (a^2 T)) = exp(-3.125).
    a = alpha, because the forward starts at 1. 20,000 paths. The estimate
    has to sit within 4 standard errors of that probability, and the terminal
    mean of Z within 4 standard errors of 1. A path that has hit 0 does not
    leave it.
    """
    model = SABR(alpha=0.8, beta=0.5, rho=0.0, nu=0.0)
    n_paths = 20_000
    paths = model.simulate(n_paths, np.linspace(0.0, 1.0, 65), seed=2)
    terminal = paths.Z[:, -1]
    hit_terminal = terminal == 0.0
    probability = float(hit_terminal.mean())
    se_p = float(np.sqrt(probability * (1.0 - probability) / n_paths))
    exact = float(np.exp(-2.0 / (0.8**2)))
    assert abs(probability - exact) < 4.0 * se_p, (probability, exact, se_p)
    mean_z = float(terminal.mean())
    se_z = float(terminal.std(ddof=1) / np.sqrt(n_paths))
    assert abs(mean_z - 1.0) < 4.0 * se_z, (mean_z, se_z)
    revived = 0
    ever = np.any(paths.Z == 0.0, axis=1)
    for row in paths.Z[ever]:
        seen = False
        for value in row:
            if value == 0.0:
                seen = True
            elif seen:
                revived += 1
                break
    assert revived == 0


def test_synthetic_calibration_recovers_each_expiry():
    """beta is fixed at 1. Each expiry recovers the prices that built that slice."""
    truth = SABR(alpha=0.22, beta=1.0, rho=-0.45, nu=0.35)
    fitted = SABRFitter(beta=1.0, n_starts=4, seed=0).fit(_quotes_from(truth))
    assert list(fitted.slice_report["T"]) == [0.5, 1.0, 1.5]
    for row in fitted.slice_report.itertuples(index=False):
        assert row.beta == pytest.approx(1.0)
        assert row.alpha == pytest.approx(0.22, abs=1e-8)
        assert row.rho == pytest.approx(-0.45, abs=1e-8)
        assert row.nu == pytest.approx(0.35, abs=1e-8)
        assert row.max_abs_price < 1e-9
    assert fitted.wall_time < 5.0


def test_check_surface_finds_a_negative_density_at_long_maturity():
    """alpha = 0.3, beta = 1, rho = -0.3, nu = 1.5, T = 10.

    Implied vol on this strike window stays positive. The call is still not
    convex: ``check_surface`` reports a butterfly, and the log-density goes
    negative. That is the Hagan expansion leaving the regime where it is a
    density, not a bug in the checker.
    """
    model = SABR(alpha=0.3, beta=1.0, rho=-0.3, nu=1.5)
    log_k = np.linspace(-1.5, 0.5, 41)
    surface = ModelSurface(model, _market(), Domain(0.5, 12.0, -2.0, 1.0))
    report = check_surface(surface, np.array([10.0]), log_k)
    assert report.ok is False
    assert "butterfly" in set(report.violations["kind"])
    assert report.min_density < 0.0
    iv = model.implied_vol(100.0, 100.0 * np.exp(log_k), 10.0)
    assert np.all(iv > 0.0)


def _market(forward: float = 100.0) -> Curves:
    def curve(level):
        return lambda t: np.full(np.shape(t), level, dtype=float)

    return Curves(F=curve(forward), df=curve(1.0))


def _quotes_from(model: SABR) -> MarketData:
    forward = 100.0
    rows = []
    for maturity in (0.5, 1.0, 1.5):
        log_k = np.linspace(-0.25, 0.25, 7)
        strike = forward * np.exp(log_k)
        kind = np.where(log_k < 0.0, "put", "call")
        price = model.price(forward, strike, maturity, 1.0, kind)
        iv = bs.implied_vol(price, forward, strike, maturity, 1.0, kind)
        for i in range(strike.size):
            rows.append(
                {
                    "T": maturity,
                    "K": float(strike[i]),
                    "kind": str(kind[i]),
                    "bid": np.nan,
                    "ask": np.nan,
                    "exercise": "european",
                    "settlement": "cash",
                    "volume": 1.0,
                    "price": float(price[i]),
                    "iv": float(iv[i]),
                    "k": float(log_k[i]),
                    "F": forward,
                    "df": 1.0,
                    "expiry": f"T{maturity}",
                }
            )
    frame = pd.DataFrame(rows)

    def flat(level):
        return lambda t: np.full(np.shape(t), level, dtype=float)

    return MarketData(
        asof=dt.datetime(2026, 10, 7, tzinfo=dt.UTC),
        quotes=frame,
        F=flat(forward),
        df=flat(1.0),
        F_err=None,
        day_count="ACT/365",
    )
