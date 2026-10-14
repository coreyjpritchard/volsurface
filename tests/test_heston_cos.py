"""Heston cosine prices, Euler full truncation, and calibration.

Fang and Oosterlee publish 5.785155450, nine digits, at T = 1. A 50-digit
Lewis integral is 5.78515543438. The library Lewis is required to 1e-10 of
that value, and the cosine price to 1e-7. ``L = 10``. ``N`` is chosen from
the interval unless a test passes ``n_terms``.
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
# 50-digit Lewis integral of the trap-free characteristic function. The
# published 5.785155450 is that value rounded to nine digits, 1.56e-8 high.
LEWIS_ATM = 5.78515543438


def test_fang_oosterlee_reference_by_lewis_and_cos():
    model = Heston(**FO)
    lewis = model.price(100.0, np.array([100.0]), 1.0)[0]
    cos = model.cos_price(100.0, np.array([100.0]), 1.0)[0]
    assert lewis == pytest.approx(LEWIS_ATM, abs=1e-10)
    assert cos == pytest.approx(LEWIS_ATM, abs=1e-7)
    assert model.feller_ratio() < 1.0


def test_lewis_and_cos_agree_on_a_grid_including_the_wings_and_one_week():
    """FO parameters, which violate Feller, on one week through two years.

    Strikes run from k = -1 to k = 1. L = 10 and N follows the interval.
    The tolerance is 1e-6 in cash price, wider than the T = 1 at-the-money
    gap and tighter than a wrong characteristic function.
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
        gap = np.max(
            np.abs(model.price(100.0, strike, maturity) - model.cos_price(100.0, strike, maturity))
        )
        assert gap < 1e-6, (maturity, gap)


def test_cos_converges_to_black_scholes_as_xi_vanishes():
    """v0 = theta and xi = 1e-4. The fourth cumulant is round-off.

    L is raised until the cosine price settles. The remaining gap against
    Black is the vol-of-vol, the same size for Lewis and for COS. The price
    gap, not the stencil, is what this test checks, and it is already inside
    5e-4 here.
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
                model,
                100.0,
                np.array([one]),
                float(maturity),
                n_paths=8_000,
                n_steps=64,
                seed=100 + node,
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


def test_euler_on_fang_oosterlee_parameters_after_the_measured_bias():
    """Smoke test. Feller ratio 0.38. 8,000 paths, 64 steps, one strike per seed.

    The budget is the mean Euler price minus Lewis at 1,000,000 paths,
    64 steps, seeds 0, 1, 2 and 3, on strikes 90, 100 and 110. Those four
    runs had sample standard deviations 0.00672, 0.00684 and 0.00372.
    An 8,000-path standard error is several times that budget, and the
    z-rule at n = 24 allows a mean z of 0.71, so this test does not have
    power to detect the budget. It fails if the scheme is grossly wrong.
    The z-rule is applied to (estimate - Lewis - budget). Nodes do not
    share paths.
    """
    model = Heston(**FO)
    assert model.feller_ratio() == pytest.approx(0.3794922785422908, rel=1e-12)
    strike = np.array([90.0, 100.0, 110.0])
    # Mean of four 1,000,000-path runs. See the docstring.
    budget = np.array([0.01593516, 0.02413260, 0.02054494])
    reference = model.price(100.0, strike, 1.0)
    estimates = []
    errors = []
    references = []
    node = 0
    for _seed_i in range(8):
        for j, one in enumerate(strike):
            result = euler_price(
                model,
                100.0,
                np.array([one]),
                1.0,
                n_paths=8_000,
                n_steps=64,
                seed=400 + node,
            )
            estimates.append(result.value[0])
            errors.append(result.stderr[0])
            references.append(reference[j] + budget[j])
            node += 1
    z = (np.array(estimates) - np.array(references)) / np.array(errors)
    assert z.size == 24
    failure = check_z(z)
    assert failure is None, failure


def test_cos_matches_lewis_at_high_kurtosis():
    """Puts, F = 100, k in {-1, -0.4, 0, 0.3}. Tolerance 1e-5.

    The first set has kurtosis about 103 at T = 1. The second has kurtosis
    about 281 at T = 0.25 and about 798 at T = 1. Both sit inside param_spec.
    """
    specs = (
        Heston(v0=0.04, kappa=0.5, theta=0.04, xi=1.5, rho=-0.9),
        Heston(v0=0.04, kappa=1.0, theta=0.04, xi=5.0, rho=-0.7),
    )
    strike = 100.0 * np.exp(np.array([-1.0, -0.4, 0.0, 0.3]))
    for model in specs:
        for maturity in (0.25, 1.0):
            lewis = model.price(100.0, strike, maturity, kind="put")
            cos = model.cos_price(100.0, strike, maturity, kind="put")
            gap = float(np.max(np.abs(lewis - cos)))
            assert gap < 1e-5, (model.xi, maturity, gap)


def test_short_maturity_fourth_cumulant_stays_finite():
    """The step is f / sqrt(c2), so a short maturity does not shrink it.

    On the Fang and Oosterlee parameters, T = 1 day, T = 0.02 and one week
    each return a positive fourth cumulant.
    """
    model = Heston(**FO)
    for maturity in (1.0 / 365.0, 0.02, 7.0 / 365.0):
        _c1, c2, c4 = model.cumulants(maturity)
        assert c2 > 0.0
        assert np.isfinite(c4) and c4 > 0.0, (maturity, c4)


def test_cos_matches_lewis_on_random_draws_including_high_kurtosis():
    """Seed 0 over a low-v0, high-xi box, plus three fixed heavy tails.

    Puts, F = 100. The fixed sets are the kurtosis-1049 corner, a one-day
    slice with xi = 5, and a T = 1 draw with kurtosis about 24,000. The
    random sample covers one day through one year. Tolerance 1e-4 is the
    cash gap. On this seed the largest random gap is 8.6e-6.
    """
    corner = Heston(v0=6.582e-4, kappa=2.1, theta=0.00999, xi=3.56, rho=-0.54)
    one_day = Heston(v0=1e-4, kappa=2.0, theta=0.01, xi=5.0, rho=-0.8)
    heavy = Heston(
        v0=1.0235984745304476e-4,
        kappa=0.8396638361927103,
        theta=7.681046417007965e-4,
        xi=3.5750252249003682,
        rho=0.2599077374058081,
    )
    _check_put_gap(corner, 0.25, 1e-4)
    _check_put_gap(one_day, 1.0 / 365.0, 1e-4)
    _check_put_gap(heavy, 1.0, 1e-4)
    assert _kurtosis(corner, 0.25) > 1000.0
    assert _kurtosis(one_day, 1.0 / 365.0) > 1000.0
    assert _kurtosis(heavy, 1.0) > 10_000.0

    rng = np.random.default_rng(0)
    n_draws = 12

    def _log_uniform(low, high):
        return np.exp(rng.uniform(np.log(low), np.log(high), n_draws))

    v0 = _log_uniform(1e-4, 0.5)
    kappa = _log_uniform(0.2, 8.0)
    theta = _log_uniform(1e-4, 0.5)
    xi = _log_uniform(0.05, 5.0)
    rho = rng.uniform(-0.95, 0.95, n_draws)
    maturities = (1.0 / 365.0, 7.0 / 365.0, 0.25, 1.0)
    saw_short = False
    saw_heavy = False
    for i in range(n_draws):
        model = Heston(
            v0=float(v0[i]),
            kappa=float(kappa[i]),
            theta=float(theta[i]),
            xi=float(xi[i]),
            rho=float(rho[i]),
        )
        for maturity in maturities:
            if maturity <= 7.0 / 365.0:
                saw_short = True
            kurtosis = _kurtosis(model, maturity)
            if kurtosis is not None and kurtosis > 500.0:
                saw_heavy = True
            _check_put_gap(model, maturity, 1e-4)
    assert saw_short
    assert saw_heavy


def _kurtosis(model: Heston, maturity: float) -> float | None:
    _c1, c2, c4 = model.cumulants(maturity)
    if not (np.isfinite(c4) and c2 > 0.0 and c4 > 0.0):
        return None
    return float(c4 / c2**2)


def _check_put_gap(model: Heston, maturity: float, tolerance: float) -> None:
    strike = 100.0 * np.exp(np.array([-0.5, -0.086, 0.0, 0.3]))
    lewis = model.price(100.0, strike, maturity, kind="put")
    cos = model.cos_price(100.0, strike, maturity, kind="put")
    gap = float(np.max(np.abs(lewis - cos)))
    assert gap < tolerance, (model.v0, model.xi, maturity, gap)


def test_fitter_refuses_american_exercise():
    truth = Heston(v0=0.03, kappa=1.2, theta=0.05, xi=0.4, rho=-0.6)
    data = _quotes_from(truth, half_spread=None)
    data.quotes.loc[data.quotes.index[:2], "exercise"] = "american"
    with pytest.raises(ValueError, match="2 quotes"):
        HestonFitter(n_starts=1).fit(data)
    fitted = HestonFitter(n_starts=1, seed=1, allow_american=True).fit(data)
    assert int(fitted.slice_report.iloc[0]["n_american"]) == 2


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
