"""Second-order Black Greeks and the dashboard's volatility environments.

The Greek tests use the ``impl`` fixture like ``test_bs.py`` and skip when an exercise
module has not written ``vanna``, ``volga`` or ``theta`` yet.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from volsurface import bs as ref_bs
from volsurface import environments as env_lib
from volsurface.models.sabr import hagan_implied_vol


def _need(module, *names):
    for name in names:
        if not hasattr(module, name):
            pytest.skip(f"{module.__name__}.{name} not written yet")


# Black Greeks ---------------------------------------------------------------------------


def test_second_order_greeks_closed_form_at_the_money(impl):
    """F = K = 100, T = 1, sigma = 0.2: d1 = 0.1, d2 = -0.1."""
    bs = impl("bs")
    _need(bs, "vanna", "volga", "theta")
    n = norm.pdf(0.1)
    F = K = 100.0
    np.testing.assert_allclose(bs.vanna(F, K, 1.0, 0.2), -n * (-0.1) / 0.2, rtol=1e-12)
    np.testing.assert_allclose(bs.volga(F, K, 1.0, 0.2), F * n * 0.1 * (-0.1) / 0.2, rtol=1e-12)
    np.testing.assert_allclose(bs.theta(F, K, 1.0, 0.2), -F * n * 0.2 / 2.0, rtol=1e-12)
    np.testing.assert_allclose(bs.vega(F, K, 1.0, 0.2), F * n, rtol=1e-12)
    np.testing.assert_allclose(bs.gamma(F, K, 1.0, 0.2), n / (F * 0.2), rtol=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_second_order_greeks_match_finite_differences(impl, kind):
    bs = impl("bs")
    _need(bs, "vanna", "volga", "theta")
    F, df = 100.0, 0.98
    K = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
    T, s = 0.75, 0.3
    hF, hs, hT = 1e-2, 1e-3, 1e-5

    def p(f=F, sig=s, t=T):
        return bs.price(f, K, t, sig, df, kind)

    fd_vanna = (p(F + hF, s + hs) - p(F + hF, s - hs) - p(F - hF, s + hs) + p(F - hF, s - hs)) / (
        4 * hF * hs
    )
    fd_volga = (p(sig=s + hs) - 2 * p() + p(sig=s - hs)) / hs**2
    fd_theta = -(p(t=T + hT) - p(t=T - hT)) / (2 * hT)
    np.testing.assert_allclose(bs.vanna(F, K, T, s, df), fd_vanna, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(bs.volga(F, K, T, s, df), fd_volga, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(bs.theta(F, K, T, s, df), fd_theta, rtol=1e-6)


def test_black_scholes_equation_links_theta_vega_and_gamma(impl):
    """Zero rates: theta = -1/2 sigma^2 F^2 gamma and vega = sigma T F^2 gamma."""
    bs = impl("bs")
    _need(bs, "theta")
    F = 100.0
    K = np.linspace(60.0, 160.0, 21)[:, None]
    T = np.array([0.02, 0.25, 1.0, 3.0])[None, :]
    for s in (0.1, 0.4):
        gamma = bs.gamma(F, K, T, s)
        np.testing.assert_allclose(bs.theta(F, K, T, s), -0.5 * s**2 * F**2 * gamma, rtol=1e-12)
        np.testing.assert_allclose(bs.vega(F, K, T, s), s * T * F**2 * gamma, rtol=1e-12)


def test_vanna_is_the_vol_derivative_of_delta(impl):
    bs = impl("bs")
    _need(bs, "vanna", "volga")
    F, K, T, s, h = 100.0, np.array([80.0, 100.0, 120.0]), 0.5, 0.25, 1e-5
    fd = (bs.delta(F, K, T, s + h) - bs.delta(F, K, T, s - h)) / (2 * h)
    np.testing.assert_allclose(bs.vanna(F, K, T, s), fd, rtol=1e-6)
    fd = (bs.vega(F, K, T, s + h) - bs.vega(F, K, T, s - h)) / (2 * h)
    np.testing.assert_allclose(bs.volga(F, K, T, s), fd, rtol=1e-6, atol=1e-9)


# Environments ---------------------------------------------------------------------------


def _iv_at(env, T, k):
    """Implied vol of ``env`` at one expiry and arbitrary log-moneyness."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    z = k / (env_lib.SIGMA_REF * np.sqrt(T))
    return env.iv(env_lib.Grid(np.array([T]), z, env_lib.SIGMA_REF))[0]


def test_greeks_grid_units_and_flat_vol_values():
    grid = env_lib.default_grid()
    env = env_lib.flat_environment(0.2)
    iv = env.iv(grid)
    np.testing.assert_allclose(iv, 0.2, atol=1e-10)
    g = env_lib.greeks_grid(grid, iv)
    F, K, T = env_lib.NOTIONAL_F, grid.strike, grid.T[:, None]
    np.testing.assert_allclose(g["vega"], ref_bs.vega(F, K, T, 0.2) / 100.0)
    np.testing.assert_allclose(g["volga"], ref_bs.volga(F, K, T, 0.2) / 1e4)
    np.testing.assert_allclose(g["theta"] * 365.0, -0.5 * 0.04 * F**2 * g["gamma"], rtol=1e-12)
    # Flat vol: the smile does not move, so both deltas agree.
    np.testing.assert_allclose(env_lib.smile_delta(env, grid, iv), g["delta"], atol=1e-8)


def test_heston_smile_delta_matches_a_price_bump():
    """Heston is homogeneous in (F, K), so the smile moves with K / F."""
    env = env_lib.heston_environment(v0=0.03, kappa=2.0, theta=0.04, xi=0.8, rho=-0.7)
    grid = env_lib.default_grid(T=np.array([0.25, 1.0]), z=np.linspace(-1.5, 1.5, 7))
    iv = env.iv(grid)
    got = env_lib.smile_delta(env, grid, iv)
    F, h = env_lib.NOTIONAL_F, 0.05
    for i, T in enumerate(grid.T):
        K = grid.strike[i]

        def call(f, T=T, K=K):
            return ref_bs.price(f, K, T, _iv_at(env, T, np.log(K / f)))

        fd = (call(F + h) - call(F - h)) / (2 * h)
        np.testing.assert_allclose(got[i], fd, atol=2e-4)
    # Downward skew: as F rises a fixed strike moves to lower k and higher vol, so the
    # smile delta sits above the sticky-strike delta (minimum-variance deltas go the
    # other way; see the dashboard text).
    sticky = env_lib.greeks_grid(grid, iv)["delta"]
    assert np.all(got[:, 3] > sticky[:, 3])


def test_sabr_smile_delta_matches_a_hagan_price_bump():
    a, b, r, n = 0.25, 0.6, -0.4, 0.7
    env = env_lib.sabr_environment(a, b, r, n)
    grid = env_lib.default_grid(T=np.array([0.5]), z=np.linspace(-1.5, 1.5, 7))
    iv = env.iv(grid)
    got = env_lib.smile_delta(env, grid, iv)[0]
    F, h = env_lib.NOTIONAL_F, 0.05
    K = grid.strike[0]
    T = float(grid.T[0])

    def call(f):
        # Hagan with the forward normalised to 1: forward f / F, strike K / F.
        sig = hagan_implied_vol(f / F, K / F, T, a, b, r, n)
        return ref_bs.price(f, K, T, sig)

    fd = (call(F + h) - call(F - h)) / (2 * h)
    np.testing.assert_allclose(got, fd, atol=2e-5)


def test_flat_and_heston_surfaces_are_free_of_static_arbitrage():
    grid = env_lib.default_grid()
    for env in (
        env_lib.flat_environment(0.25),
        env_lib.heston_environment(0.02, 2.0, 0.04, 0.6, -0.7),
    ):
        arb = env_lib.arbitrage_map(env, grid)
        assert arb.ok, (env.name, arb.counts)
        assert arb.n_checked == grid.k.size


def test_svi_presets_flag_the_arbitrage_they_are_built_for():
    grid = env_lib.default_grid()
    base = dict(atm_vol=0.2, term_slope=0.05, b=0.06, rho=-0.6, m=0.0, sigma=0.15, skew_decay=0.4)
    assert env_lib.arbitrage_map(env_lib.svi_environment(grid.T, **base), grid).ok

    calendar = dict(base, atm_vol=0.12, term_slope=-0.6, b=0.05, skew_decay=0.3)
    counts = env_lib.arbitrage_map(env_lib.svi_environment(grid.T, **calendar), grid).counts
    assert counts["calendar"] > 0 and counts["butterfly"] == 0

    butterfly = dict(base, term_slope=0.0, b=0.4, rho=-0.95, sigma=0.02, skew_decay=0.3)
    counts = env_lib.arbitrage_map(env_lib.svi_environment(grid.T, **butterfly), grid).counts
    assert counts["butterfly"] > 0 and counts["calendar"] == 0


def test_svi_short_end_is_floored():
    """Below SVI_T_FLOOR the slice parameters stop moving, so the short end stays finite."""
    grid = env_lib.default_grid()
    env = env_lib.svi_environment(
        grid.T, atm_vol=0.12, term_slope=-1.0, b=0.05, rho=-0.6, m=0.0, sigma=0.15, skew_decay=1.5
    )
    iv = env.iv(grid)
    assert np.all(np.isfinite(iv)) and iv.max() < 1.0
    atm = _iv_at(env, 7 / 365, 0.0)[0]
    np.testing.assert_allclose(atm, 0.12 * env_lib.SVI_T_FLOOR**-1.0, rtol=1e-10)


def test_atm_skew_of_heston_has_the_sign_of_rho():
    T = np.array([0.1, 0.5, 1.0])
    down = env_lib.atm_skew(env_lib.heston_environment(0.04, 2.0, 0.04, 0.6, -0.7), T)
    up = env_lib.atm_skew(env_lib.heston_environment(0.04, 2.0, 0.04, 0.6, 0.7), T)
    assert np.all(down < 0) and np.all(up > 0)
    np.testing.assert_allclose(down, -up, rtol=0.05)


@pytest.mark.skipif(not env_lib.PRESET_PATH.exists(), reason="dashboard presets not built")
def test_spx_presets_rebuild_and_sanos_is_arbitrage_free():
    presets = env_lib.load_spx_presets()
    sanos, svi = presets["spx_sanos"], presets["spx_svi"]
    grid = env_lib.spx_grid(sanos)
    iv = sanos.iv(grid)
    assert np.all(np.isfinite(iv)) and 0.05 < iv.min() and iv.max() < 0.6
    assert env_lib.arbitrage_map(sanos, grid).ok
    # Raw SVI fitted slice by slice is not arbitrage-free on the same grid.
    assert not env_lib.arbitrage_map(svi, env_lib.spx_grid(svi)).ok
