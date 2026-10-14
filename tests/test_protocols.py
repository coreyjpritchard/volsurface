"""Protocol tests for DESIGN.md sections 3 and 4.

The existing Black-Scholes, Heston, and chain tests stay in their own files.
These check the new surface of the same objects: state, parameters, the
characteristic function, and a MarketData built from the committed snapshot.
"""

import numpy as np
import pytest

from volsurface import models
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.protocols import (
    Model,
    Parametric,
    SupportsCharFunc,
    capabilities,
)
from volsurface.mc import price_with_error
from volsurface.plugins import create


def test_black_scholes_protocol_and_empty_state():
    m = models.BlackScholes(0.2)
    assert isinstance(m, Model)
    assert isinstance(m, Parametric)
    assert isinstance(m, SupportsCharFunc)
    assert capabilities(m) == {"price", "char_func", "parametric"}
    K = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(m.price(100.0, K, 1.0, state=None), m.price(100.0, K, 1.0))
    np.testing.assert_allclose(m.price(100.0, K, 1.0, state={}), m.price(100.0, K, 1.0))
    # phi(0) = 1, and the first cumulant is the mean of log(F_T / F).
    assert m.char_func(0.0, 1.0) == pytest.approx(1.0)
    c1, c2, c4 = m.cumulants(1.0)
    assert c1 == pytest.approx(-0.5 * 0.04)
    assert c2 == pytest.approx(0.04)
    assert c4 == pytest.approx(0.0)


def test_parameter_round_trip():
    original = models.Heston(v0=0.02, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
    rebuilt = models.Heston.from_unconstrained(original.to_unconstrained())
    for spec in models.Heston.param_spec():
        assert getattr(rebuilt, spec.name) == pytest.approx(getattr(original, spec.name), rel=1e-10)


def test_heston_state_replaces_v0_and_matches_the_module():
    m = models.Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.5)
    K = np.array([90.0, 100.0, 110.0])
    from_state = m.price(100.0, K, 0.5, state={"v": np.array(0.01)})
    other = models.Heston(0.01, 1.5, 0.04, 0.3, -0.5)
    np.testing.assert_allclose(from_state, other.price(100.0, K, 0.5))
    u = np.array([0.5, 1.0 + 0.25j])
    np.testing.assert_allclose(m.char_func(u, 1.0), m.char_func(u, 1.0, state=m.initial_state()))


def test_heston_cumulants_match_a_second_differentiation():
    """c2 and c4 agree with a second stencil and with central differences.

    The integrated-variance line is the expression ``cumulants`` already
    returns, rearranged. It is an algebraic consistency check on one parameter
    set (the Fang and Oosterlee values). The independent check is computed
    here from ``char_func``: a degree-8 fit at ``h = 0.01`` for ``c1``, the
    same fit at ``h = 0.02`` for ``c2`` and ``c4``, and a central difference
    at ``h = 0.02``. Tolerances are absolute and wider than the gap observed
    on this parameter set (about ``1e-12`` for the second-stencil ``c2``,
    about ``5e-8`` for ``c4``, and a few ``1e-7`` for the central difference).
    """
    m = models.Heston(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
    T = 1.0
    integ = m.theta * T + (m.v0 - m.theta) * (1.0 - np.exp(-m.kappa * T)) / m.kappa
    c1, c2, c4 = m.cumulants(T)
    assert c1 == pytest.approx(-0.5 * integ, rel=0, abs=1e-12)

    def cgf(t):
        return np.log(m.char_func(-1j * np.asarray(t, dtype=float), T)).real

    h = 0.01
    fitted = np.poly1d(np.polyfit(np.arange(-4, 5) * h, cgf(np.arange(-4, 5) * h), 8))
    assert float(fitted.deriv(1)(0.0)) == pytest.approx(c1, abs=1e-9)

    h2 = 0.02
    grid = np.arange(-4, 5) * h2
    fitted2 = np.poly1d(np.polyfit(grid, cgf(grid), 8))
    assert float(fitted2.deriv(2)(0.0)) == pytest.approx(c2, abs=1e-10)
    assert float(fitted2.deriv(4)(0.0)) == pytest.approx(c4, abs=1e-6)

    c2_fd = (cgf(h2) - 2.0 * cgf(0.0) + cgf(-h2)) / h2**2
    c4_fd = (cgf(-2 * h2) - 4 * cgf(-h2) + 6 * cgf(0.0) - 4 * cgf(h2) + cgf(2 * h2)) / h2**4
    assert c2_fd == pytest.approx(c2, abs=1e-6)
    assert c4_fd == pytest.approx(c4, abs=1e-5)


def test_price_with_error_is_deterministic_for_black_scholes():
    m = models.BlackScholes(0.2)
    result = price_with_error(m, 100.0, np.array([100.0]), 1.0, df=0.99)
    assert result.stderr is None
    assert result.n_evals == 1
    assert result.value[0] == pytest.approx(m.price(100.0, [100.0], 1.0, 0.99)[0])


def test_registry_and_entry_point_build_black_scholes():
    built = create("black_scholes", sigma=0.25)
    assert isinstance(built, models.BlackScholes)
    assert built.sigma == 0.25


def test_market_data_from_committed_snapshot():
    from volsurface import chains

    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    assert data.day_count == "ACT/365"
    assert set(data.quotes["exercise"]) == {"european"}
    assert set(data.quotes["settlement"]) == {"cash"}
    assert data.quotes["bid"].notna().all()
    assert (data.quotes["bid"] > 0).all()
    row = data.quotes.iloc[len(data.quotes) // 2]
    assert data.F(row["T"]) == pytest.approx(row["F"], rel=1e-10)
    assert data.df(np.array([row["T"]]))[0] == pytest.approx(row["df"], rel=1e-10)
    assert np.isfinite(data.F_err(row["T"]))
    assert data.F_err(row["T"]) > 0.0
    # The first expiry is after 0, so the discount curve is anchored at df(0) = 1.
    assert float(data.quotes["T"].min()) > 0.0
    assert data.df(0.0) == pytest.approx(1.0)
