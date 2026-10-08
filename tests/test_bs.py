import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def test_put_call_parity(impl):
    """C - P = df (F - K) for every strike, maturity and volatility."""
    bs = impl("bs")
    F, df = 100.0, 0.97
    K = np.linspace(50.0, 200.0, 31)[:, None]
    T = np.array([0.01, 0.25, 1.0, 5.0])[None, :]
    for sigma in (0.05, 0.2, 0.8):
        diff = bs.price(F, K, T, sigma, df, "call") - bs.price(F, K, T, sigma, df, "put")
        np.testing.assert_allclose(diff, df * (F - K) * np.ones_like(T), atol=1e-10)


def test_greeks_match_finite_differences(impl):
    bs = impl("bs")
    F, K, T, s, df, h = 100.0, np.array([80.0, 100.0, 125.0]), 0.5, 0.25, 0.99, 1e-4
    for kind in ("call", "put"):
        up, down = bs.price(F + h, K, T, s, df, kind), bs.price(F - h, K, T, s, df, kind)
        fd_delta = (up - down) / (2 * h)
        np.testing.assert_allclose(bs.delta(F, K, T, s, df, kind), fd_delta, rtol=1e-6)
    g = 1e-2  # second differences need a wider step to stay clear of rounding error
    fd_gamma = (
        bs.price(F + g, K, T, s, df) - 2 * bs.price(F, K, T, s, df) + bs.price(F - g, K, T, s, df)
    ) / g**2
    np.testing.assert_allclose(bs.gamma(F, K, T, s, df), fd_gamma, rtol=1e-5)
    fd_vega = (bs.price(F, K, T, s + h, df) - bs.price(F, K, T, s - h, df)) / (2 * h)
    np.testing.assert_allclose(bs.vega(F, K, T, s, df), fd_vega, rtol=1e-6)


@settings(max_examples=300, deadline=None)
@given(
    sigma=st.floats(0.02, 2.0),
    k=st.floats(-1.0, 1.0),
    T=st.floats(0.003, 5.0),
    kind=st.sampled_from(["call", "put"]),
)
def test_implied_vol_round_trip(impl, sigma, k, T, kind):
    """Price then invert recovers sigma wherever the price carries vol information."""
    bs = impl("bs")
    F, df = 100.0, 0.95
    K = F * np.exp(k)
    p = bs.price(F, K, T, sigma, df, kind)
    # Skip prices where the time value is below double precision resolution.
    otm = bs.price(F, K, T, sigma, df, "call" if K >= F else "put")
    if otm < 1e-10 * F:
        return
    assert bs.implied_vol(p, F, K, T, df, kind) == pytest.approx(sigma, rel=1e-6)


def test_implied_vol_vectorised(impl):
    bs = impl("bs")
    F, T = 100.0, np.array([[0.1], [1.0]])
    K = np.array([[70.0, 100.0, 140.0]])
    sigma = np.array([[0.4, 0.2, 0.3]])
    kind = np.array([["put", "call", "call"]])
    p = bs.price(F, K, T, sigma, 1.0, kind)
    np.testing.assert_allclose(bs.implied_vol(p, F, K, T, 1.0, kind), sigma * np.ones((2, 1)),
                               rtol=1e-8)


def test_implied_vol_edge_cases_are_nan(impl):
    """At or below intrinsic, at or above the upper bound, or T <= 0: no implied vol."""
    bs = impl("bs")
    F, K, T = 100.0, np.array([80.0, 80.0, 120.0, 100.0, 100.0]), 1.0
    prices = np.array([20.0, 19.0, 100.0, -1.0, np.nan])
    kinds = np.array(["call", "call", "call", "put", "put"])
    assert np.all(np.isnan(bs.implied_vol(prices, F, K, T, 1.0, kinds)))
    assert np.isnan(bs.implied_vol(5.0, F, 100.0, 0.0, 1.0, "call"))
