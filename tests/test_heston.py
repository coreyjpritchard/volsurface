import numpy as np
import pytest

# Fang and Oosterlee (2008), "A novel pricing method for European options based on
# Fourier-cosine series expansions", SIAM J. Sci. Comput. 31(2), Section 5.2: Heston with
# S0 = K = 100, r = q = 0, kappa = 1.5768, xi = 0.5751, theta = 0.0398, v0 = 0.0175,
# rho = -0.5711. Reference call prices 5.785155450 (T = 1) and 22.318945791 (T = 10).
FO_PARAMS = dict(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)


@pytest.mark.parametrize("T, reference", [(1.0, 5.785155450), (10.0, 22.318945791)])
def test_fang_oosterlee_reference(impl, T, reference):
    heston = impl("heston")
    price = heston.price(100.0, np.array([100.0]), T, **FO_PARAMS)[0]
    assert price == pytest.approx(reference, abs=1e-6)


@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.5])
def test_black_scholes_limit(impl, rho):
    """xi -> 0 with v0 = theta: variance stays at theta, so Heston = BS(sqrt(theta))."""
    heston, bs = impl("heston"), impl("bs")
    theta, F, df = 0.04, 100.0, 0.98
    K = np.array([60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 150.0])
    for T in (0.02, 0.5, 2.0):
        for kind in ("call", "put"):
            h = heston.price(F, K, T, theta, 1.5, theta, 1e-5, rho, df, kind)
            b = bs.price(F, K, T, np.sqrt(theta), df, kind)
            np.testing.assert_allclose(h, b, atol=1e-4)


def test_negative_rho_gives_downward_skew(impl):
    heston, bs = impl("heston"), impl("bs")
    F, T = 100.0, 0.5
    K = np.array([80.0, 100.0, 120.0])
    kind = np.array(["put", "call", "call"])
    for rho, sign in ((-0.7, -1), (0.7, 1)):
        p = heston.price(F, K, T, 0.04, 2.0, 0.04, 0.6, rho, 1.0, kind)
        iv = bs.implied_vol(p, F, K, T, 1.0, kind)
        assert np.sign(iv[2] - iv[0]) == sign


@pytest.mark.slow
def test_monte_carlo_agrees(impl):
    """Euler full truncation, 200k paths in 4 batches of 50k, 200 steps; within 4 standard
    errors (about 0.07 at this sample size) of the Fourier price for three strikes."""
    heston = impl("heston")
    F, T = 100.0, 1.0
    K = np.array([80.0, 100.0, 120.0])
    payoffs = []
    for seed in range(4):
        paths, _ = heston.simulate(F, T, **FO_PARAMS, n=200, m=50_000, seed=seed)
        payoffs.append(np.maximum(paths[:, -1:] - K[None, :], 0.0))
    payoffs = np.concatenate(payoffs)
    mc = payoffs.mean(axis=0)
    se = payoffs.std(axis=0, ddof=1) / np.sqrt(len(payoffs))
    exact = heston.price(F, K, T, **FO_PARAMS)
    assert np.all(np.abs(mc - exact) < 4 * se), (mc, exact, se)
