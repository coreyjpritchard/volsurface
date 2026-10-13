"""Five greek engines. Tolerances are the ones in ``docs/DESIGN.md`` section 8.

The z-rule uses independent seeds, one per strike. ``n`` is the strike count.
A pathwise digital or a pathwise gamma is refused rather than returned as zero.
"""

import numpy as np
import pytest
from scipy.stats import norm

from volsurface import bs
from volsurface.core.types import Contract, Digital, Forward, Param, Spot, Vanilla
from volsurface.greeks import Analytic, Autodiff, FiniteDifference, LikelihoodRatio, Pathwise
from volsurface.models.bs import BlackScholes
from volsurface.models.gbm import GBM


def _z_close(z, n: int) -> None:
    """max |z| < 4.5 and |mean z| < 0.5.

    ``n`` is the number of independent seeds. The 0.90 to 0.99 band on the
    fraction inside 2 is for a few hundred nodes, which is the price grid in
    the Monte Carlo tests. On a few dozen greeks that band fails whenever
    every node happens to land inside 2.
    """
    z = np.asarray(z, dtype=float)
    assert z.shape == (n,)
    assert np.max(np.abs(z)) < 4.5, float(np.max(np.abs(z)))
    assert abs(float(z.mean())) < 0.5, float(z.mean())


def test_analytic_matches_finite_differences_including_digitals():
    """Halving the bump is the check. The digital formulas are differentiated in closed form."""
    model = BlackScholes(sigma=0.25)
    engine = Analytic(F=100.0, df=0.97)
    contracts = [
        Contract(Vanilla(90.0, "call"), 0.8),
        Contract(Vanilla(110.0, "put"), 0.8),
        Contract(Digital(100.0, "call"), 0.8),
        Contract(Digital(100.0, "put"), 0.8),
    ]
    factors = [Spot(), Forward(), Param("sigma")]
    first = engine.compute(model, contracts, factors, order=1)
    second = engine.compute(model, contracts, factors, order=2)
    # Spot and forward are the same bump when df is held fixed.
    np.testing.assert_allclose(first.value[:, 0], first.value[:, 1], atol=1e-12)
    np.testing.assert_allclose(second.value[:, 0], second.value[:, 1], atol=1e-12)
    fd = FiniteDifference(F=100.0, df=0.97, bump=1e-4)
    # Digitals are not a model.price. Check the analytic derivative against a
    # central step on the cash-or-nothing price df N(d2).
    # h = 1e-3 stays above the round-off that a second difference picks up on a
    # digital gamma of size 1e-4.
    bumped = _digital_bump(100.0, 100.0, 0.8, 0.25, 0.97, h=1e-3)
    np.testing.assert_allclose(first.value[2, 1], bumped["delta"], rtol=1e-8)
    np.testing.assert_allclose(second.value[2, 1], bumped["gamma"], rtol=1e-5)
    np.testing.assert_allclose(first.value[2, 2], bumped["vega"], rtol=1e-7)
    np.testing.assert_allclose(second.value[2, 2], bumped["volga"], rtol=1e-5)
    vanilla = FiniteDifference(F=100.0, df=0.97, bump=1e-4).compute(
        model, contracts[:2], [Forward(), Param("sigma")], order=1,
    )
    np.testing.assert_allclose(vanilla.value[:, 0], first.value[:2, 1], rtol=1e-6)
    np.testing.assert_allclose(vanilla.value[:, 1], first.value[:2, 2], rtol=1e-6)
    second_fd = FiniteDifference(F=100.0, df=0.97, bump=1e-4).compute(
        model, contracts[:2], [Forward(), Param("sigma")], order=2,
    )
    np.testing.assert_allclose(second_fd.value[:, 0], second.value[:2, 1], rtol=5e-4)
    np.testing.assert_allclose(second_fd.value[:, 1], second.value[:2, 2], rtol=1e-4)
    assert not fd.supports(model, contracts[2], 1)
    assert engine.supports(model, contracts[2], 2)
    assert first.method == "analytic"
    assert first.stderr is None


def _digital_bump(F, K, T, sigma, df, h):
    """Central steps of df N(d2), the cash-or-nothing call price."""

    def price(F_, sigma_):
        s = sigma_ * np.sqrt(T)
        d2 = np.log(F_ / K) / s - 0.5 * s
        return df * norm.cdf(d2)

    return {
        "delta": (price(F + h, sigma) - price(F - h, sigma)) / (2 * h),
        "gamma": (price(F + h, sigma) - 2 * price(F, sigma) + price(F - h, sigma)) / h**2,
        "vega": (price(F, sigma + h) - price(F, sigma - h)) / (2 * h),
        "volga": (price(F, sigma + h) - 2 * price(F, sigma) + price(F, sigma - h)) / h**2,
    }


def test_finite_difference_order_is_two_above_the_roundoff_floor():
    """Bumps from 1e-2 down to 1.25e-3. Slope of log error on log bump is 2 within 0.2."""
    model = BlackScholes(sigma=0.2)
    truth = Analytic(F=100.0, df=1.0).compute(
        model, [Contract(Vanilla(100.0), 1.0)], [Forward()], order=1,
    ).value[0, 0]
    bumps = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3])
    error = np.empty(bumps.size)
    for i, h in enumerate(bumps):
        got = FiniteDifference(F=100.0, df=1.0, bump=float(h)).compute(
            model, [Contract(Vanilla(100.0), 1.0)], [Forward()], order=1,
        ).value[0, 0]
        error[i] = abs(got - truth)
    # Each halving multiplies the error by about 1/4.
    ratios = error[:-1] / error[1:]
    np.testing.assert_allclose(ratios, 4.0, atol=0.5)
    slope = float(np.polyfit(np.log(bumps), np.log(error), 1)[0])
    assert slope == pytest.approx(2.0, abs=0.2), slope


def test_common_random_numbers_cut_the_finite_difference_variance():
    """40 independent seeds, 2,000 paths, bump 1 on a forward of 100.

    The variance without a shared seed, divided by the variance with one, is
    the factor. It has to exceed 5. The runs are not the same stream.
    """
    contract = [Contract(Vanilla(100.0), 1.0)]
    model = GBM(sigma=0.2, n_paths=2_000, n_steps=1, scheme="exact")

    def once(seed, crn):
        engine = FiniteDifference(F=100.0, df=1.0, bump=1.0, crn=crn)
        return engine.compute(model, contract, [Forward()], n_paths=2_000, seed=seed).value[0, 0]

    crn = np.array([once(seed, True) for seed in range(40)])
    indep = np.array([once(1_000 + seed, False) for seed in range(40)])
    factor = float(np.var(indep, ddof=1) / np.var(crn, ddof=1))
    assert factor > 5.0, factor


def test_pathwise_matches_black_and_refuses_a_digital_and_a_gamma():
    """40 strikes, independent seeds, 8,000 paths. n = 40. max |z| < 4.5, |mean z| < 0.5."""
    F, df, sigma, T = 100.0, 0.98, 0.22, 0.75
    k = np.linspace(-0.25, 0.25, 40)
    model_bs = BlackScholes(sigma=sigma)
    analytic = Analytic(F=F, df=df)
    z_delta = np.empty(k.size)
    z_vega = np.empty(k.size)
    for i, log_k in enumerate(k):
        contract = [Contract(Vanilla(F * np.exp(log_k), "call"), T)]
        truth = analytic.compute(model_bs, contract, [Forward(), Param("sigma")]).value[0]
        mc = GBM(sigma=sigma, n_paths=8_000, n_steps=4, seed=0, scheme="exact")
        got = Pathwise(F=F, df=df).compute(mc, contract, [Forward(), Param("sigma")], seed=200 + i)
        z_delta[i] = (got.value[0, 0] - truth[0]) / got.stderr[0, 0]
        z_vega[i] = (got.value[0, 1] - truth[1]) / got.stderr[0, 1]
    _z_close(z_delta, 40)
    _z_close(z_vega, 40)
    engine = Pathwise(F=F, df=df)
    digital = Contract(Digital(F, "call"), T)
    vanilla = Contract(Vanilla(F), T)
    assert not engine.supports(mc, digital, 1)
    assert not engine.supports(mc, vanilla, 2)
    with pytest.raises(ValueError):
        engine.compute(mc, [digital], [Forward()])
    with pytest.raises(ValueError):
        engine.compute(mc, [vanilla], [Forward()], order=2)


def test_likelihood_ratio_matches_black_and_is_noisier_when_expiry_is_short():
    """Delta, vega and gamma on 20 strikes. A short digital's delta error exceeds the long one.

    n = 20, 20,000 paths, independent seeds. Volga is estimated too, on one
    strike: its standard error is reported and the z-score stays inside 4.5.
    """
    F, df, sigma = 80.0, 1.0, 0.3
    T = 1.0
    k = np.linspace(-0.2, 0.2, 20)
    analytic = Analytic(F=F, df=df)
    bs_model = BlackScholes(sigma=sigma)
    z = {name: np.empty(k.size) for name in ("delta", "vega", "gamma")}
    for i, log_k in enumerate(k):
        contract = [Contract(Vanilla(F * np.exp(log_k)), T)]
        truth_1 = analytic.compute(
            bs_model, contract, [Forward(), Param("sigma")], order=1,
        ).value[0]
        truth_g = analytic.compute(bs_model, contract, [Forward()], order=2).value[0, 0]
        mc = GBM(sigma=sigma, scheme="exact", n_steps=1)
        got_1 = LikelihoodRatio(F=F, df=df).compute(
            mc, contract, [Forward(), Param("sigma")], order=1, n_paths=20_000, seed=300 + i,
        )
        got_g = LikelihoodRatio(F=F, df=df).compute(
            mc, contract, [Forward()], order=2, n_paths=20_000, seed=300 + i,
        )
        z["delta"][i] = (got_1.value[0, 0] - truth_1[0]) / got_1.stderr[0, 0]
        z["vega"][i] = (got_1.value[0, 1] - truth_1[1]) / got_1.stderr[0, 1]
        z["gamma"][i] = (got_g.value[0, 0] - truth_g) / got_g.stderr[0, 0]
    for row in z.values():
        _z_close(row, 20)

    # A digital payoff does not shrink with expiry, and the score divides by
    # sigma^2 T, so the short-dated standard error is the larger one.
    short = _digital_delta_se(F, sigma, df, T=0.05, seed=7)
    long = _digital_delta_se(F, sigma, df, T=1.0, seed=8)
    assert short > long, (short, long)

    contract = [Contract(Vanilla(F), 1.0)]
    truth_v = analytic.compute(bs_model, contract, [Param("sigma")], order=2).value[0, 0]
    got_v = LikelihoodRatio(F=F, df=df).compute(
        GBM(sigma=sigma, scheme="exact"), contract, [Param("sigma")],
        order=2, n_paths=100_000, seed=9,
    )
    z_volga = (got_v.value[0, 0] - truth_v) / got_v.stderr[0, 0]
    assert abs(float(z_volga)) < 4.5, float(z_volga)


def _digital_delta_se(F, sigma, df, T, seed) -> float:
    contract = [Contract(Digital(F, "call"), T)]
    got = LikelihoodRatio(F=F, df=df).compute(
        GBM(sigma=sigma, scheme="exact"), contract, [Forward()],
        n_paths=30_000, seed=seed,
    )
    return float(got.stderr[0, 0])


def test_likelihood_ratio_prices_a_digital_that_pathwise_cannot():
    """One cash-or-nothing call. n = 1 node, 200,000 paths, |z| < 4.5 against df n(d2)/(F s)."""
    F, K, T, sigma, df = 100.0, 105.0, 0.5, 0.2, 0.99
    analytic = Analytic(F=F, df=df).compute(
        BlackScholes(sigma=sigma), [Contract(Digital(K, "call"), T)], [Forward()],
    )
    got = LikelihoodRatio(F=F, df=df).compute(
        GBM(sigma=sigma, scheme="exact"), [Contract(Digital(K, "call"), T)], [Forward()],
        n_paths=200_000, seed=11,
    )
    z = (got.value[0, 0] - analytic.value[0, 0]) / got.stderr[0, 0]
    assert abs(float(z)) < 4.5, float(z)


def test_autodiff_matches_pathwise_to_1e_10_in_float64():
    """Same seed, one step and four steps. The values, not just the expectations, agree."""
    F, df, sigma = 50.0, 0.96, 0.18
    contracts = [Contract(Vanilla(K), 0.4) for K in (45.0, 50.0, 55.0)]
    factors = [Forward(), Param("sigma")]
    engine = Autodiff(F=F, df=df)
    for n_steps in (1, 4):
        model = GBM(sigma=sigma, n_steps=n_steps, scheme="exact", antithetic=False)
        path = Pathwise(F=F, df=df).compute(model, contracts, factors, n_paths=2_000, seed=12)
        auto = engine.compute(model, contracts, factors, n_paths=2_000, seed=12)
        np.testing.assert_allclose(auto.value, path.value, atol=1e-10)
        assert auto.method == "autodiff"
    assert not engine.supports(model, Contract(Digital(50.0), 0.4), 1)
    assert not engine.supports(model, contracts[0], 2)


def test_put_pathwise_delta_is_the_negative_indicator():
    """A put, one strike, 30,000 paths. |z| < 4.5 against the Black put delta."""
    F, K, T, sigma, df = 100.0, 100.0, 1.0, 0.2, 1.0
    truth = float(bs.delta(F, K, T, sigma, df, "put"))
    got = Pathwise(F=F, df=df).compute(
        GBM(sigma=sigma, scheme="exact"), [Contract(Vanilla(K, "put"), T)], [Spot()],
        n_paths=30_000, seed=13,
    )
    z = (got.value[0, 0] - truth) / got.stderr[0, 0]
    assert abs(float(z)) < 4.5, float(z)
