"""Five greek engines. Tolerances are the ones in ``docs/DESIGN.md`` section 8.

Statistical checks go through :func:`volsurface.mc.zrule.check_z`, so the
maximum, the mean, and the fraction inside 2 all move with the node count.
A pathwise digital or a pathwise gamma is refused rather than returned as zero.
"""

import numpy as np
import pytest
from scipy.stats import norm

from volsurface import bs
from volsurface.core.types import Contract, Digital, Forward, Param, Spot, Vanilla
from volsurface.greeks import Analytic, Autodiff, FiniteDifference, LikelihoodRatio, Pathwise
from volsurface.mc.zrule import check_z
from volsurface.models.bs import BlackScholes
from volsurface.models.gbm import GBM


def _assert_z(z) -> None:
    failure = check_z(np.asarray(z, dtype=float))
    assert failure is None, failure


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
        model,
        contracts[:2],
        [Forward(), Param("sigma")],
        order=1,
    )
    np.testing.assert_allclose(vanilla.value[:, 0], first.value[:2, 1], rtol=1e-6)
    np.testing.assert_allclose(vanilla.value[:, 1], first.value[:2, 2], rtol=1e-6)
    second_fd = FiniteDifference(F=100.0, df=0.97, bump=1e-4).compute(
        model,
        contracts[:2],
        [Forward(), Param("sigma")],
        order=2,
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
    truth = (
        Analytic(F=100.0, df=1.0)
        .compute(
            model,
            [Contract(Vanilla(100.0), 1.0)],
            [Forward()],
            order=1,
        )
        .value[0, 0]
    )
    bumps = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3])
    error = np.empty(bumps.size)
    for i, h in enumerate(bumps):
        got = (
            FiniteDifference(F=100.0, df=1.0, bump=float(h))
            .compute(
                model,
                [Contract(Vanilla(100.0), 1.0)],
                [Forward()],
                order=1,
            )
            .value[0, 0]
        )
        error[i] = abs(got - truth)
    # Each halving multiplies the error by about 1/4.
    ratios = error[:-1] / error[1:]
    np.testing.assert_allclose(ratios, 4.0, atol=0.5)
    slope = float(np.polyfit(np.log(bumps), np.log(error), 1)[0])
    assert slope == pytest.approx(2.0, abs=0.2), slope


def test_common_random_numbers_cut_the_finite_difference_variance():
    """40 seeds, 2,000 paths, bump 1 on a forward of 100, at the money, sigma 0.2.

    Independent bumps have variance about ``Var(payoff) / (2 h² n)``, near
    0.04 here. Shared paths have variance near 1.3e-4, because the kink, not
    the bump, sets the noise. The ratio is about 300. Requiring more than 50
    still fails when the two bumps do not share normals, and it does not
    demand the theoretical factor from one sample of 40.
    """
    contract = [Contract(Vanilla(100.0), 1.0)]
    model = GBM(sigma=0.2, n_paths=2_000, n_steps=1, scheme="exact")

    def once(seed, crn):
        engine = FiniteDifference(F=100.0, df=1.0, bump=1.0, crn=crn)
        return engine.compute(model, contract, [Forward()], n_paths=2_000, seed=seed).value[0, 0]

    crn = np.array([once(seed, True) for seed in range(40)])
    indep = np.array([once(1_000 + seed, False) for seed in range(40)])
    factor = float(np.var(indep, ddof=1) / np.var(crn, ddof=1))
    assert factor > 50.0, factor


def test_common_random_numbers_do_not_advance_the_caller_seed():
    """Two computes with one ``SeedSequence`` both draw child 0.

    ``spawn`` copies the sequence, so the caller's ``n_children_spawned`` stays
    0 and the two values match, including a sigma bump that resimulates.
    """
    parent = np.random.SeedSequence(12345)
    model = GBM(sigma=0.2, n_paths=300, n_steps=1, scheme="exact")
    contract = [Contract(Vanilla(100.0), 1.0)]
    forward = FiniteDifference(F=100.0, df=1.0, bump=1.0, crn=True)
    first = forward.compute(model, contract, [Forward()], seed=parent).value
    second = forward.compute(model, contract, [Forward()], seed=parent).value
    np.testing.assert_array_equal(first, second)
    # A bump of 1 would send sigma = 0.2 negative. 0.05 stays positive and still resimulates.
    sigma = FiniteDifference(F=100.0, df=1.0, bump=0.05, crn=True)
    first_s = sigma.compute(model, contract, [Param("sigma")], seed=parent).value
    second_s = sigma.compute(model, contract, [Param("sigma")], seed=parent).value
    np.testing.assert_array_equal(first_s, second_s)
    assert parent.n_children_spawned == 0


def test_finite_difference_standard_error_follows_the_simulator():
    """A Monte Carlo bump reports a standard error. A closed form does not.

    At bump 1 on a forward of 100 the independent-seed error is the larger
    one: the shared-path difference cancels the payoff noise away from the strike.
    """
    contract = [Contract(Vanilla(100.0), 1.0)]
    model = GBM(sigma=0.2, n_paths=2_000, n_steps=1, scheme="exact")
    shared = FiniteDifference(F=100.0, df=1.0, bump=1.0, crn=True).compute(
        model,
        contract,
        [Forward()],
        seed=4,
    )
    split = FiniteDifference(F=100.0, df=1.0, bump=1.0, crn=False).compute(
        model,
        contract,
        [Forward()],
        seed=4,
    )
    assert shared.stderr is not None and shared.stderr.shape == (1, 1)
    assert split.stderr is not None and split.stderr.shape == (1, 1)
    assert np.isfinite(shared.stderr[0, 0]) and shared.stderr[0, 0] > 0.0
    assert np.isfinite(split.stderr[0, 0]) and split.stderr[0, 0] > shared.stderr[0, 0]
    closed = FiniteDifference(F=100.0, df=1.0, bump=1e-4).compute(
        BlackScholes(sigma=0.2),
        contract,
        [Forward()],
    )
    assert closed.stderr is None


def test_omitted_bump_scales_with_the_level():
    """No explicit bump: ``1e-4`` times the level on Black, and a finite Monte Carlo delta.

    The forward level is 100, so the closed-form step is 0.01. Sigma is 0.2,
    so its step is 2e-5. Both stay on the analytic delta and vega.
    """
    contract = [Contract(Vanilla(100.0), 1.0)]
    model = BlackScholes(sigma=0.2)
    got = FiniteDifference(F=100.0).compute(model, contract, [Forward(), Param("sigma")])
    truth = Analytic(F=100.0).compute(model, contract, [Forward(), Param("sigma")])
    np.testing.assert_allclose(got.value, truth.value, rtol=1e-5, atol=1e-6)
    assert got.stderr is None
    mc = FiniteDifference(F=100.0).compute(
        GBM(sigma=0.2, n_paths=400, scheme="exact"),
        contract,
        [Forward()],
        seed=1,
    )
    assert np.isfinite(mc.value[0, 0])
    assert mc.stderr is not None and np.isfinite(mc.stderr[0, 0]) and mc.stderr[0, 0] > 0.0


def test_finite_difference_standard_error_is_calibrated_over_200_seeds():
    """Sample standard deviation over mean reported standard error.

    200 seeds, numbered 10_000 to 10_199. Exact geometric Brownian motion,
    volatility 0.2, 2,000 antithetic paths, one step, common random numbers.
    The bump is 1 on the forward and 0.04 on sigma. Gamma is the forward
    second derivative at bump 2. The bands are [0.9, 1.1] for delta and vega
    and [0.8, 1.2] for gamma. The ratio is the calibration, not a slope of
    ``stderr`` against ``1/sqrt(n)``.
    """
    contract = [Contract(Vanilla(100.0), 1.0)]
    model = GBM(sigma=0.2, n_paths=2_000, n_steps=1, scheme="exact", antithetic=True)
    specs = (
        ("delta", Forward(), 1, 1.0, 0.9, 1.1),
        ("sigma", Param("sigma"), 1, 0.04, 0.9, 1.1),
        ("gamma", Forward(), 2, 2.0, 0.8, 1.2),
    )
    seeds = range(10_000, 10_200)
    for name, factor, order, bump, lo, hi in specs:
        values = np.empty(len(seeds))
        errors = np.empty(len(seeds))
        for i, seed in enumerate(seeds):
            engine = FiniteDifference(F=100.0, df=1.0, bump=bump, crn=True)
            got = engine.compute(
                model,
                contract,
                [factor],
                order=order,
                n_paths=2_000,
                seed=seed,
            )
            values[i] = got.value[0, 0]
            errors[i] = got.stderr[0, 0]
        ratio = float(np.std(values, ddof=1) / np.mean(errors))
        assert lo < ratio < hi, (name, ratio)


def test_pathwise_matches_black_and_refuses_a_digital_and_a_gamma():
    """40 strikes, independent seeds, 8,000 paths. The z-rule uses n = 40."""
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
    _assert_z(z_delta)
    _assert_z(z_vega)
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
    """Delta, vega and gamma on 20 strikes. Short digital error exceeds twice the long one.

    n = 20, 20,000 paths, independent seeds, checked with the z-rule. Volga is
    one strike and 100,000 paths, so its bound is the n = 1 maximum. The
    digital score divides by ``sigma² T``. At the money the theoretical ratio
    of the standard errors is several times two; the test asks for two so one
    unlucky pair of seeds does not fail it.
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
            bs_model,
            contract,
            [Forward(), Param("sigma")],
            order=1,
        ).value[0]
        truth_g = analytic.compute(bs_model, contract, [Forward()], order=2).value[0, 0]
        mc = GBM(sigma=sigma, scheme="exact", n_steps=1)
        got_1 = LikelihoodRatio(F=F, df=df).compute(
            mc,
            contract,
            [Forward(), Param("sigma")],
            order=1,
            n_paths=20_000,
            seed=300 + i,
        )
        got_g = LikelihoodRatio(F=F, df=df).compute(
            mc,
            contract,
            [Forward()],
            order=2,
            n_paths=20_000,
            seed=300 + i,
        )
        z["delta"][i] = (got_1.value[0, 0] - truth_1[0]) / got_1.stderr[0, 0]
        z["vega"][i] = (got_1.value[0, 1] - truth_1[1]) / got_1.stderr[0, 1]
        z["gamma"][i] = (got_g.value[0, 0] - truth_g) / got_g.stderr[0, 0]
    for row in z.values():
        _assert_z(row)

    # A digital payoff does not shrink with expiry, and the score divides by
    # sigma^2 T, so the short-dated standard error is the larger one.
    short = _digital_delta_se(F, sigma, df, T=0.05, seed=7)
    long = _digital_delta_se(F, sigma, df, T=1.0, seed=8)
    assert short > 2.0 * long, (short, long)

    contract = [Contract(Vanilla(F), 1.0)]
    truth_v = analytic.compute(bs_model, contract, [Param("sigma")], order=2).value[0, 0]
    got_v = LikelihoodRatio(F=F, df=df).compute(
        GBM(sigma=sigma, scheme="exact"),
        contract,
        [Param("sigma")],
        order=2,
        n_paths=100_000,
        seed=9,
    )
    z_volga = (got_v.value[0, 0] - truth_v) / got_v.stderr[0, 0]
    _assert_z([z_volga])


def _digital_delta_se(F, sigma, df, T, seed) -> float:
    contract = [Contract(Digital(F, "call"), T)]
    got = LikelihoodRatio(F=F, df=df).compute(
        GBM(sigma=sigma, scheme="exact"),
        contract,
        [Forward()],
        n_paths=30_000,
        seed=seed,
    )
    return float(got.stderr[0, 0])


def test_likelihood_ratio_prices_a_digital_that_pathwise_cannot():
    """One cash-or-nothing call. n = 1, 200,000 paths, z-rule against df n(d2) / (F s)."""
    F, K, T, sigma, df = 100.0, 105.0, 0.5, 0.2, 0.99
    analytic = Analytic(F=F, df=df).compute(
        BlackScholes(sigma=sigma),
        [Contract(Digital(K, "call"), T)],
        [Forward()],
    )
    got = LikelihoodRatio(F=F, df=df).compute(
        GBM(sigma=sigma, scheme="exact"),
        [Contract(Digital(K, "call"), T)],
        [Forward()],
        n_paths=200_000,
        seed=11,
    )
    z = (got.value[0, 0] - analytic.value[0, 0]) / got.stderr[0, 0]
    _assert_z([z])


def test_autodiff_matches_pathwise_to_1e_10_in_float64():
    """Same seed, one step and four steps. The values, not just the expectations, agree."""
    pytest.importorskip("torch")
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
    """A put, one strike, 30,000 paths. The z-rule at n = 1 against the Black put delta."""
    F, K, T, sigma, df = 100.0, 100.0, 1.0, 0.2, 1.0
    truth = float(bs.delta(F, K, T, sigma, df, "put"))
    got = Pathwise(F=F, df=df).compute(
        GBM(sigma=sigma, scheme="exact"),
        [Contract(Vanilla(K, "put"), T)],
        [Spot()],
        n_paths=30_000,
        seed=13,
    )
    z = (got.value[0, 0] - truth) / got.stderr[0, 0]
    _assert_z([z])


def test_simulation_greeks_refuse_a_control_variate():
    """Pathwise, likelihood ratio, and autodiff do not differentiate a controlled price."""
    model = GBM(sigma=0.2, n_paths=100, scheme="exact", control_variate=True)
    contract = Contract(Vanilla(100.0), 1.0)
    for engine in (Pathwise(100.0), LikelihoodRatio(100.0)):
        assert not engine.supports(model, contract, 1)
        with pytest.raises(ValueError, match="control variate"):
            engine.compute(model, [contract], [Forward()], n_paths=100, seed=1)
    pytest.importorskip("torch")
    engine = Autodiff(100.0)
    assert not engine.supports(model, contract, 1)
    with pytest.raises(ValueError, match="control variate"):
        engine.compute(model, [contract], [Forward()], n_paths=100, seed=1)
