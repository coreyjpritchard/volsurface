"""Market paths, pricers, and the nu-only regime chain. DESIGN sections 21 and 22."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import binom, chisquare

from volsurface import bs
from volsurface.core.protocols import capabilities
from volsurface.core.types import Contract, Digital, MarketPaths, RegimeSpec, Vanilla
from volsurface.markets.filter import hamilton_filter, regime_scores
from volsurface.markets.pricers import ClosedFormPricer, SurrogatePricer, VarianceSwapPricer
from volsurface.markets.regime import (
    RegimeSwitchingBergomi,
    _bridge_log_returns,
    _kernel_weights,
    _log_variance_increments,
    _stochastic,
    _surface_seed,
    bergomi_forward_variance,
    filter_differenced_realised_variance,
    filter_log_realised_variance,
    filter_log_variance,
    hybrid_increment_variance,
    increment_variance,
    trader_forward_variance,
)
from volsurface.markets.seeds import DRAW_ORDER, run_seed, stream_seeds
from volsurface.mc.rng import generator
from volsurface.mc.zrule import check_z
from volsurface.models.bs import BlackScholes
from volsurface.models.heston import Heston


def _spec(switch="nu") -> RegimeSpec:
    return RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": 0.4}, {"nu": 1.1}],
        p0=np.array([0.7, 0.3]),
        switch=switch,
    )


def _model(**kwargs) -> RegimeSwitchingBergomi:
    values = {"xi0": 0.04, "rho": -0.4, "H": 0.5}
    values.update(kwargs)
    return RegimeSwitchingBergomi(_spec(), **values)


def test_switch_other_than_nu_raises():
    times = np.linspace(0.0, 0.2, 3)
    for switch in ("xi_level", "both"):
        model = RegimeSwitchingBergomi(_spec(switch), xi0=0.04, rho=-0.4)
        with pytest.raises(ValueError, match="nu"):
            model.simulate_market(4, times, seed=1)


def test_missing_rough_driver_is_not_replaced(monkeypatch):
    def absent(name, **kwargs):
        raise KeyError(name)

    monkeypatch.setattr("volsurface.markets.regime.create", absent)
    model = _model(H=0.1)
    with pytest.raises(RuntimeError, match="roughvol"):
        model.simulate_market(4, np.linspace(0.0, 0.2, 3), seed=1)


def test_plugin_driver_is_called_and_not_labelled_rbergomi(monkeypatch):
    pytest.importorskip("roughvol")
    seen = []

    class _StandIn:
        def volterra_components(self, n_paths, times, *, seed, H):
            seen.append((H, n_paths))
            return np.zeros((n_paths, len(times) - 1, 2))

    monkeypatch.setattr("volsurface.markets.regime.create", lambda name, **kwargs: _StandIn())
    model = _model(H=0.1)
    paths = model.simulate_market(6, np.linspace(0.0, 0.2, 4), seed=3)
    # The market draw is the call with 6 paths. The emission pilot calls
    # the same plugin again, once per regime, and is not the market draw.
    assert (0.1, 6) in seen
    assert paths.meta["roughvol"] is False
    assert paths.meta["label"] == "_StandIn"
    assert "market_simulate" in capabilities(model)


def test_train_and_test_streams_are_disjoint():
    train = run_seed(11, 0)
    test = run_seed(11, 2)
    assert train.spawn_key == (0,)
    assert test.spawn_key == (2,)
    train_streams = stream_seeds(train)
    test_streams = stream_seeds(test)
    assert list(train_streams) == list(DRAW_ORDER)
    train_draw = generator(train_streams["regime"]).standard_normal()
    test_draw = generator(test_streams["regime"]).standard_normal()
    assert train_draw != pytest.approx(test_draw)
    child_draws = [generator(train_streams[name]).standard_normal() for name in DRAW_ORDER]
    for left in range(4):
        for right in range(left + 1, 4):
            assert child_draws[left] != pytest.approx(child_draws[right])


def test_same_seed_repeats_the_paths():
    model = _model()
    times = np.linspace(0.0, 0.5, 6)
    first = model.simulate_market(24, times, seed=run_seed(5, 0))
    second = model.simulate_market(24, times, seed=run_seed(5, 0))
    assert np.array_equal(first.Z, second.Z)
    assert np.array_equal(first.regime, second.regime)
    assert np.array_equal(first.regime_prob, second.regime_prob)


def test_chain_matches_expm():
    """Chi-square at alpha 0.001. Sample size 4,000 paths, one time."""
    model = _model()
    times = np.array([0.0, 0.5, 1.0])
    paths = model.simulate_market(4000, times, seed=run_seed(1, 2))
    counts = np.bincount(paths.regime[:, -1], minlength=2)
    # P(t) = expm(Q t) acts on the right: a row law is p0 @ P(t).
    law = _spec().p0 @ expm(_spec().generator * 1.0)
    law = np.asarray(law, dtype=float)
    law = law / law.sum()
    expected = counts.sum() * law
    stat, p_value = chisquare(counts, expected)
    assert p_value > 0.001
    assert stat < 10.83


def test_mean_z_and_variance_stay_inside_the_z_rule():
    """Eight independent runs, 2,000 paths, terminal time 1. The z-rule is the test."""
    model = _model()
    times = np.linspace(0.0, 1.0, 11)
    z_scores = []
    v_scores = []
    for replication in range(8):
        paths = model.simulate_market(2000, times, seed=run_seed(30 + replication, 2))
        terminal = paths.Z[:, -1]
        se = terminal.std(ddof=1) / math.sqrt(terminal.size)
        z_scores.append((terminal.mean() - 1.0) / se)
        variance = paths.state["v"][:, -1]
        se_v = variance.std(ddof=1) / math.sqrt(variance.size)
        v_scores.append((variance.mean() - model.xi0) / se_v)
    assert check_z(np.asarray(z_scores)) is None
    assert check_z(np.asarray(v_scores)) is None


def test_rough_mean_variance_matches_xi0_inside_the_z_rule():
    """Eight runs, 2,000 paths, H = 0.1, terminal time 1. The z-rule is the test.

    The driver is roughvol's ``(dW, I_1, ..., I_kappa)``. ``E[v] - xi0`` and
    ``E[Z] - 1`` are in standard errors of the mean.
    """
    pytest.importorskip("roughvol")
    model = _model(H=0.1)
    times = np.linspace(0.0, 1.0, 11)
    z_scores = []
    v_scores = []
    for replication in range(8):
        paths = model.simulate_market(2000, times, seed=run_seed(70 + replication, 2))
        terminal = paths.Z[:, -1]
        se = terminal.std(ddof=1) / math.sqrt(terminal.size)
        z_scores.append((terminal.mean() - 1.0) / se)
        variance = paths.state["v"][:, -1]
        se_v = variance.std(ddof=1) / math.sqrt(variance.size)
        v_scores.append((variance.mean() - model.xi0) / se_v)
    assert paths.meta["roughvol"] is True
    assert paths.meta["label"] == "RoughBergomi"
    assert check_z(np.asarray(z_scores)) is None
    assert check_z(np.asarray(v_scores)) is None


def test_constant_nu_hybrid_matches_the_roughvol_driver():
    """Constant nu factors out. The rebuilt variance matches roughvol's Y."""
    pytest.importorskip("roughvol")
    from roughvol.rbergomi import _variance_from_driver
    from roughvol.rng import generator as rough_generator
    from roughvol.volterra import hybrid_scheme_rng

    from volsurface.plugins import create

    hurst = 0.1
    nu = 1.5
    xi0 = 0.04
    times = np.linspace(0.0, 1.0, 17)
    n_paths = 32
    n_steps = times.size - 1
    vol_seed = stream_seeds(run_seed(9, 2))["vol"]
    plugin = create("rbergomi")
    components = np.asarray(
        plugin.volterra_components(n_paths, times, seed=vol_seed, H=hurst), dtype=float
    )
    kappa = components.shape[-1] - 1
    nu_step = np.full((n_paths, n_steps), nu)
    variance = hybrid_increment_variance(hurst, nu_step, xi0, times, components)
    vol = hybrid_scheme_rng(hurst, n_steps, 1.0, n_paths, rough_generator(vol_seed), kappa=kappa)
    level = np.full(times.shape, xi0)
    expected = _variance_from_driver(hurst, nu, level, times, vol.Y)
    assert variance == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_variance_swap_matches_realised_variance_within_se():
    """Left Riemann. Eight runs, 1,500 paths. The gap is in standard errors of the mean."""
    model = _model()
    pricer = VarianceSwapPricer()
    times = np.linspace(0.0, 1.0, 11)
    contract = Contract(Vanilla(100.0), T=1.0)
    scores = []
    for replication in range(8):
        paths = model.simulate_market(1500, times, seed=run_seed(40 + replication, 0))
        quoted = pricer.price_along(paths, [contract])[:, 0, 0]
        dt = np.diff(times)
        realised = paths.state["v"][:, :-1] @ dt
        gap = realised - quoted
        se = gap.std(ddof=1) / math.sqrt(gap.size)
        scores.append(gap.mean() / se)
    assert check_z(np.asarray(scores)) is None


def test_closed_form_holds_the_payoff_and_matches_the_formula():
    model = BlackScholes(sigma=0.2)
    pricer = ClosedFormPricer(model)
    times = np.array([0.0, 0.5, 1.0, 1.5])
    Z = np.array([[1.0, 1.1, 0.9, 1.2]])
    paths = MarketPaths(
        times=times,
        Z=Z,
        state={},
        F0=lambda t: np.full(np.shape(np.asarray(t, dtype=float)), 100.0),
        df=lambda t: np.exp(-0.01 * np.asarray(t, dtype=float)),
    )
    contract = Contract(Vanilla(90.0, "call"), T=1.0)
    cash = pricer.price_along(paths, [contract])[0, :, 0]
    df_ratio = math.exp(-0.01 * 1.0)
    at_start = model.price(100.0, 90.0, 1.0, df_ratio, "call")
    assert cash[0] == pytest.approx(float(at_start), abs=1e-12)
    later_df = math.exp(-0.01 * 0.5)
    at_half = model.price(110.0, 90.0, 0.5, df_ratio / later_df, "call")
    assert cash[1] == pytest.approx(float(at_half), abs=1e-12)
    # Payoff at the expiry column, held unchanged one step later.
    assert cash[2] == pytest.approx(0.0, abs=1e-12)
    assert cash[3] == pytest.approx(cash[2], abs=1e-12)
    digital = Contract(Digital(100.0, "call"), T=0.5)
    digit = pricer.price_along(paths, [digital])[0, 0, 0]
    assert 0.0 < digit < math.exp(-0.01 * 0.5)


def test_heston_pricer_uses_the_path_variance():
    model = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.5)
    pricer = ClosedFormPricer(model)
    paths = MarketPaths(
        times=np.array([0.0, 1.0]),
        Z=np.ones((1, 2)),
        state={"v": np.array([[0.09, 0.09]])},
        F0=lambda t: np.full(np.shape(np.asarray(t, dtype=float)), 100.0),
        df=lambda t: np.ones_like(np.asarray(t, dtype=float), dtype=float),
    )
    contract = Contract(Vanilla(100.0), T=1.0)
    got = pricer.price_along(paths, [contract])[0, 0, 0]
    formula = model.price(100.0, 100.0, 1.0, 1.0, "call", state={"v": np.array(0.09)})
    assert got == pytest.approx(float(formula), abs=1e-10)
    assert pricer.price_along(paths, [contract])[0, 1, 0] == pytest.approx(0.0, abs=1e-12)


def test_surrogate_refuses_an_error_above_its_bar():
    def predict(paths, contracts):
        return np.zeros((paths.Z.shape[0], paths.Z.shape[1], len(contracts)))

    kept = SurrogatePricer(predict, validation_error_bp=2.0, bar_bp=5.0)
    assert kept.validation_error_bp == 2.0
    with pytest.raises(ValueError, match="bar"):
        SurrogatePricer(predict, validation_error_bp=10.0, bar_bp=5.0)


def test_filter_is_causal_and_inside_binomial_bands():
    """Reliability at the last date, 2,500 paths.

    Bands are the 0.1 and 99.9 percent binomial quantiles.
    """
    rng = generator(run_seed(4, 1))
    n_paths, n_times = 2500, 30
    transition = np.array([[0.97, 0.03], [0.04, 0.96]])
    means = np.array([-1.0, 1.0])
    variances = np.array([0.25, 0.25])
    regime = np.empty((n_paths, n_times), dtype=int)
    regime[:, 0] = rng.random(n_paths) < 0.5
    for t in range(1, n_times):
        stay = rng.random(n_paths)
        regime[:, t] = np.where(regime[:, t - 1] == 0, stay > 0.97, stay > 0.04).astype(int)
    y = means[regime] + 0.5 * rng.standard_normal((n_paths, n_times))
    probs = hamilton_filter(y, transition, means, variances, np.array([0.5, 0.5]))
    altered = y.copy()
    altered[:, 10:] = 0.0
    earlier = hamilton_filter(altered, transition, means, variances, np.array([0.5, 0.5]))
    assert np.allclose(probs[:, :10, :], earlier[:, :10, :], atol=1e-12)
    predicted = probs[:, -1, 1]
    truth = regime[:, -1] == 1
    checked = 0
    edges = np.linspace(0.0, 1.0, 6)
    for band in range(5):
        if band < 4:
            mask = (predicted >= edges[band]) & (predicted < edges[band + 1])
        else:
            mask = (predicted >= edges[band]) & (predicted <= edges[band + 1])
        count = int(mask.sum())
        if count < 40:
            continue
        hits = int(truth[mask].sum())
        centre = float(np.clip(predicted[mask].mean(), 1e-6, 1.0 - 1e-6))
        lo = float(binom.ppf(0.001, count, centre))
        hi = float(binom.ppf(0.999, count, centre))
        assert lo <= hits <= hi
        checked += 1
    assert checked >= 2


def test_forward_variance_matches_the_closed_form_away_from_one_half():
    """One path. The weight is the exact integral of ``(u - s)^{H - 1/2}``."""
    H = 0.25
    nu = 0.8
    xi0 = 0.05
    times = np.array([0.0, 0.16, 0.36, 1.0])
    increments = np.array([[0.3, -0.15, 0.05]])
    xi = bergomi_forward_variance(H, nu, xi0, times, increments)
    assert xi[0, 0, :] == pytest.approx(xi0)
    assert np.isnan(xi[0, 2, 0])

    alpha = H - 0.5
    u = 1.0
    # Two steps have finished by t = 0.36. Each weight is the antiderivative.
    steps = ((0.0, 0.16, 0.3), (0.16, 0.36, -0.15))
    integral = 0.0
    for left, right, shock in steps:
        dt = right - left
        weight = ((u - left) ** (alpha + 1) - (u - right) ** (alpha + 1)) / ((alpha + 1) * dt)
        integral += weight * shock
    t = 0.36
    compensator = 0.5 * nu**2 * (u ** (2 * H) - (u - t) ** (2 * H))
    expected = xi0 * math.exp(nu * math.sqrt(2 * H) * integral - compensator)
    assert xi[0, 2, 3] == pytest.approx(expected, rel=1e-12)

    dt = 0.16
    y_weight = dt**alpha / (alpha + 1.0)
    y = y_weight * 0.3
    variance = xi0 * math.exp(nu * math.sqrt(2 * H) * y - 0.5 * nu**2 * dt ** (2 * H))
    assert xi[0, 1, 1] == pytest.approx(variance, rel=1e-12)
    # The spot variance, copied onto a later tenor, is the bug this replaces.
    spot = xi0 * math.exp(-0.5 * nu**2 * t ** (2 * H))
    assert xi[0, 2, 3] != pytest.approx(spot)


def test_half_hurst_forward_variance_depends_on_the_clock_only():
    """At H = 1/2, ``I_t(u) = W_t`` and the exponent has no tenor in it."""
    nu = 1.1
    times = np.array([0.0, 0.2, 0.5, 1.0])
    increments = np.array([[0.1, -0.4, 0.2]])

    def xi0(t):
        grid = np.asarray(t, dtype=float)
        return 0.03 + 0.01 * grid

    xi = bergomi_forward_variance(0.5, nu, xi0, times, increments)
    level = xi0(times)
    assert xi[0, 0, :] == pytest.approx(level)
    for i, t in enumerate(times):
        if i == 0:
            continue
        w = float(increments[0, :i].sum())
        factor = math.exp(nu * w - 0.5 * nu**2 * float(t))
        assert xi[0, i, i:] == pytest.approx(level[i:] * factor, rel=1e-12)
        assert np.all(np.isnan(xi[0, i, :i]))


def test_zero_driver_stores_the_rough_compensator(monkeypatch):
    """Near field and dW are 0. Variance is the compensator of the nu path."""
    pytest.importorskip("roughvol")

    class _StandIn:
        def volterra_components(self, n_paths, times, *, seed, H):
            return np.zeros((n_paths, len(times) - 1, 2))

    monkeypatch.setattr("volsurface.markets.regime.create", lambda name, **kwargs: _StandIn())
    H = 0.1
    model = _model(H=H)
    times = np.linspace(0.0, 0.2, 4)
    paths = model.simulate_market(6, times, seed=3)
    nus = np.array([0.4, 1.1])
    nu_step = nus[paths.regime[:, :-1]]
    weights = _kernel_weights(H, times, times)
    zeros = np.zeros((6, times.size - 1))
    expected_v = increment_variance(H, nu_step, model.xi0, times, zeros, weights)
    assert paths.state["v"] == pytest.approx(expected_v, rel=1e-12)
    expected_xi = trader_forward_variance(
        H, nus, paths.state["v"], paths.regime_prob, model.xi0, times, zeros
    )
    np.testing.assert_allclose(paths.state["xi"], expected_xi, rtol=1e-12, equal_nan=True)
    diagonal = np.arange(times.size)
    assert paths.state["xi"][:, diagonal, diagonal] == pytest.approx(paths.state["v"], rel=1e-12)
    assert np.isnan(paths.state["xi"][:, 1, 0]).all()


def test_forward_variance_matches_roughvol_on_a_known_increment():
    """Same dW, same grid. Skip only when the plugin is absent."""
    pytest.importorskip("roughvol")
    from roughvol.rbergomi import _forward_curve

    H = 0.1
    nu = 1.3
    xi0 = 0.04
    times = np.linspace(0.0, 1.0, 9)
    increments = np.random.default_rng(0).normal(scale=np.sqrt(np.diff(times)), size=(5, 8))
    ours = bergomi_forward_variance(H, nu, xi0, times, increments)
    variance = np.diagonal(ours, axis1=1, axis2=2)
    theirs = _forward_curve(H, nu, xi0, times, increments, variance, times)
    assert np.allclose(ours, theirs, equal_nan=True, rtol=1e-10, atol=1e-12)


def test_a_switch_does_not_rescale_variance_already_accumulated():
    """Two paths share dW. Variance agrees until the step whose nu differs."""
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    dW = np.array([[0.1, -0.2, 0.05, 0.03], [0.1, -0.2, 0.05, 0.03]])
    nu = np.array([[0.4, 0.4, 0.4, 1.1], [0.4, 0.4, 1.1, 1.1]])
    weights = _kernel_weights(0.5, times, times)
    variance = increment_variance(0.5, nu, 0.04, times, dW, weights)
    assert variance[0, :3] == pytest.approx(variance[1, :3])
    assert variance[0, 3] != pytest.approx(variance[1, 3])
    dt = 0.25
    step = variance[1, 2] * math.exp(1.1 * dW[1, 2] - 0.5 * 1.1**2 * dt)
    assert variance[1, 3] == pytest.approx(step)


def test_constant_nu_matches_the_bergomi_diagonal():
    """One nu on every step is the closed form, including the compensator."""
    hurst = 0.25
    nu = 0.8
    times = np.linspace(0.0, 1.0, 5)
    dW = np.array([[0.1, -0.2, 0.05, 0.02]])
    weights = _kernel_weights(hurst, times, times)
    nu_step = np.full((1, 4), nu)
    variance = increment_variance(hurst, nu_step, 0.05, times, dW, weights)
    curve = bergomi_forward_variance(hurst, nu, 0.05, times, dW)
    assert variance[0] == pytest.approx(np.diag(curve[0]), rel=1e-12)


def test_simulated_variance_uses_the_nu_in_force_on_the_step():
    """The update uses the regime at the left of the step, not the new label."""
    model = _model()
    times = np.linspace(0.0, 1.0, 11)
    paths = model.simulate_market(200, times, seed=run_seed(3, 0))
    nus = np.array([0.4, 1.1])
    nu_step = nus[paths.regime[:, :-1]]
    weights = _kernel_weights(0.5, times, times)
    expected = increment_variance(0.5, nu_step, model.xi0, times, paths.state["dW"], weights)
    assert paths.state["v"] == pytest.approx(expected, rel=1e-10)
    assert np.any(paths.regime[:, 1:] != paths.regime[:, :-1])
    nu_right = nus[paths.regime[:, 1:]]
    wrong = increment_variance(0.5, nu_right, model.xi0, times, paths.state["dW"], weights)
    assert not np.allclose(paths.state["v"], wrong)


def test_forward_curve_uses_the_closed_form_off_the_diagonal():
    """Later dates are the closed form. The diagonal is the path's variance.

    The closed-form diagonal is the kernel integral against ``dW``. It drops
    the hybrid near field. Multiplying the path's variance by the ratio of
    that closed form would lift the curve. This draw does not.
    """
    hurst = 0.1
    nu = np.array([1.5])
    times = np.linspace(0.0, 1.0, 5)
    dW = np.array([[0.2, -0.1, 0.05, 0.03]])
    variance = np.array([[0.04, 0.05, 0.03, 0.06, 0.045]])
    weights = np.ones((1, 5, 1))
    curve = trader_forward_variance(hurst, nu, variance, weights, 0.04, times, dW)
    closed = bergomi_forward_variance(hurst, float(nu[0]), 0.04, times, dW)
    assert curve[0, 2, 2] == pytest.approx(variance[0, 2])
    assert curve[0, 2, 4] == pytest.approx(closed[0, 2, 4])
    lifted = variance[0, 2] * closed[0, 2, 4] / closed[0, 2, 2]
    assert curve[0, 2, 4] != pytest.approx(lifted)


def test_constant_nu_mean_forward_curve_matches_xi0():
    """Eight runs, 500 paths, H = 0.1, clock index 8. The z-rule is the test.

    ``nu`` is 1.5 and then 3. The mean of the forward curve, over the
    dates after the clock, is ``xi0``. The diagonal is ``v_t`` and is not
    in this average. The surface log contract against the variance swap
    is not this test: both sides integrate the same curve.
    """
    pytest.importorskip("roughvol")
    times = np.linspace(0.0, 1.0, 17)
    clock = 8
    for nu in (1.5, 3.0):
        model = _single_regime(nu, n_mc=2, H=0.1, n_pilot=2)
        scores = []
        for replication in range(8):
            paths = model.simulate_market(500, times, seed=run_seed(80 + replication, 2))
            future = paths.state["xi"][:, clock, clock + 1 :]
            path_mean = future.mean(axis=1)
            se = path_mean.std(ddof=1) / math.sqrt(path_mean.size)
            scores.append((path_mean.mean() - model.xi0) / se)
        assert check_z(np.asarray(scores)) is None


def test_trader_curve_follows_the_filter_not_the_label():
    """At H other than 1/2 a later date is the mixture of the closed forms."""
    hurst = 0.25
    nus = np.array([0.4, 1.1])
    times = np.linspace(0.0, 1.0, 5)
    dW = np.array([[0.2, -0.1, 0.05, 0.0]])
    variance = np.array([[0.04, 0.05, 0.03, 0.06, 0.04]])
    low = np.zeros((1, 5, 2))
    low[:, :, 0] = 1.0
    high = np.zeros((1, 5, 2))
    high[:, :, 1] = 1.0
    curve_low = trader_forward_variance(hurst, nus, variance, low, 0.04, times, dW)
    curve_high = trader_forward_variance(hurst, nus, variance, high, 0.04, times, dW)
    assert curve_low[0, 2, 2] == pytest.approx(variance[0, 2])
    assert curve_high[0, 2, 2] == pytest.approx(variance[0, 2])
    assert curve_low[0, 2, 4] != pytest.approx(curve_high[0, 2, 4])
    mid = 0.5 * (low + high)
    curve_mid = trader_forward_variance(hurst, nus, variance, mid, 0.04, times, dW)
    assert curve_mid[0, 2, 4] == pytest.approx(0.5 * (curve_low[0, 2, 4] + curve_high[0, 2, 4]))


def test_prices_do_not_depend_on_the_true_regime():
    """Flipping the latent labels leaves the variance swap and the surface unchanged."""
    model = _model()
    times = np.linspace(0.0, 1.0, 11)
    paths = model.simulate_market(32, times, seed=run_seed(8, 0))
    flipped = replace(paths, regime=(1 - paths.regime).astype(np.int8))
    assert np.array_equal(model.surface_at(paths, 4).call, model.surface_at(flipped, 4).call)
    pricer = VarianceSwapPricer()
    contract = Contract(Vanilla(model.forward), T=1.0)
    left = pricer.price_along(paths, [contract])
    right = pricer.price_along(flipped, [contract])
    assert np.allclose(left, right)


def test_filter_reads_log_realised_variance_from_the_spot():
    """The observation is a function of Z. Time 0 is the prior, with no return yet."""
    model = _model()
    times = np.linspace(0.0, 0.5, 6)
    paths = model.simulate_market(20, times, seed=run_seed(6, 1))
    dt = np.diff(times)
    log_ret = np.log(paths.Z[:, 1:]) - np.log(paths.Z[:, :-1])
    expected = np.log(np.maximum(log_ret**2 / dt, 1e-300))
    assert paths.state["y"][:, 1:] == pytest.approx(expected)
    assert np.isnan(paths.state["y"][:, 0]).all()
    assert np.allclose(paths.regime_prob[:, 0, :], _spec().p0)


def test_log_variance_filter_matches_the_gaussian_increment():
    """The exact half-Brownian likelihood, not the latent-variance benchmark.

    One step, dt = 0.2, increment 0.05, prior (0.7, 0.3), identity transition.
    The emission is Normal(-nu^2 dt / 2, nu^2 dt). The posterior is
    (0.86427398, 0.13572602). Column 0 stays the prior. The market does
    not call this function.
    """
    from scipy.stats import norm

    dt = 0.2
    nu = np.array([0.4, 1.1])
    mean = -0.5 * nu**2 * dt
    var = nu**2 * dt
    prior = np.array([0.7, 0.3])
    like = np.exp(norm.logpdf(0.05, mean, np.sqrt(var)))
    posterior = prior * like
    posterior = posterior / posterior.sum()
    v = np.array([[0.04, 0.04 * np.exp(0.05)]])
    got = filter_log_variance(v, [np.eye(2)], nu, np.array([0.0, dt]), prior)
    assert got[0, 0] == pytest.approx(prior)
    assert got[0, 1] == pytest.approx(posterior)
    assert got[0, 1] == pytest.approx([0.86427398, 0.13572602], abs=1e-8)


def test_log_variance_filter_does_not_use_a_later_increment():
    """Changing the last increment of log variance moves only the last probability."""
    times = np.linspace(0.0, 0.3, 4)
    v = np.array([[0.04, 0.041, 0.039, 0.042]])
    step = np.array([[0.9, 0.1], [0.2, 0.8]])
    nu = np.array([0.4, 1.1])
    prior = np.array([0.7, 0.3])
    first = filter_log_variance(v, [step, step, step], nu, times, prior)
    moved = v.copy()
    moved[0, -1] = 0.2
    second = filter_log_variance(moved, [step, step, step], nu, times, prior)
    assert np.allclose(first[0, :-1], second[0, :-1])
    assert not np.allclose(first[0, -1], second[0, -1])


def test_prices_read_the_spot_filter_not_the_latent_benchmark():
    """``surface_at`` reads ``regime_prob``, the spot filter under the default.

    The windowed filter and the latent-variance benchmark are stored beside
    it. Replacing the latent array does not move the call. Replacing
    ``regime_prob`` does. The market does not call the exact half-Brownian
    likelihood.
    """
    model = _model(n_mc=16)
    times = np.linspace(0.0, 0.4, 5)
    paths = model.simulate_market(6, times, seed=run_seed(4, 2))
    assert paths.meta["price_filter"] == "spot"
    assert np.allclose(paths.regime_prob, paths.state["regime_prob_spot"])
    steps = [_stochastic_step(model, float(times[i] - times[i - 1])) for i in range(1, times.size)]
    window = filter_differenced_realised_variance(
        paths.state["y_rv"], steps, paths.meta["rv_means"], paths.meta["rv_vars"], _spec().p0
    )
    latent = filter_log_realised_variance(
        _log_variance_increments(paths.state["v"]),
        steps,
        paths.meta["latent_means"],
        paths.meta["latent_vars"],
        _spec().p0,
    )
    assert np.allclose(paths.state["regime_prob_rv"], window)
    assert np.allclose(paths.state["regime_prob_latent"], latent)
    assert not np.allclose(paths.regime_prob, paths.state["regime_prob_latent"])
    tweaked = dict(paths.state)
    benchmark = np.zeros_like(paths.state["regime_prob_latent"])
    benchmark[..., 0] = 1.0
    tweaked["regime_prob_latent"] = benchmark
    tweaked["regime_prob_rv"] = benchmark
    left = model.surface_at(paths, 2).call
    right = model.surface_at(replace(paths, state=tweaked), 2).call
    assert np.array_equal(left, right)
    flipped = paths.regime_prob[:, :, ::-1].copy()
    changed = model.surface_at(replace(paths, regime_prob=flipped), 2).call
    assert not np.allclose(left, changed)


def _stochastic_step(model: RegimeSwitchingBergomi, dt: float) -> np.ndarray:
    return _stochastic(expm(np.asarray(model.spec.generator, dtype=float) * dt))


def test_windowed_returns_sum_to_the_coarse_return():
    """``n_sub`` refines the coarse spot shock. The stored path does not move."""
    v_left = np.array([[0.04, 0.05]])
    dW = np.array([[0.1, -0.2]])
    perp = np.array([[0.3, 0.0]])
    times = np.array([0.0, 0.1, 0.3])
    rho = -0.4
    log_sub = _bridge_log_returns(
        v_left, dW, perp, times, rho, 21, generator(run_seed(1, 0))
    )
    orthogonal = math.sqrt(1.0 - rho * rho)
    for i, dt in enumerate(np.diff(times)):
        shock = rho * dW[0, i] + orthogonal * math.sqrt(dt) * perp[0, i]
        coarse = math.sqrt(v_left[0, i]) * shock - 0.5 * v_left[0, i] * dt
        assert float(log_sub[0, i].sum()) == pytest.approx(coarse)

    times = np.linspace(0.0, 0.5, 6)
    fine = _model(n_sub=21).simulate_market(8, times, seed=run_seed(6, 1))
    coarse = _model(n_sub=1).simulate_market(8, times, seed=run_seed(6, 1))
    assert np.array_equal(fine.Z, coarse.Z)
    assert np.array_equal(fine.state["v"], coarse.state["v"])
    assert coarse.state["y_rv"][:, 1:] == pytest.approx(coarse.state["y"][:, 1:])
    assert not np.allclose(fine.state["y_rv"][:, 1:], fine.state["y"][:, 1:])


def test_market_does_not_call_the_half_gaussian_likelihood(monkeypatch):
    """The latent benchmark is the pilot table, including at ``H = 1/2``."""

    def boom(*args, **kwargs):
        raise AssertionError("analytic log-variance likelihood")

    monkeypatch.setattr("volsurface.markets.regime.filter_log_variance", boom)
    paths = _model().simulate_market(4, np.linspace(0.0, 0.2, 3), seed=run_seed(2, 2))
    assert paths.state["regime_prob_latent"].shape == (4, 3, 2)
    assert not np.allclose(paths.meta["latent_means"][:, 0], paths.meta["latent_means"][:, 1])


def test_differenced_rv_filter_uses_the_regime_two_steps_back():
    """The change at date 2 is about the regime at date 0, then two transitions.

    The level is ``[nan, 0, 0.4]``. The change is 0.4. Means are -1 and 1,
    variances are 1, and the prior is (0.6, 0.4). Date 1 is the prior moved
    once, with no observation. Date 2 is the posterior moved twice.
    """
    from scipy.stats import norm

    level = np.array([[np.nan, 0.0, 0.4]])
    means = np.array([-1.0, 1.0])
    variances = np.array([1.0, 1.0])
    prior = np.array([0.6, 0.4])
    like = np.exp(norm.logpdf(0.4, means, np.sqrt(variances)))
    posterior = prior * like
    posterior = posterior / posterior.sum()
    first = np.array([[0.8, 0.2], [0.3, 0.7]])
    second = np.array([[0.9, 0.1], [0.4, 0.6]])
    got = filter_differenced_realised_variance(level, [first, second], means, variances, prior)
    assert got[0, 0] == pytest.approx(prior)
    assert got[0, 1] == pytest.approx(prior @ first)
    assert got[0, 2] == pytest.approx(posterior @ first @ second)
    assert got[0, 1] != pytest.approx(posterior @ first)


def test_differenced_rv_filter_does_not_use_a_later_change():
    """Changing the last level moves only the last probability.

    Dates 0 and 1 have no change yet, so they stay put when any later
    level moves. The first change uses the level at date 1, so moving
    that level moves every probability from date 2 on.
    """
    level = np.array([[np.nan, 0.0, 1.0, -2.0, 0.5]])
    step = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([0.0, 0.5])
    variances = np.array([1.0, 1.0])
    prior = np.array([0.5, 0.5])
    steps = [step, step, step, step]
    first = filter_differenced_realised_variance(level, steps, means, variances, prior)
    changed = level.copy()
    changed[0, -1] = 10.0
    second = filter_differenced_realised_variance(changed, steps, means, variances, prior)
    assert np.allclose(first[0, :-1], second[0, :-1])
    assert not np.allclose(first[0, -1], second[0, -1])
    early = level.copy()
    early[0, 1] = 4.0
    third = filter_differenced_realised_variance(early, steps, means, variances, prior)
    assert np.allclose(first[0, :2], third[0, :2])
    assert not np.allclose(first[0, 2:], third[0, 2:])


def _reliability_bands(predicted: np.ndarray, truth: np.ndarray) -> int:
    """Count bins whose hits sit inside the 0.1 and 99.9 percent binomial quantiles."""
    checked = 0
    edges = np.linspace(0.0, 1.0, 6)
    for band in range(5):
        if band < 4:
            mask = (predicted >= edges[band]) & (predicted < edges[band + 1])
        else:
            mask = (predicted >= edges[band]) & (predicted <= edges[band + 1])
        count = int(mask.sum())
        if count < 40:
            continue
        hits = int(truth[mask].sum())
        centre = float(np.clip(predicted[mask].mean(), 1e-6, 1.0 - 1e-6))
        lo = float(binom.ppf(0.001, count, centre))
        hi = float(binom.ppf(0.999, count, centre))
        assert lo <= hits <= hi
        checked += 1
    return checked


def test_differenced_rv_filter_is_calibrated_at_three_clocks():
    """Clocks 5, 10 and 20. Predicted probability of the high regime versus hits.

    Pre-registered settings, 1,600 paths, 21 times, seed entropy 12, test
    split. Bins with at least 40 paths. The bands are the 0.1 and 99.9
    percent binomial quantiles of the mean predicted probability.
    """
    pytest.importorskip("roughvol")
    spec = RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": 1.5}, {"nu": 3.0}],
        p0=np.array([0.7, 0.3]),
        switch="nu",
    )
    model = RegimeSwitchingBergomi(spec, xi0=0.04, rho=-0.4, H=0.1)
    times = np.linspace(0.0, 1.0, 21)
    paths = model.simulate_market(1600, times, seed=run_seed(12, 2))
    for clock in (5, 10, 20):
        predicted = paths.state["regime_prob_rv"][:, clock, 1]
        truth = paths.regime[:, clock] == 1
        assert _reliability_bands(predicted, truth) >= 1


def test_rv_filter_does_not_use_a_later_return():
    """Changing the last log realised variance moves only the last probability."""
    y = np.array([[np.nan, 0.0, 1.0, -2.0]])
    step = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([0.0, 0.0])
    variances = np.array([1.0, 4.0])
    prior = np.array([0.5, 0.5])
    first = filter_log_realised_variance(y, [step, step, step], means, variances, prior)
    changed = y.copy()
    changed[0, -1] = 10.0
    second = filter_log_realised_variance(changed, [step, step, step], means, variances, prior)
    assert np.allclose(first[0, :-1], second[0, :-1])
    assert not np.allclose(first[0, -1], second[0, -1])


def _bayes_accuracy(price: np.ndarray, regime: np.ndarray) -> tuple[float, bool]:
    """Modal regime inside eight quantile bins, and whether any bin holds both."""
    edges = np.quantile(price, np.linspace(0.0, 1.0, 9))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    correct = 0
    total = 0
    both = False
    labels = regime.astype(int)
    for band in range(8):
        if band < 7:
            mask = (price >= edges[band]) & (price < edges[band + 1])
        else:
            mask = (price >= edges[band]) & (price <= edges[band + 1])
        chosen = labels[mask]
        if chosen.size == 0:
            continue
        if np.unique(chosen).size > 1:
            both = True
        mode = int(np.bincount(chosen, minlength=2).argmax())
        correct += int(np.sum(chosen == mode))
        total += int(chosen.size)
    return correct / total, both


def test_regime_cannot_be_read_exactly_from_prices():
    """Six hundred paths, clock index 10, market-test settings.

    The settings are ``H = 1/2`` and ``nu`` of 0.4 and 1.1. The base rate
    is the larger share of the latent regime at that clock. Eight quantile
    bins of the call and of the variance swap each contain both regimes.
    Their modal accuracy beats the base rate by more than 0.05 and by less
    than 0.35, so the price is not the unconditional law and it is not the
    label. The spot filter's bar at these settings is the Brier skill
    against the constant forecast that quotes the sample shares. That skill
    is not positive: one-step log realised variance has one mean in both
    regimes, and ``pi**2 / 2`` dwarfs the gap in ``nu**2``.
    """
    model = _model()
    times = np.linspace(0.0, 1.0, 21)
    paths = model.simulate_market(600, times, seed=run_seed(12, 2))
    labels = paths.regime[:, 10]
    base = float(np.bincount(labels.astype(int), minlength=2).max() / labels.size)
    batch = model.surface_at(paths, 10)
    atm = int(np.argmin(np.abs(batch.k)))
    call_accuracy, call_both = _bayes_accuracy(batch.call[:, 0, atm], labels)
    contract = Contract(Vanilla(model.forward), T=1.0)
    swap = VarianceSwapPricer().price_along(paths, [contract])[:, 10, 0]
    swap_accuracy, swap_both = _bayes_accuracy(swap, labels)
    assert call_both and swap_both
    assert 0.05 < call_accuracy - base < 0.35
    assert 0.05 < swap_accuracy - base < 0.35
    spot = regime_scores(paths.state["regime_prob_spot"][:, 10], labels)
    assert spot["brier_skill"] <= 0.0
    assert spot["accuracy"] <= spot["base_rate"]


def _implied_vol_strip(batch, clock: float) -> np.ndarray:
    """Implied vol on expiries after ``clock``. Shape ``(m, n_exp, n_k)``."""
    columns = []
    for j, maturity in enumerate(batch.T):
        tau = float(maturity - clock)
        if tau <= 0.0:
            continue
        columns.append(bs.implied_vol(batch.call[:, j, :], 1.0, np.exp(batch.k), tau))
    return np.stack(columns, axis=1)


def _filter_draw(nus, H, n_mc):
    """Six hundred paths, 21 times, clock index 10, seed entropy 12, test split."""
    spec = RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": float(nu)} for nu in nus],
        p0=np.array([0.7, 0.3]),
        switch="nu",
    )
    model = RegimeSwitchingBergomi(spec, xi0=0.04, rho=-0.4, H=H, n_mc=n_mc)
    times = np.linspace(0.0, 1.0, 21)
    paths = model.simulate_market(600, times, seed=run_seed(12, 2))
    return model, times, paths


def _mixture_gaps(model, paths, clock_index):
    """Call gap, log-contract gap, and implied vol against the true-regime mixture."""
    labels = paths.regime[:, clock_index]
    one_hot = np.eye(2)[labels.astype(int)]
    true_weights = paths.regime_prob.copy()
    true_weights[:, clock_index, :] = one_hot
    true = model.surface_at(replace(paths, regime_prob=true_weights), clock_index)
    clock = float(paths.times[clock_index])
    iv_true = _implied_vol_strip(true, clock)
    out = {"true": true, "iv_true": iv_true}
    for name in ("regime_prob_spot", "regime_prob_rv", "regime_prob_latent"):
        weights = paths.regime_prob.copy()
        weights[:, clock_index, :] = paths.state[name][:, clock_index, :]
        batch = model.surface_at(replace(paths, regime_prob=weights), clock_index)
        out[name] = batch
        out["iv_" + name] = _implied_vol_strip(batch, clock)
    finite = np.isfinite(iv_true)
    for name in ("regime_prob_spot", "regime_prob_rv", "regime_prob_latent"):
        finite = finite & np.isfinite(out["iv_" + name])
    out["finite"] = finite
    return out


def test_market_test_filters_are_scored_against_the_base_rate():
    """H = 1/2, nu 0.4 and 1.1. Integer counts are the pin. Floats sit at 1e-9.

    The spot filter has no Brier skill. The differenced windowed filter is
    right on 345 of 600 paths. Its Brier skill is 0.0035. The latent
    benchmark is the pilot table, right on 487 of 600 paths. Prices read
    the spot filter. The variance swap does not depend on the weights.
    """
    model, times, paths = _filter_draw((0.4, 1.1), 0.5, 1024)
    assert model.price_filter == "spot"
    labels = paths.regime[:, 10]
    spot = regime_scores(paths.state["regime_prob_spot"][:, 10], labels)
    window = regime_scores(paths.state["regime_prob_rv"][:, 10], labels)
    latent = regime_scores(paths.state["regime_prob_latent"][:, 10], labels)
    assert spot["confusion"].tolist() == [[129, 148], [165, 158]]
    assert window["confusion"].tolist() == [[109, 168], [87, 236]]
    assert latent["confusion"].tolist() == [[223, 54], [59, 264]]
    assert int(spot["confusion"].sum()) == 600
    assert spot["base_rate"] == pytest.approx(323 / 600)
    assert spot["brier_skill"] < 0.0
    assert window["brier_skill"] > spot["brier_skill"]
    assert latent["brier_skill"] > 0.0
    assert spot["brier"] == pytest.approx(0.5221569169178578, abs=1e-9)
    assert window["brier"] == pytest.approx(0.4953107970789739, abs=1e-9)
    assert latent["brier"] == pytest.approx(0.2805716276927648, abs=1e-9)
    assert spot["log_score"] == pytest.approx(-0.7184579069520304, abs=1e-9)
    assert window["log_score"] == pytest.approx(-0.6892768575340644, abs=1e-9)
    assert latent["log_score"] == pytest.approx(-0.4411708645577869, abs=1e-9)
    assert spot["base_brier"] == pytest.approx(0.4970611111111111, abs=1e-9)
    assert spot["base_log_score"] == pytest.approx(-0.6902054058581284, abs=1e-9)

    gaps = _mixture_gaps(model, paths, 10)
    assert int(gaps["finite"].sum()) == 35032
    clock = float(times[10])
    assert clock == pytest.approx(0.5)
    for name, call_gap, variance_gap, iv_error in (
        ("regime_prob_spot", 0.0002897885217583061, 0.00045257763306333343, 0.005348033582997692),
        ("regime_prob_rv", 0.0002834313623806424, 0.0004396162668406603, 0.005232047160575295),
        ("regime_prob_latent", 0.00015488813465253245, 0.00024342840743678254, 0.00391528681479051),
    ):
        batch = gaps[name]
        iv = gaps["iv_" + name]
        finite = gaps["finite"]
        assert float(np.mean(np.abs(batch.call - gaps["true"].call))) == pytest.approx(
            call_gap, abs=1e-9
        )
        assert float(np.mean(np.abs(batch.total_variance - gaps["true"].total_variance))) == (
            pytest.approx(variance_gap, abs=1e-9)
        )
        got = float(np.sqrt(np.mean((iv[finite] - gaps["iv_true"][finite]) ** 2)))
        assert got == pytest.approx(iv_error, abs=1e-9)
    nus = np.array([0.4, 1.1])
    v = paths.state["v"]
    dW = paths.state["dW"]
    left = trader_forward_variance(0.5, nus, v, paths.state["regime_prob_spot"], 0.04, times, dW)
    right = trader_forward_variance(0.5, nus, v, paths.state["regime_prob_rv"], 0.04, times, dW)
    assert np.allclose(left, right, equal_nan=True)


def test_preregistered_filters_are_scored_against_the_base_rate(monkeypatch):
    """H = 0.1, nu 1.5 and 3. The spot hit rate is 410 of 600.

    That is the measured rate at these settings. The windowed filter and
    the latent benchmark both beat the base-rate forecast. The benchmark
    is the pilot table: the market does not call the half-Brownian
    likelihood. Prices still read the spot filter.
    """
    pytest.importorskip("roughvol")

    def boom(*args, **kwargs):
        raise AssertionError("analytic log-variance likelihood")

    monkeypatch.setattr("volsurface.markets.regime.filter_log_variance", boom)
    model, _times, paths = _filter_draw((1.5, 3.0), 0.1, 1024)
    assert model.price_filter == "spot"
    labels = paths.regime[:, 10]
    spot = regime_scores(paths.state["regime_prob_spot"][:, 10], labels)
    window = regime_scores(paths.state["regime_prob_rv"][:, 10], labels)
    latent = regime_scores(paths.state["regime_prob_latent"][:, 10], labels)
    assert spot["confusion"].tolist() == [[176, 101], [89, 234]]
    assert window["confusion"].tolist() == [[207, 70], [78, 245]]
    assert latent["confusion"].tolist() == [[207, 70], [66, 257]]
    assert spot["brier_skill"] > 0.0
    assert window["brier_skill"] > 0.0
    assert latent["brier_skill"] > window["brier_skill"] > spot["brier_skill"]
    assert spot["brier"] == pytest.approx(0.41927448869043304, abs=1e-9)
    assert window["brier"] == pytest.approx(0.35601895170282283, abs=1e-9)
    assert latent["brier"] == pytest.approx(0.33155110233077373, abs=1e-9)
    assert spot["log_score"] == pytest.approx(-0.6156778597945741, abs=1e-9)
    assert window["log_score"] == pytest.approx(-0.540600014719739, abs=1e-9)
    assert latent["log_score"] == pytest.approx(-0.5098968565249987, abs=1e-9)

    gaps = _mixture_gaps(model, paths, 10)
    assert int(gaps["finite"].sum()) == 31684
    for name, call_gap, iv_error in (
        ("regime_prob_spot", 0.0016502145362395097, 0.020694890419512047),
        ("regime_prob_rv", 0.001452224971440912, 0.01906455429095802),
        ("regime_prob_latent", 0.0013103675485724878, 0.018047992510532688),
    ):
        batch = gaps[name]
        iv = gaps["iv_" + name]
        finite = gaps["finite"]
        assert float(np.mean(np.abs(batch.call - gaps["true"].call))) == pytest.approx(
            call_gap, abs=1e-9
        )
        got = float(np.sqrt(np.mean((iv[finite] - gaps["iv_true"][finite]) ** 2)))
        assert got == pytest.approx(iv_error, abs=1e-9)


def test_four_thousand_path_filter_scores():
    """Clock index 10, seed entropy 12, test split, 4,000 paths. Both settings.

    The integer confusion counts are the pin. The market-test windowed
    skill is 0.0094. The pre-registered windowed skill is 0.2071. The
    pilot size is the default 1,024.
    """
    pytest.importorskip("roughvol")
    market = _score_draw((0.4, 1.1), 0.5, 4000)
    assert market["regime_prob_spot"]["confusion"].tolist() == [[894, 953], [1038, 1115]]
    assert market["regime_prob_rv"]["confusion"].tolist() == [[677, 1170], [613, 1540]]
    assert market["regime_prob_latent"]["confusion"].tolist() == [[1561, 286], [396, 1757]]
    assert market["regime_prob_spot"]["brier"] == pytest.approx(0.5116058140559726, abs=1e-9)
    assert market["regime_prob_rv"]["brier"] == pytest.approx(0.4924174573005298, abs=1e-9)
    assert market["regime_prob_latent"]["brier"] == pytest.approx(0.2592191835470902, abs=1e-9)
    rough = _score_draw((1.5, 3.0), 0.1, 4000)
    assert rough["regime_prob_spot"]["confusion"].tolist() == [[1174, 673], [684, 1469]]
    assert rough["regime_prob_rv"]["confusion"].tolist() == [[1336, 511], [670, 1483]]
    assert rough["regime_prob_latent"]["confusion"].tolist() == [[1394, 453], [572, 1581]]
    assert rough["regime_prob_spot"]["brier"] == pytest.approx(0.4385066651896362, abs=1e-9)
    assert rough["regime_prob_rv"]["brier"] == pytest.approx(0.39412496974823874, abs=1e-9)
    assert rough["regime_prob_latent"]["brier"] == pytest.approx(0.3554459765106517, abs=1e-9)


def _score_draw(nus, H, n_paths):
    spec = RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": float(nu)} for nu in nus],
        p0=np.array([0.7, 0.3]),
        switch="nu",
    )
    model = RegimeSwitchingBergomi(spec, xi0=0.04, rho=-0.4, H=H, n_mc=2)
    times = np.linspace(0.0, 1.0, 21)
    paths = model.simulate_market(n_paths, times, seed=run_seed(12, 2))
    labels = paths.regime[:, 10]
    return {name: regime_scores(paths.state[name][:, 10], labels) for name in (
        "regime_prob_spot",
        "regime_prob_rv",
        "regime_prob_latent",
    )}


def _single_regime(nu: float, n_mc: int, **kwargs) -> RegimeSwitchingBergomi:
    spec = RegimeSpec(
        generator=np.zeros((1, 1)),
        params=[{"nu": nu}],
        p0=np.array([1.0]),
        switch="nu",
    )
    values = {"xi0": 0.04, "rho": -0.4, "H": 0.5, "n_mc": n_mc}
    values.update(kwargs)
    return RegimeSwitchingBergomi(spec, **values)


def _quoted_paths(v0: float, n_steps: int, prob: np.ndarray, entropy: int = 7) -> MarketPaths:
    """One clock at time 0, variance ``v0``, filter weights ``prob``."""
    times = np.linspace(0.0, 1.0, n_steps + 1)
    weights = np.asarray(prob, dtype=float)
    regime_prob = np.zeros((1, times.size, weights.size))
    regime_prob[:, 0, :] = weights
    variance = np.full((1, times.size), float(v0))
    return MarketPaths(
        times=times,
        Z=np.ones((1, times.size)),
        state={"v": variance, "xi_tenor": times.copy(), "dW": np.zeros((1, n_steps))},
        F0=lambda t: np.full(np.shape(np.asarray(t, dtype=float)), 100.0),
        df=lambda t: np.ones(np.shape(np.asarray(t, dtype=float))),
        regime=np.zeros((1, times.size), dtype=np.int8),
        regime_prob=regime_prob,
        meta={"seed_entropy": entropy, "spawn_key": (2,)},
    )


def _last_smile(model: RegimeSwitchingBergomi, paths: MarketPaths) -> np.ndarray:
    """Implied vol of the last expiry on the unit forward."""
    batch = model.surface_at(paths, 0)
    tau = float(batch.T[-1] - paths.times[0])
    return bs.implied_vol(batch.call[0, -1], 1.0, np.exp(batch.k), tau)


def test_surface_mixes_the_filter_and_repeats():
    """A convex combination of regime probabilities is the same combination of prices."""
    low = _model(n_mc=32)
    times = np.linspace(0.0, 0.25, 5)
    paths = low.simulate_market(4, times, seed=run_seed(9, 2))
    pure = []
    for regime in (0, 1):
        weights = np.zeros_like(paths.regime_prob)
        weights[:, :, regime] = 1.0
        pure.append(low.surface_at(replace(paths, regime_prob=weights), 2).call)
    mixed_weights = 0.25 * np.ones_like(paths.regime_prob)
    mixed_weights[:, :, 0] = 0.25
    mixed_weights[:, :, 1] = 0.75
    mixed = low.surface_at(replace(paths, regime_prob=mixed_weights), 2).call
    assert mixed == pytest.approx(0.25 * pure[0] + 0.75 * pure[1], rel=0, abs=1e-12)
    assert np.array_equal(mixed, low.surface_at(replace(paths, regime_prob=mixed_weights), 2).call)


def test_surface_seed_is_common_across_hedge_dates():
    """The clock is not in the seed. Two dates share shocks. Two regimes do not."""
    times = np.linspace(0.0, 0.4, 5)
    paths = _model().simulate_market(2, times, seed=run_seed(3, 2))
    early = _surface_seed(paths, 1, 0)
    late = _surface_seed(paths, 3, 0)
    assert early.entropy == late.entropy
    assert tuple(early.spawn_key) == tuple(late.spawn_key)
    other = _surface_seed(paths, 1, 1)
    assert tuple(other.spawn_key) != tuple(early.spawn_key)


def test_half_smile_is_not_flat_and_moves_with_nu():
    """H = 1/2, one path, 32 steps, 8,192 Monte Carlo copies, seed entropy 7.

    The implied-vol span at nu = 1.5 is 0.04358, against 0.000383 at nu = 0.01.
    With rho = -0.4 the low-strike implied vol is 0.2107 and the high-strike
    implied vol is 0.1676. The control removes the Monte Carlo smile that a
    raw mean left at nu = 0.01.
    """
    paths = _quoted_paths(0.04, 32, np.array([1.0]))
    calm = _last_smile(_single_regime(0.01, 8192), paths)
    moved = _last_smile(_single_regime(1.5, 8192), paths)
    assert np.isfinite(calm).all() and np.isfinite(moved).all()
    calm_span = float(calm.max() - calm.min())
    moved_span = float(moved.max() - moved.min())
    assert moved_span > 0.03
    assert moved_span > calm_span + 0.02
    assert moved[0] > moved[-1]


def test_half_surface_does_not_call_the_plugin(monkeypatch):
    def absent(name, **kwargs):
        raise AssertionError(name)

    monkeypatch.setattr("volsurface.markets.regime.create", absent)
    paths = _quoted_paths(0.04, 4, np.array([1.0]))
    batch = _single_regime(0.4, 8).surface_at(paths, 0)
    assert np.isfinite(batch.call).all()


def test_rough_surface_ignores_the_label_and_responds_to_nu():
    """H = 0.1 restarts the hybrid factor on xi_t(u). 16 copies, 4 steps."""
    pytest.importorskip("roughvol")
    paths = _quoted_paths(0.04, 4, np.array([1.0]), entropy=11)
    calm = _single_regime(0.2, 16, H=0.1).surface_at(paths, 0)
    moved = _single_regime(1.5, 16, H=0.1).surface_at(paths, 0)
    flipped = replace(paths, regime=np.ones_like(paths.regime))
    assert np.array_equal(calm.call, _single_regime(0.2, 16, H=0.1).surface_at(flipped, 0).call)
    assert np.isfinite(calm.call).all()
    assert not np.allclose(calm.call, moved.call)


def test_rough_surface_requires_the_forward_driver():
    """Without ``dW`` the rough surface would anchor on flat ``v_t`` and drop ``xi_t(u)``."""
    pytest.importorskip("roughvol")
    paths = _quoted_paths(0.04, 4, np.array([1.0]))
    state = dict(paths.state)
    del state["dW"]
    with pytest.raises(ValueError, match="dW"):
        _single_regime(0.4, 8, H=0.1).surface_at(replace(paths, state=state), 0)


def test_surface_standard_error_is_below_the_raw_monte_carlo_noise():
    """One path, nu = 1.1, H = 1/2, 32 steps, 1,024 copies, seed entropy 7.

    The at-the-money relative standard error at expiry 1 is 0.0207. The raw
    mean at 256 copies was about 8 percent of the price. The coefficient on
    each half of the copies is fit on the other half.
    """
    paths = _quoted_paths(0.04, 32, np.array([1.0]))
    batch = _single_regime(1.1, 1024).surface_at(paths, 0)
    assert batch.stderr.shape == batch.call.shape
    assert np.isfinite(batch.stderr).all()
    assert np.all(batch.stderr >= 0.0)
    atm = int(np.argmin(np.abs(batch.k)))
    relative = float(batch.stderr[0, -1, atm] / batch.call[0, -1, atm])
    assert relative < 0.025
    assert batch.total_variance.shape == (1, batch.T.size)
    assert np.isfinite(batch.total_variance_se).all()


def _otm_variance(call: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Total variance from out-of-the-money options. ``call`` has shape ``(m, n_k)``.

    Put-call parity builds the put. A call a fraction under intrinsic makes
    that put negative, and the weight ``1/K^2`` turns the wing into a large
    negative contribution, so the out-of-the-money price is clipped at 0.
    On a uniform log-strike grid the trapezium rule overstates Black's total
    variance by ``dk^2 / 6``.
    """
    strikes = np.exp(k)
    put = call - (1.0 - strikes)
    otm = np.maximum(np.where(strikes < 1.0, put, call), 0.0)
    integrand = 2.0 * otm * np.exp(-k)
    bits = (integrand[:, 1:] + integrand[:, :-1]) * np.diff(k) / 2.0
    dk = float(k[1] - k[0])
    return bits.sum(axis=1) - dk**2 / 6.0


def test_rough_surface_variance_swap_matches_i2_within_mc_error():
    """H = 0.1, eight paths, clock 0.5, maturity 1, 4,096 copies.

    The flat anchor ``v_t (T - t)`` misses the variance swap by a median
    relative gap of 0.281 on the last expiry. The surface log contract
    ``-2 E[log Z]`` stays inside 4 standard errors of that swap on every
    expiry, with an extra ``1e-8`` for the one-step expiry, whose control
    residual is numerical noise. The largest absolute z-score on a later
    expiry is 1.80. The out-of-the-money strip, on log-moneyness from -0.8
    to 0.8, stays inside 4 standard errors of the same swap on the last
    expiry. Its largest absolute z-score is 1.56.
    """
    pytest.importorskip("roughvol")
    smile = np.linspace(-0.8, 0.8, 161)
    model = _model(H=0.1, n_mc=4096, smile_k=smile)
    times = np.linspace(0.0, 1.0, 17)
    paths = model.simulate_market(8, times, seed=run_seed(21, 2))
    clock = 8
    batch = model.surface_at(paths, clock)
    contracts = [Contract(Vanilla(model.forward), T=float(expiry)) for expiry in batch.T]
    i2 = VarianceSwapPricer().price_along(paths, contracts)[:, clock, :]
    gap = np.abs(batch.total_variance - i2)
    assert np.all(gap <= 4.0 * batch.total_variance_se + 1e-8)
    last = i2[:, -1]
    flat = paths.state["v"][:, clock] * (float(batch.T[-1]) - float(times[clock]))
    assert float(np.median(np.abs(flat - last) / last)) > 0.20
    strip = _otm_variance(batch.call[:, -1, :], batch.k)
    assert np.all(np.abs(strip - last) <= 4.0 * batch.total_variance_se[:, -1])
