"""Market paths, pricers, and the nu-only regime chain. DESIGN sections 21 and 22."""

import gc
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
from volsurface.markets.pricers import (
    ClosedFormPricer,
    SurrogatePricer,
    VarianceSwapPricer,
    integrate_forward_variance,
)
from volsurface.markets.regime import (
    RegimeSwitchingBergomi,
    _bridge_log_returns,
    _draw_paths,
    _expiry_indices,
    _future_nu,
    _kernel_weights,
    _log_variance_increments,
    _step_compensator,
    _stochastic,
    _surface_seed,
    bergomi_forward_variance,
    filter_differenced_realised_variance,
    filter_log_realised_variance,
    filter_log_variance,
    hybrid_forward_variance,
    hybrid_increment_variance,
    hybrid_path_forward_variance,
    hybrid_tail_mark_bias,
    increment_variance,
    trader_forward_variance,
)
from volsurface.markets.seeds import DRAW_ORDER, run_seed, stream_seeds
from volsurface.mc.rng import generator
from volsurface.mc.zrule import check_z, max_abs_bound, mean_abs_bound
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
    components = np.zeros((6, times.size - 1, 2))
    expected_xi = hybrid_path_forward_variance(H, nu_step, model.xi0, times, components)
    diagonal = np.arange(times.size)
    expected_xi[:, diagonal, diagonal] = paths.state["v"]
    np.testing.assert_allclose(paths.state["xi"], expected_xi, rtol=1e-12, equal_nan=True)
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


def test_hybrid_forward_diagonal_is_the_simulated_variance():
    """The clock entry is the simulated variance.

    At constant ``nu`` and ``kappa`` 1 the next tenor matches the closed
    form. On a path whose ``nu`` has switched, the next tenor uses the
    ``nu`` in force on each past step. It is not the mixture of the two
    constant-``nu`` closed forms. Rebuilding the next variance as that
    tenor times the near-field factor is a code-consistency identity.
    It checks that the curve and ``v`` share one driver. It is not a
    martingale check. Given the true ``nu`` path, the martingale holds
    up to the tail-weight bias.
    """
    pytest.importorskip("roughvol")
    nu = 1.5
    model = _constant_nu(nu, H=0.1, n_sub=1, n_pilot=2)
    times = np.linspace(0.0, 1.0, 9)
    drawn = _draw_paths(model, 4, times, run_seed(3, 0))
    curve = hybrid_forward_variance(0.1, nu, model.xi0, times, drawn["components"])
    diagonal = np.diagonal(curve, axis1=1, axis2=2)
    assert diagonal == pytest.approx(drawn["v"], rel=1e-10, abs=1e-12)
    closed = bergomi_forward_variance(0.1, nu, model.xi0, times, drawn["dW"])
    assert curve[:, 3, 4] == pytest.approx(closed[:, 3, 4], rel=1e-10, abs=1e-12)

    switching = RegimeSwitchingBergomi(
        _prereg_spec(), xi0=0.04, rho=-0.4, H=0.1, n_mc=2, n_sub=1, n_pilot=2
    )
    grid = np.linspace(0.0, 0.5, 22)
    drawn_s = _draw_paths(switching, 64, grid, run_seed(3, 1))
    nus = np.array([1.5, 3.0])
    nu_step = nus[drawn_s["regime"][:, :-1]]
    path_curve = hybrid_path_forward_variance(
        0.1, nu_step, switching.xi0, grid, drawn_s["components"]
    )
    clock = 10
    assert np.diagonal(path_curve, axis1=1, axis2=2) == pytest.approx(
        drawn_s["v"], rel=1e-10, abs=1e-12
    )
    probs = np.full((64, grid.size, 2), 0.5)
    mixture = trader_forward_variance(
        0.1, nus, drawn_s["v"], probs, switching.xi0, grid, drawn_s["dW"]
    )
    switched = np.any(drawn_s["regime"][:, : clock + 1] != drawn_s["regime"][:, :1], axis=1)
    assert switched.any()
    assert not np.allclose(
        path_curve[switched, clock, clock + 1],
        mixture[switched, clock, clock + 1],
        rtol=1e-6,
        atol=1e-8,
    )
    hurst = 0.1
    scale = math.sqrt(2.0 * hurst)
    dt = float(grid[clock + 1] - grid[clock])
    factor = np.exp(
        scale * nu_step[:, clock] * drawn_s["components"][:, clock, 1]
        - 0.5 * nu_step[:, clock] ** 2 * dt ** (2.0 * hurst)
    )
    rebuilt = path_curve[:, clock, clock + 1] * factor
    assert rebuilt == pytest.approx(drawn_s["v"][:, clock + 1], rel=1e-10, abs=1e-12)


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


def test_call_reads_the_true_regime_and_the_swap_does_not():
    """The stored variance swap ignores a flipped label. The call does not.

    ``state["xi"]`` was built at simulate time. Flipping ``paths.regime``
    does not rebuild it. The future factor of the call uses the label at
    the clock, so the price moves. ``regime_prob`` is not a weight.
    """
    model = _model()
    times = np.linspace(0.0, 1.0, 11)
    paths = model.simulate_market(32, times, seed=run_seed(8, 0))
    flipped = replace(paths, regime=(1 - paths.regime).astype(np.int8))
    assert not np.allclose(model.surface_at(paths, 4).call, model.surface_at(flipped, 4).call)
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


def test_prices_do_not_read_the_filter():
    """The call reads the true regime. It does not read a probability array.

    The windowed filter and the latent-variance benchmark are stored beside
    the spot filter. Replacing those arrays, or reversing ``regime_prob``,
    leaves the call unchanged. Replacing the latent regime moves it. The
    market does not call the exact half-Brownian likelihood.
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
    unchanged = model.surface_at(replace(paths, regime_prob=flipped), 2).call
    assert np.array_equal(left, unchanged)
    relabelled = replace(paths, regime=(1 - paths.regime).astype(np.int8))
    assert not np.allclose(left, model.surface_at(relabelled, 2).call)


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

    The settings are ``H = 1/2`` and ``nu`` of 0.4 and 1.1. The call is
    the informed price: the true regime's ``nu``, and a variance level
    flat at ``v_t``. The base rate is the larger share of the latent
    regime at that clock, 0.5383. Eight quantile bins of the call and of
    the variance swap each contain both regimes. On this draw both modal
    accuracies are 0.66, a lift of 0.122 over the base rate, inside
    (0.05, 0.35). At these settings the informed call is not a readout
    of the label. The spot filter's bar is the Brier skill against the
    constant forecast that quotes the sample shares. That skill is not
    positive: one-step log realised variance has one mean in both
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


def test_market_test_filters_are_scored_against_the_base_rate():
    """H = 1/2, nu 0.4 and 1.1. Integer counts are the pin. Floats sit at 1e-9.

    The spot filter has no Brier skill. The differenced windowed filter is
    right on 345 of 600 paths. Its Brier skill is 0.0035. The latent
    benchmark is the pilot table, right on 487 of 600 paths. The call
    reads the true regime, not these probabilities. At ``H = 1/2`` the
    variance-swap curve does not depend on the weights.
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
    assert float(times[10]) == pytest.approx(0.5)
    nus = np.array([0.4, 1.1])
    v = paths.state["v"]
    dW = paths.state["dW"]
    left = trader_forward_variance(0.5, nus, v, paths.state["regime_prob_spot"], 0.04, times, dW)
    right = trader_forward_variance(0.5, nus, v, paths.state["regime_prob_rv"], 0.04, times, dW)
    assert np.allclose(left, right, equal_nan=True)


def test_preregistered_filters_are_scored_against_the_base_rate(monkeypatch):
    """H = 0.1, nu 1.5 and 3. The spot hit rate is 410 of 600.

    That is the measured rate at these settings. The windowed filter and
    the latent benchmark both beat the base-rate forecast.     The benchmark
    is the pilot table: the market does not call the half-Brownian
    likelihood. The call reads the true regime, not the spot filter.
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


def test_surface_reads_the_regime_and_repeats():
    """Replacing the filter leaves the call unchanged. The label moves it.

    A second call on the same paths matches the first. Forcing every path
    into regime 0 prices a different book from forcing regime 1.
    """
    model = _model(n_mc=32)
    times = np.linspace(0.0, 0.25, 5)
    paths = model.simulate_market(4, times, seed=run_seed(9, 2))
    first = model.surface_at(paths, 2).call
    mixed_weights = np.zeros_like(paths.regime_prob)
    mixed_weights[:, :, 0] = 0.25
    mixed_weights[:, :, 1] = 0.75
    mixed = model.surface_at(replace(paths, regime_prob=mixed_weights), 2).call
    assert np.array_equal(first, mixed)
    assert np.array_equal(first, model.surface_at(paths, 2).call)
    low = np.zeros_like(paths.regime)
    high = np.ones_like(paths.regime)
    priced_low = model.surface_at(replace(paths, regime=low), 2).call
    priced_high = model.surface_at(replace(paths, regime=high), 2).call
    assert not np.allclose(priced_low, priced_high)


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


def test_rough_surface_reads_the_label_and_responds_to_nu():
    """H = 0.1. The future factor uses the label. A larger nu moves the smile.

    One regime in the spec cannot take a label of 1. That raises. With
    two regimes, the same path priced at label 0 and at label 1 differs.
    """
    pytest.importorskip("roughvol")
    paths = _quoted_paths(0.04, 4, np.array([1.0]), entropy=11)
    calm = _single_regime(0.2, 16, H=0.1).surface_at(paths, 0)
    moved = _single_regime(1.5, 16, H=0.1).surface_at(paths, 0)
    flipped = replace(paths, regime=np.ones_like(paths.regime))
    with pytest.raises(ValueError, match="regime"):
        _single_regime(0.2, 16, H=0.1).surface_at(flipped, 0)
    two = _model(H=0.1, n_mc=16)
    assert np.isfinite(calm.call).all()
    assert not np.allclose(calm.call, moved.call)
    assert not np.allclose(two.surface_at(paths, 0).call, two.surface_at(flipped, 0).call)


def test_rough_surface_requires_the_forward_driver():
    """Without ``dW`` and without ``state["xi"]`` the restarted curve cannot be built."""
    pytest.importorskip("roughvol")
    paths = _quoted_paths(0.04, 4, np.array([1.0]))
    state = dict(paths.state)
    del state["dW"]
    with pytest.raises(ValueError, match="dW"):
        _single_regime(0.4, 8, H=0.1).surface_at(replace(paths, state=state), 0)


def test_forward_copies_leave_the_clock_regime():
    """The forward chain uses the same generator as the simulator.

    Four thousand copies start in the low regime. The first column stays
    there. The next column matches ``expm(Q dt)`` within 4 standard errors.
    Later columns do reach the high vol of vol: a frozen ``nu`` would not.
    """
    nus = np.array([1.5, 3.0])
    generator_q = np.array([[-1.5, 1.5], [0.8, -0.8]])
    dt = 1.0 / 252.0
    n_steps = 21
    grid = np.arange(n_steps + 1) * dt
    n_sim = 4000
    start = np.zeros(n_sim, dtype=int)
    nu = _future_nu(nus, generator_q, start, grid, generator(1))
    assert nu.shape == (n_sim, n_steps)
    assert np.all(nu[:, 0] == 1.5)
    stay = float(expm(generator_q * dt)[0, 0])
    got = float(np.mean(nu[:, 1] == 1.5))
    se = math.sqrt(stay * (1.0 - stay) / n_sim)
    assert abs(got - stay) < 4.0 * se
    assert float(np.mean(nu[:, 1:] == 3.0)) > 0.02


def test_surface_copies_switch_off_the_clock_regime(monkeypatch):
    """``surface_at`` draws the forward chain from the label at the clock."""
    seen = {}
    real = _future_nu

    def wrap(*args, **kwargs):
        out = real(*args, **kwargs)
        seen["nu"] = out
        return out

    monkeypatch.setattr("volsurface.markets.regime._future_nu", wrap)
    model = _model(n_mc=8)
    times = np.linspace(0.0, 0.2, 11)
    paths = model.simulate_market(20, times, seed=run_seed(5, 2))
    regime = np.array(paths.regime, copy=True)
    regime[:, 2] = 0
    model.surface_at(replace(paths, regime=regime), 2)
    nu = seen["nu"]
    assert np.all(nu[:, 0] == 0.4)
    assert np.any(nu[:, 1:] == 1.1)


def test_gate_mode_requires_the_stored_hybrid_curve():
    """Gate mode is ``price_filter="rv"`` with ``H`` other than 1/2.

    A missing ``state["xi"]`` raises. The spot filter still builds the
    restarted curve from ``dW``, which is the quoted-path fallback.
    """
    pytest.importorskip("roughvol")
    model = RegimeSwitchingBergomi(
        _prereg_spec(),
        xi0=0.04,
        rho=-0.4,
        H=0.1,
        n_mc=4,
        n_sub=1,
        n_pilot=2,
        price_filter="rv",
        smile_k=np.array([0.0]),
    )
    times = np.linspace(0.0, 0.1, 5)
    missing = model.simulate_market(4, times, seed=run_seed(3, 2), store_forward=False)
    assert "xi" not in missing.state
    with pytest.raises(ValueError, match="xi"):
        model.surface_at(missing, 1)
    stored = model.simulate_market(4, times, seed=run_seed(3, 2))
    assert np.isfinite(model.surface_at(stored, 1).call).all()


def _coupled_replay(model, drawn, clock: int) -> tuple[np.ndarray, np.ndarray]:
    """Copy map fed this path's own future pieces. Returns variance and log Z.

    The variance row is the copy's left variance, including the terminal
    level. ``log Z`` is the sum of log increments from the clock, the same
    shock the simulator stored.
    """
    grid = np.asarray(drawn["grid"], dtype=float)
    regime = drawn["regime"]
    nu_step = drawn["nus"][regime[:, :-1]]
    v = drawn["v"]
    perp = drawn["perp"]
    rho = float(model.rho)
    ortho = math.sqrt(max(0.0, 1.0 - rho * rho))
    n_times = grid.size - clock
    log_z = np.zeros((v.shape[0], n_times))
    if model.H == 0.5:
        dW = drawn["dW"]
        variance = np.empty((v.shape[0], n_times))
        multiplier = np.ones(v.shape[0])
        variance[:, 0] = v[:, clock]
        for step in range(n_times - 1):
            dt = float(grid[clock + step + 1] - grid[clock + step])
            shock_w = dW[:, clock + step]
            nu = nu_step[:, clock + step]
            left = v[:, clock] * multiplier
            shock = rho * shock_w + ortho * math.sqrt(dt) * perp[:, clock + step]
            increment = np.sqrt(np.maximum(left, 0.0)) * shock - 0.5 * left * dt
            log_z[:, step + 1] = log_z[:, step] + increment
            multiplier = multiplier * np.exp(nu * shock_w - 0.5 * nu * nu * dt)
            variance[:, step + 1] = v[:, clock] * multiplier
        return variance, log_z
    components = drawn["components"]
    xi = hybrid_path_forward_variance(model.H, nu_step, model.xi0, grid, components)
    diagonal = np.arange(grid.size)
    xi[:, diagonal, diagonal] = v
    short = grid[clock:] - float(grid[clock])
    pieces = components[:, clock:, :]
    factor = hybrid_increment_variance(model.H, nu_step[:, clock:], 1.0, short, pieces)
    variance = xi[:, clock, clock:] * factor
    for step in range(n_times - 1):
        dt = float(short[step + 1] - short[step])
        shock_w = pieces[:, step, 0]
        shock = rho * shock_w + ortho * math.sqrt(dt) * perp[:, clock + step]
        left = np.maximum(variance[:, step], 0.0)
        log_z[:, step + 1] = log_z[:, step] + np.sqrt(left) * shock - 0.5 * left * dt
    return variance, log_z


def _coupled_gaps(model, drawn, clock: int) -> tuple[float, float]:
    variance, log_z = _coupled_replay(model, drawn, clock)
    v_gap = float(np.max(np.abs(variance - drawn["v"][:, clock:])))
    spot = np.log(drawn["Z"][:, clock:]) - np.log(drawn["Z"][:, clock])[:, None]
    z_gap = float(np.max(np.abs(log_z - spot)))
    return v_gap, z_gap


def test_copy_fed_the_paths_pieces_reproduces_variance_and_log_spot():
    """Coupling identity. A copy fed the path's own future pieces matches the draw.

    The pieces are the future ``nu`` path, the future hybrid components or
    the future ``dW`` at ``H = 1/2``, the stored orthogonal shock, and the
    forward curve at the clock. Both ``v`` and ``log Z`` from that clock
    match to ``1e-12``. Clocks are the start, the middle, and the last
    step of a 21-day grid, 32 paths, seeds ``run_seed(81, 2)`` and
    ``run_seed(82, 2)``. On this draw the largest absolute gap is
    8.95e-16, on ``log Z`` at ``H = 1/2``.
    """
    pytest.importorskip("roughvol")
    day = 1.0 / 252.0
    times = np.arange(22) * day
    clocks = (0, 10, 20)
    rough = RegimeSwitchingBergomi(
        _prereg_spec(), xi0=0.04, rho=-0.4, H=0.1, n_mc=2, n_sub=1, n_pilot=2
    )
    half = RegimeSwitchingBergomi(
        _prereg_spec(), xi0=0.04, rho=-0.4, H=0.5, n_mc=2, n_sub=1, n_pilot=2
    )
    drawn_rough = _draw_paths(rough, 32, times, run_seed(81, 2))
    drawn_half = _draw_paths(half, 32, times, run_seed(82, 2))
    for clock in clocks:
        rough_v, rough_z = _coupled_gaps(rough, drawn_rough, clock)
        half_v, half_z = _coupled_gaps(half, drawn_half, clock)
        assert rough_v < 1e-12 and rough_z < 1e-12, f"rough clock {clock} {rough_v} {rough_z}"
        assert half_v < 1e-12 and half_z < 1e-12, f"half clock {clock} {half_v} {half_z}"


def test_copy_chain_switch_rate_matches_expm():
    """One-step regime rates match ``expm(Q dt)`` under the z-rule.

    Eight replications. The copy chain is 4,000 paths from each starting
    regime, ``dt = 1/252``. The simulator chain is 2,000 paths at
    ``H = 1/2``, split by the regime at the left endpoint. The probability
    is the high-regime entry of ``_stochastic(expm(Q dt))``. That matrix
    matches ``expm(Q dt)`` to ``1e-12``. Seeds are ``run_seed(80 + r, 2)``
    for the copy and ``run_seed(90 + r, 2)`` for the simulator. On this
    draw the copy's largest absolute z is 2.07 and the simulator's is 2.39.
    """
    nus = np.array([1.5, 3.0])
    generator_q = np.array([[-1.5, 1.5], [0.8, -0.8]])
    dt = 1.0 / 252.0
    raw = expm(generator_q * dt)
    step = _stochastic(raw)
    assert float(np.max(np.abs(raw - step))) < 1e-12
    grid = np.array([0.0, dt, 2.0 * dt])
    n_sim = 4000
    copy_scores = []
    for replication in range(8):
        rng = generator(run_seed(80 + replication, 2))
        for row in (0, 1):
            start = np.full(n_sim, row, dtype=int)
            nu = _future_nu(nus, generator_q, start, grid, rng)
            freq = float(np.mean(nu[:, 1] == nus[1]))
            probability = float(step[row, 1])
            se = math.sqrt(probability * (1.0 - probability) / n_sim)
            copy_scores.append((freq - probability) / se)
    copy_z = np.asarray(copy_scores)
    failed = check_z(copy_z)
    if failed is not None:
        raise AssertionError(failed)
    model = RegimeSwitchingBergomi(
        _prereg_spec(), xi0=0.04, rho=-0.4, H=0.5, n_mc=2, n_sub=1, n_pilot=2
    )
    sim_scores = []
    for replication in range(8):
        drawn = _draw_paths(model, 2000, np.array([0.0, dt]), run_seed(90 + replication, 2))
        regime = drawn["regime"]
        for row in (0, 1):
            mask = regime[:, 0] == row
            count = int(mask.sum())
            assert count > 100
            freq = float(np.mean(regime[mask, 1] == 1))
            probability = float(step[row, 1])
            se = math.sqrt(probability * (1.0 - probability) / count)
            sim_scores.append((freq - probability) / se)
    sim_z = np.asarray(sim_scores)
    failed = check_z(sim_z)
    if failed is not None:
        raise AssertionError(failed)


def test_gate_surface_defaults_to_standardised_moneyness():
    """Gate settings, ``smile_k`` omitted, ``n_mc`` 1024, ``T`` of 21 trading days.

    The default strike grid is ``k = z * sqrt(xi0 * tau)`` with ``z`` from
    -2.5 to 2.5 and ``tau`` the time from the clock to the last expiry.
    An explicit ``smile_k`` is left as given. The spot-filter default stays
    nine strikes from -0.2 to 0.2. The out-of-the-money exact-zero rate is
    the share of last-expiry call prices with ``k > 0`` that are exactly 0.
    On eight paths, seed ``run_seed(83, 2)``, that rate at 21 days is 0
    on 80 prices. The same strikes on the one-day node are far out for
    that maturity, and 63 of those 80 prices are exactly zero.
    """
    pytest.importorskip("roughvol")
    spot = _model()
    assert np.allclose(spot.smile_k, np.linspace(-0.2, 0.2, 9))
    short = np.linspace(0.0, 0.2, 5)
    spot_paths = spot.simulate_market(2, short, seed=run_seed(4, 2))
    assert np.allclose(spot.surface_at(spot_paths, 0).k, np.linspace(-0.2, 0.2, 9))
    explicit = RegimeSwitchingBergomi(
        _prereg_spec(),
        xi0=0.04,
        rho=-0.4,
        H=0.1,
        n_mc=4,
        n_sub=1,
        n_pilot=2,
        price_filter="rv",
        smile_k=np.array([0.0]),
    )
    explicit_paths = explicit.simulate_market(2, short, seed=run_seed(4, 2))
    assert np.allclose(explicit.surface_at(explicit_paths, 1).k, np.array([0.0]))
    model = RegimeSwitchingBergomi(
        _prereg_spec(),
        xi0=0.04,
        rho=-0.4,
        H=0.1,
        n_mc=1024,
        n_sub=21,
        n_pilot=1024,
        price_filter="rv",
    )
    day = 1.0 / 252.0
    times = np.arange(22) * day
    paths = model.simulate_market(8, times, seed=run_seed(83, 2))
    batch = model.surface_at(paths, 0)
    tau = float(times[-1])
    smile = np.linspace(-2.5, 2.5, 21) * math.sqrt(0.04 * tau)
    assert np.allclose(batch.k, smile)
    wing = batch.call[:, -1, batch.k > 0.0]
    zero_rate = float(np.mean(wing == 0.0))
    assert zero_rate < 0.05, f"otm call zero rate {zero_rate:.4f} on {wing.size} prices"


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

    Strikes are standardised moneyness on the last expiry,
    ``k = z * sqrt(xi0) * sqrt(tau)`` with ``z`` from -2.5 to 2.5. The
    call is priced on the true regime, the forward chain, and
    ``state["xi"]``. The one-step expiry is an identity: the factor's
    first column is 1, so the log-contract gap is under ``1e-12``. The
    other six expiries are 48 comparisons. ``max |z|`` is below
    ``max_abs_bound(48)`` (4.25) and the mean of the eight path-mean
    z-scores is below ``mean_abs_bound(8)``. Nodes share paths, so the
    binomial fraction band is not applied. On this draw the largest
    absolute z is 2.26 and the path-mean z is 0.039. The flat anchor
    ``v_t (T - t)`` is a non-triviality check: the z-score of its mean
    absolute relative gap, not a median. On this draw that mean is 0.302,
    4.60 standard errors, and the median is 0.290. The out-of-the-money
    strip's eight path z-scores pass the z-rule. Its largest absolute z
    is 0.96. None of the 80 out-of-the-money call prices is exactly zero.
    """
    pytest.importorskip("roughvol")
    times = np.linspace(0.0, 1.0, 17)
    clock = 8
    tau = float(times[-1] - times[clock])
    width = math.sqrt(0.04 * tau)
    smile = np.linspace(-2.5, 2.5, 21) * width
    model = _model(H=0.1, n_mc=4096, smile_k=smile)
    paths = model.simulate_market(8, times, seed=run_seed(21, 2))
    batch = model.surface_at(paths, clock)
    contracts = [Contract(Vanilla(model.forward), T=float(expiry)) for expiry in batch.T]
    swap = VarianceSwapPricer().price_along(paths, contracts)[:, clock, :]
    one_step = np.abs(batch.total_variance[:, 0] - swap[:, 0])
    assert np.all(one_step < 1e-12)
    later = batch.total_variance[:, 1:] - swap[:, 1:]
    later_se = batch.total_variance_se[:, 1:]
    assert later.shape == (8, 6)
    assert np.all(later_se > 0.0)
    scores = later / later_se
    failed = check_z(scores, mean=False, fraction=False)
    path_mean = scores.mean(axis=1)
    centre = float(path_mean.mean())
    text = (
        f"max |z| {float(np.max(np.abs(scores))):.4f} "
        f"bound {max_abs_bound(scores.size):.4f} "
        f"path-mean z {centre:.4f} bound {mean_abs_bound(path_mean.size):.4f}"
    )
    if failed is not None:
        raise AssertionError(f"{text}; {failed}")
    assert abs(centre) < mean_abs_bound(path_mean.size), text
    last = swap[:, -1]
    flat = paths.state["v"][:, clock] * (float(batch.T[-1]) - float(times[clock]))
    absolute = np.abs((flat - last) / last)
    flat_se = float(absolute.std(ddof=1) / math.sqrt(absolute.size))
    flat_z = float(absolute.mean() / flat_se)
    assert flat_z > 2.0, f"flat-anchor mean {float(absolute.mean()):.4f} z {flat_z:.2f}"
    strip = _otm_variance(batch.call[:, -1, :], batch.k)
    strip_se = batch.total_variance_se[:, -1]
    assert np.all(strip_se > 0.0)
    strip_failed = check_z((strip - last) / strip_se)
    if strip_failed is not None:
        raise AssertionError(strip_failed)
    otm = batch.k > 0.0
    wing = batch.call[:, -1, otm]
    zero_rate = float(np.mean(wing == 0.0))
    assert zero_rate < 0.05, f"otm call zero rate {zero_rate:.4f}"


def _constant_nu(nu: float, **kwargs) -> RegimeSwitchingBergomi:
    spec = RegimeSpec(
        generator=np.zeros((1, 1)),
        params=[{"nu": nu}],
        p0=np.array([1.0]),
        switch="nu",
    )
    values = {"xi0": 0.04, "rho": -0.4, "H": 0.1, "n_mc": 2, "n_sub": 1, "n_pilot": 2}
    values.update(kwargs)
    return RegimeSwitchingBergomi(spec, **values)


def _prereg_spec() -> RegimeSpec:
    return RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": 1.5}, {"nu": 3.0}],
        p0=np.array([0.7, 0.3]),
        switch="nu",
    )


def test_log_variance_mean_matches_the_compensator():
    """Eight runs, 2,000 paths, H = 0.1, T = 1. The z-rule is the threshold.

    Constant nu. The target is ``log xi0 - nu^2 T^{2H} / 2``. At n = 8 the
    max-|z| bound is 3.836 and the mean bound is 1.237. On seeds
    ``run_seed(70 + r, 2)`` the largest absolute z is 1.49. The two vols of
    vol share that draw because the driver is linear in nu.
    """
    pytest.importorskip("roughvol")
    times = np.linspace(0.0, 1.0, 11)
    for nu in (1.5, 3.0):
        model = _constant_nu(nu)
        target = math.log(model.xi0) - 0.5 * nu * nu * float(times[-1]) ** (2.0 * model.H)
        log_scores = []
        for replication in range(8):
            paths = model.simulate_market(2000, times, seed=run_seed(70 + replication, 2))
            log_v = np.log(paths.state["v"][:, -1])
            se = log_v.std(ddof=1) / math.sqrt(log_v.size)
            log_scores.append((log_v.mean() - target) / se)
        assert check_z(np.asarray(log_scores)) is None


def test_switching_log_variance_mean_matches_the_path_compensator():
    """Eight runs, 2,000 paths, the pre-registered switching market, T = 1.

    The target on each path is ``log xi0`` minus that path's compensator.
    The largest absolute z on seeds ``run_seed(70 + r, 2)`` is 1.74.
    """
    pytest.importorskip("roughvol")
    model = RegimeSwitchingBergomi(
        _prereg_spec(), xi0=0.04, rho=-0.4, H=0.1, n_mc=2, n_sub=1, n_pilot=2
    )
    times = np.linspace(0.0, 1.0, 11)
    nus = np.array([1.5, 3.0])
    scores = []
    for replication in range(8):
        paths = model.simulate_market(2000, times, seed=run_seed(70 + replication, 2))
        nu_step = nus[paths.regime[:, :-1]]
        compensator = _step_compensator(model.H, nu_step, times)[:, -1]
        gap = np.log(paths.state["v"][:, -1]) - (math.log(model.xi0) - compensator)
        se = gap.std(ddof=1) / math.sqrt(gap.size)
        scores.append(gap.mean() / se)
    assert check_z(np.asarray(scores)) is None


def test_episode_grid_log_variance_mean_matches_the_compensator():
    """Episode grid, 157 daily times, eight runs of 2,000 paths, ``H = 0.1``.

    ``T = 156/252``, from time 0 through ``t0`` plus 30 trading days. The
    forward curve is not stored. Constant ``nu`` of 1.5 and of 3 share
    z-scores, because the driver is linear in ``nu``. On seeds
    ``run_seed(70 + r, 2)`` the largest absolute z is 1.46 and the mean z
    is 0.203. The switching market has largest absolute z 1.04 and mean
    z 0.032. Both pass the z-rule.
    """
    pytest.importorskip("roughvol")
    times = np.arange(126 + 30 + 1) / 252.0
    maturity = float(times[-1])
    constant_scores = None
    for nu in (1.5, 3.0):
        model = _constant_nu(nu, n_sub=1, n_pilot=2)
        target = math.log(model.xi0) - 0.5 * nu * nu * maturity ** (2.0 * model.H)
        scores = []
        for replication in range(8):
            paths = model.simulate_market(
                2000, times, seed=run_seed(70 + replication, 2), store_forward=False
            )
            log_v = np.log(paths.state["v"][:, -1])
            se = log_v.std(ddof=1) / math.sqrt(log_v.size)
            scores.append((log_v.mean() - target) / se)
        got = np.asarray(scores)
        if constant_scores is None:
            constant_scores = got
        else:
            assert np.allclose(got, constant_scores)
        assert check_z(got) is None
    model = RegimeSwitchingBergomi(
        _prereg_spec(), xi0=0.04, rho=-0.4, H=0.1, n_mc=2, n_sub=1, n_pilot=2
    )
    nus = np.array([1.5, 3.0])
    switching = []
    for replication in range(8):
        paths = model.simulate_market(
            2000, times, seed=run_seed(70 + replication, 2), store_forward=False
        )
        nu_step = nus[paths.regime[:, :-1]]
        compensator = _step_compensator(model.H, nu_step, times)[:, -1]
        gap = np.log(paths.state["v"][:, -1]) - (math.log(model.xi0) - compensator)
        se = gap.std(ddof=1) / math.sqrt(gap.size)
        switching.append(gap.mean() / se)
    assert check_z(np.asarray(switching)) is None


def test_daily_spot_is_a_martingale_and_has_no_time_slope():
    """Eight runs, 2,000 paths, 30 daily steps, H = 0.1. The z-rule is the test.

    The terminal z-score is ``(E[Z_T] - 1) / se`` on the paths. The drift
    regression is weighted least squares of the cross-sectional mean
    increment ``Z_i - Z_{i-1}`` on a constant and on the left calendar
    time. Weights are the reciprocal of that date's sample variance.
    """
    pytest.importorskip("roughvol")
    model = RegimeSwitchingBergomi(
        _prereg_spec(), xi0=0.04, rho=-0.4, H=0.1, n_mc=2, n_sub=1, n_pilot=2
    )
    times = np.arange(31) / 252.0
    terminal = []
    intercept = []
    slope = []
    for replication in range(8):
        paths = model.simulate_market(2000, times, seed=run_seed(40 + replication, 2))
        total = paths.Z[:, -1] - 1.0
        se = total.std(ddof=1) / math.sqrt(total.size)
        terminal.append(total.mean() / se)
        increment = np.diff(paths.Z, axis=1)
        means = increment.mean(axis=0)
        variance = increment.var(axis=0, ddof=1) / increment.shape[0]
        clock = times[:-1]
        design = np.column_stack([np.ones(clock.size), clock])
        weight = 1.0 / variance
        weighted = design * weight[:, None]
        coef = np.linalg.solve(design.T @ weighted, design.T @ (weight * means))
        se_coef = np.sqrt(np.diag(np.linalg.inv(design.T @ weighted)))
        z_coef = coef / se_coef
        intercept.append(z_coef[0])
        slope.append(z_coef[1])
    assert check_z(np.asarray(terminal)) is None
    assert check_z(np.asarray(intercept)) is None
    assert check_z(np.asarray(slope)) is None


def _variance_swap_mark(variance, xi, times, start, maturity):
    """Left-point model variance from ``start``, plus the remaining integral."""
    accrued = np.zeros(variance.shape[0])
    mark = np.empty_like(variance)
    grid = np.asarray(times, dtype=float)
    for i, t in enumerate(grid):
        remaining = integrate_forward_variance(xi[:, i, :], grid, float(t), maturity)
        mark[:, i] = accrued + remaining
        if i < grid.size - 1 and float(t) >= start - 1e-15 and float(t) < maturity - 1e-15:
            left = max(float(t), start)
            right = min(float(grid[i + 1]), maturity)
            accrued = accrued + variance[:, i] * (right - left)
    return mark


def _clustered_slope(increment, probability, mask=None):
    """Slope of the mark change on the probability of the high regime.

    Each row is one path and one cluster. ``mask`` drops observations, in
    row-major order. The scale on the sandwich is
    ``G / (G - 1) * (N - 1) / (N - 2)``. The return is the intercept and
    the slope, then their clustered z-scores.
    """
    y = np.asarray(increment, dtype=float)
    p = np.asarray(probability, dtype=float)
    flat_y = y.reshape(-1)
    flat_p = p.reshape(-1)
    path_id = np.repeat(np.arange(y.shape[0]), y.shape[1])
    if mask is not None:
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        flat_y = flat_y[keep]
        flat_p = flat_p[keep]
        path_id = path_id[keep]
    design = np.column_stack((np.ones(flat_y.size), flat_p))
    xtx = design.T @ design
    beta = np.linalg.solve(xtx, design.T @ flat_y)
    resid = flat_y - design @ beta
    order = np.argsort(path_id, kind="mergesort")
    ordered = path_id[order]
    scores = design[order] * resid[order, None]
    breaks = np.flatnonzero(np.diff(ordered)) + 1
    meat = np.zeros((2, 2))
    groups = 0
    start = 0
    stops = (*breaks, ordered.size)
    for stop in stops:
        total = scores[start:stop].sum(axis=0)
        meat += np.outer(total, total)
        groups += 1
        start = stop
    bread = np.linalg.inv(xtx)
    n = flat_y.size
    scale = (groups / (groups - 1)) * ((n - 1) / (n - 2))
    se = np.sqrt(np.diag(scale * (bread @ meat @ bread)))
    return beta, beta / se


def _drop_largest(increment, fraction):
    """Row-major mask that keeps the smallest ``1 - fraction`` absolute changes."""
    flat = np.abs(np.asarray(increment, dtype=float)).reshape(-1)
    keep_n = int(math.floor((1.0 - fraction) * flat.size))
    order = np.argsort(flat)
    mask = np.zeros(flat.size, dtype=bool)
    mask[order[:keep_n]] = True
    return mask


def _drift_z(increment, times_left):
    """Weighted least squares, dropping a step whose sample variance is noise."""
    variance = increment.var(axis=0, ddof=1)
    keep = variance > 1e-18
    kept = increment[:, keep]
    means = kept.mean(axis=0)
    scale = variance[keep] / kept.shape[0]
    clock = np.asarray(times_left, dtype=float)[keep]
    design = np.column_stack([np.ones(clock.size), clock])
    weight = 1.0 / scale
    weighted = design * weight[:, None]
    coef = np.linalg.solve(design.T @ weighted, design.T @ (weight * means))
    se_coef = np.sqrt(np.diag(np.linalg.inv(design.T @ weighted)))
    return coef / se_coef


# Two-sided 5 percent normal quantile, and the one-sided 80 percent power quantile.
_Z95 = 1.959963984540054
_Z80 = 0.8416212335729143


def _episode_mark_drift(model, n_days):
    """Terminal z-scores and the filter-probability drift of the mark.

    Eight runs of 1,500 paths. The drift is the signed slope of the
    one-step mark change on the probability of the high regime, times the
    number of steps, divided by the mean entry quote. ``se`` is the
    path-clustered standard error of that fraction. The trimmed figures
    drop the largest 1 percent of absolute daily changes. They are a
    diagnostic, and the sign is kept.
    """
    end = 126 + n_days
    times = np.arange(end + 1) / 252.0
    start = float(times[126])
    maturity = float(times[end])
    terminal = []
    intercept = []
    slope = []
    panels = []
    probs = []
    quotes = []
    replication_z = []
    for replication in range(8):
        paths = model.simulate_market(1500, times, seed=run_seed(40 + replication, 2))
        mark = _variance_swap_mark(paths.state["v"], paths.state["xi"], times, start, maturity)
        gap = mark[:, end] - mark[:, 126]
        se = gap.std(ddof=1) / math.sqrt(gap.size)
        terminal.append(gap.mean() / se)
        increment = np.diff(mark[:, 126 : end + 1], axis=1)
        z_coef = _drift_z(increment, times[126:end])
        intercept.append(z_coef[0])
        slope.append(z_coef[1])
        step = increment[:, :-1]
        probability = paths.regime_prob[:, 126 : end - 1, 1]
        _beta, z_filter = _clustered_slope(step, probability)
        replication_z.append(z_filter[1])
        panels.append(step)
        probs.append(probability)
        quotes.append(mark[:, 126])
    panel = np.concatenate(panels)
    probability = np.concatenate(probs)
    quote = float(np.concatenate(quotes).mean())
    beta, z_score = _clustered_slope(panel, probability)
    steps = panel.shape[1]
    z = float(z_score[1])
    drift = float(beta[1]) * steps / quote
    se = float(drift / z) if z != 0.0 else float("inf")
    trimmed_beta, trimmed_z = _clustered_slope(panel, probability, _drop_largest(panel, 0.01))
    trimmed_drift = float(trimmed_beta[1]) * steps / quote
    return {
        "terminal": np.asarray(terminal),
        "intercept": np.asarray(intercept),
        "slope": np.asarray(slope),
        "drift": drift,
        "se": se,
        "z": z,
        "quote": quote,
        "trimmed_drift": trimmed_drift,
        "trimmed_z": float(trimmed_z[1]),
        "replication_z": np.asarray(replication_z),
        "mdd": (_Z95 + _Z80) * se,
    }


def _assert_filter_z(result, label):
    """Untrimmed 95 percent interval contains 0, and the replications pass.

    The minimum detectable drift at 80 percent power is ``(z_95 + z_80)``
    times the clustered standard error. The trimmed regression is reported
    in the failure text and is not a bar.
    """
    half = _Z95 * result["se"]
    low = result["drift"] - half
    high = result["drift"] + half
    text = (
        f"{label} drift {result['drift']:.4f} se {result['se']:.4f} "
        f"ci ({low:.4f}, {high:.4f}) mdd {result['mdd']:.4f} "
        f"z {result['z']:.2f} trimmed {result['trimmed_drift']:.4f} "
        f"z {result['trimmed_z']:.2f} quote {result['quote']:.6e}"
    )
    if not (low <= 0.0 <= high):
        raise AssertionError(text)
    failed = check_z(result["replication_z"])
    if failed is not None:
        raise AssertionError(f"{text}; {failed}")


def test_near_field_variance_equals_the_compensator_piece():
    """Lag 1 has conditional mean 1. This is an identity, not a sample.

    ``2 H Var(I_1) = dt^{2H}`` at ``H = 0.1`` and ``dt = 1/252``, to
    ``1e-12``. The factor ``exp(sqrt(2H) nu I_1 - 0.5 nu^2 dt^{2H})``
    then has conditional mean 1 for every ``nu``.
    """
    pytest.importorskip("roughvol")
    from roughvol.volterra import block_covariance

    hurst = 0.1
    dt = 1.0 / 252.0
    variance = float(block_covariance(hurst, 1, dt)[1, 1])
    piece = dt ** (2.0 * hurst)
    assert abs(2.0 * hurst * variance - piece) < 1e-12


def test_hybrid_tail_mark_bias_is_under_one_percent_of_the_quote():
    """Closed form from the power-kernel tail. Lag 1 contributes nothing.

    21 trading days: -4.364e-4 at ``nu = 1.5`` and -1.745e-3 at ``nu = 3``.
    30 trading days: -4.563e-4 and -1.825e-3. The stationary mixture, with
    ``pi_high = 1.5/2.3``, is -1.290e-3 and -1.349e-3. The bar is 0.01 of
    the quote.
    """
    pytest.importorskip("roughvol")
    dt = 1.0 / 252.0
    pi_high = 1.5 / 2.3
    expected = (
        (21, -4.36427467e-4, -1.74500639e-3),
        (30, -4.56312949e-4, -1.82452522e-3),
    )
    for n_days, low, high in expected:
        got_low = hybrid_tail_mark_bias(0.1, 1.5, n_days, dt)
        got_high = hybrid_tail_mark_bias(0.1, 3.0, n_days, dt)
        assert got_low == pytest.approx(low, rel=1e-6, abs=1e-12)
        assert got_high == pytest.approx(high, rel=1e-6, abs=1e-12)
        assert abs(got_low) < 0.01
        assert abs(got_high) < 0.01
        mixed = (1.0 - pi_high) * got_low + pi_high * got_high
        assert abs(mixed) < 0.01


def _fixed_atm_mark(model, paths, start, end):
    """Mark of the at-the-money call entered at ``start`` and expiring at ``end``.

    The payoff is ``max(Z_end / Z_start - 1, 0)``. At a later clock the
    unit-forward strike is ``Z_start / Z_clock``, and the mark multiplies
    that call by ``Z_clock / Z_start``.
    """
    entry = np.asarray(paths.Z[:, start], dtype=float)
    columns = []
    for clock in range(start, end):
        if int(_expiry_indices(paths.times.size, clock)[-1]) != end:
            raise AssertionError(f"clock {clock} does not expire at index {end}")
        strike = (entry / paths.Z[:, clock])[:, None]
        call, _stderr = model.unit_forward_calls(paths, clock, strike)
        columns.append(paths.Z[:, clock] / entry * call[:, -1, 0])
    return np.column_stack(columns)


def _episode_call_drift(model, n_days, n_paths, n_reps):
    """Filter-probability drift of the fixed at-the-money call.

    The drift is the signed slope of the one-step mark change on the
    hedger's probability of the high regime, times the number of steps,
    divided by the mean entry quote. The window stops at the last price
    before expiry. ``se`` is the path-clustered standard error of that
    fraction. The trimmed figures drop the largest 1 percent of absolute
    daily changes. They are a diagnostic.
    """
    end = 126 + n_days
    times = np.arange(end + 1) / 252.0
    panels = []
    probs = []
    quotes = []
    replication_z = []
    terminal = []
    for replication in range(n_reps):
        paths = model.simulate_market(n_paths, times, seed=run_seed(40 + replication, 2))
        mark = _fixed_atm_mark(model, paths, 126, end)
        payoff = np.maximum(paths.Z[:, end] / paths.Z[:, 126] - 1.0, 0.0)
        gap = payoff - mark[:, 0]
        se_gap = gap.std(ddof=1) / math.sqrt(gap.size)
        terminal.append(gap.mean() / se_gap)
        increment = np.diff(mark, axis=1)
        probability = paths.regime_prob[:, 126 : end - 1, 1]
        if probability.shape != increment.shape:
            raise AssertionError(f"probability {probability.shape} increment {increment.shape}")
        _beta, z_filter = _clustered_slope(increment, probability)
        replication_z.append(float(z_filter[1]))
        panels.append(increment)
        probs.append(probability)
        quotes.append(mark[:, 0])
    panel = np.concatenate(panels)
    probability = np.concatenate(probs)
    quote = float(np.concatenate(quotes).mean())
    beta, z_score = _clustered_slope(panel, probability)
    steps = panel.shape[1]
    z = float(z_score[1])
    drift = float(beta[1]) * steps / quote
    se = float(drift / z) if z != 0.0 else float("inf")
    trimmed_beta, trimmed_z = _clustered_slope(panel, probability, _drop_largest(panel, 0.01))
    return {
        "terminal": np.asarray(terminal),
        "drift": drift,
        "se": se,
        "z": z,
        "quote": quote,
        "trimmed_drift": float(trimmed_beta[1]) * steps / quote,
        "trimmed_z": float(trimmed_z[1]),
        "replication_z": np.asarray(replication_z),
        "mdd": (_Z95 + _Z80) * se,
    }


def test_variance_swap_mark_on_the_daily_episode_passes_the_z_rule():
    """Eight runs, 1,500 paths, gate settings, episode start 0.5.

    The mark is the left-point model variance accrued since the episode
    start, plus the remaining integral of the hybrid forward curve. That
    curve uses the ``nu`` in force on each finished step. Rebuilding the
    next variance from the next tenor is a code-consistency identity, not
    the martingale check. Maturities are 21 and 30 trading days. The last
    step's mark change is numerical noise and is left out of the drift
    regression.

    The primary bar is :func:`hybrid_tail_mark_bias`, under 1 percent of
    the quote. The Monte Carlo check is the untrimmed regression of the
    one-step mark change on the hedger's probability of the high regime,
    with path-clustered standard errors. The 95 percent interval contains
    0. On this draw the 21-day signed drift is -0.0573 of the quote,
    standard error 0.0661, interval (-0.187, 0.072), minimum detectable
    drift 0.185. The 30-day signed drift is -0.0849, standard error
    0.0565, interval (-0.196, 0.026), minimum detectable drift 0.158.
    One z-score per replication passes the z-rule, at the gate and at the
    ``H = 1/2`` and equal-``nu`` controls. Those signed drifts are 0.0575
    and 0.0037. Dropping the largest 1 percent of absolute changes is a
    diagnostic. It cuts the right tail of the near-field factor and is
    not a bar. On this draw the trimmed drifts are -0.248 and -0.276.
    """
    pytest.importorskip("roughvol")
    model = RegimeSwitchingBergomi(
        _prereg_spec(),
        xi0=0.04,
        rho=-0.4,
        H=0.1,
        n_mc=2,
        n_sub=21,
        n_pilot=1024,
        price_filter="rv",
    )
    for n_days in (21, 30):
        result = _episode_mark_drift(model, n_days)
        assert check_z(result["terminal"]) is None
        assert check_z(result["intercept"]) is None
        assert check_z(result["slope"]) is None
        _assert_filter_z(result, f"gate {n_days}")
    half = RegimeSwitchingBergomi(
        _prereg_spec(),
        xi0=0.04,
        rho=-0.4,
        H=0.5,
        n_mc=2,
        n_sub=21,
        n_pilot=1024,
        price_filter="rv",
    )
    half_result = _episode_mark_drift(half, 21)
    _assert_filter_z(half_result, "H=1/2")
    equal = RegimeSwitchingBergomi(
        RegimeSpec(
            generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
            params=[{"nu": 1.5}, {"nu": 1.5}],
            p0=np.array([0.7, 0.3]),
            switch="nu",
        ),
        xi0=0.04,
        rho=-0.4,
        H=0.1,
        n_mc=2,
        n_sub=21,
        n_pilot=1024,
        price_filter="rv",
    )
    _assert_filter_z(_episode_mark_drift(equal, 21), "equal nu")


def test_call_mark_on_the_daily_episode_passes_the_clustered_interval():
    """Fixed at-the-money call, forward chain, episode start 0.5, 21 trading days.

    The mark is the conditional expectation of ``max(Z_T / Z_entry - 1, 0)``.
    Each copy starts at the true regime and switches with the generator.
    The model drift is 0 by construction: each copy replicates the
    simulator's law, and the variance swap's tail-weight bias does not
    enter the call. The primary bar is the coupling identity. This
    regression is a wiring smoke test. It cannot detect a 1 percent
    drift. The minimum detectable drift on this sample is about 0.35 of
    the quote.

    The Monte Carlo check is the untrimmed regression of the one-step mark
    change on the hedger's probability of the high regime, with
    path-clustered standard errors. The 95 percent interval contains 0.
    The minimum detectable drift at 80 percent power is reported in the
    failure text.
    Eight runs, ``n_pilot`` 128, 64 copies. The rough sample is 150 paths:
    signed drift -0.018, standard error 0.124, interval (-0.262, 0.226),
    minimum detectable drift 0.348, on a quote of 0.01630. ``H = 1/2``
    uses 400 paths: signed drift 0.057, standard error 0.129, interval
    (-0.196, 0.311), minimum detectable drift 0.362, on a quote of 0.01643.
    Trimmed diagnostics are -0.053 and 0.181.
    """
    pytest.importorskip("roughvol")
    rough = RegimeSwitchingBergomi(
        _prereg_spec(),
        xi0=0.04,
        rho=-0.4,
        H=0.1,
        n_mc=64,
        n_sub=21,
        n_pilot=128,
        price_filter="rv",
        smile_k=np.array([0.0]),
    )
    half = RegimeSwitchingBergomi(
        _prereg_spec(),
        xi0=0.04,
        rho=-0.4,
        H=0.5,
        n_mc=64,
        n_sub=21,
        n_pilot=128,
        price_filter="rv",
        smile_k=np.array([0.0]),
    )
    for hurst, model, n_paths in ((0.1, rough, 150), (0.5, half, 400)):
        result = _episode_call_drift(model, 21, n_paths, 8)
        _assert_filter_z(result, f"call H={hurst}")
        assert check_z(result["terminal"]) is None


def test_rough_variance_swap_hold_to_maturity_passes_the_z_rule():
    """Eight runs, 1,500 paths, H = 0.1, T = 1. Entered at the time-0 quote.

    The quote is the remaining integral, and at inception the left-point
    leg is zero. The gap is the left-point model variance minus that quote.
    Constant nu of 1.5 and of 3, and the switching market, each pass the
    z-rule.
    """
    pytest.importorskip("roughvol")
    pricer = VarianceSwapPricer()
    times = np.linspace(0.0, 1.0, 11)
    contract = Contract(Vanilla(0.04), T=1.0)
    models = [
        _constant_nu(1.5),
        _constant_nu(3.0),
        RegimeSwitchingBergomi(
            _prereg_spec(), xi0=0.04, rho=-0.4, H=0.1, n_mc=2, n_sub=1, n_pilot=2
        ),
    ]
    dt = np.diff(times)
    for model in models:
        scores = []
        for replication in range(8):
            paths = model.simulate_market(1500, times, seed=run_seed(40 + replication, 0))
            quoted = pricer.price_along(paths, [contract])[:, 0, 0]
            realised = paths.state["v"][:, :-1] @ dt
            gap = realised - quoted
            se = gap.std(ddof=1) / math.sqrt(gap.size)
            scores.append(gap.mean() / se)
        assert check_z(np.asarray(scores)) is None


def test_quotes_do_not_enter_the_filter():
    """The stored filters match a recompute from y, y_rv, and log v.

    The forward curve is on the path and is not an argument of that
    recompute. A variance-swap quote and a call mid are not inputs.
    """
    pytest.importorskip("roughvol")
    spec = _prereg_spec()
    model = RegimeSwitchingBergomi(spec, xi0=0.04, rho=-0.4, H=0.1, n_mc=2)
    times = np.arange(31) / 252.0
    paths = model.simulate_market(48, times, seed=run_seed(12, 2))
    nus = np.array([1.5, 3.0])
    steps = [
        _stochastic(expm(spec.generator * float(times[i] - times[i - 1])))
        for i in range(1, times.size)
    ]
    means, variances = model._emission(nus)
    spot = filter_log_realised_variance(paths.state["y"], steps, means, variances, spec.p0)
    window = filter_differenced_realised_variance(
        paths.state["y_rv"], steps, paths.meta["rv_means"], paths.meta["rv_vars"], spec.p0
    )
    latent = filter_log_realised_variance(
        _log_variance_increments(paths.state["v"]),
        steps,
        paths.meta["latent_means"],
        paths.meta["latent_vars"],
        spec.p0,
    )
    assert spot == pytest.approx(paths.state["regime_prob_spot"], abs=1e-12)
    assert window == pytest.approx(paths.state["regime_prob_rv"], abs=1e-12)
    assert latent == pytest.approx(paths.state["regime_prob_latent"], abs=1e-12)
    assert "xi" in paths.state


def _entropic(pnl: np.ndarray, lam: float) -> float:
    log_weight = -lam * pnl
    centre = float(np.max(log_weight))
    return (centre + math.log(float(np.mean(np.exp(log_weight - centre))))) / lam


def _top_weight_share(pnl: np.ndarray, lam: float) -> float:
    log_weight = -lam * pnl
    count = int(math.ceil(0.01 * log_weight.size))
    order = np.argpartition(log_weight, -count)[-count:]
    centre = float(np.max(log_weight))
    weight = np.exp(log_weight - centre)
    return float(weight[order].sum() / weight.sum())


def _call_book(model, n_steps):
    times = np.arange(n_steps + 1) / 252.0
    premium_paths = _draw_paths(model, 65536, times, run_seed(1, 0))
    evaluation = _draw_paths(model, 20000, times, run_seed(12, 2))
    premium_payoff = np.maximum(premium_paths["Z"][:, -1] - 1.0, 0.0)
    evaluation_payoff = np.maximum(evaluation["Z"][:, -1] - 1.0, 0.0)
    premium = float(premium_payoff.mean())
    premium_se = float(premium_payoff.std(ddof=1) / math.sqrt(premium_payoff.size))
    bare = (premium - evaluation_payoff) / premium
    tau = times[-1] - times[:-1]
    variance = evaluation["v"]
    spot = evaluation["Z"]

    def hedged(sigma):
        held = bs.delta(spot[:, :-1], 1.0, tau, sigma)
        hedge = (held * np.diff(spot, axis=1)).sum(axis=1)
        turnover = np.abs(held[:, 0]) + np.abs(np.diff(held, axis=1)).sum(axis=1)
        turnover = turnover + np.abs(held[:, -1])
        return (premium - evaluation_payoff + hedge - 1e-4 * turnover) / premium

    sqrt_v = hedged(np.sqrt(np.maximum(variance[:, :-1], 1e-16)))
    flat = hedged(np.full(variance[:, :-1].shape, math.sqrt(model.xi0)))
    return premium, premium_se, bare, sqrt_v, flat


def test_short_call_tail_weight_uses_one_evaluation_seed():
    """65,536 premium paths and 20,000 evaluation paths, one seed per book.

    The premium seed is ``run_seed(1, 0)``. The evaluation seed is
    ``run_seed(12, 2)``. On the 30-day book minus the mean is 0.004158, the
    lambda = 1 tail share is 1.000, and the lambda = 0.1 tail share is
    0.02277. The hedged lambda = 0.1 tail shares, with the spot cost, are
    0.01368 and 0.01671 on the 21-day book.
    """
    pytest.importorskip("roughvol")
    model = RegimeSwitchingBergomi(
        _prereg_spec(), xi0=0.04, rho=-0.4, H=0.1, n_mc=2, n_sub=1, n_pilot=2
    )
    month_premium, month_se, month_bare, month_sqrt, month_flat = _call_book(model, 21)
    premium, premium_se, bare, sqrt_v, flat = _call_book(model, 30)
    assert month_premium == pytest.approx(0.02070205, abs=1e-8)
    assert month_se == pytest.approx(0.00012343, abs=1e-8)
    assert _top_weight_share(month_bare, 0.1) == pytest.approx(0.02337, abs=1e-5)
    assert _top_weight_share(month_bare, 0.1) < 0.10
    assert _top_weight_share(month_sqrt, 0.1) == pytest.approx(0.013678, abs=1e-5)
    assert _top_weight_share(month_flat, 0.1) == pytest.approx(0.016711, abs=1e-5)
    assert premium == pytest.approx(0.02459316, abs=1e-8)
    assert premium_se == pytest.approx(0.00014734, abs=1e-8)
    assert -float(bare.mean()) == pytest.approx(0.004158, abs=1e-6)
    assert _top_weight_share(bare, 1.0) == pytest.approx(1.0, abs=1e-4)
    assert _entropic(bare, 1.0) == pytest.approx(17.596, abs=1e-3)
    share = _top_weight_share(bare, 0.1)
    assert share == pytest.approx(0.02277, abs=1e-5)
    assert share < 0.10
    assert _entropic(bare, 0.1) == pytest.approx(0.1495, abs=1e-4)
    assert _top_weight_share(sqrt_v, 0.1) == pytest.approx(0.013627, abs=1e-5)
    assert _top_weight_share(flat, 0.1) == pytest.approx(0.015811, abs=1e-5)


def _switched_by(regime):
    changed = np.zeros(regime.shape, dtype=bool)
    changed[:, 1:] = np.cumsum(regime[:, 1:] != regime[:, :-1], axis=1) > 0
    return changed


def _episode_draw(nus, hurst, n_paths):
    spec = RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": float(nu)} for nu in nus],
        p0=np.array([0.7, 0.3]),
        switch="nu",
    )
    model = RegimeSwitchingBergomi(spec, xi0=0.04, rho=-0.4, H=hurst, n_mc=2)
    times = np.arange(126 + 30 + 1) / 252.0
    paths = model.simulate_market(
        n_paths, times, seed=run_seed(12, 2), store_forward=False
    )
    return paths, _switched_by(paths.regime)


def test_episode_filter_scores_after_burn_in():
    """Episode start t = 0.5, daily grid, seed entropy 12, test split.

    The filter sees the path from 0. Scores are split by whether the regime
    has switched by the date. The windowed observation uses 21 intraday
    returns, and that level is left-point variance plus noise. Each draw is
    scored and then freed. The forward-variance array is not stored, because
    the scores do not read it.
    """
    pytest.importorskip("roughvol")
    # (hurst, n_paths, switched, filter) -> confusion, brier at t0.
    t0 = {
        (0.5, 600, False, "regime_prob_spot"): ([[139, 57], [62, 58]], 0.50267677),
        (0.5, 600, False, "regime_prob_rv"): ([[68, 128], [45, 75]], 0.54923517),
        (0.5, 600, False, "regime_prob_latent"): ([[196, 0], [0, 120]], 0.00435509),
        (0.5, 600, True, "regime_prob_spot"): ([[46, 26], [137, 75]], 0.64478160),
        (0.5, 600, True, "regime_prob_rv"): ([[26, 46], [77, 135]], 0.48128047),
        (0.5, 600, True, "regime_prob_latent"): ([[65, 7], [4, 208]], 0.06166986),
        (0.5, 4000, False, "regime_prob_spot"): ([[915, 446], [444, 377]], 0.52163865),
        (0.5, 4000, False, "regime_prob_rv"): ([[470, 891], [288, 533]], 0.55024360),
        (0.5, 4000, False, "regime_prob_latent"): ([[1354, 7], [6, 815]], 0.00870802),
        (0.5, 4000, True, "regime_prob_spot"): ([[317, 179], [830, 492]], 0.67056211),
        (0.5, 4000, True, "regime_prob_rv"): ([[177, 319], [480, 842]], 0.48736833),
        (0.5, 4000, True, "regime_prob_latent"): ([[452, 44], [51, 1271]], 0.09253899),
        (0.1, 600, False, "regime_prob_spot"): ([[151, 45], [7, 113]], 0.26204393),
        (0.1, 600, False, "regime_prob_rv"): ([[192, 4], [0, 120]], 0.02991517),
        (0.1, 600, False, "regime_prob_latent"): ([[193, 3], [2, 118]], 0.02996947),
        (0.1, 600, True, "regime_prob_spot"): ([[33, 39], [31, 181]], 0.39412234),
        (0.1, 600, True, "regime_prob_rv"): ([[57, 15], [14, 198]], 0.16045452),
        (0.1, 600, True, "regime_prob_latent"): ([[61, 11], [13, 199]], 0.13240425),
        (0.1, 4000, False, "regime_prob_spot"): ([[1070, 291], [54, 767]], 0.24866322),
        (0.1, 4000, False, "regime_prob_rv"): ([[1327, 34], [13, 808]], 0.03691441),
        (0.1, 4000, False, "regime_prob_latent"): ([[1341, 20], [8, 813]], 0.02633243),
        (0.1, 4000, True, "regime_prob_spot"): ([[244, 252], [263, 1059]], 0.44903988),
        (0.1, 4000, True, "regime_prob_rv"): ([[396, 100], [128, 1194]], 0.20154028),
        (0.1, 4000, True, "regime_prob_latent"): ([[406, 90], [108, 1214]], 0.17108644),
    }
    pooled = {
        21: {
            False: ([[26578, 637], [293, 16276]], 0.03668642),
            True: ([[9039, 2201], [2476, 26500]], 0.18563385),
        },
        30: {
            False: ([[37047, 874], [478, 22862]], 0.03822418),
            True: ([[13467, 3159], [3628, 38485]], 0.18514544),
        },
    }
    filters = ("regime_prob_spot", "regime_prob_rv", "regime_prob_latent")
    for hurst, nus in ((0.5, (0.4, 1.1)), (0.1, (1.5, 3.0))):
        for n_paths in (600, 4000):
            paths, changed = _episode_draw(nus, hurst, n_paths)
            assert "xi" not in paths.state
            for switched in (False, True):
                mask0 = changed[:, 126] if switched else ~changed[:, 126]
                for name in filters:
                    scored = regime_scores(
                        paths.state[name][mask0, 126], paths.regime[mask0, 126]
                    )
                    confusion, brier = t0[(hurst, n_paths, switched, name)]
                    assert scored["confusion"].tolist() == confusion
                    assert scored["brier"] == pytest.approx(brier, abs=1e-8)
            if hurst == 0.1 and n_paths == 4000:
                for horizon, splits in pooled.items():
                    for switched, (confusion, brier) in splits.items():
                        probs = []
                        labels = []
                        for date in range(126, 126 + horizon):
                            mask = changed[:, date] if switched else ~changed[:, date]
                            probs.append(paths.state["regime_prob_rv"][mask, date])
                            labels.append(paths.regime[mask, date])
                        scored = regime_scores(np.concatenate(probs), np.concatenate(labels))
                        assert scored["confusion"].tolist() == confusion
                        assert scored["brier"] == pytest.approx(brier, abs=1e-8)
            del paths, changed
            gc.collect()
