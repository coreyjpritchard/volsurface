"""Market paths, pricers, and the nu-only regime chain. DESIGN sections 21 and 22."""

import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import binom, chisquare

from volsurface.core.protocols import capabilities
from volsurface.core.types import Contract, Digital, MarketPaths, RegimeSpec, Vanilla
from volsurface.markets.filter import hamilton_filter
from volsurface.markets.pricers import ClosedFormPricer, SurrogatePricer, VarianceSwapPricer
from volsurface.markets.regime import RegimeSwitchingBergomi, bergomi_forward_variance
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
    seen = {}

    class _StandIn:
        def driver_y(self, n_paths, times, seed, H):
            seen["H"] = H
            seen["n"] = n_paths
            return np.zeros((n_paths, len(times)))

    monkeypatch.setattr("volsurface.markets.regime.create", lambda name, **kwargs: _StandIn())
    model = _model(H=0.1)
    paths = model.simulate_market(6, np.linspace(0.0, 0.2, 4), seed=3)
    assert seen == {"H": 0.1, "n": 6}
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
    """Y = 0, so the Wiener increment is 0. Future tenors are not copies of v_t."""

    class _StandIn:
        def driver_y(self, n_paths, times, seed, H):
            return np.zeros((n_paths, len(times)))

    monkeypatch.setattr("volsurface.markets.regime.create", lambda name, **kwargs: _StandIn())
    H = 0.1
    model = _model(H=H)
    times = np.linspace(0.0, 0.2, 4)
    paths = model.simulate_market(6, times, seed=3)
    row = 0
    clock = 1
    tenor = times.size - 1
    nu = float(_spec().params[int(paths.regime[row, clock])]["nu"])
    t = float(times[clock])
    u = float(times[tenor])
    expected = model.xi0 * math.exp(-0.5 * nu**2 * (u ** (2 * H) - (u - t) ** (2 * H)))
    assert paths.state["xi"][row, clock, tenor] == pytest.approx(expected, rel=1e-12)
    assert paths.state["xi"][row, clock, clock] == pytest.approx(paths.state["v"][row, clock])
    assert paths.state["xi"][row, clock, tenor] != pytest.approx(paths.state["v"][row, clock])
    assert np.isnan(paths.state["xi"][row, clock, 0])


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
