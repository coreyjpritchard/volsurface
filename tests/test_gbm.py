"""Geometric Brownian motion: z-rule against Black, inversion, parity, Euler bias.

The z-rule is ``docs/DESIGN.md`` section 7, implemented in
``volsurface.mc.zrule``. ``n`` is stated on each test that uses it. Nodes in
the fraction test have independent seeds. A shared-path grid checks the max
on one draw and the mean on independent replications of that grid.
"""

import numpy as np
from scipy.stats import norm

from volsurface import bs
from volsurface.mc.grid import price_grid
from volsurface.mc.rng import generator, spawn
from volsurface.mc.zrule import check_replication_means, check_z, max_abs_bound
from volsurface.models.gbm import GBM
from volsurface.plugins import create


def test_draw_order_is_one_normal_vector_per_step():
    """The child seed's first draw is the first step, path index fastest. The mirror is -g."""
    sigma, n_paths = 0.3, 5
    times = np.array([0.0, 0.25, 1.0])
    model = GBM(sigma=sigma, n_paths=n_paths, n_steps=2, seed=11, antithetic=True, scheme="exact")
    paths = model.simulate(n_paths, times, seed=11)
    rng = generator(spawn(11, 1)[0])
    g0 = rng.standard_normal(n_paths)
    g1 = rng.standard_normal(n_paths)
    dt0, dt1 = 0.25, 0.75
    shock0 = np.concatenate([g0, -g0])
    shock1 = np.concatenate([g1, -g1])
    log1 = -0.5 * sigma**2 * dt0 + sigma * np.sqrt(dt0) * shock0
    log2 = log1 - 0.5 * sigma**2 * dt1 + sigma * np.sqrt(dt1) * shock1
    np.testing.assert_allclose(paths.Z[:, 1], np.exp(log1), atol=1e-12)
    np.testing.assert_allclose(paths.Z[:, 2], np.exp(log2), atol=1e-12)
    assert paths.Z.shape == (2 * n_paths, 3)
    assert paths.times[0] == 0.0


def test_pathwise_put_call_parity_on_the_same_paths():
    """On each path, call minus put is df * F * (Z - K/F). The averages use that same Z."""
    F, K, T, df = 100.0, np.array([80.0, 100.0, 130.0]), 0.7, 0.96
    model = GBM(sigma=0.25, n_paths=4000, n_steps=3, seed=3, scheme="exact")
    paths = model.simulate(model.n_paths, np.linspace(0.0, T, model.n_steps + 1), seed=model.seed)
    z = paths.Z[:, -1]
    call_path = df * np.maximum(F * z[:, None] - K[None, :], 0.0)
    put_path = df * np.maximum(K[None, :] - F * z[:, None], 0.0)
    np.testing.assert_allclose(call_path - put_path, df * (F * z[:, None] - K[None, :]), atol=1e-12)
    call = model.price(F, K, T, df, "call")
    put = model.price(F, K, T, df, "put")
    np.testing.assert_allclose(call - put, df * (F * z.mean() - K), atol=1e-12)


def test_exact_scheme_matches_black_scholes_by_the_z_rule():
    """200 nodes, independent seeds, 8_000 paths each. The fraction test needs that independence."""
    F, df, sigma = 100.0, 0.98, 0.22
    k = np.linspace(-0.3, 0.3, 40)
    maturities = np.array([0.3, 0.6, 1.0, 1.5, 2.0])
    nodes = [(float(t), float(kk)) for t in maturities for kk in k]
    assert len(nodes) == 200
    z = np.empty(len(nodes))
    for i, (t, kk) in enumerate(nodes):
        K = F * np.exp(kk)
        model = GBM(sigma=sigma, n_paths=8_000, n_steps=4, seed=1_000 + i, scheme="exact")
        result = model.price_with_error(F, K, t, df, "call")
        reference = float(bs.price(F, K, t, sigma, df, "call"))
        z[i] = (float(result.value) - reference) / float(result.stderr)
    message = check_z(z)
    assert message is None, message


def test_reported_standard_error_matches_the_scatter_of_estimates():
    """R = 200 seeds, antithetic pairs and the Z control, 2_000 paths, one step.

    ``stderr = sd / sqrt(n)`` by construction, so a slope of log(stderr) on
    log(n) does not test the estimator. Here the sample standard deviation of
    the 200 prices, divided by the mean reported standard error, lies in
    [0.9, 1.1].
    """
    F, K, T, df, sigma = 100.0, 100.0, 1.0, 1.0, 0.2
    n_rep = 200
    estimates = np.empty(n_rep)
    stderr = np.empty(n_rep)
    for seed in range(n_rep):
        model = GBM(
            sigma=sigma,
            n_paths=2_000,
            n_steps=1,
            seed=seed,
            scheme="exact",
            antithetic=True,
            control_variate=True,
        )
        result = model.price_with_error(F, K, T, df, "call")
        estimates[seed] = float(result.value)
        stderr[seed] = float(result.stderr)
    ratio = float(np.std(estimates, ddof=1) / np.mean(stderr))
    assert 0.9 <= ratio <= 1.1, ratio


def test_shared_path_grid_and_iv_round_trip():
    """One simulation for the whole grid. The nodes share paths, so they are one draw.

    n = 4 maturities x 11 strikes = 44. The max bound uses that n. The mean is
    checked on R = 32 independent replications, with the observed sd of the
    replication means. Price to implied vol to price, relative 1e-12 where the
    implied vol exists.
    """
    F, df, sigma = 50.0, 0.99, 0.2
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    k = np.linspace(-0.2, 0.2, 11)
    K = F * np.exp(k)
    model = GBM(sigma=sigma, scheme="exact", antithetic=True, control_variate=True)
    reference = np.vstack([bs.price(F, K, t, sigma, df, "call") for t in maturities])

    def grid(seed: int):
        return price_grid(model, F, K, maturities, df, "call", n_paths=20_000, seed=seed)

    def z_of(seed: int) -> np.ndarray:
        result = grid(seed)
        return (result.value - reference) / result.stderr

    result = grid(7)
    z = (result.value - reference) / result.stderr
    n = int(z.size)
    assert n == 44
    message = check_z(z, mean=False, fraction=False)
    assert message is None, message
    n_rep = 32
    means = np.array([float(z_of(seed).mean()) for seed in range(n_rep)])
    message = check_replication_means(means)
    assert message is None, message

    excluded = 0
    for i, t in enumerate(maturities):
        iv = bs.implied_vol(result.value[i], F, K, t, df, "call")
        missing = ~np.isfinite(iv)
        excluded += int(missing.sum())
        if np.any(~missing):
            rebuilt = bs.price(F, K[~missing], t, iv[~missing], df, "call")
            rel = np.abs(rebuilt - result.value[i, ~missing]) / np.abs(result.value[i, ~missing])
            assert np.max(rel) < 1e-12, float(np.max(rel))
    assert excluded == 0, f"excluded {excluded} of {n} nodes with no implied vol"


def test_closed_form_iv_round_trip_on_standardised_moneyness():
    """Vol to price to vol, to 1e-8 where n(d1) sqrt(T) exceeds 1e-8.

    Strikes are built in standardised moneyness ``k / (sigma sqrt(T))`` on
    ``[-4, 4]``, 21 strikes, four expiries. The fraction of that grid whose
    normalised vega clears ``1e-8`` is asserted, not an absolute exclusion
    count. Where an implied vol is returned, the price round trip is relative
    1e-12. With a failed inversion stored as NaN, that round trip holds by
    construction on the finite nodes.
    """
    F, df, sigma = 100.0, 0.95, 0.2
    moneyness = np.linspace(-4.0, 4.0, 21)
    maturities = np.array([0.005, 0.02, 0.25, 1.0])
    kept = 0
    total = 0
    for t in maturities:
        sd = sigma * np.sqrt(t)
        k = moneyness * sd
        K = F * np.exp(k)
        total += K.size
        price = bs.price(F, K, t, sigma, df, "call")
        back = bs.implied_vol(price, F, K, t, df, "call")
        d1 = -moneyness + 0.5 * sd
        vega = norm.pdf(d1) * np.sqrt(t)
        keep = vega > 1e-8
        kept += int(keep.sum())
        assert np.all(np.isfinite(back[keep]))
        np.testing.assert_allclose(back[keep], sigma, atol=1e-8)
        finite = np.isfinite(back)
        rebuilt = bs.price(F, K[finite], t, back[finite], df, "call")
        rel = np.abs(rebuilt - price[finite]) / np.abs(price[finite])
        assert np.max(rel) < 1e-12, float(np.max(rel))
    coverage = kept / total
    assert coverage >= 0.95, f"coverage {coverage:.3f} on {total} nodes"


def test_one_step_euler_matches_the_normal_and_not_black():
    """n = 1 node, 100_000 paths. The normal formula is the Euler law. Black is not."""
    F, K, T, df, sigma = 100.0, 100.0, 1.0, 1.0, 1.0
    n_paths = 100_000
    model = GBM(sigma=sigma, n_paths=n_paths, n_steps=1, seed=9, scheme="euler")
    result = model.price_with_error(F, K, T, df, "call")
    sd = sigma * np.sqrt(T)
    # E[(Z - 1)+] for Z ~ N(1, sd^2) is sd / sqrt(2 pi). Cash price is df * F times that.
    euler = df * F * sd / np.sqrt(2.0 * np.pi)
    black = float(bs.price(F, K, T, sigma, df, "call"))
    z_euler = (float(result.value) - euler) / float(result.stderr)
    z_black = (float(result.value) - black) / float(result.stderr)
    assert check_z(np.array([z_euler])) is None, z_euler
    assert abs(z_black) > max_abs_bound(1), z_black


def test_price_grid_uses_one_seed_for_every_maturity():
    """A one-maturity grid and ``price`` with one step consume the same child and match."""
    model = GBM(sigma=0.2, n_steps=1, scheme="exact", antithetic=False)
    K = np.array([90.0, 100.0, 110.0])
    grid = price_grid(model, 100.0, K, np.array([0.5]), 0.97, "call", n_paths=3000, seed=4)
    single = model.price_with_error(100.0, K, 0.5, 0.97, "call", n_paths=3000, seed=4)
    np.testing.assert_allclose(grid.value[0], single.value, atol=1e-12)
    np.testing.assert_allclose(grid.stderr[0], single.stderr, atol=1e-12)


def test_registry_builds_gbm():
    model = create("gbm", sigma=0.2, n_paths=10, seed=1)
    assert isinstance(model, GBM)
    assert model.price(1.0, np.array([1.0]), 1.0).shape == (1,)
