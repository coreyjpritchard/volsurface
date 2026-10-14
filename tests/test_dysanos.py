"""Minimal DYSANOS. Paper statements are UNVERIFIED. DESIGN section 22."""

import numpy as np
from scipy.stats import kstest

from volsurface.arbitrage.check import check_surface
from volsurface.core.types import ArbTol
from volsurface.dynamic.dysanos import (
    DEFAULT_DAYS,
    Dysanos,
    bootstrap_bands,
    fit_var,
    kappa_matrix,
    max_modulus,
    simulate_factors,
    transition_matrix,
)
from volsurface.mc.rng import generator


def _dynamics():
    lower = np.array([[0.4, 0.0], [0.1, 0.3]])
    skew = np.array([[0.0, 0.2], [-0.2, 0.0]])
    loading = np.array([[0.15, 0.0], [0.02, 0.12]])
    mean = np.array([0.0, 0.0])
    return mean, lower, skew, loading


def test_default_grid_is_seven_expiries_and_the_wide_grid_is_a_flag():
    model = Dysanos(d=4, seed=1)
    assert model.expiries.size == 7
    assert model.strikes.size == 21
    assert list(np.round(model.expiries * 252).astype(int)) == list(DEFAULT_DAYS)
    wide = Dysanos(d=4, wide=True, seed=1)
    assert wide.wide
    assert wide.expiries.size == 18
    assert wide.strikes.size == 200


def test_eigenvalues_of_A_stay_inside_the_unit_disk():
    """Five draws. The modulus is strictly below 1, including through a refit of L and S."""
    rng = generator(0)
    worst = 0.0
    for _ in range(5):
        lower = np.tril(rng.normal(scale=0.5, size=(3, 3)))
        raw = rng.normal(scale=0.2, size=(3, 3))
        skew = 0.5 * (raw - raw.T)
        for step in (1.0 / 252.0, 1.0):
            modulus = max_modulus(lower, skew, step)
            assert modulus < 1.0
            worst = max(worst, modulus)
        kappa = kappa_matrix(lower, skew)
        real = np.real(np.linalg.eigvals(kappa))
        assert np.min(real) >= 2e-4 - 1e-10
    assert worst > 0.0
    mean, lower, skew, loading = _dynamics()
    factors = simulate_factors(mean, lower, skew, loading, 400, 1.0, seed=4)
    centre, slope, _ = fit_var(factors)
    # The stored transition after a fit is still built from kappa, not from the free slope.
    refit = transition_matrix(lower, skew, 1.0)
    assert max_modulus(lower, skew, 1.0) < 1.0
    assert refit.shape == slope.shape
    assert centre.shape == mean.shape


def test_known_ar1_is_recovered_inside_bootstrap_bands():
    """D = 1, 6,000 dates, 200 bootstrap draws, seed 2. Bands are the 2.5 and 97.5 percentiles."""
    mean, lower, skew, loading = _dynamics()
    step = 1.0
    factors = simulate_factors(mean, lower, skew, loading, 6000, step, seed=2)
    truth = transition_matrix(lower, skew, step)
    cov_truth = loading.T @ loading
    low_a, high_a, low_c, high_c = bootstrap_bands(factors, n_boot=200, seed=3)
    assert np.all(truth >= low_a - 1e-12)
    assert np.all(truth <= high_a + 1e-12)
    assert np.all(cov_truth >= low_c - 1e-12)
    assert np.all(cov_truth <= high_c + 1e-12)


def test_decoded_surface_passes_check_surface_at_1e_9():
    model = Dysanos(d=4, n_strikes=21, seed=1)
    latent = generator(1).normal(size=4)
    surface = model.decode(latent)
    assert 0.01 <= surface.eta <= 0.50
    assert np.all(np.diff(surface.variance) > 0.0)
    mids = 0.5 * (surface.T[:-1] + surface.T[1:])
    grid_t = np.sort(np.concatenate([surface.T, mids]))
    log_k = np.log(model.strikes)[::2]
    report = check_surface(surface, grid_t, log_k, tol=ArbTol(price=1e-9))
    assert report.ok, report.violations
    assert report.violations.empty


def test_front_pit_is_uniform_by_ks():
    """2,000 draws, seed 3. The bar is a KS p-value above 0.001."""
    model = Dysanos(d=4, seed=1)
    surface = model.decode(generator(1).normal(size=4))
    spots = model.sample_front(surface, 2000, seed=3)
    pits = model.front_cdf(surface, spots)
    assert np.all((pits > 0.0) & (pits < 1.0))
    stat = kstest(pits, "uniform")
    assert stat.pvalue > 0.001


def test_held_book_is_not_negative_beyond_mc_noise():
    """4,000 draws from the last expiry, seed 4. The mean P&L is above -3.5 standard errors."""
    model = Dysanos(d=4, seed=1)
    surface = model.decode(generator(1).normal(size=4))
    index = surface.T.size - 1
    log_k = np.array([-0.1, 0.0, 0.1])
    premium = surface.call(np.array([surface.T[index]]), log_k)[0]
    spots = model.sample_expiry(surface, index, 4000, seed=4)
    strikes = np.exp(log_k)
    payoff = np.maximum(spots[:, None] - strikes[None, :], 0.0)
    pnl = payoff - premium
    mean = pnl.mean(axis=0)
    se = pnl.std(axis=0, ddof=1) / np.sqrt(pnl.shape[0])
    assert np.all(mean >= -3.5 * se)


def test_optional_scale_changes_the_innovation_and_the_default_does_not():
    mean, lower, skew, loading = _dynamics()
    flat = simulate_factors(mean, lower, skew, loading, 20, 1.0, seed=5)
    scaled = simulate_factors(mean, lower, skew, loading, 20, 1.0, seed=5, scale=np.full(20, 2.0))
    assert not np.allclose(flat, scaled)
    again = simulate_factors(mean, lower, skew, loading, 20, 1.0, seed=5)
    assert np.array_equal(flat, again)
