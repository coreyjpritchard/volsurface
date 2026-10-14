"""Toy dynamic surface. Not a DYSANOS v3 replication. DESIGN section 22."""

import math

import numpy as np
import pytest

from volsurface.arbitrage.check import check_surface
from volsurface.core.types import ArbTol
from volsurface.dynamic.dysanos import (
    DEFAULT_DAYS,
    KAPPA_MIN,
    Dysanos,
    bootstrap_bands,
    simulate_factors,
    transition_matrix,
)
from volsurface.mc.rng import generator
from volsurface.surface.sanos import black_call


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


def test_one_day_transition_is_the_scalar_exponential():
    """``kappa = 0.5 + kappa_min`` per year. One day is ``exp(-kappa / 252)``.

    Two days is the square of that number, to ``1e-12``. A sign error, or a
    step that is read as days when it is already in years, misses both.
    """
    lower = np.array([[math.sqrt(0.5)]])
    skew = np.zeros((1, 1))
    day = 1.0 / 252.0
    kappa = 0.5 + KAPPA_MIN
    got = transition_matrix(lower, skew, day)
    expect = math.exp(-kappa * day)
    assert got[0, 0] == pytest.approx(expect, abs=1e-12)
    two = transition_matrix(lower, skew, 2.0 * day)
    assert two[0, 0] == pytest.approx(expect * expect, abs=1e-12)
    _, lower2, skew2, _ = _dynamics()
    half = transition_matrix(lower2, skew2, 0.5)
    full = transition_matrix(lower2, skew2, 1.0)
    assert full == pytest.approx(half @ half, abs=1e-12)


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


def test_expiry_mixture_moments_match_the_lognormal_atoms():
    """Last expiry, 20,000 draws, seed 11. Tolerance is 4 standard errors.

    ``E[Z] = q' K`` and ``E[Z^2] = q' (K^2 exp(eta V))``. Dropping the
    ``-v/2`` term in the sampler moves both moments by more than that bar
    on this expiry.
    """
    model = Dysanos(d=4, seed=1)
    surface = model.decode(generator(1).normal(size=4))
    index = surface.T.size - 1
    strikes = np.asarray(surface.strikes[index], dtype=float)
    weights = np.asarray(surface.weights[index], dtype=float)
    variance = surface.eta * float(surface.variance[index])
    n = 20_000
    spots = model.sample_expiry(surface, index, n, seed=11)
    mean = float(spots.mean())
    second = float(np.mean(spots**2))
    expected_mean = float(weights @ strikes)
    expected_second = float(weights @ (strikes**2 * math.exp(variance)))
    se_mean = float(spots.std(ddof=1) / math.sqrt(n))
    se_second = float((spots**2).std(ddof=1) / math.sqrt(n))
    assert abs(mean - expected_mean) <= 4.0 * se_mean
    assert abs(second - expected_second) <= 4.0 * se_second
    # The same draws miss the law that omits ``-v/2``.
    dropped_mean = float(weights @ (strikes * math.exp(0.5 * variance)))
    dropped_second = float(weights @ (strikes**2 * math.exp(2.0 * variance)))
    assert abs(mean - dropped_mean) > 4.0 * se_mean
    assert abs(second - dropped_second) > 4.0 * se_second


def test_pillar_call_is_the_atom_mixture_and_moves_with_the_latent():
    """The last pillar equals ``weights @ black_call`` to ``1e-10``. Two latents differ."""
    model = Dysanos(d=4, seed=1)
    calm = model.decode(np.zeros(4))
    moved = model.decode(np.ones(4))
    k = np.array([0.0])
    index = calm.T.size - 1
    strike = np.exp(k)
    kernel = calm.eta * float(calm.variance[index])
    mixture = calm.weights[index] @ black_call(
        calm.strikes[index][:, None], strike[None, :], kernel
    )
    got = float(calm.call(np.array([calm.T[index]]), k)[0, 0])
    assert got == pytest.approx(float(mixture[0]), abs=1e-10)
    other = float(moved.call(np.array([moved.T[index]]), k)[0, 0])
    assert abs(other - got) > 1e-4


def test_optional_scale_changes_the_innovation_and_the_default_does_not():
    mean, lower, skew, loading = _dynamics()
    flat = simulate_factors(mean, lower, skew, loading, 20, 1.0, seed=5)
    scaled = simulate_factors(mean, lower, skew, loading, 20, 1.0, seed=5, scale=np.full(20, 2.0))
    assert not np.allclose(flat, scaled)
    again = simulate_factors(mean, lower, skew, loading, 20, 1.0, seed=5)
    assert np.array_equal(flat, again)
