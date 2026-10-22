"""Discrete local volatility. DESIGN section 19."""

import numpy as np
import pytest

from volsurface.arbitrage import check_surface
from volsurface.core.types import ArbTol, Curves, Domain
from volsurface.fit.sanos import project_marginal
from volsurface.surface.sanos import SanosSurface
from volsurface.surface.sanos_dlv import (
    advance,
    calls_from_masses,
    implied_sigma,
    transition_matrix,
    transition_matrix_torch,
)


def _strikes() -> np.ndarray:
    return np.linspace(0.55, 1.55, 9)


def _sigma(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width = np.zeros(9)
    width[1:-1] = rng.uniform(0.05, 0.35, size=7)
    return width


def _weights(seed: int, strikes: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights, _ = project_marginal(rng.random(strikes.size), strikes)
    return weights


def test_random_sigma_is_column_stochastic_and_a_martingale():
    strikes = _strikes()
    matrix = transition_matrix(strikes, _sigma(1), 0.2)
    ones = np.ones(strikes.size)
    assert ones @ matrix == pytest.approx(ones, abs=1e-12)
    assert strikes @ matrix == pytest.approx(strikes, abs=1e-12)


def test_round_trip_reproduces_arbitrage_free_calls():
    """Calls to Sigma to Q to calls, to 1e-10."""
    strikes = _strikes()
    weights = _weights(2, strikes)
    sigma = _sigma(3)
    dT = 0.25
    nxt = transition_matrix(strikes, sigma, dT) @ weights
    prev = calls_from_masses(strikes, weights)
    now = calls_from_masses(strikes, nxt)
    recovered = implied_sigma(strikes, prev, now, dT)
    back = transition_matrix(strikes, recovered, dT) @ weights
    rebuilt = calls_from_masses(strikes, back)
    assert rebuilt == pytest.approx(now, abs=1e-10)
    assert recovered[0] == 0.0
    assert recovered[-1] == 0.0


def test_torch_matches_numpy_in_float64():
    torch = pytest.importorskip("torch")
    strikes = _strikes()
    sigma = _sigma(4)
    numpy_q = transition_matrix(strikes, sigma, 0.15)
    torch_q = transition_matrix_torch(strikes, sigma, 0.15)
    assert torch_q.dtype == torch.float64
    assert torch_q.detach().cpu().numpy() == pytest.approx(numpy_q, abs=1e-12)


def test_inhomogeneous_split_keeps_mass_and_mean_then_the_transition_does():
    source = np.array([0.7, 1.0, 1.3])
    masses, _ = project_marginal(np.array([0.2, 0.5, 0.3]), source)
    target = _strikes()
    sigma = np.zeros(target.size)
    sigma[1:-1] = 0.15
    landed = advance(source, masses, target, sigma, 0.1)
    assert landed.sum() == pytest.approx(1.0, abs=1e-12)
    assert float(landed @ target) == pytest.approx(float(masses @ source), abs=1e-12)


def test_decoded_chain_passes_check_surface():
    """Three expiries from a random nonnegative Sigma. eta = 0."""
    strikes = _strikes()
    weights = _weights(5, strikes)
    mats = [weights]
    for seed in (6, 7):
        sigma = _sigma(seed)
        mats.append(transition_matrix(strikes, sigma, 0.3) @ mats[-1])
    times = np.array([0.3, 0.6, 0.9])
    variance = np.array([0.02, 0.04, 0.07])

    def flat(level):
        def curve(t):
            return np.full(np.shape(t), level, dtype=float)

        return curve

    surface = SanosSurface(
        times,
        [strikes, strikes, strikes],
        mats,
        variance,
        0.0,
        Curves(flat(1.0), flat(1.0)),
        Domain(0.3, 0.9, float(np.log(strikes[0])), float(np.log(strikes[-1]))),
    )
    grid_t = np.array([0.3, 0.45, 0.6, 0.75, 0.9])
    log_k = np.log(strikes)
    report = check_surface(surface, grid_t, log_k, tol=ArbTol(price=1e-9))
    assert report.ok, report.violations
    assert report.violations.empty
