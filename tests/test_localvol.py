"""Dupire local variance and log-Euler Monte Carlo.

Flat Black-Scholes is analytic: local volatility equals sigma. The Vogt SVI
slice has a negative Durrleman denominator, and that cell is flagged rather
than clipped. Monte Carlo tolerances are an implied-vol budget for the
Euler bias, plus the standard error of the implied vol.
"""

import numpy as np
import pytest

from volsurface import bs
from volsurface.core.types import Curves, Domain
from volsurface.mc.zrule import check_z, max_abs_bound
from volsurface.models.bs import BlackScholes
from volsurface.models.heston import Heston
from volsurface.models.localvol import EXPLODE, LocalVol, dupire
from volsurface.plugins import create
from volsurface.surface.black import durrleman_g
from volsurface.surface.model import ModelSurface
from volsurface.surface.svi import RawSVI, SVISurface


def _market(forward=100.0):
    def curve(level):
        return lambda t: np.full(np.shape(t), level, dtype=float)

    return Curves(F=curve(forward), df=curve(1.0))


def test_registry_builds_localvol():
    """The entry point and the in-tree register both name this class."""
    built = create(
        "localvol",
        t_grid=np.array([0.0, 1.0]),
        k_grid=np.array([-0.2, 0.2]),
        variance=np.full((2, 2), 0.04),
    )
    assert isinstance(built, LocalVol)
    assert built.n_steps == 32


def test_flat_black_scholes_local_vol_equals_sigma():
    """Analytic slopes. The gap is zero, the tolerance is 1e-4 on sigma."""
    surface = ModelSurface(
        BlackScholes(0.2), _market(), Domain(0.0, 2.0, -1.0, 1.0),
    )
    report = dupire(surface, np.array([0.25, 1.0]), np.linspace(-0.4, 0.4, 9))
    sigma = np.sqrt(report.variance)
    assert np.max(np.abs(sigma - 0.2)) < 1e-4
    assert report.n_negative == 0
    assert report.n_exploding == 0
    # The expanded total-variance denominator is Durrleman's g.
    k = np.array([0.2])
    w, w_k, w_kk = np.array([[0.04]]), np.array([[0.01]]), np.array([[0.002]])
    expanded = (
        1.0
        - (k / w) * w_k
        + 0.25 * (-0.25 - 1.0 / w + k**2 / w**2) * w_k**2
        + 0.5 * w_kk
    )
    np.testing.assert_allclose(expanded, durrleman_g(k, w, w_k, w_kk))


def test_dupire_is_nan_outside_the_quoted_domain():
    surface = ModelSurface(
        BlackScholes(0.2), _market(), Domain(0.2, 1.0, -0.2, 0.2),
    )
    report = dupire(surface, np.array([0.5]), np.array([-0.5, 0.0, 0.5]))
    assert np.isnan(report.variance[0, 0])
    assert np.isnan(report.variance[0, 2])
    assert report.variance[0, 1] == pytest.approx(0.04)
    assert report.extrapolated[0, 0] and not report.extrapolated[0, 1]


def test_negative_denominator_is_flagged_on_the_vogt_slice():
    """Gatheral and Jacquier example 3.1. g is negative near k = 0.88.

    The domain includes that strike. The cell stays negative. It is counted,
    not replaced by a floor. A strike near zero, where g is positive, is not
    in the mask.
    """
    surface = SVISurface(
        [(1.0, RawSVI.vogt())], _market(), Domain(0.5, 1.5, -0.2, 1.4),
    )
    k = np.linspace(-0.2, 1.4, 161)
    report = dupire(surface, np.array([1.0]), k)
    assert report.n_negative > 0
    assert np.any(report.bad_denominator & (report.denominator <= 0.0))
    assert np.nanmin(report.denominator) < 0.0
    near_zero = int(np.argmin(np.abs(k)))
    assert not report.bad_denominator[0, near_zero]
    assert report.denominator[0, near_zero] > 0.0


def test_flat_local_vol_monte_carlo_reprices_black():
    """Log-Euler of a constant variance is the exact lognormal step.

    The Euler-bias budget is 5 bp of implied vol. On top of that, each
    strike may miss by the z-rule's max-|z| bound times its implied-vol
    standard error. A one-sigma cap would fail on an ordinary normal node.
    Twelve strikes, each its own seed, 6,000 paths. The price z-scores
    pass the z-rule at n = 12.
    """
    surface = ModelSurface(
        BlackScholes(0.2), _market(), Domain(0.0, 1.5, -1.5, 1.5),
    )
    t_grid = np.linspace(0.0, 1.5, 4)
    k_grid = np.linspace(-1.5, 1.5, 13)
    variance = dupire(surface, t_grid, k_grid).variance
    model = LocalVol(t_grid, k_grid, variance, n_paths=6_000, n_steps=4, seed=0)
    log_k = np.linspace(-0.2, 0.2, 12)
    strike = 100.0 * np.exp(log_k)
    black = BlackScholes(0.2)
    z = []
    vol_cap = max_abs_bound(strike.size)
    for i, one in enumerate(strike):
        result = model.price_with_error(100.0, np.array([one]), 1.0, seed=20 + i)
        reference = float(black.price(100.0, np.array([one]), 1.0)[0])
        iv = float(bs.implied_vol(result.value, 100.0, np.array([one]), 1.0, 1.0)[0])
        vega = float(bs.vega(100.0, one, 1.0, 0.2, 1.0))
        se_iv = float(result.stderr[0]) / vega
        assert abs(iv - 0.2) < 0.0005 + vol_cap * se_iv, (iv, se_iv, vol_cap)
        z.append((float(result.value[0]) - reference) / float(result.stderr[0]))
    failure = check_z(np.array(z))
    assert failure is None, failure


def test_heston_local_vol_monte_carlo_reprices_within_an_iv_budget():
    """Heston-generated surface, Dupire by finite difference, then log-Euler.

    The grid is inside the domain and has no non-positive cell. The budget
    is 40 bp of implied vol for the Euler bias and the difference stencil,
    plus one standard error. Strikes 95, 100, 105 at T = 1, 16,000 paths,
    48 steps, seed 4.
    """
    heston = Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.15, rho=-0.3)
    surface = ModelSurface(heston, _market(), Domain(0.0, 1.5, -0.8, 0.8))
    t_grid = np.linspace(0.15, 1.2, 8)
    k_grid = np.linspace(-0.3, 0.3, 13)
    report = dupire(surface, t_grid, k_grid)
    assert report.n_negative == 0
    assert report.n_exploding == 0
    assert np.nanmax(np.abs(report.variance)) < EXPLODE
    full_t = np.concatenate([[0.0], t_grid])
    variance = np.vstack([report.variance[0], report.variance])
    model = LocalVol(full_t, k_grid, variance, n_paths=16_000, n_steps=48, seed=4)
    strike = np.array([95.0, 100.0, 105.0])
    result = model.price_with_error(100.0, strike, 1.0, seed=4)
    reference = heston.price(100.0, strike, 1.0)
    iv_mc = bs.implied_vol(result.value, 100.0, strike, 1.0, 1.0)
    iv_ref = bs.implied_vol(reference, 100.0, strike, 1.0, 1.0)
    vega = bs.vega(100.0, strike, 1.0, iv_ref, 1.0)
    se_iv = result.stderr / vega
    gap = np.abs(iv_mc - iv_ref)
    assert np.all(gap < 0.0040 + se_iv), (gap, se_iv)
