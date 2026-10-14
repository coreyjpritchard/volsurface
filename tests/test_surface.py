"""Surfaces: normalised Black calculus, linear and spline interpolation, raw SVI."""

import numpy as np
import pytest
from scipy.stats import norm

from volsurface.core.types import Curves, Domain
from volsurface.fit.svi import fit_raw_svi
from volsurface.surface.base import SurfaceMixin
from volsurface.surface.black import (
    call_from_total_variance,
    d2call_dk2_from_w,
    dcall_dk_from_w,
    durrleman_g,
    log_density,
)
from volsurface.surface.svi import RawSVI


def _flat_market():
    def one(t):
        return np.ones_like(np.asarray(t, dtype=float), dtype=float)

    return Curves(one, one)


def test_flat_smile_density_is_normal_and_g_is_one():
    """Black-Scholes: g = 1 and phi is N(-w/2, w)."""
    k = np.linspace(-1.5, 1.5, 81)
    w = np.full_like(k, 0.04)
    g = durrleman_g(k, w, np.zeros_like(k), np.zeros_like(k))
    np.testing.assert_allclose(g, 1.0, atol=1e-12)
    phi = log_density(k, w, 0.0, 0.0)
    expected = norm.pdf(k, loc=-0.5 * 0.04, scale=np.sqrt(0.04))
    np.testing.assert_allclose(phi, expected, atol=1e-12)
    # The density integrates to one on a wide grid by the trapezoid rule.
    integral = np.trapezoid(phi, k)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_call_derivatives_match_finite_differences():
    """Chain rule through a tilted total variance, against a central step of 1e-5."""
    k = np.linspace(-0.8, 0.6, 25)
    w = 0.04 + 0.01 * k + 0.005 * k**2
    w_k = 0.01 + 0.01 * k
    w_kk = np.full_like(k, 0.01)
    h = 1e-5

    def w_of(kk):
        return 0.04 + 0.01 * kk + 0.005 * kk**2

    fd_dk = (
        call_from_total_variance(k + h, w_of(k + h))
        - call_from_total_variance(k - h, w_of(k - h))
    ) / (2 * h)
    np.testing.assert_allclose(dcall_dk_from_w(k, w, w_k), fd_dk, atol=1e-8)
    fd_dkk = (
        dcall_dk_from_w(k + h, w_of(k + h), w_k + w_kk * h)
        - dcall_dk_from_w(k - h, w_of(k - h), w_k - w_kk * h)
    ) / (2 * h)
    np.testing.assert_allclose(d2call_dk2_from_w(k, w, w_k, w_kk), fd_dkk, atol=1e-6)


def test_vogt_slice_has_negative_g():
    """Gatheral and Jacquier (2014), Example 3.1, parameters (3.8)."""
    from volsurface.surface.svi import VOGT_2014

    vogt = RawSVI.vogt()
    k = np.linspace(-1.5, 1.5, 601)
    assert vogt.g(k).min() < 0.0
    # SVI-JW ATM total variance quoted in their Example 5.1.
    assert vogt.w(0.0) == pytest.approx(0.01742625, abs=1e-8)
    # The printed tuple is (a, b, m, rho, sigma). The dataclass is (a, b, rho, m, sigma).
    a, b, m, rho, sigma = VOGT_2014
    assert RawSVI(a=a, b=b, rho=rho, m=m, sigma=sigma) == vogt
    assert RawSVI(*VOGT_2014) != vogt


def test_svi_surface_accepts_python_lists():
    """In-spread sampling passes one quote as two lists. That used to raise TypeError."""
    from volsurface.surface.svi import SVISurface

    surface = SVISurface([(1.0, RawSVI.vogt())], _flat_market(), Domain(1.0, 1.0, -1.0, 1.0))
    value = surface.call([1.0], [0.0])
    assert value.shape == (1, 1)
    assert np.isfinite(value[0, 0])
    assert np.isfinite(surface.g([1.0], [0.1])[0, 0])
    outside = surface.extrapolated([1.0], [2.0])
    assert bool(outside[0, 0])
    assert not bool(surface.extrapolated([1.0], [0.0])[0, 0])


def test_floor_penalty_keeps_a_constant_residual_length():
    from volsurface.fit.svi import _initial, _residual

    k = np.linspace(-0.2, 0.2, 8)
    total_var = np.full_like(k, 0.04)
    start = _initial(k, total_var)
    start[0] = -5.0
    gap = _residual(start, k, total_var, np.ones_like(k))
    assert gap.shape == (k.size + 1,)
    assert gap[-1] < 0.0
    safe = _residual(_initial(k, total_var), k, total_var, np.ones_like(k))
    assert safe.shape == (k.size + 1,)
    assert safe[-1] == pytest.approx(0.0)


def test_raw_svi_fit_recovers_a_slice():
    truth = RawSVI(a=0.01, b=0.15, rho=-0.4, m=0.05, sigma=0.2)
    k = np.linspace(-0.5, 0.4, 31)
    fitted, residual = fit_raw_svi(k, truth.w(k))
    np.testing.assert_allclose(fitted.w(k), truth.w(k), atol=1e-8)
    assert np.max(np.abs(residual)) < 1e-8


def test_spline_of_a_flat_smile_stays_flat_inside_the_quotes():
    from volsurface.surface.spline import SplineSurface

    k = np.linspace(-0.4, 0.3, 9)
    pillars = [{"T": 0.5, "k": k, "iv": np.full_like(k, 0.2)}]
    surface = SplineSurface(pillars, _flat_market(), Domain(0.5, 0.5, k[0], k[-1]))
    grid = np.linspace(k[0], k[-1], 15)
    iv = surface.iv([0.5], grid)[0]
    np.testing.assert_allclose(iv, 0.2, atol=1e-10)
    # Past the last quote the cubic is an extrapolation and is marked as such.
    wing = np.array([k[-1] + 0.2])
    assert surface.extrapolated([0.5], wing)[0, 0]


class _Bump(SurfaceMixin):
    """Flat Black call, with three planted derivative failures of known size."""

    def __init__(self):
        self.market = _flat_market()
        self.domain = Domain(0.1, 2.0, -1.0, 1.0)
        self.sigma = 0.2

    def call(self, T, k):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        k = np.atleast_1d(np.asarray(k, dtype=float))
        w = self.sigma**2 * T[:, None]
        return call_from_total_variance(k[None, :], w)

    def _base(self, T, k):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        k = np.atleast_1d(np.asarray(k, dtype=float))
        w = self.sigma**2 * T[:, None] * np.ones((1, k.size))
        zeros = np.zeros_like(w)
        return (
            dcall_dk_from_w(k[None, :], w, zeros),
            d2call_dk2_from_w(k[None, :], w, zeros, zeros),
        )

    def dcall_dk(self, T, k, h=1e-4):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        k = np.atleast_1d(np.asarray(k, dtype=float))
        ck, _ = self._base(T, k)
        # Slope s = 0.25 at (T, k) = (0.5, 0). s = exp(-k) dC/dk.
        mask = np.isclose(T, 0.5)[:, None] & np.isclose(k, 0.0)[None, :]
        return np.where(mask, 0.25 * np.exp(k)[None, :], ck)

    def d2call_dk2(self, T, k, h=1e-4):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        k = np.atleast_1d(np.asarray(k, dtype=float))
        ck, ckk = self._base(T, k)
        # Keep convexity at the call-spread plant equal to the Black value,
        # which means C_kk moves with the planted C_k.
        planted_ck = self.dcall_dk(T, k)
        # gamma = exp(-2k) (C_kk - C_k). Black gamma, except -0.05 at (1.0, 0.2).
        black_gamma = np.exp(-2.0 * k)[None, :] * (ckk - ck)
        gamma = black_gamma.copy()
        fly = np.isclose(T, 1.0)[:, None] & np.isclose(k, 0.2)[None, :]
        gamma = np.where(fly, -0.05, gamma)
        return planted_ck + gamma * np.exp(2.0 * k)[None, :]

    def dcall_dT(self, T, k, h=1e-4):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        k = np.atleast_1d(np.asarray(k, dtype=float))
        # Black calendar is positive. Plant -0.02 at (0.3, -0.2).
        w = self.sigma**2 * T[:, None]
        # dC/dT = dC/dw * sigma^2 for flat Black.
        from volsurface.surface.black import dcall_dw

        dt = dcall_dw(k[None, :], w) * self.sigma**2
        mask = np.isclose(T, 0.3)[:, None] & np.isclose(k, -0.2)[None, :]
        return np.where(mask, -0.02, dt)
