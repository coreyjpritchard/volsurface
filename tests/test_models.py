import numpy as np

from volsurface import models


def test_protocol_and_flat_bs_surface():
    m = models.BlackScholes(0.23)
    assert isinstance(m, models.Model)
    K = np.exp(np.linspace(-0.3, 0.2, 7))
    surface = models.implied_surface(m, 1.0, K, [0.05, 0.5, 2.0], df=0.97)
    assert surface.shape == (3, 7)
    np.testing.assert_allclose(surface, 0.23, atol=1e-10)


def test_heston_surface_per_maturity_strikes():
    m = models.Heston(v0=0.02, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)
    T = np.array([0.1, 1.0])
    K = np.exp(np.outer(np.sqrt(T), np.linspace(-0.4, 0.2, 5)))
    surface = models.implied_surface(m, 1.0, K, T)
    assert surface.shape == (2, 5)
    assert np.all(np.isfinite(surface))
    assert np.all(np.diff(surface[:, :3], axis=1) < 0)  # downward skew left of the money
