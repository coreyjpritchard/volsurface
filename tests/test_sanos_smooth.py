"""SANOS at eta > 0. DESIGN section 18. Theorem 3.1 and the PCHIP blend."""

import numpy as np
import pytest

from tests.test_sanos_eta0 import _heston_book, _known_surface
from volsurface.arbitrage import check_surface
from volsurface.core.types import ArbTol, Domain
from volsurface.fit.sanos import SanosFitter
from volsurface.surface.sanos import SanosSurface


def _smooth_from(base: SanosSurface, eta: float, time_interp: str = "linear") -> SanosSurface:
    return SanosSurface(
        base.T,
        base.strikes,
        base.weights,
        base.variance,
        eta,
        base.market,
        base.domain,
        time_interp=time_interp,
    )


def test_smooth_call_is_at_least_the_linear_call_at_fixed_weights():
    """Theorem 3.1. Same q, eta > 0 prices sit above eta = 0."""
    base = _known_surface()
    smooth = _smooth_from(base, 0.25)
    log_k = np.linspace(-1.0, 0.6, 31)
    for maturity in (0.5, 0.75, 1.0):
        gap = smooth.call([maturity], log_k) - base.call([maturity], log_k)
        assert float(np.min(gap)) >= -1e-12


def test_density_has_unit_mass_and_unit_mean():
    """At each expiry the mixture integrates to 1 and has mean 1, to 1e-8."""
    base = _known_surface()
    smooth = _smooth_from(base, 0.25)
    for j, maturity in enumerate(smooth.T):
        variance = smooth.eta * float(smooth.variance[j])
        sd = np.sqrt(variance)
        means = np.log(np.maximum(smooth.strikes[j], 1e-300)) - 0.5 * variance
        grid = np.linspace(float(means.min() - 12 * sd), float(means.max() + 12 * sd), 20_001)
        phi = smooth.density([maturity], grid)[0]
        mass = float(np.trapezoid(phi, grid))
        mean = float(np.trapezoid(np.exp(grid) * phi, grid))
        assert mass == pytest.approx(1.0, abs=1e-8)
        assert mean == pytest.approx(1.0, abs=1e-8)
        assert float(np.min(phi)) > 0.0


def test_smoothed_heston_book_has_no_static_arbitrage():
    """check_surface at 1e-9, and the density is positive on the grid."""
    book = _heston_book(0.15)
    result = SanosFitter(eta=0.25, omega=0.0, dx=0.05, time_interp="pchip").fit(book)
    times = np.linspace(0.25, 1.0, 9)
    log_k = np.linspace(-0.35, 0.35, 19)
    report = check_surface(result.surface, times, log_k, tol=ArbTol(price=1e-9))
    assert report.ok
    assert report.violations.empty
    assert report.min_density > 0.0
    assert result.detail["omega_label"] == "calendar guaranteed"


def test_pchip_time_derivative_matches_across_an_expiry_at_the_money():
    """Finite-difference jump of dC/dT at k = 0 is below 1e-4 for h = 1e-4.

    Off the money the blend weight is built from the at-the-money total
    variance, so the same jump is not required to vanish. It is measured.
    """
    weights = np.array([0.25, 0.5, 0.25])
    strikes = np.array([0.5, 1.0, 1.5])
    market = _known_surface().market
    surface = SanosSurface(
        np.array([0.25, 0.5, 1.0]),
        [strikes, strikes, strikes],
        [weights, weights, weights],
        np.array([0.02, 0.04, 0.09]),
        0.25,
        market,
        Domain(0.25, 1.0, float(np.log(0.5)), float(np.log(1.5))),
        time_interp="pchip",
    )
    maturity = 0.5
    step = 1e-4

    def slope(log_k: float, side: str) -> float:
        if side == "left":
            above = float(np.asarray(surface.call([maturity], [log_k])).reshape(-1)[0])
            below = float(np.asarray(surface.call([maturity - step], [log_k])).reshape(-1)[0])
            return (above - below) / step
        above = float(np.asarray(surface.call([maturity + step], [log_k])).reshape(-1)[0])
        below = float(np.asarray(surface.call([maturity], [log_k])).reshape(-1)[0])
        return (above - below) / step

    atm_jump = slope(0.0, "right") - slope(0.0, "left")
    wing_jump = slope(0.2, "right") - slope(0.2, "left")
    assert abs(atm_jump) < 1e-4
    # The wing jump is the measured off-ATM discontinuity. Keep it visible.
    assert abs(wing_jump) > abs(atm_jump)


@pytest.mark.slow
def test_spx_eta_025_meets_the_market_bar():
    """omega = 0, zero violations at 1e-9, in-spread fraction above 0.558."""
    from volsurface import chains
    from volsurface.core.market import market_data_from_snapshot

    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    result = SanosFitter(eta=0.25, omega=0.0, dx=0.05, time_interp="pchip").fit(data)
    assert result.detail["omega_label"] == "calendar guaranteed"
    assert result.in_spread > 0.558
    times = result.surface.T
    mids = 0.5 * (times[:-1] + times[1:])
    grid_t = np.unique(np.concatenate([times, mids]))
    log_k = np.linspace(-0.4, 0.3, 71)
    report = check_surface(result.surface, grid_t, log_k, tol=ArbTol(price=1e-9))
    assert report.ok
    assert report.violations.empty
    assert report.min_density > 0.0


def test_omega_one_is_labelled_not_guaranteed():
    book = _heston_book(0.15)
    result = SanosFitter(eta=0.25, omega=1.0, dx=0.05, method="highs-ipm").fit(book)
    assert result.detail["omega"] == 1.0
    assert result.detail["omega_label"] == "not guaranteed"
    assert "not guaranteed" in result.solver_status
