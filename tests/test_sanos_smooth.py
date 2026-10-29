"""SANOS at eta > 0. DESIGN section 18. Theorem 3.1 and the PCHIP blend."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import needs_real_snapshot
from tests.test_sanos_eta0 import _heston_book, _known_surface
from volsurface.arbitrage import check_surface
from volsurface.core.types import ArbTol, Domain, MarketData
from volsurface.fit.sanos import SanosFitter, quote_in_spread
from volsurface.fit.svi import SVIFitter
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


def test_synthetic_spx_eta_025_is_arbitrage_free_and_mostly_in_spread(synthetic_spx):
    """The market-bar configuration on the synthetic chain (about 0.86 in spread here)."""
    from volsurface.core.market import market_data_from_snapshot

    data = market_data_from_snapshot(synthetic_spx)
    result = SanosFitter(eta=0.25, omega=0.0, dx=0.05, time_interp="pchip").fit(data)
    assert result.detail["omega_label"] == "calendar guaranteed"
    assert result.in_spread > 0.75
    times = result.surface.T
    grid_t = np.unique(np.concatenate([times, 0.5 * (times[:-1] + times[1:])]))
    log_k = np.linspace(-0.4, 0.3, 71)
    report = check_surface(result.surface, grid_t, log_k, tol=ArbTol(price=1e-9))
    assert report.ok
    assert report.violations.empty


@pytest.mark.slow
@needs_real_snapshot
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


def _hold_out_strikes(
    data: MarketData, *, fraction: float = 0.2, seed: int = 0
) -> tuple[MarketData, pd.DataFrame]:
    """Leave out about ``fraction`` of the strikes in each expiry.

    One ``numpy.random.default_rng(seed)`` draws every expiry, in expiry
    order. A strike that rounds to fewer than one quote is kept. The
    returned frame is the held-out quotes.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction}")
    rng = np.random.default_rng(seed)
    quotes = data.quotes
    held_rows: list[int] = []
    for _, group in quotes.groupby("expiry", sort=True):
        strikes = np.unique(group["k"].to_numpy(dtype=float))
        n_hold = int(round(fraction * strikes.size))
        if n_hold < 1:
            continue
        chosen = rng.choice(strikes, size=n_hold, replace=False)
        mask = np.isin(group["k"].to_numpy(dtype=float), chosen)
        held_rows.extend(group.index[mask].tolist())
    held = quotes.loc[held_rows].copy()
    train = quotes.drop(index=held_rows).copy()
    return replace(data, quotes=train.reset_index(drop=True)), held.reset_index(drop=True)


def _dense_grid(surface):
    times = surface.T
    mids = 0.5 * (times[:-1] + times[1:])
    grid_t = np.unique(np.concatenate([times, mids]))
    log_k = np.linspace(-0.4, 0.3, 71)
    return grid_t, log_k


def test_holdout_draws_about_a_fifth_of_each_expiry():
    """Nine strikes, so 20 percent rounds to two. The same seed repeats."""
    book = _heston_book(0.15)
    train, held = _hold_out_strikes(book, fraction=0.2, seed=0)
    again, held_again = _hold_out_strikes(book, fraction=0.2, seed=0)
    assert train.quotes["k"].tolist() == again.quotes["k"].tolist()
    assert held["k"].tolist() == held_again["k"].tolist()
    train_keys = set(
        zip(train.quotes["expiry"], train.quotes["k"], train.quotes["kind"], strict=True)
    )
    held_keys = set(zip(held["expiry"], held["k"], held["kind"], strict=True))
    assert train_keys.isdisjoint(held_keys)
    for expiry, group in book.quotes.groupby("expiry", sort=True):
        n_strikes = int(group["k"].nunique())
        n_held = int(held.loc[held["expiry"] == expiry, "k"].nunique())
        assert n_held == int(round(0.2 * n_strikes))
        assert n_held == 2
    other, _ = _hold_out_strikes(book, fraction=0.2, seed=1)
    assert train.quotes["k"].tolist() != other.quotes["k"].tolist()


def test_heston_holdout_fits_and_prices_the_left_out_quotes():
    """The split is usable: both fitters run, and SANOS stays arbitrage-free."""
    book = _heston_book(0.15)
    train, held = _hold_out_strikes(book, fraction=0.2, seed=0)
    sanos = SanosFitter(eta=0.25, omega=0.0, dx=0.05, time_interp="pchip").fit(train)
    sanos_share = quote_in_spread(held, sanos.surface)
    assert 0.0 <= sanos_share <= 1.0
    times = np.linspace(0.25, 1.0, 9)
    log_k = np.linspace(-0.35, 0.35, 19)
    report = check_surface(sanos.surface, times, log_k, tol=ArbTol(price=1e-9))
    assert report.ok
    assert report.violations.empty


@pytest.mark.slow
@needs_real_snapshot
def test_spx_holdout_meets_the_leave_strikes_out_bar():
    """Held-out SANOS in-spread is at least raw SVI, and violations stay at 0.

    About 20 percent of strikes per expiry, seed 0. The fit is eta 0.25,
    omega 0, the market-bar configuration. Scoring is the same bid-ask
    rule on the quotes that were left out.
    """
    from volsurface import chains
    from volsurface.core.market import market_data_from_snapshot

    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    train, held = _hold_out_strikes(data, fraction=0.2, seed=0)
    sanos = SanosFitter(eta=0.25, omega=0.0, dx=0.05, time_interp="pchip").fit(train)
    svi = SVIFitter().fit(train)
    sanos_share = quote_in_spread(held, sanos.surface)
    svi_share = quote_in_spread(held, svi.surface)
    grid_t, log_k = _dense_grid(sanos.surface)
    report = check_surface(sanos.surface, grid_t, log_k, tol=ArbTol(price=1e-9))
    print(
        f"holdout sanos={sanos_share:.6f} svi={svi_share:.6f} "
        f"n_held={len(held)} n_train={len(train.quotes)} "
        f"violations={len(report.violations)}"
    )
    assert report.violations.empty
    assert sanos_share >= svi_share, f"SANOS {sanos_share} SVI {svi_share}"
    assert sanos.detail["omega_label"] == "calendar guaranteed"


def test_omega_one_is_labelled_not_guaranteed():
    book = _heston_book(0.15)
    result = SanosFitter(eta=0.25, omega=1.0, dx=0.05, method="highs-ipm").fit(book)
    assert result.detail["omega"] == 1.0
    assert result.detail["omega_label"] == "not guaranteed"
    assert "not guaranteed" in result.solver_status
