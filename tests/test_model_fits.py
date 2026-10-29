"""The fitted environments of the vol lab, rebuilt from ``data/presets/spx_model_fits.json``."""

from __future__ import annotations

import io

import numpy as np
import pytest

from volsurface import environments as env_lib
from volsurface.models.sabr import hagan_implied_vol

pytestmark = pytest.mark.skipif(
    not (env_lib.MODEL_FITS_PATH.exists() and env_lib.PRESET_PATH.exists()),
    reason="fitted presets not built",
)


@pytest.fixture(scope="module")
def fits():
    return env_lib.load_model_fits()


@pytest.fixture(scope="module")
def spx():
    return env_lib.load_spx_presets()


@pytest.fixture(scope="module")
def market():
    # Rebuilt from the snapshot: the preset files hold no quotes.
    quotes = env_lib.load_market_quotes()
    if quotes is None:
        pytest.skip("no SPX snapshot in data/snapshots")
    return quotes


def test_preset_files_hold_fitted_parameters_not_quotes(fits):
    # No per-quote array: nothing in the presets has the length of the quote set.
    n = fits["data"]["n_quotes"]
    saved = np.load(env_lib.PRESET_PATH, allow_pickle=False)
    assert not any(name.startswith("market") for name in saved.files)
    assert all(saved[name].size < n for name in saved.files)
    assert all(
        len(v) < n for m in fits["models"].values() if m for v in m.values() if isinstance(v, list)
    )


def test_fits_file_has_every_model_and_describes_the_data(fits, market):
    assert set(fits["models"]) >= {"flat", "heston", "sabr", "svi", "sanos", "rbergomi"}
    data = fits["data"]
    assert data["underlying"].startswith("SPX")
    assert data["n_quotes"] == len(market["T"]) and data["n_expiries"] == 40
    assert len(data["forward"]) == len(data["discount_factor"]) == data["n_expiries"]
    np.testing.assert_allclose(data["expiry_T"], np.unique(market["T"]))


def test_load_model_fits_reads_bytes_like_the_web_build(fits):
    raw = env_lib.MODEL_FITS_PATH.read_bytes()
    assert env_lib.load_model_fits(io.BytesIO(raw)) == fits


def test_flat_fit_is_the_mean_vol_and_its_rmse_the_spread(fits, market):
    flat = fits["models"]["flat"]
    sigma = flat["params"]["sigma"]
    # The quotes' vols are re-inverted from the snapshot's prices, so the last digits can
    # differ by platform (a few 1e-6 here); 1e-5 of vol is far below a quote's spread.
    assert sigma == pytest.approx(np.mean(market["iv"]), abs=1e-5)
    assert flat["rmse_vol_points"] == pytest.approx(100 * np.std(market["iv"]), abs=1e-3)


@pytest.mark.parametrize("key", ["flat", "sabr", "svi", "sanos", "heston"])
def test_stored_rmse_is_what_the_rebuilt_environment_gives(fits, spx, market, key):
    m = fits["models"][key]
    if key == "flat":
        env = env_lib.flat_environment(m["params"]["sigma"])
    elif key == "heston":
        env = env_lib.heston_environment(**m["params"])
    elif key == "sabr":
        env = env_lib.fitted_sabr_environment(fits)
    else:
        env = spx[f"spx_{key}"]
    q = env_lib.fit_quality(env, market)
    if key == "heston":
        # Heston vols come from inverting Fourier prices; on the 10-hour expiry the far
        # wings are tiny and the last digits (and a few usable quotes) differ by platform.
        assert q.rmse == pytest.approx(m["rmse_vol_points"], abs=0.02)
        assert abs(q.n_used - m["n_used"]) <= 20
    else:
        assert q.rmse == pytest.approx(m["rmse_vol_points"], abs=1e-3)
        assert q.n_used == m["n_used"]
    assert q.n_quotes == m["n_quotes"]


def test_heston_fit_reports_xi_on_its_bound(fits):
    h = fits["models"]["heston"]
    assert h["at_bound"] == ["xi"]
    assert h["params"]["xi"] == pytest.approx(h["bounds"]["xi"][1], abs=1e-3)


def test_fitted_models_beat_one_flat_vol(fits):
    m = fits["models"]
    flat = m["flat"]["rmse_vol_points"]
    for key in ("heston", "sabr", "svi", "sanos"):
        assert m[key]["rmse_vol_points"] < flat / 2
    assert m["sanos"]["rmse_vol_points"] < m["svi"]["rmse_vol_points"] < 1.0


def test_sabr_slices_are_hagan_at_the_pillars_and_linear_in_variance_between(fits):
    s = fits["models"]["sabr"]
    env = env_lib.fitted_sabr_environment(fits)
    surface = env.surface
    k = np.linspace(-0.2, 0.1, 7)
    for j in (5, 20, 39):
        T = s["T"][j]
        hagan = hagan_implied_vol(1.0, np.exp(k), T, s["alpha"][j], 1.0, s["rho"][j], s["nu"][j])
        np.testing.assert_allclose(surface.iv([T], k)[0], hagan, rtol=1e-12)
    t0, t1 = s["T"][20], s["T"][21]
    mid = 0.5 * (t0 + t1)
    w = surface.total_variance([t0, mid, t1], k)
    np.testing.assert_allclose(w[1], 0.5 * (w[0] + w[2]), rtol=1e-12)


def test_fitted_grid_rows_are_quoted_expiries(fits):
    expiry_T = np.asarray(fits["data"]["expiry_T"])
    grid = env_lib.fitted_grid(expiry_T, n_T=10)
    assert 5 <= grid.T.size <= 10
    assert np.all(np.isin(grid.T, expiry_T))
    assert grid.T.min() >= 1.0 / 365.0 and grid.T.max() == expiry_T.max()


def test_rbergomi_environment_needs_exactly_one_variance_input():
    with pytest.raises(ValueError):
        env_lib.rbergomi_environment(0.1, 1.5, -0.7)
    with pytest.raises(ValueError):
        env_lib.rbergomi_environment(0.1, 1.5, -0.7, sigma0=0.2, xi0=lambda t: 0.04)


def test_rbergomi_fit_has_free_h_on_the_sanos_forward_variance(fits):
    rb = fits["models"]["rbergomi"]
    if rb is None:
        pytest.skip("rough-volatility extra was not installed when the fits were built")
    assert 0.0 < rb["params"]["H"] < 0.5 and -1.0 < rb["params"]["rho"] < 0.0
    assert rb["n_paths"] == 4000 and rb["seed"] == 0


@pytest.mark.slow
def test_sabr_fit_reruns_to_the_stored_parameters(fits):
    from volsurface import chains
    from volsurface.core.market import market_data_from_snapshot
    from volsurface.fit.sabr import SABRFitter

    snapshot = chains.load_snapshot(chains.latest_snapshot("SPX"))
    report = SABRFitter(beta=1.0, n_starts=4, seed=0).fit(market_data_from_snapshot(snapshot))
    s = fits["models"]["sabr"]
    for name in ("alpha", "rho", "nu"):
        np.testing.assert_allclose(report.slice_report[name], s[name], rtol=1e-6, atol=1e-8)
