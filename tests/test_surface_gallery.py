"""The surface gallery runs on a small grid and marks strikes past the quotes.

The full SPX figures are produced by ``python docs/figures/surfaces_gallery.py``.
This test uses two expiries and a handful of strikes so the fits stay small.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from volsurface import chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import MarketData

ROOT = Path(__file__).resolve().parents[1]


def _gallery():
    path = ROOT / "docs" / "figures" / "surfaces_gallery.py"
    spec = importlib.util.spec_from_file_location("surfaces_gallery", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_clip_mask_blanks_cells_outside_the_window():
    """A cell outside the locked window is marked and is not drawn at its raw height."""
    gallery = _gallery()
    z = np.array([[0.20, 3.00], [0.01, 0.40]])
    extrapolated = np.array([[False, True], [False, False]])
    maturities = np.array([0.10, 0.20])
    k = np.array([-0.10, 0.10])
    mask = gallery.clipped_mask(z, 0.05, 0.90)
    assert mask.tolist() == [[False, True], [True, False]]
    drawn = gallery.layer_z(z, ~mask)
    assert drawn[0, 0] == 0.20 and drawn[1, 1] == 0.40
    assert np.isnan(drawn[0, 1]) and np.isnan(drawn[1, 0])
    report = gallery.clip_report(z, extrapolated, maturities, k, 0.05, 0.90)
    assert report["n_clipped"] == 2
    assert report["n_above"] == 1 and report["n_below"] == 1
    assert report["n_clipped_extrapolated"] == 1
    assert report["n_clipped_inside"] == 1
    assert report["worst"][0]["z"] == 3.0
    assert report["worst"][0]["extrapolated"]


def test_display_grid_drops_short_expiries_and_unquoted_wings():
    """Figures use k/sqrt(T) and do not draw past an expiry's own quotes."""
    gallery = _gallery()
    pillars = [
        (0.01, np.array([-0.20, 0.0, 0.10]), np.array([0.30, 0.20, 0.25])),
        (0.25, np.array([-0.20, 0.0, 0.10]), np.array([0.20, 0.16, 0.17])),
    ]
    m = np.array([-3.0, -0.40, 0.0, 1.5])
    kept = gallery.pillars_for_display(pillars)
    assert [row[0] for row in kept] == [0.25]
    mask = gallery.quote_mask(kept, m)
    # k = m * sqrt(0.25). Quotes cover k in [-0.20, 0.10], so m in [-0.40, 0.20].
    assert mask.shape == (1, 4)
    assert mask[0].tolist() == [False, True, True, False]
    raw = gallery.market_on_moneyness(kept, m)
    assert np.isnan(raw[0, 0]) and np.isnan(raw[0, 3])
    assert np.isfinite(raw[0, 1]) and np.isfinite(raw[0, 2])
    smooth = gallery.smooth_along_moneyness(raw, sigma_steps=1.0)
    assert np.isnan(smooth[0, 0]) and np.isnan(smooth[0, 3])
    assert np.isfinite(smooth[0, 2])
    axis = gallery.standardised_axis()
    assert axis[0] == -3.0 and axis[-1] == 1.5
    assert gallery.DISPLAY_MIN_DAYS == 7


def test_error_colour_limit_ignores_a_handful_of_large_cells():
    """The heatmap scale is a high percentile of |error|, not the maximum."""
    gallery = _gallery()
    abs_errors = np.concatenate([np.full(98, 1.0), np.array([50.0, 50.0])])
    limit = gallery.error_colour_limit(abs_errors, percentile=98)
    assert 1.0 <= limit < 5.0
    assert limit < float(np.max(abs_errors))


def test_extrapolation_mask_is_outside_each_expiry():
    gallery = _gallery()
    pillars = [
        (0.25, np.array([-0.20, -0.10, 0.0, 0.10]), np.array([0.2, 0.18, 0.16, 0.17])),
        (0.50, np.array([-0.05, 0.0, 0.05]), np.array([0.15, 0.14, 0.15])),
    ]
    k = np.array([-0.20, -0.10, 0.0, 0.10])
    mask = gallery.extrapolation_mask(pillars, k)
    assert mask.shape == (2, 4)
    assert not mask[0].any()
    assert mask[1, 0] and mask[1, 1] and mask[1, 3]
    assert not mask[1, 2]


def _small_book() -> MarketData:
    """Two expiries, six strikes near the money, taken from the committed SPX file."""
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    chosen = []
    for _expiry, group in data.quotes.groupby("expiry", sort=True):
        maturity = float(group["T"].iloc[0])
        if maturity < 0.2:
            continue
        group = group.drop_duplicates("k").sort_values("k")
        centre = int(np.argmin(np.abs(group["k"].to_numpy(dtype=float))))
        window = group.iloc[max(0, centre - 3) : centre + 3]
        if len(window) >= 6 and np.all(np.abs(window["k"]) <= 0.4):
            chosen.append(window.iloc[:6])
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise AssertionError("the snapshot did not yield two six-strike expiries")
    quotes = pd.concat(chosen, ignore_index=True)
    return MarketData(
        asof=data.asof,
        quotes=quotes,
        F=data.F,
        df=data.df,
        F_err=data.F_err,
        day_count=data.day_count,
    )


def test_gallery_script_runs_on_a_small_grid(tmp_path):
    """Fits, surfaces, PNG and HTML on two expiries and five log-moneyness nodes."""
    gallery = _gallery()
    data = _small_book()
    numbers = gallery.build_and_write(
        data,
        fig_dir=tmp_path / "figures",
        html_dir=tmp_path / "surfaces",
        n_k=5,
        n_paths=64,
        seed=2,
        heston_starts=1,
        sabr_starts=1,
    )
    assert numbers["n_expiries"] == 2
    assert numbers["n_k"] == 5
    assert numbers["flat_sigma"] > 0.0
    assert numbers["black_scholes"]["span"] == 0.0
    assert numbers["dupire_localvol"]["n_finite_inside"] > 0
    assert numbers["gbm_mc"]["n_missing"] < numbers["n_expiries"] * numbers["n_k"]
    assert numbers["display"]["coordinate"] == "k/sqrt(T)"
    assert numbers["display"]["m_min"] == -3.0
    assert numbers["display"]["m_max"] == 1.5
    assert numbers["display"]["min_days"] == 7
    assert numbers["display"]["n_expiries_drawn"] == 2
    assert numbers["display"]["n_expiries_omitted"] == 0
    assert numbers["display"]["implied_vol_axis"] == [0.05, 0.9]
    local = numbers["display"]["clips"]["dupire_localvol"]
    assert local["n_clipped"] == 0
    assert local["raw_max"] is None or local["hi"] >= local["raw_max"]
    heat = numbers["display"]["error_heatmap"]
    assert heat["percentile"] == 98
    assert heat["limit_vol_points"] > 0.0
    assert heat["panels"]["svi"]["n"] > 0
    assert heat["panels"]["svi"]["rmse_vol_points"] > 0.0
    assert (
        numbers["display"]["clips"]["black_scholes"]["n_finite"]
        == numbers["display"]["clips"]["market"]["n_finite"]
    )
    assert (
        numbers["display"]["clips"]["gbm_mc"]["n_finite"]
        == numbers["display"]["clips"]["market"]["n_finite"]
    )

    fig_dir = tmp_path / "figures"
    html_dir = tmp_path / "surfaces"
    for key in (
        "market",
        "black_scholes",
        "gbm_mc",
        "svi",
        "heston",
        "sabr",
        "dupire_localvol",
    ):
        png = fig_dir / f"p2_surface_{key}.png"
        html = html_dir / f"{key}.html"
        assert png.is_file() and png.stat().st_size > 1000
        assert html.is_file() and html.stat().st_size > 1000
        text = html.read_text(encoding="utf-8")
        assert "plotly" in text
        assert 'name="viewport"' in text
        assert '"autorange":false' in text or '"autorange": false' in text
        assert "7 days" in text
        assert "standardised moneyness" in text.lower()
        if key == "market":
            assert "raw quotes" in text
        if key != "dupire_localvol":
            assert "0.90" in text
    assert (fig_dir / "p2_surface_gallery.png").stat().st_size > 1000
    assert (fig_dir / "p2_surface_iv_error.png").stat().st_size > 1000
    local = html_dir / "dupire_localvol.html"
    assert "local volatility" in local.read_text(encoding="utf-8")
    assert "not implied vol" in local.read_text(encoding="utf-8")
