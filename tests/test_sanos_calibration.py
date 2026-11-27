"""Tests for the SANOS-calibrated market export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from volsurface import bs
from volsurface.fit.sanos_calibration import (
    NOTE,
    SITE_LABEL,
    SITE_SCHEMA,
    TRAIN_SCHEMA,
    atm_level_and_skew,
    calibrate_from_grid,
    interpolate_iv,
    load_grid,
    site_document,
    write_grid,
    write_json,
)
from volsurface.models.heston import Heston


def _forbidden() -> tuple[str, ...]:
    return (
        "SP" + "X",
        "SP" + "XW",
        "Cb" + "oe",
        "CB" + "OE",
        "2026-10-" + "14",
        "14 " + "October",
        "14 " + "Oct",
    )

TRUTH = Heston(v0=0.03, kappa=1.2, theta=0.05, xi=0.4, rho=-0.6)


def _synthetic_grid():
    T = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
    k = np.linspace(-0.3, 0.3, 9)
    iv = np.full((T.size, k.size), np.nan)
    strikes = 100.0 * np.exp(k)
    for i, maturity in enumerate(T):
        prices = np.asarray(
            TRUTH.price(100.0, strikes, float(maturity), df=1.0, kind="call"),
            dtype=float,
        )
        iv[i] = bs.implied_vol(prices, 100.0, strikes, float(maturity), df=1.0, kind="call")
    return T, k, iv


def test_load_grid_refuses_dates(tmp_path: Path):
    T, k, iv = _synthetic_grid()
    path = tmp_path / "grid.json"
    write_grid(path, T, k, iv)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dates"] = ["not-allowed"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dates"):
        load_grid(bad)


def test_interpolate_recovers_a_grid_node():
    T, k, iv = _synthetic_grid()
    mid_t, mid_k = T[2], k[4]
    got = interpolate_iv(T, k, iv, float(mid_t), float(mid_k))
    assert got == pytest.approx(float(iv[2, 4]), rel=1e-12, abs=1e-12)


def test_synthetic_calibration_recovers_known_parameters():
    T, k, iv = _synthetic_grid()
    train = calibrate_from_grid(T, k, iv, seed=1, n_starts=4, maturity=0.25)
    heston = train["heston"]
    assert train["schema"] == TRAIN_SCHEMA
    assert heston["v0"] == pytest.approx(TRUTH.v0, abs=1e-6)
    assert heston["kappa"] == pytest.approx(TRUTH.kappa, abs=1e-5)
    assert heston["theta"] == pytest.approx(TRUTH.theta, abs=1e-6)
    assert heston["xi"] == pytest.approx(TRUTH.xi, abs=1e-5)
    assert heston["rho"] == pytest.approx(TRUTH.rho, abs=1e-5)
    assert heston["rmse_iv"] < 1e-4
    sigma, _skew = atm_level_and_skew(T, k, iv, 0.25)
    assert train["black_scholes"]["sigma"] == pytest.approx(sigma, rel=1e-12)
    assert train["contract"]["s0"] == 100.0
    assert train["contract"]["maturity"] == pytest.approx(0.25)


def test_site_document_omits_parameters_and_forbidden_names(tmp_path: Path):
    T, k, iv = _synthetic_grid()
    train = calibrate_from_grid(T, k, iv, seed=1, n_starts=2, maturity=0.25)
    site = site_document(train)
    assert site["schema"] == SITE_SCHEMA
    assert site["label"] == SITE_LABEL
    assert site["note"] == NOTE
    text = json.dumps(site)
    for key in ("v0", "kappa", "theta", "xi", "rho", "iv", "T", "k", "dates", "expiry"):
        assert key not in site
        if key in ("v0", "kappa", "theta", "xi", "rho", "dates", "expiry"):
            assert f'"{key}"' not in text
    assert "black_scholes" in site["markets"]
    assert "heston" in site["markets"]
    assert "premium" in site["markets"]["heston"]
    path = tmp_path / "site.json"
    write_json(path, site)
    dumped = path.read_text(encoding="utf-8")
    for token in _forbidden():
        assert token not in dumped


def test_new_source_files_avoid_forbidden_names():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "volsurface" / "fit" / "sanos_calibration.py",
        root / "scripts" / "sanos_calibration.py",
        Path(__file__),
        root / "data" / "exports" / "README.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in _forbidden():
            assert token not in text, f"{token} in {path}"
