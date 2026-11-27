"""Tests for the SANOS-calibrated market export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from volsurface import bs
from volsurface.fit.sanos_calibration import (
    KAPPA_UPPER_DEFAULT,
    KAPPA_UPPER_WIDE,
    NOTE,
    SITE_LABEL,
    SITE_SCHEMA,
    SYNTHETIC_HESTON,
    SYNTHETIC_NOTE,
    TRAIN_SCHEMA,
    atm_level_and_skew,
    bound_hits,
    calibrate_from_grid,
    contract_weights,
    heston_specs,
    interpolate_iv,
    load_grid,
    site_document,
    synthetic_example_document,
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


TRUTH = SYNTHETIC_HESTON


def _synthetic_grid(model: Heston = TRUTH, T=None, k=None):
    T = np.asarray([0.05, 0.1, 0.25, 0.5, 1.0] if T is None else T, dtype=float)
    k = np.linspace(-0.3, 0.3, 9) if k is None else np.asarray(k, dtype=float)
    iv = np.full((T.size, k.size), np.nan)
    strikes = 100.0 * np.exp(k)
    for i, maturity in enumerate(T):
        prices = np.asarray(
            model.price(100.0, strikes, float(maturity), df=1.0, kind="call"),
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


def test_contract_weights_peak_at_the_hedged_maturity():
    T = np.array([0.05, 30.0 / 365.0, 1.0])
    weights = contract_weights(T, 30.0 / 365.0)
    assert weights[1] == pytest.approx(1.0)
    assert weights[1] > weights[0] > weights[2]


def test_synthetic_calibration_recovers_known_parameters():
    T, k, iv = _synthetic_grid()
    train = calibrate_from_grid(T, k, iv, seed=1, n_starts=4, maturity=0.25)
    heston = train["heston"]
    full = train["heston_fits"]["full"]
    contract = train["heston_fits"]["contract"]
    assert train["schema"] == TRAIN_SCHEMA
    assert heston is contract
    for fit in (full, contract):
        assert fit["v0"] == pytest.approx(TRUTH.v0, abs=1e-6)
        assert fit["kappa"] == pytest.approx(TRUTH.kappa, abs=1e-5)
        assert fit["theta"] == pytest.approx(TRUTH.theta, abs=1e-6)
        assert fit["xi"] == pytest.approx(TRUTH.xi, abs=1e-5)
        assert fit["rho"] == pytest.approx(TRUTH.rho, abs=1e-5)
        assert fit["rmse_iv"] < 1e-4
        assert fit["bound_hits"] == []
    assert full["kappa_upper"] == KAPPA_UPPER_DEFAULT
    assert contract["kappa_upper"] == KAPPA_UPPER_WIDE
    assert contract["weighting"] == "contract"
    assert full["weighting"] == "full"
    sigma, _skew = atm_level_and_skew(T, k, iv, 0.25)
    assert train["black_scholes"]["sigma"] == pytest.approx(sigma, rel=1e-12)
    assert train["contract"]["s0"] == 100.0
    assert train["contract"]["maturity"] == pytest.approx(0.25)


def test_wide_kappa_recovers_a_parameter_outside_the_default_box():
    truth = Heston(v0=0.04, kappa=30.0, theta=0.04, xi=0.8, rho=-0.5)
    T, k, iv = _synthetic_grid(truth, T=[0.08, 0.25, 0.5, 1.0], k=np.linspace(-0.2, 0.2, 7))
    train = calibrate_from_grid(T, k, iv, seed=0, n_starts=4, maturity=0.08)
    boxed = train["heston_fits"]["full"]
    wide = train["heston_fits"]["contract"]
    assert "kappa" in boxed["bound_hits"]
    assert boxed["kappa"] == pytest.approx(KAPPA_UPPER_DEFAULT, rel=0.02)
    assert wide["kappa"] == pytest.approx(30.0, rel=0.05, abs=0.5)
    assert "kappa" not in wide["bound_hits"]


def test_bound_hits_detects_the_numerical_box():
    specs = heston_specs(kappa_upper=20.0, xi_upper=5.0)
    params = {"v0": 0.04, "kappa": 20.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}
    assert bound_hits(params, specs) == ["kappa"]
    params["kappa"] = 1.5
    assert bound_hits(params, specs) == []


def test_site_document_omits_parameters_and_forbidden_names(tmp_path: Path):
    T, k, iv = _synthetic_grid()
    train = calibrate_from_grid(T, k, iv, seed=1, n_starts=2, maturity=0.25)
    site = site_document(train)
    assert site["schema"] == SITE_SCHEMA
    assert site["label"] == SITE_LABEL
    assert site["note"] == NOTE
    assert site["synthetic"] is False
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


def test_synthetic_example_is_labelled_and_matches_the_model():
    example = synthetic_example_document()
    assert example["synthetic"] is True
    assert example["note"] == SYNTHETIC_NOTE
    assert example["schema"] == SITE_SCHEMA
    priced = float(
        np.asarray(SYNTHETIC_HESTON.price(100.0, 100.0, 30.0 / 365.0, df=1.0, kind="call")).reshape(
            ()
        )
    )
    assert example["markets"]["heston"]["premium"] == pytest.approx(priced)
    text = json.dumps(example)
    for token in _forbidden():
        assert token not in text
    for key in ("v0", "kappa", "theta", "xi", "rho"):
        assert f'"{key}"' not in text


def test_committed_example_matches_the_synthetic_document():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "exports" / "examples" / "synthetic_snapshot_markets.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == synthetic_example_document()
    assert committed["synthetic"] is True


def test_new_source_files_avoid_forbidden_names():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "volsurface" / "fit" / "sanos_calibration.py",
        root / "scripts" / "sanos_calibration.py",
        Path(__file__),
        root / "data" / "exports" / "README.md",
        root / "data" / "exports" / "examples" / "synthetic_snapshot_markets.json",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in _forbidden():
            assert token not in text, f"{token} in {path}"
