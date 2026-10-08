import datetime as dt

import numpy as np
import pandas as pd
import pytest

from volsurface import bs, chains


def synthetic_chain(spot=500.0, r=0.04, q=0.012, rate_column=np.nan):
    """A parity-consistent chain priced by Black-Scholes with a smile, in the tidy schema."""
    quote_time = pd.Timestamp("2026-10-07 20:00", tz="UTC")
    rows = []
    for days in (7, 30, 91, 365):
        expiry = (quote_time + pd.Timedelta(days=days)).date()
        T = (chains.expiry_close_utc(expiry) - quote_time).total_seconds() / (365 * 86400)
        F, df = spot * np.exp((r - q) * T), np.exp(-r * T)
        K = np.round(F * np.exp(np.linspace(-0.3, 0.15, 61) * np.sqrt(T + 0.05)), 0)
        sigma = 0.15 - 0.3 * np.log(K / F) + 0.5 * np.log(K / F) ** 2
        for kind in ("call", "put"):
            p = bs.price(F, K, T, sigma, df, kind)
            rows.append(pd.DataFrame({"expiry": expiry, "kind": kind, "strike": K,
                                      "bid": p * 0.99, "ask": p * 1.01}))
    raw = pd.concat(rows, ignore_index=True)
    return chains.tidy(raw, underlying="TEST", quote_time=quote_time, spot=spot,
                       source="synthetic", rate=rate_column, min_price=1e-6,
                       moneyness=(0.0, np.inf))


def test_infer_forward_exact():
    K = np.arange(90.0, 111.0)
    F, df = 101.3, 0.982
    C = bs.price(F, K, 0.5, 0.2, df, "call")
    P = bs.price(F, K, 0.5, 0.2, df, "put")
    F_hat, df_hat = chains.infer_forward(K, C, P, spot=100.0)
    assert F_hat == pytest.approx(F, rel=1e-10)
    assert df_hat == pytest.approx(df, rel=1e-10)


@pytest.mark.parametrize("rate_column", [np.nan, 0.04])
def test_market_implied_vols_recovers_forward_and_smile(rate_column):
    r, q, spot = 0.04, 0.012, 500.0
    chain = synthetic_chain(spot, r, q, rate_column)
    assert list(chain.columns) == chains.COLUMNS
    out = chains.market_implied_vols(chain)
    assert out["expiry"].nunique() == 4
    for _, g in out.groupby("expiry"):
        T = g["T"].iloc[0]
        assert g["F"].iloc[0] == pytest.approx(spot * np.exp((r - q) * T), rel=1e-9)
        assert g["df"].iloc[0] == pytest.approx(np.exp(-r * T), rel=1e-9)
        expected = 0.15 - 0.3 * g["k"] + 0.5 * g["k"] ** 2
        np.testing.assert_allclose(g["iv"], expected, atol=1e-7)


def test_tidy_time_to_expiry_and_filters():
    quote_time = pd.Timestamp("2026-10-07 20:00", tz="UTC")  # 16:00 New York
    raw = pd.DataFrame({
        "expiry": [dt.date(2026, 10, 8)] * 3,
        "kind": ["call", "call", "put"],
        "strike": [100.0, 400.0, 100.0],
        "bid": [1.0, 0.0, 0.0],
        "ask": [1.2, 0.0, 0.0],
        "last": [np.nan, 0.5, 2.0],
    })
    out = chains.tidy(raw, underlying="X", quote_time=quote_time, spot=100.0, source="test")
    assert len(out) == 1  # far OTM strike and the unquoted put are dropped
    assert out["T"].iloc[0] == pytest.approx(1 / 365)
    assert out["mid"].iloc[0] == pytest.approx(1.1)
    with_last = chains.tidy(raw, underlying="X", quote_time=quote_time, spot=100.0,
                            source="test", allow_last=True)
    assert with_last["price_source"].tolist() == ["mid", "last"]


def test_committed_snapshot_loads():
    snap = chains.load_snapshot(chains.latest_snapshot("SPY"))
    assert list(snap.columns) == chains.COLUMNS
    out = chains.market_implied_vols(snap)
    assert out["expiry"].nunique() >= 8
    assert out["iv"].between(0.02, 2.0).all()


@pytest.mark.skipif(not chains.ib_reachable(), reason="no IB Gateway or TWS reachable")
def test_ibkr_fetch_schema():
    chain = chains.fetch_ibkr("SPY", max_years=0.1)
    assert list(chain.columns) == chains.COLUMNS
    assert len(chain) > 0


def test_read_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nIB_HOST=10.0.0.5\nIB_PORT='4001'\n")
    monkeypatch.delenv("IB_HOST", raising=False)
    monkeypatch.setenv("IB_PORT", "7497")
    values = chains.read_env(env)
    assert values["IB_HOST"] == "10.0.0.5"
    assert values["IB_PORT"] == "7497"  # the process environment wins
