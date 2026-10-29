import datetime as dt
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import needs_real_snapshot
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
            rows.append(
                pd.DataFrame(
                    {"expiry": expiry, "kind": kind, "strike": K, "bid": p * 0.99, "ask": p * 1.01}
                )
            )
    raw = pd.concat(rows, ignore_index=True)
    return chains.tidy(
        raw,
        underlying="TEST",
        quote_time=quote_time,
        spot=spot,
        source="synthetic",
        rate=rate_column,
        min_price=1e-6,
        moneyness=(0.0, np.inf),
    )


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
    raw = pd.DataFrame(
        {
            "expiry": [dt.date(2026, 10, 8)] * 3,
            "kind": ["call", "call", "put"],
            "strike": [100.0, 400.0, 100.0],
            "bid": [1.0, 0.0, 0.0],
            "ask": [1.2, 0.0, 0.0],
            "last": [np.nan, 0.5, 2.0],
        }
    )
    out = chains.tidy(raw, underlying="X", quote_time=quote_time, spot=100.0, source="test")
    assert len(out) == 1  # far OTM strike and the unquoted put are dropped
    assert out["T"].iloc[0] == pytest.approx(1 / 365)
    assert out["mid"].iloc[0] == pytest.approx(1.1)
    with_last = chains.tidy(
        raw, underlying="X", quote_time=quote_time, spot=100.0, source="test", allow_last=True
    )
    assert with_last["price_source"].tolist() == ["mid", "last"]


def test_synthetic_chain_loads_like_a_snapshot(synthetic_spx):
    snap = synthetic_spx
    assert list(snap.columns) == chains.COLUMNS
    assert set(snap["underlying"]) == {"SPX"}
    assert set(snap["source"]) == {"synthetic"}
    assert set(snap["price_source"]) == {"mid"}
    assert snap["bid"].notna().all() and (snap["bid"] > 0).all()
    assert (snap["ask"] >= snap["bid"]).all()
    assert snap["expiry"].nunique() == 10
    out = chains.market_implied_vols(snap)
    assert out["expiry"].nunique() == 10
    assert out["iv"].between(0.02, 2.0).all()
    # Parity on noisy mids recovers the model's discount factors to a few parts in 100.
    from tests.fixtures import synthetic_spx as fixture

    df = out.groupby("expiry", sort=True)[["T", "df"]].first()
    np.testing.assert_allclose(df["df"], np.exp(-fixture.RATE * df["T"]), atol=0.02)


def test_synthetic_chain_is_what_its_generator_writes(synthetic_spx):
    from tests.fixtures import synthetic_spx as fixture

    rebuilt = fixture.build()
    assert len(rebuilt) == len(synthetic_spx)
    for col in ("strike", "T", "bid", "ask"):
        # Model prices are rounded to a 0.05 tick; platform digits can move one tick.
        np.testing.assert_allclose(rebuilt[col], synthetic_spx[col], atol=0.051)


@needs_real_snapshot
def test_real_snapshot_loads():
    snap = chains.load_snapshot(chains.latest_snapshot("SPX"))
    assert list(snap.columns) == chains.COLUMNS
    assert set(snap["underlying"]) == {"SPX"}
    assert set(snap["source"]) == {"cboe"}
    assert set(snap["price_source"]) == {"mid"}
    assert snap["bid"].notna().all() and (snap["bid"] > 0).all()
    assert (snap["ask"] >= snap["bid"]).all()
    assert len(snap) == 18543
    assert snap["expiry"].nunique() == 40
    assert not np.isfinite(snap["rate"].iloc[0])
    out = chains.market_implied_vols(snap)
    assert out["expiry"].nunique() == 40
    assert out["iv"].between(0.02, 2.0).all()
    # Parity on this file: the front discount factor is above 1, the last is not.
    df = out.groupby("expiry", sort=True)["df"].first()
    assert float(df.iloc[0]) > 1.0
    assert float(df.iloc[-1]) < 1.0


def test_missing_snapshot_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_cboe_spx_snapshot.py"):
        chains.latest_snapshot("SPX", tmp_path)
    with pytest.raises(SystemExit, match="terms of use"):
        chains.require_snapshot("SPX", tmp_path)
    assert not chains.has_snapshot("SPX", tmp_path)


def test_ibkr_refuses_an_index():
    with pytest.raises(ValueError, match="index"):
        chains.fetch_ibkr("SPX")


@pytest.mark.skipif(
    not chains.ib_reachable() or not os.environ.get("IB_TEST_SYMBOL"),
    reason="no IB Gateway, or IB_TEST_SYMBOL is unset (the fetcher qualifies a stock)",
)
def test_ibkr_fetch_schema():
    chain = chains.fetch_ibkr(os.environ["IB_TEST_SYMBOL"], max_years=0.1)
    assert list(chain.columns) == chains.COLUMNS
    assert len(chain) > 0


def _cboe_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_cboe_spx_snapshot.py"
    spec = importlib.util.spec_from_file_location("build_cboe_spx_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cboe_builder_keeps_spxw_and_drops_the_am_root():
    """SPX and SPXW can share a calendar day. The builder keeps the PM root only."""

    def row(symbol, bid=100.0, ask=101.0):
        return {
            "option": symbol,
            "bid": bid,
            "ask": ask,
            "last_trade_price": bid,
            "volume": 1.0,
            "open_interest": 1.0,
        }

    payload = {
        "timestamp": "2026-10-14 10:01:12",
        "data": {
            "current_price": 7805.45,
            "options": [
                row("SPXW261016C07600000"),
                row("SPXW261016P07600000", bid=90.0, ask=91.0),
                row("SPXW261016C08000000"),
                row("SPXW261016P08000000", bid=80.0, ask=81.0),
                row("SPX261016C07600000"),
                row("SPX261016P07600000", bid=90.0, ask=91.0),
                row("SPXW261016C07700000", bid=50.0, ask=40.0),
                row("SPXW261013C07600000"),
            ],
        },
    }
    chain = _cboe_builder().build(payload)
    assert len(chain) == 4
    assert set(chain["underlying"]) == {"SPX"}
    assert set(chain["source"]) == {"cboe"}
    assert chain["expiry"].nunique() == 1
    assert pd.Timestamp(chain["quote_time"].iloc[0]) == pd.Timestamp(
        "2026-10-14 10:01:12", tz="UTC"
    )
    assert not np.isfinite(chain["rate"].iloc[0])
    assert set(chain["strike"]) == {7600.0, 8000.0}


def test_read_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nIB_HOST=10.0.0.5\nIB_PORT='4001'\n")
    monkeypatch.delenv("IB_HOST", raising=False)
    monkeypatch.setenv("IB_PORT", "7497")
    values = chains.read_env(env)
    assert values["IB_HOST"] == "10.0.0.5"
    assert values["IB_PORT"] == "7497"  # the process environment wins
