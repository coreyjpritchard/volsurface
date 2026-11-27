"""Committed docs carry no per-quote market values and no Cboe-derived figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUOTE_COLUMNS = {"bid", "ask", "mid", "price", "market_iv", "residual_spreads"}
# Names that would mean a published market snapshot or its derived files.
FORBIDDEN_STEMS = (
    "p2c_spx_",
    "p2d_snapshot_localvol",
    "p2f_spx_sabr",
    "rvc_spx_term",
)


def test_figure_csvs_have_no_per_quote_columns():
    for path in sorted((ROOT / "docs").rglob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        assert not columns & QUOTE_COLUMNS, path


def test_no_committed_market_snapshot_figure_names():
    docs = ROOT / "docs"
    for path in docs.rglob("*"):
        if path.is_file():
            name = path.name
            assert not name.startswith(FORBIDDEN_STEMS), path
            assert name != "00_the_smile.html", path
