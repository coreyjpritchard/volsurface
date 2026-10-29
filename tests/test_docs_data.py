"""Committed docs carry no per-quote Cboe values: summaries per expiry only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUOTE_COLUMNS = {"bid", "ask", "mid", "price", "market_iv", "residual_spreads"}


def _strip_module():
    path = ROOT / "scripts" / "strip_export_quotes.py"
    spec = importlib.util.spec_from_file_location("strip_export_quotes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure_csvs_have_no_per_quote_columns():
    for path in sorted((ROOT / "docs").rglob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        assert not columns & QUOTE_COLUMNS, path


def test_smile_export_has_no_per_quote_market_points():
    strip = _strip_module()
    page = (ROOT / "docs" / "00_the_smile.html").read_text(encoding="utf-8")
    figures = strip.figures(page)
    assert len(figures) == 3
    assert not any(strip.is_per_quote(f, t) for f in figures for t in f["data"])
    assert strip.strip(page) == page


def test_strip_drops_quotes_but_keeps_the_skew_points():
    strip = _strip_module()
    smile = {"layout": {"title": {"text": "Smile, 30 days"}}, "data": [{"name": "SPX"}]}
    skew = {"layout": {"title": {"text": "At-the-money skew"}}, "data": [{"name": "SPX"}]}
    assert strip.is_per_quote(smile, smile["data"][0])
    assert not strip.is_per_quote(skew, skew["data"][0])
