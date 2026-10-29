"""Sparse and dense quote programs on the same coefficients. DESIGN section 17."""

import json
import re
import subprocess
import sys

import numpy as np
import pytest

from tests.test_arbitrage import _market, _quote_row
from volsurface import chains
from volsurface.arbitrage import check_quotes
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import ArbTol, MarketData


def _objective(report) -> float:
    match = re.search(r"Objective \(total slack\) ([0-9.eE+-]+)\.", report.note)
    assert match, report.note
    return float(match.group(1))


def _book(n_expiries: int) -> MarketData:
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    expiries = list(data.quotes.groupby("expiry", sort=True).groups)[:n_expiries]
    subset = data.quotes[data.quotes["expiry"].isin(expiries)].copy()
    return MarketData(
        asof=data.asof,
        quotes=subset,
        F=data.F,
        df=data.df,
        F_err=data.F_err,
        day_count=data.day_count,
    )


def _same(sparse, dense) -> None:
    assert sparse.n_checked == dense.n_checked
    assert sparse.ok is True
    assert dense.ok is True
    assert _objective(sparse) == pytest.approx(0.0, abs=1e-8)
    assert _objective(dense) == pytest.approx(0.0, abs=1e-8)
    assert sparse.violations.empty
    assert dense.violations.empty


def test_sparse_matches_dense_on_two_expiries():
    """278 quotes, objective 0, the same empty violation set."""
    book = _book(2)
    sparse = check_quotes(book)
    dense = check_quotes(book, dense=True)
    assert sparse.n_checked == 278
    _same(sparse, dense)


def test_sparse_matches_dense_on_eight_expiries():
    """1,685 quotes. Dense peak RSS on this book is about 1.2 GB."""
    book = _book(8)
    sparse = check_quotes(book)
    dense = check_quotes(book, dense=True)
    assert sparse.n_checked == 1685
    assert sparse.n_checked == len(book.quotes)
    _same(sparse, dense)


@pytest.mark.slow
def test_sparse_matches_dense_on_the_first_24_expiries():
    """4,398 quotes. Dense peak RSS on this book reached 6.2 GB, so it is not fast."""
    book = _book(24)
    sparse = check_quotes(book)
    dense = check_quotes(book, dense=True)
    assert sparse.n_checked == 4398
    _same(sparse, dense)


def test_seeded_ask_is_the_gap_below_intrinsic():
    """An ask of 0.1 at normalised strike 0.5 cannot clear the intrinsic 0.5."""
    row = _quote_row(0.5, 0.5, 0.0, 0.1)
    report = check_quotes(_market([row]), side="bidask", tol=ArbTol(price=1e-8))
    assert report.ok is False
    hit = report.violations[report.violations["kind"] == "quote_ask"]
    assert len(hit) == 1
    assert hit["k"].iloc[0] == pytest.approx(float(np.log(0.5)))
    assert hit["magnitude"].iloc[0] == pytest.approx(0.4, abs=1e-6)
    assert report.violations[report.violations["kind"] == "quote_bid"].empty


@pytest.mark.slow
def test_all_forty_expiries_return_a_status_under_2gb():
    """Peak RSS of a fresh process on the full SPX book stays below 2 GB."""
    code = """
import json, resource, time
from volsurface import chains
from volsurface.arbitrage import check_quotes
from volsurface.core.market import market_data_from_snapshot

t0 = time.perf_counter()
chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
data = market_data_from_snapshot(chain)
report = check_quotes(data)
wall = time.perf_counter() - t0
# ru_maxrss survives fork/exec on Linux, so a child of a large pytest process
# would report the parent's peak. VmHWM is per address space and resets at exec.
rss_gb = None
try:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                rss_gb = int(line.split()[1]) / 1024**2
except OSError:
    pass
if rss_gb is None:
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(json.dumps({
    "ok": report.ok,
    "n_checked": report.n_checked,
    "note": report.note,
    "n_violations": int(len(report.violations)),
    "wall_s": wall,
    "rss_gb": rss_gb,
}))
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    payload = json.loads(out.strip().splitlines()[-1])
    assert "HiGHS status" in payload["note"]
    assert payload["n_checked"] > 4398
    assert payload["rss_gb"] < 2.0
