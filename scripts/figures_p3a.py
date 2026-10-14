"""Peak memory and wall time of the quote linear program, dense against sparse.

Run from the repository root:

    python scripts/figures_p3a.py

Each point is a fresh process, so the resident-set high water mark is that
run and not the sum of earlier runs. Dense stops at 24 expiries: the next
sizes are the ones the 15.6 GB kill was about.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

# Dense at 24 expiries peaked near 6 GB in the run that fixed the counts.
# 32 and 40 are left to the sparse solver.
DENSE_N = (2, 4, 8, 12, 16, 24)
SPARSE_N = (2, 4, 8, 12, 16, 24, 32, 40)


def _one(n_expiries: int, dense: bool) -> dict:
    code = f"""
import json, resource, time
from volsurface import chains
from volsurface.arbitrage import check_quotes
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import MarketData

t0 = time.perf_counter()
chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
data = market_data_from_snapshot(chain)
expiries = list(data.quotes.groupby("expiry", sort=True).groups)[:{n_expiries}]
subset = data.quotes[data.quotes["expiry"].isin(expiries)].copy()
book = MarketData(
    asof=data.asof, quotes=subset, F=data.F, df=data.df,
    F_err=data.F_err, day_count=data.day_count,
)
report = check_quotes(book, dense={dense})
wall = time.perf_counter() - t0
rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(json.dumps({{
    "n_expiries": {n_expiries},
    "dense": {dense},
    "n_checked": report.n_checked,
    "ok": report.ok,
    "wall_s": wall,
    "rss_gb": rss_gb,
    "status": report.note.split("Dropped")[0].strip(),
}}))
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    rows = []
    for n in SPARSE_N:
        rows.append(_one(n, False))
        print("sparse", rows[-1])
    for n in DENSE_N:
        rows.append(_one(n, True))
        print("dense", rows[-1])
    frame = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FIG / "p3a_lp_memory.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for dense, label, style in ((False, "sparse", "-o"), (True, "dense", "--s")):
        part = frame[frame["dense"] == dense]
        axes[0].plot(part["n_expiries"], part["rss_gb"], style, label=label)
        axes[1].plot(part["n_expiries"], part["wall_s"], style, label=label)
    axes[0].axhline(2.0, color="0.4", lw=0.8, label="2 GB")
    axes[0].set_ylabel("peak RSS (GB)")
    axes[1].set_ylabel("wall time (s)")
    for ax in axes:
        ax.set_xlabel("expiries")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Quote linear program on SPX, 14 October 2026.")
    fig.tight_layout()
    fig.savefig(FIG / "p3a_lp_memory.png", dpi=140)
    plt.close(fig)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
