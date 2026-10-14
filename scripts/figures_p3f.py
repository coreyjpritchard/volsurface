"""Sample paths of variance, the regime, and the filter probability.

Run from the repository root:

    python scripts/figures_p3f.py

The printed block is the measurement behind the log entry. The chain check
uses 4,000 paths. The martingale and variance-swap checks use eight runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.stats import binom, chisquare

from volsurface.core.types import Contract, RegimeSpec, Vanilla
from volsurface.markets.filter import hamilton_filter
from volsurface.markets.pricers import VarianceSwapPricer
from volsurface.markets.regime import RegimeSwitchingBergomi
from volsurface.markets.seeds import run_seed
from volsurface.mc.rng import generator

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def _spec() -> RegimeSpec:
    return RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": 0.4}, {"nu": 1.1}],
        p0=np.array([0.7, 0.3]),
        switch="nu",
    )


def _model() -> RegimeSwitchingBergomi:
    return RegimeSwitchingBergomi(_spec(), xi0=0.04, rho=-0.4, H=0.5)


def _chain_row() -> dict:
    paths = _model().simulate_market(4000, np.array([0.0, 0.5, 1.0]), seed=run_seed(1, 2))
    counts = np.bincount(paths.regime[:, -1], minlength=2).astype(float)
    law = np.asarray(_spec().p0 @ expm(_spec().generator * 1.0), dtype=float)
    law = law / law.sum()
    stat, p_value = chisquare(counts, counts.sum() * law)
    return {
        "counts0": counts[0],
        "counts1": counts[1],
        "expected0": 4000 * law[0],
        "expected1": 4000 * law[1],
        "chi2": float(stat),
        "p_value": float(p_value),
    }


def _mean_scores() -> tuple[np.ndarray, np.ndarray]:
    model = _model()
    times = np.linspace(0.0, 1.0, 11)
    z_scores = []
    v_scores = []
    for replication in range(8):
        paths = model.simulate_market(2000, times, seed=run_seed(30 + replication, 2))
        terminal = paths.Z[:, -1]
        se = terminal.std(ddof=1) / math.sqrt(terminal.size)
        z_scores.append((terminal.mean() - 1.0) / se)
        variance = paths.state["v"][:, -1]
        se_v = variance.std(ddof=1) / math.sqrt(variance.size)
        v_scores.append((variance.mean() - model.xi0) / se_v)
    return np.asarray(z_scores), np.asarray(v_scores)


def _swap_scores() -> np.ndarray:
    model = _model()
    pricer = VarianceSwapPricer()
    times = np.linspace(0.0, 1.0, 11)
    contract = Contract(Vanilla(100.0), T=1.0)
    scores = []
    for replication in range(8):
        paths = model.simulate_market(1500, times, seed=run_seed(40 + replication, 0))
        quoted = pricer.price_along(paths, [contract])[:, 0, 0]
        gap = paths.state["v"][:, :-1] @ np.diff(times) - quoted
        scores.append(gap.mean() / (gap.std(ddof=1) / math.sqrt(gap.size)))
    return np.asarray(scores)


def _filter_row() -> dict:
    rng = generator(run_seed(4, 1))
    n_paths, n_times = 2500, 30
    transition = np.array([[0.97, 0.03], [0.04, 0.96]])
    means = np.array([-1.0, 1.0])
    regime = np.empty((n_paths, n_times), dtype=int)
    regime[:, 0] = rng.random(n_paths) < 0.5
    for t in range(1, n_times):
        stay = rng.random(n_paths)
        regime[:, t] = np.where(regime[:, t - 1] == 0, stay > 0.97, stay > 0.04)
    y = means[regime] + 0.5 * rng.standard_normal((n_paths, n_times))
    probs = hamilton_filter(y, transition, means, np.full(2, 0.25), np.array([0.5, 0.5]))
    predicted = probs[:, -1, 1]
    truth = regime[:, -1] == 1
    worst = 0.0
    checked = 0
    edges = np.linspace(0.0, 1.0, 6)
    for band in range(5):
        upper = edges[band + 1]
        if band < 4:
            mask = (predicted >= edges[band]) & (predicted < upper)
        else:
            mask = (predicted >= edges[band]) & (predicted <= upper)
        count = int(mask.sum())
        if count < 40:
            continue
        hits = int(truth[mask].sum())
        centre = float(predicted[mask].mean())
        gap = abs(hits / count - centre)
        worst = max(worst, gap)
        lo = float(binom.ppf(0.001, count, np.clip(centre, 1e-6, 1 - 1e-6)))
        hi = float(binom.ppf(0.999, count, np.clip(centre, 1e-6, 1 - 1e-6)))
        assert lo <= hits <= hi
        checked += 1
    return {"bins": checked, "worst_abs_gap": worst}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    model = _model()
    times = np.linspace(0.0, 1.0, 53)
    paths = model.simulate_market(4, times, seed=run_seed(9, 0))
    rows = []
    for i in range(4):
        for t in range(times.size):
            rows.append(
                (
                    i,
                    times[t],
                    paths.state["v"][i, t],
                    int(paths.regime[i, t]),
                    paths.regime_prob[i, t, 1],
                    paths.Z[i, t],
                )
            )
    frame = np.array(rows)
    header = "path,t,v,regime,regime_prob_1,Z"
    np.savetxt(FIG / "p3f_paths.csv", frame, delimiter=",", header=header, comments="")

    chain = _chain_row()
    z_scores, v_scores = _mean_scores()
    swap = _swap_scores()
    bands = _filter_row()
    summary = {
        **chain,
        "z_max_abs": float(np.max(np.abs(z_scores))),
        "z_mean": float(np.mean(z_scores)),
        "v_max_abs": float(np.max(np.abs(v_scores))),
        "v_mean": float(np.mean(v_scores)),
        "swap_max_abs": float(np.max(np.abs(swap))),
        "swap_mean": float(np.mean(swap)),
        "filter_bins": bands["bins"],
        "filter_worst_abs_gap": bands["worst_abs_gap"],
    }
    with (FIG / "p3f_summary.csv").open("w") as handle:
        handle.write("name,value\n")
        for key, value in summary.items():
            handle.write(f"{key},{value}\n")
    for key, value in summary.items():
        print(f"{key}: {value}")

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.2), sharex=True)
    for i in range(4):
        axes[0].plot(times, paths.state["v"][i], lw=1.0, label=f"path {i + 1}")
        axes[1].step(times, paths.regime[i], where="post", lw=1.0)
        axes[2].plot(times, paths.regime_prob[i, :, 1], lw=1.0)
    axes[0].set_ylabel("variance")
    axes[0].set_title("Four paths. nu is 0.4 or 1.1. The filter probability is causal.")
    axes[0].legend(frameon=False, fontsize=8, ncol=4)
    axes[1].set_ylabel("regime")
    axes[1].set_yticks([0, 1])
    axes[2].set_ylabel("P(regime = 1)")
    axes[2].set_xlabel("time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p3f_paths.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
