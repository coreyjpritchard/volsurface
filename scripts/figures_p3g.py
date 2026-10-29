"""PCA of implied-vol changes from minimal DYSANOS, against a synthetic path.

Run from the repository root:

    python scripts/figures_p3g.py

The full v3 baseline is future work. The printed block is the
measurement behind the log entry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

from volsurface import bs
from volsurface.arbitrage.check import check_surface
from volsurface.core.types import ArbTol, RegimeSpec
from volsurface.dynamic.dysanos import (
    Dysanos,
    bootstrap_bands,
    max_modulus,
    simulate_factors,
    transition_matrix,
)
from volsurface.markets.pricers import integrate_forward_variance
from volsurface.markets.regime import RegimeSwitchingBergomi
from volsurface.markets.seeds import run_seed
from volsurface.mc.rng import generator

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def _dynamics():
    lower = np.array([[0.4, 0.0], [0.1, 0.3]])
    skew = np.array([[0.0, 0.2], [-0.2, 0.0]])
    loading = np.array([[0.15, 0.0], [0.02, 0.12]])
    mean = np.zeros(2)
    return mean, lower, skew, loading


def _modulus() -> float:
    rng = generator(0)
    worst = 0.0
    for _ in range(5):
        lower = np.tril(rng.normal(scale=0.5, size=(3, 3)))
        raw = rng.normal(scale=0.2, size=(3, 3))
        skew = 0.5 * (raw - raw.T)
        for step in (1.0 / 252.0, 1.0):
            worst = max(worst, max_modulus(lower, skew, step))
    return worst


def _iv_panel(surface, maturities: np.ndarray, log_k: np.ndarray) -> np.ndarray:
    """Implied vol where the 2-day wing is above intrinsic. Shorter slices are omitted."""
    call = surface.call(maturities, log_k)
    strikes = np.exp(log_k)
    out = np.empty_like(call)
    for j, maturity in enumerate(maturities):
        out[j] = bs.implied_vol(call[j], 1.0, strikes, float(maturity), 1.0)
    return out


def _first_loading(panel: np.ndarray) -> tuple[np.ndarray, float]:
    changes = np.diff(panel, axis=0).reshape(panel.shape[0] - 1, -1)
    centred = changes - changes.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    share = float(singular[0] ** 2 / np.sum(singular**2))
    return vt[0], share


def _forward_ratios() -> None:
    """Mean of the H=0.1 forward curve after the clock, divided by xi0.

    One regime, 2,000 paths, clock index 8, seed entropy 80, test split.
    Later dates are the closed form. The diagonal is v_t and is not in
    the average.
    """
    times = np.linspace(0.0, 1.0, 17)
    clock = 8
    for nu in (1.5, 3.0):
        spec = RegimeSpec(
            generator=np.zeros((1, 1)),
            params=[{"nu": nu}],
            p0=np.array([1.0]),
            switch="nu",
        )
        model = RegimeSwitchingBergomi(spec, xi0=0.04, rho=-0.4, H=0.1, n_mc=2, n_pilot=2)
        paths = model.simulate_market(2000, times, seed=run_seed(80, 2))
        future = paths.state["xi"][:, clock, clock + 1 :]
        print(f"forward_mean_ratio nu={nu:.1f}: {float(future.mean()) / 0.04:.6f}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    modulus = _modulus()
    model = Dysanos(d=4, n_strikes=21, seed=1)
    latent = generator(1).normal(size=4)
    surface = model.decode(latent)
    mids = 0.5 * (surface.T[:-1] + surface.T[1:])
    grid_t = np.sort(np.concatenate([surface.T, mids]))
    report = check_surface(surface, grid_t, np.log(model.strikes)[::2], tol=ArbTol(price=1e-9))
    spots = model.sample_front(surface, 2000, seed=3)
    ks = kstest(model.front_cdf(surface, spots), "uniform")
    index = int(surface.T.size - 1)
    log_book = np.array([-0.1, 0.0, 0.1])
    premium = surface.call(np.array([surface.T[index]]), log_book)[0]
    drawn = model.sample_expiry(surface, index, 4000, seed=4)
    payoff = np.maximum(drawn[:, None] - np.exp(log_book)[None, :], 0.0)
    pnl = payoff - premium
    mean = pnl.mean(axis=0)
    se = pnl.std(axis=0, ddof=1) / np.sqrt(pnl.shape[0])

    mean_h, lower, skew, loading = _dynamics()
    factors = simulate_factors(mean_h, lower, skew, loading, 6000, 1.0, seed=2)
    truth = transition_matrix(lower, skew, 1.0)
    cov_truth = loading.T @ loading
    low_a, high_a, low_c, high_c = bootstrap_bands(factors, 200, seed=3)

    log_k = np.linspace(-0.15, 0.15, 7)
    maturities = model.expiries[2:]
    lower4 = np.diag([0.3, 0.25, 0.2, 0.15])
    skew4 = np.zeros((4, 4))
    skew4[0, 1] = 0.1
    skew4[1, 0] = -0.1
    latent_path = simulate_factors(np.zeros(4), lower4, skew4, 0.2 * np.eye(4), 40, 1.0, seed=5)
    history = np.stack([_iv_panel(model.decode(row), maturities, log_k) for row in latent_path])
    loading_d, share_d = _first_loading(history)

    spec = RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": 0.4}, {"nu": 1.1}],
        p0=np.array([0.7, 0.3]),
        switch="nu",
    )
    regime = RegimeSwitchingBergomi(spec, xi0=0.04, rho=-0.4, H=0.5)
    n_days = 40
    horizon = float(model.expiries[-1])
    times = np.arange(0, n_days + int(horizon * 252) + 2) / 252.0
    paths = regime.simulate_market(1, times, seed=run_seed(6, 0))
    regime_iv = np.empty((n_days, maturities.size))
    for day in range(n_days):
        now = float(times[day])
        curve = paths.state["xi"][0, day][None, :]
        for j, tenor in enumerate(maturities):
            total = integrate_forward_variance(
                curve, paths.state["xi_tenor"], now, now + float(tenor)
            )
            regime_iv[day, j] = np.sqrt(max(float(total[0]), 0.0) / float(tenor))
    loading_r, share_r = _first_loading(regime_iv[:, None, :])

    print(f"max_modulus: {modulus:.8f}")
    print(f"check_ok: {report.ok} violations: {len(report.violations)} eta: {surface.eta:.4f}")
    print(f"ks_stat: {ks.statistic:.4f} ks_pvalue: {ks.pvalue:.4f}")
    for j, strike in enumerate(log_book):
        print(f"pnl_k={strike:.1f} mean: {mean[j]:.6e} se: {se[j]:.6e}")
    print(f"A_diag_truth: {truth[0, 0]:.4f} {truth[1, 1]:.4f}")
    print(f"A_diag_low: {low_a[0, 0]:.4f} {low_a[1, 1]:.4f}")
    print(f"A_diag_high: {high_a[0, 0]:.4f} {high_a[1, 1]:.4f}")
    print(f"cov_diag_truth: {cov_truth[0, 0]:.4f} {cov_truth[1, 1]:.4f}")
    print(f"cov_diag_low: {low_c[0, 0]:.4f} {low_c[1, 1]:.4f}")
    print(f"cov_diag_high: {high_c[0, 0]:.4f} {high_c[1, 1]:.4f}")
    print(f"pca_dysanos_share: {share_d:.4f}")
    print(f"pca_regime_share: {share_r:.4f}")
    print(f"iv_finite_fraction: {float(np.isfinite(history).mean()):.4f}")
    _forward_ratios()

    # One panel. The regime check has a single loading per expiry and a share of
    # 1 to four decimals, so it is reported as text rather than drawn.
    grid = loading_d.reshape(maturities.size, log_k.size)
    # The sign of a principal component is arbitrary. It is flipped for the plot
    # only, so the CSV keeps the solver's sign.
    shown = -grid if np.mean(grid) < 0.0 else grid
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    markers = ("o", "s", "^")
    for marker, j in zip(markers, (0, maturities.size // 2, maturities.size - 1), strict=True):
        ax.plot(
            log_k,
            shown[j],
            marker=marker,
            ms=4,
            lw=1.2,
            label=f"expiry {int(round(maturities[j] * 252))} business days",
        )
    ax.set_xlabel("log-moneyness k = log(K / F)")
    ax.set_ylabel("first principal component loading\n(unit vector, sign fixed positive)")
    span = float(shown.max() - shown.min())
    ax.set_ylim(float(shown.min()) - 0.1 * span, float(shown.max()) + 0.25 * span)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"In minimal DYSANOS, one component explains {100 * share_d:.2f}% of implied-vol "
        "changes,\nand it is close to a level shift: every loading lies between "
        f"{shown.min():.3f} and {shown.max():.3f},\na little higher at the money "
        f"(regime variance-swap check: {100 * share_r:.2f}%)",
        fontsize=10.5,
    )
    fig.text(
        0.01,
        0.005,
        f"Synthetic: 4 latent factors, a 40-step path (seed 5), {log_k.size} strikes from "
        f"$-0.15$ to 0.15, {maturities.size} expiries.\nPCA of step-on-step implied-vol changes "
        "across all expiries and strikes together.",
        fontsize=7.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.subplots_adjust(top=0.84)
    fig.savefig(FIG / "p3g_pca.png", dpi=140)
    plt.close(fig)

    with (FIG / "p3g_pca.csv").open("w") as handle:
        handle.write("source,expiry_days,log_k,loading\n")
        for j, maturity in enumerate(maturities):
            for a, moneyness in enumerate(log_k):
                handle.write(
                    f"dysanos,{float(maturity) * 252:.1f},{moneyness:.4f},{grid[j, a]:.8f}\n"
                )
            handle.write(f"regime,{float(maturity) * 252:.1f},,{loading_r[j]:.8f}\n")


if __name__ == "__main__":
    main()
