"""Figure for P1d: what a greek costs, and where a finite-difference bump sits.

The reference is the Black-Scholes delta. Pathwise, autodiff and the
likelihood ratio estimate it by Monte Carlo on exact geometric Brownian
motion. The middle panel bumps the closed form. The right panel bumps the
Monte Carlo price, delta and gamma, shared paths and independent paths.

Run from the repository root:

    python docs/figures/p1d_greeks.py

Writes docs/figures/p1d_greeks.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface.core.types import Contract, Forward, Vanilla
from volsurface.greeks import Analytic, Autodiff, FiniteDifference, LikelihoodRatio, Pathwise
from volsurface.models.bs import BlackScholes
from volsurface.models.gbm import GBM

OUT = Path(__file__).with_name("p1d_greeks.png")


def _rmse(engine, model, contracts, truth, n_paths, seeds) -> tuple[float, int]:
    """Root mean square error over strikes and seeds. One ``n_evals`` from the first seed."""
    errors = []
    n_evals = 0
    for seed in seeds:
        got = engine.compute(model, contracts, [Forward()], n_paths=n_paths, seed=seed)
        errors.append(got.value[:, 0] - truth)
        n_evals = got.n_evals
    stacked = np.concatenate(errors)
    return float(np.sqrt(np.mean(stacked**2))), n_evals


def _bump_rmse(crn: bool, order: int, bumps, model, contract, truth, n_paths, seeds) -> np.ndarray:
    """Root mean square error across seeds, one entry per bump."""
    out = np.empty(len(bumps))
    for i, h in enumerate(bumps):
        errors = []
        engine = FiniteDifference(F=100.0, df=1.0, bump=float(h), crn=crn)
        for seed in seeds:
            got = engine.compute(
                model,
                contract,
                [Forward()],
                order=order,
                n_paths=n_paths,
                seed=int(seed),
            )
            errors.append(float(got.value[0, 0]) - truth)
        out[i] = float(np.sqrt(np.mean(np.square(errors))))
    return out


def _derivatives(F: float, df: float, sigma: float, T: float) -> tuple[float, float]:
    """``d³C/dF³`` and ``d⁴C/dF⁴`` from the analytic gamma, step 0.01 in the forward."""

    def gamma(forward: float) -> float:
        return float(
            Analytic(F=forward, df=df)
            .compute(
                BlackScholes(sigma=sigma),
                [Contract(Vanilla(F), T)],
                [Forward()],
                order=2,
            )
            .value[0, 0]
        )

    h = 1e-2
    up, mid, down = gamma(F + h), gamma(F), gamma(F - h)
    third = (up - down) / (2.0 * h)
    fourth = (up - 2.0 * mid + down) / h**2
    return third, fourth


def main() -> None:
    F, df, sigma, T = 100.0, 1.0, 0.2, 1.0
    k = np.linspace(-0.2, 0.2, 9)
    contracts = [Contract(Vanilla(float(F * np.exp(log_k))), T) for log_k in k]
    truth = (
        Analytic(F=F, df=df)
        .compute(
            BlackScholes(sigma=sigma),
            contracts,
            [Forward()],
        )
        .value[:, 0]
    )
    model = GBM(sigma=sigma, scheme="exact", n_steps=1)
    counts = [1_000, 4_000, 16_000, 64_000]
    seeds = range(6)
    engines = [
        ("pathwise", Pathwise(F, df), "C0", "o", 7),
        ("autodiff (same derivative, lies on pathwise)", Autodiff(F, df), "C1", "x", 6),
        ("likelihood ratio", LikelihoodRatio(F, df), "C3", "s", 4),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    ax = axes[0]
    pathwise_start = None
    for label, engine, colour, marker, size in engines:
        xs, ys = [], []
        for n in counts:
            error, n_evals = _rmse(engine, model, contracts, truth, n, seeds)
            xs.append(n_evals)
            ys.append(error)
        ax.loglog(
            xs,
            ys,
            marker=marker,
            ms=size,
            color=colour,
            label=label,
            mfc="none" if marker == "o" else colour,
        )
        if pathwise_start is None:
            pathwise_start = (xs[0], ys[0])
    # Slope -1/2: four times the paths, half the error. Anchored on pathwise.
    guide_x = np.array([pathwise_start[0], counts[-1]])
    guide_y = pathwise_start[1] * np.sqrt(pathwise_start[0] / guide_x)
    ax.loglog(
        guide_x,
        guide_y,
        color="0.4",
        lw=0.8,
        label="slope $-1/2$ (error halves when paths quadruple)",
    )
    ax.set_xlabel("Monte Carlo paths")
    ax.set_ylabel("RMSE of the delta (9 strikes, 6 seeds)")
    ax.set_title("Monte Carlo delta: error against paths", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    bumps = np.geomspace(1e-1, 1e-8, 15)
    atm = [contracts[len(contracts) // 2]]
    atm_truth = float(truth[len(contracts) // 2])
    error = np.empty(bumps.size)
    for i, h in enumerate(bumps):
        got = FiniteDifference(F=F, df=df, bump=float(h)).compute(
            BlackScholes(sigma=sigma),
            atm,
            [Forward()],
        )
        error[i] = abs(float(got.value[0, 0]) - atm_truth)
    ax.loglog(bumps, error, color="C2", marker="o", ms=3.5)
    valley = int(np.argmin(error))
    ax.plot(
        bumps[valley],
        error[valley],
        "o",
        color="k",
        ms=6,
        label=f"smallest error, bump {bumps[valley]:.1e}",
    )
    ax.loglog([], [], color="C2", marker="o", ms=3.5, label="closed-form delta, central difference")
    ax.set_xlabel("bump h in the forward (forward = 100)")
    ax.set_ylabel("absolute error of the delta")
    ax.set_title("Bumping the formula: truncation right, rounding left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    mc_bumps = np.geomspace(0.05, 8.0, 8)
    mc_paths = 4_000
    mc_seeds = range(8)
    mc_model = GBM(sigma=sigma, scheme="exact", n_steps=1, n_paths=mc_paths)
    gamma_truth = float(
        Analytic(F=F, df=df)
        .compute(
            BlackScholes(sigma=sigma),
            atm,
            [Forward()],
            order=2,
        )
        .value[0, 0]
    )
    third, fourth = _derivatives(F, df, sigma, T)
    series = [
        ("delta, shared", True, 1, atm_truth, "C0", "-"),
        ("delta, independent", False, 1, atm_truth, "C0", "--"),
        ("gamma, shared", True, 2, gamma_truth, "C1", "-"),
        ("gamma, independent", False, 2, gamma_truth, "C1", "--"),
    ]
    printed = []
    for label, crn, order, target, colour, style in series:
        rmse = _bump_rmse(crn, order, mc_bumps, mc_model, atm, target, mc_paths, mc_seeds)
        ax.loglog(mc_bumps, rmse, color=colour, ls=style, marker="o", ms=3.5, label=label)
        best = int(np.argmin(rmse))
        printed.append((label, float(mc_bumps[best]), float(rmse[best]), rmse))
    # Central-difference truncation. Delta bias is h² |f'''| / 6. Gamma bias is h² |f''''| / 12.
    ax.loglog(
        mc_bumps,
        mc_bumps**2 * abs(third) / 6.0,
        color="k",
        lw=0.9,
        label="delta truncation bias, $h^2|f^{(3)}|/6$",
    )
    ax.loglog(
        mc_bumps,
        mc_bumps**2 * abs(fourth) / 12.0,
        color="k",
        lw=0.9,
        ls=":",
        label="gamma truncation bias, $h^2|f^{(4)}|/12$",
    )
    ax.set_xlabel("bump h in the forward (forward = 100)")
    ax.set_ylabel("RMSE at 4,000 paths, 8 seeds")
    ax.set_title("Bumping a Monte Carlo price", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Pathwise and autodiff deltas beat the likelihood ratio at every path count. Bumping works "
        "on the formula\n(best bump near 3e-4), but on a Monte Carlo price only shared paths keep "
        "the delta usable at small bumps",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.005,
        "Simulation: Black-Scholes, at-the-money call on a forward of 100, volatility 20%, "
        "expiry 1 year, exact GBM in one step. Left: 9 strikes, log-moneyness "
        "$-0.2$ to 0.2, seeds 0 to 5. Right: seeds 0 to 7.",
        fontsize=7.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(top=0.84)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print(f"valley bump={bumps[valley]:.3e} error={error[valley]:.3e}")
    print(f"third={third:.6e} fourth={fourth:.6e} gamma={gamma_truth:.6e}")
    for label, h_best, rmse_best, rmse in printed:
        text = " ".join(f"{v:.4e}" for v in rmse)
        print(f"{label} best_h={h_best:.4g} best_rmse={rmse_best:.4e} rmse={text}")


if __name__ == "__main__":
    main()
