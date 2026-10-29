# %% Static figure for the README: python docs/figure_00.py -> docs/00_the_smile.png
"""Session 00 figure: SPX smiles against one Heston fit, and the ATM skew term structure.

The Heston parameters are refit here. The loss is the at-the-money level and
skew at every expiry, unweighted, in one residual vector, minimised with scipy
``least_squares`` (``trf``). The box is v0 in [1e-4, 0.15], kappa in [0.1, 50],
theta in [0.001, 0.3], xi in [0.05, 5], rho in [-0.99, 0.99]. ``XI_MAX=10``
raises the xi bound and writes ``docs/00_the_smile_xi10.png`` instead.

Earlier versions used the session slider box (v0 >= 0.01, xi <= 2). The fit
ended on both of those edges, so its flat short end was set by the box and not
by the data. In the wider box v0 moves inside and only xi stays on its edge.
Starts whose residuals are not finite are skipped (one of 35), and a residual
that turns non-finite during a fit is replaced by 10.

The fit is a seeded multi-start: the three hand-picked starts used before,
plus 32 points of a scrambled Sobol sequence (seed 0) spread over the box.
The smallest cost wins, ties broken by start order, so the drawn fit does not
depend on which local minimum one start happens to fall into. The script
prints the spread of the final costs over all starts.

Model vols are inverted with a relative tolerance of 1e-8 (``IV_TOL``). With
the library default of 1e-12 the outer call strike at the 10-hour expiry is
sometimes dropped, the front skew jumps by about 0.1, and the loss is
discontinuous. Earlier versions of this figure fitted that discontinuous loss,
which is why their result depended on the optimiser's path.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from volsurface import chains, models, smile  # noqa: E402

# Wide enough that only xi, the parameter the short-end skew needs, can bind.
LO = np.array([1e-4, 0.1, 0.001, 0.05, -0.99])
# XI_MAX=10 python docs/figure_00.py reruns the fit with a wider xi bound (README).
XI_MAX = float(os.environ.get("XI_MAX", "5"))
HI = np.array([0.15, 50.0, 0.3, XI_MAX, 0.99])
NAMES = ("v0", "kappa", "theta", "xi", "rho")
HAND_STARTS = (
    np.array([0.011, 3.0, 0.04, 0.4, -0.9]),
    np.array([0.04, 1.5, 0.04, 0.5, -0.7]),
    np.array([0.02, 10.0, 0.08, 1.5, -0.99]),
)
SOBOL_SEED = 0
SOBOL_LOG2_N = 5  # 32 Sobol starts
IV_TOL = 1e-8
SOURCE = "SPXW mids, CBOE delayed, 14 Oct 2026"


def starts() -> list[np.ndarray]:
    """Hand-picked starts, then a seeded scrambled Sobol design inside the box."""
    sobol = qmc.Sobol(d=5, scramble=True, seed=SOBOL_SEED).random_base2(SOBOL_LOG2_N)
    inner_lo, inner_hi = LO + 1e-3 * (HI - LO), HI - 1e-3 * (HI - LO)
    return [*HAND_STARTS, *qmc.scale(sobol, inner_lo, inner_hi)]


# %% Data
snapshot = chains.load_snapshot(chains.latest_snapshot("SPX"))
market = chains.market_implied_vols(snapshot)
ts = smile.market_term_structure(market)
maturities = ts["T"].to_numpy()
market_atm = ts["atm"].to_numpy()
market_skew = ts["skew"].to_numpy()


def _residuals(x: np.ndarray) -> np.ndarray:
    pred = smile.model_term_structure(models.Heston(*x), maturities, iv_tol=IV_TOL)
    return np.concatenate([pred[:, 0] - market_atm, pred[:, 1] - market_skew])


# %% Fit
def _safe(x: np.ndarray) -> np.ndarray:
    r = _residuals(x)
    return np.where(np.isfinite(r), r, 10.0)


all_starts = starts()
finite_starts = [x0 for x0 in all_starts if np.all(np.isfinite(_residuals(x0)))]
print(f"starts with finite residuals: {len(finite_starts)} of {len(all_starts)}")
results = [least_squares(_safe, x0, bounds=(LO, HI), method="trf") for x0 in finite_starts]
order = sorted(range(len(results)), key=lambda i: (round(float(results[i].cost), 10), i))
best = results[order[0]]
n_hit = sum(abs(r.cost - best.cost) <= 1e-6 * best.cost for r in results)
fit = models.Heston(*best.x)
pred = smile.model_term_structure(fit, maturities, iv_tol=IV_TOL)
limit = fit.rho * fit.xi / (4.0 * np.sqrt(fit.v0))
tol = 1e-4 * (HI - LO)
at_bound = [
    n
    for n, v, lo, hi, t in zip(NAMES, best.x, LO, HI, tol, strict=True)
    if abs(v - lo) <= t or abs(v - hi) <= t
]
bound_text = ", ".join(
    f"{n} = {lo if abs(v - lo) <= t else hi:g}"
    for n, v, lo, hi, t in zip(NAMES, best.x, LO, HI, tol, strict=True)
    if n in at_bound
)
print(
    f"v0={fit.v0:.12f} kappa={fit.kappa:.12f} theta={fit.theta:.12f} "
    f"xi={fit.xi:.12f} rho={fit.rho:.12f} cost={best.cost:.6e} "
    f"nfev={best.nfev} status={best.status}"
)
print(
    f"starts={len(results)} best_start={order[0]} starts_at_best={n_hit} "
    f"at_bounds={at_bound} feller={2 * fit.kappa * fit.theta / fit.xi**2:.4f}"
)
costs = np.array([r.cost for r in results])
print(
    f"final costs over starts: min={costs.min():.6f} median={np.median(costs):.6f} "
    f"max={costs.max():.6f}; within 1% of best: {int(np.sum(costs <= 1.01 * best.cost))}"
)
spread = np.array([r.x for r in results if r.cost <= 1.01 * best.cost])
ranges = zip(NAMES, spread.min(0), spread.max(0), strict=True)
print("params within 1% of best cost:", " ".join(f"{n}={a:.4f}..{b:.4f}" for n, a, b in ranges))
print(
    f"rmse_atm={np.sqrt(np.mean((pred[:, 0] - market_atm) ** 2)):.6f} "
    f"rmse_skew={np.sqrt(np.mean((pred[:, 1] - market_skew) ** 2)):.6f} "
    f"short_skew_limit={limit:.6f}"
)
for label, target in (("front", maturities[0]), ("1d", 1 / 365), ("1w", 7 / 365), ("1y", 1.0)):
    j = int(np.argmin(np.abs(maturities - target)))
    print(
        f"{label} T={maturities[j]:.6f} mkt_atm={market_atm[j]:.6f} "
        f"mod_atm={pred[j, 0]:.6f} mkt_skew={market_skew[j]:.6f} mod_skew={pred[j, 1]:.6f}"
    )

A, alpha = smile.power_law(ts["T"], ts["skew"])
H = 0.5 - alpha
T_line = np.geomspace(ts["T"].min(), ts["T"].max(), 60)
model_line = -smile.model_term_structure(fit, T_line, iv_tol=IV_TOL)[:, 1]
# Local log-log slope of the Heston skew; "flat" where it is below half the market slope.
slope = -np.gradient(np.log(model_line), np.log(T_line))
flat = slope < 0.5 * alpha
edge = int(np.argmin(flat)) if not flat.all() else len(flat)
flat_days = 365 * T_line[edge - 1] if edge > 0 else float("nan")
cross = np.where(np.diff(np.sign(model_line - A * T_line**-alpha)))[0]
print(f"power A={A:.6f} alpha={alpha:.6f} H={H:.6f} n_expiries={len(ts)}")
print(
    f"heston_flat_below_days={flat_days:.2f} (local slope < alpha/2) "
    f"crosses_power_law_days={[round(365 * T_line[i], 2) for i in cross]}"
)
print(
    "heston local slope by day:",
    " ".join(f"{365 * t:.2f}:{s:.3f}" for t, s in zip(T_line[::6], slope[::6], strict=True)),
)

# %% Figure
days = [1, 7, 30, 91, 365]
fig = plt.figure(figsize=(12, 5.6))
ax = fig.add_subplot(1, 2, 1)
colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(days)))
handles = []
iv_seen = []
for d, c in zip(days, colours, strict=True):
    row = ts.iloc[int(np.argmin(np.abs(ts["T"].to_numpy() - d / 365)))]
    pts = market[market["expiry"] == row["expiry"]]
    pts = pts[pts["k"] >= -1.5 * np.sqrt(row["T"])]
    k = np.linspace(pts["k"].min(), pts["k"].max(), 80)
    iv = models.implied_surface(fit, 1.0, np.exp(k), [row["T"]])[0]
    ax.plot(pts["k"], pts["iv"], ".", ms=3, color=c)
    ax.plot(k, iv, color=c)
    iv_seen += [np.nanmin(pts["iv"]), np.nanmax(pts["iv"]), np.nanmin(iv), np.nanmax(iv)]
    n_days = row["T"] * 365
    label = f"{n_days:.0f} days" if n_days >= 1.5 else f"{n_days * 24:.0f} hours"
    handles.append(Line2D([], [], color=c, marker=".", label=label))
print(f"left panel iv range {min(iv_seen):.4f} to {max(iv_seen):.4f}")
handles += [
    Line2D([], [], color="0.3", marker=".", ls="none", label="dots: SPXW mid quotes"),
    Line2D([], [], color="0.3", label="lines: Heston, calibrated"),
]
ax.set(
    xlabel="log-moneyness k = log(K/F)",
    ylabel="implied volatility",
    ylim=(0.0, 0.6),
    title="Smiles: market quotes against one Heston fit",
)
ax.grid(True, alpha=0.3)
ax.legend(handles=handles, title="expiry", fontsize=8, loc="upper right")

ax = fig.add_subplot(1, 2, 2)
ax.loglog(
    365 * maturities,
    -market_skew,
    "o",
    ms=4,
    color="#d62728",
    label="SPX ATM skew, one dot per expiry",
)
ax.loglog(
    365 * T_line,
    A * T_line**-alpha,
    ":",
    color="#d62728",
    label=f"power law ${A:.3f}\\,T^{{-{alpha:.3f}}}$ (T in years), H = {H:.2f}",
)
ax.loglog(365 * T_line, model_line, color="#1f77b4", label="Heston, calibrated")
xt = [0.5, 1, 2, 7, 30, 91, 365]
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.set_xticklabels([f"{x:g}" for x in xt])
ax.xaxis.set_minor_locator(NullLocator())
yt = [0.2, 0.3, 0.5, 1, 2, 3]
ax.yaxis.set_major_locator(FixedLocator(yt))
ax.set_yticklabels([f"{y:g}" for y in yt])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set(
    xlabel="maturity (calendar days, log scale)",
    ylabel="ATM skew, -d(iv)/dk at k = 0 (log scale)",
    title="At-the-money skew against maturity",
)
ax.grid(True, which="major", alpha=0.3)
ax.legend(fontsize=8, loc="lower left")
feller = 2 * fit.kappa * fit.theta / fit.xi**2
xi_text = f"xi on its bound of {HI[3]:g}" if "xi" in at_bound else f"xi = {fit.xi:.2f}"
fig.suptitle(
    f"SPX ATM skew follows the $T^{{-1/3}}$ power law (H \u2248 {H:.2f}). Heston can follow it "
    f"only with {xi_text},\nand then its short-dated ATM vol is far too low "
    f"({100 * pred[0, 0]:.1f}% against {100 * market_atm[0]:.1f}% at 10 hours)",
    fontsize=12,
)
fig.text(
    0.5,
    0.012,
    f"{SOURCE}. {len(ts)} expiries, 10 hours to 351 days. Heston is fitted to the ATM "
    f"level and skew of every expiry with xi <= {HI[3]:g} and v0 >= {LO[0]:g}; "
    f"on the bound: {bound_text or 'none'}. Feller ratio {feller:.2f}.",
    ha="center",
    fontsize=8,
    color="0.3",
)
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
suffix = "" if XI_MAX == 5.0 else f"_xi{XI_MAX:g}"
out = Path(__file__).with_name(f"00_the_smile{suffix}.png")
fig.savefig(out, dpi=110)
print(out)
