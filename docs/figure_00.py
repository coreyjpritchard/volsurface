# %% Static figure for the README: python docs/figure_00.py -> docs/00_the_smile.png
"""Session 00 figure: SPX smiles against one Heston fit, and the ATM skew term structure.

The Heston parameters are refit here. The loss is the at-the-money level and
skew at every expiry, unweighted, in one residual vector. Three starts, scipy
``least_squares`` (``trf``). The box is the session slider box with kappa
allowed up to 20: v0 in [0.01, 0.15], kappa in [0.1, 20], theta in
[0.01, 0.15], xi in [0.05, 2], rho in [-0.99, 0.99]. The smallest cost is drawn.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from volsurface import chains, models, smile  # noqa: E402

# Slider box from sessions/00_the_smile.py, with the kappa upper used in LOG.md.
LO = np.array([0.01, 0.1, 0.01, 0.05, -0.99])
HI = np.array([0.15, 20.0, 0.15, 2.0, 0.99])
STARTS = (
    np.array([0.011, 3.0, 0.04, 0.4, -0.9]),
    np.array([0.04, 1.5, 0.04, 0.5, -0.7]),
    np.array([0.02, 10.0, 0.08, 1.5, -0.99]),
)

# %% Data
market = chains.market_implied_vols(chains.load_snapshot(chains.latest_snapshot("SPX")))
ts = smile.market_term_structure(market)
maturities = ts["T"].to_numpy()
market_atm = ts["atm"].to_numpy()
market_skew = ts["skew"].to_numpy()


def _residuals(x: np.ndarray) -> np.ndarray:
    pred = smile.model_term_structure(models.Heston(*x), maturities)
    return np.concatenate([pred[:, 0] - market_atm, pred[:, 1] - market_skew])


best = None
for start in STARTS:
    result = least_squares(_residuals, start, bounds=(LO, HI), method="trf")
    if best is None or result.cost < best.cost:
        best = result
assert best is not None
fit = models.Heston(*best.x)
pred = smile.model_term_structure(fit, maturities)
limit = fit.rho * fit.xi / (4.0 * np.sqrt(fit.v0))
print(
    f"v0={fit.v0:.12f} kappa={fit.kappa:.12f} theta={fit.theta:.12f} "
    f"xi={fit.xi:.12f} rho={fit.rho:.12f} cost={best.cost:.6e} "
    f"nfev={best.nfev} status={best.status}"
)
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

days = [1, 7, 30, 91, 365]
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(days)))
for d, c in zip(days, colours, strict=True):
    row = ts.iloc[int(np.argmin(np.abs(ts["T"].to_numpy() - d / 365)))]
    pts = market[market["expiry"] == row["expiry"]]
    pts = pts[pts["k"] >= -1.5 * np.sqrt(row["T"])]
    k = np.linspace(pts["k"].min(), pts["k"].max(), 80)
    iv = models.implied_surface(fit, 1.0, np.exp(k), [row["T"]])[0]
    ax.plot(pts["k"], pts["iv"], ".", ms=3, color=c)
    n_days = row["T"] * 365
    label = f"{n_days:.0f} days" if n_days >= 1.5 else f"{n_days * 24:.0f} hours"
    ax.plot(k, iv, color=c, label=label)
ax.set(xlabel="k = log(K/F)", ylabel="implied vol", ylim=(0, 0.9),
       title="SPX smiles (points) and one Heston fit (lines)")
ax.legend(title="expiry")

ax = fig.add_subplot(1, 2, 2)
A, alpha = smile.power_law(ts["T"], ts["skew"])
T_line = np.geomspace(ts["T"].min(), ts["T"].max(), 50)
ax.loglog(ts["T"], -ts["skew"], "o", color="#d62728", label="SPX")
ax.loglog(T_line, A * T_line**-alpha, ":", color="#d62728",
          label=f"{A:.3f} T^(-{alpha:.3f})")
ax.loglog(T_line, -smile.model_term_structure(fit, T_line)[:, 1], color="#1f77b4",
          label="Heston, best of three starts")
ax.set(xlabel="T (years)", ylabel="-d(iv)/dk at k = 0", title="At-the-money skew")
ax.legend()
fig.tight_layout()
out = Path(__file__).with_name("00_the_smile.png")
fig.savefig(out, dpi=110)
print(out)
print(f"power A={A:.6f} alpha={alpha:.6f}")
