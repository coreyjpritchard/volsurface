# %% Static figure for the README: python docs/figure_00.py -> docs/00_the_smile.png
"""Session 00 figure: SPY smiles against one Heston fit, and the ATM skew term structure.

The Heston parameters are the joint least squares fit to the SPY at-the-money vols and
skews of the 2026-10-07 snapshot recorded in LOG.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from volsurface import chains, models, smile  # noqa: E402

FIT = models.Heston(v0=0.0098, kappa=4.26, theta=0.0373, xi=0.388, rho=-0.99)

# %% Data
market = chains.market_implied_vols(chains.load_snapshot(chains.latest_snapshot("SPY")))
ts = smile.market_term_structure(market)
days = [1, 7, 30, 91, 365]
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(days)))
for d, c in zip(days, colours, strict=True):
    row = ts.iloc[int(np.argmin(np.abs(ts["T"].to_numpy() - d / 365)))]
    pts = market[market["expiry"] == row["expiry"]]
    pts = pts[pts["k"] >= -1.5 * np.sqrt(row["T"])]
    k = np.linspace(pts["k"].min(), pts["k"].max(), 80)
    iv = models.implied_surface(FIT, 1.0, np.exp(k), [row["T"]])[0]
    ax.plot(pts["k"], pts["iv"], ".", ms=3, color=c)
    n_days = round(row["T"] * 365)
    ax.plot(k, iv, color=c, label=f"{n_days} day" + ("" if n_days == 1 else "s"))
ax.set(xlabel="k = log(K/F)", ylabel="implied vol", ylim=(0, 0.6),
       title="SPY smiles (points) and one Heston fit (lines)")
ax.legend(title="expiry")

ax = fig.add_subplot(1, 2, 2)
A, alpha = smile.power_law(ts["T"], ts["skew"])
T_line = np.geomspace(ts["T"].min(), ts["T"].max(), 50)
ax.loglog(ts["T"], -ts["skew"], "o", color="#d62728", label="SPY")
ax.loglog(T_line, A * T_line**-alpha, ":", color="#d62728",
          label=f"{A:.2f} T^(-{alpha:.2f})")
ax.loglog(T_line, -smile.model_term_structure(FIT, T_line)[:, 1], color="#1f77b4",
          label="Heston, best joint fit")
ax.set(xlabel="T (years)", ylabel="-d(iv)/dk at k = 0", title="At-the-money skew")
ax.legend()
fig.tight_layout()
out = Path(__file__).with_name("00_the_smile.png")
fig.savefig(out, dpi=110)
print(out)
