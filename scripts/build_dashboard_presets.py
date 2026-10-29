"""Save the SPX SVI and SANOS fits that the vol environment dashboard loads.

Run from the repository root (about three minutes, most of it the SANOS linear
programme):

    python scripts/build_dashboard_presets.py

Writes ``data/presets/spx_fits.npz``: the raw SVI parameters of every expiry,
the SANOS (eta = 0.25, omega = 0, pchip in time) strikes, weights and ATM
variances. Fitted parameters only: no quote and no per-quote implied vol is saved
(the lab rebuilds those from the local snapshot,
:func:`volsurface.environments.load_market_quotes`). Both fits are the ones
in ``scripts/figures_p3c.py``. The dashboard rebuilds the surfaces from these
numbers, so it never refits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from volsurface import chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.fit.sanos import SanosFitter
from volsurface.fit.svi import SVIFitter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "presets" / "spx_fits.npz"
ETA = 0.25


def main() -> None:
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)

    svi = SVIFitter().fit(data).surface
    svi_T = np.array([t for t, _ in svi.pillars])
    svi_params = np.array([[s.a, s.b, s.rho, s.m, s.sigma] for _, s in svi.pillars])
    print(f"SVI: {svi_T.size} expiries", flush=True)

    sanos = (
        SanosFitter(eta=ETA, omega=0.0, dx=0.05, time_interp="pchip", method="highs-ipm")
        .fit(data)
        .surface
    )
    print(f"SANOS: {sanos.T.size} expiries", flush=True)

    arrays = {
        "svi_T": svi_T,
        "svi_params": svi_params,
        "svi_k_span": np.array([svi.domain.k_min, svi.domain.k_max]),
        "sanos_T": sanos.T,
        "sanos_variance": sanos.variance,
        "sanos_eta": np.array(sanos.eta),
        "sanos_time_interp": np.array(sanos.time_interp),
        "sanos_n": np.array(sanos.T.size),
        "sanos_domain": np.array(
            [sanos.domain.t_min, sanos.domain.t_max, sanos.domain.k_min, sanos.domain.k_max]
        ),
    }
    for j in range(sanos.T.size):
        arrays[f"sanos_strikes_{j}"] = sanos.strikes[j]
        arrays[f"sanos_weights_{j}"] = sanos.weights[j]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size / 1024:.0f} KB", flush=True)


if __name__ == "__main__":
    main()
