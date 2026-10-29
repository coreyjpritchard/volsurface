"""Log-space check of the trader's forward variance curve at constant nu.

Run from the repository root:

    python scripts/forward_log_residuals.py

For constant ``nu`` the variance at ``u`` given the curve at ``t`` is
lognormal: ``log v_u - log xi_t(u)`` is ``N(-s2/2, s2)`` with
``s2 = nu^2 (u - t)^(2H)``. The script prints the sample mean and variance of
that residual next to the theory, and the slope of ``log v_u`` on
``log xi_t(u)``, which is 1 when the curve is the conditional expectation.

Settings: one regime, ``H = 0.1``, flat ``xi0 = 0.04``, ``rho = -0.4``,
21 dates on ``[0, 1]`` (step 0.05), eight runs of 500 paths with seeds
``run_seed(81 + run, 2)`` (test split). The rows are the four ``(nu, t, u)`` pairs printed.
"""

from __future__ import annotations

import numpy as np

from volsurface.core.types import RegimeSpec
from volsurface.markets.regime import RegimeSwitchingBergomi
from volsurface.markets.seeds import run_seed

H = 0.1
XI0 = 0.04
TIMES = np.linspace(0.0, 1.0, 21)
ROWS = ((1.5, 0.4, 0.45), (1.5, 0.4, 1.0), (3.0, 0.4, 0.45), (3.0, 0.7, 1.0))
N_RUNS = 8
N_PATHS = 500


def _index(t: float) -> int:
    i = int(round(t / 0.05))
    assert abs(TIMES[i] - t) < 1e-12
    return i


def _draws(nu: float) -> tuple[np.ndarray, np.ndarray]:
    spec = RegimeSpec(
        generator=np.zeros((1, 1)),
        params=[{"nu": nu}],
        p0=np.array([1.0]),
        switch="nu",
    )
    model = RegimeSwitchingBergomi(spec, xi0=XI0, rho=-0.4, H=H, n_mc=2, n_pilot=2)
    v, xi = [], []
    for run in range(N_RUNS):
        paths = model.simulate_market(N_PATHS, TIMES, seed=run_seed(81 + run, 2))
        v.append(paths.state["v"])
        xi.append(paths.state["xi"])
    return np.concatenate(v), np.concatenate(xi)


def main() -> None:
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    print("nu  t->u  mean (theory)  variance (theory)  slope  n")
    for nu, t, u in ROWS:
        if nu not in cache:
            cache[nu] = _draws(nu)
        v, xi = cache[nu]
        i_t, i_u = _index(t), _index(u)
        log_v = np.log(v[:, i_u])
        log_xi = np.log(xi[:, i_t, i_u])
        resid = log_v - log_xi
        s2 = nu**2 * (u - t) ** (2.0 * H)
        slope = float(np.polyfit(log_xi, log_v, 1)[0])
        print(
            f"{nu:.1f}  {t:.2f}->{u:.2f}  {resid.mean():.3f} ({-s2 / 2:.3f})"
            f"  {resid.var(ddof=1):.3f} ({s2:.3f})  {slope:.3f}  {resid.size}"
        )


if __name__ == "__main__":
    main()
