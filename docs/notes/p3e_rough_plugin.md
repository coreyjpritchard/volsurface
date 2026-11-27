# The rough-volatility plug-in

Rough Bergomi lives in the `roughvol` package. The dev extra pins
`rough-volatility` at commit `3888dfc`. `create("rbergomi")` is the only
way this repository builds it. The conformance test imports `roughvol`.
A test double used to exercise the calibrator is not that plugin, and the
figure does not draw a double.

The log contract of a SANOS surface is `VS(T) = -2 E[log Z_T]`. For the
mixture that expectation is exact: an atom at strike `K` with kernel
variance `v` contributes `log K - v / 2`. Calendar order makes `VS`
increasing in `T`. The instantaneous forward variance is the slope of
that contract. Between pillars

`xi(t) = (VS(T_{j+1}) - VS(T_j)) / (T_{j+1} - T_j)`.

On `[0, T_0]` the level is `VS(T_0) / T_0`. Past the last pillar the last
level is held. That callable is the `xi0` argument of every plugin
construction inside `calibrate`. The surface is fit once, at `eta = 0.25`
and `omega = 0`, and is not refit inside the residual. The plugin default
`0.04` is a flat 20 percent vol. The fit above does not leave `xi0` there.

Calibration matches at-the-money level and skew on every expiry of the
loaded chain. `H` is free, or fixed at 0.10, or fixed at 0.15. The vol-of-vol
coordinate is `nu`. Paths use the plugin defaults, 4,000 paths and 64
steps, and seed 0. This repository does not contain the hybrid scheme.

The committed figure `docs/figures/p3e_skew.png` is drawn from the synthetic
demo chain, not from a market snapshot. Pass `--snapshot` to
`scripts/figures_p3e.py` to run the same comparison on a chain you supply;
those outputs are not committed.
