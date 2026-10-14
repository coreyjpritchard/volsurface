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

Calibration matches at-the-money level and skew on all 40 SPX expiries.
`H` is free, or fixed at 0.167, or fixed at 0.149. The vol-of-vol
coordinate is `nu`. Paths use the plugin defaults, 4,000 paths and 64
steps, and seed 0. This repository does not contain the hybrid scheme.

On this snapshot the SANOS slope is `0.01637` at the first expiry,
`0.04085` at the middle expiry, and `0.06236` at the last. The curve
rises with maturity.

The free fit reached `ftol` in 27.4 s: `H = 0.2057`, `nu = 2.326`,
`rho = -0.6593`, objective `3.039e-2`. With `H` fixed at 0.167 the fit
reached `ftol` in 29.7 s: `nu = 1.460`, `rho = -0.8231`, objective
`6.746e-2`. With `H` fixed at 0.149 it reached `ftol` in 24.7 s:
`nu = 1.247`, `rho = -0.8863`, objective `1.319e-1`. The three
correlations are inside `(-0.999, 0)`.

At `T = 0.00114` the market skew is `-2.9648`. The free rBergomi skew is
`-2.9643`, the 0.167 fit is `-3.0938`, and the 0.149 fit is `-3.1969`.
The Heston skew there is `-3.2744`. At `T = 0.96278` the market skew is
`-0.2815`, the free fit is `-0.2675`, and the Heston skew is `-0.2387`.
The front at-the-money level is `0.1511` in the market, `0.1749` in the
free fit, and `0.0879` in the Heston fit.

The power law on these 40 expiries is `0.30511 T^{-0.33283}`. The figure
is `docs/figures/p3e_skew.png`.
