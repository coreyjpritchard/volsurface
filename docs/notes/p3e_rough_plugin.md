# The rough-volatility plug-in

Rough Bergomi lives in the `roughvol` package. `create("rbergomi")` is the
only way this repository builds it. The conformance test imports `roughvol`
and skips when that import fails. A test double used to exercise the
calibrator is not that plugin, and the figure does not draw it.

The forward variance is the log contract of a SANOS surface,
`VS(T) = -2 E[log Z_T]`. For the mixture that expectation is exact: an atom
at strike `K` with kernel variance `v` contributes `log K - v / 2`. Calendar
order makes `VS` increasing in `T`. The test checks the sign of the
differences.

Calibration, when the plugin is installed, matches at-the-money level and
skew with one seed passed through to the plugin. `H` is free, or fixed at
0.167, or fixed at 0.149. The vol-of-vol coordinate is `nu`. This
repository does not contain the hybrid scheme.

The figure on this machine has no rBergomi curve. `roughvol` is not
installed, so `create("rbergomi")` is not drawn. The plotted series are the
SPX at-the-money skew, the merged Heston fit (`v0 = 0.007982`,
`kappa = 13.466`, `theta = 0.03836`, `xi = 2.703`, `rho = -0.6508`), and
the power law refit to that market skew. The power law on these 40
expiries is `0.30511 T^{-0.33283}`. At `T = 0.00114` the market skew is
`-2.9648` and the Heston skew is `-3.2744`. At `T = 0.96278` the market
skew is `-0.2815` and the Heston skew is `-0.2387`. The front at-the-money
level is `0.1511` in the market and `0.0879` in the Heston fit.
