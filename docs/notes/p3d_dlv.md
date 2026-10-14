# Discrete local volatility

On one strike grid the transition matrix `Q` is defined by its inverse.
For an interior column the three nonzero entries are built from
`Sigma^2 dT` and the neighbouring strike gaps, and the first and last
columns are unit columns. The columns of `Q` sum to 1, and each column
has mean equal to its strike, so a discrete marginal stays a martingale
when it is stepped forward.

The implied `Sigma` is the discrete Dupire ratio: twice the time slope of
the call, divided by `K^2` times the strike second difference, with `0 / 0`
read as 0. Sending calls through that ratio and back through `Q` returns
the calls. Strikes that change with expiry are handled by splitting each
source atom onto the two target strikes that bracket it, which keeps the
mean, and then applying the homogeneous operator on the target grid.

The heatmap is this ratio on the `eta = 0.25` SANOS fit of the SPX
snapshot, on one common strike grid. A large second difference along
strike is the statement that these discrete local volatilities are not a
smooth surface in the strike.
