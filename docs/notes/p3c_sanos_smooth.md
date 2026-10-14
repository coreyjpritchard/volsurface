# SANOS with a positive eta

At fixed weights, a Black call is at least the hockey-stick payoff, so the
mixture with `eta > 0` is at least the `eta = 0` call at every strike.
That is Theorem 3.1. A refit is a different object. The linear program
chooses new weights, and linear interpolation through the quotes is the
most expensive interpolation, so the refitted smooth call can sit below
the linear one between quotes.

`omega = 0` puts the undiscounted payoff into the calendar rows. That is
the choice the paper says keeps the marginals in convex order. `omega = 1`
puts Black calls of variance `eta * V` into those rows. The fit is
reported and labelled not guaranteed. It is not the market-bar fit.

Between expiries the blend weight comes from a PCHIP of at-the-money
total variance. At the money that weight makes the time derivative of the
call match from the left and from the right, because the at-the-money
call is the Black call of that total variance. Off the money the same
weight does not cancel the ratio of call gaps, so a jump in `dC/dT` is
expected and is measured.

The market bar on this SPX snapshot, with `omega = 0`, is zero
static-arbitrage violations on the checked grid at tolerance `1e-9`, and
an in-spread fraction strictly above 0.558. The 0.558 is the raw SVI
figure already in `LOG.md`. The paper's 91.4 percent is a different book.

A second bar holds out strikes. In each expiry, about 20 percent of the
strikes are drawn without replacement by `numpy.random.default_rng` with
a fixed seed, and those quotes are left out of the fit. The bar is that
the share of held-out quotes whose model call lies inside that quote's
bid and ask is at least the raw SVI share on the same held-out quotes,
and that the fitted SANOS surface still has zero static-arbitrage
violations on the dense grid. Corey has ratified this bar: SANOS
held-out in-spread at least raw SVI held-out in-spread, with zero
static-arbitrage violations, ratified 14 Oct 2026.

Seed 0 holds out 1,796 of 8,973 quotes. The mean fraction of strikes
left out of an expiry is 0.2002. SANOS at `eta = 0.25` and `omega = 0`,
fit on the other 7,177 quotes, places 0.990535 of the held-out model
calls inside the held-out bid and ask. Raw SVI on the same split places
0.541203. The fitted SANOS surface has 0 violations on the dense grid.
Both fits and that check took 86.7 s.

The solver is the HiGHS interior point method, with crossover off. The
simplex run on the smooth book returns HiGHS status 15 and no solution
vector. The objective is divided by the largest quote weight before the
solve and multiplied back afterwards, so the minimiser is unchanged and
the logged objective is still the weighted slack.
