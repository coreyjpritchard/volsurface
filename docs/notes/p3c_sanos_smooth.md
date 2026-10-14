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

The solver is the HiGHS interior point method, with crossover off. The
simplex run on the smooth book returns HiGHS status 15 and no solution
vector. The objective is divided by the largest quote weight before the
solve and multiplied back afterwards, so the minimiser is unchanged and
the logged objective is still the weighted slack.
