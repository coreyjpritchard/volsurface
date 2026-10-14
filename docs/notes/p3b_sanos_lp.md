# SANOS at eta = 0

The fit is one linear program over every expiry. At expiry `T_j` the
normalised call is `sum_i q_j^i (K^i - x)+`. That is a Black call with
zero variance, so the call is piecewise linear in the normalised strike.
The unknowns `q_j` are nonnegative, have mass 1, and have mean 1.

Quotes are out of the money. A call and a put at the same normalised
strike become one row, and the row keeps the tighter bid and the tighter
ask. The objective charges `1e-8` times the distance to the mid, in units
of the spread, and charges the full distance once the model leaves the
bid or the ask. The weight on a quote is `1 / (ask - bid)`.

Calendar rows are hard. With `omega = 0` they compare undiscounted
payoffs, which is the choice that keeps the marginals in convex order
when the strikes differ by expiry. `omega = 1` is a different program and
is not the default.

`V_j` is the at-the-money bid total variance, lifted by a cumulative
maximum where a later expiry would otherwise sit below an earlier one.
Each lifted expiry is listed on the fit. It does not enter the `eta = 0`
call. It is stored because the same fitter accepts `eta > 0`.

After the solver returns, each weight vector is clipped at 0 and tilted
so the mass and the mean are 1 again. The size of that correction is on
`FitResult.detail`. Two solves are compared on the objective and on the
quote prices from the solver weights, before that tilt. The weights
themselves are not unique.

Boundary strikes follow the wing construction in Remark 2.7, footnote 9,
with a clamp back to one tenth of the lowest quoted strike and 1.5 times
the highest when that construction falls inside the quotes. Interior
strikes are added so that no gap exceeds `dx`. The figure uses `dx = 0.05`.
