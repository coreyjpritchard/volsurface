# The quote linear program, stored sparse

The question `check_quotes` asks is whether a discrete martingale measure
sits inside the bid and the ask. The unknowns on one expiry are probabilities
`q` on `{0}` union the quoted normalised strikes, plus a tail pair `(Q, M)`.
Mass and mean are equalities. Each quote adds two slack columns, and each
shared strike of two neighbouring expiries adds a calendar slack. The
objective is the sum of the slacks.

A positive objective is a cash amount. The dual variables on the quote rows
are holdings in those options. If the primal is positive, that portfolio
super-replicates a positive cash amount at every terminal node the support
allows, which is a static arbitrage in the quoted book. The solver does not
return the dual. The objective is the size of the gap.

The coefficients of that program did not change. What changed is the
storage. The previous code allocated a dense row of length `n_var` for
every constraint and stacked the rows. On the first 24 expiries of the
committed SPX file that matrix is large enough that the dense solve's
resident set reaches several gigabytes. The same rows, written as a SciPy
CSR matrix and passed to HiGHS, are the nonzero call coefficients and the
slack columns. `dense=True` turns that CSR matrix back into an array and
solves the identical program, which is the comparison the test runs.

HiGHS returns a status on all 40 expiries. The scale bar is a peak resident
set below 2 GB, measured in a fresh process.
