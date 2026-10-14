# Minimal DYSANOS

UNVERIFIED. This note follows the plan's citations of arXiv:2608.12587. The
paper was not read in full. The module is a small replication of the
baseline, not a new model, and it does not reproduce the paper's numbers.

The transition is `A_D = exp(-kappa D)` with
`kappa = L L' + S + kappa_min I`, `kappa_min = 2e-4`, and `S`
antisymmetric. The real part of each eigenvalue of `kappa` is at least
`kappa_min`, so each eigenvalue of `A_D` has modulus `exp(-Re(lambda) D)`,
which is strictly below 1. On five random draws the largest modulus is
`0.99995719`, from a step of one business day. A refit does not leave this
parameterisation, so the bound holds for every transition the code stores.

The Gaussian maximum likelihood of the linear transition is least squares.
On 6,000 dates with `D = 1`, seed 2, and 200 residual-bootstrap draws, the
true diagonal entries `0.8353` and `0.8877` lie in `[0.8159, 0.8392]` and
`[0.8747, 0.8990]`. The innovation variances `0.0229` and `0.0144` lie in
`[0.0225, 0.0241]` and `[0.0139, 0.0150]`. The bands are the 2.5 and 97.5
percentiles. Fitting the factors by least squares, after a separate
surface stage, is a stated deviation from joint training. Joint training
is UNVERIFIED.

The decoder builds a SANOS surface. Strikes are a uniform log-moneyness
grid. The plan's strike formula is UNVERIFIED. The plan writes a Sigma
bound `(0.01^2, 3^2)` and flags the exponent. The operator squares Sigma,
so this code bounds Sigma in `[0.01, 3]`. That exponent is UNVERIFIED.
`eta` is in `[0.01, 0.50]`. One decoded surface, `eta = 0.3460`, has no
`check_surface` violation at `1e-9`.

The spot is a draw from the front-expiry mixture. No drift is added. The
paper's copula formulae are UNVERIFIED. On 2,000 draws the
probability-integral transform has Kolmogorov-Smirnov statistic `0.0230`
and p-value `0.2376`.

A three-strike book held to the last expiry, 4,000 draws, has mean P&L
`5.76e-3`, `4.60e-3` and `3.63e-3` at log-moneyness `-0.1`, `0` and `0.1`.
The standard errors are `2.76e-3`, `2.28e-3` and `1.72e-3`. Each mean is
about two standard errors above zero on shared draws. None is negative.
The test bar is `-3.5` standard errors.

The figure compares day-to-day implied-vol changes. The generated history
is 40 dates of this decoder. The synthetic history is one regime path,
priced by the Black call on integrated forward variance, which has no
strike slope. The first component takes `0.9989` of the generated changes
and `0.9520` of the regime changes. The paper's figure is UNVERIFIED, so
this panel is not a comparison with it.

The optional scale on the innovations is off by default. `G` does not
depend on the latent. The constant-vol-of-vol reading of the paper is
UNVERIFIED. There is no daily SPX file. The only history here is synthetic.
