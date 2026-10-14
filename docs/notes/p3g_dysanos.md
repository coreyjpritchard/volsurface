# Minimal DYSANOS

This module is minimal DYSANOS. The full v3 baseline is future work: a
fitted decoder, a spot copula, the paper's kappa units and expiry grid,
and joint training. arXiv:2608.12587 was not read in full, so those
paper choices stay UNVERIFIED where this note says so.

`kappa` is per year. The step `D` is in years. A one-day step is `1/252`.
Decoder expiries are business days divided by 252. The default grid is
2, 5, 10, 20, 40, 126 and 252 business days, with 21 strikes. `wide=True`
is an 18 by 200 stand-in and is not the paper's grid.

The transition is `A_D = exp(-kappa D)` with
`kappa = L L' + S + kappa_min I`, `kappa_min = 2e-4`, and `S`
antisymmetric. For the scalar `L L' = 0.5`, a one-day step is
`A = 0.998017048017`, which is `exp(-(0.5 + kappa_min) / 252)`. The
two-day matrix is that number squared, to `1e-12`. On the two-factor
example, `A_1 = A_{0.5} A_{0.5}` to `1e-12`.

The Gaussian maximum likelihood of the linear transition is least squares.
On 6,000 dates with `D = 1`, seed 2, and 200 residual-bootstrap draws, the
true diagonal entries `0.8353` and `0.8877` lie in `[0.8159, 0.8392]` and
`[0.8747, 0.8990]`. The innovation variances `0.0229` and `0.0144` lie in
`[0.0225, 0.0241]` and `[0.0139, 0.0150]`. The bands are the 2.5 and 97.5
percentiles.

The decoder builds a SANOS surface. The plan's Sigma exponent is
UNVERIFIED: the operator squares Sigma, and this code bounds Sigma in
`[0.01, 3]`. One decoded surface, `eta = 0.3460`, has no `check_surface`
violation at `1e-9`.

The last pillar call equals `weights @ black_call` to `1e-10`. The zero
latent and the unit latent differ by `0.0122` in the at-the-money call at
that expiry.

Draws from the last expiry, 20,000 paths, seed 11, kernel variance
`0.013185`. The sample mean is `0.9977` against `q' K = 1`, a z-score of
`-1.51` (standard error `1.52e-3`). The second moment is `1.0415` against
`q' (K^2 exp(eta V)) = 1.0464`, a z-score of `-1.51` (standard error
`3.24e-3`). Both sit inside 4 standard errors. The law that drops the
`-v/2` term in the sampler sits 5.9 and 5.8 standard errors away from
these draws.

The figure compares day-to-day implied-vol changes of minimal DYSANOS
with one regime path priced by the Black call on integrated forward
variance. The first component takes `0.9989` of the minimal DYSANOS
changes and `1.0000` of the regime changes. At `H = 1/2` the forward
variance is flat, so that regime vol is `sqrt(v_t)` at every expiry.
There is no daily SPX file. The only history here is synthetic.
