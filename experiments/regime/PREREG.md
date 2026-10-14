# Regime experiment, pre-registration

Status: draft. To freeze before the pilot ends. This file is not frozen.

The pilot is a CPU run. No GPU is assumed. The regime variable is `nu`
only. `xi_level` and `both` are not in the pilot. The hedger and the
experiment runner are not in this repository.

`Gap(X -> Y)` is the risk of the policy trained on `X` and evaluated on
`Y`, minus the risk of the policy trained on `Y` and evaluated on `Y`.

## H1. Transfer cost

`Gap(L -> H) > 0` for the 30-day at-the-money call hedged with spot only.
The bar is a lower 95 percent bound above 0.

## H2. Vol-of-vol content

The premium-normalised `Gap(L -> H)` is larger for the forward-start
at-the-money straddle than for the 30-day at-the-money call. The bar is a
lower 95 percent bound on that difference above 0.

## H3. Hedgeability

For the 30-day call, adding a 30-day variance swap reduces `Gap(L -> H)`.
For the forward-start straddle, the reduction from the variance swap is
smaller than the reduction from adding a 30-day at-the-money call, if that
call pricer clears its validation bar. The directions are fixed. The
variance swap hedges the level of variance.

## Not the gate

H4 and H5 are exploratory. They are not hypotheses this draft freezes.
DYSANOS statements cited as motivation for H5 remain UNVERIFIED.
