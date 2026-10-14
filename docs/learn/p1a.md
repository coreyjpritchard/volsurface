# P1a: one price, in the forward

## What to watch

The figure `docs/figures/p1a_protocol.png` is a Heston smile and a flat
Black-Scholes smile, both produced by the same `price` method. The crosses
reprice Heston with the variance passed in as state instead of as the
parameter `v0`. They sit on the line. That is the whole interface: `F`, `K`,
`T`, `df`, and an optional state. The standard error is not on the array.

## The definition

The cash call is `df * E[(F_T - K)+]`. `F_T` is the random terminal forward.
`F(T)` is today's forward curve. Divide by `df * F(T)` and the curve value
drops out. What is left depends on the strike only through `k = log(K / F(T))`.
Black-Scholes has nothing else to remember, so its state is empty. Heston
remembers the variance. `state=None` means "use the variance the model was
built with".

A Monte Carlo estimate is a random number. Putting the standard error on the
array would lose it the first time you add two prices. `price` returns the
array. `price_with_error` returns the array and the standard error side by side.

## Hypotheses

These are not results.

- The normalised call used here matches SANOS once the PDF is checked equation
  by equation. The HTML summary is all we have used.
- Labelling the SPY snapshot American and physical is correct, and it is why
  that file is the wrong input for a European martingale fit.
- The Heston cumulant stencil is accurate enough to place a cosine grid. `c1`
  matches the integrated expected variance, and `c2` and `c4` match a second
  stencil and a central difference on the Fang and Oosterlee parameter set.
  Phase 1 does not price with the cosine grid.

## Explain it back

1. Why does `price` take a forward and a discount factor, and not a spot and a rate?
2. What goes wrong if a Monte Carlo price returns an array that also carries `.stderr`?
3. What is in Heston's state, and what does `state=None` mean for Black-Scholes?
