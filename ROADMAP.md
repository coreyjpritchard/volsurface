# Roadmap

One question per session. Levels: 0 play, 1 reproduce, 2 implement, 3 extend (see
`CLAUDE.md`).

- [x] **00 The smile is the model's fingerprint** (level 0). Heston surface with sliders,
  Black-Scholes for contrast, SPY overlay. The market's short-dated skew follows a power
  law that Heston cannot match at both ends.
- [ ] **01 Fit SVI to the SPY surface** (level 1). Fit raw SVI per expiry (Gatheral 2004),
  then check for butterfly arbitrage (negative density) and calendar arbitrage (total
  variance decreasing in $T$), as in Gatheral and Jacquier (2014), *Arbitrage-free SVI
  volatility surfaces*.
- [ ] **02 Calibrate Heston to the snapshot** (level 2). Weighted least squares on implied
  vols over all expiries. Where does the fit fail, and does it fail where Session 00 said?
- [ ] **03 SABR and the short-end skew** (level 2). Hagan et al. (2002) expansion behind the
  `Model` interface; per-expiry fits and what the fitted parameters do across maturities.
- [ ] **04 Rough Bergomi by Monte Carlo** (level 1 to 2). Hybrid scheme (Bennedsen, Lunde,
  Pakkanen 2017) behind the same interface. Reproduce the ATM skew term structure against
  maturity from Bayer, Friz and Gatheral (2016) and compare it with the SPY power law.
- [ ] **05 Live IBKR chain and a daily snapshot cron** (level 0 to 1). Run the IBKR
  backend against a gateway, snapshot SPY each day at the close, and watch the skew
  exponent move.
- [ ] **06 Level 3 question** (level 3). Proposal: is the short-end skew exponent stable from
  day to day, and does it move with the level of volatility? Rough volatility with a fixed
  $H$ says the exponent should not depend on the vol level; a regime-dependent exponent
  would point to a different mechanism. Uses the daily snapshots from Session 05.
