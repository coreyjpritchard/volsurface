import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="00 The smile is the model's fingerprint")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from volsurface import chains, models, smile

    return chains, go, mo, models, np, smile


@app.cell(hide_code=True)
def _(chains, mo, np, smile):
    # Market data and the fixed (k, T) grid. Runs once; sliders never touch this cell.
    _dir = (mo.notebook_dir() or chains.REPO_ROOT / "sessions").parent / "data" / "snapshots"
    snapshot_path = chains.latest_snapshot("SPY", _dir)
    snapshot = chains.load_snapshot(snapshot_path)
    market = chains.market_implied_vols(snapshot)
    market_ts = smile.market_term_structure(market)

    # Eight expiries from the snapshot closest to 1 day, 1 and 2 weeks, 1, 3, 6, 9, 12 months.
    _targets = np.array([1, 7, 14, 30, 91, 182, 273, 365]) / 365
    _T_all = market_ts["T"].to_numpy()
    _pick = sorted({int(np.argmin(np.abs(_T_all - t))) for t in _targets})
    grid_ts = market_ts.iloc[_pick].reset_index(drop=True)
    T_grid = grid_ts["T"].to_numpy()

    # Moneyness range per maturity: the market's, clipped to [-1.5, 0.5] * sqrt(T).
    k_lo, k_hi = [], []
    for _e, _t in zip(grid_ts["expiry"], T_grid, strict=True):
        _k = market.loc[market["expiry"] == _e, "k"]
        k_lo.append(max(_k.min(), -1.5 * np.sqrt(_t)))
        k_hi.append(min(_k.max(), 0.5 * np.sqrt(_t)))
    k_lo, k_hi = np.array(k_lo), np.array(k_hi)
    k_surface = k_lo[:, None] + (k_hi - k_lo)[:, None] * np.linspace(0, 1, 25)[None, :]
    k_slice = k_lo[:, None] + (k_hi - k_lo)[:, None] * np.linspace(0, 1, 61)[None, :]
    T_surface = np.repeat(T_grid[:, None], 25, axis=1)

    market_grid = market[market["expiry"].isin(grid_ts["expiry"])]
    maturity_labels = {
        f"{e:%d %b %Y} ({t * 365:.0f} days)": i
        for i, (e, t) in enumerate(zip(grid_ts["expiry"], T_grid, strict=True))
    }
    return (
        T_grid, T_surface, grid_ts, k_slice, k_surface, market, market_grid, market_ts,
        maturity_labels, snapshot, snapshot_path,
    )


@app.cell(hide_code=True)
def _(maturity_labels, mo):
    v0 = mo.ui.slider(0.01, 0.15, step=0.001, value=0.011, label="v0, initial variance",
                      show_value=True)
    kappa = mo.ui.slider(0.1, 10.0, step=0.1, value=3.0, label="kappa, mean reversion speed",
                         show_value=True)
    theta = mo.ui.slider(0.01, 0.15, step=0.001, value=0.04, label="theta, long-run variance",
                         show_value=True)
    xi = mo.ui.slider(0.05, 2.0, step=0.05, value=0.4, label="xi, vol of variance",
                      show_value=True)
    rho = mo.ui.slider(-0.99, 0.99, step=0.01, value=-0.9, label="rho, correlation",
                       show_value=True)
    use_bs = mo.ui.switch(value=False, label="Black-Scholes instead (sigma = sqrt(v0))")
    show_market = mo.ui.checkbox(value=True, label="Overlay SPY market implied vols")
    maturity = mo.ui.dropdown(options=maturity_labels, value=list(maturity_labels)[3],
                              label="Smile at expiry")
    return kappa, maturity, rho, show_market, theta, use_bs, v0, xi


@app.cell(hide_code=True)
def _(T_grid, k_slice, k_surface, kappa, models, np, rho, theta, use_bs, v0, xi):
    # The only cell that reprices: 8 x 25 surface points and 61 slice points per slider move.
    if use_bs.value:
        model = models.BlackScholes(float(np.sqrt(v0.value)))
        model_name = f"Black-Scholes, sigma = {np.sqrt(v0.value):.3f}"
    else:
        model = models.Heston(v0.value, kappa.value, theta.value, xi.value, rho.value)
        model_name = "Heston"
    iv_surface = models.implied_surface(model, 1.0, np.exp(k_surface), T_grid)
    iv_slice = models.implied_surface(model, 1.0, np.exp(k_slice), T_grid)
    return iv_slice, iv_surface, model, model_name


@app.cell(hide_code=True)
def _(
    T_grid, T_surface, go, grid_ts, iv_slice, iv_surface, k_slice, k_surface, kappa,
    market_grid, maturity, mo, model_name, rho, show_market, theta, use_bs, v0, xi,
):
    _i = maturity.value
    surface = go.Figure(
        go.Surface(x=k_surface, y=T_surface, z=iv_surface, colorscale="Viridis",
                   cmin=0.05, cmax=0.45, showscale=False, opacity=0.95, name=model_name)
    )
    if show_market.value:
        surface.add_trace(go.Scatter3d(
            x=market_grid["k"], y=market_grid["T"], z=market_grid["iv"], mode="markers",
            marker=dict(size=2, color="#d62728"), name="SPY",
        ))
    surface.update_layout(
        title=model_name, height=560, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        uirevision="keep",
        scene=dict(xaxis_title="k = log(K/F)", yaxis_title="T (years)",
                   zaxis_title="implied vol", zaxis=dict(range=[0.0, 0.6]),
                   aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=0.8),
                   camera=dict(eye=dict(x=-1.0, y=-2.0, z=0.9))),
    )

    smile_fig = go.Figure(go.Scatter(x=k_slice[_i], y=iv_slice[_i], mode="lines",
                                     name=model_name, line=dict(width=3)))
    if show_market.value:
        _m = market_grid[market_grid["expiry"] == grid_ts["expiry"].iloc[_i]]
        smile_fig.add_trace(go.Scatter(x=_m["k"], y=_m["iv"], mode="markers", name="SPY",
                                       marker=dict(size=5, color="#d62728")))
    smile_fig.update_layout(
        title=f"Smile, {T_grid[_i] * 365:.0f} days", height=560, uirevision="keep",
        xaxis_title="k = log(K/F)", yaxis_title="implied vol", yaxis=dict(rangemode="tozero"),
        margin=dict(l=50, r=10, t=40, b=40), legend=dict(x=0.6, y=0.95),
    )

    _controls = mo.vstack([v0, kappa, theta, xi, rho, mo.md("---"), use_bs, show_market,
                           maturity])
    mo.vstack([
        mo.md("# The smile is the model's fingerprint"),
        mo.hstack([_controls, mo.ui.plotly(surface), mo.ui.plotly(smile_fig)],
                  widths=[1, 2.2, 1.6], align="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you are looking at

    The surface is the Black-Scholes implied volatility of every option a model prices,
    plotted against log-moneyness $k = \log(K/F)$ and time to expiry $T$. Implied vol is
    the single number that, put into the Black-Scholes formula, reproduces the price. A model
    that agreed with Black-Scholes would draw a flat sheet. Any shape is the model telling
    you how its distribution of returns differs from a lognormal one.

    Heston (1993) lets the variance $v_t$ of returns move on its own:

    $$
    \frac{dF_t}{F_t} = \sqrt{v_t}\, dW_t, \qquad
    dv_t = \kappa(\theta - v_t)\, dt + \xi \sqrt{v_t}\, dZ_t, \qquad
    d\langle W, Z\rangle_t = \rho\, dt .
    $$

    Each parameter leaves its own mark on the surface.

    - **rho sets the skew.** With $\rho < 0$, variance rises when the price falls, so the left
      tail is fatter than the right and low strikes carry higher implied vol.
    - **xi sets the curvature.** Vol of variance mixes many lognormals with different
      variances, which fattens both tails and bends the smile upward at both ends.
    - **kappa and theta set the term structure.** Variance starts at $v_0$ and is pulled
      towards $\theta$ at rate $\kappa$, so long-dated at-the-money vol tends to
      $\sqrt{\theta}$ and $\kappa$ says how quickly it gets there. Mean reversion also
      averages out the variance shocks, which is why the skew fades with maturity.
    - **v0 sets the level** at the short end: at-the-money vol for small $T$ is close to
      $\sqrt{v_0}$.

    Black-Scholes is flat because it is the yardstick. Its returns are Gaussian with variance
    $\sigma^2 T$ at every strike and maturity, and inverting a Black-Scholes price through
    Black-Scholes gives back $\sigma$. Flip the switch to see it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try this

    Make a guess before you move the slider, then open the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "1. Set rho to 0. Which way does the skew go, and is the smile symmetric?": mo.md(
            "The downward slope disappears and the smile becomes a U that is exactly "
            "symmetric in $k$. With zero correlation, implied vol is an even function of "
            "$\\log(K/F)$ for any stochastic volatility model (Renault and Touzi 1996, "
            "Math. Finance 6(3)). Push rho positive and the skew turns upward."
        ),
        "2. With rho at 0, raise xi from 0.4 to 1.2. Which moves more, the wings or the "
        "at-the-money vol?": mo.md(
            "At short maturities the wings rise sharply and at-the-money vol falls a "
            "little: vol of variance adds curvature, not level. At one year the whole smile "
            "drops, at-the-money most, so it still bends more. The effect is strongest at "
            "short maturities, where mean reversion has not yet averaged the variance "
            "shocks away."
        ),
        "3. Set v0 = 0.01 and theta = 0.06. Is short-dated or long-dated vol higher? Now "
        "raise kappa: what happens?": mo.md(
            "Long-dated vol is higher: variance is expected to drift up from $v_0$ to "
            "$\\theta$. With kappa = 3, at-the-money vol climbs from about 10% at one day "
            "to about 20% at one year. With kappa = 10 the climb happens sooner and one-year "
            "vol reaches 23%, close to $\\sqrt{\\theta} = 24.5\\%$; the skew also flattens "
            "at every maturity beyond a few days."
        ),
    })
    return


@app.cell(hide_code=True)
def _(T_grid, go, kappa, market_ts, mo, model, np, rho, smile, theta, use_bs, v0, xi):
    _model_ts = smile.model_term_structure(model, T_grid)
    _A, _alpha = smile.power_law(market_ts["T"], market_ts["skew"])
    _T_line = np.geomspace(market_ts["T"].min(), market_ts["T"].max(), 50)
    skew_fig = go.Figure([
        go.Scatter(x=market_ts["T"], y=-market_ts["skew"], mode="markers", name="SPY",
                   marker=dict(size=7, color="#d62728")),
        go.Scatter(x=_T_line, y=_A * _T_line ** (-_alpha), mode="lines",
                   name=f"{_A:.2f} T^(-{_alpha:.2f})", line=dict(dash="dot", color="#d62728")),
        go.Scatter(x=T_grid, y=-_model_ts[:, 1], mode="lines+markers",
                   name="Black-Scholes" if use_bs.value else "Heston (sliders)",
                   line=dict(width=3)),
    ])
    skew_fig.update_layout(
        title="At-the-money skew, -d(iv)/dk at k = 0", height=420, uirevision="keep",
        xaxis=dict(type="log", title="T (years)"),
        yaxis=dict(type="log", title="-skew", range=[np.log10(0.05), np.log10(10)]),
        margin=dict(l=50, r=10, t=40, b=40), legend=dict(x=0.55, y=0.95),
    )
    mo.hstack([mo.vstack([v0, kappa, theta, xi, rho]), mo.ui.plotly(skew_fig)],
              widths=[1, 3], align="start")
    return


@app.cell(hide_code=True)
def _(market_ts, mo, smile, snapshot, snapshot_path):
    _A, _alpha = smile.power_law(market_ts["T"], market_ts["skew"])
    _short, _long = market_ts.iloc[0], market_ts.iloc[-1]
    _time = snapshot["quote_time"].iloc[0]
    _src = "last trade prices" if (snapshot["price_source"] == "last").all() else "mid quotes"
    _d = round(_short["T"] * 365)
    _short_label = f"{_d} day" + ("" if _d == 1 else "s")
    mo.md(rf"""
    ## What the market says

    The SPY skew keeps steepening as expiry shrinks, and Heston's does not. The figure above
    plots minus the at-the-money skew $\partial \sigma / \partial k$ against maturity on log
    axes. SPY goes from {-_long["skew"]:.2f} at {_long["T"] * 365:.0f} days to
    {-_short["skew"]:.2f} at {_short_label}, close to a straight line:
    $|\psi(T)| \approx {_A:.2f}\, T^{{-{_alpha:.2f}}}$. Heston's skew flattens to a
    constant, roughly $\rho\xi / (4\sqrt{{v_0}})$, as $T \to 0$, and decays like $1/T$
    once $T$ is well past $1/\kappa$.

    No slider setting fixes both ends. Heston can make a one-day skew as steep as the
    market's (try xi = 2, rho = -0.99), but then the one-year skew is far too steep. A least
    squares fit of all five parameters to the SPY at-the-money vols and skews, at every
    expiry at once, pushes rho to its bound of $-0.99$ and still leaves the one-day skew
    near 1.0 against the market's 1.6, and the one-year skew at 0.19 against 0.27 (fit on
    the 2026-10-07 snapshot, see `LOG.md`).

    A power law in $T$ is what rough volatility predicts. If volatility is driven by a
    fractional Brownian motion with Hurst exponent $H < 1/2$, the short-dated skew behaves
    like $T^{{H - 1/2}}$ (Bayer, Friz and Gatheral 2016, *Pricing under rough volatility*,
    Quant. Finance 16(6)), and time series of realised volatility suggest $H \approx 0.1$
    (Gatheral, Jaisson and Rosenbaum 2018, *Volatility is rough*, Quant. Finance 18(6)).
    This snapshot gives an exponent of {_alpha:.2f}, so $H \approx {0.5 - _alpha:.2f}$:
    rougher than Brownian motion, less rough than the time series estimate. The exponent
    depends on the window used to measure the skew, which Session 04 takes up with rough
    Bergomi behind the same interface.

    Data: SPY options from Yahoo Finance, {_src} at {_time:%Y-%m-%d %H:%M} UTC, file
    `data/snapshots/{snapshot_path.name}`. Forwards come from put-call parity per expiry,
    discount factors from the 13-week Treasury bill rate.
    """)
    return


if __name__ == "__main__":
    app.run()
