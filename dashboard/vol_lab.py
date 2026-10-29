import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Volatility environment lab")


@app.cell(hide_code=True)
async def _():
    import sys

    import marimo as mo

    # WEB is the static build (scripts/build_public_vollab.py): Pyodide in the visitor's
    # browser, the library wheel and the SPX fits served next to the page, no server.
    WEB = sys.platform == "emscripten"
    REPO_URL = "https://github.com/coreyjpritchard/volsurface"
    if WEB:
        import micropip

        # numpy, scipy, pandas and tzdata (the chain clock uses New York time) come with
        # Pyodide; plotly and volsurface ship in public/.
        await micropip.install(["numpy", "scipy", "pandas", "tzdata"])
        for _wheel in ("plotly-7.1.0-py3-none-any.whl", "volsurface-0.1.0-py3-none-any.whl"):
            await micropip.install(str(mo.notebook_location() / "public" / _wheel), deps=False)
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from volsurface import environments as env_lib

    return REPO_URL, WEB, env_lib, go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(WEB, mo):
    if WEB:
        mo.output.append(mo.md('<small><a href="m/" target="_self">Mobile version</a></small>'))
    mo.output.append(
        mo.md(r"""
    # Volatility environment lab

    Pick a volatility environment, move its parameters, and watch three things change
    together: the implied volatility surface, the Greeks of every option on it, and whether
    the surface is still free of static arbitrage. Every number is computed live by the
    `volsurface` library: Lewis Fourier prices for Heston, Hagan's formula for SABR, raw
    SVI slices, the rough Bergomi plug-in (Monte Carlo), and the SVI and SANOS fits to the
    SPX option chain of 14 October 2026.
    """)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    ENVIRONMENTS = {
        "Black-Scholes: one flat vol": "flat",
        "Heston: stochastic variance": "heston",
        "SABR: stochastic vol per expiry": "sabr",
        "SVI: parametric smile slices": "svi",
        "Rough Bergomi: rough volatility (Monte Carlo)": "rbergomi",
        "SPX 14 Oct 2026: raw SVI fit per expiry": "spx_svi",
        "SPX 14 Oct 2026: SANOS fit (arbitrage-free)": "spx_sanos",
    }
    PRESETS = {
        "flat": {"Calm market": {"sigma": 0.15}, "Stressed market": {"sigma": 0.45}},
        "heston": {
            "Equity index": dict(v0=0.02, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7),
            "Vol of vol up": dict(v0=0.02, kappa=2.0, theta=0.04, xi=1.5, rho=-0.7),
            "No correlation (symmetric smile)": dict(
                v0=0.04, kappa=2.0, theta=0.04, xi=0.8, rho=0.0
            ),
            "Inverted term structure (stress)": dict(
                v0=0.16, kappa=3.0, theta=0.04, xi=0.8, rho=-0.8
            ),
            "SPX fit in the README (xi on its bound)": dict(
                v0=0.00287, kappa=8.72, theta=0.0642, xi=5.0, rho=-0.625
            ),
        },
        "sabr": {
            "Equity-like skew": dict(alpha=0.2, beta=0.7, rho=-0.5, nu=0.6),
            "Rates-like smile": dict(alpha=0.2, beta=0.5, rho=-0.1, nu=0.4),
            "High vol of vol (density goes negative)": dict(alpha=0.2, beta=0.3, rho=-0.5, nu=1.5),
        },
        "svi": {
            "Equity-like": dict(
                atm_vol=0.2, term_slope=0.05, b=0.06, rho=-0.6, m=0.0, sigma=0.15, skew_decay=0.4
            ),
            "Calendar arbitrage (ATM total variance falls)": dict(
                atm_vol=0.12, term_slope=-0.6, b=0.05, rho=-0.6, m=0.0, sigma=0.15, skew_decay=0.3
            ),
            "Butterfly arbitrage (wings too steep)": dict(
                atm_vol=0.2, term_slope=0.0, b=0.4, rho=-0.95, m=0.0, sigma=0.02, skew_decay=0.3
            ),
        },
        "rbergomi": {
            "Rough, as in Gatheral et al. (H = 0.1)": dict(H=0.1, eta=1.9, rho=-0.9, sigma0=0.2),
            "SPX free-H fit in the README": dict(H=0.207, eta=2.35, rho=-0.66, sigma0=0.13),
            "Almost Brownian (H = 0.45)": dict(H=0.45, eta=1.0, rho=-0.7, sigma0=0.2),
        },
        "spx_svi": {"As fitted": {}},
        "spx_sanos": {"As fitted": {}},
    }
    environment = mo.ui.dropdown(
        options=ENVIRONMENTS, value="Heston: stochastic variance", label="Environment"
    )
    return ENVIRONMENTS, PRESETS, environment


@app.cell(hide_code=True)
def _(PRESETS, environment, mo):
    _names = list(PRESETS[environment.value])
    preset = mo.ui.dropdown(options=_names, value=_names[0], label="Start from")
    return (preset,)


@app.cell(hide_code=True)
def _(PRESETS, environment, mo, preset):
    # Slider ranges and labels per environment. The preset sets the starting values.
    _specs = {
        "flat": [("sigma", 0.05, 0.8, 0.01, "sigma, the one volatility")],
        "heston": [
            ("v0", 0.0025, 0.25, 0.0025, "v0, variance today"),
            ("kappa", 0.1, 15.0, 0.1, "kappa, mean reversion speed"),
            ("theta", 0.0025, 0.25, 0.0025, "theta, long-run variance"),
            ("xi", 0.05, 5.0, 0.05, "xi, vol of variance"),
            ("rho", -0.99, 0.99, 0.01, "rho, spot-variance correlation"),
        ],
        "sabr": [
            ("alpha", 0.05, 0.8, 0.01, "alpha, vol level (forward = 1)"),
            ("beta", 0.0, 1.0, 0.05, "beta, backbone exponent"),
            ("rho", -0.99, 0.99, 0.01, "rho, forward-vol correlation"),
            ("nu", 0.0, 2.5, 0.05, "nu, vol of vol"),
        ],
        "svi": [
            ("atm_vol", 0.05, 0.6, 0.01, "ATM vol at 1 year"),
            ("term_slope", -1.0, 0.5, 0.05, "ATM vol term exponent: vol ~ T^p"),
            ("b", 0.0, 0.6, 0.01, "b, wing slope at 1 year"),
            ("rho", -0.99, 0.99, 0.01, "rho, wing asymmetry"),
            ("m", -0.3, 0.3, 0.01, "m, horizontal shift"),
            ("sigma", 0.01, 0.6, 0.01, "sigma, ATM curvature (smaller is sharper)"),
            ("skew_decay", 0.0, 1.5, 0.05, "skew decay: b ~ T^-gamma"),
        ],
        "rbergomi": [
            ("H", 0.05, 0.49, 0.01, "H, Hurst index (roughness)"),
            ("eta", 0.2, 3.0, 0.05, "eta, vol of vol"),
            ("rho", -0.99, 0.0, 0.01, "rho, spot-vol correlation"),
            ("sigma0", 0.05, 0.6, 0.01, "sigma0, flat forward vol"),
        ],
        "spx_svi": [],
        "spx_sanos": [],
    }
    _values = PRESETS[environment.value][preset.value]
    params = mo.ui.dictionary(
        {
            name: mo.ui.slider(
                lo,
                hi,
                step=step,
                value=_values[name],
                label=label,
                show_value=True,
                debounce=True,
                full_width=True,
            )
            for name, lo, hi, step, label in _specs[environment.value]
        }
    )
    return (params,)


@app.cell(hide_code=True)
async def _(WEB, env_lib, mo):
    # The SPX fits are loaded once from data/presets/spx_fits.npz. No refitting here.
    # The web build ships the same file in its public folder and reads it from there.
    if WEB:
        import io

        from pyodide.http import pyfetch

        _resp = await pyfetch(str(mo.notebook_location() / "public" / "spx_fits.npz"))
        spx = env_lib.load_spx_presets(io.BytesIO(await _resp.bytes()))
    else:
        spx = mo.cache(env_lib.load_spx_presets)()
    return (spx,)


@app.cell(hide_code=True)
def _(REPO_URL, WEB, env_lib, environment, mo, np, params, spx):
    key = environment.value
    p = {name: float(v) for name, v in params.value.items()}
    if key in ("spx_svi", "spx_sanos"):
        env = spx[key]
        grid = env_lib.spx_grid(env)
    else:
        grid = env_lib.default_grid()
        if key == "flat":
            env = env_lib.flat_environment(**p)
        elif key == "heston":
            env = env_lib.heston_environment(**p)
        elif key == "sabr":
            env = env_lib.sabr_environment(**p)
        elif key == "svi":
            env = env_lib.svi_environment(grid.T, **p)
        else:
            # The environment menu is shown with the message so the visitor can switch back.
            mo.stop(
                WEB,
                mo.vstack(
                    [
                        environment,
                        mo.callout(
                            mo.md(
                                "**Rough Bergomi is not supported on the web version: clone the "
                                f"project to run it.** [{REPO_URL}]({REPO_URL}), then "
                                "`make dashboard`. It is a Monte Carlo model from the separate "
                                "rough-volatility package."
                            ),
                            kind="warn",
                        ),
                    ]
                ),
            )
            try:
                env = env_lib.rbergomi_environment(**p)
            except KeyError:
                mo.stop(
                    True,
                    mo.vstack(
                        [
                            environment,
                            mo.callout(
                                mo.md(
                                    "Rough Bergomi needs the `rough-volatility` package, a dev "
                                    "extra: `uv sync --extra dev`, or run `make dashboard`."
                                ),
                                kind="warn",
                            ),
                        ]
                    ),
                )
    iv = env.iv(grid)
    greeks = env_lib.greeks_grid(grid, iv)
    delta_smile = env_lib.smile_delta(env, grid, iv)
    arb = env_lib.arbitrage_map(env, grid)
    T_term = np.geomspace(grid.T[0], grid.T[-1], 24)
    skew_term = env_lib.atm_skew(env, T_term)
    atm_term = np.array(
        [float(env.iv(env_lib.Grid(np.array([t]), np.array([0.0])))[0, 0]) for t in T_term]
    )
    return T_term, arb, atm_term, delta_smile, env, greeks, grid, iv, key, skew_term


@app.cell(hide_code=True)
def _(env_lib, environment, mo, spx):
    # Depends on the environment only, so moving a slider keeps the chosen expiry.
    _key = environment.value
    _T = (
        env_lib.spx_grid(spx[_key]).T
        if _key in ("spx_svi", "spx_sanos")
        else env_lib.default_grid().T
    )
    _labels = {f"{t * 365:.0f} days": i for i, t in enumerate(_T)}
    expiry = mo.ui.dropdown(
        options=_labels, value=list(_labels)[min(4, len(_labels) - 1)], label="Slice at expiry"
    )
    greek = mo.ui.dropdown(
        options={
            "Delta (call)": "delta",
            "Gamma": "gamma",
            "Vega": "vega",
            "Vanna": "vanna",
            "Volga": "volga",
            "Theta (per day)": "theta",
        },
        value="Gamma",
        label="Greek surface",
    )
    x_axis = mo.ui.radio(
        options={"log-moneyness k = log(K/F)": "k", "strike per 100 of forward": "K"},
        value="log-moneyness k = log(K/F)",
        label="Horizontal axis",
    )
    show_flags = mo.ui.checkbox(value=True, label="Mark arbitrage on the surface")
    return expiry, greek, show_flags, x_axis


@app.cell(hide_code=True)
def _(arb, env, environment, expiry, greek, mo, params, preset, show_flags, x_axis):
    _note = f"\n\n*{env.note}*" if env.note else ""
    _status = (
        "No static arbitrage on the grid"
        if arb.ok
        else "Arbitrage flagged: " + ", ".join(f"{k} {v}" for k, v in arb.counts.items() if v)
    )
    _kind = mo.callout(mo.md(_status), kind="success" if arb.ok else "danger")
    controls = mo.vstack(
        [
            environment,
            preset,
            mo.md("**Parameters**") if len(params) else mo.md("*No parameters: a fitted surface.*"),
            *params.elements.values(),
            mo.md(
                "| now | value |\n|---|---|\n"
                + "\n".join(f"| {k} | {v:.4g} |" for k, v in params.value.items())
            )
            if len(params)
            else mo.md(""),
            mo.md("---"),
            expiry,
            greek,
            x_axis,
            show_flags,
            _kind,
            mo.md(_note),
        ],
        gap=0.4,
    )
    return (controls,)


@app.cell(hide_code=True)
def _(arb, env, expiry, go, grid, iv, key, np, show_flags, x_axis):
    def xs(g=grid):
        return g.k if x_axis.value == "k" else g.strike

    X_LABEL = "k = log(K/F)" if x_axis.value == "k" else "strike (forward = 100)"
    _T2 = np.repeat(grid.T[:, None], grid.z.size, axis=1)
    surface_fig = go.Figure(
        go.Surface(
            x=xs(),
            y=_T2,
            z=iv,
            colorscale="Viridis",
            showscale=False,
            opacity=0.95,
            name="implied vol",
            hovertemplate="x %{x:.3f}<br>T %{y:.3f}<br>vol %{z:.3f}<extra></extra>",
        )
    )
    if show_flags.value and not arb.ok:
        _bad = np.zeros(iv.shape, dtype=bool)
        for _m in arb.masks.values():
            _bad |= _m
        surface_fig.add_trace(
            go.Scatter3d(
                x=xs()[_bad],
                y=_T2[_bad],
                z=np.nan_to_num(iv[_bad]),
                mode="markers",
                marker=dict(size=3, color="#d62728"),
                name="arbitrage",
            )
        )
    if key.startswith("spx"):
        _mk = env.params["market"]
        _in = (_mk["T"] >= grid.T[0]) & (_mk["T"] <= grid.T[-1])
        _in &= np.abs(_mk["k"]) <= 0.6 * np.sqrt(_mk["T"]) + 0.02
        _mx = _mk["k"][_in] if x_axis.value == "k" else 100 * np.exp(_mk["k"][_in])
        surface_fig.add_trace(
            go.Scatter3d(
                x=_mx,
                y=_mk["T"][_in],
                z=_mk["iv"][_in],
                mode="markers",
                marker=dict(size=1.5, color="black"),
                name="SPX quotes",
            )
        )
    surface_fig.update_layout(
        title="Implied volatility surface",
        height=480,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        uirevision="keep",
        scene=dict(
            xaxis_title=X_LABEL,
            yaxis_title="T (years)",
            zaxis_title="implied vol",
            camera=dict(eye=dict(x=-1.4, y=-1.7, z=0.8)),
        ),
    )

    _i = expiry.value
    smile_fig = go.Figure()
    for _j in range(grid.T.size):
        smile_fig.add_trace(
            go.Scatter(
                x=xs()[_j],
                y=iv[_j],
                mode="lines",
                line=dict(width=1, color="lightgrey"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    smile_fig.add_trace(
        go.Scatter(
            x=xs()[_i],
            y=iv[_i],
            mode="lines",
            line=dict(width=3),
            name=f"{grid.T[_i] * 365:.0f} days",
        )
    )
    if show_flags.value:
        _bad = np.zeros(grid.z.size, dtype=bool)
        for _m in arb.masks.values():
            _bad |= _m[_i]
        if _bad.any():
            smile_fig.add_trace(
                go.Scatter(
                    x=xs()[_i][_bad],
                    y=iv[_i][_bad],
                    mode="markers",
                    name="arbitrage",
                    marker=dict(color="#d62728", size=7),
                )
            )
    if key.startswith("spx"):
        _mk = env.params["market"]
        _near = np.abs(_mk["T"] - grid.T[_i]) <= 0.08 * grid.T[_i] + 1.0 / 365
        _sel = _near & (_mk["k"] >= grid.k[_i].min()) & (_mk["k"] <= grid.k[_i].max())
        _mx = _mk["k"][_sel] if x_axis.value == "k" else 100 * np.exp(_mk["k"][_sel])
        smile_fig.add_trace(
            go.Scatter(
                x=_mx,
                y=_mk["iv"][_sel],
                mode="markers",
                name="SPX quotes near this expiry",
                marker=dict(color="black", size=4),
            )
        )
    _ = smile_fig.update_layout(
        title=f"Smile at {grid.T[_i] * 365:.0f} days (grey: other expiries)",
        height=480,
        xaxis_title=X_LABEL,
        yaxis_title="implied vol",
        margin=dict(l=50, r=10, t=40, b=40),
        legend=dict(x=0.02, y=0.98),
        uirevision="keep",
    )
    return X_LABEL, smile_fig, surface_fig, xs


@app.cell(hide_code=True)
def _(T_term, atm_term, go, make_subplots, np, skew_term):
    term_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("ATM vol against expiry", "|ATM skew| against expiry, log-log"),
    )
    term_fig.add_trace(
        go.Scatter(x=T_term, y=atm_term, mode="lines+markers", name="ATM vol"), row=1, col=1
    )
    _ok = np.isfinite(skew_term) & (np.abs(skew_term) > 1e-8)
    term_fig.add_trace(
        go.Scatter(x=T_term[_ok], y=np.abs(skew_term[_ok]), mode="lines+markers", name="|skew|"),
        row=1,
        col=2,
    )
    if _ok.sum() >= 4:
        _slope = np.polyfit(np.log(T_term[_ok]), np.log(np.abs(skew_term[_ok])), 1)[0]
        term_fig.add_annotation(
            text=f"fitted slope {_slope:.2f}",
            xref="x2 domain",
            yref="y2 domain",
            x=0.95,
            y=0.95,
            showarrow=False,
        )
    term_fig.update_xaxes(title_text="T (years)", row=1, col=1)
    term_fig.update_xaxes(title_text="T (years)", type="log", row=1, col=2)
    term_fig.update_yaxes(title_text="implied vol", row=1, col=1)
    term_fig.update_yaxes(title_text="|d sigma / dk| at k = 0", type="log", row=1, col=2)
    _ = term_fig.update_layout(height=360, showlegend=False, margin=dict(l=50, r=10, t=40, b=40))
    return (term_fig,)


@app.cell(hide_code=True)
def _(X_LABEL, delta_smile, expiry, go, greek, greeks, grid, make_subplots, np, xs):
    _units = {
        "delta": "call delta",
        "gamma": "delta per 1 point of F",
        "vega": "price per vol point",
        "vanna": "delta per vol point",
        "volga": "vega per vol point",
        "theta": "price per day",
    }
    _g = greeks[greek.value]
    _T2 = np.repeat(grid.T[:, None], grid.z.size, axis=1)
    greek_fig = go.Figure(
        go.Surface(
            x=xs(),
            y=_T2,
            z=_g,
            colorscale="RdBu",
            showscale=False,
            cmid=0.0 if greek.value in ("vanna", "volga", "theta") else None,
        )
    )
    greek_fig.update_layout(
        title=f"{greek.value.capitalize()} surface ({_units[greek.value]})",
        height=480,
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision="keep",
        scene=dict(
            xaxis_title=X_LABEL,
            yaxis_title="T (years)",
            zaxis_title=greek.value,
            camera=dict(eye=dict(x=-1.4, y=-1.7, z=0.8)),
        ),
    )

    _i = expiry.value
    slices_fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Delta: sticky strike vs smile",
            "Gamma",
            "Vega",
            "Vanna",
            "Volga",
            "Theta per day",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.07,
    )
    slices_fig.add_trace(
        go.Scatter(
            x=xs()[_i], y=greeks["delta"][_i], name="sticky strike", line=dict(color="#1f77b4")
        ),
        row=1,
        col=1,
    )
    slices_fig.add_trace(
        go.Scatter(
            x=xs()[_i],
            y=delta_smile[_i],
            name="smile moves with F",
            line=dict(color="#ff7f0e", dash="dash"),
        ),
        row=1,
        col=1,
    )
    for _n, (_name, _r, _c) in enumerate(
        [("gamma", 1, 2), ("vega", 1, 3), ("vanna", 2, 1), ("volga", 2, 2), ("theta", 2, 3)]
    ):
        slices_fig.add_trace(
            go.Scatter(
                x=xs()[_i], y=greeks[_name][_i], showlegend=False, line=dict(color="#1f77b4")
            ),
            row=_r,
            col=_c,
        )
    _ = slices_fig.update_layout(
        height=480,
        title=f"All Greeks at {grid.T[_i] * 365:.0f} days",
        uirevision="keep",
        legend=dict(x=0.01, y=0.62, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=10, t=60, b=30),
    )
    return greek_fig, slices_fig


@app.cell(hide_code=True)
def _(arb, go, grid, make_subplots, np):
    _names = ["bounds", "call_spread", "butterfly", "calendar"]
    arb_fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[f"{n.replace('_', ' ')}: {arb.counts[n]}" for n in _names],
        horizontal_spacing=0.04,
    )
    for _c, _n in enumerate(_names, start=1):
        arb_fig.add_trace(
            go.Heatmap(
                z=arb.masks[_n].astype(int),
                x=grid.z,
                y=np.round(grid.T * 365).astype(int),
                colorscale=[[0, "#e8f3e8"], [1, "#d62728"]],
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate="z %{x:.2f}<br>%{y} days<br>flag %{z}<extra></extra>",
            ),
            row=1,
            col=_c,
        )
        arb_fig.update_xaxes(title_text="z = k / (0.2 sqrt T)", row=1, col=_c)
    arb_fig.update_yaxes(title_text="expiry (days)", type="category", row=1, col=1)
    for _c in range(2, 5):
        arb_fig.update_yaxes(type="category", showticklabels=False, row=1, col=_c)
    _ = arb_fig.update_layout(height=300, margin=dict(l=60, r=10, t=40, b=40))
    return (arb_fig,)


@app.cell(hide_code=True)
def _(F_check, controls, mo, smile_fig, surface_fig, term_fig):
    mo.vstack(
        [
            mo.hstack(
                [controls, mo.ui.plotly(surface_fig), mo.ui.plotly(smile_fig)],
                widths=[1.3, 1.9, 1.5],
                align="start",
            ),
            mo.ui.plotly(term_fig),
            F_check,
        ]
    )
    return


@app.cell(hide_code=True)
def _(env_lib, greeks, iv, mo, np):
    # The Black-Scholes equation with zero rates: theta = -0.5 sigma^2 F^2 gamma, node by node.
    _F = env_lib.NOTIONAL_F
    _lhs = greeks["theta"] * 365.0
    _rhs = -0.5 * iv**2 * _F**2 * greeks["gamma"]
    _gap = np.nanmax(np.abs(_lhs - _rhs))
    F_check = mo.md(
        f"Check, live: at every node, theta equals $-\\tfrac12\\sigma^2F^2\\Gamma$ to "
        f"{_gap:.1e}. With zero rates that is the Black-Scholes equation, so gamma and theta "
        "are the same surface with opposite sign."
    )
    return (F_check,)


@app.cell(hide_code=True)
def _(key, mo):
    _why = {
        "flat": r"""
**Black-Scholes, one flat vol.** Every option has the same implied vol, so the surface is
a flat sheet and the skew panel is empty. This is the reference: any shape in the other
environments is that model's distribution of returns differing from a lognormal one. Use
it to learn the Greeks on their own, because here they depend only on moneyness, expiry
and the one vol.
""",
        "heston": r"""
**Heston: variance is its own random process**, $dv = \kappa(\theta - v)\,dt +
\xi\sqrt v\,dW$, correlated $\rho$ with the forward.

- **rho tilts the smile.** Negative $\rho$ means variance rises when the index falls, so
  downside puts are dearer: the equity skew. At $\rho = 0$ the smile is symmetric.
- **xi bends it.** Vol of variance fattens both tails, which lifts the wings, and with
  $\rho$ it sets how steep the short-dated skew is.
- **kappa and theta set the term structure.** Variance drifts from $v_0$ towards
  $\theta$ at speed $\kappa$; with $v_0 > \theta$ the term structure is inverted, as in a
  sell-off. The skew is roughly flat for $T \ll 1/\kappa$ and decays like $1/T$ for
  $T \gg 1/\kappa$, so on log-log axes the skew line bends; it is not straight.
- **What Heston cannot do:** as $T \to 0$ its ATM skew tends to a finite limit,
  $\rho\xi/(4\sqrt{v_0})$. The SPX skew keeps rising like $T^{-1/3}$ down to 10 hours.
  The "SPX fit" preset is the README fit: to chase that skew it puts $\xi$ on its bound
  of 5 and $v_0$ very low, and then its short-dated ATM vol is far too low.
- The Feller ratio $2\kappa\theta/\xi^2$ below 1 means variance can touch zero. Prices
  are still fine (Fourier, no simulation), but it is a sign of a strained fit.
""",
        "sabr": r"""
**SABR: the forward's vol is lognormal**, $dF = \alpha F^\beta dW$, $d\alpha = \nu\alpha\,dZ$,
correlation $\rho$. Prices use Hagan's 2002 expansion, one parameter set at every expiry.

- **alpha** is the vol level, **rho** the tilt, **nu** the curvature (vol of vol).
- **beta is the backbone:** how the ATM vol moves when the forward moves. Below 1 the
  smile does not just slide with the forward, which is why the smile delta (orange,
  dashed, Greeks panel) differs from the sticky-strike delta even with $\rho = 0$.
- **Hagan's formula is an expansion, not an exact price.** With large $\nu$, small
  $\beta$ and long expiries the implied density at low strikes goes negative. Try the
  "High vol of vol" preset: the butterfly check flags exactly those strikes.
""",
        "svi": r"""
**Raw SVI**, $w(k) = a + b\{\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\}$ in total variance
$w = \sigma_{BS}^2 T$, one slice per expiry. Here the slices share $\rho$, $m$ and
$\sigma$; the ATM vol moves like $T^{p}$ and the wing slope $b$ like $T^{-\gamma}$, both
held flat below three months so the short end stays sane.

- **b and rho are the wings.** Far from the money $w$ is linear in $k$ with slopes
  $b(1 \pm \rho)$. Roger Lee's moment formula caps those slopes at 2 in total variance.
- **sigma rounds the bottom of the smile**; small sigma makes a sharp V. Steep wings on a
  sharp V make the implied density negative near the bottom: that is butterfly arbitrage
  (a butterfly that costs less than nothing), and the preset shows it.
- **Calendar arbitrage** means total variance falls as expiry grows at a fixed $k$:
  a later option is cheaper than an earlier one. With $p < -1/2$ ATM total variance
  falls with $T$ (past three months here); with $\gamma > 1$ the wings do.
  Try the two arbitrage presets.
""",
        "rbergomi": r"""
**Rough Bergomi**: log-variance driven by a fractional Brownian motion with Hurst index
$H$, vol of vol $\eta$, correlation $\rho$, and here a flat forward variance $\sigma_0^2$.

- **H is roughness.** The ATM skew grows like $T^{H - 1/2}$ as expiry shrinks: a straight
  line of slope $H - 1/2$ on the log-log panel. $H = 0.1$ gives slope $-0.4$; near
  $H = 0.5$ the line flattens, as in a classical diffusion. SPX on 14 October 2026 has
  slope $-0.333$.
- **eta and rho** scale and tilt the skew, as $\xi$ and $\rho$ do in Heston.
- **Monte Carlo:** 4,000 paths, 64 steps, seed 0, no error bars, so deep wings are noisy
  and an arbitrage flag here can be noise rather than the model.
""",
        "spx_svi": r"""
**SPX, raw SVI fitted slice by slice** to the SPXW mids of 14 October 2026 (one
snapshot, before the cash open). Each slice fits its own quotes well, but nothing ties
neighbouring slices together or keeps the wings inside Lee's bounds, so the checks flag
arbitrage. On the repository's full check grid that is 2,246 violations.
""",
        "spx_sanos": r"""
**SPX, SANOS** (arXiv:2601.11209) with smoothing $\eta = 0.25$: the surface is a mixture
of Black calls whose weights are fitted by a linear programme with the static-arbitrage
constraints built in. It is free of arbitrage by construction; the checks should stay
green. Compare the smile with the SVI fit on the same quotes.
""",
    }[key]
    mo.md(_why)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to read the surface panels

    **What is plotted.** Left: Black implied vol for every strike and expiry. The
    horizontal axis is log-moneyness $k = \log(K/F)$ (or the strike, with the forward at
    100); each expiry spans the same number of standard deviations, so the short end is not
    all wing. Middle: one expiry's smile, with the others in grey. Below: the ATM vol and the
    ATM skew $d\sigma/dk$ against expiry; on log-log axes a power law is a straight line.
    Red points are nodes where an arbitrage check fails.

    **Takeaway.** The surface is the model's fingerprint: tilt is correlation, curvature is
    vol of vol, and how the skew decays with expiry tells a diffusion (bending line) from a
    rough model (straight line).

    **What it does not show.** The grid is ten expiries by 45 strikes. Arbitrage checks are
    pointwise on that grid, so a pass says nothing between the nodes.
    """)
    return


@app.cell(hide_code=True)
def _(greek_fig, mo, slices_fig):
    mo.vstack(
        [
            mo.md("## Greeks across strike and expiry"),
            mo.hstack(
                [mo.ui.plotly(greek_fig), mo.ui.plotly(slices_fig)],
                widths=[1.0, 1.4],
                align="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What is plotted.** Black Greeks at each option's own implied vol, per 100 of forward,
    zero rates: the sticky-strike numbers a risk system shows. Choose the 3D surface with the
    "Greek surface" menu; the six small panels are every Greek at the chosen expiry.

    **Takeaway.**

    - **Gamma** piles up at the money and at short expiries, about $1/(\sigma F\sqrt{T})$:
      a one-week ATM option's delta swings far more than a one-year one's.
    - **Theta** is gamma's mirror image. With zero rates the Black-Scholes equation says
      $\Theta = -\tfrac12\sigma^2F^2\Gamma$ (checked live above): time decay pays for gamma.
    - **Vega** grows like $\sqrt{T}$ and peaks near the money; long-dated options carry the
      vol risk.
    - **Vanna** ($\partial\Delta/\partial\sigma$) is near zero at the money and has opposite
      signs on the two wings: it is the risk of a risk reversal.
    - **Volga** ($\partial \mathrm{vega}/\partial\sigma$) is near zero at the money and
      positive in the wings: wing options are long vol of vol, which is why they cost more
      when vol of vol is high.
    - **Delta, two ways.** The blue line holds the smile fixed at each strike. The dashed
      line lets the smile move with the forward: exactly the model's delta for Heston and
      rough Bergomi (their smiles depend on $K/F$ only) and for SABR, and the sticky
      moneyness assumption for the fitted SPX surfaces. With a downward skew the two
      differ by vega times the skew over $F$.

    **What it does not show.** These are not minimum-variance hedges: when vol moves with
    the index ($\rho \ne 0$) a hedger also shades delta for the expected vol move. Model
    Greeks for Monte Carlo environments would need pathwise or likelihood-ratio estimators,
    which the library has (`volsurface/greeks/`) but this page does not use.
    """)
    return


@app.cell(hide_code=True)
def _(arb, arb_fig, mo):
    mo.vstack(
        [
            mo.md("## Static arbitrage on the grid"),
            mo.ui.plotly(arb_fig),
            mo.md(rf"""
**What is plotted.** For each check, a red cell is a node where it fails
(`volsurface.arbitrage.check_surface`, tolerance $10^{{-9}}$ on the normalised call
$C/(F\,\mathrm{{df}})$). {arb.n_checked} nodes were checked.

- **bounds**: the call lies between intrinsic value and the forward.
- **call spread**: a higher strike call is never worth more, and not by more than the
  strike gap.
- **butterfly**: the call is convex in strike, so the implied density is not negative.
- **calendar**: at fixed moneyness a later expiry is never cheaper.

**Takeaway.** Any red cell is a set of option prices you could combine into a riskless
profit if they traded. Smooth formulas do not guarantee this: SABR's expansion and SVI
slices both fail it in reachable corners, while SANOS is built to pass.

**What it does not show.** {arb.note} Monte Carlo surfaces (rough Bergomi) can fail by noise.
"""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(REPO_URL, WEB, mo):
    _web = (
        "This web version runs entirely in your browser (Python compiled to WebAssembly with "
        "Pyodide); nothing you do is sent anywhere. The SPX data are one bundled snapshot. "
        "Rough Bergomi needs the full project. "
        if WEB
        else ""
    )
    mo.md(f"""
    ---
    {_web}Run locally: clone [{REPO_URL}]({REPO_URL}) and run `make dashboard` (or
    `uv run marimo run dashboard/vol_lab.py`). The library code behind every panel is in
    `volsurface/environments.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
