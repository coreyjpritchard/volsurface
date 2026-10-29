import marimo

__generated_with = "0.25.2"
app = marimo.App(width="compact", app_title="Volatility lab (mobile)")


@app.cell(hide_code=True)
async def _():
    import sys

    import marimo as mo

    # The phone layout of dashboard/vol_lab.py: same environments, presets and library
    # calls, one column, one plot at a time, coarser grid. WEB is the static build
    # (scripts/build_public_vollab.py), where this page lives in m/ next to the desktop one.
    WEB = sys.platform == "emscripten"
    REPO_URL = "https://github.com/coreyjpritchard/volsurface"
    BASE = mo.notebook_location()
    if WEB:
        import micropip

        if BASE.name == "m":
            BASE = BASE.parent
        await micropip.install(["numpy", "scipy", "pandas", "tzdata"])
        for _wheel in ("plotly-7.1.0-py3-none-any.whl", "volsurface-0.1.0-py3-none-any.whl"):
            await micropip.install(str(BASE / "public" / _wheel), deps=False)
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from volsurface import environments as env_lib

    # A coarser grid than the desktop page (8 expiries by 28 strikes against 10 by 45).
    MOBILE_T = np.array([7, 14, 30, 61, 91, 182, 365, 730]) / 365.0
    MOBILE_Z = np.linspace(-3.0, 2.5, 28)
    return BASE, MOBILE_T, MOBILE_Z, REPO_URL, WEB, env_lib, go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(WEB, mo):
    _switch = '<a href="../?desktop=1" target="_self">Desktop version</a> · ' if WEB else ""
    mo.md(f"""
    # Volatility lab

    Pick a model, open **Parameters** to move it, and switch views with the tabs. Every
    number is computed in your browser by the `volsurface` library.

    <small>{_switch}Best on a wide screen for the full layout.</small>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Same environments and presets as dashboard/vol_lab.py (tests/test_vol_lab_mobile.py
    # checks they match); shorter labels for a phone.
    ENVIRONMENTS = {
        "Black-Scholes (flat vol)": "flat",
        "Heston": "heston",
        "SABR": "sabr",
        "SVI": "svi",
        "Rough Bergomi": "rbergomi",
        "SPX 14 Oct 2026: SVI fit": "spx_svi",
        "SPX 14 Oct 2026: SANOS fit": "spx_sanos",
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
        options=ENVIRONMENTS, value="Heston", label="Model", full_width=True
    )
    return PRESETS, environment


@app.cell(hide_code=True)
def _(PRESETS, environment, mo):
    _names = list(PRESETS[environment.value])
    preset = mo.ui.dropdown(options=_names, value=_names[0], label="Preset", full_width=True)
    return (preset,)


@app.cell(hide_code=True)
def _(PRESETS, environment, mo, preset):
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
    # Shown from this cell, which does not re-run when a slider moves, so the panel stays
    # open while you drag.
    mo.vstack(
        [
            environment,
            preset,
            mo.accordion(
                {f"Parameters ({len(params)})": mo.vstack(list(params.elements.values()), gap=0.8)}
            )
            if len(params)
            else mo.md("*A fitted surface: no parameters.*"),
        ],
        gap=0.5,
    )
    return (params,)


@app.cell(hide_code=True)
async def _(BASE, WEB, env_lib, mo):
    if WEB:
        import io

        from pyodide.http import pyfetch

        _resp = await pyfetch(str(BASE / "public" / "spx_fits.npz"))
        spx = env_lib.load_spx_presets(io.BytesIO(await _resp.bytes()))
    else:
        spx = mo.cache(env_lib.load_spx_presets)()
    return (spx,)


@app.cell(hide_code=True)
def _(MOBILE_T, MOBILE_Z, REPO_URL, WEB, env_lib, environment, mo, params, spx):
    key = environment.value
    _p = {name: float(v) for name, v in params.value.items()}
    if key in ("spx_svi", "spx_sanos"):
        env = spx[key]
        grid = env_lib.spx_grid(env, n_T=MOBILE_T.size, z=MOBILE_Z)
    else:
        grid = env_lib.default_grid(T=MOBILE_T, z=MOBILE_Z)
        if key == "flat":
            env = env_lib.flat_environment(**_p)
        elif key == "heston":
            env = env_lib.heston_environment(**_p)
        elif key == "sabr":
            env = env_lib.sabr_environment(**_p)
        elif key == "svi":
            env = env_lib.svi_environment(grid.T, **_p)
        else:
            mo.stop(
                WEB,
                mo.callout(
                    mo.md(
                        "**Rough Bergomi is not supported on the web version: clone the "
                        f"project to run it.** [{REPO_URL}]({REPO_URL})"
                    ),
                    kind="warn",
                ),
            )
            try:
                env = env_lib.rbergomi_environment(**_p)
            except KeyError:
                mo.stop(
                    True,
                    mo.callout(
                        mo.md("Rough Bergomi needs the `rough-volatility` dev extra."),
                        kind="warn",
                    ),
                )
    iv = env.iv(grid)
    arb = env_lib.arbitrage_map(env, grid)
    _status = (
        "No static arbitrage on the grid"
        if arb.ok
        else "Arbitrage: "
        + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in arb.counts.items() if v)
    )
    _note = f"\n\n<small>{env.note}</small>" if env.note else ""
    mo.callout(mo.md(_status + _note), kind="success" if arb.ok else "danger")
    return arb, env, grid, iv, key


@app.cell(hide_code=True)
def _(mo):
    view = mo.ui.tabs(
        {"Surface": "", "Smile": "", "Term": "", "Greeks": "", "Arbitrage": ""}, value="Surface"
    )
    mo.output.replace(view)
    return (view,)


@app.cell(hide_code=True)
def _(MOBILE_T, MOBILE_Z, env_lib, environment, mo, spx):
    # Depends on the environment only, so moving a slider keeps the chosen expiry.
    _T = (
        env_lib.spx_grid(spx[environment.value], n_T=MOBILE_T.size, z=MOBILE_Z).T
        if environment.value in ("spx_svi", "spx_sanos")
        else MOBILE_T
    )
    _labels = {f"{t * 365:.0f} days": i for i, t in enumerate(_T)}
    expiry = mo.ui.dropdown(
        options=_labels,
        value=list(_labels)[min(4, len(_labels) - 1)],
        label="Expiry",
        full_width=True,
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
        label="Greek",
        full_width=True,
    )
    return expiry, greek


@app.cell(hide_code=True)
def _():
    PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}

    def style(fig, height, title):
        """Phone styling: full width, readable fonts, no drag-zoom stealing the scroll."""
        fig.update_layout(
            title=dict(text=title, font=dict(size=15), x=0.02, xanchor="left"),
            height=height,
            autosize=True,
            font=dict(size=13),
            margin=dict(l=48, r=12, t=44, b=44),
            dragmode=False,
            hovermode="closest",
            uirevision="keep",
            legend=dict(orientation="h", x=0, y=-0.22, font=dict(size=12)),
        )
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        return fig

    return PLOT_CONFIG, style


@app.cell(hide_code=True)
def _(
    PLOT_CONFIG,
    arb,
    env,
    env_lib,
    expiry,
    go,
    greek,
    grid,
    iv,
    key,
    make_subplots,
    mo,
    np,
    style,
    view,
):
    _T2 = np.repeat(grid.T[:, None], grid.z.size, axis=1)
    _bad = np.zeros(iv.shape, dtype=bool)
    for _m in arb.masks.values():
        _bad |= _m
    _mk = env.params.get("market") if key.startswith("spx") else None

    def _plot(fig):
        return mo.ui.plotly(fig, config=PLOT_CONFIG)

    def _read(text):
        return mo.accordion({"How to read this": mo.md(text)})

    if view.value == "Surface":
        _fig = go.Figure(
            go.Surface(
                x=grid.k,
                y=_T2,
                z=iv,
                colorscale="Viridis",
                showscale=False,
                hovertemplate="k %{x:.2f}<br>T %{y:.2f}<br>vol %{z:.3f}<extra></extra>",
            )
        )
        if not arb.ok:
            _fig.add_trace(
                go.Scatter3d(
                    x=grid.k[_bad],
                    y=_T2[_bad],
                    z=np.nan_to_num(iv[_bad]),
                    mode="markers",
                    marker=dict(size=3, color="#d62728"),
                    name="arbitrage",
                )
            )
        style(_fig, 400, "Implied vol surface")
        _fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
            scene=dict(
                xaxis_title="k",
                yaxis_title="T",
                zaxis_title="vol",
                camera=dict(eye=dict(x=-1.5, y=-1.8, z=0.9)),
                xaxis=dict(title_font=dict(size=12), tickfont=dict(size=10)),
                yaxis=dict(title_font=dict(size=12), tickfont=dict(size=10)),
                zaxis=dict(title_font=dict(size=12), tickfont=dict(size=10)),
            ),
        )
        _out = mo.vstack(
            [
                _plot(_fig),
                _read(
                    "Black implied vol for every strike and expiry; k = log(K/F). Drag to "
                    "rotate. Tilt is correlation, curvature is vol of vol, and how the short "
                    "end bends is the term structure of the skew. Red points fail an "
                    "arbitrage check. The grid is 8 expiries by 28 strikes, coarser than the "
                    "desktop page."
                ),
            ]
        )
    elif view.value == "Smile":
        _i = expiry.value
        _fig = go.Figure()
        for _j in range(grid.T.size):
            _fig.add_trace(
                go.Scatter(
                    x=grid.k[_j],
                    y=iv[_j],
                    mode="lines",
                    line=dict(width=1, color="lightgrey"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        _fig.add_trace(
            go.Scatter(
                x=grid.k[_i],
                y=iv[_i],
                mode="lines",
                line=dict(width=3),
                name=f"{grid.T[_i] * 365:.0f} days",
            )
        )
        if _bad[_i].any():
            _fig.add_trace(
                go.Scatter(
                    x=grid.k[_i][_bad[_i]],
                    y=iv[_i][_bad[_i]],
                    mode="markers",
                    name="arbitrage",
                    marker=dict(color="#d62728", size=8),
                )
            )
        if _mk is not None:
            _near = np.abs(_mk["T"] - grid.T[_i]) <= 0.08 * grid.T[_i] + 1.0 / 365
            _sel = _near & (_mk["k"] >= grid.k[_i].min()) & (_mk["k"] <= grid.k[_i].max())
            _fig.add_trace(
                go.Scatter(
                    x=_mk["k"][_sel],
                    y=_mk["iv"][_sel],
                    mode="markers",
                    name="SPX quotes",
                    marker=dict(color="black", size=4),
                )
            )
        style(_fig, 360, f"Smile at {grid.T[_i] * 365:.0f} days")
        _fig.update_xaxes(title_text="k = log(K/F)")
        _fig.update_yaxes(title_text="implied vol")
        _out = mo.vstack(
            [
                expiry,
                _plot(_fig),
                _read(
                    "Implied vol against log-moneyness at one expiry; grey lines are the "
                    "other expiries. A downward slope is the equity skew: puts cost more "
                    "vol than calls. For the SPX fits the black dots are the quotes the "
                    "surface was fitted to."
                ),
            ]
        )
    elif view.value == "Term":
        _Tt = np.geomspace(grid.T[0], grid.T[-1], 16)
        _atm = np.array(
            [float(env.iv(env_lib.Grid(np.array([t]), np.array([0.0])))[0, 0]) for t in _Tt]
        )
        _skew = env_lib.atm_skew(env, _Tt)
        _f1 = go.Figure(go.Scatter(x=_Tt, y=_atm, mode="lines+markers", name="ATM vol"))
        style(_f1, 280, "ATM vol against expiry")
        _f1.update_xaxes(title_text="T (years)")
        _f1.update_yaxes(title_text="implied vol")
        _ok = np.isfinite(_skew) & (np.abs(_skew) > 1e-8)
        _f2 = go.Figure(
            go.Scatter(
                x=_Tt[_ok],
                y=np.abs(_skew[_ok]),
                mode="lines+markers",
                line=dict(color="#d62728"),
                name="|skew|",
            )
        )
        _title = "|ATM skew|, log-log"
        if _ok.sum() >= 4:
            _slope = np.polyfit(np.log(_Tt[_ok]), np.log(np.abs(_skew[_ok])), 1)[0]
            _title += f" (slope {_slope:.2f})"
        style(_f2, 280, _title)
        _f2.update_xaxes(title_text="T (years)", type="log")
        _f2.update_yaxes(title_text="|d sigma / dk|", type="log")
        _out = mo.vstack(
            [
                _plot(_f1),
                _plot(_f2),
                _read(
                    "Top: at-the-money vol by expiry. Bottom: the size of the at-the-money "
                    "skew on log-log axes, where a power law is a straight line. Diffusions "
                    "such as Heston bend (flat at the short end, then like 1/T); rough "
                    "volatility stays straight with slope H - 1/2."
                ),
            ]
        )
    elif view.value == "Greeks":
        _i = expiry.value
        _g = greek.value
        _G = env_lib.greeks_grid(grid, iv)
        _units = {
            "delta": "call delta",
            "gamma": "per 1 point of F",
            "vega": "per vol point",
            "vanna": "delta per vol point",
            "volga": "vega per vol point",
            "theta": "per day",
        }
        _f1 = go.Figure(
            go.Scatter(
                x=grid.k[_i],
                y=_G[_g][_i],
                name="sticky strike",
                line=dict(color="#1f77b4", width=3),
            )
        )
        if _g == "delta":
            _ds = env_lib.smile_delta(env, grid, iv)
            _f1.add_trace(
                go.Scatter(
                    x=grid.k[_i],
                    y=_ds[_i],
                    name="smile moves with F",
                    line=dict(color="#ff7f0e", dash="dash", width=3),
                )
            )
        style(_f1, 320, f"{_g.capitalize()} at {grid.T[_i] * 365:.0f} days ({_units[_g]})")
        _f1.update_xaxes(title_text="k = log(K/F)")
        _f1.update_layout(showlegend=_g == "delta")
        _f2 = go.Figure(
            go.Surface(
                x=grid.k,
                y=_T2,
                z=_G[_g],
                colorscale="RdBu",
                showscale=False,
                cmid=0.0 if _g in ("vanna", "volga", "theta") else None,
            )
        )
        style(_f2, 340, f"{_g.capitalize()} surface")
        _f2.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis_title="k",
                yaxis_title="T",
                zaxis_title=_g,
                camera=dict(eye=dict(x=-1.5, y=-1.8, z=0.9)),
            ),
        )
        _out = mo.vstack(
            [
                greek,
                expiry,
                _plot(_f1),
                _plot(_f2),
                _read(
                    "Black Greeks at each option's own implied vol, per 100 of forward, zero "
                    "rates. Gamma piles up at the money and at short expiries; theta is its "
                    "mirror image; vega grows with expiry; vanna changes sign across the "
                    "money; volga is positive in the wings. For delta the dashed line lets "
                    "the smile move with the forward."
                ),
            ]
        )
    else:
        _names = ["bounds", "call_spread", "butterfly", "calendar"]
        _fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f"{n.replace('_', ' ')}: {arb.counts[n]}" for n in _names],
            horizontal_spacing=0.06,
            vertical_spacing=0.16,
        )
        for _n, _name in enumerate(_names):
            _fig.add_trace(
                go.Heatmap(
                    z=arb.masks[_name].astype(int),
                    x=grid.z,
                    y=np.round(grid.T * 365).astype(int),
                    colorscale=[[0, "#e8f3e8"], [1, "#d62728"]],
                    zmin=0,
                    zmax=1,
                    showscale=False,
                    hovertemplate="z %{x:.2f}<br>%{y} days<extra></extra>",
                ),
                row=_n // 2 + 1,
                col=_n % 2 + 1,
            )
        _fig.update_yaxes(type="category", tickfont=dict(size=10))
        _fig.update_xaxes(tickfont=dict(size=10))
        style(_fig, 440, "Where the checks fail (red)")
        _fig.update_annotations(font=dict(size=12))
        _out = mo.vstack(
            [
                _plot(_fig),
                _read(
                    "Each panel is one static-arbitrage check on the grid: price bounds, "
                    "call spreads (prices must fall with strike), butterflies (convexity) "
                    "and calendars (total variance must rise with expiry). Rows are "
                    "expiries in days, columns standardised strikes. The checks are "
                    "pointwise, so a pass says nothing between nodes."
                ),
            ]
        )
    mo.output.replace(_out)
    return


@app.cell(hide_code=True)
def _(REPO_URL, mo):
    mo.accordion(
        {
            "About this page": mo.md(f"""
    Runs entirely in your browser (Python compiled to WebAssembly with Pyodide); nothing
    you do is sent anywhere. Models: Black-Scholes, Heston (Lewis Fourier prices), SABR
    (Hagan's formula), raw SVI, and the SVI and SANOS fits to the SPX option chain of
    14 October 2026. Rough Bergomi needs the full project: clone
    [{REPO_URL}]({REPO_URL}) and run `make dashboard`.
    """)
        }
    )
    return


if __name__ == "__main__":
    app.run()
