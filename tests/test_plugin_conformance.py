"""Rough Bergomi is the roughvol entry point, or it is absent. DESIGN section 20."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tests.test_sanos_eta0 import _heston_book, _known_surface
from volsurface import bs
from volsurface.core.protocols import capabilities
from volsurface.core.types import ParamSpec
from volsurface.fit.rbergomi import calibrate, variance_swap
from volsurface.fit.sanos import SanosFitter
from volsurface.plugins import create
from volsurface.surface.sanos import SanosSurface


def test_plugin_conformance_when_roughvol_is_installed():
    """Skip when the separate roughvol package is not on the path."""
    pytest.importorskip("roughvol")
    model = create("rbergomi")
    assert {"price", "simulate"} <= capabilities(model)
    assert type(model).__module__.startswith("roughvol")


def test_missing_plugin_is_a_key_error_and_not_a_substitute(monkeypatch):
    def absent(name, **kwargs):
        raise KeyError(f"unknown model {name!r}")

    monkeypatch.setattr("volsurface.fit.rbergomi.create", absent)
    with pytest.raises(KeyError):
        calibrate(_heston_book(0.15), H=0.167)


def test_log_contract_is_the_mixture_and_rises_with_variance():
    base = _known_surface()
    smooth = SanosSurface(
        base.T,
        base.strikes,
        base.weights,
        base.variance,
        0.25,
        base.market,
        base.domain,
        time_interp="linear",
    )
    log_mean = float(base.weights[0] @ np.log(base.strikes[0]))
    expected = -2.0 * log_mean + smooth.eta * smooth.variance
    got = variance_swap(smooth, smooth.T)
    assert got == pytest.approx(expected, abs=1e-12)
    mid = variance_swap(smooth, np.array([0.75]))
    assert mid[0] == pytest.approx(0.5 * (expected[0] + expected[1]), abs=1e-12)
    assert got[1] > got[0]


def test_heston_sanos_log_contract_is_increasing_in_maturity():
    """Calendar order of the fit. VS(T) = -2 E[log Z_T] does not fall."""
    fitted = SanosFitter(eta=0.25, omega=0.0, dx=0.05).fit(_heston_book(0.15))
    grid = np.linspace(fitted.surface.T[0], fitted.surface.T[-1], 13)
    total = variance_swap(fitted.surface, grid)
    assert np.all(np.diff(total) >= -1e-8)


class _MockVol:
    """A test double. Not rBergomi, and not labelled as one."""

    def __init__(self, H: float = 0.1, nu: float = 0.5, rho: float = -0.4, seed: int = 0):
        self.H = H
        self.nu = nu
        self.rho = rho
        self.seed = seed

    def price(self, F, K, T, df=1.0, kind="call", state=None):
        log_k = np.log(np.asarray(K, dtype=float) / np.asarray(F, dtype=float))
        sigma = np.maximum(0.2 + 0.4 * self.nu * log_k, 1e-3)
        return bs.price(F, K, T, sigma, df, kind)

    @classmethod
    def param_spec(cls) -> list[ParamSpec]:
        return [
            ParamSpec("H", 0.01, 0.49, 0.1, "identity"),
            ParamSpec("nu", 0.05, 1.5, 0.5, "log"),
            ParamSpec("rho", -0.999, 0.0, -0.4, "identity"),
        ]


def _skew_book(nu: float) -> object:
    rows = []
    for maturity in (0.5, 1.0):
        log_k = np.linspace(-0.1, 0.1, 7)
        for k in log_k:
            rows.append(
                {
                    "expiry": f"T{maturity}",
                    "T": maturity,
                    "k": float(k),
                    "iv": 0.2 + 0.4 * nu * float(k),
                    "K": 100.0 * np.exp(float(k)),
                    "kind": "put" if k < 0 else "call",
                    "bid": 1.0,
                    "ask": 1.1,
                    "F": 100.0,
                    "df": 1.0,
                }
            )
    from volsurface.core.types import MarketData

    frame = pd.DataFrame(rows)

    def flat(level):
        def curve(t):
            return np.full(np.shape(t), level, dtype=float)

        return curve

    return MarketData(
        asof=dt.datetime(2026, 10, 14, tzinfo=dt.UTC),
        quotes=frame,
        F=flat(100.0),
        df=flat(1.0),
        F_err=None,
        day_count="ACT/365",
    )


def test_mock_calibration_is_not_labelled_rbergomi(monkeypatch):
    def factory(name, **kwargs):
        assert name == "rbergomi"
        return _MockVol()

    monkeypatch.setattr("volsurface.fit.rbergomi.create", factory)
    result = calibrate(_skew_book(0.3), H=0.167, seed=7)
    assert result.detail["label"] == "_MockVol"
    assert result.detail["rbergomi"] is False
    assert result.detail["parameters"]["nu"] == pytest.approx(0.3, abs=1e-3)
    assert result.detail["parameters"]["H"] == pytest.approx(0.167)
    again = calibrate(_skew_book(0.3), H=0.167, seed=7)
    assert again.detail["parameters"] == result.detail["parameters"]
