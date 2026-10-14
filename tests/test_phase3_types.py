"""Import check for the Phase 3 stubs in DESIGN sections 17 to 22."""

import numpy as np

from volsurface.core.protocols import Pricer, SupportsMarketSimulate, capabilities
from volsurface.core.types import MarketPaths, RegimeSpec, SurfaceBatch


def test_phase3_stubs_construct():
    paths = MarketPaths(
        times=np.array([0.0, 1.0]),
        Z=np.ones((2, 2)),
        state={},
        F0=lambda t: np.ones_like(np.asarray(t, dtype=float)),
        df=lambda t: np.ones_like(np.asarray(t, dtype=float)),
        meta={"draw_order": ("regime", "vol", "spot", "surrogate")},
    )
    assert paths.regime is None
    assert paths.regime_prob is None
    spec = RegimeSpec(
        generator=np.zeros((1, 1)),
        params=[{"nu": 0.5}],
        p0=np.array([1.0]),
        switch="nu",
    )
    assert spec.switch == "nu"
    batch = SurfaceBatch(
        T=np.array([1.0]),
        k=np.array([0.0]),
        call=np.ones((2, 1, 1)),
    )
    assert batch.call.shape == (2, 1, 1)
    assert "market_simulate" not in capabilities(object())
    assert Pricer.__name__ == "Pricer"
    assert SupportsMarketSimulate.__name__ == "SupportsMarketSimulate"
