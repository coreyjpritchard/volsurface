"""ATM level and skew measurement on model smiles."""

import numpy as np
import pytest

from volsurface import models, smile

# A short-dated Heston set where the default inversion drops the outer call strike
# at the 10-hour expiry for some nearby parameters and not for others.
FRONT_T = 0.001139269406392694
PARAMS = dict(v0=0.010439, kappa=11.552295, theta=0.051621, xi=1.964307, rho=-0.547528)


def test_iv_tol_agrees_with_the_default_away_from_the_front():
    model = models.Heston(**PARAMS)
    maturities = np.array([0.02, 0.25, 1.0])
    default = smile.model_term_structure(model, maturities)
    loose = smile.model_term_structure(model, maturities, iv_tol=1e-8)
    assert loose == pytest.approx(default, abs=1e-6)


def test_front_skew_is_continuous_in_the_parameters_with_iv_tol():
    base = smile.model_term_structure(models.Heston(**PARAMS), [FRONT_T], iv_tol=1e-8)
    nudged = dict(PARAMS, rho=PARAMS["rho"] * (1 + 2e-4))
    moved = smile.model_term_structure(models.Heston(**nudged), [FRONT_T], iv_tol=1e-8)
    assert np.all(np.isfinite(base))
    assert abs(moved[0, 1] - base[0, 1]) < 1e-2
