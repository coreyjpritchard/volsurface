"""Bounds in volsurface.mc.zrule, with no Monte Carlo draw.

The formulas are DESIGN.md section 7. These tests lock the quantiles and the
failure sentences. Path tests that consume the helper live with the models.
"""

import numpy as np
import pytest
from scipy.stats import binom, norm

from volsurface.mc.rng import spawn
from volsurface.mc.zrule import (
    check_replication_means,
    check_z,
    fraction_band,
    max_abs_bound,
    mean_abs_bound,
)


def test_bounds_match_the_normal_and_binomial_quantiles():
    n = 20
    p = 2.0 * norm.cdf(2.0) - 1.0
    assert max_abs_bound(n) == pytest.approx(norm.ppf(1.0 - 0.001 / (2.0 * n)))
    lo, hi = fraction_band(n)
    assert lo == pytest.approx(binom.ppf(0.001, n, p) / n)
    assert hi == pytest.approx(binom.ppf(0.999, n, p) / n)
    # A sample of 20 in which every |z| is below 2 used to miss a fixed 0.99 cap.
    assert hi == pytest.approx(1.0)
    assert mean_abs_bound(n) == pytest.approx(3.5 / np.sqrt(n))


def test_check_z_accepts_a_quiet_sample_and_names_a_failure():
    quiet = np.zeros(20)
    assert check_z(quiet) is None
    loud = np.zeros(20)
    loud[0] = 10.0
    message = check_z(loud)
    assert message is not None
    assert "max |z|" in message
    # Shared paths skip the mean and the fraction. The max bound still applies.
    assert check_z(np.full(20, 0.7), mean=False, fraction=False) is None


def test_replication_means_use_the_observed_sd():
    calm = np.array([0.01, -0.02, 0.0, 0.015, -0.01])
    assert check_replication_means(calm) is None
    biased = np.array([1.0, 1.1, 0.9, 1.05])
    message = check_replication_means(biased)
    assert message is not None
    assert "R=4" in message
    assert "sd=" in message


def test_seed_sequence_is_copied_before_spawn():
    """Two spawns of one SeedSequence share child 0, and the caller stays put."""
    parent = np.random.SeedSequence(123)
    first = spawn(parent, 2)
    second = spawn(parent, 2)
    draw_a = np.random.default_rng(first[0]).normal(size=5)
    draw_b = np.random.default_rng(second[0]).normal(size=5)
    np.testing.assert_array_equal(draw_a, draw_b)
    assert parent.n_children_spawned == 0
