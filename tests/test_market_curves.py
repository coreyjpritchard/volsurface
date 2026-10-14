"""Forward and discount curves outside the pillars. DESIGN.md section 10."""

import numpy as np
import pytest

from volsurface.core.market import _discount_curve, _forward_curve


def test_discount_anchors_at_one_and_extends_the_last_rate():
    # Pillars at 0.5 and 1.0 with a 5% then 10% forward rate, continuously compounded.
    times = np.array([0.5, 1.0])
    dfs = np.exp(np.array([-0.05 * 0.5, -0.05 * 0.5 - 0.10 * 0.5]))
    df = _discount_curve(times, dfs)
    assert df(0.0) == pytest.approx(1.0)
    # Halfway to the first pillar uses the anchor, so it is not stuck at df(0.5).
    assert df(0.25) == pytest.approx(np.exp(-0.05 * 0.25))
    assert df(0.25) != pytest.approx(df(0.5))
    # Past the last pillar the last gap's rate (10%) continues.
    assert df(1.5) == pytest.approx(df(1.0) * np.exp(-0.10 * 0.5))
    assert df(0.5) == pytest.approx(dfs[0])
    assert df(1.0) == pytest.approx(dfs[1])


def test_one_discount_pillar_holds_the_level_past_the_end():
    df = _discount_curve(np.array([0.5]), np.array([0.99]))
    assert df(0.0) == pytest.approx(1.0)
    assert df(0.25) == pytest.approx(np.exp(0.5 * np.log(0.99)))
    assert df(2.0) == pytest.approx(0.99)


def test_forward_extends_carry_on_both_sides_and_holds_a_single_pillar():
    times = np.array([0.5, 1.5])
    forwards = np.array([100.0, 102.0])
    carry = np.log(1.02) / 1.0
    F = _forward_curve(times, forwards)
    assert F(0.5) == pytest.approx(100.0)
    assert F(1.5) == pytest.approx(102.0)
    # Interior stays linear in the level.
    assert F(1.0) == pytest.approx(101.0)
    assert F(0.0) == pytest.approx(100.0 * np.exp(carry * (0.0 - 0.5)))
    assert F(2.5) == pytest.approx(102.0 * np.exp(carry * (2.5 - 1.5)))
    held = _forward_curve(np.array([0.5]), np.array([100.0]))
    assert held(0.0) == pytest.approx(100.0)
    assert held(2.0) == pytest.approx(100.0)
