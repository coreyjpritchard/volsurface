"""The phone page (dashboard/vol_lab_mobile.py) offers the same choices as the desktop one."""

from __future__ import annotations

import ast
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"


def _assigned(path: Path, name: str):
    """Evaluate the literal assigned to ``name`` anywhere in a marimo notebook."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return eval(compile(ast.Expression(node.value), str(path), "eval"), {"dict": dict})
    raise AssertionError(f"{name} not found in {path.name}")


def test_presets_match_desktop():
    desktop = _assigned(DASHBOARD / "vol_lab.py", "PRESETS")
    mobile = _assigned(DASHBOARD / "vol_lab_mobile.py", "PRESETS")
    assert mobile == desktop


def test_environments_match_desktop():
    desktop = _assigned(DASHBOARD / "vol_lab.py", "ENVIRONMENTS")
    mobile = _assigned(DASHBOARD / "vol_lab_mobile.py", "ENVIRONMENTS")
    assert sorted(mobile.values()) == sorted(desktop.values())


def test_both_pages_install_the_same_wheels():
    for wheel in ("plotly-7.1.0-py3-none-any.whl", "volsurface-0.1.0-py3-none-any.whl"):
        for page in ("vol_lab.py", "vol_lab_mobile.py"):
            assert wheel in (DASHBOARD / page).read_text()
