"""Shared fixtures.

``impl`` loads the implementation under test. By default it is the reference package
``volsurface``. ``pytest --impl exercises`` loads Corey's own modules from ``exercises/``
instead (``exercises/bs.py``, ``exercises/heston.py``, ...); a test whose module is not
there yet is skipped.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def pytest_addoption(parser):
    parser.addoption(
        "--impl", default="volsurface",
        help="'volsurface' (reference) or a directory such as 'exercises' holding bs.py etc.",
    )


@pytest.fixture(scope="session")
def impl(request):
    """``impl("bs")`` returns the ``bs`` module of the selected implementation."""
    choice = request.config.getoption("--impl")

    def load(name: str):
        if choice == "volsurface":
            return importlib.import_module(f"volsurface.{name}")
        path = (ROOT / choice / f"{name}.py").resolve()
        if not path.exists():
            pytest.skip(f"{path} not written yet")
        spec = importlib.util.spec_from_file_location(f"{Path(choice).name}_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
