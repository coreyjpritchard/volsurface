# One entry point: a locked environment, the fast tests, and the README figure.
.PHONY: reproduce test lint dashboard public-vollab

reproduce:
	uv sync --locked --extra dev --extra autodiff
	uv run --locked pytest -m "not slow"
	uv run --locked python docs/figure_00.py

test:
	uv run --locked pytest -m "not slow"

lint:
	uv run --locked ruff check .

# The volatility environment lab (marimo app) at http://localhost:2718.
dashboard:
	uv run --locked --extra dev marimo run dashboard/vol_lab.py

# The static web version of the lab (runs in the browser with Pyodide; no server).
public-vollab:
	uv run --locked --extra dev python scripts/build_public_vollab.py build/public-vollab
