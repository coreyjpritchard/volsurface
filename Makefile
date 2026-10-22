# One entry point: a locked environment, the fast tests, and the README figure.
.PHONY: reproduce test lint

reproduce:
	uv sync --locked --extra dev --extra autodiff
	uv run --locked pytest -m "not slow"
	uv run --locked python docs/figure_00.py

test:
	uv run --locked pytest -m "not slow"

lint:
	uv run --locked ruff check .
