# One entry point: a locked environment, the fast tests, and the README figure (synthetic demo).
.PHONY: reproduce test lint calibration synthetic-example

reproduce:
	uv sync --locked --extra dev --extra autodiff
	uv run --locked pytest -m "not slow"
	uv run --locked python docs/figure_00.py

test:
	uv run --locked pytest -m "not slow"

lint:
	uv run --locked ruff check .

# Rebuild the local training file and local site pack. GRID=... or
# SNAPSHOT=... must point at a local file; see data/exports/README.md.
# Outputs go to data/exports/local/ and are not committed.
calibration:
	@if [ -n "$(GRID)" ]; then \
		uv run --locked python scripts/sanos_calibration.py --grid "$(GRID)"; \
	elif [ -n "$(SNAPSHOT)" ]; then \
		uv run --locked python scripts/sanos_calibration.py --snapshot "$(SNAPSHOT)"; \
	else \
		echo "make calibration needs GRID=... or SNAPSHOT=... (a local path)"; \
		exit 2; \
	fi

# Rebuild the committed schema example from a labelled synthetic surface.
synthetic-example:
	uv run --locked python scripts/sanos_calibration.py --synthetic-example
