"""Build the SANOS-calibrated Black-Scholes and Heston market files.

Usage:

    uv run python scripts/sanos_calibration.py --grid path/to/grid.json
    uv run python scripts/sanos_calibration.py --snapshot path/to/local.parquet

The grid is a normalised implied-vol file (T, k, iv). The snapshot path is a
local chain, used only to rebuild that grid; quotes are never written. The
training document goes to data/exports/local/ (gitignored). The site document
goes to data/exports/site/snapshot_markets.json and omits fitted parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from volsurface.fit.sanos_calibration import (
    calibrate_from_grid,
    git_commit,
    grid_from_snapshot,
    load_grid,
    site_document,
    write_grid,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "exports" / "local" / "sanos_calibration.json"
DEFAULT_SITE = ROOT / "data" / "exports" / "site" / "snapshot_markets.json"
DEFAULT_GRID_OUT = ROOT / "data" / "exports" / "local" / "sanos_grid.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, help="normalised implied-vol grid JSON")
    parser.add_argument("--snapshot", type=Path, help="local snapshot parquet; not written")
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--site-out", type=Path, default=DEFAULT_SITE)
    parser.add_argument(
        "--write-grid",
        type=Path,
        default=None,
        help="optional path to write the normalised grid (no quotes)",
    )
    args = parser.parse_args()
    if args.grid is None and args.snapshot is None:
        raise SystemExit("pass --grid or --snapshot")
    if args.grid is not None and args.snapshot is not None:
        raise SystemExit("pass only one of --grid or --snapshot")
    if args.snapshot is not None:
        T, k, iv = grid_from_snapshot(args.snapshot)
        grid_path = args.write_grid or DEFAULT_GRID_OUT
        write_grid(grid_path, T, k, iv)
        print(f"wrote grid {grid_path} shape {iv.shape}")
    else:
        T, k, iv = load_grid(args.grid)
        if args.write_grid is not None:
            write_grid(args.write_grid, T, k, iv)
    train = calibrate_from_grid(T, k, iv, commit=git_commit())
    site = site_document(train)
    write_json(args.train_out, train)
    write_json(args.site_out, site)
    bs = train["black_scholes"]
    heston = train["heston"]
    print(
        f"black-scholes sigma {bs['sigma']:.6f} premium {bs['premium']:.6f}\n"
        f"heston premium {heston['premium']:.6f} rmse_iv {heston['rmse_iv']:.6f} "
        f"n={heston['n_points']} {heston['solver_status']}\n"
        f"wrote {args.train_out}\n"
        f"wrote {args.site_out}"
    )


if __name__ == "__main__":
    main()
