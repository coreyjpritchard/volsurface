"""Build SANOS-calibrated Black-Scholes and Heston market files.

Usage:

    uv run python scripts/sanos_calibration.py --grid path/to/grid.json
    uv run python scripts/sanos_calibration.py --snapshot path/to/local.parquet
    uv run python scripts/sanos_calibration.py --synthetic-example

The grid is a normalised implied-vol file (T, k, iv). The snapshot path is a
local chain, used only to rebuild that grid; quotes are never written. The
training document and any site pack built from a user-supplied grid go to
data/exports/local/ (gitignored). The committed example pack is built from a
labelled synthetic Heston surface, not from a historical snapshot.
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
    synthetic_example_document,
    write_grid,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "exports" / "local" / "sanos_calibration.json"
DEFAULT_SITE = ROOT / "data" / "exports" / "local" / "snapshot_markets.json"
DEFAULT_GRID_OUT = ROOT / "data" / "exports" / "local" / "sanos_grid.json"
DEFAULT_EXAMPLE = ROOT / "data" / "exports" / "examples" / "synthetic_snapshot_markets.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, help="normalised implied-vol grid JSON")
    parser.add_argument("--snapshot", type=Path, help="local snapshot parquet; not written")
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--site-out",
        type=Path,
        default=DEFAULT_SITE,
        help="local site pack (gitignored); do not commit snapshot-derived packs",
    )
    parser.add_argument(
        "--write-grid",
        type=Path,
        default=None,
        help="optional path to write the normalised grid (no quotes)",
    )
    parser.add_argument(
        "--synthetic-example",
        action="store_true",
        help="write the committed schema example from a labelled synthetic surface",
    )
    parser.add_argument("--example-out", type=Path, default=DEFAULT_EXAMPLE)
    args = parser.parse_args()
    if args.synthetic_example:
        if args.grid is not None or args.snapshot is not None:
            raise SystemExit("pass --synthetic-example on its own")
        example = synthetic_example_document()
        write_json(args.example_out, example)
        print(f"wrote synthetic example {args.example_out}")
        return
    if args.grid is None and args.snapshot is None:
        raise SystemExit("pass --grid, --snapshot or --synthetic-example")
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
    full = train["heston_fits"]["full"]
    contract = train["heston_fits"]["contract"]
    print(
        f"black-scholes sigma {bs['sigma']:.6f} premium {bs['premium']:.6f}\n"
        f"heston full     premium {full['premium']:.6f} rmse_iv {full['rmse_iv']:.6f} "
        f"atm {full['atm_vol']:.6f} kappa {full['kappa']:.4f} "
        f"bound_hits={full['bound_hits']} n={full['n_points']}\n"
        f"heston contract premium {contract['premium']:.6f} rmse_iv {contract['rmse_iv']:.6f} "
        f"rmse_iv_contract {contract['rmse_iv_contract']:.6f} "
        f"atm {contract['atm_vol']:.6f} kappa {contract['kappa']:.4f} "
        f"bound_hits={contract['bound_hits']} n_eff={contract['n_effective']:.1f}\n"
        f"wrote {args.train_out}\n"
        f"wrote {args.site_out}"
    )


if __name__ == "__main__":
    main()
