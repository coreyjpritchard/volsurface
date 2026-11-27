"""Historical session-00 Heston comparison (withdrawn from the public tree).

Earlier versions of this script compared two Heston fits on a market option
chain and printed the fitted parameters. Those snapshot-specific numbers are
no longer in this repository.

To fit Heston on a chain you supply, run:

    python docs/figure_00.py --snapshot path/to/chain.parquet

Outputs go to data/cache/figures/ and are not committed. See
docs/notes/heston_fit_old_vs_new.md.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "This comparison was withdrawn because it encoded snapshot-specific "
        "fitted parameters. Fit a chain you supply with: python docs/figure_00.py "
        "--snapshot path/to/chain.parquet"
    )


if __name__ == "__main__":
    main()
