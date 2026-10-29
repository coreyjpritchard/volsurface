"""Test data that ships with the repository. Nothing here is market data."""

import pytest

from volsurface import chains

# The Cboe snapshot is not committed (data/snapshots/README.md). Tests that check numbers
# of that one file run where a local copy exists and are skipped elsewhere, such as CI.
needs_real_snapshot = pytest.mark.skipif(
    not chains.has_snapshot("SPX"),
    reason="no local Cboe SPX snapshot in data/snapshots (it is not committed)",
)
