"""JSON presets for the site widget.

The schema name is ``volsurface.surface/v2``. One file is one model on one
``(T, k)`` grid: the normalised call and the Black implied volatility. A
missing implied vol is JSON ``null``. The file is the grid that was priced,
not a promise about points off the grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "volsurface.surface/v2"


def write_surface_preset(
    path: str | Path,
    *,
    model: str,
    params: dict,
    T,
    k,
    call,
    iv,
    forward: float,
    discount: float,
    note: str,
) -> None:
    """Write one preset. ``call`` and ``iv`` have shape ``(len(T), len(k))``."""
    call = np.asarray(call, dtype=float)
    iv = np.asarray(iv, dtype=float)
    T = np.asarray(T, dtype=float).reshape(-1)
    k = np.asarray(k, dtype=float).reshape(-1)
    if call.shape != (T.size, k.size) or iv.shape != call.shape:
        raise ValueError(
            f"call/iv shape {call.shape}, {iv.shape} does not match T {T.size} and k {k.size}"
        )
    payload = {
        "schema": SCHEMA,
        "model": model,
        "params": {key: float(value) for key, value in params.items()},
        "forward": float(forward),
        "discount": float(discount),
        "T": [float(value) for value in T],
        "k": [float(value) for value in k],
        "call": call.tolist(),
        "iv": [[None if not np.isfinite(value) else float(value) for value in row] for row in iv],
        "note": note,
    }
    text = json.dumps(payload, indent=2)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8")
    if destination.stat().st_size > 200_000:
        raise ValueError(f"{destination} is over 200 KB")
