"""Discrete local volatility for a SANOS marginal. DESIGN section 19.

The transition satisfies ``Q^{-1} = I + Omega Sigma^2 dT``, applied column
by column. End columns are unit columns. ``Q`` is column-stochastic and a
martingale: ``1' Q = 1'`` and ``K' Q = K'``. The same construction is in
NumPy and in torch float64.

The implied ``Sigma`` is the discrete Dupire ratio on one strike grid.
``0 / 0`` is read as 0. Strikes that differ by expiry are sent across by a
mean-preserving split onto the two bracketing strikes, and the homogeneous
operator then runs on the target grid.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional extra
    torch = None


def _grid(strikes: np.ndarray) -> np.ndarray:
    grid = np.asarray(strikes, dtype=float).reshape(-1)
    if grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("strikes must be strictly increasing")
    if np.any(grid <= 0.0):
        raise ValueError("strikes must be positive")
    return grid


def inverse_transition(strikes: np.ndarray, sigma: np.ndarray, dT: float) -> np.ndarray:
    """``Q^{-1}`` for one step. ``sigma`` is nonnegative and matches ``strikes``."""
    grid = _grid(strikes)
    width = np.asarray(sigma, dtype=float).reshape(-1)
    if width.shape != grid.shape:
        raise ValueError(f"sigma shape {width.shape} does not match strikes {grid.shape}")
    if dT <= 0.0:
        raise ValueError(f"dT must be positive, got {dT}")
    if np.any(width < 0.0) or not np.isfinite(width).all():
        raise ValueError("sigma must be nonnegative and finite")
    n = grid.size
    matrix = np.eye(n)
    step = width**2 * float(dT)
    for i in range(1, n - 1):
        span = grid[i + 1] - grid[i - 1]
        left = grid[i] - grid[i - 1]
        right = grid[i + 1] - grid[i]
        scale = step[i] * grid[i] ** 2 / span
        w_minus = scale / left
        w_plus = scale / right
        matrix[i - 1, i] = -w_minus
        matrix[i, i] = 1.0 + w_minus + w_plus
        matrix[i + 1, i] = -w_plus
    return matrix


def transition_matrix(strikes: np.ndarray, sigma: np.ndarray, dT: float) -> np.ndarray:
    """Column-stochastic martingale transition. Endpoint ``sigma`` is unused."""
    return np.linalg.inv(inverse_transition(strikes, sigma, dT))


def transition_matrix_torch(strikes, sigma, dT: float):
    """The same transition in torch float64."""
    if torch is None:
        raise ImportError("torch is not installed")
    grid = torch.as_tensor(strikes, dtype=torch.float64)
    width = torch.as_tensor(sigma, dtype=torch.float64)
    eye = torch.eye(grid.shape[0], dtype=torch.float64)
    step = width**2 * float(dT)
    for i in range(1, grid.shape[0] - 1):
        span = grid[i + 1] - grid[i - 1]
        left = grid[i] - grid[i - 1]
        right = grid[i + 1] - grid[i]
        scale = step[i] * grid[i] ** 2 / span
        w_minus = scale / left
        w_plus = scale / right
        eye[i - 1, i] = -w_minus
        eye[i, i] = 1.0 + w_minus + w_plus
        eye[i + 1, i] = -w_plus
    return torch.linalg.inv(eye)


def segment_slopes(strikes: np.ndarray, calls: np.ndarray) -> np.ndarray:
    """Call slopes, with ``-1`` on the left of the first strike and ``0`` on the right of the last.

    Entry ``i`` is the slope on the left of strike ``i``. The last entry is
    the slope on the right of the last strike. Length ``n + 1``.
    """
    grid = np.asarray(strikes, dtype=float)
    price = np.asarray(calls, dtype=float)
    interior = np.diff(price) / np.diff(grid)
    return np.concatenate([[-1.0], interior, [0.0]])


def masses_from_calls(strikes: np.ndarray, calls: np.ndarray) -> np.ndarray:
    """Atom mass at each strike: right slope minus left slope."""
    slopes = segment_slopes(strikes, calls)
    return slopes[1:] - slopes[:-1]


def calls_from_masses(
    strikes: np.ndarray,
    masses: np.ndarray,
    query: np.ndarray | None = None,
) -> np.ndarray:
    """Undiscounted call of a discrete marginal. ``query`` defaults to ``strikes``."""
    grid = np.asarray(strikes, dtype=float)
    weight = np.asarray(masses, dtype=float)
    points = grid if query is None else np.asarray(query, dtype=float)
    return weight @ np.maximum(grid[:, None] - points[None, :], 0.0)


def implied_sigma(
    strikes: np.ndarray,
    call_prev: np.ndarray,
    call_now: np.ndarray,
    dT: float,
) -> np.ndarray:
    """Discrete local volatility. ``0 / 0`` is 0. The endpoints are 0."""
    grid = _grid(strikes)
    if dT <= 0.0:
        raise ValueError(f"dT must be positive, got {dT}")
    theta = (np.asarray(call_now, dtype=float) - np.asarray(call_prev, dtype=float)) / float(dT)
    slopes = segment_slopes(grid, call_now)
    gamma = np.zeros(grid.size)
    span = grid[2:] - grid[:-2]
    # dC^i is the slope to the right of strike i. dC^{i-1} is the slope to its left.
    gamma[1:-1] = (slopes[2:-1] - slopes[1:-2]) / (0.5 * span)
    sigma = np.zeros(grid.size)
    numer = 2.0 * theta
    denom = grid**2 * gamma
    quiet = (np.abs(numer) <= 1e-14) & (np.abs(denom) <= 1e-14)
    usable = (denom > 1e-14) & np.isfinite(numer) & ~quiet
    sigma[usable] = np.sqrt(np.maximum(numer[usable] / denom[usable], 0.0))
    return sigma


def split_onto(
    source_strikes: np.ndarray,
    masses: np.ndarray,
    target_strikes: np.ndarray,
) -> np.ndarray:
    """Put each source atom on the two target strikes that bracket it, keeping the mean."""
    source = _grid(source_strikes)
    target = _grid(target_strikes)
    weight = np.asarray(masses, dtype=float)
    out = np.zeros(target.size)
    for strike, mass in zip(source, weight, strict=True):
        if mass == 0.0:
            continue
        if strike <= target[0] or strike >= target[-1]:
            # Outside the target there is no bracketing pair. The endpoint
            # takes the mass. Callers that need the mean kept should stay inside.
            index = 0 if strike <= target[0] else target.size - 1
            out[index] += mass
            continue
        j = int(np.searchsorted(target, strike, side="right") - 1)
        left = float(target[j])
        right = float(target[j + 1])
        width = right - left
        out[j] += mass * (right - strike) / width
        out[j + 1] += mass * (strike - left) / width
    return out


def advance(
    source_strikes: np.ndarray,
    masses: np.ndarray,
    target_strikes: np.ndarray,
    sigma: np.ndarray,
    dT: float,
) -> np.ndarray:
    """Split onto ``target_strikes``, then apply the homogeneous transition there."""
    landed = split_onto(source_strikes, masses, target_strikes)
    return transition_matrix(target_strikes, sigma, dT) @ landed
