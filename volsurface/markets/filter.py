"""Hamilton filter for a Markov-switching Gaussian. DESIGN section 22.

The observation is ``y_t | r_t = j ~ Normal(mean_j, variance_j)``. The
filter probability at date ``t`` uses ``y`` up to ``t`` and not after.
``transition[i, j]`` is ``P(r_t = j | r_{t-1} = i)`` and each row sums to 1.
"""

from __future__ import annotations

import numpy as np


def hamilton_filter(
    y: np.ndarray,
    transition: np.ndarray | list[np.ndarray],
    means: np.ndarray,
    variances: np.ndarray,
    p0: np.ndarray,
) -> np.ndarray:
    """Filtered probabilities.

    ``y`` has shape ``(n,)`` or ``(m, n)``. One transition matrix is used at
    every step. A list of length ``n - 1`` uses a different matrix on each
    step, which is the chain on an uneven time grid. The return has shape
    ``(n, R)`` or ``(m, n, R)`` and each row sums to 1.
    """
    series = np.asarray(y, dtype=float)
    single = series.ndim == 1
    if single:
        series = series[None, :]
    if series.ndim != 2:
        raise ValueError(f"y must have shape (n,) or (m, n), got {y.shape}")
    mean = np.asarray(means, dtype=float).reshape(-1)
    var = np.asarray(variances, dtype=float).reshape(-1)
    if mean.shape != var.shape:
        raise ValueError("means and variances must have the same length")
    if np.any(var <= 0.0):
        raise ValueError("variances must be positive")
    prior = np.asarray(p0, dtype=float).reshape(-1)
    if prior.shape != mean.shape:
        raise ValueError("p0 must have one entry per regime")
    n_regimes = int(mean.size)
    n_times = int(series.shape[1])
    steps = _transitions(transition, n_regimes, n_times)
    log_var = np.log(2.0 * np.pi * var)
    pred = np.broadcast_to(prior, (series.shape[0], n_regimes)).copy()
    out = np.empty((series.shape[0], n_times, n_regimes))
    for t in range(n_times):
        gap = series[:, t, None] - mean[None, :]
        log_like = -0.5 * log_var[None, :] - 0.5 * gap * gap / var[None, :]
        log_joint = np.log(np.maximum(pred, 1e-300)) + log_like
        log_joint -= log_joint.max(axis=1, keepdims=True)
        joint = np.exp(log_joint)
        post = joint / joint.sum(axis=1, keepdims=True)
        out[:, t, :] = post
        if t + 1 < n_times:
            pred = post @ steps[t]
    if single:
        return out[0]
    return out


def _transitions(transition, n_regimes: int, n_times: int) -> list[np.ndarray]:
    if n_times < 1:
        raise ValueError("y must have at least one time")
    if isinstance(transition, list):
        if len(transition) != max(n_times - 1, 0):
            raise ValueError(f"expected {n_times - 1} transition matrices, got {len(transition)}")
        return [_as_transition(step, n_regimes) for step in transition]
    matrix = _as_transition(transition, n_regimes)
    return [matrix for _ in range(max(n_times - 1, 0))]


def _as_transition(transition, n_regimes: int) -> np.ndarray:
    matrix = np.asarray(transition, dtype=float)
    if matrix.shape != (n_regimes, n_regimes):
        raise ValueError(f"transition must have shape {(n_regimes, n_regimes)}, got {matrix.shape}")
    if np.any(matrix < -1e-12):
        raise ValueError("a transition probability is negative")
    row_sum = matrix.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-8):
        raise ValueError("each transition row must sum to 1")
    return matrix / row_sum[:, None]
