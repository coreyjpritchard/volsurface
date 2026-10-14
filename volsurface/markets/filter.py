"""Hamilton filter for a Markov-switching Gaussian. DESIGN section 22.

The observation is ``y_t | r_t = j ~ Normal(mean_j, variance_j)``. The
filter probability at date ``t`` uses ``y`` up to ``t`` and not after.
``transition[i, j]`` is ``P(r_t = j | r_{t-1} = i)`` and each row sums to 1.
Means and variances may be one vector, shared by every date, or one row
per date.
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
    step, which is the chain on an uneven time grid. ``means`` and
    ``variances`` have shape ``(R,)``, or ``(n, R)`` when the emission
    changes with the date. The return has shape ``(n, R)`` or ``(m, n, R)``
    and each row sums to 1.
    """
    series = np.asarray(y, dtype=float)
    single = series.ndim == 1
    if single:
        series = series[None, :]
    if series.ndim != 2:
        raise ValueError(f"y must have shape (n,) or (m, n), got {y.shape}")
    n_times = int(series.shape[1])
    mean, var = _emission_table(means, variances, n_times)
    if np.any(var <= 0.0):
        raise ValueError("variances must be positive")
    prior = np.asarray(p0, dtype=float).reshape(-1)
    n_regimes = int(mean.shape[1])
    if prior.shape != (n_regimes,):
        raise ValueError("p0 must have one entry per regime")
    steps = _transitions(transition, n_regimes, n_times)
    log_var = np.log(2.0 * np.pi * var)
    pred = np.broadcast_to(prior, (series.shape[0], n_regimes)).copy()
    out = np.empty((series.shape[0], n_times, n_regimes))
    for t in range(n_times):
        gap = series[:, t, None] - mean[t]
        log_like = -0.5 * log_var[t] - 0.5 * gap * gap / var[t]
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


def regime_scores(prob: np.ndarray, regime: np.ndarray) -> dict:
    """Modal accuracy, confusion counts, Brier score, and log score.

    ``prob`` has shape ``(m, R)`` and each row is a probability. ``regime``
    has shape ``(m,)`` and holds the true label. The confusion matrix counts
    true regime ``i`` predicted as modal regime ``j``. The Brier score is the
    mean of ``||p - e_r||^2``, so a perfect row scores 0 and two classes
    score at most 2. The log score is the mean of ``log(p_r)``, with the
    probability floored at ``1e-300``. The base rate is the larger share of
    the true labels. ``base_brier`` and ``base_log_score`` are the same two
    scores for the constant forecast that quotes the sample shares.
    ``brier_skill`` is ``1 - brier / base_brier``. A positive skill beats
    that forecast. The bar is this skill, scored separately at each setting.
    """
    p = np.asarray(prob, dtype=float)
    labels = np.asarray(regime, dtype=int).reshape(-1)
    if p.ndim != 2 or p.shape[0] != labels.size:
        raise ValueError(f"prob shape {p.shape} does not match {labels.size} labels")
    n_regimes = int(p.shape[1])
    if np.any(labels < 0) or np.any(labels >= n_regimes):
        raise ValueError("a regime label is outside the probability columns")
    if np.any(p < -1e-12) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("prob rows must be probabilities")
    predicted = np.argmax(p, axis=1)
    confusion = np.zeros((n_regimes, n_regimes), dtype=int)
    for truth in range(n_regimes):
        for guess in range(n_regimes):
            confusion[truth, guess] = int(np.sum((labels == truth) & (predicted == guess)))
    one_hot = np.eye(n_regimes, dtype=float)[labels]
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    chosen = p[np.arange(labels.size), labels]
    log_score = float(np.mean(np.log(np.maximum(chosen, 1e-300))))
    counts = np.bincount(labels, minlength=n_regimes)
    share = counts.astype(float) / labels.size
    # Constant forecast that quotes the sample shares. For two classes the
    # Brier score of that forecast is ``2 p (1 - p)``.
    base_brier = float(np.sum(share * np.sum((share - np.eye(n_regimes)) ** 2, axis=1)))
    base_log = float(np.sum(share * np.log(np.maximum(share, 1e-300))))
    if base_brier > 0.0:
        brier_skill = float(1.0 - brier / base_brier)
    else:
        brier_skill = 0.0
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "base_rate": float(share.max()),
        "confusion": confusion,
        "brier": brier,
        "log_score": log_score,
        "base_brier": base_brier,
        "base_log_score": base_log,
        "brier_skill": brier_skill,
    }


def _emission_table(means, variances, n_times: int) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast ``(R,)`` emissions to ``(n_times, R)``. A table is used as given."""
    mean = np.asarray(means, dtype=float)
    var = np.asarray(variances, dtype=float)
    if mean.ndim == 1:
        mean = np.broadcast_to(mean, (n_times, mean.size)).copy()
    if var.ndim == 1:
        var = np.broadcast_to(var, (n_times, var.size)).copy()
    if mean.ndim != 2 or var.shape != mean.shape or mean.shape[0] != n_times:
        raise ValueError(
            f"means and variances must have shape (R,) or {(n_times, 'R')}, "
            f"got {np.shape(means)} and {np.shape(variances)}"
        )
    return mean, var


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
