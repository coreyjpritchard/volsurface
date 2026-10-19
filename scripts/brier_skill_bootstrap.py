"""Path-bootstrap intervals for the gate filter Brier skills.

The draw is the one behind the reported skills 0.1178, 0.2071 and 0.2849.
Gate settings: ``H = 0.1``, ``nu`` 1.5 and 3, 4,000 paths, 21 times on
``[0, 1]``, clock index 10, ``n_sub = 21``, seed ``run_seed(12, 2)``.
The three scores are the spot filter, the windowed realised-variance
filter, and the latent-variance benchmark.

The bootstrap resamples paths with replacement, ``B = 10,000``, seed 12,
``np.random.default_rng``. Each resample recomputes the Brier skill,
including the base-rate forecast on that resample's labels, and the hit
rate. The interval is the 2.5 and 97.5 percentiles. Paired differences
use one index draw for every filter.

A path has switched by the clock when its regime on the grid up to that
date is not constant. The stayed and switched skills resample within
that group.

Run from the repository root:

    python scripts/brier_skill_bootstrap.py
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from volsurface.core.types import RegimeSpec
from volsurface.markets.filter import regime_scores
from volsurface.markets.regime import RegimeSwitchingBergomi
from volsurface.markets.seeds import run_seed

CLOCK = 10
N_PATHS = 4000
B = 10_000
BOOT_SEED = 12
CHUNK = 250

# Confusion counts pinned by test_four_thousand_path_filter_scores.
PINNED_CONFUSION = {
    "spot": [[1174, 673], [684, 1469]],
    "windowed": [[1336, 511], [670, 1483]],
    "latent": [[1394, 453], [572, 1581]],
}
FILTERS = (
    ("spot", "regime_prob_spot"),
    ("windowed", "regime_prob_rv"),
    ("latent", "regime_prob_latent"),
)


def _model() -> RegimeSwitchingBergomi:
    spec = RegimeSpec(
        generator=np.array([[-1.5, 1.5], [0.8, -0.8]]),
        params=[{"nu": 1.5}, {"nu": 3.0}],
        p0=np.array([0.7, 0.3]),
        switch="nu",
    )
    return RegimeSwitchingBergomi(
        spec,
        xi0=0.04,
        rho=-0.4,
        H=0.1,
        n_mc=2,
        n_sub=21,
        n_pilot=1024,
    )


def _path_terms(prob: np.ndarray, regime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-path Brier term, hit indicator, and label. Shapes ``(m,)``."""
    p = np.asarray(prob, dtype=float)
    labels = np.asarray(regime, dtype=int).reshape(-1)
    n_regimes = int(p.shape[1])
    one_hot = np.eye(n_regimes, dtype=float)[labels]
    brier = np.sum((p - one_hot) ** 2, axis=1)
    hit = (np.argmax(p, axis=1) == labels).astype(float)
    return brier, hit, labels


def _skill(brier: np.ndarray, labels: np.ndarray) -> float:
    """``1 - Brier / Brier_base`` on these paths. Two regimes."""
    share = np.bincount(labels, minlength=2).astype(float) / labels.size
    base = 2.0 * float(share[0] * share[1])
    if base <= 0.0:
        return 0.0
    return float(1.0 - np.mean(brier) / base)


def _resample(
    terms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    b: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Bootstrap skills and hit rates. One index draw is shared by every filter."""
    names = list(terms)
    labels = terms[names[0]][2]
    n = int(labels.size)
    rng = np.random.default_rng(seed)
    skills = {name: np.empty(b) for name in names}
    hits = {name: np.empty(b) for name in names}
    for start in range(0, b, CHUNK):
        m = min(CHUNK, b - start)
        idx = rng.integers(0, n, size=(m, n))
        share_high = np.mean(labels[idx] == 1, axis=1)
        base = 2.0 * share_high * (1.0 - share_high)
        ok = base > 0.0
        for name in names:
            brier, hit, _label = terms[name]
            mean_brier = np.mean(brier[idx], axis=1)
            skill = np.ones(m)
            skill[ok] = 1.0 - mean_brier[ok] / base[ok]
            skills[name][start : start + m] = skill
            hits[name][start : start + m] = np.mean(hit[idx], axis=1)
    return skills, hits


def _interval(sample: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(sample, [0.025, 0.975])
    return float(low), float(high)


def _fmt(value: float) -> str:
    """Four decimals. A half rounds away from zero."""
    text = f"{value:.6f}"
    return str(Decimal(text).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _fmt_interval(low: float, high: float) -> str:
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _group_mask(regime: np.ndarray, clock: int) -> np.ndarray:
    """True where the regime on ``0 .. clock`` is not constant."""
    if clock < 1:
        return np.zeros(regime.shape[0], dtype=bool)
    changed = regime[:, 1 : clock + 1] != regime[:, :clock]
    return np.any(changed, axis=1)


def _report_block(title: str, terms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    skills, hits = _resample(terms, B, BOOT_SEED)
    print(f"## {title}")
    print()
    print("| score | estimate | 95 percent interval |")
    print("| --- | --- | --- |")
    point = {}
    for name in terms:
        brier, hit, labels = terms[name]
        skill = _skill(brier, labels)
        hit_rate = float(np.mean(hit))
        point[name] = skill
        skill_lo, skill_hi = _interval(skills[name])
        hit_lo, hit_hi = _interval(hits[name])
        print(f"| {name} skill | {_fmt(skill)} | {_fmt_interval(skill_lo, skill_hi)} |")
        print(f"| {name} hit rate | {_fmt(hit_rate)} | {_fmt_interval(hit_lo, hit_hi)} |")
    pairs = (
        ("windowed minus spot", "windowed", "spot"),
        ("latent minus windowed", "latent", "windowed"),
    )
    for label, left, right in pairs:
        diff = point[left] - point[right]
        lo, hi = _interval(skills[left] - skills[right])
        print(f"| {label} | {_fmt(diff)} | {_fmt_interval(lo, hi)} |")
    print()


def main() -> None:
    model = _model()
    times = np.linspace(0.0, 1.0, 21)
    paths = model.simulate_market(N_PATHS, times, seed=run_seed(12, 2))
    labels = paths.regime[:, CLOCK]
    terms = {}
    for name, key in FILTERS:
        scored = regime_scores(paths.state[key][:, CLOCK], labels)
        if scored["confusion"].tolist() != PINNED_CONFUSION[name]:
            got = scored["confusion"].tolist()
            raise RuntimeError(f"{name} confusion {got} is not the pinned draw")
        brier, hit, lab = _path_terms(paths.state[key][:, CLOCK], labels)
        if abs(_skill(brier, lab) - scored["brier_skill"]) > 1e-12:
            raise RuntimeError(f"{name} skill does not match regime_scores")
        if abs(float(np.mean(hit)) - scored["accuracy"]) > 1e-12:
            raise RuntimeError(f"{name} hit rate does not match regime_scores")
        terms[name] = (brier, hit, lab)
    n = int(labels.size)
    base = float(np.bincount(labels.astype(int), minlength=2).max() / n)
    print(f"paths {n}, clock {CLOCK}, t {float(times[CLOCK]):.4f}, base rate {_fmt(base)}")
    print(f"bootstrap B {B}, seed {BOOT_SEED}")
    print()
    _report_block("all paths", terms)
    switched = _group_mask(paths.regime, CLOCK)
    for title, mask in (("stayed", ~switched), ("switched", switched)):
        count = int(mask.sum())
        print(f"{title} paths {count}")
        group = {name: tuple(part[mask] for part in terms[name]) for name in terms}
        _report_block(title, group)


if __name__ == "__main__":
    main()
