# Path-bootstrap intervals for the gate filter skills

The table is the 4,000-path gate draw at clock index 10. Spot skill is
0.1178, windowed skill is 0.2071, and the latent-variance benchmark is
0.2849. The intervals are path bootstrap percentiles. Windowed minus
spot, and latent minus windowed, both sit above 0.05.

The draw is `H = 0.1`, `nu` 1.5 and 3, 21 times on `[0, 1]`, `n_sub = 21`,
seed `run_seed(12, 2)`. The clock is index 10, time 0.5. The base rate
is 0.5383, from 1,847 paths in the low regime and 2,153 in the high
regime. Brier skill is `1 - Brier / Brier_base`, against the constant
forecast that quotes the sample shares. The hit rate is the share of
paths whose modal regime is the label. The bootstrap draws paths with
replacement, `B = 10,000`, seed 12. Each resample recomputes the skill,
including that resample's base rate, and the hit rate. One index draw
is shared by the three filters, so the differences are paired. The
interval is the 2.5 and 97.5 percentiles. Figures are rounded to 4
decimals, and a half rounds away from zero. The script is
`scripts/brier_skill_bootstrap.py`.

| score | estimate | 95 percent interval |
| --- | --- | --- |
| spot skill | 0.1178 | [0.0915, 0.1439] |
| spot hit rate | 0.6608 | [0.6463, 0.6755] |
| windowed skill | 0.2071 | [0.1794, 0.2339] |
| windowed hit rate | 0.7048 | [0.6905, 0.7188] |
| latent skill | 0.2849 | [0.2556, 0.3137] |
| latent hit rate | 0.7438 | [0.7300, 0.7573] |
| windowed minus spot | 0.0893 | [0.0604, 0.1190] |
| latent minus windowed | 0.0778 | [0.0616, 0.0943] |

Hit counts are 2,643, 2,819 and 2,975 out of 4,000.

A path has switched by the clock when its regime on the grid up to that
date is not constant. 2,177 paths have stayed and 1,823 have switched.
The stayed skills are positive and ordered latent, windowed, spot. The
switched skills are negative for all three filters. On the switched
paths the windowed filter sits below the spot filter.

| score | stayed | 95 percent interval | switched | 95 percent interval |
| --- | --- | --- | --- | --- |
| spot skill | 0.2796 | [0.2408, 0.3164] | -0.3675 | [-0.4324, -0.3081] |
| spot hit rate | 0.7726 | [0.7547, 0.7901] | 0.5272 | [0.5041, 0.5502] |
| windowed skill | 0.5140 | [0.4857, 0.5403] | -0.4552 | [-0.5298, -0.3877] |
| windowed hit rate | 0.8732 | [0.8590, 0.8875] | 0.5036 | [0.4805, 0.5266] |
| latent skill | 0.5956 | [0.5684, 0.6207] | -0.3599 | [-0.4355, -0.2914] |
| latent hit rate | 0.8911 | [0.8778, 0.9040] | 0.5677 | [0.5447, 0.5902] |
| windowed minus spot | 0.2344 | [0.1978, 0.2721] | -0.0877 | [-0.1470, -0.0292] |
| latent minus windowed | 0.0816 | [0.0644, 0.0994] | 0.0953 | [0.0583, 0.1333] |

Stayed hit counts are 1,682, 1,901 and 1,940 out of 2,177. Switched hit
counts are 961, 918 and 1,035 out of 1,823.

This grid has 21 times. The daily episode at `t0` is a different path
and a different observation. On that episode the windowed skill is
0.9214 on the 2,182 paths that have stayed and 0.4921 on the 1,818 that
have switched.
