# Camera-ready results

Reported in the camera-ready manuscript supplied on 2026-10-13. These are recorded
measurements, not results rerun during code preparation. The
[project page](https://lsh3163.github.io/prism/) provides figures and videos.

| Evaluation | Baseline (%) | PRISM (%) | Protocol |
|---|---:|---:|---|
| Humanoid-Gym walking | 51.0 | 92.5 | 24 s survival; five training seeds |
| ACT manipulation | 63.5 | 67.6 | Mean success on three ManiSkill3 tasks; final 100K checkpoints |
| SmolVLA / LIBERO | 62.00 | 63.82 | 80K checkpoints; mean over three training seeds |
| Flat unseen dynamics | 78.3 | 92.3 | 60 s survival; parameter-matched MLP; six settings |
| Rough unseen dynamics | 80.0 | 87.1 | 20 s supported survival; standard MLP; six settings |
| StackCube drop recovery | 40.9 | 52.3 | 18/44 vs. 23/44 shared validated states; 5 s horizon |
| SO-101 tape pickup | 68.0 | 94.0 | 34/50 vs. 47/50 attempts |

## ACT task success

| Method | StackCube | TwoRobotStackCube | G1TransportBox | Mean |
|---|---:|---:|---:|---:|
| ACT | 58.8 ± 6.3 | 47.0 ± 3.2 | 84.8 ± 4.9 | 63.5 |
| PRISM | 68.8 ± 5.8 | 48.2 ± 2.8 | 85.8 ± 5.1 | 67.6 |

Each task uses 500 attempts per method. Task entries are mean ± sample SD across
five groups of 100 attempts for one trained checkpoint.

## Drop recovery

| Method | Recovered | Capped time (s) | Placement error (mm) |
|---|---:|---:|---:|
| ACT | 18/44 (40.9%) | 3.38 | 141.7 |
| ACT + MCC-Sensorless | 6/44 (13.6%) | 4.50 | 123.5 |
| ACT + MCC-Oracle | 15/44 (34.1%) | 3.71 | 105.4 |
| PRISM | 23/44 (52.3%) | 3.11 | 96.2 |

Diagnostics are means over all 44 attempts, including failures; capped time assigns
5 s to unrecovered attempts. MCC variants are simulation adaptations.

## BFM-Zero tracking

| Method | Nominal EMD | Low-friction EMD | Increased-mass EMD |
|---|---:|---:|---:|
| Baseline | 1.104 | 1.582 | 1.121 |
| Larger model | 1.090 | 1.589 | 1.114 |
| PRISM | 1.050 | 1.548 | 1.073 |

Lower EMD is better; all checkpoints are aligned at 9.6M environment steps and
use one training seed. Increased mass is +15%.

[Earlier result snapshot](docs/legacy/results-20260916.md) is preserved for comparison.
[Source record](docs/camera-ready-sources.json) identifies the manuscript used here.
