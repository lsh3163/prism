# Stronger-Backbone Results

These are the aligned results used for the current paper update.

## BFM-Zero Locomotion

All methods are evaluated at `9.6M` environment steps with seed `1000`.
Tracking EMD is lower-is-better.

| Method | Parameters | Nominal | Low friction | Payload mass |
|---|---:|---:|---:|---:|
| BFM-Zero | 32.109M | 1.104 | 1.582 | 1.121 |
| **PRISM** | 32.665M | **1.050** | **1.548** | **1.073** |
| Larger BFM-Zero | 49.260M | 1.090 | 1.589 | 1.114 |

PRISM adds approximately `0.556M` inference parameters over BFM-Zero and uses
approximately `16.6M` fewer parameters than the larger-capacity control.

## SmolVLA LIBERO

All methods use seed `1000` and the `80K` checkpoint. Results follow the
official multi-task `eval50` protocol: 500 episodes per suite and 2,000
episodes in total. Success is higher-is-better.

| Method | Parameters | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLA | 450.046M | 65.8 | 53.4 | 82.6 | 52.2 | 63.50 |
| **PRISM** | 451.925M | **70.0** | **57.4** | **85.4** | **53.4** | **66.55** |
| Larger SmolVLA | 456.245M | 66.6 | 55.0 | 84.0 | 54.0 | 64.90 |

The larger models are capacity controls rather than matched-size baselines.
PRISM modifies only the proprioceptive conditioner in these comparisons.
