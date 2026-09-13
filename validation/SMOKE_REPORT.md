# Simulator execution checks

The extracted G1 actor/environment and patched LeRobot Diffusion integration
ran with trained checkpoints on September 13, 2026. These checks establish
bounded execution and metrics export. No training, full benchmark reproduction,
or hardware experiment was performed.

## G1 / Isaac Gym

| Check | Outcome |
|---|---|
| Nominal degree-2 warmup actor; seed 1; one environment | Completed 2,400 control steps / 24 seconds and survived to timeout. |
| Episode diagnostics | Return 148.496765; planar velocity error 0.143841 m/s; absolute yaw-rate error 0.048508 rad/s. |
| Extraction versus original research environment | All observations, actions, rewards, termination flags, and root states matched bit for bit across a separate 200-step trace. |

The comparison did not cross an episode reset. The nominal evaluator disables
observation noise, dynamics randomization, action delay/noise, and pushes.
Training convergence, randomized physics, and reset-trajectory equivalence
remain untested.

Environment: Python 3.8, PyTorch 2.3.1/CUDA 12.1, NumPy 1.20.0, Isaac Gym
Preview 4, and an RTX 4090 Laptop GPU. Unitree RL Gym revision:
`276801e46c5d433564f24658bac64f254b7d2d4b`; RSL-RL revision:
`2ad79cf0caa85b91721abfe358105f869a784121`.
Checkpoint SHA-256:
`f05a364979651edf297c008ae9e4e00fba824aeaa56208f185a86f78c1914183`.

## Diffusion / LIBERO

Baseline and `diffusion_factorized_state_v1` policies loaded 20K checkpoints
and saved preprocessors on `libero_spatial:0`. Each ran one episode with seed
1000, batch 1, controller Kp 50/damping ratio 1, ten diffusion inference steps,
eight action steps, and a 280-step horizon. MCC and added perturbations were off.

| Full-horizon check | Baseline | PRISM |
|---|---:|---:|
| Control steps | 280 | 280 |
| Achieved task success during episode | No | Yes |
| Peak contact-force proxy | 44.478107 | 22.948658 |
| Incoming-action smoothness | 0.046734 | 0.049836 |

Both processes exited successfully. A single episode per policy does not
establish relative performance. Force is the maximum per-body external-force
norm, not an EEF wrench; success means achieved at any time during the episode.

The wrapper now honors `terminate_on_success=false` despite LIBERO signaling
success through backend `done`. Regression tests cover success continuation and
non-success termination. Actor weights were unchanged; historical success-stopped
and corrected full-horizon diagnostics have different semantics. A baseline
repeat was not bit-identical, so deterministic replay is not established.

Environment: LeRobot `d656da8ccca5989ff0a2207e81fbfa2c2d5bafb1` plus the released
patch, Python 3.12, PyTorch 2.10, MuJoCo/LIBERO, and NVIDIA EGL. Checkpoint SHA-256:

- Baseline: `cbc191620a2dae5bd71e362f37fd461fbb061456f7594fba824c1969769436d7`.
- PRISM: `8f13bd809dd6d322ebf5cecb03a5cfa993d9aceff40befce30ff46a8bb65ef7b`.

## Run the checks

See the [G1 guide](../integrations/humanoid-gym/README.md),
[LeRobot guide](../integrations/lerobot/README.md), and
[CPU validation guide](README.md). Source revisions and extraction boundaries
are recorded in the [G1 manifest](../integrations/humanoid-gym/SOURCE_MANIFEST.json)
and [LeRobot manifest](../integrations/lerobot/source_manifest.json).

Raw logs, traces, executed commands, and local audit registries are retained
outside the source distribution. Checkpoints and simulator assets are external.
