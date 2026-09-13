# G1 Humanoid-Gym simulation experiments

This directory contains the G1 task, environment overrides, actor, training,
nominal evaluation, and physics probes extracted from the research code.
It adapts the Humanoid-Gym PPO recipe to Unitree G1 using Unitree RL Gym and
RSL-RL. It is not an XBot-L reproduction of the official Humanoid-Gym repository.

The extraction preserves the historical residual polynomial actor and its
checkpoint keys under the identifier `g1_residual_poly_v1`. It does not replace
that actor with the newer gated `prism_robot.PRISMConditioner`. These are distinct implementations; a common
PRISM name does not establish architectural or experimental equivalence.

## Source and validation status

- Exact CPU parity passed against the original actor for baseline, larger MLP,
  degree 1, degree 2, degree 2 with warmup, and degree 3. Polynomial actor
  gradients also match exactly. An existing trained degree-2 warmup checkpoint
  loads strictly and gives exactly matching actions.
- Extracted noisy actor/critic history and all 17 G1-specific reward functions
  match the original source on identical synthetic states.
- Imports and all six task registrations passed in Python 3.8 with Isaac Gym
  Preview 4 using clean source trees at the upstream commits below.
- A headless GPU simulation passed on September 13, 2026 using the clean pinned
  sources and the trained degree-2 warmup checkpoint: seed 1, one environment,
  one episode, 2,400 control steps (24 simulated seconds), survived to timeout.
- A separate 200-step simulation comparison against the original research
  environment gave bit-identical observations, actions, rewards, done flags,
  and root states for the same seed and checkpoint.
- Training convergence, randomized-physics and reset trajectory equivalence,
  and paper-table reproduction remain unverified. The bounded checks do not
  establish aggregate policy performance.

`SOURCE_MANIFEST.json` records hashes of the modified research source and the
extraction boundaries. No checkpoints, recorded robot data, or physical robot
deployment implementation are included here.

The single nominal episode returned `148.496765`, with mean planar velocity
error `0.143841 m/s` and mean absolute yaw-rate error `0.048508 rad/s`. It took
151.36 seconds on the validation machine. These are smoke-test measurements,
not paper results. The checkpoint SHA-256 is
`f05a364979651edf297c008ae9e4e00fba824aeaa56208f185a86f78c1914183`.
Local ignored evidence is under
`outputs/prism_release_validation/20260913T143558Z/humanoid/`: command and source
hash manifests, simulator logs, episode metrics, both trace arrays, and their
comparison. The checkpoint and local evidence are not distributed in this release.

## Environment

Keep this simulator in a separate environment from modern LeRobot and PRISM.
The inspected working environment uses Python 3.8, PyTorch 2.3.1 with CUDA 12.1,
NumPy 1.20.0, Isaac Gym Preview 4, and RSL-RL v1.0.2. The Isaac Gym package
declares Python `<3.9`; installing PRISM's Python `>=3.10` package in this
environment is unnecessary. The actor in this directory imports only PyTorch.

Start from these upstream revisions and install their dependencies, including
the separately obtained NVIDIA Isaac Gym Preview 4 distribution:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
git -C unitree_rl_gym checkout 276801e46c5d433564f24658bac64f254b7d2d4b
git clone https://github.com/leggedrobotics/rsl_rl.git
git -C rsl_rl checkout 2ad79cf0caa85b91721abfe358105f869a784121

# Point to an existing Python 3.8 environment with the simulator dependencies.
export GYM_PYTHON=/absolute/path/to/gym-environment/bin/python
uv pip install --python "$GYM_PYTHON" -e /path/to/isaacgym/python
uv pip install --python "$GYM_PYTHON" -e /path/to/rsl_rl -e /path/to/unitree_rl_gym
export PRISM_ROOT=/absolute/path/to/prism
export GYM_CODE="$PRISM_ROOT/integrations/humanoid-gym"
```

The Unitree checkout supplies the G1 URDF/meshes, base task, simulator setup,
and generic rewards. This release supplies only the experiment-specific code.
`runtime.py` registers these tasks and the actor in the current process and
adds the PPO warmup hook when absent. It does not modify either checkout.
Use a clean pinned checkout to avoid inheriting unrelated local extensions.

## Training and ablations

| `--variant` | Actor | Polynomial degree | Warmup |
|---|---|---:|---:|
| `baseline` | MLP `[512, 256, 128]` | — | — |
| `larger` | MLP `[816, 352, 160]` | — | — |
| `degree1` | Residual polynomial + MLP | 1 | 0 |
| `degree2` | Residual polynomial + MLP | 2 | 0 |
| `prism` | Residual polynomial + MLP | 2 | 500 PPO updates |
| `degree3` | Residual polynomial + MLP | 3 | 0 |

All tasks use a 705D actor input (15 frames × 47 features), 219D privileged
critic input (3 × 73), and 12 actions. The polynomial width is 256. The critic
is a plain `[768, 256, 128]` MLP. The training config retains 4,096 environments,
60 steps per PPO iteration, 3,001 iterations, learning rate `1e-5`, two learning
epochs, four minibatches, gamma `0.994`, and lambda `0.9`.

First inspect a command, then run a short simulator check in the prepared
environment before committing to the full training matrix:

```bash
uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/train.py" \
  --variant prism --seed=1 --num_envs=16 --max_iterations=1 --headless --dry-run

uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/train.py" \
  --variant prism --seed=1 --num_envs=16 --max_iterations=1 --headless

# List the planned five-seed simulation matrix without launching training.
uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/run_suite.py" \
  --seeds 1 2 3 4 5 --dry-run
```

The suite previews commands by default; add `--execute` to launch them:

```bash
uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/run_suite.py" \
  --seeds 1 2 3 4 5 --execute
```

Choose explicit experiment names for standalone training with
`--experiment_name=...`. Upstream writes checkpoints under its
`logs/<experiment_name>/<timestamp>_<run_name>/` directory. Training refuses to
reuse an existing run directory, including an empty one. If two starts choose
the same timestamp/name, rerun with a distinct `--run_name=...`.
The five-seed matrix is a runnable experiment plan, not evidence that all
five reported paper checkpoints have been located.

Before the first PPO rollout, `train.py` writes `prism_run_manifest.json` beside
the future checkpoints. It records the resolved environment/training seed and
configurations, actor identity and degree/warmup, command, runtime versions,
imported source hashes, available git revisions, and any resumed checkpoint's
SHA-256. PRISM runs use `g1_residual_poly_v1`; MLP controls have a null PRISM actor
identifier and retain their `baseline` or `larger` recipe name. Keep this file
with checkpoints when moving or sharing a run. It records training inputs,
not evidence that training completed or reproduced a paper result. It is not
a complete dependency lock or robot-asset archive.

Use nonnegative seeds. The legacy `--seed=-1` random-seed shortcut is rejected
because upstream does not retain the sampled value. To resume, select an exact
run and checkpoint with `--resume --load_run=<run-directory> --checkpoint=<number>`;
implicit latest-run/checkpoint selection is rejected. Resume loads the original
optimizer and actor buffers through the unchanged upstream loader and writes
the continuation to a fresh run directory.

The latent polynomial calculation is

```text
h = poly_in_proj(x)
p = A[0](h)
p = p + A[i](h) * S[i-1](p)       for i = 1, ..., degree-1
z = ELU(raw_proj(x) + scale * poly_out_proj(p))
action_mean = actor_MLP(z)
```

`scale = min(update / 500, 1)` for the warmup variant. The polynomial branch
before ELU has the configured degree bound. The final actor includes ELU
activations and is not itself a polynomial function of the observation.
Warmup buffers are stored in the checkpoint and restored when loading it.

## Evaluation and probes

Use explicit checkpoint paths; some existing experiment artifacts use
`model_3001.pt`, so do not silently assume `model_3000.pt`.

```bash
uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/evaluate.py" \
  --task=g1_humanoidgym_ppo_poly_warmup \
  --model_path=/absolute/path/to/model_3001.pt \
  --variant_name=prism --seed=1 --episodes=200 --num_envs=100 --headless \
  --save_path=outputs/humanoid/seed1_nominal.json

uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/probe_future_response.py" \
  --mlp-path=/absolute/path/to/baseline.pt \
  --prism-path=/absolute/path/to/prism.pt --horizon=5

uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/probe_hidden_physics.py" \
  --mlp-path=/absolute/path/to/baseline.pt \
  --prism-path=/absolute/path/to/prism.pt --stable-only
```

The historical evaluator disables observation noise and friction/mass
randomization, delayed/noisy actions, and pushes by default. Survival means
reaching the episode time limit. Linear tracking error is mean planar
velocity-error norm; yaw error is mean absolute yaw-rate error. It retains the
original post-step metric timing, including the environment's automatic resets.
The extracted task supports the paper plane terrain; rough-terrain experiments
are not represented as equivalent evaluations.

The hidden-physics probe predicts privileged quantities from frozen actor
representations on a shared rollout. The future-response script retains its
historical five-step targets, stability filter, split and ridge-regression
implementation (`--ridge=10` by default). This is not yet reconciled with the
paper's stated OLS protocol. Its slip target is contact-weighted planar foot
speed and power is the sum of absolute joint torque × velocity. Review the
target/window construction in the script before comparing it to a paper row.

## Unresolved reproduction work

1. Extend the matched-source simulator check across episode resets and training
   randomization, including noisy/delayed actions and randomized PD gains. The
   verified 200-step nominal trace has no episode reset or randomization.
2. Identify the exact checkpoint, evaluation settings and seed aggregation for
   each paper table. The located nominal result files cover seeds 1 and 2;
   the main table, degree ablation and existing result files are not yet mapped
   to one verified protocol. Do not substitute the BFM-Zero results for G1 PPO.
3. Recover the exact low-friction/payload and fixed-command trace recipes before
   claiming those paper scenarios are reproduced. Changing an output label
   does not configure a different physical scenario.
4. Reconcile the future-response OLS/ridge and split conventions with the paper.

For source validation without a simulator rollout:

```bash
uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/verify_environment.py" \
  --reference-root=/path/to/original/modified/unitree_rl_gym

cd "$PRISM_ROOT"
RSL_RL_REFERENCE_ROOT=/path/to/original/modified/rsl_rl \
  uv run --no-project --python "$GYM_PYTHON" python \
  -m unittest discover -s tests -p test_humanoid_actor.py -v
```

Set `HUMANOID_CHECKPOINT` to an original degree-2 warmup checkpoint to also
check trained-weight parity. Preserve `LICENSE.unitree` and `LICENSE.rsl-rl`
when redistributing the extracted code.
