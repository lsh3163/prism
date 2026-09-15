# G1 Humanoid-Gym simulation experiments

This directory contains the G1 task, environment overrides, actors, training,
nominal evaluation, and physics probes.
It adapts the Humanoid-Gym PPO recipe to Unitree G1 using Unitree RL Gym and
RSL-RL. It is not an XBot-L reproduction of the official Humanoid-Gym repository.

New `prism` runs use `g1_gated_poly_v2`: every higher-order interaction has a
learned per-feature alpha, initialized to `0.01` and optimized by PPO. There is
no scheduled warmup. `gated_actor.py` implements the common gated recurrence in
Python 3.8, with the G1 ELU projection and actor/critic interfaces.

Historical residual checkpoints retain `g1_residual_poly_v1` and their original
keys in `actor.py`. Select them explicitly with `legacy-prism` or a
`legacy-degree*` recipe. Old checkpoints are rejected by the new actor; no
weight conversion or equivalence between these variants is implied.

The separate [residual gate control](RESIDUAL_GATE_CONTROL.md) compares
`legacy-prism` with `residual-learned-gate` while preserving the historical
encoder, initial shared weights, raw path and 500-update warmup. Its alpha starts
at 1 and gates only the residual interaction term. This diagnostic is not the
default shared gated architecture or a substitute for its benchmark results.

The [capacity comparison](CAPACITY_COMPARISON.md) expands the same gated-v2
formula to approximately 1.321M total parameters (`prism-1321k`). It compares
that actor with a matched MLP (`mlp-1321k`) and approximately 1.500M wide/deep
MLPs (`mlp-wide-1500k`, `mlp-deep-1500k`), retaining the same critic and PPO
budget. These optional recipes have a separate four-method, five-seed report.

The [seed-1 robustness evaluation](ROBUSTNESS.md) reuses their final checkpoints
on nominal ground, friction 0.20, and uniform link mass increased by 15%. It
records realized physics and repeats nominal evaluation to check parity. This
is one-training-seed evidence; uniform mass scaling is not an attached payload.

## Source and validation status

- New gated actors pass degree-three polynomial/gradient checks, learned-alpha
  optimizer and save/load checks, privileged-critic isolation, and rejection
  of incompatible historical checkpoints. A real two-environment, two-update
  PPO run passed on September 13, 2026: all 256 alpha features changed, Adam
  recorded 16 optimization steps, and the saved checkpoint reloaded exactly.
  This is a new training recipe requiring fresh performance results.
- Historical CPU parity passed against the original actor for baseline, larger MLP,
  degree 1, degree 2, degree 2 with warmup, and degree 3. Polynomial actor
  gradients also match exactly. An existing trained degree-2 warmup checkpoint
  loads strictly and gives exactly matching actions.
- Extracted noisy actor/critic history and all 17 G1-specific reward functions
  match the original source on identical synthetic states.
- Historical imports and task registrations passed in Python 3.8 with Isaac Gym
  Preview 4 using clean source trees at the upstream commits below.
- A historical residual-actor GPU simulation passed on September 13, 2026 using the clean pinned
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

The historical residual actor's single nominal episode returned `148.496765`, with mean planar velocity
error `0.143841 m/s` and mean absolute yaw-rate error `0.048508 rad/s`. It took
151.36 seconds on the validation machine. These are smoke-test measurements,
not paper results. The checkpoint SHA-256 is
`f05a364979651edf297c008ae9e4e00fba824aeaa56208f185a86f78c1914183`.
Local ignored evidence is under
`outputs/prism_release_validation/20260913T143558Z/humanoid/`: command and source
hash manifests, simulator logs, episode metrics, both trace arrays, and their
comparison. The checkpoint and local evidence are not distributed in this release.

New gated PPO evidence is separate, under
`outputs/gated_g1_validation/20260913T174104Z/`. The public training CLI ran
240 environment transitions in 36.80 seconds. Alpha remained finite and changed
by up to `0.004110` from its `0.01` initialization; this verifies learned PPO
updates, not a locomotion benchmark. See its `SUMMARY.md`, `execution.json`,
`gate_validation.json`, simulator log and per-run training manifest.

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
`runtime.py` registers these tasks and actors in the current process. It adds
the compatibility warmup hook for legacy actors when absent; the new gated
actor has no warmup method and is optimized directly by the ordinary PPO optimizer.
It does not modify either checkout.
Use a clean pinned checkout to avoid inheriting unrelated local extensions.

## Training and ablations

| `--variant` | Actor | Polynomial degree | Alpha / scale |
|---|---|---:|---:|
| `baseline` | MLP `[512, 256, 128]` | — | — |
| `larger` | MLP `[648, 328, 160]` | — | — |
| `degree1` | Gated encoder + MLP | 1 | No interaction gate |
| `prism`, `degree2` | Gated encoder + MLP | 2 | Learned per feature, init `0.01` |
| `degree3` | Gated encoder + MLP | 3 | Learned per feature, init `0.01` |
| `legacy-larger` | MLP `[816, 352, 160]` | — | — |
| `legacy-degree1/2/3` | Historical residual encoder + MLP | 1 / 2 / 3 | Fixed scalar `1` |
| `legacy-prism` | Historical residual encoder + MLP | 2 | Scalar warmup over 500 PPO updates |

The new degree-two actor mean has 724,876 parameters, including its encoder.
The new `larger` control has 724,932 (56 more, a 0.0077% difference).
Total degree-two actor/critic parameters, including Gaussian standard deviation,
are 1,123,737. The marker buffer is excluded from parameter counts.

All tasks use a 705D actor input (15 frames × 47 features), 219D privileged
critic input (3 × 73), and 12 actions. The polynomial width is 256. The critic
is a plain `[768, 256, 128]` MLP. The training config retains 4,096 environments,
60 steps per PPO iteration, 3,001 iterations, learning rate `1e-5`, two learning
epochs, four minibatches, gamma `0.994`, and lambda `0.9`.
Actor histories include gait phase, velocity commands, joint position/velocity,
prior actions and IMU features. Privileged physical targets remain critic-only.

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

The default suite contains baseline, larger, degree one, PRISM/degree two and
degree three. It does not duplicate the `degree2` alias or include legacy runs.
Use `--variants legacy-prism legacy-degree1 legacy-degree2 legacy-degree3`
to explicitly select a historical matrix.

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
SHA-256. The default and capacity PRISM recipes use `g1_gated_poly_v2`;
explicit legacy PRISM runs use `g1_residual_poly_v1`, and the residual learned-gate
diagnostic uses `g1_residual_learned_gate_v1`. MLP controls have a null PRISM actor
identifier and retain their selected recipe name. Keep this file
with checkpoints when moving or sharing a run. It records training inputs,
not evidence that training completed or reproduced a paper result. It is not
a complete dependency lock or robot-asset archive.

Use nonnegative seeds. The legacy `--seed=-1` random-seed shortcut is rejected
because upstream does not retain the sampled value. To resume, select an exact
run and checkpoint with `--resume --load_run=<run-directory> --checkpoint=<number>`;
implicit latest-run/checkpoint selection is rejected. Resume loads the original
optimizer and actor buffers through the unchanged upstream loader and writes
the continuation to a fresh run directory.

The new gated calculation is

```text
phi = factors[0](x)
phi = phi * (1 + alpha[i-1] * factors[i](x))   for i = 1, ..., degree-1
z = ELU(projection(phi))
action_mean = actor_MLP(z)
```

Each factor is an affine 705→256 map. Alpha has shape `(degree-1, 256)` and is
an `nn.Parameter`, initialized to `0.01`; later factor biases start at zero.
The post-interaction projection is affine 256→256, followed by ELU and actor
MLP widths `[512, 256, 128]`. Degree one omits alpha entirely. The checkpoint
contains `actor_variant_version=2`, `actor_encoder.factors.*`,
`actor_encoder.interaction_scales`, and `actor_encoder.projection.*`.
Loading requires strict schema matching; residual keys and warmup buffers do
not belong to this actor. No scheduled updates change alpha.

The explicit legacy residual calculation remains

```text
h = poly_in_proj(x)
p = A[0](h)
p = p + A[i](h) * S[i-1](p)       for i = 1, ..., degree-1
z = ELU(raw_proj(x) + scale * poly_out_proj(p))
action_mean = actor_MLP(z)
```

`scale = min(update / 500, 1)` for `legacy-prism`. The polynomial branch
before ELU has the configured degree bound. The final actor includes ELU
activations and is not itself a polynomial function of the observation.
Warmup buffers are stored in the checkpoint and restored when loading it.

## Evaluation and probes

Use explicit checkpoint paths; some existing experiment artifacts use
`model_3001.pt`, so do not silently assume `model_3000.pt`.

```bash
uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/evaluate.py" \
  --variant=prism --model_path=/absolute/path/to/new_gated_model.pt \
  --variant_name=prism --seed=1 --episodes=200 --num_envs=100 --headless \
  --save_path=outputs/humanoid/seed1_nominal.json

uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/probe_future_response.py" \
  --mlp-path=/absolute/path/to/baseline.pt \
  --prism-path=/absolute/path/to/prism.pt --horizon=5

uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/probe_hidden_physics.py" \
  --mlp-path=/absolute/path/to/baseline.pt \
  --prism-path=/absolute/path/to/prism.pt --stable-only
```

Evaluation defaults to the new `prism` recipe. For an old residual warmup
checkpoint, use `--variant=legacy-prism`. Original task IDs such as
`--task=g1_humanoidgym_ppo_poly_warmup` still select only their historical actor;
they are never redirected to the new implementation. Results record the actual
actor ID and checkpoint SHA-256, and existing `--save_path` files are rejected.

Probes also default to gated degree two. Add `--prism-variant=legacy-prism` when
using an old residual checkpoint. In the hidden-physics probe, gated `poly`
features are the raw multiplicative features before projection; `combined` and
`encoder` use the projected ELU representation. Historical feature definitions
are retained under the explicit legacy choice.

Nominal evaluation disables observation noise and friction/mass randomization,
delayed/noisy actions, and pushes. Initial states and velocity/heading commands
remain randomized. The default `--evaluation_protocol=balanced` assigns an equal
episode quota to each environment: 200 episodes with 100 environments means the
first two completed episodes from each. Survival requires reaching the actual
time limit, and terminal-step tracking errors are captured before automatic
reset. Linear error is the mean planar velocity-error norm; yaw error is the
mean absolute yaw-rate error. Result JSON retains the protocol, individual
episodes, checkpoint identity, and available training provenance.

Use `--evaluation_protocol=legacy-pooled` only to reproduce historical tables.
That protocol takes the first completed episodes across all environments, uses
the old length-based survival threshold, and retains post-reset metric timing.
It can overrepresent repeated early failures. Do not combine its results with
the balanced protocol when computing a table.
The extracted task supports the paper plane terrain; rough-terrain experiments
are not represented as equivalent evaluations.

The hidden-physics probe predicts privileged quantities from frozen actor
representations on a shared rollout. The future-response script retains its
historical five-step targets, stability filter, split and ridge-regression
implementation (`--ridge=10` by default). This is not yet reconciled with the
paper's stated OLS protocol. Its slip target is contact-weighted planar foot
speed and power is the sum of absolute joint torque × velocity. Review the
target/window construction in the script before comparing it to a paper row.

## Five-seed main table with learned alpha

See [MAIN_TABLE.md](MAIN_TABLE.md) for the complete input schema and evidence checks.

`run_main_table.py` registers exactly three methods (`baseline`, `larger`, and
new gated `prism`) with independent training seeds 1–5. Every run starts fresh
and uses 4,096 training environments and 3,001 PPO updates. Each final
checkpoint is evaluated with the shared evaluation seeds 101–103, with 200
episodes per evaluation and 100 parallel environments: 600 evaluation episodes
per trained policy. The evaluator uses `g1_balanced_timeout_v2` throughout.

The total trainable parameter counts, including the critic and Gaussian action
standard deviation, are 926,105 for MLP, 1,123,793 for Larger MLP, and 1,123,737
for gated PRISM. Thus the two larger models are reported as **1.124M**, replacing
the historical residual model's 1.321M. Alpha is a learned parameter initialized
to 0.01; its optimizer membership, updates, and final statistics are recorded.

Prepare a fresh output directory after validating the simulator environment:

```bash
uv run --no-project --python "$GYM_PYTHON" python "$GYM_CODE/run_main_table.py" \
  --output-dir "$PRISM_ROOT/outputs/g1_main_five_seeds" \
  --python "$GYM_PYTHON" --unitree-root /path/to/unitree_rl_gym \
  --rsl-root /path/to/rsl_rl --prepare

uv run --no-project --python "$GYM_PYTHON" python \
  "$PRISM_ROOT/outputs/g1_main_five_seeds/source/integration/run_main_table.py" \
  --output-dir "$PRISM_ROOT/outputs/g1_main_five_seeds" --execute
```

Preparation copies the integration and upstream source/resources into the
output directory and records their hashes. Execution uses that frozen copy,
runs one GPU job at a time, and stops on subprocess or provenance validation
failure. `queue_state.json` records progress, while each run retains its log,
training manifest, checkpoints, completion marker, and evaluation files.
Completed work can be recognized on a later `--execute`; incomplete training
requires inspection instead of silently resuming with different RNG state.
On the inspected laptop, historical full runs took roughly 2.7–3.1 hours each;
15 fresh runs and 45 evaluations are a multi-day workload.

`build_main_table.py --suite <output>/suite.json --output-dir <output>/table`
generates `main_table.tex`, `main_table.json`, and `inventory.md`. The queue calls
it as results arrive. For each metric it first averages the three evaluations
within one training seed, then reports the mean and sample standard deviation
(`ddof=1`) over the five trained policies. Survival is expressed in percent;
episode length is in control steps. Episode-level variation is not used as the
reported training-seed standard deviation. Incomplete results remain pending;
legacy actors, inconsistent budgets/protocols, and mismatched identities are
rejected. The generated caption describes the comparison without presupposing
an improvement or a causal explanation. Manuscript files are not edited.

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
  -m unittest discover -s tests -p 'test_humanoid*.py' -v
```

Set `HUMANOID_CHECKPOINT` to an original degree-2 warmup checkpoint to also
check trained-weight parity. Preserve `LICENSE.unitree` and `LICENSE.rsl-rl`
when redistributing the extracted code.
