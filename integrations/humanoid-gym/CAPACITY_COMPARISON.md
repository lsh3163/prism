# Gated PRISM capacity comparison

This study compares a roughly 1.321M-parameter gated PRISM policy with a
similarly sized MLP and two larger MLP controls. All four methods are trained
from scratch. The architecture sizes are chosen from parameter counts before
training, not selected using evaluation outcomes.

## Models and parameter budgets

`prism-1321k` expands the simplified `g1_gated_poly_v2` representation from
256 to **334 features**, with actor-head hidden widths **513, 256, 128**.
The degree-two interaction is unchanged:

~~~text
u = A1(x)
v = A2(x)
phi = u * (1 + alpha * v)
~~~

The affine factors map the same 705-dimensional observation history to 334
features. Each feature has an unconstrained trainable `alpha` initialized to
**0.01**. A 334-to-334 affine projection and ELU precede the actor head.
This retains the simplified gated architecture: there is no historical raw
residual skip or scheduled polynomial-branch warmup.

| Recipe | Representation width | MLP head hidden widths | Actor mean parameters | All trainable parameters |
|---|---:|---|---:|---:|
| `prism-1321k` | 334 | 513, 256, 128 | 921,715 | 1,320,576 |
| `mlp-1321k` | — | 816, 352, 160 | 922,092 | 1,320,953 |
| `mlp-wide-1500k` | — | 928, 384, 224 | 1,100,844 | 1,499,705 |
| `mlp-deep-1500k` | — | 768, 512, 256, 128 | 1,101,708 | 1,500,569 |

Actor mean counts include PRISM's encoder and projection where present. Every
method uses the same critic, **219 → 768 → 256 → 128 → 1**, with 398,849
parameters, plus 12 learned Gaussian standard deviations. Total counts include
these parameters and exclude buffers.

The matched MLP differs from PRISM by only **377 parameters**, so the budgets
are close rather than exactly equal. It provides the comparison at the same
approximate capacity. The two roughly 1.5M controls test larger MLPs with three
or four hidden layers and different hidden widths. Larger controls alone would
leave the comparison at matched capacity unanswered.

“Wide” and “deep” describe the MLP controls: the wide model retains three hidden
layers, while the deep model has four. This is not a claim that the deep MLP has
more layers than the entire PRISM computation graph.

## Training and evaluation

Each method uses **five fresh training seeds, 1–5**, for **20 training runs**.
All runs use 3,001 PPO updates, 4,096 training environments, and the same PPO,
reward, observation, action, and low-level control settings. No archived
checkpoint initializes these runs.

Evaluate the same final checkpoint from each run at seeds **101, 102, 103**.
Each evaluation uses the nominal `g1_balanced_timeout_v2` protocol:

- 100 environments contribute exactly two episodes each: 200 episodes per
  evaluation, with extra completed events excluded from the aggregate.
- Matching terrain and pushes off, with the task's sampled initial conditions
  and command distribution and the same nominal 24-second horizon.
- Survival uses the environment timeout flag; tracking errors are captured
  before terminal resets and averaged over the steps of each episode.

There are **600 episodes per trained model**, **3,000 per method**, and
**12,000 total** across 60 evaluations.

Average episodes within each evaluation, then average the three evaluation-seed
results within each training run. Report the mean and **sample SD (ddof=1)**
across the five training-run means. Episodes and evaluation seeds do not
substitute for independently trained seeds.

## Run the separate suite

Use the simulator Python and pinned upstream checkouts from the
[installation guide](README.md). Choose a fresh output directory:

~~~bash
uv run --no-project python integrations/humanoid-gym/run_capacity_controls.py \
  --prepare \
  --python "$ISAAC_PYTHON" \
  --unitree-root "$UNITREE_ROOT" \
  --rsl-root "$RSL_ROOT" \
  --output-dir "$CAPACITY_OUTPUT"
~~~

Preparation records inputs and freezes the source; it does not run training.
Preview the prepared workload:

~~~bash
uv run --no-project python integrations/humanoid-gym/run_capacity_controls.py \
  --dry-run --output-dir "$CAPACITY_OUTPUT"
~~~

When sharing a GPU with another suite, let its active training job finish before
starting this one:

~~~bash
uv run --no-project python integrations/humanoid-gym/run_capacity_controls.py \
  --execute --output-dir "$CAPACITY_OUTPUT"
~~~

Regenerate the report without running a simulator:

~~~bash
uv run --no-project python integrations/humanoid-gym/run_capacity_controls.py \
  --report --output-dir "$CAPACITY_OUTPUT"
~~~

Reports are written to `report/capacity_summary.json` and
`report/capacity_summary.md`. Missing training runs or evaluations remain
pending; they do not establish a completed five-seed result. Add
`--require-complete` to the report command for a final completeness assertion.

## Interpretation and other comparisons

These recipes add a separate capacity study. The ordinary `prism`,
`baseline`, and `larger` recipes and their existing suite retain their own
configurations and results.

The [residual learned-gate control](RESIDUAL_GATE_CONTROL.md) asks a different
question: it retains the historical residual architecture and 500-update
warmup, adding an internal interaction scale initialized to one. This capacity
study uses the simplified gated-v2 formula initialized at 0.01. Keep the
studies' actor identities, checkpoints, and reports separate.

No performance improvement is guaranteed by the larger PRISM representation.
Results can compare the tested architectures at these capacity budgets;
performance alone cannot prove that learned feature interactions explain a
difference or that a policy has learned a particular physical mechanism.
