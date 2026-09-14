# Residual PRISM learned-gate control

This diagnostic compares the historical residual PRISM architecture with a
version that learns a scale on its internal polynomial interaction. It tests
that change while retaining the original representation and training setup.
It does not replace the default G1 recipe or report new benchmark results.

## What changes

Both methods use the degree-two residual encoder, including its raw-observation
skip, polynomial input/output projections, affine A/S transforms, ELU, actor and
critic MLPs, and the 500-PPO-update polynomial-branch warmup.

The learned-gate variant adds one unconstrained scale per interaction feature:

~~~python
raw = raw_proj(x)
h = poly_in_proj(x)
p0 = A[0](h)
interaction = A[1](h) * S[0](p0)
p = p0 + alpha * interaction
features = ELU(raw + beta * poly_out_proj(p))
~~~

The legacy model uses the same expression with the interaction scale fixed at
one. In the new control, `alpha` is a trainable 256-element vector initialized
to **1.0**. It multiplies the internal product, not the entire polynomial branch.
Both methods retain the separate scheduled scalar
`beta = min(completed_PPO_updates / 500, 1)`.

Shared weights are initialized in the same order for a paired training seed;
creating the constant gate does not consume additional random draws. At
`alpha=1`, matched shared weights and warmup state give the legacy computation.
After training begins, gradients and subsequent trajectories can diverge.
The gate uses the existing PPO objective and may become negative or exceed one.
Its saved values are fixed during evaluation; the features still depend on the
current observation history.

| Recipe | Actor identity | Actor mean parameters | All trainable parameters |
|---|---|---:|---:|
| `legacy-prism` | `g1_residual_poly_v1` | 921,996 | 1,320,857 |
| `residual-learned-gate` | `g1_residual_learned_gate_v1` | 922,252 | 1,321,113 |

Totals include actor, critic, and Gaussian standard-deviation parameters;
buffers are excluded. Learning these scales necessarily adds **256 parameters**.
This is not an exact parameter-count match. A result would measure the effect
of adding trainable interaction scales in this retained architecture; it would
not by itself establish a general causal explanation for PRISM performance.

The default `prism` recipe remains `g1_gated_poly_v2`. That architecture uses
`u * (1 + alpha * v)`, initializes its gates to 0.01, and has different
projections, no legacy raw skip, and no scalar warmup. Its results and
checkpoints belong to a separate comparison. See the
[migration guide](../../GATED_MIGRATION.md).

## Fresh training and evaluation

The diagnostic trains **five fresh seeds, 1–5, for each method**. Seed values
are paired across methods, giving ten fresh training runs. Each run uses
3,001 PPO updates and 4,096 training
environments. It does not initialize either method from an archived checkpoint.

Each final checkpoint is evaluated at seeds **101, 102, 103**, using
`g1_balanced_timeout_v2`: 100 environments contribute two episodes each,
for 200 episodes per evaluation. Settings are `match_nopush`, matching terrain,
pushes off, and the same nominal 24-second horizon.

That gives 600 episodes per checkpoint, 3,000 per method, and **6,000 total**
across 30 evaluations. Tracking errors are averaged within each episode before
episodes are averaged. The three evaluation-seed results are averaged within
each training run; the reported mean and **sample SD (ddof=1)** then use the
five training-run means. Episodes and evaluation seeds are not treated as
independent training seeds.

Prepare the suite using the simulator Python and upstream checkouts described
in the [installation guide](README.md):

~~~bash
uv run --no-project python integrations/humanoid-gym/run_residual_gate_controls.py \
  --prepare \
  --python "$ISAAC_PYTHON" \
  --unitree-root "$UNITREE_ROOT" \
  --rsl-root "$RSL_ROOT" \
  --output-dir "$CONTROL_OUTPUT"
~~~

Choose a fresh `CONTROL_OUTPUT` directory. Preparation records the inputs and
freezes source files; it does not train or evaluate a policy. Execute the
prepared diagnostic separately:

~~~bash
uv run --no-project python integrations/humanoid-gym/run_residual_gate_controls.py \
  --execute --output-dir "$CONTROL_OUTPUT"
~~~

Regenerate its report without running a simulator:

~~~bash
uv run --no-project python integrations/humanoid-gym/run_residual_gate_controls.py \
  --report --output-dir "$CONTROL_OUTPUT"
~~~

The outputs are `report/diagnostic_summary.json` and
`report/diagnostic_summary.md`. Add `--require-complete` to the report command
for a final completeness assertion. Missing runs remain pending; they do not
supply fabricated five-seed averages. This diagnostic has its own report and
does not populate or change the baseline/larger/default-PRISM main table.

## Archived checkpoints are a separate comparison

Re-evaluating the two historical warmup checkpoints at evaluation seeds
101–103 can compare their saved policies with the new evaluator. It is not
fresh training and does not supply any of the ten diagnostic training runs.
Their directory labels suggest training seeds, but the old checkpoints did not
save an independently verified training seed or resolved training configuration.
Retain that uncertainty and keep their results separate.

Matching evaluation seeds and configurations also does not guarantee identical
command sequences after policy-dependent resets. Distinguish an observed
performance difference from a tested explanation for it, and report the full
five-seed comparison before drawing conclusions about the learned gate.
