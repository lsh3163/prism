# Seed-1 robustness of the capacity controls

This evaluation uses the final 3001-update checkpoints from **training seed 1**
of `prism-1321k`, `mlp-1321k`, `mlp-wide-1500k`, and `mlp-deep-1500k`.
It performs inference only: there is no adaptation, retraining, or gate update.
The training manifests, completion markers, actor counts, and checkpoint hashes
must match the registered [capacity comparison](CAPACITY_COMPARISON.md).

| Condition | Change from nominal |
|---|---|
| `nominal` | Existing flat-plane evaluation, with training randomization disabled |
| `low_friction` | Ground static/dynamic friction and every robot collision-shape friction fixed to 0.20 |
| `payload_mass` | Every robot rigid-body mass multiplied by 1.15 before simulation, with the upstream inertia recomputation enabled |

**`payload_mass` is uniform link-mass scaling, not an attached payload.** Geometry
and centers of mass are unchanged. The evaluator reads back every actor's body
mass, COM, inertia, inverse inertia, and shape friction. The report checks mass
and inertia scaling against nominal readback from the same creation pipeline.
The low-friction condition changes both sides of the contact because Isaac Gym
Preview 4 uses the average of the shapes' coefficients. The robot API exposes
one `RigidShapeProperties.friction` field; ground static/dynamic settings are
recorded separately. Ground coefficients are configured values, not getter data.

These are new G1 conditions inspired by the BFM scenario definitions, not a
reproduction of BFM or a claim that the two simulators realize identical dynamics.
Call them perturbation evaluations; an OOD claim requires checking the actual
training randomization ranges.

Each checkpoint/condition uses evaluation seeds **101, 102, 103**, 100 environments
and 200 episodes per seed: exactly the first two episodes of each environment.
This gives **600 episodes per condition** and **7,200 episodes across all four
models and three conditions**. Survival means reaching the environment's nominal
24-second timeout. Tracking errors are captured before terminal resets, as in
`g1_balanced_timeout_v2`. Pushes and observation/action noise remain disabled.
The hooks make no additional random draws. Equal evaluation seeds do not ensure
identical command streams after different policies cause different reset times.

Nominal is repeated with the instrumented evaluator. Where the original nominal
evaluation exists, its individual episode records must match within the stated
floating-point tolerance (relative 1e-6, absolute 1e-7); discrete fields must match
exactly. The report rejects undeclared changes to the environment, policy,
runtime, evaluator source, and episode protocol.

In the existing Isaac Gym environment, prepare a fresh frozen suite:

```bash
python integrations/humanoid-gym/run_robustness.py \
  --prepare --capacity-suite /absolute/path/to/capacity-suite \
  --output-dir /absolute/path/to/new-robustness-suite
```

Use its frozen controller thereafter. A CPU Python can run the controller; the
recorded simulator Python runs each evaluation subprocess.

```bash
python /absolute/path/to/new-robustness-suite/source/integration/run_robustness.py \
  --output-dir /absolute/path/to/new-robustness-suite --dry-run
python /absolute/path/to/new-robustness-suite/source/integration/run_robustness.py \
  --output-dir /absolute/path/to/new-robustness-suite --execute \
  --variants prism-1321k mlp-1321k mlp-wide-1500k
python /absolute/path/to/new-robustness-suite/source/integration/run_robustness.py \
  --output-dir /absolute/path/to/new-robustness-suite --execute \
  --variants mlp-deep-1500k
python /absolute/path/to/new-robustness-suite/source/integration/run_robustness.py \
  --output-dir /absolute/path/to/new-robustness-suite --report --require-complete
```

Run sequentially when the GPU is free. A selected checkpoint must be fully
trained; this controller never launches training or interrupts another job.
Existing results are validated and reused, not overwritten. Failed result/log
paths are preserved for investigation; use a fresh suite after changing code.

`robustness_summary.json` and `.md` report each condition's absolute metrics and
survival change in percentage points relative to nominal. Partial condition
means remain pending until all three evaluation seeds finish. These are results
from **one trained policy per architecture**: do not report five-training-seed
standard deviations, general seed robustness, or a causal physics claim from
these rollouts. The manuscript's historical residual-actor results are separate.

## Add the original residual PRISM

`run_residual_robustness.py` adds the freshly trained `legacy-prism` checkpoint
from training seed 1 to this comparison. It evaluates the original residual
actor with its completed 500-update warmup. It does not select the separate
`residual-learned-gate` diagnostic or adopt archived checkpoints with unknown
training seeds.

Prepare a separate extension after the four-model robustness suite is complete:

```bash
python integrations/humanoid-gym/run_residual_robustness.py \
  --prepare --residual-suite /absolute/path/to/residual-training-suite \
  --reference-robustness-suite /absolute/path/to/completed-robustness-suite \
  --output-dir /absolute/path/to/new-residual-robustness-suite
python /absolute/path/to/new-residual-robustness-suite/source/integration/run_residual_robustness.py \
  --output-dir /absolute/path/to/new-residual-robustness-suite --dry-run
python /absolute/path/to/new-residual-robustness-suite/source/integration/run_residual_robustness.py \
  --output-dir /absolute/path/to/new-residual-robustness-suite --execute
python /absolute/path/to/new-residual-robustness-suite/source/integration/run_residual_robustness.py \
  --output-dir /absolute/path/to/new-residual-robustness-suite --report --require-complete
```

The extension pins the completed reference results and the fresh residual
training checkpoint. It preserves the residual training bundle's exact
`config.py` and `runtime.py` in the new evaluation bundle; the reference versions
only add registrations for the four capacity models. Both versions and the
explicit residual architecture differences are recorded and checked. The
remaining training settings and the evaluation protocol, environment, runtime,
and realized physical conditions must match the reference comparison.

This adds **nine evaluations / 1,800 episodes**, with the same three conditions
and evaluation seeds. The original 36 reference evaluations remain unchanged.
Only completed three-evaluation-seed conditions belong in the combined table;
an archived residual nominal score must not fill a missing fresh-seed result.
The extension's `comparison_summary.json` and `.md` combine the five models and
update automatically as residual evaluations finish. The original four-model
result files remain unchanged.
