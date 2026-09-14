# Five-seed locomotion table

Use this workflow for newly trained baseline, larger MLP, and learnable-alpha
PRISM. Historical residual-policy results use a different architecture and
evaluation protocol.

Train seeds **1, 2, 3, 4, 5** independently from initialization for each method.
Evaluate every final checkpoint with the same evaluation seeds and settings.
The main suite uses evaluation seeds **101, 102, 103**, 200 episodes and 100
environments per evaluation: 600 episodes per training checkpoint.

The evaluator must write g1_balanced_timeout_v2 records containing the resolved
environment, actor identity, checkpoint and training-manifest hashes, source
hashes, runtime, and per-episode results. A training-start manifest alone is
insufficient: the evaluator must bind its final checkpoint to that manifest.

Create a suite JSON with this structure. Paths are absolute or relative to the
suite file. Add one entry per method and training seed. An absent entry, null
manifest, or missing evaluation file remains pending.

~~~json
{
  "schema_version": 1,
  "training_seeds": [1, 2, 3, 4, 5],
  "evaluation_seeds": [101, 102, 103],
  "max_iterations": 3001,
  "checkpoint_iteration": 3001,
  "num_envs": 4096,
  "evaluation": {
    "protocol_id": "g1_balanced_timeout_v2",
    "episodes": 200,
    "num_envs": 100,
    "condition_name": "match_nopush",
    "terrain_mode": "match",
    "push_mode": "off"
  },
  "runs": [
    {
      "variant": "baseline",
      "training_seed": 1,
      "training_manifest": "baseline/seed1/prism_run_manifest.json",
      "evaluations": [
        "baseline/seed1/eval101.json",
        "baseline/seed1/eval102.json",
        "baseline/seed1/eval103.json"
      ]
    }
  ]
}
~~~

The other method names are larger and prism. Generate the table without
installing a simulator:

~~~bash
uv run --no-project python integrations/humanoid-gym/build_main_table.py \
  --suite outputs/g1_main/suite.json \
  --output-dir outputs/g1_main/table
~~~

The script writes main_table.tex, main_table.json, and inventory.md. The JSON
status is complete or pending. Add **--require-complete** for a final queue
assertion: pending still writes its inventory, then exits with status 1.
Invalid evidence exits with status 2.

All three methods and all five training seeds must pass validation before any
five-seed mean or SD is printed. Missing data produces dashes. Conflicting seeds,
budgets, protocols, architecture identities, hashes, or training settings
produce an error. Per-episode records must reproduce saved aggregate means.

Statistics use equal averages across evaluation seeds within each training run,
then the mean and **sample SD (ddof=1)** across five training-run means.
Episode counts are not treated as independent training seeds.

Parameters include actor, critic, and learned Gaussian standard deviations;
buffers are excluded. The released counts are:

| Method | Actor mean | All trainable parameters |
|---|---:|---:|
| Baseline | 527,244 | 926,105 |
| Larger MLP | 724,932 | 1,123,793 |
| PRISM, gated v2 | 724,876 | 1,123,737 |

LaTeX output preserves the seven main-table columns, \sname{}, \xmark/\cmark,
and label tab:locomotion_main. Define these macros and load booktabs in the
manuscript. Complete tables bold the actual best mean for each metric, including
ties, regardless of method. Pending tables make no best-result claim. The
caption does not assert a causal explanation.
