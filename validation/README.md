# Validation tools

Run the CPU suite from the PRISM checkout:

```bash
uv sync --extra test --extra dev
make check
```

Tests cover polynomial stages, gradients, adapter interfaces, checkpoint
semantics, experiment runners, and release tools. Optional comparisons against
the original G1 actor and a trained checkpoint use:

```bash
RSL_RL_REFERENCE_ROOT=/path/to/research/rsl_rl \
HUMANOID_CHECKPOINT=/path/to/model_3001.pt \
uv run --extra test python -m unittest discover -s tests -v
```

Simulator execution checks use separate environments and are summarized in
[SMOKE_REPORT.md](SMOKE_REPORT.md). CPU checks, simulator smoke tests, and full
benchmark reproduction establish different levels of validation.

## Source and wheel builds

```bash
make build
uv run python validation/check_distribution.py dist
```

The distribution check verifies required integration files and notices, matching
core source in the wheel/archive, and exclusion of checkpoints and local runtime
artifacts. The wheel installs the standalone conditioner; use the repository or
source archive for simulator adapters. GitHub CI runs these build checks after
the CPU suite on Python 3.10 and 3.12.

## Checkpoint inventory

[checkpoint_inventory.py](checkpoint_inventory.py) inspects saved training
configs, G1 checkpoints, safetensors schemas, and evaluation JSON without
requiring a simulator. Copy [checkpoint_sources.example.json](checkpoint_sources.example.json)
and adjust its selectors to your run layout:

```bash
uv run python validation/checkpoint_inventory.py \
  --root runs=/path/to/simulation_runs \
  --spec validation/checkpoint_sources.example.json \
  --output outputs/checkpoint_registry.json
```

Each source declares an `id`, named `root`, relative `glob`, and `kind`:

| Kind | Selected input |
|---|---|
| `g1` | `model_*.pt` containing `model_state_dict`; requires PyTorch. |
| `lerobot` | Saved `train_config.json` beside policy config, preprocessing, and safetensors weights. |
| `bfm` | Saved training `config.json`; inspects actor configuration and checkpoint candidates. |

Optional `evaluation_sources` select G1 evaluator outputs or LeRobot
`eval_info.json` files. Supply additional roots as repeated `--root name=/path`
arguments. The tool never chooses a latest checkpoint automatically. Empty
matches remain visible in `coverage`; a generated registry is not proof that
all intended runs were found.

## Evidence semantics

Training seeds come from saved configurations, independently of directory labels
and evaluation seeds. Actor IDs identify compatible configurations or tensor
schemas; they do not establish the historical source revision or full inference
behavior. Missing seed/config/checkpoint links stay unknown. Different tasks,
checkpoint steps, and aliases do not establish independent training runs.

Output paths use named roots and relative paths. Files up to 32 MiB receive a
complete SHA-256 by default. Larger safetensors files receive an explicit
unhashed whole-file status plus a header SHA-256 and selected tensor shapes.
A header fingerprint identifies schema, not weights. Raise `--max-hash-mib`
when complete large-checkpoint hashes are needed.

Nonfinite source metrics become `null` (unavailable), with their original labels
and JSON pointers under `nonfinite_values`. Aggregates containing them also
remain unavailable; no value is imputed or omitted. Output uses strict JSON and
source files remain unchanged. Keep generated registries and private source
selectors under `outputs/` or external storage.
