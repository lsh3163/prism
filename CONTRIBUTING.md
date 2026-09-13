# Contributing

Keep the core small, adapters explicit, and checkpoints compatible. Read the
[actor contracts](ACTOR_CONTRACT.md) before changing a representation and the
[experiment guide](EXPERIMENTS.md) before adding a workflow.

## Source layout

| Directory | Responsibility |
|---|---|
| `src/prism_robot/` | Reusable PyTorch modules; no simulator, LeRobot, or hardware imports. |
| `integrations/` | Pinned adapters, installers, simulator entry points, and backbone recipes. |
| `configs/` | Named experiment configurations and stable actor contracts. |
| `analysis/` | Offline probes, plots, and aggregation with explicit inputs and outputs. |
| `validation/` | Portable source/checkpoint inventory and validation notes. |
| `tests/` | CPU checks of mathematics, gradients, serialization, interfaces, and release tools. |

New compatible adapters should reuse the shared implementation. Historical
actors with different formulas or state-dictionary layouts keep explicit
compatibility implementations until numerical equivalence is established.

## Python conventions

The core supports Python 3.10+. Keep optional dependencies lazy or guarded.
The Isaac Gym adapter supports its separate Python 3.8 environment and must
not import the modern package's unsupported syntax.

Use four spaces, 110-column formatting, double quotes, `snake_case` functions,
`PascalCase` classes, and type annotations on public APIs. Document tensor
shapes, feature order, units, and normalization. Prefer `pathlib.Path`, explicit
function arguments, and entry points with `main()` guards. Importing a module
must not launch training, download models, or connect to hardware.

Preserve upstream notices and avoid unrelated formatting changes in extracted
adapters and patches. Keep generated results, media, environments, datasets,
and checkpoints in ignored output directories or external storage.

## Checks

From the PRISM checkout:

```bash
uv sync --extra test --extra dev
make check
make format  # Apply formatting to maintained code when needed.
make build
```

`make check` runs Ruff lint/format checks and the CPU test suite. Optional
source/checkpoint parity tests require the inputs documented in
[validation/README.md](validation/README.md). Simulator checks run in the
integration's own environment and are reported separately.

GitHub CI runs the CPU checks on Python 3.10/PyTorch 2.1.2 and Python
3.12/PyTorch 2.6.0. Library dependency ranges live in `pyproject.toml`; local
`uv.lock` files and environments are ignored. Simulator dependencies retain
their separate upstream pins. `make format` excludes archived actor/environment
implementations and patches.

The wheel contains the standalone `prism_robot` package. The source archive
also contains the integrations, configurations, documentation, and tests.
Verify a fresh build with:

```bash
uv run --extra dev python validation/check_distribution.py dist
```

## Model and experiment changes

Update [configs/actor_contracts.json](configs/actor_contracts.json), the adapter
guide, and relevant tests when a model changes. Preserve parameter names,
normalization, activation, feature ordering, and warmup buffers for compatible
exports. Load checkpoints strictly and compare fixed-input outputs with the
original implementation. A changed computation needs a new variant ID.

Record source revisions, resolved configurations, dataset/task IDs, training
and evaluation seeds, checkpoint identity, and metric definitions for runs.
Keep unknown provenance explicitly unknown; folder labels and evaluation seeds
do not establish training seeds. Record actor and total-policy parameter counts
separately. Matched comparisons must share data, training budget, action decoding,
controller, and evaluation conditions.
