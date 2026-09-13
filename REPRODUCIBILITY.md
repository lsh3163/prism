# Training and evaluation recipes

Start with the [integration setup](integrations/README.md) for pinned source
revisions and dependencies. G1 and main Diffusion commands live in their
[Humanoid-Gym](integrations/humanoid-gym/README.md) and
[LeRobot](integrations/lerobot/README.md) guides. The complete simulation workflow
index is [EXPERIMENTS.md](EXPERIMENTS.md).

Record resolved configurations, source revisions, dataset versions, training and
evaluation seeds, checkpoint identity, and raw metrics for every run. Use the
[checkpoint inventory](validation/README.md) to inspect saved evidence. The
stronger-backbone recipes below default to seed 1000; repeating evaluations
does not create independent training seeds.

## BFM-Zero

Apply the BFM training and evaluation patches at upstream commit
`b87916f52d3d9e6eeba484f5e80851a235191837`. Install Isaac Sim and the LAFAN motion
data following upstream instructions. From that prepared environment:

```bash
set -a
source /path/to/prism/configs/bfm_zero_prism.env
set +a
uv run python -m humanoidverse.train
```

The recipe uses 512 parallel environments, 9.6M environment steps, 16 updates
per collection, a 2.56M replay buffer, and 4.8M checkpoint intervals. PRISM
transforms `history_actor` with a degree-2 gated product, gate initialization
0.01, a two-layer projection with Mish, and RMSNorm. The recorded training accelerator
was one NVIDIA A40.

| Method | Overrides after loading the shared recipe |
|---|---|
| Baseline | `BFM_HISTORY_CONDITIONER_TYPE=linear` |
| Larger baseline | Baseline plus `BFM_CORE_HIDDEN_DIM=2560`, `BFM_CORE_HIDDEN_LAYERS=6` |
| PRISM | Values in [bfm_zero_prism.env](configs/bfm_zero_prism.env) |

Evaluate the 9.6M checkpoint on all 40 LAFAN motions with 128 environments:

```bash
uv run python /path/to/prism/integrations/bfm-zero/evaluate_scenarios.py   --bfm-root=/path/to/patched/BFM-Zero   --model-folder=/path/to/aligned/run   --data-path=/path/to/lafan_29dof.pkl
```

The launcher previews all three commands; add `--execute` to run them.

| Scenario | Dynamics |
|---|---|
| Nominal | Training dynamics randomization disabled individually. |
| Low friction | Nominal settings plus static/dynamic friction fixed at 0.20. |
| Payload mass | Nominal settings plus link masses scaled by 1.15. |

Keep observation-noise settings identical across methods. Saved settings are
retained unless `--disable-obs-noise` is selected. Do not use the upstream blanket
`disable_dr=true` with friction/mass events: it disables those overrides after
Hydra composition. Inspect the effective configuration and preserve
`scenario_metadata.json`; scenario names alone do not establish applied dynamics.

## SmolVLA

Use the combined LeRobot integration at `d656da8ccca5989ff0a2207e81fbfa2c2d5bafb1`,
or the original standalone patch at `2d7a42011a4f8e05a8c85d5fb908da258d4cc7b1`.
The [LeRobot guide](integrations/lerobot/README.md) documents installation and
source routing. The launcher below targets the combined integration.

```bash
uv run python /path/to/prism/integrations/lerobot/scripts/smolvla.py train   --lerobot-root=/path/to/patched/lerobot --profile=prism --seed=1000   --output=outputs/smolvla_prism

uv run python /path/to/prism/integrations/lerobot/scripts/smolvla.py eval   --lerobot-root=/path/to/patched/lerobot --profile=prism --seed=1000   --checkpoint=/path/to/checkpoints/080000/pretrained_model   --output=eval_logs/smolvla_prism
```

Commands preview by default; add `--execute` to run them. Training uses
`HuggingFaceVLA/libero`, batch 64, eight workers, AdamW learning rate 1e-4,
100 scheduler warmup steps, and 100K updates. The pinned configuration loads
pretrained VLM weights, freezes the vision-language backbone, and trains the
action expert and state conditioner. The recorded accelerator was one A40.

Evaluate the 80K checkpoint with 50 episodes per task: 500 per suite and
2,000 total across Spatial, Object, Goal, and Long. Use these profiles:

| Profile | State conditioner |
|---|---|
| `baseline` | Linear projection. |
| `larger` | Three-layer MLP, width 2048. |
| `prism` | Gated quadratic product, gate init 0.01, two-layer projection with SiLU. |

The current launcher applies output RMSNorm to all profiles. Historical
baseline/larger saved configurations were not recovered to verify their
normalization; these commands alone do not certify reproduction of the
[archived results](RESULTS.md). Larger models are capacity controls.
