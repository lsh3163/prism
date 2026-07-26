# Reproducibility

## Common Controls

- Random seed: `1000`
- PRISM degree: `2`
- Interaction mode: gated
- Interaction-scale initialization: `1e-2`
- Output normalization: RMSNorm
- No force, wrench, tactile, contact-label, or privileged physical inputs are
  added to the deployed policy.

## BFM-Zero

Upstream:

- repository: <https://github.com/LeCAR-Lab/BFM-Zero>
- commit: `b87916f52d3d9e6eeba484f5e80851a235191837`

Paper-aligned training:

- accelerator: one NVIDIA A40
- parallel environments: `512`
- environment steps: `9,600,000`
- agent updates per collection: `16`
- replay-buffer size: `2,560,000`
- checkpoint interval: `4,800,000` environment steps
- PRISM input: deployable `history_actor` stream
- PRISM post-projection layers: `2`

Evaluation uses all `40` LAFAN motions with `128` parallel environments and
reports tracking EMD from the aligned `9.6M` checkpoint. Scenario settings are:

| Scenario | Dynamics overrides |
|---|---|
| Nominal | Disable training-time dynamics randomization |
| Low friction | Nominal settings plus static and dynamic friction fixed to `0.20` |
| Payload mass | Nominal settings plus link-mass scale fixed to `1.15` |

The evaluator receives the same motion set, rollout settings, and scenario
overrides for every method. The released `tracking_eval.py` accepts additional
Hydra overrides through `--hydra-overrides-json`.

Capacity controls change only these environment variables:

| Method | Overrides relative to the shared recipe |
|---|---|
| BFM-Zero | `BFM_HISTORY_CONDITIONER_TYPE=linear` |
| Larger BFM-Zero | Baseline plus `BFM_CORE_HIDDEN_DIM=2560` and `BFM_CORE_HIDDEN_LAYERS=6` |
| PRISM | Values in `configs/bfm_zero_prism.env` |

Load `configs/bfm_zero_prism.env` before invoking the patched BFM-Zero
training entry point:

```bash
set -a
source /path/to/prism/configs/bfm_zero_prism.env
set +a
uv run python -m humanoidverse.train
```

Dataset and simulator installation follow the upstream BFM-Zero instructions.

## SmolVLA

Upstream:

- repository: <https://github.com/huggingface/lerobot>
- commit: `2d7a42011a4f8e05a8c85d5fb908da258d4cc7b1`
- pretrained initialization: `lerobot/smolvla_base`
- dataset: `HuggingFaceVLA/libero`

Paper-aligned training:

- accelerator: one NVIDIA A40
- batch size: `64`
- training horizon: `100,000` steps
- reported checkpoint: `80,000`
- workers: `8`
- scheduler warmup: `100` steps
- vision encoder: frozen
- action expert: trainable
- proprioceptive state projection: trainable
- VLM weights: initialized from pretrained SmolVLA
- suites: Spatial, Object, Goal, and Long (`libero_10`)

Official `eval50` uses `500` episodes per suite (`2,000` total).

Capacity controls change only the proprioceptive conditioner:

| Method | Conditioner settings |
|---|---|
| SmolVLA | `state_conditioner_type=linear` |
| Larger SmolVLA | `state_conditioner_type=mlp`, hidden width `2048`, `3` layers |
| PRISM | `state_conditioner_type=prism`, degree `2`, gate init `1e-2`, RMSNorm |

After applying the LeRobot patch, the core arguments are:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.state_conditioner_type=prism \
  --policy.state_conditioner_num_layers=2 \
  --policy.state_conditioner_product_mode=gated_quadratic \
  --policy.state_conditioner_gate_scale_init=1e-2 \
  --policy.state_conditioner_use_rmsnorm=true \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=100000 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --batch_size=64 \
  --num_workers=8 \
  --seed=1000 \
  --steps=100000 \
  --policy.device=cuda \
  --output_dir=/path/to/smolvla-prism
```

Evaluate the aligned checkpoint with:

```bash
lerobot-eval \
  --policy.path=/path/to/smolvla-prism/checkpoints/080000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda \
  --seed=1000 \
  --output_dir=/path/to/smolvla-prism-eval50
```
