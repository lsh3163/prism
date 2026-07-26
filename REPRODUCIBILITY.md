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

Evaluation reports tracking EMD under nominal, low-friction, and
payload-mass conditions. The comparison uses the aligned `9.6M` checkpoint
for every method.

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

After applying the LeRobot patch, the core arguments are:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.state_conditioner_type=prism \
  --policy.state_conditioner_product_mode=gated_quadratic \
  --policy.state_conditioner_gate_scale_init=1e-2 \
  --policy.state_conditioner_use_rmsnorm=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --batch_size=64 \
  --num_workers=8 \
  --seed=1000 \
  --steps=100000
```
