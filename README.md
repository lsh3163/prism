<h1 align="center">PRISM</h1>

<h3 align="center">
  Polynomial Representations for Interaction-Structured Motor Control
</h3>

<p align="center">
  <strong>Seung Hyun Lee</strong> &middot; <strong>Stella X. Yu</strong>
</p>

<p align="center">
  <a href="https://lsh3163.github.io/prism/"><strong>Project Page</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="https://lsh3163.github.io/prism/assets/prism/prism-paper.pdf"><strong>Paper PDF</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="RESULTS.md"><strong>Results</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="REPRODUCIBILITY.md"><strong>Reproducibility</strong></a>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-315d45">
  <img alt="PyTorch 2.1+" src="https://img.shields.io/badge/PyTorch-2.1%2B-cb6d3f">
  <img alt="Backbones: BFM-Zero and SmolVLA" src="https://img.shields.io/badge/Backbones-BFM--Zero%20%7C%20SmolVLA-486b3e">
</p>

> **Motivation.** Robot policies observe individual state coordinates, while
> physical behavior depends on how joint, velocity, command, contact, and load
> effects interact. PRISM makes these couplings learnable from deployable
> proprioception, without adding force sensors, tactile input, contact labels,
> or privileged physical parameters.

PRISM is a compact, learnable polynomial representation for motor-control
policies. It changes only the proprioceptive representation presented to the
policy backbone and can be inserted into both reinforcement-learning and
vision-language-action policies.

## Method

For proprioceptive input $x$, PRISM forms a first-order path and recursively
introduces learned interaction factors:

$$
\begin{aligned}
h_1 &= W_1x+b_1, \\
h_k &= h_{k-1}\odot
\left(1+\alpha_k\odot(W_kx+b_k)\right),
\qquad k=2,\ldots,K, \\
z &= \mathrm{RMSNorm}\left(\mathrm{MLP}(h_K)\right).
\end{aligned}
$$

Each $\alpha_k$ is learned independently for every latent feature. Initializing
these scales near zero preserves a strong first-order path at the start of
training, while optimization determines where higher-order interactions are
useful. Expanding the recurrence yields a representation of degree at most
$K$. The reported stronger-backbone experiments use $K=2$; the implementation
also supports higher degrees and a direct factorized-polynomial mode.

## Stronger-Backbone Results

PRISM improves both backbones while remaining substantially closer in size to
the original model than the larger-capacity control.

| Backbone | Evaluation metric | Baseline | Larger control | **PRISM** |
|---|---|---:|---:|---:|
| BFM-Zero | Mean tracking EMD $\downarrow$ | 1.269 | 1.264 | **1.224** |
| SmolVLA | LIBERO Avg. success $\uparrow$ | 63.50 | 64.90 | **66.55** |

BFM-Zero results use the aligned `9.6M` checkpoint and average tracking EMD
over nominal, low-friction, and payload-mass evaluations. SmolVLA results use
the `80K` checkpoint and the official LIBERO multi-task `eval50` protocol
(`2,000` episodes total). Every run uses seed `1000`.

See [RESULTS.md](RESULTS.md) for scenario- and suite-level results, parameter
counts, and evaluation details.

## Quick Start

```bash
git clone https://github.com/lsh3163/prism.git
cd prism

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"

python -m unittest discover -s tests -v
```

## Minimal Usage

```python
import torch

from prism_robot import PRISMConditioner

conditioner = PRISMConditioner(
    input_dim=32,
    output_dim=1152,
    hidden_dim=1152,
    degree=2,
    interaction_mode="gated",
    gate_init=1e-2,
    post_mlp_layers=2,
    use_rmsnorm=True,
)

proprioception = torch.randn(8, 32)
conditioned_state = conditioner(proprioception)
assert conditioned_state.shape == (8, 1152)
```

Use `conditioner.polynomial_features(proprioception)` to inspect the learned
polynomial features before the downstream projection.

## Backbone Integrations

The release provides small patches against supported upstream revisions
instead of vendoring either backbone.

| Backbone | PRISM changes | Kept unchanged |
|---|---|---|
| [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) | Deployable `history_actor` representation | Actor core, actions, simulator, and objective |
| [LeRobot / SmolVLA](https://github.com/huggingface/lerobot) | Proprioceptive `state_proj` branch | VLM, visual path, and action-expert interface |

The BFM-Zero patch leaves the actor and simulator interface unchanged. The
SmolVLA patch replaces only the proprioceptive projection; the pretrained VLM,
visual encoder, and action-expert interfaces remain intact.

Follow [integrations/README.md](integrations/README.md) for exact patch
commands and supported upstream revisions.

## Reproduce BFM-Zero

First complete the upstream
[BFM-Zero installation](https://github.com/LeCAR-Lab/BFM-Zero), including its
Isaac Sim environment and LAFAN motion data. Check out the supported revision
and apply the training and evaluation patches using
[integrations/README.md](integrations/README.md).

From the patched BFM-Zero checkout, launch the paper-aligned PRISM run with:

```bash
export PRISM_ROOT=/absolute/path/to/prism

set -a
source "$PRISM_ROOT/configs/bfm_zero_prism.env"
set +a

uv run python -m humanoidverse.train
```

This recipe uses seed `1000`, `512` parallel training environments, and `9.6M`
environment steps. Evaluate an aligned checkpoint on all LAFAN motions with
the released evaluator:

```bash
uv run python -m humanoidverse.tracking_eval \
  --model-folder=/path/to/bfm-zero-run \
  --data-path=/path/to/lafan_29dof.pkl \
  --num-envs=128 \
  --output-subdir=tracking_eval_nominal \
  --eval-log-name=humanoidverse_tracking_eval_nominal
```

The low-friction and payload-mass overrides used for the paper are listed in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#bfm-zero), together with the baseline
and larger-capacity control settings.

## Reproduce SmolVLA on LIBERO

Install [LeRobot](https://github.com/huggingface/lerobot) with its LIBERO
dependencies, then apply the SmolVLA patch using
[integrations/README.md](integrations/README.md). The reported experiment is
one multi-task policy trained jointly on Spatial, Object, Goal, and Long:

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

Evaluate the aligned `80K` checkpoint with the official `eval50` protocol:

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

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full optimizer,
checkpoint, hardware, dataset, scenario, and capacity-control settings.

## Repository Layout

```text
prism/
|-- src/prism_robot/       # Standalone PyTorch implementation
|-- tests/                 # Unit tests for degree, gradients, and RMSNorm
|-- integrations/          # BFM-Zero and SmolVLA source patches
|-- configs/               # Paper-aligned PRISM settings
|-- analysis/              # Representation-analysis utilities
|-- RESULTS.md             # Aligned quantitative results
`-- REPRODUCIBILITY.md     # Training and evaluation protocols
```

Checkpoints, datasets, simulator assets, and complete upstream repositories are
not redistributed here. Their installation and usage remain governed by the
corresponding upstream projects.

The representation analysis is documented in
[analysis/README.md](analysis/README.md), and the aligned quantitative
comparisons are recorded in [RESULTS.md](RESULTS.md).

## Citation

The archival identifier will be added after the arXiv release. Until then,
please cite:

```bibtex
@misc{lee2026prism,
  title  = {PRISM: Polynomial Representations for Interaction-Structured Motor Control},
  author = {Lee, Seung Hyun and Yu, Stella X.},
  year   = {2026},
  url    = {https://lsh3163.github.io/prism/}
}
```

## Licensing

A top-level license for the standalone PRISM implementation is being
finalized. The integration patches remain subject to their respective
BFM-Zero and LeRobot upstream terms. See [NOTICE.md](NOTICE.md) before
redistribution.
