# PRISM: Polynomial Representations for Interaction-Structured Motor Control

PRISM is a compact, learnable polynomial representation for robot policies. It
exposes interactions among deployment-available proprioceptive variables
without adding force sensing, tactile input, contact labels, privileged
physical parameters, or a new low-level controller.

This repository contains:

- a standalone PyTorch implementation of the PRISM conditioner;
- patches for the exact BFM-Zero and SmolVLA integration points used in our
  stronger-backbone experiments;
- paper-aligned training configurations and evaluation details; and
- scripts for reproducing the representation analysis.

Project page: <https://lsh3163.github.io/prism/>

## Method

For an input \(x\), the paper-facing degree-2 model computes

\[
h_1 = W_1x+b_1,\qquad
h_2 = h_1 \odot \left(1+\alpha_2\odot(W_2x+b_2)\right),
\]

followed by a learned projection and, in the stronger-backbone experiments,
RMSNorm. The per-feature scale \(\alpha_2\) is initialized to \(10^{-2}\) and
optimized end-to-end. This starts the representation close to a linear map
while allowing each latent feature to learn how strongly it uses quadratic
interactions.

The standalone module supports degree \(K\):

\[
h_k=h_{k-1}\odot\left(1+\alpha_k\odot(W_kx+b_k)\right),
\quad k=2,\ldots,K.
\]

The resulting representation has degree at most \(K\). The reported BFM-Zero
and SmolVLA experiments use `degree=2`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
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
    post_mlp_layers=2,
    gate_init=1e-2,
    use_rmsnorm=True,
)

proprioception = torch.randn(8, 32)
conditioned_state = conditioner(proprioception)
```

`conditioner.polynomial_features(x)` returns the latent polynomial
representation before the downstream projection.

## Backbone Integrations

The integration patches are intentionally based on pinned upstream commits
instead of vendoring either project:

| Backbone | Upstream commit | PRISM insertion point |
|---|---|---|
| BFM-Zero | `b87916f52d3d9e6eeba484f5e80851a235191837` | deployable `history_actor` branch |
| LeRobot / SmolVLA | `2d7a42011a4f8e05a8c85d5fb908da258d4cc7b1` | proprioceptive `state_proj` branch |

See [integrations/README.md](integrations/README.md) for patch commands and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the paper-aligned settings.

## Results

The aligned stronger-backbone results are summarized in
[RESULTS.md](RESULTS.md). Checkpoints and datasets are not included in this
source release because they are governed by their respective upstream
projects.

## Citation

The archival paper entry will be added after the arXiv update. Until then,
please cite the project title and authors:

> Seung Hyun Lee and Stella X. Yu. *PRISM: Polynomial Representations for
> Interaction-Structured Motor Control*. 2026.

## Licensing

The standalone PRISM source and the integration patches have different
licensing considerations. Read [NOTICE.md](NOTICE.md) before redistribution.
