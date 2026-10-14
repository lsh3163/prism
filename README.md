# PRISM

**Polynomial Representations for Interaction-Structured Motor Control**

Seung Hyun Lee · Stella X. Yu · University of Michigan

[Paper](https://arxiv.org/abs/2607.23473) · [Project page](https://lsh3163.github.io/prism/) · [Results](https://github.com/lsh3163/prism/blob/main/RESULTS.md)

PRISM replaces an affine policy input layer with learned polynomial interactions.
It trains through the policy's existing objective. The core needs only PyTorch.

## Install

```bash
pip install git+https://github.com/lsh3163/prism.git
```

Python 3.10+ and PyTorch 2.1+. For a source checkout, use `pip install -e .`.

## Use

```python
import torch
from torch import nn
from prism_robot import PRISM

policy = nn.Sequential(
    PRISM(32, 256),       # (..., 32) -> (..., 256)
    nn.ELU(),
    nn.Linear(256, 12),
)
actions = policy(torch.randn(8, 32))
```

The default layer implements the camera-ready paper's quadratic representation:

```text
u = W1 x + b1
v = W2 x + b2
psi = u + s2 * (u * v)
```

`s2` is a learned scalar initialized to 0.01. `degree=1` gives an affine layer;
`degree=3` adds a third affine factor and scalar interaction stage. Projection,
activation and normalization follow the polynomial layer and are chosen by the
policy. Continue using the original observation preprocessing and action interface.

Run the CPU example from a checkout:

```bash
python examples/policy.py
```

## Policy integrations

[Adapter guide](docs/ADAPTERS.md): ACT and native mjlab camera-ready modules,
plus the existing Humanoid-Gym, SmolVLA and BFM-Zero integrations. Each backend
uses its own environment. Simulator dependencies are not installed with PRISM.

The existing `PRISMConditioner` API and historical checkpoint loaders remain
available. They retain their original parameter layouts; use the corresponding
adapter when loading a trained policy. The new `PRISM` layer is the small,
backbone-independent polynomial feature map, not a checkpoint converter.

## Development

```bash
pip install -e '.[test,dev]'
uv run --extra test --extra dev python -m unittest discover -s tests -v
```

See [CONTRIBUTING.md](https://github.com/lsh3163/prism/blob/main/CONTRIBUTING.md). Wheels and source distributions contain
the core, its example, documentation and core tests. Full experiment tools stay
in this Git repository. [Earlier experiment documentation](https://github.com/lsh3163/prism/blob/main/docs/legacy/README.md)
is retained separately from the camera-ready entry point.

## Citation

```bibtex
@article{lee2026prism,
  title = {PRISM: Polynomial Representations for Interaction-Structured Motor Control},
  author = {Lee, Seung Hyun and Yu, Stella X.},
  journal = {arXiv preprint arXiv:2607.23473},
  year = {2026}
}
```

## License

MIT for the original PRISM implementation. Derived integrations retain their
upstream terms; see [NOTICE.md](NOTICE.md).
