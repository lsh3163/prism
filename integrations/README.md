# Backbone Integrations

The patches in this directory reproduce the exact source-level integration
used in the stronger-backbone experiments. Apply each patch only to its pinned
upstream commit.

## BFM-Zero

```bash
git clone https://github.com/LeCAR-Lab/BFM-Zero.git
cd BFM-Zero
git checkout b87916f52d3d9e6eeba484f5e80851a235191837
git apply --check /path/to/prism/integrations/bfm-zero.patch
git apply /path/to/prism/integrations/bfm-zero.patch
```

The patch adds a PRISM filter for the deployable `history_actor` stream and
environment-variable configuration for matched baseline, larger-capacity, and
PRISM runs.

For the scenario evaluator and representation export used by the paper:

```bash
git apply --check /path/to/prism/integrations/bfm-zero-evaluation.patch
git apply /path/to/prism/integrations/bfm-zero-evaluation.patch
cp /path/to/prism/integrations/bfm-zero/tracking_eval.py humanoidverse/tracking_eval.py
```

The evaluation patch is optional for training. It adds scenario controls,
consistent CUDA checkpoint loading, tracking success export, and
`actor_representation.npz` generation for the t-SNE analysis.

## LeRobot / SmolVLA

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 2d7a42011a4f8e05a8c85d5fb908da258d4cc7b1
git apply --check /path/to/prism/integrations/lerobot-smolvla.patch
git apply /path/to/prism/integrations/lerobot-smolvla.patch
```

The patch replaces only SmolVLA's proprioceptive `state_proj` when
`state_conditioner_type=prism`. Visual-language and action-expert interfaces
remain unchanged.

## Upstream Terms

BFM-Zero and LeRobot retain their original licenses. See
[../NOTICE.md](../NOTICE.md) before redistributing either patch.
