# Backbone Integrations

The default simulation recipes use learned per-feature gated interactions across
G1, Diffusion, SmolVLA, and BFM-Zero. The original G1 and Diffusion computations
remain available through legacy recipes. Read [../ACTOR_CONTRACT.md](../ACTOR_CONTRACT.md) before
substituting representations or loading historical checkpoints.

| Integration | Main source and instructions |
|---|---|
| Humanoid-Gym-style Unitree G1 | [humanoid-gym/README.md](humanoid-gym/README.md): gated actor, historical residual actor, task/environment source, train/eval and probes; separate Python 3.8 simulator environment |
| LeRobot Diffusion and SmolVLA | [lerobot/README.md](lerobot/README.md): pinned extraction, task recipes, historical checkpoints, simulation metrics and robustness |
| BFM-Zero | Existing patches and evaluator below; representation analysis in [../analysis/README.md](../analysis/README.md) |
| Paper analysis | [../analysis/paper/README.md](../analysis/paper/README.md): contact probes, factor ablation, and metric aggregation |
| New real robot demonstration | [../REAL_DEMO.md](../REAL_DEMO.md): record, train, and deploy the same representation |

The stronger-backbone patches preserve their released gated computations and
add checkpoint schema checks. Apply each patch only to its pinned upstream commit.
Use the named training recipes to select the current gated actor explicitly;
missing fields in historical saved configurations retain legacy interpretation.

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
