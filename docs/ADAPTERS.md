# Policy adapters

The installed package contains the polynomial feature layer. Clone this repository
for optional adapters. Keep simulator and robot environments separate from the core.

| Policy | Code | Scope |
|---|---|---|
| ACT / ManiSkill3 | [ACT conditioner](https://github.com/lsh3163/prism/blob/main/integrations/act/conditioner.py) | Camera-ready affine baseline and scalar quadratic conditioner; no scheduler or simulator import. |
| Native mjlab G1 | [mjlab actor](https://github.com/lsh3163/prism/blob/main/integrations/mjlab/actors_mjlab_shared_rms.py) | Original 286-input / 29-action actor, 99-coordinate polynomial branch and 187-coordinate height bypass. |
| Humanoid-Gym | [Guide](https://github.com/lsh3163/prism/blob/main/integrations/humanoid-gym/README.md) | Existing training/evaluation tools and explicitly versioned historical actors. |
| SmolVLA | [Guide](https://github.com/lsh3163/prism/blob/main/integrations/lerobot/README.md) | Existing pinned LeRobot patch and runners. |
| BFM-Zero | [Guide](https://github.com/lsh3163/prism/blob/main/integrations/bfm-zero/README.md) | Existing pinned actor/history patch and tracking evaluation. |

## ACT

Use a PyTorch environment providing `nn.RMSNorm`. From the repository root:

```python
from integrations.act.conditioner import PrismConditioner

# native_projection is the ACT model's existing nn.Linear proprioceptive projection.
# Create it before the optimizer and before moving the complete model to its device.
model.input_proj_robot_state = PrismConditioner(model.input_proj_robot_state)
```

For the StackCube model, two 9-joint observations form 18 state inputs and produce
256 features. The conditioner copies the native affine path, initializes the
output projection to identity, and learns scalar `alpha=0.01`. It applies the
native output RMSNorm, with epsilon 1e-6, after the polynomial and projection.
`alpha=0` recovers the matched baseline conditioner exactly. Other observation
widths follow the supplied affine projection's dimensions.

Names `u`, `v`, `projection`, `alpha` and `output_norm` are preserved for strict
loading of the original conditioner state dictionary. The full ACT wrapper,
visual encoder, expert-target masking, normalizers, action chunking and registered
recovery protocol belong to the task pipeline; this module does not substitute
for that pipeline or supply its data/checkpoints.

## Native mjlab

For pretrained PRISM and MLP actors with an interactive browser viewer, use the
[G1 play example](../examples/g1/README.md). Actor-only weights and their trained
normalization are included in `checkpoints/g1`; the example requires no private
experiment paths or training files.

The module requires upstream `rsl_rl.models.MLPModel` and `rsl_rl.modules.MLP`.
The recorded mjlab source revision is
[`033ae22a`](https://github.com/mujocolab/mjlab/tree/033ae22a2c7a30a25a6fa77b16c113ed88dd1b55).
Use the experiment's compatible mjlab/RSL-RL environment, then expose the adapter:

```bash
export PYTHONPATH="$PWD/integrations/mjlab${PYTHONPATH:+:$PYTHONPATH}"
```

Its registered actor class remains
`actors_mjlab_shared_rms:NativeSharedRmsActor`. Existing `standard`, `matched`,
`wider`, `deeper` and `prism` configurations and state-dictionary keys are retained.
Actor observations are 286-dimensional, critic observations 298-dimensional,
and actions 29-dimensional. The polynomial affects only coordinates 0:99;
height coordinates 99:286 bypass it. Shared FP64-moment RMS normalization remains
after the final hidden ELU and before the action layer.

The recorded mjlab actor uses a **per-feature** `alpha` vector. Preserve that
shape when loading its checkpoint. The standalone `PRISM` API follows the paper's
**scalar** coefficients; replacing one by the other changes the trained model.

The module and play example supply trained-policy inference, not the complete
rough-terrain training or 1,000-pair evaluation launcher. Original checkpoint maps, resets,
terrain geometry, normalization statistics and evaluation horizons must accompany
benchmark reproduction. The play example is a qualitative simulation with a smaller
grid and its own deterministic reset, separate from the paper's aggregate evaluation.

## Existing APIs and source records

`PRISMConditioner` retains its original configurable projection and per-feature
scales. Use `PRISM` for new, minimal polynomial-layer integrations. Neither API
silently loads a backbone checkpoint into a different representation.

[Camera-ready source records](camera-ready-sources.json) identify the manuscript
and original modules used for the extraction. CPU source-parity validation can be
run with `validation/check_camera_ready_parity.py`; full simulator/hardware checks
remain separate. Historical recipes are indexed in [legacy/README.md](https://github.com/lsh3163/prism/blob/main/docs/legacy/README.md).
