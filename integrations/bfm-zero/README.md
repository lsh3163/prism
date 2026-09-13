# BFM-Zero gated PRISM

Apply the [training patch](../bfm-zero.patch) to BFM-Zero revision
`b87916f52d3d9e6eeba484f5e80851a235191837` using the
[installation instructions](../README.md#bfm-zero). The filter transforms only
`history_actor` before the existing observation concatenation.

## New training runs

```bash
set -a
source /path/to/prism/configs/bfm_zero_prism.env
set +a
uv run python -m humanoidverse.train
```

The recipe explicitly selects `gated_quadratic`, a learned alpha for each hidden
feature initialized to 0.01, a two-layer Mish projection, and output RMSNorm:

```text
left = left_proj(history_actor)
right = right_proj(history_actor)
features = left * (1 + quadratic_scale * right)
output = output_norm(post_mlp(features))
```

`quadratic_scale` is an optimizer parameter with shape `(hidden_dim,)`; it is
not a scheduled scalar. The first-order path remains when alpha is zero.
The right affine bias starts at zero. Observation normalization and the
surrounding actor/action interface retain their backbone behavior.

Setting `BFM_HISTORY_CONDITIONER_TYPE=prism` on a fresh training run also defaults
to this gated mode, gate init 0.01, and RMSNorm. The complete recipe keeps these
overrides explicit so that saved configurations identify the selected variant.
Use the [scenario launcher](evaluate_scenarios.py) and
[protocol guide](../../REPRODUCIBILITY.md#bfm-zero) for evaluation.

## Existing checkpoints

Serialized `HistoryActorPRISMFilterConfig` fields preserve their legacy defaults:
missing `product_mode` means `vanilla`, and missing `use_rmsnorm` means false.
Loading an old configuration therefore does not automatically enable a learned
gate or add normalization. For an explicitly requested legacy training run, set
`BFM_HISTORY_CONDITIONER_PRODUCT_MODE=vanilla` and
`BFM_HISTORY_CONDITIONER_USE_RMSNORM=false`.

Gated checkpoints must contain `quadratic_scale`; vanilla checkpoints must not.
The filter rejects either mismatch even when an upstream parent loader uses
`strict=False`. Checkpoint names and forward computations remain unchanged.
This gate guard does not turn upstream loading into full-model strict loading;
validate the remaining checkpoint configuration and keys separately.

[CPU tests](../../tests/test_gated_backbone_defaults.py) verify per-feature gate
initialization, a training update, strict representation save/load, parity with
the shared core after updating the gate, and rejection of gated/legacy schema
mismatches. These tests do not establish new simulator performance results.
