# Actor and checkpoint contracts

Select the representation by its stable ID in
[configs/actor_contracts.json](configs/actor_contracts.json). The adapters preserve
the computation and state-dictionary layout used by their respective backbones.
A shared PRISM name does not make checkpoints interchangeable.

## Representation variants

`A_i`, `P`, `R`, and `O` below denote learned affine maps. Products are elementwise.
The formulas describe the representation stage, followed by the policy backbone.

| Variant ID | Computation | Source and checkpoint interface |
|---|---|---|
| `g1_residual_poly_v1` | `h=P(x)`; `p_1=A_0(h)`; `p_k=p_(k-1)+A_(k-1)(h)*S_(k-2)(p_(k-1))`; `z=ELU(R(x)+s*O(p_K))` | [G1 actor](integrations/humanoid-gym/actor.py): preserve encoder, actor MLP, `poly_scale`, and `actor_poly_warmup_step`. |
| `diffusion_factorized_state_v1` | `u=LayerNorm(x)`; `p=A_1(u)*A_2(u)`; latent LayerNorm then SiLU MLP | [Diffusion conditioner](integrations/lerobot/diffusion_conditioner.py): preserve `input_norm`, `left_proj`, `right_proj`, and `net` keys. |
| `smolvla_gated_quadratic_v1` | `p=A_1(x)*(1+alpha*A_2(x))`; SiLU MLP then output RMSNorm | [SmolVLA patch](integrations/lerobot-smolvla.patch): preserve `left_proj`, `right_proj`, `quadratic_scale`, and external `state_output_norm`. |
| `bfm_history_gated_quadratic_v1` | Same gated product; Mish MLP and RMSNorm on `history_actor` | [BFM-Zero patch](integrations/bfm-zero.patch): keep other actor inputs and their concatenation unchanged. |
| `core_gated_v1`, `core_factorized_v1` | Configurable degree-K interactions and output projection | [Shared API](src/prism_robot/conditioner.py): `factors`, `interaction_scales`, `post_mlp`, `output_norm`; explicit weight mapping is needed for historical adapters. |

G1 uses a scheduled scalar scale; the stronger-backbone gates are learned
per-feature parameters. Normalization and nonlinear activations make the full
representation non-polynomial in raw state. Degree refers to the interaction
stage.

## Interfaces to preserve

- **G1:** 705 actor inputs (15×47), 219 privileged critic inputs (3×73),
  12 joint-target actions, and a 256-wide residual encoder. The `prism` recipe
  uses degree 2 and 500 PPO updates of scale warmup. Fixed degree-1/2/3 recipes
  use no warmup. Keep privileged observations confined to the critic.
- **Diffusion:** two observations with eight state coordinates each; the main
  conditioner maps 16 state-history values through 256 latent factors back to
  16 values. Flatten state, image, and optional environment histories separately,
  then concatenate those blocks. Upstream interleaved histories are a different
  layout even when dimensions match. Load the saved preprocessor; inspected
  main configurations use MIN_MAX state normalization.
- **SmolVLA/BFM-Zero:** preserve each patch's projection activation, RMSNorm
  location and dtype arithmetic. Older `use_poly_interaction` SmolVLA models
  and the BFM development `PrismActor` over all observations are distinct schemas.

## Changing or loading a model

1. Record the variant ID, dimensions, input order, normalization, action units,
   activation, and gate/warmup configuration with each checkpoint.
2. Load state dictionaries strictly. Reject unknown schemas and missing buffers.
3. Compare fixed-input outputs and gradients with the source implementation;
   verify save/load and warmup state before using a refactored actor.
4. Assign a new variant ID when the computation changes. Update the registry,
   adapter documentation, and relevant tests together.

The CPU suite checks float32 outputs and input gradients against the released
BFM-Zero and SmolVLA class definitions with mapped weights. These checks do not
establish mixed-precision or full-policy equivalence. Optional G1 checks compare
against the original research class and a trained checkpoint.

For state-only contact probes, use the explicitly selected, config-validated
state contribution mode described in [the analysis guide](analysis/paper/README.md).
The historical tail-column/SiLU surrogate and mean/std preprocessing are not
verified baseline embeddings; they remain available only as an audit mode.
