# Learned gates across PRISM backbones

All default PRISM training recipes use `u * (1 + alpha * v)`, where `u` and
`v` are learned affine features of the selected state/history. `alpha` is a
trainable vector initialized to 0.01, with one value per latent feature. For
degree K, multiply the first affine factor successively by K−1 gated factors.
Degree 1 is the first-order control and contains no gate.

The gate learns from the existing policy loss. Its saved value remains fixed
during evaluation while the affine features vary with observations. It is an
unconstrained scale, so values can become negative or exceed one. A zero gate
removes that interaction term; the remaining projection and normalization still
belong to the PRISM architecture.

## Actor versions

| Backbone | Default new-training actor | Historical compatibility |
|---|---|---|
| G1 PPO | `g1_gated_poly_v2`: affine factors, learned gates, linear projection and ELU | `g1_residual_poly_v1`: residual branch, scalar warmup or fixed scale; use `legacy-prism` or `legacy-degree1/2/3`. |
| Diffusion | `diffusion_gated_state_v2`: normalized state history, learned gated product, latent normalization and SiLU projection | `diffusion_factorized_state_v1`: ungated product; use `legacy-prism` for historical training settings. Evaluation reads the saved mode. |
| SmolVLA | `smolvla_gated_quadratic_v1`: gated state features, SiLU projection, external RMSNorm | Released gated computation is unchanged. Earlier vanilla-product checkpoints retain their own saved configuration. |
| BFM-Zero | `bfm_history_gated_quadratic_v1`: gated history features, Mish projection and RMSNorm | Released gated computation is unchanged. Earlier all-observation actors use a separate schema. |

The [actor contract](ACTOR_CONTRACT.md) and
[registry](configs/actor_contracts.json) specify dimensions, preprocessing,
parameter keys, and defaults. All default PRISM recipes explicitly select the
gated mode. Low-level LeRobot configuration defaults retain missing-field
semantics for old saved files; use the released runners or the explicit flags in
the [real-demo guide](REAL_DEMO.md) for new training.

## Existing checkpoints and results

Train new G1 and Diffusion models for the gated formulation. Their old actors
have different computation graphs and checkpoint schemas. Loading old weights
with a newly initialized gate does not establish equivalence or preserve the
old model's measured performance. Loaders reject incompatible gate schemas.

G1's gated actor also removes the old residual branch and scalar warmup. Its
new degree-2 actor has a different parameter count; use the current larger-MLP
control when comparing capacity. The degree-1/2/3 recipes now vary degree within
the same gated architecture. The historical larger control remains available
as `legacy-larger`.

For new G1/Diffusion results, rerun nominal evaluation, perturbation evaluation,
degree/capacity comparisons, and checkpoint-specific representation probes.
Reuse the same trained checkpoint within each nominal/perturbed comparison.
Baseline and MCC methods retain their baseline policies. MCC remains a separate
action correction and is disabled for PRISM.

The archived benchmark numbers and trained-checkpoint simulator checks are
historical evidence. They are not results for the new gated G1/Diffusion actors.
The SmolVLA/BFM formula is preserved, but that alone does not resolve their
historical checkpoint or independent-training-seed provenance gaps.

## New training and deployment

1. Install each integration at its pinned upstream revision.
2. Use the default `prism` recipe with an explicit training seed and a fresh
   output directory. Use matched data, batch size, training budget, and evaluation
   settings for the baseline comparison.
3. Save the resolved configuration, actor version, source revision, optimizer
   state, and checkpoint identity. Confirm learned gate tensors are present.
4. Evaluate the saved policy and preprocessing without overriding its actor mode.
5. Use the same trained representation in a real demonstration, adapting state
   and action interfaces through the [real-demo workflow](REAL_DEMO.md).

This source update enables the shared gated algorithm. It does not include new
converged training runs, benchmark averages, or hardware experiments.
Completed optimizer, serialization, and short training checks are recorded in
[validation/GATED_REPORT.md](validation/GATED_REPORT.md).
