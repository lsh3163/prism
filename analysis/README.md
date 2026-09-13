# Representation Analysis

`plot_scenario_tsne.py` reproduces the paper's per-method,
scenario-conditioned representation visualization.

Each input is an `actor_representation.npz` exported by the optional
BFM-Zero evaluation patch. The archive must contain the requested feature
array, which defaults to `history_conditioned`.

```bash
uv run --extra analysis python analysis/plot_scenario_tsne.py \
  --input "BFM-Zero:Nominal=/path/baseline/nominal/actor_representation.npz" \
  --input "BFM-Zero:Low friction=/path/baseline/low_friction/actor_representation.npz" \
  --input "BFM-Zero:Payload mass=/path/baseline/payload/actor_representation.npz" \
  --input "PRISM:Nominal=/path/prism/nominal/actor_representation.npz" \
  --input "PRISM:Low friction=/path/prism/low_friction/actor_representation.npz" \
  --input "PRISM:Payload mass=/path/prism/payload/actor_representation.npz" \
  --output bfm_scenario_tsne.pdf
```

Each method is embedded independently so the plot diagnoses how its own actor
representation organizes the three dynamics conditions. t-SNE is qualitative;
the paper reports quantitative neighborhood and classification diagnostics
separately.

See [the simulation analysis tools](paper/README.md) for contact diagnostics,
frozen-feature probes, factor ablations, and aggregation across training seeds.
