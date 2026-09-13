# Simulation analysis

These utilities expose the local paper analysis with configurable input paths.
They do not contain checkpoints, rollout datasets, or precomputed paper results.
See [the experiment inventory](../../EXPERIMENTS.md) before selecting a protocol;
several supplied-paper statements differ from the historical code.

Install this repository's `analysis` extra in an environment with PyTorch and
`safetensors`. Commands below run from the PRISM repository root. LeRobot is
needed to **collect** LIBERO traces, but these offline utilities read saved files
without importing LeRobot or a simulator.

## Contact-response probes — Tables 4 and 16

Copy [contact_probe_specs.example.json](contact_probe_specs.example.json) and
provide one entry per task. Each entry identifies an exact baseline checkpoint,
an exact `poly_kernel_conditioner` checkpoint, and a single-task `eval_info.json`
containing `state_traces` and `contact_force_traces`. Relative paths resolve from
the JSON file's directory. Checkpoints must include `model.safetensors` and
the saved `policy_preprocessor.json` with its referenced normalization statistics,
and `config.json` from the main Diffusion Policy integration. A SmolVLA or
residual-compliance checkpoint is not compatible with this representation extractor.

```bash
uv run python analysis/paper/make_contact_response_probe_table.py \
  --specs /path/to/contact_probe_specs.json \
  --baseline-features state-first-contribution \
  --ridge 0 \
  --output-json /path/to/results/contact_probe.json \
  --output-tex /path/to/results/contact_probe.tex
```

**Baseline feature interpretation must be chosen explicitly.**
`state-first-contribution` uses the released model's ordering: timestep embedding,
flattened state history, then image/environment history. It reads dimensions from
the saved config, validates the conditioning weight shape, and computes
`W_state @ Mish(normalized_state) + bias`. This is the state-dependent FiLM
projection contribution plus bias, **not a full U-Net embedding**: image and
timestep contributions are absent. Each policy's own saved preprocessor supplies
MIN_MAX, MEAN_STD, or IDENTITY state normalization. Config and preprocessor
normalization must agree. Use this mode only with source-verified state-first
checkpoints: dimension checks cannot distinguish feature ordering. Upstream
interleaved state/image history is not handled by this mode.

`--baseline-features historical-surrogate-audit` preserves the old analysis
solely to inspect historical numbers. It selects the final state-width columns
of the conditioning weight and computes `SiLU(W_tail @ mean_std_state + bias)`.
In the released model those columns belong to image features, and the actual
conditioning module is Mish **before** its linear layer. The archived policies
also specify MIN_MAX state normalization, whereas the old probe uses MEAN_STD
for both baseline and PRISM. This surrogate is not a certified frozen baseline
embedding, and its PRISM features need not match deployment preprocessing.
Historical Tables 4/16 therefore do not certify a fair representation comparison;
corrected feature extraction requires newly measured and separately labeled results.
JSON metadata and table labels distinguish both modes.

The window/fitting kernel is preserved: two-step state, PRISM affine factors and
their elementwise product, five-step windows containing force above 1 N,
seed-0 random window splitting
(80%/20%), at most 12,000 samples, and train-set feature/target standardization.
The force window is `[t, t+5)`, including the current force. The impulse proxy is
`log1p(sum(force))` without a timestep multiplier; work is
`sum(force * step_displacement)`. These are the historical analysis targets.

`--ridge 0` selects the original normal-equation OLS calculation. The historical
default is `10`; any positive ridge is a different probe protocol. OLS can fail
or become unstable for singular or nearly collinear representations. Preserve
that failure when auditing historical results; a different solver or a grouped
episode split requires a separately labeled experiment. Random windows from the
same episode can occur in both train and test sets. No grouped holdout or causal
future-only claim is implied by this utility.

The Table 4 contact values match a historical ten-task Spatial output. Table 16
matches a later four-suite output (Spatial:1, Object:3, Goal:7, Long:1). Their
input windows and source revision must be recorded independently. The old JSON
for Table 16 did not record the ridge value, so its exact solver setting remains
unconfirmed even though the current source defaults to 10.

## Latent-factor ablation — Table 5

```bash
uv run python analysis/paper/ablate_locomotion_factors.py \
  --checkpoint /path/to/model_3001.pt \
  --observations /path/to/humanoidgym_rollout_actor_observations.pt \
  --factors 159 194 19 154 \
  --output /path/to/results/factor_ablation.json
```

This utility detects the versioned gated or historical residual degree-2 G1
checkpoint and loads it strictly. For the gated actor it temporarily sets one
learned alpha to zero, preserving the first-order term and restoring the gate
after measurement. For the historical actor it zeroes one channel of the second
affine factor while retaining the raw branch and first polynomial term.
Both the historical mean action-vector L2 distance and coordinate-wise
MAE are reported after the 0.25-radian joint-target scale. These metrics differ.
The four example factor IDs belong to the historical seed-2 checkpoint; provide
explicit IDs for each new checkpoint. They are not universal important factors
for independently trained models. Gated output also records the saved alpha.

Input names are descriptive interpretations of weights. The historical naming
helper excluded the left coordinate when choosing the right coordinate. The
new output includes that historical name alongside each branch's independent
largest-weight coordinate, so diagonal interactions are visible.

The four historical Table 5 action distances were verified with the source
checkpoint and its 512 recorded observations using this canonical actor. This
checks the offline ablation; it is not a new simulator benchmark or five-seed
reproduction.

## Contact behavior — Figures 1 and 4

Collect traces with the [LeRobot evaluation integration](../../integrations/lerobot/README.md).
Use paired initial states and a declared episode choice for representative plots.
The plotter does not verify initial-state pairing automatically.

```bash
uv run python analysis/paper/plot_contact_response.py \
  --run 'Diffusion=/path/to/diffusion/eval_info.json' \
  --run 'PRISM=/path/to/prism/eval_info.json' \
  --episode 0 --dt 0.05 \
  --output /path/to/results/contact_response.pdf

uv run python analysis/paper/aggregate_compliance_mechanism.py \
  --run 'Diffusion=/path/to/task1/diffusion/eval_info.json' \
  --run 'PRISM=/path/to/task1/prism/eval_info.json' \
  --dt 0.05 \
  --json-output /path/to/results/contact_summary.json \
  --tex-output /path/to/results/contact_summary.tex
```

Set `--dt` to the **simulator control timestep** for the actual trace. Rendered
video FPS is not the control frequency. The aggregate utility preserves the
historical 5 N threshold and eight-step pre/post-contact windows. It reports
all-rollout success counts and contact diagnostics conditioned on successful
episodes with qualifying contact windows. Multiple `--run` entries with the same
label combine tasks. Those combined episodes do not become independent training
seeds.

## Perturbation and MCC validation aggregation

```bash
uv run python analysis/paper/summarize_robustness.py \
  --root /path/to/robustness \
  --json-output /path/to/results/robustness.json \
  --tex-output /path/to/results/robustness.tex

uv run python analysis/paper/summarize_mcc_sweep.py \
  --root /path/to/mcc_validation --configs /path/to/configs.txt \
  --json-output /path/to/results/mcc_validation.json \
  --tex-output /path/to/results/mcc_validation.tex
```

The robustness folder layout is
`{clean,action_delay,proprio_noise,image_corrupt,combined}/{suite}_task_{id}/{diffusion,diffusion_mcc_sensorless,prism}/eval_info.json`.
Missing conditions remain empty (`n=0`); they are not zero-success observations.
Inspect sample counts before treating a table as complete.

MCC configs are whitespace-separated rows:
`config_id gain delay ema noise clip`. The expected layout is
`config_id/{suite}_task_{id}/diffusion_mcc_sensorless/eval_info.json`. Selection
uses validation success first, successful contact impulse second, and peak force
third; do not tune on the reporting test split. The preserved MCC summarizer
assumes a 0.05-second control timestep.

## Independent training-seed statistics

Prepare a JSON list with one record per independently trained policy, after
aggregating that policy's tasks and episodes according to the benchmark:

```json
[
  {
    "method": "PRISM",
    "training_seed": 1000,
    "evaluation_seed": 0,
    "checkpoint": "immutable-checkpoint-path-or-hash",
    "metrics_path": "seed1000/eval_info.json",
    "metric_path": ["overall", "pc_success"]
  }
]
```

```bash
uv run python analysis/paper/aggregate_training_seeds.py \
  --manifest /path/to/seeds.json --expected-training-seeds 1 \
  --output /path/to/results/seed_summary.json
```

Supply all five verified records and set `--expected-training-seeds 5` for a
five-seed report, or three for the rebuttal's SmolVLA report. Use one metric and
one benchmark/checkpoint horizon per manifest. The example is a schema, not
evidence of a run. The tool rejects missing replicates, repeated seed labels,
and reused checkpoint identifiers; it cannot independently prove that different
identifiers correspond to different training runs. Mean is unweighted across
training seeds, and standard deviation uses `ddof=1` (undefined for one seed).

[source_manifest.json](source_manifest.json) records the original local scripts
and their hashes. Historical residual-compliance interaction-group ablations
used different 300/2,000-step policies; they are deliberately not substituted
for the main 20K proprioceptive-conditioner experiment.
