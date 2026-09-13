# LeRobot simulation integration

New Diffusion PRISM training defaults to a learned per-feature gate. The
release also preserves the original ungated Diffusion actor and recipes for
historical checkpoints, nominal/MCC/robustness evaluations, and the gated
SmolVLA integration. Hardware experiment source and results are excluded.

## Actor and checkpoint contracts

| Experiment | Representation | Preserved checkpoint names |
|---|---|---|
| Default Diffusion PRISM (`diffusion_gated_state_v2`) | LayerNorm on state history; learned gated affine product; latent LayerNorm; SiLU MLP | `diffusion.poly_kernel_conditioner.input_norm`, `left_proj`, `right_proj`, `quadratic_scale`, `net` |
| Legacy Diffusion PRISM (`diffusion_factorized_state_v1`) | LayerNorm on state history; ungated product of two affine factors; latent LayerNorm; SiLU MLP | `diffusion.poly_kernel_conditioner.input_norm`, `left_proj`, `right_proj`, `net` |
| Stronger SmolVLA PRISM | Degree-two gated product; SiLU MLP; external RMSNorm | `model.state_proj.left_proj`, `right_proj`, `quadratic_scale`, `post_mlp`; `model.state_output_norm` |

Both Diffusion recipes have two observations, eight state coordinates per
observation, 256 latent factors, and a 256-wide MLP. Its output replaces the
16-dimensional state history before concatenation with RGB features. The
diffusion objective, U-Net, and seven-dimensional action interface are retained.
The new `gated_quadratic` mode computes `left * (1 + alpha * right)` between
the two normalization stages. `alpha` is a 256-element `nn.Parameter`,
initialized to 0.01 and updated by the ordinary diffusion loss. The new mode
initializes the right affine bias to zero. It has no scheduled gate or
detached update. Setting alpha to zero retains the first affine path.

The legacy `latent_quadratic` mode keeps `left * right`, its original
initialization, and its exact checkpoint keys. Neither mode enumerates every
quadratic monomial. **Historical results belong to the legacy actor; the new
gated default requires training and evaluation before any performance claim.**

`diffusion_conditioner.py` preserves the legacy actor alongside the new gated
mode. Both retain the Diffusion-specific normalization and projection around
the product. They are distinct from
the standalone gated `prism_robot.PRISMConditioner`. The SmolVLA patch retains
its original module names and RMSNorm arithmetic. No checkpoint conversion is
performed. Missing or extra gate parameters cause a loading error, including
when LeRobot requests `strict=False`. See
[../../ACTOR_CONTRACT.md](../../ACTOR_CONTRACT.md).

The Diffusion patch preserves the reviewed experiment's feature ordering:
flatten each state/image/environment history separately, then concatenate
feature blocks. This also applies to the historical nominal checkpoint route.
Clean upstream LeRobot instead concatenates features per time step before
flattening; a new upstream checkpoint should not be assumed compatible with
the experiment's ordering. Historical training commits were not recorded, so
we cannot establish past run/source identity from checkpoint configs alone.

Inactive `poly_compliance_*` dataclass fields are retained only to parse the
saved main-experiment configs. `use_poly_compliance=true` is rejected. The
abandoned action-residual implementation and real-robot chunk blending are not
part of this release. Standalone-core aliases such as `prism_gated` are
rejected; the Diffusion mode is explicitly named `gated_quadratic`.

## Install in a separate pinned checkout

The combined integration targets LeRobot commit
`d656da8ccca5989ff0a2207e81fbfa2c2d5bafb1`. The original SmolVLA patch also
remains available on its historical `2d7a420...` route; its application to this
combined base has been checked.

```bash
export PRISM_ROOT=/absolute/path/to/prism
git clone https://github.com/huggingface/lerobot.git lerobot-prism
cd lerobot-prism
git checkout d656da8ccca5989ff0a2207e81fbfa2c2d5bafb1

# Validate source hashes and patch application without modifying files.
uv run --no-project python "$PRISM_ROOT/integrations/lerobot/install.py" "$PWD" --with-smolvla
# Install after the preflight succeeds.
uv run --no-project python "$PRISM_ROOT/integrations/lerobot/install.py" "$PWD" --with-smolvla --apply

uv sync --locked --extra training --extra evaluation --extra diffusion --extra libero --extra smolvla
```

Use Linux, Python 3.12 or newer, and the upstream LIBERO/MuJoCo installation
requirements. A working CUDA/EGL setup is needed for the released GPU commands.
Dataset downloads, model downloads, simulator installation, GPU training, and
full rollouts are separate from the CPU verification of this source release.

`install.py` checks the bundle hashes, upstream commit, and every affected file's
SHA-256 before writing. Repeating installation is a verified no-op, and
`--with-smolvla --apply` can add SmolVLA to an already installed Diffusion
checkout. Modified or partially patched target files are rejected. It applies
`diffusion_libero.patch`, optionally applies the existing
`../lerobot-smolvla.patch`, and copies the conditioner into LeRobot's Diffusion
package and the termination helper into its environment package. The patch includes the selected-episode sampler remap, current
`use_amp`/bf16 training behavior, incomplete-batch handling, LIBERO diagnostics,
MCC correction, observation perturbations, action delay, and trace export.
Use a fresh pinned checkout when moving from an earlier integration bundle;
the installer does not overwrite a different released source version.

If a Conda shell overrides the EGL vendor directory with Mesa-only libraries,
headless rendering can fail despite a working NVIDIA driver. On the validated
Linux machine, selecting the installed system NVIDIA vendor JSON for the
evaluation process fixed this without changing drivers or the global shell.
Pass the runner
`--egl-vendor-file /usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
Use this override only when that NVIDIA vendor file exists on your system.

## Main Diffusion training

Every runner prints commands by default. Add `--execute` to run them. Commands
use `uv run --no-sync python -m lerobot.scripts.lerobot_train` (or
`lerobot_eval`) inside `--lerobot-root`, with that checkout's `src` first in
`PYTHONPATH`. The runner checks its pinned Git revision and installed source
hashes before execution. Relative checkpoint and output paths resolve from
`--lerobot-root`; `--checkpoint-map` resolves from the calling directory.

To use an existing dependency environment, pass
`--python /absolute/path/to/environment/bin/python`; the command then uses
`uv run --no-project --python ...`. `--device` defaults to `cuda`. CLI help and
command previews require only Python's standard library; simulator and policy
dependencies are imported only by the executed LeRobot command.

Execution validates the entire planned sweep, including every required local
checkpoint, before starting a run or writing sweep files. Existing output
directories and command records are rejected. Choose a new output root to
repeat a run; these recipes do not resume or overwrite previous runs.

Each execution writes `RUN.command.json` beside its output directory. Its
versioned record includes the exact argument list, selected checkout and
environment overrides, source hashes, checkpoint/config/processor hashes,
actor identity, and completion status with timestamps and exit code. PRISM
uses the actor IDs in [the actor contract](../../ACTOR_CONTRACT.md); baseline
controls use `actor_variant: null` plus explicit policy/profile/architecture.
Checkpoint hashes identify inputs without asserting historical source identity
or deterministic reproduction. A failed command leaves its record for review
and stops the remaining sweep.

```bash
uv run --no-sync python "$PRISM_ROOT/integrations/lerobot/scripts/train_diffusion.py" \
  --lerobot-root "$PWD" --profile prism --suites libero_spatial --task-ids 0
```

Omit `--suites` and `--task-ids` to cover all 40 LIBERO tasks. The run directory
is `outputs/prism_diffusion/PROFILE/SUITE/task_ID`. Defaults are 20,000 steps,
training seed 0, two observations, horizon 16, eight executed actions, 100
diffusion training timesteps, ten inference steps, 128×128 images, U-Net widths
128/256/512, Adam learning rate 1e-4, and no image augmentation.

| Profile | Batch | Workers | Meaning |
|---|---:|---:|---|
| `prism` | 64 | 8 | New default learned-gate PRISM (`diffusion_gated_state_v2`) |
| `baseline` | 64 | 8 | New matched nominal control for gated PRISM |
| `legacy-prism` | 64 | 8 | Archived ungated recipe (`diffusion_factorized_state_v1`) |
| `legacy-baseline` | 8 | 4 | Archived nominal Diffusion recipe |

`matched-baseline` remains an alias of the new `baseline` settings;
`historical-baseline` remains an alias of `legacy-baseline`. Use `prism` and
`baseline` together for a new comparison. The low-level LeRobot dataclass
retains its ungated default for missing-field legacy config compatibility;
the new runner explicitly saves `poly_kernel_lift_mode=gated_quadratic`.

**The historical baseline and ungated PRISM runs did not use matched batch sizes.**
Neither the gated actor nor the new matched control inherits a historical result. The patch's bf16
and `drop_last` behavior matches the reviewed workspace, but the historical
training-code revision is unavailable. Do not claim bitwise reproduction of
past training from these recipes.

`recipes/diffusion_tasks.json` records all 40 exact episode selections and the
hashes of 80 saved training configs, explicitly labeling the old PRISM profile
`legacy-prism`. `prism_task0_train_config.json` and
`historical-baseline_task0_train_config.json` remain unchanged historical
evidence. `gated_prism_task0_train_config.json` is a new training recipe derived
from those settings, with no attached checkpoint or result.
The historical dataset revision was not pinned. Use
`scripts/libero_task_episodes.py` to verify task-to-episode mappings against the
installed dataset, and record `--dataset-revision` for a new run.

## Nominal, MCC, and robustness evaluation

```bash
uv run --no-sync python "$PRISM_ROOT/integrations/lerobot/scripts/eval_diffusion.py" \
  --lerobot-root "$PWD" --suite libero_spatial --task-id 0 \
  --baseline outputs/prism_diffusion/baseline/libero_spatial/task_0/checkpoints/020000/pretrained_model \
  --prism outputs/prism_diffusion/prism/libero_spatial/task_0/checkpoints/020000/pretrained_model \
  --output-root eval_logs/prism_diffusion/libero_spatial/task_0
```

The four rows are nominal Diffusion, MCC-Sensorless, MCC-Oracle, and PRISM.
Evaluation reads the PRISM actor mode from the saved checkpoint config and
records the corresponding gated-v2 or legacy-v1 actor ID. It never overrides
that mode. Reuse the same task checkpoint across nominal and robustness
conditions, and keep new gated results separate from legacy result sets.
MCC modifies only the first three action coordinates. Sensorless force is
estimated from actuator generalized forces and the end-effector Jacobian;
the oracle uses the sum of simulator `cfrc_ext` force components over all
bodies. That oracle signal is a scene-wide resultant with possible cancellation,
not an end-effector contact wrench. PRISM receives no force estimate.
The default MCC settings are gain 0.015, sensorless noise 0.15, delay 2,
EMA 0.9, and wrench regularization 1e-4. Oracle noise and delay are zero.
Correction clipping is optional; the robustness recipe fixes it at 0.05.

The default `--controller shared` uses stiffness 50 and damping ratio 1 for
every method. `--controller historical` reproduces the original shell's
default: only MCC rows override the controller. These protocols must be
reported separately. Evaluation runs continue after first success to measure
contact behavior. A real rollout on 2026-09-13 exposed that LIBERO returns
`done=true` on success, which made the recovered wrapper ignore
`terminate_on_success=false`. The release now honors that flag through
`libero_episode.py`; non-success backend termination is preserved. This is an
evaluation correction, with no change to actor weights or representation.
Historical early-stop diagnostics and corrected full-horizon diagnostics must
be kept separate. The force threshold is 20. Exported `contact_force_proxy`
is the maximum per-body external-force norm, a simulator diagnostic rather than
a physical sensor measurement. `action_smoothness` is computed from the incoming
policy/action-delay output before the environment applies MCC. For MCC rows it
therefore does not measure smoothness of the corrected executed action.
`contact_loss_rate` measures loss of task success after first success, rather
than a direct contact detector. Training defaults to seed 0; evaluation
defaults to seed 1000 because the original evaluation shells did not override
LeRobot's default evaluation seed.

Copy `recipes/checkpoints.example.json` and update its explicit local
`pretrained_model` directory paths to the checkpoints you intend to evaluate.
The default example pairs newly trained gated `prism` with matched `baseline`.
`checkpoints.legacy.example.json` provides separate `legacy-prism` and
`legacy-baseline` paths; edit them to your actual archived checkpoint locations.
Each directory must contain `config.json` and `model.safetensors`. The runner
verifies the requested baseline or PRISM architecture before execution; it
does not accept a Hub ID as a local checkpoint. No script chooses the latest
checkpoint implicitly. Then run a complete nominal matrix or either appendix
sweep:

```bash
uv run --no-sync python "$PRISM_ROOT/integrations/lerobot/scripts/sweep_diffusion.py" \
  --lerobot-root "$PWD" \
  --checkpoint-map "$PRISM_ROOT/integrations/lerobot/recipes/checkpoints.example.json" \
  --kind nominal --output-root eval_logs/prism_nominal
# Replace nominal with robustness or mcc, and choose a fresh output directory.
```

The nominal matrix covers all task keys present in the supplied map; the
example includes all 40. MCC maps need baseline paths for the four validation
tasks; nominal and robustness maps require both baseline and PRISM paths.

The robustness sweep uses Spatial 1, Object 3, Goal 7, and Long 1, with five
episodes per task (20 per condition), vector batch size five, and shared
controller gains. Its conditions are clean; action delay 1; proprioceptive
noise σ=0.01; image noise σ=0.05; and combined delay 1/proprioceptive σ=0.01/
image σ=0.03. It preserves the original pre-policy observation perturbation
position and zero-filled action-delay queue. The MCC validation sweep preserves
the nine `c00`–`c08` configurations on the same four tasks, with five episodes
per task and vector batch size one. Neither sweep adds dynamics perturbations.

Outputs retain the method directory names expected by
`analysis/paper/summarize_robustness.py` and `summarize_mcc_sweep.py`. MCC writes
`configs.txt` and `tasks.txt` when executed. `--save-probe-traces` in the single
evaluation runner exports state/force traces for downstream analysis.

## SmolVLA stronger-backbone experiment

```bash
uv run --no-sync python "$PRISM_ROOT/integrations/lerobot/scripts/smolvla.py" train \
  --lerobot-root "$PWD" --profile prism --seed 1000 --output outputs/smolvla_prism
uv run --no-sync python "$PRISM_ROOT/integrations/lerobot/scripts/smolvla.py" eval \
  --lerobot-root "$PWD" \
  --checkpoint outputs/smolvla_prism/checkpoints/080000/pretrained_model \
  --seed 1000 --output eval_logs/smolvla_prism_eval50
```

Profiles `baseline`, `larger`, and `prism` select the released linear,
2048-wide three-layer MLP, and gated-quadratic conditioners respectively.
Training defaults to `--profile prism`. Evaluation reads the architecture from
the explicit checkpoint; supplying `--profile` additionally checks that it
matches. `--steps` is training-only and `--checkpoint` is evaluation-only.
The registered SmolVLA PRISM identity requires the gated-quadratic, two-layer,
RMSNorm configuration. Older interaction schemas are rejected explicitly.
Its alpha is a learned per-feature parameter initialized to 0.01. Raw saved
configs with missing mode/norm fields keep the historical `vanilla`/`false`
defaults; the training runner explicitly selects the gated/RMSNorm settings.
The conditioner rejects missing or extra `quadratic_scale` even when the
parent policy loader is non-strict. Other policy keys retain upstream loading
semantics; use strict loading when checking a complete checkpoint.
The commands apply explicit common controls: RMSNorm, batch 64, eight workers, pretrained
VLM initialization, frozen vision encoder, trainable action expert/state
projection, 100,000 training steps, and a reported 80,000-step checkpoint.
Archived baseline/larger saved configs were not recovered, so the commands'
architecture settings, including RMSNorm, are not verified against those
historical checkpoints. The public release specifies changing conditioner type
relative to its shared recipe; these commands implement that stated comparison.
`--policy.type=smolvla --policy.load_vlm_weights=true` loads the VLM's pretrained
weights; it does not load an entire `lerobot/smolvla_base` policy checkpoint.
Evaluation uses 50 episodes for each of 40 tasks (2,000 total), with the
upstream success-termination behavior. Exact additional seed identities and
configs for the later three-seed manuscript claim are not established by the
available release. An explicit `--seed` enables additional runs without
presenting them as recovered results.

## Verification and provenance

A simulator smoke on 2026-09-13 completed one 280-step Spatial task-0 episode
for the historical nominal checkpoint and one for legacy ungated PRISM, with evaluation seed
1000 and shared stiffness 50/damping ratio 1. The nominal episode did not reach
task success; PRISM reached success. These are execution checks, not estimates
of paper success rates. The same-seed nominal repeat was not bitwise identical
across processes. The original early-stop run and corrected full-horizon run
are retained separately in the local validation artifacts; see
[the combined smoke report](../../validation/SMOKE_REPORT.md).

After installing the Diffusion extras, verify baseline, legacy, and gated
actor paths with a small CPU U-Net, synthetic observations, backward pass,
and action sampling. The gated check also verifies a nonzero alpha gradient
from the diffusion loss, an optimizer update, and a full-model checkpoint
round trip:

```bash
uv run --no-sync python "$PRISM_ROOT/integrations/lerobot/scripts/smoke_test.py" --lerobot-root "$PWD"
```

An optional `--conditioner-checkpoint /path/to/model.safetensors` also checks
strict loading of the historical 16→256→16 main PRISM conditioner.

`source_manifest.json` pins source files and documents included/excluded
changes. `tests/test_lerobot_integration.py` checks the historical actor's
checkpoint keys and strict round trip, polynomial interactions, gradients,
recorded task splits, matched-control command differences, and MCC oracle
separation, plus success-continuation and non-success termination behavior.
It also checks dependency-free CLI help, argument validation, source selection,
complete preflight, output protection, run records, and installer idempotence.
CPU checks validate code contracts, not paper task success rates.
Upstream licenses remain in force; see [../../NOTICE.md](../../NOTICE.md).
