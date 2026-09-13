# Simulation experiments

Use each integration's pinned environment, then start with one explicit
checkpoint and task before launching a full matrix. The release includes the
workflows below; simulator assets, datasets, and checkpoints are external.
[Completed validation](validation/SMOKE_REPORT.md) covers bounded G1 and LIBERO
execution checks, not full retraining or benchmark averages.

## Experiment-to-code map

| Workflow | Entry points | Scope |
|---|---|---|
| G1 PPO baseline, larger actor, PRISM | [Humanoid-Gym guide](integrations/humanoid-gym/README.md): `train.py`, `run_suite.py`, `evaluate.py` | Nominal locomotion; explicit training seeds and checkpoint paths. |
| G1 degree ablations | Same guide: `degree1`, `degree2`, `degree3` variants | Fixed scale; the main `prism` variant instead uses degree-2 warmup. |
| LIBERO Diffusion | [LeRobot guide](integrations/lerobot/README.md): `train_diffusion.py`, `eval_diffusion.py` | Forty task-specific policies across Spatial, Object, Goal, and Long. |
| MCC and robustness sweeps | LeRobot `sweep_diffusion.py --kind nominal`, `robustness`, or `mcc` | Frozen baseline/MCC/PRISM checkpoints; explicit checkpoint map. |
| BFM-Zero baseline, larger core, PRISM | [Reproducibility](REPRODUCIBILITY.md#bfm-zero), [scenario launcher](integrations/bfm-zero/evaluate_scenarios.py) | LAFAN tracking: nominal, low friction, and payload mass. |
| SmolVLA baseline, larger state MLP, PRISM | [LeRobot guide](integrations/lerobot/README.md): `smolvla.py` | One multi-task LIBERO policy; 80K checkpoint and 50 episodes/task. |
| G1 physics and future-response probes | Humanoid `probe_hidden_physics.py`, `probe_future_response.py` | Frozen actor features; physics, slip, and mechanical-power targets. |
| G1 latent-factor ablation | [Factor ablation](analysis/paper/ablate_locomotion_factors.py) | Strict actor loading; action-vector L2 and coordinate-wise MAE reported separately. |
| Contact probes and diagnostics | [Analysis guide](analysis/paper/README.md): contact probe, plotting, and aggregation | Saved state/force traces; explicit preprocessing, fitting protocol, and timestep. |
| BFM scenario features | [Representation analysis](analysis/README.md) | Feature export, standardization, PCA, and t-SNE; qualitative within-panel interpretation. |
| Independent-seed aggregation | [Training-seed aggregator](analysis/paper/aggregate_training_seeds.py) | Explicit run manifest; distinct training seeds and immutable checkpoint identifiers. |

Real robot deployment instructions are in [REAL_DEMO.md](REAL_DEMO.md).
Hardware experiments and their results are outside this simulation release.

## Protocol controls

**G1.** Preserve 15×47 actor history, 3×73 critic history, 12 joint targets,
action scale 0.25, and a 0.01-second control step. The main PPO recipe uses
3,001 iterations, rollout length 60, two epochs, four minibatches, learning rate
1e-5, gamma 0.994, and GAE 0.9. Warmup and fixed-degree runs are separate
configurations. The nominal evaluator disables observation noise, dynamics
randomization, action delay/noise, and pushes; rough terrain is not covered by
this protocol. Actor-only and total actor/critic parameter counts differ.

**Diffusion.** The main recipe uses two 128×128 RGB cameras, two observation
steps, horizon 16, eight action steps, 100 diffusion training timesteps, ten
inference timesteps, U-Net widths 128/256/512, and 20K training updates.
The historical baseline uses batch 8/workers 4; PRISM uses batch 64/workers 8.
`matched-baseline` is a separate batch-64 control requiring new training.
Keep the 7D relative EEF action/controller interface and saved preprocessing.

**MCC.** The sensorless controller uses actuator generalized forces and the
end-effector Jacobian; the oracle uses simulator force access. PRISM receives
neither signal. Use shared controller Kp 50/damping ratio 1 for matched runs;
the historical controller option has different method-specific overrides.
The oracle force is a resultant over all bodies, where opposing forces can
cancel. `action_smoothness` is computed before MCC correction, so it measures
incoming actions. Exported `contact_force_proxy` is the maximum per-body force
norm; neither force signal is an EEF wrench. `contact_loss_rate` measures loss
of task success after first success.

The corrected LIBERO wrapper honors `terminate_on_success=false` and continues
to the intended horizon. Historical success-stopped diagnostics use different
episode semantics and must be kept separate.

**Robustness.** The recorded subset is Spatial 1, Object 3, Goal 7, and Long 1,
with five episodes per task (20 per condition). Conditions are clean; action
delay 1; proprioceptive noise 0.01; image noise 0.05; and combined delay 1,
proprioceptive noise 0.01, image noise 0.03. Reuse evaluation seeds and initial
states across methods. These are observation/action perturbations, not physics
shifts.

**Probes and plots.** Choose baseline feature interpretation explicitly.
`state-first-contribution` extracts only the state-dependent FiLM contribution,
not a full image-conditioned U-Net embedding. Historical surrogate features do
not certify that comparison. Record ridge strength, split protocol, temporal
window, and saved normalization. Random window splits can share episodes
between train and test; the five-step force window includes its current step.
Use the actual control timestep for impulse, power, and plots. Factor-ablation
IDs are checkpoint-specific; L2 action distance and coordinate-wise MAE differ.

## Run records

Save the resolved configuration, source revision, actor ID, training and
evaluation seeds, checkpoint step/hash, dataset revision, task IDs, termination
policy, controller settings, and raw metrics. The [inventory tool](validation/README.md)
helps recover saved evidence without treating folder names as training seeds.
Repeated episodes, tasks, or checkpoint steps are not independent training runs.

Aggregate each policy's tasks before aggregating independent training seeds.
Use [aggregate_training_seeds.py](analysis/paper/aggregate_training_seeds.py)
with the intended `--expected-training-seeds` count. A runnable recipe or
successful smoke check does not establish a complete multi-seed result.
