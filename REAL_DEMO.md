# Add PRISM to a real robot demonstration

Train PRISM inside the policy, then deploy that checkpoint with its saved
configuration and preprocessing. This guide provides a new-demo workflow; no
hardware validation is claimed.

## 1. Choose the policy and its exact representation

Start with the Diffusion integration in [integrations/lerobot](integrations/lerobot/README.md)
to use the `diffusion_gated_state_v2` actor.
Its state history passes through input LayerNorm, two affine factors combined as
`left * (1 + alpha * right)`, then latent LayerNorm and a SiLU MLP. Each feature's
`alpha` starts at 0.01 and learns with the policy. Images retain the
existing visual encoder and actions retain the Diffusion Policy decoder.

For a SmolVLA demonstration, use the combined LeRobot integration's
`smolvla_gated_quadratic_v1` actor. The two representations are documented in
[ACTOR_CONTRACT.md](ACTOR_CONTRACT.md). The integration guide pins a revision on
which both patches apply; do not apply them to an arbitrary LeRobot revision.

Replacing a trained baseline's state projection with a randomly initialized PRISM
module is not a ready-to-run policy. Train or fine-tune the policy with PRISM enabled,
then deploy that checkpoint. Simulator checkpoints also require adaptation when
the robot's state, action units, kinematics, or cameras differ.

## 2. Keep a consistent data and robot interface

Record demonstrations using LeRobot's normal teleoperation and dataset pipeline.
Complete the robot's normal calibration and verify teleoperation first. Save:

- State feature names and ordering, units, calibration ID, and history length.
- Action feature names and ordering, absolute versus relative targets, and units.
- Camera names, resolution, frame rate, and preprocessing.
- Control frequency, action chunk length, and dataset normalization statistics.

For SO-101, joint-position targets from a leader/follower demonstration are a useful
starting interface. LIBERO's 7D relative end-effector actions are a different
interface; do not send them directly to SO-101 joints. Keep the action postprocessor
that belongs to the recorded dataset and trained policy.

Use the pinned LeRobot checkout's recording instructions and `lerobot-record --help`.
On the modern Diffusion integration revision, `lerobot-record` collects teleoperation
data and `lerobot-rollout` deploys policies. The recording command can stamp the
dataset ID; use the actual saved dataset ID and root for training. For local work,
set `--dataset.push_to_hub=false` while recording.

## 3. Train with PRISM enabled

Run from the patched LeRobot environment. Replace the example paths and dataset ID
with the recorded dataset. This is a starting training configuration, not an
estimate of the training needed to solve a particular task.

```bash
uv run lerobot-train \
  --dataset.repo_id=local/my_recorded_task \
  --dataset.root=/absolute/path/to/recorded_dataset \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_poly_kernel_conditioning=true \
  --policy.poly_kernel_source=state \
  --policy.poly_kernel_lift_mode=gated_quadratic \
  --policy.poly_kernel_gate_scale_init=0.01 \
  --policy.poly_kernel_latent_dim=256 \
  --policy.poly_kernel_hidden_dim=256 \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.resize_shape='[128,128]' \
  --policy.down_dims='[128,256,512]' \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=10 \
  --policy.optimizer_lr=1e-4 \
  --batch_size=8 \
  --num_workers=4 \
  --steps=10000 \
  --save_freq=1000 \
  --eval_freq=0 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=/absolute/path/to/prism_demo_training
```

For a matched baseline comparison, use a separate output directory, change
`--policy.use_poly_kernel_conditioning=false`, and keep data, training budget,
batch size, initialization protocol, and evaluation conditions matched.

For SmolVLA, the representation-specific flags instead are:

```text
--policy.type=smolvla
--policy.state_conditioner_type=prism
--policy.state_conditioner_product_mode=gated_quadratic
--policy.state_conditioner_num_layers=2
--policy.state_conditioner_gate_scale_init=1e-2
--policy.state_conditioner_use_rmsnorm=true
```

Use those flags with the SmolVLA training recipe and the demonstration's feature
configuration; they do not change an existing Diffusion checkpoint into SmolVLA.

## 4. Verify the saved policy

Load the saved policy and its saved pre/postprocessors in the same pinned
environment used to train it. On recorded observations, verify finite outputs,
the expected action dimension and units, and inference latency at the intended
control frequency. Check that the loaded configuration selects the intended
representation. Keep any normalization and action decoding outside PRISM exactly
as saved; PRISM itself does not map end-effector commands to motor targets.

For custom deployment code, the sequence is:

```text
robot observation -> dataset-compatible feature mapping -> saved preprocessor
-> policy.select_action(...) -> saved postprocessor -> normal robot action interface
```

Call `policy.reset()` at episode boundaries so observations and queued actions do
not carry over. PRISM is already inside the loaded policy; do not apply a second
conditioner around `select_action()` or add an action-residual controller.

## 5. Run a supervised demonstration

For the modern Diffusion integration revision, a deployment command has this
shape. Replace the checkpoint, calibrated robot ID, port, camera names/settings,
and frame rate with the values used for your dataset. This command moves the robot.

```bash
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=/absolute/path/to/prism_demo_training/checkpoints/last/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_calibrated_follower \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --task='the task used for training' \
  --fps=30 \
  --duration=10 \
  --display_data=true
```

Retain the robot's existing joint limits, target-change limits, and stop mechanism;
start in free space with an operator able to stop motion. PRISM is not a certified
force controller and does not guarantee a contact-force bound. Record the exact
checkpoint/configuration when evaluating a new demonstration.

The original standalone SmolVLA patch also supports an older revision, which can
have a different deployment CLI. If using that historical route, use its documented
entry point instead of assuming `lerobot-rollout` exists there.
