# Learned-gate implementation checks

The default PRISM recipes share learned per-feature gated interactions. These
checks validate implementation and short training execution. They do not
establish convergence, benchmark averages, or hardware performance.

Validation performed on September 13, 2026:

| Check | Outcome |
|---|---|
| Python 3.10 / PyTorch 2.1.2 CPU | 76 tests passed, including original G1 source and trained-checkpoint compatibility; Ruff lint and formatting passed. |
| Python 3.12 / PyTorch 2.6.0 CPU | 76 tests passed, including original G1 source and trained-checkpoint compatibility. |
| Python 3.8 G1 compatibility environment | 17 G1 tests passed, including learned-gate updates, polynomial/gradient checks, serialization, and historical checkpoint parity. |
| Actual G1 PPO training in Isaac Gym | Two PPO updates, two environments, 60 control steps per update: 240 transitions; all 256 alpha features changed, finite losses, saved optimizer state, and exact strict checkpoint roundtrip. |
| Actual LeRobot DiffusionPolicy | Two RGB cameras, state history and action sequence; real diffusion loss/backward, learned-alpha optimizer update, and full policy save/load passed for gated and legacy actors. |
| LeRobot checkpoint mode disagreement | Changing the saved actor mode fails loading in both directions, including `from_pretrained(strict=False)`. |
| SmolVLA and BFM representation branches | Gate initialization/update, checkpoint roundtrip, mapped shared-core float32 parity, and missing/extra-gate rejection passed on both CPU environments. |
| Pinned installation | LeRobot fresh combined installation and repeat no-op passed; BFM patch and SmolVLA patch applied at their documented upstream pins. |
| Historical Diffusion compatibility | Previous-release initialization, keys, outputs, input gradients, and parameter gradients match exactly for ungated modes; archived conditioner checkpoint loads strictly. |
| Analysis utilities | Gated latent extraction agrees with the shared recurrence; G1 gate ablation preserves and restores the saved actor; checkpoint inventory identifies new actor versions. |

The Diffusion full-policy check uses synthetic, correctly shaped observations
and actions with no dataset download. The G1 run is a short simulator training
check with a fresh actor. Neither is a task-performance comparison.

The G1 alpha tensors received 16 optimizer steps; their maximum absolute change
from initialization was 0.004110. No scalar warmup buffer was present. The
[G1 manifest](../integrations/humanoid-gym/SOURCE_MANIFEST.json) records the
checkpoint hash, seed, transition count, and validation environment.

G1 and Diffusion require newly trained checkpoints for the gated formulation.
The [historical simulator report](SMOKE_REPORT.md) concerns their earlier actor
versions. SmolVLA/BFM representation checks do not execute their full training
pipelines or resolve historical result provenance.

Checkpoint gates are versioned independently from backbone-specific projection
and normalization settings. Gate-schema guards reject missing/extra gates even
where upstream loaders use non-strict loading; they do not turn every upstream
full-policy loader into a strict loader. Use the saved configuration and the
[actor contract](../ACTOR_CONTRACT.md).
