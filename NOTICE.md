# Licensing Notice

The standalone PRISM implementation and original PRISM adapters are licensed
under the MIT License; see [LICENSE](LICENSE). This does not replace upstream
licenses for derived patches, extracted environments, assets, or datasets.

The integration patches are derived from, and must be used with, their
respective upstream projects:

- BFM-Zero is distributed under the Creative Commons
  Attribution-NonCommercial 4.0 International license.
- LeRobot is distributed under the Apache License 2.0.
- The focused Unitree G1 and RSL-RL extraction retains its upstream BSD-3-Clause
  notices in `integrations/humanoid-gym/LICENSE.unitree` and `LICENSE.rsl-rl`.

The released PRISM and MLP G1 actor weights in `checkpoints/g1` are licensed under
MIT. The play example loads robot assets from the separately installed mjlab
project; this repository does not redistribute those assets, datasets, or complete
upstream source trees. Users remain responsible for complying with upstream licenses
and dataset terms. The original mjlab integration is used with mjlab under Apache-2.0
and RSL-RL under BSD-3-Clause.
