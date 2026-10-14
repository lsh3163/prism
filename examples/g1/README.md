# Play the trained G1 policies

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsh3163/prism/blob/main/examples/g1/PRISM_G1_Play.ipynb)

The notebook includes condition controls, fresh PRISM/MLP videos and an optional
interactive 3D viewer. A GPU runtime is recommended; it automatically selects a
compatible PyTorch backend and device, with CPU fallback. Run its cells in order.

The repository includes the camera-ready **PRISM** and **MLP** deterministic actors,
including their trained observation normalization. Each checkpoint is less than
2 MB. Start from the repository root; no dataset, login, experiment directory or
training job is required.

## Install and play

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then:

```bash
git clone https://github.com/lsh3163/prism.git
cd prism
uv venv --python 3.13
uv pip install -e . -r examples/g1/requirements.txt --torch-backend=cpu
uv run --no-sync python examples/g1/play.py
```

Open **http://127.0.0.1:8080**. The viewer lets you pause, reset, rotate the camera
and change the velocity command. Press Ctrl+C in the terminal to stop.
The first launch compiles MuJoCo-Warp kernels and takes longer than later launches.
CPU execution needs no NVIDIA GPU; full comparison videos can take several minutes.

The environment is pinned to the native mjlab source used for these actors.
The core package has lighter requirements and does not install a simulator.
Use this separate virtual environment for the example. Python 3.13 on Linux is
the tested configuration. The browser viewer does not require a desktop display.

## Compare PRISM and MLP

Run each command in a separate terminal. Both use the same terrain seed and home
reset; open ports 8080 and 8081 in two browser windows.

```bash
uv run --no-sync python examples/g1/play.py --actor prism --friction 0.2 --port 8080
uv run --no-sync python examples/g1/play.py --actor mlp --friction 0.2 --port 8081
```

Change the physical condition or command:

```bash
uv run --no-sync python examples/g1/play.py --mass-ratio 1.4
uv run --no-sync python examples/g1/play.py --payload 8
uv run --no-sync python examples/g1/play.py --terrain flat
uv run --no-sync python examples/g1/play.py --velocity 1.0 0.0 0.0
```

`--friction` sets sliding friction on all fourteen foot collision geoms.
`--mass-ratio` scales all original robot link masses and inertias coherently.
`--payload` adds an independently specified, welded, noncolliding load to the torso;
the visible backpack has the same dimensions and pose as that load. The joint
controller, action scaling and observation interface remain native mjlab.
The inference environment computes actor observations and omits unused critic
observations, contact rewards and training metrics.

This interactive example uses a **5 × 5 random-rough grid** and a deterministic
home reset. It is a qualitative playground. The paper's quantitative evaluation
used a larger certified grid and 1,000 paired attempts per condition; a single
play rollout is not a reproduction of those aggregate results.

## Headless and GPU execution

Run one attempt without a browser. It stops at a fall or the requested horizon:

```bash
uv run --no-sync python examples/g1/play.py --headless --seconds 20
```

The final JSON line reports elapsed simulated time, fall status and forward
distance. Interactive play automatically resets after a fall.

To save a native simulation video for a notebook or headless server:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa uv run --no-sync python examples/g1/play.py \
  --headless --video outputs/g1/prism.mp4
```

CPU video rendering uses the OSMesa system library (`sudo apt install libosmesa6`
on Ubuntu). The Colab setup cell installs it automatically. GPU offscreen rendering
can use `MUJOCO_GL=egl`. The bundled FFmpeg executable writes the MP4.
CPU videos default to 5 fps to reduce rendering cost; CUDA videos default to 25 fps.
Use `--video-fps 25` to override the CPU default. Physics and policy rates remain
unchanged at 200 Hz and 50 Hz, respectively.

For an NVIDIA GPU, install the same requirements with a compatible CUDA PyTorch
backend, then select the device:

```bash
uv pip install -e . -r examples/g1/requirements.txt --torch-backend=cu130
uv run --no-sync python examples/g1/play.py --device cuda:0
```

`--viewer native` selects the desktop MuJoCo viewer and requires a local display.
For a remote machine, forward its viewer port, for example `ssh -L 8080:127.0.0.1:8080 your-host`.
The browser server listens on localhost by default.

## Checkpoints

| Actor | File | Training updates | Policy parameters |
|---|---|---:|---:|
| PRISM | [prism.pt](../../checkpoints/g1/prism.pt) | 30,000 | 391,965 |
| MLP | [mlp.pt](../../checkpoints/g1/mlp.pt) | 30,000 | 315,066 |

[manifest.json](../../checkpoints/g1/manifest.json) records SHA-256 hashes of both
released files and the original full training checkpoints. Files contain only the
original actor tensors: strict-loading keys, learned normalization, policy and
Gaussian distribution parameters. Critics, optimizers and run metadata are omitted.
They are inference checkpoints and cannot resume PPO training.

The G1 actor keeps its original per-feature interaction coefficients and shared
normalization; it loads through the [native adapter](../../integrations/mjlab/actors_mjlab_shared_rms.py).
Use the standalone `PRISM` API when building a new policy rather than converting
these trained actors to a different parameterization. Original PRISM code and
released actor weights use MIT; mjlab and its robot assets retain upstream licenses.
