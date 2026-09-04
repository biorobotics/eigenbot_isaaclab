# The `eigenbot` Isaac Lab extension

This folder is the whole project. Everything else in the repository is either
upstream Isaac Lab (`isaaclab/`), legacy ROS packages for the physical robot
(`eigenbot_v2/`), or documentation.

> **Start with the [repository README](../../README.md)** — setup, how to run
> training, the tuning knob index and the docs index all live there. This file
> covers only what is specific to the extension's internals.

It is an **external Isaac Lab extension**: it is bind-mounted into the container
at `/workspace/eigenbot` by `isaaclab/docker/eigenbot.yaml` and installed with
`pip install -e source/eigenbot`. All modifications should stay inside it — the
upstream `isaaclab/` checkout is not ours to edit.

We use `rsl_rl` for RL, the **direct** workflow (not manager-based) to keep the
port from `legged_gym` close to line-for-line, and single-agent control.
A port to the manager-based workflow is reasonable once RL is settled and the ROS
migration is done, but it is a rewrite of the RL side, not the sim side.

---

## Layout

```
eigenbot/eigenbot/
├── scripts/
│   ├── list_envs.py            print the registered tasks
│   ├── zero_agent.py           zero action — the first thing to run after any change
│   ├── random_agent.py         random action smoke test
│   ├── diag_gait.py            measure the open-loop gait in numbers
│   ├── eval_compare.py         head-to-head evaluation, per terrain type, CSV out
│   ├── test_cpg.py             9 offline CPG checks (no Isaac Sim, no GPU)
│   ├── boa_compat.sh           machine-local API patches for the lab PC — never commit its output
│   ├── ars/train.py            Augmented Random Search trainer (gradient-free)
│   └── rsl_rl/
│       ├── cli_args.py         --resume / --load_run / --checkpoint / --logger ...
│       ├── train.py            PPO training entry point
│       └── play.py             replay a checkpoint; exports policy.pt and policy.onnx
└── source/eigenbot/
    ├── setup.py                packaging metadata (how pip installs the extension)
    ├── config/extension.toml   Omniverse Kit extension manifest
    └── eigenbot/
        ├── ui_extension_example.py     Kit UI example, unused
        ├── assets/
        │   ├── eigenbot.py             EIGENBOT_CFG: URDF path, PD gains, torque limit, default pose
        │   └── eigenbot/
        │       ├── urdf/eigenbot_hexapod.urdf
        │       └── meshes/*.stl, *.png
        └── tasks/direct/eigenbot/
            ├── __init__.py                  gym.register for both tasks
            ├── eigenbot_env.py              EigenbotEnv — observations, rewards, resets, curriculum
            ├── eigenbot_env_cfg.py          the main config, shared by both tasks
            ├── cpg.py                       HopfCPG — oscillators and joint mapping
            ├── eigenbot_cpg_env.py          EigenbotCPGEnv — 7-D action → 18 joints
            ├── eigenbot_cpg_env_cfg.py      CPGCfg — every gait tunable
            └── agents/
                ├── rsl_rl_ppo_cfg.py        PPO hyperparameters, 18-D baseline
                └── rsl_rl_cpg_ppo_cfg.py    PPO hyperparameters, 7-D CPG
```

---

## The four layers

### Assets — `source/eigenbot/eigenbot/assets/`

`eigenbot.py` defines `EIGENBOT_CFG`: URDF path and conversion settings, rigid
body and articulation properties, the default standing pose, and the actuator
model (stiffness 20, damping 0.5, effort limit 8 N·m).

`eigenbot/meshes/` holds every `.stl` / `.png` the URDF references;
`eigenbot/urdf/` holds the URDF itself. When adding assets, keep naming
consistent and make sure every URDF dependency resolves — the URDF is converted
to USD on each run (`force_usd_conversion=True`), so a missing mesh shows up as a
silent geometry hole rather than an error.

> ⚠️ The PD gains and torque limit mirror the physical bendy modules. Do not
> raise them to fix a gait — see the [tuning guide](../../docs/tuning.md).

### Simulation core — `eigenbot_env_cfg.py`

Rewards, commands, terrain, domain randomization, noise, normalization, the
depth-camera config, observation-size constants, sim dt and episode length. This
config is **shared by both tasks**, which is what keeps the CPG-vs-PPO comparison
honest — and means an edit here affects both.

### RL core — `eigenbot_env.py`

`EigenbotEnv` builds the 974-dim observation, computes every reward term, handles
termination, resets, the terrain curriculum and domain randomization. It is the
most complex file here (~1000 lines), but it is just an RL environment.

Adding a reward term does **not** require understanding the whole file: add a
field to `RewardScalesCfg` and a `_reward_<field>(self)` method returning a
per-env tensor, and `_prepare_reward_functions` wires it up.

### CPG layer — `cpg.py`, `eigenbot_cpg_env.py`, `eigenbot_cpg_env_cfg.py`

`EigenbotCPGEnv` subclasses `EigenbotEnv` and overrides four methods to drive the
robot from a 7-D action. Everything else is inherited. Design notes, the
joint-mapping derivation and the geometry facts that must be right:
[`docs/cpg.md`](../../docs/cpg.md).

---

## Status of the port

Ported and working: terrain generator with a row-wise difficulty curriculum,
`RayCaster` height scan (12×11), physical domain randomization (friction, COM,
motor gains, pushes) applied to PhysX, observation history, depth camera
(implemented, `use_camera = False` by default).

Still open:

- **Action delay** — the history buffer and config exist; `action_delay` is
  `False`. Enabling it costs only training time.
- **Base mass randomization** — implemented, currently off.
- **Scan / privileged encoders** — the original Isaac Gym implementation used
  dedicated `scan_encoder` and `priv_encoder` networks; this uses a flat MLP over
  the full observation. Reproducing them needs a custom rsl-rl policy class. This
  is an RL-training change, not a sim change.
- **Per-term reward logging** — `episode_sums` is accumulated but never pushed
  into `extras["episode"]`, so reward composition is invisible in TensorBoard.
  The highest-value small change in the repo.
- **Observation noise** — `NoiseCfg` / `add_noise` / `noise_scales` are defined
  but never read; no noise reaches the observation.
- **Foot slot ordering** — `find_bodies(FEET_BODIES)` is called without
  `preserve_order=True`, so `rule_1` / `rule_3` may not be grouping the legs they
  claim to. See [`docs/architecture.md`](../../docs/architecture.md) §5.
- **Depth camera consumers** — the buffer is filled but nothing reads it yet. It
  is the hook for the perception side of the project.
