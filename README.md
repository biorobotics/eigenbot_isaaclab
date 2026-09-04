# EigenBot × Isaac Lab — Reinforcement Learning Locomotion

Simulation and RL training for the **EigenBot** hexapod (CMU Biorobotics Lab) in
[Isaac Lab](https://github.com/isaac-sim/IsaacLab), ported from the original
Isaac Gym / `legged_gym` implementation.

The robot is an 18-DOF modular hexapod: six legs, each built from three "bendy"
modules (`M1..M18`). Everything project-specific lives in one Isaac Lab
extension at [`eigenbot/eigenbot/`](eigenbot/eigenbot) — the `isaaclab/` folder
is an unmodified upstream checkout used only for the Docker build.

**Two locomotion policies are registered, and they are designed to be compared
head to head:**

| Task ID | Approach | Action space | Agent config |
|---|---|---|---|
| `Template-Eigenbot-Direct-v0` | End-to-end PPO — the policy writes all 18 joint targets | 18 | `rsl_rl_ppo_cfg.py` |
| `Template-Eigenbot-CPG-Direct-v0` | CPG+RL — a Hopf central pattern generator produces a tripod gait; the policy only *modulates* it (1 turn bias + 6 per-leg gains) | 7 (25 in residual mode) | `rsl_rl_cpg_ppo_cfg.py` |

Both tasks share the **same 974-dim observation, the same reward function, the
same terrain and the same domain randomization**, so a difference in the numbers
is a difference in method — that comparison is the point of the project.

> **Current status (2026-09):** both policies train end to end and walk. The
> first 10k-vs-10k comparison tied on reward but revealed a reward exploit (see
> [Results](#results-so-far)); rewards were re-weighted and a second comparison
> was launched. The `eval_compare.py` head-to-head has not been run yet.

---

## Start here

| I want to… | Go to |
|---|---|
| Get it running for the first time | [Quick start](#quick-start) |
| Know what file does what | [Where everything lives](#where-everything-lives) |
| Train a policy | [Running a training](#running-a-training) |
| Watch a trained policy | [Watching a policy](#watching-a-policy) |
| Change a reward / gait / terrain / hyperparameter | [**Tuning: the knob index**](#tuning-the-knob-index) ← most-asked question |
| Add a new task or a new control approach | [Adding a new policy](#adding-a-new-policy) |
| Understand *why* the env is built this way | [`docs/architecture.md`](docs/architecture.md) |
| Understand the CPG | [`docs/cpg.md`](docs/cpg.md) |
| Run on the lab training PC | [`docs/boa_runbook.md`](docs/boa_runbook.md) |
| Know what has already been tried | [`docs/experiment_log.md`](docs/experiment_log.md) |

---

## Quick start

<details>
<summary>Docker build → enter → install → smoke test → train (7 steps)</summary>

Tested on Ubuntu 20.04. Docker is the supported path — it pins the Isaac Sim /
Isaac Lab versions and sidesteps every version conflict. You need Docker, Docker
Compose and the NVIDIA Container Toolkit, plus an NVIDIA GPU with ~12 GB VRAM.

**1. Clone**

```bash
git clone https://github.com/biorobotics/eigenbot_isaaclab
cd eigenbot_isaaclab
```

**2. Build and start the container** (10–15 min the first time). The
`--files eigenbot.yaml` flag adds an overlay that bind-mounts
`eigenbot/eigenbot/` into the container at `/workspace/eigenbot`, so edits on the
host show up inside immediately.

```bash
python isaaclab/docker/container.py start --files eigenbot.yaml
```

**3. Enter the container** — this is your development shell.

```bash
python isaaclab/docker/container.py enter --files eigenbot.yaml
```

**4. Install the extension** (once per container build):

```bash
cd /workspace/eigenbot && pip install -e source/eigenbot
```

**5. Smoke test — is the robot there and does gravity work?**

```bash
python scripts/zero_agent.py --task Template-Eigenbot-Direct-v0 --num_envs 1
```

**6. See the engineered gait with no policy at all** (zero action on the CPG task
means θ=0 and b at the midpoint of `[b_min, b_max]` = 0.875 on every leg, i.e.
the pure open-loop tripod):

```bash
python scripts/zero_agent.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 1
```

**7. Train:**

```bash
python scripts/rsl_rl/train.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 2048 --headless
```

If the viewer is black or the robot is invisible, see
[Troubleshooting](#troubleshooting).

</details>

## Where everything lives

<details>
<summary>Repo layout, the files you will actually edit, and every script</summary>

Everything you will edit is under `eigenbot/eigenbot/`. Two path shorthands are
used below:

- **`EXT`** = `eigenbot/eigenbot/`
- **`TASK`** = `eigenbot/eigenbot/source/eigenbot/eigenbot/tasks/direct/eigenbot/`

```
eigenbot_isaaclab/
├── README.md                  ← you are here
├── docs/                      ← architecture, tuning, CPG design, boa runbook, experiment log
├── isaaclab/                  ← upstream Isaac Lab checkout (do not edit; docker/eigenbot.yaml is ours)
├── eigenbot_v2/               ← legacy ROS/Gazebo/CoppeliaSim packages for the physical robot (not used by RL)
└── eigenbot/eigenbot/         ← THE EXTENSION (all RL work happens here)
    ├── scripts/               ← everything you run
    └── source/eigenbot/       ← the installable python package
```

### Files you will actually touch

| File | What it is | Touch it when… |
|---|---|---|
| `TASK/eigenbot_env_cfg.py` | **The main config.** Rewards, commands, terrain, domain randomization, noise, observation sizes, sim dt, episode length. Shared by *both* tasks. | Tuning rewards, terrain, curriculum, DR — the most common edit |
| `TASK/eigenbot_env.py` | `EigenbotEnv` — the RL environment. Observation assembly, reward *functions*, termination, resets, terrain curriculum, domain-randomization application. ~1000 lines. | Adding/changing a reward function, changing observations |
| `TASK/eigenbot_cpg_env_cfg.py` | `CPGCfg` — every CPG tunable (frequency, amplitudes, gains, joint mapping, per-leg lift scales) + `EigenbotCPGEnvCfg` | Tuning the gait |
| `TASK/cpg.py` | `HopfCPG` — the oscillators, coupling, action decode, joint mapping | Changing the *structure* of the gait generator |
| `TASK/eigenbot_cpg_env.py` | `EigenbotCPGEnv` — the ~120-line subclass that drives joints from a 7-D action | Changing how CPG output reaches the joints |
| `TASK/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters + network size for the 18-D baseline | Tuning learning rate, entropy, net width, iteration count |
| `TASK/agents/rsl_rl_cpg_ppo_cfg.py` | Same, for the 7-D CPG task (smaller net) | Same |
| `TASK/__init__.py` | `gym.register(...)` for both tasks | Registering a new task |
| `EXT/source/eigenbot/eigenbot/assets/eigenbot.py` | `EIGENBOT_CFG` — URDF path, PD gains, torque limit, default standing pose, physics properties | Changing the robot model. ⚠️ **PD gains / effort limit mirror the real hardware — do not raise them to fix a gait** |
| `EXT/source/eigenbot/eigenbot/assets/eigenbot/urdf/` | The URDF and meshes | Robot geometry changed |

### Scripts (all run from `EXT`, inside the container)

| Script | What it does |
|---|---|
| `scripts/rsl_rl/train.py` | PPO training (rsl-rl). Logs to `logs/rsl_rl/<experiment_name>/<timestamp>/` |
| `scripts/rsl_rl/play.py` | Replay a checkpoint in the viewer; also exports `policy.pt` (JIT) and `policy.onnx` next to the checkpoint |
| `scripts/eval_compare.py` | **The head-to-head evaluator.** Fixed command, per-terrain metrics, CSV out |
| `scripts/diag_gait.py` | Measures the open-loop gait in numbers (per-leg clearance, stance fraction, body travel). Replaces eyeballing the viewer |
| `scripts/test_cpg.py` | 9 offline CPG unit checks. **No Isaac Sim needed — runs anywhere with PyTorch** |
| `scripts/zero_agent.py` | Zero action. The single most useful eyeball test after any mapping/sign change |
| `scripts/random_agent.py` | Random action smoke test |
| `scripts/list_envs.py` | Print the registered tasks |
| `scripts/ars/train.py` | Augmented Random Search trainer (gradient-free, the paper's optimizer). Never yet run |
| `scripts/boa_compat.sh` | Re-applies machine-local API patches on the lab PC — see [`docs/boa_runbook.md`](docs/boa_runbook.md) |

</details>

## Running a training

<details>
<summary>Commands, flags that matter, where output goes, TensorBoard, tmux</summary>

All commands run from `/workspace/eigenbot` (i.e. `EXT`) inside the container.

**Baseline PPO (18-D action):**

```bash
python scripts/rsl_rl/train.py --task Template-Eigenbot-Direct-v0 --num_envs 2048 --max_iterations 10000 --seed 42 --headless
```

**CPG+RL (7-D action):**

```bash
python scripts/rsl_rl/train.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 2048 --max_iterations 10000 --seed 42 --headless
```

Drop `--headless` to watch it train (much slower — use a small `--num_envs`).

### Flags worth knowing

| Flag | Meaning |
|---|---|
| `--num_envs` | Parallel environments. **2048 is the safe number on a 12 GB card**; 4096 nearly fills it and a second job on the same GPU will corrupt the PhysX scene |
| `--max_iterations` | Overrides `max_iterations` in the agent config |
| `--seed` | Set it, and keep it identical across policies you intend to compare |
| `--device cuda:0` / `cuda:1` | Pick a GPU. Check `nvidia-smi` first so you don't land on someone else's job |
| `--headless` | No viewer. Always use this for real runs |
| `--resume --load_run <dir> --checkpoint <file>` | Continue an earlier run |
| `--run_name <name>` | Suffix on the log directory, so runs are identifiable later |
| `--video --video_interval N --video_length L` | Record training clips |
| `--use_camera` | Enable the depth camera (off by default) |

### Where the output goes

```
logs/rsl_rl/eigenbot_locomotion/<YYYY-MM-DD_HH-MM-SS>/       ← baseline PPO
logs/rsl_rl/eigenbot_cpg_locomotion/<YYYY-MM-DD_HH-MM-SS>/   ← CPG+RL
    ├── model_<iter>.pt        every save_interval (50) iterations
    ├── params/env.yaml        the FULL resolved env config for this run
    ├── params/agent.yaml      the full agent config
    └── events.out.tfevents…   TensorBoard
```

`params/env.yaml` is the record of exactly what a run was trained with — check it
before trusting an old checkpoint.

### Monitoring

From a **second** terminal in the same container:

```bash
docker ps
```

```bash
docker exec -it <container_name> bash
```

```bash
tensorboard --logdir /workspace/eigenbot/logs/rsl_rl --port 6006
```

Pointing `--logdir` at `logs/rsl_rl` (rather than one experiment) puts both
methods on the same axes. Note that TensorBoard will not pick up runs created
after it started if it has been running for days — restart it before concluding a
run is missing.

### Long runs

Isaac Sim ignores `Ctrl+C`; kill from another terminal with
`pkill -9 -f train.py`. Put long runs in `tmux` — launch detached rather than
attaching:

```bash
tmux new -d -s cpg
```

```bash
tmux send-keys -t cpg "cd /workspace/eigenbot && python scripts/rsl_rl/train.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 2048 --max_iterations 10000 --seed 42 --headless" Enter
```

```bash
tmux capture-pane -pt cpg | tail -20
```

</details>

## Watching a policy

<details>
<summary>play.py, zero_agent, the gait measurer, and the offline CPG tests</summary>

**Replay the newest checkpoint of a task:**

```bash
python scripts/rsl_rl/play.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 32
```

**Replay a specific checkpoint:**

```bash
python scripts/rsl_rl/play.py --task Template-Eigenbot-CPG-Direct-v0 --checkpoint logs/rsl_rl/eigenbot_cpg_locomotion/2026-08-01_01-31-28/model_9999.pt --num_envs 8
```

`play.py` also writes `exported/policy.pt` (TorchScript) and
`exported/policy.onnx` next to the checkpoint — those are what a ROS 2 inference
node on the physical robot would load.

**Look at the gait with no policy** (θ=0 and b=0.875, the midpoint of
`[b_min, b_max]`):

```bash
python scripts/zero_agent.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 1
```

**Measure the gait instead of eyeballing it.** `diag_gait.py` reports per-leg
foot clearance, mean foot height, stance fraction, body travel, mean body height
and lateral drift. Run it on flat ground (`terrain_type="plane"`), because foot
height is measured in world z:

```bash
python scripts/diag_gait.py --headless --seconds 8
```

CPG parameters can be overridden on the command line, so sign/scale sweeps need
no file edits:

```bash
python scripts/diag_gait.py --headless --omega_hz 1.0 --lift_scales "1.7,1,1,1.35,1,1"
```

Available overrides: `--swing`, `--lift`, `--lift_signs`, `--lift_scales`,
`--omega_hz`, `--stiffness`, `--damping`, `--effort`, `--seconds`, `--settle`.

**Check the CPG maths without Isaac Sim** (any machine with PyTorch — 9 checks
covering tripod phase locking, limit-cycle stability, perturbation recovery, gait
direction, rear-leg lift boost, offset bounds, steering asymmetry, reset and
batching):

```bash
python scripts/test_cpg.py
```

> ⚠️ A still frame of a tripod gait always looks broken — three legs are airborne
> by design. Judge motion, never a screenshot.

</details>

## Comparing policies

<details>
<summary>eval_compare.py, the metrics it reports, and a fair-comparison protocol</summary>

Reward totals are **not** a comparison. In the first study two policies both
scored ~95: one walked, the other vibrated its legs in place. Always watch
`play.py` and then measure with `eval_compare.py`.

`scripts/eval_compare.py` loads a checkpoint, holds the command fixed (fixed
forward speed, no random heading), rolls out episodes and reports **per terrain
sub-type**:

- `forward_m` — distance along the initial heading
- `speed_mps` — achieved vs commanded speed
- `early_term` — fraction that fell / hit an illegal contact before timeout
- `roll_std`, `pitch_std` — attitude stability
- `lateral_m` — drift off the straight line

> ⚠️ **Two things to fix in the script before publishing numbers.** It sets
> `max_init_terrain_level = None`, which does *not* freeze the terrain level —
> `None` means "no cap", so evaluation envs are seeded across **all 10**
> difficulty rows while training only used rows 0–5, making evaluation harder
> than training; and the terrain curriculum keeps promoting/demoting during the
> rollout. `commands.curriculum = False` is a no-op — nothing in this env reads
> that field. Only the fixed velocity range and `rand_heading = False` take
> effect.

```bash
python scripts/eval_compare.py --task Template-Eigenbot-CPG-Direct-v0 --episodes 40 --num_envs 40 --command_vel 0.3 --seed 123 --headless --out logs/eval_cpg.csv
```

```bash
python scripts/eval_compare.py --task Template-Eigenbot-Direct-v0 --episodes 40 --num_envs 40 --command_vel 0.3 --seed 123 --headless --out logs/eval_ppo.csv
```

Then diff the two CSVs. **Keep `--episodes`, `--command_vel` and `--seed`
identical between runs or the comparison means nothing.** Use `--checkpoint` to
pin an exact `.pt` instead of taking the newest.

### Protocol for a fair comparison

1. Both policies trained with the same `--num_envs`, `--max_iterations`, `--seed`
   and the same reward config (the reward terms live in the shared
   `eigenbot_env_cfg.py`, so this is automatic unless you edit between runs).
2. `play.py` both — confirm each is actually locomoting before believing a number.
3. `eval_compare.py` both with identical flags.
4. Compare TensorBoard curves against **iterations** *and* against **wall-clock**
   — the latter is the data-efficiency claim.

</details>

## Tuning: the knob index

<details>
<summary>Every knob, the file it lives in, and a symptom → knob table</summary>

This is the "where do I go to change X" section. Full detail and the reasoning
behind current values: [`docs/tuning.md`](docs/tuning.md).

### Rewards → `TASK/eigenbot_env_cfg.py` → `RewardScalesCfg`

Set a scale to `0.0` to disable a term. Every scale is multiplied by the policy
`dt` internally, and each name maps to a `_reward_<name>` method in
`eigenbot_env.py` — **add a scale field and a matching method and the term is
picked up automatically** (`_prepare_reward_functions` does the dispatch).

| Term | Current | What it does |
|---|---|---|
| `tracking_goal_vel` | `10.0` | Exponential reward for velocity along the commanded heading. **The main locomotion signal** |
| `delta_yaw` | `1.2` | Heading alignment |
| `feet_air_time` | `0.87` | Rewards long swing phases (steps, not shuffles) |
| `rule_1` | `0.35` | Insect gait rule: no simultaneous ipsilateral liftoff |
| `rule_3` | `0.1` | Insect gait rule: swing-initiation timing |
| `termination` | `-1.0` | Terminal penalty (applied *after* the positive-only clip) |
| `lin_vel_z` | `-1.0` | Penalize bouncing |
| `ang_vel_xy` | `-0.05` | Penalize roll/pitch rate |
| `orientation` | `-0.25` | Penalize non-flat base (flat terrain only) |
| `torques` | `-0.0002` | Effort penalty |
| `dof_acc` | `-2.5e-7` | Joint acceleration penalty (smoothness) |
| `action_rate` | `-0.01` | Penalize jerky action changes |
| `collision` | `-0.25` | Contact on leg modules |
| `stumble` | `-0.25` | Feet hitting vertical surfaces |
| `stand_still` | `-0.5` | Penalize motion at zero command |
| `dof_pos_limits` | `-1.0` | Joint-limit violation |
| `dof_vel` | `0.0` | Off |

⚠️ `only_positive_rewards = True` clips the total to ≥0 *before* the termination
penalty. Combined with generous survival terms this is exactly how a policy
learns to bank reward without travelling — see [Results](#results-so-far).

Shaping parameters live next door in `RewardsCfg`: `tracking_sigma`,
`soft_dof_pos_limit`, `base_height_target`, `contact_tresh`, `stumble_tresh`,
`exp_coeff_rule3`, `torque_limit_hard`.

### Commands → `CommandsCfg` / `CommandRangesCfg` (same file)

`lin_vel_x` `(0.2, 0.4)` — commanded forward speed range. The floor is off zero
deliberately: `lin_vel_clip = 0.1` zeroes small commands, so a `(0.0, …)` range
let "stand still" satisfy the velocity term outright.
Also here: `ang_vel_yaw`, `heading`, `resampling_time` (10 s), `heading_command`,
`rand_heading`, `curriculum`.

### Terrain → `EIGENBOT_ROUGH_TERRAIN_CFG` + `EigenbotEnvCfg.terrain` (same file)

Current mix on a 10×20 grid of 8×8 m patches: flat 20% / random_rough 40% /
slopes 20% / obstacles 20%. Change the mix in `sub_terrains`.
Set `terrain_type="plane"` for flat-ground gait inspection, `"generator"` for
training. Row-wise difficulty curriculum is in
`EigenbotEnv._update_terrain_curriculum` — robots are promoted when they cover
more than half a patch.

### Domain randomization → `DomainRandCfg` (same file)

`randomize_friction` (0.5–1.25), `randomize_base_com` (±0.2), `randomize_motor`
(0.8–1.2× PD gains), `push_robots` (every 15 s, up to 1 m/s).
`randomize_base_mass` and `action_delay` are currently **off**.

### Observation scaling & noise → `NormalizationCfg`, `NoiseCfg` (same file)

`obs_scales`, `clip_observations` (100), `clip_actions` (1.0).

⚠️ **`NoiseCfg` is dead config.** `add_noise` and `noise_scales` are defined but
nothing in `eigenbot_env.py` reads them — no noise is added to observations at
all. That matters for sim-to-real; wiring it up is an open item.

### Sim & episode → `EigenbotEnvCfg` (same file)

`decimation` 4 (policy runs at 50 Hz, physics at 200 Hz), `episode_length_s` 25,
`action_scale` 0.25, `sim.dt` 0.005, `scene.num_envs`, `env_spacing`.

### Robot hardware model → `EXT/source/eigenbot/eigenbot/assets/eigenbot.py`

PD stiffness 20, damping 0.5, `effort_limit_sim` 8 N·m, default standing pose
(swing joints by leg *position*, identical on both sides: M1/M4 rear −π/4,
M2/M5 mid 0, M3/M6 front +π/4; M7–M18 all +π/4), spawn height, URDF path.
⚠️ **The PD gains and torque limit mirror the physical modules. They were raised
once to fix a gait and deliberately reverted. Fix gaits with the CPG parameters
instead, or the policy will not transfer to hardware.**

### PPO hyperparameters → `TASK/agents/rsl_rl_ppo_cfg.py` (18-D) and `rsl_rl_cpg_ppo_cfg.py` (7-D)

| | baseline | CPG |
|---|---|---|
| hidden dims | `[1024, 512, 128]` | `[256, 128]` |
| `init_noise_std` | 1.0 | 0.4 |
| `entropy_coef` | 0.01 | 0.002 |
| `max_iterations` | 20000 | 5000 |
| `experiment_name` | `eigenbot_locomotion` | `eigenbot_cpg_locomotion` |

Shared: lr 3e-4 adaptive, γ 0.99, λ 0.95, clip 0.2, 5 epochs, 4 minibatches,
24 steps/env, `desired_kl` 0.01, save every 50 iters.

### The gait (CPG task only) → `TASK/eigenbot_cpg_env_cfg.py` → `CPGCfg`

| Field | Current | What it controls |
|---|---|---|
| `omega` | `2π·1.5` | Gait frequency. **The main speed lever** — see the note below |
| `swing_amplitude` | `0.45` rad | Fore/aft stride at b=1 |
| `lift_amplitude` | `0.55` rad | Leg lift during swing at b=1 |
| `lift_scales` | `(1.7,1,1,1.35,1,1)` | Per-leg lift multiplier. **The correct lever for uneven clearance** — the rear pair (legs 0 and 3) carries the most load and drags without a boost |
| `b_min` / `b_max` | `0.5` / `1.25` | Range of the per-leg gains the policy commands |
| `turn_gain` | `0.6` | How strongly θ biases left/right stride |
| `lift_phase_sign` | `-1.0` | Which half-cycle the legs lift in. Flip if the robot walks backwards |
| `lift_joint_signs` | `(1.0, 0.5)` | Signs on the two lift joints. Wrong values make the shank trail the thigh |
| `leg_joint_indices` | `((0,6,12),…)` | Articulation order is **breadth-first**, not per-leg. Verify with `find_bodies(..., preserve_order=True)` |
| `leg_sides` | `(-1,-1,-1,1,1,1)` | Body side per leg, for differential steering |
| `alpha`, `mu`, `coupling_weight`, `phase_offsets` | 100, 1.0, 1.0, tripod | Oscillator dynamics |
| `use_residual` / `residual_scale` | `False` / `0.1` | Adds 18 per-joint corrections on top of the CPG (action space becomes 25) |

> ⚠️ **Known divergence:** the tuning that produced the measured gait in
> [`docs/experiment_log.md`](docs/experiment_log.md) used **1.0 Hz**
> (`omega = 2π·1.0`), and that change was never committed — the file still says
> 1.5 Hz. At 1.5 Hz the PD cannot track the commanded lift within a half-cycle
> under load and the loaded legs never break contact. Verify with
> `diag_gait.py --omega_hz 1.0` before trusting the committed default.

### Common symptom → knob

| Symptom | Look at |
|---|---|
| Robot vibrates legs in place, no travel | `tracking_goal_vel`, `lin_vel_x` floor, `only_positive_rewards` |
| Rear legs drag, front legs clear fine | `lift_scales` (**not** the PD gains) |
| Robot walks backwards | `lift_phase_sign` |
| Second half of each leg doesn't move forward | `lift_joint_signs` |
| Front legs fold under | `b_max`, joint-limit clamp in `EigenbotCPGEnv._apply_action` |
| Too slow, stride won't grow | `omega`; the front and rear swing joints start at ±π/4 against a ±π/2 limit so stride has little room, making frequency the practical lever — see the "frequency as an action" thread |
| Action noise std blows up (~10+) | `entropy_coef`, `init_noise_std` |
| Falls over constantly on rough terrain | terrain mix proportions, `max_init_terrain_level`, penalty weights |

</details>

## How the environment works

<details>
<summary>Control loop, the 974-dim observation, reward pipeline, resets</summary>

Full walkthrough: [`docs/architecture.md`](docs/architecture.md). The short
version:

**Control loop.** Physics at 200 Hz (`sim.dt = 0.005`), policy at 50 Hz
(`decimation = 4`). Baseline: `target = action_scale · action + default_dof_pos`.
CPG: the oscillators are stepped **once per physics substep** while the policy
action is held constant across the decimation window, then
`target = default_dof_pos + cpg_offsets`, clamped to the soft joint limits. Both
paths then compute torques with an explicit PD law and clamp to 8 N·m.

**Observation — 974 dims**, identical for both tasks:

| Block | Dims | Contents |
|---|---|---|
| proprioception | 72 | ang vel (3), roll/pitch (2), Δyaw (1), projected gravity (3), commanded vel (1), joint pos (18), joint vel (18), flat/not-flat flags (2), last action (18), foot contacts (6) |
| height scan | 132 | 12×11 ray grid, 0.15 m spacing, 0.375 m ahead of the base |
| privileged explicit | 9 | base linear velocity + zero padding |
| privileged latent | 41 | mass/COM params (4), friction (1), motor strength P and D (36) |
| history | 720 | the last 10 proprioception frames |

On the CPG task the "last action" slot is fed the **CPG's 18 joint offsets**, not
the 7-D policy action — so the observation size never changes and the comparison
stays clean.

**Reward.** Each active term's function is called, multiplied by
`scale · dt`, and accumulated into `episode_sums`. The total is clipped to ≥0
when `only_positive_rewards` is set, *then* the termination penalty is added.

**Termination.** Any contact force >1 N on `base_link`, or timeout at 25 s.

**Reset.** Joint positions scaled by U[0.5, 1.5] of default, root velocity
U[-0.5, 0.5], commands resampled, domain-randomization parameters redrawn,
terrain curriculum updated.

> **Known gap:** the env does not log per-term reward scalars
> (`extras["episode"]["rew_*"]`), so reward composition is invisible in
> TensorBoard. Adding that would have caught the reward exploit in minutes
> instead of after two 10k-iteration runs. It is the highest-value small change
> in the repo.

</details>

## Adding a new policy

<details>
<summary>Five-step recipe, with the CPG task as the worked example</summary>

The CPG task is the worked example: it changes the action space and the actuation
path while inheriting everything else. Copy that shape.

**1. Config — subclass `EigenbotEnvCfg`** so rewards, observations, terrain and
domain randomization stay shared (that is what keeps comparisons fair):

```python
# TASK/eigenbot_myapproach_env_cfg.py
@configclass
class MyCfg:
    my_param: float = 1.0

@configclass
class EigenbotMyEnvCfg(EigenbotEnvCfg):
    action_space: int = 12          # whatever your policy emits
    my: MyCfg = MyCfg()
```

**2. Environment — subclass `EigenbotEnv`** and override only the actuation path:
`_pre_physics_step` (store the policy action), `_apply_action` (turn it into
joint targets + torques), `_reset_idx` (reset your internal state).

⚠️ **The `_init_buffers` gotcha.** The parent sizes every joint-dimension buffer
from `cfg.action_space` and finishes by applying domain randomization to
`(num_envs, 18)` robot data. If your action space is not 18, swap it around the
`super()` call exactly as `EigenbotCPGEnv._init_buffers` does, or construction
crashes on a shape mismatch:

```python
policy_dim = self.cfg.action_space
self.cfg.action_space = NUM_JOINTS
try:
    super()._init_buffers()
finally:
    self.cfg.action_space = policy_dim
```

Also feed the observation's 18-dim "last action" slot with your *joint offsets*
(`self.actions = offsets`), not the raw policy action, so the observation size
and the comparison stay intact.

**3. Agent config** — copy `agents/rsl_rl_cpg_ppo_cfg.py`, give it a **unique
`experiment_name`** (that is the `logs/rsl_rl/<name>/` folder) and size the
network to the action space.

**4. Register it** in `TASK/__init__.py`:

```python
gym.register(
    id="Template-Eigenbot-MyApproach-Direct-v0",
    entry_point=f"{__name__}.eigenbot_myapproach_env:EigenbotMyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eigenbot_myapproach_env_cfg:EigenbotMyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_myapproach_ppo_cfg:MyPPORunnerCfg",
    },
)
```

**5. Verify, in this order:**

```bash
python scripts/list_envs.py
```

```bash
python scripts/zero_agent.py --task Template-Eigenbot-MyApproach-Direct-v0 --num_envs 1
```

```bash
python scripts/diag_gait.py --headless --task Template-Eigenbot-MyApproach-Direct-v0
```

Only then train. If your approach has pure-maths components, add an offline test
next to `scripts/test_cpg.py` — it needs no GPU and catches sign errors in
seconds instead of after a 6-hour run.

### Just retuning, not adding a method?

You do **not** need a new task to change rewards or gait parameters — edit
`eigenbot_env_cfg.py` / `eigenbot_cpg_env_cfg.py`, use `--run_name` to label the
run, and compare in TensorBoard. Add a new task only when the *action space or
actuation path* changes.

</details>

## Running on the lab training PC (boa)

<details>
<summary>Conda env, the compat patches, GPU etiquette</summary>

Full detail: [`docs/boa_runbook.md`](docs/boa_runbook.md). The essentials:

- Everything runs as user **loganzhang** (`su - loganzhang`), clone at
  `~loganzhang/Documents/cpg/eigenbot_isaaclab_cpg`.
- `conda activate env_isaaclab` in **every** new shell — a `(base)` prompt is the
  tell. Add `export DISPLAY=:0` for anything with a viewer.
- boa runs **Isaac Lab 2.0.1 / rsl-rl 2.1.2**, older than this repo targets. The
  API differences are re-applied by `bash scripts/boa_compat.sh` — the script is
  committed, but **the edits it makes are machine-local and must never be
  committed**. Any new `TypeError: unexpected keyword argument` on boa is the
  same version mismatch.
- Every `git pull` / `checkout` wipes the patches — just re-run the script:

```bash
bash scripts/boa_compat.sh
```

- Two RTX 3080 12 GB. Check `nvidia-smi`, pick a free one with `--device cuda:0`
  or `cuda:1`, and run **one Isaac Sim process per GPU**.
- **Never `git stash` on boa** — it takes the compat patches with it.

</details>

## Troubleshooting

<details>
<summary>Every failure mode hit so far, and the fix</summary>

| Symptom | Cause / fix |
|---|---|
| Robot invisible / black viewer in Docker | X11 authorization. Run `xhost +local:` on the host, and export `DISPLAY` inside the container. Also check camera framing — the default view can put the robot off-screen |
| `TypeError: unexpected keyword argument …` on the lab PC | Isaac Lab / rsl-rl version mismatch. Run `bash scripts/boa_compat.sh` |
| Crash on env construction with a shape mismatch | A non-18 action space reaching `_init_buffers`. See [Adding a new policy](#adding-a-new-policy) |
| `"Scene state is corrupted"`, CUDA error 700/2 | Two Isaac Sim processes on one GPU. One job per GPU; 2048 envs leaves headroom |
| Ctrl+C does nothing | Isaac Sim ignores it. `pkill -9 -f train.py` from another terminal |
| tmux detach doesn't work in VS Code | VS Code steals `Ctrl+B`. Use `tmux detach-client -s <name>` from another terminal, or launch detached with `tmux new -d` + `send-keys` and never attach |
| TensorBoard doesn't show a new run | It doesn't pick up runs created after it started if it has been up for days. Restart it or use a new port |
| Import still points at the old clone | `pip install -e source/eigenbot` re-points the `eigenbot` package only for **new** processes. Launch each run from the clone you intend it to use |
| Everything looks broken in a screenshot | A tripod gait always has three legs airborne. Judge motion, not stills |
| `.git` corruption, vanishing `.git/config`, stale `index.lock` | OneDrive. Keep working clones **outside** OneDrive; sync through GitHub |
| Reward is high but the robot isn't going anywhere | Believe the video, not the number. See [Results](#results-so-far) |

</details>

## Results so far

<details>
<summary>The measured gait, the first comparison, and the reward exploit it exposed</summary>

Full history, every fix and why: [`docs/experiment_log.md`](docs/experiment_log.md).

### Measured open-loop CPG gait (flat ground, zero action, 1.0 Hz)

Per-leg clearance 1.3 / 2.6 / 2.7 / 2.0 / 2.2 / 2.6 cm, stance 64–78 %, travel
0.50 m in 6 s (**0.083 m/s**), lateral drift ~0.00 m, body height ~11.5 cm — all
six legs clearing and the body moving.

### First comparison (2026-08-01, 10k iterations, 2048 envs, seed 42, mixed terrain)

Final mean reward **95.42 (CPG) vs 95.19 (PPO)** — a statistical tie, both with
episode length pinned at the maximum. But `play.py` showed they were not doing
the same thing at all: **the CPG genuinely locomoted, while the 18-DOF PPO
baseline vibrated its legs in place with no forward progress.**

The reward function was not discriminating locomotion. With
`tracking_goal_vel = 4.0` alongside generous survival terms and
`only_positive_rewards = True`, staying upright and cycling the legs banked most
of the available reward without travelling.

**This is itself the most interesting result so far: the CPG's structural gait
prior made it immune to a reward exploit that the unconstrained 18-DOF policy
fell straight into.**

### Second comparison (launched 2026-08-08)

`tracking_goal_vel` 4.0 → 10.0, `lin_vel_x` (0.0, 0.5) → (0.2, 0.4), CPG `b_max`
1.5 → 1.25 plus a joint-limit clamp. Both tasks share the reward changes, so the
comparison stays fair. **Expect lower absolute rewards than 95** — the scale
changed and standing no longer pays. The signal to look for is whether a *gap*
opens.

**Not done yet:** `eval_compare.py` has not been run on either policy. That is
the next step.

</details>

## Open threads

<details>
<summary>What to do next, in priority order</summary>

Roughly in priority order:

1. **Run `eval_compare.py` on both second-comparison policies.** The head-to-head
   numbers are the deliverable and nothing else is blocked on them.
2. **Per-term reward logging** (`extras["episode"]["rew_*"]`) so reward
   composition shows up in TensorBoard. Cheap; would have saved two 10k runs.
3. **Commit the 1.0 Hz gait frequency** — the biggest single gait improvement
   still only exists in the lab PC's working tree.
4. **Frequency as an action** (7 → 8 dims, an ω scale of ~0.6–2.0×). Addresses
   the 0.083 m/s speed ceiling directly; stride has little room to grow because
   the front and rear swing joints start at ±π/4 against a ±π/2 limit (the mid
   pair starts at 0), so frequency is the practical lever. Matches Bellegarda &
   Ijspeert's CPG-RL.
5. **Entropy / std control** — `entropy_coef` below 0.002 or an std ceiling; the
   action noise std reached ~10–13 by the end of both first-comparison runs.
6. **Residual mode** (`cpg.use_residual = True`, action space 25) as the
   flexibility hedge.
7. **ARS trainer** (`scripts/ars/train.py`) — the paper's gradient-free
   optimizer, implemented but never run.
8. **Perception hook** — the six per-leg `b` gains are the intended interface for
   terrain features from the vision side of the project.
9. **`base_height_target = 0.25` is inert and misleading.** There is no
   base-height reward term at all — the field is read only by `diag_gait.py`,
   for a printed note — and the value is unreachable anyway (the robot rides
   ~11.5 cm). Wire up a height term with a reachable target, or delete the field.
10. **ROS 2 policy inference node** — the remaining piece before deployment to the
    physical robot.

### Bugs found in the 2026-09-04 documentation pass

11. **Foot slots are not order-pinned.** `find_bodies(FEET_BODIES)` is called
    *without* `preserve_order=True`, so the six contact slots come back in
    articulation order, not `FEET_BODIES` order. `_reward_rule_1` and
    `_reward_rule_3` treat slots `[0,3]` as front, `[1,4]` as middle and `[2,5]`
    as hind, which does not match the URDF (foot M25 belongs to a mid leg, M26 to
    a rear leg). **The gait-rule rewards are probably not rewarding the
    coordination they claim to.** Check before any writeup leans on them.
12. **Observation noise is never applied** — `NoiseCfg` / `add_noise` /
    `noise_scales` are defined and unread. Sim-to-real relevant.
13. **`scripts/test_cpg.py`'s stub config has drifted from `CPGCfg`**
    (`b_max = 1.5` vs `1.25`, `lift_scales[0] = 1.35` vs `1.7`), so the offline
    checks no longer exercise the committed values.
14. **Other dead config:** the `ang_vel_yaw` command range is never sampled
    (`commands[:, 2]` is derived from heading error every step);
    `soft_dof_vel_limit`, `soft_torque_limit`, `max_contact_force` and
    `stumble_tresh` are unread.

</details>

## Conventions

<details>
<summary>Branches, what never to commit, OneDrive, line endings</summary>

- **Branches.** `main` is the documented trunk. Feature work goes on a branch and
  merges back via PR. (`cgp-rl`, a typo of `cpg-rl`, is dead — ignore it.)
- **Never commit the machine-local compat edits** made by
  `scripts/boa_compat.sh` — they break on newer Isaac Lab. (The script itself is
  tracked; its output is not.)
- **Never commit** `__pycache__/`, `*.pyc`, checkpoints or `logs/`.
- **Keep working clones outside OneDrive.** OneDrive has corrupted `.git` in this
  repo more than once (stale `index.lock`, vanishing `.git/config`).
- **Line endings.** Editing on Windows can rewrite whole files as CRLF and turn a
  one-line change into a 1000-line diff. Check the diff before committing.
- **When you change gait or reward behaviour, update the numbers in
  [`docs/experiment_log.md`](docs/experiment_log.md)** — that file is the record
  of what was tried and what it did, and it is the first thing the next person
  will read.

</details>

---

## Reference

- Li, Wei & Qiu (2023), "Combined Reinforcement Learning and CPG Algorithm to
  Generate Terrain-Adaptive Gait of Hexapod Robots", *MDPI Actuators* 12(4):157 —
  the CPG+RL method this project reproduces and compares.
- Bellegarda & Ijspeert, *CPG-RL* — the joint-space mapping and the
  frequency-as-an-action idea.
- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/)
