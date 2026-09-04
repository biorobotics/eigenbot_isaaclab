# Architecture — how the EigenBot environment actually works

Companion to the [root README](../README.md). This is the "read before you change
anything structural" document: what runs when, what shape everything is, and why
the pieces are arranged this way.

Path shorthand used throughout:
**`TASK`** = `eigenbot/eigenbot/source/eigenbot/eigenbot/tasks/direct/eigenbot/`

---

## 1. Why a *direct* workflow

Isaac Lab offers two styles: **manager-based** (rewards/observations declared as
composable term objects) and **direct** (one `DirectRLEnv` subclass that computes
everything itself). This project uses **direct**, because it was ported from a
`legged_gym` (Isaac Gym) implementation whose `LeggedRobot` class has the same
shape — the port is close to line-for-line, which made it verifiable against the
original.

The cost is that adding a reward term means editing a class rather than composing
config objects. The mitigation is the reward dispatch described in §5: adding a
scale field plus a matching method is enough, no wiring.

A future port to the manager-based workflow is reasonable once RL is settled and
the ROS migration is done, but it is a rewrite of the RL side, not the sim side.

---

## 2. Class layout

```
DirectRLEnv                        (Isaac Lab)
└── EigenbotEnv                    TASK/eigenbot_env.py       ~1000 lines
    └── EigenbotCPGEnv             TASK/eigenbot_cpg_env.py   ~120 lines

DirectRLEnvCfg                     (Isaac Lab)
└── EigenbotEnvCfg                 TASK/eigenbot_env_cfg.py
    └── EigenbotCPGEnvCfg          TASK/eigenbot_cpg_env_cfg.py
```

`EigenbotCPGEnv` overrides exactly four methods — `_init_buffers`,
`_pre_physics_step`, `_apply_action`, `_reset_idx`. **Everything else —
observations, rewards, terrain, curriculum, domain randomization, termination —
is inherited unchanged.** That inheritance is the experimental control: any
difference between the two policies is a difference in the action space and the
actuation path, not in the task.

Registration happens in `TASK/__init__.py` via `gym.register`, pointing each task
ID at its env class, its env config and its rsl-rl agent config.

---

## 3. The control loop

| Rate | What happens |
|---|---|
| **200 Hz** (`sim.dt = 0.005`) | Physics step. `_apply_action()` is called every substep |
| **50 Hz** (`decimation = 4`) | Policy step. `_pre_physics_step()` stores a new action; observations and rewards are computed once |

`self._dt = sim.dt * decimation = 0.02 s` is the **policy dt**, and it is what
reward scales are multiplied by.

### Baseline actuation path (`EigenbotEnv`)

```
action (18) ──clamp ±1──► target = action_scale·action + default_dof_pos
                          └─► set_joint_position_target(target)   ← this drives the robot
                          └─► self.torques = Kp·(target − q) − Kd·q̇, clamped ±8 N·m
                                                                ← bookkeeping only (see note)
```

`action_scale = 0.25`, so the policy commands at most ±0.25 rad off the default
pose per joint.

### CPG actuation path (`EigenbotCPGEnv`)

```
action (7) ──held constant across the decimation window
   │
   └─► every physics substep:  cpg.step(action, dt=0.005) ──► offsets (18)
                               target = default_dof_pos + offsets
                               target = clamp(target, soft joint limits)   ← stops leg folding
                               same self.torques bookkeeping
```

Two things matter here:

1. **The oscillators are integrated at the physics dt, not the policy dt.** The
   gait is smooth at 200 Hz even though the policy only speaks at 50 Hz.
2. **`self.actions` is set to the 18 joint offsets**, not the 7-D policy action.
   That keeps the observation's "last action" slot 18-dim, so
   `observation_space` is identical for both tasks and the comparison holds.

If domain randomization of motor strength is on, `Kp` and `Kd` are scaled
per-joint per-env by `motor_strength` before the torque computation — in both
paths.

> **`self.torques` does not drive the robot.** Nothing ever calls
> `set_joint_effort_target`. The joints are driven by PhysX's implicit actuator
> (`ImplicitActuatorCfg(stiffness=20, damping=0.5, effort_limit_sim=8.0)`), which
> is what enforces the real effort limit. The explicit PD computation exists so
> the `torques` reward has an effort estimate — and, as a side effect, motor
> strength randomization currently shows up in the *penalty* rather than in the
> dynamics. Worth knowing before you change `torque_limit_hard` expecting a
> behavioural difference.

---

## 4. The observation vector — 974 dims

Assembled in `EigenbotEnv._compute_observations()`. Sizes are constants at the
top of `eigenbot_env_cfg.py`; change one and `NUM_OBSERVATIONS` follows.

| Block | Dims | Contents (in order) |
|---|---|---|
| **proprioception** (`N_PROPRIO`) | **72** | base angular velocity ×`ang_vel` scale (3) · roll, pitch (2) · Δyaw to commanded heading (1) · projected gravity (3) · commanded forward velocity ×`lin_vel` scale (1) · (q − q_default) ×`dof_pos` scale (18) · q̇ ×`dof_vel` scale (18) · flat flag (1) · not-flat flag (1) · last action (18) · foot contacts − 0.5 (6) |
| **height scan** (`N_SCAN`) | **132** | 12×11 ray grid, 0.15 m spacing, centred 0.375 m ahead of `base_link`, yaw-aligned. Value = `clamp(base_z − 0.3 − ray_hit_z, −1, 1)`. Rays that miss fall back to base height |
| **privileged explicit** (`N_PRIV`) | **9** | base linear velocity in body frame ×`lin_vel` scale (3) + two zero blocks (6), kept for shape compatibility with the legged_gym original |
| **privileged latent** (`N_PRIV_LATENT`) | **41** | added mass + COM offset (4) · friction coefficient (1) · motor strength P − 1 (18) · motor strength D − 1 (18) |
| **history** | **720** | the last `HISTORY_LEN = 10` proprioception frames, flattened |

`72 + 132 + 9 + 41 + 720 = 974`.

Notes:

- The **flat / not-flat flags** come from the height scan
  (`|mean(measured_heights)| > 0.05`), and `flat_tensor` also gates the
  `orientation` reward — the robot is only asked to stay level on flat ground.
- Privileged blocks are fed to the same MLP as everything else. The original
  Isaac Gym implementation used dedicated `scan_encoder` / `priv_encoder`
  networks; reproducing that needs a custom rsl-rl policy class and has not been
  done here.
- The whole vector is `nan_to_num`'d and clipped to ±`clip_observations` (100),
  and proprioception is sanitized *before* entering the history buffer so a
  single NaN cannot persist for ten frames.
- On reset the history is filled with ten copies of the current frame rather than
  zeros.

---

## 5. Rewards

### The dispatch

`_prepare_reward_functions()` walks every field of `RewardScalesCfg`, skips
zeros, multiplies each scale by the policy `dt`, and binds it to the method named
`_reward_<field>`. So:

> **Adding a reward term = add a field to `RewardScalesCfg` + add a
> `_reward_<name>(self)` method returning a per-env tensor.** Nothing else.
> A missing method prints a warning and is skipped.

### The accumulation, in order

```python
total = Σ  scale[name] · _reward_name()          # every non-zero, non-termination term
if only_positive_rewards:  total = clamp(total, min=0)
total += scale["termination"] · _reward_termination()
```

`episode_sums[name]` accumulates each term for logging — **but note that the env
does not currently push these into `extras["episode"]`, so per-term rewards do
not appear in TensorBoard.** Adding that is the single highest-value small change
in the repo; without it the only visible signal is the scalar total, and a total
can hide everything (see [`experiment_log.md`](experiment_log.md)).

### The `only_positive_rewards` trap

Clipping the total to ≥0 before the terminal penalty means **penalties can never
push an episode's step reward below zero**. A policy that survives and cycles its
legs collects the survival-flavoured terms (`feet_air_time`, `rule_1`, `rule_3`,
`delta_yaw`) at full value while paying nothing for going nowhere. That is
exactly the failure the first comparison exposed. If you weaken
`tracking_goal_vel` again, expect it back.

### Term-by-term

| Term | Function does | Notes |
|---|---|---|
| `tracking_goal_vel` | `exp(−(v·ĥ − v_cmd)² / tracking_sigma)` where `ĥ` is the commanded heading unit vector | The locomotion signal. Uses **world-frame** velocity projected on the commanded heading |
| `delta_yaw` | `exp(−|wrap(cmd_heading − yaw)|)` | Heading hold |
| `feet_air_time` | `Σ (air_time − 0.5)` at each touchdown, zeroed when `‖cmd‖ < 0.1` | Rewards steps rather than shuffles |
| `rule_1` | Wilson's stepping rules over contact slots: slots `[1,4]` airborne while `[0,3]` are down, and `[2,5]` airborne while `[1,4]` are down | ⚠️ see the ordering caveat below |
| `rule_3` | Exponentially decaying reward on ipsilateral/contralateral liftoff timing | Uses `exp_coeff_rule3 = −10` |
| `termination` | 1 on illegal contact, 0 on timeout | Applied after the clip |
| `lin_vel_z`, `ang_vel_xy`, `orientation` | Bouncing / roll-pitch rate / non-flat base | `orientation` masked by `flat_tensor` |
| `torques`, `dof_vel`, `dof_acc`, `action_rate` | Effort and smoothness | `dof_vel` currently off |
| `collision` | Contact >0.1 N on any of the 18 leg modules | |
| `stumble` | Foot lateral force > 5× vertical | Toe stubbing |
| `stand_still` | `Σ|q − q_default|` when `‖cmd‖ < 0.1` | |
| `dof_pos_limits` | Excursion beyond the soft limits (`soft_dof_pos_limit = 0.9` of the URDF range) | |

Feet are identified by name (`FEET_BODIES`), and the contact indices are resolved
once at construction via `find_bodies`. **Body order in the articulation is
breadth-first, not per-leg** — this bit the project once already.

> ⚠️ **Open issue — the foot slots are not pinned.** `find_bodies(FEET_BODIES)`
> is called *without* `preserve_order=True`, so the six slots come back in
> articulation order rather than in `FEET_BODIES` order. `_reward_rule_1` and
> `_reward_rule_3` then treat slots `[0,3]` as front, `[1,4]` as middle and
> `[2,5]` as hind — which does not line up with the URDF (foot M25 belongs to a
> mid leg, M26 to a rear leg). **The gait-rule rewards are therefore probably not
> rewarding the coordination their docstrings describe.** Add
> `preserve_order=True` and re-derive the slot groups before any writeup relies
> on `rule_1` / `rule_3`.

---

## 6. Commands

Four numbers per env: `[lin_vel_x, lin_vel_y, ang_vel_yaw, heading]`, resampled
every `resampling_time = 10 s` and on reset.

- Heading is drawn as an offset from the robot's *current* heading, within
  ±π/3, so the robot is never asked for a 180° turn from a standstill.
- `heading_command = True` converts the heading error into the yaw-rate command.
- **`lin_vel_clip = 0.1` zeroes any commanded speed below 0.1 m/s.** This is why
  the `lin_vel_x` range floor was raised off zero: with `(0.0, 0.5)`, a sizeable
  fraction of episodes commanded "stand still", which the velocity term rewarded
  for doing nothing.

---

## 7. Terrain and curriculum

`EIGENBOT_ROUGH_TERRAIN_CFG` builds a **10 rows × 20 cols** grid of 8×8 m
patches. Sub-terrains are laid out **along columns**, difficulty **along rows**:

| Sub-terrain | Proportion | Parameters |
|---|---|---|
| `flat` | 20% | — |
| `random_rough` | 40% | noise 0.02–0.10 m, step 0.02 |
| `slopes` | 20% | pyramid, slope 0–0.25, platform 2 m |
| `obstacles` | 20% | 40 discrete boxes, 0.4–1.0 m wide, 0.02–0.08 m tall |

One policy trains across all four simultaneously. `max_init_terrain_level = 5`
**caps** the starting difficulty — envs are seeded uniformly across rows 0–5.

`EigenbotEnv._update_terrain_curriculum()` runs on every reset: an env that
travelled more than half a patch is promoted a row; one that travelled less than
`|cmd_vel| · episode_length · 0.5` is demoted. This is the legacy `legged_gym`
heuristic, unchanged.

`eval_compare.py` sets `max_init_terrain_level = None` and
`commands.curriculum = False`. Be careful about what that actually does: `None`
means *no cap*, so evaluation envs are seeded across **all 10** rows — harder
than the 0–5 they trained on — and the terrain curriculum keeps promoting and
demoting them mid-evaluation. `commands.curriculum` is not read by this env at
all, so setting it changes nothing. Only the fixed velocity range and
`rand_heading = False` take effect. Fix this before publishing per-terrain
numbers.

For gait inspection set `terrain_type="plane"` — `diag_gait.py` measures foot
height in world z, which only equals clearance on flat ground.

---

## 8. Domain randomization

Split across two places: parameters are drawn into buffers in `_init_buffers` /
`_reset_idx`, and pushed into PhysX by `_apply_domain_randomization(env_ids)`.

| Knob | Default | Applied to |
|---|---|---|
| `randomize_friction` | on, 0.5–1.25 | material properties |
| `randomize_base_com` | on, ±0.2 m | body COM |
| `randomize_motor` | on, 0.8–1.2× | `Kp`/`Kd` used in the torque law, and the privileged obs |
| `push_robots` | on, every 15 s, ≤1 m/s | root velocity |
| `randomize_base_mass` | **off** | |
| `action_delay` | **off** | ring buffer of `action_buf_len = 8` policy steps |

All drawn values also appear in the privileged latent observation block, so a
future teacher-student / privileged-distillation setup has what it needs.

Everything is re-randomized on reset, relative to cached defaults captured at
construction (`_default_masses`, `_default_coms`, …) — randomization is never
compounded run to run.

---

## 9. Termination and reset

**Terminated** — any contact force > 1 N on `base_link` (the body touched
something). **Timed out** — `episode_length_s = 25 s` (1250 policy steps).

On reset: joint positions are `default · U[0.5, 1.5]` (note: *scaled*, not offset
— joints whose default is 0 stay at 0), root velocities `U[−0.5, 0.5]` in all six
components, commands resampled, all gait-timing and contact buffers cleared,
observation history refilled, DR redrawn, terrain curriculum updated.

---

## 10. Sensors

- **Contact sensor** on every body (`prim_path=".../Robot/.*"`), history 2,
  `track_air_time=True`. Feet, penalized modules and the termination body are
  index subsets of it.
- **Height scanner** (`RayCaster`) attached to `base_link`, offset 0.375 m
  forward and 20 m up, yaw-aligned only, 12×11 grid at 0.15 m.
- **Depth camera** (`TiledCamera`) — **off by default** (`use_camera = False`).
  When on: 106×60 front-mounted pinhole, 87° HFOV, cropped/resized to 50×50,
  clipped to 2 m, normalized to [−0.5, 0.5], updated every 3 steps into a
  2-frame buffer, with per-env mount pitch randomized ±5°. Enable with
  `--use_camera`. Nothing consumes the buffer yet — it is the hook for the
  perception side of the project.

---

## 11. Known gaps

Carried over from the port, still open:

- **No per-term reward logging.** See §5.
- **No scan/privileged encoders.** The original used dedicated encoder networks;
  this uses a flat MLP over the full 974 vector.
- **Action delay is implemented but disabled.**
- **`base_height_target = 0.25` is inert.** There is no base-height reward term
  — the field is read only by `diag_gait.py`, for a printed note. (The value is
  unreachable anyway: the robot rides ~0.115 m in its sprawled default pose.)
- **Privileged explicit block is 6/9 zeros**, kept only for shape compatibility.
- **Observation noise is dead config.** `NoiseCfg`, `add_noise` and
  `noise_scales` are defined but never read — no noise reaches the observation.
- **Foot slots are not order-pinned** — see the caveat in §5.
- **`ang_vel_yaw` command range is never sampled**; `commands[:, 2]` is derived
  from the heading error each step.
- **Unread config fields:** `soft_dof_vel_limit`, `soft_torque_limit`,
  `max_contact_force`, `stumble_tresh`.
