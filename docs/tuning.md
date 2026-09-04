# Tuning guide — every knob, where it lives, and what it does

Companion to the [knob index in the README](../README.md#tuning-the-knob-index),
which is the quick lookup. This file adds the *reasoning*: why the current values
are what they are, what happened when they were something else, and which knob is
the right one for a given symptom.

Path shorthand:
**`TASK`** = `eigenbot/eigenbot/source/eigenbot/eigenbot/tasks/direct/eigenbot/`
**`EXT`** = `eigenbot/eigenbot/`

> **Ground rule.** `eigenbot_env_cfg.py` is shared by *both* tasks. Anything you
> change there changes the baseline PPO task and the CPG task together. That is
> deliberate — it is what keeps the head-to-head comparison honest — but it means
> a mid-study reward edit invalidates every checkpoint trained before it. Record
> the change in [`experiment_log.md`](experiment_log.md) when you make one.

---

## 1. Rewards — `TASK/eigenbot_env_cfg.py`

### Scales: `RewardScalesCfg`

Every field is multiplied by the policy dt (0.02 s) internally, so the numbers
here are *per second*. A scale of `0.0` disables the term entirely (its function
is never called).

```python
termination        = -1.0     tracking_goal_vel  = 10.0    delta_yaw       =  1.2
lin_vel_z          = -1.0     ang_vel_xy         = -0.05   orientation     = -0.25
torques            = -0.0002  dof_vel            =  0.0    dof_acc         = -2.5e-7
feet_air_time      =  0.87    collision          = -0.25   stumble         = -0.25
action_rate        = -0.01    stand_still        = -0.5    dof_pos_limits  = -1.0
rule_1             =  0.35    rule_3             =  0.1
```

**`tracking_goal_vel` is the one that decides whether the robot walks.** It was
`4.0` for the first comparison. At that weight, the survival-flavoured terms
(`feet_air_time` 0.87 + `rule_1` 0.35 + `rule_3` 0.1 + `delta_yaw` 1.2) added up
to more reward than travelling was worth, and the 18-DOF policy learned to
vibrate its legs in place for ~95 reward. Raising it to `10.0` was the fix. If
you lower it again, expect the exploit back.

**`torques = -0.0002`** is an EigenBot-specific override — 20× the legged_gym
base of `-1e-5` — because the bendy modules are weak (8 N·m) and an unpenalized
policy will happily saturate them.

### Adding a term

Add a field to `RewardScalesCfg` and a method `_reward_<field>(self)` on
`EigenbotEnv` returning a `(num_envs,)` tensor. The dispatch in
`_prepare_reward_functions` picks it up automatically. Return the *unscaled*
quantity — the scale and dt multiply are applied for you.

### Shaping parameters: `RewardsCfg`

| Field | Value | Effect |
|---|---|---|
| `only_positive_rewards` | `True` | Clips the per-step total to ≥0 **before** the terminal penalty. See the trap in [`architecture.md` §5](architecture.md#5-rewards) |
| `tracking_sigma` | `0.25` | Width of the velocity-tracking exponential. Smaller = sharper = harder |
| `soft_dof_pos_limit` | `0.9` | Fraction of the URDF joint range treated as "legal". EigenBot override |
| `soft_dof_vel_limit` | `2.5` | EigenBot override. **Currently unread** — as are `soft_torque_limit` and `max_contact_force` |
| `base_height_target` | `0.25` | **Inert** — there is no base-height reward term; only `diag_gait.py` reads this, for a printed note. The robot rides ~0.115 m anyway |
| `contact_tresh` | `0.5` | Vertical force (N) that counts as foot contact |
| `stumble_tresh` | `2.5` | Unused by the current `_reward_stumble` (it hard-codes 5×) |
| `exp_coeff_rule3` | `-10.0` | Decay rate of the rule-3 timing reward |
| `torque_limit_hard` | `8.0` | Clamp on `self.torques`, the PD *estimate* the `torques` reward uses. The actual effort limit is `effort_limit_sim = 8.0` on the implicit actuator — keep the two in step. **Mirrors the physical module** |

---

## 2. Commands — `CommandsCfg` / `CommandRangesCfg`

```python
lin_vel_x   = (0.2, 0.4)      # m/s forward
lin_vel_y   = (0.0, 0.0)      # no strafing
ang_vel_yaw = (-1.0, 1.0)
heading     = (-pi/3, pi/3)
resampling_time = 10.0        # s
lin_vel_clip    = 0.1         # commands below this are zeroed
```

The `lin_vel_x` floor is `0.2`, not `0.0`, on purpose: `lin_vel_clip` zeroes
anything under 0.1 m/s, so a range starting at zero handed a large fraction of
episodes a "stand still" command that the velocity term rewarded for free.

`ang_vel_yaw` is **never sampled**. `_resample_commands` only draws `lin_vel_x`
and the heading; `commands[:, 2]` is overwritten from the heading error every
step. Steering is entirely heading-driven, and changing that range does nothing.

`0.4 m/s` is an aspiration, not a measured capability — the open-loop gait tops
out near **0.083 m/s** (see §5). Every episode is currently asking for roughly
3–5× what the gait can deliver. That is fine as a gradient direction, but do not
read "tracking reward is low" as "the policy is bad".

---

## 3. Terrain — `EIGENBOT_ROUGH_TERRAIN_CFG` + `EigenbotEnvCfg.terrain`

Change the mix in `sub_terrains` (proportions need not sum to 1; they are
normalized). Grid is `num_rows = 10` (difficulty) × `num_cols = 20` (type),
patches `size = (8, 8)` m.

```python
terrain_type = "generator"   # training
terrain_type = "plane"       # flat ground, for diag_gait.py and eyeballing
```

On the lab PC, `bash scripts/boa_compat.sh plane|generator` flips this for you.

`max_init_terrain_level = 5` **caps** the starting difficulty: envs are seeded
uniformly across rows 0–5 of 10. The promotion / demotion rule lives in
`EigenbotEnv._update_terrain_curriculum`.

**Making terrain harder is usually the wrong first move.** If the robot is
failing, check that it can walk on `plane` first with `diag_gait.py`.

---

## 4. Domain randomization — `DomainRandCfg`

| Knob | State | Notes |
|---|---|---|
| `randomize_friction` | on, `(0.5, 1.25)` | The most important one for sim-to-real |
| `randomize_base_com` | on, `(-0.2, 0.2)` m | |
| `randomize_motor` | on, `(0.8, 1.2)` | Scales `Kp`/`Kd` per joint per env |
| `push_robots` | on, 15 s, ≤1 m/s | Recovery robustness |
| `randomize_base_mass` | **off**, `(-1, 1)` kg | Turn on before hardware transfer |
| `action_delay` | **off**, buffer 8 | Simulates actuator latency. Implemented; enabling it costs nothing but training time |

Turning DR *up* makes training slower and the policy more conservative. Turning
it *off* is a legitimate debugging step when diagnosing a gait — just remember to
put it back before drawing a sim-to-real conclusion.

---

## 5. The gait (CPG task only) — `TASK/eigenbot_cpg_env_cfg.py` → `CPGCfg`

This is the section that gets edited most. Full derivation of the mapping is in
[`cpg.md`](cpg.md).

### Frequency — `omega`

```python
omega = 2.0 * math.pi * 1.5   # committed value: 1.5 Hz
```

> ⚠️ **The tuned gait used 1.0 Hz and that change was never committed.** At
> 1.5 Hz the PD controller cannot track the commanded lift within a half-cycle
> under load — the loaded rear legs never break contact (stance ~90%). Dropping
> to 1.0 Hz raised every leg's clearance and was **the single biggest gait
> improvement of the project**. Verify with
> `python scripts/diag_gait.py --headless --omega_hz 1.0` and commit whichever
> value you confirm.

Frequency is also the **practical speed lever**: stride has little room to grow
because the front and rear swing joints start at ±π/4 against a ±π/2 limit, so
`swing_amplitude × b` caps near 0.78 rad there (the mid pair starts at 0 and has
more headroom). A 1.5× amplitude test made things *worse* (rear clearance
0.6 cm, travel 0.22 m). The principled fix is to hand the policy frequency
authority — action space 7 → 8 with an ω scale of ~0.6–2.0× — which is what
Bellegarda & Ijspeert's CPG-RL does. That is open thread #4.

### Amplitudes and gains

| Field | Value | Meaning |
|---|---|---|
| `swing_amplitude` | `0.45` rad | Fore/aft protraction at b = 1 |
| `lift_amplitude` | `0.55` rad | Leg lift during swing at b = 1 |
| `b_min`, `b_max` | `0.5`, `1.25` | Range of the per-leg gains the policy commands |
| `turn_gain` | `0.6` | How strongly θ biases left/right stride |

`b_max` was `1.5`. At that value the swing offset (0.45 × 1.5 = 0.675 rad) plus
the ±π/4 default stance exceeded the ±π/2 joint limit and folded the front legs.
Measurements also showed `b > 1` *slowed the robot down* (more load on the rear
pair), so the top of the old range was useless. It is capped at `1.25`, and
`EigenbotCPGEnv._apply_action` additionally clamps `default_dof_pos + offsets` to
`dof_pos_limits` as a hard backstop.

### Per-leg lift — `lift_scales`

```python
lift_scales = (1.7, 1.0, 1.0, 1.35, 1.0, 1.0)
```

Legs 0 and 3 are the **rear pair** — their URDF attachment origins sit furthest
back along the body axis (`connection_*_attachment` **z** = −0.13 vs +0.05 for
the front pair; the x component is −0.041 for every leg), so they carry the most
body weight and drag with uniform lift. (The comment in `CPGCfg` says `x`; that
is a typo in the source, worth correcting.) Measured at 1.0 Hz, leg 0 was the worst (1.3 cm clearance
/ 78% stance vs ~2.5 cm / 64% for the front legs), so it gets the largest boost.

**`lift_scales` is the correct lever for uneven clearance.** The joint PD gains
(stiffness 20, damping 0.5, effort 8 N·m) mirror the physical modules; they were
raised once to fix exactly this and deliberately reverted. Raising them produces
a gait that will not transfer to hardware.

### Signs and geometry

| Field | Value | What goes wrong if it's off |
|---|---|---|
| `lift_phase_sign` | `-1.0` | Robot walks **backwards**. Selects which half-cycle the legs lift in |
| `lift_joint_signs` | `(1.0, 0.5)` | The shank trails the thigh — "the second half of each leg doesn't move forward", then the loaded rear legs slip flat and never recover. Was `(1.0, -0.5)` |
| `leg_joint_indices` | `((0,6,12),(1,7,13),…)` | Wrong joints move. **Articulation order is breadth-first**: leg *k* = modules M(k+1), M(k+7), M(k+13) |
| `leg_sides` | `(-1,-1,-1,1,1,1)` | Steering turns the wrong way. M1–M3 one side, M4–M6 the other; M1/M4 rear, M2/M5 mid, M3/M6 front |

Why `lift_joint_signs` is `(1.0, 0.5)` and not `(1.0, -0.5)`: the URDF rotates
the distal module's frame so its axis is **antiparallel** to the middle module's
(`connection_12` Rx(π/2), then `connection_18` Rz(π/2)Rx(π/2)). Same-signed
commands therefore rotate the two segments in opposite world directions — which
is what folds the leg into its Z stance, and why the default pose sets M7–M18 all
to +π/4. If lift still looks wrong after a URDF change, try `(1.0, 1.0)` or
`(-1.0, -0.5)`, and confirm with `diag_gait.py --lift_signs "..."` rather than by
eye.

### Oscillator dynamics

`alpha = 100.0` (convergence speed onto the limit cycle), `mu = 1.0` (squared
radius), `coupling_weight = 1.0` (**all-to-all** phase coupling — despite the
"ring" wording in `cpg.py`'s docstring, each oscillator is pulled toward the
average of all five others; 0 disables it),
`phase_offsets = (0, π, 0, π, 0, π)` (tripod). A different gait — wave, ripple —
is a different `phase_offsets` tuple and nothing else.

There is an Euler-integration guard in `cpg.py` that clamps the radial
contraction rate: without it, a large state kick at `alpha=100, dt=0.005`
diverges. Leave it in if you raise `alpha`.

### Residual mode

```python
use_residual  = False
residual_scale = 0.1     # rad of per-joint correction at |action| = 1
```

Turning it on makes the action space 7 + 18 = 25: the policy gets per-joint
corrections on top of the rhythmic backbone, recovering some of PPO's
flexibility. Keep `residual_scale` small or the CPG stops being the thing under
test. Never run.

---

## 6. PPO — `TASK/agents/`

| | `rsl_rl_ppo_cfg.py` (18-D) | `rsl_rl_cpg_ppo_cfg.py` (7-D) |
|---|---|---|
| `experiment_name` | `eigenbot_locomotion` | `eigenbot_cpg_locomotion` |
| hidden dims (actor & critic) | `[1024, 512, 128]` | `[256, 128]` |
| `init_noise_std` | `1.0` | `0.4` |
| `entropy_coef` | `0.01` | `0.002` |
| `max_iterations` | `20000` | `5000` |

Shared: `learning_rate` 3e-4 with adaptive schedule, `gamma` 0.99, `lam` 0.95,
`clip_param` 0.2, `num_learning_epochs` 5, `num_mini_batches` 4,
`num_steps_per_env` 24, `desired_kl` 0.01, `max_grad_norm` 1.0,
`save_interval` 50, `empirical_normalization` False.

**The noise-std problem.** A 20k-iteration CPG run failed when the action noise
std grew to ~2.8–3 against a ±1 clip range: actions went bang-bang, the hard
clamp passed zero gradient, and the policy mean parked at the extremes. Three
changes fixed it — `tanh` instead of a hard clamp in `_decode_action`, no
pre-clipping in `_pre_physics_step`, and `init_noise_std` 1.0 → 0.4 with
`entropy_coef` 0.01 → 0.002. It is not fully solved: std still reached ~10–13 by
the end of the first-comparison runs (the entropy bonus keeps inflating it, and
`tanh` saturation means there is no cost to doing so). The deterministic policy
used at play time still behaves. Next lever is `entropy_coef` below 0.002 or an
explicit std ceiling.

`--max_iterations` on the command line overrides the config, so the config value
is really just a default.

---

## 7. Robot hardware model — `EXT/source/eigenbot/eigenbot/assets/eigenbot.py`

`EIGENBOT_CFG` holds the URDF path, conversion settings, physics properties, the
default standing pose and the actuator model.

```python
stiffness = 20.0      damping = 0.5      effort_limit_sim = 8.0
```

> ⚠️ **These three mirror the physical bendy modules. Do not raise them to fix a
> gait.** They were raised once during gait debugging and deliberately reverted —
> a policy trained against stronger-than-real actuators does not transfer.

Default pose: the swing joints are set by leg *position*, identically on both
sides — M1/M4 (rear) at **−π/4**, M2/M5 (mid) at **0**, M3/M6 (front) at
**+π/4** — and M7–M18 all at **+π/4**, the counter-bend that folds each leg into
its Z stance (the same geometry fact that sets `lift_joint_signs`). Spawn height
0.42 m.

Other fields worth knowing: `merge_fixed_joints=False` and
`enabled_self_collisions=True` (both matter for a modular robot with static
elbow links), and `force_usd_conversion=True` (the URDF is re-converted to USD on
each run — 98 mesh prims when it works).

---

## 8. Sim and episode — `EigenbotEnvCfg`

| Field | Value | Notes |
|---|---|---|
| `sim.dt` | `0.005` | 200 Hz physics |
| `decimation` | `4` | → 50 Hz policy, policy dt 0.02 s |
| `episode_length_s` | `25.0` | 1250 policy steps |
| `action_scale` | `0.25` | Baseline task only — rad per unit action |
| `scene.num_envs` | `4096` | **Override on the command line**; 2048 is the safe number on a 12 GB card |
| `env_spacing` | `4.0` | Ignored when the terrain generator places envs |
| `sim.render_interval` | `4` | |

Changing `decimation` or `sim.dt` changes the reward scale (everything is
multiplied by policy dt) and invalidates comparisons with earlier runs.

---

## 9. Debugging order

When the robot does something wrong, work down this list before touching PPO:

1. **`python scripts/test_cpg.py`** — 9 offline checks, no GPU, catches sign and
   phase errors in seconds.
2. **`zero_agent.py` on the CPG task** — zero action means θ=0 and b at the
   midpoint of `[b_min, b_max]` (0.875), i.e. the pure engineered gait with no
   policy involved. If that looks wrong, no amount of RL will fix it.
3. **`diag_gait.py --headless` on `terrain_type="plane"`** — numbers, not
   eyeballs: per-leg clearance, stance fraction, body travel. Sweep parameters
   from the command line (`--omega_hz`, `--lift_scales`, `--lift_signs`,
   `--swing`, `--lift`) without editing files.
4. **`play.py` on the trained checkpoint** — is it locomoting, or farming reward?
5. **TensorBoard** — and remember the total can hide everything until per-term
   logging exists.
6. **Only then** reward weights and PPO hyperparameters.
