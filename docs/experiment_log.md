# Experiment log

The record of what was tried, what it did, and what the numbers were. This is the
first file the next person should read after the [README](../README.md) — the
code says *what* the configuration is, and this says *why*.

**Keep it current.** When you change a gait or reward parameter and run
something, add a dated entry. A value with no story behind it will be "cleaned
up" by somebody in six months.

---

## Timeline

| Date | Event |
|---|---|
| 2026-06 | Isaac Lab port of the legged_gym env; terrain, raycast height scan, domain randomization |
| 2026-06-26 | First working baseline PPO checkpoint (`eigenbot_locomotion/2026-06-26_17-52-43/model_19999.pt`) |
| 2026-07 | CPG+RL task added (`Template-Eigenbot-CPG-Direct-v0`), offline test suite, implementation guide |
| 2026-07-29 | Gait fixed: distal joint sign, 1.0 Hz frequency, per-leg lift scaling |
| 2026-08-01 | **First comparison** (10k vs 10k) — tie on reward, exposed a reward exploit |
| 2026-08-08 | Reward re-weighting + `b_max` cap + joint-limit clamp; **second comparison** launched |
| 2026-09-04 | Documentation pass; `eval_compare.py` head-to-head still outstanding |

---

## Fixes, and what each one was reacting to

### Buffer crash on construction (CPG task)

The parent `_init_buffers` sizes every joint-dimension buffer from
`cfg.action_space` and then runs domain randomization against `(ne, 18)` robot
data — so a 7-D action space crashed on a shape mismatch.
**Fix:** `EigenbotCPGEnv._init_buffers` swaps `cfg.action_space` to 18 around the
`super()` call and restores it after. Any future non-18 action space needs the
same.

### Joint mapping (critical — verified in sim)

Articulation order is **breadth-first**, so leg *k* owns joints
`(k, k+6, k+12)` = modules M(k+1), M(k+7), M(k+13):

```python
leg_joint_indices = ((0,6,12),(1,7,13),(2,8,14),(3,9,15),(4,10,16),(5,11,17))
leg_sides         = (-1,-1,-1, 1, 1, 1)     # M1–M3 one side, M4–M6 the other
```

M1/M4 rear, M2/M5 mid, M3/M6 front. URDF chain:
`body → M(k) → M(k+6) → static elbow → M(k+12) → foot`.

Related: an earlier bug came from `find_bodies` returning indices in a different
order than the name list. Always pass `preserve_order=True`.

### Gait direction — the robot walked backwards

`lift_phase_sign = -1.0` (lift during the protraction half-cycle). With `+1.0`
the leg strokes forward on the ground and back through the air.

### Distal joint sign (2026-07-29)

`lift_joint_signs (1.0, -0.5) → (1.0, 0.5)`. The URDF leaves M(k+12)'s axis
**antiparallel** to M(k+6)'s, so same-signed commands counter-bend the leg —
which is why the default pose sets M7–M18 all to +π/4. The old opposite-signed
pair made the shank trail the thigh. Symptom: *"the second half of each leg
doesn't move forward"*, followed by the loaded rear legs slipping flat and never
recovering. Verify with `scripts/diag_gait.py`; if lift still looks wrong after a
URDF change, try `(1.0, 1.0)` or `(-1.0, -0.5)`.

### Gait frequency (2026-07-29) — the biggest single improvement

`omega` 1.5 Hz → **1.0 Hz**. At 1.5 Hz the PD could not track the commanded lift
within a half-cycle under load, so the loaded legs never broke contact (rear
stance ~90%). Dropping to 1.0 Hz raised every leg's clearance.

> ⚠️ **This change was never committed.** `eigenbot_cpg_env_cfg.py` still reads
> `omega = 2π·1.5`. It exists only in the lab PC's working tree. Confirm with
> `diag_gait.py --omega_hz 1.0` and commit it.

### Per-leg lift scaling

`lift_scales = (1.7, 1.0, 1.0, 1.35, 1.0, 1.0)`. Legs 0 and 3 are the rear pair
(attachment origins at **z** = −0.13 vs +0.05 for the front pair; x is −0.041 for
every leg), carry the most load, and dragged under uniform lift.

**The joint PD gains (stiffness 20, damping 0.5, effort 8 N·m) mirror the
physical modules. They were raised once to fix this and deliberately reverted.
`lift_scales` is the correct lever; the gains are not.**

### Saturation fix (after a 20k-iteration run failed)

Noise std grew to ~2.8–3 against a ±1 clip range → bang-bang actions, zero
gradient through the hard clamp, policy mean parked at the extremes.
**Fix:** `tanh` instead of a hard clamp in `_decode_action`; no pre-clipping in
`_pre_physics_step` (just `nan_to_num`); residual clamp → `tanh`;
`init_noise_std` 1.0 → 0.4; `entropy_coef` 0.01 → 0.002.

### Euler guard in `cpg.py`

Radial contraction rate clamped, so a large oscillator kick cannot NaN the state
at α = 100, dt = 0.005.

### Joint-limit clamp + `b_max` cap (2026-08-08)

`b_max` 1.5 → **1.25**, plus a hard clamp of `default_dof_pos + offsets` to
`dof_pos_limits` in `EigenbotCPGEnv._apply_action`. At b = 1.5 the swing offset
(0.675 rad) plus the ±π/4 default stance exceeded the ±π/2 joint limit and folded
the front legs. Measurements also showed b > 1 *slowed* the robot (more rear
load), so the top of the old range was useless anyway.

### Reward re-weighting (2026-08-08, shared by both tasks)

`tracking_goal_vel` 4.0 → **10.0**; command range `lin_vel_x` (0.0, 0.5) →
**(0.2, 0.4)**. Reasoning under "First comparison" below.

---

## Measured open-loop gait

`scripts/diag_gait.py`, flat ground (`terrain_type="plane"`), zero action, final
tuned config at **1.0 Hz**:

| Metric | Value |
|---|---|
| Foot clearance, legs 0–5 | 1.3, 2.6, 2.7, 2.0, 2.2, 2.6 cm |
| Stance fraction | 64–78 % |
| Body travel | 0.50 m in 6 s → **0.083 m/s** |
| Lateral drift | ~0.00 m |
| Body height | ~11.5 cm |

Verdict line: *"all six legs clearing and the body is moving."*

Two standing observations from this measurement:

**Speed ceiling.** ~0.083 m/s at the default mid gain (zero action → b = 0.875),
and the stride has little room to grow — the front and rear swing joints start at
±π/4 against a ±π/2 limit, so `swing_amplitude × b` caps near 0.78 rad there
(the mid pair starts at 0 and has more headroom). A 1.5× amplitude test made things *worse* (rear clearance 0.6 cm,
travel 0.22 m). Frequency is the only real lever and it trades against clearance.
The principled fix is to give the policy **frequency authority** (action space
7 → 8, ω scale ~0.6–2.0×), as in Bellegarda & Ijspeert's CPG-RL.

**Body height.** The robot rides ~11.5 cm against `base_height_target = 0.25`,
inherited from legged_gym. Note that **no reward term reads that field** — there
is no `_reward_base_height` — so nothing is actually penalised; `diag_gait.py`
just prints it as a reference. Either wire up a height term with a reachable
target or delete the field, and footnote the ride height in any writeup.

---

## Terrain

`EIGENBOT_ROUGH_TERRAIN_CFG` mixes four sub-terrains — flat 20% / random_rough
40% / slopes 20% / obstacles 20% — on a 10×20 grid of 8×8 m patches. One policy
per method trains across all of them simultaneously; the row-wise curriculum in
`EigenbotEnv._update_terrain_curriculum` promotes robots as they cover distance.
Verified visually in the viewer.

Use `terrain_type="generator"` for training and `"plane"` only for gait
inspection.

---

## FIRST COMPARISON — 2026-08-01

**Setup:** 10k iterations, 2048 envs, seed 42, mixed terrain, identical settings.

| Policy | Checkpoint |
|---|---|
| CPG | `logs/rsl_rl/eigenbot_cpg_locomotion/2026-08-01_01-31-28/model_9999.pt` |
| PPO baseline | `logs/rsl_rl/eigenbot_locomotion/2026-08-01_07-57-24/` |

**Result: final mean reward 95.42 (CPG) vs 95.19 (PPO)** — statistically a tie,
both with mean episode length pinned at 1249 (the maximum; almost nothing falls).

**But `play.py` showed the two are not doing the same thing at all:**

- **CPG genuinely locomotes** — forward progress, rhythmic tripod. One front leg
  folded, since traced to the joint-limit issue and fixed by the `b_max` cap plus
  the `dof_pos_limits` clamp.
- **PPO baseline vibrates its legs rapidly in place with no forward progress.**

### Conclusion: the reward function was not discriminating locomotion

With `tracking_goal_vel = 4.0` alongside `feet_air_time 0.87`, `delta_yaw 1.2`,
`rule_1 0.35` and `only_positive_rewards = True`, a policy that stays upright and
cycles its legs banks most of the available reward without travelling. Commands
below `lin_vel_clip = 0.1` are also zeroed, so "stand still" satisfied the
velocity term outright on a sizeable fraction of episodes.

**This is itself a reportable finding: the CPG's structural gait prior made it
immune to a reward exploit that the unconstrained 18-DOF policy fell straight
into.**

Two process lessons:

- **No `Episode/rew_*` per-term scalars are logged by this env**, so the reward
  could not be decomposed in TensorBoard. Adding that logging would have caught
  the exploit in minutes rather than after two 10k runs.
- **Reward totals can hide everything.** Always watch `play.py` before believing
  a number.

**Caveat carried forward:** action noise std reached ~10–13 by the end of both
runs (tanh saturates, the entropy bonus keeps inflating it). The deterministic
policy used at play time still behaves, but `entropy_coef` may need to go below
0.002.

---

## SECOND COMPARISON — launched 2026-08-08

Changes made in response (all committed; both tasks share the reward changes so
the comparison stays fair):

- `tracking_goal_vel` 4.0 → 10.0
- `lin_vel_x` command range (0.0, 0.5) → (0.2, 0.4)
- CPG `b_max` 1.5 → 1.25 + joint-limit clamp in `_apply_action`

Launched sequentially (2048 envs, 10k iterations, seed 42, ~6.5 h each; the
baseline chained to auto-start when the CPG process exits — see
[`boa_runbook.md` §5](boa_runbook.md#5-tmux)).

**Expect lower absolute rewards than 95** — the scale changed and standing no
longer pays. The signal to look for is whether a *gap* opens between the two.

### What to do when they finish

1. `play.py` both policies. Confirm the CPG front leg no longer folds and that
   the baseline actually travels.
2. Run the head-to-head:

```bash
python scripts/eval_compare.py --task Template-Eigenbot-CPG-Direct-v0 --episodes 40 --num_envs 40 --command_vel 0.3 --seed 123 --headless --out logs/eval_cpg.csv
```

```bash
python scripts/eval_compare.py --task Template-Eigenbot-Direct-v0 --episodes 40 --num_envs 40 --command_vel 0.3 --seed 123 --headless --out logs/eval_ppo.csv
```

   Per-terrain forward distance, early-termination rate, roll/pitch std, lateral
   drift. Keep the flags identical between the two.
   (The script has progress prints; it previously appeared to hang before the
   rollout loop with no output — if that recurs, the prints will localise it.)
3. Compare TensorBoard curves against **iterations** and against **wall-clock** —
   the latter is the data-efficiency claim.
4. Record the numbers here.

**Status as of 2026-09-04: step 2 has not been run yet.** It is the top open item.

---

## Ideas and open threads, roughly in priority order

1. **Run the `eval_compare.py` head-to-head.** Nothing else is blocked on it, and
   it is the deliverable.
2. **Frequency as an action** (7 → 8 dims, ω scale ~0.6–2.0×). Directly addresses
   the 0.083 m/s speed ceiling, lets RL trade speed against clearance per
   terrain, and matches Bellegarda & Ijspeert. Biggest upside available.
3. **Per-term reward logging** (`extras["episode"]["rew_*"]`) so reward
   composition is visible in TensorBoard.
4. **Commit the 1.0 Hz frequency** — the biggest gait win is uncommitted.
5. **Entropy / std control** — `entropy_coef` below 0.002, or an std ceiling, to
   stop the noise blow-up.
6. **Residual mode** as the flexibility hedge: `cpg.use_residual = True` (action
   space 25). Prefer the PPO pipeline there.
7. **ARS trainer** (`scripts/ars/train.py`) — the paper's gradient-free
   optimiser, implemented but never run; needs no rsl-rl.
8. **Perception hook** — the six per-leg `b` gains are the intended interface for
   terrain features from the vision side of the project.
9. **`base_height_target = 0.25` is inert** — no base-height reward term exists;
   only `diag_gait.py` reads the field. Wire one up with a reachable target, or
   delete it.
10. **ROS 2 policy inference node** — the remaining piece before deployment. A
    ROS 2 Humble Docker environment with all six nodes is already up on boa.

---

## Bugs found in the 2026-09-04 documentation pass

Found by reading the code against the docs, none of them yet fixed. Listed
roughly by how much they could distort a result.

1. **Foot slots are not order-pinned.** `_resolve_body_indices` calls
   `find_bodies(FEET_BODIES)` *without* `preserve_order=True`, so the six contact
   slots come back in articulation order rather than in `FEET_BODIES` order.
   `_reward_rule_1` and `_reward_rule_3` group them as front `[0,3]`, middle
   `[1,4]`, hind `[2,5]`, which does not match the URDF — foot M25 belongs to a
   mid leg and M26 to a rear leg. **The gait-coordination rewards are very
   probably not rewarding the coordination they describe.** Same class of bug as
   the earlier `preserve_order` issue, in a different place.
2. **`eval_compare.py` makes evaluation harder than training.**
   `max_init_terrain_level = None` means *no cap*, so evaluation envs are seeded
   across all 10 difficulty rows while training used 0–5; the terrain curriculum
   also keeps running during the rollout. `commands.curriculum = False` is a
   no-op — nothing reads that field. Fix before publishing per-terrain numbers.
3. **Observation noise is never applied.** `NoiseCfg`, `add_noise` and
   `noise_scales` are defined in the config and read nowhere. Directly relevant
   to sim-to-real robustness.
4. **`base_height_target` is inert** — see above.
5. **`self.torques` is bookkeeping, not actuation.** Nothing calls
   `set_joint_effort_target`; joints are driven by the implicit actuator, whose
   `effort_limit_sim = 8.0` is the real limit. So `torque_limit_hard` shapes the
   `torques` *penalty* only, and motor-strength randomization currently perturbs
   the penalty rather than the dynamics.
6. **`scripts/test_cpg.py`'s stub config has drifted from `CPGCfg`**
   (`b_max = 1.5` vs `1.25`; `lift_scales[0] = 1.35` vs `1.7`), so the offline
   checks no longer test the committed values.
7. **`ang_vel_yaw` command range is never sampled** — `commands[:, 2]` is
   overwritten from the heading error every step.
8. **Unread config fields:** `soft_dof_vel_limit`, `soft_torque_limit`,
   `max_contact_force`, `stumble_tresh`.
9. **Two wrong source comments:** `CPGCfg.lift_scales` says the rear/front
   attachment difference is in `x` (it is `z`; x is −0.041 for every leg), and
   `cpg.py`'s docstring calls the coupling a "ring" when it is all-to-all.

---

## Gotchas learned the hard way

- **OneDrive + git = corruption** (stale `index.lock`, vanishing `.git/config`).
  Keep clones outside OneDrive; sync via GitHub.
- **`pip install -e source/eigenbot` re-points the `eigenbot` package for NEW
  processes only** — launch each run from the clone you intend it to use.
- **Don't commit `__pycache__` / `*.pyc`** (they showed up in GitHub Desktop
  once).
- **`zero_agent` on the CPG task = θ=0, b=(b_min+b_max)/2 → pure CPG gait.** The
  single most useful eyeball test after any mapping or sign change.
- **A still frame of a tripod always looks broken** (three legs airborne). Judge
  motion.
- **Never `git stash` on boa** — it takes the compat patches with it.
- **Every new shell needs `conda activate env_isaaclab`** (and `export DISPLAY=:0`
  for a viewer). A `(base)` prompt is the tell.
- **One Isaac Sim process per GPU.** A second job starved the first and corrupted
  its PhysX scene mid-run. 2048 envs leaves headroom on a 12 GB card.
- **Isaac Sim ignores Ctrl+C** — `pkill -9 -f <script>.py` from another terminal.
- **VS Code steals Ctrl+B**, so tmux detach doesn't work there. Launch with
  `tmux new -d` + `send-keys` and never attach.
- **TensorBoard doesn't pick up runs created after it started** if it has been
  running for days.
- **Reward totals can hide everything.** Two policies scored 95 while one walked
  and the other vibrated in place.
