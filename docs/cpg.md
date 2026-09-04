# CPG+RL — design and implementation

The `Template-Eigenbot-CPG-Direct-v0` task. Why it exists, how the oscillators
map to 18 joints, and what every design choice was reacting to.

Reference: Li, Wei & Qiu (2023), *"Combined Reinforcement Learning and CPG
Algorithm to Generate Terrain-Adaptive Gait of Hexapod Robots"*, MDPI Actuators
12(4):157. The joint-space mapping follows Bellegarda & Ijspeert's CPG-RL.

Files: `TASK/cpg.py` (the oscillators), `TASK/eigenbot_cpg_env.py` (the env
subclass), `TASK/eigenbot_cpg_env_cfg.py` (every tunable),
`scripts/test_cpg.py` (9 offline checks).

---

## 1. The idea

The baseline policy writes 18 joint targets at 50 Hz and has to discover
rhythmic, coordinated stepping from scratch. The CPG approach hands it the rhythm
for free:

- **Six Hopf oscillators**, one per leg, coupled into a fixed tripod phase
  pattern, generate the gait.
- **The RL policy outputs only 7 numbers**: a turning bias `θ ∈ [−1, 1]` and one
  per-leg foot-mapping gain `b_i ∈ [b_min, b_max]`.
- `θ` steers by shortening one side's stride; `b_i` scales an individual leg's
  stride and clearance, which is the terrain-adaptation channel.

The claim under test is that this structural prior buys data efficiency and
robustness at the cost of peak flexibility. The first comparison produced a
sharper version of that claim than expected — see §7.

---

## 2. Oscillator dynamics

Each leg *i* carries a 2-D state `(x_i, y_i)` obeying Hopf dynamics:

```
r²   = x² + y²
conv = α (μ − r²)                     ← radial convergence, clamped (see below)
ẋ    = conv·x − ω·y  +  coupling
ẏ    = conv·y + ω·x  +  coupling
```

`α = 100` sets how fast a perturbed oscillator returns to the limit cycle of
radius `√μ = 1`. `ω = 2π·f` is the gait frequency.

**Coupling.** Each oscillator is pulled toward the average of **all five
others**, each rotated into its own target phase frame — standard diffusive phase
coupling, all-to-all. (`cpg.py`'s docstring calls it a "ring"; the implementation
is all-to-all. Worth correcting in the source.) `phase_offsets = (0, π, 0, π, 0, π)` gives a tripod: legs {0, 2, 4}
in phase, legs {1, 3, 5} half a cycle behind. Changing that tuple is how you get
a wave or ripple gait; nothing else needs to change.

**The Euler guard.** Integration is explicit Euler at the physics dt. Without a
bound on `conv`, a large state kick (r ≳ 2.2 at α=100, dt=0.005) overshoots the
fixed point and diverges to NaN. `cpg.py` clamps `conv` to `±0.9/dt`. This
matters more than it looks: it is what makes it safe to later wire sensory
feedback into the oscillators, which is the standard next step for CPG models.

Oscillators are initialised **on** the limit cycle at their tripod phases, and
`reset(env_ids)` puts them back there — so the very first step of every episode
is already a clean gait, not a transient.

---

## 3. Action decode

```python
action = tanh(raw_action)              # (ne, 7)
theta  = action[:, 0]                  # in [-1, 1]
b      = b_mid + b_half * action[:, 1:7]   # in [b_min, b_max]
```

`tanh` rather than a hard clamp is deliberate. An earlier version clamped, and a
20k-iteration run failed because the PPO noise std grew to ~3 against a ±1 clip:
actions became bang-bang, the clamp passed **zero gradient**, and the policy mean
parked at the extremes. `tanh` keeps a gradient everywhere. Note the consequence
for testing: "maximum" commands in `test_cpg.py` use ±3, since tanh(3) ≈ 0.995.

---

## 4. Mapping oscillator state to 18 joints

This is the part that is specific to EigenBot, and the part that took the most
debugging.

The oscillator state is normalised to the unit circle (`x_n`, `y_n`) so
amplitudes stay stable regardless of transient radius, then:

```
steer_i    = clamp(1 − turn_gain · θ · side_i, 0, 2)
swing_i    = swing_amplitude · b_i · steer_i · x_n_i
clearance_i= max(lift_phase_sign · y_n_i, 0)        ← nonzero on one half-cycle only
lift_i     = lift_amplitude · lift_scales_i · b_i · clearance_i
```

and these scatter into the 18-dim offset vector as

```
offset[swing joint of leg i]  = swing_i
offset[lift joint A of leg i] = lift_joint_signs[0] · lift_i
offset[lift joint B of leg i] = lift_joint_signs[1] · lift_i
```

The final joint command is `default_dof_pos + offsets`, clamped to the soft joint
limits.

### Why joint space and not foot-tip XYZ

The paper maps oscillator output to a foot trajectory and solves inverse
kinematics. EigenBot's modular "bendy" legs have no clean analytic IK, and the
existing actuation path is already `target = default + offset`. Mapping straight
to joint offsets avoids an IK model entirely and plugs into the baseline env
unchanged. **If an IK module is ever written, only `HopfCPG._map_to_joints` needs
to change** — everything else is agnostic.

### The three geometry facts you have to get right

**1. Articulation order is breadth-first, not per-leg.** Isaac Lab orders the 18
joints as M1…M18 across the tree level by level, so leg *k* owns indices
`(k, k+6, k+12)` — modules M(k+1), M(k+7), M(k+13):

```python
leg_joint_indices = ((0,6,12), (1,7,13), (2,8,14), (3,9,15), (4,10,16), (5,11,17))
```

URDF chain per leg: `body → M(k) → M(k+6) → static elbow → M(k+12) → foot`.
M1/M4 are rear, M2/M5 mid, M3/M6 front; M1–M3 sit on one side, M4–M6 on the
other, hence `leg_sides = (−1,−1,−1, 1, 1, 1)`. If you ever resolve body or joint
indices yourself, pass `preserve_order=True` to `find_bodies` — the ordering
assumption has silently broken this project before.

**2. The distal joint axis is antiparallel to the middle one.** The URDF applies
`connection_12` Rx(π/2), then `connection_18` Rz(π/2)Rx(π/2), so same-signed
commands rotate the two leg segments in **opposite world directions** — which is
precisely what folds the leg into its Z stance, and why the robot's default pose
sets M7–M18 all to +π/4. Hence `lift_joint_signs = (1.0, 0.5)`. The original
`(1.0, −0.5)` swung both segments the same way: the shank trailed the thigh, the
foot scuffed backward through swing, and the loaded rear legs slipped flat and
never recovered.

**3. Lift must happen on the protraction half-cycle.** `lift_phase_sign = -1.0`
lifts while `y_n < 0`. The oscillator rotates counter-clockwise, so `x_n` is
increasing there — the leg swings forward through the air and strokes rearward on
the ground, and the body travels forward. Set it to `+1.0` and the robot walks
backwards. (It originally did.)

### Why per-leg lift scaling exists

`lift_scales = (1.7, 1.0, 1.0, 1.35, 1.0, 1.0)`.

Legs 0 and 3 are the rear pair; their URDF attachment origins
(`connection_*_attachment`) sit at **z = −0.13** versus **+0.05** for the front
pair along the body axis — the x component is −0.041 for every leg — so they
carry more of the body weight. (The comment in `CPGCfg` says `x`; that is a typo
in the source.) With
uniform lift they simply dragged. Measured at 1.0 Hz, leg 0 had 1.3 cm clearance
and 78% stance against ~2.5 cm and 64% for the front legs.

This is fixed **open-loop, in the CPG**, rather than left for RL to discover,
because the velocity reward is reachable on four legs — a policy is perfectly
happy to leave the rear pair flat. And it is fixed here rather than by raising
the joint PD gains, because those mirror the physical modules.

---

## 5. The environment subclass

`EigenbotCPGEnv` is ~120 lines and overrides four methods.

**`_init_buffers` — the gotcha.** The parent sizes every joint-dimension buffer
(actions, torques, motor strength, action history) from `cfg.action_space`, and
finishes by applying domain randomization to `(num_envs, 18)` robot data. With
`action_space = 7` that crashes on a shape mismatch. The fix is to swap the value
around the `super()` call:

```python
policy_dim = self.cfg.action_space
self.cfg.action_space = NUM_JOINTS      # 18
try:
    super()._init_buffers()
finally:
    self.cfg.action_space = policy_dim
```

Any future task with a non-18 action space needs the same dance.

**`_pre_physics_step`** stores the policy action with `nan_to_num` and **no
pre-clipping** — clipping here was part of the saturation failure described in
§3.

**`_apply_action`** steps the CPG once per physics substep (so the gait is smooth
at 200 Hz while the policy speaks at 50 Hz), adds residual corrections if
enabled, sets `self.actions = offsets` so the observation's "last action" slot
stays 18-dim, clamps targets to the joint limits, and runs the same PD-to-torque
law as the baseline.

**`_reset_idx`** resets the oscillators of the reset envs back onto the limit
cycle and zeroes the policy-action buffers.

Everything else — 974-dim observation, all reward terms, terrain, curriculum,
domain randomization, termination — is inherited untouched. That is what makes
the comparison meaningful.

---

## 6. Residual mode

`cpg.use_residual = True` extends the action space to 7 + 18 = 25: the policy
emits per-joint corrections added on top of the CPG offsets, scaled by
`residual_scale = 0.1` rad. It is the flexibility hedge — a middle point between
the rigid 7-D CPG and the free 18-D baseline. Implemented, never run.

---

## 7. What the CPG has actually shown

Measured open-loop gait (`diag_gait.py`, flat ground, zero action, 1.0 Hz):
clearance 1.3 / 2.6 / 2.7 / 2.0 / 2.2 / 2.6 cm across legs 0–5, stance 64–78%,
travel 0.50 m in 6 s (**0.083 m/s**), lateral drift ~0.00 m, body height ~11.5 cm.

In the first 10k-vs-10k comparison the CPG policy and the 18-DOF PPO baseline
scored an identical ~95 reward — but the CPG **walked** while the baseline
**vibrated its legs in place**. The reward function was not discriminating
locomotion, and the unconstrained policy found the exploit immediately while the
CPG could not: its action space simply does not contain "vibrate without
travelling".

That is the most interesting result the project has produced so far, and it is a
result *about* structural priors, not a bug. Full write-up in
[`experiment_log.md`](experiment_log.md).

### The speed ceiling

At the default mid gain (zero action → b = 0.875) the gait moves 0.083 m/s, and
the stride has little room to grow: the front and rear swing joints start at
±π/4 against a ±π/2 limit, so `swing_amplitude × b` caps near 0.78 rad there
(the mid pair starts at 0 and has more headroom).
A 1.5× amplitude test made things worse (rear clearance 0.6 cm, travel 0.22 m).
Frequency is the only real lever, and it trades against clearance — which is the
argument for **giving the policy frequency authority** (action space 7 → 8, an ω
scale of ~0.6–2.0×), exactly as CPG-RL does. That is the highest-upside open
thread.

---

## 8. Testing without a GPU

`python scripts/test_cpg.py` loads the real `cpg.py` by file path (so nothing
from `isaaclab` is imported) and runs it against a stub config that *should*
mirror `CPGCfg`. Nine checks:

1. tripod phase locking — legs {0,2,4} vs {1,3,5} exactly half a cycle apart
2. stable limit cycle — radius stays ~1 over 10 s of integration
3. perturbation recovery — a large kick decays instead of diverging (the Euler
   guard)
4. gait direction — the leg protracts while lifted (regression test for
   `lift_phase_sign`)
5. rear-leg lift boost — `lift_scales` gives legs 0/3 more clearance
6. bounded offsets — worst-case action keeps |offset| inside ±π/2
7. steering asymmetry — θ scales left/right strides oppositely
8. `reset()` restores selected envs onto the limit cycle
9. batching — shapes correct for `num_envs > 1`

> ⚠️ **The stub has drifted from `CPGCfg`.** It still uses `b_max = 1.5` (config:
> `1.25`) and `lift_scales = (1.35, 1.0, 1.0, 1.35, 1.0, 1.0)` (config: `1.7`
> for leg 0). So check 5 never exercises the committed rear-leg boost, and check
> 6 tests a `b` range the env no longer allows. Resync it.

Run it before every sim run after touching signs, indices or amplitudes. It takes
seconds; a bad sign costs a six-hour training run.
