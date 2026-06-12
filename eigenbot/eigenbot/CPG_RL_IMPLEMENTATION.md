# CPG + RL for Eigenbot — Implementation Guide

This document describes the CPG+RL locomotion variant added to the Eigenbot
Isaac Lab project, why this approach was chosen, exactly what files were added
(and the one existing file edited), and how to run, tune, and extend it.

## Why CPG+RL (and not RL+IL) for this workspace

Both papers in the research summary were considered against the actual codebase:

- **RL+IL (MBRL, Whitman & Choset)** needs (a) usable demonstration trajectories
  from Ivan's PPO runs and (b) a model-based RL loop or a graph-structured
  modular policy. Neither exists here, and adding MBRL means replacing the
  `rsl_rl` on-policy pipeline wholesale — the opposite of a non-invasive change,
  and gated on demo data we can't verify is good.
- **CPG+RL (Li, Wei & Qiu)** directly attacks the stated problem — PPO struggling
  with the 18-DOF action space — by compressing the policy's job to 7 numbers on
  top of a structured rhythmic gait. It slots cleanly into the existing
  `target = default_dof_pos + offset` actuation path and reuses the existing
  observations, rewards, terrain, and domain randomization untouched.

So CPG+RL is both the better fit for this repo *and* the summary's primary
recommendation. The implementation here makes one deliberate departure from the
paper: the CPG output is mapped **directly to joint-space offsets** rather than
to a foot-tip XYZ trajectory plus inverse kinematics. The Eigenbot's modular
"bendy" legs don't have an analytic IK model in this repo, and a joint-space CPG
(à la Bellegarda & Ijspeert's CPG-RL) is more robust and far less invasive. If
you later add IK, only `HopfCPG._map_to_joints` changes.

## What was added

All new files, plus a single ~12-line addition to one existing file:

```
source/eigenbot/eigenbot/tasks/direct/eigenbot/
  cpg.py                      # NEW: batched Hopf CPG (6 oscillators -> 18 joints)
  eigenbot_cpg_env_cfg.py     # NEW: EigenbotCPGEnvCfg (action_space=7) + CPGCfg
  eigenbot_cpg_env.py         # NEW: EigenbotCPGEnv(EigenbotEnv) overrides
  agents/rsl_rl_cpg_ppo_cfg.py# NEW: small-net PPO cfg for the 7D task
  __init__.py                 # EDITED: register Template-Eigenbot-CPG-Direct-v0
scripts/ars/
  train.py                    # NEW: standalone ARS (V2) trainer (the paper's optimizer)
```

The baseline `Template-Eigenbot-Direct-v0` task, `eigenbot_env.py`, and the PPO
config are all left exactly as they were. The two tasks share the same env code
path, observations, and rewards, so any result difference is attributable to the
action parameterization.

## Architecture

```
policy action a in R^7  =  [ theta , b_1 ... b_6 ]
        |                     turn      per-leg foot-mapping gains
        v
HopfCPG  (6 oscillators, tripod-coupled, integrated at sim dt)
        |   phase per leg -> swing (fore/aft) + lift (clearance during swing)
        v
joint offsets in R^18  ->  target = default_dof_pos + offset  ->  PD -> torque
```

- **Oscillators (`cpg.py`):** one Hopf oscillator per leg, diffusively coupled in
  a ring so legs {0,2,4} stay in phase and {1,3,5} stay half a cycle out of
  phase (tripod gait). State is integrated with forward Euler at the physics dt;
  the policy action is held constant across the decimation window.
- **Action decode:** `theta in [-1,1]` biases left/right stride for steering;
  each `b_i` is rescaled to `[b_min, b_max]` and scales that leg's stride and
  ground clearance — this is the clean per-leg interface for terrain adaptation
  and, later, perception conditioning.
- **Env (`eigenbot_cpg_env.py`):** subclasses `EigenbotEnv` and overrides only
  `_init_buffers`, `_pre_physics_step`, `_apply_action`, and `_reset_idx`. The
  18-dim "previous action" slot in the observation is fed the CPG's joint
  offsets, so `observation_space` (974) is unchanged and the policy still sees
  what was commanded.

### One subtlety to know about

`EigenbotEnv._init_buffers` sizes *every* joint-dimension buffer (`actions`,
`torques`, `motor_strength`, the action-delay ring buffer, ...) from
`cfg.action_space`, and it ends by applying domain randomization, which
multiplies those buffers against `(num_envs, 18)` robot data. Since the CPG cfg
sets `action_space = 7`, `EigenbotCPGEnv._init_buffers` temporarily swaps
`cfg.action_space` to 18 for the parent call (so all joint buffers come out the
right size and DR doesn't crash), restores it afterwards, and then allocates
the separate 7D `policy_actions` buffers. This is why the subclass overrides
`_init_buffers` rather than just changing the cfg.

## How to run

Inside the Isaac Lab container (see top-level `README.md` for container setup):

```bash
# Step 0 — offline CPG sanity tests (any machine with PyTorch, no Isaac Sim):
python scripts/test_cpg.py

cd /workspace/eigenbot
pip install -e source/eigenbot            # one-time, picks up the new modules

# Option A — train the CPG policy with the existing PPO pipeline (quick start).
python scripts/rsl_rl/train.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 4096 --headless

# Option B — train with ARS, the paper's gradient-free optimizer (linear policy).
python scripts/ars/train.py --task Template-Eigenbot-CPG-Direct-v0 \
    --n_directions 8 --horizon 1000 --n_iterations 1000 --headless

# Visualize / replay a PPO-trained CPG checkpoint.
python scripts/rsl_rl/play.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 32

# Sanity-check the gait with zero actions (legs should hold default stance).
python scripts/zero_agent.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 1
```

ARS uses `2 * n_directions` environments as parallel rollouts, so `--num_envs`
is derived automatically and should not be set for `scripts/ars/train.py`.

## Tuning checklist (do this first)

The CPG geometry assumes 18 joints grouped 3-per-leg in articulation order, with
each leg's first joint as the swing (coxa/yaw) joint and the next two as lift
(femur/tibia) joints. **Verify this against the real kinematics before training**
— it is the single most important thing to get right. All of it lives in
`CPGCfg` in `eigenbot_cpg_env_cfg.py`:

- `leg_joint_indices` — confirm the 3-joint grouping and which joint swings vs
  lifts. Print `env.robot.data.joint_names` once and map them to legs.
- `lift_joint_signs` / `swing_amplitude` / `lift_amplitude` — set so a swinging
  leg clears the ground and protracts forward, not sideways or into the body.
- `lift_phase_sign` — which oscillator half-cycle the leg lifts in. The default
  (`-1.0`) lifts while the swing offset is increasing, i.e. the leg protracts
  through the air and strokes rearward during stance (forward travel when a
  positive swing offset means "leg forward"). If the robot walks backward in
  sim, flip this to `1.0` or negate `swing_amplitude`.
- `leg_sides` — left/right assignment used for differential steering.
- `omega` — gait frequency (default ~1.5 Hz); `b_min/b_max` — stride/clearance range.

Quick way to eyeball it: run `zero_agent.py`, then a constant non-zero action,
and watch one robot; the legs should produce a clean tripod walk in place.

## Baseline comparison (the summary's metrics)

Train `Template-Eigenbot-Direct-v0` (PPO, 18-DOF) and
`Template-Eigenbot-CPG-Direct-v0` (CPG, 7-DOF) under identical settings and
compare: average forward distance per episode, number of early terminations,
base roll/pitch variance, and lateral offset from straight-line travel. These
are already implicit in the existing reward terms and episode logging.

## Residual mode (CPG backbone + learned per-joint corrections)

Pure 7-D modulation can only express gaits the CPG produces. If that turns out
too restrictive, enable **residual mode**: the policy keeps the 7 CPG params and
additionally emits 18 small per-joint corrections that are added on top of the
CPG offsets. This recovers some of PPO's flexibility while keeping the rhythmic,
self-stabilizing CPG backbone.

```python
# in your task cfg (or a thin subclass)
env_cfg.cpg.use_residual = True
env_cfg.cpg.residual_scale = 0.1   # rad of correction at |action| = 1; keep small
```

How it works mechanically:

- The action space grows from 7 to `7 + 18 = 25` automatically. `EigenbotCPGEnvCfg.__post_init__`
  sets `action_space` based on `cpg.use_residual`, so the policy network and the
  ARS/PPO wrappers size themselves correctly with no other changes.
- In `_apply_action`, the first 7 dims drive the CPG as before; the remaining 18
  are clamped to `[-1, 1]`, scaled by `residual_scale`, and **added** to the CPG
  joint offsets before the `default_dof_pos + offset` actuation:
  `offset = cpg(action[:7]) + residual_scale * clamp(action[7:])`.
- Because `residual_scale` is small (0.1 rad ≈ 6°), the CPG still dominates the
  motion; the residual only nudges individual joints to handle terrain or contact
  the fixed gait can't. Set it to 0 to recover pure CPG mode; raise it gradually
  if the policy is starved for authority.

This is the recommended middle ground between the structured 7-D CPG policy and
unconstrained 18-D PPO. It trains nearly as easily as pure CPG (the residual
starts at zero and the gait works immediately) while not being locked to a single
gait topology. ARS is less suited here once the action space is 25-D and the
residual makes the policy nonlinear — prefer the PPO pipeline for residual mode.

## Extending to perception (Big Task 3)

The `b_i` gains are the clean interface the summary calls out: once Shishir's
terrain features are available, concatenate them into the observation (they flow
through unchanged) and/or let a small head modulate `b_i` per leg based on
predicted terrain height/contact — no change to the CPG or actuation path.

## Notes / caveats

- This code was written against the existing env API but **has not been run in
  Isaac Lab here** (no GPU/sim in this environment). The CPG math is covered by
  `scripts/test_cpg.py`, which runs the real `cpg.py` offline: tripod phasing is
  exact (groups π apart), the limit cycle is stable and recovers from large
  state kicks, joint offsets stay bounded (≤ ~1.08 rad worst-case at θ=1,
  b=b_max — under the ±π/2 limits), the leg protracts while lifted, and
  steering produces the expected left/right asymmetry. Expect to do the
  kinematic tuning above on first run.
- The ring coupling in `cpg.py` is computed with a small python loop over the 6
  legs each substep (vectorized over environments). It's fine at 4096 envs, but
  if you profile a bottleneck, precompute the constant rotation coefficients
  once in `__init__` instead of per step.
