# CPG + RL for Eigenbot — Overview & Status

A plain-language summary of the CPG+RL locomotion work: what it is, why we're
trying it, how it differs from our current PPO setup, the benefits and
tradeoffs, what was actually built, and where it stands. A deeper, code-level
companion lives in `CPG_RL_IMPLEMENTATION.md`.

---

## TL;DR

I implemented a **CPG+RL** locomotion approach as an alternative to our current
PPO setup. It shrinks the policy's job from controlling all **18 joints** down to
just **7 numbers** by letting a Central Pattern Generator produce the rhythmic
gait, while RL only handles terrain adaptation on top. It's on a separate branch
(`cpg-rl`) so we can test it on the remote PC before merging to main. The
baseline PPO task is left untouched, so the two are directly comparable.

---

## The problem we're solving

Our PPO policy outputs a target angle for all 18 joints every timestep and has to
learn coordinated walking from scratch, purely from reward. That's a large, noisy
search space: most random 18-D joint commands produce flailing or falling, so the
reward signal for a clean, regular gait is hard to shape. This is why the PPO
weight experiments have been underwhelming — it's an action-space problem, not a
tuning problem.

---

## What a CPG is

A **Central Pattern Generator** is a small dynamical system that produces stable,
periodic signals on its own — the engineering analog of the neural circuits that
make animals' legs cycle without conscious thought. In our implementation it's
**six coupled Hopf oscillators**, one per leg:

- Each oscillator orbits a circular **limit cycle**. The key property: if you
  perturb it (push the robot, hit rough terrain), it naturally returns to its
  rhythm instead of falling apart — built-in robustness.
- The coupling between oscillators **locks in the tripod gait**: legs 0/2/4 swing
  while legs 1/3/5 plant, then they alternate.

So before any learning happens, the robot already *wants* to walk in a
coordinated, rhythmic, self-stabilizing way. The gait structure is engineered in,
not discovered by RL.

---

## How CPG+RL differs from PPO

The honest framing: "CPG+RL vs PPO" is really about **what the policy controls**,
not which learning algorithm is smarter.

| | PPO (current) | CPG + RL |
|---|---|---|
| Action space | 18 joint targets | 7 params (1 turn + 6 per-leg gains) |
| Who creates the gait | The policy, from scratch | The CPG (built-in tripod rhythm) |
| Policy's actual job | Invent + execute walking | Only adapt a working gait to terrain |
| Robustness to pushes | Learned (if at all) | Inherent (limit-cycle self-correction) |
| Reward shaping | Hard (sparse signal for gaits) | Easier (reward adaptation, not gait) |
| Data efficiency | Low | Higher |

With CPG+RL, the oscillators supply rhythm and coordination for free. The policy
emits just 7 numbers: a **turning bias** plus six **per-leg gains** that scale
each leg's stride and ground clearance. The learning problem collapses from "find
a good 18-D trajectory at every step" to "find 7 modulation parameters."

Because that space is small and nearly linear, we can also drop PPO for **ARS**
(Augmented Random Search) — a much simpler gradient-free optimizer the source
paper uses. ARS would be hopeless on raw 18-D joint control but works well here.
The code supports both ARS and our existing PPO pipeline for a clean comparison.

---

## Benefits

- **Faster, more reliable convergence.** A 7-D action space over a pre-structured
  gait is far easier to optimize than 18-D from scratch.
- **Better locomotion metrics.** The reference paper (Li, Wei & Qiu, hexapod
  CPG+RL) reports better forward distance, tighter body sway, and far fewer early
  failures than direct-joint RL.
- **Inherent robustness.** The limit cycle self-corrects after disturbances, so
  the robot recovers from pushes and bad footing more gracefully.
- **Easier reward shaping.** We reward terrain adaptation on top of a working
  gait, rather than trying to coax a gait into existence.
- **Clean perception hook.** The six per-leg gains are a natural interface for
  later conditioning on Shishir's terrain features — modulate the gains per leg
  from perceived terrain height/contact, no architectural rewrite.

---

## Tradeoffs (being honest)

- **Structure vs. flexibility.** The policy can only express gaits the CPG can
  produce. PPO over 18 joints could in principle discover an unconventional gait
  that's optimal for some weird terrain; CPG+RL cannot. For highly irregular
  terrain, climbing, or recovery from a flipped state, that rigidity can hurt.
- **Engineering priors must be right.** The joint-to-leg mapping, swing-vs-lift
  assignment, amplitudes, and frequency are set by hand and must match the real
  kinematics. Get the joint grouping wrong and it won't walk no matter how well
  RL converges — this is the first thing to verify on the test PC.
- **More moving parts.** We now reason about oscillator dynamics + a mapping
  function + a learner, versus PPO's single monolithic network.

### The hedge: residual mode

To recover some flexibility without giving up the rhythmic backbone, there's a
**residual mode** flag. It keeps the CPG gait but lets the policy add small
per-joint corrections on top (action space becomes 7 + 18 = 25, corrections kept
small so the CPG still dominates). It's the middle ground between rigid CPG and
unconstrained PPO, and trains almost as easily as pure CPG because the residual
starts near zero and the gait works immediately.

---

## What was actually built

Almost entirely **new files**, with a single ~12-line edit to register the new
task. The baseline PPO task and env are untouched.

- `cpg.py` — batched 6-oscillator Hopf CPG; maps the 7D action to 18 joint offsets.
- `eigenbot_cpg_env.py` — `EigenbotCPGEnv`, a thin subclass of the existing env;
  overrides only the action/buffer/reset hooks.
- `eigenbot_cpg_env_cfg.py` — config with `action_space = 7` (or 25 in residual
  mode) plus all the tunable CPG parameters.
- `agents/rsl_rl_cpg_ppo_cfg.py` — a smaller PPO network sized for the 7D task.
- `scripts/ars/train.py` — standalone ARS (gradient-free) trainer, the paper's
  optimizer.
- `__init__.py` — **edited** (~12 lines) to register `Template-Eigenbot-CPG-Direct-v0`.
- `CPG_RL_IMPLEMENTATION.md` — the code-level guide; this file is the overview.

Design note: the CPG maps **directly to joint-space offsets** rather than to a
foot-tip trajectory + inverse kinematics. The modular "bendy" legs have no
analytic IK model in the repo, and a joint-space CPG is more robust and far less
invasive. If we add IK later, only one function (`HopfCPG._map_to_joints`)
changes.

---

## Status & next steps

- **Done:** Full implementation written; CPG math validated offline (correct
  tripod phasing — leg groups exactly half a cycle apart, stable limit cycle,
  bounded joint offsets, working steering). Committed on the `cpg-rl` branch
  (based on `origin/main`). Post-review fixes applied: joint-dimension
  buffers are now sized correctly during init (previously crashed on
  construction with `randomize_motor`), legs now lift during the protraction
  half-cycle (`lift_phase_sign`, previously the gait would have walked
  backward/dragged feet), and the oscillator integration is guarded against
  divergence under large state perturbations.
- **Not yet done:** Hasn't been run in Isaac Lab (no GPU in the dev environment).
- **Next:**
  1. Pull the `cpg-rl` branch on the remote PC and run it in sim.
  2. **Verify the joint-to-leg mapping** against the real kinematics — the one
     critical thing to get right before training.
  3. Train CPG+RL and the PPO baseline under identical settings and compare:
     average forward distance, early terminations, body roll/pitch variance, and
     lateral offset from straight-line travel.
  4. If pure 7-D feels too restrictive, flip on residual mode.
  5. Later: wire the per-leg gains to Shishir's perception features (Big Task 3).

---

## Meeting talking points

1. "PPO struggles because it controls all 18 joints and learns walking from
   scratch — that's the real reason the weight experiments underwhelmed."
2. "CPG+RL drops the action space from 18 to 7: the oscillators make the gait,
   RL just tunes terrain adaptation on top."
3. "It self-stabilizes — the robot recovers its rhythm after a push instead of
   relearning balance."
4. "ARS is simpler than PPO and fits because the policy is nearly linear over 7
   parameters; I can also run it through our existing PPO pipeline for a fair
   comparison."
5. "It's on its own branch, almost all new files, baseline untouched — low risk
   to try, and the per-leg gains give us a clean hook for perception later."
6. "Honest tradeoff: it's locked to a tripod gait topology. I added a residual
   mode as a hedge if we need more flexibility."

---

## References

- Li, Wei & Qiu (2023), *Combined Reinforcement Learning and CPG Algorithm to
  Generate Terrain-Adaptive Gait of Hexapod Robots*, MDPI Actuators 12(4):157.
- Whitman & Choset (2023), *Learning Modular Robot Locomotion from
  Demonstrations*, ICRA (the RL+IL alternative we considered but did not pursue).
- Bellegarda & Ijspeert, CPG-RL (joint-space CPG formulation this port follows).
