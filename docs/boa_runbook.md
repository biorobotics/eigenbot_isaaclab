# Runbook — the lab training PC ("boa")

Everything machine-specific about running this repo on the lab's training
workstation. Nothing here belongs in the code; if you are on your own machine,
you want the [Quick start](../README.md#quick-start) instead.

> **Why this file exists separately:** boa runs an **older Isaac Lab / rsl-rl**
> than this repo targets. The differences are re-applied by a committed script
> whose *edits* must never be committed. Getting that wrong is the single most
> common way to lose an afternoon here.

---

## 1. Accounts and where things live

- Two users. The desktop session logs in as **siemens**; everything for this
  project runs as **loganzhang** (`su - loganzhang`).
- Working clone: `~loganzhang/Documents/cpg/eigenbot_isaaclab_cpg`
- Edit code with VS Code launched **as loganzhang**:

```bash
export DISPLAY=:0 && code ~/Documents/cpg/eigenbot_isaaclab_cpg &
```

The siemens VS Code cannot save into loganzhang's files. If it offers
**"Retry as Sudo"**, decline — it writes root-owned files that then break Docker
and git for everyone.

- AnyDesk is fine for looking at the viewer; SSH is better for running commands.

---

## 2. Environment

boa does **not** use the Docker path. It has a conda env:

```bash
conda activate env_isaaclab
```

Isaac Sim 4.5 + Isaac Lab 2.0.1 (at `~/IsaacLab/IsaacLab`) + rsl-rl 2.1.2.

> **Every new shell, tmux window and VS Code terminal needs that activate line.**
> A `(base)` prompt is the tell.

For anything with a viewer, also:

```bash
export DISPLAY=:0
```

X11, not Wayland. The siemens side ran `xhost +SI:localuser:loganzhang` once per
boot to permit this.

---

## 3. The compatibility patches

The repo targets a newer Isaac Lab and rsl-rl 3.0.1 than boa has. Six API
differences are patched **in the working tree only**:

| # | Patch |
|---|---|
| 1 | `assets/eigenbot.py`: `effort_limit_sim=8.0` → `effort_limit=8.0` |
| 2 | `eigenbot_env_cfg.py`: `ray_alignment="yaw"` → `attach_yaw_only=True`. **Currently a no-op** — the committed file already says `attach_yaw_only=True`; kept for when the repo moves to the newer API |
| 3 | `scripts/rsl_rl/train.py`: `RSL_RL_VERSION = "3.0.1"` → `"2.1.2"` |
| 4 | Drop `actor_obs_normalization` / `critic_obs_normalization` (rsl-rl 2.x uses the runner-level `empirical_normalization`, already `False`, so behaviour is unchanged). The sed targets both agent cfgs, but only `rsl_rl_ppo_cfg.py` declares them |
| 5 | `train.py` + `play.py`: import only `OnPolicyRunner` (no `DistillationRunner`); alias `RslRlOnPolicyRunnerCfg as RslRlBaseRunnerCfg`; `RslRlVecEnvWrapper(env)` without `clip_actions=`; `getattr(..., "class_name", ...)` guards |
| 6 | `play.py`: drop the `pretrained_checkpoint` import (`resume_path = None` in that branch); `obs, _ = env.get_observations()` (the old wrapper returns a tuple) |

All six are applied by one committed script:

```bash
bash scripts/boa_compat.sh
```

It is idempotent — safe to run repeatedly. It also flips the terrain:

```bash
bash scripts/boa_compat.sh plane
```

```bash
bash scripts/boa_compat.sh generator
```

### The rule

> **The patches live in the working tree, so every `git stash`, `git checkout`
> and `git pull` wipes them. They are disposable — regenerate, never stash.**

Recommended pull sequence on boa:

```bash
git checkout -- source/eigenbot/eigenbot/assets/eigenbot.py scripts/rsl_rl/
```

```bash
git pull origin main
```

```bash
bash scripts/boa_compat.sh
```

**Never `git stash` on boa** — it takes the patches with it and the next run dies
on `effort_limit_sim`. Note that `scripts/boa_compat.sh` itself is *tracked* in
the repo; it is the **edits it makes** that are machine-local and must never be
committed.

### Diagnosing a new one

Any fresh `TypeError: unexpected keyword argument …` on boa is this same version
mismatch. Find the Isaac Lab 2.0 name for the argument, rename or remove it, and
add the edit to `scripts/boa_compat.sh` so the next person gets it for free.

**Delete this whole file's §3 once the lab upgrades to Isaac Lab 2.1+ / rsl-rl
3.0.1.**

---

## 4. GPUs

Two RTX 3080, 12 GB each. Check before launching:

```bash
nvidia-smi
```

Pick a free card explicitly:

```bash
--device cuda:0     # or cuda:1
```

Rules learned the hard way:

- **One Isaac Sim process per GPU.** 4096 envs nearly fills a 12 GB card;
  launching a second job starved the first and corrupted its PhysX scene mid-run
  (`"Scene state is corrupted"`, CUDA error 700/2). **2048 envs leaves headroom.**
- A teammate's job may be sitting on both cards. Look before you launch.
- Isaac Sim ignores `Ctrl+C`. Kill from another terminal:

```bash
pkill -9 -f train.py
```

- Check disk before a long run — it has filled up before:

```bash
df -h /
```

---

## 5. tmux

Long runs go in tmux. **Launch detached and never attach** — VS Code steals
`Ctrl+B`, so detaching from inside it does not work.

```bash
tmux new -d -s cpg
```

```bash
tmux send-keys -t cpg "conda activate env_isaaclab && cd ~/Documents/cpg/eigenbot_isaaclab_cpg/eigenbot/eigenbot && python scripts/rsl_rl/train.py --task Template-Eigenbot-CPG-Direct-v0 --num_envs 2048 --max_iterations 10000 --seed 42 --headless --device cuda:1" Enter
```

```bash
tmux capture-pane -pt cpg | tail -20
```

If you did attach and are stuck, detach from another terminal:

```bash
tmux detach-client -s cpg
```

### Chaining a second run behind the first

To run the CPG job and then the baseline on the same GPU without babysitting:

```bash
CPGPID=$(pgrep -f "task Template-Eigenbot-CPG-Direct-v0" | head -1)
```

```bash
tmux new -d -s ppo
```

```bash
tmux send-keys -t ppo "while kill -0 $CPGPID 2>/dev/null; do sleep 60; done; conda activate env_isaaclab && cd ~/Documents/cpg/eigenbot_isaaclab_cpg/eigenbot/eigenbot && python scripts/rsl_rl/train.py --task Template-Eigenbot-Direct-v0 --num_envs 2048 --max_iterations 10000 --seed 42 --headless --device cuda:1" Enter
```

Roughly 6.5 h per 10k-iteration run at 2048 envs.

---

## 6. TensorBoard

```bash
tensorboard --logdir logs/rsl_rl --port 6007
```

Point it at `logs/rsl_rl` rather than one experiment so both methods share axes.
**It will not pick up runs created after it started if it has been running for
days** — restart it, or use a new port, before concluding a run is missing.

---

## 7. Gotchas specific to this machine

| Symptom | Cause / fix |
|---|---|
| `pip install -e source/eigenbot` didn't seem to take effect | It re-points the `eigenbot` package for **new** processes only. Launch each run from the clone you intend it to use |
| Permission errors on files Docker wrote | Multi-user machine; Docker writes root-owned files. Fix ownership rather than running as root |
| Robot invisible / black viewer | X11 authorization plus default camera framing. `xhost +local:` on the host side, persistent `DISPLAY` export in the shell |
| A run dies on `effort_limit_sim` | The compat patches were wiped by a git operation. `bash scripts/boa_compat.sh` |
| Disk full mid-run | `df -h /`; clear old `logs/` checkpoints |

---

## 8. Machines to keep straight

| Machine | Role | Notes |
|---|---|---|
| boa | Training | conda env, Isaac Lab 2.0.1, needs the compat patches |
| A dev laptop / desktop | Editing, docs, git | Docker path in the README. **Keep clones outside OneDrive** — it has corrupted `.git` here more than once |
| The physical EigenBot | Deployment target | ROS 2 Humble; the policy-inference node is the remaining piece |
