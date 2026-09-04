# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Head-to-head evaluation of a trained policy, broken down by terrain type.

Loads an rsl-rl checkpoint, rolls out a fixed number of episodes with a fixed
commanded velocity, and reports the metrics the CPG+RL study calls for:

  * forward distance per episode      (progress along the commanded direction)
  * early-termination rate            (falls / illegal contacts before timeout)
  * body roll & pitch std             (attitude stability)
  * lateral offset from straight line (heading hold)
  * mean achieved vs commanded speed  (does the gait actually track the command)

Results are grouped by terrain sub-type (flat / random_rough / slopes /
obstacles), because a single average hides the interesting differences.

Run it once per policy, then diff the CSVs:

    python scripts/eval_compare.py --task Template-Eigenbot-CPG-Direct-v0 \\
        --episodes 40 --headless --out logs/eval_cpg.csv
    python scripts/eval_compare.py --task Template-Eigenbot-Direct-v0 \\
        --episodes 40 --headless --out logs/eval_ppo.csv

By default the newest checkpoint of the task's experiment is used; pass
--checkpoint to pick a specific .pt file. Keep --episodes, --command_vel and
--seed identical between the two runs or the comparison is meaningless.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained policy per terrain type.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint (default: newest).")
parser.add_argument("--episodes", type=int, default=40, help="Episodes to run (spread over parallel envs).")
parser.add_argument("--num_envs", type=int, default=40, help="Parallel envs (one episode each per batch).")
parser.add_argument("--command_vel", type=float, default=0.3, help="Commanded forward velocity, m/s.")
parser.add_argument("--seed", type=int, default=123, help="Evaluation seed.")
parser.add_argument("--out", type=str, default=None, help="CSV output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import math
import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import eigenbot.tasks  # noqa: F401  (registers the gym tasks)

TERRAIN_NAMES = ["flat", "random_rough", "slopes", "obstacles"]


def _euler_rp(quat: torch.Tensor):
    """Roll and pitch from a (N, 4) wxyz quaternion."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_p = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, torch.asin(sin_p)


def _resolve_checkpoint(agent_cfg) -> str:
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    return get_checkpoint_path(os.path.abspath(log_root), ".*", "model_.*.pt")


def main():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "depth_camera"):
        env_cfg.depth_camera.use_camera = False
    # Fixed straight-ahead command so every policy is asked for the same thing.
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.ranges.lin_vel_x = (args_cli.command_vel, args_cli.command_vel)
        env_cfg.commands.ranges.ang_vel_yaw = (0.0, 0.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
        env_cfg.commands.rand_heading = False
        env_cfg.commands.curriculum = False
    # Freeze the terrain curriculum so robots stay on their assigned patch.
    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.max_init_terrain_level = None

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    resume_path = _resolve_checkpoint(agent_cfg)
    print(f"[eval] task={args_cli.task}\n[eval] checkpoint={resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    print("[eval] env created; wrapping...", flush=True)
    wrapped = RslRlVecEnvWrapper(env)
    print("[eval] wrapped; building runner...", flush=True)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    print("[eval] runner built; loading checkpoint...", flush=True)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=u.device)
    print("[eval] policy ready; starting rollouts", flush=True)

    # Which sub-terrain each env sits on. `terrain_types` holds the COLUMN index
    # (0..num_cols-1), NOT the sub-terrain index: TerrainGenerator spreads the
    # sub-terrains across columns by proportion, so 4 sub-terrains over 20 columns
    # are not the same number. Clamping the column index to 0..3 (what this used
    # to do) put every column above 3 in the last bucket and made the per-terrain
    # breakdown meaningless. Rebuild the map the way TerrainGenerator does.
    terrain = getattr(u, "_terrain", None)
    types = getattr(terrain, "terrain_types", None) if terrain is not None else None
    gen_cfg = getattr(u.cfg.terrain, "terrain_generator", None)
    terrain_names = list(TERRAIN_NAMES)
    if types is not None and gen_cfg is not None:
        terrain_names = list(gen_cfg.sub_terrains.keys())
        props = [gen_cfg.sub_terrains[n].proportion for n in terrain_names]
        total = float(sum(props))
        cumsum, acc = [], 0.0
        for pr in props:
            acc += pr / total
            cumsum.append(acc)
        num_cols = int(gen_cfg.num_cols)
        col_to_sub = [
            next(i for i, c in enumerate(cumsum) if col / num_cols + 0.001 < c)
            for col in range(num_cols)
        ]
        lut = torch.tensor(col_to_sub, device=u.device, dtype=torch.long)
        env_terrain = lut[types.to(u.device).long().clamp(0, num_cols - 1)]
        spans = {n: col_to_sub.count(i) for i, n in enumerate(terrain_names)}
        print(f"[eval] columns per sub-terrain: {spans}", flush=True)
    else:
        env_terrain = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
        terrain_names = ["all"]
        print("[warn] no terrain_types available (flat plane?) — reporting a single group")

    max_len = int(u.max_episode_length)
    n_batches = max(1, math.ceil(args_cli.episodes / u.num_envs))
    rows = []

    for batch in range(n_batches):
        reset_out = wrapped.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        start_xy = u.robot.data.root_pos_w[:, :2].clone()
        yaw0 = u._compute_yaw().clone() if hasattr(u, "_compute_yaw") else torch.zeros(u.num_envs, device=u.device)
        alive = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
        steps = torch.zeros(u.num_envs, device=u.device)
        roll_acc, pitch_acc = [], []
        end_xy = start_xy.clone()

        with torch.inference_mode():
            for t in range(max_len):
                if t % 250 == 0:
                    print(f"[eval]   batch {batch + 1}: step {t}/{max_len}, {int(alive.sum())} alive", flush=True)
                actions = policy(obs)
                step_out = wrapped.step(actions)
                obs, dones = step_out[0], step_out[2]
                if isinstance(obs, dict):
                    obs = obs.get("policy", next(iter(obs.values())))
                # An env that finishes on THIS step has already been reset by
                # Isaac Lab before step() returned, so root_pos_w now holds its
                # respawn position — and the terrain curriculum may have shifted
                # its origin by a whole 8 m patch on the way. Sampling it here
                # made every episode report the origin shift (a suspiciously
                # round +/-8.00 m) instead of the distance actually travelled.
                # Only sample envs that are still running.
                still = alive & ~dones.bool()
                roll, pitch = _euler_rp(u.robot.data.root_quat_w)
                roll_acc.append(torch.where(still, roll, torch.zeros_like(roll)))
                pitch_acc.append(torch.where(still, pitch, torch.zeros_like(pitch)))
                steps += alive.float()
                end_xy = torch.where(still.unsqueeze(1), u.robot.data.root_pos_w[:, :2], end_xy)
                alive = still
                if not alive.any():
                    break

        R = torch.stack(roll_acc)    # (T, N)
        P = torch.stack(pitch_acc)
        disp = end_xy - start_xy
        # forward axis = initial heading; lateral = perpendicular
        fwd = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=1)
        lat = torch.stack([-torch.sin(yaw0), torch.cos(yaw0)], dim=1)
        forward = (disp * fwd).sum(dim=1)
        lateral = (disp * lat).sum(dim=1)
        duration = steps * u.step_dt
        early = steps < (max_len - 1)

        for i in range(u.num_envs):
            rows.append(
                {
                    "episode": batch * u.num_envs + i,
                    "terrain": terrain_names[env_terrain[i].item()],
                    "forward_m": round(forward[i].item(), 3),
                    "lateral_m": round(lateral[i].item(), 3),
                    "speed_mps": round((forward[i] / duration[i].clamp_min(1e-6)).item(), 3),
                    "duration_s": round(duration[i].item(), 2),
                    "early_term": int(early[i].item()),
                    "roll_std": round(R[:, i].std().item(), 4),
                    "pitch_std": round(P[:, i].std().item(), 4),
                }
            )
        print(f"[eval] batch {batch + 1}/{n_batches} done")

    # ---- summary -----------------------------------------------------
    print("\n" + "=" * 86)
    print(f"EVALUATION — {args_cli.task}   (commanded {args_cli.command_vel} m/s, {len(rows)} episodes)")
    print("=" * 86)
    hdr = f"{'terrain':<14}{'n':>4}{'fwd (m)':>10}{'speed':>9}{'early%':>9}{'roll sd':>10}{'pitch sd':>10}{'lat (m)':>9}"
    print(hdr)
    print("-" * 86)

    groups = sorted({r["terrain"] for r in rows})
    for g in groups + ["ALL"]:
        sel = rows if g == "ALL" else [r for r in rows if r["terrain"] == g]
        if not sel:
            continue
        n = len(sel)
        mean = lambda k: sum(r[k] for r in sel) / n  # noqa: E731
        print(
            f"{g:<14}{n:>4}{mean('forward_m'):>10.2f}{mean('speed_mps'):>9.3f}"
            f"{100 * mean('early_term'):>8.0f}%{mean('roll_std'):>10.4f}"
            f"{mean('pitch_std'):>10.4f}{mean('lateral_m'):>9.2f}"
        )
    print("=" * 86 + "\n")

    out = args_cli.out or os.path.join("logs", f"eval_{args_cli.task.replace('-', '_')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] per-episode results written to {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
