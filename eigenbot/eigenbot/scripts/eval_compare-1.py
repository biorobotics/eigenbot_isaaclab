# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Head-to-head evaluation of a trained policy, broken down by terrain type.  [v2]

v2 rewrite.  The v1 metrics were not measuring locomotion: forward distance was
sampled from `root_pos_w` on the step an episode ended, by which point Isaac Lab
had already reset the env AND the terrain curriculum had shifted its origin by a
whole 8 m patch, so every episode reported +/-8.00 m (the level change) instead
of distance travelled.  Lateral came out exactly 0.00 for the same reason (a
level change moves along rows only).  The per-terrain split was equally bogus:
the terrain COLUMN index was clamped to 0..3, so with num_cols=20 every column
above 3 landed in the last bucket (17 cols x 2 envs = the famous n=34).

What changed:

  * every distance/velocity quantity is INTEGRATED PER STEP from base and world
    velocities, so a teleport, a reset or a curriculum shift cannot corrupt it;
  * the commanded velocity is forced on `env.commands` every step, so both
    policies are provably asked for the same thing regardless of cfg plumbing;
  * falls are read from `terminated & ~time_outs` instead of being guessed from
    episode length;
  * the terrain curriculum is neutralised for the duration of the eval, and each
    episode records the terrain LEVEL it actually ran on;
  * attitude statistics are masked to live steps (v1 padded dead envs with zeros
    and then took a std over the whole series);
  * results are reported as mean +/- 95% CI, because n=10 per terrain is not a
    number you quote without one.

Metrics reported per episode:

  fall            terminated for a reason other than timeout
  dist_m          path length, integral of |v_xy| dt
  fwd_m           net progress along the commanded heading
  lat_m           net drift perpendicular to it
  speed_mps       fwd_m / duration
  vtrack_mae      mean |v_along_heading - commanded|, the honest tracking error
  vtrack_rmse     root-mean-square of the same
  yaw_drift_deg   |heading - commanded heading| at end of episode
  pitch_rms_deg   attitude excursion, masked to live steps
  roll_rms_deg
  vz_rms          RMS vertical body velocity: bounciness, terrain-agnostic
  cot             cost of transport, sum|tau.qd| dt / (m g dist)   (if available)

Run it once per policy with IDENTICAL --episodes / --command_vel / --seed:

    python scripts/eval_compare.py --task Template-Eigenbot-Direct-v0 \\
        --episodes 200 --headless --out logs/eval_ppo.csv
    python scripts/eval_compare.py --task Template-Eigenbot-CPG-Direct-v0 \\
        --episodes 200 --headless --out logs/eval_cpg.csv
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained policy per terrain type.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint (default: newest).")
parser.add_argument("--episodes", type=int, default=200, help="Episodes to run (spread over parallel envs).")
parser.add_argument("--num_envs", type=int, default=40, help="Parallel envs (one episode each per batch).")
parser.add_argument("--command_vel", type=float, default=0.3, help="Commanded forward velocity, m/s.")
parser.add_argument("--seed", type=int, default=123, help="Evaluation seed.")
parser.add_argument("--terrain_level", type=int, default=None, help="Pin every env to this terrain level (row).")
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

RAD2DEG = 180.0 / math.pi


def _euler_rp(quat: torch.Tensor):
    """Roll and pitch from a (N, 4) wxyz quaternion."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_p = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, torch.asin(sin_p)


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _resolve_checkpoint(agent_cfg) -> str:
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    return get_checkpoint_path(os.path.abspath(log_root), ".*", "model_.*.pt")


def _mean_ci(vals):
    """Mean and half-width of the 95% CI."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def _neutralise_curriculum(u) -> list[str]:
    """No-op every terrain-curriculum method so difficulty cannot drift mid-eval."""
    patched = []
    for name in dir(u):
        if "curric" not in name.lower():
            continue
        try:
            attr = getattr(u, name)
        except Exception:
            continue
        if callable(attr):
            try:
                setattr(u, name, lambda *a, **k: None)
                patched.append(name)
            except Exception:
                pass
    return patched


def _terrain_map(u):
    """Return (env->sub_terrain index tensor, names).  Mirrors TerrainGenerator."""
    terrain = getattr(u, "_terrain", None)
    types = getattr(terrain, "terrain_types", None) if terrain is not None else None
    gen_cfg = getattr(u.cfg.terrain, "terrain_generator", None)
    if types is None or gen_cfg is None:
        print("[warn] no terrain_types available (flat plane?) - reporting a single group")
        return torch.zeros(u.num_envs, dtype=torch.long, device=u.device), ["all"]

    names = list(gen_cfg.sub_terrains.keys())
    props = [gen_cfg.sub_terrains[n].proportion for n in names]
    total = float(sum(props))
    cumsum, acc = [], 0.0
    for pr in props:
        acc += pr / total
        cumsum.append(acc)
    num_cols = int(gen_cfg.num_cols)
    # Exactly how TerrainGenerator._generate_curriculum_terrains assigns columns.
    col_to_sub = [
        next((i for i, c in enumerate(cumsum) if col / num_cols + 0.001 < c), len(names) - 1)
        for col in range(num_cols)
    ]
    lut = torch.tensor(col_to_sub, device=u.device, dtype=torch.long)
    env_terrain = lut[types.to(u.device).long().clamp(0, num_cols - 1)]
    spans = {n: col_to_sub.count(i) for i, n in enumerate(names)}
    print(f"[eval] columns per sub-terrain: {spans}")
    return env_terrain, names


def main():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    v_cmd = args_cli.command_vel

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "depth_camera"):
        env_cfg.depth_camera.use_camera = False
    # Fixed straight-ahead command so every policy is asked for the same thing.
    # This is belt; the braces are forcing env.commands every step below.
    if hasattr(env_cfg, "commands"):
        c = env_cfg.commands
        if hasattr(c, "ranges"):
            c.ranges.lin_vel_x = (v_cmd, v_cmd)
            if hasattr(c.ranges, "lin_vel_y"):
                c.ranges.lin_vel_y = (0.0, 0.0)
            if hasattr(c.ranges, "ang_vel_yaw"):
                c.ranges.ang_vel_yaw = (0.0, 0.0)
            if hasattr(c.ranges, "heading"):
                c.ranges.heading = (0.0, 0.0)
        for attr, val in (("rand_heading", False), ("curriculum", False)):
            if hasattr(c, attr):
                setattr(c, attr, val)
            else:
                print(f"[warn] cfg.commands has no '{attr}' - not set")
    if hasattr(env_cfg, "terrain") and args_cli.terrain_level is not None:
        env_cfg.terrain.max_init_terrain_level = args_cli.terrain_level

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    resume_path = _resolve_checkpoint(agent_cfg)
    print(f"[eval] task={args_cli.task}\n[eval] checkpoint={resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=u.device)

    patched = _neutralise_curriculum(u)
    print(f"[eval] curriculum methods neutralised: {patched or 'NONE FOUND - levels may drift'}")

    env_terrain, terrain_names = _terrain_map(u)
    levels = getattr(getattr(u, "_terrain", None), "terrain_levels", None)

    dt = float(u.step_dt)
    max_len = int(u.max_episode_length)
    n_batches = max(1, math.ceil(args_cli.episodes / u.num_envs))
    print(f"[eval] dt={dt:.4f}s  max_episode_length={max_len} steps ({max_len * dt:.1f}s)")
    print(f"[eval] commanded forward velocity forced to {v_cmd} m/s every step")

    # total mass for cost of transport
    try:
        mass = u.robot.data.default_mass.sum(dim=1).to(u.device)
    except Exception:
        mass = None

    rows = []
    for batch in range(n_batches):
        reset_out = wrapped.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))

        N = u.num_envs
        z = lambda: torch.zeros(N, device=u.device)  # noqa: E731

        yaw0 = _yaw_of(u.robot.data.root_quat_w).clone()
        heading_vec = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=1)
        lat_vec = torch.stack([-torch.sin(yaw0), torch.cos(yaw0)], dim=1)
        lvl = levels.clone() if levels is not None else torch.zeros(N, dtype=torch.long, device=u.device)

        alive = torch.ones(N, dtype=torch.bool, device=u.device)
        fell = torch.zeros(N, dtype=torch.bool, device=u.device)
        steps, path, fwd, lat, energy = z(), z(), z(), z(), z()
        err_abs, err_sq, vz_sq, roll_sq, pitch_sq = z(), z(), z(), z(), z()
        yaw_end = yaw0.clone()

        def force_cmd():
            if not hasattr(u, "commands"):
                return
            u.commands[:, 0] = v_cmd
            if u.commands.shape[1] > 1:
                u.commands[:, 1] = 0.0
            if u.commands.shape[1] > 3:
                u.commands[:, 3] = yaw0

        force_cmd()

        with torch.inference_mode():
            for t in range(max_len):
                if t % 250 == 0:
                    print(f"[eval]   batch {batch + 1}/{n_batches}: step {t}/{max_len}, {int(alive.sum())} alive", flush=True)
                actions = policy(obs)
                step_out = wrapped.step(actions)
                obs, dones, extras = step_out[0], step_out[2], step_out[3]
                if isinstance(obs, dict):
                    obs = obs.get("policy", next(iter(obs.values())))

                dones_b = dones.bool()
                time_out = extras.get("time_outs", None) if isinstance(extras, dict) else None
                time_out = time_out.bool() if time_out is not None else torch.zeros_like(dones_b)
                # a real failure = done for any reason other than the clock
                fell |= alive & dones_b & ~time_out

                # sample state only for envs that have NOT just been reset
                m = (alive & ~dones_b).float()

                v_w = u.robot.data.root_lin_vel_w                       # world frame
                v_xy = v_w[:, :2]
                v_head = (v_xy * heading_vec).sum(dim=1)                # along commanded heading
                roll, pitch = _euler_rp(u.robot.data.root_quat_w)
                yaw = _yaw_of(u.robot.data.root_quat_w)

                path += m * torch.norm(v_xy, dim=1) * dt
                fwd += m * v_head * dt
                lat += m * (v_xy * lat_vec).sum(dim=1) * dt
                e = v_head - v_cmd
                err_abs += m * e.abs()
                err_sq += m * e * e
                vz_sq += m * v_w[:, 2] ** 2
                roll_sq += m * roll * roll
                pitch_sq += m * pitch * pitch
                yaw_end = torch.where(m.bool(), yaw, yaw_end)
                steps += m

                if mass is not None:
                    try:
                        p = (u.robot.data.applied_torque * u.robot.data.joint_vel).abs().sum(dim=1)
                        energy += m * p * dt
                    except Exception:
                        mass = None

                alive = alive & ~dones_b
                force_cmd()
                if not alive.any():
                    break

        n_steps = steps.clamp_min(1.0)
        duration = steps * dt
        cot = (
            energy / (mass * 9.81 * path.clamp_min(1e-3))
            if mass is not None
            else torch.full_like(path, float("nan"))
        )

        for i in range(u.num_envs):
            rows.append(
                {
                    "episode": batch * u.num_envs + i,
                    "terrain": terrain_names[env_terrain[i].item()],
                    "level": int(lvl[i].item()),
                    "fall": int(fell[i].item()),
                    "duration_s": round(duration[i].item(), 2),
                    "dist_m": round(path[i].item(), 3),
                    "fwd_m": round(fwd[i].item(), 3),
                    "lat_m": round(lat[i].item(), 3),
                    "speed_mps": round((fwd[i] / duration[i].clamp_min(1e-6)).item(), 4),
                    "vtrack_mae": round((err_abs[i] / n_steps[i]).item(), 4),
                    "vtrack_rmse": round(math.sqrt((err_sq[i] / n_steps[i]).item()), 4),
                    "yaw_drift_deg": round(abs(_wrap_pi(yaw_end[i] - yaw0[i]).item()) * RAD2DEG, 2),
                    "roll_rms_deg": round(math.sqrt((roll_sq[i] / n_steps[i]).item()) * RAD2DEG, 3),
                    "pitch_rms_deg": round(math.sqrt((pitch_sq[i] / n_steps[i]).item()) * RAD2DEG, 3),
                    "vz_rms": round(math.sqrt((vz_sq[i] / n_steps[i]).item()), 4),
                    "cot": round(cot[i].item(), 3),
                }
            )
        print(f"[eval] batch {batch + 1}/{n_batches} done")

    # ---- summary -----------------------------------------------------
    W = 104
    print("\n" + "=" * W)
    print(f"EVALUATION - {args_cli.task}   (commanded {v_cmd} m/s, {len(rows)} episodes, seed {args_cli.seed})")
    print("=" * W)
    print(
        f"{'terrain':<14}{'n':>4}{'fall%':>7}{'dist(m)':>9}{'speed':>8}{'vtrk_mae':>10}"
        f"{'lat(m)':>8}{'yaw(deg)':>10}{'pitch_rms':>11}{'vz_rms':>8}{'CoT':>8}"
    )
    print("-" * W)

    groups = sorted({r["terrain"] for r in rows})
    for g in groups + ["ALL"]:
        sel = rows if g == "ALL" else [r for r in rows if r["terrain"] == g]
        if not sel:
            continue
        n = len(sel)
        mean = lambda k: sum(r[k] for r in sel) / n  # noqa: E731
        cot_vals = [r["cot"] for r in sel if r["cot"] == r["cot"]]
        cot_m = sum(cot_vals) / len(cot_vals) if cot_vals else float("nan")
        print(
            f"{g:<14}{n:>4}{100 * mean('fall'):>6.0f}%{mean('dist_m'):>9.2f}{mean('speed_mps'):>8.3f}"
            f"{mean('vtrack_mae'):>10.4f}{mean('lat_m'):>8.2f}{mean('yaw_drift_deg'):>10.1f}"
            f"{mean('pitch_rms_deg'):>11.2f}{mean('vz_rms'):>8.4f}{cot_m:>8.2f}"
        )
    print("-" * W)

    print("\nheadline metrics, mean +/- 95% CI:")
    for g in groups + ["ALL"]:
        sel = rows if g == "ALL" else [r for r in rows if r["terrain"] == g]
        if not sel:
            continue
        sm, sc = _mean_ci([r["speed_mps"] for r in sel])
        em, ec = _mean_ci([r["vtrack_mae"] for r in sel])
        fm, fc = _mean_ci([float(r["fall"]) for r in sel])
        print(
            f"  {g:<14} speed {sm:6.3f} +/- {sc:.3f} m/s   "
            f"vtrack_mae {em:6.3f} +/- {ec:.3f} m/s   "
            f"fall {100 * fm:5.1f} +/- {100 * fc:.1f} %"
        )
    print("=" * W + "\n")

    out = args_cli.out or os.path.join("logs", f"eval_{args_cli.task.replace('-', '_')}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] per-episode results written to {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
