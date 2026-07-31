# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Quantitative gait diagnostic — measure the open-loop CPG gait in numbers.

Runs the task with a **zero action** (for the CPG task that means theta = 0 and
b = 1.0 on every leg, i.e. the pure engineered gait) and reports, per leg:

* foot clearance   — how far that foot actually rises off the ground
* mean foot height — whether the leg is carrying or dragging
* stance fraction  — share of time the foot sits at its own lowest level

plus body travel, mean body height, and lateral drift. This replaces eyeballing
the viewer: a dragging rear leg shows up as clearance ≈ 0 while the front legs
report several centimetres.

CPG parameters can be overridden from the command line, so sign/scale sweeps
need no file edits:

    python scripts/diag_gait.py --headless                      # current cfg
    python scripts/diag_gait.py --headless --lift_signs "1.0,0.5"
    python scripts/diag_gait.py --headless --swing 0.0          # lift only
    python scripts/diag_gait.py --headless --lift_scales "1.6,1,1,1.6,1,1"

Run it on flat ground (``terrain_type="plane"`` in eigenbot_env_cfg.py) — foot
heights are measured in world z, which only equals clearance on a flat floor.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure per-leg foot clearance of the open-loop gait.")
parser.add_argument("--task", type=str, default="Template-Eigenbot-CPG-Direct-v0", help="Task name.")
parser.add_argument("--seconds", type=float, default=8.0, help="Total simulated seconds.")
parser.add_argument("--settle", type=float, default=2.0, help="Seconds to discard while the robot settles.")
parser.add_argument("--swing", type=float, default=None, help="Override CPGCfg.swing_amplitude.")
parser.add_argument("--lift", type=float, default=None, help="Override CPGCfg.lift_amplitude.")
parser.add_argument("--lift_signs", type=str, default=None, help='Override CPGCfg.lift_joint_signs, e.g. "1.0,0.5".')
parser.add_argument("--lift_scales", type=str, default=None, help='Override CPGCfg.lift_scales, 6 values.')
parser.add_argument("--omega_hz", type=float, default=None, help="Override gait frequency in Hz.")
parser.add_argument("--stiffness", type=float, default=None, help="Override joint PD stiffness.")
parser.add_argument("--damping", type=float, default=None, help="Override joint PD damping.")
parser.add_argument("--effort", type=float, default=None, help="Override joint effort (torque) limit.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import eigenbot.tasks  # noqa: F401  (registers the gym tasks)

# URDF wiring: leg 1 (M1/M7/M13) ends in foot module M26, leg 2 in M25, and legs
# 3..6 in M27..M30. FEET_BODIES is listed M25..M30, so the first two columns are
# swapped relative to CPG leg order.
FOOT_MODULE_TO_LEG = {25: 1, 26: 0, 27: 2, 28: 3, 29: 4, 30: 5}
LEG_LABELS = [
    "leg 0  REAR  side A",
    "leg 1  mid   side A",
    "leg 2  front side A",
    "leg 3  REAR  side B",
    "leg 4  mid   side B",
    "leg 5  front side B",
]


def _apply_overrides(cfg) -> None:
    cpg = getattr(cfg, "cpg", None)
    overrides = (args_cli.swing, args_cli.lift, args_cli.lift_signs, args_cli.lift_scales, args_cli.omega_hz)
    if cpg is None:
        if any(o is not None for o in overrides):
            print("[warn] task has no CPG config; ignoring CPG overrides")
        return
    if args_cli.swing is not None:
        cpg.swing_amplitude = args_cli.swing
    if args_cli.lift is not None:
        cpg.lift_amplitude = args_cli.lift
    if args_cli.lift_signs is not None:
        cpg.lift_joint_signs = tuple(float(v) for v in args_cli.lift_signs.split(","))
    if args_cli.lift_scales is not None:
        cpg.lift_scales = tuple(float(v) for v in args_cli.lift_scales.split(","))
    if args_cli.omega_hz is not None:
        cpg.omega = 2.0 * math.pi * args_cli.omega_hz
    print(
        f"[cfg] swing={cpg.swing_amplitude} lift={cpg.lift_amplitude} "
        f"signs={tuple(cpg.lift_joint_signs)} scales={tuple(cpg.lift_scales)} "
        f"freq={cpg.omega / (2.0 * math.pi):.2f} Hz"
    )


def _apply_actuator_overrides(cfg) -> None:
    """Patch joint PD gains / torque limit so actuator strength can be swept
    without editing assets/eigenbot.py."""
    if all(v is None for v in (args_cli.stiffness, args_cli.damping, args_cli.effort)):
        return
    robot = getattr(cfg, "robot", None)
    actuators = getattr(robot, "actuators", None) if robot is not None else None
    if not actuators:
        print("[warn] no robot actuators found on cfg; skipping actuator overrides")
        return
    for name, act in actuators.items():
        if args_cli.stiffness is not None:
            act.stiffness = args_cli.stiffness
        if args_cli.damping is not None:
            act.damping = args_cli.damping
        if args_cli.effort is not None:
            # attribute is effort_limit on Isaac Lab 2.0, effort_limit_sim on 2.1+
            for attr in ("effort_limit", "effort_limit_sim"):
                if hasattr(act, attr) and getattr(act, attr) is not None:
                    setattr(act, attr, args_cli.effort)
        print(f"[cfg] actuator '{name}': stiffness={act.stiffness} damping={act.damping} effort={args_cli.effort}")
    # keep the reward-side torque model consistent with the real limit
    if args_cli.effort is not None and hasattr(cfg, "rewards"):
        cfg.rewards.torque_limit_hard = args_cli.effort


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
    if hasattr(env_cfg, "depth_camera"):
        env_cfg.depth_camera.use_camera = False
    _apply_overrides(env_cfg)
    _apply_actuator_overrides(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped

    feet = u.feet_indices
    foot_names = [u.robot.data.body_names[i] for i in feet.tolist()]
    # column in `feet` -> CPG leg index
    col_to_leg = {}
    for col, name in enumerate(foot_names):
        try:
            module = int(name.split("_M")[1].split("_")[0])
            col_to_leg[col] = FOOT_MODULE_TO_LEG[module]
        except (IndexError, KeyError, ValueError):
            col_to_leg[col] = col  # unknown naming: fall back to positional
    leg_to_col = {leg: col for col, leg in col_to_leg.items()}

    action = torch.zeros(1, int(env_cfg.action_space), device=u.device)
    env.reset()

    n_steps = int(args_cli.seconds / u.step_dt)
    n_settle = int(args_cli.settle / u.step_dt)
    foot_z, base_p = [], []

    with torch.inference_mode():
        for t in range(n_steps):
            env.step(action)
            if t >= n_settle:
                foot_z.append(u.robot.data.body_pos_w[0, feet, 2].clone())
                base_p.append(u.robot.data.root_pos_w[0, :3].clone())

    if not foot_z:
        print("[error] no samples collected — increase --seconds")
        env.close()
        return

    Z = torch.stack(foot_z)            # (T, 6) world foot heights
    B = torch.stack(base_p)            # (T, 3) body position
    ground = Z.min().item()            # lowest point any foot reached

    print("\n" + "=" * 74)
    print(f"GAIT DIAGNOSTIC — {args_cli.task}")
    print(f"{args_cli.seconds - args_cli.settle:.1f} s analysed, zero action (pure CPG gait)")
    print("=" * 74)
    print(f"{'leg':<22}{'foot body':<22}{'clearance':>11}{'mean h':>10}{'stance':>9}")
    print("-" * 74)

    clearances = {}
    for leg in range(len(LEG_LABELS)):
        col = leg_to_col.get(leg, leg)
        z = Z[:, col]
        low = z.min().item()
        clearance = z.max().item() - low
        mean_h = (z.mean().item() - ground) * 100.0
        stance = (z < low + 0.005).float().mean().item() * 100.0
        clearances[leg] = clearance
        print(
            f"{LEG_LABELS[leg]:<22}{foot_names[col]:<22}"
            f"{clearance * 100.0:>9.1f}cm{mean_h:>8.1f}cm{stance:>8.0f}%"
        )

    travel = (B[-1, :2] - B[0, :2]).norm().item()
    forward = (B[-1, 0] - B[0, 0]).item()
    lateral = (B[-1, 1] - B[0, 1]).item()
    duration = args_cli.seconds - args_cli.settle
    height = B[:, 2].mean().item() - ground
    target_h = getattr(getattr(env_cfg, "rewards", None), "base_height_target", None)
    print("-" * 74)
    print(
        f"body: travel {travel:.2f} m (fwd {forward:+.2f}, lat {lateral:+.2f}) "
        f"= {travel / duration:.3f} m/s | mean height {height * 100.0:.1f}cm"
        + (f" (reward target {target_h * 100.0:.0f}cm)" if target_h else "")
    )
    if target_h and height < 0.7 * target_h:
        print(
            f"NOTE: body is riding {(1 - height / target_h) * 100:.0f}% below the reward's "
            "height target — the legs are squatting under load. Try --stiffness/--effort."
        )

    rear = [clearances[0], clearances[3]]
    others = [clearances[1], clearances[2], clearances[4], clearances[5]]
    rear_avg, other_avg = sum(rear) / 2.0, sum(others) / 4.0
    print(
        f"rear pair clearance {rear_avg * 100:.1f}cm vs mid/front {other_avg * 100:.1f}cm "
        f"(ratio {rear_avg / max(other_avg, 1e-9):.2f})"
    )
    if rear_avg < 0.01:
        print("VERDICT: rear feet are dragging (<1cm). Check lift_joint_signs, then lift_scales.")
    elif rear_avg < 0.6 * other_avg:
        print("VERDICT: rear legs under-lifting relative to the others — raise lift_scales.")
    elif travel < 0.1:
        print("VERDICT: legs cycle but the body is not travelling — check swing/traction.")
    else:
        print("VERDICT: all six legs clearing and the body is moving. Good to train.")
    print("=" * 74 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
