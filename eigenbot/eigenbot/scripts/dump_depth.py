# SPDX-License-Identifier: BSD-3-Clause
"""Dump Eigenbot depth-camera frames to PNG and print per-frame statistics.

Diagnostic for the depth camera: confirms the sensor renders, that it is aimed
at the terrain in front of the robot, and that the processed frame handed to the
policy is not blank. Run this before trusting any camera training run.

Usage (from <repo>/eigenbot/eigenbot):
    python scripts/dump_depth.py --num_envs 16 --steps 120 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump Eigenbot depth camera frames to PNG.")
parser.add_argument("--task", type=str, default="Template-Eigenbot-Direct-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=120, help="Environment steps to run.")
parser.add_argument("--every", type=int, default=15, help="Capture a timeline frame every N steps.")
parser.add_argument("--out", type=str, default="depth_dump", help="Output directory for PNGs.")
parser.add_argument("--action", type=str, default="zero", choices=["zero", "random"], help="Action source.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# the depth camera needs the renderer, exactly as train.py does it
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import eigenbot.tasks  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_grid(frames, titles, path, vmin, vmax, suptitle, label):
    """Save a grid of 2-D arrays as one annotated PNG."""
    n = len(frames)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False)
    im = None
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < n:
            im = ax.imshow(frames[i], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(titles[i], fontsize=8)
    fig.suptitle(suptitle, fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=label)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if not hasattr(env_cfg, "depth_camera"):
        raise SystemExit(f"Task {args_cli.task} has no depth_camera config.")
    env_cfg.depth_camera.use_camera = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    cam = getattr(u, "_depth_camera", None)
    if cam is None:
        raise SystemExit("Depth camera was not created -- use_camera plumbing is broken.")

    dc = u.cfg.depth_camera
    far = float(dc.far_clip)
    print(f"[INFO] camera prim   : {dc.sensor.prim_path}")
    print(f"[INFO] mount pos     : {dc.position}  rot: {dc.offset_rot}  pitch jitter: {dc.angle} deg")
    print(f"[INFO] native WxH    : {dc.original} -> processed {dc.resized}")
    print(f"[INFO] clip range    : {dc.near_clip} .. {far} m   capture every {dc.update_interval} steps")
    print(f"[INFO] obs width     : {env.observation_space}")

    os.makedirs(args_cli.out, exist_ok=True)
    env.reset()

    timeline, timeline_titles = [], []

    def grab():
        """Return raw depth in metres, NaN/Inf replaced by far_clip."""
        raw = cam.data.output["depth"]
        if raw is None:
            return None, None
        r = raw.squeeze(-1).float()
        finite = torch.isfinite(r)
        return torch.where(finite, r, torch.full_like(r, far)), finite

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if args_cli.action == "random":
                act = 2 * torch.rand(env.action_space.shape, device=u.device) - 1
            else:
                act = torch.zeros(env.action_space.shape, device=u.device)
            env.step(act)

            if step % args_cli.every != 0:
                continue
            rr, finite = grab()
            if rr is None:
                print(f"[step {step:4d}] camera output is None -- nothing rendered")
                continue
            at_far = (rr >= far - 1e-3).float().mean().item()
            near = (rr <= 0.05).float().mean().item()
            print(
                f"[step {step:4d}] min={rr.min():.3f} max={rr.max():.3f} mean={rr.mean():.3f} m | "
                f"frac_at_far={at_far:.3f} frac_under_5cm={near:.3f} nonfinite={(~finite).float().mean():.3f}"
            )
            timeline.append(rr[0].cpu().numpy())
            timeline_titles.append(f"env0 step {step}")

    rr, _ = grab()
    if rr is not None:
        n = min(8, rr.shape[0])
        save_grid(
            [rr[i].cpu().numpy() for i in range(n)],
            [f"env {i}" for i in range(n)],
            os.path.join(args_cli.out, "depth_across_envs.png"),
            0.0, far,
            "Front depth camera -- across envs (final step)", "depth (m)",
        )

    if timeline:
        save_grid(
            timeline, timeline_titles,
            os.path.join(args_cli.out, "depth_timeline_env0.png"),
            0.0, far,
            "Front depth camera -- env 0 over time", "depth (m)",
        )

    if hasattr(u, "depth_buffer"):
        pb = u.depth_buffer[:, -1].float()
        n = min(8, pb.shape[0])
        print(
            f"[INFO] processed buffer: shape {tuple(pb.shape)} "
            f"min={pb.min():.3f} max={pb.max():.3f} mean={pb.mean():.3f}"
        )
        save_grid(
            [pb[i].cpu().numpy() for i in range(n)],
            [f"env {i}" for i in range(n)],
            os.path.join(args_cli.out, "depth_processed_obs.png"),
            -0.5, 0.5,
            "Processed frame as fed to the policy", "normalized (-0.5 .. 0.5)",
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
