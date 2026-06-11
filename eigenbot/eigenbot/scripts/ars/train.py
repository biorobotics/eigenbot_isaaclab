# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Augmented Random Search (ARS, V2) trainer for the Eigenbot CPG task.

This is the gradient-free optimizer recommended in Li, Wei & Qiu (2023). It
trains a **linear** policy mapping observations -> 7 CPG parameters, which is
exactly the regime ARS excels at (small action space, structured gait from the
CPG). No autodiff, no value function: each iteration perturbs the policy in a
handful of random directions, rolls them out in parallel, and nudges the policy
toward the perturbations that scored higher.

The N parallel directions use 2*N environments (one for the + perturbation, one
for the -). Defaults mirror the paper: 8 directions, lr 0.015, noise 0.05.

Run inside the Isaac Lab container, e.g.:

    python scripts/ars/train.py --task Template-Eigenbot-CPG-Direct-v0 \
        --n_directions 8 --horizon 1000 --headless

Checkpoints (linear policy + observation normalizer) are saved as .pt files
under logs/ars/<experiment>/.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Eigenbot CPG policy with ARS.")
parser.add_argument("--task", type=str, default="Template-Eigenbot-CPG-Direct-v0", help="Name of the task.")
parser.add_argument("--n_directions", type=int, default=8, help="Number of random directions per iteration (N).")
parser.add_argument("--n_top", type=int, default=None, help="Use the best b directions (default: all N).")
parser.add_argument("--lr", type=float, default=0.015, help="ARS step size.")
parser.add_argument("--noise", type=float, default=0.05, help="Exploration noise (nu).")
parser.add_argument("--horizon", type=int, default=1000, help="Rollout length (policy steps) per evaluation.")
parser.add_argument("--n_iterations", type=int, default=1000, help="Number of ARS iterations.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--use_camera", action="store_true", default=False, help="Enable the Eigenbot depth camera.")
parser.add_argument("--save_interval", type=int, default=20, help="Save a checkpoint every N iterations.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 2*N envs: one per (+/-) perturbation.
NUM_ENVS = 2 * args_cli.n_directions
if args_cli.use_camera:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import eigenbot.tasks  # noqa: F401  (registers the gym tasks)


class RunningNormalizer:
    """Welford running mean/std for ARS V2 observation normalization."""

    def __init__(self, dim, device):
        self.n = torch.zeros(1, device=device)
        self.mean = torch.zeros(dim, device=device)
        self.m2 = torch.zeros(dim, device=device)
        self.device = device

    def update(self, x):  # x: (batch, dim)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (row - self.mean)

    @property
    def std(self):
        var = self.m2 / torch.clamp(self.n, min=1.0)
        return torch.sqrt(var).clamp_min(1e-6)

    def normalize(self, x):
        return (x - self.mean) / self.std


def main():
    device = app_launcher.device if hasattr(app_launcher, "device") else "cuda:0"
    torch.manual_seed(args_cli.seed)

    # --- build the vectorized env with exactly 2*N parallel rollouts ----
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=NUM_ENVS)
    if hasattr(env_cfg, "depth_camera"):
        env_cfg.depth_camera.use_camera = args_cli.use_camera
    env = gym.make(args_cli.task, cfg=env_cfg)

    obs_dim = env_cfg.observation_space
    act_dim = env_cfg.action_space
    N = args_cli.n_directions
    b = args_cli.n_top or N
    nu = args_cli.noise
    lr = args_cli.lr

    # Linear policy M: (act_dim, obs_dim), started at zero (ARS convention).
    M = torch.zeros(act_dim, obs_dim, device=device)
    normalizer = RunningNormalizer(obs_dim, device)

    log_dir = os.path.join("logs", "ars", "eigenbot_cpg", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[ARS] logging to {log_dir} | obs={obs_dim} act={act_dim} envs={NUM_ENVS}")

    def get_obs(obs_dict):
        return obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    for it in range(args_cli.n_iterations):
        # Sample N directions; build the 2N perturbed linear policies.
        deltas = torch.randn(N, act_dim, obs_dim, device=device)
        # Env e in [0, N) uses +delta_e ; env e in [N, 2N) uses -delta_(e-N).
        perturbed = torch.cat([M.unsqueeze(0) + nu * deltas, M.unsqueeze(0) - nu * deltas], dim=0)  # (2N, a, o)

        obs_dict, _ = env.reset()
        obs = get_obs(obs_dict)
        returns = torch.zeros(NUM_ENVS, device=device)

        for _ in range(args_cli.horizon):
            normalizer.update(obs)
            norm_obs = normalizer.normalize(obs)  # (2N, o)
            # Per-env linear action: bmm( (2N,a,o), (2N,o,1) ) -> (2N,a)
            action = torch.bmm(perturbed, norm_obs.unsqueeze(-1)).squeeze(-1)
            action = torch.clamp(action, -1.0, 1.0)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = get_obs(obs_dict)
            returns += reward  # env auto-resets on done; rewards still accumulate

        r_plus = returns[:N]
        r_minus = returns[N:]

        # Pick the top-b directions by their best side.
        best = torch.maximum(r_plus, r_minus)
        top_idx = torch.argsort(best, descending=True)[:b]
        rp, rm = r_plus[top_idx], r_minus[top_idx]
        d = deltas[top_idx]  # (b, a, o)

        sigma_r = torch.cat([rp, rm]).std().clamp_min(1e-6)
        update = ((rp - rm).view(b, 1, 1) * d).sum(dim=0)
        M = M + (lr / (b * sigma_r)) * update

        print(
            f"[ARS] iter {it:4d} | mean_return {returns.mean().item():8.2f} | "
            f"best {best.max().item():8.2f} | sigma_R {sigma_r.item():6.2f}"
        )

        if (it + 1) % args_cli.save_interval == 0:
            ckpt = {"M": M, "mean": normalizer.mean, "std": normalizer.std, "iter": it}
            torch.save(ckpt, os.path.join(log_dir, f"policy_{it+1:05d}.pt"))

    torch.save({"M": M, "mean": normalizer.mean, "std": normalizer.std}, os.path.join(log_dir, "policy_final.pt"))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
