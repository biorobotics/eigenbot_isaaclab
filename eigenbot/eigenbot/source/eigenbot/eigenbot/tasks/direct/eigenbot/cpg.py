# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Batched Hopf Central Pattern Generator (CPG) for the Eigenbot hexapod.

This is the rhythmic-gait generator described in:
    Li, Wei & Qiu (2023), "Combined Reinforcement Learning and CPG Algorithm to
    Generate Terrain-Adaptive Gait of Hexapod Robots", MDPI Actuators 12(4):157.

Design choices for this port
----------------------------
* One Hopf oscillator per leg (6 total), diffusively coupled in a fixed
  tripod-phase pattern (legs {0,2,4} vs {1,3,5} half a period apart).
* The oscillator output is mapped **directly to joint-space offsets** (a swing
  joint + two lift joints per leg) rather than to a foot-tip XYZ trajectory.
  This deliberately avoids needing an analytic inverse-kinematics model for the
  modular "bendy" legs, and plugs straight into the existing
  ``target = default_dof_pos + offset`` actuation path. If you later add an IK
  module, only ``_map_to_joints`` needs to change.
* The RL action is the paper's 7D vector: turning scalar ``theta in [-1, 1]``
  plus one foot-mapping gain ``b_i`` per leg. ``theta`` biases left/right stride
  for steering; ``b_i`` scales per-leg stride/clearance for terrain adaptation.

Everything is vectorized over environments and runs on the sim device. The CPG
state is integrated at the *physics* dt (call ``step`` once per physics
substep), while the action is held constant across the decimation window.

All geometry/sign assumptions live in the ``*_JOINTS`` / amplitude constants of
:class:`~eigenbot.tasks.direct.eigenbot.eigenbot_cpg_env_cfg.CPGCfg` so they can
be tuned to the real kinematics without touching this file.
"""

from __future__ import annotations

import torch


class HopfCPG:
    """Vectorized ring of Hopf oscillators mapped to hexapod joint offsets.

    Parameters
    ----------
    cfg:
        A ``CPGCfg`` instance (see ``eigenbot_cpg_env_cfg``).
    num_envs:
        Number of parallel environments.
    device:
        Torch device string.
    """

    def __init__(self, cfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.num_legs = cfg.num_legs  # 6

        # --- oscillator state: cartesian (x, y) per leg -------------------
        # Initialise on the limit cycle (radius sqrt(mu)) at the tripod phases
        # so the very first steps already produce a clean gait.
        phase0 = torch.tensor(cfg.phase_offsets, device=device)  # (num_legs,)
        r0 = cfg.mu ** 0.5
        self.x = (r0 * torch.cos(phase0)).expand(num_envs, -1).clone()
        self.y = (r0 * torch.sin(phase0)).expand(num_envs, -1).clone()

        # --- static mapping tensors --------------------------------------
        # Per-leg joint index triples [swing, lift_a, lift_b].
        self._leg_joint_idx = torch.tensor(cfg.leg_joint_indices, dtype=torch.long, device=device)  # (num_legs, 3)
        # Sign / amplitude applied to each of the 3 joints in a leg.
        self._swing_amp = cfg.swing_amplitude
        self._lift_amp = cfg.lift_amplitude
        self._lift_signs = torch.tensor(cfg.lift_joint_signs, device=device)  # (2,)
        # Static per-leg lift multiplier: the rear pair carries more of the body
        # weight (its attachment points sit furthest back), so equal commanded
        # lift leaves those feet dragging. See CPGCfg.lift_scales.
        self._lift_scales = torch.tensor(cfg.lift_scales, device=device).unsqueeze(0)  # (1, num_legs)
        # Side of body for each leg (+1 left, -1 right) for differential steering.
        self._leg_side = torch.tensor(cfg.leg_sides, device=device)  # (num_legs,)

        self.num_joints = cfg.num_joints  # 18

    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset oscillators of the given envs back onto the limit cycle."""
        phase0 = torch.tensor(self.cfg.phase_offsets, device=self.device)
        r0 = self.cfg.mu ** 0.5
        x0 = r0 * torch.cos(phase0)
        y0 = r0 * torch.sin(phase0)
        if env_ids is None:
            self.x[:] = x0
            self.y[:] = y0
        else:
            self.x[env_ids] = x0
            self.y[env_ids] = y0

    # ------------------------------------------------------------------
    def _decode_action(self, action: torch.Tensor):
        """Split the 7D policy action into (theta, b) and rescale to ranges.

        action: (num_envs, 7) roughly in [-1, 1].
        Returns theta (num_envs,) in [-1, 1] and b (num_envs, num_legs) in
        [b_min, b_max].
        """
        action = torch.tanh(action)
        theta = action[:, 0]
        b_raw = action[:, 1 : 1 + self.num_legs]  # (ne, num_legs) in [-1, 1]
        b_mid = 0.5 * (self.cfg.b_max + self.cfg.b_min)
        b_half = 0.5 * (self.cfg.b_max - self.cfg.b_min)
        b = b_mid + b_half * b_raw
        return theta, b

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the oscillators by ``dt`` and return joint offsets.

        Returns a (num_envs, num_joints) tensor of position offsets to be added
        to ``default_dof_pos``.
        """
        cfg = self.cfg
        theta, b = self._decode_action(action)

        # --- Hopf dynamics with diffusive ring coupling ------------------
        r2 = self.x * self.x + self.y * self.y  # (ne, num_legs)
        conv = cfg.alpha * (cfg.mu - r2)
        # Euler-integration guard: bound the radial contraction rate so a large
        # state perturbation (e.g. future sensory feedback wired into the
        # oscillators) cannot overshoot the fixed point and diverge. Without
        # this, kicks pushing r beyond ~2.2 blow up at alpha=100, dt=0.005.
        conv = torch.clamp(conv, min=-0.9 / dt, max=0.9 / dt)
        omega = cfg.omega  # rad/s (single intrinsic frequency)

        # Coupling: pull each oscillator toward its desired phase offset
        # relative to a reference (leg 0). Implemented as rotation of the
        # neighbour state into this leg's frame (standard CPG phase coupling).
        dx = conv * self.x - omega * self.y
        dy = conv * self.y + omega * self.x
        if cfg.coupling_weight > 0.0:
            # desired phase difference matrix is encoded by rotating leg j's
            # (x, y) by (phase_i - phase_j) before averaging.
            phase = torch.tensor(cfg.phase_offsets, device=self.device)  # (L,)
            for i in range(self.num_legs):
                cx = torch.zeros(self.num_envs, device=self.device)
                cy = torch.zeros(self.num_envs, device=self.device)
                for j in range(self.num_legs):
                    if i == j:
                        continue
                    d = (phase[i] - phase[j]).item()
                    cos_d, sin_d = torch.cos(torch.tensor(d)).item(), torch.sin(torch.tensor(d)).item()
                    # rotate leg j into leg i's target frame
                    cx = cx + (cos_d * self.x[:, j] - sin_d * self.y[:, j])
                    cy = cy + (sin_d * self.x[:, j] + cos_d * self.y[:, j])
                cx = cx / (self.num_legs - 1)
                cy = cy / (self.num_legs - 1)
                dx[:, i] = dx[:, i] + cfg.coupling_weight * (cx - self.x[:, i])
                dy[:, i] = dy[:, i] + cfg.coupling_weight * (cy - self.y[:, i])

        # Euler integration (dt is the physics dt, small enough for stability).
        self.x = self.x + dx * dt
        self.y = self.y + dy * dt

        # --- map oscillator state -> joint offsets -----------------------
        return self._map_to_joints(theta, b)

    # ------------------------------------------------------------------
    def _map_to_joints(self, theta: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Convert current oscillator state + (theta, b) into 18 joint offsets.

        Swing joint follows the horizontal oscillator component (fore/aft
        protraction). Lift joints engage only during the swing half-cycle
        (selected by ``cfg.lift_phase_sign``) to give ground clearance, scaled
        by the per-leg gain ``b`` and differential steering from ``theta``.
        """
        ne = self.num_envs
        offsets = torch.zeros(ne, self.num_joints, device=self.device)

        # Normalize oscillator state to unit limit cycle for stable amplitudes.
        r = torch.sqrt(self.x * self.x + self.y * self.y).clamp_min(1e-6)
        x_n = self.x / r
        y_n = self.y / r

        # Differential steering: scale stride of each leg by its side.
        # theta>0 turns one way by shortening one side's stride.
        side = self._leg_side.unsqueeze(0)  # (1, L)
        steer = 1.0 - self.cfg.turn_gain * theta.unsqueeze(1) * side  # (ne, L)
        steer = steer.clamp(0.0, 2.0)

        swing = self._swing_amp * b * steer * x_n            # (ne, L)
        # Lift during the half-cycle selected by cfg.lift_phase_sign. The
        # oscillator rotates counter-clockwise, so x_n *increases* while
        # y_n < 0: lifting there (sign = -1, the default) makes the leg
        # protract through the air and stroke rearward during stance, i.e. the
        # body travels forward when a positive swing offset means "leg forward".
        clearance = torch.clamp(self.cfg.lift_phase_sign * y_n, min=0.0)
        lift = self._lift_amp * self._lift_scales * b * clearance   # (ne, L)

        # Scatter into the 18-dim joint vector.
        for leg in range(self.num_legs):
            j_sw, j_la, j_lb = self._leg_joint_idx[leg]
            offsets[:, j_sw] = swing[:, leg]
            offsets[:, j_la] = self._lift_signs[0] * lift[:, leg]
            offsets[:, j_lb] = self._lift_signs[1] * lift[:, leg]

        return offsets

    # ------------------------------------------------------------------
    @property
    def phases(self) -> torch.Tensor:
        """Current oscillator phases (num_envs, num_legs), for logging/obs."""
        return torch.atan2(self.y, self.x)
