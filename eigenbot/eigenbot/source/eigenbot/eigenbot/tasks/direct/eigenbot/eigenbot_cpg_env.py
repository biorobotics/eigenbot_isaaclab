# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""CPG+RL Eigenbot environment.

Subclasses :class:`EigenbotEnv` and overrides only what is needed to drive the
robot from a 7D CPG action instead of 18 raw joint targets:

* ``_init_buffers``  – keep joint-dim buffers at 18 while the policy action is 7D,
  and instantiate the Hopf CPG.
* ``_pre_physics_step`` – store the 7D policy action.
* ``_apply_action``  – step the CPG each physics substep and command joints.
* ``_reset_idx``     – reset oscillator phases for reset envs.

The full observation vector (974 dims) is inherited unchanged: the 18-dim
"previous action" slot is fed the CPG's joint offsets, so the policy still sees
what was commanded and ``observation_space`` does not change. This keeps the CPG
policy directly comparable to the PPO baseline on identical observations and
rewards.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from .cpg import HopfCPG
from .eigenbot_env import EigenbotEnv
from .eigenbot_env_cfg import NUM_JOINTS
from .eigenbot_cpg_env_cfg import EigenbotCPGEnvCfg, CPG_ACTION_DIM


class EigenbotCPGEnv(EigenbotEnv):
    """Eigenbot locomotion driven by a Hopf CPG with a 7D RL action."""

    cfg: EigenbotCPGEnvCfg

    # ------------------------------------------------------------------
    def _init_buffers(self):
        # The parent sizes every joint-dimension buffer (actions, torques,
        # motor_strength, action_history_buf, ...) from ``cfg.action_space``
        # and finishes by calling ``_apply_domain_randomization()``, which
        # multiplies (ne, num_joints) robot data by those buffers. It must
        # therefore run with the true joint count (18), not the policy's 7,
        # or construction crashes on a shape mismatch. Temporarily swap the
        # value, then restore it for the gym/policy side.
        policy_dim = self.cfg.action_space
        self.cfg.action_space = NUM_JOINTS
        try:
            super()._init_buffers()
        finally:
            self.cfg.action_space = policy_dim

        ne, dev = self.num_envs, self.device
        self.num_joints = NUM_JOINTS  # parent set this from the swapped value; keep explicit

        # Policy-space action buffers (7D, or 25D in residual mode).
        self._full_action_dim = policy_dim
        self.policy_actions = torch.zeros(ne, policy_dim, device=dev)
        self.last_policy_actions = torch.zeros(ne, policy_dim, device=dev)

        # The rhythmic-gait generator.
        self.cpg = HopfCPG(self.cfg.cpg, ne, dev)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        clip_val = self.cfg.normalization.clip_actions
        self.last_policy_actions[:] = self.policy_actions
        self.policy_actions = torch.clamp(actions, -clip_val, clip_val)

    # ------------------------------------------------------------------
    def _apply_action(self) -> None:
        # Advance the CPG by one physics dt (this runs `decimation` times per
        # policy step, so the oscillators evolve at the full sim rate while the
        # policy action is held constant across the window). Only the first 7
        # action dims drive the CPG.
        cpg_action = self.policy_actions[:, :CPG_ACTION_DIM]
        offsets = self.cpg.step(cpg_action, dt=self.cfg.sim.dt)  # (ne, 18)

        # Residual mode: add small per-joint corrections from the remaining 18
        # action dims, on top of the rhythmic CPG gait.
        if self.cfg.cpg.use_residual:
            residual = self.policy_actions[:, CPG_ACTION_DIM : CPG_ACTION_DIM + NUM_JOINTS]
            offsets = offsets + self.cfg.cpg.residual_scale * torch.clamp(residual, -1.0, 1.0)

        # Expose the commanded joint offsets as the "actions" the obs/reward use.
        self.actions = offsets

        targets = self.default_dof_pos + offsets
        self.robot.set_joint_position_target(targets)

        # PD -> torque, identical to the baseline env.
        if self.cfg.domain_rand.randomize_motor:
            p_gains = self._p_gain * self.motor_strength[0]
            d_gains = self._d_gain * self.motor_strength[1]
        else:
            p_gains = self._p_gain
            d_gains = self._d_gain
        self.torques = (
            p_gains * (targets - self.robot.data.joint_pos)
            - d_gains * self.robot.data.joint_vel
        )
        torque_limit = self.cfg.rewards.torque_limit_hard
        self.torques = torch.clamp(self.torques, -torque_limit, torque_limit)

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            ids = None
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.cpg.reset(ids)
        if ids is None:
            self.policy_actions[:] = 0.0
            self.last_policy_actions[:] = 0.0
        else:
            self.policy_actions[ids] = 0.0
            self.last_policy_actions[ids] = 0.0
