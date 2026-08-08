# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Config for the CPG+RL variant of the Eigenbot locomotion task.

Subclasses :class:`EigenbotEnvCfg` and changes only the action space (18 -> 7)
plus a nested :class:`CPGCfg`. Everything else (observations, rewards, terrain,
domain randomization) is inherited unchanged so the CPG policy is directly
comparable to the PPO baseline.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from .eigenbot_env_cfg import EigenbotEnvCfg, NUM_JOINTS

# 7D action: turning theta + one foot-mapping gain b per leg.
CPG_ACTION_DIM = 7


@configclass
class CPGCfg:
    """Hopf-CPG hyperparameters and joint-mapping geometry."""

    # --- oscillator dynamics -----------------------------------------
    num_legs: int = 6
    num_joints: int = NUM_JOINTS  # 18
    alpha: float = 100.0          # convergence speed to the limit cycle
    mu: float = 1.0               # squared limit-cycle radius (amplitude^2)
    omega: float = 2.0 * math.pi * 1.5  # intrinsic frequency (rad/s) ~1.5 Hz gait
    coupling_weight: float = 1.0  # ring coupling strength (0 disables coupling)

    # Tripod gait: legs 0,2,4 in phase; legs 1,3,5 half a cycle out of phase.
    phase_offsets: tuple = (0.0, math.pi, 0.0, math.pi, 0.0, math.pi)

    # --- action -> gait mapping --------------------------------------
    b_min: float = 0.5            # per-leg foot-mapping gain range
    # Capped 1.5 -> 1.25: at b=1.5 the swing offset (0.45*1.5 = 0.675 rad) plus
    # the +/-pi/4 default stance runs into the +/-pi/2 joint limit, which is what
    # made the front legs fold. Measurements also showed b>1 slowed the robot
    # down (more load on the rear pair), so the top of the old range was useless.
    b_max: float = 1.25
    turn_gain: float = 0.6        # how strongly theta biases L/R stride
    swing_amplitude: float = 0.45  # rad of fore/aft protraction at b=1
    lift_amplitude: float = 0.55   # rad of leg lift during swing at b=1
    # Which half-cycle the legs lift in: -1.0 lifts while y < 0, pairing
    # airborne protraction with a rearward stance stroke (forward travel when a
    # positive swing offset = leg forward). Flip to 1.0 if the robot's joint
    # sign conventions turn out to be reversed in sim/hardware.
    lift_phase_sign: float = -1.0

    # --- joint geometry assumptions (TUNE to real kinematics) --------
    # 18 joints are grouped 3-per-leg in articulation order. Within each leg the
    # first index is treated as the swing (coxa/yaw) joint and the next two as
    # lift (femur/tibia) joints. Verify against the URDF joint order and adjust.
    leg_joint_indices: tuple = (
        (0, 6, 12),   # leg 0: M1 swing, M7 lift, M13 lift  (rear,  side A)
        (1, 7, 13),   # leg 1: M2, M8, M14                  (mid,   side A)
        (2, 8, 14),   # leg 2: M3, M9, M15                  (front, side A)
        (3, 9, 15),   # leg 3: M4, M10, M16                 (rear,  side B)
        (4, 10, 16),  # leg 4: M5, M11, M17                 (mid,   side B)
        (5, 11, 17),  # leg 5: M6, M12, M18                 (front, side B)
    )
    # Signs applied to the two lift joints [M(k+6), M(k+12)].
    # The URDF rotates the distal module's frame so its axis is ANTIPARALLEL to
    # the middle module's (connection_12 Rx(pi/2) then connection_18
    # Rz(pi/2)Rx(pi/2)). Same-signed commands therefore rotate the two segments
    # in OPPOSITE world directions — that is what folds the leg into its Z
    # stance, and it is why the robot's default pose sets M7..M18 all to +pi/4.
    # An opposite-signed pair (the original (1.0, -0.5)) instead swings both
    # segments the same way, so the shank trails the thigh, the foot scuffs
    # backward through swing, and the loaded rear legs slip out.
    lift_joint_signs: tuple = (1.0, 0.5)
    # Static per-leg lift multiplier, indexed like leg_joint_indices (legs 0..5 =
    # modules M1..M6). Legs 0 and 3 are the REAR pair: their attachments sit
    # furthest back (URDF x = -0.13 vs +0.05 for the front pair), so they carry
    # the most load and drag with uniform lift. Boosting them here fixes the gait
    # open-loop instead of hoping RL discovers it (velocity reward is reachable
    # on four legs, so the policy happily leaves the rears flat).
    # Measured with scripts/diag_gait.py at 1.0 Hz: leg 0 was the last laggard
    # (1.3cm clearance / 78% stance vs ~2.5cm / 64% for the front legs), so it
    # gets the largest boost. Tuning these is the correct lever for uneven
    # clearance — the joint PD gains and torque limit mirror the real hardware
    # and must NOT be raised to compensate.
    lift_scales: tuple = (1.7, 1.0, 1.0, 1.35, 1.0, 1.0)
    # Body side per leg: +1 left, -1 right. Used for differential steering.
    leg_sides: tuple = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)   # M1-3 one side, M4-6 the other

    # --- residual mode -----------------------------------------------
    # If True, the policy outputs an additional 18 per-joint corrections that are
    # added (scaled by residual_scale) on top of the CPG offsets. This recovers
    # some of PPO's flexibility while keeping the rhythmic CPG backbone. Action
    # space becomes 7 + 18 = 25. Keep residual_scale small so the CPG dominates.
    use_residual: bool = False
    residual_scale: float = 0.1  # rad of per-joint correction at |action| = 1


@configclass
class EigenbotCPGEnvCfg(EigenbotEnvCfg):
    """Eigenbot env where the policy outputs 7 CPG parameters instead of 18 joints.

    With ``cpg.use_residual = True`` the policy additionally emits 18 per-joint
    residual corrections, making the action space 7 + 18 = 25.
    """

    # The policy emits 7 CPG numbers (+ 18 residuals if enabled). The exact size
    # is finalized in __post_init__ since it depends on cpg.use_residual.
    action_space: int = CPG_ACTION_DIM

    cpg: CPGCfg = CPGCfg()

    def __post_init__(self):
        # configclass augments (does not replace) this hook, so we only set the
        # action-space size here based on whether residual mode is enabled.
        self.action_space = CPG_ACTION_DIM + (NUM_JOINTS if self.cpg.use_residual else 0)
