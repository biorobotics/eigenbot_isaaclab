# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Offline sanity tests for the Hopf CPG — no Isaac Sim or GPU required.

Runs the *actual* ``cpg.py`` (loaded by file path so nothing from isaaclab is
imported) against a stub config that mirrors the ``CPGCfg`` defaults, and
checks the properties the implementation guide promises:

  1. tripod phase locking   — legs {0,2,4} vs {1,3,5} exactly half a cycle apart
  2. stable limit cycle     — radius stays ~1 over 10 s of integration
  3. perturbation recovery  — a large oscillator-state kick decays instead of
                              diverging (exercises the Euler guard)
  4. gait direction         — the leg protracts (swing offset increases) while
                              lifted; regression test for ``lift_phase_sign``
  5. bounded offsets        — worst-case action keeps |offset| under the ±π/2
                              joint limits
  6. steering asymmetry     — theta scales left/right strides oppositely
  7. reset()                — restores selected envs onto the limit cycle
  8. batching               — shapes correct for num_envs > 1

Usage (any machine with PyTorch):

    python scripts/test_cpg.py
"""

from __future__ import annotations

import importlib.util
import math
import os

import torch

# --- load cpg.py directly, bypassing the package __init__ (which imports gym/isaaclab)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CPG_PATH = os.path.join(
    _HERE, "..", "source", "eigenbot", "eigenbot", "tasks", "direct", "eigenbot", "cpg.py"
)
_spec = importlib.util.spec_from_file_location("cpg_offline", _CPG_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HopfCPG = _mod.HopfCPG


class StubCfg:
    """Mirror of the ``CPGCfg`` defaults in ``eigenbot_cpg_env_cfg.py``.

    Kept as a plain class so this script has zero isaaclab dependencies. If you
    change a default in ``CPGCfg``, change it here too.
    """

    num_legs = 6
    num_joints = 18
    alpha = 100.0
    mu = 1.0
    omega = 2.0 * math.pi * 1.5
    coupling_weight = 1.0
    phase_offsets = (0.0, math.pi, 0.0, math.pi, 0.0, math.pi)
    b_min = 0.5
    b_max = 1.5
    turn_gain = 0.6
    swing_amplitude = 0.45
    lift_amplitude = 0.55
    lift_phase_sign = -1.0
    leg_joint_indices = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (9, 10, 11),
        (12, 13, 14),
        (15, 16, 17),
    )
    lift_joint_signs = (1.0, -0.5)
    leg_sides = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)


DT = 0.005  # physics dt the env integrates the CPG at
DEVICE = "cpu"
TRIPOD = torch.tensor(StubCfg.phase_offsets)


def _zero_action(ne: int) -> torch.Tensor:
    # theta = 0, raw b = 0 -> b = (b_min + b_max) / 2 = 1.0
    return torch.zeros(ne, 7)


def _phase_diff(phases: torch.Tensor) -> torch.Tensor:
    """Per-leg phase offset vs leg 0, wrapped to [-pi, pi]."""
    d = phases - phases[:, [0]]
    return (d + math.pi) % (2 * math.pi) - math.pi


def test_tripod_phase_locking():
    cpg = HopfCPG(StubCfg, num_envs=2, device=DEVICE)
    for _ in range(int(10.0 / DT)):
        cpg.step(_zero_action(2), DT)
    d = _phase_diff(cpg.phases).abs()
    expected = TRIPOD.unsqueeze(0).expand_as(d)
    assert torch.allclose(d, expected, atol=0.05), f"phase offsets {d[0]} != tripod"
    print("[ok] tripod phase locking (groups exactly pi apart)")


def test_limit_cycle_stable():
    cpg = HopfCPG(StubCfg, num_envs=1, device=DEVICE)
    max_err = 0.0
    for _ in range(int(10.0 / DT)):
        cpg.step(_zero_action(1), DT)
        r = (cpg.x**2 + cpg.y**2).sqrt()
        max_err = max(max_err, (r - 1.0).abs().max().item())
    assert max_err < 0.02, f"limit cycle drifted: max |r-1| = {max_err}"
    print(f"[ok] limit cycle stable (max |r-1| = {max_err:.4f})")


def test_perturbation_recovery():
    cpg = HopfCPG(StubCfg, num_envs=1, device=DEVICE)
    cpg.x[:, 2] += 2.5  # large radial kick; diverged before the Euler guard
    cpg.y[:, 2] -= 2.0
    for _ in range(int(3.0 / DT)):
        cpg.step(_zero_action(1), DT)
    assert torch.isfinite(cpg.x).all() and torch.isfinite(cpg.y).all(), "diverged to NaN"
    r = (cpg.x**2 + cpg.y**2).sqrt()
    assert (r - 1.0).abs().max() < 0.05, f"did not return to limit cycle: r = {r}"
    d = _phase_diff(cpg.phases).abs()
    assert torch.allclose(d, TRIPOD.unsqueeze(0), atol=0.1), "tripod phasing not recovered"
    print("[ok] perturbation recovery (kick of 3.2 decays, phasing restored)")


def test_protraction_during_lift():
    """Regression test for lift_phase_sign: swing offset must INCREASE while
    the leg is lifted (protract through the air, stroke rearward in stance)."""
    cpg = HopfCPG(StubCfg, num_envs=1, device=DEVICE)
    deltas = []
    prev_swing = None
    lift_max = StubCfg.lift_amplitude
    for _ in range(int(2.0 / DT)):
        off = cpg.step(_zero_action(1), DT)[0]
        swing, lift = off[0].item(), off[1].item()  # leg 0: swing joint, +lift joint
        if prev_swing is not None and lift > 0.3 * lift_max:
            deltas.append(swing - prev_swing)
        prev_swing = swing
    mean_d = sum(deltas) / len(deltas)
    assert mean_d > 0, (
        f"swing moves {mean_d:+.4f}/step while lifted -> leg retracts in the air "
        "(robot would walk backward); check cfg.lift_phase_sign"
    )
    print(f"[ok] gait direction (swing {mean_d:+.4f}/step while lifted -> protracts)")


def test_bounded_offsets():
    cpg = HopfCPG(StubCfg, num_envs=1, device=DEVICE)
    worst_action = torch.ones(1, 7)  # theta = 1, b = b_max on every leg
    worst = 0.0
    for _ in range(int(4.0 / DT)):
        off = cpg.step(worst_action, DT)
        worst = max(worst, off.abs().max().item())
    limit = math.pi / 2
    assert worst < limit, f"offset {worst:.2f} rad exceeds ±{limit:.2f} joint limit"
    print(f"[ok] bounded offsets (worst case {worst:.2f} rad < {limit:.2f})")


def test_steering_asymmetry():
    cpg = HopfCPG(StubCfg, num_envs=1, device=DEVICE)
    action = torch.zeros(1, 7)
    action[0, 0] = 1.0  # full turn
    left_max, right_max = 0.0, 0.0
    for _ in range(int(4.0 / DT)):
        off = cpg.step(action, DT)
        left_max = max(left_max, off[0, 0].abs().item())   # leg 0 (left) swing
        right_max = max(right_max, off[0, 3].abs().item())  # leg 1 (right) swing
    # steer = 1 -/+ turn_gain -> 0.4 vs 1.6 stride scaling
    ratio = right_max / max(left_max, 1e-9)
    assert 3.0 < ratio < 5.0, f"L/R stride ratio {ratio:.2f}, expected ~4 (1.6/0.4)"
    print(f"[ok] steering asymmetry (R/L stride ratio {ratio:.2f} ~ 4)")


def test_reset():
    cpg = HopfCPG(StubCfg, num_envs=3, device=DEVICE)
    for _ in range(100):
        cpg.step(_zero_action(3), DT)
    cpg.reset(torch.tensor([1]))
    r0 = StubCfg.mu**0.5
    exp_x = r0 * torch.cos(TRIPOD)
    exp_y = r0 * torch.sin(TRIPOD)
    assert torch.allclose(cpg.x[1], exp_x, atol=1e-5) and torch.allclose(cpg.y[1], exp_y, atol=1e-5)
    assert not torch.allclose(cpg.x[0], exp_x, atol=1e-3), "env 0 should NOT have been reset"
    print("[ok] reset() restores only the selected envs onto the limit cycle")


def test_batched_shapes():
    ne = 4
    cpg = HopfCPG(StubCfg, num_envs=ne, device=DEVICE)
    off = cpg.step(torch.randn(ne, 7), DT)
    assert off.shape == (ne, StubCfg.num_joints), f"bad offset shape {off.shape}"
    assert cpg.phases.shape == (ne, StubCfg.num_legs)
    print("[ok] batched shapes (num_envs=4 -> offsets (4, 18))")


if __name__ == "__main__":
    test_tripod_phase_locking()
    test_limit_cycle_stable()
    test_perturbation_recovery()
    test_protraction_during_lift()
    test_bounded_offsets()
    test_steering_asymmetry()
    test_reset()
    test_batched_shapes()
    print("\nAll CPG offline tests passed.")
