# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat,wrap_to_pi


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
General.
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)

def contact_length(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, duration_threshold: float, duration_cap: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hist_count = contact_sensor.data.force_history_reward_cnt

    contact_forces_histroy = torch.norm(contact_sensor.data.net_forces_w_history_reward[:, :, sensor_cfg.body_ids],dim=-1)
    contact_mask = (contact_forces_histroy > threshold)
    # import ipdb;ipdb.set_trace()
    # sum_history_time = torch.sum(contact_forces_histroy*contact_forces_histroy,dim=1)
    contact_duration = torch.sum(contact_mask, dim=1)
    # import ipdb;ipdb.set_trace()
    contact_duration = torch.clamp(contact_duration.float() * env.cfg.sim.dt, 0, duration_cap)
    reward = torch.sum(contact_duration - duration_threshold, dim=-1)
    # may need to cap this

    return reward


def weight_distribution(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hist_count = contact_sensor.data.force_history_reward_cnt

    contact_forces_histroy = torch.norm(contact_sensor.data.net_forces_w_history_reward[:, :, sensor_cfg.body_ids],dim=-1)
    nonzero_mask = (contact_forces_histroy != 0)
    sum_history_time = torch.sum(contact_forces_histroy*contact_forces_histroy,dim=1)
    count_values = torch.sum(nonzero_mask, dim=1)
    contact_forces_histroy_avg_time = sum_history_time/torch.clamp(count_values, min=1)
    contact_forces_histroy_avg_part = torch.mean(contact_forces_histroy_avg_time, dim=-1)
    # import ipdb;ipdb.set_trace()
    return torch.sum(torch.abs(contact_forces_histroy_avg_time - contact_forces_histroy_avg_part.reshape(contact_forces_histroy_avg_part.shape[0],1)),dim=-1)

"""
Velocity-tracking rewards.
"""
# def _reward_tracking_lin_vel(self):
#         # Tracking of linear velocity commands (xy axes)
#         lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
#         return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

# def _reward_tracking_ang_vel(self):
#         # Tracking of angular velocity commands (yaw) 
#         ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
#         return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    # print()
    # import ipdb;ipdb.set_trace()
    return torch.exp(-ang_vel_error / std**2)

def reward_stand_still(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        # import ipdb;ipdb.set_trace()
        # Penalize motion at zero commands
        articulation_asset: Articulation = env.scene[asset_cfg.name]
        curr_dof_pos = articulation_asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_dof_pos = articulation_asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        # return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
        # angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        command = env.command_manager.get_command(command_name)[:, :2]
        # import ipdb;ipdb.set_trace()
        return torch.sum(torch.abs(curr_dof_pos - default_dof_pos), dim=1) * (torch.norm(command, dim=1) < threshold) # small command gives 0

def penalty_energy(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        # import ipdb;ipdb.set_trace()
        # Penalize motion for high energy consumption
        articulation_asset: Articulation = env.scene[asset_cfg.name]
        curr_dof_vel = articulation_asset.data.joint_vel[:, asset_cfg.joint_ids]
        curr_dof_torque = articulation_asset.data.computed_torque[:, asset_cfg.joint_ids]
        energy = curr_dof_vel*curr_dof_torque
        command = env.command_manager.get_command(command_name)[:, :2]
        # import ipdb;ipdb.set_trace()
        return torch.sum(torch.abs(energy), dim=1) * (torch.norm(command, dim=1) < threshold) # small command gives 0

# def reward_inactivity(
#     env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
#     return

def reward_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # Penalize feet hitting vertical surfaces
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # import ipdb;ipdb.set_trace()
    # is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.any(torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids, :2], dim=2) >\
            4 *torch.abs(net_contact_forces[:, -1, sensor_cfg.body_ids, 2]), dim=1)


def reward_velocity_magnitude(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        # Calculate the magnitude of the base linear velocity in the horizontal plane
        asset: RigidObject = env.scene[asset_cfg.name]
    
        vel_magnitude = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        
        # Reward based on velocity magnitude
        magnitude_reward = vel_magnitude
        
        return magnitude_reward

def heading_vector_from_yaw(yaw_tensor):
    """
    Given a yaw angle tensor (Nx1), return the 2D unit heading vectors as an Nx2 tensor.
    """
    x = torch.cos(yaw_tensor)
    y = torch.sin(yaw_tensor)
    return torch.stack((x, y), dim=1)  # Stack along the second dimension to get an Nx2 tensor

def reward_tracking_goal_vel(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), tracking_sigma: float = 0.25) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    yaw = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    command_heading = env.command_manager.get_term(command_name).heading_target
    target_vec = heading_vector_from_yaw(command_heading)
    norm = torch.norm(target_vec, dim=-1, keepdim=True)
    target_vec_norm = target_vec / (norm + 1e-5)
    cur_vel = asset.data.root_lin_vel_w[:,:2]
    
    vel_rel_heading = torch.sum(target_vec_norm * cur_vel, dim=-1)
    # import ipdb;ipdb.set_trace()
    command_vel = env.command_manager.get_command(command_name)
    lin_vel_error = torch.square(vel_rel_heading - command_vel[:, 0])
    return torch.exp(-lin_vel_error/tracking_sigma)
def reward_delta_yaw(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize high delta_yaw values
    asset = env.scene[asset_cfg.name]
    yaw = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    command_heading = env.command_manager.get_term(command_name).heading_target
    rew = torch.exp(-torch.abs(wrap_to_pi(command_heading - yaw)))
    return rew


def reward_rule_1(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg):
    """
    Rule 1: Prevents simultaneous ipsilateral liftoff in anterior legs.

    Returns:
        Tensor: Penalty (negative reward).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)[:,0,:] > threshold
    last_contact = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)[:,1,:] > threshold
    contact_filt =  torch.logical_or(is_contact, last_contact)

    middle_leg_contact = contact_filt[:, [1, 4]]
    front_leg_contact = contact_filt[:, [0, 3]]  # Front legs
    hind_leg_contact = contact_filt[:, [2, 5]] 

    middle_up = ~middle_leg_contact  # Middle leg lifted
    # front_down = front_leg_contact | hind_leg_contact  # At least one of front/hind in stance
    rule_1_a = torch.any(middle_up & front_leg_contact, dim=1)

    hind_up = ~hind_leg_contact  # Either front or hind lifted
    rule_1_b = torch.any(hind_up & middle_leg_contact, dim=1)
    # Total reward
    rule_1_reward = rule_1_a.float() + rule_1_b.float()
    return rule_1_reward



def reward_rule_3(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, exp_coeff_rule3: float=-10):
    """
    Rule 3: Encourages swing initiation in posterior & contralateral legs based on stance progress.

    Returns:
        Tensor: Reward (positive).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)[:,0,:] > threshold
    last_contact = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)[:,1,:] > threshold
    # import ipdb;ipdb.set_trace()
    # sum over contacts for each environment
    contact_flag = torch.logical_or(is_contact, last_contact)

    # Define leg contacts
    middle_leg_contact = contact_flag[:, [1, 4]]  # Middle legs
    front_leg_contact = contact_flag[:, [0, 3]]  # Front legs
    hind_leg_contact = contact_flag[:, [2, 5]]  # Hind legs

    # --- Extract Only Middle Leg's Contact Flag ---
    middle_liftoff_flag = ~contact_flag[:, [1, 4]]  # 1 if middle leg is up, 0 if on ground
    front_liftoff_flag = ~contact_flag[:, [0, 3]]  # 1 if front leg is up, 0 if on ground
    hind_liftoff_flag = ~contact_flag[:, [2, 5]]  # 1 if hind leg is up, 0 if on ground



    # --- Track Time for Middle Leg Liftoff ---
    middle_liftoff_time_lpsi, middle_liftoff_time_contra, front_liftoff_time_contra, hind_liftoff_time_lpsi, hind_liftoff_time_contra = contact_sensor.update_liftoff_time(~middle_leg_contact * ~front_leg_contact, 
                                ~middle_leg_contact[:, [0,1]]*~middle_leg_contact[:, [1,0]],
                                ~front_leg_contact[:, [0,1]]*~front_leg_contact[:, [1,0]],
                                ~hind_leg_contact*~middle_leg_contact,
                                ~hind_leg_contact[:, [0,1]]*~hind_leg_contact[:, [1,0]],
                                )
    exp_coeff = exp_coeff_rule3
    # --- Exponentially Decreasing Reward for hind Liftoff ---
    hind_reward = torch.sum(torch.exp(exp_coeff * hind_liftoff_time_lpsi)+torch.exp(exp_coeff * hind_liftoff_time_contra) ,dim=1) # Decay over time
    middle_reward = torch.sum(torch.exp(exp_coeff * middle_liftoff_time_lpsi)+torch.exp(exp_coeff * middle_liftoff_time_contra),dim=1) # Decay over time
    front_reward = torch.sum(torch.exp(exp_coeff * front_liftoff_time_contra),dim=1) # Decay over time
    # Total reward
    rule_3_reward = hind_reward+middle_reward+front_reward

    return rule_3_reward