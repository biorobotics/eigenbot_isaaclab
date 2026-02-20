# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots.

The following configuration parameters are available:

* :obj:`ANYMAL_B_CFG`: The ANYmal-B robot with ANYdrives 3.0
* :obj:`ANYMAL_C_CFG`: The ANYmal-C robot with ANYdrives 3.0
* :obj:`ANYMAL_D_CFG`: The ANYmal-D robot with ANYdrives 3.0

Reference:

* https://github.com/ANYbotics/anymal_b_simple_description
* https://github.com/ANYbotics/anymal_c_simple_description
* https://github.com/ANYbotics/anymal_d_simple_description

"""

from eigenbot.assets.sensors.realsense_435 import REAL_SENSE_435
from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetLSTMCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
import numpy as np
from pathlib import Path

current_path = Path(__file__).parent.parent.resolve()

EIGENBOT_USD_PATH = f"{current_path}/data/usd/eigenbot.usd"

##
# Configuration - Actuators.
##

Dynamixel_xh430_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=["bendy_joint.*"],
    saturation_effort=8,
    effort_limit=4,#3.3, #8,
    velocity_limit=2,
    stiffness={".*": 20.0},
    damping={".*": 0.5},
)
"""Configuration for Dynamixel xh430-v350 with DC actuator model."""



##
# Configuration - Articulation.
##

EIGENBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=EIGENBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=2, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        # joint_pos={
        #     'bendy_joint_M1_S1': -np.pi/4,   # [rad]
        #     'bendy_joint_M2_S2': 0.0,   # [rad]
        #     'bendy_joint_M3_S3': np.pi/4,
        #     'bendy_joint_M4_S4': -np.pi/4,
        #     'bendy_joint_M5_S5': 0.0,
        #     'bendy_joint_M6_S6': np.pi/4,
        #     'bendy_joint_M7_S7': np.pi/4,
        #     'bendy_joint_M8_S8': np.pi/4,
        #     'bendy_joint_M9_S9': np.pi/4,
        #     'bendy_joint_M10_S10': np.pi/4,
        #     'bendy_joint_M11_S11': np.pi/4,
        #     'bendy_joint_M12_S12': np.pi/4,
        #     'bendy_joint_M13_S13': np.pi/4,
        #     'bendy_joint_M14_S14': np.pi/4,
        #     'bendy_joint_M15_S15': np.pi/4,
        #     'bendy_joint_M16_S16': np.pi/4,
        #     'bendy_joint_M17_S17': np.pi/4,
        #     'bendy_joint_M18_S18': np.pi/4,
        # },
        # joint_pos={
        #     'bendy_joint_.*': 0,   # [rad]
        # },

        joint_pos={
            'bendy_joint_M1_S1': 0.0,   # [rad]
            'bendy_joint_M2_S2': 0.0,   # [rad]
            'bendy_joint_M3_S3': 0.0,
            'bendy_joint_M4_S4': 0.0,
            'bendy_joint_M5_S5': 0.0,
            'bendy_joint_M6_S6': 0.0,
            'bendy_joint_M7_S7': 0.0,
            'bendy_joint_M8_S8': 0.0,
            'bendy_joint_M9_S9': 0.0,
            'bendy_joint_M10_S10': 0.0,
            'bendy_joint_M11_S11': 0.0,
            'bendy_joint_M12_S12': 0.0,
            'bendy_joint_M13_S13': 0.0,
            'bendy_joint_M14_S14': 0.0,
            'bendy_joint_M15_S15': 0.0,
            'bendy_joint_M16_S16': 0.0,
            'bendy_joint_M17_S17': 0.0,
            'bendy_joint_M18_S18': 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": Dynamixel_xh430_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-B robot using actuator-net."""

"""Configuration of ANYmal-D robot using actuator-net.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""


##
# Configuration - Sensors.
##
# eigen TODO: needs modification
# EIGENBOT_CAMERA_CFG = REAL_SENSE_435.replace(
#     offset=RayCasterCfg.OffsetCfg(pos=(-0.310, 0.000, 0.159), rot=(0.0, 0.0, 0.0, 1.0))
# )

# EIGENBOT_LIDAR_CFG = VELODYNE_VLP_16_RAYCASTER_CFG.replace(
#     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.000, 0.0), rot=(0.0, 0.0, 0.0, 1.0))
# )
"""Configuration for the Velodyne VLP-16 sensor mounted on the ANYmal robot's base."""
