# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Velodyne LiDAR sensors."""


from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
import isaaclab.sim as sim_utils

##
# Configuration
##
# TODO: needs modification
REAL_SENSE_435 = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"), 
    data_types=["depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=0.193, focus_distance=0.5, horizontal_aperture=3.89, clipping_range=(0.01, 20)
    ),
    width=1920,
    height=1080,
)
# tiled_camera: TiledCameraCfg = TiledCameraCfg(
#     prim_path="/World/envs/env_.*/Camera",
#     offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
#     data_types=["rgb"],
#     spawn=sim_utils.PinholeCameraCfg(
#         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
#     ),
#     width=80,
#     height=80,
# )

# camera = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
#         update_period=0.1,
#         height=480,
#         width=640,
#         data_types=["rgb", "distance_to_image_plane"],
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
#         ),
#         offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
#     )
"""Configuration for Velodyne Puck LiDAR (VLP-16) as a :class:`RayCasterCfg`.

Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_sensors.html#isaac-assets-sensors
https://www.intelrealsense.com/depth-camera-d435/ dataseet for d430 model
https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_sensors.html#isaac-assets-sensors for D450 from isaac sim
"""
