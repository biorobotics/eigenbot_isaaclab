# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# Baseline: end-to-end PPO over all 18 joint targets. This is the trunk task and
# the reference every other approach is measured against.
#
# NOTE: keep this pointed at `eigenbot_env_cfg`. It is the config the CPG task
# inherits from, so sharing it is what makes the CPG-vs-PPO comparison a
# comparison of methods rather than of reward functions. The alternative reward
# config lives in its own task below.
gym.register(
    id="Template-Eigenbot-Direct-v0",
    entry_point=f"{__name__}.eigenbot_env:EigenbotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eigenbot_env_cfg:EigenbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# CPG+RL variant: 7D action space (turning + 6 foot-mapping gains) driven by a
# Hopf central pattern generator. See eigenbot_cpg_env.py / cpg.py.
gym.register(
    id="Template-Eigenbot-CPG-Direct-v0",
    entry_point=f"{__name__}.eigenbot_cpg_env:EigenbotCPGEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eigenbot_cpg_env_cfg:EigenbotCPGEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cpg_ppo_cfg:CPGPPORunnerCfg",
    },
)

# Alternative reward config (Eshan): same 18-DOF env and PPO agent as the
# baseline, but `eigenbot_env_mycfg.py` weights leg lifting differently
# (feet_air_time 1.8, tracking_goal_vel 4.0, lin_vel_x (0.0, 0.5)).
#
# It gets its own task rather than replacing the baseline's config, so it stays
# runnable without silently changing what `Template-Eigenbot-Direct-v0` trains
# against. It is NOT part of the CPG-vs-PPO head-to-head — its rewards differ, so
# its reward totals are not comparable with the other two tasks.
#
# It shares `PPORunnerCfg`, so its runs land in logs/rsl_rl/eigenbot_locomotion/
# alongside the baseline's. Pass --run_name mycfg to tell them apart.
gym.register(
    id="Template-Eigenbot-MyCfg-Direct-v0",
    entry_point=f"{__name__}.eigenbot_env:EigenbotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eigenbot_env_mycfg:EigenbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
