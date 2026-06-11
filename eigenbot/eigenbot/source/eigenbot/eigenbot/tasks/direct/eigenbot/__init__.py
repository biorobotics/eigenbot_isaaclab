# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


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
