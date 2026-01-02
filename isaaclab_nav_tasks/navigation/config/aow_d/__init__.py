# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, navigation_env_cfg

##
# Register Gym environments.
##

##############################################################################################################
# MDPO

gym.register(
    id="Isaac-Nav-MDPO-AoW-D-v0",
    entry_point="isaaclab_nav_tasks.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AowDNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AowDNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-AoW-D-Play-v0",
    entry_point="isaaclab_nav_tasks.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AowDNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AowDNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-AoW-D-Dev-v0",
    entry_point="isaaclab_nav_tasks.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AowDNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AowDNavMDPORunnerDevCfg,
    },
)

######################################################################################
# PPO

gym.register(
    id="Isaac-Nav-PPO-AoW-D-v0",
    entry_point="isaaclab_nav_tasks.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AowDNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AowDNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-AoW-D-Play-v0",
    entry_point="isaaclab_nav_tasks.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AowDNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AowDNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-AoW-D-Dev-v0",
    entry_point="isaaclab_nav_tasks.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.AowDNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AowDNavPPORunnerDevCfg,
    },
)
