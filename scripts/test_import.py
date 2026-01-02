#!/usr/bin/env python3
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quick test script to verify the isaaclab_nav_tasks extension imports correctly."""

import argparse
from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Test isaaclab_nav_tasks extension import")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Test imports after Isaac Sim is initialized
print("\n" + "="*60)
print("Testing isaaclab_nav_tasks extension imports...")
print("="*60 + "\n")

try:
    # Test main module import
    import isaaclab_nav_tasks
    print("[OK] isaaclab_nav_tasks imported successfully")

    # Test navigation module
    from isaaclab_nav_tasks import navigation
    print("[OK] isaaclab_nav_tasks.navigation imported successfully")

    # Test MDP module
    from isaaclab_nav_tasks.navigation import mdp
    print("[OK] isaaclab_nav_tasks.navigation.mdp imported successfully")

    # Test specific components
    from isaaclab_nav_tasks.navigation.mdp import rewards
    print("[OK] rewards module imported successfully")

    from isaaclab_nav_tasks.navigation.mdp import observations
    print("[OK] observations module imported successfully")

    from isaaclab_nav_tasks.navigation.mdp import curriculums
    print("[OK] curriculums module imported successfully")

    from isaaclab_nav_tasks.navigation.mdp import terminations
    print("[OK] terminations module imported successfully")

    from isaaclab_nav_tasks.navigation.mdp.navigation import goal_commands
    print("[OK] goal_commands module imported successfully")

    from isaaclab_nav_tasks.navigation.mdp.navigation.actions import navigation_se2_actions
    print("[OK] navigation_se2_actions module imported successfully")

    # Test RSL-RL configs
    from isaaclab_nav_tasks.navigation.config.b2w.agents import rsl_rl_cfg as b2w_cfg
    print("[OK] B2W RSL-RL config imported successfully")

    from isaaclab_nav_tasks.navigation.config.aow_d.agents import rsl_rl_cfg as aow_cfg
    print("[OK] AoW-D RSL-RL config imported successfully")

    # Check if tasks are registered
    import gymnasium as gym

    print("\n" + "-"*60)
    print("Registered navigation tasks:")
    print("-"*60)

    nav_tasks = [env_id for env_id in gym.envs.registry.keys() if "Navigation" in env_id]
    if nav_tasks:
        for task in sorted(nav_tasks):
            print(f"  - {task}")
    else:
        print("  (No navigation tasks found - tasks may need to be registered)")

    print("\n" + "="*60)
    print("All imports successful!")
    print("="*60 + "\n")

except Exception as e:
    print(f"\n[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()

# Close the simulation
simulation_app.close()
