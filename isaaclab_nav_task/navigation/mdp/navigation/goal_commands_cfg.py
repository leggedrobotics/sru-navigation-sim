# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT


from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg
from .goal_commands import RobotNavigationGoalCommand
from .path_generator import GlobalPathConfig
from .prm import PrmConfig


"""
Base command generator.
"""

@configclass
class RobotNavigationGoalCommandCfg(CommandTermCfg):
    """Configuration for the robot goal command generator."""

    class_type: type = RobotNavigationGoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    robot_to_goal_line_vis: bool = True
    """If true, visualize the line from the robot to the goal."""

    path_cfg: GlobalPathConfig = GlobalPathConfig()
    """Configuration for the global path (PRM + A*) generated from spawn to goal."""

    prm_cfg: PrmConfig = PrmConfig()
    """Configuration for the per-terrain-tile probabilistic roadmap the path is planned over."""

