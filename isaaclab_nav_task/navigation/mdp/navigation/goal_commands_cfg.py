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

    track_spl: bool = False
    """If true, measure geodesic path length and expose SPL metrics.
    Note that this is quite expensive"""

    track_sct: bool = False
    """If true, expose approximate SCT metrics using A* geodesic distance / v_ref."""

    sct_reference_speeds: tuple[float, ...] = (0.75, 1.0, 1.5)
    """Reference speeds in m/s for approximate SCT metrics."""

    track_turn_efficiency: bool = False
    """If true, expose cumulative-yaw and turn-efficiency metrics."""

    metric_goal_distance_threshold: float = 0.5
    """Distance threshold in meters used to record first time-to-goal for metrics."""
