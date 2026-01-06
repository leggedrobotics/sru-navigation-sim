# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom robot configurations and assets for navigation tasks."""

import os

# Path to the local data directory containing robots and policies
ISAACLAB_NAV_TASKS_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
"""Path to the navigation tasks assets data directory."""

from .b2w import *
from .aow_d import *

__all__ = ["ISAACLAB_NAV_TASKS_ASSETS_DIR", "B2W_CFG", "ANYMAL_D_ON_WHEELS_CFG"]
