# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation task environments for Isaac Lab."""

from .navigation_env import NavigationEnv
from .navigation_env_cfg import *

# Import robot-specific configurations
from .config import *
