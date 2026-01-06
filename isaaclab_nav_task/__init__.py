# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Isaac Lab Navigation Tasks extension.

This extension provides navigation tasks for robot learning with visual perception.

On import, this module:
1. Applies monkey-patches to Isaac Lab terrain system for height field storage
2. Registers maze terrain types
3. Registers navigation task environments
"""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_NAV_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_NAV_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_NAV_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_NAV_TASKS_METADATA["package"]["version"]

##
# Apply patches to Isaac Lab terrain system.
# This must be done before any terrain generation happens.
# IMPORTANT: Import patches directly from the module file, NOT through
# terrains/__init__.py, to avoid triggering terrain imports before patching.
##

# Direct import from patches module to avoid loading terrains/__init__.py
import importlib.util
import os as _os
_patches_path = _os.path.join(_os.path.dirname(__file__), "terrains", "patches.py")
_spec = importlib.util.spec_from_file_location("patches", _patches_path)
_patches_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_patches_module)
_patches_module.apply_terrain_patches()
del _patches_path, _spec, _patches_module

##
# Export terrain types for convenient access.
##

from .terrains import (
    HfMazeTerrainCfg,
    MAZE_TERRAIN_CFG,
)

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", "terrains"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
