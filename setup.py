# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Installation script for the 'isaaclab_nav_task' python package.

IsaacLab task extension for SRU (Spatially-enhanced Recurrent Unit) visual navigation project.
Provides hierarchical control architecture, maze terrain generation with curriculum learning,
and depth-based reinforcement learning for legged robot navigation.
"""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy<2",
    "torch>=2.5.1",
    "torchvision>=0.14.1",
    # io
    "h5py",
    # visualization
    "tensorboard",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

# Installation operation
setup(
    name="isaaclab_nav_task",
    author="Fan Yang, Per Frivik",
    author_email="fanyang1@ethz.ch, pfrivik@ethz.ch",
    maintainer="Fan Yang, Per Frivik",
    maintainer_email="fanyang1@ethz.ch, pfrivik@ethz.ch",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["isaaclab_nav_task"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
        "Isaac Lab :: 2.1.1",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    zip_safe=False,
    license="MIT",
)
