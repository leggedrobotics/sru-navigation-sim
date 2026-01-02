# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the B2W (Unitree B2W) robot.

The following configuration parameters are available:

* :obj:`B2W_CFG`: The B2W robot with wheels.

Reference:
    The B2W is a bipedal wheeled robot.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Local assets directory for this extension
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

__all__ = ["B2W_CFG"]


B2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSETS_DIR}/Robots/B2W/b2w_rsl.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=None,
            max_angular_velocity=None,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*hip_joint": 0.0,
            ".*thigh_joint": 0.4,
            ".*foot_joint": 0.0,
            ".*calf_joint": -1.3,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_joint", ".*thigh_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=23.0,
            stiffness={".*": 100.0},
            damping={".*": 3.5},
        ),
        "legs_calf": ImplicitActuatorCfg(
            joint_names_expr=[".*calf_joint"],
            effort_limit_sim=320.0,
            velocity_limit_sim=14.0,
            stiffness={".*": 100.0},
            damping={".*": 3.5},
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*foot_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=50.0,
            stiffness={".*": 0.0},
            damping={".*": 3.0},
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of B2W robot using ImplicitActuatorCfg."""
