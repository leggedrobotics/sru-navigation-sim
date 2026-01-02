# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots with wheels.

The following configuration parameters are available:

* :obj:`ANYMAL_D_ON_WHEELS_CFG`: The ANYmal-D on wheels with ImplicitActuatorCfg.

"""

from . import ISAACLAB_NAV_TASKS_ASSETS_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

__all__ = ["ANYMAL_D_ON_WHEELS_CFG"]


ANYMAL_D_ON_WHEELS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NAV_TASKS_ASSETS_DIR}/Robots/AoW-D/aow_d.usd",
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
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
            ".*WHEEL": 0.0,  # all WHEEL
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            effort_limit_sim=80.0,  # From URDF: HAA, HFE, KFE joints all have 80.0 Nm effort limit
            velocity_limit_sim=8.5,  # From URDF: HAA, HFE, KFE joints all have 8.5 rad/s velocity limit
            stiffness={".*": 100.0},  # Typical stiffness value for legged robots
            damping={".*": 3.5},      # Typical damping value for legged robots
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*WHEEL"],
            effort_limit_sim=28.0,  # From URDF: wheel joints have 28.0 Nm effort limit
            velocity_limit_sim=200.0,  # From URDF: wheel joints have 200.0 rad/s velocity limit
            stiffness={".*": 0.0},  # Wheels typically have no stiffness (direct velocity control)
            damping={".*": 5.0},    # From original configuration
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of wheelified ANYmal-D using ImplicitActuatorCfg."""
