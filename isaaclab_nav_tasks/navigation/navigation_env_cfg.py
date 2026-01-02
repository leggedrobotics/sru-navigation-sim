# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation environment configuration.

This module provides the base environment configuration for navigation tasks
with visual perception using depth cameras.
"""

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, RayCasterCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_nav_tasks.navigation.mdp as mdp
from isaaclab_nav_tasks.navigation.mdp.custom_noise import DeltaTransformationNoiseCfg
from isaaclab_nav_tasks.navigation.mdp.delay_manager import ObservationDelayManagerCfg
from isaaclab_nav_tasks.navigation.assets import ISAACLAB_NAV_TASKS_ASSETS_DIR

##
# Pre-defined configs
##
from isaaclab_nav_tasks.terrains import MAZE_TERRAIN_CFG  # isort: skip

# Constants
LEG_JOINT_NAMES = [".*HAA", ".*HFE", ".*KFE"]
LEG_BODY_NAMES = [".*HIP", ".*THIGH", ".*SHANK"]
WHEEL_JOINT_NAMES = [".*WHEEL"]
WHEEL_BODY_NAMES = [".*WHEEL_L"]
PLANNING_FREQ = 5.0  # Hz

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MAZE_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=0.1,
            static_friction=1.0,
            dynamic_friction=0.8,
            compliant_contact_stiffness=5e5,
            compliant_contact_damping=300.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING

    raycast_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.4761, 0.0035, 0.1055), rot=(0.9848078, 0.0, 0.1736482, 0.0), convention="world"  # 20 degrees
        ),
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        max_distance=11.0,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_ros_camera_info(
            # ZED camera parameters from ROS camera_info topic
            fx=72.7025,
            fy=72.7025,
            cx=94.4457,
            cy=62.5424,
            width=192,
            height=120,
            downsample_factor=3,  # Downsample from 192x120 to 64x40
        ),
    )

    height_scanner_critic = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        # attach_yaw_only=True,
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[10.0, 10.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    robot_goal = mdp.RobotNavigationGoalCommandCfg(
        asset_name="robot",
        # Large value to disable automatic resampling - goals only change on episode reset
        # Note: math.inf doesn't work with PyTorch's uniform_(), so we use 1e9 (~31 years)
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_command = mdp.PerceptiveNavigationSE2ActionCfg(
        asset_name="robot",
        low_level_position_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*HAA", ".*HFE", ".*KFE"], scale=0.5, use_default_offset=True
        ),
        low_level_velocity_action=mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=[".*WHEEL"], scale=5.0, use_default_offset=True
        ),
        low_level_decimation=4,
        low_level_policy_file=os.path.join(
            ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "aow_d", "policy_blind_3_1.pt"
        ),
        observation_group="low_level_policy",
        policy_scaling=[1.5, 1.0, 1.0],
        use_raw_actions=True,
        policy_distr_type="gaussian",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel_delayed, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_delayed, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_delayed, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        last_action = ObsTerm(func=mdp.last_action)
        target_position = ObsTerm(
            func=mdp.generated_commands_reshaped_delayed,
            params={"command_name": "robot_goal", "flatten": True},
            noise=DeltaTransformationNoiseCfg(rotation=0.1, translation=0.5, noise_prob=0.1, remove_dist=False),
        )
        depth_image = ObsTerm(
            func=mdp.depth_image_noisy_delayed, params={"sensor_cfg": SceneEntityCfg("raycast_camera")}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        last_action = ObsTerm(func=mdp.last_action)
        target_position = ObsTerm(
            func=mdp.generated_commands_reshaped, params={"command_name": "robot_goal", "flatten": True}
        )
        time_normalized = ObsTerm(func=mdp.time_normalized, params={"command_name": "robot_goal"})
        height_scan_critic = ObsTerm(
            func=mdp.height_scan_feat, params={"sensor_cfg": SceneEntityCfg("height_scanner_critic")}
        )
        depth_image = ObsTerm(func=mdp.depth_image_prefect, params={"sensor_cfg": SceneEntityCfg("raycast_camera")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class LowLevelPolicyCfg(ObsGroup):
        """Observations for low-level policy."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(func=mdp.generated_actions, params={"action_name": "velocity_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_low_level_action, params={"action_term": "velocity_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class MetricsCfg(ObsGroup):
        """Observations for metrics tracking."""

        in_goal = ObsTerm(func=mdp.in_goal)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Observation groups
    metrics: MetricsCfg = MetricsCfg()
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    low_level_policy: LowLevelPolicyCfg = LowLevelPolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # Startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    # Reset events
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )

    randomize_camera_height = EventTerm(
        func=mdp.randomize_camera_height,
        mode="reset",
        params={
            "height_range": 0.1,
            "roll_angle_range": 5.0,
            "pitch_angle_range": 5.0,
            "yaw_angle_range": 5.0,
            "asset_cfg": SceneEntityCfg("raycast_camera"),
        },
    )

    randomize_action_scale = EventTerm(
        func=mdp.randomize_action_scale,
        mode="reset",
        params={
            "scale_range_x": (0.8, 1.2),
            "scale_range_y": (0.6, 1.0),
            "scale_range_theta": (0.8, 1.2),
            "scale_range_xb": 0.1,
            "scale_range_yb": 0.2,
            "scale_range_thetab": 0.1,
            "action_term": "velocity_command",
        },
    )

    reset_delay_buffer = EventTerm(
        func=mdp.reset_and_randomize_delay_buffer,
        mode="reset",
    )

    randomize_low_pass_filter_alpha = EventTerm(
        func=mdp.randomize_low_pass_filter_alpha,
        mode="reset",
        params={
            "alpha_range": (0.4, 0.9),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.4, 0.9),
            "alpha_range_vy": (0.4, 0.9),
            "alpha_range_omega": (0.4, 0.9),
        },
    )

    # Interval events
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.2, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Penalties
    joint_acc_l2_joint = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES)},
    )

    lateral_movement = RewTerm(func=mdp.lateral_movement, weight=-0.1)
    rot_movement = RewTerm(func=mdp.rot_movement, weight=-1e-5)
    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.1)
    episode_termination = RewTerm(func=mdp.is_terminated, weight=-50.0)

    # Goal rewards
    reach_goal_xy_soft = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=0.25,
        params={"command_name": "robot_goal", "sigmoid": 2.5, "T_r": 1.0, "probability": 0.01, "flat": False, "ratio": False},
    )
    reach_goal_xy_tight = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=1.5,
        params={"command_name": "robot_goal", "sigmoid": 0.25, "T_r": 0.1, "probability": 0.01, "flat": True, "ratio": False},
    )

    # Backward movement penalty (disabled by default, can be enabled via curriculum)
    backward_movement_penalty = RewTerm(func=mdp.backward_movement_penalty, weight=-0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out_navigation, time_out=True, params={"distance_threshold": 0.5})
    base_contact = DoneTerm(
        func=mdp.illegal_contact_navigation,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*HIP", ".*THIGH"]), "threshold": 1.0},
    )
    large_pitch_angle = DoneTerm(func=mdp.large_angle_termination_navigation, params={"threshold": 40})
    early_termination = DoneTerm(func=mdp.at_goal_navigation, time_out=True, params={"distance_threshold": 0.5})
    # Terrain fall termination (robot fell off terrain or into deep pit)
    terrain_fall = DoneTerm(
        func=mdp.terrain_fall,
        time_out=True,
        params={"fall_height_threshold": -2.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    disable_backward_penalty = CurrTerm(
        func=mdp.disable_backward_penalty_after_steps,
        params={"disable_after_steps": 500, "action_term": "velocity_command"},
    )

##
# Environment configuration
##


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment with velocity-tracking."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Observation delay configuration
    delay_cfg: ObservationDelayManagerCfg = ObservationDelayManagerCfg()

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings: low_level_policy runs at 50Hz
        self.sim.dt = 0.005
        self.is_finite_horizon = True
        self.low_level_decimation = 4
        self.decimation = int((1 / self.sim.dt) / PLANNING_FREQ)
        self.episode_length_s = 60.0
        self.sim.render_interval = self.low_level_decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # Update sensor periods
        if self.scene.height_scanner_critic is not None:
            self.scene.height_scanner_critic.update_period = self.decimation * self.sim.dt
        if self.scene.raycast_camera is not None:
            self.scene.raycast_camera.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Terrain curriculum settings
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False
