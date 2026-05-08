import copy
import os
import math

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.assets.x2t2d5_accvel_accarma import (
    X2T2D5_LINKS,
    X2T2D5_CYLINDER_CFG,
    X2T2D5_ACTION_SCALE,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets, NoisyGroupedRayCasterCameraCfg
from instinctlab.tasks.parkour.config.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg
from instinctlab.managers import MultiRewardCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    path = os.path.expanduser("/home/agiuser/projects/Instinct/InstinctLab/data/hiking-in-the-wild_Data&Model/data&model/parkour_motion_reference")
    retargetting_func = None
    filtered_motion_selection_filepath = os.path.expanduser("/home/agiuser/projects/Instinct/InstinctLab/data/hiking-in-the-wild_Data&Model/data&model/parkour_motion_reference/parkour_motion_without_run.yaml")
    motion_start_from_middle_range = [0.0, 0.9]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    robot_model_path=X2T2D5_CYLINDER_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/base_link",
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={
        "run_walk": AmassMotionCfg(),
    },
    link_of_interests=[
        "base_link",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    mp_split_method="Even",
)


ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class X2T2D5Rewards:
    """Reward terms for the MDP."""

    # Task rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0, params={"command_name": "base_velocity"})
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.3, params={"command_name": "base_velocity", "offset": 4.0})

    # Regularization rewards
    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "vel_threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=mdp.contact_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 1.0,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_square,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    pelvis_orientation_l2 = RewTerm(
        func=mdp.link_orientation, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")}
    )
    feet_flat_ori = RewTerm(
        func=mdp.feet_orientation_contact,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_at_plane = RewTerm(
        func=mdp.feet_at_plane,
        weight=-0.1,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "left_height_scanner_cfg": SceneEntityCfg("left_height_scanner"),
            "right_height_scanner_cfg": SceneEntityCfg("right_height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "height_offset": 0.035,
        },
    )
    feet_close_xy = RewTerm(
        func=mdp.feet_close_xy_gauss,
        weight=0.4,
        params={
            "threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "std": math.sqrt(0.05),
        },
    )
    energy = RewTerm(
        func=mdp.motors_power_square,
        weight=-5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            "normalize_by_stiffness": True,
        },
    )
    freeze_upper_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.004,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*"]
            ),
        },
    )

    # Safety rewards
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_ankle_roll_link).*"),
            "threshold": 1.0,
        },
    )


@configclass
class RewardsCfg(MultiRewardCfg):
    rewards: X2T2D5Rewards = X2T2D5Rewards()


@configclass
class X2T2D5ParkourRoughEnvCfg(ParkourEnvCfg):
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = X2T2D5_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # camera
        self.scene.camera.max_distance = 6.0
        self.scene.camera.update_period = 1 / 30
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(X2T2D5_LINKS))
        self.scene.camera.pattern_cfg = PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=1.8556,  # fovx
            vertical_aperture=1.0441,  # fovy
            height=int(270 / 10),
            width=int(480 / 10),
        )
        self.scene.camera.offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(
                0.0576096 + 0.0083477, # add the offset between torso_link and head_pitch_link
                -0.0111832,
                -0.0483697 + 0.3982971, # add the offset between torso_link and head_pitch_link
            ),
            rot=(0.640857, -0.298835, -0.298835, 0.640857),
            convention="opengl",
        )
        
        self.scene.motion_reference = motion_reference_cfg
        
        self.actions.joint_pos.scale = X2T2D5_ACTION_SCALE
        
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**16


@configclass
class X2T2D5ParkourRoughEnvCfg_PLAY(X2T2D5ParkourRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.viewer = ViewerCfg(
            eye=[4.0, 0.75, 1.0],
            lookat=[0.0, 0.75, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )

        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 10

        self.scene.leg_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
