import copy
import os

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass
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
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=X2T2D5_CYLINDER_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
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
class X2T2D5ParkourRoughEnvCfg(ParkourEnvCfg):   
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
