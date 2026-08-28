import copy
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.f1 import F1_CFG, F1_LINKS, F1_ACTION_SCALE
from robolab.sensors import get_link_prim_targets, Grid3dPointsGeneratorCfg
from robolab.tasks.manager_based.parkour.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg

AMP_NUM_STEPS = 3

F1_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
]
F1_LEG_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.05,
    x_max=0.13,
    x_num=19,
    y_min=-0.03,
    y_max=0.03,
    y_num=7,
    z_min=-0.04,
    z_max=-0.02,
    z_num=3,
)
F1_KNEE_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.04,
    x_max=0.04,
    x_num=9,
    y_min=-0.04,
    y_max=0.04,
    y_num=9,
    z_min=-0.36,
    z_max=-0.02,
    z_num=35,
)
F1_ROUGH_TERRAINS_CFG = copy.deepcopy(ROUGH_TERRAINS_CFG)
ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(F1_ROUGH_TERRAINS_CFG)
# 关闭随机墙体
for sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.values():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]

@configclass
class F1ParkourEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = F1_ROUGH_TERRAINS_CFG
        self.scene.robot = F1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = F1_ACTION_SCALE

        self.scene.camera.offset.pos = (0.15, 0.0, 0.25)
        self.scene.camera.offset.rot = (0.9238795, 0.0, 0.3826834, 0.0)
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(F1_LINKS))

        self.scene.leg_volume_points.points_generator = F1_LEG_VOLUME_POINTS_GRID
        self.scene.knee_volume_points.points_generator = F1_KNEE_VOLUME_POINTS_GRID

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "f1_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "36_01": 1,
            "36_11": 1,
            "114_08": 1,
            "114_09": 1,
            "A1-_Stand_stageii": 1,
            "B9_-__Walk_turn_left_90_stageii": 1,
            "B10_-__Walk_turn_left_45_stageii": 1,
            "B13_-__Walk_turn_right_90_stageii": 1,
            "B14_-__Walk_turn_right_45_t2_stageii": 1,
            "B15_-__Walk_turn_around_stageii": 1,
            "turn_l": 1,
            "turn_r": 1,
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc.key_body_pos_b.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=F1_KEY_BODY_NAMES, preserve_order=True
        )

        # Replace RPO-only selectors with F1 joint/body names.
        self.rewards.rewards.rpo_thigh_yaw_joint_sign_penalty = None
        self.rewards.rewards.joint_deviation_upper_body.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
                "waist_.*_joint",
            ],
        )
        self.rewards.rewards.freeze_upper_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_.*_joint"]
        )
        self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.036
        self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["torso_link", "pelvis"]
        )
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["pelvis", "torso_link"]
        )

@configclass
class F1ParkourEnvCfg_PLAY(F1ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None

        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1

        self.scene.leg_volume_points.debug_vis = True
        self.scene.knee_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
