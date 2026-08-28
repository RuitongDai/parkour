import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

F1_ACTION_SCALE = {
    ".*_hip_(pitch|roll|yaw)_joint": 0.125,
    ".*_knee_joint": 0.125,
    ".*_ankle_(pitch|roll)_joint": 0.625,
    "waist_(yaw|roll)_joint": 1.0 / 12.0,
    ".*_shoulder_(pitch|roll|yaw)_joint": 5.0 / 24.0,
    ".*_elbow_joint": 5.0 / 24.0,
    ".*_wrist_roll_joint": 5.0 / 24.0,
    ".*_wrist_(yaw|pitch)_joint": 0.0625,
}

F1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/f1/f1_1.urdf",
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.87),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.2,
            ".*_ankle_pitch_joint": -0.1,

            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,

            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,

            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,
            ".*_wrist_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_(yaw|roll)_joint", ".*_knee_joint"],
            effort_limit_sim=75.0,
            stiffness=150.0,
            damping=2.0,
            armature=0.01,
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit_sim=75.0,
            stiffness=150.0,
            damping=2.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_(pitch|roll)_joint"],
            effort_limit_sim=75.0,
            stiffness=30.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint"],
            effort_limit_sim=50.0,
            stiffness=150.0,
            damping=2.0,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_(pitch|roll|yaw)_joint"],
            effort_limit_sim=25.0,
            stiffness=30.0,
            damping=2.0,
            armature=0.008,
        ),
        "forearms": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=25.0,
            stiffness=30.0,
            damping=2.0,
            armature=0.005,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_(yaw|pitch)_joint"],
            effort_limit_sim=5.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.005,
        ),
    },
)


F1_LINKS = [
    "pelvis",

    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",

    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",

    "waist_yaw_link",
    "torso_link",

    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_hand_link",

    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_hand_link",

    "head_yaw_link",
    "head_pitch_link",
]
