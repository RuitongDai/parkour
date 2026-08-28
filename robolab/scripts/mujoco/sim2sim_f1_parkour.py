# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo sim2sim deployment for the 28-DoF ``f1-Parkour`` policy.

The depth preprocessing, history buffers, keyboard commands, visualization, and
separate depth-encoder/actor ONNX inference are shared with the proven RPO
Parkour deployment path.  This entry point supplies only F1-specific model,
observation, joint-order, action-scale, PD, and torque-limit configuration.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOLAB_REPO_DIR = SCRIPT_DIR.parents[1]
WORKSPACE_DIR = ROBOLAB_REPO_DIR.parent
F1_MJCF_DIR = ROBOLAB_REPO_DIR / "data" / "robots" / "f1"

NUM_ACTIONS = 28
FRAME_STACK = 8
PROPRIO_DIM = FRAME_STACK * (3 + 3 + 3 + NUM_ACTIONS + NUM_ACTIONS + NUM_ACTIONS)

# MuJoCo qpos/qvel/actuator order in f1_1.xml.
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
)

# Resolved Isaac Lab action order for f1-Parkour.  This is intentionally
# explicit: Isaac applies regex actuator/action selectors in this order, while
# the native MJCF groups complete left/right kinematic chains.
ISAAC_ACTION_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
)

ISAAC_TO_MUJOCO = tuple(MUJOCO_JOINT_NAMES.index(name) for name in ISAAC_ACTION_JOINT_NAMES)


def _joint_group_values() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    default_by_name = {name: 0.0 for name in MUJOCO_JOINT_NAMES}
    for side in ("left", "right"):
        default_by_name[f"{side}_hip_pitch_joint"] = -0.1
        default_by_name[f"{side}_knee_joint"] = 0.2
        default_by_name[f"{side}_ankle_pitch_joint"] = -0.1
        default_by_name[f"{side}_elbow_joint"] = 1.4

    kp_by_name: dict[str, float] = {}
    kd_by_name: dict[str, float] = {}
    torque_by_name: dict[str, float] = {}
    scale_by_name: dict[str, float] = {}
    for name in MUJOCO_JOINT_NAMES:
        if any(part in name for part in ("hip_", "knee")):
            kp_by_name[name], kd_by_name[name], torque_by_name[name], scale_by_name[name] = 150.0, 2.0, 75.0, 0.125
        elif "ankle_" in name:
            kp_by_name[name], kd_by_name[name], torque_by_name[name], scale_by_name[name] = 30.0, 2.0, 75.0, 0.625
        elif name.startswith("waist_"):
            kp_by_name[name], kd_by_name[name], torque_by_name[name], scale_by_name[name] = (
                150.0,
                2.0,
                50.0,
                1.0 / 12.0,
            )
        elif any(part in name for part in ("shoulder_", "elbow", "wrist_roll")):
            kp_by_name[name], kd_by_name[name], torque_by_name[name], scale_by_name[name] = 30.0, 2.0, 25.0, 5.0 / 24.0
        elif any(part in name for part in ("wrist_yaw", "wrist_pitch")):
            kp_by_name[name], kd_by_name[name], torque_by_name[name], scale_by_name[name] = 20.0, 1.0, 5.0, 0.0625
        else:  # pragma: no cover - all F1 joints must be classified above.
            raise ValueError(f"Unclassified F1 joint: {name}")

    return (
        np.array([default_by_name[name] for name in MUJOCO_JOINT_NAMES], dtype=np.float64),
        np.array([kp_by_name[name] for name in MUJOCO_JOINT_NAMES], dtype=np.float64),
        np.array([kd_by_name[name] for name in MUJOCO_JOINT_NAMES], dtype=np.float64),
        np.array([torque_by_name[name] for name in MUJOCO_JOINT_NAMES], dtype=np.float64),
        np.array([scale_by_name[name] for name in ISAAC_ACTION_JOINT_NAMES], dtype=np.float64),
    )


DEFAULT_POS, KPS, KDS, TORQUE_LIMITS, ACTION_SCALE = _joint_group_values()
EXPECTED_ARMATURE = np.array(
    [0.01] * 14 + [0.008] * 3 + [0.005] * 4 + [0.008] * 3 + [0.005] * 4,
    dtype=np.float64,
)
EXPECTED_TOTAL_MASS = 30.071
EXPECTED_WRIST_RANGES = {
    "left_wrist_roll_joint": (-3.7524, 1.5708),
    "left_wrist_yaw_joint": (-0.6109, 1.1344),
    "right_wrist_roll_joint": (-1.5708, 3.7524),
    "right_wrist_yaw_joint": (-1.1344, 0.6109),
}


def get_f1_obs(data: mujoco.MjData, model: mujoco.MjModel):
    """Return the F1 articulation observation in the same frames as Isaac Lab."""
    del model
    q = data.qpos.astype(np.float64)
    dq = data.qvel.astype(np.float64)
    quat_wxyz = q[3:7]
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]].copy()
    rotation = R.from_quat(quat_xyzw)
    base_lin_vel_b = rotation.apply(data.qvel[:3], inverse=True).astype(np.float64)
    base_ang_vel_b = data.sensor("imu_ang_vel").data.astype(np.float64).copy()
    projected_gravity_b = rotation.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float64)
    return q, dq, quat_xyzw, base_lin_vel_b, base_ang_vel_b, projected_gravity_b


def _scene_path(scene: str, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return F1_MJCF_DIR / {
        "stairs": "f1_stairs.xml",
        "stairs30": "f1_stairs.xml",
        "stairs40": "f1_stairs_40.xml",
        "stairs50": "f1_stairs_50.xml",
        "plane": "f1_plane.xml",
    }[scene]


def _latest_f1_export_dir() -> Path | None:
    candidates: list[Path] = []
    for root in (WORKSPACE_DIR / "logs", ROBOLAB_REPO_DIR / "logs"):
        candidates.extend(root.glob("rsl_rl/f1_parkour/*/exported"))
    complete = [
        path
        for path in candidates
        if (path / "0-depth_encoder.onnx").is_file() and (path / "actor.onnx").is_file()
    ]
    return max(complete, key=lambda path: path.parent.name) if complete else None


def _resolve_onnx_paths(depth_encoder: str | None, actor: str | None) -> tuple[Path | None, Path | None]:
    latest = _latest_f1_export_dir()
    encoder_path = Path(depth_encoder).expanduser().resolve() if depth_encoder else None
    actor_path = Path(actor).expanduser().resolve() if actor else None
    if latest is not None:
        encoder_path = encoder_path or latest / "0-depth_encoder.onnx"
        actor_path = actor_path or latest / "actor.onnx"
    return encoder_path, actor_path


def _select_onnx_providers(mode: str) -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on deployment environment.
        raise RuntimeError("onnxruntime or onnxruntime-gpu is required for sim2sim inference") from exc

    available = ort.get_available_providers()
    cuda_load_error: OSError | None = None
    if "CUDAExecutionProvider" in available:
        capi_dir = Path(ort.__file__).resolve().parent / "capi"
        cuda_libraries = list(capi_dir.glob("libonnxruntime_providers_cuda.so*"))
        cuda_libraries.extend(capi_dir.glob("onnxruntime_providers_cuda.dll"))
        if cuda_libraries:
            try:
                ctypes.CDLL(str(cuda_libraries[0]), mode=ctypes.RTLD_LOCAL)
            except OSError as exc:
                cuda_load_error = exc
    if mode == "auto":
        if "CUDAExecutionProvider" in available and cuda_load_error is None:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if cuda_load_error is not None:
            print(f"[WARN] CUDAExecutionProvider is installed but cannot load ({cuda_load_error}); using CPU.")
        return ["CPUExecutionProvider"]
    requested = "CUDAExecutionProvider" if mode == "cuda" else "CPUExecutionProvider"
    if requested not in available:
        raise RuntimeError(f"Requested {requested}, but available ONNX Runtime providers are: {available}")
    if requested == "CUDAExecutionProvider" and cuda_load_error is not None:
        raise RuntimeError(f"CUDAExecutionProvider is installed but its shared library cannot load: {cuda_load_error}")
    return [requested]


def _model_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(model.joint(index).name for index in range(1, model.njnt))


def _actuator_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(model.joint(int(model.actuator_trnid[index, 0])).name for index in range(model.nu))


def _body_subtree_mass(model: mujoco.MjModel, root_body_name: str) -> float:
    """Return body mass below one root without counting static scene bodies."""
    root_id = model.body(root_body_name).id
    body_ids = {root_id}
    for body_id in range(root_id + 1, model.nbody):
        if int(model.body_parentid[body_id]) in body_ids:
            body_ids.add(body_id)
    return float(np.sum(model.body_mass[list(body_ids)]))


def validate_mujoco_model(xml_path: Path, *, smoke_steps: int = 250) -> None:
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML does not exist: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if (model.nq, model.nv, model.nu) != (35, 34, NUM_ACTIONS):
        raise ValueError(f"F1 model dimensions must be nq=35, nv=34, nu=28; got {model.nq}, {model.nv}, {model.nu}")
    if _model_joint_names(model) != MUJOCO_JOINT_NAMES:
        raise ValueError(
            f"MuJoCo joint order mismatch:\nactual={_model_joint_names(model)}\nexpected={MUJOCO_JOINT_NAMES}"
        )
    if _actuator_joint_names(model) != MUJOCO_JOINT_NAMES:
        raise ValueError("F1 actuator order does not match the MuJoCo joint order")
    if not np.all(model.actuator_ctrllimited):
        raise ValueError("Every F1 torque actuator must have a control limit")
    if not np.allclose(model.actuator_ctrlrange[:, 1], TORQUE_LIMITS):
        raise ValueError(f"F1 positive actuator limits do not match training: {model.actuator_ctrlrange[:, 1]}")
    if not np.allclose(model.actuator_ctrlrange[:, 0], -TORQUE_LIMITS):
        raise ValueError(f"F1 negative actuator limits do not match training: {model.actuator_ctrlrange[:, 0]}")
    if set(ISAAC_TO_MUJOCO) != set(range(NUM_ACTIONS)):
        raise ValueError(f"Invalid Isaac-to-MuJoCo joint permutation: {ISAAC_TO_MUJOCO}")
    if model.body("pelvis").id < 0 or model.body("torso_link").id < 0:
        raise ValueError("F1 model must contain pelvis and torso_link")
    if model.sensor("imu_ang_vel").id < 0:
        raise ValueError("F1 model must contain the imu_ang_vel gyro")
    if not np.allclose(model.dof_armature[6:], EXPECTED_ARMATURE):
        raise ValueError(f"F1 armature mismatch: {model.dof_armature[6:]}")
    if not np.isclose(model.body("torso_link").mass[0], 7.609):
        raise ValueError(f"F1 torso mass must match f1_1.urdf (7.609 kg), got {model.body('torso_link').mass[0]}")
    robot_mass = _body_subtree_mass(model, "pelvis")
    if not np.isclose(robot_mass, EXPECTED_TOTAL_MASS):
        raise ValueError(
            f"F1 total mass must match f1_1.urdf ({EXPECTED_TOTAL_MASS} kg), "
            f"got {robot_mass}"
        )
    for joint_name, expected_range in EXPECTED_WRIST_RANGES.items():
        if not np.allclose(model.joint(joint_name).range, expected_range):
            raise ValueError(
                f"F1 {joint_name} range does not match training: "
                f"{model.joint(joint_name).range} != {expected_range}"
            )
    geom_names = {model.geom(index).name for index in range(model.ngeom)}
    required_geoms = {
        "pelvis_collision",
        "left_thigh_collision",
        "right_thigh_collision",
        "left_shin_collision",
        "right_shin_collision",
        "left_foot_collision",
        "right_foot_collision",
        "torso_collision",
    }
    if missing_geoms := required_geoms - geom_names:
        raise ValueError(f"F1 training-matched collision geoms are missing: {sorted(missing_geoms)}")
    if model.geom("left_foot_collision").type[0] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("F1 left foot collision must use the ankle-roll mesh")
    if model.geom("right_foot_collision").type[0] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("F1 right foot collision must use the ankle-roll mesh")

    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        data.qpos[2] = 0.9
        data.qpos[3] = 1.0
        data.qpos[-NUM_ACTIONS:] = DEFAULT_POS
    mujoco.mj_forward(model, data)
    get_f1_obs(data, model)
    for _ in range(max(0, smoke_steps)):
        q = data.qpos[-NUM_ACTIONS:]
        dq = data.qvel[-NUM_ACTIONS:]
        data.ctrl[:] = np.clip((DEFAULT_POS - q) * KPS - dq * KDS, -TORQUE_LIMITS, TORQUE_LIMITS)
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise RuntimeError("F1 MuJoCo PD smoke test produced non-finite state")
    print(
        f"[OK] F1 MJCF: {xml_path} | nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"joint mapping=28/28, smoke pelvis_z={data.qpos[2]:.3f} m"
    )


def validate_onnx_sessions(depth_encoder: Any, actor: Any) -> None:
    encoder_input = np.zeros((1, FRAME_STACK, 18, 32), dtype=np.float32)
    latent = depth_encoder.run(None, {depth_encoder.get_inputs()[0].name: encoder_input})[0]
    if latent.ndim != 2 or latent.shape[0] != 1:
        raise ValueError(f"Unexpected depth encoder output shape: {latent.shape}")
    actor_input = np.zeros((1, PROPRIO_DIM + latent.shape[-1]), dtype=np.float32)
    actions = actor.run(None, {actor.get_inputs()[0].name: actor_input})[0]
    if actions.shape != (1, NUM_ACTIONS):
        raise ValueError(
            f"F1 actor output must be (1, {NUM_ACTIONS}), got {actions.shape}; "
            "the ONNX may have been exported from an RPO checkpoint"
        )
    print(
        f"[OK] F1 ONNX: depth {encoder_input.shape} -> latent {latent.shape}; "
        f"actor {actor_input.shape} -> actions {actions.shape}"
    )


def _load_shared_parkour_module(*, headless: bool):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    # pynput opens X during import.  The dummy backend keeps headless inference usable.
    if headless:
        os.environ["PYNPUT_BACKEND"] = "dummy"
    module = importlib.import_module("sim2sim_rpo_parkour")
    if headless:
        module.start_keyboard_listener = lambda: SimpleNamespace(stop=lambda: None)
        module.print_controls_guide = lambda: None
    return module


def _install_f1_hooks(shared: Any, *, root_height: float) -> None:
    shared.get_obs = get_f1_obs
    shared._CHASE_BODY_NAME = "pelvis"
    # Match F1ParkourEnvCfg. Training randomizes this nominal pose, while
    # deployment intentionally uses the deterministic nominal transform.
    shared._CAMERA_OFFSET_POS_BODY = np.array([0.15, 0.0, 0.25], dtype=np.float64)
    shared._CAMERA_OFFSET_QUAT_WXYZ = np.array(
        [0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float64
    )

    base_command = shared.cmd

    class LatchedF1Command(base_command):
        """Increment a retained velocity target by 0.1 for every key press."""

        velocity_step = 0.1
        _target_vx = 0.0
        _target_vy = 0.0
        _target_dyaw = 0.0

        @classmethod
        def latch_key(cls, key: str) -> None:
            if key == "8":
                cls._target_vx += cls.velocity_step
            elif key == "2":
                cls._target_vx -= cls.velocity_step
            elif key == "4":
                cls._target_vy += cls.velocity_step
            elif key == "6":
                cls._target_vy -= cls.velocity_step
            elif key == "7":
                cls._target_dyaw += cls.velocity_step
            elif key == "9":
                cls._target_dyaw -= cls.velocity_step
            cls._target_vx = round(float(np.clip(cls._target_vx, -cls.hold_vx, cls.hold_vx)), 10)
            cls._target_vy = round(float(np.clip(cls._target_vy, -cls.hold_vy, cls.hold_vy)), 10)
            cls._target_dyaw = round(
                float(np.clip(cls._target_dyaw, -cls.hold_dyaw, cls.hold_dyaw)),
                10,
            )

        @classmethod
        def stop_velocity(cls) -> None:
            cls._target_vx = 0.0
            cls._target_vy = 0.0
            cls._target_dyaw = 0.0

        @classmethod
        def _keyboard_target(cls) -> tuple[float, float, float]:
            return cls._target_vx, cls._target_vy, cls._target_dyaw

        @classmethod
        def reset(cls) -> None:
            super().reset()
            cls.stop_velocity()

    pressed_motion_keys: set[str] = set()

    def on_f1_key_press(key: Any) -> None:
        try:
            if not hasattr(key, "char") or key.char is None:
                return
            char = key.char.lower()
            if char in LatchedF1Command._MOVE_KEYS and char not in pressed_motion_keys:
                pressed_motion_keys.add(char)
                LatchedF1Command.latch_key(char)
            elif char == "5":
                LatchedF1Command.stop_velocity()
            elif char == "f":
                LatchedF1Command.toggle_camera_follow()
            elif char == "r":
                LatchedF1Command.toggle_chase_camera()
            elif char == "0":
                pressed_motion_keys.clear()
                LatchedF1Command.reset_requested = True
        except AttributeError:
            pass

    def on_f1_key_release(key: Any) -> None:
        try:
            if hasattr(key, "char") and key.char is not None:
                pressed_motion_keys.discard(key.char.lower())
        except AttributeError:
            pass

    def print_f1_controls() -> None:
        print(
            "F1 latched velocity controls: 8/2=vx +/-, 4/6=vy +/-, "
            "7/9=wz +/-, 5=stop, 0=reset, R=chase camera, F=follow camera."
        )
        print(
            f"Each key press changes its retained target by {LatchedF1Command.velocity_step}; "
            f"limits: vx=+/-{LatchedF1Command.hold_vx} m/s, "
            f"vy=+/-{LatchedF1Command.hold_vy} m/s, "
            f"wz=+/-{LatchedF1Command.hold_dyaw} rad/s."
        )

    shared.cmd = LatchedF1Command
    shared.on_press = on_f1_key_press
    shared.on_release = on_f1_key_release
    shared.print_controls_guide = print_f1_controls

    # Isaac's camera casts against /visuals and /World/ground, and ignores
    # intersections closer than 0.1 m.  F1 collision geoms are group 3 and
    # visuals are group 2, so exclude collision proxies and advance each ray
    # by the same 0.1 m before asking MuJoCo for its first intersection.
    def capture_f1_depth_mj_ray(
        data: mujoco.MjData,
        model: mujoco.MjModel,
        torso_body_name: str,
        vertical_fov_deg: float,
        horizontal_fov_deg: float,
    ) -> np.ndarray:
        pose = shared.get_depth_camera_pose(data, model, torso_body_name)
        eye = pose["eye"]
        rotation_world = pose["r_world"]
        dirs_cam = shared.pinhole_ray_dirs_cam(
            shared._RAW_DEPTH_H,
            shared._RAW_DEPTH_W,
            vertical_fov_deg,
            horizontal_fov_deg,
        )
        dirs_world = dirs_cam @ rotation_world.T
        min_ray_distance = 0.1
        depth = np.full(
            (shared._RAW_DEPTH_H, shared._RAW_DEPTH_W),
            shared._DEPTH_RAY_MAX,
            dtype=np.float64,
        )
        geom_id = np.zeros(1, dtype=np.int32)
        geom_groups = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        for row in range(shared._RAW_DEPTH_H):
            for col in range(shared._RAW_DEPTH_W):
                direction_world = dirs_world[row, col]
                ray_start = eye + min_ray_distance * direction_world
                distance = mujoco.mj_ray(
                    model,
                    data,
                    ray_start,
                    direction_world,
                    geom_groups,
                    1,
                    -1,
                    geom_id,
                )
                if distance is not None and distance >= 0.0:
                    radial_distance = min_ray_distance + float(distance)
                    depth[row, col] = min(
                        radial_distance * float(dirs_cam[row, col, 0]),
                        shared._DEPTH_RAY_MAX,
                    )
        return depth

    original_depth_converter = shared.raw_depth_to_metric_grid

    def f1_depth_to_metric_grid(depth_hw: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        # mj_ray already returns metric distance_to_image_plane.  Running it
        # through OpenGL depth-buffer linearization would corrupt all values <1 m.
        if shared._DEPTH_BACKEND == "ray":
            depth = np.asarray(depth_hw, dtype=np.float64)
            if depth.shape != (shared._RAW_DEPTH_H, shared._RAW_DEPTH_W):
                depth = shared.cv2.resize(
                    depth,
                    (shared._RAW_DEPTH_W, shared._RAW_DEPTH_H),
                    interpolation=shared.cv2.INTER_AREA,
                )
            return np.abs(depth)
        return original_depth_converter(depth_hw, model)

    shared.capture_depth_mj_ray = capture_f1_depth_mj_ray
    shared.raw_depth_to_metric_grid = f1_depth_to_metric_grid

    original_macro_step = shared.mj_macro_step
    first_step = True

    def initialized_macro_step(model: mujoco.MjModel, data: mujoco.MjData, *, n_sub: int) -> None:
        nonlocal first_step
        if first_step:
            data.qpos[2] = root_height
            data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
            first_step = False
        original_macro_step(model, data, n_sub=n_sub)

    shared.mj_macro_step = initialized_macro_step


def _build_cfg(xml_path: Path, *, duration: float) -> SimpleNamespace:
    return SimpleNamespace(
        sim_config=SimpleNamespace(
            mujoco_model_path=str(xml_path),
            sim_duration=duration,
            dt=0.005,
            decimation=4,
            depth_camera_body="torso_link",
        ),
        robot_config=SimpleNamespace(
            kps=KPS,
            kds=KDS,
            default_pos=DEFAULT_POS,
            tau_limit=TORQUE_LIMITS,
            frame_stack=FRAME_STACK,
            num_actions=NUM_ACTIONS,
            action_scale=ACTION_SCALE,
            usd2urdf=list(ISAAC_TO_MUJOCO),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 Parkour sim2sim (depth encoder + actor ONNX).")
    parser.add_argument(
        "--depth-encoder",
        "--depth_encoder",
        dest="depth_encoder",
        default=None,
        help="F1 0-depth_encoder.onnx path. Default: newest complete f1_parkour export.",
    )
    parser.add_argument(
        "--actor",
        default=None,
        help="F1 actor.onnx path. Default: newest complete f1_parkour export.",
    )
    parser.add_argument(
        "--scene",
        choices=("stairs", "stairs30", "stairs40", "stairs50", "plane"),
        default="stairs",
        help="Bundled F1 scene; stairs is the backward-compatible 0.30 m tread alias.",
    )
    parser.add_argument(
        "--mujoco-xml",
        "--mujoco_xml",
        dest="mujoco_xml",
        default=None,
        help="Override the bundled F1 scene with another MJCF path.",
    )
    parser.add_argument(
        "--onnx-provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="ONNX Runtime provider; auto prefers CUDA and falls back to CPU.",
    )
    parser.add_argument("--root-height", type=float, default=0.9, help="Initial pelvis height in meters.")
    parser.add_argument("--duration", type=float, default=1_000_000.0, help="Maximum simulated duration in seconds.")
    parser.add_argument("--headless", action="store_true", help="Disable GUI and record simulation_parkour.mp4.")
    parser.add_argument(
        "--no-depth-vis",
        "--no_depth_vis",
        dest="no_depth_vis",
        action="store_true",
        help="Disable the OpenCV depth preview.",
    )
    parser.add_argument("--no-realtime", action="store_true", help="Run without wall-clock synchronization.")
    parser.add_argument("--debug-obs", action="store_true", help="Print policy observation tensor dimensions.")
    parser.add_argument("--full-ui", action="store_true", help="Show the full MuJoCo viewer side panels.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Compile/check MJCF and any available ONNX files, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = _scene_path(args.scene, args.mujoco_xml)
    encoder_path, actor_path = _resolve_onnx_paths(args.depth_encoder, args.actor)

    validate_mujoco_model(xml_path)
    if (encoder_path is None) != (actor_path is None):
        raise ValueError("Pass both --depth-encoder and --actor, or neither")
    if encoder_path is None or actor_path is None:
        if args.validate_only:
            print("[INFO] No complete f1_parkour ONNX export found; MJCF-only validation completed.")
            return
        raise FileNotFoundError(
            "No complete F1 Parkour ONNX export was found. Export a trained checkpoint with:\n"
            "  python robolab/scripts/rsl_rl/play_parkour.py --task f1-Parkour-Play "
            "--num_envs 1 --checkpoint /absolute/path/to/model_<N>.pt --exportonnx --headless\n"
            "Then rerun this script, or pass --depth-encoder and --actor explicitly."
        )
    if not encoder_path.is_file() or not actor_path.is_file():
        raise FileNotFoundError(f"Missing F1 ONNX file(s): {encoder_path}, {actor_path}")

    providers = _select_onnx_providers(args.onnx_provider)
    print(f"[INFO] Requested ONNX providers: {providers}")
    shared = _load_shared_parkour_module(headless=args.headless or args.validate_only)
    depth_encoder, actor = shared.build_onnx_sessions(str(encoder_path), str(actor_path), providers=providers)
    print(f"[INFO] Active ONNX providers: {depth_encoder.get_providers()}")
    validate_onnx_sessions(depth_encoder, actor)
    if args.validate_only:
        return

    _install_f1_hooks(shared, root_height=args.root_height)
    cfg = _build_cfg(xml_path, duration=args.duration)
    shared.run_mujoco_onnx(
        depth_encoder,
        actor,
        cfg,
        headless=args.headless,
        debug_obs=args.debug_obs,
        show_depth_vis=not args.no_depth_vis,
        depth_vis_scale=max(1, shared._SIM2SIM_PERF_DEPTH_VIS_SCALE),
        realtime_sync=not args.no_realtime,
        quiet=shared._SIM2SIM_PERF_QUIET,
        depth_vis_every_step=shared._SIM2SIM_PERF_DEPTH_VIS_EVERY_STEP,
        depth_vis_policy_stride=max(1, shared._SIM2SIM_PERF_DEPTH_VIS_POLICY_STRIDE),
        viewer_sync_every=shared._SIM2SIM_PERF_VIEWER_SYNC_EVERY,
        viewer_fallback_width=max(320, shared._SIM2SIM_PERF_VIEWER_FALLBACK_W),
        viewer_fallback_height=max(240, shared._SIM2SIM_PERF_VIEWER_FALLBACK_H),
        mujoco_full_ui=args.full_ui,
    )


if __name__ == "__main__":
    main()
