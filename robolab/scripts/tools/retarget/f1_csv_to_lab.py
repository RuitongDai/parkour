# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert F1 GMR CSV motions to RoboLab AMP pickle files.

The input CSV layout is headerless and contains one frame per row::

    root_pos_xyz (3), root_quat_xyzw (4), F1 GMR/URDF-order joint_pos (28)

The output pickle layout is the one consumed by ``MotionDataTerm``::

    fps, root_pos, root_rot (wxyz), dof_pos (Isaac Lab order),
    loop_mode, key_body_pos

The Isaac Lab joint order is queried from the instantiated articulation rather
than hard-coded. Key-body world positions are calculated by PhysX articulation
forward kinematics using the same URDF imported by the training environment.
No joint values are zeroed or otherwise altered.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("robolab/data/motions/f1_gmr_csv"),
        help="Directory containing headerless F1 GMR CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("robolab/data/motions/f1_lab"),
        help="Directory for generated AMP pickle files.",
    )
    parser.add_argument(
        "--fps_metadata_dir",
        type=Path,
        default=Path("robolab/data/motions/rpo_lab"),
        help=(
            "Optional directory of same-name motion pickles whose fps field is reused. "
            "Files without metadata use --default_fps."
        ),
    )
    parser.add_argument(
        "--default_fps",
        type=float,
        default=120.0,
        help="Frame rate used when no same-name metadata pickle exists.",
    )
    parser.add_argument(
        "--loop",
        choices=("clamp", "wrap"),
        default="clamp",
        help="Motion playback mode written to every output pickle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output pickle files.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from robolab.assets.robots.f1 import F1_CFG, F1_JOINT_NAMES


F1_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
]

# Keep the hands down beside the body.  In F1's joint convention elbow=0
# points the forearm forward, while 1.4 rad points it mostly downward.  Wrist
# joints remain neutral.  Apply this before forward kinematics so saved key
# bodies and joint positions are generated from the same pose.
F1_NEUTRAL_ARM_JOINT_POS = {
    "left_elbow_joint": 1.4,
    "right_elbow_joint": 1.4,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Minimal scene used only for articulation forward kinematics."""

    robot: ArticulationCfg = F1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _load_pickle(path: Path) -> dict:
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError):
        return joblib.load(path)


def _get_fps(csv_path: Path) -> float:
    metadata_path = args_cli.fps_metadata_dir / f"{csv_path.stem}.pkl"
    if metadata_path.is_file():
        metadata = _load_pickle(metadata_path)
        fps = float(metadata["fps"])
    else:
        fps = float(args_cli.default_fps)
        print(f"[WARNING] No fps metadata for {csv_path.name}; using {fps:g} FPS.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{csv_path.name}: fps must be positive and finite, got {fps}.")
    return fps


def _load_csv_motions(csv_paths: list[Path]) -> list[dict]:
    motions = []
    for csv_path in csv_paths:
        frames = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2)
        if frames.ndim != 2 or frames.shape[1] != 35:
            raise ValueError(
                f"{csv_path.name}: expected (frames, 35), got {frames.shape}. "
                "Required layout is root_pos(3) + root_quat_xyzw(4) + dof_pos(28)."
            )
        if frames.shape[0] < 2:
            raise ValueError(f"{csv_path.name}: at least two frames are required.")
        if not np.isfinite(frames).all():
            raise ValueError(f"{csv_path.name}: data contains NaN or infinity.")

        quat_norm = np.linalg.norm(frames[:, 3:7], axis=1)
        max_quat_norm_error = float(np.max(np.abs(quat_norm - 1.0)))
        if max_quat_norm_error > 1.0e-3:
            raise ValueError(
                f"{csv_path.name}: quaternion norm error {max_quat_norm_error:.3g} exceeds 1e-3."
            )

        motions.append(
            {
                "name": csv_path.stem,
                "fps": _get_fps(csv_path),
                "root_pos": frames[:, 0:3],
                "root_rot_xyzw": frames[:, 3:7],
                "dof_pos_gmr": frames[:, 7:35],
            }
        )
    return motions


def _validate_joint_sets(lab_joint_names: list[str]) -> list[int]:
    source_names = list(F1_JOINT_NAMES)
    if len(source_names) != 28 or len(set(source_names)) != 28:
        raise ValueError("F1 source joint list must contain 28 unique names.")
    if set(source_names) != set(lab_joint_names):
        missing = sorted(set(lab_joint_names) - set(source_names))
        extra = sorted(set(source_names) - set(lab_joint_names))
        raise ValueError(
            "F1 CSV and Isaac Lab joint sets differ: "
            f"missing_from_csv={missing}, extra_in_csv={extra}"
        )
    return [source_names.index(name) for name in lab_joint_names]


def _calculate_lab_motion_data(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    motions: list[dict],
) -> None:
    robot: Articulation = scene["robot"]
    lab_joint_names = list(robot.data.joint_names)
    gmr_to_lab_indices = _validate_joint_sets(lab_joint_names)

    print("[INFO] F1 GMR/URDF joint order:")
    for index, name in enumerate(F1_JOINT_NAMES):
        print(f"  {index:2d}: {name}")
    print("[INFO] F1 Isaac Lab articulation order:")
    for index, name in enumerate(lab_joint_names):
        print(f"  {index:2d}: {name}")

    body_names = list(robot.data.body_names)
    missing_bodies = [name for name in F1_KEY_BODY_NAMES if name not in body_names]
    if missing_bodies:
        raise ValueError(f"F1 articulation is missing key bodies: {missing_bodies}")
    key_body_indices = [body_names.index(name) for name in F1_KEY_BODY_NAMES]

    root_pos_tensors = []
    root_quat_tensors = []
    dof_pos_tensors = []
    key_body_pos_tensors = []
    num_frames = []

    for motion in motions:
        root_pos = torch.as_tensor(motion["root_pos"], dtype=torch.float32, device=scene.device)
        root_quat_xyzw = torch.as_tensor(
            motion["root_rot_xyzw"], dtype=torch.float32, device=scene.device
        )
        root_quat_wxyz = math_utils.convert_quat(root_quat_xyzw, "wxyz")
        root_quat_wxyz = math_utils.quat_unique(math_utils.normalize(root_quat_wxyz))
        dof_pos_gmr = torch.as_tensor(
            motion["dof_pos_gmr"], dtype=torch.float32, device=scene.device
        )
        dof_pos_lab = dof_pos_gmr[:, gmr_to_lab_indices]
        for joint_name, joint_pos in F1_NEUTRAL_ARM_JOINT_POS.items():
            dof_pos_lab[:, lab_joint_names.index(joint_name)] = joint_pos

        root_pos_tensors.append(root_pos)
        root_quat_tensors.append(root_quat_wxyz)
        dof_pos_tensors.append(dof_pos_lab)
        key_body_pos_tensors.append(
            torch.empty(
                (len(root_pos), len(F1_KEY_BODY_NAMES), 3),
                dtype=torch.float32,
                device=scene.device,
            )
        )
        num_frames.append(len(root_pos))

    max_num_frames = max(num_frames)
    for frame_index in range(max_num_frames):
        root_state = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)

        for motion_index, motion_num_frames in enumerate(num_frames):
            source_frame = min(frame_index, motion_num_frames - 1)
            root_state[motion_index, 0:3] = root_pos_tensors[motion_index][source_frame]
            root_state[motion_index, 0:3] += scene.env_origins[motion_index]
            root_state[motion_index, 3:7] = root_quat_tensors[motion_index][source_frame]
            root_state[motion_index, 7:13] = 0.0
            joint_pos[motion_index] = dof_pos_tensors[motion_index][source_frame]

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.forward()
        scene.update(sim.get_physics_dt())

        body_pos_w = robot.data.body_pos_w[:, key_body_indices, :]
        for motion_index, motion_num_frames in enumerate(num_frames):
            if frame_index < motion_num_frames:
                key_body_pos_tensors[motion_index][frame_index] = (
                    body_pos_w[motion_index] - scene.env_origins[motion_index]
                )

        if frame_index == 0 or (frame_index + 1) % 500 == 0 or frame_index + 1 == max_num_frames:
            print(f"[INFO] Forward kinematics: {frame_index + 1}/{max_num_frames} frames")

    for index, motion in enumerate(motions):
        motion["root_rot"] = root_quat_tensors[index].cpu().numpy()
        motion["dof_pos"] = dof_pos_tensors[index].cpu().numpy()
        motion["key_body_pos"] = key_body_pos_tensors[index].cpu().numpy()
        motion["lab_joint_names"] = lab_joint_names


def _save_motions(motions: list[dict]) -> None:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    loop_mode = 0 if args_cli.loop == "clamp" else 1

    for motion in motions:
        output_path = args_cli.output_dir / f"{motion['name']}.pkl"
        if output_path.exists() and not args_cli.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it.")

    for motion in motions:
        output_path = args_cli.output_dir / f"{motion['name']}.pkl"
        output_data = {
            "fps": float(motion["fps"]),
            "root_pos": np.asarray(motion["root_pos"], dtype=np.float32),
            "root_rot": np.asarray(motion["root_rot"], dtype=np.float32),
            "dof_pos": np.asarray(motion["dof_pos"], dtype=np.float32),
            "loop_mode": loop_mode,
            "key_body_pos": np.asarray(motion["key_body_pos"], dtype=np.float32),
        }
        with output_path.open("wb") as file:
            pickle.dump(output_data, file)
        print(
            f"[INFO] Saved {output_path}: frames={len(output_data['root_pos'])}, "
            f"fps={output_data['fps']:g}"
        )


def main() -> None:
    csv_paths = sorted(args_cli.input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args_cli.input_dir}")

    motions = _load_csv_motions(csv_paths)
    print(f"[INFO] Loaded {len(motions)} F1 CSV motions.")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / max(motion["fps"] for motion in motions), device=args_cli.device)
    )
    scene = InteractiveScene(
        ReplayMotionsSceneCfg(num_envs=len(motions), env_spacing=3.0)
    )
    sim.reset()

    _calculate_lab_motion_data(sim, scene, motions)
    _save_motions(motions)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
