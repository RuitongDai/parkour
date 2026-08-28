# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize F1 RoboLab AMP pickle motions in Isaac Lab.

Each input pickle must contain ``fps``, ``root_pos``, ``root_rot`` in WXYZ
quaternion order, and ``dof_pos`` in the F1 Isaac Lab articulation order.
The player writes those states directly to the articulation, so it shows the
stored reference motion instead of running a policy or simulating tracking.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep this group non-required while AppLauncher performs its preliminary
    # parse_known_args() call; enforce it after the final argument parse below.
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--motion_file",
        type=Path,
        help="Path to one F1 Lab-format motion pickle.",
    )
    source_group.add_argument(
        "--motion_dir",
        type=Path,
        help="Directory containing F1 Lab-format motion pickles.",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--all",
        action="store_true",
        help="Play every .pkl file in --motion_dir in filename order.",
    )
    selection_group.add_argument(
        "--names",
        nargs="+",
        help="Motion stems to play from --motion_dir, in the supplied order.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the selected motion or playlist until the window is closed.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First frame to play from every selected motion (default: 0).",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Exclusive final frame; omitted means the end of each motion.",
    )
    parser.add_argument(
        "--no_follow_camera",
        action="store_true",
        help="Keep the initial camera fixed instead of following the root.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.motion_file is None and args.motion_dir is None:
        parser.error("one of --motion_file or --motion_dir is required.")
    if args.speed <= 0.0 or not np.isfinite(args.speed):
        parser.error("--speed must be positive and finite.")
    if args.start_frame < 0:
        parser.error("--start_frame must be non-negative.")
    if args.end_frame is not None and args.end_frame <= args.start_frame:
        parser.error("--end_frame must be greater than --start_frame.")
    if args.motion_file is not None and (args.all or args.names):
        parser.error("--all/--names can only be used together with --motion_dir.")
    if args.motion_dir is not None and not (args.all or args.names):
        parser.error("--motion_dir requires either --all or --names.")
    return args


def _resolve_motion_paths(args: argparse.Namespace) -> list[Path]:
    if args.motion_file is not None:
        paths = [args.motion_file]
    else:
        if not args.motion_dir.is_dir():
            raise NotADirectoryError(f"Motion directory does not exist: {args.motion_dir}")
        if args.all:
            paths = sorted(args.motion_dir.glob("*.pkl"))
        else:
            paths = [args.motion_dir / f"{Path(name).stem}.pkl" for name in args.names]

    if not paths:
        raise FileNotFoundError("No .pkl motion files were selected.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Motion files do not exist: {missing}")
    return [path.resolve() for path in paths]


def _load_pickle(path: Path) -> dict:
    try:
        with path.open("rb") as file:
            data = pickle.load(file)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError):
        data = joblib.load(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path.name}: expected a dictionary, got {type(data).__name__}.")
    return data


def _load_motion(path: Path, start_frame: int, end_frame: int | None) -> dict:
    raw = _load_pickle(path)
    missing = sorted({"fps", "root_pos", "root_rot", "dof_pos"} - set(raw))
    if missing:
        raise KeyError(f"{path.name}: missing required fields: {missing}")

    fps = float(raw["fps"])
    root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
    root_rot = np.asarray(raw["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(raw["dof_pos"], dtype=np.float32)

    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path.name}: fps must be positive and finite, got {fps}.")
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path.name}: root_pos must have shape (frames, 3), got {root_pos.shape}.")
    if root_rot.shape != (len(root_pos), 4):
        raise ValueError(f"{path.name}: root_rot must have shape ({len(root_pos)}, 4), got {root_rot.shape}.")
    if dof_pos.shape != (len(root_pos), 28):
        raise ValueError(f"{path.name}: dof_pos must have shape ({len(root_pos)}, 28), got {dof_pos.shape}.")
    if not (np.isfinite(root_pos).all() and np.isfinite(root_rot).all() and np.isfinite(dof_pos).all()):
        raise ValueError(f"{path.name}: motion contains NaN or infinity.")

    quat_norm = np.linalg.norm(root_rot, axis=1)
    if np.any(quat_norm < 1.0e-8):
        raise ValueError(f"{path.name}: root_rot contains a zero-length quaternion.")
    max_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
    if max_quat_error > 1.0e-2:
        raise ValueError(
            f"{path.name}: quaternion norm error {max_quat_error:.3g} exceeds 1e-2; "
            "expected root_rot in normalized WXYZ format."
        )

    stop = len(root_pos) if end_frame is None else min(end_frame, len(root_pos))
    if start_frame >= stop:
        raise ValueError(
            f"{path.name}: selected frame range [{start_frame}, {stop}) is empty; "
            f"motion has {len(root_pos)} frames."
        )

    return {
        "name": path.stem,
        "path": path,
        "fps": fps,
        "root_pos": np.ascontiguousarray(root_pos[start_frame:stop]),
        "root_rot": np.ascontiguousarray(root_rot[start_frame:stop] / quat_norm[start_frame:stop, None]),
        "dof_pos": np.ascontiguousarray(dof_pos[start_frame:stop]),
        "source_start_frame": start_frame,
        "source_stop_frame": stop,
    }


args_cli = _parse_args()
motion_paths = _resolve_motion_paths(args_cli)
motions = [
    _load_motion(path, args_cli.start_frame, args_cli.end_frame)
    for path in motion_paths
]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from robolab.assets.robots.f1 import F1_CFG


@configclass
class F1MotionPlaybackSceneCfg(InteractiveSceneCfg):
    """Minimal scene containing the F1 articulation and a flat ground plane."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/"
                "PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )
    robot: ArticulationCfg = F1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _set_camera(sim: sim_utils.SimulationContext, root_pos_w: np.ndarray) -> None:
    target = root_pos_w + np.array([0.0, 0.0, 0.45], dtype=np.float32)
    eye = target + np.array([2.4, 2.4, 1.2], dtype=np.float32)
    sim.set_camera_view(eye.tolist(), target.tolist())


def _play_motion(
    motion: dict,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
) -> bool:
    root_pos = torch.as_tensor(motion["root_pos"], dtype=torch.float32, device=scene.device)
    root_rot = torch.as_tensor(motion["root_rot"], dtype=torch.float32, device=scene.device)
    dof_pos = torch.as_tensor(motion["dof_pos"], dtype=torch.float32, device=scene.device)

    fps = motion["fps"]
    frame_period = 1.0 / (fps * args_cli.speed)
    print(
        f"[PLAY] {motion['name']}: frames={len(root_pos)}, fps={fps:g}, "
        f"source_range=[{motion['source_start_frame']}, {motion['source_stop_frame']}), "
        f"duration={len(root_pos) / fps:.3f}s"
    )

    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    env_origin = scene.env_origins[0]
    start_time = time.perf_counter()

    for frame_index in range(len(root_pos)):
        if not simulation_app.is_running():
            return False

        root_state[0, 0:3] = root_pos[frame_index] + env_origin
        root_state[0, 3:7] = root_rot[frame_index]
        root_state[0, 7:13] = 0.0
        joint_pos[0] = dof_pos[frame_index]

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.forward()
        scene.update(1.0 / fps)

        if not args_cli.no_follow_camera:
            _set_camera(sim, root_state[0, 0:3].detach().cpu().numpy())
        # Headless validation does not need a renderer and calling render() can
        # block indefinitely on CPU-only Isaac Sim backends.
        if not args_cli.headless:
            sim.render()

        deadline = start_time + (frame_index + 1) * frame_period
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

    return True


def main() -> None:
    max_fps = max(motion["fps"] for motion in motions)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / max_fps, device=args_cli.device)
    )
    scene = InteractiveScene(F1MotionPlaybackSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()

    robot: Articulation = scene["robot"]
    if len(robot.data.joint_names) != 28:
        raise ValueError(
            f"F1 articulation must have 28 joints, got {len(robot.data.joint_names)}: "
            f"{robot.data.joint_names}"
        )

    print(f"[INFO] Loaded {len(motions)} motion(s). Close the window or press Ctrl+C to stop.")
    while simulation_app.is_running():
        for motion in motions:
            if not _play_motion(motion, sim, scene, robot):
                return
        if not args_cli.loop:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Playback interrupted.")
    finally:
        simulation_app.close()
