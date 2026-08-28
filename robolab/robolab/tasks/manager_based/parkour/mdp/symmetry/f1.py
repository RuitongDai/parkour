"""Left-right symmetry transformations for the 28-DoF F1 Parkour task."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]

F1_NUM_JOINTS = 28

# Isaac Lab 中 F1 的 action/joint 顺序：
#
#  0 left_hip_pitch_joint
#  1 right_hip_pitch_joint
#  2 waist_yaw_joint
#  3 left_hip_roll_joint
#  4 right_hip_roll_joint
#  5 waist_roll_joint
#  6 left_hip_yaw_joint
#  7 right_hip_yaw_joint
#  8 left_shoulder_pitch_joint
#  9 right_shoulder_pitch_joint
# 10 left_knee_joint
# 11 right_knee_joint
# 12 left_shoulder_roll_joint
# 13 right_shoulder_roll_joint
# 14 left_ankle_pitch_joint
# 15 right_ankle_pitch_joint
# 16 left_shoulder_yaw_joint
# 17 right_shoulder_yaw_joint
# 18 left_ankle_roll_joint
# 19 right_ankle_roll_joint
# 20 left_elbow_joint
# 21 right_elbow_joint
# 22 left_wrist_roll_joint
# 23 right_wrist_roll_joint
# 24 left_wrist_yaw_joint
# 25 right_wrist_yaw_joint
# 26 left_wrist_pitch_joint
# 27 right_wrist_pitch_joint

# 镜像后每个位置应该从原数组哪个位置取值。
F1_MIRROR_INDICES = (
    1, 0,        # hip pitch
    2,           # waist yaw
    4, 3,        # hip roll
    5,           # waist roll
    7, 6,        # hip yaw
    9, 8,        # shoulder pitch
    11, 10,      # knee
    13, 12,      # shoulder roll
    15, 14,      # ankle pitch
    17, 16,      # shoulder yaw
    19, 18,      # ankle roll
    21, 20,      # elbow
    23, 22,      # wrist roll
    25, 24,      # wrist yaw
    27, 26,      # wrist pitch
)

# 绕 x、z 轴旋转的关节镜像后取反；绕 y 轴的不取反。
F1_MIRROR_SIGNS = (
    1, 1,        # hip pitch: y
    -1,          # waist yaw: z
    -1, -1,      # hip roll: x
    -1,          # waist roll: x
    -1, -1,      # hip yaw: z
    1, 1,        # shoulder pitch: y
    1, 1,        # knee: y
    -1, -1,      # shoulder roll: x
    1, 1,        # ankle pitch: y
    -1, -1,      # shoulder yaw: z
    -1, -1,      # ankle roll: x
    1, 1,        # elbow: y
    -1, -1,      # wrist roll: x
    -1, -1,      # wrist yaw: z
    1, 1,        # wrist pitch: y
)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Append a left-right mirrored copy of observations and actions."""

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"])

        # critic observation group
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        # -- left-right
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs["critic"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)

    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _height_scan_left_right_dims(env: ManagerBasedRLEnv) -> tuple[int, int, int]:
    """(history_length, ny, nx) from the running env cfg (dataclass fields are not on config *classes*)."""
    cfg = getattr(env, "unwrapped", env).cfg
    pat = cfg.scene.height_scanner.pattern_cfg
    if pat.ordering != "xy":
        raise NotImplementedError(
            "height_scan L-R symmetry only supports GridPatternCfg ordering 'xy';"
            f" extend layouts if pattern uses ordering {pat.ordering!r}."
        )
    hist = cfg.observations.critic.height_scan.history_length
    res = float(pat.resolution)
    s0, s1 = float(pat.size[0]), float(pat.size[1])
    # Match ``isaaclab.sensors.ray_caster.patterns.grid_pattern`` (same arange endpoints / step).
    nx = int(torch.arange(-s0 / 2, s0 / 2 + 1.0e-9, res).numel())
    ny = int(torch.arange(-s1 / 2, s1 / 2 + 1.0e-9, res).numel())
    return hist, ny, nx


def _transform_height_scan_left_right(env: ManagerBasedRLEnv, hs: torch.Tensor) -> torch.Tensor:
    """Mirror lateral (y) grid rows; scalars only reorder (unlike ``depth_image``, scan is flat w.r.t. spatial axes)."""
    hist, ny, nx = _height_scan_left_right_dims(env)
    out = hs.view(hs.shape[0], hist, ny, nx).flip(dims=[2])
    return out.reshape(hs.shape)


def _transform_policy_obs_left_right(obs: TensorDict) -> TensorDict:
    """Left-right mirror for policy observations (``ObservationsCfg.PolicyCfg`` with ``concatenate_terms=False``)."""
    obs = obs.clone()
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    if "depth_image" in obs:
        obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: TensorDict) -> TensorDict:
    """Left-right mirror for critic observations."""
    obs = obs.clone()
    obs["base_lin_vel"] = _apply_xyz_sign(obs["base_lin_vel"], [1, -1, 1])
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    if "depth_image" in obs:
        obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    if "height_scan" in obs:
        obs["height_scan"] = _transform_height_scan_left_right(env, obs["height_scan"])
    return obs


def _transform_depth_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to depth image observations."""
    return torch.flip(obs, dims=(-1,))


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(
        *joint_data_shape[:-1],
        -1,
        F1_NUM_JOINTS,
    )
    joint_data = _switch_joints_left_right(joint_data)
    return joint_data.reshape(joint_data_shape)


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


def _switch_joints_left_right(
    joint_data: torch.Tensor,
) -> torch.Tensor:
    """Swap F1 left/right joints and apply coordinate signs."""

    indices = torch.tensor(
        F1_MIRROR_INDICES,
        device=joint_data.device,
        dtype=torch.long,
    )
    signs = joint_data.new_tensor(F1_MIRROR_SIGNS)

    return joint_data.index_select(-1, indices) * signs