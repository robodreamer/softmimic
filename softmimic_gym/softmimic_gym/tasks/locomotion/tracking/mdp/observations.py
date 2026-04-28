"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import softmimic_gym.tasks.locomotion.tracking.mdp as mdp
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

_GRAVITY_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _gravity_vector(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _GRAVITY_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
        _GRAVITY_CACHE[key] = cached
    return cached


def generated_commands_slice(env: ManagerBasedRLEnv, command_name: str, start: int, end: int, repeat_length: int = 0) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    if repeat_length > 0:
        sliceable_command = env.command_manager.get_command(command_name)
        sliceable_command = sliceable_command.reshape(env.num_envs, repeat_length, -1)
        sliced_command = sliceable_command[..., start:end].transpose(1, 2).reshape(env.num_envs, -1)
        return sliced_command
    return env.command_manager.get_command(command_name)[..., start:end]

def projected_gravity_bodies(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    body_id = asset_cfg.body_ids[0] # just use the first body id for now ( waist yaw is pelvis)
    asset = env.scene[asset_cfg.name]
    body_states_w = asset.data.body_state_w[:, body_id]
    body_quat_w = body_states_w[:, 3:7]
    gravity_vec_w = _gravity_vector(body_quat_w.device, body_quat_w.dtype).expand(body_quat_w.shape[0], -1)
    body_projected_gravity_b = math_utils.quat_rotate_inverse(body_quat_w, gravity_vec_w)
    return body_projected_gravity_b

def keypoint_error(env: ManagerBasedRLEnv, command_name: str, keypoint_body_ids: list[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The error between the keypoint position and the target position."""
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # TODO(gmargo): Why are the ankles swapped??
    # hardcoded_keypoint_body_ids = [
    #     0, # pelvis
    #     9, # left_hip_yaw_link
    #     12, # left_knee_link
    #     26, # right_ankle_roll_link
    #     10, # right_hip_yaw_link
    #     13, # right_knee_link
    #     23, # left_ankle_roll_link
    #     11, # torso_link
    #     28, # left_shoulder_yaw_link
    #     30, # left_elbow_link
    #     36, # left_wrist_yaw_link
    #     29, # right_shoulder_yaw_link
    #     31, # right_elbow_link
    #     37, # right_wrist_yaw_link
    # ]

    commanded_keypoints = env.command_manager.get_term(command_name).keypoints
    # current_keypoints = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]  # Get the current keypoints from the asset's body positions
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids, :3]  # Get the current keypoints from the asset's body positions
    
    keypoint_error = current_keypoints - commanded_keypoints
    keypoint_error = keypoint_error.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_keypoints * 3)
    return keypoint_error


def keypoint_error_local(env: ManagerBasedRLEnv, command_name: str, keypoint_body_ids: list[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The error between the keypoint position and the target position."""
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    current_root_pos = asset.data.root_pos_w[:, :3].clone()
    current_root_pos[:, 2] = 0.0 # ignore z position
    current_root_rot = asset.data.root_quat_w[:, :4]
    # current_root_rot = current_root_rot[:, [1, 2, 3, 0]]  # convert to xyzw
    _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
    
    # commanded_root_pos = env.command_manager.get_term(command_name).root_state["root_pos"][:, :3].clone()
    # commanded_root_pos[:, 2] = 0.0 # ignore z position
    # commanded_root_rot = env.command_manager.get_term(command_name).root_state["root_rot"][:, :4]
    # commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]  # convert to wxyz
    # _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)
    
    # yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
    current_yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), current_yaw)
    
    
    # TODO(gmargo): Why are the ankles swapped??
    # hardcoded_keypoint_body_ids = [
    #     0, # pelvis
    #     9, # left_hip_yaw_link
    #     12, # left_knee_link
    #     26, # right_ankle_roll_link
    #     10, # right_hip_yaw_link
    #     13, # right_knee_link
    #     23, # left_ankle_roll_link
    #     11, # torso_link
    #     28, # left_shoulder_yaw_link
    #     30, # left_elbow_link
    #     36, # left_wrist_yaw_link
    #     29, # right_shoulder_yaw_link
    #     31, # right_elbow_link
    #     37, # right_wrist_yaw_link
    # ]

    commanded_keypoints = env.command_manager.get_term(command_name).keypoints
    # current_keypoints = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]  # Get the current keypoints from the asset's body positions
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids, :3]  # Get the current keypoints from the asset's body positions

    keypoint_error = current_keypoints - commanded_keypoints

    # Rotate the keypoint error to the local yaw frame
    keypoint_error = math_utils.quat_rotate_inverse(
        current_yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints.shape[1], 1),
        keypoint_error,
    )

    keypoint_error = keypoint_error.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_keypoints * 3)
    return keypoint_error

def keypoint_error_local_shifted(env: ManagerBasedRLEnv, command_name: str, keypoint_body_ids: list[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The error between the keypoint position and the target position."""
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    current_root_pos = asset.data.root_pos_w[:, :3].clone()
    current_root_pos[:, 2] = 0.0 # ignore z position
    current_root_rot = asset.data.root_quat_w[:, :4].clone()
    # current_root_rot = current_root_rot[:, [1, 2, 3, 0]]  # convert to xyzw
    _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
    
    commanded_root_pos = env.command_manager.get_term(command_name).root_state["root_pos"][:, 0, :3].clone()
    # print(commanded_root_pos[0])
    commanded_root_pos[:, 2] = 0.0 # ignore z position
    commanded_root_rot = env.command_manager.get_term(command_name).root_state["root_rot"][:,0, :4].clone()
    commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]  # convert to wxyz
    # print("root rot", commanded_root_rot.shape)
    _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)
    
    yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
    yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), yaw_diff)
    current_yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), current_yaw)
    
    # TODO(gmargo): Why are the ankles swapped??
    # hardcoded_keypoint_body_ids = [
    #     0, # pelvis
    #     9, # left_hip_yaw_link
    #     12, # left_knee_link
    #     26, # right_ankle_roll_link
    #     10, # right_hip_yaw_link
    #     13, # right_knee_link
    #     23, # left_ankle_roll_link
    #     11, # torso_link
    #     28, # left_shoulder_yaw_link
    #     30, # left_elbow_link
    #     36, # left_wrist_yaw_link
    #     29, # right_shoulder_yaw_link
    #     31, # right_elbow_link
    #     37, # right_wrist_yaw_link
    # ]
    # for i, body_name in enumerate(asset.data.body_names):
    #     print(i, body_name)
    # input()

    commanded_keypoints = env.command_manager.get_term(command_name).keypoints
    # current_keypoints = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]  # Get the current keypoints from the asset's body positions
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids, :3]  # Get the current keypoints from the asset's body positions
    commanded_keypoints_local = commanded_keypoints[:, 0,:,:] - commanded_root_pos.unsqueeze(1)

    commanded_keypoints_local = math_utils.quat_rotate(
        yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints.shape[2], 1),
        commanded_keypoints_local,
    )
    current_keypoints_local = current_keypoints - current_root_pos.unsqueeze(1)

    keypoint_error = current_keypoints_local - commanded_keypoints_local

    # Rotate the keypoint error to the local yaw frame
    keypoint_error = math_utils.quat_rotate_inverse(
        current_yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints.shape[2], 1),
        keypoint_error,
    )

    keypoint_error = keypoint_error.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_keypoints * 3)
    return keypoint_error

def foot_contacts(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), cse_cfg = None) -> torch.Tensor:
    """The foot contacts of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["contact_forces"]
    
    # Get the contact forces for the feet
    foot_contact_forces = contact_sensor.data.net_forces_w_history[:, :, asset_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    
    # Check if the contact forces are above a threshold (e.g., 1.0)
    foot_contacts = (foot_contact_forces > 1.0).float()
    
    return foot_contacts

def reference_root_height(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ref_height = env.command_manager.get_term(command_name).original_root_state['root_pos'][:,future_steps,2:3].reshape(env.num_envs, -1)
    return ref_height

def reference_dof_pos(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    dof_pos = env.command_manager.get_term(command_name).original_dof_pos[:,future_steps,:].reshape(env.num_envs, -1)
    return dof_pos

def reference_xy_vel(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    xy_vel = env.command_manager.get_term(command_name).original_root_state['root_vel'][:, future_steps, 0:2].reshape(env.num_envs, -1)
    return xy_vel

def reference_ang_vel(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ang_vel = env.command_manager.get_term(command_name).original_root_state['root_ang_vel'][:,future_steps,2:3].reshape(env.num_envs, -1)
    return ang_vel

def reference_gravity_vector(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ref_gravity_vector = env.command_manager.get_term(command_name).original_root_state['projected_gravity_b'][:,future_steps,0:3].reshape(env.num_envs, -1)
    return ref_gravity_vector

def reference_foot_contacts(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The foot contacts of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_contacts = env.command_manager.get_term(command_name).original_foot_contacts[:, future_steps, :].reshape(env.num_envs, -1)
    return foot_contacts

def adapted_reference_root_height(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ref_height = env.command_manager.get_term(command_name).adapted_root_state['root_pos'][:,future_steps,2:3].reshape(env.num_envs, -1)
    return ref_height

def adapted_reference_dof_pos(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    dof_pos = env.command_manager.get_term(command_name).adapted_dof_pos[:,future_steps,:].reshape(env.num_envs, -1)
    return dof_pos

def adapted_reference_xy_vel(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    xy_vel = env.command_manager.get_term(command_name).adapted_root_state['root_vel'][:, future_steps, 0:2].reshape(env.num_envs, -1)
    return xy_vel

def adapted_reference_ang_vel(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ang_vel = env.command_manager.get_term(command_name).adapted_root_state['root_ang_vel'][:,future_steps,2:3].reshape(env.num_envs, -1)
    return ang_vel

def adapted_reference_gravity_vector(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    ref_gravity_vector = env.command_manager.get_term(command_name).adapted_root_state['projected_gravity_b'][:,future_steps,0:3].reshape(env.num_envs, -1)
    return ref_gravity_vector

def adapted_reference_foot_contacts(env: ManagerBasedRLEnv, command_name: str, future_steps: list[int] = [0], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The foot contacts of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_contacts = env.command_manager.get_term(command_name).adapted_foot_contacts[:, future_steps, :].reshape(env.num_envs, -1)
    return foot_contacts

def foot_contact_forces(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The foot contacts of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["contact_forces"]
    
    # Get the contact forces for the feet
    foot_contact_forces = contact_sensor.data.net_forces_w_history[:, :, asset_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    
    # Check if the contact forces are above a threshold (e.g., 1.0)
    # foot_contacts = (foot_contact_forces > 1.0).float()
    
    return foot_contact_forces.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_feet * 3)

def foot_velocities(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The foot velocities of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the foot velocities from the asset's body velocities
    foot_velocities = asset.data.body_vel_w[:, asset_cfg.body_ids, :3]  # Get the linear velocities of the feet
    
    return foot_velocities.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_feet * 3)

def root_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), cse_cfg = None) -> torch.Tensor:
    """The height of the root of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w[:, :3]  # Get the root position
    return root_pos[:, 2].unsqueeze(1)  # Return the z-coordinate (height) as a tensor of shape (num_envs, 1)
  
def root_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), cse_cfg = None) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b

def root_lin_vel_xy(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), cse_cfg = None) -> torch.Tensor:
    """Root linear velocity in the asset's root frame, only x and y components."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :2]  # Return only x and y components

def root_ang_vel_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), cse_cfg = None) -> torch.Tensor:
    """Root angular velocity around the z-axis in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2].unsqueeze(1)  # Return only z component as a tensor of shape (num_envs, 1)

def force_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The force command for the robot."""
    
    # Get the commanded forces from the command manager
    command_term = env.command_manager.get_term(command_name)
    target_forces_b = command_term.target_forces_b
    body_ids = command_term.body_ids

    all_body_ids = command_term.sorted_keypoint_ids  # Use the sorted keypoint IDs from the command term
    num_bodies = len(all_body_ids)
    num_active_envs = env.num_envs
    force_commands = torch.zeros(num_active_envs, num_bodies, 3, device=env.device)
    env_row_indices = torch.arange(num_active_envs, device=env.device)

    force_commands[env_row_indices, body_ids, :] = target_forces_b
    
    return force_commands.reshape(num_active_envs, -1)  # Reshape to (num_envs, num_bodies * 3)

def forcefield_force_applied(env: ManagerBasedRLEnv, command_name: str, observe_force_body_ids: list[int], cse_cfg = None) -> torch.Tensor:
    """The force field force for the robot."""
    
    # Get the commanded forces from the command manager
    command_term = env.command_manager.get_term(command_name)
    forcefield_forces_b = command_term.forcefield_forces_b
    body_ids = command_term.body_ids

    all_body_ids = command_term.sorted_keypoint_ids  # Use the sorted keypoint IDs from the command term
    num_bodies = len(all_body_ids)
    num_active_envs = env.num_envs
    force_field_forces = torch.zeros(num_active_envs, num_bodies, 3, device=env.device)
    env_row_indices = torch.arange(num_active_envs, device=env.device)

    # Get the body col indices
    # Convert to tensors if they aren't already
    if not isinstance(all_body_ids, torch.Tensor):
        all_body_ids = torch.tensor(all_body_ids, device=env.device)
    if not isinstance(body_ids, torch.Tensor):
        body_ids = torch.tensor(body_ids, device=env.device)
    
    # Find indices using tensor operations
    body_col_indices = torch.searchsorted(all_body_ids, body_ids)

    force_field_forces[env_row_indices, body_col_indices, :] = forcefield_forces_b
    
    force_field_forces = force_field_forces[:, observe_force_body_ids, :]
    # print(force_field_forces)
    
    return force_field_forces.reshape(num_active_envs, -1)  # Reshape to (num_envs, num_bodies * 3)
    

def forcefield_torque_applied(env: ManagerBasedRLEnv, command_name: str, observe_force_body_ids: list[int], cse_cfg = None) -> torch.Tensor:
    """The force field torque for the robot."""

    # Get the commanded torques from the command manager
    command_term = env.command_manager.get_term(command_name)
    forcefield_torques_b = command_term.forcefield_torques_b
    body_ids = command_term.body_ids

    all_body_ids = command_term.sorted_keypoint_ids  # Use the sorted keypoint IDs from the command term
    num_bodies = len(all_body_ids)
    num_active_envs = env.num_envs
    force_field_torques = torch.zeros(num_active_envs, num_bodies, 3, device=env.device)
    env_row_indices = torch.arange(num_active_envs, device=env.device)

    # Get the body col indices
    # Convert to tensors if they aren't already
    if not isinstance(all_body_ids, torch.Tensor):
        all_body_ids = torch.tensor(all_body_ids, device=env.device)
    if not isinstance(body_ids, torch.Tensor):
        body_ids = torch.tensor(body_ids, device=env.device)
    
    # Find indices using tensor operations
    body_col_indices = torch.searchsorted(all_body_ids, body_ids)

    force_field_torques[env_row_indices, body_col_indices, :] = forcefield_torques_b
    
    force_field_torques = force_field_torques[:, observe_force_body_ids, :]
    # print(force_field_torques)
    
    return force_field_torques.reshape(num_active_envs, -1)  # Reshape to (num_envs, num_bodies * 3)
    
def forcefield_force_desired(env: ManagerBasedRLEnv, command_name: str, observe_force_body_ids: list[int], cse_cfg = None) -> torch.Tensor:
    """The desired force field force for the robot."""
    
    # Get the commanded forces from the command manager
    command_term = env.command_manager.get_term(command_name)
    desired_forcefield_forces_b = command_term.target_forces_b
    body_ids = command_term.body_ids

    all_body_ids = command_term.sorted_keypoint_ids  # Use the sorted keypoint IDs from the command term
    num_bodies = len(all_body_ids)
    num_active_envs = env.num_envs
    desired_force_field_forces = torch.zeros(num_active_envs, num_bodies, 3, device=env.device)
    env_row_indices = torch.arange(num_active_envs, device=env.device)

    # Get the body col indices
    # Convert to tensors if they aren't already
    if not isinstance(all_body_ids, torch.Tensor):
        all_body_ids = torch.tensor(all_body_ids, device=env.device)
    if not isinstance(body_ids, torch.Tensor):
        body_ids = torch.tensor(body_ids, device=env.device)
    
    # Find indices using tensor operations
    body_col_indices = torch.searchsorted(all_body_ids, body_ids)

    desired_force_field_forces[env_row_indices, body_col_indices, :] = desired_forcefield_forces_b
    
    desired_force_field_forces = desired_force_field_forces[:, observe_force_body_ids, :]
    # print(desired_force_field_forces)
    
    return desired_force_field_forces.reshape(num_active_envs, -1)  # Reshape to (num_envs, num_bodies * 3)

def forcefield_torque_desired(env: ManagerBasedRLEnv, command_name: str, observe_force_body_ids: list[int], cse_cfg = None) -> torch.Tensor:
    """The desired force field torque for the robot."""

    # Get the commanded torques from the command manager
    command_term = env.command_manager.get_term(command_name)
    desired_forcefield_torques_b = command_term.target_torques_b
    body_ids = command_term.body_ids
    
    all_body_ids = command_term.sorted_keypoint_ids  # Use the sorted keypoint IDs from the command term
    num_bodies = len(all_body_ids)
    num_active_envs = env.num_envs
    desired_force_field_torques = torch.zeros(num_active_envs, num_bodies, 3, device=env.device)
    env_row_indices = torch.arange(num_active_envs, device=env.device)
    
    # Get the body col indices
    # Convert to tensors if they aren't already
    if not isinstance(all_body_ids, torch.Tensor):
        all_body_ids = torch.tensor(all_body_ids, device=env.device)
    if not isinstance(body_ids, torch.Tensor):
        body_ids = torch.tensor(body_ids, device=env.device)
        
    # Find indices using tensor operations
    body_col_indices = torch.searchsorted(all_body_ids, body_ids)
    
    desired_force_field_torques[env_row_indices, body_col_indices, :] = desired_forcefield_torques_b
    
    desired_force_field_torques = desired_force_field_torques[:, observe_force_body_ids, :]
    
    return desired_force_field_torques.reshape(num_active_envs, -1)  # Reshape to (num_envs, num_bodies * 3)
    
def forcefield_stiffness_log(env: ManagerBasedRLEnv, command_name: str, cse_cfg = None) -> torch.Tensor:
    """The force field stiffness for the robot."""
    
    # Get the commanded stiffness from the command manager
    command_term = env.command_manager.get_term(command_name)
    forcefield_stiffness = command_term.forcefield_stiffness
    
    return torch.log(forcefield_stiffness + 1e-6).reshape(env.num_envs, -1)  # Add a small value to avoid log(0)

def forcefield_rotational_stiffness_log(env: ManagerBasedRLEnv, command_name: str, cse_cfg = None) -> torch.Tensor:
    """The force field rotational stiffness for the robot."""

    # Get the commanded stiffness from the command manager
    command_term = env.command_manager.get_term(command_name)
    forcefield_rotational_stiffness = command_term.forcefield_rotational_stiffness

    return torch.log(forcefield_rotational_stiffness + 1e-6).reshape(env.num_envs, -1)  # Add a small value to avoid log(0)
    
def desired_stiffness(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The desired stiffness for the robot."""
    
    # Get the commanded stiffness from the command manager
    command_term = env.command_manager.get_term(command_name)
    desired_stiffness = command_term.desired_stiffness
    
    return desired_stiffness.reshape(env.num_envs, -1)

def desired_rotational_stiffness(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The desired rotational stiffness for the robot."""

    # Get the commanded stiffness from the command manager
    command_term = env.command_manager.get_term(command_name)
    desired_rotational_stiffness = command_term.desired_rotational_stiffness

    return desired_rotational_stiffness.reshape(env.num_envs, -1)

def desired_stiffness_inv(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The inverse of the desired stiffness for the robot."""
    
    # Get the commanded stiffness from the command manager
    command_term = env.command_manager.get_term(command_name)
    desired_stiffness = command_term.desired_stiffness
    
    # Apply inverse transformation
    inv_desired_stiffness = 1.0 / (desired_stiffness + 1e-6)  # Add a small value to avoid division by zero
    
    return inv_desired_stiffness.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_joints)

def desired_rotational_stiffness_inv(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The inverse of the desired rotational stiffness for the robot."""

    # Get the commanded stiffness from the command manager
    command_term = env.command_manager.get_term(command_name)
    desired_rotational_stiffness = command_term.desired_rotational_stiffness

    # Apply inverse transformation
    inv_desired_rotational_stiffness = 1.0 / (desired_rotational_stiffness + 1e-6)  # Add a small value to avoid division by zero

    return inv_desired_rotational_stiffness.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_joints)

def desired_stiffness_log(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The log of the desired stiffness for the robot."""
    
    # Get the commanded stiffness from the command manager
    command_term = env.command_manager.get_term(command_name)
    desired_stiffness = command_term.desired_stiffness
    
    # Apply log transformation
    log_desired_stiffness = torch.log(desired_stiffness + 1e-6)  # Add a small value to avoid log(0)
    
    # print(desired_stiffness, log_desired_stiffness)

    return log_desired_stiffness.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_joints)

def desired_rotational_stiffness_log(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The log of the desired rotational stiffness for the robot."""

    # Get the commanded stiffness from the command manager
    command_term = env.command_manager.get_term(command_name)
    desired_rotational_stiffness = command_term.desired_rotational_stiffness

    # Apply log transformation
    log_desired_rotational_stiffness = torch.log(desired_rotational_stiffness + 1e-6)  # Add a small value to avoid log(0)

    # print(desired_rotational_stiffness, log_desired_rotational_stiffness)

    return log_desired_rotational_stiffness.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_joints)

def joint_torques_scaled(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint torques of the robot, scaled by the torque limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the current joint torques from the asset's data
    current_joint_torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    
    # Scale the joint torques by the torque limits
    # torque_limits = asset.get_joint_limits().effort_max[asset_cfg.joint_ids]
    torque_limits = torch.tensor([
        200, 200, 200, 150, 150, 200, 150, 150, 200, 200, 200, 40, 40, 20, 20, 40, 40, 20, 20,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    ], device = asset.device)
        
    scaled_joint_torques = current_joint_torques / (torque_limits + 1e-6)  # Add a small value to avoid division by zero
    
    return scaled_joint_torques.reshape(env.num_envs, -1)  # Reshape to (num_envs, num_joints)


def all_adapted_keypoints_pos_error_local(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Observation of the error between current and commanded keypoint positions for all links,
    transformed into the robot's anchored frame from the start of a force event.

    This observation is computed in three steps:
    1. The target keypoint positions from the reference motion are transformed into the world
       frame, anchored to the robot's position and orientation at the start of the force event.
       This logic mirrors the target computation in `force_link_keypoint_tracking_local`.
    2. A world-frame error vector is calculated by subtracting the robot's current keypoint
       positions from these anchored target positions.
    3. This world-frame error vector is then rotated into a local frame defined by the
       anchor rotation, making the observation invariant to the global heading at the event start.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    env_ids = torch.arange(env.num_envs, device=command_term.device)
    device = command_term.ff_anchor_rot.device

    # -- Get the static anchor transform captured at the beginning of the force event --
    anchor_rot = command_term.ff_anchor_rot         # Robot's yaw delta vs. ref at event start
    anchor_pos_robot = command_term.ff_anchor_pos   # Robot's world pos at event start
    anchor_pos_ref = command_term.ff_anchor_ref_pos # Reference motion's world pos at event start

    # -- Transform the commanded keypoint positions using the static anchor transform --
    # 1. Get the commanded keypoints for the current time step
    commanded_keypoints = command_term.adapted_keypoints[:, 1]

    # 2. Calculate commanded keypoints relative to the reference motion's start position
    keypoints_relative_to_ref_start = commanded_keypoints - anchor_pos_ref.unsqueeze(1)

    # 3. Rotate these relative keypoints by the anchor rotation
    keypoints_relative_rotated = math_utils.quat_rotate(
        anchor_rot.unsqueeze(1), keypoints_relative_to_ref_start
    )

    # 4. Translate them to the robot's world frame, anchored to its start position
    # This gives the desired world-frame keypoint positions for the robot to track.
    transformed_commanded_keypoints = keypoints_relative_rotated + anchor_pos_robot.unsqueeze(1)

    # -- Calculate the error in the world frame --
    # Get the robot's current keypoints in the world frame
    keypoint_body_ids_tensor = command_term.keypoint_body_ids_tensor
    if keypoint_body_ids_tensor is None:
        raise RuntimeError("Command term does not provide keypoint body ids required for observation computation.")
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids_tensor, :3]

    # Compute the world-frame error vector
    world_frame_error = transformed_commanded_keypoints - current_keypoints

    # -- Transform the error vector into the local anchored frame --
    # Rotate the world-frame error by the inverse of the anchor rotation. This expresses the error
    # in a frame aligned with the reference motion's heading at the start of the event,
    # making the observation independent of the initial robot heading.
    local_frame_error = math_utils.quat_rotate_inverse(
        anchor_rot.unsqueeze(1), world_frame_error
    )

    # -- Select and mask the error for the active force link --
    force_link_id = command_term.force_link_id[:, 1, 0].clone()
    # Create a valid index even when no force is applied (error will be masked out later)
    force_link_id[force_link_id < 0] = 0
    force_link_idxs = (keypoint_body_ids_tensor.unsqueeze(0) == force_link_id.unsqueeze(1)).nonzero()[:, 1]

    local_frame_error = local_frame_error[env_ids, force_link_idxs]
    local_frame_error[~command_term.active_force_mask] = 0.0

    # Reshape to (num_envs, num_keypoints * 3) for the observation space
    return local_frame_error.reshape(env.num_envs, -1)


def all_adapted_keypoints_quat_error_local(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Observation of the orientation error between current and commanded keypoints for all links,
    represented as a 3D axis-angle vector.

    The target keypoint orientations are first rotated by the anchor yaw from the start of the
    force event. The error is then computed as the relative rotation between the robot's
    current keypoint orientations and these transformed targets. This relative error is
    converted to a 3D vector (axis-angle representation) suitable for an observation space.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    env_ids = torch.arange(env.num_envs, device=command_term.device)
    device = command_term.ff_anchor_rot.device

    # -- Get the static anchor rotation captured at the beginning of the force event --
    anchor_rot = command_term.ff_anchor_rot

    # -- Transform the commanded keypoint orientations using the static anchor rotation --
    # 1. Get commanded orientations for the current time step (in wxyz format)
    commanded_keypoint_quats = command_term.adapted_keypoint_rotations[:, 1]

    # 2. Apply the anchor rotation by broadcasting it across all keypoints
    num_keypoints = commanded_keypoint_quats.shape[1]
    anchor_rot_broadcastable = anchor_rot.unsqueeze(1).repeat(1, num_keypoints, 1)
    transformed_commanded_quats = math_utils.quat_mul(
        anchor_rot_broadcastable, commanded_keypoint_quats
    )

    # -- Calculate the orientation error --
    # Get the robot's current keypoint orientations
    keypoint_body_ids_tensor = command_term.keypoint_body_ids_tensor
    if keypoint_body_ids_tensor is None:
        raise RuntimeError("Command term does not provide keypoint body ids required for observation computation.")
    current_keypoint_quats = asset.data.body_quat_w[:, keypoint_body_ids_tensor, :4]

    # Compute the error quaternion: q_error = q_target * q_current_inverse
    # This gives the rotation needed to get from the current to the target orientation.
    error_quat = math_utils.quat_mul(
        transformed_commanded_quats, math_utils.quat_conjugate(current_keypoint_quats)
    )

    # Convert the error quaternion to a 3D axis-angle representation for the observation.
    # A positive scalar part (w) of the quaternion corresponds to the shortest rotation path.
    mask = error_quat[..., 0] < 0.0
    error_quat[mask] = -error_quat[mask]
    # The error vector is 2 * the vector part of the quaternion, which approximates the axis-angle vector.
    orientation_error_vec = 2.0 * error_quat[..., 1:]

    # -- Select and mask the error for the active force link --
    force_link_id = command_term.force_link_id[:, 1, 0].clone()
    force_link_id[force_link_id < 0] = 0
    force_link_idxs = (keypoint_body_ids_tensor.unsqueeze(0) == force_link_id.unsqueeze(1)).nonzero()[:, 1]

    orientation_error_vec = orientation_error_vec[env_ids, force_link_idxs]

    # Reshape to (num_envs, num_keypoints * 3) for the observation space
    return orientation_error_vec.reshape(env.num_envs, -1)
