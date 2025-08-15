"""Pick Orange Mimic Environment with DifferentialIKController implementation."""

from isaaclab.envs import ManagerBasedRLMimicEnv
import isaaclab.utils.math as PoseUtils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_from_matrix, matrix_from_quat
import torch
import gymnasium as gym
from typing import Dict, Any, Optional, Sequence
import numpy as np
import os
import traceback
import sys

# Logging utilities
import carb

class PickOrangeMimicEnv(ManagerBasedRLMimicEnv):
    """Pick Orange environment with Mimic support using DifferentialIKController."""
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        """Initialize environment with DifferentialIKController setup."""
        print("[DEBUG] Initializing PickOrangeMimicEnv with DifferentialIKController")
        
        try:
            super().__init__(cfg, render_mode, **kwargs)
            
            # Get robot reference
            robot = self.scene["robot"]
            gripper_body_id = robot.find_bodies("gripper")[0]
            
            # Calculate home position
            ee_pos_world = robot.data.body_pos_w[0, gripper_body_id].cpu().numpy()
            ee_pos_relative = ee_pos_world - self.scene.env_origins[0].cpu().numpy()
            self.isaac_ee_home = ee_pos_relative.flatten()
            
            print(f"[DEBUG] Gripper body ID: {gripper_body_id}")
            print(f"[DEBUG] EE home position (relative): {self.isaac_ee_home}")
            
            # Setup DifferentialIKController
            self._setup_diff_ik()
            
            # Get joint and body IDs for IK computation
            self._setup_robot_indices()
            
            print("[DEBUG] PickOrangeMimicEnv initialization complete")
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to initialize PickOrangeMimicEnv")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            raise

    def _setup_robot_indices(self):
        """Setup robot joint and body indices for IK computation."""
        try:
            robot = self.scene["robot"]
            
            # Hardcoded joint IDs for the 5 DOF arm (excluding gripper)
            self.arm_joint_ids = [0, 1, 2, 3, 4]  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
            self.gripper_joint_id = 5  # gripper joint controls jaw
            
            # Hardcoded body ID for jaw (gripper end-effector)
            self.jaw_body_id = 6  # jaw is the 7th body (0-indexed)
            
            # For this fixed-base robot, Jacobian index is body index - 1
            self.ee_jacobi_idx = self.jaw_body_id - 1
            
            print(f"[DEBUG] Robot indices setup:")
            print(f"  - Arm joint IDs: {self.arm_joint_ids}")
            print(f"  - Jaw body ID: {self.jaw_body_id}")
            print(f"  - EE Jacobian index: {self.ee_jacobi_idx}")
            print(f"  - Is fixed base: {robot.is_fixed_base}")
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to setup robot indices")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            raise

    def _setup_diff_ik(self):
        """Initialize DifferentialIKController."""
        try:
            print("[DEBUG] Setting up DifferentialIKController")
            
            # Configure the controller
            self.diff_ik_cfg = DifferentialIKControllerCfg(
                command_type="pose",        # Control both position and orientation
                use_relative_mode=False,    # Use absolute pose commands
                ik_method="dls",            # Use damped least-squares (most robust)
                ik_params={
                    "lambda_val": 0.1,     # Damping factor for DLS
                }
            )
            
            # Get number of environments
            num_envs = self.scene.num_envs if hasattr(self.scene, 'num_envs') else 1
            
            # Create the controller
            self.diff_ik_controller = DifferentialIKController(
                self.diff_ik_cfg,
                num_envs=num_envs,
                device=self.device
            )
            
            print(f"[DEBUG] DifferentialIKController created successfully")
            print(f"  - Command type: {self.diff_ik_cfg.command_type}")
            print(f"  - IK method: {self.diff_ik_cfg.ik_method}")
            print(f"  - Relative mode: {self.diff_ik_cfg.use_relative_mode}")
            print(f"  - Number of environments: {num_envs}")
            print(f"  - Device: {self.device}")
            
            self.use_diff_ik = True
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to setup DifferentialIKController")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            self.use_diff_ik = False
            raise

    def _solve_diff_ik(self, target_pos_b, target_quat_b, jacobian, current_joints, env_id=0):
        """
        Solve IK using DifferentialIKController.
        
        Args:
            target_pos_b: Target position in base frame
            target_quat_b: Target quaternion in base frame (w, x, y, z)
            jacobian: Jacobian matrix from PhysX
            current_joints: Current joint positions
            env_id: Environment ID for debugging
            
        Returns:
            Joint positions solution or None if failed
        """
        try:
            print(f"\n[DEBUG] Solving IK with DifferentialIKController for env {env_id}")
            
            # Convert inputs to tensors if needed
            if isinstance(target_pos_b, np.ndarray):
                target_pos_b = torch.tensor(target_pos_b, device=self.device, dtype=torch.float32)
            if isinstance(target_quat_b, np.ndarray):
                target_quat_b = torch.tensor(target_quat_b, device=self.device, dtype=torch.float32)
            if isinstance(current_joints, np.ndarray):
                current_joints = torch.tensor(current_joints, device=self.device, dtype=torch.float32)
                
            # Ensure correct shapes
            if target_pos_b.dim() == 1:
                target_pos_b = target_pos_b.unsqueeze(0)
            if target_quat_b.dim() == 1:
                target_quat_b = target_quat_b.unsqueeze(0)
            if current_joints.dim() == 1:
                current_joints = current_joints.unsqueeze(0)
            if jacobian.dim() == 2:
                jacobian = jacobian.unsqueeze(0)
                
            print(f"[DEBUG] Input shapes:")
            print(f"  - Target position: {target_pos_b.shape} = {target_pos_b.squeeze().cpu().numpy()}")
            print(f"  - Target quaternion: {target_quat_b.shape} = {target_quat_b.squeeze().cpu().numpy()}")
            print(f"  - Current joints: {current_joints.shape} = {current_joints.squeeze().cpu().numpy()}")
            print(f"  - Jacobian shape: {jacobian.shape}")
            
            # Set the desired command
            command = torch.cat([target_pos_b, target_quat_b], dim=-1)
            self.diff_ik_controller.set_command(command)
            
            print(f"[DEBUG] Command set: {command.squeeze().cpu().numpy()}")
            
            # Get current EE pose (needed for error computation)
            # For now, we'll compute it from forward kinematics if needed
            # This is a simplified version - you might need to adjust based on your robot
            
            # Compute IK solution
            joint_solution = self.diff_ik_controller.compute(
                ee_pos=target_pos_b,
                ee_quat=target_quat_b,
                jacobian=jacobian,
                joint_pos=current_joints
            )
            
            print(f"[DEBUG] IK solution computed: {joint_solution.squeeze().cpu().numpy()}")
            
            # Check for NaN or inf values
            if torch.isnan(joint_solution).any() or torch.isinf(joint_solution).any():
                carb.log_warn(f"[WARNING] IK solution contains NaN or Inf values")
                return None
                
            return joint_solution.squeeze().cpu().numpy()
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to solve IK with DifferentialIKController")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            return None

    def _get_jacobian_for_env(self, env_id=0):
        """Get Jacobian matrix from PhysX for specific environment."""
        try:
            robot = self.scene["robot"]
            
            # Get Jacobian from PhysX
            # Shape: [num_envs, num_bodies, 6, num_joints]
            all_jacobians = robot.root_physx_view.get_jacobians()
            
            # Extract Jacobian for specific environment and end-effector
            jacobian = all_jacobians[env_id, self.ee_jacobi_idx, :, self.arm_joint_ids]
            
            print(f"[DEBUG] Jacobian retrieved from PhysX:")
            print(f"  - Full Jacobian shape: {all_jacobians.shape}")
            print(f"  - Extracted Jacobian shape: {jacobian.shape}")
            print(f"  - Jacobian norm: {torch.norm(jacobian).item():.4f}")
            
            # Check for validity
            if torch.isnan(jacobian).any() or torch.isinf(jacobian).any():
                carb.log_error(f"[ERROR] Jacobian contains NaN or Inf values!")
                return None
                
            return jacobian
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to get Jacobian from PhysX")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            return None
        
    # def target_eef_pose_to_action(
    #     self,
    #     target_eef_pose_dict: dict,
    #     gripper_action_dict: dict,
    #     action_noise_dict: dict | None = None,
    #     env_id: int = 0,
    # ) -> torch.Tensor:
    #     """DIRTY FIX - Return exactly 6 values to match action space."""
    #     robot = self.scene["robot"]
    #     # Get first 6 joint positions (5 arm + 1 gripper)
    #     all_joints = robot.data.joint_pos[env_id, :6]  # First 6 joints
        
    #     # Return exactly 6 values
    #     return all_joints

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EE pose to joint positions - FIXED VERSION."""
        
        robot = self.scene["robot"]
        num_envs = self.scene.num_envs
        
        # Get current joints for ALL envs
        current_joints_all = robot.data.joint_pos[:, self.arm_joint_ids]
        
        # Get CURRENT end-effector poses for ALL envs
        current_ee_pos = robot.data.body_pos_w[:, self.jaw_body_id]  # Current EE positions
        current_ee_quat = robot.data.body_quat_w[:, self.jaw_body_id]  # Current EE orientations
        
        # Convert to relative coordinates
        env_origins = self.scene.env_origins
        current_ee_pos = current_ee_pos - env_origins
        
        # Get Jacobian for ALL envs
        all_jacobians = robot.root_physx_view.get_jacobians()
        jacobian_all = all_jacobians[:, self.ee_jacobi_idx, :, self.arm_joint_ids]
        
        # Process TARGET pose
        eef_name = "eef"
        target_eef_pose = target_eef_pose_dict[eef_name]
        target_pos, target_rot_matrix = PoseUtils.unmake_pose(target_eef_pose)
        
        # Handle batch dimensions
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0).repeat(num_envs, 1)
            target_rot_matrix = target_rot_matrix.unsqueeze(0).repeat(num_envs, 1, 1)
        
        target_quat = quat_from_matrix(target_rot_matrix)
        
        # Set the TARGET command
        command = torch.cat([target_pos, target_quat], dim=-1)
        self.diff_ik_controller.set_command(command)
        
        # Compute IK with CURRENT pose and TARGET command
        joint_solution = self.diff_ik_controller.compute(
            ee_pos=current_ee_pos,      # ✅ CURRENT position
            ee_quat=current_ee_quat,    # ✅ CURRENT orientation  
            jacobian=jacobian_all,
            joint_pos=current_joints_all
        )
        
        # Extract solution for specific env
        solution_for_env = joint_solution[env_id]
        
        # Add gripper
        gripper_action = gripper_action_dict[eef_name]
        return torch.cat([solution_for_env, gripper_action], dim=0)

    def _fallback_action(self, current_joints, gripper_action):
        """Create a safe fallback action when IK fails."""
        try:
            if isinstance(current_joints, np.ndarray):
                current_joints = torch.tensor(current_joints, device=self.device, dtype=torch.float32)
            
            if current_joints.dim() > 1:
                current_joints = current_joints.squeeze()
                
            return torch.cat([current_joints, gripper_action], dim=0)
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to create fallback action")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            # Return zeros as last resort
            return torch.zeros(6, device=self.device, dtype=torch.float32)

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose as 4x4 transformation matrix."""
        try:
            robot = self.scene["robot"]
            
            if env_ids is None:
                env_ids = slice(None)
            
            gripper_pos = robot.data.body_pos_w[env_ids, self.jaw_body_id]
            gripper_quat = robot.data.body_quat_w[env_ids, self.jaw_body_id]
            
            # Ensure proper tensor dimensions
            if gripper_pos.dim() == 1:
                gripper_pos = gripper_pos.unsqueeze(0)
            if gripper_quat.dim() == 1:
                gripper_quat = gripper_quat.unsqueeze(0)
            
            # Convert to environment-relative coordinates
            env_origins = self.scene.env_origins[env_ids]
            if env_origins.dim() == 1:
                env_origins = env_origins.unsqueeze(0)
            gripper_pos = gripper_pos - env_origins
            
            # Create pose matrix
            pose_matrix = PoseUtils.make_pose(
                gripper_pos, 
                matrix_from_quat(gripper_quat)
            )
            
            return pose_matrix
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to get robot EE pose")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            raise
    
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert joint position action to target EE pose using FK."""
        try:
            eef_name = "eef"
            robot = self.scene["robot"]
            
            # Store original joint positions
            current_joint_pos = robot.data.joint_pos.clone()
            
            # Set target joint positions
            target_joint_pos = current_joint_pos.clone()
            target_joint_pos[:, self.arm_joint_ids] = action[:, :len(self.arm_joint_ids)]
            
            # Temporarily update robot state for FK computation
            original_joint_pos = current_joint_pos.clone()
            robot.data.joint_pos[:] = target_joint_pos
            
            # Update to compute forward kinematics
            robot.update(dt=0.0)
            
            # Get resulting EE pose
            target_ee_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
            
            # Restore original joint positions
            robot.data.joint_pos[:] = original_joint_pos
            robot.update(dt=0.0)
            
            return {eef_name: target_ee_pose}
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed in action_to_target_eef_pose")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            raise
    
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from action sequence."""
        try:
            if actions.dim() == 3:
                gripper_actions = actions[:, :, -1:]
            elif actions.dim() == 2:
                gripper_actions = actions[:, -1:]
            else:
                raise ValueError(f"Unexpected action shape: {actions.shape}")
        
            return {"eef": gripper_actions}
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to extract gripper actions")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            raise
    
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Get object poses as 4x4 transformation matrices."""
        try:
            if env_ids is None:
                env_ids = slice(None)
            
            cube = self.scene["cube"]
            cube_pos = cube.data.root_pos_w[env_ids]
            cube_quat = cube.data.root_quat_w[env_ids]
            
            # Ensure proper tensor dimensions
            if cube_pos.dim() == 1:
                cube_pos = cube_pos.unsqueeze(0)
            if cube_quat.dim() == 1:
                cube_quat = cube_quat.unsqueeze(0)
            
            # Convert to environment-relative coordinates
            env_origins = self.scene.env_origins[env_ids]
            if env_origins.dim() == 1:
                env_origins = env_origins.unsqueeze(0)
            cube_pos = cube_pos - env_origins
            
            # Create pose matrix
            cube_pose_matrix = PoseUtils.make_pose(
                cube_pos,
                matrix_from_quat(cube_quat)
            )
            
            return {"cube": cube_pose_matrix}
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to get object poses")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            raise

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals."""
        try:
            if env_ids is None:
                env_ids = slice(None)
            
            signals = {}
            
            if hasattr(self.cfg, 'subtask_configs'):
                cube = self.scene["cube"]
                robot = self.scene["robot"]
                
                cube_z = cube.data.root_pos_w[env_ids, 2]
                gripper_pos = robot.data.joint_pos[env_ids, -1]
                
                initial_cube_height = 0.78
                pick_threshold = 0.05
                gripper_closed_threshold = 0.02
                
                cube_picked = (cube_z > initial_cube_height + pick_threshold) & (gripper_pos < gripper_closed_threshold)
                signals["pick_cube"] = cube_picked
                
                print(f"[DEBUG] Subtask signals computed:")
                print(f"  - Cube height: {cube_z.mean().item():.3f}")
                print(f"  - Gripper position: {gripper_pos.mean().item():.3f}")
                print(f"  - Cube picked: {cube_picked.sum().item()}/{len(cube_picked)}")
            
            return signals
            
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to get subtask signals")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            return {}
    
    def serialize(self):
        """Save environment info for re-instantiation."""
        try:
            return dict(env_name=self.spec.id, type=2, env_kwargs=dict())
        except Exception as e:
            carb.log_error(f"[ERROR] Failed to serialize environment")
            carb.log_error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            return dict(env_name="PickOrangeMimicEnv", type=2, env_kwargs=dict())