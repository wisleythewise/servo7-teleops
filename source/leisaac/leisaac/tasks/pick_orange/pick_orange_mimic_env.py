"""Pick Orange Mimic Environment for annotation support."""

from isaaclab.envs import ManagerBasedRLMimicEnv
import isaaclab.utils.math as PoseUtils
import torch
import gymnasium as gym
from typing import Dict, Any, Optional, Sequence

class PickOrangeMimicEnv(ManagerBasedRLMimicEnv):
    """Pick Orange environment with Mimic support for annotation."""
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose as 4x4 transformation matrix.
        
        Args:
            eef_name: Name of the end effector (e.g., "eef")
            env_ids: Environment indices. If None, all envs are considered.
            
        Returns:
            EE pose matrices of shape (len(env_ids), 4, 4)
        """
        robot = self.scene["robot"]
        gripper_body_id = robot.find_bodies("gripper")[0]
        
        if env_ids is None:
            env_ids = slice(None)
        
        # Get position and quaternion
        gripper_pos = robot.data.body_pos_w[env_ids, gripper_body_id]  # (N, 3)
        gripper_quat = robot.data.body_quat_w[env_ids, gripper_body_id]  # (N, 4)
        
        # Convert to 4x4 transformation matrix
        pose_matrix = PoseUtils.make_pose(
            gripper_pos, 
            PoseUtils.matrix_from_quat(gripper_quat)
        )
        
        return pose_matrix  # Shape: (N, 4, 4)
    
    def compute_jacobian_based_ik(self, target_ee_pos, target_ee_rot, env_ids=None):
        """Compute joint positions using Jacobian-based inverse kinematics.
        
        Args:
            target_ee_pos: Target end-effector position (N, 3)
            target_ee_rot: Target end-effector rotation matrix (N, 3, 3)
            env_ids: Environment indices
            
        Returns:
            Joint positions for arm joints (N, 5)
        """
        if env_ids is None:
            env_ids = slice(None)
            
        robot = self.scene["robot"]
        
        # Get current joint positions (excluding gripper)
        current_joint_pos = robot.data.joint_pos[env_ids, :5].clone()  # First 5 joints are arm
        
        # Get current EE pose
        current_ee_pose = self.get_robot_eef_pose("eef", env_ids=env_ids)
        current_ee_pos, current_ee_rot = PoseUtils.unmake_pose(current_ee_pose)
        
        # Compute position error
        pos_error = target_ee_pos - current_ee_pos  # (N, 3)
        
        # Compute orientation error using axis-angle
        rot_error_mat = torch.matmul(target_ee_rot, current_ee_rot.transpose(-1, -2))
        rot_error_quat = PoseUtils.quat_from_matrix(rot_error_mat)
        rot_error_axis_angle = PoseUtils.axis_angle_from_quat(rot_error_quat)  # (N, 3)
        
        # Combine errors into task-space velocity
        ee_error = torch.cat([pos_error, rot_error_axis_angle], dim=-1)  # (N, 6)
        
        # Get Jacobian - this is robot-specific
        # For SO101 5-DOF arm, we need the geometric Jacobian
        gripper_body_id = robot.find_bodies("gripper")[0]
        
        # Get jacobian from IsaacLab (check exact API for your version)
        # This might be stored as robot.data.jacobian or need to be computed
        try:
            # Try to get pre-computed Jacobian
            jacobian = robot.data.body_jacobian_w[env_ids, gripper_body_id, :, :5]  # (N, 6, 5)
        except:
            # If not available, compute it manually (simplified version)
            jacobian = self._compute_geometric_jacobian(robot, env_ids)
        
        # Damped least squares (more stable than pseudo-inverse)
        lambda_dls = 0.01  # Damping factor
        batch_size = jacobian.shape[0]
        
        joint_delta = torch.zeros((batch_size, 5), device=self.device)
        
        for i in range(batch_size):
            J = jacobian[i]  # (6, 5)
            JJT = torch.matmul(J, J.T)  # (6, 6)
            JJT_damped = JJT + lambda_dls * torch.eye(6, device=self.device)
            
            # Solve for joint velocities
            try:
                J_damped_inv = torch.matmul(J.T, torch.linalg.inv(JJT_damped))
                joint_delta[i] = torch.matmul(J_damped_inv, ee_error[i])
            except:
                # If inversion fails, use small random perturbation
                joint_delta[i] = 0.01 * torch.randn(5, device=self.device)
        
        # Scale down the delta for stability
        joint_delta = joint_delta * 0.1
        
        # Apply joint limits
        target_joint_pos = current_joint_pos + joint_delta
        
        # Clamp to joint limits (you should set these based on your robot)
        joint_limits_lower = torch.tensor([-3.14, -3.14, -3.14, -3.14, -3.14], device=self.device)
        joint_limits_upper = torch.tensor([3.14, 3.14, 3.14, 3.14, 3.14], device=self.device)
        
        target_joint_pos = torch.clamp(target_joint_pos, joint_limits_lower, joint_limits_upper)
        
        return target_joint_pos
    
    def _compute_geometric_jacobian(self, robot, env_ids):
        """Compute geometric Jacobian manually if not available.
        
        Simplified version - you should replace with actual FK computation.
        """
        batch_size = len(env_ids) if isinstance(env_ids, list) else robot.num_instances
        
        # Create a dummy Jacobian (you need to implement actual computation)
        # For a 5-DOF arm mapping to 6-DOF task space
        jacobian = torch.zeros((batch_size, 6, 5), device=self.device)
        
        # Fill with reasonable values for testing
        for i in range(batch_size):
            # Position part (top 3 rows)
            jacobian[i, 0:3, :] = 0.1 * torch.randn(3, 5, device=self.device)
            # Orientation part (bottom 3 rows)  
            jacobian[i, 3:6, :] = 0.05 * torch.randn(3, 5, device=self.device)
            
        return jacobian
    
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EE pose to joint position action using IK.
        
        Args:
            target_eef_pose_dict: Dict with eef_name -> 4x4 target pose matrix
            gripper_action_dict: Dict with eef_name -> gripper action
            action_noise_dict: Optional noise parameters
            env_id: Environment index
            
        Returns:
            Action tensor compatible with env.step() - joint positions
        """
        eef_name = "eef"
        
        # Get target pose
        target_eef_pose = target_eef_pose_dict[eef_name]
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        
        # Compute IK to get joint positions
        target_joint_pos = self.compute_jacobian_based_ik(
            target_pos.unsqueeze(0), 
            target_rot.unsqueeze(0), 
            env_ids=[env_id]
        )[0]  # Get first (and only) result
        
        # Get gripper action
        gripper_action = gripper_action_dict[eef_name]
        
        # Add noise if specified
        if action_noise_dict is not None and eef_name in action_noise_dict:
            noise_scale = action_noise_dict[eef_name]
            joint_noise = noise_scale * torch.randn_like(target_joint_pos)
            target_joint_pos = target_joint_pos + joint_noise
            target_joint_pos = torch.clamp(target_joint_pos, -3.14, 3.14)
        
        # Combine arm joints with gripper
        action = torch.cat([target_joint_pos, gripper_action], dim=0)
        
        return action
    
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert joint position action to target EE pose using FK.
        
        Args:
            action: Environment action of shape (num_envs, action_dim)
            
        Returns:
            Dictionary mapping eef_name to target pose matrices (4x4)
        """
        eef_name = "eef"
        
        # Action is [5 arm joints, 1 gripper]
        # We need to compute forward kinematics
        
        robot = self.scene["robot"]
        
        # Get current joint positions
        current_joint_pos = robot.data.joint_pos.clone()
        
        # Create target joint positions by replacing arm joints
        target_joint_pos = current_joint_pos.clone()
        target_joint_pos[:, :5] = action[:, :5]  # Update arm joints
        
        # Temporarily set joint positions to compute FK
        # Note: This is a workaround - ideally you'd compute FK analytically
        original_joint_pos = current_joint_pos.clone()
        robot.data.joint_pos[:] = target_joint_pos
        
        # Update kinematic chain
        robot.update(dt=0.0)
        
        # Get resulting EE pose
        target_ee_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        
        # Restore original joint positions
        robot.data.joint_pos[:] = original_joint_pos
        robot.update(dt=0.0)
        
        return {eef_name: target_ee_pose}
    
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from action sequence.
        
        Args:
            actions: Actions of shape (num_envs, num_steps, action_dim) or (num_envs, action_dim)
            
        Returns:
            Dictionary mapping eef_name to gripper actions
        """
        # Handle both 2D and 3D tensors
        if actions.dim() == 3:
            # (num_envs, num_steps, action_dim)
            gripper_actions = actions[:, :, -1:]  # Keep dimension: (N, T, 1)
        elif actions.dim() == 2:
            # (num_envs, action_dim)
            gripper_actions = actions[:, -1:]  # Keep dimension: (N, 1)
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}")
    
        return {"eef": gripper_actions}
    
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Get object poses as 4x4 transformation matrices.
        
        Args:
            env_ids: Environment indices. If None, all envs.
            
        Returns:
            Dictionary mapping object names to 4x4 pose matrices
        """
        if env_ids is None:
            env_ids = slice(None)
        
        # Get cube pose
        cube = self.scene["cube"]
        cube_pos = cube.data.root_pos_w[env_ids]  # (N, 3)
        cube_quat = cube.data.root_quat_w[env_ids]  # (N, 4)
        
        # Convert to 4x4 matrix
        cube_pose_matrix = PoseUtils.make_pose(
            cube_pos,
            PoseUtils.matrix_from_quat(cube_quat)
        )
        
        return {"cube": cube_pose_matrix}
    
    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals.
        
        Args:
            env_ids: Environment indices. If None, all envs.
            
        Returns:
            Dictionary of boolean tensors for subtask completion
        """
        if env_ids is None:
            env_ids = slice(None)
        
        signals = {}
        
        if hasattr(self.cfg, 'subtask_configs'):
            cube = self.scene["cube"]
            robot = self.scene["robot"]
            
            cube_z = cube.data.root_pos_w[env_ids, 2]
            gripper_pos = robot.data.joint_pos[env_ids, -1]
            
            # Check if cube is picked
            initial_cube_height = 0.78
            pick_threshold = 0.05
            gripper_closed_threshold = 0.02
            
            cube_picked = (cube_z > initial_cube_height + pick_threshold) & (gripper_pos < gripper_closed_threshold)
            signals["pick_cube"] = cube_picked
        
        return signals
    
    def serialize(self):
        """Save environment info for re-instantiation."""
        return dict(env_name=self.spec.id, type=2, env_kwargs=dict())
