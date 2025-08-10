"""Pick Orange Mimic Environment for annotation support."""

from isaaclab.envs import ManagerBasedRLMimicEnv
import isaaclab.utils.math as PoseUtils
import torch
import gymnasium as gym
from typing import Dict, Any, Optional, Sequence
import pybullet as p
import numpy as np
import os

class PickOrangeMimicEnv(ManagerBasedRLMimicEnv):
    """Pick Orange environment with Mimic support for annotation."""
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get Isaac home EE position for coordinate transformation
        robot = self.scene["robot"]
        gripper_body_id = robot.find_bodies("gripper")[0]
        ee_pos_world = robot.data.body_pos_w[0, gripper_body_id].cpu().numpy()
        ee_pos_relative = ee_pos_world - self.scene.env_origins[0].cpu().numpy()
        
        # IMPORTANT: Flatten to 1D array
        self.isaac_ee_home = ee_pos_relative.flatten()
        
        print(f"[INFO] Isaac EE home position (relative): {self.isaac_ee_home}")
        
        # Initialize PyBullet IK solver
        self._setup_pybullet_ik()
        
    def _setup_pybullet_ik(self):
        """Initialize PyBullet IK solver."""
        # Connect to PyBullet in DIRECT mode (no GUI)
        self.pb_client = p.connect(p.DIRECT)
        
        # Find the URDF file - adjust this path to your actual URDF location
        urdf_path = os.path.join(
            os.path.dirname(__file__), 
            "robot.urdf"  # Adjust this path
        )
        
        if not os.path.exists(urdf_path):
            print(f"[WARNING] URDF not found at {urdf_path}, trying alternative paths...")
            # Try other possible locations
            possible_paths = [
                "./robot.urdf",
                "../../../robot.urdf",
                "/home/ubuntu/Desktop/isaaclab_projects/lighthouse/leisaac/source/leisaac/leisaac/tasks/pick_orange/robot.urdf"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    urdf_path = path
                    break
        
        print(f"[INFO] Loading URDF from: {urdf_path}")
        
        # Load robot in PyBullet
        self.pb_robot = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.pb_client
        )
        
        # Identify arm joints (first 5 revolute joints, excluding jaw)
        self.pb_arm_joint_ids = []
        num_joints = p.getNumJoints(self.pb_robot, physicsClientId=self.pb_client)
        
        for i in range(num_joints):
            info = p.getJointInfo(self.pb_robot, i, physicsClientId=self.pb_client)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            if joint_type == p.JOINT_REVOLUTE and 'jaw' not in joint_name.lower():
                self.pb_arm_joint_ids.append(i)
                
            if len(self.pb_arm_joint_ids) >= 5:
                break
        
        # End-effector link ID (Wrist_Roll's child)
        self.pb_ee_link_id = 4
        
        print(f"[INFO] PyBullet IK initialized with arm joints: {self.pb_arm_joint_ids}")
        

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EE pose to joint positions - TESTING COORDINATE SYSTEMS."""
        eef_name = "eef"
        
        # Get target pose from MimicGen
        target_eef_pose = target_eef_pose_dict[eef_name]
        target_pos_raw, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        
        if target_pos_raw.dim() > 1:
            target_pos_raw = target_pos_raw.squeeze(0)
        
        # TEST: Is it already relative or is it world coordinates?
        env_origin = self.scene.env_origins[env_id]
        
        print(f"\n[COORDINATE TEST]")
        print(f"  Env {env_id} origin: {env_origin.cpu().numpy()}")
        print(f"  Target from MimicGen: {target_pos_raw.cpu().numpy()}")
        
        # Option 1: Assume it's already relative (DON'T subtract origin)
        target_pos_relative = target_pos_raw
        
        # Option 2: Assume it's world (DO subtract origin) 
        target_pos_world_adjusted = target_pos_raw - env_origin
        
        # Get current pose (already relative)
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, _ = PoseUtils.unmake_pose(curr_pose)
        if curr_pos.dim() > 1:
            curr_pos = curr_pos.squeeze(0)
        
        print(f"  Current EE (relative): {curr_pos.cpu().numpy()}")
        
        # Check which makes more sense
        movement_if_relative = target_pos_relative - curr_pos
        movement_if_world = target_pos_world_adjusted - curr_pos
        
        dist_if_relative = torch.norm(movement_if_relative).item()
        dist_if_world = torch.norm(movement_if_world).item()
        
        print(f"\n  If target is RELATIVE:")
        print(f"    Movement: {movement_if_relative.cpu().numpy()}")
        print(f"    Distance: {dist_if_relative*1000:.1f}mm")
        
        print(f"\n  If target is WORLD:")
        print(f"    Movement: {movement_if_world.cpu().numpy()}")  
        print(f"    Distance: {dist_if_world*1000:.1f}mm")
        
        # USE THE ONE THAT MAKES SENSE (smaller movement)
        if dist_if_relative < 1.0:  # Less than 1 meter
            print(f"\n  -> Using RELATIVE interpretation (makes more sense)")
            target_pos = target_pos_relative
        else:
            print(f"\n  -> Movement too large, something is wrong!")
            # Try using relative anyway
            target_pos = target_pos_relative
        
        # Continue with IK
        robot = self.scene["robot"]
        current_joints = robot.data.joint_pos[env_id, :5].cpu().numpy()
        
        target_pos_np = target_pos.cpu().numpy()
        target_pos_list = target_pos_np.tolist()
        
        # Get quaternion
        if target_rot.dim() == 2:
            target_quat = PoseUtils.quat_from_matrix(target_rot.unsqueeze(0))[0]
        else:
            target_quat = PoseUtils.quat_from_matrix(target_rot)[0] if target_rot.shape[0] == 1 else PoseUtils.quat_from_matrix(target_rot)
        
        if target_quat.dim() > 0:
            target_quat = target_quat.reshape(4)
        target_quat_np = target_quat.cpu().numpy()
        pb_quat = [float(target_quat_np[1]), float(target_quat_np[2]), 
                float(target_quat_np[3]), float(target_quat_np[0])]
        
        # Solve IK
        solution = self._solve_pybullet_ik(target_pos_list, pb_quat, current_joints)
        
        if solution is None:
            print("[IK] Failed - returning current joints")
            solution = current_joints
        
        target_joints = torch.tensor(solution, device=self.device, dtype=torch.float32)
        
        # Add gripper
        gripper_action = gripper_action_dict[eef_name]
        action = torch.cat([target_joints, gripper_action], dim=0)
        
        return action


    def _solve_pybullet_ik(self, target_pos, target_quat, current_joints=None):
        """Solve IK using PyBullet - POSITION ONLY VERSION."""
        
        # Ensure target_pos is 1D numpy array
        if isinstance(target_pos, list):
            target_pos = np.array(target_pos)
        if target_pos.ndim > 1:
            target_pos = target_pos.flatten()
        
        # Get coordinate systems
        isaac_ee_home = self.isaac_ee_home
        pybullet_ee_home = np.array([0.02, -0.30, 0.27])
        
        # Calculate offset from home in Isaac space
        offset_from_home = target_pos - isaac_ee_home
        
        # Apply transformation (direct mapping for now)
        target_pos_pb = pybullet_ee_home + offset_from_home
        
        print(f"[IK] Target: Isaac={target_pos}, PyBullet={target_pos_pb}")
        print(f"[IK] Movement: {np.linalg.norm(offset_from_home)*1000:.1f}mm")
        
        # Set current joint state if provided
        if current_joints is not None:
            for i, joint_pos in enumerate(current_joints[:5]):
                p.resetJointState(self.pb_robot, self.pb_arm_joint_ids[i], 
                                joint_pos, physicsClientId=self.pb_client)
        
        # Get current EE orientation to maintain it
        ee_state = p.getLinkState(self.pb_robot, self.pb_ee_link_id, 
                                physicsClientId=self.pb_client)
        current_quat = ee_state[1]  # Use current orientation instead of target
        
        # Try IK with position only (ignoring target orientation)
        best_solution = None
        best_error = float('inf')
        
        param_sets = [
            {'maxNumIterations': 100, 'residualThreshold': 0.001},
            {'maxNumIterations': 500, 'residualThreshold': 0.0001},
        ]
        
        for params in param_sets:
            target_pos_list = target_pos_pb.tolist() if isinstance(target_pos_pb, np.ndarray) else list(target_pos_pb)
            
            # Use CURRENT orientation, not target
            ik_solution = p.calculateInverseKinematics(
                self.pb_robot,
                self.pb_ee_link_id,
                target_pos_list,
                current_quat,  # IGNORING target_quat, using current
                **params,
                physicsClientId=self.pb_client
            )
            
            arm_solution = [ik_solution[i] for i in self.pb_arm_joint_ids]
            
            # Verify solution
            for i, joint_id in enumerate(self.pb_arm_joint_ids):
                p.resetJointState(self.pb_robot, joint_id, arm_solution[i], 
                                physicsClientId=self.pb_client)
            
            ee_state = p.getLinkState(self.pb_robot, self.pb_ee_link_id, 
                                    physicsClientId=self.pb_client)
            actual_pos = np.array(ee_state[0])
            error = np.linalg.norm(target_pos_pb - actual_pos)
            
            if error < best_error:
                best_error = error
                best_solution = arm_solution
                
            if error < 0.005:  # 5mm threshold
                print(f"[SUCCESS] IK converged with {best_error*1000:.1f}mm error")
                return np.array(arm_solution)
        
        # Return best solution if reasonable
        if best_error < 0.01:  # 10mm threshold
            print(f"[SUCCESS] Position-only IK with {best_error*1000:.1f}mm error")
            return np.array(best_solution)
        
        print(f"[FAILED] IK failed with {best_error:.3f}m error")
        return None

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose as 4x4 transformation matrix."""
        robot = self.scene["robot"]
        gripper_body_id = robot.find_bodies("gripper")[0]
        
        if env_ids is None:
            env_ids = slice(None)
        
        # Get position and quaternion
        gripper_pos = robot.data.body_pos_w[env_ids, gripper_body_id]  # (N, 3)
        gripper_quat = robot.data.body_quat_w[env_ids, gripper_body_id]  # (N, 4)
        
        # IMPORTANT: Subtract environment origins to get relative positions!
        env_origins = self.scene.env_origins[env_ids]
        gripper_pos = gripper_pos - env_origins
        
        # Convert to 4x4 transformation matrix
        pose_matrix = PoseUtils.make_pose(
            gripper_pos, 
            PoseUtils.matrix_from_quat(gripper_quat)
        )
        
        return pose_matrix 
    
    ##### DEBUG
    # def target_eef_pose_to_action(
    #     self,
    #     target_eef_pose_dict: dict,
    #     gripper_action_dict: dict,
    #     action_noise_dict: dict | None = None,
    #     env_id: int = 0,
    # ) -> torch.Tensor:
    #     """TESTING: Skip IK and just send fixed joint positions."""
        
    #     # Ignore the target_eef_pose completely for now
    #     # Just create some joint positions that should make the robot move
        
    #     # Get current joint positions to see what they are
    #     robot = self.scene["robot"]
    #     current_joints = robot.data.joint_pos[env_id, :5]
    #     print(f"[DEBUG] Current joints: {current_joints.cpu().numpy()}")
        
    #     # Create a simple test action - move first joint by 0.5 radians
    #     test_joints = current_joints.clone()
    #     test_joints[0] += 0.5  # Move shoulder_pan
    #     test_joints[1] += 0.2  # Move shoulder_lift a bit
        
    #     # Add gripper (keep it open for now)
    #     action = torch.cat([test_joints, torch.tensor([0.0], device=self.device)], dim=0)
        
    #     print(f"[DEBUG] Sending test action: {action.cpu().numpy()}")
        
    #     return action

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
        """Get object poses as 4x4 transformation matrices."""
        if env_ids is None:
            env_ids = slice(None)
        
        # Get cube pose
        cube = self.scene["cube"]
        cube_pos = cube.data.root_pos_w[env_ids]  # (N, 3)
        cube_quat = cube.data.root_quat_w[env_ids]  # (N, 4)
        
        # Subtract environment origins for relative positions!
        env_origins = self.scene.env_origins[env_ids]
        cube_pos = cube_pos - env_origins
        
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
