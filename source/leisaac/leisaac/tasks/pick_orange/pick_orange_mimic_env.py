# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pick Orange Mimic Environment for annotation support."""

from isaaclab.envs import ManagerBasedRLMimicEnv
import torch
import gymnasium as gym
from typing import Dict, Any

class PickOrangeMimicEnv(ManagerBasedRLMimicEnv):
    """Pick Orange environment with Mimic support for annotation."""
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
    def get_robot_eef_pose(self, eef_name: str = "eef"):
        """Get the end-effector pose (position + quaternion)."""
        # Get the gripper/end-effector body from the robot
        robot = self.scene["robot"]
        
        # Get the gripper link pose (assuming gripper is the end-effector)
        # You may need to adjust the body name based on your robot structure
        gripper_body_id = robot.find_bodies("gripper")[0]
        
        # Get position and orientation
        gripper_pos = robot.data.body_pos_w[:, gripper_body_id]  # (num_envs, 3)
        gripper_quat = robot.data.body_quat_w[:, gripper_body_id]  # (num_envs, 4) 
        
        # Concatenate position and quaternion
        eef_pose = torch.cat([gripper_pos, gripper_quat], dim=-1)  # (num_envs, 7)
        
        return eef_pose
    
    def get_object_poses(self):
        """Get poses of manipulated objects (the cube)."""
        # Get cube position and orientation
        cube = self.scene["cube"]
        cube_pos = cube.data.root_pos_w  # (num_envs, 3)
        cube_quat = cube.data.root_quat_w  # (num_envs, 4)
        
        # Return as a dictionary or concatenated tensor
        object_poses = {
            "cube": torch.cat([cube_pos, cube_quat], dim=-1)  # (num_envs, 7)
        }
        
        return object_poses
    
    def action_to_target_eef_pose(self, action):
        """Convert action to target end-effector pose.
        
        For joint position control, this would require forward kinematics.
        For now, return current EE pose as a placeholder.
        """
        # For accurate implementation, you'd need to:
        # 1. Get current joint positions
        # 2. Add action deltas to joints
        # 3. Compute forward kinematics to get target EE pose
        
        # Placeholder: return current EE pose
        return self.get_robot_eef_pose("eef")
    
    def get_subtask_term_signals(self):
        """Check if subtasks are completed.
        
        For manual mode, this won't be called.
        For auto mode, implement your subtask detection logic here.
        """
        signals = {}
        
        if hasattr(self.cfg, 'subtask_configs'):
            # Check if cube is picked (gripper closed and cube is elevated)
            cube = self.scene["cube"]
            robot = self.scene["robot"]
            
            # Get cube height
            cube_z = cube.data.root_pos_w[:, 2]  # z-position
            
            # Get gripper joint position (assuming last joint is gripper)
            gripper_pos = robot.data.joint_pos[:, -1]
            
            # Simple heuristic: cube is picked if it's above initial height and gripper is closed
            initial_cube_height = 0.78  # from your config
            pick_threshold = 0.05  # 5cm above initial position
            gripper_closed_threshold = 0.02  # adjust based on your gripper
            
            cube_picked = (cube_z > initial_cube_height + pick_threshold) & (gripper_pos < gripper_closed_threshold)
            
            signals["pick_cube"] = cube_picked
            
            # Task complete is handled by your termination condition
            signals["task_complete"] = torch.zeros_like(cube_picked, dtype=torch.bool)
        
        return signals

