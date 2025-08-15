"""Pick Orange Mimic Environment with correct LULA solver implementation."""

from isaaclab.envs import ManagerBasedRLMimicEnv
import isaaclab.utils.math as PoseUtils
import torch
import gymnasium as gym
from typing import Dict, Any, Optional, Sequence
import numpy as np
import os

# Enable extensions before importing motion generation
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")

from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader

import carb

class PickOrangeMimicEnv(ManagerBasedRLMimicEnv):
    """Pick Orange environment with Mimic support for annotation."""
    
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        robot = self.scene["robot"]
        gripper_body_id = robot.find_bodies("gripper")[0]
        ee_pos_world = robot.data.body_pos_w[0, gripper_body_id].cpu().numpy()
        ee_pos_relative = ee_pos_world - self.scene.env_origins[0].cpu().numpy()
        
        self.isaac_ee_home = ee_pos_relative.flatten()
        
        self.articulation_solvers = {}
        self.lula_solver = None
        
        self._setup_lula_ik()
        
    def _setup_lula_ik(self):
        """Initialize LULA IK solver following Isaac Sim documentation pattern."""
        urdf_path = os.path.join(
            os.path.dirname(__file__), 
            "robot.urdf"
        )
        
        yaml_path = os.path.join(
            os.path.dirname(__file__), 
            "lulu.yaml"
        )
        
        if not os.path.exists(yaml_path):
            self._create_basic_descriptor(urdf_path, yaml_path)
        
        try:
            self.lula_solver = LulaKinematicsSolver(
                robot_description_path=yaml_path,
                urdf_path=urdf_path
            )
            
            robot = self.scene["robot"]
            
            self.articulation_solver = ArticulationKinematicsSolver(
                robot,
                self.lula_solver,
                "gripper"
            )
            
            self.use_lula = True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize LULA: {e}")
            self.use_lula = False
    

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EE pose to joint positions using LULA IK."""
        eef_name = "eef"
        
        target_eef_pose = target_eef_pose_dict[eef_name]
        target_pos_raw, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        
        if target_pos_raw.dim() > 1:
            target_pos_raw = target_pos_raw.squeeze(0)
        
        env_origin = self.scene.env_origins[env_id]
        
        robot = self.scene["robot"]
        
        robot_base_position_world = env_origin.cpu().numpy()
        robot_base_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        
        print(f"\n[DEBUG - Env {env_id}]")
        print(f"  Robot base (world): {robot_base_position_world}")
        
        target_pos_world = target_pos_raw.cpu().numpy() + robot_base_position_world
        
        print(f"  Target (relative): {target_pos_raw.cpu().numpy()}")
        print(f"  Target (world): {target_pos_world}")
        
        current_joints = robot.data.joint_pos[env_id, :5].cpu().numpy()
        
        if target_rot.dim() == 2:
            target_quat = PoseUtils.quat_from_matrix(target_rot.unsqueeze(0))[0]
        else:
            target_quat = PoseUtils.quat_from_matrix(target_rot)[0] if target_rot.shape[0] == 1 else PoseUtils.quat_from_matrix(target_rot)
        
        if target_quat.dim() > 0:
            target_quat = target_quat.reshape(4)
        target_quat_np = target_quat.cpu().numpy()
        
        if hasattr(self, 'use_lula') and self.use_lula:
            solution = self._solve_lula_ik(
                target_pos_world, 
                target_quat_np, 
                robot_base_position_world,
                robot_base_orientation,
                current_joints
            )
        else:
            solution = self._solve_simple_ik(target_pos_raw.cpu().numpy(), current_joints)
        
        if solution is None:
            solution = current_joints
        
        target_joints = torch.tensor(solution, device=self.device, dtype=torch.float32)
        
        gripper_action = gripper_action_dict[eef_name]
        action = torch.cat([target_joints, gripper_action], dim=0)
        
        return action


    def _solve_lula_ik(self, target_pos_world, target_quat, robot_base_pos, robot_base_quat, current_joints=None):
        """
        Solve IK using LULA following Isaac Sim documentation pattern.
        """
        try:
            print(f"  LULA target (world): {target_pos_world}")
            
            self.lula_solver.set_robot_base_pose(
                robot_base_pos.tolist(),
                robot_base_quat.tolist()
            )
            
            if current_joints is not None:
                self.articulation_solver.set_joint_positions(current_joints.tolist() if hasattr(current_joints, 'tolist') else list(current_joints))
            
            success, joint_positions = self.articulation_solver.compute_inverse_kinematics(
                target_position=target_pos_world.tolist(),
                target_orientation=target_quat.tolist()
            )
            
            if success:
                print(f"  IK solution: {joint_positions[:5]}")
                return joint_positions[:5]
            else:
                carb.log_warn("IK did not converge to a solution")
                
                target_pos_relative = target_pos_world - robot_base_pos
                
                self.lula_solver.set_robot_base_pose(
                    [0, 0, 0],
                    [1, 0, 0, 0]
                )
                
                success_retry, joint_positions_retry = self.articulation_solver.compute_inverse_kinematics(
                    target_position=target_pos_relative.tolist(),
                    target_orientation=target_quat.tolist()
                )
                
                if success_retry:
                    print(f"  IK solution (relative): {joint_positions_retry[:5]}")
                    return joint_positions_retry[:5]
                else:
                    return current_joints
                
        except Exception as e:
            carb.log_error(f"LULA IK Error: {e}")
            return current_joints
            
    def _solve_simple_ik(self, target_pos, current_joints):
        """Simple fallback IK solver."""
        return current_joints

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose as 4x4 transformation matrix."""
        robot = self.scene["robot"]
        gripper_body_id = robot.find_bodies("gripper")[0]
        
        if env_ids is None:
            env_ids = slice(None)
        
        gripper_pos = robot.data.body_pos_w[env_ids, gripper_body_id]
        gripper_quat = robot.data.body_quat_w[env_ids, gripper_body_id]
        
        env_origins = self.scene.env_origins[env_ids]
        gripper_pos = gripper_pos - env_origins
        
        pose_matrix = PoseUtils.make_pose(
            gripper_pos, 
            PoseUtils.matrix_from_quat(gripper_quat)
        )
        
        return pose_matrix 
    
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert joint position action to target EE pose using FK."""
        eef_name = "eef"
        
        robot = self.scene["robot"]
        
        current_joint_pos = robot.data.joint_pos.clone()
        
        target_joint_pos = current_joint_pos.clone()
        target_joint_pos[:, :5] = action[:, :5]
        
        original_joint_pos = current_joint_pos.clone()
        robot.data.joint_pos[:] = target_joint_pos
        
        robot.update(dt=0.0)
        
        target_ee_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        
        robot.data.joint_pos[:] = original_joint_pos
        robot.update(dt=0.0)
        
        return {eef_name: target_ee_pose}
    
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from action sequence."""
        if actions.dim() == 3:
            gripper_actions = actions[:, :, -1:]
        elif actions.dim() == 2:
            gripper_actions = actions[:, -1:]
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}")
    
        return {"eef": gripper_actions}
    
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Get object poses as 4x4 transformation matrices."""
        if env_ids is None:
            env_ids = slice(None)
        
        cube = self.scene["cube"]
        cube_pos = cube.data.root_pos_w[env_ids]
        cube_quat = cube.data.root_quat_w[env_ids]
        
        env_origins = self.scene.env_origins[env_ids]
        cube_pos = cube_pos - env_origins
        
        cube_pose_matrix = PoseUtils.make_pose(
            cube_pos,
            PoseUtils.matrix_from_quat(cube_quat)
        )
        
        return {"cube": cube_pose_matrix}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals."""
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
        
        return signals
    
    def serialize(self):
        """Save environment info for re-instantiation."""
        return dict(env_name=self.spec.id, type=2, env_kwargs=dict())