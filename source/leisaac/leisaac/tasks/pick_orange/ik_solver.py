#!/usr/bin/env python3
"""
Fixed PyBullet IK solver for SO101 robot with proper workspace understanding.
"""

import pybullet as p
import numpy as np
import time
import argparse
from pathlib import Path

class SO101IKSolver:
    def __init__(self, urdf_path, visualize=False):
        """Initialize the IK solver for use in IsaacLab."""
        
        # Connect to PyBullet (DIRECT mode for use in IsaacLab)
        self.pb_client = p.connect(p.GUI if visualize else p.DIRECT)
        
        # Load the robot
        self.robot_id = p.loadURDF(
            str(urdf_path),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.pb_client
        )
        
        # Identify joints
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.pb_client)
        self.arm_joint_ids = []
        
        print("\n[INFO] Robot configuration:")
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.pb_client)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # First 5 revolute joints (excluding jaw)
            if joint_type == p.JOINT_REVOLUTE and 'jaw' not in joint_name.lower():
                self.arm_joint_ids.append(i)
                print(f"  Arm joint {len(self.arm_joint_ids)-1}: {joint_name} (ID: {i})")
            
            if len(self.arm_joint_ids) >= 5:
                break
        
        # End-effector is the gripper link (child of Wrist_Roll)
        self.ee_link_id = 4  # Wrist_Roll's child link
        print(f"[INFO] End-effector link ID: {self.ee_link_id}")
        
        # Calibrate the robot's actual workspace
        self._calibrate_workspace()
    
    def _calibrate_workspace(self):
        """Understand the robot's actual workspace and offsets."""
        # Set to home position
        for joint_id in self.arm_joint_ids:
            p.resetJointState(self.robot_id, joint_id, 0.0, physicsClientId=self.pb_client)
        
        # Get EE position at home
        ee_state = p.getLinkState(self.robot_id, self.ee_link_id, physicsClientId=self.pb_client)
        self.home_ee_pos = np.array(ee_state[0])
        
        print(f"\n[INFO] Calibration:")
        print(f"  Home EE position: {self.home_ee_pos}")
        
        # Test reachable positions
        test_configs = [
            ([0, 0.5, 0, 0, 0], "Shoulder lifted"),
            ([0, -0.5, 0, 0, 0], "Shoulder lowered"),
            ([0.5, 0, 0, 0, 0], "Base rotated"),
            ([0, 0, 0.5, 0, 0], "Elbow bent"),
        ]
        
        reach_points = []
        for config, name in test_configs:
            for i, angle in enumerate(config):
                p.resetJointState(self.robot_id, self.arm_joint_ids[i], angle, physicsClientId=self.pb_client)
            ee_state = p.getLinkState(self.robot_id, self.ee_link_id, physicsClientId=self.pb_client)
            pos = np.array(ee_state[0])
            reach_points.append(pos)
            print(f"  {name}: {pos}")
        
        # Estimate workspace center and size
        reach_points = np.array(reach_points)
        self.workspace_center = np.mean(reach_points, axis=0)
        self.workspace_radius = np.max(np.linalg.norm(reach_points - self.workspace_center, axis=1))
        
        print(f"\n[INFO] Workspace analysis:")
        print(f"  Center: {self.workspace_center}")
        print(f"  Radius: {self.workspace_radius:.3f}m")
    
    def solve_ik(self, target_pos, target_quat=[0, 0, 0, 1], current_joints=None):
        """Solve IK for target pose.
        
        Args:
            target_pos: Target position [x, y, z] in meters
            target_quat: Target quaternion [x, y, z, w] (PyBullet format)
            current_joints: Current joint positions (optional)
            
        Returns:
            Joint angles for the 5 arm joints, or None if failed
        """
        # Set current joint state if provided
        if current_joints is not None:
            for i, joint_pos in enumerate(current_joints[:5]):
                p.resetJointState(self.robot_id, self.arm_joint_ids[i], joint_pos, physicsClientId=self.pb_client)
        
        # Try IK with multiple attempts and parameters
        best_solution = None
        best_error = float('inf')
        
        # Try different IK parameters
        param_sets = [
            {'maxNumIterations': 100, 'residualThreshold': 0.001},
            {'maxNumIterations': 200, 'residualThreshold': 0.0001},
            {'maxNumIterations': 500, 'residualThreshold': 0.00001},
        ]
        
        for params in param_sets:
            ik_solution = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_id,
                target_pos,
                target_quat,
                **params,
                physicsClientId=self.pb_client
            )
            
            # Extract arm joint solutions
            arm_solution = [ik_solution[i] for i in self.arm_joint_ids]
            
            # Verify solution
            for i, joint_id in enumerate(self.arm_joint_ids):
                p.resetJointState(self.robot_id, joint_id, arm_solution[i], physicsClientId=self.pb_client)
            
            ee_state = p.getLinkState(self.robot_id, self.ee_link_id, physicsClientId=self.pb_client)
            actual_pos = np.array(ee_state[0])
            error = np.linalg.norm(target_pos - actual_pos)
            
            if error < best_error:
                best_error = error
                best_solution = arm_solution
                
            # Good enough?
            if error < 0.005:  # 5mm threshold
                return arm_solution
        
        # Return best solution even if not perfect
        if best_error < 0.05:  # 5cm threshold for "acceptable"
            print(f"[WARNING] IK solution has {best_error:.3f}m error")
            return best_solution
        
        return None
    
    def transform_isaac_to_pybullet(self, isaac_pos):
        """Transform position from IsaacSim coordinates to PyBullet IK coordinates.
        
        This handles any scale or offset differences.
        """
        # Based on the calibration, we know the robot's workspace
        # IsaacSim might have the robot at a different position/scale
        
        # For now, assume IsaacSim uses the same coordinates
        # but you might need to adjust based on your setup
        return isaac_pos
    
    def cleanup(self):
        """Disconnect PyBullet."""
        p.disconnect(self.pb_client)


# Integration with your IsaacLab environment
class PickOrangeMimicEnvIK:
    """IK module for PickOrangeMimicEnv."""
    
    def __init__(self, urdf_path):
        """Initialize IK solver."""
        self.ik_solver = SO101IKSolver(urdf_path, visualize=False)
        
    def solve_ik_for_target(self, target_pos, target_quat, current_joints=None):
        """Solve IK for IsaacLab environment.
        
        Args:
            target_pos: Target position from IsaacLab (torch tensor or numpy)
            target_quat: Target quaternion [w,x,y,z] (IsaacLab format)
            current_joints: Current joint positions
            
        Returns:
            Joint positions as torch tensor
        """
        import torch
        
        # Convert to numpy if needed
        if torch.is_tensor(target_pos):
            target_pos_np = target_pos.cpu().numpy()
        else:
            target_pos_np = target_pos
            
        if torch.is_tensor(target_quat):
            target_quat_np = target_quat.cpu().numpy()
        else:
            target_quat_np = target_quat
            
        # Convert quaternion from IsaacLab [w,x,y,z] to PyBullet [x,y,z,w]
        pb_quat = [target_quat_np[1], target_quat_np[2], target_quat_np[3], target_quat_np[0]]
        
        # Transform position if needed
        pb_pos = self.ik_solver.transform_isaac_to_pybullet(target_pos_np)
        
        # Solve IK
        if current_joints is not None and torch.is_tensor(current_joints):
            current_joints_np = current_joints.cpu().numpy()
        else:
            current_joints_np = current_joints
            
        solution = self.ik_solver.solve_ik(pb_pos, pb_quat, current_joints_np)
        
        if solution is None:
            return None
            
        # Convert back to torch tensor
        return torch.tensor(solution, dtype=torch.float32)


def test_solver():
    """Test the IK solver with known good positions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, required=True)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    solver = SO101IKSolver(args.urdf, visualize=args.visualize)
    
    print("\n" + "="*50)
    print("TESTING IK SOLVER")
    print("="*50)
    
    # Test positions based on actual robot workspace
    # Using positions relative to the home position
    home = solver.home_ee_pos
    
    test_positions = [
        ("Home", home, [0, 0, 0, 1]),
        ("Forward 5cm", home + [0.05, 0, 0], [0, 0, 0, 1]),
        ("Up 5cm", home + [0, 0, 0.05], [0, 0, 0, 1]),
        ("Right 5cm", home + [0, -0.05, 0], [0, 0, 0, 1]),
        ("Diagonal", home + [0.03, 0.03, 0.03], [0, 0, 0, 1]),
    ]
    
    success_count = 0
    for name, pos, quat in test_positions:
        print(f"\n{name}: target={pos}")
        solution = solver.solve_ik(pos, quat)
        
        if solution:
            print(f"  Solution: {solution}")
            
            # Verify
            for i, joint_id in enumerate(solver.arm_joint_ids):
                p.resetJointState(solver.robot_id, joint_id, solution[i], physicsClientId=solver.pb_client)
            
            ee_state = p.getLinkState(solver.robot_id, solver.ee_link_id, physicsClientId=solver.pb_client)
            actual_pos = np.array(ee_state[0])
            error = np.linalg.norm(pos - actual_pos)
            
            print(f"  Actual: {actual_pos}")
            print(f"  Error: {error:.6f}m")
            
            if error < 0.01:
                print("  ✓ SUCCESS")
                success_count += 1
            else:
                print("  ⚠ PARTIAL SUCCESS")
        else:
            print("  ✗ FAILED")
    
    print(f"\n[RESULTS] {success_count}/{len(test_positions)} successful")
    
    solver.cleanup()


if __name__ == "__main__":
    test_solver()