import torch

from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
# from leisaac.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_CFG, KITCHEN_WITH_ORANGE_USD_PATH
# from leisaac.assets.scenes.jappies_first_scene import JappiesFirstSceneCfg  # Import Jappie's scene
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
# from leisaac.utils.general_assets import parse_usd_and_create_subassets  # Not needed for programmatic scene
from leisaac.utils.domain_randomization import randomize_object_uniform, randomize_camera_uniform, domain_randomization

from . import mdp


@configclass
class PickOrangeSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the pick orange task using Jappie's table and cube scene."""

    # Ground plane - shared across all environments
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # Table 
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table", 
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.75),  # Table dimensions
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.6, 0.4),
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.37),  # Center table
        )
    )
    
    # Cube to pick
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # Lighter mass (50g) for easier pickup
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            # Add physics material for better grip
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,  # High static friction for better grip
                dynamic_friction=1.5,  # Dynamic friction when sliding
                restitution=0.0,  # No bounce
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.0,
                roughness=0.8,  # Visual roughness
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -.05, 0.78),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Low wall along x-axis on the table
    wall: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.02, 0.05),  # Length along x-axis, thin in y, low height
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Static wall
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.1,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.6),  # Gray color
                metallic=0.0,
                roughness=0.9,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.765),  # On table surface (table height 0.74 + wall height/2)
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # The robot
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Lighting - shared across all environments
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.5, 0.6), rot=(0.1650476, -0.9862856, 0.0, 0.0), convention="ros"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# @configclass
# class ActionsCfg:
#     """Configuration for the actions."""
#     arm_action: mdp.ActionTermCfg = MISSING
#     gripper_action: mdp.ActionTermCfg = MISSING

from dataclasses import dataclass

from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@configclass
class ActionsCfg:
    """Configuration for the actions."""
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        scale=1.0,
        # Note: no use_default_offset parameter (not in your teleop version)
    )
    gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=1.0,
    )

@configclass
class EventCfg:
    """Configuration for the events."""

    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        wrist = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False})
        front = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Configuration for the rewards"""


@configclass
class TerminationsCfg:
    """Configuration for the termination"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Comment out the old success term that looks for oranges/plate
    # success = DoneTerm(func=mdp.task_done, params={
    #     "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
    #     "plate_cfg": SceneEntityCfg("Plate")
    # })
    
    # Add a simple success term for cube picking
    success = DoneTerm(func=mdp.cube_picked_success, params={
        "cube_cfg": SceneEntityCfg("cube"),
        "height_threshold": 0.785,  # Cube is picked if above this height
    })


from isaaclab.envs import MimicEnvCfg, DataGenConfig, SubTaskConfig

    
@configclass
class PickOrangeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick orange environment."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

        # Add subtask configuration
    subtask_configs: dict = {
        "eef": [
            SubTaskConfig(
                subtask_term_signal="pick_cube",
                subtask_term_offset_range=(0, 0),
                object_ref="cube",  # Add this - it's required by SubTaskConfig
            ),
        ]
    }
    datagen_config: DataGenConfig = DataGenConfig()

    task_constraint_configs: list = [] 


    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-0.5, -1.0, 1.5)  # Better view of table and robot
        self.viewer.lookat = (0.6, 0.0, 0.7)  # Look at table center

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        # Position robot on the table (table height is 0.74m)
        self.scene.robot.init_state.pos = (0.35, 0.0, 0.73)  # x: back from table edge, y: centered, z: on table
        
        # No need for parse_usd_and_create_subassets since we're creating objects programmatically
        # parse_usd_and_create_subassets(KITCHEN_WITH_ORANGE_USD_PATH, self, specific_name_list=['Orange001', 'Orange002', 'Orange003', 'Plate'])

        # Domain randomization for the cube
        # domain_randomization(self, random_options=[
        #     randomize_object_uniform("cube", pose_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)}),
        #     randomize_camera_uniform("front", pose_range={
        #         "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05),
        #         "roll": (-5 * torch.pi / 180, 5 * torch.pi / 180),
        #         "pitch": (-5 * torch.pi / 180, 5 * torch.pi / 180),
        #         "yaw": (-5 * torch.pi / 180, 5 * torch.pi / 180)}, convention="ros"),
        # ])
        self.scene.cube.init_state.pos = list(self.scene.cube.init_state.pos)
        self.scene.cube.init_state.pos[0] -= torch.rand(1).item() * .05  
        self.scene.cube.init_state.pos[1] +=torch.rand(1).item() * .05 
        self.scene.cube.init_state.pos = tuple(self.scene.cube.init_state.pos)

    def use_teleop_device(self, teleop_device) -> None:
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device == "keyboard":
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)
