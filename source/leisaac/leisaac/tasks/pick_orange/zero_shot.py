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
"""
python ./scripts/environments/teleoperation/teleop_se3_agent.py --task LeIsaac-SO101-ZeroShot-v0 --teleop_device keyboard --enable_cameras
"""

@configclass
class ZeroShotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the zero shot task in a white studio environment."""

    # Simple room environment from Isaac Sim
    simple_room = AssetBaseCfg(
        prim_path="/World/SimpleRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Simple_Room/simple_room.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 1.0, 0.0),  # Move entire room 1m in +y direction
        )
    )
    
    # Studio floor - white floor inside the cube (on existing table_low_327)
    studio_floor: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StudioFloor",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.125, 0.05),  # Studio floor dimensions (25% less deep)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.80, 0.78),  # Even more off-white, worn look
                roughness=0.85,  # Very rough, almost chalk-like surface
                metallic=0.0,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 1.0, 0.025),  # On existing table at x=0, y=1, z=0
        )
    )
    
    # Studio left wall
    studio_left_wall: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StudioLeftWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 1.125, 0.75),  # Thin wall, half height, 25% less deep
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.80, 0.78),  # Even more off-white, worn look
                roughness=0.85,  # Very rough, almost chalk-like surface
                metallic=0.0,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.725, 1.0, 0.425),  # Left wall on existing table
        )
    )
    
    # Studio right wall
    studio_right_wall: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StudioRightWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 1.125, 0.75),  # Thin wall, half height, 25% less deep
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.80, 0.78),  # Even more off-white, worn look
                roughness=0.85,  # Very rough, almost chalk-like surface
                metallic=0.0,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.725, 1.0, 0.425),  # Right wall on existing table
        )
    )
    
    # Studio back wall (backplate)
    studio_back_wall: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StudioBackWall",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.05, 0.75),  # Backplate, half height
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.80, 0.78),  # Even more off-white, worn look
                roughness=0.85,  # Very rough, almost chalk-like surface
                metallic=0.0,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 1.5625, 0.425),  # Back wall on existing table (25% less deep)
        )
    )
    
    # Aluminum profile edges - vertical edges
    aluminum_edge_front_left: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeFrontLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.75),  # Square aluminum profile, half height
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.7),  # Aluminum color
                metallic=0.8,
                roughness=0.3,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 0.4375, 0.425),  # Front left vertical edge on existing table
        )
    )
    
    aluminum_edge_front_right: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeFrontRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.4375, 0.425),  # Front right vertical edge on existing table
        )
    )
    
    aluminum_edge_back_left: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeBackLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 1.5625, 0.425),  # Back left vertical edge on existing table
        )
    )
    
    aluminum_edge_back_right: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeBackRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 1.5625, 0.425),  # Back right vertical edge on existing table
        )
    )
    
    # Horizontal aluminum edges - bottom edges
    aluminum_edge_bottom_left: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeBottomLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 1.125, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 1.0, 0.025),  # Bottom left horizontal edge on existing table
        )
    )
    
    aluminum_edge_bottom_right: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeBottomRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 1.125, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 1.0, 0.025),  # Bottom right horizontal edge on existing table
        )
    )
    
    aluminum_edge_bottom_back: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeBottomBack",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 1.5625, 0.025),  # Bottom back horizontal edge on existing table
        )
    )
    
    # Top aluminum edges
    aluminum_edge_top_left: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeTopLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 1.125, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 1.0, 0.8),  # Top left horizontal edge on existing table
        )
    )
    
    aluminum_edge_top_right: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeTopRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 1.125, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 1.0, 0.8),  # Top right horizontal edge on existing table
        )
    )
    
    aluminum_edge_top_back: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AluminumEdgeTopBack",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.75, 0.77),  # Slightly bluish aluminum
                metallic=0.65,  # Less metallic for realism
                roughness=0.35,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 1.5625, 0.8),  # Top back horizontal edge on existing table
        )
    )
    
    # Cube to pick - positioned on studio floor
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,
                dynamic_friction=1.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.55, 0.15),  # More realistic orange
                metallic=0.02,  # Tiny bit of sheen
                roughness=0.75,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 1.0, 0.065),  # On studio floor on existing table, centered between robots
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Black plastic corner connectors
    corner_front_left_bottom: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerFrontLeftBottom",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),  # Slightly larger than aluminum profiles
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1),  # Black plastic
                metallic=0.0,
                roughness=0.7,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 0.4375, 0.025),  # Front left bottom corner on existing table
        )
    )
    
    corner_front_right_bottom: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerFrontRightBottom",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.16),  # Not pure black - more realistic
                metallic=0.05,  # Slight sheen on plastic
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.4375, 0.025),  # Front right bottom corner on existing table
        )
    )
    
    corner_back_left_bottom: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerBackLeftBottom",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.16),  # Not pure black - more realistic
                metallic=0.05,  # Slight sheen on plastic
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 1.5625, 0.025),  # Back left bottom corner on existing table
        )
    )
    
    corner_back_right_bottom: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerBackRightBottom",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.16),  # Not pure black - more realistic
                metallic=0.05,  # Slight sheen on plastic
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 1.5625, 0.025),  # Back right bottom corner on existing table
        )
    )

    # Top corner connectors
    corner_front_left_top: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerFrontLeftTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.16),  # Not pure black - more realistic
                metallic=0.05,  # Slight sheen on plastic
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 0.4375, 0.8),  # Front left top corner on existing table
        )
    )
    
    corner_front_right_top: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerFrontRightTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.16),  # Not pure black - more realistic
                metallic=0.05,  # Slight sheen on plastic
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.4375, 0.8),  # Front right top corner on existing table
        )
    )
    
    corner_back_left_top: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerBackLeftTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.16),  # Not pure black - more realistic
                metallic=0.05,  # Slight sheen on plastic
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.75, 1.5625, 0.8),  # Back left top corner on existing table
        )
    )
    
    corner_back_right_top: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerBackRightTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.16),  # Not pure black - more realistic
                metallic=0.05,  # Slight sheen on plastic
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 1.5625, 0.8),  # Back right top corner on existing table
        )
    )

    # The robots - two side by side
    robot_left: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotLeft")
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Ambient dome light - soft, natural lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.88, 0.82, 0.70),  # Even warmer, more muted
            intensity=50.0,  # Much dimmer ambient
        ),
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

    # Directional key light - creates shadows and depth
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KeyLight",
        spawn=sim_utils.DistantLightCfg(
            color=(0.90, 0.85, 0.75),  # Very warm, muted directional
            intensity=120.0,  # Much dimmer directional light
            angle=20.0,  # Extremely diffused shadows
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, -2.0, 3.0),  # Position for angled lighting
            rot=(0.924, -0.383, 0.0, 0.0),  # Angled down and to the side
        )
    )


# @configclass
# class ActionsCfg:
#     """Configuration for the actions."""
#     arm_action: mdp.ActionTermCfg = MISSING
#     gripper_action: mdp.ActionTermCfg = MISSING



@configclass
class ActionsCfg:
    """Configuration for the actions."""
    # Main robot actions (right robot only - left robot is visual only)
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        scale=1.0,
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
        "height_threshold": 0.9,  # Cube is picked if above this height (studio floor is at 0.8m)
    })


from isaaclab.envs import DataGenConfig, SubTaskConfig

    
@configclass
class ZeroShotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the zero shot environment."""

    scene: ZeroShotSceneCfg = ZeroShotSceneCfg(env_spacing=8.0)

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
        self.viewer.eye = (-1.5, -1.5, 1.2)  # Better view of studio from front-left
        self.viewer.lookat = (0.0, 0.0, 0.5)  # Look at studio center

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        # Position both robots on the studio floor (on table) at the front edge, rotated 180° around z-axis
        # robot_left is visual only - not controlled by teleoperation
        self.scene.robot_left.init_state.pos = (-0.25, 0.5, 0.05)  # Left robot on existing table (visual only)
        self.scene.robot_left.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # 180° rotation around z-axis (w, x, y, z)
        # robot is the main controllable robot
        self.scene.robot.init_state.pos = (0.25, 0.5, 0.05)  # Right robot on existing table (main controllable)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # 180° rotation around z-axis (w, x, y, z)
        
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
            self.scene.robot_left.spawn.rigid_props.disable_gravity = True
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)
