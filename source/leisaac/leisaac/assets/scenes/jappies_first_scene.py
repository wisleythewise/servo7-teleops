from pathlib import Path

from leisaac.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

"""Configuration for Jappie's First Scene with Table and Cube"""

# Create a complete scene configuration class
@configclass
class JappiesFirstSceneCfg(InteractiveSceneCfg):
    """Scene configuration with table and cube created programmatically."""
    
    # Note: We don't need a base "scene" asset since we're creating everything programmatically
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    
    # Table 
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table", 
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.74),  # Table dimensions (width, depth, height)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Table doesn't move
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.6, 0.4),  # Wood-like brown color
                roughness=0.8,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.37),  # Center at 0.6m forward, height/2 to place bottom at ground
        )
    )
    
    # Cube to pick
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),  # 5cm cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100g
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red cube
                metallic=0.0,
                roughness=0.5,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.77),  # On top of table (table_height + cube_size/2)
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz quaternion
        )
    )
    
    # Optional: Add a second cube if you want
    cube2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube2",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # Blue cube
                metallic=0.0,
                roughness=0.5,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.1, 0.77),  # Slightly to the side
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

# Export the configuration for easy import
# Note: We don't instantiate it here, the @configclass decorator handles that
# JAPPIES_FIRST_SCENE_CFG = JappiesFirstSceneCfg()