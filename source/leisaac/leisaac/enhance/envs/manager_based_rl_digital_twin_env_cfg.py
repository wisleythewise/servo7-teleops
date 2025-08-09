from typing import Dict, Optional, Literal, List

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


@configclass
class ManagerBasedRLDigitalTwinEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ManagerBasedRLDigitalTwinEnv."""

    rgb_overlay_paths: Dict[str, str] | None = None
    """A dictionary of rgb overlay paths.

    The key is the name of the rgb sensor, and the value is the path to the background image.
    example:{"camera_name": "path/to/greenscreen/background.png"}
    """

    rgb_overlay_mode: Optional[Literal["none", "debug", "background"]] = "none"
    """The mode of the rgb overlay.

    The mode can be "none", "debug", or "background".
    - "none": No overlay.
    - "debug": fuse the background image(0.5 opacity) and original render image(0.5 opacity).
    - "background": overlay the background image(1.0 opacity) on the original render image.
    """

    render_objects: Optional[List[SceneEntityCfg]] = []
    """Objects need to be rendered on the background image. If render objects are empty, then the background image will be used as the foreground image.

    The objects are rendered in the order of the list.
    example: [SceneEntityCfg("cube"), SceneEntityCfg("robot")]
    """
