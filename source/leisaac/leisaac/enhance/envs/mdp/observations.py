import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.observations import image


def overlay_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Overlay the background image on the sim render image.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb", and only "rgb" is supported for overlay_image.
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step, with the background image overlayed.
    """
    assert data_type == "rgb", "Only 'rgb' is supported for overlay_image."

    sim_image = image(env, sensor_cfg, data_type, convert_perspective_to_orthogonal, normalize)

    def image_overlapping(
        back_image: torch.Tensor,  # [num_env, H, W, C]
        fore_image: torch.Tensor,  # [num_env, H, W, C]
        back_mask: torch.Tensor | None = None,   # [num_env, H, W]
        fore_mask: torch.Tensor | None = None,   # [num_env, H, W]
        back_alpha: float = 0.5,
        fore_alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Overlap two images with masks.

        Args:
            back_image: background image [num_env, H, W, C]
            fore_image: foreground image [num_env, H, W, C]
            back_mask: background mask [num_env, H, W]
            fore_mask: foreground mask [num_env, H, W]
            back_alpha: background opacity (0-1)
            fore_alpha: foreground opacity (0-1)

        Returns:
            Overlapped image [num_env, H, W, C]
        """
        if back_mask is None:
            back_mask = torch.ones_like(back_image[:, :, :, 0], dtype=torch.bool, device=back_image.device).unsqueeze(-1)
        if fore_mask is None:
            fore_mask = torch.ones_like(fore_image[:, :, :, 0], dtype=torch.bool, device=fore_image.device).unsqueeze(-1)
        image = back_alpha * back_image * back_mask + fore_alpha * fore_image * fore_mask
        return torch.clamp(image, 0.0, 255.0).to(torch.uint8)

    semantic_id = env.foreground_semantic_id_mapping.get(sensor_cfg.name, None)
    if semantic_id is not None:
        camera_output = env.scene.sensors[sensor_cfg.name].data.output
        if env.cfg.rgb_overlay_mode == 'background':
            semantic_mask = camera_output["semantic_segmentation"]
            overlay_mask = semantic_mask == semantic_id
            sim_image = image_overlapping(
                back_image=env.rgb_overlay_images[sensor_cfg.name],
                fore_image=sim_image,
                back_mask=torch.logical_not(overlay_mask),
                fore_mask=overlay_mask,
                back_alpha=1.0,
                fore_alpha=1.0,
            )
        elif env.cfg.rgb_overlay_mode == 'debug':
            sim_image = image_overlapping(
                back_image=env.rgb_overlay_images[sensor_cfg.name],
                fore_image=sim_image,
                back_alpha=0.5,
                fore_alpha=0.5,
            )

    return sim_image
