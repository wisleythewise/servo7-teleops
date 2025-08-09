import gymnasium as gym

gym.register(
    id='LeIsaac-SO101-PickOrange-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_orange_env_cfg:PickOrangeEnvCfg",
    },
)
