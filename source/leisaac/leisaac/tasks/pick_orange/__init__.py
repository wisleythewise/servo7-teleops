import gymnasium as gym

gym.register(
    id='LeIsaac-SO101-PickOrange-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_orange_env_cfg:PickOrangeEnvCfg",
    },
)

# Register the ZeroShot environment
gym.register(
    id="LeIsaac-SO101-ZeroShot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.zero_shot:ZeroShotEnvCfg",
    },
)

# Register the Mimic environment
gym.register(
    id="LeIsaac-SO101-PickOrange-Mimic-v0",
    entry_point="leisaac.tasks.pick_orange.pick_orange_mimic_env:PickOrangeMimicEnv",
    kwargs={
        "env_cfg_entry_point": "leisaac.tasks.pick_orange.pick_orange_env_cfg:PickOrangeEnvCfg",
    },
    disable_env_checker=True,
)