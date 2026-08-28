import gymnasium as gym

from . import agents

gym.register(
    id="RPO-Parkour",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_parkour_env_cfg:RPOParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_parkour_agent_cfg:RPOParkourAmpRunnerCfg",
    },
)

gym.register(
    id="RPO-Parkour-Play",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_parkour_env_cfg:RPOParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_parkour_agent_cfg:RPOParkourAmpRunnerCfg",
    },
)

gym.register(
    id="F1-Parkour",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.f1_parkour_env_cfg:F1ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.f1_parkour_agent_cfg:F1ParkourAmpRunnerCfg",
    },
)

gym.register(
    id="F1-Parkour-Play",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.f1_parkour_env_cfg:F1ParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.f1_parkour_agent_cfg:F1ParkourAmpRunnerCfg",
    },
)
