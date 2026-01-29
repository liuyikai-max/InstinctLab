import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.shadowing.perceptive.config.x2t2d5"

gym.register(
    id="Instinct-Perceptive-Shadowing-X2T2D5-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_cfg:X2T2D5PerceptiveShadowingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2T2D5PerceptiveShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:X2T2D5PerceptiveShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Shadowing-X2T2D5-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_cfg:X2T2D5PerceptiveShadowingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2T2D5PerceptiveShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:X2T2D5PerceptiveShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Vae-X2T2D5-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_cfg:X2T2D5PerceptiveVaeEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_cfg:X2T2D5PerceptiveVaePPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Vae-X2T2D5-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_cfg:X2T2D5PerceptiveVaeEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_cfg:X2T2D5PerceptiveVaePPORunnerCfg",
    },
)
