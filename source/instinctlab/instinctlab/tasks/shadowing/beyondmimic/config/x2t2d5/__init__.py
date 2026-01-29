# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for X2T2D5 robot in BeyondMimic task."""

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.shadowing.beyondmimic.config.x2t2d5"

gym.register(
    id="Instinct-BeyondMimic-Plane-X2T2D5-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.beyondmimic_plane_cfg:X2T2D5BeyondMimicPlaneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:X2T2D5BeyondMimicPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:X2T2D5BeyondMimicPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-BeyondMimic-Plane-X2T2D5-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.beyondmimic_plane_cfg:X2T2D5BeyondMimicPlaneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:X2T2D5BeyondMimicPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:X2T2D5BeyondMimicPPORunnerCfg",
    },
)
