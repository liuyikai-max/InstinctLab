# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.x2t2d5"


gym.register(
    id="Instinct-Parkour-Target-Amp-X2T2D5-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.x2t2d5_parkour_target_amp_cfg:X2T2D5ParkourRoughEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:X2T2D5ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-X2T2D5-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.x2t2d5_parkour_target_amp_cfg:X2T2D5ParkourRoughEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:X2T2D5ParkourPPORunnerCfg",
    },
)
