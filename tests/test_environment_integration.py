from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action


def test_environment_checker_and_rule_based_success() -> None:
    env = G1ElbowTargetEnv(goal_angle=-0.8)

    try:
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=42)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = choose_rule_based_action(
                observation=np.asarray(observation),
                controller_target=float(
                    info["controller_target"]
                ),
                action_increment=env.action_increment,
            )
            (
                observation,
                _reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

        assert terminated is True
        assert truncated is False
        assert info["is_success"] is True
    finally:
        env.close()
