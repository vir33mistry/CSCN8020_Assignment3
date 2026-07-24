"""Short required smoke test for all core DQN operations."""

from __future__ import annotations

import numpy as np
import torch

from dqn import DQNAgent, DQNHyperparameters, ReplayBuffer
from dqn.experiment import set_reproducible_seeds


def main() -> None:
    seed = 8020
    set_reproducible_seeds(seed)

    print("1. Testing replay insertion and bounded capacity...")
    replay = ReplayBuffer(capacity=70, seed=seed)
    for index in range(80):
        state = np.asarray(
            [index / 100, 0.0, 0.4, 0.4 - index / 100],
            dtype=np.float32,
        )
        replay.push(
            state=state,
            action=index % 3,
            reward=float(-abs(state[3])),
            next_state=state + 0.001,
            terminated=(index % 31 == 0),
            truncated=(index % 47 == 0),
        )
    assert len(replay) == 70

    print("2. Testing random mini-batch sampling and tensors...")
    batch = replay.sample(64)
    tensors = batch.to_tensors(torch.device("cpu"))
    assert tensors.states.shape == (64, 4)
    assert tensors.actions.shape == (64, 1)
    assert tensors.terminated.dtype == torch.bool
    assert tensors.truncated.dtype == torch.bool

    print("3. Testing epsilon-greedy and deterministic greedy actions...")
    smoke_hyperparameters = DQNHyperparameters(
        warmup_transitions=64,
        target_update_steps=1,
    )
    agent = DQNAgent(
        hyperparameters=smoke_hyperparameters,
        seed=seed,
        device="cpu",
    )
    test_state = np.asarray(
        [0.0, 0.0, 0.4, 0.4],
        dtype=np.float32,
    )
    random_action = agent.select_action(test_state, epsilon=1.0)
    greedy_action_1 = agent.select_action(
        test_state,
        epsilon=0.0,
    )
    greedy_action_2 = agent.select_action(
        test_state,
        epsilon=0.0,
    )
    assert 0 <= random_action < 3
    assert greedy_action_1 == greedy_action_2

    print("4. Testing one Bellman optimization and target update...")
    for transition in replay._memory:  # noqa: SLF001 - smoke-test only
        agent.remember(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            terminated=transition.terminated,
            truncated=transition.truncated,
        )

    loss = agent.optimize_model()
    assert loss is not None
    assert np.isfinite(loss)
    assert agent.optimization_steps == 1

    for online_parameter, target_parameter in zip(
        agent.online_network.parameters(),
        agent.target_network.parameters(),
    ):
        assert torch.equal(online_parameter, target_parameter)

    print(f"Smoke test passed. Optimization loss: {loss:.6f}")


if __name__ == "__main__":
    main()
