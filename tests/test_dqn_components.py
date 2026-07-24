from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dqn import (
    DQNAgent,
    DQNHyperparameters,
    QNetwork,
    ReplayBuffer,
)


def make_state(value: float = 0.0) -> np.ndarray:
    return np.asarray(
        [value, 0.0, 0.4, 0.4 - value],
        dtype=np.float32,
    )


def test_q_network_has_required_shape_and_no_softmax() -> None:
    network = QNetwork()
    output = network(torch.zeros((5, 4)))

    assert output.shape == (5, 3)
    assert not any(
        isinstance(module, torch.nn.Softmax)
        for module in network.modules()
    )


def test_replay_buffer_capacity_and_tensor_batch() -> None:
    replay = ReplayBuffer(capacity=70, seed=8020)

    for index in range(80):
        replay.push(
            state=make_state(index / 100),
            action=index % 3,
            reward=float(index),
            next_state=make_state((index + 1) / 100),
            terminated=(index == 79),
            truncated=False,
        )

    assert len(replay) == 70
    tensor_batch = replay.sample(64).to_tensors(
        torch.device("cpu")
    )
    assert tensor_batch.states.shape == (64, 4)
    assert tensor_batch.actions.shape == (64, 1)
    assert tensor_batch.rewards.shape == (64, 1)
    assert tensor_batch.terminated.dtype == torch.bool


def test_replay_rejects_oversized_sample() -> None:
    replay = ReplayBuffer(capacity=10)
    replay.push(
        make_state(),
        1,
        0.0,
        make_state(0.01),
        False,
        False,
    )

    with pytest.raises(ValueError):
        replay.sample(2)


def test_greedy_action_is_repeatable() -> None:
    agent = DQNAgent(seed=8020, device="cpu")
    state = make_state()

    assert agent.select_action(
        state,
        epsilon=0.0,
    ) == agent.select_action(
        state,
        epsilon=0.0,
    )


def test_optimize_and_target_sync() -> None:
    hyperparameters = DQNHyperparameters(
        warmup_transitions=64,
        target_update_steps=1,
    )
    agent = DQNAgent(
        hyperparameters=hyperparameters,
        seed=8020,
        device="cpu",
    )

    for index in range(64):
        agent.remember(
            state=make_state(index / 100),
            action=index % 3,
            reward=-0.1,
            next_state=make_state((index + 1) / 100),
            terminated=(index % 17 == 0),
            truncated=(index % 29 == 0),
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


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = DQNAgent(seed=8020, device="cpu")
    checkpoint = agent.save_checkpoint(
        tmp_path / "agent.pt",
        metadata={"experiment": "unit-test"},
    )

    loaded, metadata = DQNAgent.load_checkpoint(
        checkpoint,
        device="cpu",
    )
    state = make_state()

    np.testing.assert_allclose(
        agent.q_values(state),
        loaded.q_values(state),
    )
    assert metadata["experiment"] == "unit-test"
