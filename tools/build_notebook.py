"""Build and optionally execute the completed Assignment 3 notebook."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
from pathlib import Path
from typing import Any

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "CSCN8020_Assignment3.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }

    notebook["cells"] = [
        markdown(
            """
# CSCN8020 Assignment 3

## Deep Q-Network Control of the Unitree G1 Left Elbow

**Student:** Viraj Dipakkumar Mistry  
**Student ID:** 9088985  
**Repository:** <https://github.com/vir33mistry/CSCN8020_Assignment3>  
**Clone URL:** <https://github.com/vir33mistry/CSCN8020_Assignment3.git>

This notebook documents the complete student-written PyTorch DQN workflow:
environment validation, baseline evidence, network and replay implementation,
controlled exploration-decay experiments, greedy multi-goal evaluation,
rule-based comparison, plots, checkpoint loading, interpretation, and
reproducibility.
"""
        ),
        markdown(
            """
## 1. Imports, paths, and reproducibility

The final run uses Python 3.12, CPU execution, seed 8020, Gymnasium 1.3.0,
MuJoCo 3.10.0, and PyTorch 2.7.1. All commands assume that the notebook is
opened from the repository root.
"""
        ),
        code(
            """
from pathlib import Path
import json
import random
import sys

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from IPython.display import Image, Video, display
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").is_dir():
    raise RuntimeError("Open this notebook from the CSCN8020_Assignment3 repository root.")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dqn import DQNAgent, DQNHyperparameters, QNetwork, ReplayBuffer
from dqn.experiment import BENCHMARK_GOALS, set_reproducible_seeds
from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action

SEED = 8020
set_reproducible_seeds(SEED)

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Gymnasium:", gym.__version__)
print("MuJoCo:", mujoco.__version__)
print("Device: cpu")
"""
        ),
        markdown(
            """
## 2. Approved reinforcement-learning task

The approved environment is not redesigned.

| Component | Definition |
|---|---|
| Observation | `[elbow angle, angular velocity, goal, goal - angle]` |
| Action 0 | Decrease the internal controller target |
| Action 1 | Hold the internal controller target |
| Action 2 | Increase the internal controller target |
| Training goals | Uniform samples from `[-0.8, +0.8]` rad |
| Success | Error at most 0.04 rad for 8 consecutive environment steps |
| Termination | Task success |
| Truncation | 150-step time limit |
| Low-level control | Approved PD controller plus MuJoCo bias compensation |

The DQN chooses a discrete target adjustment. The low-level controller converts
that target into bounded actuator torque. This separation lets the agent learn
high-level decisions without relearning deterministic gravity compensation.
"""
        ),
        code(
            """
environment = G1ElbowTargetEnv(
    goal_angle=None,
    goal_range=(-0.8, 0.8),
)
check_env(environment, skip_render_check=True)
observation, info = environment.reset(
    seed=SEED,
    options={"goal_angle": -0.8},
)

print("Environment checker passed.")
print("Observation:", observation)
print("Observation shape:", environment.observation_space.shape)
print("Action count:", environment.action_space.n)
print("Controlled joint:", environment.CONTROLLED_JOINT)
print("Controlled actuator:", environment.CONTROLLED_ACTUATOR)
environment.close()
"""
        ),
        markdown(
            """
## 3. Rule-based baseline before training

The assignment requires a validated baseline before DQN training. The same four
goals and five episodes per goal are used for all policies.
"""
        ),
        code(
            """
baseline_by_goal = pd.read_csv(
    PROJECT_ROOT / "results/baseline/summary_by_goal.csv"
)
baseline_summary = json.loads(
    (PROJECT_ROOT / "results/baseline/summary.json").read_text()
)

display(baseline_by_goal)
print(
    f"Overall baseline: {baseline_summary['successes']}/"
    f"{baseline_summary['episodes']} successes "
    f"({baseline_summary['success_rate']:.0%})"
)
"""
        ),
        markdown(
            """
## 4. Student-written Q-network

The required architecture is `4 -> 64 ReLU -> 64 ReLU -> 3`. The final layer
does not use softmax because DQN outputs unconstrained expected returns, not
probabilities. The online and target networks have the same architecture but
separate parameters.
"""
        ),
        code(
            """
network = QNetwork(
    observation_size=4,
    action_size=3,
    hidden_size=64,
)
sample_output = network(torch.zeros((2, 4)))

print(network)
print("Output shape for two observations:", tuple(sample_output.shape))
print(
    "Contains softmax:",
    any(isinstance(module, torch.nn.Softmax) for module in network.modules()),
)
"""
        ),
        markdown(
            """
## 5. Experience replay and Bellman optimization

Each replay transition stores `(state, action, reward, next_state, terminated,
truncated)`. Capacity is 50,000, training starts after 500 transitions, and
mini-batches contain 64 randomly sampled transitions.

The target is:

$$
y = r + \\gamma (1-\\text{terminated})\\max_{a'}Q_{target}(s',a')
$$

where $\\gamma=0.95$. Episodes stop on either ending signal. Only a true
termination blocks bootstrapping; a time-limit truncation is stored explicitly
but its final physical state remains non-terminal for the Bellman target.
Huber loss, Adam, gradient clipping, and target synchronization every 250
optimization steps stabilize learning.
"""
        ),
        code(
            """
buffer = ReplayBuffer(capacity=70, seed=SEED)
for index in range(80):
    state = np.array([index / 100, 0.0, 0.4, 0.4 - index / 100], dtype=np.float32)
    buffer.push(
        state=state,
        action=index % 3,
        reward=-abs(float(state[3])),
        next_state=state + 0.001,
        terminated=(index % 31 == 0),
        truncated=(index % 47 == 0),
    )

batch = buffer.sample(64).to_tensors(torch.device("cpu"))
print("Bounded replay size:", len(buffer))
print("State batch:", tuple(batch.states.shape))
print("Action batch:", tuple(batch.actions.shape))
print("Terminal dtype:", batch.terminated.dtype)
"""
        ),
        markdown(
            """
## 6. Controlled exploration-decay study

Both runs use 600 episodes, seed 8020, CPU, gamma 0.95, learning rate 0.001,
batch size 64, replay capacity 50,000, epsilon 1.0 to 0.05, 500-transition
warm-up, and 250-step target updates. Only epsilon decay differs:

- Configuration A: `0.995`
- Configuration B: `0.985`

Training is headless and contains a per-run time limit.
"""
        ),
        code(
            """
RUN_TRAINING = False

if RUN_TRAINING:
    import subprocess
    subprocess.run(
        [
            sys.executable,
            "src/train_dqn.py",
            "--config", "all",
            "--episodes", "600",
            "--device", "cpu",
            "--seed", str(SEED),
        ],
        check=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )
else:
    print("Using submitted metrics and checkpoints. Set RUN_TRAINING=True to repeat both experiments.")
"""
        ),
        code(
            """
training_rows = []
for configuration in ("config_a", "config_b"):
    summary = json.loads(
        (
            PROJECT_ROOT
            / "results"
            / configuration
            / "training_summary.json"
        ).read_text()
    )
    training_rows.append(
        {
            "configuration": configuration,
            "episodes": summary["completed_episodes"],
            "time_seconds": summary["wall_clock_seconds"],
            "final_epsilon": summary["final_epsilon"],
            "final_20_mean_reward": summary["mean_reward_final_20"],
            "final_50_success_rate": summary["training_success_rate_final_50"],
            "optimization_steps": summary["total_optimization_steps"],
        }
    )

training_summary_table = pd.DataFrame(training_rows)
display(training_summary_table)
"""
        ),
        markdown("## 7. Required training plots"),
        code(
            """
for plot_name in [
    "training_reward.png",
    "training_success_rate.png",
    "epsilon_decay.png",
    "loss_curve.png",
    "configuration_comparison.png",
]:
    print(plot_name)
    display(Image(filename=str(PROJECT_ROOT / "plots" / plot_name), width=900))
"""
        ),
        markdown(
            """
### Training interpretation

Both rolling success curves reach 100%. Configuration B reaches low exploration
earlier; Configuration A collects exploratory data longer. Huber loss is
non-monotonic because Q-target magnitudes and replay composition change as
terminal bonus transitions become frequent. The stable reward curves,
final-50 success rates, and independent greedy evaluation are stronger policy
evidence than loss alone.
"""
        ),
        markdown(
            """
## 8. Greedy 20-episode multi-goal evaluation

Evaluation loads each saved checkpoint, sets epsilon to `0.0`, and runs five
episodes at each required goal. Fixed common seeds make the comparison fair.
"""
        ),
        code(
            """
evaluation_by_goal = pd.read_csv(
    PROJECT_ROOT / "results/evaluation/config_a_by_goal.csv"
)
comparison = pd.read_csv(
    PROJECT_ROOT / "results/evaluation/policy_comparison.csv"
)
selection = json.loads(
    (
        PROJECT_ROOT
        / "results/evaluation/selection_summary.json"
    ).read_text()
)

display(evaluation_by_goal)
display(comparison)
print("Selected configuration:", selection["selected_configuration"])
print("Selected meets 80% threshold:", selection["selected_meets_threshold"])
"""
        ),
        code(
            """
display(
    Image(
        filename=str(PROJECT_ROOT / "plots/evaluation_success_by_goal.png"),
        width=900,
    )
)
display(
    Image(
        filename=str(PROJECT_ROOT / "plots/rule_based_vs_selected_dqn.png"),
        width=900,
    )
)
"""
        ),
        markdown(
            """
## 9. Checkpoint loading and greedy action

The checkpoint is loaded on CPU without retraining. Its metadata identifies the
selected experiment and preserves all hyperparameters.
"""
        ),
        code(
            """
selected_agent, checkpoint_metadata = DQNAgent.load_checkpoint(
    PROJECT_ROOT / "models/selected_dqn.pt",
    device="cpu",
)
sample_state = np.array([0.0, 0.0, 0.4, 0.4], dtype=np.float32)

print("Checkpoint configuration:", checkpoint_metadata["configuration"])
print("Evaluation epsilon: 0.0")
print("Sample Q-values:", selected_agent.q_values(sample_state))
print("Greedy action:", selected_agent.greedy_action(sample_state))
"""
        ),
        markdown(
            """
## 10. Rule-based versus DQN discussion

The rule-based policy is more sample efficient because it requires no training
and encodes the correct target direction. It is also more stable in action
space, averaging one action change versus 6.75 for selected DQN. The DQN,
however, succeeds at all four goals, finishes sooner, earns higher mean reward,
and reaches lower final error. Configuration A uses HOLD in 46.99% of actions,
showing that it learned to settle near goals, although it remains more
corrective than the baseline.

A hand-written policy can outperform learned behaviour on a simple deterministic
task because it starts with perfect structural knowledge. DQN's benefit is that
one network learns the multi-goal mapping from experience rather than receiving
an explicit target-update rule.
"""
        ),
        markdown(
            """
## 11. Recommendation, limitations, and conclusion

Configuration A is selected. Both policies achieve 100%, but A has higher
final-20 training reward, higher greedy mean reward, lower final absolute error,
and much greater HOLD use. Its small extra training time remains far below the
five-hour limit.

Limitations include one joint, fixed-base deterministic physics, one training
seed, 20 deterministic evaluation episodes, no disturbances, and more action
switching than the rule-based policy. Future work should repeat several seeds,
test denser unseen goals and perturbations, and evaluate approved extensions
such as Double DQN.

**Final result:** 20/20 greedy successes, 100% overall success, mean reward
13.1497, and mean final absolute error 0.0071 rad.
"""
        ),
        markdown(
            """
## 12. Saved-policy video and reproducibility

`video/selected_dqn_demo.mp4` is 2 minutes 15 seconds, loads
`models/selected_dqn.pt`, uses epsilon 0.0, and shows all four goals without
retraining. `src/render_dqn_policy.py` provides the interactive MuJoCo viewer
version for WSLg screen recording.
"""
        ),
        code(
            """
video_path = PROJECT_ROOT / "video/selected_dqn_demo.mp4"
print("Video exists:", video_path.is_file())
print("Video size (MB):", round(video_path.stat().st_size / 1_000_000, 2))
display(Video(filename=str(video_path), embed=False, width=900))
"""
        ),
        markdown(
            """
## References and AI-use acknowledgement

1. Mnih et al. (2015), *Human-level control through deep reinforcement
   learning*, Nature 518, 529-533.
2. PyTorch, *Reinforcement Learning (DQN) Tutorial*.
3. Farama Foundation, *Handling Time Limits*.
4. Google DeepMind, *MuJoCo Python Documentation*.
5. CSCN8020 Assignment 3 specification and G1 Primer Workshop.

Generative AI assistance supported scaffolding, debugging, test design,
formatting, and draft writing. The student remains responsible for verifying
all outputs and understanding every submitted component.
"""
        ),
    ]
    return notebook


def display_output(value: Any):
    """Convert common notebook display objects into rich output."""

    if hasattr(value, "to_html") and hasattr(value, "to_string"):
        return nbf.v4.new_output(
            output_type="display_data",
            data={
                "text/plain": value.to_string(),
                "text/html": value.to_html(),
            },
            metadata={},
        )

    representation_png = getattr(value, "_repr_png_", None)
    if callable(representation_png):
        png_data = representation_png()
        if isinstance(png_data, tuple):
            png_data = png_data[0]
        if png_data:
            if isinstance(png_data, str):
                encoded = png_data
            else:
                encoded = base64.b64encode(
                    png_data
                ).decode("ascii")
            return nbf.v4.new_output(
                output_type="display_data",
                data={
                    "image/png": encoded,
                    "text/plain": repr(value),
                },
                metadata={},
            )

    representation_html = getattr(value, "_repr_html_", None)
    if callable(representation_html):
        html = representation_html()
        if html:
            return nbf.v4.new_output(
                output_type="display_data",
                data={
                    "text/html": html,
                    "text/plain": repr(value),
                },
                metadata={},
            )

    return nbf.v4.new_output(
        output_type="display_data",
        data={"text/plain": repr(value)},
        metadata={},
    )


def execute_in_process(notebook):
    """Validate code cells and embed outputs without a Jupyter socket."""

    namespace: dict[str, Any] = {
        "__name__": "__notebook__",
    }
    execution_count = 0

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        execution_count += 1
        captured = io.StringIO()
        rich_outputs = []

        def capture_display(*values, **_kwargs):
            for value in values:
                rich_outputs.append(display_output(value))

        # Replace IPython.display.display after the import cell and
        # before every later cell.
        if execution_count > 1:
            namespace["display"] = capture_display

        try:
            with contextlib.redirect_stdout(captured):
                exec(
                    compile(
                        cell["source"],
                        f"<notebook-cell-{cell_index}>",
                        "exec",
                    ),
                    namespace,
                    namespace,
                )
        except Exception as error:
            cell["outputs"] = [
                nbf.v4.new_output(
                    output_type="error",
                    ename=type(error).__name__,
                    evalue=str(error),
                    traceback=[],
                )
            ]
            cell["execution_count"] = execution_count
            raise RuntimeError(
                f"Notebook cell {cell_index} failed."
            ) from error

        # The import cell replaces display with IPython's function.
        namespace["display"] = capture_display
        outputs = []
        text_output = captured.getvalue()
        if text_output:
            outputs.append(
                nbf.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=text_output,
                )
            )
        outputs.extend(rich_outputs)
        cell["outputs"] = outputs
        cell["execution_count"] = execution_count

    return notebook


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute all cells after building the notebook.",
    )
    parser.add_argument(
        "--execute-in-process",
        action="store_true",
        help=(
            "Execute and capture cells without launching a "
            "Jupyter network kernel."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    notebook = build_notebook()

    if args.execute and args.execute_in_process:
        raise ValueError(
            "Choose only one notebook execution mode."
        )

    if args.execute:
        from nbclient import NotebookClient

        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={
                "metadata": {"path": str(PROJECT_ROOT)}
            },
        )
        notebook = client.execute()
    elif args.execute_in_process:
        notebook = execute_in_process(notebook)

    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Created: {NOTEBOOK_PATH}")
    if args.execute or args.execute_in_process:
        print("Notebook execution completed successfully.")


if __name__ == "__main__":
    main()
