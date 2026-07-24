# CSCN8020 Assignment 3 — Deep Q-Network Control of the Unitree G1 Left Elbow

**Student:** Viraj Dipakkumar Mistry  
**Student ID:** 9088985  
**Course:** CSCN8020 — Reinforcement Learning  
**Repository:** <https://github.com/vir33mistry/CSCN8020_Assignment3>  
**Clone URL:** <https://github.com/vir33mistry/CSCN8020_Assignment3.git>

## Project overview

This project implements a student-written Deep Q-Network (DQN) in PyTorch to
control the left elbow joint of a fixed-base Unitree G1 humanoid robot in
MuJoCo. The agent receives a four-value observation and selects one of three
discrete actions: decrease the controller target, hold it, or increase it.

The implementation includes:

- A `4 → 64 ReLU → 64 ReLU → 3` Q-network with no softmax.
- A bounded replay buffer with random mini-batch sampling.
- Online and target networks.
- Epsilon-greedy exploration.
- Bellman target calculation and Huber loss.
- Gradient clipping and scheduled target-network synchronization.
- Checkpoint save/load support.
- Headless CPU training and greedy evaluation.
- Two controlled epsilon-decay experiments.
- A comparison with the supplied rule-based controller.
- Training metrics, evaluation results, plots, report, notebook, and video.

## Quick start: move the robot arm

Run commands from the repository root with the virtual environment activated.

For Ubuntu/WSL:

```bash
PYTHONPATH=src python src/render_dqn_policy.py --checkpoint models/selected_dqn.pt --goals -0.8 -0.4 0.4 0.8
```

For automatic transitions between the four targets:

```bash
PYTHONPATH=src python src/render_dqn_policy.py \
  --checkpoint models/selected_dqn.pt \
  --goals -0.8 -0.4 0.4 0.8 \
  --auto \
  --pause-seconds 3
```

The correct environment-variable name is **`PYTHONPATH`**. Do not type
`PYTPYTHONPATH`.

The viewer loads `models/selected_dqn.pt`, uses greedy action selection with
epsilon `0.0`, and does not retrain the model. Without `--auto`, position the
camera and press `Enter` in the terminal when prompted.

## Validated results

| Policy | Successes | Success rate | Mean reward | Mean length | Mean final absolute error |
|---|---:|---:|---:|---:|---:|
| Config A — decay 0.995 | 20/20 | 100% | 13.1497 | 20.75 | 0.0071 rad |
| Config B — decay 0.985 | 20/20 | 100% | 13.0880 | 19.75 | 0.0136 rad |
| Rule-based baseline | 20/20 | 100% | 12.8666 | 24.00 | 0.0122 rad |

Configuration A was selected because both DQN configurations met the success
target, while Configuration A achieved the higher mean reward and lower final
absolute error. Its checkpoint is saved as `models/selected_dqn.pt`.

## Validated environment

- Python 3.12.13 for the submitted training run.
- Python 3.14 compatibility verified with the newer CPU-only PyTorch wheel.
- PyTorch 2.7.1 CPU for Python versions below 3.14.
- PyTorch 2.9.1 CPU for Python 3.14.
- Gymnasium 1.3.0.
- MuJoCo 3.10.0.
- Linux x86-64, headless CPU training.
- Target platform: Windows 11 with WSL 2 and Ubuntu.
- Training goal range: `[-0.8, +0.8]` radians.
- Evaluation goals: `-0.8`, `-0.4`, `+0.4`, and `+0.8` radians.

The supplied `requirements.txt` automatically selects the compatible
CPU-only PyTorch version for the active Python interpreter.

## WSL installation

### 1. Install WSL

Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --shutdown
```

### 2. Install Ubuntu system packages

```bash
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-dev \
  build-essential \
  libglfw3 \
  libgl1-mesa-dev \
  libegl1-mesa-dev \
  ffmpeg
```

### 3. Open the project on the D drive

For the local project location used during development:

```bash
cd /mnt/d/CSCN8020_Assignment3/CSCN8020_Assignment3
```

For a repository cloned elsewhere, change into that repository root instead.

### 4. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

If a different local environment name is used, activate that environment
instead. For example:

```bash
source .venv_new/bin/activate
```

Confirm the interpreter and packages:

```bash
which python
python -c "import torch, gymnasium, mujoco; print('PyTorch:', torch.__version__); print('Gymnasium:', gymnasium.__version__); print('MuJoCo:', mujoco.__version__)"
```

## Optional native Windows setup

If WSL or WSLg is unavailable, the project can be run from Windows PowerShell.
Use a separate Windows virtual environment; a Linux virtual environment cannot
be reused on Windows.

```powershell
cd D:\CSCN8020_Assignment3\CSCN8020_Assignment3
py -3.11 -m venv .venv_win
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv_win\Scripts\Activate.ps1
python -m pip install --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

Run the graphical DQN demonstration in PowerShell:

```powershell
$env:PYTHONPATH="src"
python src\render_dqn_policy.py --checkpoint models\selected_dqn.pt --goals -0.8 -0.4 0.4 0.8 --auto --pause-seconds 3
```

## Verify the environment

Run the environment checker and rule-based baseline:

```bash
PYTHONPATH=src python src/test_g1_elbow_env.py
PYTHONPATH=src python src/verify_baseline.py
```

Expected result:

- Environment checker passes.
- Rule-based policy succeeds in 20/20 benchmark episodes.

## Run the automated tests

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python src/smoke_test_dqn.py
```

Expected result:

- `7 passed`
- `Smoke test passed`

The smoke test covers replay insertion, bounded capacity, random sampling,
tensor conversion, epsilon-greedy action selection, one Bellman optimization
step, and target-network synchronization.

## Evaluate the saved policies

Evaluate both DQN checkpoints with epsilon `0.0` and compare them against the
rule-based controller on the same 20 episodes:

```bash
PYTHONPATH=src python src/evaluate_dqn.py
```

Expected selected policy:

```text
Selected configuration: config_a
```

Expected success result:

```text
20/20 successes
```

Confirm that the selected checkpoint loads without retraining:

```bash
PYTHONPATH=src python -c "from dqn import DQNAgent; import numpy as np; a,m=DQNAgent.load_checkpoint('models/selected_dqn.pt', device='cpu'); print(m['configuration'], a.greedy_action(np.array([0,0,0.4,0.4], dtype='float32')))"
```

## Run the basic MuJoCo viewer

Use this test before running the complete G1 model:

```bash
python test_mujoco_viewer.py
```

A MuJoCo window containing a blue box should open.

## Run the rule-based G1 demonstration

```bash
PYTHONPATH=src python src/demo_g1_elbow_env.py --goal -0.8
```

After the viewer opens, position the camera and press `Enter` in the terminal.

## Run the completed notebook

```bash
jupyter notebook CSCN8020_Assignment3.ipynb
```

Select the active virtual-environment kernel. The notebook loads the submitted
evidence by default and does not retrain unless `RUN_TRAINING` is changed to
`True`.

## Train both required configurations

The repository already includes trained checkpoints. Retraining is optional
unless required by the evaluator.

```bash
PYTHONPATH=src python src/train_dqn.py \
  --config all \
  --episodes 600 \
  --device cpu \
  --seed 8020
```

Train one configuration:

```bash
PYTHONPATH=src python src/train_dqn.py --config config_a --episodes 600
PYTHONPATH=src python src/train_dqn.py --config config_b --episodes 600
```

Configuration A uses epsilon decay `0.995`; Configuration B uses `0.985`.
Every other baseline hyperparameter and the seed policy are held constant. The
training script runs headlessly and applies a 140-minute limit per experiment.

## Regenerate plots

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python src/plot_results.py
```

The script creates reward, moving-average reward, rolling success, epsilon,
loss, configuration-comparison, evaluation-by-goal, and baseline-comparison
plots.

## Video demonstration

The ready-made 2-minute 15-second demonstration is:

```text
video/selected_dqn_demo.mp4
```

It uses actual headless MuJoCo states and shows all four evaluation goals,
actions, Q-values, controller target, elbow angle, error, reward, and success.

Regenerate the portable video without retraining:

```bash
PYTHONPATH=src python src/record_dqn_video.py \
  --checkpoint models/selected_dqn.pt \
  --duration-seconds 135
```

For a strict screen recording of the graphical MuJoCo window, follow
`VIDEO_RECORDING_GUIDE.md`.

## DQN implementation details

### State and actions

The environment provides four observation values. The agent produces three
Q-values corresponding to:

1. Decrease the elbow controller target.
2. Hold the current controller target.
3. Increase the elbow controller target.

### Learning components

- Discount factor: `0.95`.
- Adam learning rate: `0.001`.
- Batch size: `64`.
- Replay-buffer capacity: `50,000`.
- Replay warm-up: at least `500` transitions.
- Initial epsilon: `1.0`.
- Minimum epsilon: `0.05`.
- Configuration A epsilon decay: `0.995`.
- Configuration B epsilon decay: `0.985`.
- Target update frequency: every `250` optimization steps.
- Maximum episode length: `150` steps.
- Huber loss and gradient clipping.

Episodes stop when either `terminated` or `truncated` becomes true. Bellman
bootstrapping is disabled only for a true task termination. Time-limit
truncations remain physically valid non-terminal transitions, so their targets
may bootstrap. Both flags are stored separately in replay memory.

## Repository structure

| Path | Purpose |
|---|---|
| `CSCN8020_Assignment3.ipynb` | Completed and executed assignment notebook |
| `src/g1_rl/g1_elbow_env.py` | Unitree G1 elbow Gymnasium environment |
| `src/dqn/` | Q-network, replay buffer, agent, and experiment helpers |
| `src/train_dqn.py` | Headless training workflow |
| `src/evaluate_dqn.py` | Greedy evaluation and policy comparison |
| `src/render_dqn_policy.py` | Interactive saved-DQN MuJoCo demonstration |
| `src/demo_g1_elbow_env.py` | Rule-based graphical demonstration |
| `models/selected_dqn.pt` | Selected Configuration A checkpoint |
| `results/` | Training, loss, baseline, and evaluation metrics |
| `plots/` | Required assignment plots |
| `report/DQN_Assignment_Report.pdf` | Nine-page technical report |
| `submission/CSCN8020_Assignment3_Brightspace.pdf` | One-page submission PDF |
| `video/selected_dqn_demo.mp4` | Saved-policy demonstration video |
| `tests/` | Unit and integration tests |

## Troubleshooting

### `ModuleNotFoundError: dqn` or `g1_rl`

Run from the repository root and include the correct prefix:

```bash
PYTHONPATH=src python src/render_dqn_policy.py --checkpoint models/selected_dqn.pt --goals -0.8 -0.4 0.4 0.8
```

### MuJoCo viewer does not open in WSL

Confirm WSLg variables:

```bash
echo $DISPLAY
echo $WAYLAND_DISPLAY
```

Test the basic viewer:

```bash
python test_mujoco_viewer.py
```

Software-rendering fallback:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw python test_mujoco_viewer.py
```

### VS Code reconnects to WSL

Run the headless evaluator from the standalone Ubuntu terminal:

```bash
PYTHONPATH=src python src/evaluate_dqn.py
```

If WSL still restarts, check available Windows/WSL disk space and memory before
retrying. Native Windows execution is available as described above.

### Virtual environments and Git

The repository `.gitignore` excludes:

```text
.venv/
.venv_new/
.venv_win/
.venv_broken*/
```

Do not commit virtual environments to GitHub.

## Submission artifacts

- `report/DQN_Assignment_Report.pdf`
- `submission/CSCN8020_Assignment3_Brightspace.pdf`
- `CSCN8020_Assignment3.ipynb`
- `models/selected_dqn.pt`
- `video/selected_dqn_demo.mp4`
- Public repository URL
- Cloneable `.git` URL

Before submitting, test the public repository from a clean folder and verify
that both GitHub links are accessible.

## Academic integrity and AI use

AI assistance was used for scaffolding, debugging, testing support, and draft
writing. The student remains responsible for validating the repository,
understanding every DQN and MuJoCo component, following course disclosure
requirements, and explaining the submitted work. See `AI_USE_DISCLOSURE.md`.
