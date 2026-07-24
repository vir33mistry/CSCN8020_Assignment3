# Step-by-Step Windows and WSL Run Instructions

These instructions assume Windows 11 and Ubuntu 24.04 under WSL 2.

## A. Check WSL

Open PowerShell as Administrator:

```powershell
wsl --status
wsl -l -v
wsl --update
```

Ubuntu should show `VERSION 2`. If Ubuntu is not installed:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if requested.

## B. Place and extract the ZIP

1. Download `CSCN8020_Assignment3.zip`.
2. Put it in `D:\`, producing `D:\CSCN8020_Assignment3.zip`.
3. Open Ubuntu.
4. Run:

```bash
cd /mnt/d
unzip CSCN8020_Assignment3.zip
cd CSCN8020_Assignment3
pwd
```

The final command should end with:

```text
/mnt/d/CSCN8020_Assignment3
```

If the ZIP is under Downloads instead:

```bash
cd /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads
unzip CSCN8020_Assignment3.zip
cd CSCN8020_Assignment3
```

## C. Install system packages

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

## D. Create the Python environment

From the project root:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The terminal prompt should begin with `(.venv)`.

Whenever a new terminal is opened:

```bash
cd /mnt/d/CSCN8020_Assignment3
source .venv/bin/activate
```

## E. Run the quick validation

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python src/smoke_test_dqn.py
PYTHONPATH=src python src/verify_baseline.py
```

Expected indicators:

- `7 passed`
- `Smoke test passed`
- `Environment checker passed`
- `20/20 successes`

## F. Evaluate the included trained model

Retraining is not required to check the submitted checkpoint:

```bash
PYTHONPATH=src python src/evaluate_dqn.py
```

Expected selected policy:

```text
Selected configuration: config_a
```

Expected DQN evaluation:

```text
20/20 successes (100%)
```

## G. Open the notebook

```bash
jupyter notebook CSCN8020_Assignment3.ipynb
```

In the browser:

1. Open the notebook.
2. Choose the Python kernel from `.venv`.
3. Select **Run All**.
4. Leave `RUN_TRAINING = False` unless both experiments should be repeated.

## H. Retrain both experiments

```bash
PYTHONPATH=src python src/train_dqn.py \
  --config all \
  --episodes 600 \
  --device cpu \
  --seed 8020
```

Then:

```bash
PYTHONPATH=src python src/evaluate_dqn.py
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python src/plot_results.py
```

## I. Open the MuJoCo viewer

Check the WSLg display:

```bash
echo $DISPLAY
echo $WAYLAND_DISPLAY
```

Then run:

```bash
PYTHONPATH=src python src/render_dqn_policy.py \
  --checkpoint models/selected_dqn.pt \
  --goals -0.8 -0.4 0.4 0.8
```

If the viewer cannot open, close Ubuntu and run in PowerShell:

```powershell
wsl --shutdown
```

Reopen Ubuntu, activate `.venv`, and retry.

## J. GitHub upload

Create a public empty repository named exactly:

```text
CSCN8020_Assignment3
```

Then run from the project root:

```bash
git init
git branch -M main
git add .
git commit -m "Complete CSCN8020 Assignment 3"
git remote add origin https://github.com/vir33mistry/CSCN8020_Assignment3.git
git push -u origin main
```

Confirm both links work:

- <https://github.com/vir33mistry/CSCN8020_Assignment3>
- <https://github.com/vir33mistry/CSCN8020_Assignment3.git>

## K. Brightspace upload

Upload:

1. `submission/CSCN8020_Assignment3_Brightspace.pdf`
2. The public repository URL
3. The cloneable `.git` URL
4. `video/selected_dqn_demo.mp4`, or the WSLg viewer recording required by the
   instructor

Test the repository from a new folder before submitting.
