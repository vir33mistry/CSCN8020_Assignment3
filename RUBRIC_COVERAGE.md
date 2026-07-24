# Rubric Coverage Map

| Criterion | Marks | Evidence in this repository |
|---|---:|---|
| A. Environment understanding and baseline | 8 | `src/g1_rl/g1_elbow_env.py`, `src/verify_baseline.py`, `results/baseline/`, notebook Sections 2-3, report Section 1 |
| B. Q-network and PyTorch | 12 | `src/dqn/q_network.py`, `src/dqn/agent.py`, `models/*.pt`, tests, notebook Section 4 |
| C. Replay buffer and transitions | 10 | `src/dqn/replay_buffer.py`, terminal flags, tensor conversion, capacity and sample tests |
| D. Action selection and exploration | 10 | `DQNAgent.select_action()`, epsilon decay/minimum, fixed seeds, greedy epsilon 0.0 evaluation |
| E. Bellman update and optimization | 15 | `DQNAgent.optimize_model()`, gather, target bootstrap, termination mask, Huber loss, Adam, gradient clipping, target sync |
| F. Training and reproducibility | 10 | `src/train_dqn.py`, CPU/headless default, time limit, metrics, checkpoints, README commands |
| G. Exploration-decay comparison | 10 | `results/config_a/`, `results/config_b/`, `plots/configuration_comparison.png`, report Sections 4-5 |
| H. Final evaluation and performance | 10 | `src/evaluate_dqn.py`, `results/evaluation/`, 20/20 selected-DQN success, per-goal table |
| I. Rule-based versus DQN | 5 | `results/evaluation/policy_comparison.csv`, comparison plot, report Section 7 |
| J. Report, plots, interpretation | 7 | Nine-page `report/DQN_Assignment_Report.pdf`, seven plot files, completed notebook |
| K. Video and submission quality | 3 | `video/selected_dqn_demo.mp4`, `src/render_dqn_policy.py`, viewer recording guide, organized README |

## Required methodology

- [x] Online and target networks
- [x] Target initialized from online network
- [x] Bounded replay memory
- [x] 500-transition warm-up
- [x] Epsilon-greedy training
- [x] Random batches of 64
- [x] Selected Q-values obtained with `gather`
- [x] Target network used for next-state values
- [x] True terminations mask bootstrapping
- [x] Truncation is stored and treated explicitly
- [x] Huber loss
- [x] Gradient clipping
- [x] Target update every 250 optimization steps
- [x] Per-episode epsilon decay with minimum 0.05
- [x] Checkpoint save and reload

## Required outputs

- [x] Episode-level training metrics for both configurations
- [x] Optimization loss metrics for both configurations
- [x] Raw and moving-average reward plot
- [x] Rolling training success plot
- [x] Epsilon plot
- [x] Loss plot
- [x] Controlled-configuration comparison
- [x] Evaluation success by target angle
- [x] Final evaluation table
- [x] Rule-based comparison table
- [x] Selected saved model
- [x] Completed notebook
- [x] Nine-page technical report
- [x] 2-minute 15-second saved-policy video
- [x] Exact run commands
- [x] One-page Brightspace PDF
