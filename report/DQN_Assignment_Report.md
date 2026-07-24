# Deep Q-Network Control of the Unitree G1 Left Elbow

**Viraj Dipakkumar Mistry - Student ID 9088985**  
**CSCN8020 Reinforcement Learning - Assignment 3**

## Executive summary

This project extends the completed Unitree MuJoCo G1 Primer Workshop with a
student-written PyTorch Deep Q-Network. The approved four-value observation,
three discrete actions, PD controller, bias compensation, reward function, and
success logic were kept unchanged. Two 600-episode CPU experiments differed
only in epsilon decay: 0.995 for Configuration A and 0.985 for Configuration B.
Both DQNs achieved 20/20 successes under greedy evaluation across goals -0.8,
-0.4, +0.4, and +0.8 rad. Configuration A was selected because its mean
evaluation reward was 13.1497 and its mean final absolute error was 0.0071 rad,
compared with 13.0880 and 0.0136 rad for Configuration B.

## 1. Connection to the G1 Primer and environment

The primer established a fixed-base G1 model, joint and actuator mappings, a
low-level PD controller, MuJoCo `qfrc_bias` compensation, and a compliant
Gymnasium environment. This assignment does not redesign that platform.
Instead, the DQN replaces the rule-based high-level policy that modifies an
internal elbow target.

The observation is `[angle, angular velocity, goal, goal - angle]`. The actions
are 0 (decrease target), 1 (hold), and 2 (increase target). Each action is
translated by the approved controller into bounded actuator torque. Reward is
negative absolute error, plus a success-region bonus, a small non-HOLD penalty
near the goal, and a 10-point terminal success bonus. Success requires the
actual elbow to remain within 0.04 rad for eight consecutive environment steps.
`terminated` therefore indicates success, while `truncated` indicates the
150-step time limit.

The Gymnasium checker passed. Before training, the rule-based policy achieved
20/20 successes, mean reward 12.8666, mean episode length 24.0, and mean final
absolute error 0.0122 rad.

## 2. Q-network and DQN methodology

The Q-network follows the required architecture: four inputs, two fully
connected 64-unit hidden layers with ReLU, and three linear outputs. The output
layer has no softmax because Q-values are unconstrained estimates of expected
discounted return.

The agent maintains separate online and target networks. Both begin with
identical weights. A bounded replay buffer stores 50,000 transitions containing
state, action, reward, next state, `terminated`, and `truncated`. Learning begins
after 500 transitions and samples random mini-batches of 64. For selected action
`a`, the update target is:

`y = r + gamma * (1 - terminated) * max_a' Q_target(s', a')`

with gamma 0.95. The target is calculated under `torch.no_grad`, so it is
detached. Smooth L1 (Huber) loss compares the selected online Q-value with the
target. Adam uses learning rate 0.001. Gradient norm is clipped to 10, and the
target network is synchronized every 250 optimization steps.

Episodes stop on either signal. Only true terminations mask bootstrapping. A
time-limit truncation is stored explicitly and ends data collection, but its
final physical state is treated as non-terminal in the Bellman target. This
matches Gymnasium's distinction between task termination and externally imposed
truncation.

## 3. Exploration, training, and reproducibility

Python, NumPy, PyTorch, replay sampling, network initialization, and Gymnasium
resets use seed 8020. Training runs headlessly on CPU. Both configurations use
600 episodes, gamma 0.95, learning rate 0.001, batch size 64, capacity 50,000,
epsilon start 1.0, epsilon minimum 0.05, target update 250, warm-up 500, and the
same goal sequence sampled from [-0.8, +0.8] rad. The only controlled change is
epsilon decay.

Configuration A uses 0.995 and reaches the minimum near the end of training.
Configuration B uses 0.985 and reaches the minimum near episode 200.
Configuration A completed in 53.70 seconds and Configuration B in 40.00 seconds
in the validated CPU environment, for a total far below five hours.

## 4. Exploration-decay results

| Required metric | Config A: 0.995 | Config B: 0.985 |
|---|---:|---:|
| Training episodes | 600 | 600 |
| Wall-clock time | 53.70 s | 40.00 s |
| Final epsilon | 0.05 | 0.05 |
| Final-20 mean reward | 15.5566 | 15.3677 |
| Final-50 training success | 100% | 100% |
| Greedy successes | 20/20 | 20/20 |
| Greedy success rate | 100% | 100% |
| Mean evaluation reward | 13.1497 | 13.0880 |
| Mean final absolute error | 0.0071 rad | 0.0136 rad |
| Mean episode length | 20.75 | 19.75 |
| HOLD action fraction | 46.99% | 11.39% |

Both schedules converge. Faster decay improves training speed and yields
slightly shorter evaluation episodes, but the longer exploration schedule
produces the higher final-20 reward, higher evaluation reward, lower final
error, and substantially more HOLD use. These are stability and control-quality
advantages rather than reliance on one unusually high reward.

## 5. Training-curve interpretation

Early random exploration produces long, negative-reward episodes. Rolling
success then rises to 100% for both agents. Configuration B crosses the success
target sooner because it reduces random action selection faster. Configuration
A improves more gradually but continues collecting exploratory transitions for
longer.

Huber loss does not decrease monotonically. Its later increase reflects growing
Q-target magnitude and a replay distribution that changes as successful,
bonus-bearing transitions become common. It is not by itself evidence of
failure: rewards remain stable, final-50 training success is 100%, and both
independent greedy evaluations reach 100%. Evaluation metrics therefore provide
the decisive evidence of policy quality.

## 6. Final greedy evaluation

Evaluation uses epsilon 0.0 and five episodes at each goal, for 20 episodes.

| Goal | Episodes | Successes | Success rate | Mean reward |
|---:|---:|---:|---:|---:|
| -0.8 rad | 5 | 5 | 100% | 10.5898 |
| -0.4 rad | 5 | 5 | 100% | 15.5033 |
| +0.4 rad | 5 | 5 | 100% | 15.5175 |
| +0.8 rad | 5 | 5 | 100% | 10.9882 |
| Overall | 20 | 20 | 100% | 13.1497 |

Configuration A exceeds the required 80% threshold by 20 percentage points and
generalizes across every required positive and negative goal. Repeated episodes
are deterministic under fixed seeds and fixed benchmark goals.

## 7. Rule-based versus selected DQN

| Metric | Rule-based | Selected DQN |
|---|---:|---:|
| Successes | 20/20 | 20/20 |
| Success rate | 100% | 100% |
| Mean cumulative reward | 12.8666 | 13.1497 |
| Mean episode length | 24.00 | 20.75 |
| Mean final absolute error | 0.0122 rad | 0.0071 rad |
| Mean action changes | 1.00 | 6.75 |
| HOLD fraction | 68.75% | 46.99% |

The rule-based policy is more sample efficient because it needs no training and
directly encodes how to move the controller target. It is also more stable in
action space: it changes action once and then holds. The selected DQN reaches
all goals, finishes sooner on average, earns higher reward, and ends closer to
the goal, but changes actions more often. This indicates some oscillatory or
corrective behaviour even though physical success is stable.

The DQN uses HOLD appropriately, especially compared with Configuration B, but
not as consistently as the rule-based controller. A hand-written policy can
outperform a learned policy in a simple task because it has perfect structural
knowledge, while DQN estimates values from finite sampled experience. DQN's
advantage is that it learns one multi-goal mapping without being explicitly
told the target-update rule.

## 8. Recommendation, limitations, and future improvements

Configuration A is recommended. Success rate is tied, so selection uses
stability and generalization evidence: A has higher final training reward,
higher evaluation reward, lower final error, and much more HOLD use. Its extra
13.70 seconds of measured training time is immaterial relative to the five-hour
limit.

Limitations include deterministic fixed-base physics, one controlled joint,
four benchmark goals, one training seed, and no disturbances or sensor noise.
The success-rate estimate has only 20 episodes, and deterministic repeats do not
measure robustness to stochastic changes. Action switching remains higher than
the rule-based baseline.

Future work should repeat each configuration across several seeds, add a small
action-switch penalty only with instructor approval, evaluate denser unseen
goals, test random perturbations, and study Double DQN or prioritized replay as
separate approved extensions. A later robotics stage could move from fixed-base
elbow control toward multi-joint coordination and sim-to-real validation.

## Reproducibility and checkpointing

The repository contains exact commands, source files, tests, structured
training/evaluation metrics, all plots, and separate checkpoints for both
configurations. `models/selected_dqn.pt` loads Configuration A without
retraining. The evaluation script uses fixed common goals and seeds for the DQN
and rule-based policy. The video also loads this checkpoint and uses epsilon
0.0.

## AI-use acknowledgement

Generative AI assistance supported scaffolding, debugging, test design,
formatting, and draft writing. The student is responsible for verifying the
results, understanding every submitted component, and adapting the disclosure
to course and institutional requirements.

## References

1. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control
   through deep reinforcement learning. *Nature*, 518, 529-533.
   <https://doi.org/10.1038/nature14236>
2. PyTorch. Reinforcement Learning (DQN) Tutorial.
   <https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>
3. Farama Foundation. Handling Time Limits.
   <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>
4. Google DeepMind. MuJoCo Python Documentation.
   <https://mujoco.readthedocs.io/en/stable/python.html>
5. CSCN8020 Assignment 3 specification and Unitree MuJoCo G1 Primer Workshop.
