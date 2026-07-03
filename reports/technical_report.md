# A MuJoCo Mini-Lab for Studying Gait Priors, Residual PPO, and Robustness in Ant Locomotion

Date: 2026-07-03

## 1. Abstract

This project is a MuJoCo `Ant-v4` robot learning mini-lab for studying
handcrafted gait priors, residual PPO, reward shaping, domain randomization,
and push recovery. The central idea is to avoid training PPO from fully random
motor commands. Instead, a handcrafted sinusoidal gait provides a structured
locomotion prior, and PPO learns small residual corrections around that prior.

The best shaped residual policy reached a mean forward displacement of `60.34`
over 20 evaluation seeds and 1000 simulation steps, compared with `6.59` for
the handcrafted gait baseline. A domain-randomized continuation with checkpoint
selection improved strong-randomization robustness, but push-aware training did
not outperform the original 500k policy. These negative results are informative:
domain robustness and push recovery are related but distinct problems.

## 2. Introduction

Learning quadruped locomotion from scratch is difficult because random actions
often produce unstable or uninformative behavior. This mini-lab studies a more
structured approach: start with a simple gait prior, then let reinforcement
learning improve it.

The project is intentionally scoped as a learning and research-style portfolio
project. `Ant-v4` is not a real robot, but it is useful for studying locomotion
policy design, reward shaping, robustness testing, and failure analysis.

## 3. Method

### 3.1 Handcrafted Gait Prior

The handcrafted prior is a sinusoidal Ant gait. It coordinates diagonal leg
pairs and uses separate amplitudes for hip and knee-like joints:

```text
front_left + rear_right: same phase
front_right + rear_left: opposite phase
```

The current gait parameters are:

```text
frequency = 1.00
action_sign = -1
action_scale = 0.20
knee_action_scale = 0.10
```

This prior is not intended to be optimal. Its role is to give learning a
structured starting point.

### 3.2 Residual PPO

The learned policy outputs a residual action added to the handcrafted gait:

```text
final_action = handcrafted_gait_action + residual_policy_action
```

The residual range is deliberately small:

```text
residual_action in [-0.05, 0.05]
```

This keeps early PPO updates from destroying the gait prior and makes the
learning problem closer to policy refinement than full behavior discovery.

### 3.3 Reward Shaping

The original Ant reward can produce policies that survive but barely move
forward. The shaped reward adds directed locomotion terms:

```text
shaped_reward = original_reward
              + forward_weight * x_velocity
              - lateral_weight * abs(y_velocity)
              - energy_weight * sum(action^2)
```

The strongest setting used in the project is:

```text
forward_weight = 0.8
lateral_weight = 1.0
energy_weight = 0.01
```

### 3.4 Domain-Randomized Continuation

Domain-randomized continuation randomizes MuJoCo physical parameters during
training, including body mass, ground friction, and joint damping. Early direct
continuation was not consistently better, so the project used a narrower
curriculum, lower learning rates, and checkpoint selection.

The best robustness candidate is:

```text
results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip
```

### 3.5 Push Recovery Evaluation

Push recovery is evaluated by applying short external forces to the Ant torso
through MuJoCo `xfrc_applied`. The recovery metric checks whether the policy
survives and continues moving forward after the push.

This is different from domain randomization. A policy can be robust to mass or
friction changes while still failing to recover from a short external impulse.

## 4. Experiments

### 4.1 Locomotion Baseline Comparison

Evaluation setup:

```text
Ant-v4
20 seeds
1000 max steps
deterministic policy
```

| Policy | Success | Mean x | Mean abs y | Forward score |
| --- | ---: | ---: | ---: | ---: |
| shaped residual PPO 500k | 0.90 | 60.34 | 2.30 | 59.88 |
| handcrafted gait | 1.00 | 6.59 | 9.23 | 4.74 |
| vanilla PPO | 1.00 | 0.01 | 0.15 | -0.02 |
| residual PPO | 1.00 | -2.35 | 5.33 | -3.42 |
| random | 0.00 | 0.61 | 0.98 | 0.41 |

### 4.2 Performance vs Robustness Benchmark

Two policies were compared:

- Performance policy: `results/logs/ant_shaped_residual_ppo_best_500k.zip`
- Robustness policy: `results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip`

| Policy | Condition | Success | Mean x | Mean abs y | Forward score |
| --- | --- | ---: | ---: | ---: | ---: |
| Performance | nominal | 0.90 | 60.34 | 2.30 | 59.88 |
| Performance | moderate | 0.85 | 56.15 | 2.48 | 55.65 |
| Performance | strong | 0.65 | 44.24 | 2.95 | 43.65 |
| Robustness | nominal | 0.80 | 53.74 | 3.45 | 53.05 |
| Robustness | moderate | 0.70 | 52.39 | 3.82 | 51.62 |
| Robustness | strong | 0.75 | 52.48 | 4.65 | 51.55 |

### 4.3 Push Recovery Stress Test

Representative performance-policy results:

| Condition | Success | Recovery | Forward score |
| --- | ---: | ---: | ---: |
| no push | 0.80 | 0.80 | 56.24 |
| +x 10N | 1.00 | 1.00 | 60.33 |
| -x 10N | 0.90 | 0.00 | 16.11 |
| +x 25N | 0.00 | 0.00 | 22.25 |

The most important failure mode is `-x 10N`: the policy often survives but does
not recover forward progress.

### 4.4 Push-Aware Training Attempts

Three push-aware variants were tested:

1. Random push-aware continuation with 5-10N pushes and high push probability.
2. A gentler random-push curriculum with 3-5N pushes and lower push probability.
3. A fixed `-x 10N` recovery-reward curriculum.

None outperformed the original 500k policy. The fixed `-x 10N` attempt produced
a small positive signal, increasing `-x 10N` recovery from `0.00` to `0.10`, but
it also reduced no-push locomotion quality.

## 5. Results

The main positive result is that shaped residual PPO turns a weak handcrafted
gait into a strong forward-locomotion policy. The main robustness result is that
domain-randomized continuation can improve strong-randomization performance
when combined with lower learning rates and checkpoint selection.

The main negative result is that push recovery did not improve by simply adding
external pushes during PPO continuation. Recovery appears to require a more
specific reward and curriculum.

## 6. Discussion

The project highlights several practical lessons:

- A gait prior makes policy search more structured.
- Reward shaping is essential when the default environment reward does not
  align with the intended behavior.
- More training is not automatically better; checkpoint selection matters.
- Robustness is multidimensional. Parameter robustness and impulse recovery
  should be evaluated separately.
- Negative results are useful when they identify a mismatch between the
  training objective and the evaluation objective.

## 7. Limitations

- `Ant-v4` is a benchmark morphology, not a real quadruped robot.
- Results are based on simulation only.
- The project does not yet report confidence intervals or full learning curves.
- Push-aware training did not outperform the original 500k policy.
- The policies do not include actuator delay, sensor noise, terrain variation,
  or realistic hardware constraints.

## 8. Future Work

- Run an ablation for shaped PPO without the gait prior.
- Add checkpoint learning curves and mean/std or confidence intervals.
- Try a more realistic quadruped XML or Unitree-like morphology.
- Add actuator delay, sensor noise, and terrain variation.
- Improve push recovery with a narrower recovery curriculum and a reward term
  tied directly to post-push recovery.
- Add a ROS2 interface later for system-integration demonstration.

## 9. Reproducibility Notes

Environment setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Key commands:

```bash
python 05_rl_baselines/compare_locomotion_baselines.py --episodes 20 --max-steps 1000 --shaped-residual-model results/logs/ant_shaped_residual_ppo_best_500k.zip --deterministic
python 07_domain_randomization/evaluate_domain_randomization.py --model results/logs/ant_shaped_residual_ppo_best_500k.zip --episodes 20 --max-steps 1000
python 08_push_recovery/evaluate_push_recovery.py --model results/logs/ant_shaped_residual_ppo_best_500k.zip --episodes 10 --max-steps 1000 --push-forces 5,10,25,50 --directions +x,-x,+y,-y
python reports/plot_formal_benchmark.py
```

Video recording:

```bash
python 05_rl_baselines/record_residual_policy_video.py \
  --model results/logs/ant_shaped_residual_ppo_best_500k.zip \
  --steps 1000 \
  --deterministic \
  --output results/videos/formal_benchmark_performance_best_1000step.mp4
```

On macOS, MuJoCo video rendering may need a normal Terminal or VS Code terminal
so the renderer can access the graphics stack.
