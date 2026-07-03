# Robot Learning Mini-Lab 技术报告

日期：2026-07-03

## 摘要

本项目实现了一个基于 MuJoCo `Ant-v4` 的四足机器人学习 mini-lab。技术路线不是让 PPO 从完全随机动作开始学习，而是先用传统周期步态构造一个可解释的 gait prior，再让 PPO 学习 residual correction，并通过 reward shaping 强化向前运动目标。

当前主线成果：

```text
handcrafted gait prior
-> residual PPO
-> shaped residual PPO
-> domain randomization evaluation/training
-> push recovery evaluation/training attempt
```

最重要的结果是：500k shaped residual PPO 在 20-seed、1000-step 评估中，将平均前进距离提升到 `60.34`，明显超过 handcrafted gait baseline 的 `6.59`。后续 domain-randomized training 得到了一个 strong randomization 下更鲁棒的候选策略，但 push-aware training 暂时没有超过原始 500k policy。

## 1. 项目目标

项目目标不是单纯跑通强化学习 demo，而是建立一条更接近机器人学习研究流程的技术路线：

1. 理解 MuJoCo 环境与 Ant 状态/action。
2. 用运动学和控制建立基础直觉。
3. 设计传统周期 gait prior。
4. 用 residual PPO 在传统步态上学习补偿。
5. 用 reward shaping 修正“活着但不前进”的 reward mismatch。
6. 系统评估 nominal、domain randomization、push recovery。
7. 通过 checkpoint selection 分析训练过程中策略的性能变化。

## 2. 代码结构

| 阶段 | 目录 | 内容 |
| --- | --- | --- |
| MuJoCo basics | `01_mujoco_basics/` | random rollout、state inspection、video recording |
| Kinematics | `02_kinematics/` | 2-link FK、IK、Jacobian、visualization |
| Control | `03_control/` | PD control、gain comparison、trajectory tracking |
| Gait prior | `04_gait_controller/` | sinusoidal Ant gait、video、parameter sweep |
| RL baselines | `05_rl_baselines/` | vanilla PPO、residual PPO、evaluation、video |
| Reward shaping | `06_reward_shaping/` | shaped residual PPO、reward-weight sweep |
| Domain randomization | `07_domain_randomization/` | evaluation、training、checkpoint selection |
| Push recovery | `08_push_recovery/` | push evaluation、push-aware training attempts |

## 3. 核心方法

### 3.1 Handcrafted gait prior

传统 gait prior 使用 Ant 的 8 维 action 构造周期动作：

```text
front_left + rear_right 同相
front_right + rear_left 反相
hip/knee 使用不同幅值
```

当前 gait 参数：

```text
frequency = 1.00
action_sign = -1
action_scale = 0.20
knee_action_scale = 0.10
```

这个 gait 不是最优控制器，但它提供了稳定的初始运动结构，让 PPO 不必从零开始探索四足步态。

### 3.2 Residual PPO

Residual PPO 的动作形式：

```text
final_action = handcrafted_gait_action + residual_policy_action
```

residual action 被限制在小范围：

```text
residual_action in [-0.05, 0.05]
```

这样 PPO 学的是“在已有步态上怎么补偿”，而不是重新发明整个步态。

### 3.3 Reward shaping

原始 Ant reward 容易出现“活着但不前进”的策略。因此项目加入 locomotion reward shaping：

```text
shaped_reward = original_reward
              + forward_weight * x_velocity
              - lateral_weight * abs(y_velocity)
              - energy_weight * sum(action^2)
```

当前效果最好的权重：

```text
forward_weight = 0.8
lateral_weight = 1.0
energy_weight = 0.01
```

## 4. 主要模型

| 模型 | 文件 | 定位 |
| --- | --- | --- |
| Performance best | `results/logs/ant_shaped_residual_ppo_best_500k.zip` | 当前常规 locomotion / 展示主力模型 |
| Robustness best | `results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip` | strong domain randomization 下更鲁棒的候选 |
| Push-aware attempts | `results/logs/ant_push_aware_lr5e5_100k.zip` 等 | 没有超过 original 500k，作为负结果保留 |

## 5. Locomotion Baseline 结果

评估设置：

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

结论：

```text
reward-shaped residual PPO 是当前 locomotion 性能提升的关键。
```

相比传统 gait，500k shaped residual PPO 的前进距离从 `6.59` 提升到 `60.34`。

## 6. Domain Randomization 实验

### 6.1 Evaluation

最初只是把 500k performance policy 拿去不同物理参数下测试：

| Condition | Success | Mean x | Mean abs y | Forward score |
| --- | ---: | ---: | ---: | ---: |
| nominal | 0.80 | 56.73 | 2.49 | 56.24 |
| moderate randomization | 0.80 | 55.41 | 2.34 | 54.94 |
| strong randomization | 0.50 | 40.87 | 2.55 | 40.36 |

结论：原始 500k policy 对中等参数扰动比较稳，但 strong randomization 下性能下降明显。

### 6.2 Training 与 checkpoint selection

直接做 50k domain-randomized continuation 没有变好，说明“加随机化”不自动等于“更鲁棒”。

后续采用更温和的设置：

```text
model_in = 500k performance policy
learning_rate = 1e-4
mass = 0.9-1.1
friction = 0.85-1.15
damping = 0.85-1.15
checkpoint_freq = 25k
```

再从较好的 +25k checkpoint 出发：

```text
learning_rate = 5e-5
train = 200k
checkpoint_freq = 25k
```

最终选出 robustness candidate：

```text
results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip
```

## 7. Performance vs Robustness 正式对比

正式 benchmark 使用 20 seeds、1000 max steps。

| Policy | Condition | Success | Mean x | Mean abs y | Forward score |
| --- | --- | ---: | ---: | ---: | ---: |
| Performance | nominal | 0.90 | 60.34 | 2.30 | 59.88 |
| Performance | moderate | 0.85 | 56.15 | 2.48 | 55.65 |
| Performance | strong | 0.65 | 44.24 | 2.95 | 43.65 |
| Robustness | nominal | 0.80 | 53.74 | 3.45 | 53.05 |
| Robustness | moderate | 0.70 | 52.39 | 3.82 | 51.62 |
| Robustness | strong | 0.75 | 52.48 | 4.65 | 51.55 |

结论：

- Performance policy 在 nominal 和 moderate 下最好。
- Robustness policy 在 strong randomization 下更强。
- 鲁棒性提升带来了常规性能下降，这是典型 tradeoff。

汇总图：

```text
results/plots/formal_policy_benchmark.png
```

## 8. Push Recovery 实验

### 8.1 Evaluation 结论

Performance policy 的 push recovery 表现：

| Condition | Success | Recovery | Forward score |
| --- | ---: | ---: | ---: |
| no push | 0.80 | 0.80 | 56.24 |
| +x 10N | 1.00 | 1.00 | 60.33 |
| -x 10N | 0.90 | 0.00 | 16.11 |
| +x 25N | 0.00 | 0.00 | 22.25 |

最明显的问题是：

```text
-x 10N backward push: survival 还可以，但 recovery = 0.00
```

也就是说，机器人没立刻摔，但被向后推后不能重新恢复向前运动。

### 8.2 Push-aware training 尝试

做了三类训练：

1. 随机 push-aware training，`push_probability=0.8`，`force=5-10N`。
2. 温和 push-aware curriculum，`push_probability=0.3`，`force=3-5N`。
3. 固定 `-x 10N` recovery training，并加入 post-push recovery reward。

结果：

| Attempt | 最好结果 | 判断 |
| --- | --- | --- |
| random push-aware | small recovery 下降到约 0.48 | 失败，破坏 gait |
| gentle push-aware | small recovery 最好约 0.57 | 仍不如 original 500k |
| fixed -x 10N recovery | -x10 recovery 从 0.00 到 0.10 | 有一点信号，但不可用 |

当前结论：

```text
Domain randomization 可以靠物理参数随机化改善；
push recovery 不能只靠随机外力 continuation，需要更明确的恢复奖励和更窄的 curriculum。
```

## 9. 视频与可视化产物

Performance policy 1000-step video：

```text
results/videos/formal_benchmark_performance_best_1000step.mp4
x_displacement = 62.5334
y_displacement = 4.6134
```

Robustness policy 1000-step video：

```text
results/videos/formal_benchmark_robustness_best_1000step.mp4
x_displacement = 65.0841
y_displacement = -4.4961
```

主要图表：

```text
results/plots/stage_03_results_summary.png
results/plots/formal_policy_benchmark.png
results/plots/locomotion_baseline_comparison_500k.png
```

## 10. 技术结论

### 10.1 成功结论

1. 传统 gait prior + residual PPO 比 vanilla PPO 更适合这个学习路径。
2. Reward shaping 是让 Ant 真正向前走的关键。
3. 500k shaped residual PPO 是当前最强常规 locomotion policy。
4. Domain-randomized training 配合低学习率、窄范围 curriculum、checkpoint selection，可以得到 strong randomization 下更强的策略。
5. Checkpoint selection 很重要，最后一个模型不一定最好。

### 10.2 失败与负结果

1. Vanilla PPO 可以活很久，但几乎不前进。
2. 直接 domain-randomized continuation 会破坏原策略。
3. 直接 push-aware random training 没有改善 push recovery。
4. 固定 `-x 10N` recovery reward 有一点信号，但仍然远不足以形成新 best。
5. Domain robustness 和 push recovery 是两类问题，不能互相替代。

## 11. 当前推荐模型

| 目标 | 推荐模型 |
| --- | --- |
| 常规 locomotion / 视频展示 / portfolio 主结果 | `results/logs/ant_shaped_residual_ppo_best_500k.zip` |
| strong domain randomization 研究 | `results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip` |
| push recovery | 暂无新 best，仍以 original 500k 作为 baseline |

## 12. 后续计划

短期建议：

1. 暂停无目标继续训练。
2. 把当前项目整理成 portfolio 材料。
3. 保留 performance/robustness 两条模型线。
4. 如果继续 push recovery，优先改 reward/curriculum，而不是单纯加训练步数。

Push recovery 下一步更合理的路线：

```text
从更稳定的 100k shaped policy 出发
-> 只训练 -x 5N
-> 加小 recovery reward
-> checkpoint selection
-> 再扩展到 -x 10N
-> 最后扩展到随机方向 5-10N
```

中长期路线：

1. 迁移到更真实的 quadruped XML 或 Unitree 类模型。
2. 增加 observation 中的 push phase / disturbance estimate。
3. 做 actuator delay、sensor noise、terrain variation。
4. 后续再考虑 ROS2 接口，用于系统集成展示。

## 13. 项目表述草稿

英文：

```text
Built a MuJoCo Ant-v4 robot learning mini-lab combining handcrafted gait priors,
residual PPO, and reward shaping. The shaped residual policy improved 1000-step
forward displacement from 6.6 for a handcrafted gait baseline to 60.3 across
20 evaluation seeds. A domain-randomized continuation with checkpoint selection
improved strong-randomization robustness, increasing strong-domain forward score
from 43.7 to 51.6. Push-recovery experiments showed that domain robustness does
not automatically transfer to disturbance recovery, motivating a separate
recovery-reward curriculum.
```

中文：

```text
本项目实现了一个基于 MuJoCo Ant-v4 的四足机器人学习 mini-lab。项目采用传统周期步态作为 gait prior，并通过 residual PPO 与 reward shaping 学习补偿策略。当前最强 performance policy 在 20 个 seed、1000-step 评估中达到 60.34 的平均前进距离，明显超过传统 gait baseline。进一步的 domain-randomized continuation 和 checkpoint selection 得到了 strong randomization 下更鲁棒的候选策略。同时，push recovery 实验表明物理参数鲁棒性并不会自动转化为外力恢复能力，因此后续需要专门的 recovery reward 和扰动课程。
```
