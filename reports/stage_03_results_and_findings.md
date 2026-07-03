# Robot Learning Mini-Lab 阶段性成果与实验结论

日期：2026-07-02

## 1. 当前项目定位

这个项目目前已经从“跑通 MuJoCo 和 PPO”推进到了一个比较完整的四足机器人学习 mini-lab：

```text
MuJoCo Ant-v4
-> 运动学与控制基础
-> 传统周期步态
-> residual PPO
-> reward shaping
-> baseline comparison
-> domain randomization / push recovery robustness check
```

核心路线是：先用传统机电和控制直觉构造一个可解释的 gait prior，再让 PPO 学 residual correction，而不是让 PPO 从完全随机动作开始摸索。

```text
final_action = handcrafted_gait_action + residual_policy_action
```

这条路线的意义是：传统控制负责给机器人一个稳定、有结构的初始行为，强化学习负责在这个基础上补偿速度、姿态和轨迹误差。

## 2. 已完成内容

| 阶段 | 内容 | 当前产物 |
| --- | --- | --- |
| 01 MuJoCo basics | Ant-v4 随机 rollout、状态检查、视频录制 | `01_mujoco_basics/` |
| 02 Kinematics | 2-link FK、IK、Jacobian、可视化 | `02_kinematics/` |
| 03 Control | 单关节 PD、增益对比、轨迹跟踪 | `03_control/` |
| 04 Gait controller | Ant-like sinusoidal gait、视频、参数扫描 | `04_gait_controller/` |
| 05 RL baselines | vanilla PPO、residual PPO、policy evaluation | `05_rl_baselines/` |
| 06 Reward shaping | shaped residual PPO、reward 权重扫描 | `06_reward_shaping/` |
| 07 Robustness | domain randomization evaluation | `07_domain_randomization/` |
| 08 Disturbance | push recovery evaluation | `08_push_recovery/` |

当前最强模型：

```text
results/logs/ant_shaped_residual_ppo_best_500k.zip
```

对应视频：

```text
results/videos/ant_shaped_residual_ppo_best_500k.mp4
```

## 3. 技术路径总结

### 3.1 传统 gait prior

我们先写了一个 sinusoidal trot-like gait，把 Ant 的 8 个关节动作组织成周期信号：

```text
front_left + rear_right 同相
front_right + rear_left 反相
hip/knee 使用不同幅值
```

通过参数扫描得到一个比较稳定的传统步态：

```text
frequency = 1.00
action_sign = -1
action_scale = 0.20
knee_action_scale = 0.10
```

这个 gait 本身不是最优策略，但它给 PPO 提供了一个“已经会动”的起点。

### 3.2 Residual PPO

Residual PPO 不直接输出完整 8 维 action，而是输出小范围 correction：

```text
residual_action in [-0.05, 0.05]
```

这样可以避免早期随机策略把传统 gait 完全破坏。我们之前观察到 residual 太大时，机器人容易活着但失去明确前进趋势，所以最终保留了比较保守的 residual range。

### 3.3 Reward shaping

原始 Ant reward 容易让策略学到“活着但不怎么向前”的行为，因此我们加入了更明确的 locomotion reward：

```text
shaped_reward = original_reward
              + forward_weight * x_velocity
              - lateral_weight * abs(y_velocity)
              - energy_weight * sum(action^2)
```

当前使用效果最好的权重：

```text
forward_weight = 0.8
lateral_weight = 1.0
energy_weight = 0.01
```

## 4. 主要实验结果

### 4.1 Locomotion baseline comparison

评估设置：

```text
Ant-v4
1000 max steps
20 seeds
deterministic policy
```

结果文件：

```text
results/tables/locomotion_baseline_comparison_best_500k_20seed_1000step.csv
```

| Policy | Episodes | Mean return | Mean length | Success | Mean x | Mean abs y | Forward score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shaped residual PPO | 20 | 2097.44 | 940.45 | 0.90 | 60.34 | 2.30 | 59.88 |
| handcrafted gait | 20 | 1081.72 | 1000.00 | 1.00 | 6.59 | 9.23 | 4.74 |
| vanilla PPO | 20 | 937.08 | 1000.00 | 1.00 | 0.01 | 0.15 | -0.02 |
| residual PPO | 20 | 892.39 | 1000.00 | 1.00 | -2.35 | 5.33 | -3.42 |
| random | 20 | -14.38 | 76.20 | 0.00 | 0.61 | 0.98 | 0.41 |

结论：500k shaped residual PPO 是目前最强 locomotion policy。它的平均前进距离约 60.34，明显超过传统 gait 的 6.59，也说明 reward shaping 对“真正向前走”非常关键。

### 4.2 100k vs 500k 的 tradeoff

100k shaped residual PPO 更稳，500k shaped residual PPO 更快、更直，但长 horizon 成功率略降。

| Model | Evaluation | Mean x | Mean abs y | Forward score | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| 100k shaped residual PPO | 1000 steps, 10 seeds | 47.04 | 11.62 | 44.72 | 1.00 |
| 500k shaped residual PPO | 1000 steps, 20 seeds | 60.34 | 2.30 | 59.88 | 0.90 |

结论：继续训练带来了更激进、更强的高速步态，但也引入少量稳定性风险。后续可以把 100k 当作 safer policy，把 500k 当作 performance policy。

### 4.3 Domain randomization

评估对象：500k shaped residual PPO。

| Condition | Randomization range | Success | Mean x | Mean abs y | Forward score |
| --- | --- | ---: | ---: | ---: | ---: |
| nominal | none | 0.80 | 56.73 | 2.49 | 56.24 |
| moderate | mass 0.8-1.2, friction 0.7-1.3, damping 0.7-1.3 | 0.80 | 55.41 | 2.34 | 54.94 |
| strong | mass 0.6-1.4, friction 0.5-1.5, damping 0.5-1.5 | 0.50 | 40.87 | 2.55 | 40.36 |

结论：中等程度的物理参数扰动对当前 policy 影响不大；强随机化会明显降低成功率。说明 policy 有一定鲁棒性，但还不是 robust locomotion policy。

### 4.4 Push recovery

评估对象：500k shaped residual PPO。

小扰动实验结果：

| Condition | Success | Recovery | Mean x | Mean post-push x | Forward score |
| --- | ---: | ---: | ---: | ---: | ---: |
| no push | 0.80 | 0.80 | 57.82 | 39.33 | 57.25 |
| +x 5N | 1.00 | 1.00 | 64.20 | 45.28 | 63.78 |
| -x 5N | 0.80 | 0.80 | 56.44 | 38.36 | 56.11 |
| +x 10N | 1.00 | 1.00 | 64.76 | 45.41 | 64.38 |
| -x 10N | 1.00 | 0.00 | 16.37 | -1.27 | 15.95 |
| +x 25N | 0.00 | 0.00 | 20.96 | 8.55 | 20.64 |
| +x 50N | 0.00 | 0.00 | 22.74 | 22.74 | 22.39 |

结论：当前 policy 可以恢复部分小扰动，尤其是顺向推力；但对反向 10N 推力已经出现 recovery failure，对 25N 及以上扰动基本不具备可靠恢复能力。

## 5. 当前图表

汇总图：

```text
results/plots/stage_03_results_summary.png
```

生成方式：

```bash
python reports/plot_stage_03_summary.py
```

这张图同时展示：

- 500k shaped residual PPO 与各 baseline 的 locomotion score；
- nominal / moderate / strong domain randomization 下的性能变化；
- push recovery 中成功率和恢复率的边界。

## 6. 目前尝试得到的结论

| 尝试 | 观察 | 结论 |
| --- | --- | --- |
| vanilla PPO | episode 能活很久，但几乎不前进 | 原始 reward 不足以保证学到 locomotion |
| handcrafted gait | 能稳定运动，但速度和方向都有限 | 传统控制可以作为不错的 prior |
| residual PPO | 如果只加 residual，不改 reward，提升有限 | residual 结构需要配合目标明确的 reward |
| reward shaping | 前进距离大幅提升 | shaping 是当前性能提升的关键 |
| 继续训练到 500k | 更快、更直，但成功率从 100% 降到 90% 左右 | 性能和稳定性之间存在 tradeoff |
| domain randomization | 中等扰动影响小，强扰动明显掉性能 | 已补训练脚本，下一步需要长训练和复测 |
| push recovery | 小扰动可以，反向/大扰动不稳 | 当前 policy 不是抗扰控制器 |

## 7. 发现的问题

1. 当前最强策略主要优化前进 locomotion，不是专门的 balance recovery。
2. 500k 模型虽然强，但在 1000-step 评估里仍有终止风险。
3. domain-randomized training 脚本已经补上，但还没有进行长时间训练和复测。
4. push recovery 目前还是 evaluation，policy 没有见过外力扰动训练。
5. 结果还集中在 Ant-v4，没有迁移到更真实的 quadruped morphology 或 ROS2/Isaac/MuJoCo XML 自定义机器人。

## 8. 下一步计划

建议下一阶段不要只做“再多训练一点”，而是围绕 robustness 和项目表达继续推进：

| 优先级 | 下一步 | 目的 |
| --- | --- | --- |
| 高 | 生成完整阶段报告和结果图 | 把项目成果沉淀成 portfolio 材料 |
| 高 | 录制 1000-step best policy video | 有可视化证据展示 learned locomotion |
| 已开始 | domain-randomized training | 让策略训练时就见到 mass/friction/damping 变化 |
| 中 | push-aware training | 训练中加入小扰动，提高 recovery ability |
| 中 | 对比 100k safer policy 和 500k performance policy | 明确稳定性和速度 tradeoff |
| 低 | 后续再接 ROS2 或更真实机器人模型 | 做系统集成，不急于当前阶段 |

## 9. Portfolio 表述草稿

```text
Built a MuJoCo Ant-v4 robot learning mini-lab combining handcrafted gait priors,
residual PPO, and reward shaping. The shaped residual policy improved 1000-step
forward displacement from 6.6 for a handcrafted gait baseline to 60.3 across
20 evaluation seeds. Additional robustness tests showed that the learned policy
remains effective under moderate mass/friction/damping randomization, while
push-recovery tests revealed clear limitations under backward and large external
disturbances.
```

中文版本：

```text
本项目实现了一个基于 MuJoCo Ant-v4 的四足机器人学习 mini-lab。技术路线是先用传统周期步态构造 gait prior，再用 residual PPO 和 reward shaping 学习补偿策略。当前最强 shaped residual PPO 在 20 个 seed、1000-step 评估中将平均前进距离提升到 60.3，明显超过传统 gait baseline 的 6.6。同时，项目进一步评估了 domain randomization 和 push recovery，发现策略具有初步物理参数鲁棒性，但在反向和大幅外力扰动下仍存在恢复能力不足的问题。
```

## 10. 追加更新：Domain-randomized training

已经新增训练脚本：

```text
07_domain_randomization/train_domain_randomized_shaped_residual_ppo.py
```

它和之前的 `evaluate_domain_randomization.py` 不同：之前是在训练后测试不同物理参数；现在是在每个训练 episode reset 时随机化 MuJoCo 的 body mass、ground friction 和 joint damping。

默认范围：

```text
mass = 0.8-1.2
friction = 0.7-1.3
damping = 0.7-1.3
```

烟雾测试运行方式：

```bash
python 07_domain_randomization/train_domain_randomized_shaped_residual_ppo.py --total-timesteps 5000 --n-steps 512
```

从当前 500k best policy 继续训练：

```bash
python 07_domain_randomization/train_domain_randomized_shaped_residual_ppo.py \
  --model-in results/logs/ant_shaped_residual_ppo_best_500k.zip \
  --total-timesteps 50000 \
  --output results/logs/ant_domain_randomized_shaped_residual_ppo_50k
```

实际已经完成一次 50k continuation：

```text
results/logs/ant_domain_randomized_shaped_residual_ppo_50k.zip
```

复测结果：

| Model | Condition | Success | Mean x | Mean abs y | Forward score |
| --- | --- | ---: | ---: | ---: | ---: |
| original 500k | nominal | 0.80 | 56.73 | 2.49 | 56.24 |
| original 500k | moderate randomized | 0.80 | 55.41 | 2.34 | 54.94 |
| original 500k | strong randomized | 0.50 | 40.87 | 2.55 | 40.36 |
| DR continuation 50k | nominal | 0.70 | 49.63 | 4.58 | 48.71 |
| DR continuation 50k | moderate randomized | 0.70 | 35.83 | 4.20 | 34.99 |
| DR continuation 50k | strong randomized | 0.50 | 38.80 | 6.37 | 37.53 |

结论：这次 50k domain-randomized continuation 没有超过原始 500k policy。它说明 domain-randomized training 不是简单“加随机化就会变强”，而是需要更仔细的 curriculum 和超参数设置。当前不建议替换 best policy，仍然保留：

```text
results/logs/ant_shaped_residual_ppo_best_500k.zip
```

下一次优化方向：

1. 降低 learning rate，避免 continuation 把原策略破坏太快。
2. 先用窄范围 randomization，再逐步扩大范围。
3. 混合 nominal 和 randomized episodes，不要一开始每个 episode 都随机。
4. 每隔固定步数同时评估 nominal、moderate、strong 三组，选择综合最好的 checkpoint。

## 11. 追加实验：窄范围 + 低学习率 continuation

按照上面的判断，继续做了一次更温和的 domain-randomized continuation：

```text
model_in = results/logs/ant_shaped_residual_ppo_best_500k.zip
total_timesteps = 100000
learning_rate = 0.0001
mass = 0.9-1.1
friction = 0.85-1.15
damping = 0.85-1.15
```

输出模型：

```text
results/logs/ant_domain_randomized_narrow_lr1e4_100k.zip
```

复测结果：

| Model | Condition | Success | Mean x | Mean abs y | Forward score |
| --- | --- | ---: | ---: | ---: | ---: |
| original 500k | nominal | 0.80 | 56.73 | 2.49 | 56.24 |
| original 500k | moderate randomized | 0.80 | 55.41 | 2.34 | 54.94 |
| original 500k | strong randomized | 0.50 | 40.87 | 2.55 | 40.36 |
| narrow LR 100k | nominal | 0.70 | 58.74 | 6.98 | 57.34 |
| narrow LR 100k | moderate randomized | 0.50 | 46.11 | 5.71 | 44.96 |
| narrow LR 100k | strong randomized | 0.70 | 43.91 | 6.47 | 42.62 |

结论：这次比直接 moderate randomization continuation 更合理。它在 strong randomization 下有改善，success 从 0.50 到 0.70，forward score 从 40.36 到 42.62；但 moderate randomization 下仍然弱于原 500k。说明低学习率和窄范围 curriculum 是正确方向，但还没有得到全面更优的 robust policy。

目前策略选择：

```text
performance best: results/logs/ant_shaped_residual_ppo_best_500k.zip
strong-randomization candidate: results/logs/ant_domain_randomized_narrow_lr1e4_100k.zip
```

下一步如果继续优化，建议不要直接替换 best policy，而是做 checkpoint selection：每训练 25k-50k 保存一次，同时评估 nominal、moderate、strong，选择综合分最高的模型。

## 12. 追加实验：Checkpoint selection

为了验证“最后一个模型不一定最好”，新增了 checkpoint selection 流程：

```text
07_domain_randomization/run_domain_checkpoint_selection.py
07_domain_randomization/select_domain_randomization_checkpoint.py
```

训练设置：

```text
model_in = results/logs/ant_shaped_residual_ppo_best_500k.zip
total_timesteps = 100000
checkpoint_freq = 25000
learning_rate = 0.0001
mass = 0.9-1.1
friction = 0.85-1.15
damping = 0.85-1.15
```

保存的 checkpoint：

```text
results/logs/dr_checkpoint_selection_narrow_lr1e4/narrow_lr1e4_525736_steps.zip
results/logs/dr_checkpoint_selection_narrow_lr1e4/narrow_lr1e4_550736_steps.zip
results/logs/dr_checkpoint_selection_narrow_lr1e4/narrow_lr1e4_575736_steps.zip
results/logs/dr_checkpoint_selection_narrow_lr1e4/narrow_lr1e4_600736_steps.zip
```

5-seed 快速 selection 排名：

| Checkpoint | Robust score | Nominal | Moderate | Strong | Avg success |
| --- | ---: | ---: | ---: | ---: | ---: |
| original 500k | 61.58 | 57.25 | 59.00 | 48.22 | 0.73 |
| narrow +25k | 61.28 | 56.82 | 62.00 | 43.43 | 0.80 |
| narrow +100k checkpoint | 57.80 | 63.90 | 56.37 | 35.23 | 0.80 |
| final 100k | 54.52 | 53.78 | 53.41 | 39.29 | 0.67 |
| narrow +75k | 49.36 | 44.42 | 55.83 | 35.11 | 0.47 |
| narrow +50k | 46.23 | 36.67 | 44.41 | 42.14 | 0.47 |

这个结果说明：训练过程确实不是单调变好的。+25k checkpoint 和原 500k 很接近；后续 +50k、+75k、final 100k 反而变差。

对 +25k checkpoint 做了 10-seed 复测：

| Model | Condition | Success | Mean x | Mean abs y | Forward score |
| --- | --- | ---: | ---: | ---: | ---: |
| original 500k | nominal | 0.80 | 56.73 | 2.49 | 56.24 |
| original 500k | moderate randomized | 0.80 | 55.41 | 2.34 | 54.94 |
| original 500k | strong randomized | 0.50 | 40.87 | 2.55 | 40.36 |
| narrow +25k | nominal | 0.80 | 54.70 | 2.04 | 54.29 |
| narrow +25k | moderate randomized | 0.80 | 55.33 | 1.62 | 55.01 |
| narrow +25k | strong randomized | 0.60 | 45.83 | 2.68 | 45.29 |

结论：

```text
performance / original best:
results/logs/ant_shaped_residual_ppo_best_500k.zip

robustness candidate:
results/logs/dr_checkpoint_selection_narrow_lr1e4/narrow_lr1e4_525736_steps.zip
```

原 500k 仍然是整体非常强的 baseline；+25k checkpoint 的 nominal 略低，但 strong randomization 下更好，说明它确实朝鲁棒性方向移动了一点。后续如果目标是 robustness，可以围绕 +25k checkpoint 继续做更细的 curriculum，而不是继续无脑训练到 100k。

## 13. 追加实验：从 +25k 出发继续低学习率训练

为了回答“是不是训练步数不够”，从上一轮 robustness candidate 继续训练：

```text
model_in = results/logs/dr_checkpoint_selection_narrow_lr1e4/narrow_lr1e4_525736_steps.zip
total_timesteps = 200000
checkpoint_freq = 25000
learning_rate = 0.00005
mass = 0.9-1.1
friction = 0.85-1.15
damping = 0.85-1.15
```

输出模型：

```text
results/logs/ant_domain_randomized_from_25k_lr5e5_200k.zip
```

checkpoint 目录：

```text
results/logs/dr_checkpoint_selection_from_25k_lr5e5/
```

5-seed selection 排名前几：

| Checkpoint | Robust score | Nominal | Moderate | Strong |
| --- | ---: | ---: | ---: | ---: |
| from25k +50k | 66.30 | 57.21 | 53.45 | 61.56 |
| final 200k | 63.41 | 56.64 | 52.03 | 57.59 |
| from25k +75k | 62.15 | 53.02 | 64.33 | 44.28 |
| original 500k | 61.58 | 57.25 | 59.00 | 48.22 |

其中 `from25k +50k` 对应文件：

```text
results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip
```

对这个 checkpoint 做 10-seed 复测：

| Model | Condition | Success | Mean x | Mean abs y | Forward score |
| --- | --- | ---: | ---: | ---: | ---: |
| original 500k | nominal | 0.80 | 56.73 | 2.49 | 56.24 |
| original 500k | moderate randomized | 0.80 | 55.41 | 2.34 | 54.94 |
| original 500k | strong randomized | 0.50 | 40.87 | 2.55 | 40.36 |
| from25k +50k | nominal | 0.70 | 54.23 | 3.92 | 53.44 |
| from25k +50k | moderate randomized | 0.80 | 49.82 | 3.81 | 49.06 |
| from25k +50k | strong randomized | 0.80 | 54.43 | 4.16 | 53.60 |

结论：训练步数确实有作用。继续从 +25k candidate 出发，用更低学习率训练后，找到了一个明显更强的 strong-randomization policy：strong score 从原 500k 的 40.36 提升到 53.60，success 从 0.50 提升到 0.80。但代价是 nominal 和 moderate 表现下降。

当前模型选择：

```text
performance best:
results/logs/ant_shaped_residual_ppo_best_500k.zip

robustness best:
results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip
```

这说明后续如果项目目标是“鲁棒四足 locomotion”，应该以 robustness best 为起点继续优化；如果目标是视频展示和最快前进，仍然使用 performance best。
