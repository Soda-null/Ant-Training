# Robot Learning Mini-Lab 阶段性报告 02

日期：2026-07-02

## 1. 阶段目标

这一阶段的目标是从普通 PPO demo 往更像机器人学习项目的路线推进：

```text
传统 gait prior -> residual PPO -> reward shaping -> baseline comparison
```

核心想法不是让 PPO 从随机动作开始摸索，而是先用传统机电/控制直觉做出一个能走的周期步态，再让 PPO 学一个小的 residual correction。

最终动作形式是：

```text
final_action = handcrafted_gait_action + residual_policy_action
```

这样 PPO 学的不是“从零开始走路”，而是：

- 身体偏了怎么补
- 速度不够怎么推
- 横漂太大怎么修
- 怎么在已有周期步态上改得更好

## 2. 当前技术路径

### 2.1 传统步态先验

先用 Ant 的 8 维 action 构造一个 sinusoidal trot-like gait：

```text
front_left + rear_right 同相
front_right + rear_left 反相
hip/knee 使用不同幅值
```

然后用参数搜索代替手调：

```text
frequency
action_sign
hip action amplitude
knee action amplitude
```

当前用于 residual PPO 的 gait prior 是：

```text
frequency = 1.00
action_sign = -1
action_scale = 0.20
knee_action_scale = 0.10
```

这个 prior 的意义是给 PPO 一个已经能稳定向前的起点。

### 2.2 Residual PPO

Residual PPO 使用一个 wrapper，把 PPO 的 action space 限制成小范围 residual：

```text
residual_action in [-0.05, 0.05]
```

这样做的原因是：如果 residual 太大，PPO 早期随机策略会直接破坏传统步态。之前测试过 `residual_scale=0.15`，结果机器人能活着但容易失去前进趋势，所以后续默认改为 `0.05`。

### 2.3 Reward Shaping

原始 Ant reward 有一个问题：机器人只要不摔，也能拿到不少奖励。因此 vanilla PPO 和 early residual PPO 可能学到“活着但不向前”的策略。

为了解决这个目标错位，加入 shaped reward：

```text
shaped_reward = original_reward
              + forward_weight * x_velocity
              - lateral_weight * abs(y_velocity)
              - energy_weight * sum(action^2)
```

当前 best setting：

```text
forward_weight = 0.8
lateral_weight = 1.0
energy_weight = 0.01
```

## 3. 已完成代码

传统 gait:

- `04_gait_controller/sinusoidal_gait_controller.py`
- `04_gait_controller/record_gait_video.py`
- `04_gait_controller/sweep_gait_params.py`

PPO baseline:

- `05_rl_baselines/train_ant_ppo.py`
- `05_rl_baselines/evaluate_policy.py`
- `05_rl_baselines/record_policy_video.py`

Residual PPO:

- `05_rl_baselines/residual_gait_wrapper.py`
- `05_rl_baselines/train_residual_ppo.py`
- `05_rl_baselines/evaluate_residual_policy.py`
- `05_rl_baselines/record_residual_policy_video.py`
- `05_rl_baselines/compare_locomotion_baselines.py`

Reward shaping:

- `06_reward_shaping/locomotion_reward_wrapper.py`
- `06_reward_shaping/train_shaped_residual_ppo.py`
- `06_reward_shaping/sweep_reward_weights.py`

## 4. 关键实验结果

### 4.1 100k shaped residual PPO

训练配置：

```text
total_timesteps = 100000
residual_scale = 0.05
forward_weight = 0.8
lateral_weight = 1.0
energy_weight = 0.01
```

模型文件：

```text
results/logs/ant_shaped_residual_ppo_best_100k.zip
```

视频：

```text
results/videos/ant_shaped_residual_ppo_best_100k.mp4
```

单条 300-step 视频结果：

```text
episode_length = 300
terminated = False
x_displacement = 12.5873
y_displacement = -5.4035
```

### 4.2 300-step baseline comparison

文件：

```text
results/tables/locomotion_baseline_comparison_best_100k.csv
```

结果：

```text
policy                 x_disp   abs_y_disp   forward_score
shaped_residual_ppo    12.712     5.392          11.634
residual_ppo            5.348     4.924           4.363
handcrafted_gait        3.672     3.429           2.986
vanilla_ppo            -0.040     0.107          -0.062
```

结论：在短 horizon 中，shaped residual PPO 明显超过传统 gait 和 unshaped residual PPO。

### 4.3 1000-step comparison, 10 seeds

文件：

```text
results/tables/locomotion_baseline_comparison_best_100k_10seed_1000step.csv
```

结果：

```text
policy                 x_disp   abs_y_disp   forward_score
shaped_residual_ppo    47.042    11.616          44.718
handcrafted_gait        8.018    11.212           5.776
vanilla_ppo             0.015     0.085          -0.002
unshaped_residual      -1.808     5.554          -2.918
```

结论：在更长 horizon 和更多 seed 下，shaped residual PPO 仍然保持明显优势。它不只是短期冲刺，而是真的学到了比传统 gait 更强的向前 locomotion 行为。

## 5. 当前发现的问题

### 5.1 横向漂移仍然存在

虽然 shaped residual PPO 的 forward score 很高，但 `abs_y_displacement` 仍不小。说明机器人会强力向前，但方向控制还没有完全稳定。

### 5.2 Reward 权重有明显 tradeoff

实验发现：

```text
lateral_weight 太小 -> 前进强，但横漂大
lateral_weight 太大 -> 横漂小，但前进被压制
```

所以 reward shaping 不是简单地“惩罚越大越好”，而是要在 forward velocity 和 lateral stability 之间找平衡。

### 5.3 目前还是 Ant-v4 单环境

当前结果只在 `Ant-v4` 上验证，还没有做：

- domain randomization
- push recovery
- 不同 friction/mass 的鲁棒性测试
- Ant-v5 迁移

所以现在可以说 policy 在当前仿真环境下有效，但还不能说它有强鲁棒性。

## 6. 当前阶段结论

这一阶段已经完成一个比较完整的 robot learning pipeline：

```text
traditional gait prior
-> residual PPO
-> shaped reward
-> reward-weight sweep
-> stronger learned locomotion policy
-> quantitative baseline comparison
```

最重要的结果是：

```text
100k shaped residual PPO 在 10-seed、1000-step evaluation 中，
mean_x_displacement 达到 47.042，
明显超过 handcrafted gait 的 8.018。
```

这说明项目已经从“能跑 PPO”进入到“有技术路线、有实验对比、有改进结果”的阶段。

## 7. 下一步计划

建议下一步不要马上继续无脑加训练时间，而是做结果展示和鲁棒性扩展：

1. 生成正式对比图表。
2. 录制更长的 1000-step policy video。
3. 增加更多 seed 的 evaluation table。
4. 开始 Phase 7：domain randomization。
5. 后续再做 push recovery。

如果作为 portfolio 项目，可以重点表述为：

```text
Built a MuJoCo quadruped locomotion pipeline combining handcrafted gait priors,
residual PPO, and reward shaping. The shaped residual policy improved
1000-step forward displacement from 8.0 to 47.0 over a handcrafted gait baseline
across 10 seeds in Ant-v4.
```

## 8. 追加训练更新

在原 100k shaped residual PPO 基础上继续训练 400k steps，得到：

```text
results/logs/ant_shaped_residual_ppo_best_500k.zip
```

500k 模型 300-step、10-seed 结果：

```text
shaped_residual_ppo: x=17.982, abs_y=1.985, forward_score=17.585
handcrafted_gait:    x=3.189,  abs_y=3.703, forward_score=2.449
```

500k 模型 1000-step、20-seed 结果：

```text
shaped_residual_ppo: x=60.344, abs_y=2.301, forward_score=59.884, success_rate=0.90
handcrafted_gait:    x=6.586,  abs_y=9.232, forward_score=4.740,  success_rate=1.00
```

视频：

```text
results/videos/ant_shaped_residual_ppo_best_500k.mp4
```

单条 300-step 视频结果：

```text
episode_length=300
terminated=False
x_displacement=17.2503
y_displacement=-1.1391
```

结论：500k 模型比 100k 模型前进更强、横漂更小，但长 horizon 成功率从 100% 降到约 90%。这说明继续训练带来了更激进的高速步态，同时也引入了少量稳定性风险。
