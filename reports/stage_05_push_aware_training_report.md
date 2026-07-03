# Robot Learning Mini-Lab Push-Aware Training 实验报告

日期：2026-07-03

## 1. 实验目标

上一阶段 formal benchmark 说明：

```text
domain randomization 可以提升强物理参数扰动下的鲁棒性，
但不能自动提升 push recovery。
```

因此本阶段开始做 push-aware training：让 PPO 在训练过程中就遇到短时外力扰动，而不是训练完之后才做 push recovery evaluation。

## 2. 新增代码

新增训练脚本：

```text
08_push_recovery/train_push_aware_shaped_residual_ppo.py
```

新增 checkpoint selection 工具：

```text
08_push_recovery/run_push_aware_checkpoint_selection.py
08_push_recovery/select_push_aware_checkpoint.py
```

核心 wrapper：

```text
PushAwareTrainingWrapper
```

它会在每个 episode 中随机选择：

```text
push_probability
push force
push direction
push start step
push duration
```

然后通过 MuJoCo 的 `xfrc_applied` 对 Ant torso 施加外力。

## 3. 实验一：较强 push-aware continuation

训练设置：

```text
model_in = results/logs/ant_shaped_residual_ppo_best_500k.zip
total_timesteps = 100000
learning_rate = 0.00005
push_probability = 0.8
force_range = 5-10N
push_step_range = 200-500
push_duration = 10
```

输出模型：

```text
results/logs/ant_push_aware_lr5e5_100k.zip
```

checkpoint 目录：

```text
results/logs/push_aware_checkpoint_selection/
```

5-seed checkpoint selection 结果：

| Checkpoint | Push score | Small push recovery | No-push score |
| --- | ---: | ---: | ---: |
| original 500k | 67.62 | 0.78 | 57.25 |
| final 100k | 51.17 | 0.48 | 55.06 |
| push-aware +100k checkpoint | 49.00 | 0.48 | 49.92 |
| push-aware +75k | 42.64 | 0.33 | 48.13 |
| push-aware +25k | 42.31 | 0.40 | 44.18 |

结论：较强 push-aware continuation 失败。它没有提高小推力恢复率，反而明显破坏了原来的 no-push locomotion。

## 4. 实验二：温和 push-aware curriculum

由于第一轮扰动太强、太频繁，第二轮改成更温和的 curriculum。

训练设置：

```text
model_in = results/logs/ant_shaped_residual_ppo_best_500k.zip
total_timesteps = 100000
learning_rate = 0.00005
push_probability = 0.3
force_range = 3-5N
push_step_range = 300-600
push_duration = 10
```

输出模型：

```text
results/logs/ant_push_aware_gentle_lr5e5_100k.zip
```

checkpoint 目录：

```text
results/logs/push_aware_gentle_checkpoint_selection/
```

5-seed checkpoint selection 结果：

| Checkpoint | Push score | Small push recovery | No-push score |
| --- | ---: | ---: | ---: |
| original 500k | 67.62 | 0.78 | 57.25 |
| gentle +75k | 53.69 | 0.57 | 47.42 |
| gentle +50k | 48.52 | 0.58 | 41.56 |
| final 100k | 46.17 | 0.40 | 49.55 |
| gentle +100k checkpoint | 44.56 | 0.38 | 48.59 |

结论：温和 push-aware curriculum 仍然没有超过 original 500k。它比第一轮好一些，但仍然降低了原来的 locomotion 能力，而且没有解决 `-x_10` 这类关键恢复失败。

## 5. 当前结论

这次实验得到一个很有用的负结果：

```text
直接在 PPO continuation 中加入随机外力，不会自动得到更好的 push recovery。
```

可能原因：

1. 原来的 500k policy 是高速稳定步态，外力扰动会让 PPO 很容易破坏已有 gait。
2. 当前 reward 仍然主要奖励 forward locomotion，没有显式奖励“被推后恢复”。
3. push recovery 评价关心 post-push forward displacement，但训练 reward 没有这个项。
4. 外力扰动是短时非平稳事件，比 mass/friction/damping randomization 更难。
5. 训练中随机 push 的时机和方向太宽，policy 可能学到保守或混乱的动作，而不是清晰恢复策略。

## 6. 当前最优模型状态

| 目标 | 当前推荐 |
| --- | --- |
| 常规 locomotion / 展示视频 | `results/logs/ant_shaped_residual_ppo_best_500k.zip` |
| strong domain randomization | `results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip` |
| push recovery | 暂时仍用 original 500k 作为 baseline，push-aware training 尚未改进 |

## 7. 下一步建议

下一步不建议继续直接加 push-aware 训练步数。更合理的路线是改训练目标：

1. 新增 push recovery reward wrapper：

```text
如果 push 后还能保持 alive，并且 post-push x displacement 继续增加，就给 bonus。
```

2. 从更窄的问题开始：

```text
只训练 -x 10N recovery
```

因为当前最明确的失败点就是 backward push。

3. 分阶段 curriculum：

```text
no push warmup
-> fixed -x 5N
-> fixed -x 10N
-> random direction 5-10N
```

4. 每阶段都做 checkpoint selection，不看最后一个模型。

一句话总结：

```text
Domain randomization 可以靠环境参数随机化改善；
push recovery 需要更明确的恢复奖励和更窄的扰动课程，不能只靠随机外力 continuation。
```

## 8. 追加实验：固定 -x 10N recovery reward

按照上面的判断，新增了更窄的训练脚本：

```text
08_push_recovery/train_backward_push_recovery_ppo.py
```

这次不再随机所有方向，而是固定训练最明确的失败点：

```text
push direction = -x
push force = 10N
push_step = 300
push_duration = 10
```

并且新增 post-push recovery reward：

```text
push 后继续产生正向 x displacement -> 给 recovery bonus
push 后仍然 alive -> 给 alive bonus
```

训练设置：

```text
model_in = results/logs/ant_shaped_residual_ppo_best_500k.zip
total_timesteps = 100000
learning_rate = 0.00005
recovery_weight = 2.0
alive_after_push_bonus = 0.2
checkpoint_freq = 25000
```

输出模型：

```text
results/logs/ant_backward_push_recovery_lr5e5_100k.zip
```

checkpoint 目录：

```text
results/logs/backward_push_recovery_checkpoint_selection/
```

5-seed general push ranking：

| Checkpoint | Push score | Small push recovery | No-push score |
| --- | ---: | ---: | ---: |
| original 500k | 67.62 | 0.78 | 57.25 |
| backward +75k | 65.35 | 0.65 | 62.98 |
| backward +50k | 61.22 | 0.73 | 49.55 |
| backward +100k checkpoint | 57.03 | 0.58 | 56.31 |
| backward +25k | 50.02 | 0.50 | 50.44 |

针对 `-x_10` 的观察：

| Model | Episodes | -x 10N recovery | -x 10N score | No-push score |
| --- | ---: | ---: | ---: | ---: |
| original 500k | 10 | 0.00 | 16.11 | 56.24 |
| backward final 100k | 10 | 0.10 | 15.38 | 53.00 |
| backward +50k | 10 | 0.10 | 18.39 | 48.54 |

结论：固定 `-x_10` recovery reward 有一点点正向信号，recovery 从 0.00 提到 0.10，但还远远不够，而且 no-push locomotion 明显下降。当前不能把它当作新的 push-recovery best。

当前判断：

```text
push recovery baseline 仍然是 original 500k；
fixed -x training 证明方向可能有效，但 reward/curriculum 还需要继续改。
```

下一次更合理的改法：

1. 不从 performance 500k 直接训，可以从更慢但更稳的 100k shaped policy 开始。
2. recovery reward 不只奖励 post-push dx，还要惩罚 push 后 base height / torso instability。
3. 对 `-x_10` 单独做更长训练，但必须每 25k 保存并只按 `-x_10` 指标选 checkpoint。
4. 降低 recovery reward 对原 locomotion reward 的破坏，例如 `recovery_weight=0.5-1.0`。
