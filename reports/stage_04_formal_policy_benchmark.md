# Robot Learning Mini-Lab 正式策略对比报告

日期：2026-07-03

## 1. 对比对象

本次正式 benchmark 对比两个当前最重要的策略：

| 名称 | 文件 | 定位 |
| --- | --- | --- |
| Performance policy | `results/logs/ant_shaped_residual_ppo_best_500k.zip` | 当前常规环境与中等扰动下的主力 best policy |
| Robustness policy | `results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip` | 从 performance policy 继续做低学习率 domain-randomized training 后选出的 strong-randomization candidate |

## 2. 评估设置

Domain randomization:

```text
episodes = 20
max_steps = 1000
nominal: mass/friction/damping = 1.0
moderate: mass 0.8-1.2, friction 0.7-1.3, damping 0.7-1.3
strong: mass 0.6-1.4, friction 0.5-1.5, damping 0.5-1.5
```

Push recovery:

```text
episodes = 10 per condition
max_steps = 1000
push_step = 300
push_duration = 10
push_forces = 5, 10, 25, 50 N
directions = +x, -x, +y, -y
```

## 3. Domain Randomization 结果

| Policy | Condition | Success | Mean x | Mean abs y | Forward score |
| --- | --- | ---: | ---: | ---: | ---: |
| Performance | nominal | 0.90 | 60.34 | 2.30 | 59.88 |
| Performance | moderate | 0.85 | 56.15 | 2.48 | 55.65 |
| Performance | strong | 0.65 | 44.24 | 2.95 | 43.65 |
| Robustness | nominal | 0.80 | 53.74 | 3.45 | 53.05 |
| Robustness | moderate | 0.70 | 52.39 | 3.82 | 51.62 |
| Robustness | strong | 0.75 | 52.48 | 4.65 | 51.55 |

结论：

- Performance policy 在 nominal 和 moderate 中明显更强。
- Robustness policy 在 strong randomization 中更强，forward score 从 43.65 提升到 51.55，success 从 0.65 提升到 0.75。
- Robustness policy 的代价是 nominal/moderate 表现下降，尤其横向漂移更大。

## 4. Push Recovery 结果

小扰动平均恢复率：

| Policy | 5/10N push recovery avg | 25/50N push recovery avg |
| --- | ---: | ---: |
| Performance | 0.75 | 0.01 |
| Robustness | 0.58 | 0.03 |

代表性条件：

| Policy | Condition | Success | Recovery | Forward score |
| --- | --- | ---: | ---: | ---: |
| Performance | no push | 0.80 | 0.80 | 56.24 |
| Performance | -x 10N | 0.90 | 0.00 | 16.11 |
| Performance | +y 10N | 0.80 | 0.80 | 43.78 |
| Performance | +x 25N | 0.00 | 0.00 | 22.25 |
| Robustness | no push | 0.70 | 0.70 | 53.44 |
| Robustness | -x 10N | 0.90 | 0.10 | 17.99 |
| Robustness | +y 10N | 0.40 | 0.30 | 29.59 |
| Robustness | +x 25N | 0.00 | 0.00 | 18.95 |

结论：

- Domain-randomized robustness 不等于 push recovery robustness。
- Robustness policy 对强物理参数扰动更强，但对外力推恢复并没有明显变强。
- 两个 policy 对 25N 和 50N 推力基本都不可靠，说明下一阶段如果要做抗扰，需要专门的 push-aware training。

## 5. 视频产物

Performance policy:

```text
results/videos/formal_benchmark_performance_best_1000step.mp4
episode_length = 1000
x_displacement = 62.5334
y_displacement = 4.6134
```

Robustness policy:

```text
results/videos/formal_benchmark_robustness_best_1000step.mp4
episode_length = 1000
x_displacement = 65.0841
y_displacement = -4.4961
```

汇总图：

```text
results/plots/formal_policy_benchmark.png
```

生成方式：

```bash
python reports/plot_formal_benchmark.py
```

## 6. 当前最终判断

| 目标 | 推荐模型 |
| --- | --- |
| 展示视频、常规 locomotion、portfolio 主结果 | `results/logs/ant_shaped_residual_ppo_best_500k.zip` |
| 强 domain randomization 鲁棒性研究 | `results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip` |
| push recovery | 两者都不是最终答案，需要新训练 |

一句话总结：

```text
Performance policy 仍然是当前主力 best policy；
Robustness policy 是 strong domain randomization 下更好的研究候选；
但 push recovery 需要单独训练，不能靠 domain randomization 自动解决。
```

## 7. 下一步建议

下一阶段不建议继续盲目 domain-randomized training。更合理的路线是：

1. 保留 performance policy 作为最终展示模型。
2. 保留 robustness policy 作为 strong randomization 实验模型。
3. 新增 push-aware training wrapper，让策略在训练中见到 5-10N 的随机短时外力。
4. 继续使用 checkpoint selection，而不是只保存最后一个模型。
