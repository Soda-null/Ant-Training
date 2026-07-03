# Results Index

This file indexes the main local artifacts for the Robot Learning Mini-Lab.
It is a local organization aid; no files listed here are intended to be pushed
or deleted as part of this cleanup.

## 1. Best Policies

For nominal locomotion and portfolio demonstration:

```text
results/logs/ant_shaped_residual_ppo_best_500k.zip
```

For strong domain randomization robustness:

```text
results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip
```

For push recovery:

```text
No new best policy has been found.
Use the original 500k policy as the baseline and report push-aware models as negative results.
```

Push-aware attempts include:

```text
results/logs/ant_push_aware_lr5e5_100k.zip
results/logs/ant_push_aware_gentle_lr5e5_100k.zip
results/logs/ant_backward_push_recovery_lr5e5_100k.zip
```

These are retained as negative-result artifacts, not recommended demonstration
models.

## 2. Videos

Main benchmark videos:

```text
results/videos/formal_benchmark_performance_best_1000step.mp4
results/videos/formal_benchmark_robustness_best_1000step.mp4
```

Additional videos:

```text
results/videos/ant_shaped_residual_ppo_best_500k.mp4
results/videos/ant_shaped_residual_ppo_best_100k.mp4
results/videos/ant_shaped_residual_ppo_best_20k.mp4
results/videos/ant_ppo_policy.mp4
results/videos/ant_residual_ppo_policy.mp4
results/videos/sinusoidal_gait_ant_best.mp4
results/videos/sinusoidal_gait_ant.mp4
results/videos/random_ant.mp4
```

## 3. Plots

Main figures:

```text
results/plots/formal_policy_benchmark.png
results/plots/stage_03_results_summary.png
results/plots/locomotion_baseline_comparison_500k.png
```

Additional plots:

```text
results/plots/locomotion_baseline_comparison.png
results/plots/sinusoidal_gait_controller.png
results/plots/compare_pd_gains.png
results/plots/pd_joint_control.png
results/plots/trajectory_tracking.png
results/plots/two_link_arm.png
```

## 4. Tables

Locomotion and reward shaping:

```text
results/tables/locomotion_baseline_comparison_best_500k_20seed_1000step.csv
results/tables/locomotion_baseline_comparison_best_500k_10seed_1000step.csv
results/tables/locomotion_baseline_comparison_best_500k_10seed_300step.csv
results/tables/locomotion_baseline_comparison_best_100k_10seed_1000step.csv
results/tables/locomotion_baseline_comparison_best_100k_1000step.csv
results/tables/locomotion_baseline_comparison_best_100k.csv
results/tables/locomotion_baseline_comparison.csv
results/tables/reward_weight_sweep.csv
```

Gait parameter sweeps:

```text
results/tables/gait_param_sweep.csv
results/tables/gait_param_sweep_broad.csv
```

Domain randomization:

```text
results/tables/domain_randomization_summary.csv
results/tables/domain_randomization_strong_summary.csv
results/tables/domain_randomization_dr50k_summary.csv
results/tables/domain_randomization_dr50k_strong_summary.csv
results/tables/domain_randomization_narrow_lr1e4_100k_summary.csv
results/tables/domain_randomization_narrow_lr1e4_100k_strong_summary.csv
results/tables/checkpoint_selection/domain_randomization_checkpoint_ranking.csv
results/tables/checkpoint_selection_from_25k_lr5e5/domain_randomization_checkpoint_ranking.csv
```

Formal benchmark:

```text
results/tables/formal_benchmark/performance_domain_nominal_summary.csv
results/tables/formal_benchmark/performance_domain_moderate_summary.csv
results/tables/formal_benchmark/performance_domain_strong_summary.csv
results/tables/formal_benchmark/robust_domain_nominal_summary.csv
results/tables/formal_benchmark/robust_domain_moderate_summary.csv
results/tables/formal_benchmark/robust_domain_strong_summary.csv
results/tables/formal_benchmark/performance_push_summary.csv
results/tables/formal_benchmark/robust_push_summary.csv
```

Push recovery:

```text
results/tables/push_recovery_summary.csv
results/tables/push_recovery_small_summary.csv
results/tables/push_aware_selection/push_aware_checkpoint_ranking.csv
results/tables/push_aware_gentle_selection/push_aware_checkpoint_ranking.csv
results/tables/backward_push_recovery_selection/push_aware_checkpoint_ranking.csv
results/tables/backward_push_recovery/backward_push_lr5e5_100k_push_summary.csv
results/tables/backward_push_recovery/backward_push_lr5e5_550736_steps_minusx10_summary.csv
```

Episode-level CSVs are also stored in the same folders for auditability.

## 5. Push-Aware Attempts

Push-aware training is currently a negative result:

- Random push-aware continuation damaged no-push locomotion and reduced small-push recovery.
- Gentler random push training was better than the first attempt but still did not beat the original 500k policy.
- Fixed `-x 10N` recovery training produced a small signal but did not create a usable new best policy.

Current recommendation:

```text
Use results/logs/ant_shaped_residual_ppo_best_500k.zip as the push-recovery baseline.
Report push-aware models as experimental negative results.
```

Future push-recovery work should use a narrower curriculum and a recovery
reward that directly measures post-push forward recovery.
