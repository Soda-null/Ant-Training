# Reproduce Commands

Run commands from the repository root after activating the virtual environment:

```bash
source .venv/bin/activate
```

## Setup

Create and populate the Python environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Baseline Comparison

Compare random actions, handcrafted gait, vanilla PPO, residual PPO, and the
500k shaped residual PPO policy.

```bash
python 05_rl_baselines/compare_locomotion_baselines.py \
  --episodes 20 \
  --max-steps 1000 \
  --shaped-residual-model results/logs/ant_shaped_residual_ppo_best_500k.zip \
  --deterministic \
  --output results/tables/locomotion_baseline_comparison.csv
```

TODO: if model files are not present locally, rerun or download the relevant
training artifacts before evaluation.

## Domain Randomization Evaluation

Evaluate the performance policy under moderate domain randomization.

```bash
python 07_domain_randomization/evaluate_domain_randomization.py \
  --model results/logs/ant_shaped_residual_ppo_best_500k.zip \
  --episodes 20 \
  --max-steps 1000 \
  --summary-output results/tables/domain_randomization_summary.csv \
  --episodes-output results/tables/domain_randomization_episodes.csv
```

Evaluate the performance policy under strong domain randomization.

```bash
python 07_domain_randomization/evaluate_domain_randomization.py \
  --model results/logs/ant_shaped_residual_ppo_best_500k.zip \
  --episodes 20 \
  --max-steps 1000 \
  --mass-range 0.6,1.4 \
  --friction-range 0.5,1.5 \
  --damping-range 0.5,1.5 \
  --summary-output results/tables/domain_randomization_strong_summary.csv \
  --episodes-output results/tables/domain_randomization_strong_episodes.csv
```

## Formal Policy Benchmark

Regenerate the formal benchmark plot from existing formal benchmark CSV files.

```bash
python reports/plot_formal_benchmark.py
```

TODO: the script expects CSV files under `results/tables/formal_benchmark/`.
If those files are missing, rerun the domain and push evaluations first.

## Push Recovery Evaluation

Evaluate push recovery for the performance policy.

```bash
python 08_push_recovery/evaluate_push_recovery.py \
  --model results/logs/ant_shaped_residual_ppo_best_500k.zip \
  --episodes 10 \
  --max-steps 1000 \
  --push-forces 5,10,25,50 \
  --directions +x,-x,+y,-y \
  --summary-output results/tables/push_recovery_summary.csv \
  --episodes-output results/tables/push_recovery_episodes.csv
```

## Plot Generation

Generate the stage-03 summary plot.

```bash
python reports/plot_stage_03_summary.py
```

Generate the formal benchmark plot.

```bash
python reports/plot_formal_benchmark.py
```

Generate the locomotion baseline comparison plot.

```bash
python 06_reward_shaping/plot_locomotion_comparison.py \
  --input results/tables/locomotion_baseline_comparison_best_500k_20seed_1000step.csv \
  --output results/plots/locomotion_baseline_comparison_500k.png
```

## Video Recording

Record a 1000-step video for the performance policy.

```bash
python 05_rl_baselines/record_residual_policy_video.py \
  --model results/logs/ant_shaped_residual_ppo_best_500k.zip \
  --steps 1000 \
  --deterministic \
  --output results/videos/formal_benchmark_performance_best_1000step.mp4
```

Record a 1000-step video for the robustness policy.

```bash
python 05_rl_baselines/record_residual_policy_video.py \
  --model results/logs/dr_checkpoint_selection_from_25k_lr5e5/from25k_lr5e5_575736_steps.zip \
  --steps 1000 \
  --deterministic \
  --output results/videos/formal_benchmark_robustness_best_1000step.mp4
```

On macOS, video recording may need to run from a normal Terminal or VS Code
terminal so MuJoCo can access the graphics stack.

## Training Commands

Train shaped residual PPO from scratch or continue an existing model.

```bash
python 06_reward_shaping/train_shaped_residual_ppo.py \
  --total-timesteps 5000 \
  --n-steps 512 \
  --forward-weight 0.8 \
  --lateral-weight 1.0 \
  --energy-weight 0.01
```

TODO: full training runs used longer budgets such as 100k and 500k timesteps.
Use checkpointing and evaluation before replacing any best policy.
