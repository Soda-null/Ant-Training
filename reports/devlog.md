# Devlog

## Day 01

### Goal

Start the Robot Learning Mini-Lab with a minimal MuJoCo basics workflow.

### What I did

- Created the project structure.
- Added dependency and ignore files.
- Added random Ant rollout, state inspection, and random video scripts.

### Results

- Python 3.11.15 was installed with Homebrew.
- A local `.venv` was created.
- Project dependencies installed successfully.
- `Ant-v4` random rollout ran successfully.
- MuJoCo state inspection ran successfully.
- Random Ant video was recorded at `results/videos/random_ant.mp4`.

### Problems

- `Ant-v4` prints a Gymnasium deprecation warning suggesting `v5`, but it still
  runs and matches the project plan.
- Video recording needs normal macOS graphics permissions. It failed in a
  restricted sandbox process but worked when run with normal app permissions.

### Next

- Review the state inspection output and explain `qpos`, `qvel`, `ctrl`, and
  `step()`.
- Start Phase 2 with the 2-link arm forward kinematics script.

## Day 02

### Goal

Start Phase 2 with a minimal forward kinematics demo.

### What I did

- Added `02_kinematics/two_link_fk.py`.
- Added command-line inputs for link lengths and joint angles.
- Updated README and VS Code tasks.

### Results

- The script computes shoulder, elbow, and end-effector coordinates for a
  simple 2-link planar arm.

### Problems

- None yet.

### Next

- Add inverse kinematics for reachable target points.

## Day 03

### Goal

Add analytical inverse kinematics for the 2-link planar arm.

### What I did

- Added `02_kinematics/two_link_ik.py`.
- Implemented reachability checking.
- Printed elbow-up and elbow-down solutions.
- Used FK to report a check error for each IK solution.

### Results

- Reachable targets produce two joint-angle solutions.
- Unreachable targets report `reachable=False`.

### Problems

- At fully stretched or folded singular configurations, the two mathematical
  branches can collapse to the same physical pose. This is expected.

### Next

- Add the Jacobian demo to map joint velocity to end-effector velocity.

## Day 04

### Goal

Add the 2-link arm Jacobian demo.

### What I did

- Added `02_kinematics/jacobian_demo.py`.
- Implemented the analytical 2x2 Jacobian.
- Mapped joint velocity to end-effector velocity.
- Added a finite-difference check against FK.

### Results

- The Jacobian demo runs from the command line and reports a small numerical
  check error.

### Problems

- None yet.

### Next

- Add a simple 2-link arm visualization with Matplotlib.

## Day 05

### Goal

Add a simple visualization for the 2-link planar arm.

### What I did

- Added `02_kinematics/visualize_two_link_arm.py`.
- Plotted shoulder, elbow, end-effector, and optional target point.
- Saved the plot to `results/plots/two_link_arm.png` by default.

### Results

- The visualization script runs without requiring a GUI window by default.

### Problems

- None yet.

### Next

- Move to Phase 3 with a minimal PD joint-control demo.

## Day 06

### Goal

Start Phase 3 with a minimal single-joint PD controller.

### What I did

- Added `03_control/pd_joint_control.py`.
- Simulated a single rotary joint with inertia and passive damping.
- Used `tau = Kp * (q_target - q) - Kd * qdot`.
- Saved angle, velocity, and torque curves to `results/plots/pd_joint_control.png`.

### Results

- The script runs from the command line and reports final error, overshoot, and
  max absolute torque.

### Problems

- This is a simple educational dynamics model, not a MuJoCo joint yet.

### Next

- Add trajectory tracking or compare several PD gain settings.

## Day 07

### Goal

Compare several PD gain settings on the same single-joint model.

### What I did

- Added `03_control/compare_pd_gains.py`.
- Reused the single-joint PD simulation from `pd_joint_control.py`.
- Compared several `Kp/Kd` settings in one plot.
- Printed final error, overshoot, settling time, and max torque for each case.

### Results

- The script saves a comparison plot to `results/plots/compare_pd_gains.png`.

### Problems

- This comparison still uses the simple educational joint model, not a MuJoCo
  joint.

### Next

- Add trajectory tracking for a time-varying target.

## Day 08

### Goal

Add PD tracking for a sinusoidal joint target.

### What I did

- Added `03_control/trajectory_tracking.py`.
- Simulated a time-varying target trajectory.
- Plotted target angle, actual angle, tracking error, and torque.
- Printed RMS error, max error, and max torque.

### Results

- The script saves a tracking plot to `results/plots/trajectory_tracking.png`.

### Problems

- Tracking has visible phase lag, which is expected for a simple PD controller
  following a moving target.

### Next

- Move toward multi-joint trajectory tracking or a simple gait-style controller.

## Day 09

### Goal

Start Phase 4 with a simple Ant-like sinusoidal gait controller.

### What I did

- Added `04_gait_controller/sinusoidal_gait_controller.py`.
- Generated eight periodic joint targets with a trot-like phase pattern.
- Simulated independent PD tracking for each joint.
- Saved target and tracked joint curves to
  `results/plots/sinusoidal_gait_controller.png`.

### Results

- The script produces a multi-joint gait-target plot and reports aggregate RMS
  tracking error.

### Problems

- This is not yet a MuJoCo locomotion controller. It only generates and tracks
  periodic joint targets in a simplified independent-joint model.

### Next

- Connect the periodic action pattern to `Ant-v4` and record a short video, or
  add a safer action-space inspection step first.

## Day 10

### Goal

Connect the handcrafted sinusoidal action pattern to `Ant-v4`.

### What I did

- Added `04_gait_controller/record_gait_video.py`.
- Generated an 8-dimensional periodic action for the Ant action space.
- Recorded a short MuJoCo video at
  `results/videos/sinusoidal_gait_ant.mp4`.
- Printed return, episode length, termination state, and base displacement.

### Results

- The handcrafted periodic action pattern can be run directly in MuJoCo.

### Problems

- The controller is open-loop and uses direct actuator actions, not true joint
  angle targets. It may fall or move poorly, which is expected.

### Next

- Start the PPO baseline in Phase 5 so the project has a learned locomotion
  policy to compare against random and handcrafted behavior.

## Day 11

### Goal

Start Phase 5 with a minimal PPO baseline pipeline.

### What I did

- Added `05_rl_baselines/train_ant_ppo.py`.
- Added `05_rl_baselines/evaluate_policy.py`.
- Added `05_rl_baselines/record_policy_video.py`.
- Updated README and VS Code tasks.

### Results

- The project now has scripts for PPO training, saved-model evaluation, and
  learned-policy video recording.
- A 5,000-timestep smoke run saved a model at
  `results/logs/ant_ppo_smoke.zip`.
- The smoke policy can run for full evaluation episodes, but the short recorded
  video shows almost no forward displacement yet.

### Problems

- A 5,000-timestep PPO run is only a smoke test. It proves the pipeline works,
  but it is not enough to learn a strong locomotion policy.
- Return alone can be misleading because Ant includes survival-style reward
  terms. For locomotion, displacement and video behavior should be checked too.

### Next

- Run a short PPO smoke training.
- Evaluate the saved model.
- Record a short policy video and compare it with random and handcrafted gait
  videos.

## Day 12

### Goal

Tune the traditional handcrafted gait before using it as a PPO prior.

### What I did

- Added `04_gait_controller/sweep_gait_params.py`.
- Swept frequency, hip action amplitude, and knee action amplitude.
- Ranked results with a score that combines forward displacement and survival
  length.
- Saved the ranked table to `results/tables/gait_param_sweep.csv`.

### Results

- The project can now search for a better handcrafted gait candidate instead
  of choosing sinusoidal parameters by hand.
- The first single-seed sweep found this best row:

```text
frequency=0.80, action_scale=0.25, knee_action_scale=0.35
x_displacement=3.031, y_displacement=-1.283, forward_score=2.774
```
- Direct video recording with a different seed showed much larger sideways
  drift, so the sweep was updated to average multiple seeds per parameter
  combination.
- Short unstable motions can still get positive displacement, so the ranking
  now uses a score that combines forward motion, episode length, and success
  rate.
- After fair multi-seed action-sign search, the current best default row is:

```text
frequency=1.00, action_sign=-1, action_scale=0.20, knee_action_scale=0.10
mean_x_displacement=3.672, mean_y_displacement=-3.429, success_rate=1.00
```
- The broader sweep was saved at `results/tables/gait_param_sweep_broad.csv`
  and showed that smaller knee amplitudes give a more stable forward prior.
- The current best gait video was recorded at
  `results/videos/sinusoidal_gait_ant_best.mp4`:

```text
episode_length=300, terminated=False
x_displacement=6.5906, y_displacement=-1.4653
```
- Seed scheduling was corrected so every parameter combination is tested on
  the same seed set instead of a different seed range.

### Problems

- The sweep is still open-loop. A good row here is only a better prior, not a
  robust locomotion controller.
- Single-seed search can overfit to one lucky initial condition, so default
  sweeps now use multiple seeds.

### Next

- Record a video using the best sweep row.
- Use the best handcrafted gait as the base action for residual PPO.

## Day 13

### Goal

Start residual PPO from the tuned handcrafted gait instead of training from
random actions.

### What I did

- Added `05_rl_baselines/residual_gait_wrapper.py`.
- Added `05_rl_baselines/train_residual_ppo.py`.
- Added `05_rl_baselines/evaluate_residual_policy.py`.
- Added `05_rl_baselines/record_residual_policy_video.py`.
- Updated README and VS Code tasks.

### Results

- The project now has the residual-learning structure:

```text
final_action = handcrafted_gait_action + residual_policy_action
```
- A conservative residual PPO smoke run with `residual_scale=0.05` saved a
  model at `results/logs/ant_residual_ppo_smoke.zip`.
- Deterministic evaluation after 5,000 timesteps:

```text
mean_return=938.676
mean_length=1000.0
mean_x_displacement=-0.0424
mean_y_displacement=2.5773
```

- A residual PPO video was recorded at
  `results/videos/ant_residual_ppo_policy.mp4`:

```text
episode_length=300, terminated=False
x_displacement=0.9439, y_displacement=-7.5262
```

### Problems

- A short residual PPO smoke run only verifies the pipeline. It does not prove
  the residual policy is better than the handcrafted gait yet.
- The first residual smoke evaluation stayed alive but lost forward movement,
  so the default residual scale was reduced from `0.15` to `0.05`.
- The conservative residual policy still drifts sideways. The residual PPO
  pipeline works, but reward and evaluation need to better emphasize forward
  displacement and lateral stability.

### Next

- Run residual PPO smoke training.
- Compare handcrafted gait, vanilla PPO, and residual PPO with the same seeds.

## Day 14

### Goal

Compare locomotion baselines with the same seeds and metrics.

### What I did

- Added `05_rl_baselines/compare_locomotion_baselines.py`.
- Compared random, handcrafted gait, vanilla PPO, and residual PPO.
- Saved a ranked table to
  `results/tables/locomotion_baseline_comparison.csv`.

### Results

- The project now has one command to check whether a policy actually moves
  forward instead of only surviving.
- The first 300-step deterministic comparison produced:

```text
policy             mean_x   mean_abs_y   forward_score
residual_ppo        5.348       4.924          4.363
handcrafted_gait    3.672       3.429          2.986
random              1.461       0.844          1.292
vanilla_ppo        -0.040       0.107         -0.062
```

### Problems

- Residual PPO still needs reward shaping before it can reliably improve on the
  handcrafted gait prior.
- The residual policy can improve short-horizon forward movement, but it also
  increases lateral drift and was less reliable in longer 1000-step evaluation.

### Next

- Add a shaped reward wrapper that rewards forward displacement and penalizes
  lateral drift.

## Day 15

### Goal

Start reward shaping for residual PPO.

### What I did

- Added `06_reward_shaping/locomotion_reward_wrapper.py`.
- Added `06_reward_shaping/train_shaped_residual_ppo.py`.
- Updated the locomotion comparison script to include shaped residual PPO when
  the shaped model exists.
- Updated README and VS Code tasks.

### Results

- The shaped reward uses:

```text
shaped_reward = original_reward
              + forward_weight * x_velocity
              - lateral_weight * abs(y_velocity)
              - energy_weight * sum(action^2)
```
- A 5,000-timestep shaped residual smoke run saved
  `results/logs/ant_shaped_residual_ppo_smoke.zip`.
- The first comparison after shaping was:

```text
policy               mean_x   mean_abs_y   forward_score
residual_ppo          5.348       4.924          4.363
handcrafted_gait      3.672       3.429          2.986
shaped_residual_ppo   3.927       4.705          2.986
vanilla_ppo          -0.040       0.107         -0.062
```

### Problems

- Reward weights are now explicit experiment knobs and need empirical tuning.
- The first shaped run did not beat the unshaped residual policy. It preserved
  forward movement but did not reduce lateral drift enough.
- A stronger lateral penalty experiment (`lateral_weight=1.5`) reduced lateral
  drift but also suppressed forward movement:

```text
mean_x_displacement=0.328
mean_abs_y_displacement=1.465
mean_forward_score=0.036
```

### Next

- Run shaped residual PPO smoke training.
- Compare shaped residual PPO against handcrafted gait and unshaped residual
  PPO.
- Try a stronger lateral penalty and compare again.
- Sweep reward weights instead of relying on a single shaped reward guess.

## Day 16

### Goal

Automate reward-weight tuning for shaped residual PPO.

### What I did

- Added `06_reward_shaping/sweep_reward_weights.py`.
- The script trains short shaped residual PPO runs across reward-weight grids.
- It evaluates each model with the same seeds and saves a ranked table.
- It saves the best short-run model to
  `results/logs/ant_shaped_residual_ppo_best.zip`.

### Results

- Reward tuning is now a repeatable experiment instead of manual one-off
  guesses.
- The first 24-config reward sweep found this best short-run setting:

```text
forward_weight=0.8
lateral_weight=1.0
energy_weight=0.01
mean_x_displacement=8.556
mean_abs_y_displacement=2.564
mean_forward_score=8.043
```

- Retraining that setting for 20,000 timesteps produced a more conservative but
  steadier model at `results/logs/ant_shaped_residual_ppo_best_20k.zip`.
- 300-step comparison for the 20k model:

```text
shaped_residual_ppo: x=5.223, abs_y=2.938, forward_score=4.635
unshaped_residual:   x=5.348, abs_y=4.924, forward_score=4.363
handcrafted_gait:    x=3.672, abs_y=3.429, forward_score=2.986
```

- 1000-step comparison showed the shaped 20k model fixes the unshaped residual
  policy's long-horizon failure mode:

```text
shaped_residual_ppo: x=4.732, abs_y=2.063, forward_score=4.320
unshaped_residual:   x=-3.614, abs_y=11.308, forward_score=-5.875
handcrafted_gait:    x=15.881, abs_y=10.702, forward_score=13.740
```

- A video was recorded at
  `results/videos/ant_shaped_residual_ppo_best_20k.mp4`:

```text
episode_length=300, terminated=False
x_displacement=5.4224, y_displacement=-3.4045
```

### Problems

- Short sweeps are noisy. A top row from 4,096 timesteps is a candidate, not a
  final policy.
- The handcrafted gait still wins over 1000 steps. The shaped residual model is
  better than unshaped residual, but not yet better than the traditional prior
  for long-horizon travel.

### Next

- Run the reward-weight sweep.
- Compare the best shaped model against handcrafted gait and unshaped residual
  PPO.

## Day 17

### Goal

Train the best shaped residual PPO setting longer.

### What I did

- Trained `forward_weight=0.8`, `lateral_weight=1.0`,
  `energy_weight=0.01` for 100,000 timesteps.
- Saved the model at `results/logs/ant_shaped_residual_ppo_best_100k.zip`.
- Compared it against random, handcrafted gait, vanilla PPO, and unshaped
  residual PPO at 300 and 1000 steps.
- Recorded a video at `results/videos/ant_shaped_residual_ppo_best_100k.mp4`.

### Results

- 300-step comparison:

```text
shaped_residual_ppo: x=12.712, abs_y=5.392, forward_score=11.634
unshaped_residual:   x=5.348,  abs_y=4.924, forward_score=4.363
handcrafted_gait:    x=3.672,  abs_y=3.429, forward_score=2.986
```

- 1000-step comparison:

```text
shaped_residual_ppo: x=46.512, abs_y=7.162,  forward_score=45.080
handcrafted_gait:    x=15.881, abs_y=10.702, forward_score=13.740
unshaped_residual:   x=-3.614, abs_y=11.308, forward_score=-5.875
```

- The 100k shaped residual model now clearly beats the traditional gait on
  forward locomotion while also reducing long-horizon lateral drift.
- A larger 10-seed, 1000-step comparison confirmed the result:

```text
shaped_residual_ppo: x=47.042, abs_y=11.616, forward_score=44.718
handcrafted_gait:    x=8.018,  abs_y=11.212, forward_score=5.776
unshaped_residual:   x=-1.808, abs_y=5.554,  forward_score=-2.918
```

### Problems

- Lateral displacement in the 300-step video is still visible, so it is not a
  polished final gait.
- This is still evaluated in one environment version and a small seed set.

### Next

- Run a larger evaluation seed set.
- Record a 1000-step video or build a comparison plot/table for the report.

## Day 18

### Goal

Continue training the shaped residual PPO policy beyond 100k.

### What I did

- Added resume support to `06_reward_shaping/train_shaped_residual_ppo.py` with
  `--model-in`.
- Continued training from
  `results/logs/ant_shaped_residual_ppo_best_100k.zip` for 400,000 additional
  timesteps.
- Saved the continued model at
  `results/logs/ant_shaped_residual_ppo_best_500k.zip`.
- Evaluated the 500k model over 10 seeds at 300 steps and 20 seeds at 1000
  steps.
- Recorded `results/videos/ant_shaped_residual_ppo_best_500k.mp4`.

### Results

- 300-step, 10-seed comparison:

```text
shaped_residual_ppo: x=17.982, abs_y=1.985, forward_score=17.585
handcrafted_gait:    x=3.189,  abs_y=3.703, forward_score=2.449
```

- 1000-step, 20-seed comparison:

```text
shaped_residual_ppo: x=60.344, abs_y=2.301, forward_score=59.884, success_rate=0.90
handcrafted_gait:    x=6.586,  abs_y=9.232, forward_score=4.740,  success_rate=1.00
```

- 300-step video result:

```text
x_displacement=17.2503, y_displacement=-1.1391, terminated=False
```

### Problems

- The 500k model is faster and straighter than 100k, but it has a lower
  long-horizon success rate. It is a better performance policy, not necessarily
  the safest policy.

### Next

- Keep both 100k and 500k models as comparison points.
- For robustness, test domain randomization or add a survival/upright margin
  term before training even longer.

## Day 19

### Goal

Start Phase 7 with domain randomization evaluation.

### What I did

- Added `07_domain_randomization/evaluate_domain_randomization.py`.
- The script compares nominal evaluation against randomized MuJoCo physics.
- Randomized parameters include body mass, sliding friction, and joint damping.

### Results

- The project can now measure whether the current shaped residual PPO policy is
  robust to simple dynamics changes.
- Default randomization (`mass=0.8..1.2`, `friction=0.7..1.3`,
  `damping=0.7..1.3`) barely reduced performance:

```text
nominal:    success_rate=0.80, x=56.733, forward_score=56.236
randomized: success_rate=0.80, x=55.407, forward_score=54.939
```

- Stronger randomization (`mass=0.6..1.4`, `friction=0.5..1.5`,
  `damping=0.5..1.5`) exposed the robustness boundary:

```text
nominal:    success_rate=0.80, x=56.733, forward_score=56.236
randomized: success_rate=0.50, x=40.874, forward_score=40.365
```

### Problems

- This is evaluation only, not domain-randomized training yet.
- The 500k policy handles moderate dynamics variation, but strong variation
  reduces long-horizon success rate.

### Next

- Run the domain randomization evaluation.
- If performance drops sharply, add domain-randomized training.

## Day 20

### Goal

Start Phase 8 with push recovery evaluation.

### What I did

- Added `08_push_recovery/evaluate_push_recovery.py`.
- The script applies short external force pulses to the Ant torso through
  MuJoCo `xfrc_applied`.
- It evaluates no-push and pushed episodes with shared seeds.

### Results

- The project can now measure whether the policy continues moving forward
  after external pushes.
- Large pushes (`100/200/300N` for 10 steps) are too severe for the current
  policy: all pushed conditions terminate shortly after the push.
- Smaller pushes reveal a clearer recovery boundary:

```text
5N pushes:  generally recover, success_rate 0.80-1.00
10N pushes: direction-dependent, +x/+y recover well, -x weak recovery
25N pushes: mostly fail
50N pushes: fail across directions
```

- The detailed small-force table is saved at
  `results/tables/push_recovery_small_summary.csv`.

### Problems

- This is an evaluation script only. It does not train a push-robust policy.
- The 500k policy is strong at forward locomotion but not robust to moderate or
  large external pushes.

### Next

- Run the push recovery evaluation.
- If recovery is weak, add push perturbations during training.
