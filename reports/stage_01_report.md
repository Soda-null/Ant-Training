# Robot Learning Mini-Lab 阶段性报告

日期：2026-07-02

## 1. 当前定位

本项目目前定位为一个 Mac 友好的机器人学习 mini-lab。它不是一开始就追求复杂强化学习结果，而是按下面这条路线逐步建立机器人学习所需的基础：

```text
MuJoCo simulation -> kinematics -> PD control -> gait generation -> RL baseline
```

目前已经完成从 MuJoCo 基础运行，到 2-link 运动学，再到单关节控制和简化四足周期动作的第一轮闭环。项目主线仍然是四足机器人 locomotion，2-link arm 部分只是用于快速建立 FK、IK、Jacobian 和控制直觉。

## 2. 技术路径

### 2.1 MuJoCo 基础

第一阶段先确认仿真工具链可用：

- 使用 Python 3.11 和项目内 `.venv`
- 安装 `mujoco`、`gymnasium[mujoco]`、`stable-baselines3`、`torch`、`matplotlib`、`imageio`
- 使用 Gymnasium 的 `Ant-v4` 作为四足机器人环境
- 实现随机动作 rollout、状态检查和视频录制

这一阶段解决的是：环境能不能跑、视频能不能录、`Ant-v4` 的 observation/action/state 大概长什么样。

### 2.2 运动学基础

第二阶段用 2-link planar arm 做最小机器人模型：

- FK：关节角到末端位置
- IK：目标点到关节角
- Jacobian：关节速度到末端速度
- Visualization：把机械臂姿态画出来

这里的目标不是转向机械臂项目，而是用最小模型把机器人几何和速度映射讲清楚。后续四足机器人每条腿也会遇到类似概念，只是维度和接触动力学更复杂。

### 2.3 控制基础

第三阶段实现了单关节 PD 控制：

```text
tau = Kp * (q_target - q) - Kd * qdot
```

已经完成：

- 固定目标角度 tracking
- 多组 `Kp/Kd` 对比
- 正弦轨迹 tracking

这一阶段建立的核心直觉是：更大的 `Kp` 通常响应更快但可能超调；合适的 `Kd` 可以提供阻尼；对于移动目标，PD 控制会出现相位滞后。

### 2.4 简化步态生成

第四阶段开始把单关节 tracking 推到 Ant-like 8 关节周期信号：

- 4 条腿，每条腿 hip/knee 两个关节
- 使用 trot-like 相位模式
- 先画出 8 个关节的周期目标和独立 PD 跟踪曲线
- 再把 8 维周期 action 直接接入 `Ant-v4`
- 录制 handcrafted gait 视频

这一阶段不是为了证明手写步态已经很好，而是为了观察：

```text
周期控制信号 -> MuJoCo 四足机器人运动
```

这也为后续 PPO baseline 提供了对照：random policy、handcrafted periodic policy、learned policy。

## 3. 已完成成果

### 3.1 代码模块

MuJoCo basics:

- `01_mujoco_basics/random_rollout_ant.py`
- `01_mujoco_basics/inspect_mujoco_state.py`
- `01_mujoco_basics/record_random_video.py`

Kinematics:

- `02_kinematics/two_link_fk.py`
- `02_kinematics/two_link_ik.py`
- `02_kinematics/jacobian_demo.py`
- `02_kinematics/visualize_two_link_arm.py`

Control:

- `03_control/pd_joint_control.py`
- `03_control/compare_pd_gains.py`
- `03_control/trajectory_tracking.py`

Gait controller:

- `04_gait_controller/sinusoidal_gait_controller.py`
- `04_gait_controller/record_gait_video.py`

### 3.2 图像和视频产物

已生成的 plot：

- `results/plots/two_link_arm.png`
- `results/plots/pd_joint_control.png`
- `results/plots/compare_pd_gains.png`
- `results/plots/trajectory_tracking.png`
- `results/plots/sinusoidal_gait_controller.png`

已生成的视频：

- `results/videos/random_ant.mp4`
- `results/videos/sinusoidal_gait_ant.mp4`

### 3.3 关键验证结果

随机 Ant rollout 已经可以正常运行。

MuJoCo state inspection 已经确认：

```text
observation_space = Box(-inf, inf, (27,), float64)
action_space = Box(-1.0, 1.0, (8,), float32)
qpos_shape = (15,)
qvel_shape = (14,)
ctrl_shape = (8,)
```

PD gain 对比中，目前较平衡的一组是：

```text
Kp=25, Kd=6
```

它相比低阻尼组合有更小的超调和更稳定的收敛表现。

Handcrafted Ant gait smoke test 已经录制成功，短测试结果为：

```text
episode_return = 98.753
episode_length = 120
terminated = False
truncated = False
x_displacement = 0.8866
y_displacement = -1.6034
```

这说明手写周期 action 能够驱动 `Ant-v4` 运动，并且在短时间内没有提前摔倒。不过它仍然是 open-loop controller，不代表已经学会稳定行走。

## 4. 当前发现的问题和限制

### 4.1 `Ant-v4` 版本提示

Gymnasium 会提示 `Ant-v4` 已过时，建议使用 `v5`。目前项目计划明确使用 `Ant-v4`，而且它仍能正常运行，所以暂时不切换。后续如果需要长期维护，可以考虑统一升级到 `Ant-v5`。

### 4.2 macOS 渲染权限

MuJoCo 视频录制在受限 sandbox 下可能失败，但在正常 VS Code 或 Terminal 图形权限下可以运行。这说明脚本逻辑没问题，但视频录制依赖 macOS 图形栈权限。

### 4.3 简化控制模型和 MuJoCo 实际控制不同

目前 `03_control` 里的 PD 控制是教育用途的单关节二阶模型。它帮助理解控制参数，但并不等价于 MuJoCo Ant 的真实动力学。

类似地，`record_gait_video.py` 里给 `Ant-v4` 的 8 维 action 是直接 actuator action，不是真正的 joint angle target。因此 handcrafted gait 可能移动不稳定、方向不准或能量效率较差。

### 4.4 Open-loop gait 局限

当前 sinusoidal gait 是 open-loop 的：

- 不看身体姿态
- 不看触地状态
- 不根据速度或跌倒风险调整动作
- 没有反馈恢复能力

这正是后面引入 RL policy 的动机。

## 5. 后续计划

### 5.1 Phase 5：PPO baseline

下一步建议实现最小 PPO baseline：

- `05_rl_baselines/train_ant_ppo.py`
- `05_rl_baselines/evaluate_policy.py`
- `05_rl_baselines/record_policy_video.py`

先做 smoke test：

```bash
python 05_rl_baselines/train_ant_ppo.py --total-timesteps 5000 --output results/logs/ant_ppo_smoke
```

目标不是马上得到强策略，而是确认：

- Stable-Baselines3 PPO pipeline 可运行
- 模型能保存和加载
- 训练结果能被评估
- 训练策略能录视频

### 5.2 建立 baseline 对比

后续应比较三类策略：

```text
random action
handcrafted sinusoidal action
PPO learned policy
```

初期可比较：

- episode return
- episode length
- base x/y displacement
- whether terminated early
- video behavior

### 5.3 进入 reward shaping

当 PPO baseline 跑通后，再开始 reward shaping，而不是提前改环境。初步 reward variants 可以是：

- velocity only
- velocity + alive/upright
- velocity + alive/upright + energy penalty

### 5.4 后期扩展

更后面可以继续做：

- domain randomization
- push recovery
- robustness evaluation table
- final technical report
- CV bullet and project summary

## 6. 阶段性结论

目前项目已经完成了一个扎实的前置基础：

```text
仿真能跑 -> 状态能看 -> 图能画 -> 视频能录
FK/IK/Jacobian 能算 -> PD 能控制 -> 周期步态能生成 -> Ant 能被手写 action 驱动
```

这已经足够支撑进入 PPO baseline。下一阶段的重点应该从“机器人基础模块”转向“学习型 locomotion pipeline”，也就是训练、评估、录制和对比 learned policy。

