"""Train shaped residual PPO for a fixed backward-push recovery task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

BASELINE_DIR = Path(__file__).resolve().parents[1] / "05_rl_baselines"
REWARD_DIR = Path(__file__).resolve().parents[1] / "06_reward_shaping"
for directory in (BASELINE_DIR, REWARD_DIR):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

from locomotion_reward_wrapper import LocomotionRewardWrapper  # noqa: E402
from residual_gait_wrapper import make_residual_ant_env  # noqa: E402


class BackwardPushRecoveryRewardWrapper(gym.Wrapper):
    """Apply one backward push and reward post-push forward recovery."""

    def __init__(
        self,
        env: gym.Env,
        push_force: float = 10.0,
        push_step: int = 300,
        push_duration: int = 10,
        recovery_weight: float = 2.0,
        alive_after_push_bonus: float = 0.2,
    ) -> None:
        super().__init__(env)
        if push_force < 0:
            raise ValueError("push_force must be non-negative.")
        if push_step < 0 or push_duration <= 0:
            raise ValueError("push_step must be non-negative and push_duration positive.")
        if recovery_weight < 0 or alive_after_push_bonus < 0:
            raise ValueError("reward weights must be non-negative.")

        self.push_force = push_force
        self.push_step = push_step
        self.push_duration = push_duration
        self.recovery_weight = recovery_weight
        self.alive_after_push_bonus = alive_after_push_bonus
        self.step_index = 0
        self.previous_x = 0.0
        self.torso_id = self.env.unwrapped.model.body("torso").id

    def reset(self, **kwargs):
        self.step_index = 0
        self.env.unwrapped.data.xfrc_applied[:] = 0.0
        observation, info = self.env.reset(**kwargs)
        self.previous_x = self.base_x()
        return observation, info

    def step(self, action):
        if self.push_step <= self.step_index < self.push_step + self.push_duration:
            self.env.unwrapped.data.xfrc_applied[self.torso_id, 0:3] = np.array(
                [-self.push_force, 0.0, 0.0],
                dtype=np.float64,
            )
        else:
            self.env.unwrapped.data.xfrc_applied[:] = 0.0

        observation, reward, terminated, truncated, info = self.env.step(action)
        current_x = self.base_x()
        dx = current_x - self.previous_x
        self.previous_x = current_x

        recovery_reward = 0.0
        alive_bonus = 0.0
        if self.step_index >= self.push_step + self.push_duration:
            recovery_reward = self.recovery_weight * max(0.0, dx)
            alive_bonus = self.alive_after_push_bonus
            reward = float(reward + recovery_reward + alive_bonus)

        info = dict(info)
        info["backward_push_active"] = self.push_step <= self.step_index < self.push_step + self.push_duration
        info["post_push_recovery_reward"] = float(recovery_reward)
        info["post_push_alive_bonus"] = float(alive_bonus)

        self.step_index += 1
        if terminated or truncated:
            self.env.unwrapped.data.xfrc_applied[:] = 0.0
        return observation, reward, terminated, truncated, info

    def base_x(self) -> float:
        return float(self.env.unwrapped.data.qpos[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train for fixed -x push recovery.")
    parser.add_argument("--env-id", default="Ant-v4")
    parser.add_argument("--total-timesteps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--action-sign", type=float, default=-1.0)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--knee-action-scale", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--forward-weight", type=float, default=0.8)
    parser.add_argument("--lateral-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--push-force", type=float, default=10.0)
    parser.add_argument("--push-step", type=int, default=300)
    parser.add_argument("--push-duration", type=int, default=10)
    parser.add_argument("--recovery-weight", type=float, default=2.0)
    parser.add_argument("--alive-after-push-bonus", type=float, default=0.2)
    parser.add_argument("--model-in", default=None)
    parser.add_argument("--output", default="results/logs/ant_backward_push_recovery_smoke")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-prefix", default="backward_push")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def model_path(path: str) -> Path:
    output = Path(path)
    if output.suffix != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0 or args.n_envs <= 0 or args.n_steps <= 0:
        raise SystemExit("total-timesteps, n-envs, and n-steps must be positive.")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive.")
    if args.checkpoint_freq <= 0:
        raise SystemExit("checkpoint-freq must be positive.")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run `python -m pip install -r requirements.txt`.") from exc

    def make_env():
        env = make_residual_ant_env(
            env_id=args.env_id,
            frequency=args.frequency,
            action_sign=args.action_sign,
            action_scale=args.action_scale,
            knee_action_scale=args.knee_action_scale,
            residual_scale=args.residual_scale,
        )
        env = BackwardPushRecoveryRewardWrapper(
            env,
            push_force=args.push_force,
            push_step=args.push_step,
            push_duration=args.push_duration,
            recovery_weight=args.recovery_weight,
            alive_after_push_bonus=args.alive_after_push_bonus,
        )
        return LocomotionRewardWrapper(
            env,
            forward_weight=args.forward_weight,
            lateral_weight=args.lateral_weight,
            energy_weight=args.energy_weight,
        )

    output = model_path(args.output)
    env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
    try:
        if args.model_in:
            model_in = Path(args.model_in)
            if not model_in.exists():
                raise SystemExit(f"model-in not found: {model_in}")
            model = PPO.load(
                model_in,
                env=env,
                device="auto",
                verbose=args.verbose,
                custom_objects={"learning_rate": args.learning_rate},
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                seed=args.seed,
                verbose=args.verbose,
                device="auto",
                n_steps=args.n_steps,
                learning_rate=args.learning_rate,
            )

        callback = None
        if args.checkpoint_dir:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callback = CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq // args.n_envs),
                save_path=str(checkpoint_dir),
                name_prefix=args.checkpoint_prefix,
            )
        model.learn(
            total_timesteps=args.total_timesteps,
            reset_num_timesteps=not args.model_in,
            callback=callback,
        )
        model.save(output)
    finally:
        env.close()

    print(f"saved_model={output}")
    print(f"model_in={args.model_in}")
    print(f"total_timesteps={args.total_timesteps}")
    print(f"learning_rate={args.learning_rate}")
    print(f"push_force={args.push_force}")
    print(f"push_step={args.push_step}")
    print(f"push_duration={args.push_duration}")
    print(f"recovery_weight={args.recovery_weight}")
    print(f"alive_after_push_bonus={args.alive_after_push_bonus}")
    print(f"checkpoint_dir={args.checkpoint_dir}")


if __name__ == "__main__":
    main()
