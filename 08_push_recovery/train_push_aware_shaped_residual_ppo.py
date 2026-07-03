"""Train shaped residual PPO with random external pushes."""

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


class PushAwareTrainingWrapper(gym.Wrapper):
    """Apply one random short push during each training episode."""

    def __init__(
        self,
        env: gym.Env,
        push_probability: float,
        force_range: tuple[float, float],
        push_step_range: tuple[int, int],
        push_duration: int,
        directions: tuple[str, ...],
    ) -> None:
        super().__init__(env)
        self.push_probability = push_probability
        self.force_range = force_range
        self.push_step_range = push_step_range
        self.push_duration = push_duration
        self.directions = directions
        self.rng = np.random.default_rng()
        self.step_index = 0
        self.push_start = 0
        self.push_end = 0
        self.force_xyz = np.zeros(3, dtype=np.float64)
        self.torso_id = self.env.unwrapped.model.body("torso").id

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        observation, info = self.env.reset(**kwargs)
        self.step_index = 0
        self.env.unwrapped.data.xfrc_applied[:] = 0.0
        self.sample_push()
        return observation, info

    def step(self, action):
        if self.push_start <= self.step_index < self.push_end:
            self.env.unwrapped.data.xfrc_applied[self.torso_id, 0:3] = self.force_xyz
        else:
            self.env.unwrapped.data.xfrc_applied[:] = 0.0

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_index += 1
        if terminated or truncated:
            self.env.unwrapped.data.xfrc_applied[:] = 0.0
        return observation, reward, terminated, truncated, info

    def sample_push(self) -> None:
        if self.rng.random() > self.push_probability:
            self.push_start = 10**9
            self.push_end = 10**9
            self.force_xyz[:] = 0.0
            return

        low_step, high_step = self.push_step_range
        force = float(self.rng.uniform(*self.force_range))
        direction = str(self.rng.choice(self.directions))
        self.push_start = int(self.rng.integers(low_step, high_step + 1))
        self.push_end = self.push_start + self.push_duration
        self.force_xyz[:] = self.direction_vector(direction, force)

    @staticmethod
    def direction_vector(direction: str, magnitude: float) -> tuple[float, float, float]:
        mapping = {
            "+x": (magnitude, 0.0, 0.0),
            "-x": (-magnitude, 0.0, 0.0),
            "+y": (0.0, magnitude, 0.0),
            "-y": (0.0, -magnitude, 0.0),
        }
        if direction not in mapping:
            raise ValueError(f"unsupported direction: {direction}")
        return mapping[direction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train push-aware shaped residual PPO.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
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
    parser.add_argument("--push-probability", type=float, default=0.8)
    parser.add_argument("--force-range", default="5,10")
    parser.add_argument("--push-step-range", default="200,500")
    parser.add_argument("--push-duration", type=int, default=10)
    parser.add_argument("--directions", default="+x,-x,+y,-y")
    parser.add_argument("--model-in", default=None)
    parser.add_argument(
        "--output",
        default="results/logs/ant_push_aware_shaped_residual_ppo_smoke",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-prefix", default="push_aware")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def parse_float_range(raw: str, name: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 2:
        raise SystemExit(f"{name} must be two comma-separated values.")
    low, high = values
    if low < 0 or high < 0 or low > high:
        raise SystemExit(f"{name} must be non-negative and ordered low,high.")
    return low, high


def parse_int_range(raw: str, name: str) -> tuple[int, int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 2:
        raise SystemExit(f"{name} must be two comma-separated integer values.")
    low, high = values
    if low < 0 or low > high:
        raise SystemExit(f"{name} must be non-negative and ordered low,high.")
    return low, high


def model_path(path: str) -> Path:
    output = Path(path)
    if output.suffix != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0:
        raise SystemExit("total-timesteps must be positive.")
    if args.n_envs <= 0:
        raise SystemExit("n-envs must be positive.")
    if args.n_steps <= 0:
        raise SystemExit("n-steps must be positive.")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive.")
    if not 0.0 <= args.push_probability <= 1.0:
        raise SystemExit("push-probability must be in [0, 1].")
    if args.push_duration <= 0:
        raise SystemExit("push-duration must be positive.")
    if args.checkpoint_freq <= 0:
        raise SystemExit("checkpoint-freq must be positive.")

    force_range = parse_float_range(args.force_range, "force-range")
    push_step_range = parse_int_range(args.push_step_range, "push-step-range")
    directions = tuple(item.strip() for item in args.directions.split(",") if item.strip())
    if not directions:
        raise SystemExit("directions must contain at least one direction.")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    def make_env():
        env = make_residual_ant_env(
            env_id=args.env_id,
            frequency=args.frequency,
            action_sign=args.action_sign,
            action_scale=args.action_scale,
            knee_action_scale=args.knee_action_scale,
            residual_scale=args.residual_scale,
        )
        env = PushAwareTrainingWrapper(
            env,
            push_probability=args.push_probability,
            force_range=force_range,
            push_step_range=push_step_range,
            push_duration=args.push_duration,
            directions=directions,
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
                save_replay_buffer=False,
                save_vecnormalize=False,
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
    print(f"push_probability={args.push_probability}")
    print(f"force_range={args.force_range}")
    print(f"push_step_range={args.push_step_range}")
    print(f"push_duration={args.push_duration}")
    print(f"directions={args.directions}")
    print(f"checkpoint_dir={args.checkpoint_dir}")


if __name__ == "__main__":
    main()
