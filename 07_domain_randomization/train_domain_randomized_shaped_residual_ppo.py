"""Train shaped residual PPO with simple MuJoCo domain randomization."""

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


class DomainRandomizationTrainingWrapper(gym.Wrapper):
    """Randomize mass, friction, and damping once at each episode reset."""

    def __init__(
        self,
        env: gym.Env,
        mass_range: tuple[float, float],
        friction_range: tuple[float, float],
        damping_range: tuple[float, float],
    ) -> None:
        super().__init__(env)
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.damping_range = damping_range
        model = self.env.unwrapped.model
        self.base_body_mass = model.body_mass.copy()
        self.base_geom_friction = model.geom_friction.copy()
        self.base_dof_damping = model.dof_damping.copy()
        self.rng = np.random.default_rng()

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        observation, info = self.env.reset(**kwargs)
        self.randomize_domain()
        return observation, info

    def randomize_domain(self) -> None:
        model = self.env.unwrapped.model
        mass_scale = float(self.rng.uniform(*self.mass_range))
        friction_scale = float(self.rng.uniform(*self.friction_range))
        damping_scale = float(self.rng.uniform(*self.damping_range))

        model.body_mass[:] = self.base_body_mass * mass_scale
        model.geom_friction[:] = self.base_geom_friction
        model.geom_friction[:, 0] = self.base_geom_friction[:, 0] * friction_scale
        model.dof_damping[:] = self.base_dof_damping * damping_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train domain-randomized shaped residual PPO.")
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
    parser.add_argument("--mass-range", default="0.8,1.2")
    parser.add_argument("--friction-range", default="0.7,1.3")
    parser.add_argument("--damping-range", default="0.7,1.3")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--model-in",
        default=None,
        help="Optional existing PPO model to continue training.",
    )
    parser.add_argument(
        "--output",
        default="results/logs/ant_domain_randomized_shaped_residual_ppo_smoke",
        help="Output model path without or with .zip.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional directory for intermediate checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25_000,
        help="Save a checkpoint every N environment steps when checkpoint-dir is set.",
    )
    parser.add_argument("--checkpoint-prefix", default="dr_checkpoint")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def parse_range(raw: str, name: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 2:
        raise SystemExit(f"{name} must be two comma-separated values, e.g. 0.8,1.2")
    low, high = values
    if low <= 0 or high <= 0 or low > high:
        raise SystemExit(f"{name} must be positive and ordered low,high.")
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
    if args.checkpoint_freq <= 0:
        raise SystemExit("checkpoint-freq must be positive.")

    mass_range = parse_range(args.mass_range, "mass-range")
    friction_range = parse_range(args.friction_range, "friction-range")
    damping_range = parse_range(args.damping_range, "damping-range")

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
        env = DomainRandomizationTrainingWrapper(
            env,
            mass_range=mass_range,
            friction_range=friction_range,
            damping_range=damping_range,
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
    print(f"env_id={args.env_id}")
    print(f"total_timesteps={args.total_timesteps}")
    print(f"model_in={args.model_in}")
    print(f"mass_range={args.mass_range}")
    print(f"friction_range={args.friction_range}")
    print(f"damping_range={args.damping_range}")
    print(f"learning_rate={args.learning_rate}")
    print(f"forward_weight={args.forward_weight}")
    print(f"lateral_weight={args.lateral_weight}")
    print(f"energy_weight={args.energy_weight}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"checkpoint_freq={args.checkpoint_freq}")


if __name__ == "__main__":
    main()
