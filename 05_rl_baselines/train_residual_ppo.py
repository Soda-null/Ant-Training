"""Train PPO residual actions on top of the handcrafted Ant gait."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_gait_wrapper import make_residual_ant_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual PPO on Ant-v4.")
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
    parser.add_argument(
        "--output",
        default="results/logs/ant_residual_ppo_smoke",
        help="Output model path without or with .zip.",
    )
    return parser.parse_args()


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

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    def make_env():
        return make_residual_ant_env(
            env_id=args.env_id,
            frequency=args.frequency,
            action_sign=args.action_sign,
            action_scale=args.action_scale,
            knee_action_scale=args.knee_action_scale,
            residual_scale=args.residual_scale,
        )

    output = model_path(args.output)
    env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    try:
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=1,
            device="auto",
            n_steps=args.n_steps,
        )
        model.learn(total_timesteps=args.total_timesteps)
        model.save(output)
    finally:
        env.close()

    print(f"saved_model={output}")
    print(f"env_id={args.env_id}")
    print(f"total_timesteps={args.total_timesteps}")
    print(f"n_steps={args.n_steps}")
    print(f"residual_scale={args.residual_scale}")
    print(f"base_frequency={args.frequency}")
    print(f"base_action_sign={args.action_sign}")
    print(f"base_action_scale={args.action_scale}")
    print(f"base_knee_action_scale={args.knee_action_scale}")


if __name__ == "__main__":
    main()
