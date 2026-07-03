"""Evaluate a residual PPO policy on top of the handcrafted Ant gait."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_gait_wrapper import make_residual_ant_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual PPO on Ant-v4.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument("--model", default="results/logs/ant_residual_ppo_smoke.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--action-sign", type=float, default=-1.0)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--knee-action-scale", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def base_xy(env) -> tuple[float, float]:
    qpos = env.unwrapped.data.qpos
    return float(qpos[0]), float(qpos[1])


def main() -> None:
    args = parse_args()

    if args.episodes <= 0:
        raise SystemExit("episodes must be positive.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    try:
        import numpy as np
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    model = PPO.load(model_path)
    returns: list[float] = []
    lengths: list[int] = []
    x_displacements: list[float] = []
    y_displacements: list[float] = []

    env = make_residual_ant_env(
        env_id=args.env_id,
        frequency=args.frequency,
        action_sign=args.action_sign,
        action_scale=args.action_scale,
        knee_action_scale=args.knee_action_scale,
        residual_scale=args.residual_scale,
    )
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            del info
            start_x, start_y = base_xy(env)

            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _state = model.predict(
                    observation,
                    deterministic=args.deterministic,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                del info
                episode_return += float(reward)
                episode_length += 1

            end_x, end_y = base_xy(env)
            x_displacement = end_x - start_x
            y_displacement = end_y - start_y
            returns.append(episode_return)
            lengths.append(episode_length)
            x_displacements.append(x_displacement)
            y_displacements.append(y_displacement)
            print(
                f"episode={episode + 1} "
                f"return={episode_return:.3f} "
                f"length={episode_length} "
                f"x_displacement={x_displacement:.4f} "
                f"y_displacement={y_displacement:.4f}"
            )
    finally:
        env.close()

    print(f"mean_return={float(np.mean(returns)):.3f}")
    print(f"std_return={float(np.std(returns)):.3f}")
    print(f"mean_length={float(np.mean(lengths)):.1f}")
    print(f"mean_x_displacement={float(np.mean(x_displacements)):.4f}")
    print(f"mean_y_displacement={float(np.mean(y_displacements)):.4f}")


if __name__ == "__main__":
    main()
