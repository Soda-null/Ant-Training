"""Evaluate a saved Stable-Baselines3 policy on Ant-v4."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO policy.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument(
        "--model",
        default="results/logs/ant_ppo_smoke.zip",
        help="Path to a saved Stable-Baselines3 model.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Evaluation episodes.")
    parser.add_argument("--seed", type=int, default=100, help="Environment seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions.",
    )
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
        import gymnasium as gym
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

    env = gym.make(args.env_id)
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

            returns.append(episode_return)
            lengths.append(episode_length)
            end_x, end_y = base_xy(env)
            x_displacement = end_x - start_x
            y_displacement = end_y - start_y
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
