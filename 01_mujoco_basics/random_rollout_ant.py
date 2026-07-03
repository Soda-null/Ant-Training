"""Run random actions in Gymnasium's Ant-v4 environment."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random rollout for Ant-v4.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Gymnasium is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    env = gym.make("Ant-v4")
    env.action_space.seed(args.seed)

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            del observation, info

            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                del observation, info

                episode_reward += float(reward)
                episode_length += 1

                if args.max_steps is not None and episode_length >= args.max_steps:
                    truncated = True

            print(
                f"episode={episode} "
                f"reward={episode_reward:.2f} "
                f"length={episode_length} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()

