"""Inspect basic state and spaces for Gymnasium's Ant-v4 environment."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Ant-v4 state values.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="Number of observation values to print.",
    )
    return parser.parse_args()


def format_values(values, count: int) -> str:
    import numpy as np

    sample = values[:count]
    return np.array2string(sample, precision=3, suppress_small=True)


def main() -> None:
    args = parse_args()

    try:
        import gymnasium as gym
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "A dependency is missing. Run `python -m pip install -r requirements.txt`."
        ) from exc

    env = gym.make("Ant-v4")
    env.action_space.seed(args.seed)

    try:
        observation, info = env.reset(seed=args.seed)
        del info

        print(f"observation_space={env.observation_space}")
        print(f"action_space={env.action_space}")
        print(f"first_observation_shape={observation.shape}")
        print(
            "first_observation_sample="
            f"{format_values(np.asarray(observation), args.sample_count)}"
        )

        data = getattr(env.unwrapped, "data", None)
        if data is not None:
            # qpos and qvel are MuJoCo's generalized position and velocity vectors.
            print(f"qpos_shape={data.qpos.shape}")
            print(f"qvel_shape={data.qvel.shape}")
            print(f"ctrl_shape={data.ctrl.shape}")
            print(f"qpos_sample={format_values(data.qpos, args.sample_count)}")
            print(f"qvel_sample={format_values(data.qvel, args.sample_count)}")

        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        del next_observation, info

        print("after_one_random_step:")
        print(f"  action_shape={action.shape}")
        print(f"  reward={float(reward):.4f}")
        print(f"  terminated={terminated}")
        print(f"  truncated={truncated}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
