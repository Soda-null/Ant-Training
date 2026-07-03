"""Record a short random-action Ant-v4 video."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record random Ant-v4 video.")
    parser.add_argument(
        "--output",
        default="results/videos/random_ant.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument("--steps", type=int, default=300, help="Maximum video steps.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import gymnasium as gym
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        env = gym.make("Ant-v4", render_mode="rgb_array")
    except Exception as exc:
        raise SystemExit(
            "Could not create Ant-v4 with rgb_array rendering. On macOS, check that "
            "`mujoco` and `gymnasium[mujoco]` installed successfully."
        ) from exc

    env.action_space.seed(args.seed)

    try:
        observation, info = env.reset(seed=args.seed)
        del observation, info

        with imageio.get_writer(output_path, fps=args.fps) as writer:
            for _ in range(args.steps):
                frame = env.render()
                writer.append_data(frame)

                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                del observation, reward, info

                if terminated or truncated:
                    break
    except Exception as exc:
        raise SystemExit(
            "Video recording failed. If this happens on macOS, try running from a "
            "normal Terminal or VS Code terminal after reinstalling `mujoco`, "
            "`imageio`, and `imageio-ffmpeg`."
        ) from exc
    finally:
        env.close()

    print(f"saved_video={output_path}")


if __name__ == "__main__":
    main()

