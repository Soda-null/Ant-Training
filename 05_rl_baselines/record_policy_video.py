"""Record an Ant-v4 video from a saved PPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a saved PPO policy video.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument(
        "--model",
        default="results/logs/ant_ppo_smoke.zip",
        help="Path to a saved Stable-Baselines3 model.",
    )
    parser.add_argument("--steps", type=int, default=300, help="Maximum video steps.")
    parser.add_argument("--seed", type=int, default=200, help="Environment seed.")
    parser.add_argument("--fps", type=int, default=None, help="Override video FPS.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions.",
    )
    parser.add_argument(
        "--output",
        default="results/videos/ant_ppo_policy.mp4",
        help="Output MP4 path.",
    )
    return parser.parse_args()


def base_xy(env) -> tuple[float, float]:
    qpos = env.unwrapped.data.qpos
    return float(qpos[0]), float(qpos[1])


def main() -> None:
    args = parse_args()

    if args.steps <= 0:
        raise SystemExit("steps must be positive.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    try:
        import gymnasium as gym
        import imageio.v2 as imageio
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO.load(model_path)

    try:
        env = gym.make(args.env_id, render_mode="rgb_array")
    except Exception as exc:
        raise SystemExit(
            f"Could not create {args.env_id} with rgb_array rendering. "
            "On macOS, try running from a normal Terminal or VS Code terminal."
        ) from exc

    fps = args.fps or int(env.metadata.get("render_fps", 30))
    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False

    try:
        observation, info = env.reset(seed=args.seed)
        del info
        start_x, start_y = base_xy(env)

        with imageio.get_writer(output_path, fps=fps) as writer:
            for _step in range(args.steps):
                action, _state = model.predict(
                    observation,
                    deterministic=args.deterministic,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                del info

                episode_return += float(reward)
                episode_length += 1
                writer.append_data(env.render())

                if terminated or truncated:
                    break

        end_x, end_y = base_xy(env)
    except Exception as exc:
        raise SystemExit(
            "Video recording failed. If this happens on macOS, rerun from a normal "
            "Terminal or VS Code terminal so MuJoCo can access the graphics stack."
        ) from exc
    finally:
        env.close()

    print(f"saved_video={output_path}")
    print(f"model={model_path}")
    print(f"episode_return={episode_return:.3f}")
    print(f"episode_length={episode_length}")
    print(f"terminated={terminated}")
    print(f"truncated={truncated}")
    print(f"x_displacement={end_x - start_x:.4f}")
    print(f"y_displacement={end_y - start_y:.4f}")


if __name__ == "__main__":
    main()
