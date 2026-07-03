"""Record an Ant-v4 video using a handcrafted sinusoidal action pattern."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from sinusoidal_gait_controller import default_trot_patterns, target_at_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a handcrafted Ant-v4 gait video.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument("--steps", type=int, default=300, help="Maximum simulation steps.")
    parser.add_argument("--frequency", type=float, default=1.0, help="Action frequency in Hz.")
    parser.add_argument(
        "--action-sign",
        type=float,
        default=-1.0,
        help="Sign applied to all gait amplitudes.",
    )
    parser.add_argument("--action-scale", type=float, default=0.20, help="Hip action amplitude.")
    parser.add_argument(
        "--knee-action-scale",
        type=float,
        default=0.10,
        help="Knee action amplitude.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--fps", type=int, default=None, help="Override video FPS.")
    parser.add_argument(
        "--output",
        default="results/videos/sinusoidal_gait_ant.mp4",
        help="Output MP4 path.",
    )
    return parser.parse_args()


def make_action(patterns, frequency: float, t: float, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    action = np.array(
        [target_at_time(pattern, frequency, t) for pattern in patterns],
        dtype=np.float32,
    )
    return np.clip(action, low, high)


def base_xy(env) -> tuple[float, float]:
    qpos = env.unwrapped.data.qpos
    return float(qpos[0]), float(qpos[1])


def main() -> None:
    args = parse_args()

    if args.steps <= 0:
        raise SystemExit("steps must be positive.")
    if args.frequency < 0:
        raise SystemExit("frequency must be non-negative.")
    if args.action_sign == 0:
        raise SystemExit("action-sign must be non-zero.")
    if args.action_scale < 0 or args.knee_action_scale < 0:
        raise SystemExit("action amplitudes must be non-negative.")

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
        env = gym.make(args.env_id, render_mode="rgb_array")
    except Exception as exc:
        raise SystemExit(
            f"Could not create {args.env_id} with rgb_array rendering. "
            "On macOS, try running from a normal Terminal or VS Code terminal."
        ) from exc

    patterns = default_trot_patterns(
        amplitude_rad=args.action_sign * args.action_scale,
        knee_amplitude_rad=args.action_sign * args.knee_action_scale,
    )

    if len(patterns) != env.action_space.shape[0]:
        env.close()
        raise SystemExit(
            f"Pattern has {len(patterns)} actions, but env action space has "
            f"{env.action_space.shape[0]} dimensions."
        )

    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)
    fps = args.fps or int(env.metadata.get("render_fps", 30))
    sim_dt = float(getattr(env.unwrapped, "dt", 1.0 / fps))

    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False

    try:
        observation, info = env.reset(seed=args.seed)
        del observation, info
        start_x, start_y = base_xy(env)

        with imageio.get_writer(output_path, fps=fps) as writer:
            for step in range(args.steps):
                t = step * sim_dt
                action = make_action(patterns, args.frequency, t, low, high)
                observation, reward, terminated, truncated, info = env.step(action)
                del observation, info

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
    print(f"episode_return={episode_return:.3f}")
    print(f"episode_length={episode_length}")
    print(f"terminated={terminated}")
    print(f"truncated={truncated}")
    print(f"x_displacement={end_x - start_x:.4f}")
    print(f"y_displacement={end_y - start_y:.4f}")


if __name__ == "__main__":
    main()
