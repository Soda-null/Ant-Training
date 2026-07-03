"""Evaluate whether a residual PPO Ant policy recovers after external pushes."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASELINE_DIR = Path(__file__).resolve().parents[1] / "05_rl_baselines"
if str(BASELINE_DIR) not in sys.path:
    sys.path.append(str(BASELINE_DIR))

from residual_gait_wrapper import make_residual_ant_env  # noqa: E402


@dataclass(frozen=True)
class PushConfig:
    label: str
    force_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class EpisodeResult:
    condition: str
    seed: int
    push_force: float
    push_direction: str
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    x_displacement: float
    y_displacement: float
    post_push_x_displacement: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ant push recovery.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument(
        "--model",
        default="results/logs/ant_shaped_residual_ppo_best_500k.zip",
        help="Path to a shaped residual PPO model.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per push condition.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="First evaluation seed.")
    parser.add_argument("--push-step", type=int, default=300, help="First step with push force.")
    parser.add_argument("--push-duration", type=int, default=10, help="Push duration in steps.")
    parser.add_argument("--push-forces", default="100,200,300", help="Comma-separated force magnitudes.")
    parser.add_argument("--directions", default="+x,-x,+y,-y", help="Comma-separated push directions.")
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--action-sign", type=float, default=-1.0)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--knee-action-scale", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--episodes-output", default="results/tables/push_recovery_episodes.csv")
    parser.add_argument("--summary-output", default="results/tables/push_recovery_summary.csv")
    return parser.parse_args()


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one value.")
    if any(value < 0 for value in values):
        raise SystemExit(f"{name} must be non-negative.")
    return values


def direction_vector(direction: str, magnitude: float) -> tuple[float, float, float]:
    mapping = {
        "+x": (magnitude, 0.0, 0.0),
        "-x": (-magnitude, 0.0, 0.0),
        "+y": (0.0, magnitude, 0.0),
        "-y": (0.0, -magnitude, 0.0),
    }
    if direction not in mapping:
        raise SystemExit(f"unsupported direction: {direction}")
    return mapping[direction]


def push_configs(forces: list[float], directions: list[str]) -> list[PushConfig]:
    configs = [PushConfig(label="no_push", force_xyz=(0.0, 0.0, 0.0))]
    for force in forces:
        for direction in directions:
            configs.append(
                PushConfig(
                    label=f"{direction}_{force:g}",
                    force_xyz=direction_vector(direction, force),
                )
            )
    return configs


def base_xy(env) -> tuple[float, float]:
    qpos = env.unwrapped.data.qpos
    return float(qpos[0]), float(qpos[1])


def apply_push(env, torso_id: int, force_xyz: tuple[float, float, float]) -> None:
    env.unwrapped.data.xfrc_applied[:] = 0.0
    env.unwrapped.data.xfrc_applied[torso_id, 0:3] = np.array(force_xyz, dtype=np.float64)


def run_episode(
    env,
    model,
    config: PushConfig,
    seed: int,
    max_steps: int,
    push_step: int,
    push_duration: int,
) -> EpisodeResult:
    observation, info = env.reset(seed=seed)
    del info
    torso_id = env.unwrapped.model.body("torso").id
    start_x, start_y = base_xy(env)
    post_push_start_x = start_x

    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False

    for step in range(max_steps):
        if push_step <= step < push_step + push_duration:
            apply_push(env, torso_id, config.force_xyz)
        else:
            apply_push(env, torso_id, (0.0, 0.0, 0.0))

        if step == push_step + push_duration:
            post_push_start_x, _post_push_y = base_xy(env)

        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        del info
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            break

    apply_push(env, torso_id, (0.0, 0.0, 0.0))
    end_x, end_y = base_xy(env)
    direction = config.label.split("_")[0] if config.label != "no_push" else "none"
    force = float(np.linalg.norm(np.array(config.force_xyz)))
    return EpisodeResult(
        condition=config.label,
        seed=seed,
        push_force=force,
        push_direction=direction,
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=terminated,
        truncated=truncated,
        x_displacement=end_x - start_x,
        y_displacement=end_y - start_y,
        post_push_x_displacement=end_x - post_push_start_x,
    )


def summarize(condition: str, results: list[EpisodeResult]) -> dict[str, float | str | int]:
    returns = np.array([result.episode_return for result in results], dtype=np.float64)
    lengths = np.array([result.episode_length for result in results], dtype=np.float64)
    x_disp = np.array([result.x_displacement for result in results], dtype=np.float64)
    abs_y = np.array([abs(result.y_displacement) for result in results], dtype=np.float64)
    post_push_x = np.array([result.post_push_x_displacement for result in results], dtype=np.float64)
    success = np.array([not result.terminated for result in results], dtype=np.float64)
    recovery = np.array(
        [not result.terminated and result.post_push_x_displacement > 0.0 for result in results],
        dtype=np.float64,
    )
    forward_score = x_disp - 0.2 * abs_y
    return {
        "condition": condition,
        "episodes": len(results),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(success)),
        "recovery_rate": float(np.mean(recovery)),
        "mean_x_displacement": float(np.mean(x_disp)),
        "mean_abs_y_displacement": float(np.mean(abs_y)),
        "mean_post_push_x_displacement": float(np.mean(post_push_x)),
        "mean_forward_score": float(np.mean(forward_score)),
    }


def write_episode_csv(results: list[EpisodeResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "seed",
        "push_force",
        "push_direction",
        "episode_return",
        "episode_length",
        "terminated",
        "truncated",
        "x_displacement",
        "y_displacement",
        "post_push_x_displacement",
        "forward_score",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            forward_score = result.x_displacement - 0.2 * abs(result.y_displacement)
            writer.writerow(
                {
                    "condition": result.condition,
                    "seed": result.seed,
                    "push_force": f"{result.push_force:.4f}",
                    "push_direction": result.push_direction,
                    "episode_return": f"{result.episode_return:.6f}",
                    "episode_length": result.episode_length,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "x_displacement": f"{result.x_displacement:.6f}",
                    "y_displacement": f"{result.y_displacement:.6f}",
                    "post_push_x_displacement": f"{result.post_push_x_displacement:.6f}",
                    "forward_score": f"{forward_score:.6f}",
                }
            )


def write_summary_csv(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "episodes",
        "mean_return",
        "mean_length",
        "success_rate",
        "recovery_rate",
        "mean_x_displacement",
        "mean_abs_y_displacement",
        "mean_post_push_x_displacement",
        "mean_forward_score",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "condition": row["condition"],
                    "episodes": row["episodes"],
                    "mean_return": f"{row['mean_return']:.6f}",
                    "mean_length": f"{row['mean_length']:.2f}",
                    "success_rate": f"{row['success_rate']:.4f}",
                    "recovery_rate": f"{row['recovery_rate']:.4f}",
                    "mean_x_displacement": f"{row['mean_x_displacement']:.6f}",
                    "mean_abs_y_displacement": f"{row['mean_abs_y_displacement']:.6f}",
                    "mean_post_push_x_displacement": f"{row['mean_post_push_x_displacement']:.6f}",
                    "mean_forward_score": f"{row['mean_forward_score']:.6f}",
                }
            )


def print_summary(rows: list[dict[str, float | str | int]]) -> None:
    print("condition episodes return length success recovery x_disp abs_y post_push_x score")
    for row in rows:
        print(
            f"{row['condition']:<10} "
            f"{row['episodes']:>8} "
            f"{row['mean_return']:>7.2f} "
            f"{row['mean_length']:>6.1f} "
            f"{row['success_rate']:>7.2f} "
            f"{row['recovery_rate']:>8.2f} "
            f"{row['mean_x_displacement']:>7.3f} "
            f"{row['mean_abs_y_displacement']:>7.3f} "
            f"{row['mean_post_push_x_displacement']:>11.3f} "
            f"{row['mean_forward_score']:>7.3f}"
        )


def main() -> None:
    args = parse_args()

    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes and max-steps must be positive.")
    if args.push_step < 0 or args.push_duration <= 0:
        raise SystemExit("push-step must be non-negative and push-duration must be positive.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    forces = parse_float_list(args.push_forces, "push-forces")
    directions = [item.strip() for item in args.directions.split(",") if item.strip()]
    configs = push_configs(forces, directions)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    model = PPO.load(model_path)
    env = make_residual_ant_env(
        env_id=args.env_id,
        frequency=args.frequency,
        action_sign=args.action_sign,
        action_scale=args.action_scale,
        knee_action_scale=args.knee_action_scale,
        residual_scale=args.residual_scale,
    )

    all_results: list[EpisodeResult] = []
    try:
        for config in configs:
            for episode in range(args.episodes):
                all_results.append(
                    run_episode(
                        env=env,
                        model=model,
                        config=config,
                        seed=args.seed + episode,
                        max_steps=args.max_steps,
                        push_step=args.push_step,
                        push_duration=args.push_duration,
                    )
                )
    finally:
        env.close()

    rows = [summarize(config.label, [r for r in all_results if r.condition == config.label]) for config in configs]
    rows.sort(key=lambda row: (float(row["push_force"]) if "push_force" in row else 0.0))
    write_episode_csv(all_results, Path(args.episodes_output))
    write_summary_csv(rows, Path(args.summary_output))
    print(f"saved_episodes={args.episodes_output}")
    print(f"saved_summary={args.summary_output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
