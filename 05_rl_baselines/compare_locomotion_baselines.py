"""Compare random, handcrafted, vanilla PPO, and residual PPO locomotion."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GAIT_DIR = Path(__file__).resolve().parents[1] / "04_gait_controller"
if str(GAIT_DIR) not in sys.path:
    sys.path.append(str(GAIT_DIR))

from residual_gait_wrapper import make_residual_ant_env  # noqa: E402
from sinusoidal_gait_controller import default_trot_patterns, target_at_time  # noqa: E402


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    seed: int
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    x_displacement: float
    y_displacement: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare locomotion baselines on Ant-v4.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per policy.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="First evaluation seed.")
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--action-sign", type=float, default=-1.0)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--knee-action-scale", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--vanilla-model", default="results/logs/ant_ppo_smoke.zip")
    parser.add_argument("--residual-model", default="results/logs/ant_residual_ppo_smoke.zip")
    parser.add_argument(
        "--shaped-residual-model",
        default="results/logs/ant_shaped_residual_ppo_best_100k.zip",
    )
    parser.add_argument(
        "--output",
        default="results/tables/locomotion_baseline_comparison.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PPO actions.")
    return parser.parse_args()


def base_xy(env) -> tuple[float, float]:
    qpos = env.unwrapped.data.qpos
    return float(qpos[0]), float(qpos[1])


def gait_action(
    env,
    frequency: float,
    action_sign: float,
    action_scale: float,
    knee_action_scale: float,
    step: int,
) -> np.ndarray:
    patterns = default_trot_patterns(
        amplitude_rad=action_sign * action_scale,
        knee_amplitude_rad=action_sign * knee_action_scale,
    )
    sim_dt = float(getattr(env.unwrapped, "dt", 1.0 / 30.0))
    t = step * sim_dt
    action = np.array(
        [target_at_time(pattern, frequency, t) for pattern in patterns],
        dtype=np.float32,
    )
    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)
    return np.clip(action, low, high)


def run_episode(env, policy: str, seed: int, max_steps: int, action_fn) -> EpisodeResult:
    observation, info = env.reset(seed=seed)
    del info
    start_x, start_y = base_xy(env)

    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False

    for step in range(max_steps):
        action = action_fn(env, observation, step)
        observation, reward, terminated, truncated, info = env.step(action)
        del info
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            break

    end_x, end_y = base_xy(env)
    return EpisodeResult(
        policy=policy,
        seed=seed,
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=terminated,
        truncated=truncated,
        x_displacement=end_x - start_x,
        y_displacement=end_y - start_y,
    )


def summarize(policy: str, results: list[EpisodeResult]) -> dict[str, float | str | int]:
    returns = np.array([result.episode_return for result in results], dtype=np.float64)
    lengths = np.array([result.episode_length for result in results], dtype=np.float64)
    x_disp = np.array([result.x_displacement for result in results], dtype=np.float64)
    y_disp = np.array([result.y_displacement for result in results], dtype=np.float64)
    success = np.array([not result.terminated for result in results], dtype=np.float64)
    forward_scores = x_disp - 0.2 * np.abs(y_disp)
    return {
        "policy": policy,
        "episodes": len(results),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(success)),
        "mean_x_displacement": float(np.mean(x_disp)),
        "mean_abs_y_displacement": float(np.mean(np.abs(y_disp))),
        "mean_forward_score": float(np.mean(forward_scores)),
    }


def write_csv(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "episodes",
        "mean_return",
        "mean_length",
        "success_rate",
        "mean_x_displacement",
        "mean_abs_y_displacement",
        "mean_forward_score",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "policy": row["policy"],
                    "episodes": row["episodes"],
                    "mean_return": f"{row['mean_return']:.6f}",
                    "mean_length": f"{row['mean_length']:.2f}",
                    "success_rate": f"{row['success_rate']:.4f}",
                    "mean_x_displacement": f"{row['mean_x_displacement']:.6f}",
                    "mean_abs_y_displacement": f"{row['mean_abs_y_displacement']:.6f}",
                    "mean_forward_score": f"{row['mean_forward_score']:.6f}",
                }
            )


def print_rows(rows: list[dict[str, float | str | int]]) -> None:
    print("policy episodes return length success x_disp abs_y_disp forward_score")
    for row in rows:
        print(
            f"{row['policy']:<18} "
            f"{row['episodes']:>8} "
            f"{row['mean_return']:>7.2f} "
            f"{row['mean_length']:>6.1f} "
            f"{row['success_rate']:>7.2f} "
            f"{row['mean_x_displacement']:>7.3f} "
            f"{row['mean_abs_y_displacement']:>10.3f} "
            f"{row['mean_forward_score']:>13.3f}"
        )


def main() -> None:
    args = parse_args()

    if args.episodes <= 0:
        raise SystemExit("episodes must be positive.")
    if args.max_steps <= 0:
        raise SystemExit("max-steps must be positive.")

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    seeds = [args.seed + index for index in range(args.episodes)]
    grouped: dict[str, list[EpisodeResult]] = {}

    env = gym.make(args.env_id)
    try:
        grouped["random"] = [
            run_episode(
                env,
                policy="random",
                seed=seed,
                max_steps=args.max_steps,
                action_fn=lambda env, _obs, _step: env.action_space.sample(),
            )
            for seed in seeds
        ]
    finally:
        env.close()

    env = gym.make(args.env_id)
    try:
        grouped["handcrafted_gait"] = [
            run_episode(
                env,
                policy="handcrafted_gait",
                seed=seed,
                max_steps=args.max_steps,
                action_fn=lambda env, _obs, step: gait_action(
                    env=env,
                    frequency=args.frequency,
                    action_sign=args.action_sign,
                    action_scale=args.action_scale,
                    knee_action_scale=args.knee_action_scale,
                    step=step,
                ),
            )
            for seed in seeds
        ]
    finally:
        env.close()

    vanilla_model_path = Path(args.vanilla_model)
    if vanilla_model_path.exists():
        vanilla_model = PPO.load(vanilla_model_path)
        env = gym.make(args.env_id)
        try:
            grouped["vanilla_ppo"] = [
                run_episode(
                    env,
                    policy="vanilla_ppo",
                    seed=seed,
                    max_steps=args.max_steps,
                    action_fn=lambda _env, obs, _step: vanilla_model.predict(
                        obs,
                        deterministic=args.deterministic,
                    )[0],
                )
                for seed in seeds
            ]
        finally:
            env.close()
    else:
        print(f"skip vanilla_ppo: model not found at {vanilla_model_path}")

    residual_model_path = Path(args.residual_model)
    if residual_model_path.exists():
        residual_model = PPO.load(residual_model_path)
        env = make_residual_ant_env(
            env_id=args.env_id,
            frequency=args.frequency,
            action_sign=args.action_sign,
            action_scale=args.action_scale,
            knee_action_scale=args.knee_action_scale,
            residual_scale=args.residual_scale,
        )
        try:
            grouped["residual_ppo"] = [
                run_episode(
                    env,
                    policy="residual_ppo",
                    seed=seed,
                    max_steps=args.max_steps,
                    action_fn=lambda _env, obs, _step: residual_model.predict(
                        obs,
                        deterministic=args.deterministic,
                    )[0],
                )
                for seed in seeds
            ]
        finally:
            env.close()
    else:
        print(f"skip residual_ppo: model not found at {residual_model_path}")

    shaped_residual_model_path = Path(args.shaped_residual_model)
    if shaped_residual_model_path.exists():
        shaped_residual_model = PPO.load(shaped_residual_model_path)
        env = make_residual_ant_env(
            env_id=args.env_id,
            frequency=args.frequency,
            action_sign=args.action_sign,
            action_scale=args.action_scale,
            knee_action_scale=args.knee_action_scale,
            residual_scale=args.residual_scale,
        )
        try:
            grouped["shaped_residual_ppo"] = [
                run_episode(
                    env,
                    policy="shaped_residual_ppo",
                    seed=seed,
                    max_steps=args.max_steps,
                    action_fn=lambda _env, obs, _step: shaped_residual_model.predict(
                        obs,
                        deterministic=args.deterministic,
                    )[0],
                )
                for seed in seeds
            ]
        finally:
            env.close()
    else:
        print(f"skip shaped_residual_ppo: model not found at {shaped_residual_model_path}")

    rows = [summarize(policy, results) for policy, results in grouped.items()]
    rows.sort(key=lambda row: float(row["mean_forward_score"]), reverse=True)

    output_path = Path(args.output)
    write_csv(rows, output_path)
    print(f"saved_table={output_path}")
    print_rows(rows)


if __name__ == "__main__":
    main()
