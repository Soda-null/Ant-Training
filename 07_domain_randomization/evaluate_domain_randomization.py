"""Evaluate a residual PPO policy under simple MuJoCo domain randomization."""

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
class DomainParams:
    mass_scale: float
    friction_scale: float
    damping_scale: float


@dataclass(frozen=True)
class EpisodeResult:
    condition: str
    seed: int
    mass_scale: float
    friction_scale: float
    damping_scale: float
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    x_displacement: float
    y_displacement: float


class DomainRandomizer:
    def __init__(self, env) -> None:
        model = env.unwrapped.model
        self.env = env
        self.body_mass = model.body_mass.copy()
        self.geom_friction = model.geom_friction.copy()
        self.dof_damping = model.dof_damping.copy()

    def apply(self, params: DomainParams) -> None:
        model = self.env.unwrapped.model
        model.body_mass[:] = self.body_mass * params.mass_scale
        model.geom_friction[:] = self.geom_friction
        model.geom_friction[:, 0] = self.geom_friction[:, 0] * params.friction_scale
        model.dof_damping[:] = self.dof_damping * params.damping_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate domain randomization robustness.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument(
        "--model",
        default="results/logs/ant_shaped_residual_ppo_best_500k.zip",
        help="Path to a shaped residual PPO model.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per condition.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="First evaluation seed.")
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--action-sign", type=float, default=-1.0)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--knee-action-scale", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--mass-range", default="0.8,1.2")
    parser.add_argument("--friction-range", default="0.7,1.3")
    parser.add_argument("--damping-range", default="0.7,1.3")
    parser.add_argument(
        "--episodes-output",
        default="results/tables/domain_randomization_episodes.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/tables/domain_randomization_summary.csv",
    )
    return parser.parse_args()


def parse_range(raw: str, name: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise SystemExit(f"{name} must be two comma-separated values, e.g. 0.8,1.2")
    low, high = parts
    if low <= 0 or high <= 0 or low > high:
        raise SystemExit(f"{name} must be positive and ordered low,high.")
    return low, high


def base_xy(env) -> tuple[float, float]:
    qpos = env.unwrapped.data.qpos
    return float(qpos[0]), float(qpos[1])


def sample_params(
    rng: np.random.Generator,
    mass_range: tuple[float, float],
    friction_range: tuple[float, float],
    damping_range: tuple[float, float],
) -> DomainParams:
    return DomainParams(
        mass_scale=float(rng.uniform(*mass_range)),
        friction_scale=float(rng.uniform(*friction_range)),
        damping_scale=float(rng.uniform(*damping_range)),
    )


def run_episode(env, model, randomizer: DomainRandomizer, condition: str, seed: int, params: DomainParams, max_steps: int) -> EpisodeResult:
    observation, info = env.reset(seed=seed)
    del info
    randomizer.apply(params)
    start_x, start_y = base_xy(env)

    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False

    for _step in range(max_steps):
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        del info
        episode_return += float(reward)
        episode_length += 1
        if terminated or truncated:
            break

    end_x, end_y = base_xy(env)
    return EpisodeResult(
        condition=condition,
        seed=seed,
        mass_scale=params.mass_scale,
        friction_scale=params.friction_scale,
        damping_scale=params.damping_scale,
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=terminated,
        truncated=truncated,
        x_displacement=end_x - start_x,
        y_displacement=end_y - start_y,
    )


def summarize(condition: str, results: list[EpisodeResult]) -> dict[str, float | str | int]:
    returns = np.array([result.episode_return for result in results], dtype=np.float64)
    lengths = np.array([result.episode_length for result in results], dtype=np.float64)
    x_disp = np.array([result.x_displacement for result in results], dtype=np.float64)
    abs_y = np.array([abs(result.y_displacement) for result in results], dtype=np.float64)
    success = np.array([not result.terminated for result in results], dtype=np.float64)
    forward_score = x_disp - 0.2 * abs_y
    return {
        "condition": condition,
        "episodes": len(results),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(success)),
        "mean_x_displacement": float(np.mean(x_disp)),
        "mean_abs_y_displacement": float(np.mean(abs_y)),
        "mean_forward_score": float(np.mean(forward_score)),
    }


def write_episode_csv(results: list[EpisodeResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "seed",
        "mass_scale",
        "friction_scale",
        "damping_scale",
        "episode_return",
        "episode_length",
        "terminated",
        "truncated",
        "x_displacement",
        "y_displacement",
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
                    "mass_scale": f"{result.mass_scale:.6f}",
                    "friction_scale": f"{result.friction_scale:.6f}",
                    "damping_scale": f"{result.damping_scale:.6f}",
                    "episode_return": f"{result.episode_return:.6f}",
                    "episode_length": result.episode_length,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "x_displacement": f"{result.x_displacement:.6f}",
                    "y_displacement": f"{result.y_displacement:.6f}",
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
                    "condition": row["condition"],
                    "episodes": row["episodes"],
                    "mean_return": f"{row['mean_return']:.6f}",
                    "mean_length": f"{row['mean_length']:.2f}",
                    "success_rate": f"{row['success_rate']:.4f}",
                    "mean_x_displacement": f"{row['mean_x_displacement']:.6f}",
                    "mean_abs_y_displacement": f"{row['mean_abs_y_displacement']:.6f}",
                    "mean_forward_score": f"{row['mean_forward_score']:.6f}",
                }
            )


def print_summary(rows: list[dict[str, float | str | int]]) -> None:
    print("condition episodes return length success x_disp abs_y_disp forward_score")
    for row in rows:
        print(
            f"{row['condition']:<12} "
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

    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes and max-steps must be positive.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    mass_range = parse_range(args.mass_range, "mass-range")
    friction_range = parse_range(args.friction_range, "friction-range")
    damping_range = parse_range(args.damping_range, "damping-range")

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
    randomizer = DomainRandomizer(env)
    rng = np.random.default_rng(args.seed)

    all_results: list[EpisodeResult] = []
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            all_results.append(
                run_episode(
                    env=env,
                    model=model,
                    randomizer=randomizer,
                    condition="nominal",
                    seed=seed,
                    params=DomainParams(1.0, 1.0, 1.0),
                    max_steps=args.max_steps,
                )
            )

        for index in range(args.episodes):
            seed = args.seed + index
            params = sample_params(rng, mass_range, friction_range, damping_range)
            all_results.append(
                run_episode(
                    env=env,
                    model=model,
                    randomizer=randomizer,
                    condition="randomized",
                    seed=seed,
                    params=params,
                    max_steps=args.max_steps,
                )
            )
    finally:
        env.close()

    rows = [
        summarize("nominal", [result for result in all_results if result.condition == "nominal"]),
        summarize("randomized", [result for result in all_results if result.condition == "randomized"]),
    ]
    write_episode_csv(all_results, Path(args.episodes_output))
    write_summary_csv(rows, Path(args.summary_output))
    print(f"saved_episodes={args.episodes_output}")
    print(f"saved_summary={args.summary_output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
