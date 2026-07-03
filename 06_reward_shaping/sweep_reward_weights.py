"""Sweep shaped residual PPO reward weights and save a ranked table."""

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

from locomotion_reward_wrapper import LocomotionRewardWrapper  # noqa: E402
from residual_gait_wrapper import make_residual_ant_env  # noqa: E402


@dataclass(frozen=True)
class SweepRow:
    rank: int
    forward_weight: float
    lateral_weight: float
    energy_weight: float
    mean_return: float
    mean_length: float
    success_rate: float
    mean_x_displacement: float
    mean_abs_y_displacement: float
    mean_forward_score: float
    rank_score: float


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep shaped residual PPO reward weights.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument("--total-timesteps", type=int, default=4_096)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--forward-weights", default="0.8,1.0,1.2")
    parser.add_argument("--lateral-weights", default="0.5,0.8,1.0,1.2")
    parser.add_argument("--energy-weights", default="0.005,0.01")
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--action-sign", type=float, default=-1.0)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--knee-action-scale", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="results/tables/reward_weight_sweep.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--best-model",
        default="results/logs/ant_shaped_residual_ppo_best",
        help="Best model path without or with .zip.",
    )
    return parser.parse_args()


def model_path(path: str) -> Path:
    output = Path(path)
    if output.suffix != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def base_xy(env) -> tuple[float, float]:
    qpos = env.unwrapped.data.qpos
    return float(qpos[0]), float(qpos[1])


def make_train_env(args: argparse.Namespace, forward: float, lateral: float, energy: float):
    env = make_residual_ant_env(
        env_id=args.env_id,
        frequency=args.frequency,
        action_sign=args.action_sign,
        action_scale=args.action_scale,
        knee_action_scale=args.knee_action_scale,
        residual_scale=args.residual_scale,
    )
    return LocomotionRewardWrapper(
        env,
        forward_weight=forward,
        lateral_weight=lateral,
        energy_weight=energy,
    )


def make_eval_env(args: argparse.Namespace):
    return make_residual_ant_env(
        env_id=args.env_id,
        frequency=args.frequency,
        action_sign=args.action_sign,
        action_scale=args.action_scale,
        knee_action_scale=args.knee_action_scale,
        residual_scale=args.residual_scale,
    )


def evaluate_model(model, args: argparse.Namespace) -> dict[str, float]:
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    x_displacements: list[float] = []
    abs_y_displacements: list[float] = []
    forward_scores: list[float] = []

    env = make_eval_env(args)
    try:
        for episode in range(args.eval_episodes):
            observation, info = env.reset(seed=args.seed + episode)
            del info
            start_x, start_y = base_xy(env)
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            for _step in range(args.eval_steps):
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                del info
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated:
                    break

            end_x, end_y = base_xy(env)
            x_displacement = end_x - start_x
            y_displacement = end_y - start_y
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(float(not terminated))
            x_displacements.append(x_displacement)
            abs_y_displacements.append(abs(y_displacement))
            forward_scores.append(x_displacement - 0.2 * abs(y_displacement))
    finally:
        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "mean_x_displacement": float(np.mean(x_displacements)),
        "mean_abs_y_displacement": float(np.mean(abs_y_displacements)),
        "mean_forward_score": float(np.mean(forward_scores)),
    }


def write_csv(rows: list[SweepRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "forward_weight",
        "lateral_weight",
        "energy_weight",
        "mean_return",
        "mean_length",
        "success_rate",
        "mean_x_displacement",
        "mean_abs_y_displacement",
        "mean_forward_score",
        "rank_score",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row.rank,
                    "forward_weight": f"{row.forward_weight:.4f}",
                    "lateral_weight": f"{row.lateral_weight:.4f}",
                    "energy_weight": f"{row.energy_weight:.4f}",
                    "mean_return": f"{row.mean_return:.6f}",
                    "mean_length": f"{row.mean_length:.2f}",
                    "success_rate": f"{row.success_rate:.4f}",
                    "mean_x_displacement": f"{row.mean_x_displacement:.6f}",
                    "mean_abs_y_displacement": f"{row.mean_abs_y_displacement:.6f}",
                    "mean_forward_score": f"{row.mean_forward_score:.6f}",
                    "rank_score": f"{row.rank_score:.6f}",
                }
            )


def print_rows(rows: list[SweepRow], top_k: int = 8) -> None:
    print("rank fw lat energy return length success x_disp abs_y score rank_score")
    for row in rows[:top_k]:
        print(
            f"{row.rank:>4} "
            f"{row.forward_weight:>4.2f} "
            f"{row.lateral_weight:>4.2f} "
            f"{row.energy_weight:>6.3f} "
            f"{row.mean_return:>7.2f} "
            f"{row.mean_length:>6.1f} "
            f"{row.success_rate:>7.2f} "
            f"{row.mean_x_displacement:>7.3f} "
            f"{row.mean_abs_y_displacement:>7.3f} "
            f"{row.mean_forward_score:>7.3f} "
            f"{row.rank_score:>10.3f}"
        )


def main() -> None:
    args = parse_args()

    if args.total_timesteps <= 0 or args.n_steps <= 0:
        raise SystemExit("total-timesteps and n-steps must be positive.")
    if args.eval_episodes <= 0 or args.eval_steps <= 0:
        raise SystemExit("eval-episodes and eval-steps must be positive.")

    forward_weights = parse_float_list(args.forward_weights, "forward-weights")
    lateral_weights = parse_float_list(args.lateral_weights, "lateral-weights")
    energy_weights = parse_float_list(args.energy_weights, "energy-weights")

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    rows: list[SweepRow] = []
    best_model = None
    best_score = -float("inf")
    config_index = 0

    for forward in forward_weights:
        for lateral in lateral_weights:
            for energy in energy_weights:
                train_env = make_train_env(args, forward, lateral, energy)
                try:
                    model = PPO(
                        "MlpPolicy",
                        train_env,
                        seed=args.seed + config_index,
                        verbose=0,
                        device="auto",
                        n_steps=args.n_steps,
                    )
                    model.learn(total_timesteps=args.total_timesteps)
                finally:
                    train_env.close()

                metrics = evaluate_model(model, args)
                rank_score = (
                    metrics["mean_forward_score"]
                    + 0.005 * metrics["mean_length"]
                    + 0.5 * metrics["success_rate"]
                )
                rows.append(
                    SweepRow(
                        rank=0,
                        forward_weight=forward,
                        lateral_weight=lateral,
                        energy_weight=energy,
                        mean_return=metrics["mean_return"],
                        mean_length=metrics["mean_length"],
                        success_rate=metrics["success_rate"],
                        mean_x_displacement=metrics["mean_x_displacement"],
                        mean_abs_y_displacement=metrics["mean_abs_y_displacement"],
                        mean_forward_score=metrics["mean_forward_score"],
                        rank_score=rank_score,
                    )
                )

                if rank_score > best_score:
                    best_score = rank_score
                    best_model = model

                config_index += 1
                print(
                    f"finished config={config_index} "
                    f"fw={forward} lat={lateral} energy={energy} "
                    f"rank_score={rank_score:.3f}"
                )

    rows.sort(key=lambda row: row.rank_score, reverse=True)
    ranked_rows = [
        SweepRow(rank=index + 1, **{field: getattr(row, field) for field in row.__dataclass_fields__ if field != "rank"})
        for index, row in enumerate(rows)
    ]

    output_path = Path(args.output)
    write_csv(ranked_rows, output_path)

    best_output = model_path(args.best_model)
    if best_model is not None:
        best_model.save(best_output)

    print(f"saved_table={output_path}")
    print(f"saved_best_model={best_output}")
    print_rows(ranked_rows)


if __name__ == "__main__":
    main()
