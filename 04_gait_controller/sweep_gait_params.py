"""Sweep handcrafted Ant-v4 gait parameters and save a ranked CSV table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sinusoidal_gait_controller import default_trot_patterns, target_at_time


@dataclass(frozen=True)
class TrialResult:
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    x_displacement: float
    y_displacement: float
    forward_score: float


@dataclass(frozen=True)
class SweepResult:
    frequency: float
    action_sign: float
    action_scale: float
    knee_action_scale: float
    trials: int
    mean_return: float
    mean_length: float
    success_rate: float
    mean_x_displacement: float
    mean_y_displacement: float
    mean_forward_score: float
    std_forward_score: float
    rank_score: float


def parse_float_list(raw: str, name: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))

    if not values:
        raise SystemExit(f"{name} must contain at least one value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep handcrafted Ant-v4 gait parameters.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument("--steps", type=int, default=300, help="Maximum steps per trial.")
    parser.add_argument(
        "--frequencies",
        default="0.4,0.6,0.8,1.0",
        help="Comma-separated gait frequencies in Hz.",
    )
    parser.add_argument(
        "--action-signs",
        default="1,-1",
        help="Comma-separated signs applied to all gait amplitudes.",
    )
    parser.add_argument(
        "--action-scales",
        default="0.10,0.20,0.30,0.40,0.50",
        help="Comma-separated hip action amplitudes.",
    )
    parser.add_argument(
        "--knee-action-scales",
        default="0.10,0.20,0.30,0.40,0.50",
        help="Comma-separated knee action amplitudes.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base environment seed.")
    parser.add_argument(
        "--trials-per-config",
        type=int,
        default=3,
        help="Seeds to evaluate for each parameter combination.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Rows to print after ranking.")
    parser.add_argument(
        "--output",
        default="results/tables/gait_param_sweep.csv",
        help="Output CSV path.",
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


def run_trial(
    env,
    frequency: float,
    action_sign: float,
    action_scale: float,
    knee_action_scale: float,
    steps: int,
    seed: int,
) -> TrialResult:
    patterns = default_trot_patterns(
        amplitude_rad=action_sign * action_scale,
        knee_amplitude_rad=action_sign * knee_action_scale,
    )

    if len(patterns) != env.action_space.shape[0]:
        raise SystemExit(
            f"Pattern has {len(patterns)} actions, but env action space has "
            f"{env.action_space.shape[0]} dimensions."
        )

    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)
    sim_dt = float(getattr(env.unwrapped, "dt", 1.0 / 30.0))

    observation, info = env.reset(seed=seed)
    del observation, info
    start_x, start_y = base_xy(env)

    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False

    for step in range(steps):
        t = step * sim_dt
        action = make_action(patterns, frequency, t, low, high)
        observation, reward, terminated, truncated, info = env.step(action)
        del observation, info

        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            break

    end_x, end_y = base_xy(env)
    x_displacement = end_x - start_x
    y_displacement = end_y - start_y
    forward_score = x_displacement - 0.2 * abs(y_displacement)

    return TrialResult(
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=terminated,
        truncated=truncated,
        x_displacement=x_displacement,
        y_displacement=y_displacement,
        forward_score=forward_score,
    )


def summarize_trials(
    frequency: float,
    action_sign: float,
    action_scale: float,
    knee_action_scale: float,
    trials: list[TrialResult],
) -> SweepResult:
    forward_scores = [trial.forward_score for trial in trials]
    mean_length = float(np.mean([trial.episode_length for trial in trials]))
    mean_forward_score = float(np.mean(forward_scores))
    success_count = sum(not trial.terminated and not trial.truncated for trial in trials)
    return SweepResult(
        frequency=frequency,
        action_sign=action_sign,
        action_scale=action_scale,
        knee_action_scale=knee_action_scale,
        trials=len(trials),
        mean_return=float(np.mean([trial.episode_return for trial in trials])),
        mean_length=mean_length,
        success_rate=success_count / len(trials),
        mean_x_displacement=float(np.mean([trial.x_displacement for trial in trials])),
        mean_y_displacement=float(np.mean([trial.y_displacement for trial in trials])),
        mean_forward_score=mean_forward_score,
        std_forward_score=float(np.std(forward_scores)),
        rank_score=mean_forward_score + 0.005 * mean_length + 0.5 * success_count / len(trials),
    )


def write_csv(results: list[SweepResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "frequency",
        "action_sign",
        "action_scale",
        "knee_action_scale",
        "trials",
        "mean_return",
        "mean_length",
        "success_rate",
        "mean_x_displacement",
        "mean_y_displacement",
        "mean_forward_score",
        "std_forward_score",
        "rank_score",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "frequency": f"{result.frequency:.4f}",
                    "action_sign": f"{result.action_sign:.1f}",
                    "action_scale": f"{result.action_scale:.4f}",
                    "knee_action_scale": f"{result.knee_action_scale:.4f}",
                    "trials": result.trials,
                    "mean_return": f"{result.mean_return:.6f}",
                    "mean_length": f"{result.mean_length:.2f}",
                    "success_rate": f"{result.success_rate:.4f}",
                    "mean_x_displacement": f"{result.mean_x_displacement:.6f}",
                    "mean_y_displacement": f"{result.mean_y_displacement:.6f}",
                    "mean_forward_score": f"{result.mean_forward_score:.6f}",
                    "std_forward_score": f"{result.std_forward_score:.6f}",
                    "rank_score": f"{result.rank_score:.6f}",
                }
            )


def print_top(results: list[SweepResult], top_k: int) -> None:
    print("rank freq sign hip_amp knee_amp trials return length success x_disp y_disp fwd_score rank_score")
    for rank, result in enumerate(results[:top_k], start=1):
        print(
            f"{rank:>4} "
            f"{result.frequency:>4.2f} "
            f"{result.action_sign:>4.0f} "
            f"{result.action_scale:>7.2f} "
            f"{result.knee_action_scale:>8.2f} "
            f"{result.trials:>6} "
            f"{result.mean_return:>7.2f} "
            f"{result.mean_length:>6.1f} "
            f"{result.success_rate:>7.2f} "
            f"{result.mean_x_displacement:>7.3f} "
            f"{result.mean_y_displacement:>7.3f} "
            f"{result.mean_forward_score:>7.3f} "
            f"{result.rank_score:>10.3f}"
        )


def main() -> None:
    args = parse_args()

    if args.steps <= 0:
        raise SystemExit("steps must be positive.")
    if args.trials_per_config <= 0:
        raise SystemExit("trials-per-config must be positive.")
    if args.top_k <= 0:
        raise SystemExit("top-k must be positive.")

    frequencies = parse_float_list(args.frequencies, "frequencies")
    action_signs = parse_float_list(args.action_signs, "action-signs")
    action_scales = parse_float_list(args.action_scales, "action-scales")
    knee_action_scales = parse_float_list(args.knee_action_scales, "knee-action-scales")

    if any(value < 0 for value in frequencies):
        raise SystemExit("frequencies must be non-negative.")
    if any(value == 0 for value in action_signs):
        raise SystemExit("action-signs must be non-zero.")
    if any(value < 0 for value in action_scales + knee_action_scales):
        raise SystemExit("action amplitudes must be non-negative.")

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    env = gym.make(args.env_id)
    results: list[SweepResult] = []

    try:
        config_index = 0
        for frequency in frequencies:
            for action_sign in action_signs:
                for action_scale in action_scales:
                    for knee_action_scale in knee_action_scales:
                        trials = []
                        for trial in range(args.trials_per_config):
                            seed = args.seed + trial
                            trials.append(
                                run_trial(
                                    env=env,
                                    frequency=frequency,
                                    action_sign=action_sign,
                                    action_scale=action_scale,
                                    knee_action_scale=knee_action_scale,
                                    steps=args.steps,
                                    seed=seed,
                                )
                            )
                        results.append(
                            summarize_trials(
                                frequency=frequency,
                                action_sign=action_sign,
                                action_scale=action_scale,
                                knee_action_scale=knee_action_scale,
                                trials=trials,
                            )
                        )
                        config_index += 1
    finally:
        env.close()

    results.sort(key=lambda result: result.rank_score, reverse=True)
    output_path = Path(args.output)
    write_csv(results, output_path)

    print(f"saved_table={output_path}")
    print(f"trials={len(results)}")
    print_top(results, min(args.top_k, len(results)))


if __name__ == "__main__":
    main()
