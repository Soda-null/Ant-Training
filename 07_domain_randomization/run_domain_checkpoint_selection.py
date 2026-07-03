"""Evaluate multiple domain-randomization checkpoints and rank them."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SCRIPT = ROOT / "07_domain_randomization" / "evaluate_domain_randomization.py"
SELECT_SCRIPT = ROOT / "07_domain_randomization" / "select_domain_randomization_checkpoint.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run domain-randomization checkpoint selection.")
    parser.add_argument(
        "--checkpoint-dir",
        default="results/logs/dr_checkpoint_selection_narrow_lr1e4",
    )
    parser.add_argument(
        "--baseline-model",
        default="results/logs/ant_shaped_residual_ppo_best_500k.zip",
    )
    parser.add_argument(
        "--final-model",
        default="results/logs/ant_domain_randomized_checkpoint_selection_narrow_lr1e4_100k.zip",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        default="results/tables/checkpoint_selection",
    )
    return parser.parse_args()


def model_entries(args: argparse.Namespace) -> list[tuple[str, Path]]:
    entries = [("original_500k", ROOT / args.baseline_model)]
    checkpoint_dir = ROOT / args.checkpoint_dir
    for path in sorted(checkpoint_dir.glob("*.zip")):
        entries.append((path.stem, path))
    final_model = ROOT / args.final_model
    if final_model.exists():
        entries.append(("final_100k", final_model))
    return entries


def condition_args(condition: str) -> list[str]:
    if condition == "nominal":
        return [
            "--mass-range",
            "1.0,1.0",
            "--friction-range",
            "1.0,1.0",
            "--damping-range",
            "1.0,1.0",
        ]
    if condition == "moderate":
        return []
    if condition == "strong":
        return [
            "--mass-range",
            "0.6,1.4",
            "--friction-range",
            "0.5,1.5",
            "--damping-range",
            "0.5,1.5",
        ]
    raise ValueError(condition)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = model_entries(args)
    if not entries:
        raise SystemExit("no models found for checkpoint selection")

    for label, model_path in entries:
        if not model_path.exists():
            raise SystemExit(f"model not found: {model_path}")
        for condition in ("nominal", "moderate", "strong"):
            summary_output = output_dir / f"{label}_{condition}_summary.csv"
            episodes_output = output_dir / f"{label}_{condition}_episodes.csv"
            command = [
                sys.executable,
                str(EVALUATE_SCRIPT),
                "--model",
                str(model_path),
                "--episodes",
                str(args.episodes),
                "--max-steps",
                str(args.max_steps),
                "--summary-output",
                str(summary_output),
                "--episodes-output",
                str(episodes_output),
                *condition_args(condition),
            ]
            print(f"evaluating {label} / {condition}")
            subprocess.run(command, cwd=ROOT, check=True)

    ranking_output = output_dir / "domain_randomization_checkpoint_ranking.csv"
    subprocess.run(
        [
            sys.executable,
            str(SELECT_SCRIPT),
            "--input-dir",
            str(output_dir),
            "--output",
            str(ranking_output),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
