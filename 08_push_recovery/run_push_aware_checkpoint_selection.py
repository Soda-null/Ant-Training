"""Evaluate push-aware checkpoints and rank them."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SCRIPT = ROOT / "08_push_recovery" / "evaluate_push_recovery.py"
SELECT_SCRIPT = ROOT / "08_push_recovery" / "select_push_aware_checkpoint.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run push-aware checkpoint selection.")
    parser.add_argument("--checkpoint-dir", default="results/logs/push_aware_checkpoint_selection")
    parser.add_argument("--baseline-model", default="results/logs/ant_shaped_residual_ppo_best_500k.zip")
    parser.add_argument("--final-model", default="results/logs/ant_push_aware_lr5e5_100k.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--push-forces", default="5,10,25,50")
    parser.add_argument("--directions", default="+x,-x,+y,-y")
    parser.add_argument("--output-dir", default="results/tables/push_aware_selection")
    return parser.parse_args()


def model_entries(args: argparse.Namespace) -> list[tuple[str, Path]]:
    entries = [("original_500k", ROOT / args.baseline_model)]
    for path in sorted((ROOT / args.checkpoint_dir).glob("*.zip")):
        entries.append((path.stem, path))
    final_model = ROOT / args.final_model
    if final_model.exists():
        entries.append(("final_100k", final_model))
    return entries


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, model_path in model_entries(args):
        if not model_path.exists():
            raise SystemExit(f"model not found: {model_path}")
        summary_output = output_dir / f"{label}_push_summary.csv"
        episodes_output = output_dir / f"{label}_push_episodes.csv"
        command = [
            sys.executable,
            str(EVALUATE_SCRIPT),
            "--model",
            str(model_path),
            "--episodes",
            str(args.episodes),
            "--max-steps",
            str(args.max_steps),
            "--push-forces",
            args.push_forces,
            "--directions",
            args.directions,
            "--summary-output",
            str(summary_output),
            "--episodes-output",
            str(episodes_output),
        ]
        print(f"evaluating {label}")
        subprocess.run(command, cwd=ROOT, check=True)

    subprocess.run(
        [
            sys.executable,
            str(SELECT_SCRIPT),
            "--input-dir",
            str(output_dir),
            "--output",
            str(output_dir / "push_aware_checkpoint_ranking.csv"),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
