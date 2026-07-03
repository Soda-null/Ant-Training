"""Rank push-aware checkpoint evaluation summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SMALL_PUSHES = {"+x_5", "-x_5", "+y_5", "-y_5", "+x_10", "-x_10", "+y_10", "-y_10"}
HARD_PUSHES = {"+x_25", "-x_25", "+y_25", "-y_25", "+x_50", "-x_50", "+y_50", "-y_50"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank push-aware checkpoint summaries.")
    parser.add_argument("--input-dir", default="results/tables/push_aware_selection")
    parser.add_argument(
        "--output",
        default="results/tables/push_aware_selection/push_aware_checkpoint_ranking.csv",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    ranked_rows: list[dict[str, float | str]] = []

    for path in sorted(input_dir.glob("*_push_summary.csv")):
        checkpoint = path.stem.removesuffix("_push_summary")
        rows = read_rows(path)
        no_push = [row for row in rows if row["condition"] == "no_push"]
        small = [row for row in rows if row["condition"] in SMALL_PUSHES]
        hard = [row for row in rows if row["condition"] in HARD_PUSHES]
        if not no_push or not small or not hard:
            print(f"skipping {checkpoint}: incomplete conditions")
            continue

        no_push_score = float(no_push[0]["mean_forward_score"])
        no_push_success = float(no_push[0]["success_rate"])
        small_recovery = mean(small, "recovery_rate")
        small_score = mean(small, "mean_forward_score")
        hard_recovery = mean(hard, "recovery_rate")
        hard_score = mean(hard, "mean_forward_score")
        push_score = 35.0 * small_recovery + 5.0 * hard_recovery
        push_score += 0.30 * no_push_score + 0.35 * small_score + 0.05 * hard_score
        push_score += 5.0 * no_push_success

        ranked_rows.append(
            {
                "checkpoint": checkpoint,
                "push_score": push_score,
                "no_push_score": no_push_score,
                "no_push_success": no_push_success,
                "small_recovery": small_recovery,
                "small_score": small_score,
                "hard_recovery": hard_recovery,
                "hard_score": hard_score,
            }
        )

    ranked_rows.sort(key=lambda row: float(row["push_score"]), reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "push_score",
        "no_push_score",
        "no_push_success",
        "small_recovery",
        "small_score",
        "hard_recovery",
        "hard_score",
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(
                {
                    "checkpoint": row["checkpoint"],
                    "push_score": f"{float(row['push_score']):.6f}",
                    "no_push_score": f"{float(row['no_push_score']):.6f}",
                    "no_push_success": f"{float(row['no_push_success']):.4f}",
                    "small_recovery": f"{float(row['small_recovery']):.4f}",
                    "small_score": f"{float(row['small_score']):.6f}",
                    "hard_recovery": f"{float(row['hard_recovery']):.4f}",
                    "hard_score": f"{float(row['hard_score']):.6f}",
                }
            )

    print(f"saved_ranking={output}")
    for row in ranked_rows[:5]:
        print(
            f"{row['checkpoint']}: push={float(row['push_score']):.2f}, "
            f"small_recovery={float(row['small_recovery']):.2f}, "
            f"no_push={float(row['no_push_score']):.2f}"
        )


if __name__ == "__main__":
    main()
