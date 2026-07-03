"""Rank domain-randomization checkpoint evaluation summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best evaluated DR checkpoint.")
    parser.add_argument(
        "--input-dir",
        default="results/tables/checkpoint_selection",
        help="Directory containing per-checkpoint summary CSV files.",
    )
    parser.add_argument(
        "--output",
        default="results/tables/checkpoint_selection/domain_randomization_checkpoint_ranking.csv",
    )
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, str]:
    with path.open() as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"empty summary: {path}")
    if len(rows) == 1:
        return rows[0]
    for row in rows:
        if row["condition"] == "randomized":
            return row
    return rows[0]


def group_summaries(input_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for path in sorted(input_dir.glob("*_summary.csv")):
        stem = path.stem.removesuffix("_summary")
        if stem.endswith("_nominal"):
            checkpoint = stem.removesuffix("_nominal")
            condition = "nominal"
        elif stem.endswith("_moderate"):
            checkpoint = stem.removesuffix("_moderate")
            condition = "moderate"
        elif stem.endswith("_strong"):
            checkpoint = stem.removesuffix("_strong")
            condition = "strong"
        else:
            continue
        grouped.setdefault(checkpoint, {})[condition] = read_summary(path)
    return grouped


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    grouped = group_summaries(input_dir)
    if not grouped:
        raise SystemExit(f"no checkpoint summaries found in {input_dir}")

    ranked_rows: list[dict[str, str | float]] = []
    for checkpoint, summaries in grouped.items():
        missing = {"nominal", "moderate", "strong"} - set(summaries)
        if missing:
            print(f"skipping {checkpoint}: missing {sorted(missing)}")
            continue

        nominal = summaries["nominal"]
        moderate = summaries["moderate"]
        strong = summaries["strong"]
        nominal_score = to_float(nominal, "mean_forward_score")
        moderate_score = to_float(moderate, "mean_forward_score")
        strong_score = to_float(strong, "mean_forward_score")
        nominal_success = to_float(nominal, "success_rate")
        moderate_success = to_float(moderate, "success_rate")
        strong_success = to_float(strong, "success_rate")
        average_success = (nominal_success + moderate_success + strong_success) / 3.0
        robust_score = 0.25 * nominal_score + 0.35 * moderate_score + 0.40 * strong_score
        robust_score += 10.0 * average_success

        ranked_rows.append(
            {
                "checkpoint": checkpoint,
                "nominal_score": nominal_score,
                "moderate_score": moderate_score,
                "strong_score": strong_score,
                "nominal_success": nominal_success,
                "moderate_success": moderate_success,
                "strong_success": strong_success,
                "average_success": average_success,
                "robust_score": robust_score,
            }
        )

    ranked_rows.sort(key=lambda row: float(row["robust_score"]), reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "robust_score",
        "nominal_score",
        "moderate_score",
        "strong_score",
        "nominal_success",
        "moderate_success",
        "strong_success",
        "average_success",
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(
                {
                    "checkpoint": row["checkpoint"],
                    "robust_score": f"{float(row['robust_score']):.6f}",
                    "nominal_score": f"{float(row['nominal_score']):.6f}",
                    "moderate_score": f"{float(row['moderate_score']):.6f}",
                    "strong_score": f"{float(row['strong_score']):.6f}",
                    "nominal_success": f"{float(row['nominal_success']):.4f}",
                    "moderate_success": f"{float(row['moderate_success']):.4f}",
                    "strong_success": f"{float(row['strong_success']):.4f}",
                    "average_success": f"{float(row['average_success']):.4f}",
                }
            )

    print(f"saved_ranking={output}")
    for row in ranked_rows[:5]:
        print(
            f"{row['checkpoint']}: robust={float(row['robust_score']):.2f}, "
            f"nominal={float(row['nominal_score']):.2f}, "
            f"moderate={float(row['moderate_score']):.2f}, "
            f"strong={float(row['strong_score']):.2f}"
        )


if __name__ == "__main__":
    main()
