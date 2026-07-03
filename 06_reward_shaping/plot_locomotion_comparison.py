"""Plot locomotion baseline comparison metrics from a CSV table."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot locomotion baseline comparison.")
    parser.add_argument(
        "--input",
        default="results/tables/locomotion_baseline_comparison_best_100k_10seed_1000step.csv",
        help="Input comparison CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/plots/locomotion_baseline_comparison.png",
        help="Output plot path.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def pretty_name(name: str) -> str:
    return {
        "shaped_residual_ppo": "Shaped residual PPO",
        "handcrafted_gait": "Handcrafted gait",
        "residual_ppo": "Residual PPO",
        "vanilla_ppo": "Vanilla PPO",
        "random": "Random",
    }.get(name, name)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = read_rows(input_path)
    if not rows:
        raise SystemExit(f"no rows found in {input_path}")

    cache_dir = Path("results/.cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = [pretty_name(row["policy"]) for row in rows]
    x_disp = [float(row["mean_x_displacement"]) for row in rows]
    abs_y = [float(row["mean_abs_y_displacement"]) for row in rows]
    score = [float(row["mean_forward_score"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    metrics = [
        ("Forward displacement", x_disp, "#2f6f9f"),
        ("Lateral drift", abs_y, "#b35c44"),
        ("Forward score", score, "#3d7f4f"),
    ]

    for ax, (title, values, color) in zip(axes, metrics):
        ax.barh(policies, values, color=color)
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.invert_yaxis()
        for index, value in enumerate(values):
            ax.text(value, index, f" {value:.2f}", va="center", fontsize=8)

    fig.suptitle("Ant-v4 locomotion baseline comparison, 1000 steps, 10 seeds")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
