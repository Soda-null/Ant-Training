"""Plot the formal benchmark comparing performance and robustness policies."""

from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables" / "formal_benchmark"
OUTPUT = ROOT / "results" / "plots" / "formal_policy_benchmark.png"


def read_randomized_score(path: Path) -> tuple[float, float]:
    with path.open() as file:
        rows = list(csv.DictReader(file))
    row = rows[-1]
    return float(row["mean_forward_score"]), float(row["success_rate"])


def read_push_average(path: Path, labels: list[str]) -> tuple[float, float]:
    with path.open() as file:
        rows = list(csv.DictReader(file))
    selected = [row for row in rows if row["condition"] in labels]
    recovery = sum(float(row["recovery_rate"]) for row in selected) / len(selected)
    score = sum(float(row["mean_forward_score"]) for row in selected) / len(selected)
    return recovery, score


def main() -> None:
    cache_dir = ROOT / "results" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = ["Performance", "Robustness"]
    domain_conditions = ["nominal", "moderate", "strong"]
    domain_scores = {
        "Performance": [
            read_randomized_score(TABLES / f"performance_domain_{condition}_summary.csv")[0]
            for condition in domain_conditions
        ],
        "Robustness": [
            read_randomized_score(TABLES / f"robust_domain_{condition}_summary.csv")[0]
            for condition in domain_conditions
        ],
    }
    domain_success = {
        "Performance": [
            read_randomized_score(TABLES / f"performance_domain_{condition}_summary.csv")[1]
            for condition in domain_conditions
        ],
        "Robustness": [
            read_randomized_score(TABLES / f"robust_domain_{condition}_summary.csv")[1]
            for condition in domain_conditions
        ],
    }

    small_push = ["+x_5", "-x_5", "+y_5", "-y_5", "+x_10", "-x_10", "+y_10", "-y_10"]
    hard_push = ["+x_25", "-x_25", "+y_25", "-y_25", "+x_50", "-x_50", "+y_50", "-y_50"]
    push_metrics = {
        "Performance": [
            read_push_average(TABLES / "performance_push_summary.csv", small_push)[0],
            read_push_average(TABLES / "performance_push_summary.csv", hard_push)[0],
        ],
        "Robustness": [
            read_push_average(TABLES / "robust_push_summary.csv", small_push)[0],
            read_push_average(TABLES / "robust_push_summary.csv", hard_push)[0],
        ],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.patch.set_facecolor("#f7f4ef")

    x_positions = range(len(domain_conditions))
    width = 0.36
    axes[0].bar([x - width / 2 for x in x_positions], domain_scores["Performance"], width, label="Performance", color="#2f6f73")
    axes[0].bar([x + width / 2 for x in x_positions], domain_scores["Robustness"], width, label="Robustness", color="#526d9f")
    axes[0].set_title("Domain forward score")
    axes[0].set_xticks(list(x_positions), ["Nominal", "Moderate", "Strong"])
    axes[0].set_ylabel("Forward score")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar([x - width / 2 for x in x_positions], domain_success["Performance"], width, label="Performance", color="#2f6f73")
    axes[1].bar([x + width / 2 for x in x_positions], domain_success["Robustness"], width, label="Robustness", color="#526d9f")
    axes[1].set_title("Domain success rate")
    axes[1].set_xticks(list(x_positions), ["Nominal", "Moderate", "Strong"])
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    push_labels = ["5/10N pushes", "25/50N pushes"]
    x_positions = range(len(push_labels))
    axes[2].bar([x - width / 2 for x in x_positions], push_metrics["Performance"], width, label="Performance", color="#2f6f73")
    axes[2].bar([x + width / 2 for x in x_positions], push_metrics["Robustness"], width, label="Robustness", color="#526d9f")
    axes[2].set_title("Push recovery rate")
    axes[2].set_xticks(list(x_positions), push_labels)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle("Formal benchmark: performance policy vs robustness policy", fontsize=14)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={OUTPUT}")


if __name__ == "__main__":
    main()
