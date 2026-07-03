"""Create a compact summary figure for the current locomotion experiments."""

from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "plots" / "stage_03_results_summary.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pretty_policy(name: str) -> str:
    return {
        "shaped_residual_ppo": "Shaped residual PPO",
        "handcrafted_gait": "Handcrafted gait",
        "vanilla_ppo": "Vanilla PPO",
        "residual_ppo": "Residual PPO",
        "random": "Random",
    }.get(name, name)


def main() -> None:
    cache_dir = ROOT / "results" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_rows = read_rows(TABLES / "locomotion_baseline_comparison_best_500k_20seed_1000step.csv")
    domain_rows = read_rows(TABLES / "domain_randomization_summary.csv")
    domain_strong_rows = read_rows(TABLES / "domain_randomization_strong_summary.csv")
    push_rows = read_rows(TABLES / "push_recovery_small_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.patch.set_facecolor("#f7f4ef")

    baseline_rows = sorted(baseline_rows, key=lambda row: as_float(row, "mean_forward_score"))
    policies = [pretty_policy(row["policy"]) for row in baseline_rows]
    baseline_scores = [as_float(row, "mean_forward_score") for row in baseline_rows]
    axes[0].barh(policies, baseline_scores, color="#2f6f73")
    axes[0].set_title("Locomotion score")
    axes[0].set_xlabel("x displacement - 0.2 * abs(y)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.25)
    for index, value in enumerate(baseline_scores):
        axes[0].text(value, index, f" {value:.1f}", va="center", fontsize=8)

    domain_map = {
        "Nominal": domain_rows[0],
        "Moderate randomization": domain_rows[1],
        "Strong randomization": domain_strong_rows[1],
    }
    domain_labels = list(domain_map)
    domain_scores = [as_float(row, "mean_forward_score") for row in domain_map.values()]
    domain_success = [as_float(row, "success_rate") for row in domain_map.values()]
    x_positions = list(range(len(domain_labels)))
    axes[1].bar(x_positions, domain_scores, color="#526d9f", label="Forward score")
    axes[1].set_title("Domain randomization")
    axes[1].set_ylabel("Forward score")
    axes[1].set_xticks(x_positions, domain_labels, rotation=18, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    for index, value in enumerate(domain_scores):
        axes[1].text(index, value + 1.0, f"{value:.1f}", ha="center", fontsize=8)
    ax_success = axes[1].twinx()
    ax_success.plot(x_positions, domain_success, color="#b45f45", marker="o", linewidth=2, label="Success")
    ax_success.set_ylim(0, 1.08)
    ax_success.set_ylabel("Success rate")

    selected_push = ["no_push", "+x_5", "-x_10", "+x_25", "+x_50"]
    push_map = {row["condition"]: row for row in push_rows}
    recovery = [as_float(push_map[label], "recovery_rate") for label in selected_push]
    success = [as_float(push_map[label], "success_rate") for label in selected_push]
    width = 0.38
    x_positions = list(range(len(selected_push)))
    axes[2].bar([x - width / 2 for x in x_positions], success, width=width, color="#7a8f52", label="Success")
    axes[2].bar([x + width / 2 for x in x_positions], recovery, width=width, color="#b45f45", label="Recovery")
    axes[2].set_title("Push recovery boundary")
    axes[2].set_xticks(x_positions, selected_push)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Rate")
    axes[2].grid(axis="y", linestyle="--", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Robot Learning Mini-Lab: current results summary", fontsize=14)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={OUTPUT}")


if __name__ == "__main__":
    main()
