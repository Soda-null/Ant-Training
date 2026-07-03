"""Visualize a 2-link planar arm configuration."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from two_link_fk import Point, forward_kinematics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a 2-link planar arm.")
    parser.add_argument("--theta1", type=float, default=30.0, help="Joint 1 angle.")
    parser.add_argument("--theta2", type=float, default=45.0, help="Joint 2 angle.")
    parser.add_argument("--l1", type=float, default=1.0, help="Link 1 length.")
    parser.add_argument("--l2", type=float, default=1.0, help="Link 2 length.")
    parser.add_argument("--target-x", type=float, default=None, help="Optional target x.")
    parser.add_argument("--target-y", type=float, default=None, help="Optional target y.")
    parser.add_argument(
        "--output",
        default="results/plots/two_link_arm.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Interpret theta1 and theta2 as radians instead of degrees.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    return parser.parse_args()


def set_equal_workspace(ax, reach: float) -> None:
    margin = max(0.25, 0.15 * reach)
    limit = reach + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")


def draw_arm(ax, shoulder: Point, elbow: Point, end_effector: Point) -> None:
    xs = [shoulder.x, elbow.x, end_effector.x]
    ys = [shoulder.y, elbow.y, end_effector.y]
    ax.plot(xs, ys, "-o", linewidth=3, markersize=8, label="arm")
    ax.scatter([shoulder.x], [shoulder.y], s=80, label="shoulder")
    ax.scatter([elbow.x], [elbow.y], s=80, label="elbow")
    ax.scatter([end_effector.x], [end_effector.y], s=80, label="end effector")


def main() -> None:
    args = parse_args()

    if args.l1 <= 0 or args.l2 <= 0:
        raise SystemExit("Link lengths must be positive.")
    if (args.target_x is None) != (args.target_y is None):
        raise SystemExit("Provide both --target-x and --target-y, or neither.")

    cache_dir = Path("results/.cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    theta1 = args.theta1 if args.radians else math.radians(args.theta1)
    theta2 = args.theta2 if args.radians else math.radians(args.theta2)
    pose = forward_kinematics(theta1, theta2, args.l1, args.l2)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_arm(ax, pose.shoulder, pose.elbow, pose.end_effector)

    if args.target_x is not None and args.target_y is not None:
        ax.scatter(
            [args.target_x],
            [args.target_y],
            marker="x",
            s=100,
            linewidths=2,
            label="target",
        )

    angle_unit = "rad" if args.radians else "deg"
    ax.set_title(
        f"2-link arm: theta1={args.theta1:.1f}{angle_unit}, "
        f"theta2={args.theta2:.1f}{angle_unit}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    set_equal_workspace(ax, args.l1 + args.l2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"saved_plot={output_path}")
    print(f"end_effector=({pose.end_effector.x:.4f}, {pose.end_effector.y:.4f})")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
