"""Analytical inverse kinematics for a simple 2-link planar arm."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from two_link_fk import forward_kinematics


@dataclass(frozen=True)
class IKSolution:
    name: str
    theta1_rad: float
    theta2_rad: float
    fk_error: float


def solve_inverse_kinematics(
    x: float,
    y: float,
    link1: float,
    link2: float,
) -> list[IKSolution]:
    """Return elbow-up and elbow-down IK solutions when the target is reachable."""
    distance_sq = x * x + y * y
    max_reach = link1 + link2
    min_reach = abs(link1 - link2)
    distance = math.sqrt(distance_sq)

    if distance > max_reach or distance < min_reach:
        return []

    cos_theta2 = (distance_sq - link1 * link1 - link2 * link2) / (2 * link1 * link2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    theta2_options = [math.acos(cos_theta2), -math.acos(cos_theta2)]
    names = ["elbow_down", "elbow_up"]
    solutions: list[IKSolution] = []

    for name, theta2 in zip(names, theta2_options):
        theta1 = math.atan2(y, x) - math.atan2(
            link2 * math.sin(theta2),
            link1 + link2 * math.cos(theta2),
        )
        pose = forward_kinematics(theta1, theta2, link1, link2)
        error = math.hypot(pose.end_effector.x - x, pose.end_effector.y - y)
        solutions.append(
            IKSolution(
                name=name,
                theta1_rad=theta1,
                theta2_rad=theta2,
                fk_error=error,
            )
        )

    return solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2-link planar arm IK demo.")
    parser.add_argument("--x", type=float, default=1.0, help="Target x position.")
    parser.add_argument("--y", type=float, default=1.0, help="Target y position.")
    parser.add_argument("--l1", type=float, default=1.0, help="Link 1 length.")
    parser.add_argument("--l2", type=float, default=1.0, help="Link 2 length.")
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Print joint angles in radians instead of degrees.",
    )
    return parser.parse_args()


def format_angle(angle_rad: float, radians: bool) -> float:
    return angle_rad if radians else math.degrees(angle_rad)


def main() -> None:
    args = parse_args()

    if args.l1 <= 0 or args.l2 <= 0:
        raise SystemExit("Link lengths must be positive.")

    solutions = solve_inverse_kinematics(args.x, args.y, args.l1, args.l2)
    print(f"target=({args.x:.4f}, {args.y:.4f})")
    print(f"link_lengths=({args.l1:.3f}, {args.l2:.3f})")

    if not solutions:
        print("reachable=False")
        print("reason=target is outside the arm workspace")
        return

    print("reachable=True")
    angle_unit = "rad" if args.radians else "deg"
    for solution in solutions:
        theta1 = format_angle(solution.theta1_rad, args.radians)
        theta2 = format_angle(solution.theta2_rad, args.radians)
        print(
            f"{solution.name}: "
            f"theta1_{angle_unit}={theta1:.4f} "
            f"theta2_{angle_unit}={theta2:.4f} "
            f"fk_error={solution.fk_error:.6f}"
        )


if __name__ == "__main__":
    main()

