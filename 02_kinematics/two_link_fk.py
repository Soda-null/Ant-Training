"""Forward kinematics for a simple 2-link planar arm."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class ArmPose:
    shoulder: Point
    elbow: Point
    end_effector: Point


def forward_kinematics(
    theta1_rad: float,
    theta2_rad: float,
    link1: float,
    link2: float,
) -> ArmPose:
    """Map joint angles to the elbow and end-effector positions."""
    shoulder = Point(0.0, 0.0)
    elbow = Point(link1 * math.cos(theta1_rad), link1 * math.sin(theta1_rad))
    end_effector = Point(
        elbow.x + link2 * math.cos(theta1_rad + theta2_rad),
        elbow.y + link2 * math.sin(theta1_rad + theta2_rad),
    )
    return ArmPose(shoulder=shoulder, elbow=elbow, end_effector=end_effector)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2-link planar arm FK demo.")
    parser.add_argument("--theta1", type=float, default=30.0, help="Joint 1 angle.")
    parser.add_argument("--theta2", type=float, default=45.0, help="Joint 2 angle.")
    parser.add_argument("--l1", type=float, default=1.0, help="Link 1 length.")
    parser.add_argument("--l2", type=float, default=1.0, help="Link 2 length.")
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Interpret theta1 and theta2 as radians instead of degrees.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.l1 <= 0 or args.l2 <= 0:
        raise SystemExit("Link lengths must be positive.")

    theta1 = args.theta1 if args.radians else math.radians(args.theta1)
    theta2 = args.theta2 if args.radians else math.radians(args.theta2)
    pose = forward_kinematics(theta1, theta2, args.l1, args.l2)

    angle_unit = "rad" if args.radians else "deg"
    print(f"joint_angles_{angle_unit}=({args.theta1:.3f}, {args.theta2:.3f})")
    print(f"link_lengths=({args.l1:.3f}, {args.l2:.3f})")
    print(f"shoulder=({pose.shoulder.x:.4f}, {pose.shoulder.y:.4f})")
    print(f"elbow=({pose.elbow.x:.4f}, {pose.elbow.y:.4f})")
    print(f"end_effector=({pose.end_effector.x:.4f}, {pose.end_effector.y:.4f})")


if __name__ == "__main__":
    main()

