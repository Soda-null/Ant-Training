"""Jacobian demo for a simple 2-link planar arm."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from two_link_fk import forward_kinematics


@dataclass(frozen=True)
class Matrix2x2:
    a11: float
    a12: float
    a21: float
    a22: float


@dataclass(frozen=True)
class Velocity2D:
    x_dot: float
    y_dot: float


def jacobian(theta1_rad: float, theta2_rad: float, link1: float, link2: float) -> Matrix2x2:
    """Return the planar 2-link arm Jacobian at the current joint angles."""
    total_angle = theta1_rad + theta2_rad
    return Matrix2x2(
        a11=-link1 * math.sin(theta1_rad) - link2 * math.sin(total_angle),
        a12=-link2 * math.sin(total_angle),
        a21=link1 * math.cos(theta1_rad) + link2 * math.cos(total_angle),
        a22=link2 * math.cos(total_angle),
    )


def end_effector_velocity(
    matrix: Matrix2x2,
    theta1_dot: float,
    theta2_dot: float,
) -> Velocity2D:
    """Map joint velocity to end-effector velocity."""
    return Velocity2D(
        x_dot=matrix.a11 * theta1_dot + matrix.a12 * theta2_dot,
        y_dot=matrix.a21 * theta1_dot + matrix.a22 * theta2_dot,
    )


def finite_difference_velocity(
    theta1_rad: float,
    theta2_rad: float,
    theta1_dot: float,
    theta2_dot: float,
    link1: float,
    link2: float,
    dt: float,
) -> Velocity2D:
    """Approximate end-effector velocity by stepping FK forward by dt."""
    pose_now = forward_kinematics(theta1_rad, theta2_rad, link1, link2)
    pose_next = forward_kinematics(
        theta1_rad + theta1_dot * dt,
        theta2_rad + theta2_dot * dt,
        link1,
        link2,
    )
    return Velocity2D(
        x_dot=(pose_next.end_effector.x - pose_now.end_effector.x) / dt,
        y_dot=(pose_next.end_effector.y - pose_now.end_effector.y) / dt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2-link planar arm Jacobian demo.")
    parser.add_argument("--theta1", type=float, default=30.0, help="Joint 1 angle.")
    parser.add_argument("--theta2", type=float, default=45.0, help="Joint 2 angle.")
    parser.add_argument("--theta1-dot", type=float, default=0.5, help="Joint 1 velocity.")
    parser.add_argument("--theta2-dot", type=float, default=0.2, help="Joint 2 velocity.")
    parser.add_argument("--l1", type=float, default=1.0, help="Link 1 length.")
    parser.add_argument("--l2", type=float, default=1.0, help="Link 2 length.")
    parser.add_argument("--dt", type=float, default=1e-5, help="Finite-difference step.")
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
    if args.dt <= 0:
        raise SystemExit("dt must be positive.")

    theta1 = args.theta1 if args.radians else math.radians(args.theta1)
    theta2 = args.theta2 if args.radians else math.radians(args.theta2)

    matrix = jacobian(theta1, theta2, args.l1, args.l2)
    velocity = end_effector_velocity(matrix, args.theta1_dot, args.theta2_dot)
    fd_velocity = finite_difference_velocity(
        theta1,
        theta2,
        args.theta1_dot,
        args.theta2_dot,
        args.l1,
        args.l2,
        args.dt,
    )
    check_error = math.hypot(
        velocity.x_dot - fd_velocity.x_dot,
        velocity.y_dot - fd_velocity.y_dot,
    )

    angle_unit = "rad" if args.radians else "deg"
    print(f"joint_angles_{angle_unit}=({args.theta1:.3f}, {args.theta2:.3f})")
    print(f"joint_velocity=({args.theta1_dot:.4f}, {args.theta2_dot:.4f})")
    print("jacobian=")
    print(f"  [{matrix.a11:.6f}, {matrix.a12:.6f}]")
    print(f"  [{matrix.a21:.6f}, {matrix.a22:.6f}]")
    print(f"end_effector_velocity=({velocity.x_dot:.6f}, {velocity.y_dot:.6f})")
    print(
        "finite_difference_velocity="
        f"({fd_velocity.x_dot:.6f}, {fd_velocity.y_dot:.6f})"
    )
    print(f"finite_difference_error={check_error:.8f}")


if __name__ == "__main__":
    main()

