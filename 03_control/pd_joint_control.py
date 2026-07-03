"""Minimal PD control demo for a single rotary joint."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationConfig:
    q0: float
    qdot0: float
    q_target: float
    kp: float
    kd: float
    inertia: float
    damping: float
    dt: float
    duration: float


@dataclass(frozen=True)
class SimulationResult:
    time: list[float]
    position: list[float]
    velocity: list[float]
    torque: list[float]


def simulate_pd_joint(config: SimulationConfig) -> SimulationResult:
    q = config.q0
    qdot = config.qdot0
    steps = int(config.duration / config.dt)

    time: list[float] = []
    position: list[float] = []
    velocity: list[float] = []
    torque: list[float] = []

    for step in range(steps + 1):
        t = step * config.dt
        tau = config.kp * (config.q_target - q) - config.kd * qdot

        time.append(t)
        position.append(q)
        velocity.append(qdot)
        torque.append(tau)

        qddot = (tau - config.damping * qdot) / config.inertia
        qdot = qdot + qddot * config.dt
        q = q + qdot * config.dt

    return SimulationResult(time=time, position=position, velocity=velocity, torque=torque)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-joint PD control demo.")
    parser.add_argument("--q0", type=float, default=0.0, help="Initial angle.")
    parser.add_argument("--qdot0", type=float, default=0.0, help="Initial angular velocity.")
    parser.add_argument("--target", type=float, default=60.0, help="Target angle.")
    parser.add_argument("--kp", type=float, default=25.0, help="Proportional gain.")
    parser.add_argument("--kd", type=float, default=6.0, help="Derivative gain.")
    parser.add_argument("--inertia", type=float, default=1.0, help="Joint inertia.")
    parser.add_argument("--damping", type=float, default=0.1, help="Passive damping.")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation timestep.")
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration.")
    parser.add_argument(
        "--output",
        default="results/plots/pd_joint_control.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Interpret q0, qdot0, and target as radians/rad-s instead of degrees/deg-s.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    return parser.parse_args()


def to_radians(value: float, radians: bool) -> float:
    return value if radians else math.radians(value)


def from_radians(value: float, radians: bool) -> float:
    return value if radians else math.degrees(value)


def plot_result(
    result: SimulationResult,
    target_rad: float,
    output_path: Path,
    radians: bool,
    show: bool,
) -> None:
    cache_dir = Path("results/.cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    position = [from_radians(q, radians) for q in result.position]
    velocity = [from_radians(qdot, radians) for qdot in result.velocity]
    target = from_radians(target_rad, radians)
    unit = "rad" if radians else "deg"
    velocity_unit = "rad/s" if radians else "deg/s"

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(result.time, position, label="q")
    axes[0].axhline(target, color="tab:red", linestyle="--", label="target")
    axes[0].set_ylabel(f"angle ({unit})")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(result.time, velocity, color="tab:orange")
    axes[1].set_ylabel(f"velocity ({velocity_unit})")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(result.time, result.torque, color="tab:green")
    axes[2].set_ylabel("torque")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Single-joint PD control")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.kp < 0 or args.kd < 0:
        raise SystemExit("PD gains must be non-negative.")
    if args.inertia <= 0 or args.dt <= 0 or args.duration <= 0:
        raise SystemExit("inertia, dt, and duration must be positive.")

    config = SimulationConfig(
        q0=to_radians(args.q0, args.radians),
        qdot0=to_radians(args.qdot0, args.radians),
        q_target=to_radians(args.target, args.radians),
        kp=args.kp,
        kd=args.kd,
        inertia=args.inertia,
        damping=args.damping,
        dt=args.dt,
        duration=args.duration,
    )
    result = simulate_pd_joint(config)
    plot_result(result, config.q_target, Path(args.output), args.radians, args.show)

    final_error = config.q_target - result.position[-1]
    max_position = max(result.position)
    overshoot = max(0.0, max_position - config.q_target)
    print(f"saved_plot={args.output}")
    print(f"final_angle={from_radians(result.position[-1], args.radians):.4f}")
    print(f"final_error={from_radians(final_error, args.radians):.6f}")
    print(f"max_overshoot={from_radians(overshoot, args.radians):.6f}")
    print(f"max_abs_torque={max(abs(tau) for tau in result.torque):.4f}")


if __name__ == "__main__":
    main()

