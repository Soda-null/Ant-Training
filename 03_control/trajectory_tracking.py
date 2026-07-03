"""Track a sinusoidal joint trajectory with a PD controller."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

from pd_joint_control import from_radians, to_radians


@dataclass(frozen=True)
class TrackingConfig:
    q0: float
    qdot0: float
    offset: float
    amplitude: float
    frequency: float
    kp: float
    kd: float
    inertia: float
    damping: float
    dt: float
    duration: float


@dataclass(frozen=True)
class TrackingResult:
    time: list[float]
    target: list[float]
    position: list[float]
    velocity: list[float]
    torque: list[float]
    error: list[float]


def target_at_time(t: float, offset: float, amplitude: float, frequency: float) -> float:
    return offset + amplitude * math.sin(2.0 * math.pi * frequency * t)


def simulate_tracking(config: TrackingConfig) -> TrackingResult:
    q = config.q0
    qdot = config.qdot0
    steps = int(config.duration / config.dt)

    time: list[float] = []
    target: list[float] = []
    position: list[float] = []
    velocity: list[float] = []
    torque: list[float] = []
    error: list[float] = []

    for step in range(steps + 1):
        t = step * config.dt
        q_target = target_at_time(t, config.offset, config.amplitude, config.frequency)
        q_error = q_target - q
        tau = config.kp * q_error - config.kd * qdot

        time.append(t)
        target.append(q_target)
        position.append(q)
        velocity.append(qdot)
        torque.append(tau)
        error.append(q_error)

        qddot = (tau - config.damping * qdot) / config.inertia
        qdot = qdot + qddot * config.dt
        q = q + qdot * config.dt

    return TrackingResult(
        time=time,
        target=target,
        position=position,
        velocity=velocity,
        torque=torque,
        error=error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PD tracking of a sinusoidal joint target.")
    parser.add_argument("--q0", type=float, default=0.0, help="Initial angle.")
    parser.add_argument("--qdot0", type=float, default=0.0, help="Initial angular velocity.")
    parser.add_argument("--offset", type=float, default=0.0, help="Target trajectory offset.")
    parser.add_argument("--amplitude", type=float, default=30.0, help="Target amplitude.")
    parser.add_argument("--frequency", type=float, default=0.5, help="Target frequency in Hz.")
    parser.add_argument("--kp", type=float, default=25.0, help="Proportional gain.")
    parser.add_argument("--kd", type=float, default=6.0, help="Derivative gain.")
    parser.add_argument("--inertia", type=float, default=1.0, help="Joint inertia.")
    parser.add_argument("--damping", type=float, default=0.1, help="Passive damping.")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation timestep.")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration.")
    parser.add_argument(
        "--output",
        default="results/plots/trajectory_tracking.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Interpret angle arguments as radians/rad-s instead of degrees/deg-s.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    return parser.parse_args()


def plot_tracking(
    result: TrackingResult,
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

    target = [from_radians(q, radians) for q in result.target]
    position = [from_radians(q, radians) for q in result.position]
    error = [from_radians(e, radians) for e in result.error]
    unit = "rad" if radians else "deg"

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(result.time, target, linestyle="--", label="target")
    axes[0].plot(result.time, position, label="q")
    axes[0].set_ylabel(f"angle ({unit})")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(result.time, error, color="tab:red")
    axes[1].set_ylabel(f"error ({unit})")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(result.time, result.torque, color="tab:green")
    axes[2].set_ylabel("torque")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("PD trajectory tracking")
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
    if args.frequency < 0:
        raise SystemExit("frequency must be non-negative.")

    config = TrackingConfig(
        q0=to_radians(args.q0, args.radians),
        qdot0=to_radians(args.qdot0, args.radians),
        offset=to_radians(args.offset, args.radians),
        amplitude=to_radians(args.amplitude, args.radians),
        frequency=args.frequency,
        kp=args.kp,
        kd=args.kd,
        inertia=args.inertia,
        damping=args.damping,
        dt=args.dt,
        duration=args.duration,
    )
    result = simulate_tracking(config)
    plot_tracking(result, Path(args.output), args.radians, args.show)

    rms_error = math.sqrt(sum(e * e for e in result.error) / len(result.error))
    max_abs_error = max(abs(e) for e in result.error)
    print(f"saved_plot={args.output}")
    print(f"rms_error={from_radians(rms_error, args.radians):.6f}")
    print(f"max_abs_error={from_radians(max_abs_error, args.radians):.6f}")
    print(f"max_abs_torque={max(abs(tau) for tau in result.torque):.4f}")


if __name__ == "__main__":
    main()

