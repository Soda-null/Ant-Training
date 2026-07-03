"""Generate and track simple sinusoidal joint targets for an Ant-like robot."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path


JOINT_NAMES = [
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
]


@dataclass(frozen=True)
class JointPattern:
    name: str
    offset: float
    amplitude: float
    phase: float


@dataclass(frozen=True)
class GaitConfig:
    frequency: float
    kp: float
    kd: float
    inertia: float
    damping: float
    dt: float
    duration: float


@dataclass(frozen=True)
class JointTrace:
    name: str
    target: list[float]
    position: list[float]
    torque: list[float]


def default_trot_patterns(amplitude_rad: float, knee_amplitude_rad: float) -> list[JointPattern]:
    trot_a = 0.0
    trot_b = math.pi
    knee_lag = math.pi / 2.0
    return [
        JointPattern("front_left_hip", 0.0, amplitude_rad, trot_a),
        JointPattern("front_left_knee", 0.0, knee_amplitude_rad, trot_a + knee_lag),
        JointPattern("front_right_hip", 0.0, amplitude_rad, trot_b),
        JointPattern("front_right_knee", 0.0, knee_amplitude_rad, trot_b + knee_lag),
        JointPattern("rear_left_hip", 0.0, amplitude_rad, trot_b),
        JointPattern("rear_left_knee", 0.0, knee_amplitude_rad, trot_b + knee_lag),
        JointPattern("rear_right_hip", 0.0, amplitude_rad, trot_a),
        JointPattern("rear_right_knee", 0.0, knee_amplitude_rad, trot_a + knee_lag),
    ]


def target_at_time(pattern: JointPattern, frequency: float, t: float) -> float:
    return pattern.offset + pattern.amplitude * math.sin(
        2.0 * math.pi * frequency * t + pattern.phase
    )


def simulate_joint(pattern: JointPattern, config: GaitConfig) -> JointTrace:
    q = pattern.offset
    qdot = 0.0
    steps = int(config.duration / config.dt)

    target: list[float] = []
    position: list[float] = []
    torque: list[float] = []

    for step in range(steps + 1):
        t = step * config.dt
        q_target = target_at_time(pattern, config.frequency, t)
        tau = config.kp * (q_target - q) - config.kd * qdot

        target.append(q_target)
        position.append(q)
        torque.append(tau)

        qddot = (tau - config.damping * qdot) / config.inertia
        qdot = qdot + qddot * config.dt
        q = q + qdot * config.dt

    return JointTrace(name=pattern.name, target=target, position=position, torque=torque)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple sinusoidal gait target demo.")
    parser.add_argument("--frequency", type=float, default=0.5, help="Gait frequency in Hz.")
    parser.add_argument("--amplitude", type=float, default=25.0, help="Hip amplitude.")
    parser.add_argument("--knee-amplitude", type=float, default=35.0, help="Knee amplitude.")
    parser.add_argument("--kp", type=float, default=25.0, help="Proportional gain.")
    parser.add_argument("--kd", type=float, default=6.0, help="Derivative gain.")
    parser.add_argument("--inertia", type=float, default=1.0, help="Joint inertia.")
    parser.add_argument("--damping", type=float, default=0.1, help="Passive damping.")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation timestep.")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration.")
    parser.add_argument(
        "--output",
        default="results/plots/sinusoidal_gait_controller.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Interpret amplitudes as radians instead of degrees.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    return parser.parse_args()


def to_radians(value: float, radians: bool) -> float:
    return value if radians else math.radians(value)


def from_radians(value: float, radians: bool) -> float:
    return value if radians else math.degrees(value)


def plot_gait(
    time: list[float],
    traces: list[JointTrace],
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

    unit = "rad" if radians else "deg"
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, trace in zip(axes_flat, traces):
        target = [from_radians(q, radians) for q in trace.target]
        position = [from_radians(q, radians) for q in trace.position]
        ax.plot(time, target, linestyle="--", linewidth=1.4, label="target")
        ax.plot(time, position, linewidth=1.2, label="tracked")
        ax.set_title(trace.name)
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"angle ({unit})")
    axes_flat[0].legend(loc="upper right")
    fig.suptitle("Sinusoidal gait targets with independent PD tracking")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.frequency < 0:
        raise SystemExit("frequency must be non-negative.")
    if args.kp < 0 or args.kd < 0:
        raise SystemExit("PD gains must be non-negative.")
    if args.inertia <= 0 or args.dt <= 0 or args.duration <= 0:
        raise SystemExit("inertia, dt, and duration must be positive.")

    config = GaitConfig(
        frequency=args.frequency,
        kp=args.kp,
        kd=args.kd,
        inertia=args.inertia,
        damping=args.damping,
        dt=args.dt,
        duration=args.duration,
    )
    patterns = default_trot_patterns(
        amplitude_rad=to_radians(args.amplitude, args.radians),
        knee_amplitude_rad=to_radians(args.knee_amplitude, args.radians),
    )
    traces = [simulate_joint(pattern, config) for pattern in patterns]
    steps = int(args.duration / args.dt)
    time = [step * args.dt for step in range(steps + 1)]

    plot_gait(time, traces, Path(args.output), args.radians, args.show)

    all_errors = [
        target - position
        for trace in traces
        for target, position in zip(trace.target, trace.position)
    ]
    rms_error = math.sqrt(sum(error * error for error in all_errors) / len(all_errors))
    max_abs_torque = max(abs(tau) for trace in traces for tau in trace.torque)
    print(f"saved_plot={args.output}")
    print(f"joints={len(traces)}")
    print(f"rms_tracking_error={from_radians(rms_error, args.radians):.6f}")
    print(f"max_abs_torque={max_abs_torque:.4f}")


if __name__ == "__main__":
    main()

