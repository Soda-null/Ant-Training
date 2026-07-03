"""Compare several PD gain settings on a single rotary joint."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

from pd_joint_control import (
    SimulationConfig,
    SimulationResult,
    from_radians,
    simulate_pd_joint,
    to_radians,
)


@dataclass(frozen=True)
class GainCase:
    kp: float
    kd: float


@dataclass(frozen=True)
class GainSummary:
    gain: GainCase
    final_error: float
    max_overshoot: float
    max_abs_torque: float
    settling_time: float | None


DEFAULT_GAINS = [
    GainCase(kp=10.0, kd=2.0),
    GainCase(kp=25.0, kd=2.0),
    GainCase(kp=25.0, kd=6.0),
    GainCase(kp=50.0, kd=6.0),
]


def parse_gain(text: str) -> GainCase:
    try:
        kp_text, kd_text = text.split(",", maxsplit=1)
        return GainCase(kp=float(kp_text), kd=float(kd_text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use --gain Kp,Kd, for example --gain 25,6") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare single-joint PD gains.")
    parser.add_argument(
        "--gain",
        type=parse_gain,
        action="append",
        default=None,
        help="Gain pair as Kp,Kd. Can be passed multiple times.",
    )
    parser.add_argument("--q0", type=float, default=0.0, help="Initial angle.")
    parser.add_argument("--qdot0", type=float, default=0.0, help="Initial angular velocity.")
    parser.add_argument("--target", type=float, default=60.0, help="Target angle.")
    parser.add_argument("--inertia", type=float, default=1.0, help="Joint inertia.")
    parser.add_argument("--damping", type=float, default=0.1, help="Passive damping.")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation timestep.")
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration.")
    parser.add_argument(
        "--settling-band",
        type=float,
        default=1.0,
        help="Error band for settling time, in degrees unless --radians is used.",
    )
    parser.add_argument(
        "--output",
        default="results/plots/compare_pd_gains.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Interpret q0, qdot0, target, and settling band as radians/rad-s.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    return parser.parse_args()


def compute_settling_time(
    result: SimulationResult,
    target_rad: float,
    band_rad: float,
) -> float | None:
    for index, _ in enumerate(result.time):
        tail = result.position[index:]
        if all(abs(q - target_rad) <= band_rad for q in tail):
            return result.time[index]
    return None


def summarize_result(
    gain: GainCase,
    result: SimulationResult,
    target_rad: float,
    band_rad: float,
) -> GainSummary:
    final_error = target_rad - result.position[-1]
    max_overshoot = max(0.0, max(result.position) - target_rad)
    max_abs_torque = max(abs(tau) for tau in result.torque)
    settling_time = compute_settling_time(result, target_rad, band_rad)
    return GainSummary(
        gain=gain,
        final_error=final_error,
        max_overshoot=max_overshoot,
        max_abs_torque=max_abs_torque,
        settling_time=settling_time,
    )


def plot_comparison(
    results: list[tuple[GainCase, SimulationResult]],
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

    unit = "rad" if radians else "deg"
    target = from_radians(target_rad, radians)

    fig, ax = plt.subplots(figsize=(8, 5))
    for gain, result in results:
        position = [from_radians(q, radians) for q in result.position]
        ax.plot(result.time, position, label=f"Kp={gain.kp:g}, Kd={gain.kd:g}")

    ax.axhline(target, color="black", linestyle="--", linewidth=1.2, label="target")
    ax.set_title("PD gain comparison")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"joint angle ({unit})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    gains = args.gain if args.gain is not None else DEFAULT_GAINS
    if any(gain.kp < 0 or gain.kd < 0 for gain in gains):
        raise SystemExit("PD gains must be non-negative.")
    if args.inertia <= 0 or args.dt <= 0 or args.duration <= 0:
        raise SystemExit("inertia, dt, and duration must be positive.")
    if args.settling_band <= 0:
        raise SystemExit("settling band must be positive.")

    q0 = to_radians(args.q0, args.radians)
    qdot0 = to_radians(args.qdot0, args.radians)
    target = to_radians(args.target, args.radians)
    settling_band = to_radians(args.settling_band, args.radians)

    runs: list[tuple[GainCase, SimulationResult]] = []
    summaries: list[GainSummary] = []
    for gain in gains:
        config = SimulationConfig(
            q0=q0,
            qdot0=qdot0,
            q_target=target,
            kp=gain.kp,
            kd=gain.kd,
            inertia=args.inertia,
            damping=args.damping,
            dt=args.dt,
            duration=args.duration,
        )
        result = simulate_pd_joint(config)
        runs.append((gain, result))
        summaries.append(summarize_result(gain, result, target, settling_band))

    plot_comparison(runs, target, Path(args.output), args.radians, args.show)

    print(f"saved_plot={args.output}")
    print("gain_summary:")
    for summary in summaries:
        settling = (
            "not_settled"
            if summary.settling_time is None
            else f"{summary.settling_time:.3f}s"
        )
        print(
            f"  Kp={summary.gain.kp:g} Kd={summary.gain.kd:g} "
            f"final_error={from_radians(summary.final_error, args.radians):.4f} "
            f"overshoot={from_radians(summary.max_overshoot, args.radians):.4f} "
            f"settling={settling} "
            f"max_abs_torque={summary.max_abs_torque:.4f}"
        )


if __name__ == "__main__":
    main()

