"""Train residual PPO with a shaped forward-locomotion reward."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parents[1] / "05_rl_baselines"
if str(BASELINE_DIR) not in sys.path:
    sys.path.append(str(BASELINE_DIR))

from residual_gait_wrapper import make_residual_ant_env  # noqa: E402
from locomotion_reward_wrapper import LocomotionRewardWrapper  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shaped residual PPO on Ant-v4.")
    parser.add_argument("--env-id", default="Ant-v4", help="Gymnasium MuJoCo env id.")
    parser.add_argument("--total-timesteps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--action-sign", type=float, default=-1.0)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--knee-action-scale", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--forward-weight", type=float, default=1.0)
    parser.add_argument("--lateral-weight", type=float, default=0.5)
    parser.add_argument("--energy-weight", type=float, default=0.01)
    parser.add_argument(
        "--model-in",
        default=None,
        help="Optional existing PPO model to continue training.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Stable-Baselines3 verbosity level.",
    )
    parser.add_argument(
        "--output",
        default="results/logs/ant_shaped_residual_ppo_smoke",
        help="Output model path without or with .zip.",
    )
    return parser.parse_args()


def model_path(path: str) -> Path:
    output = Path(path)
    if output.suffix != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main() -> None:
    args = parse_args()

    if args.total_timesteps <= 0:
        raise SystemExit("total-timesteps must be positive.")
    if args.n_envs <= 0:
        raise SystemExit("n-envs must be positive.")
    if args.n_steps <= 0:
        raise SystemExit("n-steps must be positive.")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from exc

    def make_env():
        env = make_residual_ant_env(
            env_id=args.env_id,
            frequency=args.frequency,
            action_sign=args.action_sign,
            action_scale=args.action_scale,
            knee_action_scale=args.knee_action_scale,
            residual_scale=args.residual_scale,
        )
        return LocomotionRewardWrapper(
            env,
            forward_weight=args.forward_weight,
            lateral_weight=args.lateral_weight,
            energy_weight=args.energy_weight,
        )

    output = model_path(args.output)
    env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    try:
        if args.model_in:
            model_in = Path(args.model_in)
            if not model_in.exists():
                raise SystemExit(f"model-in not found: {model_in}")
            model = PPO.load(model_in, env=env, device="auto", verbose=args.verbose)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                seed=args.seed,
                verbose=args.verbose,
                device="auto",
                n_steps=args.n_steps,
            )
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=not args.model_in)
        model.save(output)
    finally:
        env.close()

    print(f"saved_model={output}")
    print(f"env_id={args.env_id}")
    print(f"total_timesteps={args.total_timesteps}")
    print(f"n_steps={args.n_steps}")
    print(f"residual_scale={args.residual_scale}")
    print(f"forward_weight={args.forward_weight}")
    print(f"lateral_weight={args.lateral_weight}")
    print(f"energy_weight={args.energy_weight}")
    print(f"model_in={args.model_in}")


if __name__ == "__main__":
    main()
