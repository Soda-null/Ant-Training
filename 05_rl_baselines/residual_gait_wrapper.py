"""Gymnasium wrapper for residual learning on top of a handcrafted Ant gait."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


GAIT_DIR = Path(__file__).resolve().parents[1] / "04_gait_controller"
if str(GAIT_DIR) not in sys.path:
    sys.path.append(str(GAIT_DIR))

from sinusoidal_gait_controller import default_trot_patterns, target_at_time  # noqa: E402


class ResidualGaitWrapper(gym.ActionWrapper):
    """Add learned residual actions to a fixed handcrafted gait."""

    def __init__(
        self,
        env: gym.Env,
        frequency: float = 1.0,
        action_sign: float = -1.0,
        action_scale: float = 0.20,
        knee_action_scale: float = 0.10,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__(env)

        if frequency < 0:
            raise ValueError("frequency must be non-negative.")
        if action_sign == 0:
            raise ValueError("action_sign must be non-zero.")
        if action_scale < 0 or knee_action_scale < 0 or residual_scale < 0:
            raise ValueError("action amplitudes must be non-negative.")

        self.frequency = frequency
        self.action_sign = action_sign
        self.action_scale = action_scale
        self.knee_action_scale = knee_action_scale
        self.residual_scale = residual_scale
        self.step_index = 0

        self.patterns = default_trot_patterns(
            amplitude_rad=action_sign * action_scale,
            knee_amplitude_rad=action_sign * knee_action_scale,
        )

        if len(self.patterns) != self.env.action_space.shape[0]:
            raise ValueError(
                f"Pattern has {len(self.patterns)} actions, but env action space has "
                f"{self.env.action_space.shape[0]} dimensions."
            )

        shape = self.env.action_space.shape
        self.action_space = spaces.Box(
            low=-residual_scale,
            high=residual_scale,
            shape=shape,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        self.step_index = 0
        return self.env.reset(**kwargs)

    def action(self, residual_action) -> np.ndarray:
        residual = np.asarray(residual_action, dtype=np.float32)
        residual = np.clip(residual, self.action_space.low, self.action_space.high)

        base = self.base_action()
        low = self.env.action_space.low.astype(np.float32)
        high = self.env.action_space.high.astype(np.float32)
        return np.clip(base + residual, low, high).astype(np.float32)

    def step(self, action):
        result = super().step(action)
        self.step_index += 1
        return result

    def base_action(self) -> np.ndarray:
        sim_dt = float(getattr(self.env.unwrapped, "dt", 1.0 / 30.0))
        t = self.step_index * sim_dt
        return np.array(
            [target_at_time(pattern, self.frequency, t) for pattern in self.patterns],
            dtype=np.float32,
        )


def make_residual_ant_env(
    env_id: str = "Ant-v4",
    render_mode: str | None = None,
    frequency: float = 1.0,
    action_sign: float = -1.0,
    action_scale: float = 0.20,
    knee_action_scale: float = 0.10,
    residual_scale: float = 0.05,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    return ResidualGaitWrapper(
        env,
        frequency=frequency,
        action_sign=action_sign,
        action_scale=action_scale,
        knee_action_scale=knee_action_scale,
        residual_scale=residual_scale,
    )
