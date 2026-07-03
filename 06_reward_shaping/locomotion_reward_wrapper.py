"""Reward shaping wrapper for forward Ant locomotion."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def base_xy(env) -> tuple[float, float]:
    qpos = env.unwrapped.data.qpos
    return float(qpos[0]), float(qpos[1])


class LocomotionRewardWrapper(gym.Wrapper):
    """Reward forward velocity, penalize lateral drift and large actions."""

    def __init__(
        self,
        env: gym.Env,
        forward_weight: float = 1.0,
        lateral_weight: float = 0.5,
        energy_weight: float = 0.01,
    ) -> None:
        super().__init__(env)
        if forward_weight < 0 or lateral_weight < 0 or energy_weight < 0:
            raise ValueError("reward weights must be non-negative.")

        self.forward_weight = forward_weight
        self.lateral_weight = lateral_weight
        self.energy_weight = energy_weight

    def step(self, action):
        x_before, y_before = base_xy(self.env)
        observation, reward, terminated, truncated, info = self.env.step(action)
        x_after, y_after = base_xy(self.env)

        dt = float(getattr(self.env.unwrapped, "dt", 1.0))
        x_velocity = (x_after - x_before) / dt
        y_velocity = (y_after - y_before) / dt
        action_energy = float(np.sum(np.square(action)))

        forward_bonus = self.forward_weight * x_velocity
        lateral_penalty = self.lateral_weight * abs(y_velocity)
        energy_penalty = self.energy_weight * action_energy
        shaped_reward = float(reward + forward_bonus - lateral_penalty - energy_penalty)

        info = dict(info)
        info["original_reward"] = float(reward)
        info["forward_bonus"] = float(forward_bonus)
        info["lateral_penalty"] = float(lateral_penalty)
        info["energy_penalty"] = float(energy_penalty)
        info["shaped_reward"] = shaped_reward
        info["x_velocity"] = float(x_velocity)
        info["y_velocity"] = float(y_velocity)

        return observation, shaped_reward, terminated, truncated, info
