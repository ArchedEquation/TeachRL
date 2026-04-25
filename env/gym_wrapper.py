"""gym_wrapper.py — Gymnasium wrapper for SB3/PPO training on TeachRL v2.
Clean 30-action space: 10 concepts × 3 difficulties.
Archetype guessing handled separately by the PPOAgent wrapper.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
from env.environment import TeachRLEnv, TASK_REGISTRY
from env.archetypes import CONCEPTS, DIFFICULTY_LEVELS, ALL_ARCHETYPES

N_CONCEPTS = len(CONCEPTS)           # 10
N_DIFFS    = len(DIFFICULTY_LEVELS)  # 3
N_ACTIONS  = N_CONCEPTS * N_DIFFS    # 30

CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}
DIFF_IDX    = {d: i for i, d in enumerate(DIFFICULTY_LEVELS)}

# Obs: 10 × 4 + 5 = 45
OBS_DIM = N_CONCEPTS * 4 + 5


def int_to_action(a: int) -> Tuple[str, str]:
    return CONCEPTS[a // N_DIFFS], DIFFICULTY_LEVELS[a % N_DIFFS]


def obs_to_vector(obs_dict: dict, max_steps: int) -> np.ndarray:
    vec = []
    for c in CONCEPTS:
        vec.append(obs_dict["concept_success_rates"].get(c, 0.0))
        vec.append(min(obs_dict["concept_attempt_counts"].get(c, 0) / 20.0, 1.0))
        vec.append(min(obs_dict["concept_streaks"].get(c, 0) / 5.0, 1.0))
        vec.append(obs_dict["prerequisite_readiness"].get(c, 1.0))
    vec.append(obs_dict["engagement"])
    vec.append(obs_dict["fatigue"])
    vec.append(1.0 - obs_dict["step_count"] / max(max_steps, 1))
    vec.append(obs_dict.get("hint_reliability", 0.7))
    vec.append(1.0 if obs_dict.get("last_correct", False) else 0.0)
    return np.array(vec, dtype=np.float32)


class TeachRLGymEnv(gym.Env):
    """Gymnasium wrapper — Discrete(30) actions, 45-dim Box observations."""
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, task_id: str = "blind_teaching", seed: int = 42,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.task_id     = task_id
        self._seed       = seed
        self.render_mode = render_mode
        self._max_steps  = TASK_REGISTRY[task_id]["max_steps"]
        self._env        = TeachRLEnv(task_id=task_id, seed=seed)

        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs = self._env.reset(seed=seed if seed is not None else self._seed)
        return obs_to_vector(obs.model_dump(), self._max_steps), {}

    def step(self, action: int):
        c, d = int_to_action(int(action))
        result = self._env.step({
            "concept":         c,
            "difficulty":      d,
            "hint_given":      False,
            "archetype_guess": None,
        })
        vec  = obs_to_vector(result.observation.model_dump(), self._max_steps)
        info = result.info
        info["task_score"] = self._env._task_score()
        return vec, float(result.reward), result.done, False, info

    def render(self):
        if self.render_mode in ("human", "ansi"):
            print(self._env.render())

    def close(self): pass