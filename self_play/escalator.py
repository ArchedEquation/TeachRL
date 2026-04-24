"""
self_play/escalator.py — Self-Play Difficulty Escalator.
Theme 4 core: tracks agent performance per archetype,
auto-generates harder variants when agent consistently scores > 0.70.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from env.archetypes import ArchetypeID, ALL_ARCHETYPES


@dataclass
class EpisodeResult:
    archetype_id:      ArchetypeID
    task_score:        float
    mastery_achieved:  Dict[str, float]
    engagement_final:  float
    fatigue_final:     float
    steps_used:        int
    inference_correct: bool


@dataclass
class ArchetypeVariant:
    base_archetype:   ArchetypeID
    generation:       int
    difficulty_delta: Dict[str, float]
    label:            str


class SelfPlayEscalator:
    """
    Monitors agent scores per archetype.
    Escalates difficulty when agent scores >= THRESHOLD across MIN_EPS consecutive episodes.
    Implements recursive self-improvement: environment grows with the agent.
    """
    THRESHOLD = 0.70
    MIN_EPS   = 2

    def __init__(self):
        self._history:     Dict[ArchetypeID, List[EpisodeResult]] = {a: [] for a in ALL_ARCHETYPES}
        self._generations: Dict[ArchetypeID, int]                 = {a: 0 for a in ALL_ARCHETYPES}
        self._variants:    Dict[ArchetypeID, ArchetypeVariant]    = {}
        self._total = 0

    def record(self, result: EpisodeResult):
        aid = result.archetype_id
        self._history[aid].append(result)
        self._total += 1
        recent = self._history[aid][-self.MIN_EPS:]
        if len(recent) >= self.MIN_EPS:
            if np.mean([r.task_score for r in recent]) >= self.THRESHOLD:
                self._escalate(aid)

    def _escalate(self, aid: ArchetypeID):
        self._generations[aid] += 1
        g = self._generations[aid]
        MAP = {
            ArchetypeID.OVERCONFIDENT: {"label": f"OverconfidentLearner Gen{g}", "delta": {"p_guess_boost": min(g*0.05, 0.25)}},
            ArchetypeID.ANXIOUS:       {"label": f"AnxiousPerfectionist Gen{g}", "delta": {"hard_slip_boost": min(g*0.05, 0.20)}},
            ArchetypeID.ADHD:          {"label": f"ADHDSprinter Gen{g}",         "delta": {"boredom_onset": max(2-g, 1)}},
            ArchetypeID.SLOW_STEADY:   {"label": f"SlowSteadyBuilder Gen{g}",    "delta": {"p_learn_reduce": min(g*0.02, 0.06)}},
            ArchetypeID.GAMER:         {"label": f"StrategicGamer Gen{g}",        "delta": {"easy_guess_boost": min(g*0.05, 0.15)}},
            ArchetypeID.EMOTIONAL:     {"label": f"EmotionalLearner Gen{g}",     "delta": {"mood_cycle": max(10-g*2, 4)}},
            ArchetypeID.UNEVEN:        {"label": f"UnevenGenius Gen{g}",          "delta": {"n_blind_spots": min(3+g, 6)}},
            ArchetypeID.IMPOSTOR:      {"label": f"Impostor Gen{g}",              "delta": {"p_init_boost": min(g*0.04, 0.12)}},
        }
        cfg = MAP[aid]
        self._variants[aid] = ArchetypeVariant(aid, g, cfg["delta"], cfg["label"])

    def generation(self, aid: ArchetypeID) -> int:
        return self._generations[aid]

    def total_generation_sum(self) -> int:
        return sum(self._generations.values())

    def weakest_archetype(self) -> ArchetypeID:
        scores = {a: float(np.mean([r.task_score for r in h[-3:]])) if h else 0.0
                  for a, h in self._history.items()}
        return min(scores, key=scores.get)

    def summary(self) -> Dict:
        return {
            "total_episodes": self._total,
            "generations":    {a.value: g for a, g in self._generations.items()},
            "avg_scores":     {a.value: float(np.mean([r.task_score for r in h[-5:]])) if h else 0.0
                               for a, h in self._history.items()},
        }
