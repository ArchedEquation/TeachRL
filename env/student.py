"""
env/student.py — Archetype-driven BKT Student Simulator for TeachRL v2.
Archetype is hidden. Agent observes only success rates, streaks, engagement, fatigue.
answer_question() returns (correct: bool, mastery_delta: float).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from env.archetypes import (
    BaseArchetype, ArchetypeID, BKTParams,
    CONCEPTS, DIFFICULTY_LEVELS, PREREQUISITES,
    make_archetype, sample_archetype,
)

__all__ = ["CONCEPTS", "DIFFICULTY_LEVELS", "PREREQUISITES", "StudentState", "StudentSimulator"]


@dataclass
class StudentState:
    mastery:    Dict[str, float]           = field(default_factory=dict)
    archetype:  Optional[ArchetypeID]      = None
    attempts:   Dict[str, Dict[str, int]]  = field(default_factory=dict)
    correct:    Dict[str, Dict[str, int]]  = field(default_factory=dict)
    engagement: float = 1.0
    fatigue:    float = 0.0
    step_count: int   = 0
    streak:     Dict[str, int]             = field(default_factory=dict)


class StudentSimulator:
    """
    Simulates a student using archetype-specific BKT dynamics.
    Hidden: true mastery + archetype identity.
    Observable: success rates, attempt counts, streaks, engagement, fatigue.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._arch: Optional[BaseArchetype] = None
        self.state = self._fresh(ArchetypeID.OVERCONFIDENT)

    def reset(self, seed: Optional[int] = None,
              archetype_id: Optional[ArchetypeID] = None) -> StudentState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        chosen = archetype_id or sample_archetype(self.rng)
        self._arch = make_archetype(chosen, self.rng)
        self.state = self._fresh(chosen)
        return self.state

    def answer_question(self, concept: str, difficulty: str) -> Tuple[bool, float]:
        """Returns (correct: bool, mastery_delta: float)."""
        assert concept in CONCEPTS, f"Unknown concept: {concept}"
        assert difficulty in DIFFICULTY_LEVELS, f"Unknown difficulty: {difficulty}"

        p = self._arch.get_params(concept, difficulty)
        ps = p.slip(difficulty)
        pg = p.guess(difficulty)
        m  = self.state.mastery[concept]

        pc      = m * (1 - ps) + (1 - m) * pg
        correct = bool(self.rng.random() < pc)

        post  = (m * (1 - ps)) / max(pc, 1e-9) if correct else (m * ps) / max(1 - pc, 1e-9)
        new_m = post + (1 - post) * p.p_learn
        new_m = new_m * (1 - p.p_forget)
        prev  = m
        self.state.mastery[concept] = float(np.clip(new_m, 0.0, 1.0))
        delta = self.state.mastery[concept] - prev

        self.state.attempts[concept][difficulty] += 1
        if correct:
            self.state.correct[concept][difficulty] += 1
            self.state.streak[concept] = self.state.streak.get(concept, 0) + 1
        else:
            self.state.streak[concept] = 0

        ed = self._arch.engagement_delta(
            concept, difficulty, correct,
            self.state.streak.get(concept, 0),
            self.state.mastery[concept],
        )
        self.state.engagement = float(np.clip(self.state.engagement + ed, 0.05, 1.0))
        self.state.fatigue    = float(np.clip(self.state.fatigue + self._arch.fatigue_delta(), 0.0, 1.0))
        self._arch.on_step(concept, difficulty, correct)
        self.state.step_count += 1
        return correct, delta

    def prerequisite_readiness(self, concept: str) -> float:
        prereqs = PREREQUISITES.get(concept, [])
        return 1.0 if not prereqs else float(np.mean([self.state.mastery[p] for p in prereqs]))

    def get_expert_hint(self) -> str:
        return self._arch.hint if self._arch else ""

    def _ideal_difficulty_score(self, concept: str, difficulty: str) -> float:
        m = self.state.mastery[concept]
        t = {"easy": 0.3, "medium": 0.55, "hard": 0.8}[difficulty]
        return float(np.exp(-4 * (m - t) ** 2))

    @property
    def archetype_id(self) -> Optional[ArchetypeID]:
        return self.state.archetype

    def _fresh(self, aid: ArchetypeID) -> StudentState:
        arch = make_archetype(aid, self.rng)
        mastery = {
            c: float(np.clip(self.rng.normal(arch.get_params(c, "medium").p_init, 0.04), 0.02, 0.95))
            for c in CONCEPTS
        }
        return StudentState(
            mastery=mastery, archetype=aid,
            attempts={c: {d: 0 for d in DIFFICULTY_LEVELS} for c in CONCEPTS},
            correct= {c: {d: 0 for d in DIFFICULTY_LEVELS} for c in CONCEPTS},
            streak=  {c: 0 for c in CONCEPTS},
        )
