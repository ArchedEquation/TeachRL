"""
env/archetypes.py
8 Student Archetypes — each with unique BKT parameters breaking a different RL assumption.
Theme 4: Self-Improvement — the environment escalates these archetypes as agent improves.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

# ── Constants ─────────────────────────────────────────────────────────────────

CONCEPTS = [
    "algebra_basics", "linear_equations", "quadratic_equations", "functions",
    "trigonometry", "probability", "statistics", "calculus_intro",
    "geometry", "number_theory",
]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
PREREQUISITES: Dict[str, List[str]] = {
    "algebra_basics":      [],
    "linear_equations":    ["algebra_basics"],
    "quadratic_equations": ["algebra_basics", "linear_equations"],
    "functions":           ["algebra_basics", "linear_equations"],
    "trigonometry":        ["functions", "geometry"],
    "probability":         ["algebra_basics"],
    "statistics":          ["algebra_basics", "probability"],
    "calculus_intro":      ["functions", "trigonometry"],
    "geometry":            ["algebra_basics"],
    "number_theory":       ["algebra_basics"],
}

# ── Archetype IDs ─────────────────────────────────────────────────────────────

class ArchetypeID(str, Enum):
    OVERCONFIDENT = "overconfident_learner"
    ANXIOUS       = "anxious_perfectionist"
    ADHD          = "adhd_sprinter"
    SLOW_STEADY   = "slow_steady_builder"
    GAMER         = "strategic_gamer"
    EMOTIONAL     = "emotional_learner"
    UNEVEN        = "uneven_genius"
    IMPOSTOR      = "impostor"

ALL_ARCHETYPES = list(ArchetypeID)

# ── BKT Parameter Bundle ──────────────────────────────────────────────────────

@dataclass
class BKTParams:
    p_init: float; p_learn: float; p_forget: float; p_slip: float; p_guess: float

    def slip(self, difficulty: str) -> float:
        return float(np.clip(self.p_slip + {"easy":-0.05,"medium":0.0,"hard":0.08}[difficulty], 0.01, 0.58))

    def guess(self, difficulty: str) -> float:
        return float(np.clip(self.p_guess - {"easy":-0.05,"medium":0.0,"hard":0.08}[difficulty], 0.01, 0.58))

# ── Base Archetype ────────────────────────────────────────────────────────────

class BaseArchetype:
    archetype_id: ArchetypeID
    description:  str
    hint:         str
    DEFAULT = BKTParams(0.25, 0.22, 0.05, 0.12, 0.16)

    def __init__(self, rng: np.random.Generator):
        self.rng = rng; self._step = 0

    def get_params(self, concept: str, difficulty: str) -> BKTParams:
        return self.DEFAULT

    def engagement_delta(self, concept: str, difficulty: str,
                         correct: bool, streak: int, mastery: float) -> float:
        t = {"easy": 0.3, "medium": 0.55, "hard": 0.8}[difficulty]
        zpd = float(np.exp(-4 * (mastery - t) ** 2))
        return float(np.clip(0.05 * zpd - 0.02 * (not correct), -0.15, 0.10))

    def fatigue_delta(self) -> float: return 0.02

    def on_step(self, concept: str, difficulty: str, correct: bool):
        self._step += 1

# ── 1. Overconfident Learner ──────────────────────────────────────────────────

class OverconfidentLearner(BaseArchetype):
    archetype_id = ArchetypeID.OVERCONFIDENT
    description  = "High p_guess + p_slip. Gets lucky, makes careless errors."
    hint         = "Student appears confident but makes careless mistakes on known material."
    P = BKTParams(0.30, 0.10, 0.04, 0.35, 0.40)

    def get_params(self, c, d): return self.P
    def engagement_delta(self, c, d, correct, streak, mastery):
        if not correct and mastery > 0.5: return -0.12
        return super().engagement_delta(c, d, correct, streak, mastery)

# ── 2. Anxious Perfectionist ──────────────────────────────────────────────────

class AnxiousPerfectionist(BaseArchetype):
    archetype_id = ArchetypeID.ANXIOUS
    description  = "p_slip spikes on hard. Learns fast when calm. 2x fatigue."
    hint         = "Student performs well on easy questions but struggles significantly under pressure."
    CALM    = BKTParams(0.20, 0.35, 0.04, 0.05, 0.15)
    STRESSED = BKTParams(0.20, 0.35, 0.06, 0.40, 0.15)

    def get_params(self, c, d): return self.STRESSED if d == "hard" else self.CALM
    def engagement_delta(self, c, d, correct, streak, mastery):
        if d == "hard" and not correct: return -0.20
        if correct and streak >= 2:     return 0.08
        return super().engagement_delta(c, d, correct, streak, mastery)
    def fatigue_delta(self): return 0.04

# ── 3. ADHD Sprinter ──────────────────────────────────────────────────────────

class ADHDSprinter(BaseArchetype):
    archetype_id = ArchetypeID.ADHD
    description  = "High p_forget+p_learn. Bored by repetition, energised by novelty."
    hint         = "Student shows rapid learning but high forgetting. Engages best with variety."
    P = BKTParams(0.25, 0.45, 0.20, 0.12, 0.18)

    def __init__(self, rng):
        super().__init__(rng); self._counts: Dict[str, int] = {}

    def get_params(self, c, d): return self.P
    def engagement_delta(self, c, d, correct, streak, mastery):
        rep = self._counts.get(c, 0)
        return float(np.clip(
            super().engagement_delta(c, d, correct, streak, mastery)
            - 0.04 * max(rep - 2, 0) + (0.15 if rep == 0 else 0.0),
            -0.25, 0.20))
    def on_step(self, c, d, correct):
        super().on_step(c, d, correct); self._counts[c] = self._counts.get(c, 0) + 1

# ── 4. Slow Steady Builder ────────────────────────────────────────────────────

class SlowSteadyBuilder(BaseArchetype):
    archetype_id = ArchetypeID.SLOW_STEADY
    description  = "Very low p_learn, very low p_forget. Many exposures needed."
    hint         = "Student learns slowly but retains everything perfectly once mastered."
    P = BKTParams(0.15, 0.08, 0.01, 0.05, 0.12)

    def get_params(self, c, d): return self.P
    def engagement_delta(self, c, d, correct, streak, mastery):
        return 0.01 if correct else -0.01

# ── 5. Strategic Gamer ────────────────────────────────────────────────────────

class StrategicGamer(BaseArchetype):
    archetype_id = ArchetypeID.GAMER
    description  = "p_guess=0.55 on easy. High correct rate, zero real learning."
    hint         = "Student has high correct rate on easy questions but mastery is not growing."
    EASY = BKTParams(0.20, 0.05, 0.03, 0.10, 0.55)
    MED  = BKTParams(0.20, 0.05, 0.03, 0.10, 0.30)
    HARD = BKTParams(0.20, 0.05, 0.03, 0.10, 0.10)

    def get_params(self, c, d):
        return {"easy": self.EASY, "medium": self.MED, "hard": self.HARD}[d]
    def engagement_delta(self, c, d, correct, streak, mastery): return 0.02

# ── 6. Emotional Learner ──────────────────────────────────────────────────────

class EmotionalLearner(BaseArchetype):
    archetype_id = ArchetypeID.EMOTIONAL
    description  = "BKT params shift every 10 steps. Engagement is leading indicator."
    hint         = "Student performance varies cyclically. Watch engagement as a leading indicator."
    GOOD = BKTParams(0.25, 0.40, 0.03, 0.05, 0.18)
    BAD  = BKTParams(0.25, 0.05, 0.08, 0.45, 0.18)
    CYCLE = 10

    def __init__(self, rng): super().__init__(rng); self._good = True
    def get_params(self, c, d): return self.GOOD if self._good else self.BAD
    def on_step(self, c, d, correct):
        super().on_step(c, d, correct)
        if self._step % self.CYCLE == 0: self._good = not self._good
    def engagement_delta(self, c, d, correct, streak, mastery):
        warn = -0.06 if (self.CYCLE - self._step % self.CYCLE) <= 5 and self._good else 0.0
        base = super().engagement_delta(c, d, correct, streak, mastery)
        return float(np.clip(base + warn + (-0.05 if not self._good else 0.0), -0.25, 0.15))

# ── 7. Uneven Genius ──────────────────────────────────────────────────────────

class UnevenGenius(BaseArchetype):
    archetype_id = ArchetypeID.UNEVEN
    description  = "3 gift concepts (fast learn), 3 blind spots (barely learn). Random each episode."
    hint         = "Student shows unusually high ability in some areas and unexpected weakness in others."
    GIFT  = BKTParams(0.70, 0.60, 0.02, 0.05, 0.20)
    BLIND = BKTParams(0.05, 0.05, 0.08, 0.20, 0.12)
    NORM  = BKTParams(0.25, 0.22, 0.05, 0.12, 0.16)

    def __init__(self, rng):
        super().__init__(rng)
        sh = list(rng.permutation(CONCEPTS))
        self.gifts = set(sh[:3]); self.blinds = set(sh[3:6])

    def get_params(self, c, d):
        if c in self.gifts:  return self.GIFT
        if c in self.blinds: return self.BLIND
        return self.NORM

# ── 8. Impostor ───────────────────────────────────────────────────────────────

class Impostor(BaseArchetype):
    archetype_id = ArchetypeID.IMPOSTOR
    description  = "Appears mastered (p_init=0.85) but crumbles on hard questions."
    hint         = "Student claims prior knowledge but performance degrades significantly on harder problems."
    SURFACE  = BKTParams(0.85, 0.02, 0.03, 0.50, 0.40)
    EXPOSED  = BKTParams(0.85, 0.02, 0.03, 0.50, 0.10)

    def __init__(self, rng): super().__init__(rng); self._exposed: Dict[str, bool] = {}
    def get_params(self, c, d):
        if d == "hard": self._exposed[c] = True
        return self.EXPOSED if self._exposed.get(c) else self.SURFACE
    def engagement_delta(self, c, d, correct, streak, mastery): return 0.01

# ── Registry ──────────────────────────────────────────────────────────────────

ARCHETYPE_CLASSES = {
    ArchetypeID.OVERCONFIDENT: OverconfidentLearner,
    ArchetypeID.ANXIOUS:       AnxiousPerfectionist,
    ArchetypeID.ADHD:          ADHDSprinter,
    ArchetypeID.SLOW_STEADY:   SlowSteadyBuilder,
    ArchetypeID.GAMER:         StrategicGamer,
    ArchetypeID.EMOTIONAL:     EmotionalLearner,
    ArchetypeID.UNEVEN:        UnevenGenius,
    ArchetypeID.IMPOSTOR:      Impostor,
}

def make_archetype(aid: ArchetypeID, rng: np.random.Generator) -> BaseArchetype:
    return ARCHETYPE_CLASSES[aid](rng)

def sample_archetype(rng: np.random.Generator,
                     exclude: Optional[List[ArchetypeID]] = None) -> ArchetypeID:
    pool = [a for a in ALL_ARCHETYPES if a not in (exclude or [])]
    return pool[rng.integers(len(pool))]
