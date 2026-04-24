"""
env/environment.py — TeachRL v2 OpenEnv-compliant Environment.
4 tasks: archetype_identification / adaptive_curriculum / blind_teaching / self_play_escalation
API: reset() / step() / state()
Theme 4: Self-Improvement via self-play escalation.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from env.archetypes import (
    ArchetypeID, ALL_ARCHETYPES, CONCEPTS, DIFFICULTY_LEVELS,
    make_archetype, sample_archetype,
)
from env.student import StudentSimulator
from self_play.escalator import SelfPlayEscalator, EpisodeResult

# ── Pydantic Models ───────────────────────────────────────────────────────────

class TutorAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concept:         str
    difficulty:      str
    hint_given:      bool = False
    archetype_guess: Optional[str] = None

class TutorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concept_success_rates:  Dict[str, float]
    concept_attempt_counts: Dict[str, int]
    concept_streaks:        Dict[str, int]
    prerequisite_readiness: Dict[str, float]
    engagement:             float
    fatigue:                float
    step_count:             int
    last_correct:           bool
    last_concept:           str
    last_difficulty:        str
    expert_hint:            str   = ""
    hint_reliability:       float = 0.7
    current_generation:     Dict[str, int] = Field(default_factory=dict)

class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: TutorObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)

class EpisodeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id:           str
    task_difficulty:   str
    max_steps:         int
    steps_remaining:   int
    cumulative_reward: float
    observation:       TutorObservation
    true_archetype:    Optional[str]          = None
    mastery_snapshot:  Optional[Dict[str, float]] = None

# ── Task Registry ─────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict] = {
    "archetype_identification": {
        "difficulty": "easy", "max_steps": 20, "score_fn": "identification",
        "reveal_archetype": False,
        "description": "Identify the hidden student archetype in 20 steps.",
    },
    "adaptive_curriculum": {
        "difficulty": "medium", "max_steps": 50, "score_fn": "curriculum",
        "reveal_archetype": True,
        "target_concepts": ["algebra_basics","linear_equations","functions","probability","geometry"],
        "mastery_threshold": 0.70,
        "description": "Teach a revealed archetype to mastery across 5 concepts in 50 steps.",
    },
    "blind_teaching": {
        "difficulty": "hard", "max_steps": 80, "score_fn": "blind",
        "reveal_archetype": False,
        "target_concepts": CONCEPTS, "mastery_threshold": 0.65,
        "description": "Infer AND teach unknown archetype across all 10 concepts in 80 steps.",
    },
    "self_play_escalation": {
        "difficulty": "expert", "max_steps": 80, "score_fn": "self_play",
        "reveal_archetype": False,
        "target_concepts": CONCEPTS, "mastery_threshold": 0.65,
        "n_episodes": 5,
        "description": "Face auto-escalated student variants across 5 sequential episodes.",
    },
}

# ── Environment ───────────────────────────────────────────────────────────────

class TeachRLEnv:
    """
    TeachRL v2 — OpenEnv-compliant adaptive tutoring environment.
    The agent must infer hidden student archetype and adapt curriculum.
    Self-play escalator makes archetypes harder as agent improves.
    """
    VERSION = "2.0.0"
    ENV_ID  = "TeachRL-v2"

    def __init__(self, task_id: str = "blind_teaching",
                 seed: int = 42, eval_mode: bool = False,
                 escalator: Optional[SelfPlayEscalator] = None):
        assert task_id in TASK_REGISTRY, f"Unknown task: {task_id}. Choose from {list(TASK_REGISTRY)}"
        self.task_id   = task_id
        self.task_cfg  = TASK_REGISTRY[task_id]
        self.seed      = seed
        self.eval_mode = eval_mode
        self.escalator = escalator or SelfPlayEscalator()

        self._sim            = StudentSimulator(seed=seed)
        self._step_count     = 0
        self._cum_reward     = 0.0
        self._done           = False
        self._prev_mastery:  Dict[str, float] = {}
        self._last_obs:      Optional[TutorObservation] = None
        self._last_correct   = False
        self._last_concept   = CONCEPTS[0]
        self._last_diff      = "easy"
        self._agent_guesses: List[Optional[str]] = []
        self._hint_reliability = 0.7
        self._sp_episode     = 0
        self._sp_scores:     List[float] = []

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              archetype_id: Optional[ArchetypeID] = None) -> TutorObservation:
        _seed = seed if seed is not None else self.seed
        self._step_count   = 0
        self._cum_reward   = 0.0
        self._done         = False
        self._agent_guesses = []
        self._last_correct = False
        self._last_concept = CONCEPTS[0]
        self._last_diff    = "easy"

        rng = np.random.default_rng(_seed + self._sp_episode)
        self._hint_reliability = float(rng.uniform(0.4, 0.95))

        forced = archetype_id
        if forced is None and self.task_cfg.get("reveal_archetype"):
            forced = ALL_ARCHETYPES[rng.integers(len(ALL_ARCHETYPES))]

        self._sim.reset(seed=_seed, archetype_id=forced)
        self._prev_mastery = dict(self._sim.state.mastery)
        obs = self._build_obs()
        self._last_obs = obs
        return obs

    def step(self, action: TutorAction | Dict) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        if isinstance(action, dict):
            action = TutorAction(**action)
        if action.concept not in CONCEPTS:
            raise ValueError(f"Invalid concept: {action.concept}")
        if action.difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(f"Invalid difficulty: {action.difficulty}")

        if action.archetype_guess:
            self._agent_guesses.append(action.archetype_guess)

        correct, delta = self._sim.answer_question(action.concept, action.difficulty)
        self._last_correct = correct
        self._last_concept = action.concept
        self._last_diff    = action.difficulty
        self._step_count  += 1

        reward = self._compute_reward(action, correct, delta)
        self._cum_reward += reward

        done = self._check_done()
        self._done = done
        if done:
            self._on_episode_end()

        obs = self._build_obs()
        self._last_obs = obs
        self._prev_mastery = dict(self._sim.state.mastery)

        return StepResult(
            observation=obs, reward=reward, done=done,
            info={
                "mastery":              dict(self._sim.state.mastery),
                "task_score":           self._task_score(),
                "archetype_identified": self._inf_score(),
                "step":                 self._step_count,
                "escalation":           self.escalator.summary(),
            }
        )

    def state(self) -> EpisodeInfo:
        obs = self._last_obs or self._build_obs()
        return EpisodeInfo(
            task_id=self.task_id,
            task_difficulty=self.task_cfg["difficulty"],
            max_steps=self.task_cfg["max_steps"],
            steps_remaining=self.task_cfg["max_steps"] - self._step_count,
            cumulative_reward=self._cum_reward,
            observation=obs,
            true_archetype=self._sim.archetype_id.value if self.eval_mode else None,
            mastery_snapshot=dict(self._sim.state.mastery) if self.eval_mode else None,
        )

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: TutorAction, correct: bool, delta: float) -> float:
        c   = action.concept
        pm  = self._prev_mastery.get(c, 0.0)
        thr = self.task_cfg.get("mastery_threshold", 0.70)
        tgt = self.task_cfg.get("target_concepts", CONCEPTS)

        is_tgt = c in tgt
        done   = pm >= thr
        wt     = 0.0 if not is_tgt else (0.5 if done else 3.0)

        r  = max(delta, 0.0) * wt
        r += -0.08 if (is_tgt and done) else 0.0
        r += self._sim.state.engagement * 0.08
        r += self._sim.prerequisite_readiness(c) * 0.05
        t  = {"easy": 0.3, "medium": 0.55, "hard": 0.8}[action.difficulty]
        r += float(np.exp(-4 * (self._sim.state.mastery[c] - t) ** 2)) * 0.08
        r += 0.10 if correct else 0.0

        if action.archetype_guess:
            aid = self._sim.archetype_id
            r += 0.20 if (aid and action.archetype_guess == aid.value) else -0.05

        if self._step_count >= self.task_cfg["max_steps"] - 1:
            r += self._task_score() * 0.5

        r += -0.05 if action.hint_given else 0.0
        r += -self._sim.state.fatigue * 0.04
        return float(np.clip(r, 0.0, 1.0))

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _task_score(self) -> float:
        fn  = self.task_cfg["score_fn"]
        m   = self._sim.state.mastery
        thr = self.task_cfg.get("mastery_threshold", 0.70)
        tgt = self.task_cfg.get("target_concepts", CONCEPTS)

        if fn == "identification":
            return float(np.clip(self._inf_score(), 0.001, 0.999))

        elif fn == "curriculum":
            sc  = float(np.mean([min(m[c]/thr, 1.0) for c in tgt]))
            eng = float(np.clip(self._sim.state.engagement / 0.5, 0.5, 1.0))
            return float(np.clip(sc * eng, 0.001, 0.999))

        elif fn == "blind":
            ms  = float(np.mean([min(m[c]/thr, 1.0) for c in tgt]))
            ia  = self._inf_score()
            sr  = 1.0 - 0.15 * (self._step_count / self.task_cfg["max_steps"])
            return float(np.clip(ms * 0.7 + ia * 0.2 + sr * 0.1, 0.001, 0.999))

        elif fn == "self_play":
            if not self._sp_scores:
                return 0.001
            gb = self.escalator.total_generation_sum() * 0.02
            return float(np.clip(float(np.mean(self._sp_scores)) + gb, 0.001, 0.999))

        return 0.001

    def _inf_score(self) -> float:
        if not self._agent_guesses:
            return 0.0
        aid = self._sim.archetype_id
        if aid is None:
            return 0.0
        w    = np.linspace(0.5, 1.0, len(self._agent_guesses))
        hits = [1.0 if g == aid.value else 0.0 for g in self._agent_guesses]
        return float(np.clip(np.average(hits, weights=w), 0.0, 1.0))

    def _on_episode_end(self):
        thr = self.task_cfg.get("mastery_threshold", 0.65)
        tgt = self.task_cfg.get("target_concepts", CONCEPTS)
        m   = self._sim.state.mastery
        raw = float(np.mean([min(m[c]/thr, 1.0) for c in tgt]))
        eng = float(np.clip(self._sim.state.engagement / 0.5, 0.5, 1.0))
        ep_score = float(np.clip(raw * eng, 0.001, 0.999))
        self._sp_scores.append(ep_score)
        self.escalator.record(EpisodeResult(
            archetype_id=self._sim.archetype_id or ArchetypeID.OVERCONFIDENT,
            task_score=ep_score,
            mastery_achieved=dict(self._sim.state.mastery),
            engagement_final=self._sim.state.engagement,
            fatigue_final=self._sim.state.fatigue,
            steps_used=self._step_count,
            inference_correct=self._inf_score() > 0.5,
        ))
        self._sp_episode += 1

    def _check_done(self) -> bool:
        return (self._step_count >= self.task_cfg["max_steps"] or
                self._sim.state.engagement < 0.10)

    def _build_obs(self) -> TutorObservation:
        s = self._sim.state
        rates, counts, streaks, prereqs = {}, {}, {}, {}
        for c in CONCEPTS:
            ta = sum(s.attempts[c].values())
            tc = sum(s.correct[c].values())
            rates[c]   = round(tc / max(ta, 1), 4)
            counts[c]  = ta
            streaks[c] = s.streak.get(c, 0)
            prereqs[c] = round(self._sim.prerequisite_readiness(c), 4)

        rng  = np.random.default_rng(self.seed + self._step_count)
        hint = self._sim.get_expert_hint()
        if rng.random() > self._hint_reliability:
            fake = make_archetype(sample_archetype(rng), rng)
            hint = fake.hint

        return TutorObservation(
            concept_success_rates=rates, concept_attempt_counts=counts,
            concept_streaks=streaks, prerequisite_readiness=prereqs,
            engagement=round(s.engagement, 4), fatigue=round(s.fatigue, 4),
            step_count=self._step_count, last_correct=self._last_correct,
            last_concept=self._last_concept, last_difficulty=self._last_diff,
            expert_hint=hint, hint_reliability=round(self._hint_reliability, 3),
            current_generation={a.value: self.escalator.generation(a) for a in ALL_ARCHETYPES},
        )

    @property
    def action_space(self) -> Dict:
        return {
            "type": "discrete_composite",
            "concept":         {"type": "categorical", "values": CONCEPTS},
            "difficulty":      {"type": "categorical", "values": DIFFICULTY_LEVELS},
            "hint_given":      {"type": "boolean"},
            "archetype_guess": {"type": "categorical",
                                "values": [a.value for a in ALL_ARCHETYPES] + [None]},
        }

    @property
    def observation_space(self) -> Dict:
        return {
            "type": "dict",
            "concept_success_rates":  {"type": "float", "shape": (10,), "range": [0, 1]},
            "concept_attempt_counts": {"type": "int",   "shape": (10,)},
            "concept_streaks":        {"type": "int",   "shape": (10,)},
            "prerequisite_readiness": {"type": "float", "shape": (10,), "range": [0, 1]},
            "engagement":             {"type": "float", "shape": (1,),  "range": [0, 1]},
            "fatigue":                {"type": "float", "shape": (1,),  "range": [0, 1]},
            "hint_reliability":       {"type": "float", "shape": (1,),  "range": [0, 1]},
        }

    def render(self, mode: str = "text") -> str:
        s  = self._sim.state
        at = self._sim.archetype_id
        lines = [
            f"\n{'='*62}", f"  TeachRL v2 | Task: {self.task_id}",
            f"  Step:{self._step_count}/{self.task_cfg['max_steps']} "
            f"Reward:{self._cum_reward:.3f} Score:{self._task_score():.3f}",
            f"  Engagement:{s.engagement:.2f} Fatigue:{s.fatigue:.2f}",
            f"  Archetype:{at.value if (at and self.eval_mode) else '???'}",
            f"  Guesses:{self._agent_guesses[-3:] if self._agent_guesses else 'none'}",
            f"{'─'*62}",
        ]
        for c in CONCEPTS:
            ta = sum(s.attempts[c].values())
            tc = sum(s.correct[c].values())
            sr = tc / max(ta, 1)
            m  = s.mastery[c]
            bar = "█" * int(m * 8) + "░" * (8 - int(m * 8))
            lines.append(f"  {c:<25}{bar} {m:.2f} sr={sr:.2f} att={ta}")
        lines.append("=" * 62)
        return "\n".join(lines)
