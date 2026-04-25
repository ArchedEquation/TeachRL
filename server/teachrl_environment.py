"""
server/teachrl_environment.py — TeachRL v2 Environment using OpenEnv base class.

Inherits from openenv.core.env_server.interfaces.Environment
as required by the OpenEnv spec.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uuid import uuid4
from typing import Any, Optional
import numpy as np

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False
    # Fallback base class
    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = True
        def __init__(self, transform=None, rubric=None):
            self.transform = transform; self.rubric = rubric
    class State:
        def __init__(self, **kw):
            for k,v in kw.items(): setattr(self,k,v)

try:
    from models import TutorAction, TutorObservation, TutorState
except ImportError:
    from server.models import TutorAction, TutorObservation, TutorState

from env.archetypes import (
    ArchetypeID, ALL_ARCHETYPES, CONCEPTS, DIFFICULTY_LEVELS,
    make_archetype, sample_archetype,
)
from env.student import StudentSimulator
from self_play.escalator import SelfPlayEscalator, EpisodeResult
from env.environment import TASK_REGISTRY


class TeachRLEnvironment(Environment):
    """
    TeachRL v2 — OpenEnv Environment base class implementation.
    Theme 4: Self-Improvement via self-play difficulty escalation.

    8 hidden student archetypes modelled via Bayesian Knowledge Tracing.
    Agent must infer archetype AND teach simultaneously.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        task_id:   str  = "blind_teaching",
        seed:      int  = 42,
        eval_mode: bool = False,
        transform  = None,
        rubric     = None,
    ):
        super().__init__(transform=transform, rubric=rubric)
        assert task_id in TASK_REGISTRY
        self.task_id   = task_id
        self.task_cfg  = TASK_REGISTRY[task_id]
        self.seed      = seed
        self.eval_mode = eval_mode
        self.escalator = SelfPlayEscalator()

        self._sim            = StudentSimulator(seed=seed)
        self._step_count     = 0
        self._cum_reward     = 0.0
        self._done           = False
        self._prev_mastery   = {}
        self._last_correct   = False
        self._last_concept   = CONCEPTS[0]
        self._last_diff      = "easy"
        self._agent_guesses  = []
        self._hint_reliability = 0.7
        self._sp_episode     = 0
        self._sp_scores      = []
        self._episode_id     = str(uuid4())

        # State for openenv.core.env_server.types.State
        self._state = State(episode_id=self._episode_id, step_count=0) if OPENENV_AVAILABLE else None

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: Optional[str] = None, **kwargs) -> TutorObservation:
        """Reset environment and return initial observation."""
        _seed = seed if seed is not None else self.seed
        if task_id and task_id in TASK_REGISTRY:
            self.task_id  = task_id
            self.task_cfg = TASK_REGISTRY[task_id]

        self._step_count    = 0
        self._cum_reward    = 0.0
        self._done          = False
        self._agent_guesses = []
        self._last_correct  = False
        self._last_concept  = CONCEPTS[0]
        self._last_diff     = "easy"
        self._episode_id    = episode_id or str(uuid4())

        rng = np.random.default_rng(_seed + self._sp_episode)
        self._hint_reliability = float(rng.uniform(0.4, 0.95))

        forced = None
        if self.task_cfg.get("reveal_archetype"):
            forced = ArchetypeID(list(ArchetypeID)[rng.integers(len(ArchetypeID))])

        self._sim.reset(seed=_seed, archetype_id=forced)
        self._prev_mastery = dict(self._sim.state.mastery)

        if OPENENV_AVAILABLE:
            self._reset_rubric()
            self._state = State(episode_id=self._episode_id, step_count=0)

        return self._build_obs()

    def step(self, action: TutorAction, timeout_s: Optional[float] = None,
             **kwargs) -> TutorObservation:
        """Execute one tutoring step."""
        if self._done:
            obs = self._build_obs()
            obs.done   = True
            obs.reward = 0.0
            return obs

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

        reward           = self._compute_reward(action, correct, delta)
        self._cum_reward += reward
        done             = self._check_done()
        self._done       = done

        if done:
            self._on_episode_end()

        if OPENENV_AVAILABLE:
            self._state = State(episode_id=self._episode_id, step_count=self._step_count)

        self._prev_mastery = dict(self._sim.state.mastery)

        obs        = self._build_obs()
        obs.done   = done
        obs.reward = reward
        obs.metadata = {
            "task_score":   self._task_score(),
            "arch_acc":     self._inf_score(),
            "escalation":   self.escalator.total_generation_sum(),
            "cum_reward":   self._cum_reward,
        }
        return obs

    @property
    def state(self) -> TutorState:
        """Get current episode state."""
        obs = self._build_obs()
        return TutorState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self.task_id,
            task_difficulty=self.task_cfg["difficulty"],
            max_steps=self.task_cfg["max_steps"],
            cumulative_reward=self._cum_reward,
            true_archetype=self._sim.archetype_id.value if self.eval_mode else None,
            mastery_snapshot=dict(self._sim.state.mastery) if self.eval_mode else None,
        )

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: TutorAction, correct: bool, delta: float) -> float:
        c         = action.concept
        prev_m    = self._prev_mastery.get(c, 0.0)
        thr       = self.task_cfg.get("mastery_threshold", 0.70)
        tgts      = self.task_cfg.get("target_concepts", CONCEPTS)
        is_tgt    = c in tgts
        done_already = prev_m >= thr

        r  = max(delta, 0.0) * (0.0 if not is_tgt else (0.5 if done_already else 3.0))
        r += -0.08 if (is_tgt and done_already) else 0.0
        r += self._sim.state.engagement * 0.08
        r += self._sim.prerequisite_readiness(c) * 0.05
        r += self._sim._ideal_difficulty_score(c, action.difficulty) * 0.08
        r += 0.10 if correct else 0.0

        if action.archetype_guess:
            r += 0.20 if (self._sim.archetype_id and
                          action.archetype_guess == self._sim.archetype_id.value) else -0.05

        if self._step_count >= self.task_cfg["max_steps"] - 1:
            r += self._task_score() * 0.5

        r += -0.05 if action.hint_given else 0.0
        r += -self._sim.state.fatigue * 0.04
        return float(np.clip(r, 0.0, 1.0))

    # ── Task score ────────────────────────────────────────────────────────────

    def _task_score(self) -> float:
        fn   = self.task_cfg["score_fn"]
        m    = self._sim.state.mastery
        thr  = self.task_cfg.get("mastery_threshold", 0.70)
        tgts = self.task_cfg.get("target_concepts", CONCEPTS)

        if fn == "identification":
            return float(np.clip(self._inf_score(), 0.001, 0.999))
        elif fn == "curriculum":
            sc  = float(np.mean([min(m[c]/thr, 1.0) for c in tgts]))
            eng = float(np.clip(self._sim.state.engagement/0.5, 0.5, 1.0))
            return float(np.clip(sc * eng, 0.001, 0.999))
        elif fn == "blind":
            ms = float(np.mean([min(m[c]/thr, 1.0) for c in tgts]))
            ia = self._inf_score()
            sr = 1.0 - 0.15 * (self._step_count / self.task_cfg["max_steps"])
            return float(np.clip(ms*0.7 + ia*0.2 + sr*0.1, 0.001, 0.999))
        elif fn == "self_play":
            if not self._sp_scores: return 0.001
            gb = self.escalator.total_generation_sum() * 0.02
            return float(np.clip(float(np.mean(self._sp_scores)) + gb, 0.001, 0.999))
        return 0.001

    def _inf_score(self) -> float:
        if not self._agent_guesses: return 0.0
        true_id = self._sim.archetype_id
        if true_id is None: return 0.0
        w    = np.linspace(0.5, 1.0, len(self._agent_guesses))
        hits = [1.0 if g == true_id.value else 0.0 for g in self._agent_guesses]
        return float(np.clip(np.average(hits, weights=w), 0.0, 1.0))

    def _on_episode_end(self):
        thr      = self.task_cfg.get("mastery_threshold", 0.65)
        tgts     = self.task_cfg.get("target_concepts", CONCEPTS)
        m        = self._sim.state.mastery
        raw      = float(np.mean([min(m[c]/thr, 1.0) for c in tgts]))
        eng_mult = float(np.clip(self._sim.state.engagement/0.5, 0.5, 1.0))
        ep_score = float(np.clip(raw * eng_mult, 0.001, 0.999))
        self._sp_scores.append(ep_score)
        self.escalator.record(EpisodeResult(
            archetype_id=self._sim.archetype_id or ArchetypeID.OVERCONFIDENT,
            task_score=ep_score, mastery_achieved=dict(self._sim.state.mastery),
            engagement_final=self._sim.state.engagement, fatigue_final=self._sim.state.fatigue,
            steps_used=self._step_count, inference_correct=self._inf_score() > 0.5,
        ))
        self._sp_episode += 1

    def _check_done(self) -> bool:
        return (self._step_count >= self.task_cfg["max_steps"] or
                self._sim.state.engagement < 0.10)

    def _build_obs(self) -> TutorObservation:
        s = self._sim.state
        rates, counts, streaks, prereqs = {}, {}, {}, {}
        for c in CONCEPTS:
            ta        = sum(s.attempts[c].values())
            tc        = sum(s.correct[c].values())
            rates[c]  = round(tc / max(ta, 1), 4)
            counts[c] = ta
            streaks[c] = s.streak.get(c, 0)
            prereqs[c] = round(self._sim.prerequisite_readiness(c), 4)

        rng  = np.random.default_rng(self.seed + self._step_count)
        hint = self._sim.get_expert_hint()
        if rng.random() > self._hint_reliability:
            fake = make_archetype(sample_archetype(rng), rng)
            hint = fake.observable_hint

        return TutorObservation(
            concept_success_rates=rates, concept_attempt_counts=counts,
            concept_streaks=streaks, prerequisite_readiness=prereqs,
            engagement=round(s.engagement, 4), fatigue=round(s.fatigue, 4),
            step_count=self._step_count, last_correct=self._last_correct,
            last_concept=self._last_concept, last_difficulty=self._last_diff,
            expert_hint=hint, hint_reliability=round(self._hint_reliability, 3),
            current_generation={a.value: self.escalator.generation(a) for a in ALL_ARCHETYPES},
            done=self._done, reward=None,
        )