"""baseline/agents.py — 4 baseline agents of increasing sophistication."""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from env.archetypes import CONCEPTS, DIFFICULTY_LEVELS, PREREQUISITES, ALL_ARCHETYPES, ArchetypeID

class RandomAgent:
    name = "RandomAgent"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def __call__(self, obs: Dict) -> Dict:
        return {"concept": self.rng.choice(CONCEPTS),
                "difficulty": self.rng.choice(DIFFICULTY_LEVELS),
                "hint_given": False, "archetype_guess": None}

class HeuristicAgent:
    name = "HeuristicAgent"
    def __call__(self, obs: Dict) -> Dict:
        rates   = obs["concept_success_rates"]
        prereqs = obs["prerequisite_readiness"]
        fatigue = obs["fatigue"]
        eligible = [c for c in CONCEPTS if prereqs.get(c, 1.0) >= 0.35] or CONCEPTS
        concept  = min(eligible, key=lambda c: rates.get(c, 0.0))
        sr = rates.get(concept, 0.0)
        diff = "easy" if (fatigue > 0.65 or sr < 0.40) else "medium" if sr < 0.70 else "hard"
        return {"concept": concept, "difficulty": diff, "hint_given": False, "archetype_guess": None}

class GreedyArchetypeAgent:
    name = "GreedyArchetypeAgent"
    def __init__(self):
        self._topo = self._sort(CONCEPTS)

    def _sort(self, concepts):
        def depth(c, vis=frozenset()):
            pr = [p for p in PREREQUISITES.get(c, []) if p in concepts]
            return 0 if not pr else 1 + max(depth(p, vis | {c}) for p in pr)
        return sorted(concepts, key=depth)

    def __call__(self, obs: Dict) -> Dict:
        rates   = obs["concept_success_rates"]
        attempts = obs["concept_attempt_counts"]
        streaks = obs["concept_streaks"]
        prereqs = obs["prerequisite_readiness"]
        fatigue = obs["fatigue"]
        best = None; best_score = -1.0
        for i, c in enumerate(self._topo):
            sr  = rates.get(c, 0.0); att = attempts.get(c, 0)
            gap = max(0.70 - sr, 0.0); pr = prereqs.get(c, 1.0)
            if pr < 0.45 and att == 0: continue
            score = gap * (pr**2) * (1.2 if att == 0 else 1.0) * (1.0 + 0.08*(len(self._topo)-i))
            if score > best_score: best_score = score; best = c
        concept = best or min(CONCEPTS, key=lambda c: rates.get(c, 0.0))
        sr = rates.get(concept, 0.0); streak = streaks.get(concept, 0)
        diff = "easy" if fatigue > 0.65 else "hard" if (streak >= 3 or sr > 0.70) else "medium" if sr > 0.40 else "easy"
        return {"concept": concept, "difficulty": diff, "hint_given": False, "archetype_guess": None}

class ArchetypeInferenceAgent:
    """Actively identifies archetype then applies archetype-specific strategy."""
    name = "ArchetypeInferenceAgent"
    STRATEGIES = {
        "overconfident_learner":  {"pref_diff": "hard"},
        "anxious_perfectionist":  {"pref_diff": "easy"},
        "adhd_sprinter":          {"switch_every": 2, "pref_diff": "medium"},
        "slow_steady_builder":    {"pref_diff": "easy"},
        "strategic_gamer":        {"pref_diff": "hard"},
        "emotional_learner":      {"pref_diff": "medium"},
        "uneven_genius":          {"probe_all": True, "pref_diff": "medium"},
        "impostor":               {"pref_diff": "hard"},
    }

    def __init__(self):
        self._votes     = {a.value: 0.0 for a in ALL_ARCHETYPES}
        self._identified: Optional[str] = None
        self._step      = 0
        self._topo      = self._sort(CONCEPTS)

    def _sort(self, concepts):
        def depth(c, vis=frozenset()):
            pr = [p for p in PREREQUISITES.get(c, []) if p in concepts]
            return 0 if not pr else 1 + max(depth(p, vis | {c}) for p in pr)
        return sorted(concepts, key=depth)

    def _update_votes(self, obs: Dict):
        rates  = obs["concept_success_rates"]; eng = obs["engagement"]
        fatigue = obs["fatigue"]; streaks = list(obs["concept_streaks"].values())
        hint = obs.get("expert_hint", ""); reliability = obs.get("hint_reliability", 0.7)
        avg_sr = float(np.mean(list(rates.values())))
        if avg_sr > 0.55 and eng < 0.75:                          self._votes["overconfident_learner"] += 0.3
        if eng < 0.60 and fatigue > 0.3:                          self._votes["anxious_perfectionist"] += 0.3
        if np.std(streaks) > 1.5:                                  self._votes["adhd_sprinter"]         += 0.3
        if sum(obs["concept_attempt_counts"].values())>10 and avg_sr<0.35 and eng>0.65:
                                                                   self._votes["slow_steady_builder"]   += 0.3
        if avg_sr > 0.65 and abs(eng - 0.80) < 0.15:             self._votes["strategic_gamer"]        += 0.3
        if hint and reliability > 0.6:
            for aid in ALL_ARCHETYPES:
                kws = aid.value.split("_")
                if any(kw in hint.lower() for kw in kws):
                    self._votes[aid.value] += reliability * 0.5

    def __call__(self, obs: Dict) -> Dict:
        self._step += 1
        self._update_votes(obs)
        if self._identified is None and self._step >= 6:
            top = max(self._votes, key=self._votes.get)
            if self._votes[top] > 0.2: self._identified = top
        guess = self._identified or max(self._votes, key=self._votes.get)
        rates   = obs["concept_success_rates"]
        prereqs = obs["prerequisite_readiness"]
        fatigue = obs["fatigue"]
        attempts = obs["concept_attempt_counts"]
        strat = self.STRATEGIES.get(guess, {"pref_diff": "medium"})
        if strat.get("probe_all"):
            unvisited = [c for c in self._topo if attempts.get(c,0)==0 and prereqs.get(c,1.0)>=0.35]
            concept   = unvisited[0] if unvisited else min(CONCEPTS, key=lambda c: rates.get(c,0.0))
        elif strat.get("switch_every"):
            concept = self._topo[(self._step // strat["switch_every"]) % len(self._topo)]
        else:
            eligible = [c for c in self._topo if prereqs.get(c,1.0)>=0.35]
            concept  = min(eligible or CONCEPTS, key=lambda c: rates.get(c,0.0))
        sr   = rates.get(concept, 0.0); pref = strat.get("pref_diff", "medium")
        if fatigue > 0.65:           diff = "easy"
        elif pref == "hard" and sr > 0.40: diff = "hard"
        elif pref == "easy":         diff = "easy"
        else:                        diff = "medium" if sr > 0.35 else "easy"
        return {"concept": concept, "difficulty": diff, "hint_given": False, "archetype_guess": guess}

def get_agent(name: str, **kwargs):
    registry = {"random": RandomAgent, "heuristic": HeuristicAgent,
                 "greedy": GreedyArchetypeAgent, "inference": ArchetypeInferenceAgent}
    if name not in registry: raise ValueError(f"Unknown agent: {name}")
    return registry[name](**kwargs)
