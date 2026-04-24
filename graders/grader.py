"""graders/grader.py — 4 task graders returning scores in (0.001, 0.999)."""
from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List
from env.environment import TeachRLEnv, TASK_REGISTRY
from self_play.escalator import SelfPlayEscalator

AgentFn = Callable[[Dict], Dict]

@dataclass
class GraderResult:
    task_id: str; difficulty: str; score: float
    mean_task_score: float; std_task_score: float
    mean_cum_reward: float; mean_steps: float
    mean_engagement: float; mean_fatigue: float
    episodes_run: int; elapsed: float
    per_episode_scores: List[float]; archetype_accuracy: float = 0.0

    def summary(self) -> str:
        bar = "█" * int(self.score * 30) + "░" * (30 - int(self.score * 30))
        return (f"\n╔{'═'*52}╗\n"
                f"║  Task: {self.task_id:<43}║\n"
                f"║  Difficulty: {self.difficulty:<39}║\n"
                f"╠{'═'*52}╣\n"
                f"║  Score [{bar}] {self.score:.3f}  ║\n"
                f"║  Mean:{self.mean_task_score:.4f} ±{self.std_task_score:.4f}  "
                f"ArchAcc:{self.archetype_accuracy:.3f}  ║\n"
                f"╚{'═'*52}╝")

class BaseGrader:
    def __init__(self, task_id: str, n_episodes: int = 10, seeds=None):
        self.task_id    = task_id
        self.n_episodes = n_episodes
        self.seeds      = seeds or list(range(n_episodes))

    def grade(self, agent_fn: AgentFn) -> GraderResult:
        t0 = time.time()
        scores, rewards, steps, eng, fat, inf_acc = [], [], [], [], [], []
        for seed in self.seeds[:self.n_episodes]:
            env  = TeachRLEnv(task_id=self.task_id, seed=seed, eval_mode=True)
            obs  = env.reset(seed=seed); done = False
            while not done:
                result = env.step(agent_fn(obs.model_dump()))
                obs = result.observation; done = result.done
            scores.append(env._task_score())
            rewards.append(env._cum_reward)
            steps.append(env._step_count)
            eng.append(env._sim.state.engagement)
            fat.append(env._sim.state.fatigue)
            inf_acc.append(env._inf_score())
        return GraderResult(
            task_id=self.task_id, difficulty=TASK_REGISTRY[self.task_id]["difficulty"],
            score=float(np.clip(np.mean(scores), 0.001, 0.999)),
            mean_task_score=float(np.mean(scores)), std_task_score=float(np.std(scores)),
            mean_cum_reward=float(np.mean(rewards)), mean_steps=float(np.mean(steps)),
            mean_engagement=float(np.mean(eng)), mean_fatigue=float(np.mean(fat)),
            episodes_run=self.n_episodes, elapsed=time.time()-t0,
            per_episode_scores=scores, archetype_accuracy=float(np.mean(inf_acc)),
        )

class EasyGrader(BaseGrader):
    def __init__(self, n_episodes=10): super().__init__("archetype_identification", n_episodes)

class MediumGrader(BaseGrader):
    def __init__(self, n_episodes=10): super().__init__("adaptive_curriculum", n_episodes)

class HardGrader(BaseGrader):
    def __init__(self, n_episodes=10): super().__init__("blind_teaching", n_episodes)

class ExpertGrader:
    def __init__(self, n_episodes=5):
        self.task_id    = "self_play_escalation"
        self.n_episodes = n_episodes

    def grade(self, agent_fn: AgentFn) -> GraderResult:
        t0 = time.time(); escalator = SelfPlayEscalator()
        env = TeachRLEnv(task_id=self.task_id, seed=42, eval_mode=True, escalator=escalator)
        scores, inf_acc = [], []
        for ep in range(self.n_episodes):
            obs = env.reset(seed=42+ep); done = False
            while not done:
                result = env.step(agent_fn(obs.model_dump()))
                obs = result.observation; done = result.done
            scores.append(env._task_score()); inf_acc.append(env._inf_score())
        return GraderResult(
            task_id=self.task_id, difficulty="expert",
            score=float(np.clip(env._task_score(), 0.001, 0.999)),
            mean_task_score=float(np.mean(scores)), std_task_score=float(np.std(scores)),
            mean_cum_reward=env._cum_reward, mean_steps=float(env._step_count),
            mean_engagement=env._sim.state.engagement, mean_fatigue=env._sim.state.fatigue,
            episodes_run=self.n_episodes, elapsed=time.time()-t0,
            per_episode_scores=scores, archetype_accuracy=float(np.mean(inf_acc)),
        )

GRADER_REGISTRY: Dict[str, type] = {
    "archetype_identification": EasyGrader,
    "adaptive_curriculum":      MediumGrader,
    "blind_teaching":           HardGrader,
    "self_play_escalation":     ExpertGrader,
}

def run_all_graders(agent_fn: AgentFn, n_episodes: int = 10) -> Dict[str, GraderResult]:
    results = {}
    for task_id, Cls in GRADER_REGISTRY.items():
        n = min(n_episodes, 5) if task_id == "self_play_escalation" else n_episodes
        print(f"  Grading {task_id}...", end=" ", flush=True)
        r = Cls(n_episodes=n).grade(agent_fn); results[task_id] = r
        print(f"score={r.score:.4f} arch_acc={r.archetype_accuracy:.3f}")
    return results
