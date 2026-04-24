"""
baseline/rl_agent.py — PPO training and evaluation for TeachRL v2.
Usage:
    python baseline/rl_agent.py --train --task blind_teaching
    python baseline/rl_agent.py --eval --task all
    python baseline/rl_agent.py --train --eval --task blind_teaching
"""
import os, sys, time, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from env.gym_wrapper import TeachRLGymEnv, int_to_action, obs_to_vector
from env.environment import TASK_REGISTRY
from env.archetypes import ALL_ARCHETYPES
from graders.grader import GRADER_REGISTRY
from baseline.agents import RandomAgent, HeuristicAgent, GreedyArchetypeAgent, ArchetypeInferenceAgent

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TASK_CFG = {
    "archetype_identification": {"total_timesteps": 80_000,  "n_envs": 4, "policy_kwargs": dict(net_arch=[64, 64])},
    "adaptive_curriculum":      {"total_timesteps": 150_000, "n_envs": 4, "policy_kwargs": dict(net_arch=[128, 128])},
    "blind_teaching":           {"total_timesteps": 300_000, "n_envs": 4, "policy_kwargs": dict(net_arch=[256, 256, 128])},
    "self_play_escalation":     {"total_timesteps": 400_000, "n_envs": 4, "policy_kwargs": dict(net_arch=[256, 256, 128])},
}

class ProgressCallback(BaseCallback):
    def __init__(self, log_interval=50_000):
        super().__init__(); self._last = 0; self._t0 = time.time(); self.log_interval = log_interval
    def _on_step(self):
        if self.num_timesteps - self._last >= self.log_interval:
            fps = self.num_timesteps / max(time.time() - self._t0, 1)
            print(f"  step={self.num_timesteps:>7,} fps={fps:>5.0f} elapsed={time.time()-self._t0:>5.0f}s")
            self._last = self.num_timesteps
        return True

def train_ppo(task_id: str, seed: int = 42) -> PPO:
    cfg = TASK_CFG[task_id]
    print(f"\n{'='*55}\n  Training PPO: {task_id}\n  Steps:{cfg['total_timesteps']:,} Envs:{cfg['n_envs']}\n{'='*55}")
    max_steps = TASK_REGISTRY[task_id]["max_steps"]
    n_steps   = max(512, max_steps * 4)
    vec_env   = make_vec_env(lambda: TeachRLGymEnv(task_id=task_id, seed=seed), n_envs=cfg["n_envs"], seed=seed)
    eval_env  = make_vec_env(lambda: TeachRLGymEnv(task_id=task_id, seed=seed+999), n_envs=1, seed=seed+999)
    eval_cb   = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, log_path=MODEL_DIR,
                             eval_freq=max(10_000//cfg["n_envs"], 1), n_eval_episodes=5, verbose=0)
    model = PPO("MlpPolicy", vec_env, learning_rate=2e-4, n_steps=n_steps, batch_size=256,
                n_epochs=10, gamma=0.999, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
                vf_coef=0.5, max_grad_norm=0.5, policy_kwargs=cfg["policy_kwargs"], verbose=0, seed=seed)
    t0 = time.time()
    model.learn(cfg["total_timesteps"], callback=[eval_cb, ProgressCallback()], progress_bar=False)
    path = os.path.join(MODEL_DIR, f"ppo_{task_id}")
    model.save(path)
    print(f"\n  Done in {time.time()-t0:.0f}s → {path}.zip")
    return model

class PPOAgent:
    name = "PPO (trained)"
    def __init__(self, task_id: str):
        path = os.path.join(MODEL_DIR, f"ppo_{task_id}.zip")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model at {path}. Run --train first.")
        self._model     = PPO.load(path)
        self._max_steps = TASK_REGISTRY[task_id]["max_steps"]
    def __call__(self, obs_dict: dict) -> dict:
        vec = obs_to_vector(obs_dict, self._max_steps)
        action, _ = self._model.predict(vec, deterministic=True)
        c, d = int_to_action(int(action))
        return {"concept": c, "difficulty": d, "hint_given": False, "archetype_guess": None}

def evaluate_all(task_id: str, n_episodes: int = 10, seed: int = 42) -> dict:
    Cls = GRADER_REGISTRY[task_id]
    n   = min(n_episodes, 5) if task_id == "self_play_escalation" else n_episodes
    results = {}
    for name, AgentCls in [("Random", RandomAgent), ("Heuristic", HeuristicAgent),
                            ("Greedy", GreedyArchetypeAgent), ("Inference", ArchetypeInferenceAgent)]:
        print(f"  Grading {name:<12}...", end=" ", flush=True)
        r = Cls(n_episodes=n).grade(AgentCls())
        results[name] = r.score
        print(f"score={r.score:.4f} arch_acc={r.archetype_accuracy:.3f}")
    try:
        ppo = PPOAgent(task_id)
        print(f"  Grading PPO        ...", end=" ", flush=True)
        r = Cls(n_episodes=n).grade(ppo)
        results["PPO"] = r.score
        print(f"score={r.score:.4f} arch_acc={r.archetype_accuracy:.3f}")
    except FileNotFoundError as e:
        print(f"  PPO: {e}")
    return results

def print_comparison(task_id: str, results: dict):
    diff = TASK_REGISTRY[task_id]["difficulty"]
    icon = {"easy":"🟢","medium":"🟡","hard":"🔴","expert":"🔴🔴"}.get(diff,"")
    print(f"\n  {icon} {task_id} ({diff})\n  {'─'*48}")
    best = max(results.values())
    for name, score in results.items():
        bar = "█" * int(score * 25)
        tag = " ← 🏆" if score == best else ""
        print(f"  {name:<14}[{bar:<25}] {score:.4f}{tag}")
    if "PPO" in results:
        delta = results["PPO"] - max(v for k,v in results.items() if k != "PPO")
        print(f"  PPO vs best baseline: {'+' if delta>=0 else ''}{delta:.4f}")

def main():
    p = argparse.ArgumentParser(description="TeachRL v2 PPO Agent")
    p.add_argument("--train",    action="store_true")
    p.add_argument("--eval",     action="store_true")
    p.add_argument("--task",     default="all", choices=["all"] + list(TASK_REGISTRY))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   default=None)
    args = p.parse_args()
    tasks = list(TASK_REGISTRY) if args.task == "all" else [args.task]
    all_results = {}
    for task_id in tasks:
        if args.train: train_ppo(task_id, seed=args.seed)
        if args.eval:
            print(f"\n  Evaluating: {task_id}")
            r = evaluate_all(task_id, args.episodes, args.seed)
            all_results[task_id] = r
            print_comparison(task_id, r)
    if args.output and all_results:
        with open(args.output, "w") as f: json.dump(all_results, f, indent=2)
    if not args.train and not args.eval:
        p.print_help()

if __name__ == "__main__": main()
