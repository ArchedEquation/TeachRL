"""
baseline/baseline_inference.py — Reproducible baseline evaluation for TeachRL v2.
Usage: python baseline/baseline_inference.py --episodes 10 --seed 42
"""
import argparse, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
from env.environment import TASK_REGISTRY
from graders.grader import GRADER_REGISTRY
from baseline.agents import RandomAgent, HeuristicAgent, GreedyArchetypeAgent, ArchetypeInferenceAgent

AGENTS = {"random": RandomAgent, "heuristic": HeuristicAgent,
          "greedy": GreedyArchetypeAgent, "inference": ArchetypeInferenceAgent}

def main():
    p = argparse.ArgumentParser(description="TeachRL v2 Baseline Evaluation")
    p.add_argument("--agent",    default="all", choices=["all"] + list(AGENTS))
    p.add_argument("--task",     default="all", choices=["all"] + list(TASK_REGISTRY))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   default=None)
    args = p.parse_args()

    agents = list(AGENTS)        if args.agent == "all" else [args.agent]
    tasks  = list(TASK_REGISTRY) if args.task  == "all" else [args.task]

    print(f"\n  TeachRL v2 — Baseline Evaluation")
    print(f"  Agents:{agents}  Tasks:{tasks}  Episodes:{args.episodes}\n")

    all_results = []; t0 = time.time()
    for task_id in tasks:
        for agent_name in agents:
            agent = AGENTS[agent_name]()
            n     = min(args.episodes, 5) if task_id == "self_play_escalation" else args.episodes
            Cls   = GRADER_REGISTRY[task_id]
            print(f"  {agent_name:<18} {task_id}...", end=" ", flush=True)
            r = Cls(n_episodes=n).grade(agent)
            print(f"score={r.score:.3f} ±{r.std_task_score:.3f} arch_acc={r.archetype_accuracy:.3f}")
            all_results.append({"agent": agent_name, "task": task_id, "score": r.score,
                                 "std": r.std_task_score, "arch_acc": r.archetype_accuracy,
                                 "steps": r.mean_steps})

    print(f"\n{'═'*80}")
    print(f"  {'Agent':<18}{'Task':<32}{'Diff':<10}{'Score':>7}{'ArchAcc':>9}")
    print(f"{'─'*80}")
    for r in all_results:
        diff = TASK_REGISTRY[r['task']]['difficulty']
        icon = {"easy":"🟢","medium":"🟡","hard":"🔴","expert":"🔴🔴"}.get(diff,"")
        print(f"  {r['agent']:<18}{r['task']:<32}{icon}{diff:<6}{r['score']:>7.3f}{r['arch_acc']:>9.3f}")
    print(f"{'═'*80}\n  Total time: {time.time()-t0:.1f}s")

    if args.output:
        with open(args.output, "w") as f: json.dump(all_results, f, indent=2)
        print(f"  Saved → {args.output}")

if __name__ == "__main__": main()
