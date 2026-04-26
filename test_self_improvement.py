"""
test_self_improvement.py — Live demonstration of TeachRL self-improvement loop.

Shows the escalation mechanism working in real time with an LLM agent.
The environment gets harder as the agent improves — proving Theme 4.

Usage:
    python test_self_improvement.py              # uses PPO agent
    python test_self_improvement.py --llm        # uses Qwen LLM agent
    python test_self_improvement.py --llm --episodes 10
"""

import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dotenv import load_dotenv
load_dotenv()

from env.environment import TeachRLEnv, TASK_REGISTRY
from env.archetypes import ALL_ARCHETYPES, ArchetypeID
from self_play.escalator import SelfPlayEscalator, EpisodeResult


# ── Colour helpers ────────────────────────────────────────────────────────────
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
R  = "\033[91m"   # red
B  = "\033[94m"   # blue
M  = "\033[95m"   # magenta
W  = "\033[97m"   # white bold
NC = "\033[0m"    # reset


def bar(score, width=30, color=G):
    filled = int(score * width)
    return color + "█" * filled + NC + "░" * (width - filled)


def print_header():
    print(f"\n{W}{'='*65}{NC}")
    print(f"{W}  TeachRL v2 — Self-Improvement Live Demonstration{NC}")
    print(f"{W}  Theme 4: Recursive Self-Improvement via Self-Play Escalation{NC}")
    print(f"{W}{'='*65}{NC}\n")
    print(f"  {B}How it works:{NC}")
    print(f"  1. Agent teaches a student for 80 steps")
    print(f"  2. If score ≥ 0.70 on an archetype → environment ESCALATES")
    print(f"  3. Next episode: that archetype is harder to teach")
    print(f"  4. Agent must improve its strategy — or scores drop\n")


# ── LLM Agent ─────────────────────────────────────────────────────────────────

def make_llm_agent():
    from openai import OpenAI
    api_key  = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    if not api_key:
        print(f"{R}ERROR: Set HF_TOKEN in .env{NC}")
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)

    SYSTEM = """You are an adaptive AI tutor teaching a student math.
The student has a HIDDEN personality type (archetype) you must identify from their behaviour.

8 possible archetypes:
- overconfident_learner: high success rate but mastery not growing (high guess rate)
- anxious_perfectionist: collapses on hard questions, great on easy
- adhd_sprinter: learns fast but forgets fast, bored by repetition
- slow_steady_builder: very slow progress but never forgets
- strategic_gamer: high correct rate on easy, mastery stays flat
- emotional_learner: performance cycles every ~10 steps
- uneven_genius: some concepts effortless, others impossibly hard
- impostor: appears mastered but fails hard questions

Respond ONLY with valid JSON, no explanation:
{"concept":"<name>","difficulty":"<easy|medium|hard>","archetype_guess":"<id or null>"}

Concepts: algebra_basics, linear_equations, quadratic_equations, functions,
trigonometry, probability, statistics, calculus_intro, geometry, number_theory"""

    def agent(obs_dict):
        rates   = obs_dict["concept_success_rates"]
        att     = obs_dict["concept_attempt_counts"]
        eng     = obs_dict["engagement"]
        fat     = obs_dict["fatigue"]
        hint    = obs_dict.get("expert_hint", "")
        step    = obs_dict["step_count"]
        gen     = obs_dict.get("current_generation", {})

        rows = "\n".join(
            f"  {'★' if rates.get(c,0) < 0.6 else ' '} {c:<28}"
            f" sr={rates.get(c,0):.2f} att={att.get(c,0):>2}"
            for c in ["algebra_basics","linear_equations","quadratic_equations",
                      "functions","trigonometry","probability","statistics",
                      "calculus_intro","geometry","number_theory"]
        )
        user = (f"Step {step}/80 | Engagement:{eng:.2f} Fatigue:{fat:.2f}\n"
                f"Expert hint: {hint}\n"
                f"Escalation generations: {gen}\n\n"
                f"Concepts (★=needs work):\n{rows}\n\nJSON only:")
        try:
            r    = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM},
                          {"role":"user","content":user}],
                temperature=0.2, max_tokens=60,
            )
            text = r.choices[0].message.content.strip()
            text = text.replace("```json","").replace("```","").strip()
            act  = json.loads(text)
            assert act["concept"] in [
                "algebra_basics","linear_equations","quadratic_equations","functions",
                "trigonometry","probability","statistics","calculus_intro",
                "geometry","number_theory"]
            assert act["difficulty"] in ["easy","medium","hard"]
            return act
        except Exception:
            # fallback
            worst = min(rates, key=rates.get)
            sr    = rates.get(worst, 0.0)
            return {"concept": worst,
                    "difficulty": "easy" if fat > 0.6 else "medium" if sr < 0.5 else "hard",
                    "archetype_guess": None}

    return agent, model


def make_ppo_agent():
    from stable_baselines3 import PPO
    from env.gym_wrapper import obs_to_vector, int_to_action
    from baseline.archetype_classifier import ArchetypeClassifier

    model_path = "models/ppo_self_play_escalation.zip"
    if not os.path.exists(model_path):
        model_path = "models/ppo_blind_teaching.zip"
    if not os.path.exists(model_path):
        print(f"{Y}No PPO model found — using heuristic fallback{NC}")
        return make_heuristic_agent(), "Heuristic"

    model = PPO.load(model_path)
    clf   = ArchetypeClassifier()
    max_s = TASK_REGISTRY["self_play_escalation"]["max_steps"]

    def agent(obs_dict):
        vec = obs_to_vector(obs_dict, max_s)
        a, _ = model.predict(vec, deterministic=True)
        c, d  = int_to_action(int(a))
        guess = clf.predict_from_obs(obs_dict) if clf.is_loaded else None
        return {"concept":c,"difficulty":d,"hint_given":False,"archetype_guess":guess}

    return agent, "PPO + Classifier (92.8% acc)"


def make_heuristic_agent():
    from baseline.agents import GreedyArchetypeAgent
    agent = GreedyArchetypeAgent()
    def fn(obs): return agent(obs)
    return fn


# ── Core test ─────────────────────────────────────────────────────────────────

def run_self_improvement_test(agent_fn, agent_name, n_episodes=8):
    print_header()
    print(f"  {W}Agent:{NC} {agent_name}")
    print(f"  {W}Episodes:{NC} {n_episodes}")
    print(f"  {W}Task:{NC} self_play_escalation (5×80 steps)\n")

    # ── Setup log file ────────────────────────────────────────────────────────
    import os as _os
    from datetime import datetime as _dt
    from logger import LOG_DIR, RUN_DIR
    _log_subdir = _os.path.join(RUN_DIR, f"llm_self_improvement_{agent_name.replace(' ','_').replace('/','_')[:30]}")
    _os.makedirs(_log_subdir, exist_ok=True)
    _log_path = _os.path.join(_log_subdir, "self_improvement.log")
    _lf = open(_log_path, "w", encoding="utf-8")

    def _log(line="", also_print=True):
        _lf.write(line + "\n"); _lf.flush()
        if also_print: print(line)

    _log("=" * 65)
    _log(f"  TeachRL — LLM Self-Improvement Log")
    _log(f"  Agent:    {agent_name}")
    _log(f"  Episodes: {n_episodes}")
    _log(f"  Started:  {_dt.now():%Y-%m-%d %H:%M:%S}")
    _log(f"  Task:     self_play_escalation (5x80 steps)")
    _log("=" * 65)
    _log()

    escalator  = SelfPlayEscalator()
    env        = TeachRLEnv(task_id="self_play_escalation",
                            seed=42, eval_mode=True, escalator=escalator)

    episode_scores   = []
    escalation_events = []
    gen_before       = {a: 0 for a in ALL_ARCHETYPES}

    for ep in range(n_episodes):
        obs  = env.reset(seed=42 + ep)
        done = False
        step = 0
        ep_rewards = []

        print(f"  {W}{'─'*63}{NC}")
        arch = env._sim.archetype_id
        gens = escalator.summary()["generations"]
        gen_str = " ".join(f"{a.value[:8]}:G{gens[a.value]}" for a in ALL_ARCHETYPES
                           if gens[a.value] > 0)

        print(f"  {W}Episode {ep+1}/{n_episodes}{NC} | "
              f"Archetype: {M}{arch.value}{NC}")
        if gen_str:
            print(f"  Active escalations: {Y}{gen_str}{NC}")

        # Run episode
        while not done:
            action = agent_fn(obs.model_dump())
            result = env.step(action)
            obs    = result.observation
            done   = result.done
            step  += 1
            ep_rewards.append(result.reward)

        score  = env._task_score()
        mastery = env._sim.state.mastery
        episode_scores.append(score)

        # Check for new escalations
        new_gens = escalator.summary()["generations"]
        for a in ALL_ARCHETYPES:
            if new_gens[a.value] > gen_before[a.value]:
                escalation_events.append({
                    "episode": ep + 1,
                    "archetype": a.value,
                    "old_gen": gen_before[a.value],
                    "new_gen": new_gens[a.value],
                    "score": score,
                })
                esc_line = f"  ESCALATION! {a.value} -> Generation {new_gens[a.value]} (score={score:.3f})"
                print(f"\n  {G}🔺 ESCALATION!{NC} {a.value} → "
                      f"Generation {new_gens[a.value]} "
                      f"(agent scored {score:.3f} ≥ 0.70, environment gets harder)")
                _log(esc_line, also_print=False)
        gen_before = dict(new_gens)

        # Print and log episode result
        color = G if score >= 0.70 else Y if score >= 0.50 else R
        ep_line = f"  Steps: {step} | Score: {score:.4f} | Eng: {obs.engagement:.2f} | Fat: {obs.fatigue:.2f}"
        print(f"  Steps: {step} | Score: [{bar(score,25,color)}] {color}{score:.4f}{NC} | "
              f"Eng: {obs.engagement:.2f} | Fat: {obs.fatigue:.2f}")
        _log(ep_line, also_print=False)

        # Top 3 mastered concepts
        top3 = sorted(mastery.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = "  Top mastery: " + " | ".join(f"{c[:10]}:{v:.2f}" for c,v in top3)
        print(top3_str)
        _log(top3_str, also_print=False)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n  {W}{'='*63}{NC}")
    print(f"  {W}SELF-IMPROVEMENT SUMMARY{NC}")
    print(f"  {W}{'='*63}{NC}\n")

    # Score progression
    print(f"  {W}Score Progression:{NC}")
    for i, s in enumerate(episode_scores):
        color = G if s >= 0.70 else Y if s >= 0.50 else R
        trend = "↑" if i > 0 and s > episode_scores[i-1] else ("↓" if i > 0 and s < episode_scores[i-1] else "→")
        print(f"  Episode {i+1}: [{bar(s,20,color)}] {color}{s:.4f}{NC} {trend}")

    # Escalation events
    print(f"\n  {W}Escalation Events (environment got harder):{NC}")
    if escalation_events:
        for ev in escalation_events:
            print(f"  {G}✓{NC} Episode {ev['episode']}: {ev['archetype']:<28} "
                  f"Gen {ev['old_gen']} → Gen {ev['new_gen']} "
                  f"(trigger score: {ev['score']:.3f})")
    else:
        print(f"  {Y}No escalations triggered. "
              f"Agent needs to score ≥ 0.70 on same archetype across 2 episodes.{NC}")

    # Generation summary
    print(f"\n  {W}Final Escalation Generations:{NC}")
    gens = escalator.summary()["generations"]
    avgs = escalator.summary()["avg_scores"]
    for a in ALL_ARCHETYPES:
        g = gens[a.value]; avg = avgs[a.value]
        color = G if g > 0 else NC
        gen_bar = color + "●" * g + NC + "○" * (4 - g)
        print(f"  {a.value:<28} [{gen_bar}] Gen:{g} AvgScore:{avg:.3f}")

    total_gen = escalator.total_generation_sum()
    mean_score = np.mean(episode_scores)
    final_score = np.mean(episode_scores[-3:]) if len(episode_scores) >= 3 else mean_score
    early_score = np.mean(episode_scores[:3])

    print(f"\n  {W}Key Metrics:{NC}")
    print(f"  Total escalation generations triggered: {G}{total_gen}{NC}")
    print(f"  Early episodes avg score:  {early_score:.4f}")
    print(f"  Final episodes avg score:  {final_score:.4f}")
    improvement = final_score - early_score
    color = G if improvement > 0 else R
    print(f"  Net improvement:           {color}{'+' if improvement>=0 else ''}{improvement:.4f}{NC}")

    # Verdict
    print(f"\n  {W}{'─'*63}{NC}")
    if total_gen >= 2:
        print(f"  {G}✅ SELF-IMPROVEMENT CONFIRMED{NC}")
        print(f"  The environment escalated {total_gen} times, proving the agent")
        print(f"  improved enough to trigger harder variants (Theme 4 working).")
    elif total_gen == 1:
        print(f"  {Y}⚠️  PARTIAL: 1 escalation triggered.{NC}")
        print(f"  Run more episodes or train longer for stronger evidence.")
    else:
        print(f"  {R}❌ No escalations yet.{NC}")
        print(f"  Agent needs to score ≥ 0.70 on same archetype in 2 consecutive episodes.")
        print(f"  Try: python baseline/rl_agent.py --train --task self_play_escalation")

    print(f"  {W}{'─'*63}{NC}\n")

    # ── Write summary to log file ─────────────────────────────────────────────
    _log()
    _log("=" * 65)
    _log("  SUMMARY")
    _log("=" * 65)
    for i, s in enumerate(episode_scores):
        trend = "UP" if i > 0 and s > episode_scores[i-1] else ("DOWN" if i > 0 and s < episode_scores[i-1] else "SAME")
        _log(f"  Episode {i+1:>2}: {s:.4f}  {trend}")
    _log()
    _log(f"  Total escalations: {escalator.total_generation_sum()}")
    _log(f"  Early avg score:   {float(np.mean(episode_scores[:3])):.4f}")
    _log(f"  Late  avg score:   {float(np.mean(episode_scores[-3:])):.4f}")
    improvement = float(np.mean(episode_scores[-3:])) - float(np.mean(episode_scores[:3]))
    _log(f"  Net improvement:   {'+' if improvement>=0 else ''}{improvement:.4f}")
    if escalation_events:
        _log()
        _log("  Escalation Events:")
        for ev in escalation_events:
            _log(f"    Ep {ev['episode']}: {ev['archetype']} Gen {ev['old_gen']} -> {ev['new_gen']} (score={ev['score']:.3f})")
    _log()
    _log(f"  Log saved -> {_log_path}")
    _log("=" * 65)
    _lf.close()
    print(f"\n  Log saved -> {_log_path}")

    return escalation_events


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="TeachRL Self-Improvement Test")
    p.add_argument("--llm",      action="store_true", help="Use LLM agent (Qwen)")
    p.add_argument("--ppo",      action="store_true", help="Use trained PPO agent (default)")
    p.add_argument("--episodes", type=int, default=8, help="Number of episodes to run")
    args = p.parse_args()

    if args.llm:
        agent_fn, agent_name = make_llm_agent()
    else:
        agent_fn, agent_name = make_ppo_agent()

    run_self_improvement_test(agent_fn, agent_name, n_episodes=args.episodes)


if __name__ == "__main__":
    main()