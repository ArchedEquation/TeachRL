"""
inference.py — TeachRL v2 OpenEnv Inference Script.
Uses LLM (OpenAI-compatible) as the tutoring agent with archetype inference.
Emits required [START]/[STEP]/[END] stdout format.

Env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""
import os, json, sys
from typing import List, Optional
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "teachrl-v2"

CONCEPTS   = ["algebra_basics","linear_equations","quadratic_equations","functions",
              "trigonometry","probability","statistics","calculus_intro","geometry","number_theory"]
DIFFS      = ["easy","medium","hard"]
ARCHETYPES = ["overconfident_learner","anxious_perfectionist","adhd_sprinter","slow_steady_builder",
              "strategic_gamer","emotional_learner","uneven_genius","impostor"]

TASKS = [
    {"id":"archetype_identification","targets":CONCEPTS,"goal":"Identify student archetype in 20 steps","max_steps":20},
    {"id":"adaptive_curriculum","targets":["algebra_basics","linear_equations","functions","probability","geometry"],"goal":"Master 5 concepts in 50 steps","max_steps":50},
    {"id":"blind_teaching","targets":CONCEPTS,"goal":"Infer and teach unknown archetype in 80 steps","max_steps":80},
    {"id":"self_play_escalation","targets":CONCEPTS,"goal":"Teach auto-escalated student variants across 5x80 steps","max_steps":80},
]

def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)
def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

def build_system_prompt(task: dict) -> str:
    targets = ", ".join(task["targets"])
    return f"""You are an expert AI tutor. Task: {task['id']}
GOAL: {task['goal']}
TARGET CONCEPTS: {targets}
MAX STEPS: {task['max_steps']}

Observe student state and:
1. Select concept + difficulty to maximise mastery
2. Infer the hidden student archetype from signals
3. Use expert_hint but check hint_reliability before trusting

Archetype clues:
- overconfident_learner: high success but careless errors
- anxious_perfectionist: collapses on hard, good on easy
- adhd_sprinter: learns fast, forgets fast, bored by repetition
- slow_steady_builder: very slow growth, never forgets
- strategic_gamer: high easy success but mastery not growing
- emotional_learner: engagement cycles every ~10 steps
- uneven_genius: some concepts suspiciously easy, others impossible
- impostor: appears mastered but fails hard questions

Respond ONLY with JSON, no markdown:
{{"concept":"<n>","difficulty":"<easy|medium|hard>","hint_given":false,"archetype_guess":"<id or null>"}}"""

def build_user_prompt(obs: dict, step: int, task: dict) -> str:
    rates   = obs["concept_success_rates"]
    targets = set(task["targets"])
    rows    = "\n".join(
        f"  {'★' if c in targets else ' '} {c:<28} sr={rates.get(c,0):.2f} "
        f"att={obs['concept_attempt_counts'].get(c,0)} streak={obs['concept_streaks'].get(c,0)} "
        f"prereq={obs['prerequisite_readiness'].get(c,1):.2f}"
        for c in CONCEPTS)
    return (f"Step {step}/{task['max_steps']} | Eng:{obs['engagement']:.2f} Fat:{obs['fatigue']:.2f} | "
            f"Last:{obs['last_concept']}({obs['last_difficulty']})={'✓' if obs['last_correct'] else '✗'}\n"
            f"Hint(rel={obs.get('hint_reliability',0.7):.2f}): {obs.get('expert_hint','')}\n\n"
            f"Concepts (★=target):\n{rows}\n\nJSON only:")

def get_llm_action(client, obs: dict, step: int, task: dict) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":build_system_prompt(task)},
                      {"role":"user","content":build_user_prompt(obs,step,task)}],
            temperature=0.1, max_tokens=80, stream=False)
        text   = (completion.choices[0].message.content or "").strip().replace("```json","").replace("```","").strip()
        action = json.loads(text)
        assert action["concept"] in CONCEPTS
        assert action["difficulty"] in DIFFS
        if action.get("archetype_guess") not in ARCHETYPES + [None]: action["archetype_guess"] = None
        return action
    except Exception:
        rates   = obs["concept_success_rates"]
        prereqs = obs["prerequisite_readiness"]
        eligible = [c for c in task["targets"] if prereqs.get(c,1.0)>=0.35 and rates.get(c,0.0)<0.85]
        concept  = min(eligible or task["targets"], key=lambda c: rates.get(c,0.0))
        sr = rates.get(concept, 0.0)
        return {"concept": concept, "difficulty": "easy" if obs["fatigue"]>0.65 or sr<0.4 else "medium" if sr<0.7 else "hard",
                "hint_given": False, "archetype_guess": None}

def run_task(task: dict, client) -> dict:
    from env.environment import TeachRLEnv
    env = TeachRLEnv(task_id=task["id"], seed=42, eval_mode=True)
    rewards: List[float] = []; steps_taken = 0; score = 0.0; success = False
    log_start(task["id"], BENCHMARK, MODEL_NAME)
    try:
        obs = env.reset(seed=42); done = False
        for step in range(1, task["max_steps"]+1):
            if done: break
            action     = get_llm_action(client, obs.model_dump(), step, task)
            action_str = f"concept={action['concept']},difficulty={action['difficulty']},guess={action.get('archetype_guess')}"
            try:
                result = env.step(action); obs = result.observation
                reward = result.reward; done = result.done; err = None
            except Exception as e:
                reward = 0.0; done = True; err = str(e)
            rewards.append(reward); steps_taken = step
            log_step(step, action_str, reward, done, err)
        score = env._task_score(); success = score >= 0.5
    except Exception as e:
        print(f"[DEBUG] {e}", flush=True)
    finally:
        log_end(success, steps_taken, score, rewards)
    return {"task": task["id"], "score": score, "success": success, "steps": steps_taken}

def main():
    if not API_KEY:
        print("[ERROR] Set HF_TOKEN or OPENAI_API_KEY", flush=True); sys.exit(1)
    client   = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_env = os.getenv("TEACHRL_TASK", "all")
    tasks    = TASKS if task_env == "all" else [t for t in TASKS if t["id"] == task_env]
    results  = [run_task(t, client) for t in tasks]
    print("\n[SUMMARY]", flush=True)
    for r in results:
        print(f"  task={r['task']} score={r['score']:.3f} success={str(r['success']).lower()} steps={r['steps']}", flush=True)

if __name__ == "__main__": main()