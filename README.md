---
title: TeachRL-v2
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🎓 TeachRL-v2 — The AI Tutor That Learns How to Teach

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-green)](https://openenv.ai)
[![Theme](https://img.shields.io/badge/Theme-4%3A%20Self--Improvement-purple)](https://huggingface.co/spaces/ArchedEquation/TeachRL-v2)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

> **What if you had to teach 8 completely different students — without knowing which one you're facing?**

TeachRL is a real-world OpenEnv environment where an AI agent acts as an adaptive tutor for a student learning 10 high-school math concepts. The student's **hidden personality type (archetype)** determines how they respond to teaching. The agent must **infer the archetype** and **adapt its curriculum simultaneously** — while a self-play loop makes students progressively harder to teach.

---

## 🔗 Links

| Resource | URL |
|---|---|
| **HuggingFace Space (Live API)** | https://huggingface.co/spaces/ArchedEquation/TeachRL-v2 |
| **GitHub Repository** | https://github.com/ArchedEquation/TeachRL-v2 |
| **Training Notebook (Colab)** | *(link your Colab notebook here)* |
| **Mini-Blog / Writeup** | *(link your HF blog post or YouTube here)* |

---

## 🧠 The Problem

Most AI tutoring systems treat every student identically. But students are fundamentally different:

- One student gets **bored instantly** if questions repeat
- Another **panics under pressure** and makes errors they wouldn't otherwise make
- A third has **secretly not done the homework** and is faking prior knowledge

TeachRL forces an AI agent to solve two coupled problems simultaneously:
1. **Infer** which of 8 student archetypes it's teaching (hidden state)
2. **Teach** that student efficiently within a step budget

Neither problem is solvable without solving the other.

---

## 👥 The 8 Student Archetypes (Hidden)

Each archetype has a unique Bayesian Knowledge Tracing parameter signature that breaks a different RL assumption:

| # | Archetype | Key Trait | RL Challenge |
|---|---|---|---|
| 1 | **Overconfident Learner** | High p_guess (0.40) + p_slip (0.35) | Reward signal unreliable |
| 2 | **Anxious Perfectionist** | p_slip spikes to 0.40 on hard | Non-linear difficulty response |
| 3 | **ADHD Sprinter** | p_forget=0.20, bored by repetition | Stationarity broken |
| 4 | **Slow Steady Builder** | p_learn=0.08, p_forget=0.01 | Long-horizon patience required |
| 5 | **Strategic Gamer** | p_guess=0.55 on easy | Surface signals lie |
| 6 | **Emotional Learner** | BKT params shift every 10 steps | Non-stationary dynamics |
| 7 | **Uneven Genius** | 3 gifts + 3 blind spots (random) | Exploration required |
| 8 | **Impostor** | p_init=0.85, crumbles on hard | Trust calibration |

---

## 🏆 The 4 Tasks

| Task | Difficulty | Steps | Goal |
|---|---|---|---|
| `archetype_identification` | 🟢 Easy | 20 | Identify the hidden archetype |
| `adaptive_curriculum` | 🟡 Medium | 50 | Teach a revealed archetype to mastery |
| `blind_teaching` | 🔴 Hard | 80 | Infer + teach simultaneously |
| `self_play_escalation` | 🔴🔴 Expert | 5×80 | Face auto-escalated harder variants |

---

## 🔄 Self-Play Escalation (Theme 4: Self-Improvement)

The `self_play_escalation` task implements **recursive self-improvement**: when the agent consistently scores ≥0.70 on an archetype, the escalator makes that archetype harder:

```
Agent masters OverconfidentLearner (p_guess=0.40)
    → Escalator raises p_guess to 0.45
    → Agent must learn a more sophisticated strategy
    → If mastered again, p_guess raised to 0.50
    → This continues until training ends
```

Each archetype escalates independently — the environment and the agent co-evolve.

---

## 📊 Results

### Training Curves

![Reward Curves](training_plots/reward_curves.png)

*PPO shows three distinct learning phases on the Hard task: basic signals → prerequisite ordering → archetype inference*

### Agent Comparison

![Agent Comparison](training_plots/agent_comparison.png)

*PPO beats all hand-coded baselines across every task after full training*

### Self-Play Escalation

![Self-Play Escalation](training_plots/self_play_escalation.png)

*ADHD Sprinter reaches Generation 4 (hardest variant) — the agent mastered it most quickly*

### Archetype Identification Accuracy

![Archetype Accuracy](training_plots/archetype_accuracy.png)

*PPO learns to identify archetypes with ~74% accuracy — far above random chance (12.5%)*

### Baseline Scores (seed=42, 10 episodes)

| Agent | Easy (ID) | Medium | Hard | Expert |
|---|---|---|---|---|
| Random | 0.001 | 0.619 | 0.586 | 0.613 |
| Heuristic | 0.001 | 0.783 | 0.689 | 0.658 |
| Greedy | 0.001 | 0.794 | 0.676 | 0.614 |
| Inference Agent | 0.300 | 0.794 | 0.748 | 0.663 |
| **PPO (trained)** | **0.520** | **0.965** | **0.820** | **0.730** |

---

## 🚀 Quickstart

```bash
git clone https://github.com/ArchedEquation/TeachRL-v2
cd TeachRL-v2
pip install -r requirements.txt
```

### Run one episode

```python
from env.environment import TeachRLEnv, TutorAction

env = TeachRLEnv(task_id="blind_teaching", seed=42, eval_mode=True)
obs = env.reset()

done = False
while not done:
    action = TutorAction(concept="algebra_basics", difficulty="medium")
    result = env.step(action)
    obs    = result.observation
    done   = result.done

print(env.render())
```

### Run baseline evaluation

```bash
python baseline/baseline_inference.py --episodes 10 --seed 42
```

### Train PPO

```bash
python baseline/rl_agent.py --train --task blind_teaching
python baseline/rl_agent.py --train --task self_play_escalation
python baseline/rl_agent.py --eval  --task all
```

### Run via HTTP API

```bash
python app.py  # starts on port 7860
```

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "blind_teaching", "seed": 42}'
```

---

## 💰 Reward Function

Dense reward at every step across 9 components:

| Component | Weight | Purpose |
|---|---|---|
| Mastery gain (unmastered) | ×3.0 | Primary learning signal |
| Mastery gain (mastered) | ×0.5 | Diminishing returns |
| Terminal bonus | +0.5× score | Long-horizon credit assignment |
| Archetype guess (correct) | +0.20 | Rewards identification |
| Archetype guess (wrong) | −0.05 | Penalises overconfidence |
| Engagement | +0.08 | Keep student engaged |
| ZPD alignment | +0.08 | Zone of Proximal Development |
| Prerequisite alignment | +0.05 | Respect dependency graph |
| Overdrill penalty | −0.08 | Stop wasting steps |
| Fatigue penalty | −0.04 | Manage student energy |

---

## 📁 Project Structure

```
TeachRL-v2/
├── env/
│   ├── archetypes.py        # 8 student archetypes (BKT params)
│   ├── student.py           # BKT student simulator
│   ├── environment.py       # OpenEnv API: reset/step/state
│   └── gym_wrapper.py       # Gymnasium wrapper for PPO
├── self_play/
│   └── escalator.py         # Self-play difficulty escalation
├── graders/
│   └── grader.py            # 4 task graders (0.001–0.999)
├── baseline/
│   ├── agents.py            # 4 heuristic baseline agents
│   ├── baseline_inference.py
│   └── rl_agent.py          # PPO train + eval
├── training_plots/          # Required: reward + loss curves
│   ├── reward_curves.png
│   ├── agent_comparison.png
│   ├── self_play_escalation.png
│   └── archetype_accuracy.png
├── tests/
│   └── test_env.py          # 28 tests (all passing)
├── inference.py             # OpenAI client inference script
├── app.py                   # FastAPI server
├── openenv.yaml             # OpenEnv spec
├── pyproject.toml           # openenv validate compliance
├── Dockerfile
└── README.md
```

---

## 🐳 Docker

```bash
docker build -t teachrl-v2 .
docker run -p 7860:7860 teachrl-v2
# Swagger UI: http://localhost:7860/docs
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health + env info |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take one action |
| GET | `/state?session_id=` | Episode snapshot |
| GET | `/render?session_id=` | Text render |
| GET | `/archetypes` | List all archetypes |
| GET | `/tasks` | List all tasks |
| GET | `/docs` | Swagger UI |

---

## 📜 License

MIT © VIT-AP University | Meta PyTorch Hackathon x Scaler 2025
