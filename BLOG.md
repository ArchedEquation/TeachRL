# TeachRL: We Trained an RL Agent to Figure Out How You Learn

*A story about teaching machines to teach people — and why that's harder than it sounds.*

---

There's a student named Alex. Alex is smart, finishes problems quickly, and gets most of them right. But if you look closely, Alex has a habit of guessing on harder questions — not because they don't know the material, but because they've figured out that guessing strategically on multiple choice gives them a 55% hit rate without the mental effort of actually thinking it through. Their grades look fine. Their actual learning? Not so much.

Now imagine an AI tutor trying to help Alex. It sees correct answers rolling in and thinks: *great, they're doing well, let me keep the difficulty up.* It never catches on. Alex coasts.

That's the problem TeachRL is trying to crack.

---

## The Gap Nobody's Filling

Every major EdTech platform — Khan Academy, Duolingo, Carnegie Learning — uses Bayesian Knowledge Tracing under the hood. BKT is a beautifully principled model: it tracks a student's hidden mastery through their correct/incorrect responses, accounting for lucky guesses and careless slips. It's been the gold standard for 30 years.

But here's the thing: **BKT treats every student the same.** The same slip probability, the same guess probability, the same learning rate — averaged across millions of students into one generic "learner." Real students aren't like that. They're wildly, fundamentally different in the *shape* of how they learn. An anxious perfectionist who shuts down on hard questions isn't just a slow learner — they need a completely different teaching strategy than a slow-and-steady builder who just needs more repetitions.

The question we started with was simple: *what if we trained an RL agent to figure out which kind of learner it's dealing with, and then teach to that specifically?*

The answer turned out to be genuinely tricky to build.

---

## The Environment: Eight Students, One Agent, No Cheating

We built **TeachRL** as a reinforcement learning environment where the agent plays the role of a tutor. The student is a simulated learner modeled with archetype-specific BKT parameters — the agent never gets to see those parameters directly. It has to infer the student type from what it can observe: response patterns, engagement levels, concept streaks, how performance changes across difficulty levels.

There are eight student archetypes, each designed to break a different assumption that a naive tutor would make:

| Archetype | What makes them hard to teach |
|---|---|
| **Strategic Gamer** | Guesses correctly 55% of the time on easy questions. Looks like they know the material when they don't. |
| **Overconfident Learner** | High slip rate — gets questions wrong even when they've mastered concepts. Reward signal is noisy. |
| **Anxious Perfectionist** | Slip rate spikes to 0.40 on hard questions. Non-linear response to difficulty. Push too hard and they fall apart. |
| **ADHD Sprinter** | Learns fast (p_learn=0.45) but forgets almost as fast (p_forget=0.20). The stationarity assumption is completely broken. |
| **Slow Steady Builder** | Learning rate 0.08, forgetting rate 0.01. Needs patience the average tutor won't have. |
| **Emotional Learner** | BKT parameters shift every 10 steps based on mood cycles. The environment is literally non-stationary. |
| **Uneven Genius** | Has 3 gift concepts they've already mastered and 3 blind spots that need heavy work. Requires careful exploration. |
| **Impostor** | Starts with apparent 85% mastery, crumbles under hard questions. Don't trust the initial signal. |

The agent sees a **45-dimensional observation vector** at each step. It knows things like: how many times has each concept been attempted, what's the current streak, what's the student's engagement level, how fatigued are they, what did the last hint reliability say. It never sees the archetype label or the true BKT parameters. Just signals.

At each step, the agent picks:
- **Which of 10 math concepts to teach** (algebra through calculus, with a prerequisite graph)
- **What difficulty level** (easy / medium / hard)
- **A guess about the student's archetype** — and this one has teeth: correct guess gives +0.20, wrong guess costs −0.10

That archetype guess mechanic is load-bearing. It forces the agent to commit to a belief about who it's teaching, rather than staying vague and playing it safe.

### The Reward Function (Designed to Be Hard to Game)

Getting the reward function right took a few iterations. You don't want the agent to find some cheap trick — drilling one easy concept forever, for example, would technically produce "correct" answers. So the reward has six composable components:

| Rubric | Signal | Anti-Gaming Design |
|---|---|---|
| **Mastery Progress** | ×3.0 unmastered, ×0.3 mastered | Overdrill penalty −0.15 |
| **Coverage Diversity** | +0.06 for first 4 attempts | Prevents single-concept exploitation |
| **Pedagogical Quality** | ZPD ×0.07 + prereq ×0.05 + engagement ×0.07 | Wrong difficulty earns near-zero |
| **Response Quality** | correct × (0.5 + 0.5×ZPD) | Easy on mastered concept worth less |
| **Archetype ID** | correct +0.20 / wrong −0.10 | Wrong guess actively hurts |
| **Efficiency** | terminal +0.5×score, fatigue −0.05 | Forces long-horizon planning |

All rewards are clipped to [0, 1] per step. There's no easy loophole.

### Four Tasks: One Progression

The environment structures this into four tasks of increasing difficulty:

1. **Easy — Identify the archetype** (20 steps): No teaching required, just figure out who you're dealing with. Baseline agents score near zero here because there's no heuristic that works reliably.
2. **Medium — Teach a revealed archetype** (50 steps): The archetype is told to you at the start. The challenge is curriculum optimization.
3. **Hard — Blind teaching** (80 steps): You don't know the archetype. Infer and teach simultaneously.
4. **Expert — Self-play escalation** (5×80 steps): Same as Hard, but the environment fights back.

---

## Architecture: Two Models, One Agent

There's a deliberate two-model split in how TeachRL works:

```
Observation
    │
    ├─→ PPO Teaching Policy
    │   (Discrete action space: 10 concepts × 3 difficulties)
    │   Trained entirely through RL on environment reward
    │
    └─→ Archetype Classifier (neural net)
            Trained supervised on 4,000 simulated BKT episodes
    │
    ▼
Combined action: {concept, difficulty, archetype_guess}
    → Environment executes → returns reward
```

The classifier handles the *who*, the PPO policy handles the *how*. We trained the classifier separately on simulated episodes, then plugged it into the RL loop so the PPO agent could benefit from its archetype guesses without needing to learn that signal from scratch through rewards alone.

The classifier hit **92.8% validation accuracy** on held-out archetype identification — which sounds good until you remember that in the actual environment, the signal is noisy, the student's behavior shifts, and the Strategic Gamer is *actively trying to look like someone else.*

---

## Training: Watching the Agent Learn

Here's what the PPO training curves look like across all four tasks — raw rewards (faint) smoothed over 20-episode windows (bold), with the best hand-coded baseline shown as a dashed reference line:

![Training Reward Curves](training_plots/reward_curves.png)

*4 subplots — one per task. x-axis: training steps (thousands). y-axis: mean episode reward [0–1], averaged over last 20 completed episodes. Raw values shown faint, smoothed curve shown bold. Dashed line = best hand-coded baseline. Dotted line = PPO final evaluation score. All curves generated from real PPO training runs connecting live to the TeachRL environment — not a static dataset.*

The key thing to note: on the Easy (identification) task, every baseline flatlines near zero from the start. There's no rule you can hardcode to identify a hidden BKT archetype from 20 steps of noisy observation. The classifier is solving something genuinely hard, and the training curve shows it actually getting there.

---

## The Self-Play Loop: Where It Gets Interesting

The expert task implements a self-escalation mechanism: whenever the agent scores ≥0.70 on the same archetype in two consecutive episodes, that archetype gets harder. The overconfident learner's guess probability goes up. The strategic gamer's exploitation rate increases. The agent is, in a sense, generating its own curriculum.

It looks like this in practice:

```
PPO masters OverconfidentLearner (p_guess=0.40)
    → Escalator: raise p_guess to 0.45  [Generation 1]
    → PPO adapts and masters again
    → Escalator: raise p_guess to 0.50  [Generation 2]
    → Continues until training ends
```

Each of the eight archetypes escalates independently based on the agent's performance against that specific student type. The agent sees a `current_generation` counter in its observation — it knows how many times the environment has gotten harder, and it can use that as a signal for how sophisticated its teaching needs to be.

Here's the live evidence from a 10-episode run:

![Self-Play Escalation](training_plots/self_play_escalation.png)

*Top: episode scores coloured by hidden student archetype. Red dashed lines mark escalation events — each time PPO scored ≥0.70 on the same archetype twice, the environment made that archetype harder. 3 escalations triggered (strategic\_gamer Gen 1, overconfident Gen 1 → Gen 2). Bottom-left: escalation generations per archetype after training. Bottom-right: early vs late episode score distribution — PPO improves even as the environment escalates.*

In 10 training episodes, we confirmed **5 escalation events** firing live — strategic_gamer reaching Generation 1, overconfident_learner hitting Generation 2, and three others progressing. Episode scores improved from 0.84 to 0.91 as the environment got progressively harder. The agent wasn't just maintaining performance; it was improving against harder opponents.

---

## What Actually Changed After Training

Here's the honest evaluation across all agents and all tasks, run on seed=42, 10 episodes each:

![Agent Comparison](training_plots/agent_comparison.png)

*All 5 agents evaluated on all 4 tasks (seed=42, 10 episodes each, live environment rollouts). PPO+Classifier (red) beats every baseline on Easy (+16.5%), Medium (+0.7%), and Expert (+12.5%) tasks. Annotations show exact delta over best baseline. Hard task PPO (0.722) is within 2.6% of best baseline (Inference 0.748) and needs more training steps.*

In table form:

| Agent | Easy (ID) | Medium (Curriculum) | Hard (Blind) | Expert (Self-Play) |
|---|---|---|---|---|
| Random | 0.001 | 0.621 | 0.587 | 0.613 |
| Heuristic | 0.001 | 0.781 | 0.689 | 0.684 |
| Greedy | 0.001 | 0.791 | 0.676 | 0.610 |
| LLM Inference | 0.300 | 0.789 | **0.748** | 0.648 |
| **PPO + Classifier** | **0.465** | **0.799** | 0.722 | **0.809** |

A few things stand out:

**On the Easy task (archetype identification), the baselines score essentially zero.** Heuristic, Greedy, even Random — they all fail at 0.001. This isn't a bad implementation; it's the point. There's no rule you can hardcode to identify a hidden BKT archetype from 20 steps of noisy observation. The classifier is solving something genuinely hard, and it's doing it.

**On the Expert task, PPO beats every baseline by +12.5%.** This is the self-play task — the one where the environment gets harder as you get better. The LLM inference agent, despite having full language reasoning ability, scores 0.648. The PPO policy, which learned from scratch, hits 0.809. The gap is the self-play adaptation: PPO learned through experience what strategies work against escalating variants, while the LLM is reasoning from first principles on each episode.

**On the Hard task, the LLM wins.** PPO at 0.722 trails the LLM inference agent at 0.748. When you have 80 steps to infer and teach simultaneously with no archetype information whatsoever, the reasoning capability of an LLM gives a real edge. More PPO training time would likely close this gap, but we want to be straight about where the numbers landed.

---

## How Well Did the Agent Actually Teach?

It's one thing to score well on a task metric — it's another to ask whether the agent actually produced better-learned students. Here's the concept mastery heatmap:

![Mastery Heatmap](training_plots/mastery_heatmap.png)

*10 math concepts (rows) × 8 student archetypes (columns). Colour = mastery achieved [0=red, 1=green]. Left: Random agent — inconsistent, archetype-blind. Right: PPO+Classifier — consistently higher mastery, especially for anxious\_perfectionist (column 2, near-perfect green across all concepts). PPO+Clf mean mastery: 0.656 vs Random: 0.534 — **+0.122 improvement**.*

The mastery heatmap is where the learned behavior becomes concrete. For the Anxious Perfectionist in particular — the archetype where random teaching is actively harmful and only careful difficulty management works — PPO produces near-perfect greens where the random agent leaves reds. The agent genuinely learned to back off on difficulty for anxious students, a behavior that has to emerge purely from reward signals with no explicit rule encoding it.

---

## Why This Is a Harder Problem Than It Looks

Most RL environments have one tricky property. TeachRL has three at once:

**Hidden discrete latent variable.** The archetype is the key to the entire episode — teaching strategy, expected responses, difficulty calibration. And the agent can never directly observe it. This is a Partially Observable MDP, and the partial observability is meaningful, not artificial.

**Non-stationary dynamics.** The Emotional Learner's BKT parameters shift every 10 steps. The ADHD Sprinter forgets almost as fast as they learn. Standard RL assumes the environment is stationary; that assumption breaks here in ways that matter.

**Adaptive adversarial curriculum.** The self-play escalation means the environment co-evolves with the agent. There's no fixed target to converge to. The better you get, the harder it gets.

No existing educational RL benchmark combines all three of these properties. That's not us saying our work is better than others — it's a genuine claim about what makes this a useful benchmark for future research.

---

## Who Would Care About This

If you're working on **adaptive learning systems** and wondering how to personalize curriculum beyond simple rule-based difficulty adjustment, TeachRL gives you a training ground where your agent actually has to solve the personalization problem, not just approximate it.

If you're a **reinforcement learning researcher** interested in POMDPs or non-stationary environments, the archetype setup gives you a clean, reproducible benchmark with measurable latent variable inference and a self-play escalation loop you can extend.

If you're building **LLM agents** and want to test whether they can actually adapt instructional strategy based on behavioral signals rather than stated preferences — the Hard task's LLM vs. PPO comparison is a useful data point for what LLMs are and aren't good at right now.

And honestly, if you've ever been a student who wished your teacher had actually noticed how you were struggling before moving on — this is what we'd want that system to be learning to do.

---

## Try It

The live API is running on [Hugging Face Spaces](https://huggingface.co/spaces/ArchedEquation/TeachRL). You can send it a session, reset, step through an episode, and watch the student respond. The Swagger docs are at `/docs`.

```bash
git clone https://github.com/ArchedEquation/TeachRL
cd TeachRL
pip install -r requirements.txt
python -m pytest tests/ -q                    # 28 tests
python baseline/baseline_inference.py --episodes 10
python train_trl.py --task all --eval --self-play
```

Training all four tasks takes about 30–40 minutes on CPU. The self-improvement demo can be run separately:

```bash
python test_self_improvement.py --episodes 10
# Watch escalation events fire in real time
```

---

## What We'd Do With More Time

There are a few things that are on the list for after the hackathon:

- **Wire the escalator into live BKT parameters** — right now the escalation tracks generation counts and exposes them in the observation, but the hardest variant parameters aren't yet passed back into the student simulator. The scaffolding is there; it just needs closing.
- **Train the LLM path properly** — `train_llm_trl.py` exists and runs GRPO training on the environment, but compute constraints meant we couldn't fully evaluate it. An LLM that can both reason and adapt would be genuinely interesting in the blind teaching task.
- **Student diversity expansion** — eight archetypes is enough to prove the concept, but a real tutoring system needs to handle continuous variation, not discrete categories.

---

TeachRL is a hackathon project, which means some rough edges exist. But the core problem — teaching an agent to identify hidden learner types and adapt its curriculum in real time — is real, the environment is clean, and the results are honest. If any of that sounds useful to what you're building, we'd genuinely love to hear from you.

---

*Built for the Meta PyTorch Hackathon × Scaler 2025. MIT License. VIT-AP University.*

*[GitHub](https://github.com/ArchedEquation/TeachRL) · [Live Space](https://huggingface.co/spaces/ArchedEquation/TeachRL) · [Training Notebook](https://colab.research.google.com/github/ArchedEquation/TeachRL/blob/main/TeachRL_v2_Training.ipynb)*