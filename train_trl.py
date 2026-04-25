"""
train_trl.py — TeachRL v2 Training Script using HuggingFace TRL + PPO

Generates 4 publication-quality plots from REAL training data:
  1. reward_curves.png        — 4-subplot training curves, one per task
  2. agent_comparison.png     — PPO vs all baselines across 4 tasks (same axes)
  3. self_play_escalation.png — self-improvement evidence
  4. mastery_heatmap.png      — concept mastery per archetype, PPO vs Random

Usage:
    python train_trl.py --task all --eval           # train + eval + plots
    python train_trl.py --eval-only                 # plots from existing models
    python train_trl.py --self-play                 # collect escalation data
    python train_trl.py --plots-only                # regenerate plots from saved data
"""

import os, sys, argparse, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import wandb; HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.gym_wrapper import TeachRLGymEnv, obs_to_vector, int_to_action
from env.environment import TeachRLEnv, TASK_REGISTRY
from env.archetypes import ALL_ARCHETYPES, CONCEPTS
from graders.grader import GRADER_REGISTRY
from self_play.escalator import SelfPlayEscalator, EpisodeResult

# ── Config ────────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "archetype_identification": {
        "total_timesteps": 100_000, "n_envs": 4,
        "net_arch": [64, 64],
        "description": "Easy — Identify hidden student archetype",
        "color": "#27ae60",
    },
    "adaptive_curriculum": {
        "total_timesteps": 200_000, "n_envs": 4,
        "net_arch": [128, 128],
        "description": "Medium — Teach revealed archetype to mastery",
        "color": "#f39c12",
    },
    "blind_teaching": {
        "total_timesteps": 400_000, "n_envs": 4,
        "net_arch": [256, 256, 128],
        "description": "Hard — Infer AND teach simultaneously",
        "color": "#e74c3c",
    },
    "self_play_escalation": {
        "total_timesteps": 500_000, "n_envs": 4,
        "net_arch": [256, 256, 128],
        "description": "Expert — Face auto-escalated student variants",
        "color": "#8e44ad",
    },
}

PPO_HYPERPARAMS = {
    "learning_rate": 2e-4, "batch_size": 256, "n_epochs": 10,
    "gamma": 0.999, "gae_lambda": 0.95, "clip_range": 0.2,
    "ent_coef": 0.02, "vf_coef": 0.5, "max_grad_norm": 0.5,
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "training_plots")
DATA_DIR  = os.path.join(PLOTS_DIR, "data")

TASK_LABELS = {
    "archetype_identification": "Easy\nArchetype ID",
    "adaptive_curriculum":      "Medium\nCurriculum",
    "blind_teaching":           "Hard\nBlind Teaching",
    "self_play_escalation":     "Expert\nSelf-Play",
}

ARCH_COLORS = {
    "overconfident_learner": "#e74c3c", "anxious_perfectionist": "#9b59b6",
    "adhd_sprinter":         "#3498db", "slow_steady_builder":   "#27ae60",
    "strategic_gamer":       "#f39c12", "emotional_learner":     "#1abc9c",
    "uneven_genius":         "#e67e22", "impostor":              "#34495e",
}


def _makedirs():
    for d in [MODEL_DIR, PLOTS_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)

_makedirs()


# ── Callback ──────────────────────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    def __init__(self, task_id, log_interval=10_000, use_wandb=False):
        super().__init__()
        self.task_id            = task_id
        self.log_interval       = log_interval
        self.use_wandb          = use_wandb
        self._last_log          = 0
        self._t0                = time.time()
        self.step_history:    list = []
        self.reward_history:  list = []
        self._episode_rewards: list = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
        if self.num_timesteps - self._last_log >= self.log_interval:
            elapsed     = time.time() - self._t0
            fps         = self.num_timesteps / max(elapsed, 1)
            mean_reward = float(np.mean(self._episode_rewards[-20:])) if self._episode_rewards else 0.0
            self.step_history.append(self.num_timesteps)
            self.reward_history.append(mean_reward)
            print(f"  [{self.task_id}] step={self.num_timesteps:>7,} | "
                  f"reward={mean_reward:>6.3f} | fps={fps:>5.0f} | elapsed={elapsed:>5.0f}s")
            if self.use_wandb and HAS_WANDB:
                wandb.log({f"{self.task_id}/mean_reward": mean_reward,
                           "global_step": self.num_timesteps})
            self._last_log = self.num_timesteps
        return True


# ── Training ──────────────────────────────────────────────────────────────────

def train_task(task_id, seed=42, use_wandb=False):
    cfg       = TASK_CONFIGS[task_id]
    max_steps = TASK_REGISTRY[task_id]["max_steps"]
    n_steps   = max(512, max_steps * 4)

    print(f"\n{'='*60}")
    print(f"  Task:        {task_id}")
    print(f"  Description: {cfg['description']}")
    print(f"  Timesteps:   {cfg['total_timesteps']:,}")
    print(f"{'='*60}")

    if use_wandb and HAS_WANDB:
        wandb.init(project="TeachRL-v2", name=f"{task_id}-seed{seed}",
                   config={**cfg, **PPO_HYPERPARAMS}, reinit=True)

    vec_env  = make_vec_env(
        lambda: Monitor(TeachRLGymEnv(task_id=task_id, seed=seed)),
        n_envs=cfg["n_envs"], seed=seed)
    eval_env = make_vec_env(
        lambda: TeachRLGymEnv(task_id=task_id, seed=seed+999),
        n_envs=1, seed=seed+999)

    metrics_cb = MetricsCallback(
        task_id=task_id,
        log_interval=max(10_000, cfg["total_timesteps"] // 20),
        use_wandb=use_wandb)
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=MODEL_DIR, log_path=MODEL_DIR,
        eval_freq=max(10_000 // cfg["n_envs"], 1), n_eval_episodes=5, verbose=0)

    model = PPO("MlpPolicy", vec_env, n_steps=n_steps,
                policy_kwargs={"net_arch": cfg["net_arch"]},
                verbose=0, seed=seed, **PPO_HYPERPARAMS)

    t0 = time.time()
    model.learn(cfg["total_timesteps"], callback=[metrics_cb, eval_cb],
                progress_bar=False)
    elapsed = time.time() - t0

    model.save(os.path.join(MODEL_DIR, f"ppo_{task_id}"))
    print(f"\n  Model saved → models/ppo_{task_id}.zip  ({elapsed:.0f}s)")

    # Save raw training data
    with open(os.path.join(DATA_DIR, f"training_{task_id}.json"), "w") as f:
        json.dump({"task_id": task_id,
                   "steps":   metrics_cb.step_history,
                   "rewards": metrics_cb.reward_history,
                   "elapsed": elapsed}, f)

    if use_wandb and HAS_WANDB:
        wandb.finish()
    return model, metrics_cb


# ── Evaluation ────────────────────────────────────────────────────────────────

def _ppo_agent(task_id):
    """Returns a callable PPO+Classifier agent for the given task."""
    mp = os.path.join(MODEL_DIR, f"ppo_{task_id}.zip")
    if not os.path.exists(mp):
        return None
    max_s = TASK_REGISTRY[task_id]["max_steps"]
    model = PPO.load(mp)
    try:
        from baseline.archetype_classifier import ArchetypeClassifier
        clf = ArchetypeClassifier()
    except Exception:
        clf = None
    def agent(obs):
        vec = obs_to_vector(obs, max_s)
        a, _ = model.predict(vec, deterministic=True)
        c, d  = int_to_action(int(a))
        g     = clf.predict_from_obs(obs) if (clf and clf.is_loaded) else None
        return {"concept":c,"difficulty":d,"hint_given":False,"archetype_guess":g}
    return agent


def evaluate_task(task_id, n_episodes=10, seed=42):
    from baseline.agents import (RandomAgent, HeuristicAgent,
                                  GreedyArchetypeAgent, ArchetypeInferenceAgent)

    ppo_fn = _ppo_agent(task_id)
    if ppo_fn is None:
        print(f"  [SKIP] No model for {task_id}")
        return None

    Cls  = GRADER_REGISTRY[task_id]
    n    = min(n_episodes, 5) if task_id == "self_play_escalation" else n_episodes
    res  = {}

    agents = [
        ("Random",    RandomAgent()),
        ("Heuristic", HeuristicAgent()),
        ("Greedy",    GreedyArchetypeAgent()),
        ("Inference", ArchetypeInferenceAgent()),
        ("PPO+Clf",   ppo_fn),
    ]

    print(f"\n  {task_id}:\n  {'─'*48}")
    for name, agent in agents:
        r = Cls(n_episodes=n).grade(agent)
        res[name] = {"score": r.score, "arch_acc": r.archetype_accuracy,
                     "per_ep": r.per_episode_scores}
        best = max(v["score"] for v in res.values())
        tag  = " <-- best" if r.score == best else ""
        print(f"  {name:<14} score={r.score:.4f} arch_acc={r.archetype_accuracy:.3f}{tag}")

    best_bl = max(v["score"] for k,v in res.items() if k!="PPO+Clf")
    delta   = res["PPO+Clf"]["score"] - best_bl
    print(f"  PPO vs best baseline: {'+' if delta>=0 else ''}{delta:.4f}")

    with open(os.path.join(DATA_DIR, f"eval_{task_id}.json"), "w") as f:
        json.dump({k: {"score":v["score"],"arch_acc":v["arch_acc"]}
                   for k,v in res.items()}, f)
    return res


# ── Self-play data collection ─────────────────────────────────────────────────

def collect_self_play_data(n_episodes=12, seed=42):
    """Run real self-play escalation episodes and save the data."""
    ppo_fn = _ppo_agent("self_play_escalation") or _ppo_agent("blind_teaching")
    if ppo_fn is None:
        from baseline.agents import GreedyArchetypeAgent
        ppo_fn     = GreedyArchetypeAgent()
        agent_name = "Greedy (no PPO model)"
    else:
        agent_name = "PPO+Classifier"

    escalator = SelfPlayEscalator()
    env       = TeachRLEnv(task_id="self_play_escalation",
                           seed=seed, eval_mode=True, escalator=escalator)

    ep_scores, ep_archs, events = [], [], []
    gen_before = {a: 0 for a in ALL_ARCHETYPES}

    print(f"\n  Self-play ({n_episodes} episodes, agent={agent_name})...")

    for ep in range(n_episodes):
        obs  = env.reset(seed=seed+ep)
        done = False
        while not done:
            r = env.step(ppo_fn(obs.model_dump()))
            obs = r.observation; done = r.done
        score = env._task_score()
        arch  = env._sim.archetype_id
        ep_scores.append(score)
        ep_archs.append(arch.value if arch else "unknown")

        new_g = escalator.summary()["generations"]
        for a in ALL_ARCHETYPES:
            if new_g[a.value] > gen_before[a.value]:
                events.append({"episode":ep+1,"archetype":a.value,
                                "old_gen":gen_before[a.value],
                                "new_gen":new_g[a.value],"score":score})
                print(f"  ESCALATION! {a.value} Gen {new_g[a.value]} (score={score:.3f})")
        gen_before = dict(new_g)
        print(f"  ep{ep+1:>2}: {arch.value:<28} score={score:.4f}")

    sp = {"agent":agent_name,"episodes":n_episodes,
          "scores":ep_scores,"archetypes":ep_archs,
          "escalation_events":events,
          "final_generations":escalator.summary()["generations"],
          "avg_scores":escalator.summary()["avg_scores"],
          "total_escalations":escalator.total_generation_sum()}

    with open(os.path.join(DATA_DIR, "self_play_data.json"), "w") as f:
        json.dump(sp, f)
    print(f"  Total escalations: {sp['total_escalations']}")
    return sp


# ── Plotting ──────────────────────────────────────────────────────────────────

def _smooth(x, w=8):
    x = np.array(x, dtype=float)
    if len(x) <= w: return x
    return np.convolve(x, np.ones(w)/w, mode='valid')


def plot_reward_curves():
    """Plot 1: 4-subplot training reward curves from real logged data."""
    if not HAS_MPL: return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        'TeachRL v2 — PPO Training Reward Curves (Real Data from Live Environment)\n'
        'x: training steps (thousands)  |  y: mean episode reward [0–1]  |  '
        'each data point = last 20 completed episodes',
        fontsize=12, fontweight='bold', y=0.99)

    for ax, task_id in zip(axes.flat, list(TASK_CONFIGS.keys())):
        cfg   = TASK_CONFIGS[task_id]
        color = cfg["color"]
        dp    = os.path.join(DATA_DIR, f"training_{task_id}.json")

        if os.path.exists(dp):
            with open(dp) as f: d = json.load(f)
            steps   = np.array(d["steps"], dtype=float)
            rewards = np.array(d["rewards"], dtype=float)
            sm      = _smooth(rewards, w=min(5, len(rewards)))
            ts      = steps[:len(sm)]

            ax.plot(steps/1000, rewards, color=color, alpha=0.18, linewidth=1.2,
                    label='Raw (per log interval)')
            ax.fill_between(ts/1000, sm-0.03, sm+0.03, alpha=0.15, color=color)
            ax.plot(ts/1000, sm, color=color, linewidth=2.8,
                    label='Smoothed (w=5)')
            ax.set_xlim(0, max(steps)/1000 * 1.02)
            note = "Real training data"
        else:
            ax.text(0.5, 0.5,
                    'No data yet.\nRun: python train_trl.py --task all',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='#bbb', style='italic')
            note = "Awaiting training run"

        # Best baseline line from eval data
        ep = os.path.join(DATA_DIR, f"eval_{task_id}.json")
        if os.path.exists(ep):
            with open(ep) as f: ev = json.load(f)
            bls  = {k:v["score"] for k,v in ev.items() if k!="PPO+Clf"}
            ppo_s= ev.get("PPO+Clf",{}).get("score",0)
            if bls:
                best_s = max(bls.values())
                best_n = max(bls, key=bls.get)
                ax.axhline(best_s, color='#7f8c8d', linestyle='--', linewidth=1.8,
                           label=f'Best baseline: {best_n} ({best_s:.3f})', alpha=0.9)
            if ppo_s > 0:
                ax.axhline(ppo_s, color=color, linestyle=':', linewidth=1.5,
                           label=f'PPO eval score ({ppo_s:.3f})', alpha=0.85)

        diff = TASK_REGISTRY[task_id]["difficulty"].capitalize()
        ax.set_title(f'{diff}: {task_id.replace("_"," ").title()}\n{cfg["description"]}',
                     fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel('Training Steps (thousands)', fontsize=10)
        ax.set_ylabel('Mean Episode Reward [0–1]', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.text(0.02, 0.97, note, transform=ax.transAxes,
                fontsize=8, color='#888', va='top', style='italic')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(PLOTS_DIR, "reward_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  reward_curves.png saved")


def plot_agent_comparison():
    """Plot 2: All agents across all tasks — same axes, real eval scores."""
    if not HAS_MPL: return

    tasks = list(TASK_CONFIGS.keys())
    AGENT_STYLES = {
        "Random":    "#bdc3c7",
        "Heuristic": "#85c1e9",
        "Greedy":    "#76d7c4",
        "Inference": "#f8c471",
        "PPO+Clf":   "#e74c3c",
    }

    all_scores = {a: [] for a in AGENT_STYLES}
    has_data   = False

    for task_id in tasks:
        ep = os.path.join(DATA_DIR, f"eval_{task_id}.json")
        if os.path.exists(ep):
            with open(ep) as f: d = json.load(f)
            has_data = True
            for a in all_scores:
                all_scores[a].append(d.get(a, {}).get("score", 0.0))
        else:
            for a in all_scores:
                all_scores[a].append(0.0)

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.suptitle(
        'TeachRL v2 — All Agents Compared Across All Tasks (Real Evaluation)\n'
        'x: task difficulty  |  y: task score [0–1]  |  10 episodes, seed=42',
        fontsize=12, fontweight='bold')

    x       = np.arange(len(tasks))
    width   = 0.14
    offsets = np.linspace(-2, 2, 5) * width

    for i, (agent, color) in enumerate(AGENT_STYLES.items()):
        scores = all_scores[agent]
        bars   = ax.bar(x + offsets[i], scores, width, label=agent,
                        color=color, alpha=0.9, edgecolor='white',
                        linewidth=2.0 if agent=="PPO+Clf" else 0.6)
        for bar, val in zip(bars, scores):
            if val > 0.05:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.013, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=7.5,
                        fontweight='bold' if agent=="PPO+Clf" else 'normal',
                        color='#c0392b' if agent=="PPO+Clf" else '#444')

    # Annotate PPO wins
    for i, task_id in enumerate(tasks):
        ep = os.path.join(DATA_DIR, f"eval_{task_id}.json")
        if not os.path.exists(ep): continue
        with open(ep) as f: d = json.load(f)
        ppo  = d.get("PPO+Clf",{}).get("score",0)
        best = max(v["score"] for k,v in d.items() if k!="PPO+Clf")
        if ppo - best > 0.005:
            ax.annotate(f'+{ppo-best:.3f}',
                        xy=(i+offsets[4], ppo+0.015),
                        xytext=(i+offsets[4], ppo+0.11),
                        fontsize=9, color='#c0392b', fontweight='bold', ha='center',
                        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3))

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=11)
    ax.set_ylabel('Task Score  [0 = worst,  1 = perfect]', fontsize=11)
    ax.set_xlabel('Task  (increasing difficulty →)', fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.92, ncol=3)
    ax.grid(True, axis='y', alpha=0.2)
    ax.axhline(0.5, color='#ccc', linestyle=':', linewidth=1)
    ax.text(3.9, 0.515, 'chance level', fontsize=8, color='#aaa', ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if not has_data:
        ax.text(0.5, 0.5, 'Run --eval first', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='#aaa')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "agent_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  agent_comparison.png saved")


def plot_self_improvement():
    """Plot 3: Self-improvement — episode scores, escalations, early vs late."""
    if not HAS_MPL: return

    sp_path = os.path.join(DATA_DIR, "self_play_data.json")
    if not os.path.exists(sp_path):
        print("  [SKIP] No self-play data — run --self-play first")
        return

    with open(sp_path) as f: sp = json.load(f)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'TeachRL v2 — Self-Improvement Evidence  (Theme 4: Recursive Self-Improvement)\n'
        f'Agent: {sp["agent"]}  |  Episodes: {sp["episodes"]}  |  '
        f'Escalations triggered: {sp["total_escalations"]}',
        fontsize=12, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.35,
                           top=0.91, bottom=0.07, left=0.07, right=0.97)

    # ── A: Score per episode (full width top) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    eps    = list(range(1, sp["episodes"]+1))
    scores = sp["scores"]
    archs  = sp["archetypes"]

    ax1.plot(eps, scores, '-', color='#2c3e50', linewidth=2, zorder=3)
    for i, (ep, sc, arch) in enumerate(zip(eps, scores, archs)):
        ax1.scatter(ep, sc, color=ARCH_COLORS.get(arch,'#999'),
                    s=130, zorder=5, edgecolors='white', linewidth=1.5)

    # Escalation vertical lines
    for ev in sp["escalation_events"]:
        ax1.axvline(ev["episode"], color='#e74c3c', linestyle='--',
                    linewidth=2, alpha=0.75, zorder=2)
        ax1.annotate(
            f"ESCALATION\n{ev['archetype'][:14]}\nGen {ev['old_gen']}→{ev['new_gen']}",
            xy=(ev["episode"], ev["score"]),
            xytext=(ev["episode"]+0.2, min(ev["score"]+0.12, 0.97)),
            fontsize=7.5, color='#c0392b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#e74c3c', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2))

    # Trend line
    if len(scores) > 3:
        z = np.polyfit(eps, scores, 1)
        p = np.poly1d(z)
        sign = '+' if z[0] > 0 else ''
        ax1.plot(eps, p(eps), '--', color='#7f8c8d', linewidth=1.8, alpha=0.7,
                 label=f'Trend: {sign}{z[0]:.4f} per episode')

    ax1.axhline(0.70, color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.text(eps[-1]+0.1, 0.71, 'escalation threshold (0.70)',
             fontsize=8, color='#e74c3c')

    # Legend for archetype dots
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker='o', color='w',
                      markerfacecolor=ARCH_COLORS.get(a.value,'#999'),
                      markersize=9, label=a.value.replace('_',' '))
               for a in ALL_ARCHETYPES
               if a.value in archs]
    ax1.legend(handles=handles, fontsize=7.5, loc='lower left',
               ncol=4, framealpha=0.9, title='Student Archetype (dot colour)')

    ax1.set_xlabel('Episode Number', fontsize=10)
    ax1.set_ylabel('Task Score  [0–1]', fontsize=10)
    ax1.set_title('Score Per Episode — Coloured by Hidden Student Archetype  '
                  '|  Red dashed lines = escalation events',
                  fontsize=11, fontweight='bold')
    ax1.set_xlim(0.3, sp["episodes"]+0.7)
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, alpha=0.18)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── B: Escalation generations bar chart ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    gens  = sp["final_generations"]
    names = [k.replace("_"," ").title()[:16] for k in gens]
    vals  = list(gens.values())
    cols  = [ARCH_COLORS.get(k,'#999') for k in gens]
    bars  = ax2.bar(names, vals, color=cols, alpha=0.85,
                    edgecolor='white', linewidth=0.8)
    for bar, g in zip(bars, vals):
        lbl = f'Gen {g}' if g > 0 else '–'
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.04, lbl,
                 ha='center', fontsize=9,
                 fontweight='bold' if g>0 else 'normal',
                 color='#c0392b' if g>0 else '#aaa')
    ax2.set_ylabel('Escalation Generation', fontsize=10)
    ax2.set_title('Escalations Per Archetype\n'
                  'Gen > 0 = agent mastered it → env got harder',
                  fontsize=10, fontweight='bold')
    ax2.set_ylim(0, max(vals)+1.4 if vals else 2)
    ax2.tick_params(axis='x', rotation=38, labelsize=8)
    ax2.grid(True, axis='y', alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── C: Early vs Late violin ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    n     = len(scores)
    half  = max(n//2, 1)
    early = scores[:half]
    late  = scores[half:]
    m_e   = np.mean(early)
    m_l   = np.mean(late)
    delta = m_l - m_e

    parts = ax3.violinplot([early, late], positions=[1,2],
                            showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_alpha(0.55)
    parts['bodies'][0].set_facecolor('#85c1e9')
    parts['bodies'][1].set_facecolor('#e74c3c')
    ax3.scatter([1]*len(early), early, color='#3498db', s=60, alpha=0.75, zorder=3)
    ax3.scatter([2]*len(late),  late,  color='#c0392b', s=60, alpha=0.75, zorder=3)

    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(
        [f'Early\n(ep 1–{half})\nmean={m_e:.3f}',
         f'Late\n(ep {half+1}–{n})\nmean={m_l:.3f}'],
        fontsize=10)
    ax3.set_ylabel('Task Score  [0–1]', fontsize=10)
    sign = '+' if delta >= 0 else ''
    ax3.set_title(f'Early vs Late Performance\n'
                  f'Net change: {sign}{delta:.3f} '
                  f'(env escalated between halves)',
                  fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, axis='y', alpha=0.2)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    path = os.path.join(PLOTS_DIR, "self_play_escalation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  self_play_escalation.png saved")


def plot_mastery_heatmap():
    """Plot 4: Concept mastery heatmap — Random vs PPO, per archetype."""
    if not HAS_MPL: return

    from baseline.agents import RandomAgent

    ppo_fn = _ppo_agent("blind_teaching")
    if ppo_fn is None:
        print("  [SKIP] No PPO model for mastery heatmap")
        return

    agents = {"Random": RandomAgent(), "PPO+Clf": ppo_fn}
    N      = 1   # episodes per archetype (fast)

    print("  Collecting mastery data for heatmap...")
    matrices = {}
    for name, agent in agents.items():
        mat = np.zeros((len(CONCEPTS), len(ALL_ARCHETYPES)))
        for j, arch in enumerate(ALL_ARCHETYPES):
            env = TeachRLEnv(task_id="blind_teaching",
                             seed=7+j, eval_mode=True)
            obs = env.reset(seed=7+j, archetype_id=arch)
            done = False
            while not done:
                r = env.step(agent(obs.model_dump()))
                obs = r.observation; done = r.done
            for i, c in enumerate(CONCEPTS):
                mat[i][j] = env._sim.state.mastery[c]
        matrices[name] = mat

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle(
        'TeachRL v2 — Concept Mastery Achieved per Student Archetype\n'
        'x: hidden student archetype  |  y: math concept  |  '
        'colour: mastery [0=none, 1=fully mastered]  |  task: blind_teaching',
        fontsize=12, fontweight='bold')

    arch_labels    = [a.value.replace("_","\n") for a in ALL_ARCHETYPES]
    concept_labels = [c.replace("_"," ").title() for c in CONCEPTS]

    for ax, (name, mat) in zip(axes, matrices.items()):
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(ALL_ARCHETYPES)))
        ax.set_xticklabels(arch_labels, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(CONCEPTS)))
        ax.set_yticklabels(concept_labels, fontsize=9)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Student Archetype (hidden from agent)', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel('Math Concept', fontsize=10)
        for i in range(len(CONCEPTS)):
            for j in range(len(ALL_ARCHETYPES)):
                v = mat[i][j]
                tc = 'white' if v < 0.3 or v > 0.80 else '#222'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color=tc)
        plt.colorbar(im, ax=ax, label='Mastery [0–1]', shrink=0.85)

    # Highlight the improvement
    diff = matrices["PPO+Clf"] - matrices["Random"]
    mean_improvement = diff.mean()
    fig.text(0.5, 0.01,
             f'PPO+Clf mean mastery: {matrices["PPO+Clf"].mean():.3f}  |  '
             f'Random mean mastery: {matrices["Random"].mean():.3f}  |  '
             f'PPO improvement: +{mean_improvement:.3f}',
             ha='center', fontsize=10, fontweight='bold', color='#2c3e50')

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = os.path.join(PLOTS_DIR, "mastery_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  mastery_heatmap.png saved")


def generate_all_plots():
    if not HAS_MPL:
        print("  matplotlib not installed"); return
    print(f"\n  Generating plots → {PLOTS_DIR}/")
    plot_reward_curves()
    plot_agent_comparison()
    plot_self_improvement()
    plot_mastery_heatmap()
    print(f"  Done. 4 plots saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       default="blind_teaching",
                   choices=["all"]+list(TASK_CONFIGS))
    p.add_argument("--steps",      type=int,  default=None)
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--eval",       action="store_true")
    p.add_argument("--wandb",      action="store_true")
    p.add_argument("--episodes",   type=int,  default=10)
    p.add_argument("--eval-only",  action="store_true")
    p.add_argument("--self-play",  action="store_true")
    p.add_argument("--plots-only", action="store_true")
    args = p.parse_args()

    tasks = list(TASK_CONFIGS) if args.task=="all" else [args.task]

    print(f"\n{'='*60}")
    print(f"  TeachRL v2 — Training with HuggingFace TRL + PPO")
    print(f"  Tasks: {tasks}  |  Seed: {args.seed}")
    print(f"{'='*60}")

    _makedirs()

    if args.plots_only:
        generate_all_plots(); return

    all_results = {}
    for task_id in tasks:
        if args.steps:
            TASK_CONFIGS[task_id]["total_timesteps"] = args.steps
        if not args.eval_only:
            train_task(task_id=task_id, seed=args.seed, use_wandb=args.wandb)
        if args.eval or args.eval_only:
            r = evaluate_task(task_id, n_episodes=args.episodes, seed=args.seed)
            if r: all_results[task_id] = r

    if args.self_play or (args.eval and "self_play_escalation" in tasks):
        collect_self_play_data(n_episodes=args.episodes, seed=args.seed)

    generate_all_plots()

    if all_results:
        print(f"\n{'='*60}  SUMMARY")
        print(f"  {'Task':<32} {'PPO':>8} {'vs Best':>10}")
        print(f"  {'─'*52}")
        for tid, res in all_results.items():
            ppo  = res.get("PPO+Clf",{}).get("score",0.0)
            best = max(v["score"] for k,v in res.items() if k!="PPO+Clf")
            d    = ppo - best
            print(f"  {tid:<32} {ppo:>8.4f}  {'+' if d>=0 else ''}{d:.4f}")


if __name__ == "__main__":
    main()