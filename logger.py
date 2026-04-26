"""
logger.py — TeachRL Training Logger

Writes structured, human-readable training logs to logs/ folder.
One log file per task, plus a master summary log.

Log format:
    logs/
    ├── training_archetype_identification.log
    ├── training_adaptive_curriculum.log
    ├── training_blind_teaching.log
    ├── training_self_play_escalation.log
    └── training_summary.log
"""

import os, sys, time, json, logging
from datetime import datetime
import numpy as np

# Use project root, not wherever logger.py is imported from
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR       = os.path.join(_PROJECT_ROOT, "logs")

# Each training run gets its own timestamped subfolder
_RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR  = os.path.join(LOG_DIR, f"run_{_RUN_TS}")
os.makedirs(RUN_DIR, exist_ok=True)


def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar(value, width=30, max_val=1.0):
    filled = int((value / max_val) * width)
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


class TeachRLLogger:
    """
    Structured logger for one training task.
    Writes to both console and a log file simultaneously.
    """

    def __init__(self, task_id: str, task_cfg: dict, ppo_cfg: dict, seed: int):
        self.task_id   = task_id
        self.task_cfg  = task_cfg
        self.ppo_cfg   = ppo_cfg
        self.seed      = seed
        self.t_start   = time.time()
        self.step_logs = []   # raw step-level data for JSON export
        self.eval_logs = []   # eval checkpoint data

        # File handler
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Each task gets its own subfolder inside the run folder
        task_dir  = os.path.join(RUN_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        log_path  = os.path.join(task_dir, f"training_{task_id}.log")
        self._file = open(log_path, "w", encoding="utf-8")
        self._path = log_path

        self._write_header()

    def _w(self, line="", also_print=True):
        """Write to file and optionally print."""
        self._file.write(line + "\n")
        self._file.flush()
        if also_print:
            print(line)

    def _write_header(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_ts = self.task_cfg["total_timesteps"]
        diff     = {
            "archetype_identification": "🟢 Easy",
            "adaptive_curriculum":      "🟡 Medium",
            "blind_teaching":           "🔴 Hard",
            "self_play_escalation":     "🔴🔴 Expert",
        }.get(self.task_id, "")

        self._w("=" * 70)
        self._w(f"  TeachRL — PPO Training Log")
        self._w(f"  Task:       {self.task_id}  {diff}")
        self._w(f"  Started:    {now}")
        self._w(f"  Seed:       {self.seed}")
        self._w(f"  Timesteps:  {total_ts:,}")
        self._w(f"  Envs:       {self.task_cfg['n_envs']} parallel")
        self._w(f"  Net arch:   {self.task_cfg['net_arch']}")
        self._w("-" * 70)
        self._w(f"  PPO Hyperparameters:")
        for k, v in self.ppo_cfg.items():
            self._w(f"    {k:<22} {v}")
        self._w("=" * 70)
        self._w()

    def log_step(self, step: int, total_steps: int, mean_reward: float,
                 episode_rewards: list, fps: float, elapsed: float):
        """Log one training checkpoint."""
        pct      = step / total_steps * 100
        norm_rew = mean_reward / self.task_cfg.get("max_steps", 80)
        norm_rew = max(0.0, min(1.0, norm_rew))
        bar      = _bar(norm_rew, width=25)
        eta      = (elapsed / max(step, 1)) * (total_steps - step)

        line = (
            f"  Step {step:>7,}/{total_steps:,} ({pct:>5.1f}%) │ "
            f"reward={mean_reward:>7.3f} │ "
            f"[{bar}] │ "
            f"fps={fps:>5.0f} │ "
            f"elapsed={_fmt_time(elapsed)} │ "
            f"eta={_fmt_time(eta)}"
        )
        self._w(line)

        # Store for JSON export
        self.step_logs.append({
            "step":          step,
            "pct":           round(pct, 2),
            "mean_reward":   round(mean_reward, 4),
            "norm_reward":   round(norm_rew, 4),
            "fps":           round(fps, 1),
            "elapsed_s":     round(elapsed, 1),
            "eta_s":         round(eta, 1),
            "n_episodes":    len(episode_rewards),
            "reward_std":    round(float(np.std(episode_rewards[-20:])) if len(episode_rewards) >= 2 else 0.0, 4),
            "reward_min":    round(float(np.min(episode_rewards[-20:])) if episode_rewards else 0.0, 4),
            "reward_max":    round(float(np.max(episode_rewards[-20:])) if episode_rewards else 0.0, 4),
        })

    def log_eval_checkpoint(self, step: int, eval_scores: dict):
        """Log evaluation checkpoint results."""
        self._w()
        self._w(f"  ── Eval Checkpoint @ step {step:,} {'─'*35}")
        for agent, score in eval_scores.items():
            marker = " ← 🏆" if score == max(eval_scores.values()) else ""
            bar    = _bar(score, width=20)
            self._w(f"  {agent:<16} [{bar}] {score:.4f}{marker}")
        self._w()

        self.eval_logs.append({"step": step, "scores": eval_scores})

    def log_training_complete(self, model_path: str):
        """Log training completion summary."""
        elapsed = time.time() - self.t_start
        self._w()
        self._w("=" * 70)
        self._w(f"  TRAINING COMPLETE")
        self._w(f"  Total time:    {_fmt_time(elapsed)}")
        self._w(f"  Model saved:   {model_path}")
        self._w(f"  Log file:      {self._path}")

        if self.step_logs:
            rewards = [s["mean_reward"] for s in self.step_logs]
            first3  = np.mean(rewards[:3])
            last3   = np.mean(rewards[-3:])
            delta   = last3 - first3
            self._w(f"  Reward start:  {first3:.4f} (mean of first 3 checkpoints)")
            self._w(f"  Reward end:    {last3:.4f} (mean of last 3 checkpoints)")
            self._w(f"  Improvement:   {'+' if delta>=0 else ''}{delta:.4f}")
            self._w(f"  Peak reward:   {max(rewards):.4f} @ step {self.step_logs[np.argmax(rewards)]['step']:,}")

        self._w("=" * 70)

    def save_data_for_plots(self):
        """Save raw training data for plot generation only (to data/ folder)."""
        data = {
            "task_id": self.task_id,
            "steps":   [s["step"]        for s in self.step_logs],
            "rewards": [s["mean_reward"] for s in self.step_logs],
            "elapsed": round(time.time() - self.t_start, 1),
        }
        from train_trl import DATA_DIR
        path = os.path.join(DATA_DIR, f"training_{self.task_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def close(self):
        self._file.close()


class SummaryLogger:
    """
    Master summary log across all 4 tasks.
    Written after all tasks complete.
    """

    def __init__(self):
        self._path   = os.path.join(RUN_DIR, "training_summary.log")
        self._results = {}

    def add_task_result(self, task_id: str, elapsed: float,
                        rewards: list, eval_scores: dict):
        self._results[task_id] = {
            "elapsed":     elapsed,
            "rewards":     rewards,
            "eval_scores": eval_scores,
        }

    def write(self, seed: int):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        TASK_ORDER = [
            "archetype_identification",
            "adaptive_curriculum",
            "blind_teaching",
            "self_play_escalation",
        ]
        DIFF = {
            "archetype_identification": "Easy   ",
            "adaptive_curriculum":      "Medium ",
            "blind_teaching":           "Hard   ",
            "self_play_escalation":     "Expert ",
        }

        lines = []
        lines.append("=" * 70)
        lines.append("  TeachRL — Training Summary")
        lines.append(f"  Generated: {now}  |  Seed: {seed}")
        lines.append("=" * 70)
        lines.append("")

        # Per-task summary
        total_time = 0
        for task_id in TASK_ORDER:
            if task_id not in self._results:
                continue
            r       = self._results[task_id]
            elapsed = r["elapsed"]
            rewards = r["rewards"]
            total_time += elapsed

            first3  = np.mean(rewards[:3])  if len(rewards) >= 3 else rewards[0]  if rewards else 0
            last3   = np.mean(rewards[-3:]) if len(rewards) >= 3 else rewards[-1] if rewards else 0
            delta   = last3 - first3

            lines.append(f"  {DIFF[task_id]} | {task_id}")
            lines.append(f"  {'─'*66}")
            lines.append(f"  Training time:  {_fmt_time(elapsed)}")
            lines.append(f"  Checkpoints:    {len(rewards)}")
            lines.append(f"  Reward start:   {first3:.4f}")
            lines.append(f"  Reward end:     {last3:.4f}  ({'+' if delta>=0 else ''}{delta:.4f})")
            lines.append(f"  Peak reward:    {max(rewards):.4f}")

            if r["eval_scores"]:
                lines.append(f"  Final Eval Scores:")
                es = r["eval_scores"]
                for agent in ["Random","Heuristic","Greedy","Inference","PPO+Clf"]:
                    if agent in es:
                        score  = es[agent]
                        best   = max(es.values())
                        marker = " ← 🏆" if score == best else ""
                        bar    = _bar(score, width=20)
                        lines.append(f"    {agent:<14} [{bar}] {score:.4f}{marker}")
                ppo  = es.get("PPO+Clf", 0)
                best_bl = max(v for k,v in es.items() if k != "PPO+Clf") if len(es)>1 else 0
                delta_ppo = ppo - best_bl
                lines.append(f"  PPO vs best baseline: {'+' if delta_ppo>=0 else ''}{delta_ppo:.4f}")
            lines.append("")

        # Overall summary table
        lines.append("=" * 70)
        lines.append("  FINAL RESULTS TABLE")
        lines.append("=" * 70)
        lines.append(f"  {'Agent':<16} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Expert':>8}")
        lines.append(f"  {'─'*52}")

        agents = ["Random","Heuristic","Greedy","Inference","PPO+Clf"]
        for agent in agents:
            row = f"  {agent:<16}"
            for task_id in TASK_ORDER:
                es = self._results.get(task_id, {}).get("eval_scores", {})
                sc = es.get(agent, None)
                row += f" {sc:>8.4f}" if sc is not None else f" {'—':>8}"
            if agent == "PPO+Clf":
                row += "  ← trained"
            lines.append(row)

        lines.append("")
        lines.append(f"  Total training time: {_fmt_time(total_time)}")
        lines.append("")

        lines.append(f"  Log files: {LOG_DIR}/")
        lines.append("=" * 70)

        content = "\n".join(lines)
        with open(self._path, "w", encoding="utf-8") as f:
            f.write(content)
        print("\n" + content)
        print(f"\n  Summary saved → {self._path}")
        print(f"  Run logs    → {RUN_DIR}/")