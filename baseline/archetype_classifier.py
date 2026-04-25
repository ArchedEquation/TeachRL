"""
archetype_classifier.py — Supervised archetype identification model.

Trains a small neural network to classify student archetype from
observable signals (success rates, streaks, engagement patterns).

Separate from PPO teaching policy — clean separation of concerns.
Training data generated from the student simulator directly.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Tuple, Optional

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from env.archetypes import ALL_ARCHETYPES, ArchetypeID, CONCEPTS, DIFFICULTY_LEVELS
from env.student import StudentSimulator


# ── Feature extraction ────────────────────────────────────────────────────────

ARCH_TO_IDX = {a: i for i, a in enumerate(ALL_ARCHETYPES)}
IDX_TO_ARCH = {i: a for i, a in enumerate(ALL_ARCHETYPES)}
N_CLASSES   = len(ALL_ARCHETYPES)  # 8

# Features per concept: success_rate, attempt_norm, streak_norm, prereq
# Global: engagement, fatigue, hint_reliability
FEATURE_DIM = len(CONCEPTS) * 4 + 3


def extract_features(sim: StudentSimulator) -> np.ndarray:
    """Extract observable features from student state."""
    s   = sim.state
    vec = []
    for c in CONCEPTS:
        ta  = sum(s.attempts[c].values())
        tc  = sum(s.correct[c].values())
        vec.append(tc / max(ta, 1))                          # success rate
        vec.append(min(ta / 15.0, 1.0))                     # attempt count (norm)
        vec.append(min(s.streak.get(c, 0) / 5.0, 1.0))     # streak (norm)
        vec.append(sim.prerequisite_readiness(c))            # prereq readiness
    vec.append(s.engagement)
    vec.append(s.fatigue)
    vec.append(0.7)  # hint reliability placeholder
    return np.array(vec, dtype=np.float32)


# ── Data generation ───────────────────────────────────────────────────────────

def generate_training_data(
    n_episodes: int = 2000,
    steps_per_episode: int = 15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (features, labels) by simulating episodes for each archetype.
    Features are extracted mid-episode (after ~15 interactions).
    This is when enough signal exists to identify the archetype.
    """
    rng   = np.random.default_rng(seed)
    X, y  = [], []
    eps_per_arch = n_episodes // N_CLASSES

    for arch_idx, arch_id in enumerate(ALL_ARCHETYPES):
        for ep in range(eps_per_arch):
            ep_seed = seed + arch_idx * 10000 + ep
            sim     = StudentSimulator(seed=ep_seed)
            sim.reset(seed=ep_seed, archetype_id=arch_id)

            # Random probe actions to generate observable signal
            for _ in range(steps_per_episode):
                concept    = CONCEPTS[rng.integers(len(CONCEPTS))]
                difficulty = DIFFICULTY_LEVELS[rng.integers(len(DIFFICULTY_LEVELS))]
                sim.answer_question(concept, difficulty)

            X.append(extract_features(sim))
            y.append(arch_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── Model ─────────────────────────────────────────────────────────────────────

class ArchetypeNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────

def train_classifier(
    n_episodes: int  = 4000,
    epochs:     int  = 80,
    lr:         float = 1e-3,
    seed:       int  = 42,
) -> Optional["ArchetypeNet"]:
    if not HAS_TORCH:
        print("  [SKIP] PyTorch not available")
        return None

    print(f"\n  Training archetype classifier...")
    print(f"  Episodes: {n_episodes} | Epochs: {epochs}")

    X, y = generate_training_data(n_episodes=n_episodes, seed=seed)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, {N_CLASSES} classes")

    # Train / val split
    idx   = np.random.default_rng(seed).permutation(len(X))
    split = int(0.85 * len(X))
    X_tr, y_tr = X[idx[:split]], y[idx[:split]]
    X_va, y_va = X[idx[split:]], y[idx[split:]]

    ds_tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)

    model   = ArchetypeNet()
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        # Validate
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_va)
            yv = torch.tensor(y_va)
            preds    = model(xv).argmax(dim=1)
            val_acc  = (preds == yv).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs} | val_acc={val_acc:.3f} | best={best_val_acc:.3f}")

    model.load_state_dict(best_state)
    print(f"\n  Best val accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")

    # Save
    path = os.path.join(MODEL_DIR, "archetype_classifier.pt")
    torch.save({"model_state": model.state_dict(),
                "feature_dim": FEATURE_DIM,
                "n_classes":   N_CLASSES,
                "val_acc":     best_val_acc}, path)
    print(f"  Saved → {path}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

class ArchetypeClassifier:
    """
    Wraps the trained ArchetypeNet for inference.
    Used by PPOAgent to emit archetype_guess at each step.
    """

    def __init__(self):
        self._model: Optional[ArchetypeNet] = None
        self._loaded = False
        self._load()

    def _load(self):
        path = os.path.join(MODEL_DIR, "archetype_classifier.pt")
        if not os.path.exists(path):
            return
        if not HAS_TORCH:
            return
        ckpt         = torch.load(path, map_location="cpu")
        self._model  = ArchetypeNet(ckpt["feature_dim"], ckpt["n_classes"])
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        self._loaded = True
        print(f"  [Classifier] Loaded (val_acc={ckpt.get('val_acc', '?'):.3f})")

    def predict(self, sim: StudentSimulator) -> Optional[str]:
        """Return archetype ID string or None if model not loaded."""
        if not self._loaded or self._model is None:
            return None
        with torch.no_grad():
            feat   = torch.tensor(extract_features(sim)).unsqueeze(0)
            logits = self._model(feat)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            # Only guess when confident enough
            if conf.item() < 0.30:
                return None
            return IDX_TO_ARCH[idx.item()].value

    def predict_from_obs(self, obs_dict: dict, sim: Optional[StudentSimulator] = None) -> Optional[str]:
        """Predict from raw obs dict (no sim access needed)."""
        if not self._loaded or self._model is None:
            return None
        # Build feature vector from obs dict
        vec = []
        for c in CONCEPTS:
            sr  = obs_dict["concept_success_rates"].get(c, 0.0)
            att = obs_dict["concept_attempt_counts"].get(c, 0)
            stk = obs_dict["concept_streaks"].get(c, 0)
            prq = obs_dict["prerequisite_readiness"].get(c, 1.0)
            vec.extend([sr, min(att/15.0, 1.0), min(stk/5.0, 1.0), prq])
        vec.extend([
            obs_dict.get("engagement", 1.0),
            obs_dict.get("fatigue", 0.0),
            obs_dict.get("hint_reliability", 0.7),
        ])
        feat = torch.tensor(np.array(vec, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(feat)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            if conf.item() < 0.30:
                return None
            return IDX_TO_ARCH[idx.item()].value

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--epochs",   type=int, default=80)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--test",     action="store_true")
    args = p.parse_args()

    if args.test:
        clf = ArchetypeClassifier()
        if not clf.is_loaded:
            print("No classifier found. Run without --test first.")
        else:
            print("\nTesting classifier on fresh episodes:")
            correct = 0; total = 0
            for arch_id in ALL_ARCHETYPES:
                sim = StudentSimulator(seed=999)
                sim.reset(seed=999, archetype_id=arch_id)
                rng = np.random.default_rng(999)
                for _ in range(15):
                    c = CONCEPTS[rng.integers(len(CONCEPTS))]
                    d = DIFFICULTY_LEVELS[rng.integers(len(DIFFICULTY_LEVELS))]
                    sim.answer_question(c, d)
                pred = clf.predict(sim)
                hit  = pred == arch_id.value
                correct += hit; total += 1
                print(f"  {arch_id.value:<28} pred={pred:<28} {'✓' if hit else '✗'}")
            print(f"\n  Accuracy: {correct}/{total} = {correct/total:.1%}")
    else:
        train_classifier(n_episodes=args.episodes, epochs=args.epochs, seed=args.seed)