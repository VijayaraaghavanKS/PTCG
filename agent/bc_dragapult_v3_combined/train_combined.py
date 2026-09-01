"""
Train a Dragapult-specific GBDT BC model on a combination of:
  - rows_public.jsonl   (real top-player replay wins, Dragapult-tagged, from the public dataset)
  - rows_own_ladder.jsonl (our own real live-ladder win replays, 100% Dragapult)
  - rows_selfplay.jsonl (self-play wins by dragapult_patched_v1 vs 5 local reference decks)

Self-play is volume/diversity augmentation, not a primary signal (a pointwise
classifier distilling its own teacher's outputs can't exceed the teacher --
see this project's bc_v1-v4 findings) -- so real rows get much higher sample
weight, and self-play is subsampled (by GROUP, keeping full decisions intact)
rather than used at its full multi-million-row scale.

Usage:
  python3 train_combined.py <mode> <out_prefix>
    mode = "real_only" | "combined"
"""
import json
import os
import pickle
import random
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def renumber_groups(rows, offset):
    """Groups from different source files collide (all start at 1) -- renumber
    to a disjoint range and return the new next-offset."""
    remap = {}
    next_id = offset
    for r in rows:
        if r["group"] not in remap:
            remap[r["group"]] = next_id
            next_id += 1
        r["group"] = remap[r["group"]]
    return next_id


def subsample_selfplay_groups(rows, n_groups, seed=42):
    groups = sorted(set(r["group"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(groups)
    keep = set(groups[:n_groups])
    return [r for r in rows if r["group"] in keep]


def split_groups(rows, held_out_frac=0.2, seed=42):
    """Split by underlying GAME (game_key), not by decision group, to avoid leakage."""
    games = sorted(set(r["game_key"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(games)
    n_held = max(1, int(len(games) * held_out_frac))
    held = set(games[:n_held])
    train = set(games[n_held:])
    return train, held


def build_arrays(rows, game_set, real_weight=1.0, selfplay_weight=0.15):
    X, y, groups, w = [], [], [], []
    for r in rows:
        if r["game_key"] not in game_set:
            continue
        X.append(r["feat"])
        y.append(r["label"])
        groups.append(r["group"])
        w.append(real_weight if r.get("source") != "selfplay" else selfplay_weight)
    return (np.array(X, dtype=np.float64), np.array(y, dtype=np.int32),
            np.array(groups, dtype=np.int64), np.array(w, dtype=np.float64))


def decision_level_eval(scores, y, groups):
    order = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    n, correct_model, correct_first = 0, 0, 0
    for g, idxs in order.items():
        true_idxs = [i for i in idxs if y[i] == 1]
        if not true_idxs:
            continue
        n += 1
        true_set = set(true_idxs)
        best_i = max(idxs, key=lambda i: scores[i])
        if best_i in true_set:
            correct_model += 1
        if idxs[0] in true_set:
            correct_first += 1
    return {
        "n_decisions": n,
        "model_acc": correct_model / n if n else 0.0,
        "first_option_baseline_acc": correct_first / n if n else 0.0,
    }


def main():
    mode = sys.argv[1]
    out_prefix = sys.argv[2]

    public_rows = load_rows(os.path.join(HERE, "rows_public.jsonl"))
    own_rows = load_rows(os.path.join(HERE, "rows_own_ladder.jsonl"))

    next_off = 1
    next_off = renumber_groups(public_rows, next_off)
    next_off = renumber_groups(own_rows, next_off)
    real_rows = public_rows + own_rows
    print(f"Real rows: {len(real_rows)} ({len(set(r['group'] for r in real_rows))} groups, "
          f"{len(set(r['game_key'] for r in real_rows))} games)")

    all_rows = list(real_rows)
    if mode == "combined":
        selfplay_rows = load_rows(os.path.join(HERE, "rows_selfplay.jsonl"))
        n_selfplay_groups_total = len(set(r["group"] for r in selfplay_rows))
        target_groups = min(n_selfplay_groups_total, 40000)
        selfplay_sub = subsample_selfplay_groups(selfplay_rows, target_groups)
        next_off = renumber_groups(selfplay_sub, next_off)
        print(f"Self-play subsample: {len(selfplay_sub)} rows ({target_groups} of {n_selfplay_groups_total} groups)")
        all_rows += selfplay_sub

    train_games, held_games = split_groups(all_rows, held_out_frac=0.2, seed=42)
    X_train, y_train, g_train, w_train = build_arrays(all_rows, train_games)
    X_held, y_held, g_held, w_held = build_arrays(all_rows, held_games)
    print(f"Train rows: {len(X_train)}, Held rows: {len(X_held)}")

    # Hold-out eval restricted to REAL games only (selfplay-vs-selfplay accuracy
    # is not informative -- we care whether real-decision prediction improved).
    held_real_games = held_games & set(r["game_key"] for r in real_rows)
    X_held_real, y_held_real, g_held_real, _ = build_arrays(real_rows, held_real_games)

    t0 = time.time()
    class_bal = np.where(y_train == 1, (len(y_train) - y_train.sum()) / max(1, y_train.sum()), 1.0)
    sw_train = w_train * class_bal
    gbdt = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.8,
        min_samples_leaf=10, random_state=42,
    )
    gbdt.fit(X_train, y_train, sample_weight=sw_train)
    t_fit = time.time() - t0
    print(f"Train time: {t_fit:.1f}s")

    scores_held = gbdt.decision_function(X_held)
    auc_held = roc_auc_score(y_held, scores_held) if len(set(y_held)) > 1 else float("nan")
    dec_held = decision_level_eval(scores_held, y_held, g_held)
    print(f"Held (all sources) decision acc: {dec_held}, row AUC: {auc_held:.4f}")

    if len(X_held_real) > 0:
        scores_held_real = gbdt.decision_function(X_held_real)
        dec_held_real = decision_level_eval(scores_held_real, y_held_real, g_held_real)
        print(f"Held (REAL games only) decision acc: {dec_held_real}")
    else:
        dec_held_real = None

    with open(os.path.join(HERE, f"{out_prefix}_model.pkl"), "wb") as f:
        pickle.dump(gbdt, f)

    # Final: retrain on ALL rows (train+held) for deployment.
    X_all, y_all, g_all, w_all = build_arrays(all_rows, train_games | held_games)
    class_bal_all = np.where(y_all == 1, (len(y_all) - y_all.sum()) / max(1, y_all.sum()), 1.0)
    gbdt_final = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.8,
        min_samples_leaf=10, random_state=42,
    )
    gbdt_final.fit(X_all, y_all, sample_weight=w_all * class_bal_all)
    with open(os.path.join(HERE, f"{out_prefix}_model_final.pkl"), "wb") as f:
        pickle.dump(gbdt_final, f)

    report = {
        "mode": mode,
        "n_real_rows": len(real_rows),
        "n_train_rows": len(X_train), "n_held_rows": len(X_held),
        "train_time_s": t_fit,
        "auc_held": auc_held,
        "decision_held_all": dec_held,
        "decision_held_real_only": dec_held_real,
    }
    with open(os.path.join(HERE, f"{out_prefix}_train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out_prefix}_model.pkl, {out_prefix}_model_final.pkl, {out_prefix}_train_report.json")


if __name__ == "__main__":
    main()
