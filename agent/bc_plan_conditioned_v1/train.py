"""
Train a plan-conditioned GBDT BC model on the 144-dim (138 base + 6 turn-plan
context) real Dragapult-specific win-filtered dataset (rows_public.jsonl +
rows_own.jsonl). Mirrors agent/bc_dragapult_v3_combined/train_combined.py's
real_only mode, no self-play (already shown not to help in that project).
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
    remap = {}
    next_id = offset
    for r in rows:
        if r["group"] not in remap:
            remap[r["group"]] = next_id
            next_id += 1
        r["group"] = remap[r["group"]]
    return next_id


def split_groups(rows, held_out_frac=0.2, seed=42):
    games = sorted(set(r["game_key"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(games)
    n_held = max(1, int(len(games) * held_out_frac))
    held = set(games[:n_held])
    train = set(games[n_held:])
    return train, held


def build_arrays(rows, game_set):
    X, y, groups = [], [], []
    for r in rows:
        if r["game_key"] not in game_set:
            continue
        X.append(r["feat"])
        y.append(r["label"])
        groups.append(r["group"])
    return (np.array(X, dtype=np.float64), np.array(y, dtype=np.int32),
            np.array(groups, dtype=np.int64))


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
    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "plan_conditioned"

    public_rows = load_rows(os.path.join(HERE, "rows_public.jsonl"))
    own_rows = load_rows(os.path.join(HERE, "rows_own.jsonl"))
    next_off = 1
    next_off = renumber_groups(public_rows, next_off)
    next_off = renumber_groups(own_rows, next_off)
    all_rows = public_rows + own_rows
    print(f"Total rows: {len(all_rows)} ({len(set(r['group'] for r in all_rows))} groups, "
          f"{len(set(r['game_key'] for r in all_rows))} games)")

    train_games, held_games = split_groups(all_rows, held_out_frac=0.2, seed=42)
    X_train, y_train, g_train = build_arrays(all_rows, train_games)
    X_held, y_held, g_held = build_arrays(all_rows, held_games)
    print(f"Train rows: {len(X_train)}, Held rows: {len(X_held)}")

    t0 = time.time()
    class_bal = np.where(y_train == 1, (len(y_train) - y_train.sum()) / max(1, y_train.sum()), 1.0)
    gbdt = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.8,
        min_samples_leaf=10, random_state=42,
    )
    gbdt.fit(X_train, y_train, sample_weight=class_bal)
    t_fit = time.time() - t0
    print(f"Train time: {t_fit:.1f}s")

    scores_held = gbdt.decision_function(X_held)
    auc_held = roc_auc_score(y_held, scores_held) if len(set(y_held)) > 1 else float("nan")
    dec_held = decision_level_eval(scores_held, y_held, g_held)
    print(f"Held decision acc: {dec_held}, row AUC: {auc_held:.4f}")

    with open(os.path.join(HERE, f"{out_prefix}_model.pkl"), "wb") as f:
        pickle.dump(gbdt, f)

    X_all, y_all, g_all = build_arrays(all_rows, train_games | held_games)
    class_bal_all = np.where(y_all == 1, (len(y_all) - y_all.sum()) / max(1, y_all.sum()), 1.0)
    gbdt_final = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.8,
        min_samples_leaf=10, random_state=42,
    )
    gbdt_final.fit(X_all, y_all, sample_weight=class_bal_all)
    with open(os.path.join(HERE, f"{out_prefix}_model_final.pkl"), "wb") as f:
        pickle.dump(gbdt_final, f)

    report = {
        "n_rows": len(all_rows),
        "n_train_rows": len(X_train), "n_held_rows": len(X_held),
        "train_time_s": t_fit,
        "auc_held": auc_held,
        "decision_held": dec_held,
    }
    with open(os.path.join(HERE, f"{out_prefix}_train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out_prefix}_model.pkl / _model_final.pkl / _train_report.json")


if __name__ == "__main__":
    main()
