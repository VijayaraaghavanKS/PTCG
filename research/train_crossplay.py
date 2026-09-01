"""
Train a Dragapult-specific GBDT BC reranker model combining:
  - real ladder-replay rows (agent/bc_dragapult_v3_combined/rows_public.jsonl +
    rows_own_ladder.jsonl -- 50,679 rows / 7,774 groups / 83 real games, the
    project's best real dataset)
  - cross-play rows (research/crossplay_run1/rows_shard_*.jsonl -- decisions
    from the WINNING side of games between DIFFERENT independently-evolved
    Dragapult-line policy variants)

Same architecture/hyperparameters as agent/bc_dragapult_v3_combined/train_combined.py
(GBDT 300 trees depth 4 lr 0.06) and the SAME held-out split (seed=42,
held_out_frac=0.2, split by game_key) so "real_only" reproduces that script's
own 63.5% figure and the crossplay-combined model is evaluated on an IDENTICAL
held-out real-game set -- a genuine apples-to-apples comparison, not a fresh
random split that could differ by luck.

Usage:
  python3 research/train_crossplay.py <mode> <out_prefix> [crossplay_group_cap] [crossplay_weight]
    mode = "real_only" | "crossplay_combined"
"""
import glob
import json
import os
import pickle
import random
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = "/Users/vijayaraaghavanks/Dev/PTCG"
REAL_DIR = os.path.join(ROOT, "agent", "bc_dragapult_v3_combined")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_rows_glob(pattern):
    rows = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
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


def subsample_groups(rows, n_groups, seed=42):
    groups = sorted(set(r["group"] for r in rows))
    if len(groups) <= n_groups:
        return rows
    rng = random.Random(seed)
    rng.shuffle(groups)
    keep = set(groups[:n_groups])
    return [r for r in rows if r["group"] in keep]


def split_groups(rows, held_out_frac=0.2, seed=42):
    games = sorted(set(r["game_key"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(games)
    n_held = max(1, int(len(games) * held_out_frac))
    held = set(games[:n_held])
    train = set(games[n_held:])
    return train, held


def build_arrays(rows, game_set, real_weight=1.0, crossplay_weight=0.15):
    X, y, groups, w = [], [], [], []
    for r in rows:
        if r["game_key"] not in game_set:
            continue
        X.append(r["feat"])
        y.append(r["label"])
        groups.append(r["group"])
        w.append(real_weight if r.get("source") != "crossplay" else crossplay_weight)
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
    crossplay_group_cap = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    crossplay_weight = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

    public_rows = load_rows(os.path.join(REAL_DIR, "rows_public.jsonl"))
    own_rows = load_rows(os.path.join(REAL_DIR, "rows_own_ladder.jsonl"))

    next_off = 1
    next_off = renumber_groups(public_rows, next_off)
    next_off = renumber_groups(own_rows, next_off)
    real_rows = public_rows + own_rows
    print(f"Real rows: {len(real_rows)} ({len(set(r['group'] for r in real_rows))} groups, "
          f"{len(set(r['game_key'] for r in real_rows))} games)")

    # held-out split computed from REAL rows only, BEFORE crossplay rows are
    # added -- guarantees the held-out real-game set is identical between
    # real_only and crossplay_combined runs (same seed=42/frac=0.2 as the
    # original train_combined.py), so the comparison is apples-to-apples.
    train_games, held_games = split_groups(real_rows, held_out_frac=0.2, seed=42)

    all_rows = list(real_rows)
    n_cp_groups_total = 0
    n_cp_kept = 0
    if mode == "crossplay_combined":
        # Pre-subsampled by research/subsample_crossplay.py (memory-safe
        # two-pass streaming sampler -- the raw shard pool has ~3.3M decision
        # groups / 21M rows / 19GB, far too large to load whole into memory).
        cp_sub = load_rows(os.path.join(ROOT, "research", "crossplay_run1", "crossplay_subsampled.jsonl"))
        if crossplay_group_cap < len(set(r["group"] for r in cp_sub)):
            cp_sub = subsample_groups(cp_sub, crossplay_group_cap)
        n_cp_groups_total = 3340901  # from subsample_crossplay.py's pass-1 count, logged for the report
        n_cp_kept = len(set(r["group"] for r in cp_sub))
        next_off = renumber_groups(cp_sub, next_off)
        print(f"Cross-play subsample: {len(cp_sub)} rows ({n_cp_kept} of {n_cp_groups_total} groups), "
              f"weight={crossplay_weight}")
        # crossplay games are ALWAYS added to the train pool only -- never
        # held out (held_games is defined purely over real games), so the
        # held-out eval set stays 100% real and identical across both modes.
        train_games = train_games | set(r["game_key"] for r in cp_sub)
        all_rows += cp_sub

    X_train, y_train, g_train, w_train = build_arrays(all_rows, train_games, crossplay_weight=crossplay_weight)
    X_held, y_held, g_held, w_held = build_arrays(real_rows, held_games)
    print(f"Train rows: {len(X_train)}, Held rows (real-only): {len(X_held)}")

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
    print(f"Held (REAL games only) decision acc: {dec_held}, row AUC: {auc_held:.4f}")

    with open(os.path.join(HERE, f"{out_prefix}_model.pkl"), "wb") as f:
        pickle.dump(gbdt, f)

    # Final deployment model: retrain on ALL available rows (train+held real,
    # plus the same crossplay subsample if applicable) for the shipped artifact.
    final_game_set = train_games | held_games
    X_all, y_all, g_all, w_all = build_arrays(all_rows, final_game_set, crossplay_weight=crossplay_weight)
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
        "n_real_groups": len(set(r["group"] for r in real_rows)),
        "n_real_games": len(set(r["game_key"] for r in real_rows)),
        "n_crossplay_groups_total": n_cp_groups_total,
        "n_crossplay_groups_kept": n_cp_kept,
        "crossplay_weight": crossplay_weight if mode == "crossplay_combined" else None,
        "n_train_rows": len(X_train), "n_held_rows": len(X_held),
        "train_time_s": t_fit,
        "auc_held": auc_held,
        "decision_held_real_only": dec_held,
    }
    with open(os.path.join(HERE, f"{out_prefix}_train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved {out_prefix}_model.pkl, {out_prefix}_model_final.pkl, {out_prefix}_train_report.json")


if __name__ == "__main__":
    main()
