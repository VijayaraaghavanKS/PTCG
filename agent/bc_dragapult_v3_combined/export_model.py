"""
Export a trained sklearn GradientBoostingClassifier (final.pkl) into a pure
stdlib-loadable JSON tree dump, verified faithful against sklearn's own
decision_function(). Adapted from bc_v2/export_model.py to take explicit
in/out paths (this dir trains multiple model variants).

Usage: python3 export_model.py <model_final.pkl> <out_model.json> [<rows.jsonl for faithfulness check>]
"""
import json
import os
import pickle
import random
import sys

import numpy as np


def export_trees(gbdt):
    trees = []
    for i in range(gbdt.n_estimators):
        t = gbdt.estimators_[i, 0].tree_
        trees.append({
            "feature": [int(v) for v in t.feature],
            "threshold": [round(float(v), 8) for v in t.threshold],
            "left": [int(v) for v in t.children_left],
            "right": [int(v) for v in t.children_right],
            "value": [round(float(v), 10) for v in t.value[:, 0, 0]],
        })
    return trees


def tree_predict_one(tree_json, x):
    node = 0
    feature = tree_json["feature"]
    threshold = tree_json["threshold"]
    left = tree_json["left"]
    right = tree_json["right"]
    value = tree_json["value"]
    while feature[node] != -2:
        f = feature[node]
        if x[f] <= threshold[node]:
            node = left[node]
        else:
            node = right[node]
    return value[node]


def gbdt_score_stdlib(model_json, x):
    s = model_json["init"]
    lr = model_json["learning_rate"]
    total = 0.0
    for t in model_json["trees"]:
        total += tree_predict_one(t, x)
    return s + lr * total


def main():
    pkl_path = sys.argv[1]
    out_path = sys.argv[2]
    rows_path = sys.argv[3] if len(sys.argv) > 3 else None

    with open(pkl_path, "rb") as f:
        gbdt = pickle.load(f)

    X = gbdt._raw_predict_init(np.zeros((1, gbdt.n_features_in_)))
    init_const = float(X[0, 0])

    model_json = {
        "n_features": int(gbdt.n_features_in_),
        "learning_rate": float(gbdt.learning_rate),
        "init": init_const,
        "trees": export_trees(gbdt),
    }
    with open(out_path, "w") as f:
        json.dump(model_json, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.0f} KB, {len(model_json['trees'])} trees)")

    rng = random.Random(7)
    n_feat = model_json["n_features"]
    samples = []
    for _ in range(500):
        samples.append([rng.random() if rng.random() < 0.3 else 0.0 for _ in range(n_feat)])
    if rows_path and os.path.exists(rows_path):
        with open(rows_path) as f:
            lines = f.readlines()
        rng.shuffle(lines)
        for line in lines[:500]:
            row = json.loads(line)
            samples.append(row["feat"])

    Xs = np.array(samples, dtype=np.float32)
    sk_scores = gbdt.decision_function(Xs)
    max_abs_diff = 0.0
    for x, sk_s in zip(samples, sk_scores):
        my_s = gbdt_score_stdlib(model_json, x)
        diff = abs(my_s - float(sk_s))
        max_abs_diff = max(max_abs_diff, diff)
    print(f"Max abs diff (stdlib vs sklearn): {max_abs_diff:.3e}")
    print("PASS" if max_abs_diff < 1e-4 else "WARNING: export may not be faithful")


if __name__ == "__main__":
    main()
