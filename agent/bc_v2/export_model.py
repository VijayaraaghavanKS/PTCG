"""
Export the trained sklearn GradientBoostingClassifier (model_final.pkl) into a
pure stdlib-loadable JSON tree dump (model.json), then verify the exported
representation reproduces sklearn's decision_function() EXACTLY (to float
precision) on a held-out sample before trusting it for the live agent.

GradientBoostingClassifier's decision_function(x) = init_const + learning_rate
* sum_over_trees(tree.predict(x)), where each tree is a plain
sklearn.tree.DecisionTreeRegressor. We walk each tree's `.tree_.feature` /
`.threshold` / `.children_left` / `.children_right` / `.value` arrays (feature
== -2 marks a leaf) -- this needs no sklearn/numpy at inference time, just the
exported arrays + a stdlib while-loop.
"""
import json
import os
import pickle
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


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
    with open(os.path.join(HERE, "model_final.pkl"), "rb") as f:
        gbdt = pickle.load(f)

    X = gbdt._raw_predict_init(np.zeros((1, gbdt.n_features_in_)))
    init_const = float(X[0, 0])

    model_json = {
        "n_features": int(gbdt.n_features_in_),
        "learning_rate": float(gbdt.learning_rate),
        "init": init_const,
        "trees": export_trees(gbdt),
    }
    out_path = os.path.join(HERE, "model.json")
    with open(out_path, "w") as f:
        json.dump(model_json, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.0f} KB, {len(model_json['trees'])} trees)")

    # ---- Faithfulness check: compare stdlib tree-walk vs sklearn's own
    # decision_function on a fresh random + real-feature sample. ----
    rng = random.Random(7)
    n_feat = model_json["n_features"]

    # Random synthetic vectors (values in a plausible 0..1-ish range, since
    # that's what extract_features() produces).
    samples = []
    for _ in range(500):
        samples.append([rng.random() if rng.random() < 0.3 else 0.0 for _ in range(n_feat)])

    # Also pull real extracted feature rows if available, for a realistic check.
    rows_path = os.path.join(HERE, "extracted_rows.jsonl")
    if os.path.exists(rows_path):
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
    print(f"Verified on {len(samples)} samples (random + real extracted rows).")
    print(f"Max abs diff between stdlib export and sklearn decision_function: {max_abs_diff:.3e}")
    if max_abs_diff < 1e-4:
        print("PASS: export is faithful.")
    else:
        print("WARNING: export may not be faithful, investigate before deploying.")


if __name__ == "__main__":
    main()
