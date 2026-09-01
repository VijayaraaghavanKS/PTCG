import json
import sys

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import numpy as np

rows = []
with open("turn_rows_all.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))

X = np.array([r["feat"] for r in rows])
y = np.array([r["label"] for r in rows])
groups = np.array([r["game"] for r in rows])

gkf = GroupKFold(n_splits=5)
maes = []
baseline_maes = []
for tr, te in gkf.split(X, y, groups):
    model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    maes.append(mean_absolute_error(y[te], pred))
    baseline_maes.append(mean_absolute_error(y[te], np.full_like(y[te], y[tr].mean())))

print(f"GroupKFold MAE (model): {np.mean(maes):.3f} vs mean-baseline: {np.mean(baseline_maes):.3f}")

# final model on all data
final = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0)
final.fit(X, y)

# export to plain-stdlib-walkable JSON (sklearn tree structure)
trees = []
for est in final.estimators_[:, 0]:
    tree = est.tree_
    trees.append({
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, 0].tolist(),
    })

export = {
    "init_value": float(final.init_.constant_[0][0]) if hasattr(final.init_, "constant_") else float(np.mean(y)),
    "learning_rate": final.learning_rate,
    "trees": trees,
    "feature_names": ["pre_my_hp", "pre_op_hp", "pre_my_prize_left", "pre_op_prize_left",
                       "pre_my_bench_n", "pre_op_bench_n", "pre_my_energy_total", "turn_num",
                       "attacked", "n_evolves", "n_attaches", "n_trainers", "n_ability"],
}
with open("value_model.json", "w") as f:
    json.dump(export, f)

# verify faithful export
import pickle
with open("value_model.pkl", "wb") as f:
    pickle.dump(final, f)

def walk_tree(t, x):
    node = 0
    while t["children_left"][node] != -1:
        f = t["feature"][node]
        thr = t["threshold"][node]
        node = t["children_left"][node] if x[f] <= thr else t["children_right"][node]
    return t["value"][node]

max_diff = 0.0
for i in range(min(200, len(X))):
    pred_sklearn = final.predict(X[i:i+1])[0]
    pred_manual = export["init_value"] + export["learning_rate"] * sum(walk_tree(t, X[i]) for t in export["trees"])
    max_diff = max(max_diff, abs(pred_sklearn - pred_manual))
print(f"export faithfulness max abs diff: {max_diff:.2e}")
print(f"n_rows={len(rows)} n_games={len(set(groups))}")
