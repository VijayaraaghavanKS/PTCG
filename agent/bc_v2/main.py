import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.api import all_card_data, all_attack, to_observation_class

"""
bc_v1: behavior-cloning agent. Option-scoring comes from a GradientBoosting
tree ensemble trained offline (see train.py) via imitation learning on real
Kaggle top-episode replays (research/episodes/*/*.json), filtered to ONLY the
eventual WINNER's decisions in each game (unlike the rejected ml_v1 attempt,
which trained on unfiltered winner+loser decisions with a linear model).

Deck-agnostic model: only 2 of the 61 usable replay games had a Dragapult ex
winner (nowhere near enough data for an archetype-specific model), so this
was trained as a general "what would a winning player do" scorer using the
same deck-agnostic feature extractor as ml_v1 (features.py, unmodified),
following the task's documented fallback plan.

For every legal Option, extract_features() builds a fixed-length vector; the
exported tree ensemble (model.json, pure data -- no sklearn/numpy needed)
scores it via a stdlib tree-walk faithfully reproducing the original sklearn
GradientBoostingClassifier.decision_function() (verified to ~1e-10 agreement
offline, see export_model.py). We act on the highest-scoring option(s).

Safety: the real Kaggle sandbox is stdlib-only, so this file has zero
third-party runtime dependencies. Per this project's hard "must never crash"
rule, EVERY decision is wrapped in a 3-layer fallback:
  1. BC model scoring (primary).
  2. dragapult_day1's proven hand-tuned heuristic (heuristic_fallback.py),
     imported unmodified, only invoked if (1) raises.
  3. Trivial "select the first minCount options" (guaranteed legal), only
     invoked if BOTH (1) and (2) raise.
"""

# Load deck.csv in the dataset
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

# Card / attack metadata tables (global, deck-independent)
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
all_atk = all_attack()
attack_table = {a.attackId: a for a in all_atk}

_MODEL = None
_MODEL_LOAD_ERROR = None
try:
    model_path = "model.json"
    if not os.path.exists(model_path):
        model_path = "/kaggle_simulations/agent/" + model_path
    with open(model_path, "r") as f:
        _MODEL = json.load(f)
except Exception as e:  # noqa: BLE001
    _MODEL_LOAD_ERROR = e
    _MODEL = None

try:
    from features import extract_features, FEATURE_DIM
except Exception:  # noqa: BLE001
    extract_features = None
    FEATURE_DIM = None

try:
    from heuristic_fallback import heuristic_agent
except Exception:  # noqa: BLE001
    heuristic_agent = None


def _tree_predict_one(tree, x):
    node = 0
    feature = tree["feature"]
    threshold = tree["threshold"]
    left = tree["left"]
    right = tree["right"]
    value = tree["value"]
    # Bounded loop as an extra safety net against any malformed/cyclic export.
    for _ in range(64):
        f = feature[node]
        if f == -2:
            return value[node]
        if x[f] <= threshold[node]:
            node = left[node]
        else:
            node = right[node]
    return value[node]


def score_option(obs, option) -> float:
    feats = extract_features(obs, option, card_table, attack_table)
    if len(feats) != _MODEL["n_features"]:
        raise ValueError("feature dim mismatch")
    total = 0.0
    for tree in _MODEL["trees"]:
        total += _tree_predict_one(tree, feats)
    return _MODEL["init"] + _MODEL["learning_rate"] * total


def _bc_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    select = obs.select
    options = select.option
    if not options:
        return []

    scores = [score_option(obs, o) for o in options]
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    max_c = select.maxCount if select.maxCount is not None else 1
    k = max(min(max_c, len(options)), 0)
    return desc_indices[:k]


def _trivial_fallback(obs_dict: dict) -> list[int]:
    """Last-resort, dependency-free legal move: select() is unavailable so
    fall back to the raw dict shape directly."""
    sel = obs_dict.get("select")
    if sel is None:
        return my_deck
    options = sel.get("option") or []
    min_c = sel.get("minCount") or 0
    n = max(min_c, 1 if options else 0)
    n = min(n, len(options))
    return list(range(n))


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount
    (inclusive), with no duplicate elements.
    """
    if _MODEL is not None and extract_features is not None:
        try:
            return _bc_agent(obs_dict)
        except Exception:  # noqa: BLE001
            pass

    if heuristic_agent is not None:
        try:
            return heuristic_agent(obs_dict)
        except Exception:  # noqa: BLE001
            pass

    return _trivial_fallback(obs_dict)
