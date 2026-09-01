import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.api import all_card_data, all_attack, to_observation_class

"""
bc_plan_conditioned_v1: pure-BC-primary agent, same GBDT-tree-ensemble
architecture as bc_dragapult_v3_combined/primary_test, but the feature
vector includes 6 additional "what's already committed this turn" features
(see features.py's TurnContextTracker) alongside the proven 138-dim base
vector. Tests whether the 4x-confirmed pure-BC failure (5-24% win rate,
bc_v1/bc_v2/bc_v3_ko_override/bc_v4_ensemble/bc_dragapult_v3_combined's own
primary test) is fixable by giving the pointwise classifier turn-history
context, rather than more data/better lethal features (both already tried,
neither moved the needle).

The tracker is maintained as global state across agent() calls within a
single match (reset on state.turn==0 or a turn-number change), and is
updated using the model's OWN chosen action after each decision -- causal,
no lookahead, matches how a live agent can only ever know its own past
actions this turn.

Safety: identical 3-layer fallback pattern to primary_test/main.py.
"""

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
all_atk = all_attack()
attack_table = {a.attackId: a for a in all_atk}

_MODEL = None
try:
    model_path = "model.json"
    if not os.path.exists(model_path):
        model_path = "/kaggle_simulations/agent/" + model_path
    with open(model_path, "r") as f:
        _MODEL = json.load(f)
except Exception:
    _MODEL = None

try:
    from features import extract_features, FEATURE_DIM, TurnContextTracker
except Exception:
    extract_features = None
    FEATURE_DIM = None
    TurnContextTracker = None

try:
    from heuristic_fallback import heuristic_agent
except Exception:
    heuristic_agent = None

_tracker = None  # persists across agent() calls within a match


def _get_tracker():
    global _tracker
    if _tracker is None and TurnContextTracker is not None:
        _tracker = TurnContextTracker()
    return _tracker


def _tree_predict_one(tree, x):
    node = 0
    feature = tree["feature"]
    threshold = tree["threshold"]
    left = tree["left"]
    right = tree["right"]
    value = tree["value"]
    for _ in range(64):
        f = feature[node]
        if f == -2:
            return value[node]
        if x[f] <= threshold[node]:
            node = left[node]
        else:
            node = right[node]
    return value[node]


def score_option(obs, option, tracker) -> float:
    feats = extract_features(obs, option, card_table, attack_table, tracker)
    if len(feats) != _MODEL["n_features"]:
        raise ValueError("feature dim mismatch")
    total = 0.0
    for tree in _MODEL["trees"]:
        total += _tree_predict_one(tree, feats)
    return _MODEL["init"] + _MODEL["learning_rate"] * total


def _bc_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        global _tracker
        _tracker = None  # new match starting
        return my_deck

    select = obs.select
    options = select.option
    if not options:
        return []

    tracker = _get_tracker()

    scores = [score_option(obs, o, tracker) for o in options]
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    max_c = select.maxCount if select.maxCount is not None else 1
    k = max(min(max_c, len(options)), 0)
    chosen = desc_indices[:k]

    # Advance the tracker using OUR OWN chosen option(s) -- causal, only
    # informs decisions later in this same turn.
    if tracker is not None:
        try:
            for idx in chosen:
                tracker.observe_chosen(obs, options[idx])
        except Exception:
            pass

    return chosen


def _trivial_fallback(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select")
    if sel is None:
        return my_deck
    options = sel.get("option") or []
    min_c = sel.get("minCount") or 0
    n = max(min_c, 1 if options else 0)
    n = min(n, len(options))
    return list(range(n))


def agent(obs_dict: dict) -> list[int]:
    if _MODEL is not None and extract_features is not None:
        try:
            return _bc_agent(obs_dict)
        except Exception:
            pass

    if heuristic_agent is not None:
        try:
            return heuristic_agent(obs_dict)
        except Exception:
            pass

    return _trivial_fallback(obs_dict)
