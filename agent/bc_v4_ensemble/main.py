import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.api import all_card_data, all_attack, to_observation_class

"""
bc_v4_ensemble: a genuinely different architecture from both bc_v2 (pure BC
policy) and bc_reranker_v2 (heuristic-primary, BC-tiebreak-only-on-safe-
near-ties). Here neither model is "primary" -- dragapult_day1's proven
heuristic (heuristic_scored.py, a copy of bc_v2's heuristic_fallback.py with
one additive change: it exposes its own raw per-option score vector via
_LAST_SCORES instead of only its own committed top-K choice) and the BC
GBDT model each independently rank every legal option for a decision. The
agent picks whichever option has the best COMBINED rank (lowest sum of the
two 0-indexed rank positions, heuristic rank + BC rank, ties broken by
heuristic rank since it's the better-understood/more reliable signal).

Why this is worth testing separately from the reranker: bc_reranker_v2 only
lets the BC model act when the heuristic is already "confused" (near-tied
top options in a restricted safe-context allowlist) -- everywhere else the
heuristic's own top pick wins outright, no matter how the BC model would
have ranked the alternatives. A rank-sum vote instead lets the BC model
exert a *graded* pull on every decision (nudging the pick toward whatever
the BC model likes second-most when the heuristic's own top-2/3 are close),
without ever fully overriding the heuristic's clearly-best option (since a
huge heuristic score gap still means heuristic rank 0 vs everything else,
which the BC model's rank alone is very unlikely to overcome in a sum).

Safety: same 3-layer fallback convention as bc_v2/bc_reranker_v2 --
(1) ensemble scoring, (2) plain heuristic_agent() call (this file's own
heuristic_scored.py, called directly -- its return value is unaffected by
the _LAST_SCORES capture), (3) trivial first-legal-options fallback.
"""

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = [int(csv[i]) for i in range(60)]

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
except Exception:  # noqa: BLE001
    _MODEL = None

try:
    from features import extract_features
except Exception:  # noqa: BLE001
    extract_features = None

try:
    import heuristic_scored
    heuristic_agent = heuristic_scored.heuristic_agent
except Exception:  # noqa: BLE001
    heuristic_scored = None
    heuristic_agent = None


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


def score_option(obs, option) -> float:
    feats = extract_features(obs, option, card_table, attack_table)
    if len(feats) != _MODEL["n_features"]:
        raise ValueError("feature dim mismatch")
    total = 0.0
    for tree in _MODEL["trees"]:
        total += _tree_predict_one(tree, feats)
    return _MODEL["init"] + _MODEL["learning_rate"] * total


def _ranks_from_scores(scores):
    """0-indexed rank per option, best score = rank 0. Stable on ties (earlier
    option index wins ties, matching this project's other agents' sort
    convention)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def _ensemble_refine(obs_dict: dict, heuristic_choice: list[int]) -> list[int]:
    """Given the heuristic's own already-computed choice (and the _LAST_SCORES
    side effect it just left behind), try to replace it with the combined-rank
    ensemble pick. Never re-invokes heuristic_agent() -- that must run exactly
    once per decision (it has cumulative turn/plan/log state), so any failure
    here just falls back to the heuristic_choice already in hand, not a fresh
    heuristic call."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return heuristic_choice

    select = obs.select
    options = select.option
    if not options:
        return heuristic_choice

    h_scores = heuristic_scored._LAST_SCORES
    if h_scores is None or len(h_scores) != len(options):
        return heuristic_choice

    bc_scores = [score_option(obs, o) for o in options]

    h_ranks = _ranks_from_scores(h_scores)
    bc_ranks = _ranks_from_scores(bc_scores)
    combined = [(h_ranks[i] + bc_ranks[i], h_ranks[i], i) for i in range(len(options))]
    combined.sort()
    order = [c[2] for c in combined]

    max_c = select.maxCount if select.maxCount is not None else 1
    k = max(min(max_c, len(options)), 0)
    out = order[:k]
    while len(out) < select.minCount:
        for i in order:
            if i not in out:
                out.append(i)
                break
        else:
            break
    return out


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
    if heuristic_agent is not None:
        try:
            heuristic_choice = heuristic_agent(obs_dict)  # runs exactly once per decision
        except Exception:  # noqa: BLE001
            heuristic_choice = None

        if heuristic_choice is not None:
            if _MODEL is not None and extract_features is not None:
                try:
                    return _ensemble_refine(obs_dict, heuristic_choice)
                except Exception:  # noqa: BLE001
                    pass
            return heuristic_choice

    return _trivial_fallback(obs_dict)
