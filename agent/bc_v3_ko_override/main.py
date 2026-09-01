import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.api import all_card_data, all_attack, to_observation_class, OptionType, Pokemon

"""
bc_v3_ko_override: identical to bc_v2 (pure BC decision policy, same model,
same features, same 3-layer fallback) EXCEPT for one minimal addition on top
of the BC ranking: if any legal option in an ATTACK decision is a CONFIRMED
KO on the opponent's active Pokemon (computed directly from engine state --
not learned, not a heuristic plan, just "would this attack's damage meet or
exceed the target's current HP"), always take the best such KO (highest
prize value, tie-broken by BC score) instead of whatever the BC model
would have picked on its own.

Purpose: bc_v2 (pure BC, no override) and bc_reranker_v2 (heuristic-
primary, BC-tiebreak-only) sit at two extremes. This tests a narrow,
specific hypothesis from the bc_v1/bc_v2 postmortems -- that the model's
weak local win rate comes SPECIFICALLY from failing to reliably take
available KOs (despite the lethal-aware features already being in its
input vector), not from a broader failure to sequence/plan. If adding
*only* a hard KO-override meaningfully closes the gap, that pinpoints the
missing piece as "step 1 of KO-sequencing" specifically. If it barely moves
the needle, the deficit is broader than just missing KOs.

This is NOT a heuristic in the ga_v1/dragapult_day1 sense -- it has no
plan state, no evolve/attach/retreat logic, no draw-supporter prioritization,
nothing beyond "take the KO if one exists, else trust the BC model for
literally everything else" (which is exactly what bc_v2 already does when
no KO is available).
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
except Exception:  # noqa: BLE001
    _MODEL = None

try:
    from features import extract_features
except Exception:  # noqa: BLE001
    extract_features = None

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


def _effective_damage(attacker_data, atk, target_data):
    if attacker_data is None or atk is None or target_data is None:
        return atk.damage if atk is not None else 0.0
    dmg = float(atk.damage or 0)
    if dmg <= 0:
        return dmg
    if target_data.weakness is not None and target_data.weakness == attacker_data.energyType:
        dmg *= 2
    elif target_data.resistance is not None and target_data.resistance == attacker_data.energyType:
        dmg = max(0.0, dmg - 30)
    return dmg


def _prize_value(card_data):
    if card_data is None:
        return 1
    if getattr(card_data, "megaEx", False):
        return 3
    if getattr(card_data, "ex", False):
        return 2
    return 1


def _confirmed_ko_options(obs, options):
    """Indices of ATTACK options that would KO the opponent's active Pokemon
    right now, computed directly from engine state (weakness/resistance-aware,
    generic -- same math as features.py's lethal block, kept independent here
    on purpose so this override never silently depends on the model/feature
    pipeline it's meant to be a check on)."""
    state = obs.current
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None
    if my_active is None or op_active is None:
        return []
    op_active_data = card_table.get(op_active.id)
    if op_active_data is None or not op_active.hp or op_active.hp <= 0:
        return []
    my_active_data = card_table.get(my_active.id)

    ko_indices = []
    for i, o in enumerate(options):
        if o.type != OptionType.ATTACK or o.attackId is None:
            continue
        atk = attack_table.get(o.attackId)
        if atk is None:
            continue
        dmg = _effective_damage(my_active_data, atk, op_active_data)
        if dmg >= op_active.hp:
            ko_indices.append(i)
    return ko_indices


def _bc_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    select = obs.select
    options = select.option
    if not options:
        return []

    scores = [score_option(obs, o) for o in options]

    # --- minimal KO override: if a legal option is a confirmed lethal KO on
    # the opponent's active, always take the best one (by prize value, then
    # BC score), regardless of what the BC model itself ranked highest.
    ko_indices = []
    try:
        ko_indices = _confirmed_ko_options(obs, options)
    except Exception:  # noqa: BLE001
        ko_indices = []

    if ko_indices:
        state = obs.current
        op_state = state.players[1 - state.yourIndex]
        op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None
        op_active_data = card_table.get(op_active.id) if op_active is not None else None
        pv = _prize_value(op_active_data)
        best_ko = max(ko_indices, key=lambda i: (pv, scores[i]))
        desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
        remaining = [i for i in desc_indices if i != best_ko]
        desc_indices = [best_ko] + remaining
    else:
        desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    max_c = select.maxCount if select.maxCount is not None else 1
    k = max(min(max_c, len(options)), 0)
    return desc_indices[:k]


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
        except Exception:  # noqa: BLE001
            pass

    if heuristic_agent is not None:
        try:
            return heuristic_agent(obs_dict)
        except Exception:  # noqa: BLE001
            pass

    return _trivial_fallback(obs_dict)
