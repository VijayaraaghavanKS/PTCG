import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.api import AreaType, OptionType, SelectContext, all_card_data, all_attack, to_observation_class

"""
bc_reranker_v1: dragapult_day1's proven hand-tuned heuristic (unmodified logic,
see heuristic_scored.py -- a mechanical extraction into compute_option_scores()
+ select_from_scores(), NOT a rewrite) stays the primary decision-maker for
every choice. The behavior-cloning tree ensemble (same model as bc_v1, trained
on win-filtered real ladder replays) is layered on top ONLY as a margin-gated
reranker for single-pick (maxCount==1), non bench-setup decisions where the
heuristic's own top-2 candidates are close in score ("genuinely a close call")
-- exactly the untried direction flagged in AGENT_LOG.md's ml_v1 postmortem,
and the same additive "propose top-K, override only on a wide margin" pattern
already proven safe by this project's v2 search upgrades (dragapult_v2,
mega_lucario_v2, bellibolt_v2).

Why this design: bc_v1 (BC model as the SOLE policy) tested at 5-15% win rate
locally -- pure imitation learning can't replicate the heuristic's explicit
multi-turn KO-sequencing/lethal-detection logic from a pointwise per-option
classifier trained on only ~60 games. This reranker can never do worse than
the heuristic on the (large majority of) decisions with a clear heuristic
winner, since those never enter the tie-break; it can only change outcomes on
genuinely close calls, where the BC model's real-player-imitation signal is
most plausibly useful and least likely to override correct tactical logic.

Safety: heuristic logic (compute_option_scores/select_from_scores) is called
FIRST and is what's actually returned if the BC layer errors in any way --
this file can never play worse than plain dragapult_day1 due to a BC bug.
Full 3-layer fallback, same as bc_v1: BC-augmented -> plain heuristic_agent()
-> trivial legal-move fallback.
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

from heuristic_scored import compute_option_scores, select_from_scores, heuristic_agent  # noqa: E402

# --- Reranker tuning knobs ---
TOP_K = 3          # consider at most this many top-heuristic-scoring candidates
TIE_ABS_MARGIN = 50.0    # OR gap under this absolute amount counts as "close"
TIE_REL_MARGIN = 0.15    # gap under this fraction of the top score counts as "close"

# IMPORTANT: dragapult_day1's heuristic is NOT a stateless per-option scorer --
# several decision types share cross-decision plan state (module globals
# `plan_a`/`plan_b`/`bench_attacker`/`can_switch` etc, computed in
# main_option_proc() and consumed by LATER selects in the same turn, e.g. which
# bench Pokemon to place damage counters on to complete an already-decided
# attack plan). Overriding the heuristic's pick on one of those plan-linked
# decisions (measured in an early build: turn-action choice (MAIN) and
# damage-counter placement together accounted for ~80% of overrides and this
# scope brought the mirror-match win rate to a flat, unproven 50%) risks
# leaving the rest of the turn acting on stale plan state. So reranking is
# restricted to an ALLOWLIST of contexts that are self-contained, single-shot
# choices with no such downstream dependency.
RERANK_ALLOWED_CONTEXTS = {
    SelectContext.ATTACH_FROM,       # which Pokemon to attach an energy/tool to
    SelectContext.TO_HAND,           # which card to add to hand
    SelectContext.DISCARD,           # which card in play to discard
    SelectContext.TO_DECK,           # which card to shuffle back into the deck
    SelectContext.TO_DECK_BOTTOM,
    SelectContext.TO_PRIZE,
    SelectContext.NOT_MOVE,
    SelectContext.HEAL,
    SelectContext.REMOVE_DAMAGE_COUNTER,
    SelectContext.EFFECT_TARGET,
    SelectContext.LOOK,
    SelectContext.DISCARD_ENERGY_CARD,
    SelectContext.DISCARD_TOOL_CARD,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
}


def _is_near_tie(top1, top2):
    if top2 is None:
        return False
    gap = top1 - top2
    if gap <= TIE_ABS_MARGIN:
        return True
    if top1 > 0 and (gap / abs(top1)) <= TIE_REL_MARGIN:
        return True
    return False


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


def _bc_score(obs, option) -> float:
    feats = extract_features(obs, option, card_table, attack_table)
    if len(feats) != _MODEL["n_features"]:
        raise ValueError("feature dim mismatch")
    total = 0.0
    for tree in _MODEL["trees"]:
        total += _tree_predict_one(tree, feats)
    return _MODEL["init"] + _MODEL["learning_rate"] * total


def _bc_augmented_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    select = obs.select
    scores = compute_option_scores(obs)

    if (_MODEL is not None and extract_features is not None
            and select.maxCount == 1 and len(select.option) >= 2
            and select.context in RERANK_ALLOWED_CONTEXTS):
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top1_idx = ranked[0]
        top2_score = scores[ranked[1]] if len(ranked) > 1 else None
        if _is_near_tie(scores[top1_idx], top2_score):
            candidate_idxs = ranked[:TOP_K]
            try:
                bc_scores = {i: _bc_score(obs, select.option[i]) for i in candidate_idxs}
                best_i = max(candidate_idxs, key=lambda i: bc_scores[i])
                if best_i != top1_idx:
                    return [best_i]
            except Exception:  # noqa: BLE001
                pass  # fall through to plain heuristic selection below

    return select_from_scores(select, scores)


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
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount
    (inclusive), with no duplicate elements.
    """
    try:
        return _bc_augmented_agent(obs_dict)
    except Exception:  # noqa: BLE001
        pass

    try:
        return heuristic_agent(obs_dict)
    except Exception:  # noqa: BLE001
        pass

    return _trivial_fallback(obs_dict)
