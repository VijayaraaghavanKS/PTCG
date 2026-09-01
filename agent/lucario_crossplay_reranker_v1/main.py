import importlib.util
import json
import os
import sys

from cg.api import AreaType, OptionType, SelectContext, all_card_data, all_attack, to_observation_class

# The Kaggle sandbox execs main.py's source directly (no proper module load),
# so __file__ is undefined here -- unlike a normal script run, which is why
# this can't use os.path.dirname(os.path.abspath(__file__)) (that crashed a
# live submission, see AGENT_LOG.md: "NameError: name '__file__' is not
# defined"). Deriving the directory from an imported package's __file__ (e.g.
# cg.__file__) is ALSO unreliable for local testing specifically: this
# project's harness (run_match.py) does its own top-level `from cg.game
# import ...` before loading any agent, so `cg` ends up cached in
# sys.modules pointing at the HARNESS's copy, not this agent's -- meaning a
# derived path would silently point at the wrong directory there (this
# doesn't affect the real Kaggle sandbox, which only ever loads one agent
# per process, but it means that particular approach can't be trusted locally
# either). Sidestep both problems entirely: load features.py/heuristic_scored.py
# directly by file path, reusing the exact same bare-relative-path-with-
# "/kaggle_simulations/agent/"-fallback pattern already proven correct for
# deck.csv/model.json below, instead of relying on sys.path or __file__ at all.
def _load_sibling_module(mod_name):
    path = f"{mod_name}.py"
    if not os.path.exists(path):
        path = f"/kaggle_simulations/agent/{mod_name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

"""
bc_reranker_lucario_v1: same reranker architecture as bc_reranker_v2 (see
AGENT_LOG.md's "bc_reranker_v2 live result: 848.1" entry -- this is the
direct reason this build exists), ported onto Mega Lucario ex instead of
Dragapult ex. mega_lucario_ref's hand-tuned heuristic (unmodified logic,
heuristic_scored.py -- a mechanical extraction into compute_option_scores()
+ select_from_scores(), NOT a rewrite) stays the PRIMARY decision-maker for
every choice. The SAME already-trained deck-agnostic BC-GBDT tree ensemble
and feature extractor from bc_reranker_v2 (model.json / features.py, copied
byte-for-byte, not retrained) is layered on top ONLY as a margin-gated
reranker for single-pick, non-plan-critical decisions where the heuristic's
own top-2 candidates are close in score.

Why mega_lucario_ref, not mega_lucario_v2: mega_lucario_v2 already layers
its own belief-tracking + fusion-safe search override on top of this same
base heuristic (see AGENT_LOG.md's "Mega Lucario v2" entry). Stacking a
SECOND independent override layer (this BC reranker) on top of a heuristic
that already conditionally overrides its own top pick via search would make
both the "is this decision safe to rerank" analysis and the interaction
between the two override layers (which one wins, does the reranker's
near-tie margin even mean anything once search has already picked a
non-obvious move) much harder to reason about correctly under this
project's deadline. This mirrors bc_reranker_v2's own choice of
dragapult_day1 over dragapult_v2/v3 for the identical reason -- always layer
BC-reranking onto the simplest proven single-heuristic decision-maker for a
given archetype, not onto a build that already has its own override layer.

Cross-decision state check (see heuristic_scored.py's docstring for the full
analysis): Mega Lucario's heuristic has real cross-decision plan state (a
`plan` AttackPlan global), and -- unlike Dragapult, where only the
OPPONENT-side branch of SWITCH/TO_ACTIVE read plan state -- here BOTH the
own-side and opponent-side branches read `plan.attacker`/`plan.target`. This
means the "own-side-only" restriction that made SWITCH/TO_ACTIVE safe for
Dragapult is NOT sufficient here. Consequently SWITCH/TO_ACTIVE are excluded
from the allowlist entirely for this deck (not even own-side-only) -- only
SETUP_ACTIVE_POKEMON (no plan reference, occurs before any MAIN decision
this game) is kept from that group.

Safety: heuristic logic (compute_option_scores/select_from_scores) is called
FIRST and is what's actually returned if the BC layer errors in any way --
this file can never play worse than plain mega_lucario_ref due to a BC bug.
Full 3-layer fallback, same as bc_reranker_v2: BC-augmented -> plain
heuristic_agent() -> trivial legal-move fallback.
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
    extract_features = _load_sibling_module("features").extract_features
except Exception:  # noqa: BLE001
    extract_features = None

_heuristic_mod = _load_sibling_module("heuristic_scored")
compute_option_scores = _heuristic_mod.compute_option_scores
select_from_scores = _heuristic_mod.select_from_scores
heuristic_agent = _heuristic_mod.heuristic_agent

# --- Reranker tuning knobs (same values as bc_reranker_v2) ---
TOP_K = 3          # consider at most this many top-heuristic-scoring candidates
TIE_ABS_MARGIN = 50.0    # OR gap under this absolute amount counts as "close"
TIE_REL_MARGIN = 0.15    # gap under this fraction of the top score counts as "close"

# See module docstring + heuristic_scored.py's docstring for why each of
# these is safe to include. MAIN and SWITCH/TO_ACTIVE are deliberately NOT
# here -- MAIN builds+consumes `plan` in one call (excluded on architectural
# principle, same as the other two rerankers), and SWITCH/TO_ACTIVE both
# consume `plan.attacker`/`plan.target` set by a PRIOR MAIN decision this
# turn on BOTH the own-side and opponent-side branches (unlike Dragapult,
# where only the opponent-side branch did) -- so no own-side-only carve-out
# is safe here the way it was for bc_reranker_v2.
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

# Only SETUP_ACTIVE_POKEMON survives from bc_reranker_v2's "own-side-only"
# group for this deck -- see docstring above for why SWITCH/TO_ACTIVE don't.
RERANK_ALLOWED_CONTEXTS_OWN_SIDE_ONLY = {
    SelectContext.SETUP_ACTIVE_POKEMON,
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


def _context_is_rerankable(select, my_index) -> bool:
    ctx = select.context
    if ctx in RERANK_ALLOWED_CONTEXTS:
        return True
    if ctx in RERANK_ALLOWED_CONTEXTS_OWN_SIDE_ONLY:
        for o in select.option:
            pidx = o.playerIndex if o.playerIndex is not None else my_index
            if pidx != my_index:
                return False
        return True
    return False


def _bc_augmented_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    select = obs.select
    my_index = obs.current.yourIndex
    scores = compute_option_scores(obs)

    if (_MODEL is not None and extract_features is not None
            and select.maxCount == 1 and len(select.option) >= 2
            and _context_is_rerankable(select, my_index)):
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

    return select_from_scores(obs, scores)


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
