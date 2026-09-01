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
bc_reranker_bellibolt_v1: same reranker architecture as bc_reranker_v2 (see
AGENT_LOG.md's "bc_reranker_v2 live result: 848.1" entry -- this is the
direct reason this build exists), ported onto Iono's Bellibolt ex instead of
Dragapult ex. iono_day1's proven hand-tuned heuristic (unmodified logic,
heuristic_scored.py -- a mechanical extraction into compute_option_scores()
+ select_from_scores(), NOT a rewrite) stays the PRIMARY decision-maker for
every choice. The SAME already-trained deck-agnostic BC-GBDT tree ensemble
and feature extractor from bc_reranker_v2 (model.json / features.py, copied
byte-for-byte, not retrained -- see train.py's own framing of this as a
"good decision in general" model, confirmed deck-agnostic by construction:
it never reads card IDs directly, only generic stats/one-hot context) is
layered on top ONLY as a margin-gated reranker for single-pick, non-plan-
critical decisions where the heuristic's own top-2 candidates are close in
score.

Why iono_day1, not bellibolt_v2: bellibolt_v2 adds its own belief-tracking +
fusion-safe search layer on top of this exact same base heuristic (see
AGENT_LOG.md's "Bellibolt v2" entry) -- stacking a second override layer (BC
reranker) on top of an ALREADY self-overriding heuristic would make the
"does reranking corrupt cross-decision state" safety analysis far harder,
and bellibolt_v2's own live debut (478->517.6, well below iono_day1's stable
699.7 peak, plus a known real weak matchup vs Mega Lucario ex) makes it the
less-proven, not more-proven, base. This mirrors bc_reranker_v2's own choice
of dragapult_day1 over dragapult_v2/v3 for the exact same reason.

Cross-decision state check (the whole point of a per-deck safe-context
allowlist -- see AGENT_LOG.md's bc_v1/bc_reranker_v1 postmortem, which
cratered win rate to 0/6 the first time this wasn't respected): read
heuristic_scored.py's docstring for the full analysis. Summary: unlike
Dragapult (plan_a/plan_b) and Mega Lucario (a `plan` AttackPlan global),
Iono/Bellibolt's heuristic has NO persistent attack-plan state at all -- the
only global (`can_attack`) is recomputed fresh from the available options
every MAIN-context call, never from which option actually gets chosen. The
allowlist below is nonetheless kept IDENTICAL to bc_reranker_v2's proven one
(same contexts, same MAIN exclusion) rather than exploiting that extra
safety margin -- this is a faithful port, not a redesign.

Safety: heuristic logic (compute_option_scores/select_from_scores) is called
FIRST and is what's actually returned if the BC layer errors in any way --
this file can never play worse than plain iono_day1 due to a BC bug. Full
3-layer fallback, same as bc_reranker_v2: BC-augmented -> plain
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

# Same allowlist as bc_reranker_v2, MINUS ATTACH_FROM -- see module docstring
# + heuristic_scored.py's docstring for why this deck has no cross-decision
# plan-state hazard at all (that part of the port is a conservative, not a
# load-bearing, restriction). ATTACH_FROM was cut for a DIFFERENT, deck-
# specific reason found via direct empirical testing while building this
# agent (not present in the original bc_reranker_v2 design, which is why
# it's called out here specifically): for Dragapult, attach_score()'s
# candidate scores are widely separated (distinct id-based buckets), so
# "near tie" (TIE_REL_MARGIN=0.15 of the top score) rarely fires there. For
# this deck, ATTACH_FROM's scores cluster tightly around a flat 40000
# baseline with small +1000..+13000 bonuses layered on -- so the SAME margin
# knobs, copied unchanged, flagged 60-75% of ATTACH_FROM decisions as "near
# tied" and let the reranker override roughly half of all energy-attachment
# choices made all game (measured via direct instrumentation: 235-358
# overrides out of ~20 games' worth of decisions). That's a wholesale
# re-decision of the deck's core "which Pokemon gets the energy this turn"
# logic, not an occasional tie-break -- and it measurably lost games (10/50,
# 20% vs iono_day1, well below this build's own ~40% mirror-noise baseline
# measured with the reranker fully disabled). Cutting TIE_REL_MARGIN to 0
# only partially fixed it (489 seen / 314 ties / 235 overrides still, 27.5%
# at n=40) since the absolute-margin-50 clusters are ALSO tight here (many
# genuinely-near-identical attach targets by this heuristic's own coarse
# scoring). Removing ATTACH_FROM from the allowlist (keeping the ORIGINAL
# margin knobs unchanged, since they're not actually the root cause) directly
# fixed it: 25/60 (41.7%) vs iono_day1, back in line with the ~40-50%
# no-regression band. See AGENT_LOG.md-bound report for full instrumentation
# numbers -- this is the kind of per-deck safe-context finding the porting
# task asked to be investigated, not assumed to transfer unchanged.
RERANK_ALLOWED_CONTEXTS = {
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

# Contexts allowed ONLY when every option in the decision targets our own
# side (playerIndex == my_index). SWITCH/TO_ACTIVE were ALSO cut here for the
# same empirically-found reason as ATTACH_FROM above (see that comment):
# iono_day1's SWITCH/TO_ACTIVE scoring is `score -= c.hp; score -= energy*100`
# plus small id-based bonuses -- continuous-valued and often genuinely close
# between two similar-HP benched Pokemon, unlike Dragapult's widely-separated
# id buckets there. Direct instrumentation showed a 67% override rate among
# SWITCH ties (40/59 in one 50-game sample) and a measurable win-rate cost:
# with SWITCH/TO_ACTIVE included, 19/50 (38%) vs iono_day1; with them
# excluded (this final config), 45/100 (45%), back in line with the ~48%
# (72/150) pure-heuristic mirror-noise baseline measured with the reranker
# disabled entirely. Only SETUP_ACTIVE_POKEMON survives from bc_reranker_v2's
# original three-context group -- no Boss's Orders in this deck's decklist,
# so the opponent-side sub-case that group's "own-side-only" restriction was
# built to guard against (in Dragapult's case) never actually occurs here,
# and SETUP_ACTIVE_POKEMON itself showed no override-rate problem (11/21
# ties led to only 2 overrides in the same instrumentation pass).
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
