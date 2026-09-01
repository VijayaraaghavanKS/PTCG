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
bc_reranker_v2: direct follow-up to bc_reranker_v1 (see AGENT_LOG.md's "bc_v1
and bc_reranker_v1" entry). dragapult_day1's proven hand-tuned heuristic
(unmodified logic, heuristic_scored.py -- a mechanical extraction into
compute_option_scores() + select_from_scores(), NOT a rewrite) stays the
PRIMARY decision-maker for every choice. The behavior-cloning tree ensemble
(GBDT trained offline on real ladder replays -- see train.py -- now on a
larger pull of episodes AND with an added lethal/KO-aware feature block,
features.py) is layered on top ONLY as a margin-gated reranker for
single-pick, non-plan-linked decisions where the heuristic's own top-2
candidates are close in score.

v2 changes vs v1:
  1. Larger training set (see train_report.json for the actual episode/row
     counts used) -- v1 was trained on only 61 games / 34K rows.
  2. features.py now includes 8 explicit lethal/KO features (effective
     weakness-aware damage, is-this-a-KO, prize value, "does this win the
     game", opponent's next-turn threat, "are we in KO range") instead of
     forcing the model to infer that from generic stats.
  3. The safe-context allowlist is widened by exactly ONE case, justified by
     re-reading heuristic_scored.py's main_option_proc(): the SWITCH /
     TO_ACTIVE / SETUP_ACTIVE_POKEMON select context is used for TWO
     unrelated sub-decisions that happen to share a context enum --
     (a) picking OUR OWN new active Pokemon (self-contained, no downstream
     plan dependency), and (b) selecting the OPPONENT's Pokemon to force
     active via Boss's Orders (which reads `plan_a.attack`, set earlier this
     turn by main_option_proc() during the MAIN-context decision, and is
     exactly the kind of cross-decision plan-state dependency that broke
     v1's win rate to 0/6 when it was in scope). We only add this context to
     the allowlist when EVERY option in the decision has playerIndex ==
     my_index -- i.e. it is structurally impossible for this to be the
     Boss's-Orders-target sub-case, since that sub-case's options are all
     opponent-side (playerIndex != my_index). This lets the reranker use the
     new switch_target_ko_range feature on "which of my own Pokemon do I
     send up" decisions without touching the plan-linked branch at all.
     MAIN (turn-action choice) and DAMAGE_COUNTER/DAMAGE_COUNTER_ANY
     (damage-counter placement, reads plan_b.counter) remain excluded, same
     as v1 -- those are the ones that actually broke the heuristic's plan
     state when overridden.

Why this design (unchanged from v1): bc_v2 (BC model as the SOLE policy,
tested separately) still trails the heuristic in local head-to-head testing
-- pure imitation learning from ~a few hundred games can't fully replicate
the heuristic's explicit multi-turn KO-sequencing logic. This reranker can
never do worse than the heuristic on the large majority of decisions with a
clear heuristic winner (those never enter the tie-break); it can only change
outcomes on genuinely close calls, where the BC model's real-player-
imitation signal is most plausibly useful and least likely to override
correct tactical logic.

Safety: heuristic logic (compute_option_scores/select_from_scores) is called
FIRST and is what's actually returned if the BC layer errors in any way --
this file can never play worse than plain dragapult_day1 due to a BC bug.
Full 3-layer fallback, same as v1: BC-augmented -> plain heuristic_agent()
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
    extract_features = _load_sibling_module("features").extract_features
except Exception:  # noqa: BLE001
    extract_features = None

_heuristic_mod = _load_sibling_module("heuristic_scored")
compute_option_scores = _heuristic_mod.compute_option_scores
select_from_scores = _heuristic_mod.select_from_scores
heuristic_agent = _heuristic_mod.heuristic_agent

# --- Reranker tuning knobs (same values as bc_reranker_v1) ---
TOP_K = 3          # consider at most this many top-heuristic-scoring candidates
TIE_ABS_MARGIN = 50.0    # OR gap under this absolute amount counts as "close"
TIE_REL_MARGIN = 0.15    # gap under this fraction of the top score counts as "close"

# See module docstring for why each of these is safe to include. MAIN and the
# DAMAGE_COUNTER contexts are deliberately NOT here -- those are the ones
# that consume this-turn cross-decision plan state (plan_a/plan_b) and broke
# win rate to 0/6 in early bc_reranker_v1 testing when in scope.
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

# Contexts allowed ONLY when every option in the decision targets our own
# side (playerIndex == my_index) -- see docstring. Structurally excludes the
# plan_a-consuming "select opponent's forced-active target" sub-case.
RERANK_ALLOWED_CONTEXTS_OWN_SIDE_ONLY = {
    SelectContext.SWITCH,
    SelectContext.TO_ACTIVE,
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
