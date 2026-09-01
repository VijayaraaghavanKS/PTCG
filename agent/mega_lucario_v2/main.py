import os
import json
import time
import random
from collections import defaultdict, Counter

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
                     Card, Pokemon, all_card_data, all_attack, to_observation_class)

try:
    from cg.api import search_begin, search_step, search_end
    _SEARCH_IMPORT_OK = True
except Exception:
    _SEARCH_IMPORT_OK = False

"""
Mega Lucario v2 - Mega Lucario ex Deck (deck.csv identical to mega_lucario_ref /
ga_v1: this build is a methodology upgrade, not a deck-novelty experiment).

Clean-room implementation, structurally descended from ga_v1's staged policy
(forced/trivial moves -> lethal/KO detection -> loss-shielding retreat logic ->
value-progress fallback scoring), starting from ga_v1's GA-evolved coefficients
(ga_best_weights.json, baked in below as DEFAULT_WEIGHTS) since local testing
showed those roughly on par with hand-guessing. Two techniques are layered on
top, both required by design brief, both optional/best-effort so a failure in
either NEVER prevents returning a legal move:

  1. Persistent opponent-archetype belief tracking (_update_belief / _belief_evidence):
     a monotonic evidence accumulator over revealed opponent cards (Pokemon lines,
     archetype-unique Trainers, dominant energy type), turned into a smoothed
     probability distribution over {dragapult, iono, lucario, unknown}. This
     belief is recomputed every single agent() call (not gated to MAIN / to the
     search layer) and folded into the *ordinary* heuristic via belief-scaled
     effective weights (_effective_weights) and a belief-scaled danger margin
     inside compute_plan - so it influences every decision on every turn, with
     or without the search layer firing.

  2. A fusion-safe shallow forward-search layer (_search_decide). Root candidate
     actions (top few by plain heuristic score) are fixed BEFORE any hidden-info
     determinization is sampled. Each candidate is then evaluated against every
     sampled determinization and the per-candidate scores are averaged - this
     is the specific fix for the "strategy fusion" failure mode of naive
     determinized search (where independently-best moves per-world get compared
     against each other instead of the same fixed action set). Determinization
     sampling is biased by the belief distribution from (1). The search is
     gated to MAIN phase, a moderate option count, turn >= 2, and a strict
     per-decision time budget far below the ~0.8s used by the public reference
     notebook this technique's *mechanics* were studied from (see
     research/public_notebooks/romanrozen_strong-start-baseline-agent-v10-lb-950);
     the code here is an independent reimplementation, not copied.
"""

# ---------------------------------------------------------------------------
# Deck loading
# ---------------------------------------------------------------------------
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = [int(csv[i]) for i in range(60)]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
attack_table = {a.attackId: a for a in all_attack()}

# Decklist (identical to mega_lucario_ref/deck.csv)
Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Gravity_Mountain = 1252
Basic_Fighting_Energy = 6

COSMIC_BEAM = 980  # Solrock, needs Lunatone in play, ignores weak/resist

ATTACKER_LINE_1 = (Riolu, Mega_Lucario_ex)
ATTACKER_LINE_2 = (Makuhita, Hariyama)
ATTACKER_LINE_3 = (Lunatone, Solrock)
DRAW_SUPPORTERS = (Carmine, Lillie_Determination)
SEARCH_ITEMS = (Dusk_Ball, Fighting_Gong, Poke_Pad)

# ---------------------------------------------------------------------------
# Tunable weight vector - baked in from ga_v1's GA-evolved ga_best_weights.json
# (starting point per design brief: testing showed it roughly on par with
# hand-guessed weights, so there's no value re-deriving it here). Still
# override-able the same way ga_v1 supports, in case further tuning is wanted.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "damage_progress": 63.12372262323399,
    "ko_prize_value": 901.7377130657528,
    "energy_on_attacker": 5.986848186560567,
    "active_attacker_bonus": 41.016693128576854,
    "gust_bonus": 263.102306692876,
    "energy_attach_readiness": 46.99507924717307,
    "energy_attach_active_bonus": 25.821545086794455,
    "evolution_progress": 147.61216597764565,
    "card_draw_value": 75.21385210269779,
    "bench_development": 39.466783343232336,
    "search_value": 54.37903977208673,
    "retreat_danger_weight": 37.96498736150028,
    "switch_target_health": 10.034185236895384,
    "tool_hp_value": 31.510040014675152,
    "stadium_value": 16.98868541869651,
    "discard_synergy": 13.806578573943494,
}

WEIGHT_KEYS = list(DEFAULT_WEIGHTS.keys())


def _load_weights() -> dict:
    w = dict(DEFAULT_WEIGHTS)
    paths = []
    env_path = os.environ.get("GA_WEIGHTS_FILE")
    if env_path:
        paths.append(env_path)
    paths.append("weights.json")
    for p in paths:
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    data = json.load(f)
                for k in WEIGHT_KEYS:
                    if k in data:
                        w[k] = float(data[k])
            except Exception:
                pass
            break
    return w


WEIGHTS = _load_weights()

# Feature toggles for A/B ablation testing (default both on).
_SEARCH_ENABLED = os.environ.get("MEGA_V2_SEARCH", "1") != "0"
_BELIEF_ENABLED = os.environ.get("MEGA_V2_BELIEF", "1") != "0"

# Instrumentation (read by test scripts via `import main as mod; mod.SEARCH_STATS`).
SEARCH_STATS = {"decisions_seen": 0, "attempts": 0, "overrides": 0, "fails": 0, "total_ms": 0.0}

# Debug trace instrumentation (opt-in via MEGA_V2_DEBUG=1). Collects one entry per
# search override so a test script can dump a per-game trace to find root causes
# empirically instead of guessing. Never affects behavior when off.
_DEBUG = os.environ.get("MEGA_V2_DEBUG", "0") == "1"
DEBUG_EVENTS = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_card(obs: Observation, area, index, player_index):
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return ps.hand[index]
    if area == AreaType.DISCARD:
        return ps.discard[index]
    if area == AreaType.ACTIVE:
        return ps.active[index]
    if area == AreaType.BENCH:
        return ps.bench[index]
    if area == AreaType.PRIZE:
        return ps.prize[index]
    if area == AreaType.STADIUM:
        return obs.current.stadium[index]
    if area == AreaType.LOOKING:
        return obs.current.looking[index]
    return None


def prize_value(pokemon_id: int) -> int:
    data = card_table.get(pokemon_id)
    if data is None:
        return 1
    if data.megaEx:
        return 3
    if data.ex:
        return 2
    return 1


def stage_value(card_id: int) -> int:
    data = card_table.get(card_id)
    if data is None:
        return 0
    if data.stage2:
        return 2
    if data.stage1:
        return 1
    return 0


class Plan:
    attacker = -1        # -1 = none, 0 = active, 1+ = bench index+1
    attack_id = -1
    target_active = True
    target_bench_index = -1
    needs_energy = False  # needs 1 more energy attached this turn to fire
    wins_game = False
    value = -1.0
    use_gust = False
    danger = False        # my current active is likely to be KO'd next opp turn


def energy_ready(pokemon: Pokemon, attack) -> tuple:
    """Return (currently_ready, needs_one_more)."""
    req = Counter(attack.energies)
    have = Counter(pokemon.energies)
    colorless_needed = req.get(EnergyType.COLORLESS, 0)
    missing = 0
    leftover_have = dict(have)
    for t, c in req.items():
        if t == EnergyType.COLORLESS:
            continue
        h = leftover_have.get(t, 0)
        use = min(h, c)
        leftover_have[t] = h - use
        missing += (c - use)
    if colorless_needed:
        leftover = sum(leftover_have.values())
        use = min(leftover, colorless_needed)
        missing += colorless_needed - use
    return (missing == 0, missing == 1)


def effective_damage(attacker: Pokemon, attack, target: Pokemon) -> int:
    dmg = attack.damage
    if attack.attackId == COSMIC_BEAM:
        return dmg  # ignores weakness/resistance per card text
    tdata = card_table.get(target.id)
    if tdata is None:
        return dmg
    if tdata.weakness is not None and tdata.weakness == EnergyType.FIGHTING:
        dmg *= 2
    elif tdata.resistance is not None and tdata.resistance == EnergyType.FIGHTING:
        dmg = max(0, dmg - 30)
    return dmg


def opponent_best_damage_estimate(op_active, my_active) -> int:
    """Rough estimate of the most damage the opponent's active could deal next turn.

    Improvement over ga_v1: also applies weakness doubling using the opponent
    Pokemon's own type (CardData.energyType) against *our* active's weakness -
    this is the correct TCG mechanic (weakness is a property of the attacking
    Pokemon's type, not the energy used to pay for the attack), and it matters
    concretely here: e.g. Dragapult's own line has no weakness, but a Psychic
    tech like Latias ex in that same archetype does hit our Psychic-weak
    Mega Lucario ex for double.
    """
    if op_active is None:
        return 0
    data = card_table.get(op_active.id)
    if data is None:
        return 0
    my_weak = None
    if my_active is not None:
        mdata = card_table.get(my_active.id)
        if mdata is not None:
            my_weak = mdata.weakness
    best = 0
    have = len(op_active.energies)
    for aid in data.attacks:
        atk = attack_table.get(aid)
        if atk is None:
            continue
        req = len(atk.energies)
        if have + 1 >= req:  # assume they can attach one more energy next turn
            dmg = atk.damage
            if my_weak is not None and data.energyType == my_weak:
                dmg *= 2
            best = max(best, dmg)
    return best


# ---------------------------------------------------------------------------
# Technique 1: persistent opponent-archetype belief tracking
# ---------------------------------------------------------------------------
_DRAGAPULT_DECK = [
    119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 140, 184, 235, 235,
    1071, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1120,
    1121, 1121, 1121, 1121, 1152, 1152, 1152, 1156, 1182, 1182, 1182, 1198, 1198, 1198,
    1198, 1210, 1210, 1227, 1227, 1227, 1227, 1256, 1256, 2, 2, 2, 2, 5, 5, 5, 5,
]
_IONO_DECK = [
    265, 265, 265, 268, 268, 268, 269, 269, 269, 270, 270, 270, 271, 271, 271,
    1086, 1086, 1086, 1097, 1097, 1110, 1118, 1121, 1121, 1121, 1152, 1152, 1227, 1227,
    1227, 1227, 1233, 1233, 1233, 1233, 1254, 1254, 1254,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
]
# _LUCARIO_DECK is my_deck itself (same list, mega_lucario_ref unchanged).

ARCHETYPES = ("dragapult", "iono", "lucario")
_ARCH_DECKS = {"dragapult": _DRAGAPULT_DECK, "iono": _IONO_DECK, "lucario": my_deck}
_ARCH_BASIC_MON = {"dragapult": 119, "iono": 265, "lucario": Riolu}  # fallback face-down guess


def _pokemon_ids(deck):
    return {cid for cid in set(deck) if card_table.get(cid) and card_table[cid].cardType == CardType.POKEMON}


def _trainer_ids(deck):
    trainer_types = (CardType.ITEM, CardType.TOOL, CardType.SUPPORTER, CardType.STADIUM)
    return {cid for cid in set(deck) if card_table.get(cid) and card_table[cid].cardType in trainer_types}


_ARCH_MONS = {name: _pokemon_ids(deck) for name, deck in _ARCH_DECKS.items()}
_ARCH_TRAINERS_RAW = {name: _trainer_ids(deck) for name, deck in _ARCH_DECKS.items()}
# Only count Trainers unique to exactly one of the three known archetypes as
# evidence - shared meta staples (Poke Pad, Boss's Orders, Lillie's
# Determination, ...) carry no discriminating signal.
_ARCH_TRAINERS = {}
for _name in ARCHETYPES:
    _others = set()
    for _other in ARCHETYPES:
        if _other != _name:
            _others |= _ARCH_TRAINERS_RAW[_other]
    _ARCH_TRAINERS[_name] = _ARCH_TRAINERS_RAW[_name] - _others

_ARCH_ENERGY_TYPES = {}
for _name in ARCHETYPES:
    _types = set()
    for cid in set(_ARCH_DECKS[_name]):
        d = card_table.get(cid)
        if d is not None and d.cardType == CardType.BASIC_ENERGY:
            _types.add(int(d.energyType))
    _ARCH_ENERGY_TYPES[_name] = _types

_belief_evidence = {name: 0.0 for name in ARCHETYPES}
_BELIEF_CONF_SATURATION = 12.0  # raw evidence points at which confidence saturates to 1.0


def _reset_belief():
    global _belief_evidence
    _belief_evidence = {name: 0.0 for name in ARCHETYPES}


def _scan_opponent_evidence(state, op_index) -> dict:
    """Raw per-archetype evidence score from currently-visible opponent cards."""
    op = state.players[op_index]
    seen_ids = Counter()
    energy_types = Counter()
    for c in op.discard:
        seen_ids[c.id] += 1
    mons = ([op.active[0]] if (op.active and op.active[0] is not None) else []) + list(op.bench)
    for p in mons:
        seen_ids[p.id] += 1
        for c in p.energyCards:
            seen_ids[c.id] += 1
        for c in p.tools:
            seen_ids[c.id] += 1
        for c in p.preEvolution:
            seen_ids[c.id] += 1
        for e in p.energies:
            energy_types[int(e)] += 1
    if state.stadium and state.stadium[0].playerIndex == op_index:
        seen_ids[state.stadium[0].id] += 1
    for c in op.prize:
        if c is not None:
            seen_ids[c.id] += 1

    raw = {}
    for name in ARCHETYPES:
        score = 0.0
        for cid in seen_ids:
            if cid in _ARCH_MONS[name]:
                score += 3.0
            elif cid in _ARCH_TRAINERS[name]:
                score += 1.0
        raw[name] = score

    if energy_types:
        etop = max(energy_types.items(), key=lambda x: x[1])[0]
        for name in ARCHETYPES:
            if etop in _ARCH_ENERGY_TYPES[name]:
                raw[name] += 0.5
    return raw


def _update_belief(state, op_index) -> dict:
    """Update the persistent monotonic evidence accumulator from the REAL
    (non-simulated) game state, and return a smoothed belief distribution
    over {dragapult, iono, lucario, unknown} summing to 1.0.

    Monotonic max (never decreases) rather than plain overwrite so a rare
    "un-reveal" effect (e.g. a card returned from discard to deck) can't
    erase evidence we already legitimately observed earlier in the match.
    """
    global _belief_evidence
    raw = _scan_opponent_evidence(state, op_index)
    for name in ARCHETYPES:
        if raw[name] > _belief_evidence[name]:
            _belief_evidence[name] = raw[name]
    total = sum(_belief_evidence.values())
    confidence = min(1.0, total / _BELIEF_CONF_SATURATION) if total > 0 else 0.0
    dist = {"unknown": 1.0 - confidence}
    for name in ARCHETYPES:
        dist[name] = confidence * (_belief_evidence[name] / total) if total > 0 else 0.0
    return dist


def _belief_blend(belief: dict, table: dict) -> float:
    return sum(belief.get(k, 0.0) * table.get(k, 1.0) for k in table)


# How the belief distribution biases the ordinary (non-search) heuristic.
# Dragapult: no weakness, high burst (Phantom Dive 200 for 2 energy) -> be
#   more cautious with loss-shielding and race tempo harder.
# Iono/Bellibolt: weak to Fighting (our whole deck) -> favorable matchup,
#   so be slightly less cautious / more willing to trade.
# Lucario mirror: symmetric, moderate caution.
_ARCH_DANGER_MARGIN = {"dragapult": 45.0, "iono": -10.0, "lucario": 15.0, "unknown": 15.0}
_ARCH_GUST_MULT = {"dragapult": 1.35, "iono": 1.0, "lucario": 1.1, "unknown": 1.0}
_ARCH_TEMPO_MULT = {"dragapult": 1.25, "iono": 0.95, "lucario": 1.1, "unknown": 1.0}
_ARCH_RETREAT_MULT = {"dragapult": 1.3, "iono": 0.85, "lucario": 1.1, "unknown": 1.0}


def _effective_weights(belief: dict) -> dict:
    w = dict(WEIGHTS)
    gust_mult = _belief_blend(belief, _ARCH_GUST_MULT)
    tempo_mult = _belief_blend(belief, _ARCH_TEMPO_MULT)
    retreat_mult = _belief_blend(belief, _ARCH_RETREAT_MULT)
    w["gust_bonus"] *= gust_mult
    w["active_attacker_bonus"] *= tempo_mult
    w["energy_attach_readiness"] *= tempo_mult
    w["energy_attach_active_bonus"] *= tempo_mult
    w["retreat_danger_weight"] *= retreat_mult
    return w


# ---------------------------------------------------------------------------
# compute_plan (pure - returns a Plan, doesn't mutate module globals) so it's
# safe to call from inside the search layer's hypothetical rollouts without
# corrupting the real decision's state.
# ---------------------------------------------------------------------------
def compute_plan(obs, state, select, my_index, my_state, op_state, field_counts, hand_counts,
                  belief, eff_weights) -> Plan:
    plan = Plan()

    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None
    if my_active is None or op_active is None:
        return plan

    threat = opponent_best_damage_estimate(op_active, my_active)
    danger_margin = _belief_blend(belief, _ARCH_DANGER_MARGIN)
    if my_active.hp <= threat + danger_margin:
        plan.danger = True

    can_reposition = (not state.retreated)
    my_cards = [(0, my_active)] + [(i + 1, p) for i, p in enumerate(my_state.bench)]

    op_bench = list(op_state.bench)
    has_boss = hand_counts[Boss_Orders] > 0

    best_value = -1.0
    for idx, pokemon in my_cards:
        if idx != 0 and not can_reposition:
            continue
        data = card_table.get(pokemon.id)
        if data is None:
            continue
        for aid in data.attacks:
            atk = attack_table.get(aid)
            if atk is None:
                continue
            if aid == COSMIC_BEAM and field_counts[Lunatone] == 0:
                continue
            ready, needs_one = energy_ready(pokemon, atk)
            if not ready and not (needs_one and not state.energyAttached and hand_counts[Basic_Fighting_Energy] > 0):
                continue

            targets = [(True, -1, op_active)]
            if has_boss:
                for bi, bp in enumerate(op_bench):
                    targets.append((False, bi, bp))

            for is_active_target, bench_idx, target in targets:
                dmg = effective_damage(pokemon, atk, target)
                is_ko = dmg >= target.hp
                pv = prize_value(target.id)
                if is_ko:
                    value = eff_weights["ko_prize_value"] * pv
                    wins = is_ko and len(op_state.prize) <= pv
                else:
                    value = eff_weights["damage_progress"] * min(1.0, dmg / max(1, target.hp))
                    wins = False
                value += eff_weights["energy_on_attacker"] * len(pokemon.energies)
                if idx == 0:
                    value += eff_weights["active_attacker_bonus"]
                if not is_active_target:
                    value += eff_weights["gust_bonus"] * (pv / 3.0)
                if wins:
                    value = 1e9

                if value > best_value:
                    best_value = value
                    plan.attacker = idx
                    plan.attack_id = aid
                    plan.target_active = is_active_target
                    plan.target_bench_index = bench_idx
                    plan.needs_energy = needs_one and not ready
                    plan.wins_game = wins
                    plan.use_gust = not is_active_target
                    plan.value = value
    return plan


# ---------------------------------------------------------------------------
# Option scoring (pure - same staged-heuristic structure as ga_v1, ported to
# take `plan`/`eff_weights` as parameters instead of reading module globals,
# so it's safe to reuse both for the real decision and for search rollouts).
# ---------------------------------------------------------------------------
def _score_options(obs, state, select, context, my_index, my_state, op_state,
                    field_counts, hand_counts, discard_counts, stadium_id, plan, eff_weights):
    my_active = my_state.active[0] if my_state.active else None

    scores = []
    for o in select.option:
        score = 0.0
        if o.type == OptionType.NUMBER:
            score = float(o.number)
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = -1.0  # prefer going second: first player skips their turn-1 draw/attack in this ruleset
            else:
                score = 1.0
        elif o.type == OptionType.NO:
            score = 0.0
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                hp = card.hp if isinstance(card, Pokemon) else 0
                maxhp = card.maxHp if isinstance(card, Pokemon) else 1
                hp_frac = (hp / maxhp) if isinstance(card, Pokemon) and maxhp > 0 else 0.0

                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if o.playerIndex == my_index:
                        score = 100.0
                        score += eff_weights["energy_on_attacker"] * energy_count
                        score += eff_weights["switch_target_health"] * hp_frac * 10.0
                        if plan.attacker >= 1 and (plan.attacker - 1) == o.index:
                            score += 500.0
                    else:
                        score = eff_weights["gust_bonus"] * (prize_value(card.id) / 3.0)
                        score += (100.0 - hp_frac * 100.0)
                        if plan.use_gust and plan.target_bench_index == o.index:
                            score += 1000.0
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Riolu:
                        score = 40.0
                    elif card.id == Makuhita:
                        score = 30.0
                    elif card.id == Solrock:
                        score = 20.0 if state.firstPlayer != my_index else 15.0
                    elif card.id == Lunatone:
                        score = 10.0
                    else:
                        score = 5.0
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = eff_weights["bench_development"]
                    if card.id in (Riolu, Makuhita):
                        score += eff_weights["evolution_progress"] * 0.3
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.LOOK):
                    score = _hand_target_score(card.id, field_counts, hand_counts, eff_weights)
                elif context == SelectContext.DISCARD:
                    score = -_hand_target_score(card.id, field_counts, hand_counts, eff_weights) * 0.5
                    if card.id == Basic_Fighting_Energy:
                        score += eff_weights["discard_synergy"]
                elif context == SelectContext.ATTACH_FROM:
                    score = _attach_score(pokemon=card, active=(o.area == AreaType.ACTIVE), eff_weights=eff_weights)
                elif context in (SelectContext.EVOLVES_FROM,):
                    score = 100.0 + eff_weights["energy_on_attacker"] * energy_count + hp_frac * 20.0
                elif context in (SelectContext.EVOLVES_TO,):
                    score = 50.0 + eff_weights["evolution_progress"] * stage_value(card.id)
                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if hp > 0:
                        score = 1000.0 - hp
                else:
                    score = _hand_target_score(card.id, field_counts, hand_counts, eff_weights)
        elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY, OptionType.TOOL_CARD):
            score = 1.0 if o.playerIndex != my_index else 0.5
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table.get(card.id)
            score = _play_score(card.id, data, field_counts, hand_counts, discard_counts, stadium_id, state,
                                 plan, eff_weights)
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is not None and card.id == Hero_Cape:
                score = 7000.0 + eff_weights["tool_hp_value"]
                if pokemon is not None and pokemon.id == Mega_Lucario_ex:
                    score += eff_weights["tool_hp_value"]
            else:
                score = _attach_score(pokemon, o.inPlayArea == AreaType.ACTIVE, eff_weights)
                is_planned_target = (
                    (plan.attacker == 0 and o.inPlayArea == AreaType.ACTIVE) or
                    (plan.attacker >= 1 and o.inPlayArea == AreaType.BENCH and o.inPlayIndex == plan.attacker - 1)
                )
                if plan.needs_energy and is_planned_target:
                    score += 5000.0
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            card = get_card(obs, o.area, o.index, my_index)
            score = 9000.0 + eff_weights["evolution_progress"] * stage_value(card.id if card else 0)
            if pokemon is not None:
                score += eff_weights["energy_on_attacker"] * len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None and card.id == Lunatone:
                if hand_counts[Basic_Fighting_Energy] > 0:
                    score = 25000.0 + eff_weights["card_draw_value"] + eff_weights["discard_synergy"]
                else:
                    score = -1.0
            elif card is not None and card.id == Hariyama:
                score = 28000.0 + eff_weights["gust_bonus"] * 0.2
            else:
                score = 26000.0
        elif o.type == OptionType.DISCARD:
            score = 0.0
        elif o.type == OptionType.RETREAT:
            if my_active is not None and my_active.maxHp > 0:
                hp_frac = my_active.hp / my_active.maxHp
            else:
                hp_frac = 1.0
            score = -1.0
            if plan.danger:
                score = eff_weights["retreat_danger_weight"] * (1.5 - hp_frac)
            if plan.attacker >= 1:
                score = max(score, 2000.0)
        elif o.type == OptionType.ATTACK:
            if plan.wins_game and o.attackId == plan.attack_id:
                score = 1e9
            elif o.attackId == plan.attack_id and not plan.needs_energy:
                score = 1000.0 + max(0.0, plan.value) * 0.01
            else:
                score = 200.0
        elif o.type == OptionType.END:
            score = -5.0

        scores.append(score)
    return scores


def _pick_from_scores(select, scores):
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    out = order[:select.maxCount]
    while len(out) < select.minCount:
        for i in order:
            if i not in out:
                out.append(i)
                break
        else:
            break
    return order, out


def _hand_target_score(card_id: int, field_counts, hand_counts, eff_weights) -> float:
    if card_id == Riolu:
        if field_counts[Riolu] + field_counts[Mega_Lucario_ex] >= 3:
            return 100.0
        return 3000.0 + eff_weights["evolution_progress"]
    if card_id == Mega_Lucario_ex:
        if field_counts[Riolu] >= 1:
            return 4000.0 + eff_weights["evolution_progress"] * 2
        return 500.0
    if card_id == Makuhita:
        if field_counts[Makuhita] + field_counts[Hariyama] >= 2:
            return 100.0
        return 2500.0 + eff_weights["evolution_progress"] * 0.5
    if card_id == Hariyama:
        if field_counts[Makuhita] >= 1:
            return 3800.0 + eff_weights["evolution_progress"] * 2
        return 300.0
    if card_id in (Lunatone, Solrock):
        if field_counts[card_id] >= 1:
            return 50.0
        return 1800.0
    if card_id == Basic_Fighting_Energy:
        return 2200.0 + eff_weights["energy_attach_readiness"] * 0.3
    if card_id in SEARCH_ITEMS:
        return 2000.0 + eff_weights["search_value"]
    if card_id in DRAW_SUPPORTERS:
        return 2400.0 + eff_weights["card_draw_value"]
    if card_id == Boss_Orders:
        return 2100.0 + eff_weights["gust_bonus"] * 0.3
    if card_id == Switch:
        return 1500.0
    if card_id == Hero_Cape:
        return 1400.0 + eff_weights["tool_hp_value"]
    if card_id == Premium_Power_Pro:
        return 1600.0
    if card_id == Gravity_Mountain:
        return 800.0 + eff_weights["stadium_value"]
    return 500.0


def _attach_score(pokemon, active: bool, eff_weights) -> float:
    if pokemon is None:
        return -1.0
    if pokemon.id not in (Makuhita, Hariyama, Lunatone, Solrock, Riolu, Mega_Lucario_ex):
        return -1.0
    data = card_table.get(pokemon.id)
    if data is None:
        return -1.0
    max_req = 0
    for aid in data.attacks:
        atk = attack_table.get(aid)
        if atk is not None:
            max_req = max(max_req, len(atk.energies))
    have = len(pokemon.energies)
    if have >= max_req and max_req > 0:
        return 100.0
    score = 8000.0
    score += eff_weights["energy_attach_readiness"] * (have + 1)
    if active:
        score += eff_weights["energy_attach_active_bonus"]
    return score


def _play_score(card_id, data, field_counts, hand_counts, discard_counts, stadium_id, state,
                 plan, eff_weights) -> float:
    if data is None:
        return -1.0
    if data.cardType == CardType.POKEMON:
        score = 20000.0
        if card_id in (Lunatone, Solrock) and field_counts[card_id] >= 1:
            score = -1.0
        elif card_id == Riolu and field_counts[Riolu] + field_counts[Mega_Lucario_ex] >= 3:
            score = -1.0
        elif card_id == Makuhita and field_counts[Makuhita] + field_counts[Hariyama] >= 2:
            score = -1.0
        return score

    if data.cardType == CardType.SUPPORTER and state.supporterPlayed:
        return -1.0

    if card_id == Switch:
        if plan.attacker >= 1:
            return 6000.0
        return -1.0
    if card_id == Premium_Power_Pro:
        if plan.attacker >= -1 and not plan.wins_game:
            return 5500.0
        return -1.0
    if card_id == Boss_Orders:
        if plan.use_gust:
            return 6200.0 + eff_weights["gust_bonus"]
        return -1.0
    if card_id in DRAW_SUPPORTERS:
        hand_size = len(getattr(state.players[state.yourIndex], "hand", None) or [])
        need = max(0.0, 5.0 - hand_size)
        return 5000.0 + eff_weights["card_draw_value"] * (1.0 + need * 0.2)
    if card_id in SEARCH_ITEMS:
        return 4800.0 + eff_weights["search_value"]
    if card_id == Hero_Cape:
        return -1.0
    if card_id == Gravity_Mountain:
        if stadium_id == Gravity_Mountain:
            return -1.0
        return 3000.0 + eff_weights["stadium_value"]

    return 3000.0


def _decide(obs, belief):
    """Compute (plan, scores, order, choice) for the given observation, pure
    function of (obs, belief) - safe to call for the real decision and for
    hypothetical search rollouts alike (works for whichever player's turn the
    observation's select actually belongs to)."""
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)
    for c in ([my_state.active[0]] if my_state.active and my_state.active[0] else []) + list(my_state.bench):
        field_counts[c.id] += 1
    for c in (my_state.hand or []):
        hand_counts[c.id] += 1
    for c in my_state.discard:
        discard_counts[c.id] += 1
    stadium_id = state.stadium[0].id if state.stadium else 0

    eff_weights = _effective_weights(belief)

    plan = Plan()
    if context == SelectContext.MAIN:
        plan = compute_plan(obs, state, select, my_index, my_state, op_state, field_counts, hand_counts,
                             belief, eff_weights)

    scores = _score_options(obs, state, select, context, my_index, my_state, op_state,
                             field_counts, hand_counts, discard_counts, stadium_id, plan, eff_weights)
    order, choice = _pick_from_scores(select, scores)
    return plan, scores, order, choice


# ---------------------------------------------------------------------------
# Technique 2: fusion-safe shallow forward-search layer
# ---------------------------------------------------------------------------
# Escalated from the original N_DET=2/K_CAND=3/TIME_BUDGET_S=0.35: measured
# instrumentation showed massive unused time headroom (avg ~4-18ms/decision,
# worst full match ~1.2s wall-clock, against a 600s/match hard budget and the
# public reference notebook's own 0.8s/decision figure), so there was no
# reason not to sample more determinizations and consider more candidates.
# Escalation was validated empirically (see report): no regression on ga_v1
# A/B parity or the dragapult_day1 matchup at n=100+, each tested after the
# change, not just before it.
N_DET = 6                  # hidden-info determinizations sampled per decision
K_CAND = 5                  # fixed root candidates evaluated across ALL determinizations
TIME_BUDGET_S = 0.6         # per-decision wall clock budget (well under the ~0.8s reference)
MAX_ROLLOUT_STEPS = 14      # cap on sub-selects greedily resolved to finish our own turn
SEARCH_MIN_OPTS = 3
SEARCH_MAX_OPTS = 18
SEARCH_MIN_TURN = 2
# Override the plain heuristic's top pick only when the averaged search score
# beats it by at least this much. Originally 450 (~half a KO's prize value
# under our weights, ~900/2). A/B testing at n=100-300/matchup showed 300 was
# safe (no regression vs ga_v1 parity or dragapult_day1) while letting
# marginally-confident search findings through more often - matches the
# spirit of the public reference notebook's own margin gate, just tuned looser
# now that N_DET=6 (up from 2) gives each candidate's average far less noise.
SEARCH_OVERRIDE_MARGIN = 300.0

_search_ok = _SEARCH_IMPORT_OK


def _leaf_eval(cs, my_index):
    """Evaluate a (possibly mid-turn) game state from my_index's perspective."""
    if cs is None:
        return None
    if cs.result is not None and cs.result >= 0:
        if cs.result == my_index:
            return 1_000_000.0
        if cs.result == 2:
            return 0.0
        return -1_000_000.0
    me = cs.players[my_index]
    op = cs.players[1 - my_index]
    my_mons = ([me.active[0]] if me.active and me.active[0] else []) + list(me.bench)
    op_mons = ([op.active[0]] if op.active and op.active[0] else []) + list(op.bench)
    my_hp = sum(p.hp for p in my_mons if p)
    op_hp = sum(p.hp for p in op_mons if p)
    my_en = sum(len(p.energies) for p in my_mons if p)
    op_en = sum(len(p.energies) for p in op_mons if p)

    value = WEIGHTS["ko_prize_value"] * (len(op.prize) - len(me.prize))
    value += (my_hp - op_hp)
    value += WEIGHTS["energy_on_attacker"] * 0.5 * (my_en - op_en)
    if not (me.active and me.active[0]):
        value -= 5000.0
    if not (op.active and op.active[0]):
        value += 3000.0

    # One-ply-of-static-lookahead patch: the rollout only completes OUR turn
    # and never simulates the opponent's reply, which left a blind spot around
    # exactly the danger/retreat situations the base heuristic already knows
    # to avoid (a candidate that "develops the board" could look great here
    # while walking straight into a KO next turn). Penalize resulting states
    # where our post-rollout active is exposed to the opponent's best next hit,
    # using the same weakness-aware estimator compute_plan's danger flag uses -
    # cheap (no extra search_step calls) since it only reads the already-
    # returned state.
    if me.active and me.active[0] and op.active and op.active[0]:
        threat = opponent_best_damage_estimate(op.active[0], me.active[0])
        if threat >= me.active[0].hp:
            value -= WEIGHTS["retreat_danger_weight"] * 3.0
        elif threat > 0:
            value -= threat * 0.5
    return value


def _sample_hidden(state, my_index, belief):
    """One determinization of all hidden zones, biased toward the
    belief-weighted archetype, returned as kwargs for search_begin()."""
    me = state.players[my_index]
    op_i = 1 - my_index
    op = state.players[op_i]

    # --- our own deck + facedown prize (we know our own decklist exactly) ---
    seen = Counter()
    for c in (me.hand or []):
        seen[c.id] += 1
    for c in me.discard:
        seen[c.id] += 1
    for c in me.prize:
        if c is not None:
            seen[c.id] += 1
    for p in ([me.active[0]] if me.active and me.active[0] else []) + list(me.bench):
        seen[p.id] += 1
        for c in p.energyCards:
            seen[c.id] += 1
        for c in p.tools:
            seen[c.id] += 1
        for c in p.preEvolution:
            seen[c.id] += 1
    if state.stadium and state.stadium[0].playerIndex == my_index:
        seen[state.stadium[0].id] += 1

    remainder = []
    for cid, n in Counter(my_deck).items():
        remainder.extend([cid] * max(0, n - seen.get(cid, 0)))
    n_prize_hidden = sum(1 for c in me.prize if c is None)
    need = me.deckCount + n_prize_hidden
    if len(remainder) < need:
        remainder += [Basic_Fighting_Energy] * (need - len(remainder))
    random.shuffle(remainder)
    your_deck = remainder[:me.deckCount]
    fill = iter(remainder[me.deckCount:need])
    your_prize = [c.id if c is not None else next(fill, Basic_Fighting_Energy) for c in me.prize]

    # --- opponent hidden info: sample an archetype according to belief, then
    # fill from that archetype's decklist minus what we've already seen ---
    op_seen = Counter()
    op_energy_types = Counter()
    for c in op.discard:
        op_seen[c.id] += 1
    for p in ([op.active[0]] if op.active and op.active[0] else []) + list(op.bench):
        op_seen[p.id] += 1
        for c in p.energyCards:
            op_seen[c.id] += 1
        for c in p.tools:
            op_seen[c.id] += 1
        for c in p.preEvolution:
            op_seen[c.id] += 1
        for e in p.energies:
            op_energy_types[int(e)] += 1
    if state.stadium and state.stadium[0].playerIndex == op_i:
        op_seen[state.stadium[0].id] += 1
    for c in op.prize:
        if c is not None:
            op_seen[c.id] += 1

    names = list(ARCHETYPES) + [None]
    weights = [belief.get(k, 0.0) for k in ARCHETYPES] + [belief.get("unknown", 0.0)]
    pick = random.choices(names, weights=weights, k=1)[0] if sum(weights) > 0 else None

    if pick is not None:
        pool = []
        for cid, n in Counter(_ARCH_DECKS[pick]).items():
            pool.extend([cid] * max(0, n - op_seen.get(cid, 0)))
    else:
        etop = max(op_energy_types.items(), key=lambda x: x[1])[0] if op_energy_types else None
        energy_id = etop if (etop is not None and 1 <= etop <= 8) else Basic_Fighting_Energy
        top_mons = [cid for cid in op_seen if card_table.get(cid) and card_table[cid].cardType == CardType.POKEMON]
        pool = []
        for cid in top_mons:
            pool.extend([cid] * 4)
        pool += [energy_id] * 30
        pool += [Riolu] * 4  # generic basic filler so a setup-style sub-select never wedges

    n_op_prize_hidden = sum(1 for c in op.prize if c is None)
    op_need = op.deckCount + n_op_prize_hidden + op.handCount
    if len(pool) < op_need:
        pool += [Basic_Fighting_Energy] * (op_need - len(pool))
    random.shuffle(pool)
    opponent_deck = pool[:op.deckCount]
    off = op.deckCount
    fill_op = iter(pool[off:off + n_op_prize_hidden])
    opponent_prize = [c.id if c is not None else next(fill_op, Basic_Fighting_Energy) for c in op.prize]
    off += n_op_prize_hidden
    opponent_hand = pool[off:off + op.handCount]

    opponent_active = []
    if op.active and op.active[0] is None:
        guess = _ARCH_BASIC_MON.get(pick, Riolu)
        opponent_active = [guess]

    return dict(your_deck=your_deck, your_prize=your_prize,
                opponent_deck=opponent_deck, opponent_prize=opponent_prize,
                opponent_hand=opponent_hand, opponent_active=opponent_active)


def _rollout_my_turn(sid, cur, my_index, belief, deadline):
    """Greedily resolve sub-selects using the plain heuristic until either our
    turn hands off to the opponent, the game ends, the step cap is hit, or the
    time budget runs out. This is what keeps the search "shallow": we only
    look at the resulting board state after finishing our OWN turn, never
    simulate the opponent's reply."""
    for _ in range(MAX_ROLLOUT_STEPS):
        if time.monotonic() > deadline:
            return sid, cur
        cs = cur.current
        if cs is None or cur.select is None:
            return sid, cur
        if cs.result is not None and cs.result >= 0:
            return sid, cur
        if cs.yourIndex != my_index:
            return sid, cur
        try:
            _, _, _, choice = _decide(cur, belief)
        except Exception:
            return sid, cur
        if not choice:
            return sid, cur
        try:
            ss = search_step(sid, choice)
        except Exception:
            return sid, cur
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _search_decide(obs, state, my_index, belief, order, scores, plan=None):
    """Fusion-safe shallow search: fix candidate root actions from the plain
    heuristic BEFORE sampling any hidden-info determinization, then evaluate
    every candidate against every sampled determinization and average - never
    let a determinization pick its own independent best move (that's the
    ISMCTS "strategy fusion" failure mode this is explicitly built to avoid).
    Returns an option index to override with, or None to keep the heuristic's
    own top pick (including on any error/timeout - this must never crash or
    fail to return a legal choice from the caller's perspective, since None
    just means "use the normal heuristic answer")."""
    global _search_ok
    if not (_search_ok and _SEARCH_IMPORT_OK):
        return None
    select = obs.select
    n = len(select.option)
    if not (SEARCH_MIN_OPTS <= n <= SEARCH_MAX_OPTS):
        return None
    if state.turn < SEARCH_MIN_TURN:
        return None
    if not (select.minCount <= 1 <= select.maxCount):
        return None
    if getattr(obs, "search_begin_input", None) is None:
        return None

    heur_top = order[0]
    if scores[heur_top] >= 1e8:
        return None  # already a detected forced win / forced move - search adds nothing, save the time
    if scores[heur_top] < 0:
        return None  # heuristic itself has nothing good here - not worth searching among "avoid" options

    # Only ever compare candidates the plain heuristic considers plausible
    # (score >= 0). A sibling build (bellibolt_v2) hit a concrete bug from
    # NOT doing this: with only END filtered out, a noisy few-determinization
    # search could occasionally rank a heuristic-flagged "avoid this" option
    # (score < 0, e.g. a Trainer with no legal target right now) above a
    # legitimate one purely by chance, since a negative-score option could
    # still land in the fixed top-K_CAND candidate set whenever most other
    # options were also negative. Filtering to score >= 0 here closes the
    # same hole pre-emptively.
    cand = [i for i in order if scores[i] >= 0][:K_CAND]
    if len(cand) < 2:
        return None

    SEARCH_STATS["decisions_seen"] += 1
    t0 = time.monotonic()
    deadline = t0 + TIME_BUDGET_S
    acc = {i: 0.0 for i in cand}
    n_eval = {i: 0 for i in cand}
    began = False
    SEARCH_STATS["attempts"] += 1
    try:
        for _det in range(N_DET):
            if time.monotonic() > deadline:
                break
            hidden = _sample_hidden(state, my_index, belief)
            try:
                ss0 = search_begin(obs, **hidden)
                began = True
            except Exception:
                _search_ok = False
                return None
            root_sid = ss0.searchId

            for idx in cand:
                if time.monotonic() > deadline:
                    break
                try:
                    ss = search_step(root_sid, [idx])
                except Exception:
                    continue
                sid, cur = ss.searchId, ss.observation
                sid, cur = _rollout_my_turn(sid, cur, my_index, belief, deadline)
                val = _leaf_eval(cur.current if cur is not None else None, my_index)
                if val is None:
                    continue
                acc[idx] += val
                n_eval[idx] += 1

            try:
                search_end()
            except Exception:
                pass
            began = False

        n_top = n_eval.get(heur_top, 0)
        if n_top == 0:
            return None
        # Fair comparison: only compare candidates evaluated across the same
        # number of determinizations as the heuristic's own top pick.
        evaluated = [i for i in cand if n_eval[i] == n_top]
        avg = {i: acc[i] / n_eval[i] for i in evaluated}
        best = max(evaluated, key=lambda i: avg[i])
        if best == heur_top:
            return None
        if avg[best] < avg[heur_top] + SEARCH_OVERRIDE_MARGIN:
            return None
        SEARCH_STATS["overrides"] += 1
        if _DEBUG:
            try:
                heur_opt = select.option[heur_top]
                best_opt = select.option[best]
                DEBUG_EVENTS.append({
                    "turn": state.turn,
                    "heur_top_type": int(heur_opt.type), "heur_top_score": scores[heur_top],
                    "heur_top_neg": scores[heur_top] < 0,
                    "best_type": int(best_opt.type),
                    "avg_heur": avg[heur_top], "avg_best": avg[best],
                    "margin": avg[best] - avg[heur_top],
                    "plan_danger": bool(plan.danger) if plan is not None else None,
                    "n_cand": len(cand), "n_det_used": n_top,
                })
            except Exception:
                pass
        return best
    except Exception:
        SEARCH_STATS["fails"] += 1
        return None
    finally:
        SEARCH_STATS["total_ms"] += (time.monotonic() - t0) * 1000.0
        if began:
            try:
                search_end()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
def agent(obs_dict: dict) -> list:
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return my_deck

    if obs.select is None:
        _reset_belief()
        if _DEBUG:
            DEBUG_EVENTS.clear()
        return my_deck

    state = obs.current
    my_index = state.yourIndex
    op_index = 1 - my_index

    if _BELIEF_ENABLED:
        try:
            belief = _update_belief(state, op_index)
        except Exception:
            belief = {"unknown": 1.0, "dragapult": 0.0, "iono": 0.0, "lucario": 0.0}
    else:
        belief = {"unknown": 1.0, "dragapult": 0.0, "iono": 0.0, "lucario": 0.0}

    try:
        plan, scores, order, choice = _decide(obs, belief)
    except Exception:
        # Ultra-safe last-resort fallback: never fail to return a legal move.
        try:
            sel = obs.select
            n = len(sel.option)
            k = min(max(1, sel.minCount), n) if n else 0
            return list(range(k)) if k else [0]
        except Exception:
            return [0]

    if _SEARCH_ENABLED and obs.select.context == SelectContext.MAIN:
        try:
            override = _search_decide(obs, state, my_index, belief, order, scores, plan)
        except Exception:
            override = None
        if override is not None:
            return [override]

    return choice
