import os
from collections import defaultdict

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
                     Card, Pokemon, all_card_data, all_attack, to_observation_class)

"""
Latias ex / Xerneas ex - Psychic type-weakness-exploit deck (clean-room build).

Why this archetype (see agent/AGENT_LOG.md for the full history this responds
to): every "high card-value" deck we tried before this (Alakazam, Mist
Energy, Greninja ex, Inteleon, Jolteon ex) needed multiple turns of
evolution/ramp setup before threatening real damage and got run over by our
rating band's fast starter-kit decks. This deck is deliberately built to
avoid that failure mode: all 4 Pokemon lines (Latias ex, Xerneas ex, Scream
Tail ex, Cresselia) are Basic Pokemon - zero evolution, so every copy drawn
is an immediately-playable attacker from turn 1.

The new lever: we checked the real card data and found Mega Lucario ex
(the single most common opponent in our rating band, ~28-29% share) is
Fighting-type with Weakness {P} (Psychic). None of our decks or reference
opponents have ever exploited a type weakness. This deck is mono-Psychic,
so every one of our attacks gets the engine's automatic weakness-doubling
against Mega Lucario ex. Concretely: Latias ex's Eon Blade (200 base) or
Xerneas ex's Rising Horns (120 base, +100 more since Lucario is a Pokemon ex
= 220 base) both double to 400/440 against Lucario's 340 HP - a clean
one-shot from full health, before Lucario's own attacks (130/270) can reply.

Latias ex's Skyliner ability ("Your Basic Pokemon in play have no Retreat
Cost") is the deck's second pillar: since every Pokemon in the deck is
Basic, once Latias ex is in play (bench or active - the ability isn't
active-spot-gated) our whole team can pivot for free. That lets us play our
fragile-for-their-HP (120-210 HP) ex attackers aggressively and still yank
them out of danger before a big hit lands, without spending a turn's worth
of energy on retreat cost the way most decks have to.

Structure: staged decision policy (forced/trivial -> lethal/KO detection,
weakness- and ex-bonus-aware -> loss-shielding retreat -> value-progress
fallback for search/draw/development), same shape as mega_lucario_v2's
compute_plan/_score_options split, reimplemented clean-room for this deck's
specific cards. The native engine (cg.dll/.so) resolves actual battle damage
(including weakness/resistance) automatically - what we own is making sure
OUR OWN pre-attack lethal-detection heuristic doesn't systematically
under-estimate its own kill potential by ignoring that bonus, which is
exactly the gap `effective_damage()` below closes generically (compares the
attacker's own energyType against the target's Weakness/Resistance, not
hardcoded to one type - unlike the Fighting-only special case in
mega_lucario_v2, this generalizes to whichever type our own attacker is).
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

card_table = {c.cardId: c for c in all_card_data()}
attack_table = {a.attackId: a for a in all_attack()}

Latias_ex = 184
Xerneas_ex = 331
Scream_Tail_ex = 969
Cresselia = 764
ATTACKER_IDS = (Latias_ex, Xerneas_ex, Scream_Tail_ex, Cresselia)

Ultra_Ball = 1121
Dusk_Ball = 1102
Lillie_Determination = 1227
Carmine = 1192
Crispin = 1198
Boss_Orders = 1182
Night_Stretcher = 1097
Switch = 1123
Waitress = 1235
Mystery_Garden = 1263
Poke_Pad = 1152
Basic_Psychic_Energy = 5

DRAW_SUPPORTERS = (Lillie_Determination, Carmine)
SEARCH_ITEMS = (Ultra_Ball, Dusk_Ball, Poke_Pad)

RISING_HORNS = 461  # Xerneas ex: +100 damage if opponent's Active is a Pokemon ex

# ---------------------------------------------------------------------------
# Hand-picked value weights (not GA-tuned - kept simple and legible for a
# first build; scale matches the same rough order of magnitude used by
# mega_lucario_v2's GA-evolved weights, as a sanity anchor).
# ---------------------------------------------------------------------------
W = {
    "damage_progress": 65.0,
    "ko_prize_value": 900.0,
    "energy_on_attacker": 6.0,
    "active_attacker_bonus": 40.0,
    "gust_bonus": 260.0,
    "energy_attach_readiness": 47.0,
    "energy_attach_active_bonus": 26.0,
    "card_draw_value": 75.0,
    "bench_development": 40.0,
    "search_value": 55.0,
    "retreat_danger_weight": 40.0,
    "switch_target_health": 10.0,
    "stadium_value": 17.0,
    "ramp_value": 600.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_card(obs, area, index, player_index):
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


def prize_value(pokemon_id):
    data = card_table.get(pokemon_id)
    if data is None:
        return 1
    if data.megaEx:
        return 3
    if data.ex:
        return 2
    return 1


def energy_ready(pokemon, attack):
    """Return (currently_ready, needs_one_more)."""
    from collections import Counter
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


def attack_ready(pokemon, atk, state, hand_counts):
    """(is_ready_now, needs_one_more_this_turn). Accounts for the fact we may
    still have this turn's single manual energy attachment available."""
    ready, needs_one = energy_ready(pokemon, atk)
    can_via_attach = needs_one and not state.energyAttached and hand_counts.get(Basic_Psychic_Energy, 0) > 0
    return (ready or can_via_attach), (needs_one and not ready)


def is_ramp_attack(atk):
    if atk.damage:
        return False
    t = (atk.text or "").lower()
    return "energy" in t and "attach" in t


def conditional_bonus(atk, target_data):
    """Damage bonuses the engine applies conditionally on attack text that we
    must replicate ourselves so our own lethal-detection doesn't
    underestimate kill potential (mirrors the weakness-bonus gap called out
    for this task, generalized to other conditional text on our own cards)."""
    if atk.attackId == RISING_HORNS and target_data is not None and target_data.ex:
        return 100
    return 0


def effective_damage(attacker_data, atk, target_data):
    """Generalized weakness/resistance-aware damage estimate: compares the
    ATTACKING Pokemon's own type against the TARGET's Weakness/Resistance
    (the correct TCG mechanic - weakness is a property of the attacker's
    type, not the energy that paid for the attack). Unlike a hardcoded
    single-type special case, this works for whichever type our own attacker
    happens to be, so it's correct for every matchup, not just the one this
    deck was built around."""
    if attacker_data is None or target_data is None:
        return atk.damage
    dmg = atk.damage + conditional_bonus(atk, target_data)
    if target_data.weakness is not None and target_data.weakness == attacker_data.energyType:
        dmg *= 2
    elif target_data.resistance is not None and target_data.resistance == attacker_data.energyType:
        dmg = max(0, dmg - 30)
    return dmg


def opponent_best_damage_estimate(op_active, my_active):
    """Rough estimate of the most damage the opponent's active could deal
    next turn, applying weakness doubling using the OPPONENT Pokemon's own
    type against OUR active's weakness (same generalized mechanic as
    effective_damage, just from the other side)."""
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
        if have + 1 >= req:  # assume they attach one more energy next turn
            dmg = atk.damage
            if my_weak is not None and data.energyType == my_weak:
                dmg *= 2
            best = max(best, dmg)
    return best


class Plan:
    attacker = -1
    attack_id = -1
    target_active = True
    target_bench_index = -1
    needs_energy = False
    wins_game = False
    value = -1.0
    use_gust = False
    danger = False
    ramp_attacker = -1
    ramp_attack_id = -1


def compute_plan(state, my_index, my_state, op_state, hand_counts):
    plan = Plan()
    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None
    if my_active is None or op_active is None:
        return plan

    op_data = card_table.get(op_active.id)
    threat = opponent_best_damage_estimate(op_active, my_active)
    if my_active.hp <= threat + 20.0:
        plan.danger = True

    op_bench = list(op_state.bench)
    has_boss = hand_counts.get(Boss_Orders, 0) > 0

    my_cards = [(0, my_active)] + [(i + 1, p) for i, p in enumerate(my_state.bench)]

    best_value = -1.0
    best_ramp_value = -1.0
    for idx, pokemon in my_cards:
        data = card_table.get(pokemon.id)
        if data is None:
            continue
        for aid in data.attacks:
            atk = attack_table.get(aid)
            if atk is None:
                continue
            ready, needs_one = attack_ready(pokemon, atk, state, hand_counts)
            if not ready:
                continue

            if atk.damage <= 0:
                # Non-damaging attacks: either a genuine ramp move (handled
                # specially below) or a narrow conditional effect (e.g. Scream
                # Tail ex's "Scream", first-turn-if-going-second-only, no
                # damage). Neither is a real "plan" candidate for the main
                # damage/KO comparison loop below - a bug here previously let
                # a 0-damage attack register a nonzero plan value purely from
                # the unconditional gust-bonus arithmetic on a bench target,
                # hijacking plan.attacker away from real attackers.
                if is_ramp_attack(atk):
                    max_dmg_req = max(
                        (len(attack_table[a2].energies) for a2 in data.attacks
                         if attack_table.get(a2) is not None and attack_table[a2].damage > 0),
                        default=0,
                    )
                    if len(pokemon.energies) < max_dmg_req and W["ramp_value"] > best_ramp_value:
                        best_ramp_value = W["ramp_value"]
                        plan.ramp_attacker = idx
                        plan.ramp_attack_id = aid
                continue

            targets = [(True, -1, op_active, op_data)]
            if has_boss:
                for bi, bp in enumerate(op_bench):
                    targets.append((False, bi, bp, card_table.get(bp.id)))

            for is_active_target, bench_idx, target, target_data in targets:
                dmg = effective_damage(data, atk, target_data)
                is_ko = dmg >= target.hp
                pv = prize_value(target.id)
                if is_ko:
                    value = W["ko_prize_value"] * pv
                    wins = len(op_state.prize) <= pv
                else:
                    value = W["damage_progress"] * min(1.0, dmg / max(1, target.hp))
                    wins = False
                value += W["energy_on_attacker"] * len(pokemon.energies)
                if idx == 0:
                    value += W["active_attacker_bonus"]
                if not is_active_target:
                    value += W["gust_bonus"] * (pv / 3.0)
                if wins:
                    value = 1e9

                if value > best_value:
                    best_value = value
                    plan.attacker = idx
                    plan.attack_id = aid
                    plan.target_active = is_active_target
                    plan.target_bench_index = bench_idx
                    plan.needs_energy = needs_one
                    plan.wins_game = wins
                    plan.use_gust = not is_active_target
                    plan.value = value
    return plan


def hand_target_score(card_id, field_counts):
    if card_id in ATTACKER_IDS:
        n_in_play = field_counts.get(card_id, 0)
        if n_in_play >= 2:
            return 300.0
        base = {Latias_ex: 3200.0, Xerneas_ex: 3000.0, Cresselia: 2200.0, Scream_Tail_ex: 2000.0}[card_id]
        return base
    if card_id == Basic_Psychic_Energy:
        return 2400.0 + W["energy_attach_readiness"] * 0.3
    if card_id in SEARCH_ITEMS:
        return 2000.0 + W["search_value"]
    if card_id in DRAW_SUPPORTERS or card_id == Waitress:
        return 2300.0 + W["card_draw_value"]
    if card_id == Crispin:
        return 2350.0 + W["energy_attach_readiness"] * 0.3
    if card_id == Boss_Orders:
        return 2100.0 + W["gust_bonus"] * 0.3
    if card_id == Night_Stretcher:
        return 1800.0
    if card_id == Switch:
        return 1200.0
    if card_id == Mystery_Garden:
        return 900.0 + W["stadium_value"]
    return 500.0


def attach_score(pokemon, active):
    if pokemon is None or pokemon.id not in ATTACKER_IDS:
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
    score += W["energy_attach_readiness"] * (have + 1)
    if active:
        score += W["energy_attach_active_bonus"]
    return score


def play_score(card_id, data, state, plan, stadium_id):
    if data is None:
        return -1.0
    if data.cardType == CardType.POKEMON:
        return 20000.0
    if data.cardType == CardType.SUPPORTER and state.supporterPlayed:
        return -1.0
    if card_id == Switch:
        return 6000.0 if plan.attacker >= 1 else -1.0
    if card_id == Boss_Orders:
        return (6200.0 + W["gust_bonus"]) if plan.use_gust else -1.0
    if card_id in DRAW_SUPPORTERS:
        hand_size = len(getattr(state.players[state.yourIndex], "hand", None) or [])
        need = max(0.0, 5.0 - hand_size)
        return 5000.0 + W["card_draw_value"] * (1.0 + need * 0.2)
    if card_id in SEARCH_ITEMS or card_id in (Crispin, Waitress):
        return 4800.0 + W["search_value"]
    if card_id == Night_Stretcher:
        return 4000.0
    if card_id == Mystery_Garden:
        return -1.0 if stadium_id == Mystery_Garden else (3000.0 + W["stadium_value"])
    return 3000.0


def _score_options(obs, state, select, context, my_index, my_state, op_state, field_counts, hand_counts,
                    stadium_id, plan):
    my_active = my_state.active[0] if my_state.active else None
    scores = []
    for o in select.option:
        score = 0.0
        if o.type == OptionType.NUMBER:
            score = float(o.number)
        elif o.type == OptionType.YES:
            score = -1.0 if context == SelectContext.IS_FIRST else 1.0
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
                        score = 100.0 + W["energy_on_attacker"] * energy_count + W["switch_target_health"] * hp_frac * 10.0
                        if plan.attacker >= 1 and (plan.attacker - 1) == o.index:
                            score += 500.0
                    else:
                        score = W["gust_bonus"] * (prize_value(card.id) / 3.0) + (100.0 - hp_frac * 100.0)
                        if plan.use_gust and plan.target_bench_index == o.index:
                            score += 1000.0
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    score = {Latias_ex: 40.0, Xerneas_ex: 30.0, Cresselia: 15.0, Scream_Tail_ex: 10.0}.get(card.id, 5.0)
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = W["bench_development"]
                    if card.id == Latias_ex:
                        score += 20.0
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.LOOK):
                    score = hand_target_score(card.id, field_counts)
                elif context == SelectContext.DISCARD:
                    score = -hand_target_score(card.id, field_counts) * 0.5
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(pokemon=card, active=(o.area == AreaType.ACTIVE))
                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if hp > 0:
                        score = 1000.0 - hp
                else:
                    score = hand_target_score(card.id, field_counts)
        elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY, OptionType.TOOL_CARD):
            score = 1.0 if o.playerIndex != my_index else 0.5
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table.get(card.id)
            score = play_score(card.id, data, state, plan, stadium_id)
        elif o.type == OptionType.ATTACH:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
            is_planned_target = (
                (plan.attacker == 0 and o.inPlayArea == AreaType.ACTIVE)
                or (plan.attacker >= 1 and o.inPlayArea == AreaType.BENCH and o.inPlayIndex == plan.attacker - 1)
            )
            if plan.needs_energy and is_planned_target:
                score += 5000.0
        elif o.type == OptionType.EVOLVE:
            score = 100.0  # unused by this deck (no evolutions) - defensive fallback only
        elif o.type == OptionType.ABILITY:
            score = 100.0  # unused by this deck (Skyliner is passive) - defensive fallback only
        elif o.type == OptionType.DISCARD:
            score = 0.0
        elif o.type == OptionType.RETREAT:
            hp_frac = (my_active.hp / my_active.maxHp) if (my_active is not None and my_active.maxHp > 0) else 1.0
            score = -1.0
            if plan.danger:
                score = W["retreat_danger_weight"] * (1.5 - hp_frac)
            if plan.attacker >= 1:
                score = max(score, 2000.0)
        elif o.type == OptionType.ATTACK:
            if plan.wins_game and o.attackId == plan.attack_id:
                score = 1e9
            elif o.attackId == plan.attack_id and not plan.needs_energy:
                score = 1000.0 + max(0.0, plan.value) * 0.01
            elif o.attackId == plan.ramp_attack_id:
                score = 700.0
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
    return out


def _decide(obs):
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    for c in ([my_state.active[0]] if my_state.active and my_state.active[0] else []) + list(my_state.bench):
        field_counts[c.id] += 1
    for c in (my_state.hand or []):
        hand_counts[c.id] += 1
    stadium_id = state.stadium[0].id if state.stadium else 0

    plan = Plan()
    if context == SelectContext.MAIN:
        plan = compute_plan(state, my_index, my_state, op_state, hand_counts)

    scores = _score_options(obs, state, select, context, my_index, my_state, op_state,
                             field_counts, hand_counts, stadium_id, plan)
    picked = _pick_from_scores(select, scores)
    if os.environ.get("LX_DEBUG") == "1" and context == SelectContext.MAIN:
        opts = [(i, int(o.type), getattr(o, "attackId", None)) for i, o in enumerate(select.option)]
        print(f"[LX_DEBUG] plan(attacker={plan.attacker},attack_id={plan.attack_id},needs_energy={plan.needs_energy},"
              f"value={plan.value:.1f},wins={plan.wins_game},ramp={plan.ramp_attacker},ramp_id={plan.ramp_attack_id}) "
              f"opts={list(zip(opts, scores))} picked={picked}")
    return picked


def agent(obs_dict: dict) -> list:
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return my_deck

    if obs.select is None:
        return my_deck

    try:
        return _decide(obs)
    except Exception:
        try:
            sel = obs.select
            n = len(sel.option)
            k = min(max(1, sel.minCount), n) if n else 0
            return list(range(k)) if k else [0]
        except Exception:
            return [0]
