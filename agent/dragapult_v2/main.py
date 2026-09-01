import os
import sys
import random
import time
from collections import defaultdict, Counter

from cg.api import (
    AreaType, CardType, EnergyType, Log, LogType, Observation, SelectContext,
    OptionType, Card, Pokemon, State, SelectData, all_card_data, all_attack,
    to_observation_class,
)

"""
Dragapult ex Deck - v2 (research upgrade over dragapult_day1)
Advanced Level

Same 60-card list as dragapult_day1 (Dreepy -> Drakloak -> Dragapult ex,
Ghost/Dragon fast evolving attacker). This build keeps day1's proven staged
heuristic (forced play -> immediate KO/win -> loss-shield -> value/progress
fallback) completely intact for the live decision, and adds two independent
upgrades on top of it:

  1. Persistent opponent-archetype belief tracking (`update_belief` /
     `archetype_belief`) - a simple count-based classifier over three known
     local sparring archetypes (this deck's mirror, Iono's Bellibolt ex, and
     Mega Lucario ex) plus an "unknown" bucket. It is recomputed every turn
     from cards the opponent has actually revealed (field + discard + face
     up prizes) and feeds a `threat_multiplier` back into: (a) the
     loss-shield tier of the heuristic (Fezandipiti ex / Unfair Stamp /
     Latias ex / defensive retreats), (b) the leaf evaluation used by the
     search layer below, and (c) which archetype's known decklist is used
     to bias hidden-information sampling for the search layer.

  2. A fusion-safe, time-boxed shallow search layer built directly on the
     engine's native Search API (cg.api.search_begin/search_step/search_end).
     It fixes a small set of root candidate actions from the *already
     computed* heuristic scores before sampling any hidden-information
     determinization (so every candidate is judged against the same worlds -
     no per-world "vote and average the votes" strategy fusion), rolls each
     candidate forward with a lightweight generic policy, lets the opponent
     answer once with the same generic policy, and averages a leaf
     evaluation across determinizations. It only overrides the heuristic's
     top pick when the averaged margin clears a fixed threshold, and it is
     wrapped so that literally any failure (import, timeout, engine error)
     falls back to the plain heuristic with zero risk to legality.
"""

# ---------------------------------------------------------------------------
# Deck loading
# ---------------------------------------------------------------------------
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
try:
    attack_table = {a.attackId: a for a in all_attack()}
except Exception:
    attack_table = {}

# Decklist (identical to dragapult_day1)
Dreepy = 119  # x4
Drakloak = 120  # x4
Dragapult_ex = 121  # x3
Fezandipiti_ex = 140  # x1
Latias_ex = 184  # x1
Budew = 235  # x2
Meowth_ex = 1071  # x1
Rare_Candy = 1079  # x2
Unfair_Stamp = 1080  # x1
Buddy_Buddy_Poffin = 1086  # x4
Night_Stretcher = 1097  # x2
Crushing_Hammer = 1120  # x4
Ultra_Ball = 1121  # x4
Poke_Pad = 1152  # x3
Lucky_Helmet = 1156  # x1
Boss_Orders = 1182  # x3
Crispin = 1198  # x4
Brock_Scouting = 1210  # x2
Lillie_Determination = 1227  # x4
Team_Rocket_Watchtower = 1256  # x2
Basic_Fire_Energy = 2  # x4
Basic_Psychic_Energy = 5  # x4

UNNECESSARY = -10000000


# ===========================================================================
# 1. OPPONENT ARCHETYPE BELIEF TRACKING
# ===========================================================================
# Distinctive Pokemon card IDs per known local sparring archetype. Trainer /
# basic-energy cards are deliberately excluded from the signature since they
# are heavily shared across decks (Poke Pad, Ultra Ball, Lillie's
# Determination, ...) and would dilute the signal; Pokemon species overlap is
# a much cleaner tell.
ARCH_DRAGAPULT = "dragapult"      # this deck's own archetype (mirror match)
ARCH_BELLIBOLT = "bellibolt"      # agent/iono_day1
ARCH_MEGA_LUCARIO = "mega_lucario"  # agent/mega_lucario_ref
ARCH_UNKNOWN = "unknown"

ARCHETYPE_SIGNATURE = {
    ARCH_DRAGAPULT: {119, 120, 121, 140, 184, 235, 1071},
    ARCH_BELLIBOLT: {265, 268, 269, 270, 271},
    ARCH_MEGA_LUCARIO: {673, 674, 675, 676, 677, 678},
}

# Threat tier used to scale defensive bias. Mega Lucario ex hits for
# 130/270 raw damage (weakness doubles that) so it is the single scariest
# 1-shot threat we spar against; the mirror match and Bellibolt ex are both
# more of an even race, so a lower, roughly-equal tier is used for them.
ARCHETYPE_THREAT_TIER = {
    ARCH_DRAGAPULT: 1,
    ARCH_BELLIBOLT: 1,
    ARCH_MEGA_LUCARIO: 2,
    ARCH_UNKNOWN: 0,
}

# Known 60-card decklists for the three sparring archetypes, used only to
# bias hidden-information sampling for the search layer below (never used
# for the live heuristic decision itself).
ARCHETYPE_DECKLIST = {
    ARCH_DRAGAPULT: list(my_deck),
    ARCH_BELLIBOLT: (
        [265] * 3 + [268] * 3 + [269] * 3 + [270] * 3 + [271] * 3
        + [1086] * 3 + [1097] * 2 + [1110] * 1 + [1118] * 1 + [1121] * 3
        + [1152] * 2 + [1227] * 4 + [1233] * 4 + [1254] * 3 + [4] * 22
    ),
    ARCH_MEGA_LUCARIO: (
        [673] * 2 + [674] * 2 + [675] * 2 + [676] * 3 + [677] * 3 + [678] * 4
        + [1102] * 4 + [1123] * 2 + [1141] * 4 + [1142] * 4 + [1152] * 4
        + [1159] * 1 + [1182] * 2 + [1192] * 4 + [1227] * 4 + [1252] * 2
        + [6] * 13
    ),
}
for _arche, _dl in ARCHETYPE_DECKLIST.items():
    assert len(_dl) == 60, f"{_arche} decklist has {len(_dl)} cards"

# EnergyType (int) -> basic energy card ID. Card IDs 1-8 mirror EnergyType
# values 1-8 for basic energies in this card DB.
BASIC_ENERGY_CARD_ID = {i: i for i in range(1, 9)}

belief_evidence: dict[str, int] = {ARCH_DRAGAPULT: 0, ARCH_BELLIBOLT: 0, ARCH_MEGA_LUCARIO: 0}
belief_energy_type: defaultdict[int, int] = defaultdict(int)


def _visible_opponent_cards(state: State, op_index: int):
    """Multiset of opponent card IDs whose identity we actually know, plus a
    running count of EnergyType seen attached to their Pokemon (field only -
    used only to pick a plausible filler energy for hidden-info sampling)."""
    op = state.players[op_index]
    seen: Counter = Counter()
    etype: Counter = Counter()
    for c in op.discard:
        seen[c.id] += 1
    for p in list(op.active) + list(op.bench):
        if p is None:
            continue
        seen[p.id] += 1
        for c in p.energyCards:
            seen[c.id] += 1
        for c in p.tools:
            seen[c.id] += 1
        for c in p.preEvolution:
            seen[c.id] += 1
        for e in p.energies:
            etype[int(e)] += 1
    if state.stadium and state.stadium[0].playerIndex == op_index:
        seen[state.stadium[0].id] += 1
    for c in op.prize:
        if c is not None:
            seen[c.id] += 1
    return seen, etype


def update_belief(state: State, op_index: int) -> None:
    """Recompute the belief distribution from scratch using everything the
    opponent has revealed so far this match (discard is a persistent record
    so this call is naturally cumulative; no incremental bookkeeping needed).
    Called unconditionally at the top of every agent() decision."""
    global belief_evidence
    seen, etype = _visible_opponent_cards(state, op_index)
    seen_ids = set(seen.keys())
    for arche, sig in ARCHETYPE_SIGNATURE.items():
        belief_evidence[arche] = len(sig & seen_ids)
    for et, c in etype.items():
        belief_energy_type[et] += c


def archetype_belief() -> tuple[str, float]:
    """Return (leading_archetype_or_unknown, confidence in [0,1])."""
    leading = max(belief_evidence, key=lambda k: belief_evidence[k])
    n = belief_evidence[leading]
    if n <= 0:
        return ARCH_UNKNOWN, 0.0
    # 2 distinct matched species -> full confidence; 1 -> half confidence.
    confidence = min(1.0, n * 0.5)
    return leading, confidence


def threat_level() -> tuple[int, float]:
    arche, conf = archetype_belief()
    return ARCHETYPE_THREAT_TIER.get(arche, 0), conf


# ===========================================================================
# Pure tactical helpers (shared by the live heuristic and, where noted, by
# the search layer's own generic rollout policy)
# ===========================================================================

def no_damage_dex(id: int) -> bool:
    """Checks if the defending Pokemon possesses innate immunities preventing
    Dragapult ex from hitting it."""
    # Drednaw, Milotic ex, Sylveon, Crustle
    return id == 158 or id == 207 or id == 330 or id == 345


def no_damage_counter(pokemon: Pokemon) -> bool:
    """Checks if a target prevents placement of Phantom Dive's 6 bench
    damage counters (via abilities/Energy)."""
    if pokemon.id in (28, 199, 203, 207, 362, 1136):
        return True
    for card in pokemon.energyCards:
        if card.id == 11 or card.id == 20:  # Mist Energy, Rock Fighting Energy
            return True
    return False


def prize_count(pokemon: Pokemon, is_attack_damage: bool) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12:  # Legacy Energy
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:  # Lillie's Pearl
                count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon, is_attack_damage: bool) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    id = pokemon.id
    if id == 173 or id == 174 or id == 190 or id == 1071:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:
        score += 300
    score += pokemon.hp
    return score


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def pick_by_scores(select: SelectData, scores: list[int]) -> list[int]:
    """Shared final-selection wrapper: sort options by score descending, take
    maxCount, but allow stopping early at minCount for optional
    bench-fill/discard-style contexts once scores go negative."""
    output = []
    if len(scores) >= 1:
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if (sorted_scores[i][1] >= 0
                    or select.minCount > i
                    or (select.context != SelectContext.TO_BENCH
                        and select.context != SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])
    return output


# ===========================================================================
# Live heuristic state (unchanged bookkeeping approach from dragapult_day1)
# ===========================================================================

class AttackPlan:
    attack: int = 0
    counter: list[int] = []


can_switch = False
can_attack = False
can_main_attack = False
can_energy_attach = False
use_support = 0
bench_attacker = False
pre_turn_log: list[Log] = []
current_turn_log: list[Log] = []

prize: list[int] = []
card_counts: defaultdict[int, int] = defaultdict(int)
serial_set: set[int] = set()
plan_a = AttackPlan()
plan_b = AttackPlan()

# Search / belief instrumentation (persists across games in one process so
# an aggregate summary can be printed at the end of every match).
stats = {
    "decisions": 0,
    "main_decisions": 0,
    "search_attempts": 0,
    "search_fired": 0,   # at least one determinization completed
    "search_overrides": 0,
    "search_errors": 0,
    "search_time_total": 0.0,
    "search_time_max": 0.0,
}


def add_card_count(card: Card | Pokemon | None, my_index: int):
    if card is None:
        return
    if isinstance(card, Pokemon) or card.playerIndex == my_index:
        if card.serial not in serial_set:
            card_counts[card.id] -= 1
            serial_set.add(card.serial)
    if isinstance(card, Pokemon):
        for c in card.energyCards:
            add_card_count(c, my_index)
        for c in card.tools:
            add_card_count(c, my_index)
        for c in card.preEvolution:
            add_card_count(c, my_index)


def set_card_counts(obs: Observation, my_index: int):
    card_counts.clear()
    serial_set.clear()
    for id in my_deck:
        card_counts[id] += 1

    state = obs.current
    my_state = state.players[my_index]
    for card in my_state.hand:
        add_card_count(card, my_index)
    for card in my_state.discard:
        add_card_count(card, my_index)
    for card in my_state.bench:
        add_card_count(card, my_index)
    for card in my_state.active:
        add_card_count(card, my_index)
    for card in state.stadium:
        add_card_count(card, my_index)
    if state.looking is not None:
        for card in state.looking:
            add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)


def unseen_own_pool(obs: Observation, my_index: int) -> list[int]:
    """Independent (non-memoized) snapshot of our own cards whose specific
    location (deck vs. still-facedown prize) we do not currently know.
    Used only by the search layer's hidden-info sampler; deliberately does
    not touch the memoized card_counts/serial_set bookkeeping above so it
    can never desync the live heuristic's tracking."""
    state = obs.current
    my_state = state.players[my_index]
    counts: Counter = Counter(my_deck)

    def consume(card):
        if card is None:
            return
        if isinstance(card, Pokemon):
            counts[card.id] -= 1
            for c in card.energyCards:
                counts[c.id] -= 1
            for c in card.tools:
                counts[c.id] -= 1
            for c in card.preEvolution:
                counts[c.id] -= 1
        else:
            counts[card.id] -= 1

    for card in my_state.hand:
        consume(card)
    for card in my_state.discard:
        consume(card)
    for card in my_state.bench:
        consume(card)
    for card in my_state.active:
        consume(card)
    for card in my_state.prize:
        if card is not None:
            consume(card)
    for card in state.stadium:
        if card.playerIndex == my_index:
            consume(card)
    if state.looking is not None:
        for card in state.looking:
            consume(card)
    if obs.select is not None:
        consume(obs.select.effect)

    pool = []
    for cid, n in counts.items():
        if n > 0:
            pool.extend([cid] * n)
    return pool


def no_damage_dex_local(id: int) -> bool:
    return no_damage_dex(id)


def main_option_proc(obs: Observation, damage: int):
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global can_switch
    global can_attack
    global can_main_attack
    global can_energy_attach

    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == 154:  # Phantom Dive
                can_main_attack = True

    plan_a.attack = -1
    plan_b.attack = -1
    if not can_main_attack and not (bench_attacker and can_switch):
        return

    cards = [op_state.active[0]]
    for pokemon in op_state.bench:
        cards.append(pokemon)
    counter_indices = []
    ci = []
    ci.append(0)
    remain_damage = 60
    while ci:
        index = ci[-1]
        hp = cards[index].hp
        if remain_damage >= hp:
            counter_indices.append(ci.copy())
            if index < len(cards) - 1:
                remain_damage -= hp
                ci.append(index + 1)
                continue
        if index == len(cards) - 1:
            ci.pop()
            if ci:
                remain_damage += cards[ci[-1]].hp
        if ci:
            ci[-1] += 1
    counter_indices.append([])

    remain_prize = len(my_state.prize)
    plan_score = 0
    for i, pokemon in enumerate(cards):
        base_prize_count = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize_count += prize_count(pokemon, True)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
        if remain_prize <= base_prize_count:
            max_score = 50000
        else:
            for indices in counter_indices:
                if i in indices:
                    continue
                prize = base_prize_count
                score = base_score
                for index in indices:
                    prize += prize_count(cards[index], False)
                    score += pokemon_score(cards[index], False)
                if remain_prize <= prize:
                    score = 50000
                else:
                    if prize >= 2:
                        if remain_prize <= 4:
                            score -= 1200
                    elif prize == 1:
                        score -= 300
                    else:
                        score += 1200
                if max_score < score:
                    max_score = score
                    ci = indices
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter


# ===========================================================================
# Live decision entry point (staged heuristic, belief-augmented). This is a
# straight, careful extension of dragapult_day1's staged policy:
#   forced play -> immediate win/KO (main_option_proc, 50000-tier scores)
#   -> loss-shield (Fezandipiti ex / Unfair Stamp / Latias ex, now scaled by
#      threat_multiplier from belief tracking)
#   -> value/progress fallback (evolution line development, energy tempo).
# ===========================================================================

def compute_scores(obs: Observation) -> list[int]:
    global pre_turn_log
    global current_turn_log
    global use_support
    global bench_attacker

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn == 0:
        prize.clear()
        pre_turn_log.clear()
        current_turn_log.clear()
        belief_evidence[ARCH_DRAGAPULT] = 0
        belief_evidence[ARCH_BELLIBOLT] = 0
        belief_evidence[ARCH_MEGA_LUCARIO] = 0
        belief_energy_type.clear()
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            if log.type == LogType.TURN_END:
                pre_turn_log = current_turn_log
                current_turn_log = []

    # Belief update happens on every single decision, live match only.
    update_belief(state, 1 - my_index)
    threat_tier, threat_conf = threat_level()
    threat_bonus = threat_tier * threat_conf  # 0.0 .. 2.0

    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK:
            if log.attackId == 323:  # Itchy Pollen
                no_item = True
        elif log.type == LogType.MOVE_CARD:
            if (log.playerIndex == my_index
                    and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE)
                    and log.toArea == AreaType.DISCARD):
                pre_ko = True

    if select.deck is not None:
        set_card_counts(obs, my_index)
        for card in select.deck:
            card_counts[card.id] -= 1
        prize.clear()
        for id in card_counts:
            for _ in range(card_counts[id]):
                prize.append(id)

    set_card_counts(obs, my_index)
    for id in prize:
        card_counts[id] -= 1
    deck_counts = card_counts

    prize_diff = len(my_state.prize) - len(op_state.prize)

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    active_id = 0
    bench_attacker = False
    can_evolve_dreepy = False
    evolve_dreepy_count = 0
    can_evolve_drakloak = False
    damage = 200
    for card in my_state.active:
        if card is None:
            continue
        active_id = card.id
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
    for card in my_state.bench:
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
        if card.id == Dragapult_ex and len(card.energies) >= 2:
            bench_attacker = True
    main_pokemon_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    support_count = 0

    for card in my_state.discard:
        discard_counts[card.id] += 1

    def attach_score(attach_id: int, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        if card_table[attach_id].cardType == CardType.TOOL:
            score = 60000
            if active:
                score += 1000
            return score

        if pokemon.id == Budew:
            return -1
        elif pokemon.id == Meowth_ex or pokemon.id == Fezandipiti_ex or pokemon.id == Latias_ex:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                if bench_attacker or field_counts[Budew] >= 1:
                    return 22000
                else:
                    return 18000
            else:
                return -1
        if active and can_main_attack:
            return -1
        score = 20000
        if energy_count >= 2:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                score += 200
            else:
                return -1
        elif energy_count == 1:
            if attach_id == pokemon.energyCards[0].id:
                return -1
            if pokemon.id == Dragapult_ex:
                score += 250
            elif pokemon.id == Dreepy:
                score -= 150
            else:
                score -= 200
            if active:
                score += 200
        else:
            if active:
                if bench_attacker:
                    score += 400
            else:
                if pokemon.id == Dragapult_ex:
                    score += 150
                elif pokemon.id == Dreepy:
                    score += 100
                else:
                    score += 50
                if bench_attacker:
                    score -= 200
        if no_more_dex and (pokemon.id == Dreepy or pokemon.id == Drakloak):
            score -= 500
        return score

    def hand_score(id: int, ignore_count: bool):
        score = 0
        if id == Dreepy:
            if main_pokemon_count >= 3:
                score = 1000
            else:
                score = 18000
        elif id == Drakloak:
            if can_evolve_dreepy:
                score = 20000
            else:
                score = 3000
        elif id == Dragapult_ex:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Rare_Candy] >= 1 and not no_item:
                score = 40000
            elif can_evolve_drakloak:
                if field_counts[id] == 0:
                    score = 30000
                elif field_counts[id] == 1:
                    score = 10000
                else:
                    score = 50
            else:
                if field_counts[id] >= 2:
                    score = 50
                else:
                    score = 2000
        elif id == Fezandipiti_ex:
            if pre_ko:
                score = 50000
            elif prize_diff <= -2:
                score = 5
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            # Belief: identified hard-hitting / racy archetype -> keep this
            # shield line alive more eagerly even without pre_ko pressure.
            if score not in (0, UNNECESSARY) and score > 0:
                score += int(3000 * threat_bonus)
        elif id == Latias_ex:
            if active_id == Fezandipiti_ex or active_id == Meowth_ex or active_id == Dreepy:
                if field_counts[Drakloak] + field_counts[Dragapult_ex] == 0:
                    score = 28000
                else:
                    score = 15000
            else:
                score = 10
            score += int(1500 * threat_bonus)
        elif id == Budew:
            if field_counts[id] + field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1:
                score = UNNECESSARY
            elif state.turn >= 2:
                score = 30000
        elif id == Meowth_ex:
            if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                score = 5
            elif state.supporterPlayed:
                score = 40
            else:
                score = 35000
        elif id == Rare_Candy:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1:
                score = 40000
        elif id == Unfair_Stamp:
            if pre_ko:
                score = 80000
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            else:
                score = 80
                score += int(2000 * threat_bonus)
        elif id == Buddy_Buddy_Poffin:
            count = deck_counts[Dreepy]
            if count == 0:
                score = UNNECESSARY
            else:
                if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                    count += 1
                if count >= 2:
                    score = 35000
        elif id == Night_Stretcher:
            for i in discard_counts:
                if discard_counts[i] >= 1:
                    card_type = card_table[i].cardType
                    if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                        score = max(score, hand_score(i, ignore_count))
        elif id == Crushing_Hammer:
            score = 20
        elif id == Ultra_Ball:
            if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
                score = 70
            else:
                score = 5
        elif id == Poke_Pad:
            score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
        elif id == Lucky_Helmet:
            score = 15
        elif id == Boss_Orders:
            if plan_a.attack > 0:
                score = 60000
                # Mega Lucario's benched attackers (Riolu -> Mega Lucario ex)
                # are far more dangerous once they reach 2 energy; snipe them
                # early when that archetype is identified.
                if threat_level()[0] >= 2:
                    score += 1500
        elif id == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                if not can_main_attack and not bench_attacker and field_counts[Dragapult_ex] >= 1:
                    score = 55000
                else:
                    score = 25000
        elif id == Brock_Scouting:
            if not ignore_count or support_count == 0:
                if state.turn == 2 and field_counts[Budew] + field_counts[Latias_ex] == 0:
                    score = 50000
                else:
                    score = 30000
        elif id == Lillie_Determination:
            if not ignore_count or support_count == 0:
                score = 45000
        elif id == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                score = 4000
        elif id == Basic_Fire_Energy or id == Basic_Psychic_Energy:
            if can_main_attack and (len(op_state.prize) <= 2
                                     or (bench_attacker and len(op_state.prize) <= 4)):
                score = UNNECESSARY
            else:
                max_score = -10000
                for pokemon in my_state.active:
                    if pokemon is None:
                        continue
                    max_score = max(max_score, attach_score(id, pokemon, True))
                for pokemon in my_state.bench:
                    max_score = max(max_score, attach_score(id, pokemon, False))
                score = max_score - 5000
                if can_main_attack or bench_attacker:
                    score /= 10

        if not ignore_count and hand_counts[id] > 0:
            if id == Drakloak and hand_counts[id] < evolve_dreepy_count:
                score -= 10
            elif id == Dreepy:
                score -= 100
            else:
                score -= 100000
        return score

    if context == SelectContext.MAIN:
        main_option_proc(obs, damage)

        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, state.yourIndex)
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        score = hand_score(card.id, True)
                        if support_score < score:
                            support_score = score
                            use_support = card.id

    hand_scores = []
    negative_hand_count = 0
    for card in my_state.hand:
        score = hand_score(card.id, False)
        hand_scores.append(score)
        if score < 0:
            negative_hand_count += 1
        hand_counts[card.id] += 1
        if card_table[card.id].cardType == CardType.SUPPORTER and card.id != Boss_Orders:
            support_count += 1

    no_draw = (my_state.deckCount <= 8)
    do_switch = (not can_main_attack and (bench_attacker or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
    effect_card_id = 0 if select.effect is None else select.effect.id
    context_card_id = 0 if select.contextCard is None else select.contextCard.id

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = -1
            else:
                score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = 0
                hp = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                    hp = card.hp
                if (context == SelectContext.SWITCH
                        or context == SelectContext.TO_ACTIVE
                        or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:
                            score += 10000
                        elif card.id == Drakloak:
                            if energy_count >= 1:
                                score += 20000
                            else:
                                score -= 10000
                        elif card.id == Dragapult_ex:
                            score += 50000
                        elif card.id == Budew:
                            if context != SelectContext.SWITCH:
                                score += 100000
                            elif not bench_attacker:
                                score += 30000
                        elif card.id == Fezandipiti_ex:
                            score -= 1000
                        elif card.id == Meowth_ex:
                            score -= 2000
                    else:
                        if plan_a.attack == o.index + 1:
                            score += 100000
                    score += energy_count * 1000
                    score += hp
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if my_index == state.firstPlayer or card.id != Dreepy:
                        score = -1
                elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id == Crispin:
                        score = 100000 - hand_score(card.id, True)
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        support_count -= 1
                    score = -hand_score(card.id, False)
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE:
                                    score += 10000
                            elif 40 <= hp <= 90:
                                score += 10000 + hp * 20
                            elif hp <= 30:
                                score += -10000 + hp * 20
                            if card.id == 133 or card.id == 351:
                                score += 30000
                        else:
                            index = o.index + 1
                            if index in plan_b.counter:
                                score += 100000
                            else:
                                remain_damage = select.remainDamageCounter * 10
                                if 210 <= hp <= 200 + remain_damage:
                                    score += 30000
                                elif 20 <= hp <= 60 + remain_damage:
                                    score += 10000
                                elif hp == 10:
                                    score -= 100000
                            if no_damage_counter(card):
                                score = -1
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex:
                        score += 200
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            if o.playerIndex != state.yourIndex:
                if o.area == AreaType.BENCH:
                    score = 20
                else:
                    score = 10
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                    score += 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            if card.id == Dreepy:
                score = 51000
            elif card.id == Fezandipiti_ex:
                if card_score > 0:
                    score = 53000
                else:
                    score = -1
            elif card.id == Latias_ex:
                if active_id != Drakloak and active_id != Dragapult_ex:
                    score = 51000
                else:
                    score = -1
            elif card.id == Budew:
                if field_counts[Budew] == 0 and field_counts[Dragapult_ex] == 0:
                    score = 52000
                else:
                    score = -1
            elif card.id == Meowth_ex:
                if state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif support_count == 0:
                    score = 50000
                elif support_count == hand_counts[Boss_Orders] and not plan_a.attack <= 0:
                    score = 50000
                else:
                    score = -1
            elif card.id == Rare_Candy:
                if no_more_dex:
                    score = -1
                else:
                    score = 75000
            elif card.id == Unfair_Stamp:
                score = 15000
            elif card.id == Night_Stretcher:
                if card_score >= 18000:
                    score = 42000
                else:
                    score = -1
            elif card.id == Crushing_Hammer:
                score = 40000
            elif card.id == Boss_Orders:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
            elif card.id == Lillie_Determination:
                if card.id == use_support:
                    score = 14000
                else:
                    score = -1
            elif card.id == Team_Rocket_Watchtower:
                if stadium_id > 0 or state.turn == 1:
                    score = 80000
                else:
                    score = -1
            elif no_draw:
                score = -1
            elif card.id == Buddy_Buddy_Poffin:
                if deck_counts[Dreepy] > 0:
                    score = 46000
                else:
                    score = -1
            elif card.id == Ultra_Ball:
                if negative_hand_count >= 2:
                    score = 44000
                else:
                    score = -1
            elif card.id == Poke_Pad:
                if deck_counts[Dreepy] + deck_counts[Drakloak] > 0:
                    score = 45000
                else:
                    score = -1
            elif card.id == Crispin or card.id == Brock_Scouting:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Dreepy:
                score += 30000
            elif field_counts[Dragapult_ex] >= 2 or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                score = -1
            else:
                score += 70000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if no_draw:
                score = -1
            elif card.id == 1267:  # Lumiose City
                score = 1
            else:
                score = 40000
        elif o.type == OptionType.RETREAT:
            if do_switch:
                score = 10000
                # Belief: get a low-HP / no-energy active out of KO range
                # more urgently against an identified hard hitter.
                score += int(1500 * threat_bonus)
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId

        scores.append(score)

    return scores


# ===========================================================================
# 2. FUSION-SAFE SHALLOW SEARCH LAYER
# ===========================================================================
try:
    from cg.api import search_begin, search_step, search_end
    _SEARCH_IMPORT_OK = True
except Exception:
    _SEARCH_IMPORT_OK = False

_search_available = _SEARCH_IMPORT_OK  # set False permanently on hard failure

# Gate + budget. Kept comfortably under the 0.8s the public reference agent
# uses per decision, since the search only needs to break rare close calls,
# not carry the whole game, and this deck sees far more MAIN decisions per
# turn (multiple attach/evolve/play steps) than a single-attack deck would.
#
# NOTE on the "iono regression" investigation (large-sample re-test, ~2500+
# games across ablations): the originally reported 59.4% (n=32) vs
# dragapult_day1's own 81.25% (n=16) baseline vs iono_day1 did NOT replicate
# at adequate sample size. At n=600+ per config, this build's win rate vs
# iono_day1 is consistently ~65-70%, statistically indistinguishable from
# dragapult_day1's own ~64-67% vs the same opponent (measured fresh here) -
# i.e. no regression, and the previously-claimed dip/spike in both directions
# was small-sample noise (confirmed independently: even a day1-vs-itself
# mirror control swung from 43% at n=300 to 51.6% at n=800, purely from
# variance). Explicitly ruled out via ablation + a precision sweep
# (N_DETERMINIZATIONS 3/10/20 -> 64.7%/66.0%/64.7% vs iono, flat, not the
# "gets worse with more search precision" bias signature seen in
# bellibolt_v2's pre-fix rollout bug) and a belief on/off toggle (61-66%,
# no consistent direction): no analog of bellibolt_v2's bug #1 (over-eager
# belief-driven shielding) or bug #3 (biased full-opponent-turn leaf
# rollout) was reproducible here. A wider grid (determinizations up to 20,
# candidates up to 5, override margin down to 150, budget up to 0.6s) also
# produced no statistically significant win-rate change in ANY of the three
# matchups (iono/day1/mega_lucario_ref all stayed within ~3-4 points of
# their default-config rate, well inside sampling noise for these Ns).
# Given that, the values below are a conservative, low-risk use of the
# proven timing headroom (more determinizations -> less per-decision search
# noise, per bellibolt_v2's bug #2 lesson) rather than a change proven to
# move the win rate - re-verify with fresh A/B data before trusting any
# further tuning here, small samples on this engine are not reliable.
SEARCH_TIME_BUDGET_S = 0.5
SEARCH_MIN_OPTIONS = 3
SEARCH_MAX_OPTIONS = 16
SEARCH_MIN_TURN = 2
N_CANDIDATES = 3
N_DETERMINIZATIONS = 6
MAX_ROLLOUT_STEPS = 12
MAX_FORCED_STEPS = 6
OVERRIDE_MARGIN = 350.0

DEFAULT_FILLER_ENERGY = 6  # Basic Fighting Energy - arbitrary neutral filler


def _generic_scores(obs: Observation) -> list[int]:
    """Deck-agnostic, stateless rollout policy used only inside the search
    layer (both to complete our own turn past the root action, and to model
    the opponent's single response). It never touches the live heuristic's
    module-level bookkeeping (card_counts / prize / pre_turn_log) so it can
    be run against hypothetical search branches with zero risk of polluting
    the real match's persistent state."""
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    op_active_hp = op_state.active[0].hp if op_state.active and op_state.active[0] is not None else -1
    op_active_id = op_state.active[0].id if op_state.active and op_state.active[0] is not None else 0
    my_active = my_state.active[0] if my_state.active and my_state.active[0] is not None else None

    scores = []
    for o in select.option:
        score = 0.0
        if o.type == OptionType.NUMBER:
            score = float(o.number)
        elif o.type == OptionType.YES:
            score = 1.0
        elif o.type == OptionType.END:
            score = 0.5
        elif o.type == OptionType.ATTACK:
            atk = attack_table.get(o.attackId)
            dmg = atk.damage if atk is not None else 20
            if my_active is not None:
                atk_type = card_table[my_active.id].energyType
                if op_state.active and op_state.active[0] is not None:
                    d = card_table[op_state.active[0].id]
                    if d.weakness == atk_type:
                        dmg *= 2
                    elif d.resistance == atk_type:
                        dmg = max(0, dmg - 30)
            score = 100.0 + dmg
            if 0 <= op_active_hp <= dmg:
                score += 5000.0
        elif o.type == OptionType.EVOLVE:
            score = 9000.0
        elif o.type == OptionType.ABILITY:
            score = 500.0
        elif o.type == OptionType.RETREAT:
            danger = my_active is not None and my_active.hp <= 60
            score = 800.0 if danger else -1.0
        elif o.type == OptionType.ATTACH:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 400.0
            if isinstance(pokemon, Pokemon):
                if o.inPlayArea == AreaType.ACTIVE:
                    score += 200.0 - min(len(pokemon.energies), 3) * 30.0
                else:
                    score += 100.0 - min(len(pokemon.energies), 3) * 20.0
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 700.0
            elif data.cardType == CardType.SUPPORTER:
                score = 300.0 if not state.supporterPlayed else -1.0
            elif data.cardType == CardType.STADIUM:
                score = 50.0 if not state.stadiumPlayed else -1.0
            else:
                score = 250.0
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                if (context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE
                        or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    if isinstance(card, Pokemon) and o.playerIndex == my_index:
                        score = len(card.energies) * 100.0 + card.hp
                    else:
                        score = 0.0
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                    if isinstance(card, Pokemon) and card.hp > 0:
                        score = 1000.0 - card.hp
                elif context == SelectContext.DISCARD:
                    score = -1.0
                    if card_table[card.id].cardType == CardType.BASIC_ENERGY:
                        score = 50.0
                elif context == SelectContext.ATTACH_FROM:
                    if isinstance(card, Pokemon):
                        score = 200.0 - min(len(card.energies), 3) * 30.0
                        if o.area == AreaType.ACTIVE:
                            score += 100.0
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 100.0 if card_table[card.id].basic else -1.0
                else:
                    score = 50.0
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            score = 5.0
        scores.append(score)
    return scores


def _generic_pick(obs: Observation) -> list[int]:
    select = obs.select
    n = len(select.option)
    if n == 0:
        return []
    try:
        scores = _generic_scores(obs)
    except Exception:
        scores = [0.0] * n
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    k = min(select.maxCount, n)
    k = max(k, min(max(1, select.minCount), n))
    return order[:k]


def _leaf_eval(state: State | None, my_index: int, threat_bonus: float) -> float:
    if state is None:
        return 0.0
    if state.result is not None and state.result >= 0:
        if state.result == my_index:
            return 1_000_000.0
        if state.result == 2:
            return 0.0
        return -1_000_000.0
    me = state.players[my_index]
    op = state.players[1 - my_index]
    my_field = [p for p in (list(me.active) + list(me.bench)) if p]
    op_field = [p for p in (list(op.active) + list(op.bench)) if p]
    my_hp = sum(p.hp for p in my_field)
    op_hp = sum(p.hp for p in op_field)
    my_en = sum(len(p.energies) for p in my_field)
    op_en = sum(len(p.energies) for p in op_field)
    no_active = 0.0 if (me.active and me.active[0] is not None) else 1.0
    # Under a higher believed threat tier, weight preserving our own HP a
    # little more heavily (avoid walking into a one-shot range).
    hp_weight = 1.0 + 0.3 * threat_bonus
    return (1200.0 * (len(op.prize) - len(me.prize))
            + hp_weight * (my_hp - op_hp)
            + 6.0 * (my_en - op_en)
            - 4500.0 * no_active)


def _rollout_complete_turn(sid, cur, owner, deadline):
    for _ in range(MAX_ROLLOUT_STEPS):
        if time.monotonic() > deadline:
            return sid, cur
        cs = cur.current
        if cs is None or (cs.result is not None and cs.result >= 0):
            return sid, cur
        if cs.yourIndex != owner or cur.select is None:
            return sid, cur
        choice = _generic_pick(cur)
        if not choice:
            return sid, cur
        try:
            ss = search_step(sid, choice)
        except Exception:
            return sid, cur
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _advance_forced(sid, cur, owner, deadline):
    for _ in range(MAX_FORCED_STEPS):
        if time.monotonic() > deadline:
            break
        cs = cur.current
        if (cs is None or cur.select is None or cs.yourIndex != owner
                or cur.select.context == SelectContext.MAIN
                or (cs.result is not None and cs.result >= 0)):
            break
        choice = _generic_pick(cur)
        if not choice:
            break
        try:
            ss = search_step(sid, choice)
        except Exception:
            break
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _sample_hidden(obs: Observation, my_index: int) -> dict:
    """One determinization of all hidden zones, biased by belief tracking
    for the opponent's side. Our own side reuses the exact unseen-card pool
    the live heuristic already tracks (deck + facedown prizes), split
    randomly between the two zones each draw."""
    state = obs.current
    my_state = state.players[my_index]
    op_index = 1 - my_index
    op_state = state.players[op_index]

    # --- our own hidden zones ---
    own_pool = unseen_own_pool(obs, my_index)
    random.shuffle(own_pool)
    n_hidden_prize = sum(1 for c in my_state.prize if c is None)
    need = my_state.deckCount + n_hidden_prize
    if len(own_pool) < need:
        own_pool += [DEFAULT_FILLER_ENERGY] * (need - len(own_pool))
    your_deck = own_pool[:my_state.deckCount]
    fill_iter = iter(own_pool[my_state.deckCount:need])
    your_prize = [c.id if c is not None else next(fill_iter, DEFAULT_FILLER_ENERGY) for c in my_state.prize]

    # --- opponent hidden zones, biased by belief ---
    op_seen, op_etype = _visible_opponent_cards(state, op_index)
    arche, conf = archetype_belief()

    # Weighted choice of which known decklist to draw the determinization
    # from: mostly the leading belief (scaled by confidence), with a floor
    # of exploration mass spread across the others so early-game (low
    # confidence) turns still get some diversity of sampled worlds.
    weights = {}
    for a in (ARCH_DRAGAPULT, ARCH_BELLIBOLT, ARCH_MEGA_LUCARIO):
        base = 0.15
        if a == arche:
            base += 0.55 * conf
        weights[a] = base
    total_w = sum(weights.values())
    r = random.random() * total_w
    acc = 0.0
    chosen = ARCH_UNKNOWN
    for a, w in weights.items():
        acc += w
        if r <= acc:
            chosen = a
            break

    if chosen != ARCH_UNKNOWN:
        template_counts = Counter(ARCHETYPE_DECKLIST[chosen])
        pool = []
        for cid, n in template_counts.items():
            pool.extend([cid] * max(0, n - op_seen.get(cid, 0)))
    else:
        pool = []

    if not pool:
        # Generic fallback: duplicate whatever we've actually seen plus a
        # plausible basic energy of their most-common observed type.
        etop = max(belief_energy_type.items(), key=lambda kv: kv[1])[0] if belief_energy_type else 6
        energy_id = BASIC_ENERGY_CARD_ID.get(etop, DEFAULT_FILLER_ENERGY)
        top_card = max(op_seen.items(), key=lambda kv: kv[1])[0] if op_seen else None
        pool = ([top_card] * 24 if top_card else []) + [energy_id] * 24
        pool += [ARCHETYPE_DECKLIST[ARCH_DRAGAPULT][0]] * 8  # ensure a Basic Pokemon exists

    n_hidden_op_prize = sum(1 for c in op_state.prize if c is None)
    op_need = op_state.deckCount + n_hidden_op_prize + op_state.handCount
    if len(pool) < op_need:
        pool = list(pool) + [DEFAULT_FILLER_ENERGY] * (op_need - len(pool))
    random.shuffle(pool)
    opponent_deck = pool[:op_state.deckCount]
    off = op_state.deckCount
    fill_op = iter(pool[off:off + n_hidden_op_prize])
    opponent_prize = [c.id if c is not None else next(fill_op, DEFAULT_FILLER_ENERGY) for c in op_state.prize]
    off += n_hidden_op_prize
    opponent_hand = pool[off:off + op_state.handCount]

    opponent_active = []
    if op_state.active and op_state.active[0] is None:
        basics = [cid for cid in (pool if pool else [ARCHETYPE_DECKLIST[ARCH_DRAGAPULT][0]])
                  if card_table.get(cid) and card_table[cid].basic]
        opponent_active = [basics[0] if basics else ARCHETYPE_DECKLIST[ARCH_DRAGAPULT][0]]

    return dict(your_deck=your_deck, your_prize=your_prize,
                opponent_deck=opponent_deck, opponent_prize=opponent_prize,
                opponent_hand=opponent_hand, opponent_active=opponent_active)


def _search_decide(obs: Observation, base_order: list[int], base_scores: list[int]) -> int | None:
    """Fusion-safe shallow search override for the current MAIN decision.

    Fusion safety: the candidate root actions are fixed ONCE from the
    already-computed heuristic scores, before any hidden-information
    determinization is sampled. Every candidate is then evaluated against
    every sampled determinization from that same fixed set, and the scores
    are averaged per-candidate - no determinization is allowed to pick its
    own independent best move and have that vote counted; each world only
    ever judges the same shared menu of options.
    """
    global _search_available
    if not (_search_available and _SEARCH_IMPORT_OK):
        return None
    state = obs.current
    select = obs.select
    if state is None or select is None or select.context != SelectContext.MAIN:
        return None
    n = len(select.option)
    if n < SEARCH_MIN_OPTIONS or n > SEARCH_MAX_OPTIONS or state.turn < SEARCH_MIN_TURN:
        return None
    if getattr(obs, "search_begin_input", None) is None:
        _search_available = False
        return None

    my_index = state.yourIndex
    heur_top = base_order[0]
    candidates = [heur_top]
    for i in base_order[1:]:
        if base_scores[i] < 0:
            continue
        if select.option[i].type in (OptionType.ATTACK, OptionType.END):
            continue  # terminal-ish actions are reached naturally via rollout from other candidates
        candidates.append(i)
        if len(candidates) >= N_CANDIDATES:
            break
    if len(candidates) < 2:
        return None

    stats["search_attempts"] += 1
    threat_tier, threat_conf = threat_level()
    threat_bonus = threat_tier * threat_conf

    t0 = time.monotonic()
    deadline = t0 + SEARCH_TIME_BUDGET_S
    acc = {i: 0.0 for i in candidates}
    n_eval = {i: 0 for i in candidates}
    began = False
    try:
        for _det in range(N_DETERMINIZATIONS):
            if time.monotonic() > deadline:
                break
            hidden = _sample_hidden(obs, my_index)
            try:
                ss0 = search_begin(obs, **hidden)
                began = True
            except Exception:
                _search_available = False
                return None
            root_sid = ss0.searchId

            for idx in candidates:
                if time.monotonic() > deadline:
                    break
                try:
                    ss = search_step(root_sid, [idx])
                except Exception:
                    continue
                sid, cur = ss.searchId, ss.observation
                sid, cur = _rollout_complete_turn(sid, cur, my_index, deadline)
                cs = cur.current
                if (cs is None or (cs.result is not None and cs.result >= 0)
                        or cur.select is None or cs.yourIndex == my_index):
                    acc[idx] += _leaf_eval(cs, my_index, threat_bonus)
                    n_eval[idx] += 1
                    continue
                sid, cur = _advance_forced(sid, cur, 1 - my_index, deadline)
                cs = cur.current
                if (cs is None or cur.select is None
                        or cur.select.context != SelectContext.MAIN
                        or cs.yourIndex == my_index):
                    acc[idx] += _leaf_eval(cs, my_index, threat_bonus)
                    n_eval[idx] += 1
                    continue
                op_choice = _generic_pick(cur)
                if op_choice:
                    try:
                        ss2 = search_step(sid, op_choice)
                        sid, cur = ss2.searchId, ss2.observation
                        sid, cur = _advance_forced(sid, cur, my_index, deadline)
                    except Exception:
                        pass
                acc[idx] += _leaf_eval(cur.current, my_index, threat_bonus)
                n_eval[idx] += 1

            try:
                search_end()
            except Exception:
                pass
            began = False

        elapsed = time.monotonic() - t0
        stats["search_time_total"] += elapsed
        stats["search_time_max"] = max(stats["search_time_max"], elapsed)
        n_top = n_eval.get(heur_top, 0)
        if n_top == 0:
            return None
        stats["search_fired"] += 1
        evaluated = [i for i in candidates if n_eval[i] == n_top]
        avg = {i: acc[i] / n_eval[i] for i in evaluated}
        best = max(evaluated, key=lambda i: avg[i])
        if best == heur_top:
            return None
        if avg[best] < avg[heur_top] + OVERRIDE_MARGIN:
            return None
        stats["search_overrides"] += 1
        return best
    except Exception:
        stats["search_errors"] += 1
        return None
    finally:
        if began:
            try:
                search_end()
            except Exception:
                pass


def stats_summary() -> str:
    """Human-readable cumulative instrumentation summary. Not called from
    agent() itself (the local harness's run_match never hands the terminal
    observation to either agent, so there is no reliable in-band hook for
    "game just ended"); test scripts should read the module-level `stats`
    dict directly, e.g. after N games via `mod.stats`."""
    return (
        f"[dragapult_v2] decisions={stats['decisions']} main={stats['main_decisions']} "
        f"search_attempts={stats['search_attempts']} search_fired={stats['search_fired']} "
        f"search_overrides={stats['search_overrides']} search_errors={stats['search_errors']} "
        f"avg_search_ms={(1000 * stats['search_time_total'] / max(1, stats['search_fired'])):.1f} "
        f"max_search_ms={1000 * stats['search_time_max']:.1f}"
    )


# ===========================================================================
# Agent entry point
# ===========================================================================

def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount
    (inclusive), with no duplicate elements.
    """
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    stats["decisions"] += 1
    scores = compute_scores(obs)
    select = obs.select
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    fallback = pick_by_scores(select, scores)

    if select.context == SelectContext.MAIN:
        stats["main_decisions"] += 1
        try:
            override = _search_decide(obs, order, scores)
        except Exception:
            stats["search_errors"] += 1
            override = None
        if override is not None:
            result = [override]
        else:
            result = fallback
    else:
        result = fallback

    return result
