import os
from collections import defaultdict

from cg.api import (AreaType, CardType, Observation, SelectContext, OptionType,
                     Card, Pokemon, all_card_data, all_attack, to_observation_class)

"""
Abra / Kadabra / Alakazam Deck
Win condition: evolve into Alakazam and use Powerful Hand (damage = hand size x 20,
since each of the 2 damage counters per card = 20 damage) to one-shot the opponent's
Active. Kadabra's Super Psy Bolt (30 dmg, 1 Energy) finishes already-weakened targets
cheaply. Dudunsparce's Run Away Draw and the two evolution Abilities (Kadabra draws 2,
Alakazam draws 3, both trigger when playing the card from hand to evolve) are the
engine that builds a big enough hand.
"""

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = [int(csv[i]) for i in range(60)]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
attack_table = {a.attackId: a for a in all_attack()}

# Decklist
Basic_Psychic_Energy = 5  # x2
Telepath_Psychic_Energy = 19  # x4 (special energy; attaching to a {P} Pokemon fetches up to 2 Basic {P} Pokemon to bench)
Dunsparce60 = 65  # x2 (Gnaw 10/1en, Dig 30/2en+coin-guard)
Dudunsparce = 66  # x4 (Run Away Draw: draw 3, then shuffle self back into deck)
Dunsparce305 = 305  # x1 (Trading Places: free switch, Ram 20/2en)
Abra = 741  # x4 (Teleportation Attack: 10 dmg + free switch, 1 energy)
Kadabra = 742  # x4 (Psychic Draw on evolve: draw 2; Super Psy Bolt 30/1en)
Alakazam = 743  # x4 (Psychic Draw on evolve: draw 3; Powerful Hand: hand_size*20 dmg)
Rare_Candy = 1079  # x3
Enhanced_Hammer = 1081  # x3 (discard opponent's Special Energy)
Buddy_Buddy_Poffin = 1086  # x4 (bench up to 2 Basic <=70HP)
Night_Stretcher = 1097  # x3 (Pokemon or Basic Energy: discard -> hand)
Sacred_Ash = 1129  # x1 (up to 5 Pokemon: discard -> deck)
Poke_Pad = 1152  # x4 (search 1 non-Rule-Box Pokemon -> hand)
Boss_Orders = 1182  # x3 (gust)
Lanas_Aid = 1184  # x1 (up to 3 Pokemon/Basic Energy: discard -> hand)
Xerosics_Machinations = 1197  # x3 (opponent discards down to 3 in hand)
Hilda = 1225  # x4 (search 1 Evolution Pokemon + 1 Energy -> hand)
Lillies_Determination = 1227  # x1 (shuffle hand into deck, draw 6, or 8 at exactly 6 prizes)
Dawn = 1231  # x4 (search 1 Basic + 1 Stage1 + 1 Stage2 -> hand)
Neutralization_Zone = 1247  # x1 (Stadium: opponent's ex/V attacks deal 0 to non-Rule-Box Pokemon, i.e. shields our whole board)

DUNSPARCE_NAMES = {Dunsparce60, Dunsparce305}
ABRA_LINE = {Abra, Kadabra, Alakazam}

TELEPORT_ATTACK = 1070
SUPER_PSY_BOLT = 1071
POWERFUL_HAND = 1072

UNNECESSARY = -10000000

pre_turn_log = []
current_turn_log = []


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
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


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn_log, current_turn_log

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn == 0:
        pre_turn_log.clear()
        current_turn_log.clear()
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            from cg.api import LogType
            if log.type == LogType.TURN_END:
                pre_turn_log = current_turn_log
                current_turn_log = []

    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None

    field_counts = defaultdict(int)
    if my_active:
        field_counts[my_active.id] += 1
    for p in my_state.bench:
        field_counts[p.id] += 1

    hand_counts = defaultdict(int)
    for c in my_state.hand:
        hand_counts[c.id] += 1

    discard_counts = defaultdict(int)
    for c in my_state.discard:
        discard_counts[c.id] += 1

    my_evo_line_count = field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam]
    op_has_special_energy = False
    op_pokemon = ([op_active] if op_active else []) + [p for p in op_state.bench]
    for p in op_pokemon:
        for e in p.energyCards:
            edata = card_table.get(e.id)
            if edata is not None and edata.cardType == CardType.SPECIAL_ENERGY:
                op_has_special_energy = True

    no_draw = my_state.deckCount <= 3  # avoid decking ourselves out

    hand_size_now = len(my_state.hand)

    def powerful_hand_lethal():
        if op_active is None:
            return False
        return hand_size_now * 20 >= op_active.hp

    def psy_bolt_lethal():
        if op_active is None:
            return False
        return 30 >= op_active.hp

    def is_evolution_pokemon_in_hand():
        return hand_counts[Kadabra] > 0 or hand_counts[Alakazam] > 0 or hand_counts[Dudunsparce] > 0

    stadium_id = state.stadium[0].id if state.stadium else 0

    def hand_score(id: int, ignore_count: bool):
        score = 0
        if id == Abra:
            if field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam] == 0:
                score = 55000
            else:
                score = 8000
        elif id == Dunsparce60 or id == Dunsparce305:
            if field_counts[Dunsparce60] + field_counts[Dunsparce305] + field_counts[Dudunsparce] >= 2:
                score = 500
            else:
                score = 20000
        elif id == Kadabra:
            if field_counts[Abra] >= 1:
                score = 70000
            else:
                score = 100
        elif id == Alakazam:
            if field_counts[Kadabra] >= 1:
                score = 75000
            elif field_counts[Abra] >= 1 and hand_counts[Rare_Candy] > 0:
                score = 71000
            else:
                score = 100
        elif id == Dudunsparce:
            if field_counts[Dunsparce60] + field_counts[Dunsparce305] >= 1:
                score = 60000
            else:
                score = 100
        elif id == Rare_Candy:
            if field_counts[Abra] >= 1 and hand_counts[Alakazam] > 0:
                score = 76000
            else:
                score = -1
        elif id == Telepath_Psychic_Energy:
            score = 24000
        elif id == Basic_Psychic_Energy:
            score = 15000
        elif id == Buddy_Buddy_Poffin:
            score = 34000
        elif id == Poke_Pad:
            score = 32000
        elif id == Dawn:
            if not ignore_count or state.supporterPlayed is False:
                score = 48000
        elif id == Hilda:
            if not ignore_count or state.supporterPlayed is False:
                score = 44000
        elif id == Lillies_Determination:
            if not ignore_count or state.supporterPlayed is False:
                score = 20000 if hand_size_now >= 3 else 42000
        elif id == Xerosics_Machinations:
            if not ignore_count or state.supporterPlayed is False:
                score = 15000 if len(op_state.hand or []) >= 4 or op_state.handCount >= 4 else 3000
        elif id == Boss_Orders:
            if len(op_state.bench) > 0:
                score = 28000
        elif id == Enhanced_Hammer:
            score = 18000 if op_has_special_energy else 2000
        elif id == Night_Stretcher:
            score = 22000 if (discard_counts and my_evo_line_count < 3) else 3000
        elif id == Sacred_Ash:
            dead_pokemon_in_discard = sum(1 for c in my_state.discard if card_table.get(c.id) and card_table[c.id].cardType == CardType.POKEMON)
            score = 25000 if dead_pokemon_in_discard >= 3 else 500
        elif id == Lanas_Aid:
            score = 21000
        elif id == Neutralization_Zone:
            score = 33000 if stadium_id != Neutralization_Zone else -1

        if no_draw and id in (Dawn, Hilda, Lillies_Determination, Buddy_Buddy_Poffin, Poke_Pad, Night_Stretcher, Sacred_Ash, Lanas_Aid):
            score = -1

        if not ignore_count and hand_counts[id] > 0:
            score -= 100000
        return score

    def attach_score(pokemon: Pokemon, active: bool) -> int:
        if pokemon.id not in ABRA_LINE and pokemon.id not in (Dunsparce60, Dunsparce305, Dudunsparce):
            return -1
        energy_count = len(pokemon.energies)
        if energy_count >= 1:
            return -1
        score = 20000
        if pokemon.id == Alakazam:
            score += 500
        elif pokemon.id == Kadabra:
            score += 300
        if active:
            score += 200
        return score

    global_support_used = 0

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = -1
            elif context == SelectContext.MULLIGAN:
                score = 1
            elif context == SelectContext.COIN_HEAD:
                score = 1
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
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
                    if o.playerIndex == my_index:
                        if card.id == Alakazam and energy_count >= 1:
                            score += 60000
                        elif card.id == Kadabra and energy_count >= 1:
                            score += 40000
                        elif card.id == Abra:
                            score += 20000
                        elif card.id in DUNSPARCE_NAMES or card.id == Dudunsparce:
                            score += 15000
                        else:
                            score += 5000
                        score += energy_count * 1000
                        score += hp
                    else:
                        # choosing opponent's Pokemon (e.g. Boss's Orders target)
                        score += (200 - hp) + energy_count * 500 + len(card.tools) * 300
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if card.id == Abra:
                        score = 10000
                    elif card.id in DUNSPARCE_NAMES:
                        score = 8000
                    else:
                        score = -1
                elif context in (SelectContext.TO_BENCH, SelectContext.TO_HAND):
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    score = -hand_score(card.id, False)
                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if hp > 0:
                        score = 100000 - 10 * hp
                        if 0 < hp <= 30:
                            score += 20000
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.EVOLVES_FROM:
                    score = 1000 + energy_count * 500 + hp
                elif context in (SelectContext.EVOLVES_TO,):
                    if card.id == Alakazam:
                        score = 50000
                    elif card.id == Kadabra:
                        score = 40000
                    elif card.id == Dudunsparce:
                        score = 30000
                    else:
                        score = 100
                elif context == SelectContext.LOOK:
                    score = hand_score(card.id, True)
                else:
                    score = hand_score(card.id, True)
        elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            if o.playerIndex != my_index:
                score = 10
            else:
                score = 5
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table.get(card.id)
            is_supporter = data is not None and data.cardType == CardType.SUPPORTER
            if is_supporter and state.supporterPlayed:
                score = -1
            elif no_draw and card.id in (Dawn, Hilda, Lillies_Determination, Buddy_Buddy_Poffin, Poke_Pad, Night_Stretcher, Sacred_Ash, Lanas_Aid):
                score = -1
            else:
                score = hand_score(card.id, True)
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Alakazam:
                score = 90000
            elif card.id == Kadabra:
                score = 85000
            elif card.id == Dudunsparce:
                score = 80000
            else:
                score = 60000
            score += len(pokemon.energies) * 100
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if no_draw:
                score = -1
            elif card.id == Dudunsparce:
                score = 41000
            else:
                score = 40000
        elif o.type == OptionType.RETREAT:
            can_win_now = False
            if my_active is not None:
                if my_active.id == Alakazam:
                    can_win_now = powerful_hand_lethal()
                elif my_active.id == Kadabra:
                    can_win_now = psy_bolt_lethal()
            in_danger = (
                my_active is not None and my_active.maxHp > 0
                and my_active.hp <= my_active.maxHp * 0.4
            )
            if can_win_now:
                score = -1
            elif in_danger and len(my_state.bench) > 0:
                score = 12000
            elif my_active is not None and my_active.id != Alakazam and my_active.id != Kadabra:
                bench_has_better = any(
                    p.id in (Alakazam, Kadabra) and len(p.energies) >= 1 for p in my_state.bench
                )
                score = 9000 if bench_has_better else -1
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            if o.attackId == POWERFUL_HAND:
                score = 200000 if powerful_hand_lethal() else 400
            elif o.attackId == SUPER_PSY_BOLT:
                score = 150000 if psy_bolt_lethal() else 300
            elif o.attackId == TELEPORT_ATTACK:
                score = 200
            else:
                score = o.attackId

        scores.append(score)

    output = []
    if len(scores) >= 1:
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if (sorted_scores[i][1] >= 0
                or select.minCount > i
                or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])

    return output
