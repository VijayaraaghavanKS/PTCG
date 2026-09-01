import os
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Mega Kangaskhan ex / Crustle / Petrel Deck  (EX_06, "Mega Kangaskhan disruption")
Local test-harness reference opponent only -- NOT a Kaggle submission candidate.

This is a minimal, reasonably-competent rule-based pilot for the Crustle-wall
control archetype, built to give the win-rate signal against a true-immunity
wall (Crustle's "Mysterious Rock Inn": blocks all damage from opponent `ex`
attacks) some meaning in local A/B testing. It is not tuned/optimized beyond
"attach energy, evolve to Crustle, attack with whatever does the most
damage/has the best matchup, wall with Crustle when it makes sense". Zero
crashes is the bar, not competitive quality.
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

# Decklist
Mega_Kangaskhan_ex = 756  # x4
Dwebble = 344  # x3
Crustle = 345  # x3
Ogerpon_ex = 117  # x1
Psyduck = 858  # x1
Buddy_Buddy_Poffin = 1086  # x2
Crushing_Hammer = 1120  # x4
Ultra_Ball = 1121  # x1
Pokegear = 1122  # x3
Switch = 1123  # x1
Jumbo_Ice_Cream = 1147  # x4
Heros_Cape = 1159  # x1 (tool)
Boss_Orders = 1182  # x4
Eri = 1186  # x1
Bianca_Devotion = 1190  # x1
Xerosic_Machinations = 1197  # x1
Petrel = 1219  # x4
Hilda = 1225  # x2
Lillie_Determination = 1227  # x4
TR_Factory = 1257  # x1 (stadium)
Basic_Fighting_Energy = 6  # x2
Mist_Energy = 11  # x4
Spiky_Energy = 14  # x4
Grow_Grass_Energy = 18  # x4

# Attack IDs
Rapid_Fire_Combo = 1092  # Kangaskhan, 3 colorless, 200+ dmg
Ascension = 478  # Dwebble, 1 colorless, evolves to Crustle via deck search
Superb_Scissors = 479  # Crustle, 1 grass + 2 colorless, 120 dmg
Demolish = 148  # Ogerpon, 1 fighting + 2 colorless, 140 dmg, ignores W/R/effects
Ram = 1237  # Psyduck, 2 colorless, 20 dmg

SMALL_BASICS = (Dwebble, Psyduck)  # <=70 HP, valid Buddy-Buddy Poffin targets


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


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    return 3 if data.megaEx else 2 if data.ex else 1


def pokemon_score(pokemon: Pokemon) -> int:
    """Rough tactical value of a target Pokemon -- prize value, investment, HP."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage1:
        score += 130
    score += pokemon.hp
    return score


class AttackPlan:
    attacker = -1  # index into my_cards (0 = active)
    target = -1  # index into op_cards (0 = active)
    attack_id = -1
    lethal = False


plan = AttackPlan()
pre_turn = -1
ability_used = False


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global plan, pre_turn, ability_used
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        ability_used = False

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    my_cards = [c for c in my_state.active if c is not None] + [c for c in my_state.bench if c is not None]
    # Pad so index 0 is always "active" even if somehow empty (setup phase)
    op_cards = [c for c in op_state.active if c is not None] + [c for c in op_state.bench if c is not None]

    for card in my_cards:
        field_counts[card.id] += 1
    for card in my_state.hand:
        hand_counts[card.id] += 1
    for card in my_state.discard:
        discard_counts[card.id] += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    has_crustle_in_play = field_counts[Crustle] >= 1
    has_dwebble_in_play = field_counts[Dwebble] >= 1
    op_has_ex_attacker = any(card_table[c.id].ex or card_table[c.id].megaEx for c in op_cards)

    # ---- Attack planning (only matters when it's our MAIN decision) ----
    if context == SelectContext.MAIN:
        can_attack = False
        can_switch = False
        for o in select.option:
            if o.type == OptionType.ATTACK:
                can_attack = True
            elif o.type == OptionType.RETREAT:
                can_switch = True

        plan.attacker = -1
        plan.target = -1
        plan.attack_id = -1
        plan.lethal = False

        if can_attack and len(my_state.active) > 0 and my_state.active[0] is not None:
            attacker = my_state.active[0]
            attacker_type = card_table[attacker.id].energyType
            base_damage = {
                Rapid_Fire_Combo: 200,
                Superb_Scissors: 120,
                Demolish: 140,
                Ram: 20,
                Ascension: 0,
            }
            best_score = -1
            for j, op_pokemon in enumerate(op_cards):
                op_data = card_table[op_pokemon.id]
                for attack_id, dmg in base_damage.items():
                    if attack_id == Ascension:
                        continue  # handled separately below, not a damage attack
                    damage = dmg
                    if op_data.weakness == attacker_type:
                        damage *= 2
                    elif op_data.resistance == attacker_type:
                        damage = max(0, damage - 30)
                    # Crustle's own wall only protects itself, not other opponents
                    score = pokemon_score(op_pokemon)
                    lethal = op_pokemon.hp <= damage and damage > 0
                    if lethal:
                        score += 5000
                    else:
                        score *= damage / max(1, op_pokemon.hp)
                    if j == 0:
                        score += 300  # prefer hitting active (only legal target most of the time)
                    if best_score < score:
                        best_score = score
                        plan.attacker = 0
                        plan.target = j
                        plan.attack_id = attack_id
                        plan.lethal = lethal

    def energy_score(pokemon: Pokemon, active: bool) -> int:
        """How much we want to attach an energy card to this Pokemon."""
        energy_count = len(pokemon.energies)
        pid = pokemon.id
        score = 5000
        if pid == Mega_Kangaskhan_ex:
            score += 300
            if energy_count < 3:
                score += 200
        elif pid == Crustle:
            score += 200
            if energy_count < 3:
                score += 150
        elif pid == Ogerpon_ex:
            score += 100
            if energy_count < 3:
                score += 100
        elif pid == Dwebble:
            if energy_count < 1:
                score += 120  # enough to Ascension into Crustle
            else:
                score -= 50
        elif pid == Psyduck:
            score -= 100
        if active:
            score += 20
        if energy_count >= 4:
            score -= 1000  # already over-invested
        return score

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.NO:
            score = 0
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                hp = card.hp if isinstance(card, Pokemon) else 0
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
                    if o.playerIndex == my_index:
                        if card.id == Crustle and op_has_ex_attacker:
                            score += 60000  # wall vs an ex-attacking opponent
                        elif card.id == Mega_Kangaskhan_ex and energy_count >= 3:
                            score += 50000
                        elif card.id == Ogerpon_ex and energy_count >= 3:
                            score += 40000
                        elif card.id == Crustle:
                            score += 25000
                        else:
                            score += 10000
                        score += energy_count * 100 + hp
                    else:
                        if plan.target >= 0 and o.index == plan.target - (1 if len(op_state.active) > 0 else 0):
                            score += 100000
                        score += pokemon_score(card)
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if card.id in SMALL_BASICS:
                        score = 20
                    elif card.id == Mega_Kangaskhan_ex:
                        score = 10
                    else:
                        score = 5
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
                    if card.id == Crustle:
                        score = 900 if not has_crustle_in_play else 100
                    elif card.id == Mega_Kangaskhan_ex:
                        score = 800 if field_counts[Mega_Kangaskhan_ex] < 2 else 200
                    elif card.id == Dwebble:
                        score = 700 if not has_crustle_in_play and not has_dwebble_in_play else 150
                    elif card.id == Ogerpon_ex:
                        score = 500
                    elif card.id in (Grow_Grass_Energy, Basic_Fighting_Energy, Mist_Energy, Spiky_Energy):
                        score = 400
                    elif card.id == Boss_Orders or card.id == Petrel or card.id == Lillie_Determination:
                        score = 300
                    else:
                        score = 200 - hand_counts[card.id] * 50
                elif context == SelectContext.DISCARD:
                    # Discard least valuable cards first (Ultra Ball cost etc.)
                    if card.id in (Basic_Fighting_Energy, Mist_Energy, Spiky_Energy, Grow_Grass_Energy):
                        score = 50
                    elif card.id in (Crushing_Hammer, Jumbo_Ice_Cream):
                        score = 40
                    elif card.id == Psyduck or card.id == Ogerpon_ex:
                        score = 30
                    else:
                        score = 10
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card)
                else:
                    score = pokemon_score(card) if isinstance(card, Pokemon) else 10
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            score = 10
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id in (Mega_Kangaskhan_ex,) and field_counts[card.id] >= 2:
                    score = 3000
                elif card.id == Crustle and (has_crustle_in_play or field_counts[Dwebble] == 0):
                    score = -1 if has_crustle_in_play else score
            elif data.cardType == CardType.SUPPORTER:
                if state.supporterPlayed:
                    score = -1
                elif card.id == Lillie_Determination:
                    score = 15000
                elif card.id == Petrel:
                    score = 14000
                elif card.id == Hilda:
                    score = 13500 if not has_crustle_in_play else 8000
                elif card.id == Bianca_Devotion:
                    lowest_hp_ok = any(
                        c is not None and 0 < c.hp <= 30
                        for c in (my_state.active + my_state.bench)
                    )
                    score = 16000 if lowest_hp_ok else -1
                elif card.id == Eri:
                    score = 6000 if op_state.handCount >= 3 else -1
                elif card.id == Xerosic_Machinations:
                    score = 5000 if op_state.handCount >= 4 else -1
                else:
                    score = 9000
            elif data.cardType == CardType.STADIUM:
                score = 2000 if stadium_id != TR_Factory else -1
            elif data.cardType == CardType.TOOL:
                score = -1  # handled via ATTACH
            else:  # ITEM
                if card.id == Buddy_Buddy_Poffin:
                    score = 12000 if field_counts[Dwebble] + field_counts[Psyduck] < 2 else 500
                elif card.id == Ultra_Ball:
                    score = 11000 if len(my_state.hand) >= 3 else -1
                elif card.id == Pokegear:
                    score = 10500
                elif card.id == Crushing_Hammer:
                    score = 4000 if len(op_state.active) > 0 and op_state.active[0] is not None and len(op_state.active[0].energies) > 0 else -1
                elif card.id == Switch:
                    score = 8000 if (op_has_ex_attacker and my_state.active and my_state.active[0] is not None and my_state.active[0].id != Crustle and field_counts[Crustle] >= 1) else -1
                elif card.id == Jumbo_Ice_Cream:
                    healable = (
                        my_state.active and my_state.active[0] is not None
                        and len(my_state.active[0].energies) >= 3
                        and my_state.active[0].hp < card_table[my_state.active[0].id].hp
                    )
                    score = 7000 if healable else -1
                else:
                    score = 3000
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Heros_Cape:
                score = 9000 if pokemon.id in (Mega_Kangaskhan_ex, Crustle, Ogerpon_ex) else -1
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 30000 + len(pokemon.energies) * 10
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Mega_Kangaskhan_ex and not ability_used:
                score = 40000
            else:
                score = -1
        elif o.type == OptionType.RETREAT:
            if op_has_ex_attacker and my_state.active and my_state.active[0] is not None and my_state.active[0].id != Crustle and field_counts[Crustle] >= 1:
                score = 15000
            elif my_state.active and my_state.active[0] is not None and my_state.active[0].hp <= 20:
                score = 5000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            if o.attackId == Ascension:
                # Only worth using the attack-slot on Ascension if Crustle isn't
                # already established -- otherwise attack for damage instead.
                score = 25000 if not has_crustle_in_play else -1
            elif plan.attack_id == o.attackId:
                score = 20000 + (5000 if plan.lethal else 0)
            else:
                score = 1000
        elif o.type == OptionType.END:
            score = 0
        elif o.type == OptionType.DISCARD:
            score = -1
        else:
            score = 0

        scores.append(score)

    output = []
    if len(scores) >= 1:
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if (sorted_scores[i][1] >= 0
                or select.minCount > i
                or context not in (SelectContext.TO_BENCH, SelectContext.SETUP_BENCH_POKEMON, SelectContext.DISCARD)):
                output.append(sorted_scores[i][0])

    if context == SelectContext.MAIN and output:
        o = select.option[output[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Mega_Kangaskhan_ex:
                ability_used = True

    return output
