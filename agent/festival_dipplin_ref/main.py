import os
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, all_attack, to_observation_class

"""
Festival Lead / Dipplin / Thwackey Deck  (EX_12, "Festival Lead single-Prize swarm")
Local test-harness reference opponent only -- NOT a Kaggle submission candidate.

This is a minimal, reasonably-competent rule-based pilot for a single-Prize
swarm/rush archetype: fill the bench with cheap Basics, get Festival Grounds
in play, evolve Applin -> Dipplin, and repeatedly attack with "Do the Wave"
(20 dmg per own Benched Pokemon, usable twice per turn under Festival Lead
while Festival Grounds is in play). Thwackey's Boom Boom Groove searches a
card whenever a Festival-Lead Pokemon (Dipplin/Goldeen/Seaking) is active.
Not tuned/optimized beyond "keep the bench full, keep the stadium down,
attack with whatever does the most damage". Zero crashes is the bar.
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
attack_table = {a.attackId: a for a in all_attack()}

# Decklist (key card IDs)
Basic_Grass_Energy = 1  # x5
Grookey = 89  # x4
Thwackey = 90  # x4  (Boom Boom Groove ability; Beat 50 dmg)
Applin = 149  # x4
Dipplin = 93  # x3  (Festival Lead ability; Do the Wave 20x bench)
Rellor = 73  # x1
Rabsca = 74  # x1  (Spherical Shield -- passive bench-damage wall)
Goldeen = 100  # x2
Seaking = 240  # x1  (Festival Lead ability; Rapid Draw)
Budew = 235  # x1
Festival_Grounds = 1245  # x4 (stadium)
Lillie_Determination = 1227  # x4
Poke_Pad = 1152  # x4
Bug_Catching_Set = 1094  # x4
Buddy_Buddy_Poffin = 1086  # x4
Boss_Orders = 1182  # x3
Night_Stretcher = 1097  # x2
Dawn = 1231  # x1
Black_Belt_Training = 1211  # x1 (+40 dmg vs ex this turn)
Kieran = 1191  # x1
Lana_Aid = 1184  # x1
Brave_Bangle = 1175  # x1 (tool, +30 dmg vs ex)
Air_Balloon = 1174  # x1 (tool, retreat -2)
Maximum_Belt = 1158  # x1 (tool ACE SPEC, +50 dmg vs ex)
Sacred_Ash = 1129  # x1
Switch = 1123  # x1

FESTIVAL_LEAD_MONS = (Dipplin, Goldeen, Seaking)
SMALL_BASICS = (Grookey, Applin, Goldeen, Rellor, Budew)  # cheap bench-fillers for Do the Wave scaling

# Attack IDs
Smash_Kick = 109
Branch_Poke = 110
Beat = 111
Do_the_Wave = 115
Whirlpool = 126
Spray_Fluid = 194
Itchy_Pollen = 323
Rapid_Draw = 330
Slight_Intrusion = 87
Psychic = 88


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
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage1:
        score += 130
    score += pokemon.hp
    return score


def compute_damage(attack_id: int, attacker: Pokemon, my_state, op_active_energy: int) -> int:
    base = attack_table[attack_id].damage if attack_id in attack_table else 0
    if attack_id == Do_the_Wave:
        bench_n = sum(1 for c in my_state.bench if c is not None)
        return 20 * bench_n
    if attack_id == Psychic:
        return 10 + 30 * op_active_energy
    return base


class AttackPlan:
    attacker = -1
    target = -1
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

    my_cards = [c for c in my_state.active if c is not None] + [c for c in my_state.bench if c is not None]
    op_cards = [c for c in op_state.active if c is not None] + [c for c in op_state.bench if c is not None]

    for card in my_cards:
        field_counts[card.id] += 1
    for card in my_state.hand:
        hand_counts[card.id] += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id
    festival_in_play = stadium_id == Festival_Grounds

    op_has_ex_active = bool(op_state.active) and op_state.active[0] is not None and (card_table[op_state.active[0].id].ex or card_table[op_state.active[0].id].megaEx)
    my_active_is_festival_lead = bool(my_state.active) and my_state.active[0] is not None and my_state.active[0].id in FESTIVAL_LEAD_MONS
    op_active_energy = len(op_state.active[0].energies) if op_state.active and op_state.active[0] is not None else 0

    # ---- Attack planning ----
    if context == SelectContext.MAIN:
        can_attack = any(o.type == OptionType.ATTACK for o in select.option)

        plan.attacker = -1
        plan.target = -1
        plan.attack_id = -1
        plan.lethal = False

        if can_attack and my_state.active and my_state.active[0] is not None:
            attacker = my_state.active[0]
            attacker_data = card_table[attacker.id]
            attacker_type = attacker_data.energyType
            best_score = -1
            for attack_id in attacker_data.attacks:
                dmg = compute_damage(attack_id, attacker, my_state, op_active_energy)
                op_pokemon = op_cards[0] if op_cards else None
                if op_pokemon is None:
                    continue
                op_data = card_table[op_pokemon.id]
                damage = dmg
                if op_data.weakness == attacker_type:
                    damage *= 2
                elif op_data.resistance == attacker_type:
                    damage = max(0, damage - 30)
                score = pokemon_score(op_pokemon)
                lethal = op_pokemon.hp <= damage and damage > 0
                if lethal:
                    score += 5000
                elif damage > 0:
                    score *= damage / max(1, op_pokemon.hp)
                else:
                    score *= 0.05
                if best_score < score:
                    best_score = score
                    plan.attacker = 0
                    plan.target = 0
                    plan.attack_id = attack_id
                    plan.lethal = lethal

    def energy_score(pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        pid = pokemon.id
        score = 5000
        if pid == Dipplin:
            score += 300
            if energy_count < 1:
                score += 300
        elif pid == Thwackey:
            score += 200
            if energy_count < 2:
                score += 150
        elif pid == Rabsca:
            score += 100
            if energy_count < 1:
                score += 100
        elif pid == Applin or pid == Grookey:
            if energy_count < 1:
                score += 80
            else:
                score -= 50
        if active:
            score += 20
        if energy_count >= 3:
            score -= 1000
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
                        if card.id == Dipplin and energy_count >= 1:
                            score += 50000
                        elif card.id in FESTIVAL_LEAD_MONS:
                            score += 30000
                        elif card.id == Thwackey:
                            score += 20000
                        else:
                            score += 10000
                        score += energy_count * 100 + hp
                    else:
                        score += pokemon_score(card)
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 30 if card.id in SMALL_BASICS else (25 if card.id == Rabsca else 10)
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
                    if card.id == Applin:
                        score = 900 if field_counts[Dipplin] + field_counts[Applin] < 2 else 300
                    elif card.id == Grookey:
                        score = 800 if field_counts[Thwackey] + field_counts[Grookey] < 2 else 250
                    elif card.id == Rabsca or card.id == Rellor:
                        score = 700 if field_counts[Rabsca] < 1 else 150
                    elif card.id == Goldeen or card.id == Seaking:
                        score = 500
                    elif card.id == Budew:
                        score = 400
                    elif card.id == Basic_Grass_Energy:
                        score = 450
                    elif card.id in (Boss_Orders, Lillie_Determination, Dawn):
                        score = 350
                    else:
                        score = 200 - hand_counts[card.id] * 50
                elif context == SelectContext.DISCARD:
                    if card.id in (Budew, Rellor):
                        score = 50
                    elif card.id == Basic_Grass_Energy:
                        score = 40
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
                if card.id == Applin and field_counts[Applin] + field_counts[Dipplin] >= 3:
                    score = 3000
                elif card.id == Grookey and field_counts[Grookey] + field_counts[Thwackey] >= 3:
                    score = 3000
            elif data.cardType == CardType.SUPPORTER:
                if state.supporterPlayed:
                    score = -1
                elif card.id == Lillie_Determination:
                    score = 15000
                elif card.id == Dawn and field_counts[Dipplin] < 1:
                    score = 14500
                elif card.id == Boss_Orders and len(op_state.bench) > 0:
                    score = 14000
                elif card.id == Black_Belt_Training and op_has_ex_active:
                    score = 13000
                elif card.id == Lana_Aid:
                    score = 6000
                else:
                    score = 9000
            elif data.cardType == CardType.STADIUM:
                score = 12000 if stadium_id != Festival_Grounds else -1
            elif data.cardType == CardType.TOOL:
                score = -1  # handled via ATTACH
            else:  # ITEM
                if card.id == Buddy_Buddy_Poffin:
                    score = 12000 if sum(field_counts[c] for c in SMALL_BASICS) < 4 else 500
                elif card.id == Poke_Pad:
                    score = 11500
                elif card.id == Bug_Catching_Set:
                    score = 11000
                elif card.id == Night_Stretcher:
                    score = 5000
                elif card.id == Sacred_Ash:
                    score = 4000
                elif card.id == Switch:
                    score = 3000 if my_state.active and my_state.active[0] is not None and my_state.active[0].hp <= 20 else -1
                else:
                    score = 3000
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id in (Brave_Bangle, Maximum_Belt):
                score = 9000 if pokemon.id == Dipplin and op_has_ex_active else -1
            elif card.id == Air_Balloon:
                score = 8500 if pokemon.id in (Dipplin, Thwackey) else -1
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 30000 + len(pokemon.energies) * 10
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Thwackey and not ability_used and my_active_is_festival_lead:
                score = 40000
            else:
                score = -1
        elif o.type == OptionType.RETREAT:
            if my_state.active and my_state.active[0] is not None and my_state.active[0].hp <= 20:
                score = 5000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            if plan.attack_id == o.attackId:
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
            if card.id == Thwackey:
                ability_used = True

    return output
