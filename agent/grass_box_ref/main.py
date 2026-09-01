import os
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, all_attack, to_observation_class

"""
Grass Box / Ogerpon / Hydrapple Deck  (EX_05, "Grass energy toolbox")
Local test-harness reference opponent only -- NOT a Kaggle submission candidate.

This is a minimal, reasonably-competent rule-based pilot for a combo/
resource-allocation archetype distinct from everything else in our local
reference set: it spreads Basic Grass Energy across a wide bench (Ogerpon
ex's Teal Dance and Hydrapple ex's Ripening Charge each attach-from-hand
and draw/heal once per turn) and converts that distributed energy board
into scaling damage (Hydrapple's Syrup Storm: +30 per G Energy on ALL own
Pokemon; Ogerpon's Myriad Leaf Shower: +30 per Energy on both Actives).
Not tuned/optimized beyond "spread energy, use the free-attach abilities
every turn, attack with whatever does the most damage". Zero crashes is
the bar, not competitive quality.
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
Basic_Grass_Energy = 1  # x15
Ogerpon_ex = 96  # x4 (Teal Dance: attach G + draw; Myriad Leaf Shower 30+30/energy-on-both-actives)
Applin = 92  # x2
Dipplin = 93  # x2 (Festival Lead; Do the Wave 20x bench -- tech copy, not the main plan here)
Chikorita = 708  # x1 (+917 x1, same role)
Chikorita_B = 917
Bayleef = 709  # x1 (+918 x1, same role)
Bayleef_B = 918
Meganium = 710  # x2 (Wild Growth passive; Solar Beam 140)
Hydrapple_ex = 150  # x2 (Ripening Charge: attach G + heal 30; Syrup Storm 30+30/total-G-energy)
Celebi = 655  # x1 (search)
Hoothoot = 172  # x1
Noctowl = 173  # x1 (Jewel Seeker search-on-evolve)
Tapu_Bulu = 920  # x1 (Wood Hammer 220, self-dmg 30)
Fezandipiti_ex = 140  # x1 (Flip the Script draw3; Cruel Arrow 100 dmg to any target)
Meowth_ex = 1071  # x1 (Last-Ditch Catch: search supporter on bench-play)
Forest_of_Vitality = 1261  # x3 (stadium)
Lillie_Determination = 1227  # x4
Bug_Catching_Set = 1094  # x4
Dawn = 1231  # x2
Boss_Orders = 1182  # x2
Poke_Pad = 1152  # x2
Ultra_Ball = 1121  # x2
Briar = 1201  # x1
Lana_Aid = 1184  # x1
Night_Stretcher = 1097  # x1
Unfair_Stamp = 1080  # x1

CHIKORITA_LINE = (Chikorita, Chikorita_B, Bayleef, Bayleef_B, Meganium)

# Attack IDs
Tumbling_Attack = 114
Do_the_Wave = 115
Myriad_Leaf_Shower = 120
Cruel_Arrow = 183
Syrup_Storm = 195
Triple_Stab = 228
Speed_Wing = 229
Traverse_Time = 945
Solar_Cutter = 946
Razor_Leaf = 1026
Push_Down = 1027
Solar_Beam = 1028
Growl = 1322
Seed_Bomb = 1323
Leaf_Step = 1324
Wood_Hammer = 1326
Tuck_Tail = 1546

IGNORES_WR = {Cruel_Arrow}
BENCH_TARGETABLE = {Cruel_Arrow}


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


def compute_damage(attack_id: int, attacker: Pokemon, my_state, op_pokemon: Pokemon) -> int:
    base = attack_table[attack_id].damage if attack_id in attack_table else 0
    if attack_id == Do_the_Wave:
        bench_n = sum(1 for c in my_state.bench if c is not None)
        return 20 * bench_n
    if attack_id == Myriad_Leaf_Shower:
        op_energy = len(op_pokemon.energies)
        return base + 30 * (len(attacker.energies) + op_energy)
    if attack_id == Syrup_Storm:
        total_g = sum(
            sum(1 for e in c.energyCards if e.id == Basic_Grass_Energy)
            for c in ([my_state.active[0]] if my_state.active and my_state.active[0] is not None else []) + [c for c in my_state.bench if c is not None]
        )
        return base + 30 * total_g
    if attack_id == Tumbling_Attack:
        return base + 10
    if attack_id == Triple_Stab:
        return 15
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

    # ---- Attack planning ----
    if context == SelectContext.MAIN:
        can_attack = any(o.type == OptionType.ATTACK for o in select.option)

        plan.attacker = -1
        plan.target = -1
        plan.attack_id = -1
        plan.lethal = False

        if can_attack and my_state.active and my_state.active[0] is not None and op_cards:
            attacker = my_state.active[0]
            attacker_data = card_table[attacker.id]
            attacker_type = attacker_data.energyType
            best_score = -1
            for attack_id in attacker_data.attacks:
                targets = range(len(op_cards)) if attack_id in BENCH_TARGETABLE else (0,)
                for j in targets:
                    op_pokemon = op_cards[j]
                    op_data = card_table[op_pokemon.id]
                    dmg = compute_damage(attack_id, attacker, my_state, op_pokemon)
                    damage = dmg
                    if attack_id not in IGNORES_WR:
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
                    if j == 0:
                        score += 200
                    if best_score < score:
                        best_score = score
                        plan.attacker = 0
                        plan.target = j
                        plan.attack_id = attack_id
                        plan.lethal = lethal

    def energy_score(pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        pid = pokemon.id
        score = 5000
        if pid == Hydrapple_ex:
            score += 300
        elif pid == Ogerpon_ex:
            score += 280
        elif pid == Meganium:
            score += 220
        elif pid == Tapu_Bulu:
            score += 200
            if energy_count < 4:
                score += 100
        elif pid in (Dipplin, Bayleef, Bayleef_B):
            if energy_count < 1:
                score += 80
        if active:
            score += 20
        if energy_count >= 4:
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
                        if card.id == Hydrapple_ex and energy_count >= 2:
                            score += 50000
                        elif card.id == Ogerpon_ex and energy_count >= 2:
                            score += 45000
                        elif card.id == Meganium:
                            score += 35000
                        elif card.id == Tapu_Bulu:
                            score += 25000
                        else:
                            score += 10000
                        score += energy_count * 100 + hp
                    else:
                        if plan.target >= 0 and o.index == plan.target - (1 if len(op_state.active) > 0 else 0):
                            score += 100000
                        score += pokemon_score(card)
                elif context == SelectContext.EFFECT_TARGET:
                    if o.playerIndex != my_index and plan.target >= 0 and o.index == plan.target:
                        score += 100000
                    score += pokemon_score(card) if isinstance(card, Pokemon) else 10
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 30 if card.id in (Applin, Chikorita, Chikorita_B, Hoothoot) else (20 if card.id in (Ogerpon_ex,) else 10)
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
                    if card.id == Ogerpon_ex:
                        score = 900 if field_counts[Ogerpon_ex] < 2 else 350
                    elif card.id == Applin or card.id == Dipplin:
                        score = 750 if field_counts[Applin] + field_counts[Dipplin] < 2 else 300
                    elif card.id in (Chikorita, Chikorita_B):
                        score = 800 if field_counts[Meganium] + field_counts[Bayleef] + field_counts[Bayleef_B] < 1 else 300
                    elif card.id in (Bayleef, Bayleef_B):
                        score = 850
                    elif card.id == Meganium:
                        score = 900
                    elif card.id == Hydrapple_ex:
                        score = 850
                    elif card.id == Basic_Grass_Energy:
                        score = 500
                    elif card.id in (Boss_Orders, Lillie_Determination, Dawn):
                        score = 350
                    else:
                        score = 200 - hand_counts[card.id] * 50
                elif context == SelectContext.DISCARD:
                    if card.id == Basic_Grass_Energy:
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
                if card.id == Ogerpon_ex and field_counts[Ogerpon_ex] >= 2:
                    score = 3000
            elif data.cardType == CardType.SUPPORTER:
                if state.supporterPlayed:
                    score = -1
                elif card.id == Lillie_Determination:
                    score = 15000
                elif card.id == Dawn and (field_counts[Meganium] < 1 or field_counts[Hydrapple_ex] < 1):
                    score = 14500
                elif card.id == Boss_Orders and len(op_state.bench) > 0:
                    score = 14000
                elif card.id == Briar and len(op_state.prize) == 2:
                    score = 13500
                elif card.id == Lana_Aid:
                    score = 6000
                else:
                    score = 9000
            elif data.cardType == CardType.STADIUM:
                score = 12000 if stadium_id != Forest_of_Vitality else -1
            elif data.cardType == CardType.TOOL:
                score = -1
            else:  # ITEM
                if card.id == Bug_Catching_Set:
                    score = 12000
                elif card.id == Poke_Pad:
                    score = 11500
                elif card.id == Ultra_Ball:
                    score = 11000 if len(my_state.hand) >= 3 else -1
                elif card.id == Night_Stretcher:
                    score = 5000
                elif card.id == Unfair_Stamp:
                    score = 4000
                else:
                    score = 3000
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 30000 + len(pokemon.energies) * 10
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Ogerpon_ex and not ability_used and hand_counts[Basic_Grass_Energy] > 0:
                score = 41000
            elif card.id == Hydrapple_ex and not ability_used and hand_counts[Basic_Grass_Energy] > 0:
                score = 40500
            elif card.id == Meowth_ex and not ability_used:
                score = 40000
            elif card.id == Fezandipiti_ex and not ability_used:
                score = 15000
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
            if card.id in (Ogerpon_ex, Hydrapple_ex, Meowth_ex, Fezandipiti_ex):
                ability_used = True

    return output
