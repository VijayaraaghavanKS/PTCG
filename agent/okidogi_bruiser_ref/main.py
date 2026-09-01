import os
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, all_attack, to_observation_class

"""
Okidogi / Barbaracle / Battle Cage Deck  (EX_15, "Okidogi Battle Cage bruiser")
Local test-harness reference opponent only -- NOT a Kaggle submission candidate.

This is a minimal, reasonably-competent rule-based pilot for a midrange
single-Prize bruiser archetype: a sturdy non-ex Basic (Okidogi, 130 HP,
Good Punch for 70, +100 if it has a Darkness-providing Energy attached) is
the primary attacker, backed by the Solrock/Lunatone draw-and-hit-hard
combo and a Barbaracle/Binacle Fighting-Energy-acceleration package, all
under Battle Cage (protects the bench from damage-counter effects). Not
tuned/optimized beyond "keep attaching Fighting Energy, attack with
whatever does the most damage, use free-attach/draw abilities when
available". Zero crashes is the bar.
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
Basic_Fighting_Energy = 6  # x9
Prism_Energy = 16  # x4 (provides {C})
Legacy_Energy = 12  # x1 (ACE SPEC, provides any 1 type -- can feed Okidogi's Adrena-Power)
Okidogi = 116  # x3 (Adrena-Power ability; Good Punch 70(+100 w/ {D}))
Solrock = 676  # x3 (Cosmic Beam 70, ignores W/R, needs Lunatone on bench)
Lunatone = 675  # x2 (Lunar Cycle ability; Power Gem 50)
Binacle = 1051  # x2 (Double Draw / Scratch)
Barbaracle = 1052  # x2 (Stone Arms ability -- free F-energy attach; Hammer In 80)
Cornerstone_Ogerpon_ex = 117  # x1 (Cornerstone Stance passive wall; Demolish 140 ignores W/R + effects)
Munkidori = 112  # x1 (Adrena-Brain ability; Mind Bend 60 confuse)
Bloodmoon_Ursaluna = 135  # x1 (Battle-Hardened; Mad Bite 100+)
Moltres = 791  # x1 (Fighting Wings 20(+90 vs ex))
Battle_Cage = 1264  # x3 (stadium)
Fighting_Gong = 1142  # x4 (search F energy/pokemon)
Poke_Pad = 1152  # x4
Lillie_Determination = 1227  # x4
Boss_Orders = 1182  # x3
Tarragon = 1238  # x2 (recursion)
Morty_Conviction = 1187  # x2
Air_Balloon = 1174  # x2
Pokegear = 1122  # x2
Night_Stretcher = 1097  # x2
Ciphermaniac_Codebreaking = 1188  # x1
Energy_Retrieval = 1118  # x1

FIGHTING_BASICS = (Okidogi, Solrock, Lunatone, Binacle, Moltres)

# Attack IDs
Mind_Bend = 141
Good_Punch = 147
Demolish = 148
Mad_Bite = 175
Power_Gem = 979
Cosmic_Beam = 980
Fighting_Wings = 1143
Double_Draw = 1519
Scratch = 1520
Hammer_In = 1521

IGNORES_WR = {Demolish, Cosmic_Beam}


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


def compute_damage(attack_id: int, attacker: Pokemon, field_counts: dict, op_pokemon: Pokemon) -> int:
    base = attack_table[attack_id].damage if attack_id in attack_table else 0
    if attack_id == Good_Punch:
        has_dark = any(c.id == Legacy_Energy for c in attacker.energyCards)
        return base + (100 if has_dark else 0)
    if attack_id == Cosmic_Beam:
        return base if field_counts[Lunatone] >= 1 else 0
    if attack_id == Fighting_Wings:
        op_data = card_table[op_pokemon.id]
        return base + (90 if (op_data.ex or op_data.megaEx) else 0)
    if attack_id == Mad_Bite:
        dmg_taken = max(0, op_pokemon.maxHp - op_pokemon.hp)
        return base + 30 * (dmg_taken // 10)
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

    op_has_ex_active = bool(op_state.active) and op_state.active[0] is not None and (card_table[op_state.active[0].id].ex or card_table[op_state.active[0].id].megaEx)
    have_f_energy_in_hand = hand_counts[Basic_Fighting_Energy] >= 1

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
            op_pokemon = op_cards[0]
            op_data = card_table[op_pokemon.id]
            best_score = -1
            for attack_id in attacker_data.attacks:
                dmg = compute_damage(attack_id, attacker, field_counts, op_pokemon)
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
        if pid == Okidogi:
            score += 300
            if energy_count < 3:
                score += 200
        elif pid == Cornerstone_Ogerpon_ex:
            score += 250
            if energy_count < 3:
                score += 150
        elif pid == Barbaracle or pid == Binacle:
            score += 150
            if energy_count < 2:
                score += 100
        elif pid == Solrock or pid == Lunatone:
            score += 100
            if energy_count < 2:
                score += 100
        elif pid == Bloodmoon_Ursaluna:
            score += 120
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
                        if card.id == Okidogi and energy_count >= 2:
                            score += 50000
                        elif card.id == Cornerstone_Ogerpon_ex and energy_count >= 2:
                            score += 45000
                        elif card.id in (Solrock, Lunatone, Barbaracle):
                            score += 25000
                        else:
                            score += 10000
                        score += energy_count * 100 + hp
                    else:
                        score += pokemon_score(card)
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 30 if card.id in (Solrock, Lunatone, Binacle) else (20 if card.id == Okidogi else 10)
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
                    if card.id == Okidogi:
                        score = 900 if field_counts[Okidogi] < 2 else 300
                    elif card.id == Solrock:
                        score = 800 if field_counts[Solrock] < 1 else 400
                    elif card.id == Lunatone:
                        score = 800 if field_counts[Lunatone] < 1 and field_counts[Solrock] >= 1 else 350
                    elif card.id == Barbaracle or card.id == Binacle:
                        score = 700
                    elif card.id == Cornerstone_Ogerpon_ex:
                        score = 500
                    elif card.id == Basic_Fighting_Energy:
                        score = 450
                    elif card.id in (Boss_Orders, Lillie_Determination):
                        score = 350
                    else:
                        score = 200 - hand_counts[card.id] * 50
                elif context == SelectContext.DISCARD:
                    if card.id == Prism_Energy:
                        score = 50
                    elif card.id == Moltres or card.id == Munkidori:
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
                if card.id == Okidogi and field_counts[Okidogi] >= 2:
                    score = 3000
            elif data.cardType == CardType.SUPPORTER:
                if state.supporterPlayed:
                    score = -1
                elif card.id == Lillie_Determination:
                    score = 15000
                elif card.id == Boss_Orders and len(op_state.bench) > 0:
                    score = 14000
                elif card.id == Tarragon and hand_counts[Basic_Fighting_Energy] == 0:
                    score = 13000
                elif card.id == Morty_Conviction and len(my_state.hand) >= 2:
                    score = 8500
                elif card.id == Ciphermaniac_Codebreaking:
                    score = 12500
                else:
                    score = 9000
            elif data.cardType == CardType.STADIUM:
                score = 12000 if stadium_id != Battle_Cage else -1
            elif data.cardType == CardType.TOOL:
                score = -1  # handled via ATTACH
            else:  # ITEM
                if card.id == Fighting_Gong:
                    score = 12000
                elif card.id == Poke_Pad:
                    score = 11500
                elif card.id == Pokegear:
                    score = 11000
                elif card.id == Night_Stretcher:
                    score = 5000
                elif card.id == Energy_Retrieval:
                    score = 4500
                else:
                    score = 3000
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Air_Balloon:
                score = 8500 if pokemon.id in (Okidogi, Cornerstone_Ogerpon_ex) else -1
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 30000 + len(pokemon.energies) * 10
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Barbaracle and not ability_used and have_f_energy_in_hand:
                score = 41000
            elif card.id == Lunatone and not ability_used and field_counts[Solrock] >= 1 and have_f_energy_in_hand:
                score = 40000
            elif card.id == Munkidori and not ability_used:
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
            if card.id in (Barbaracle, Lunatone, Munkidori):
                ability_used = True

    return output
