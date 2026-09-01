import os
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, all_attack, to_observation_class

"""
Team Rocket's Honchkrow / Porygon Deck  (EX_11, "Team Rocket Honchkrow control")
Local test-harness reference opponent only -- NOT a Kaggle submission candidate.

This is a minimal, reasonably-competent rule-based pilot for a resource-denial
control archetype, distinct from the Crustle wall (which walls via a
damage-immunity Ability): this deck disrupts through card/hand denial
(Porygon's Hacking, Torment attack-lock) and converts a dense "Team Rocket"
Supporter engine into damage via Honchkrow's Rocket Feathers (60 dmg per
Team-Rocket Supporter discarded from hand) and Porygon2's R Command (20 dmg
per Team-Rocket Supporter already in the discard pile). Not tuned/optimized
beyond "keep the TR supporter engine running, attack with whatever does the
most damage, disrupt when a disruption Supporter is legal to play". Zero
crashes is the bar.
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
Team_Rocket_Energy = 15  # x4 (only attaches to Team Rocket's Pokemon)
Ignition_Energy = 17  # x4 (discards itself end of turn)
Murkrow = 463  # x4 (Deceit search supporter / Torment 30 dmg + attack-lock)
Honchkrow = 891  # x3 (Rocket Feathers 60x discarded TR supporters / Hammer In 100)
Porygon = 473  # x2 (Hacking -- mutual hand discard)
Porygon2 = 474  # x1 (R Command 20x TR supporters in discard)
Articuno = 414  # x2 (Repelling Veil passive wall; Dark Frost 60(+60 w/ TR Energy))
Proton = 1220  # x4 supporter (extra first-turn play)
Petrel = 1219  # x4 supporter (search trainer)
Giovanni = 1218  # x4 supporter (switch own+force opp switch -- gust)
Archer = 1217  # x4 supporter (conditional on a TR KO last turn)
Ariana = 1216  # x4 supporter (draw to 5, or 8 if all-TR board)
Team_Rocket_Factory = 1257  # x3 (stadium, draw 2 on TR supporter play)
Poke_Pad = 1152  # x4
Transceiver = 1134  # x4 (search TR supporter)
Roto_Stick = 1077  # x4 (dig top 4 for supporters)
Night_Stretcher = 1097  # x3
Ultra_Ball = 1121  # x1
Miracle_Headset = 1109  # x1 (ACE SPEC, 2 supporters from discard to hand)

TR_SUPPORTERS = (Proton, Petrel, Giovanni, Archer, Ariana)

# Attack IDs
Dark_Frost = 583
Deceit = 652
Torment = 653
Hacking = 669
R_Command = 670
Rocket_Feathers = 1285
Hammer_In = 1286


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


def compute_damage(attack_id: int, attacker: Pokemon, hand_counts: dict, discard_counts: dict) -> int:
    base = attack_table[attack_id].damage if attack_id in attack_table else 0
    if attack_id == Rocket_Feathers:
        n = sum(hand_counts[c] for c in TR_SUPPORTERS)
        return 60 * n
    if attack_id == R_Command:
        n = sum(discard_counts[c] for c in TR_SUPPORTERS)
        return 20 * n
    if attack_id == Dark_Frost:
        has_tr_energy = any(c.id == Team_Rocket_Energy for c in attacker.energyCards)
        return base + (60 if has_tr_energy else 0)
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
    discard_counts = defaultdict(int)

    my_cards = [c for c in my_state.active if c is not None] + [c for c in my_state.bench if c is not None]
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
                dmg = compute_damage(attack_id, attacker, hand_counts, discard_counts)
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
        if pid == Honchkrow:
            score += 300
            if energy_count < 2:
                score += 200
        elif pid == Articuno:
            score += 250
            if energy_count < 2:
                score += 150
        elif pid == Murkrow:
            if energy_count < 1:
                score += 100
            else:
                score -= 30
        elif pid == Porygon2:
            score += 100
        if active:
            score += 20
        if energy_count >= 2:
            score -= 800
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
                        if card.id == Honchkrow and energy_count >= 1:
                            score += 50000
                        elif card.id == Articuno:
                            score += 30000
                        elif card.id == Porygon2:
                            score += 20000
                        else:
                            score += 10000
                        score += energy_count * 100 + hp
                    else:
                        score += pokemon_score(card)
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 30 if card.id in (Murkrow, Porygon) else (20 if card.id == Articuno else 10)
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
                    if card.id == Murkrow:
                        score = 900 if field_counts[Murkrow] + field_counts[Honchkrow] < 2 else 300
                    elif card.id == Porygon:
                        score = 700 if field_counts[Porygon] + field_counts[Porygon2] < 1 else 300
                    elif card.id == Articuno:
                        score = 650
                    elif card.id in TR_SUPPORTERS:
                        score = 550
                    elif card.id in (Team_Rocket_Energy, Ignition_Energy):
                        score = 450
                    else:
                        score = 200 - hand_counts[card.id] * 50
                elif context == SelectContext.DISCARD:
                    if card.id == Ignition_Energy:
                        score = 60
                    elif card.id in TR_SUPPORTERS:
                        score = 45  # feeds R Command / already-used copies
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
                if card.id == Murkrow and field_counts[Murkrow] + field_counts[Honchkrow] >= 3:
                    score = 3000
            elif data.cardType == CardType.SUPPORTER:
                if state.supporterPlayed:
                    score = -1
                elif card.id == Ariana:
                    score = 15000
                elif card.id == Giovanni and len(op_state.bench) > 0:
                    score = 14500
                elif card.id == Petrel:
                    score = 13500
                elif card.id == Archer:
                    score = 13000
                elif card.id == Proton:
                    score = 12500
                else:
                    score = 9000
            elif data.cardType == CardType.STADIUM:
                score = 12000 if stadium_id != Team_Rocket_Factory else -1
            elif data.cardType == CardType.TOOL:
                score = -1
            else:  # ITEM
                if card.id == Transceiver:
                    score = 12000
                elif card.id == Poke_Pad:
                    score = 11500
                elif card.id == Roto_Stick:
                    score = 11000
                elif card.id == Night_Stretcher:
                    score = 5000
                elif card.id == Ultra_Ball:
                    score = 10500 if len(my_state.hand) >= 3 else -1
                elif card.id == Miracle_Headset:
                    score = 4500
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

    return output
