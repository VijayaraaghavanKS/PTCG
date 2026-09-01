import os
from collections import defaultdict

from cg.api import AreaType, CardType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Mega Lopunny ex / Mega Froslass ex "Mist Energy" Deck
Mined from a real rank-9 ladder replay. Win condition (from real card text,
not assumption):
  - Mega Lopunny ex: Gale Thrust (60, or 60+170=230 if it just moved from
    Bench to Active this turn) / Spiky Hopper (160, ignores opponent Active
    effects). Cheap (1-2 colorless), the reliable primary attacker.
  - Mega Froslass ex: Absolute Snow (150 dmg + Asleep) is the reliable line;
    Resentful Refrain (50 x opponent's hand size) is a situational finisher
    against hand-heavy opponents, not the primary plan.
  - Dudunsparce's Run Away Draw (draw 3, then shuffle self+attachments back
    into deck) is the consistency engine.
  - Wally's Compassion fully heals a damaged Mega ex and returns its Energy
    to hand -- the resilience/tempo tool protecting the one-per-turn Mega
    investment.
  - Battle Cage protects the bench (Dunsparce/basics) from spread damage.
  - Air Balloon cuts retreat cost, supporting repeated bench<->active swaps
    to re-trigger Gale Thrust's moved-this-turn bonus.
"""

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = [int(csv[i]) for i in range(60)]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist
Mist_Energy = 11  # x4
Basic_W_Energy = 3  # x3
Enriching_Energy = 13  # x1
Dunsparce = 305  # x4
Dudunsparce = 66  # x3 (evolves from Dunsparce)
Buneary = 848  # x2
Mega_Lopunny_ex = 849  # x2 (evolves from Buneary)
Snorunt = 860  # x2
Mega_Froslass_ex = 861  # x2 (evolves from Snorunt)
Fan_Rotom = 174  # x1
Buddy_Buddy_Poffin = 1086  # x4
Ultra_Ball = 1121  # x4
Poke_Pad = 1152  # x4
Lillie_Determination = 1227  # x4
Wallys_Compassion = 1229  # x4
Hand_Trimmer = 1087  # x3
Air_Balloon = 1174  # x3
Hilda = 1225  # x3
Battle_Cage = 1264  # x3
Pokegear = 1122  # x2
Boss_Orders = 1182  # x2

GALE_THRUST = 1225
SPIKY_HOPPER = 1226
RESENTFUL_REFRAIN = 1240
ABSOLUTE_SNOW = 1241
LAND_CRUSH = 76
RAM = 424

MEGA_LINE = {Buneary, Mega_Lopunny_ex, Snorunt, Mega_Froslass_ex}
BASIC_ATTACKERS = {Dunsparce, Buneary, Snorunt}


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


def pokemon_threat_score(pokemon: Pokemon) -> int:
    """How valuable is it to KO / target this opposing Pokemon."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    score -= pokemon.hp  # lower remaining HP -> easier/better to finish
    return score


def best_attack_damage(pokemon: Pokemon, just_moved_to_active: bool, op_hand_size: int = 0) -> int:
    """Max damage `pokemon` can currently deal, given energies attached (rough, ignores exact typing gate)."""
    if pokemon.id == Mega_Lopunny_ex:
        n = len(pokemon.energies)
        best = 0
        if n >= 1:
            best = max(best, 60 + (170 if just_moved_to_active else 0))
        if n >= 2:
            best = max(best, 160)
        return best
    if pokemon.id == Mega_Froslass_ex:
        n = len(pokemon.energies)
        best = 0
        if n >= 1:
            best = max(best, 50 * op_hand_size)  # Resentful Refrain, 1 energy
        if n >= 3:
            best = max(best, 150)  # Absolute Snow
        return best
    if pokemon.id == Dudunsparce:
        return 90 if len(pokemon.energies) >= 3 else 0
    if pokemon.id == Dunsparce:
        return 20 if len(pokemon.energies) >= 2 else 0
    if pokemon.id == Buneary:
        return 20 if len(pokemon.energies) >= 2 else 0
    if pokemon.id == Snorunt:
        return 10 if len(pokemon.energies) >= 1 else 0
    return 0


def attack_damage_estimate(attack_id: int, active_just_moved: bool, op_hand_size: int) -> int:
    """Estimated damage of a specific attack choice (ignores weakness/resistance -- conservative)."""
    if attack_id == GALE_THRUST:
        return 60 + (170 if active_just_moved else 0)
    if attack_id == SPIKY_HOPPER:
        return 160
    if attack_id == ABSOLUTE_SNOW:
        return 150
    if attack_id == RESENTFUL_REFRAIN:
        return 50 * op_hand_size
    if attack_id == LAND_CRUSH:
        return 90
    if attack_id == RAM:
        return 20
    return 0


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

    hand_counts = defaultdict(int)
    for card in my_state.hand:
        hand_counts[card.id] += 1

    op_hand_size = op_state.handCount

    field_counts = defaultdict(int)
    active_mega_ready = False
    active_id = 0
    active_just_moved = False
    for card in my_state.active:
        if card is None:
            continue
        active_id = card.id
        active_just_moved = card.appearThisTurn
        field_counts[card.id] += 1
        if card.id in (Mega_Lopunny_ex, Mega_Froslass_ex) and best_attack_damage(card, active_just_moved, op_hand_size) > 0:
            active_mega_ready = True
    for card in my_state.bench:
        field_counts[card.id] += 1

    damaged_mega_on_field = None
    for card in list(my_state.active) + list(my_state.bench):
        if card is not None and card.id in (Mega_Lopunny_ex, Mega_Froslass_ex):
            max_hp = card_table[card.id].hp
            if card.hp < max_hp:
                if damaged_mega_on_field is None or card.hp < damaged_mega_on_field.hp:
                    damaged_mega_on_field = card

    no_draw = my_state.deckCount <= 8
    can_evolve_lopunny = field_counts[Buneary] >= 1
    can_evolve_froslass = field_counts[Snorunt] >= 1
    stadium_id = state.stadium[0].id if state.stadium else 0

    can_attack = False
    can_switch = False
    for o in select.option:
        if o.type == OptionType.ATTACK:
            can_attack = True
        elif o.type == OptionType.RETREAT:
            can_switch = True

    use_support = 0
    if context == SelectContext.MAIN and not state.supporterPlayed:
        best_support_score = -1
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card_table[card.id].cardType == CardType.SUPPORTER:
                    s = 0
                    if card.id == Wallys_Compassion and damaged_mega_on_field is not None:
                        s = 90000
                    elif card.id == Hilda:
                        s = 40000
                    elif card.id == Boss_Orders and len(op_state.bench) > 0:
                        s = 60000 if not can_attack else 30000
                    elif card.id == Hand_Trimmer and op_hand_size >= 6:
                        s = 25000
                    elif card.id == Lillie_Determination:
                        s = 10000
                    if s > best_support_score:
                        best_support_score = s
                        use_support = card.id

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = -1 if context == SelectContext.IS_FIRST else 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                hp = card.hp if isinstance(card, Pokemon) else 0
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
                    if o.playerIndex == my_index:
                        if card.id in (Mega_Lopunny_ex, Mega_Froslass_ex):
                            score += 50000 + best_attack_damage(card, True, op_hand_size) * 10
                        elif card.id == Dudunsparce:
                            score += 20000
                        elif card.id in (Buneary, Snorunt):
                            score += 10000
                        elif card.id == Dunsparce:
                            score += 8000
                        score += energy_count * 1000 + hp
                    else:
                        score += pokemon_threat_score(card)
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 5000 if card.id in BASIC_ATTACKERS else 1000
                elif context in (SelectContext.TO_BENCH, SelectContext.TO_HAND):
                    score = 100
                    if card.id in MEGA_LINE:
                        score = 5000
                    elif card.id == Dudunsparce:
                        score = 3000
                elif context == SelectContext.DISCARD:
                    score = -100
                    if card.id in (Hand_Trimmer, Basic_W_Energy):
                        score = -10
                    if card.id in MEGA_LINE or card.id == Dudunsparce:
                        score = -5000
                elif context == SelectContext.ATTACH_FROM:
                    if card_table[card.id].cardType == CardType.TOOL:
                        score = 40000 + (2000 if o.area == AreaType.ACTIVE else 0)
                    else:
                        score = 30000 if o.area == AreaType.ACTIVE else 10000
                        if card.id in (Mega_Lopunny_ex, Mega_Froslass_ex):
                            score += 5000
                else:
                    score = pokemon_threat_score(card) if isinstance(card, Pokemon) else 0
        elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            score = 10
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None and card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                score += 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            cid = card.id
            if cid in (Mega_Lopunny_ex, Mega_Froslass_ex):
                score = -1  # played via EVOLVE, not PLAY
            elif cid in (Buneary, Snorunt, Dunsparce):
                score = 51000
            elif cid == Fan_Rotom:
                score = 52000 if state.turn <= 2 else -1
            elif cid == Wallys_Compassion:
                score = 49000 if (card.id == use_support and damaged_mega_on_field is not None) else -1
            elif cid == Boss_Orders:
                score = 49000 if card.id == use_support else -1
            elif cid == Hilda:
                score = 48000 if card.id == use_support else -1
            elif cid == Hand_Trimmer:
                score = 47000 if card.id == use_support else -1
            elif cid == Lillie_Determination:
                score = 46000 if card.id == use_support else -1
            elif no_draw:
                score = -1
            elif cid == Buddy_Buddy_Poffin:
                score = 45000 if (field_counts[Buneary] + field_counts[Snorunt] + field_counts[Dunsparce] < 4) else 5000
            elif cid == Ultra_Ball:
                score = 44000 if len(my_state.hand) >= 2 else -1
            elif cid == Poke_Pad:
                score = 43000
            elif cid == Pokegear:
                score = 42000 if use_support == 0 else -1
            elif cid == Air_Balloon:
                score = -1  # attached via ATTACH
            elif cid == Battle_Cage:
                score = 41000 if stadium_id != Battle_Cage else -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card_table[card.id].cardType == CardType.TOOL:
                score = 30000 + (2000 if o.inPlayArea == AreaType.ACTIVE else 0)
                if card.id == Air_Balloon and pokemon.id in (Mega_Lopunny_ex, Buneary):
                    score += 5000
            else:
                score = 20000 + (5000 if o.inPlayArea == AreaType.ACTIVE else 0)
                if pokemon.id in (Mega_Lopunny_ex, Mega_Froslass_ex):
                    score += 4000
                elif pokemon.id in (Buneary, Snorunt, Dunsparce):
                    score += 1000
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Buneary:
                score += 70000
            elif pokemon.id == Snorunt:
                score += 69000
            elif pokemon.id == Dunsparce:
                score += 60000
            else:
                score += 50000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if no_draw and card.id == Dudunsparce:
                score = -1
            elif card.id == Dudunsparce:
                score = 35000
            elif card.id == Fan_Rotom:
                score = 55000
            else:
                score = 30000
        elif o.type == OptionType.RETREAT:
            score = 8000 if (not can_attack and not active_mega_ready) else -1
        elif o.type == OptionType.ATTACK:
            op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None
            dmg = attack_damage_estimate(o.attackId, active_just_moved, op_hand_size)
            if op_active is not None and dmg >= op_active.hp:
                score = 200000  # this attack KOs the opponent's active right now -- always take it
            elif o.attackId == RESENTFUL_REFRAIN:
                score = 1000 + op_hand_size * 200
            elif o.attackId in (SPIKY_HOPPER, ABSOLUTE_SNOW):
                score = 900
            elif o.attackId == GALE_THRUST:
                score = 950 if active_just_moved else 700
            elif o.attackId == LAND_CRUSH:
                score = 600
            else:
                score = 500

        scores.append(score)

    output = []
    if scores:
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if (sorted_scores[i][1] >= 0
                or select.minCount > i
                or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])

    return output
