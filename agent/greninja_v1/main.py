import os
from collections import defaultdict

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
                     Card, Pokemon, all_card_data, all_attack, to_observation_class)

"""
Greninja ex Snipe Deck (Froakie -> Frogadier -> Greninja ex)

Win condition: get Greninja ex online and fire Shinobi Blade (single {W} energy,
170 damage -- the best damage-per-energy in the whole card pool) every turn.
Mirage Barrage ({W} + 2 colorless, discards 2 of Greninja ex's own attached
energy) hits 120 to TWO of the opponent's Pokemon at once, which is how we
punch through walls and clear a path to double-prize turns. Greninja ex is
also a Tera Pokemon: while it sits on the Bench it takes zero damage from any
attack, so it is safe to park it there half-charged while Froakie/Frogadier
soak hits up front, and only walk it into the Active Spot once it can swing
immediately.

Speed is the deck's biggest risk (Stage 2 + multi-prize ex is a slow, costly
combo), so the trainer line leans hard into evolution acceleration: Rare
Candy skips Frogadier outright, Buddy-Buddy Poffin and Ultra Ball dig for
Froakie/Greninja ex, and Dawn fetches the whole Basic/Stage1/Stage2 line in
one card.
"""

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = [int(csv[i]) for i in range(60)]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
attack_by_id = {a.attackId: a for a in all_attack()}

# Decklist
Froakie = 33            # x4 (Flock: bench up to 2 more Froakie, 1en; Flop: 10 dmg, 1en)
Frogadier = 34           # x4 (Numbing Water: 20 dmg + coin-flip paralyze, 1en)
Greninja_ex = 40         # x4 (Shinobi Blade: 170 dmg + search 1 card, 1{W}; Mirage Barrage: 120 to
                         #     2 opp Pokemon, discards 2 of its own energy, {W}+2 colorless; Tera:
                         #     immune to attack damage while Benched)
Rare_Candy = 1079        # x4 (skip Frogadier -> straight to Greninja ex)
Buddy_Buddy_Poffin = 1086  # x4 (search up to 2 Basic <=70 HP -> bench; only Froakie qualifies here)
Ultra_Ball = 1121        # x4 (discard 2, search any Pokemon -> hand)
Dawn = 1231              # x4 (search 1 Basic + 1 Stage1 + 1 Stage2 -> hand, all at once)
Boss_Orders = 1182       # x4 (gust an opponent Benched Pokemon into Active)
Night_Stretcher = 1097   # x4 (Pokemon or Basic Energy: discard -> hand)
Pokegear = 1122          # x3 (look at top 7, may take a Supporter)
Hilda = 1225             # x3 (search 1 Evolution Pokemon + 1 Energy -> hand)
Carmine = 1192           # x2 (discard hand, draw 5; usable turn 1 even going first)
Switch = 1123            # x2 (free retreat)
Lillies_Determination = 1227  # x2 (shuffle hand into deck, draw 6, or 8 at exactly 6 prizes)
Basic_Water_Energy = 3   # x12

GRENINJA_LINE = {Froakie, Frogadier, Greninja_ex}
SUPPORTERS = {Dawn, Boss_Orders, Hilda, Carmine, Lillies_Determination}

def _attacks_by_name(card_id: int) -> dict:
    """Look up move names -> attackId scoped to ONE card's own attacks list, since move
    names like 'Flop' are reused across many unrelated cards in the full attack table."""
    return {attack_by_id[aid].name: aid for aid in card_table[card_id].attacks}


_froakie_atk = _attacks_by_name(Froakie)
_frogadier_atk = _attacks_by_name(Frogadier)
_greninja_atk = _attacks_by_name(Greninja_ex)

FLOCK = _froakie_atk["Flock"]
FLOP = _froakie_atk["Flop"]
NUMBING_WATER = _frogadier_atk["Numbing Water"]
SHINOBI_BLADE = _greninja_atk["Shinobi Blade"]
MIRAGE_BARRAGE = _greninja_atk["Mirage Barrage"]

SHINOBI_DAMAGE = 170
MIRAGE_DAMAGE = 120
MIRAGE_ENERGY_NEEDED = 3

pre_turn = -1


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


def prize_value(pokemon: Pokemon) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def is_tera_safe_on_bench(card_id: int) -> bool:
    data = card_table.get(card_id)
    return bool(data and data.tera)


def default_fallback(select) -> list[int]:
    """Guaranteed-legal minimal selection used if anything above throws."""
    n = max(select.minCount, 0)
    n = min(n, select.maxCount, len(select.option))
    return list(range(n))


def agent(obs_dict: dict) -> list[int]:
    try:
        return _agent_inner(obs_dict)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return my_deck
            return default_fallback(obs.select)
        except Exception:
            return my_deck


def _agent_inner(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn != pre_turn:
        pre_turn = state.turn

    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None

    field_counts = defaultdict(int)
    if my_active:
        field_counts[my_active.id] += 1
    for p in my_state.bench:
        field_counts[p.id] += 1

    hand_counts = defaultdict(int)
    for c in my_state.hand or []:
        hand_counts[c.id] += 1

    discard_counts = defaultdict(int)
    for c in my_state.discard:
        discard_counts[c.id] += 1

    my_pokemon_in_play = ([my_active] if my_active else []) + list(my_state.bench)
    op_pokemon_in_play = ([op_active] if op_active else []) + list(op_state.bench)

    have_greninja_in_play = field_counts[Greninja_ex] > 0
    have_ready_greninja_bench = any(
        p.id == Greninja_ex and len(p.energies) >= 1 for p in my_state.bench
    )
    no_draw = my_state.deckCount <= 3  # avoid decking ourselves out

    # Total bodies currently in play across the whole evolution line. Getting knocked down
    # to 0 with an empty bench is an instant loss ("no active Pokemon") regardless of the
    # prize count -- a real risk for a deck built around one Stage 2 ex attacker with no
    # secondary lines. board_thin means we should urgently add spare bodies even ahead of
    # further evolving/searching for Greninja ex copies.
    total_bodies = field_counts[Froakie] + field_counts[Frogadier] + field_counts[Greninja_ex]
    board_thin = total_bodies <= 1

    def can_shinobi(pokemon: Pokemon) -> bool:
        return pokemon is not None and pokemon.id == Greninja_ex and len(pokemon.energies) >= 1

    def can_mirage(pokemon: Pokemon) -> bool:
        return pokemon is not None and pokemon.id == Greninja_ex and len(pokemon.energies) >= MIRAGE_ENERGY_NEEDED

    def mirage_double_ko_count() -> int:
        return sum(1 for p in op_pokemon_in_play if p is not None and p.hp <= MIRAGE_DAMAGE)

    # -------------------------------------------------------------- search/hand priority
    def pick_score(card_id: int, ignore_hand_dupe: bool) -> int:
        """How much we want a given card ID in hand / played, given board state."""
        score = -1
        if card_id == Froakie:
            if total_bodies == 0:
                score = 45000
            elif board_thin:
                score = 42000  # only 1 body in play -- one bad KO away from bench-out, refill now
            elif field_counts[Froakie] < 3:
                score = 18000
            else:
                score = 4000
        elif card_id == Frogadier:
            if field_counts[Froakie] >= 1 and hand_counts[Rare_Candy] == 0:
                score = 22000
            else:
                score = 3000
        elif card_id == Greninja_ex:
            if field_counts[Froakie] >= 1 or field_counts[Frogadier] >= 1:
                score = 90000
            else:
                score = 15000
        elif card_id == Rare_Candy:
            if field_counts[Froakie] >= 1 and (hand_counts[Greninja_ex] > 0 or not ignore_hand_dupe):
                score = 85000
            elif field_counts[Froakie] >= 1:
                score = 60000
            else:
                score = 500
        elif card_id == Buddy_Buddy_Poffin:
            if board_thin:
                score = 65000  # outranks Rare Candy / Dawn: refilling bench is the emergency
            elif total_bodies < 3:
                score = 32000
            else:
                score = 6000
        elif card_id == Ultra_Ball:
            if board_thin:
                score = 55000
            elif not have_greninja_in_play:
                score = 30000
            else:
                score = 20000
        elif card_id == Dawn:
            score = 50000
        elif card_id == Hilda:
            score = 27000
        elif card_id == Boss_Orders:
            score = 26000 if len(op_state.bench) > 0 else 300
        elif card_id == Night_Stretcher:
            has_recyclable = discard_counts[Froakie] + discard_counts[Frogadier] + discard_counts[Greninja_ex] + discard_counts[Basic_Water_Energy] > 0
            score = 21000 if has_recyclable else 500
        elif card_id == Pokegear:
            score = 24000
        elif card_id == Carmine:
            score = 33000 if state.turn <= 3 else 4000
        elif card_id == Lillies_Determination:
            score = 19000 if len(my_state.hand or []) >= 2 else 41000
        elif card_id == Switch:
            score = 9000 if my_active is not None and my_active.id != Greninja_ex and have_ready_greninja_bench else 2000
        elif card_id == Basic_Water_Energy:
            total_energy_in_play = sum(len(p.energies) for p in my_pokemon_in_play)
            score = 26000 if total_energy_in_play < 6 else 8000

        if no_draw and card_id in (Dawn, Hilda, Carmine, Lillies_Determination, Buddy_Buddy_Poffin, Pokegear, Night_Stretcher):
            score = -1

        if not ignore_hand_dupe and hand_counts[card_id] > 0 and card_id != Basic_Water_Energy:
            score -= 60000
        return score

    def discard_score(card_id: int) -> int:
        # Discard cost for Ultra Ball / similar: give up the least valuable card.
        base = pick_score(card_id, True)
        if card_id == Basic_Water_Energy:
            base -= 5000  # keep some energy in hand, but it's the safest thing to pitch
        if card_id in (Froakie,) and field_counts[Froakie] + hand_counts[Froakie] > 2:
            base -= 3000
        return -base

    # -------------------------------------------------------------- energy attach priority
    def attach_score(pokemon: Pokemon, is_active: bool) -> int:
        if pokemon is None or pokemon.id not in GRENINJA_LINE:
            return -1
        energy_count = len(pokemon.energies)
        score = 4000
        if pokemon.id == Greninja_ex:
            score += 4000
            if energy_count < 1:
                score += 3000
            elif energy_count < MIRAGE_ENERGY_NEEDED:
                score += 900
            else:
                score -= 1500
        elif pokemon.id == Frogadier:
            score += 1200
            if energy_count < 1:
                score += 600
            else:
                score -= 800
        else:  # Froakie
            score += 200
            if energy_count < 1:
                score += 300
            else:
                score -= 800
        if is_active:
            score += 1800
        return score

    # -------------------------------------------------------------- our own repositioning target
    def own_field_target_score(pokemon: Pokemon) -> int:
        if pokemon is None:
            return -1
        energy_count = len(pokemon.energies)
        score = 3000
        if pokemon.id == Greninja_ex:
            score += 6000
            if can_shinobi(pokemon):
                score += 5000
            score += energy_count * 400
        elif pokemon.id == Frogadier:
            score += 2000 + energy_count * 200
        elif pokemon.id == Froakie:
            score += 500 + energy_count * 200
        score += pokemon.hp
        return score

    # -------------------------------------------------------------- opponent-side targeting
    def opponent_target_score(pokemon: Pokemon) -> int:
        if pokemon is None:
            return -1
        hp = pokemon.hp
        score = prize_value(pokemon) * 1500
        if hp <= SHINOBI_DAMAGE:
            score += 6000
        if hp <= MIRAGE_DAMAGE:
            score += 4000
        score += len(pokemon.energies) * 150
        score += max(0, 260 - hp)
        return score

    # -------------------------------------------------------------- retreat/loss-shielding
    def attack_is_lethal_now() -> bool:
        if my_active is None or op_active is None:
            return False
        if can_shinobi(my_active) and op_active.hp <= SHINOBI_DAMAGE:
            return True
        if can_mirage(my_active) and mirage_double_ko_count() >= 1:
            return True
        if my_active.id == Frogadier and len(my_active.energies) >= 1 and op_active.hp <= 20:
            return True
        if my_active.id == Froakie and len(my_active.energies) >= 1 and op_active.hp <= 10:
            return True
        return False

    def retreat_score() -> int:
        if my_active is None or len(my_state.bench) == 0:
            return -1
        if attack_is_lethal_now():
            return -1  # take the KO this turn, don't waste it retreating

        max_hp = my_active.maxHp or 1
        hp_frac = my_active.hp / max_hp
        has_decent_bench = any(p.hp > 0 for p in my_state.bench)

        if my_active.id == Greninja_ex:
            # Protect the 2-prize attacker: it takes zero attack damage while Benched
            # (Tera), so once it's eaten a big hit it's safer parked behind a cheaper
            # sacrificial body than left active to be finished off for a 2-prize swing.
            # This score must outrank a routine (non-lethal) ATTACK score (~100000) or the
            # protection never actually fires, since RETREAT and ATTACK compete directly
            # in the same MAIN selection.
            if hp_frac <= 0.55 and has_decent_bench:
                return 150000
            return -1
        if have_ready_greninja_bench:
            return 45000  # bring the finisher in over a weaker attacker's routine swing
        if hp_frac <= 0.35 and has_decent_bench:
            return 7000  # get a non-Greninja attacker out of danger
        return -1

    # ================================================================ main scoring loop
    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = -1  # prefer going second: extra draw, no turn-1 attack restriction
            elif context == SelectContext.MULLIGAN:
                score = 1
            elif context == SelectContext.COIN_HEAD:
                score = 1
            elif context == SelectContext.ACTIVATE:
                score = 1  # e.g. Shinobi Blade's optional search: always take it
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

                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if o.playerIndex == my_index:
                        score = own_field_target_score(card)
                    else:
                        score = opponent_target_score(card)
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    score = 100 if card.id == Froakie else 1
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 100 if card.id == Froakie else -1
                elif context in (SelectContext.TO_BENCH, SelectContext.TO_HAND, SelectContext.LOOK):
                    score = pick_score(card.id, True)
                    hand_counts[card.id] += 1
                elif context == SelectContext.DISCARD:
                    score = discard_score(card.id)
                    hand_counts[card.id] -= 1
                elif context in (SelectContext.DAMAGE, SelectContext.EFFECT_TARGET,
                                  SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if o.playerIndex == my_index:
                        score = -1 if hp > 0 else 1
                    else:
                        score = opponent_target_score(card) if hp > 0 else -1
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.EVOLVES_FROM:
                    score = 1000 + energy_count * 400 + hp
                    if o.area == AreaType.ACTIVE:
                        score += 500
                elif context == SelectContext.EVOLVES_TO:
                    if card.id == Greninja_ex:
                        score = 60000
                    elif card.id == Frogadier:
                        score = 40000
                    else:
                        score = 100
                else:
                    score = pick_score(card.id, True)
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
            elif no_draw and card.id in (Dawn, Hilda, Carmine, Lillies_Determination, Buddy_Buddy_Poffin, Pokegear, Night_Stretcher):
                score = -1
            else:
                score = pick_score(card.id, True)
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Greninja_ex:
                score = 95000
            elif card.id == Frogadier:
                score = 70000
            else:
                score = 40000
            score += len(pokemon.energies) * 200
            if o.inPlayArea == AreaType.ACTIVE:
                score += 300
        elif o.type == OptionType.ABILITY:
            score = -1 if no_draw else 20000
        elif o.type == OptionType.RETREAT:
            score = retreat_score()
        elif o.type == OptionType.ATTACK:
            if o.attackId == SHINOBI_BLADE:
                lethal = op_active is not None and op_active.hp <= SHINOBI_DAMAGE
                score = 500000 if lethal else 100000
            elif o.attackId == MIRAGE_BARRAGE:
                kos = mirage_double_ko_count()
                if kos >= 2:
                    score = 600000
                elif kos == 1:
                    score = 90000
                else:
                    # burns 2 of our own energy for chip damage to 2 targets; only worth it
                    # when we have energy to spare (won't cripple next Shinobi Blade)
                    spare_energy = len(my_active.energies) - 2 if my_active else 0
                    score = 45000 if spare_energy >= 1 else 15000
            elif o.attackId == NUMBING_WATER:
                lethal = op_active is not None and op_active.hp <= 20
                score = 60000 if lethal else 20000
            elif o.attackId == FLOP:
                lethal = op_active is not None and op_active.hp <= 10
                score = 30000 if lethal else 8000
            elif o.attackId == FLOCK:
                # Setup tool, not a real attack: only worth using early while still
                # short on Froakie copies in play, since it forfeits this turn's attack.
                score = 25000 if (state.turn <= 6 and field_counts[Froakie] < 3 and not have_greninja_in_play) else 500
            else:
                score = 1000
        elif o.type == OptionType.DISCARD:
            card = get_card(obs, o.area, o.index, my_index)
            score = discard_score(card.id) if card is not None else 0

        scores.append(score)

    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    output = []
    for i in range(select.maxCount):
        idx, sc = sorted_scores[i]
        if sc >= 0 or select.minCount > i or context not in (
            SelectContext.TO_BENCH, SelectContext.SETUP_BENCH_POKEMON, SelectContext.DISCARD,
        ):
            output.append(idx)

    if len(output) < select.minCount:
        # top off with whatever's left to stay legal
        used = set(output)
        for idx, _ in sorted_scores:
            if idx not in used:
                output.append(idx)
                used.add(idx)
            if len(output) >= select.minCount:
                break

    return output
