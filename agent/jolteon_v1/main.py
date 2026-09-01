import os
from collections import defaultdict

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
    Card, Pokemon, all_card_data, all_attack, to_observation_class,
)

"""
Jolteon ex Burst-Damage Deck (Lightning)
-----------------------------------------
Win condition: get Eevee active on turn 1 and use its "Boosted Evolution"
ability (evolve during your first turn or the turn you play it) to skip the
usual evolve-sickness turn, then ride Jolteon ex's "Flashing Spear" ({L}+1,
base 60, +90 per Basic Energy discarded from the Bench, up to 2) for burst
damage up to 240. Misty's Magikarp / Poltchageist sit on the Bench purely as
disposable Basic-Energy "ammo dumps" -- they're immune to all damage while
benched (own Ability), so loading spare Lightning onto them and discarding it
for Flashing Spear is free upside with zero risk of losing tempo to a snipe.
Jolteon ex also carries "Dravite" ({R}{W}{L}, 280 dmg) as a rare finisher for
when the mixed-type cost is affordable, at the cost of locking its own
attacks next turn -- only used when it's the last hit we need.
"""

# ---------------------------------------------------------------------------
# Deck loading
# ---------------------------------------------------------------------------
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as f:
    _csv = f.read().split("\n")
my_deck = [int(_csv[i]) for i in range(60)]

# ---------------------------------------------------------------------------
# Static card / attack metadata (engine-provided, queried once at import)
# ---------------------------------------------------------------------------
card_table = {c.cardId: c for c in all_card_data()}
attack_table = {a.attackId: a for a in all_attack()}

# Decklist card IDs
EEVEE = 317              # x4  Boosted Evolution: evolve turn 1 / turn played, while Active
JOLTEON_EX = 244         # x4  Flashing Spear (60 + up to 180), Dravite (280, locks next turn)
MAGIKARP = 362           # x4  Misty's Magikarp - immune while benched, cheap ammo/Poffin target
POLTCHAGEIST = 28        # x1  Poltchageist - immune while benched, cheap ammo/Poffin target

BUDDY_BUDDY_POFFIN = 1086   # x4  search up to 2 Basic <=70HP -> bench
ULTRA_BALL = 1121           # x4  discard 2 -> search any Pokemon -> hand
BOSS_ORDERS = 1182          # x4  gust opponent's benched Pokemon to Active
NIGHT_STRETCHER = 1097      # x4  discard pile Pokemon or Basic Energy -> hand
ENERGY_SEARCH_PRO = 1100    # x1  ACE SPEC: search any # of different-type Basic Energy -> hand
HILDA = 1225                # x4  search 1 Evolution Pokemon + 1 Energy -> hand
CHEREN = 1224                # x4  draw 3
CARMINE = 1192                # x4  discard hand, draw 5 (also usable turn 1 if going first)
POKE_PAD = 1152                # x4  search non-Rule-Box Pokemon -> hand
SWITCH = 1123                  # x2  swap Active with a Benched Pokemon

BASIC_L = 4   # x8
BASIC_R = 2   # x2 (Dravite fuel)
BASIC_W = 3   # x2 (Dravite fuel)

BASICS = {EEVEE, MAGIKARP, POLTCHAGEIST}
AMMO_TANKS = {MAGIKARP, POLTCHAGEIST}
BASIC_ENERGY_IDS = {BASIC_L, BASIC_R, BASIC_W}


def _find_attack_id(card_id: int, name_substr: str) -> int | None:
    for aid in card_table[card_id].attacks:
        if name_substr.lower() in attack_table[aid].name.lower():
            return aid
    return None


FLASHING_SPEAR = _find_attack_id(JOLTEON_EX, "Flashing Spear")
DRAVITE = _find_attack_id(JOLTEON_EX, "Dravite")


# ---------------------------------------------------------------------------
# Per-turn plan (reset whenever the turn counter changes)
# ---------------------------------------------------------------------------
class Plan:
    def __init__(self):
        self.attack_id: int | None = None      # attack we intend to use this turn
        self.discard_target: int = 0            # desired Flashing Spear bench-energy discards
        self.retreat_wanted: bool = False        # loss-shielding wants us off the field
        self.retreat_to_id: int | None = None    # preferred switch-in card id


plan = Plan()
_pre_turn = -1


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


def is_basic_energy(card_id: int) -> bool:
    data = card_table.get(card_id)
    return data is not None and data.cardType == CardType.BASIC_ENERGY


def prize_value(pokemon_id: int) -> int:
    data = card_table.get(pokemon_id)
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def apply_weakness_resistance(damage: int, attacker_type: EnergyType, defender_id: int) -> int:
    data = card_table.get(defender_id)
    if data is None or damage <= 0:
        return damage
    if data.weakness is not None and data.weakness == attacker_type:
        damage *= 2
    if data.resistance is not None and data.resistance == attacker_type:
        damage = max(0, damage - 30)
    return damage


def bench_basic_energy_count(bench: list[Pokemon]) -> int:
    total = 0
    for p in bench:
        for e in p.energyCards:
            if is_basic_energy(e.id):
                total += 1
    return total


def best_flashing_spear(available_ammo: int, defender_id: int) -> tuple[int, int]:
    """Return (discard_count, damage) maximizing damage without wasting discards
    beyond what's needed for a KO once a KO is already reachable."""
    defender_hp = card_table.get(defender_id).hp if defender_id in card_table else 999999
    best = (0, apply_weakness_resistance(60, EnergyType.LIGHTNING, defender_id))
    for n in range(1, min(2, available_ammo) + 1):
        dmg = apply_weakness_resistance(60 + 90 * n, EnergyType.LIGHTNING, defender_id)
        # Prefer the smallest n that still gets the KO; otherwise prefer more damage.
        if best[1] < defender_hp and dmg >= defender_hp:
            best = (n, dmg)
            break
        if dmg > best[1]:
            best = (n, dmg)
    return best


def affordable_attacks(pokemon: Pokemon):
    """Attacks `pokemon` can currently pay for, given its attached energies."""
    data = card_table.get(pokemon.id)
    if data is None:
        return []
    have = list(pokemon.energies)
    out = []
    for aid in data.attacks:
        atk = attack_table[aid]
        need = list(atk.energies)
        pool = list(have)
        ok = True
        # match typed requirements first, colorless (0) last
        for req in sorted(need, key=lambda t: t == EnergyType.COLORLESS):
            if req == EnergyType.COLORLESS:
                if pool:
                    pool.pop()
                else:
                    ok = False
                    break
            elif req in pool:
                pool.remove(req)
            else:
                ok = False
                break
        if ok:
            out.append(atk)
    return out


def threat_to(pokemon_id: int, opponent_active: Pokemon | None) -> int:
    """Max damage `opponent_active` could currently deal to a Pokemon of `pokemon_id`."""
    if opponent_active is None:
        return 0
    best = 0
    for atk in affordable_attacks(opponent_active):
        dmg = apply_weakness_resistance(atk.damage, card_table[opponent_active.id].energyType, pokemon_id)
        best = max(best, dmg)
    return best


def agent(obs_dict: dict) -> list[int]:
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return my_deck
    if obs.select is None:
        return my_deck
    try:
        return _decide(obs)
    except Exception:
        sel = obs.select
        n = max(0, min(sel.minCount, len(sel.option)))
        return list(range(n))


def _decide(obs: Observation) -> list[int]:
    global _pre_turn, plan

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn != _pre_turn:
        _pre_turn = state.turn
        plan = Plan()

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

    op_prize_left = len(op_state.prize)
    my_prize_left = len(my_state.prize)

    # -----------------------------------------------------------------
    # MAIN phase: build this turn's attack plan before scoring options.
    # -----------------------------------------------------------------
    if context == SelectContext.MAIN:
        _build_plan(obs, my_state, op_state, my_active, op_active, field_counts)

    scores = []
    for o in select.option:
        scores.append(_score_option(
            obs, o, state, select, context, my_index, my_state, op_state,
            my_active, op_active, field_counts, hand_counts, discard_counts,
            op_prize_left, my_prize_left,
        ))

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    output = []
    for rank, idx in enumerate(order):
        if rank >= select.maxCount:
            break
        if scores[idx] >= 0 or rank < select.minCount:
            output.append(idx)
    return output


def _build_plan(obs, my_state, op_state, my_active, op_active, field_counts):
    """Decide which attack (if any) we want to use, and loss-shielding intent.

    These two are mutually exclusive for a given decision: we either commit to
    the best attack available, or -- if staying active would likely get our
    current Pokemon knocked out next turn for less value than it's worth --
    we retreat to a safer bench Pokemon instead and skip the attack.
    """
    plan.attack_id = None
    plan.discard_target = 0
    plan.retreat_wanted = False
    plan.retreat_to_id = None

    select = obs.select
    attack_options = [o for o in select.option if o.type == OptionType.ATTACK]

    best_choice = None  # (score, attack_id, discard_target, is_ko, is_win, dmg)
    if my_active is not None and attack_options and op_active is not None:
        ammo = bench_basic_energy_count(my_state.bench)
        for o in attack_options:
            aid = o.attackId
            atk = attack_table.get(aid)
            if atk is None:
                continue
            if aid == FLASHING_SPEAR:
                n, dmg = best_flashing_spear(ammo, op_active.id)
                discard_target = n
            elif aid == DRAVITE:
                dmg = apply_weakness_resistance(280, EnergyType.LIGHTNING, op_active.id)
                discard_target = 0
            else:
                dmg = apply_weakness_resistance(atk.damage, card_table[my_active.id].energyType, op_active.id)
                discard_target = 0

            is_ko = dmg >= op_active.hp
            is_win = is_ko and prize_value(op_active.id) >= len(op_state.prize)
            # Dravite locks our own attacks next turn -- only worth it as a
            # clean KO when Flashing Spear can't also get there, or as the
            # final winning blow.
            penalty = 0
            if aid == DRAVITE and not is_win:
                penalty = -50 if is_ko else -1000
            score = (3 if is_win else 2 if is_ko else 0) * 1_000_000 + dmg + penalty
            if best_choice is None or score > best_choice[0]:
                best_choice = (score, aid, discard_target, is_ko, is_win)

    # Find the safest bench retreat candidate, if any, and whether swapping to
    # it is actually an upgrade over staying put.
    incoming = threat_to(my_active.id, op_active) if (my_active is not None and op_active is not None) else 0
    we_are_exposed = my_active is not None and incoming >= my_active.hp
    retreat_improves = False
    best_id = None
    if my_active is not None and op_active is not None and my_state.bench:
        current_margin = my_active.hp - incoming
        best_margin = None
        for cand in my_state.bench:
            margin = cand.hp - threat_to(cand.id, op_active)
            if best_margin is None or margin > best_margin:
                best_margin = margin
                best_id = cand.id
        retreat_improves = best_margin is not None and best_margin > current_margin

    if best_choice is not None:
        _, aid, discard_target, is_ko, is_win = best_choice
        commit = True
        # Only skip a *committed* attack for retreat when it's a knockout
        # that's a bad trade (we're risking a pricier Pokemon than the one
        # we'd KO) -- ordinary chip damage is still worth doing even while
        # exposed, since passing up all offense guarantees we never catch up.
        if is_ko and not is_win and we_are_exposed and retreat_improves:
            if prize_value(my_active.id) > prize_value(op_active.id):
                commit = False
        if commit:
            plan.attack_id = aid
            plan.discard_target = discard_target

    if plan.attack_id is None and we_are_exposed and retreat_improves:
        plan.retreat_wanted = True
        plan.retreat_to_id = best_id


def _pokemon_field_priority(card_id: int) -> int:
    if card_id == JOLTEON_EX:
        return 100
    if card_id == EEVEE:
        return 70
    if card_id in AMMO_TANKS:
        return 40
    return 10


def _score_option(obs, o, state, select, context, my_index, my_state, op_state,
                   my_active, op_active, field_counts, hand_counts, discard_counts,
                   op_prize_left, my_prize_left) -> float:
    score = 0

    if o.type == OptionType.NUMBER:
        score = o.number

    elif o.type == OptionType.YES:
        if context == SelectContext.IS_FIRST:
            score = -1  # prefer going second for the extra draw; free retreat covers early tempo
        else:
            score = 1

    elif o.type == OptionType.NO:
        score = 0

    elif o.type == OptionType.CARD:
        card = get_card(obs, o.area, o.index, o.playerIndex)
        if card is None:
            score = -1
        else:
            is_mine = (o.playerIndex == my_index)
            energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
            hp = card.hp if isinstance(card, Pokemon) else 0

            if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
                if is_mine:
                    if plan.retreat_to_id is not None and card.id == plan.retreat_to_id:
                        score = 90000
                    elif card.id == JOLTEON_EX:
                        score = 50000 + energy_count * 100
                    elif card.id == EEVEE:
                        score = 30000 + energy_count * 100
                    else:
                        score = 10000 + energy_count * 100
                else:
                    # choosing opponent's Pokemon, e.g. Boss's Orders target:
                    # prefer their weakest-HP / most-invested Pokemon (easiest KO, best value)
                    score = 20000 - hp + energy_count * 300 + len(card.tools) * 200

            elif context == SelectContext.SETUP_BENCH_POKEMON:
                score = _pokemon_field_priority(card.id) if card.id in BASICS else -1

            elif context in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                if card.id not in BASICS:
                    score = -1
                elif card.id == EEVEE:
                    score = 50000 if field_counts[JOLTEON_EX] + field_counts[EEVEE] < 2 else 20000
                else:
                    score = 30000

            elif context == SelectContext.TO_HAND:
                score = _search_card_score(card.id, field_counts, hand_counts, discard_counts)

            elif context == SelectContext.DISCARD:
                # Hand discard (e.g. for Ultra Ball / Carmine forced discard): give up
                # the least useful card. Higher "keep" value -> lower discard score.
                score = -_search_card_score(card.id, field_counts, hand_counts, discard_counts)

            elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                if hp > 0:
                    score = 100000 - hp * 10

            elif context == SelectContext.ATTACH_FROM:
                score = _attach_target_score(card, o.area == AreaType.ACTIVE, my_state)

            elif context == SelectContext.EVOLVES_FROM:
                score = 1000 + energy_count * 200 + hp

            elif context == SelectContext.EVOLVES_TO:
                score = 50000 if card.id == JOLTEON_EX else 100

            elif context in (SelectContext.LOOK, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM,
                              SelectContext.NOT_MOVE, SelectContext.TO_PRIZE):
                score = _search_card_score(card.id, field_counts, hand_counts, discard_counts)

            else:
                score = _search_card_score(card.id, field_counts, hand_counts, discard_counts)

    elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
        # For these option types, area/index identify the Pokemon *holding* the
        # energy/tool, not a standalone card -- see cg.api's OptionType docstring.
        pokemon = None
        if o.area in (AreaType.ACTIVE, AreaType.BENCH):
            holder_player = o.playerIndex if o.playerIndex is not None else my_index
            pokemon = get_card(obs, o.area, o.index, holder_player)
        pid = pokemon.id if pokemon is not None else None
        # Bench-energy discard for Flashing Spear (SelectContext.DISCARD_ENERGY): prefer
        # discarding from ammo tanks first, then Eevee, keep Jolteon's own energy last.
        if context == SelectContext.DISCARD_ENERGY:
            if pid in AMMO_TANKS:
                score = 300
            elif pid == EEVEE:
                score = 200
            elif pid == JOLTEON_EX:
                score = 100
            else:
                score = 150
        else:
            score = 10

    elif o.type == OptionType.PLAY:
        card = get_card(obs, AreaType.HAND, o.index, my_index)
        data = card_table.get(card.id)
        if data is not None and data.cardType == CardType.POKEMON:
            score = _play_pokemon_score(card.id, field_counts, my_state)
        else:
            score = _play_trainer_score(card.id, state, my_state, op_state, hand_counts, discard_counts, field_counts)

    elif o.type == OptionType.ATTACH:
        card = get_card(obs, o.area, o.index, my_index)
        pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
        score = _attach_target_score(pokemon, o.inPlayArea == AreaType.ACTIVE, my_state, card)

    elif o.type == OptionType.EVOLVE:
        pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
        card = get_card(obs, o.area, o.index, my_index)
        score = 95000 + len(pokemon.energies) * 50 if card.id == JOLTEON_EX else 60000

    elif o.type == OptionType.ABILITY:
        score = 40000

    elif o.type == OptionType.DISCARD:
        score = -1

    elif o.type == OptionType.RETREAT:
        if plan.retreat_wanted:
            score = 85000
        else:
            score = -1

    elif o.type == OptionType.ATTACK:
        if plan.attack_id is not None and o.attackId == plan.attack_id:
            score = 500000
        else:
            score = 100

    elif o.type == OptionType.END:
        score = 0

    elif o.type == OptionType.SPECIAL_CONDITION:
        score = 0

    else:
        score = 0

    return score


def _search_card_score(card_id: int, field_counts, hand_counts, discard_counts) -> int:
    if card_id == JOLTEON_EX:
        return 90000 if field_counts[EEVEE] + field_counts[JOLTEON_EX] > 0 else 80000
    if card_id == EEVEE:
        return 85000 if field_counts[EEVEE] + field_counts[JOLTEON_EX] < 2 else 40000
    if card_id in AMMO_TANKS:
        return 30000
    if card_id in BASIC_ENERGY_IDS:
        return 25000 if card_id == BASIC_L else 8000
    if card_id in (BUDDY_BUDDY_POFFIN, ULTRA_BALL, POKE_PAD):
        return 20000
    if card_id == HILDA:
        return 22000
    if card_id in (CHEREN, CARMINE):
        return 18000
    if card_id == BOSS_ORDERS:
        return 16000
    if card_id == NIGHT_STRETCHER:
        return 15000
    if card_id == ENERGY_SEARCH_PRO:
        return 24000
    if card_id == SWITCH:
        return 5000
    return 1000


def _attach_target_score(pokemon: Pokemon, is_active: bool, my_state, card=None) -> int:
    if pokemon is None:
        return -1
    if pokemon.id not in ({EEVEE, JOLTEON_EX} | AMMO_TANKS):
        return -1

    if card is not None:
        data = card_table.get(card.id)
        if data is not None and data.cardType not in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            return -1

    energy_count = len(pokemon.energies)

    # Only one manual Energy attachment happens per turn, so getting it onto a
    # combat-ready attacker is usually more valuable than any single routine
    # trainer play -- outrank the generic ~20000-34000 trainer band whenever
    # our attack line still needs its combat energy.
    if pokemon.id == JOLTEON_EX:
        # Flashing Spear wants exactly {L}+1 (2 energy); extra copies on Jolteon
        # itself don't feed the discard bonus (that comes from the Bench), so
        # once it's loaded, deprioritize hard.
        if energy_count < 2:
            score = 75000 if is_active else 55000
        else:
            score = 300
    elif pokemon.id == EEVEE:
        # Energy attached now carries over through evolution into Jolteon ex,
        # so loading an Eevee early is never wasted.
        if energy_count < 2:
            score = 65000 if is_active else 45000
        else:
            score = 300
    else:  # ammo tanks: happy to stockpile Basic Energy as Flashing Spear fuel,
        # but only once our real attacker's own requirement is already met.
        score = 500 + energy_count * 50
        if is_active:
            score -= 5000  # never want an ammo tank stuck active

    return score


def _play_pokemon_score(card_id: int, field_counts, my_state) -> int:
    bench_open = len(my_state.bench) < my_state.benchMax if hasattr(my_state, "benchMax") else True
    if not bench_open:
        return -1
    if card_id == JOLTEON_EX:
        return -1  # Jolteon ex only ever comes in via Evolve, never played as a Basic
    return 20000 + _pokemon_field_priority(card_id)


def _play_trainer_score(card_id, state, my_state, op_state, hand_counts, discard_counts, field_counts) -> int:
    if state.supporterPlayed:
        supporter_ids = {HILDA, CHEREN, CARMINE, BOSS_ORDERS}
        if card_id in supporter_ids:
            return -1

    if card_id == BUDDY_BUDDY_POFFIN:
        return 34000 if len(my_state.bench) < 5 else 5000
    if card_id == POKE_PAD:
        return 32000 if field_counts[EEVEE] + field_counts[JOLTEON_EX] < 4 else 10000
    if card_id == ULTRA_BALL:
        if len(my_state.hand) >= 3 and field_counts[JOLTEON_EX] + field_counts[EEVEE] < 6:
            return 31000
        return 4000
    if card_id == HILDA:
        return 30000 if field_counts[EEVEE] > 0 and field_counts[JOLTEON_EX] == 0 else 15000
    if card_id == ENERGY_SEARCH_PRO:
        return 28000
    if card_id == CHEREN:
        return 27000 if len(my_state.hand) <= 4 else 3000
    if card_id == CARMINE:
        return 26000 if len(my_state.hand) <= 3 else 2000
    if card_id == BOSS_ORDERS:
        return 24000 if len(op_state.bench) > 0 else -1
    if card_id == NIGHT_STRETCHER:
        has_target = discard_counts[JOLTEON_EX] > 0 or discard_counts[EEVEE] > 0 or any(
            discard_counts[e] > 0 for e in BASIC_ENERGY_IDS
        )
        return 21000 if has_target else -1
    if card_id == SWITCH:
        return 23000 if plan.retreat_wanted else -1
    return 5000
