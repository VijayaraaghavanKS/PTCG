import os
from collections import defaultdict

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
    Card, Pokemon, all_card_data, all_attack, to_observation_class,
)

"""
Inteleon 1-Prize Attacker Deck (Water)
---------------------------------------
Win condition: rush the Sobble -> Drizzile -> Inteleon line (Rare Candy skips
Drizzile whenever possible) and lean on Inteleon's Water Shot ({W}, 110 dmg,
then discards the attached Energy) for repeated high-damage attacks, backed
by Bring Down ({W}, no listed damage: automatically KOs whatever Pokemon in
play -- either side -- currently has the least remaining HP, excluding
Inteleon itself) to snipe already-weak targets for free without losing the
attached Energy. Since Inteleon only costs the opponent 1 Prize when it dies,
we can afford to trade down and still win the Prize race.

Because Water Shot discards its own Energy, every turn effectively resets
Inteleon back to 0 Energy: the recurring pattern is "attach a fresh Basic
{W} Energy, then attack" -- so the deck leans hard on Trainers that either
find more Water Energy (Hilda) or recur discarded ones (Night Stretcher).

Deck consistency (mitigating the classic Stage-2 speed problem):
  - 4x Rare Candy: skip Drizzile entirely once a Sobble has survived to our
    second turn.
  - 4x Buddy-Buddy Poffin + 1x Precious Trolley (our single ACE SPEC slot):
    both put Basic Pokemon (Sobble, HP 70 <= the Poffin's 70 HP cap) straight
    onto the bench without passing through hand, so they dodge hand-size
    pressure entirely. Precious Trolley in particular can flood the bench
    with every remaining Sobble in one card.
  - 4x Ultra Ball / 4x Poke Pad / 4x Dawn / 4x Hilda: four different ways to
    fish for whichever evolution-line piece (or Energy) is currently
    missing. Dawn is a near-perfect fit since it searches exactly
    Basic + Stage 1 + Stage 2 in one card -- our exact evolution line.
  - 4x Night Stretcher: recycles discarded Water Energy (which Water Shot
    keeps consuming) or a KO'd evolution-line Pokemon back to hand.
  - 4x Boss's Orders: gust up a weak/juicy bench target for Bring Down or a
    lethal Water Shot.
  - 3x Lillie's Determination: emergency hand refresh if the opening draws
    are dead.

This is a from-scratch, clean-room decision policy: for every legal option
the engine offers we compute an independent heuristic score (favoring lethal
attacks, then safe Bring Down snipes, then loss-shielding retreats, then
general evolution/consistency progress), and submit the highest-scoring
option(s). No code was copied from any other agent in this repository.
"""

# ---------------------------------------------------------------------------
# Load our deck (list of 60 Card IDs, one per line, no header).
# ---------------------------------------------------------------------------
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    _csv_lines = file.read().split("\n")
my_deck = [int(_csv_lines[i]) for i in range(60)]

# ---------------------------------------------------------------------------
# Static reference data straight from the engine (never guessed/hardcoded
# damage numbers -- attack IDs/damage come from all_attack(), card flags
# such as ex/megaEx/basic/stage1/stage2 come from all_card_data()).
# ---------------------------------------------------------------------------
card_table = {c.cardId: c for c in all_card_data()}
attack_table = {a.attackId: a for a in all_attack()}
_attack_id_by_name = {a.name: a.attackId for a in all_attack()}

WATER_SHOT_ID = _attack_id_by_name.get("Water Shot")
BRING_DOWN_ID = _attack_id_by_name.get("Bring Down")
DOUBLE_STAB_ID = _attack_id_by_name.get("Double Stab")
SURPRISE_ATTACK_ID = _attack_id_by_name.get("Surprise Attack")

# Decklist (Card IDs verified against EN Card Data.csv via CardTable.name_to_ids)
SOBBLE = 726                    # x4  Basic, 70 HP, Surprise Attack {W}/30 (coin flip)
DRIZZILE = 727                  # x4  Stage 1, 100 HP, Double Stab {W}/30x (2 coins)
INTELEON = 728                  # x4  Stage 2, 150 HP, Water Shot {W}/110 (discards its own
                                 #     Energy after use) + Bring Down {W} (auto-KO lowest HP)
RARE_CANDY = 1079                # x4  skip Drizzile, evolve Sobble straight to Inteleon
BUDDY_BUDDY_POFFIN = 1086        # x4  fetch up to 2 Basic <=70HP straight to bench
ULTRA_BALL = 1121                # x4  discard 2, search any Pokemon to hand
POKE_PAD = 1152                  # x4  search a non-Rule-Box Pokemon to hand
NIGHT_STRETCHER = 1097           # x4  discard-pile Pokemon or Basic Energy -> hand
PRECIOUS_TROLLEY = 1126          # x1  ACE SPEC: search any number of Basic Pokemon -> bench
DAWN = 1231                      # x4  search Basic + Stage1 + Stage2 Pokemon -> hand
HILDA = 1225                     # x4  search an Evolution Pokemon + an Energy card -> hand
BOSS_ORDERS = 1182               # x4  gust an opponent's benched Pokemon to Active
LILLIES_DETERMINATION = 1227     # x3  shuffle hand into deck, draw 6 (8 at exactly 6 prizes)
BASIC_WATER_ENERGY = 3           # x12 Basic {W} Energy

EVOLUTION_LINE = (SOBBLE, DRIZZILE, INTELEON)


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    """Safely resolve a Card/Pokemon object out of a given zone."""
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index] if obs.select.deck is not None else None
    if area == AreaType.HAND:
        return ps.hand[index] if ps.hand is not None else None
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
        return obs.current.looking[index] if obs.current.looking is not None else None
    return None


def in_play(player_state) -> list:
    """Active + bench Pokemon actually on the field for a given PlayerState."""
    pk = []
    if player_state.active and player_state.active[0] is not None:
        pk.append(player_state.active[0])
    for p in player_state.bench:
        if p is not None:
            pk.append(p)
    return pk


def prize_value(card_id: int) -> int:
    data = card_table.get(card_id)
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def effective_damage(base_damage: int, defender_id: int) -> int:
    """Apply the defender's Weakness/Resistance to a {W} attack's base damage."""
    data = card_table.get(defender_id)
    dmg = base_damage
    if data is not None:
        if data.weakness == EnergyType.WATER:
            dmg *= 2
        elif data.resistance == EnergyType.WATER:
            dmg = max(0, dmg - 30)
    return dmg


def target_value(pokemon: Pokemon) -> int:
    """How juicy a KO/gust target this Pokemon is -- more invested & more Prizes = better."""
    data = card_table.get(pokemon.id)
    value = prize_value(pokemon.id) * 1000
    value += len(pokemon.energies) * 120
    value += len(pokemon.tools) * 80
    if data is not None:
        if data.stage2:
            value += 200
        elif data.stage1:
            value += 100
    value += pokemon.hp
    return value


# ---------------------------------------------------------------------------
# Main decision policy
# ---------------------------------------------------------------------------

def agent(obs_dict: dict) -> list[int]:
    """Each returned index must be >= 0 and < len(obs.select.option); the list
    length must be between obs.select.minCount and obs.select.maxCount
    inclusive with no duplicates. Defensive fallback below guarantees this
    even if anything above throws.
    """
    try:
        return _decide(obs_dict)
    except Exception:
        return _fallback(obs_dict)


def _fallback(obs_dict: dict) -> list[int]:
    """Last-resort legal move: never crash, always return something the
    engine will accept."""
    select = obs_dict.get("select")
    if select is None:
        return my_deck
    options = select.get("option", []) or []
    mn = max(int(select.get("minCount", 0) or 0), 0)
    mx = select.get("maxCount", mn)
    mx = mn if mx is None else int(mx)
    count = min(max(mn, 0), mx, len(options))
    return list(range(count))


def _decide(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        # Initial deck selection.
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None

    field_counts = defaultdict(int)
    for p in in_play(my_state):
        field_counts[p.id] += 1

    hand_counts = defaultdict(int)
    for c in (my_state.hand or []):
        hand_counts[c.id] += 1

    discard_counts = defaultdict(int)
    for c in my_state.discard:
        discard_counts[c.id] += 1

    my_first_turn = 1 if my_index == 0 else 2

    # -----------------------------------------------------------------
    # Stage: evolution-line search/fetch scoring, shared across many
    # different SelectContexts (TO_HAND, TO_BENCH, LOOK, SETUP, ...).
    # -----------------------------------------------------------------
    def pokemon_fetch_score(card_id: int) -> int:
        if card_id == INTELEON:
            have_ready_basic = field_counts[SOBBLE] > 0 or field_counts[DRIZZILE] > 0
            base = 9000 if have_ready_basic else 4000
            return base - hand_counts[INTELEON] * 1500
        if card_id == DRIZZILE:
            base = 6000 if field_counts[SOBBLE] > 0 else 2500
            return base - hand_counts[DRIZZILE] * 1200
        if card_id == SOBBLE:
            line_in_play = field_counts[SOBBLE] + field_counts[DRIZZILE] + field_counts[INTELEON]
            return 8500 if line_in_play == 0 else max(500, 5000 - line_in_play * 1500)
        return 100  # not part of our plan; shouldn't normally occur (mono-line deck)

    def energy_fetch_score(card_id: int) -> int:
        return 7000 if card_id == BASIC_WATER_ENERGY else 0

    def pick_score(card_id: int) -> int:
        data = card_table.get(card_id)
        if data is not None and data.cardType == CardType.POKEMON:
            return pokemon_fetch_score(card_id)
        if card_id == BASIC_WATER_ENERGY:
            return energy_fetch_score(card_id)
        return 50

    def discard_score(card_id: int) -> int:
        """Used both for Ultra Ball's discard-2 cost and OptionType.DISCARD.
        Higher = more willing to part with this card."""
        if card_id == BASIC_WATER_ENERGY:
            # Cheapest thing to discard: Night Stretcher brings it right back,
            # and we usually hold more copies than we need at once anyway.
            return 9000 - hand_counts[BASIC_WATER_ENERGY] * 10
        if card_id in EVOLUTION_LINE:
            # Strongly avoid discarding our own evolution pieces unless we're
            # already holding redundant extra copies.
            return -5000 + hand_counts[card_id] * 500
        return 3000  # spare Trainers are the next most disposable

    def evolve_to_score(card_id: int) -> int:
        if card_id == INTELEON:
            return 90000
        if card_id == DRIZZILE:
            return 80000
        return 100

    def bench_priority(pokemon: Pokemon) -> int:
        score = 0
        if pokemon.id == INTELEON:
            score += 5000
        elif pokemon.id == DRIZZILE:
            score += 3000
        elif pokemon.id == SOBBLE:
            score += 1000
        score += len(pokemon.energies) * 300
        score += pokemon.hp
        return score

    def attach_score(pokemon: Pokemon, is_active: bool) -> int:
        if pokemon.id not in EVOLUTION_LINE:
            return -1
        if len(pokemon.energies) > 0:
            return -1  # every attack in our line only ever needs exactly 1 Energy
        score = 10000
        if is_active:
            score += 500
        if pokemon.id == INTELEON:
            score += 300
        elif pokemon.id == DRIZZILE:
            score += 150
        return score

    def retreat_score() -> int:
        if my_active is None:
            return -1
        has_backup = any(p is not None for p in my_state.bench)
        if not has_backup:
            return -1
        in_danger = my_active.maxHp > 0 and my_active.hp <= max(1, my_active.maxHp * 0.35)
        if in_danger:
            return 8000  # loss-shielding: pull a nearly-dead attacker (esp. an
                          # Energy-loaded Inteleon) out of the kill zone
        # No immediate danger: still worth swapping a pre-evolution attacker
        # out for a ready Inteleon sitting on the bench.
        if my_active.id in (SOBBLE, DRIZZILE) and any(
            p is not None and p.id == INTELEON and len(p.energies) > 0 for p in my_state.bench
        ):
            return 6000
        return -1

    def best_boss_target():
        candidates = [p for p in op_state.bench if p is not None]
        if not candidates:
            return None
        return max(candidates, key=target_value)

    def boss_orders_worth_it() -> bool:
        tgt = best_boss_target()
        if tgt is None:
            return False
        if my_active is not None and my_active.id == INTELEON:
            energy_ready = len(my_active.energies) > 0 or (
                not state.energyAttached and hand_counts[BASIC_WATER_ENERGY] > 0
            )
            if energy_ready and effective_damage(110, tgt.id) >= tgt.hp:
                return True
        return target_value(tgt) >= 1300  # a Stage-2/energy-loaded mon is worth disrupting

    def bring_down_target():
        """Which opponent Pokemon Bring Down would remove, or None if it's
        not safe/worthwhile to fire this turn.

        Bring Down auto-KOs whichever Pokemon in play (either side, except
        the attacking Inteleon) currently has the least remaining HP. We
        only ever fire it when every Pokemon tied for that minimum HP is on
        the OPPONENT's side -- if our own board could be the one selected
        (uniquely, or tied with nothing of the opponent's at that HP), we
        refuse rather than risk self-KO-ing our own Pokemon for nothing.
        """
        if my_active is None:
            return None
        pool = []
        for pi, ps in enumerate(state.players):
            for p in in_play(ps):
                if p.serial == my_active.serial:
                    continue
                pool.append((pi, p))
        if not pool:
            return None
        min_hp = min(p.hp for _, p in pool)
        opp_candidates = [p for pi, p in pool if pi != my_index and p.hp == min_hp]
        if not opp_candidates:
            return None
        return max(opp_candidates, key=target_value)

    def attack_score(attack_id) -> int:
        if attack_id == WATER_SHOT_ID and my_active is not None and my_active.id == INTELEON:
            if op_active is None:
                return 100
            dmg = effective_damage(110, op_active.id)
            score = 5000 + dmg
            if dmg >= op_active.hp:
                score += 50000  # KO's their Active this turn
                if len(op_state.prize) <= prize_value(op_active.id):
                    score += 200000  # ... and that KO ends the game right now
            return score
        if attack_id == BRING_DOWN_ID and my_active is not None and my_active.id == INTELEON:
            victim = bring_down_target()
            if victim is None:
                return -1  # never risk self-sniping our own board for nothing
            score = 4000 + prize_value(victim.id) * 2000
            if len(op_state.prize) <= prize_value(victim.id):
                score += 200000  # automatic KO that ends the game right now
            return score
        if attack_id == DOUBLE_STAB_ID:
            return 2000  # Drizzile: 2 coins x 30, ~30 dmg expected
        if attack_id == SURPRISE_ATTACK_ID:
            return 1500  # Sobble: 50% chance of 30 dmg
        return 800

    def play_priority(card_id: int) -> int:
        data = card_table.get(card_id)
        is_supporter = data is not None and data.cardType == CardType.SUPPORTER
        if is_supporter and state.supporterPlayed:
            return -1

        if card_id == SOBBLE:
            line_in_play = field_counts[SOBBLE] + field_counts[DRIZZILE] + field_counts[INTELEON]
            return 40000 if line_in_play < my_state.benchMax else 5000

        if card_id == RARE_CANDY:
            have_target = any(
                p.id == SOBBLE and not p.appearThisTurn for p in in_play(my_state)
            )
            if have_target and hand_counts[INTELEON] > 0 and state.turn != my_first_turn:
                return 100000
            return -1

        if card_id == PRECIOUS_TROLLEY:
            already_committed = (
                field_counts[SOBBLE] + field_counts[DRIZZILE] + field_counts[INTELEON]
                + hand_counts[SOBBLE] + discard_counts[SOBBLE]
            )
            sobbles_left = 4 - already_committed
            bench_room = my_state.benchMax - len(my_state.bench)
            return 60000 if (sobbles_left > 0 and bench_room > 0) else -1

        if card_id == BUDDY_BUDDY_POFFIN:
            line_in_play = field_counts[SOBBLE] + field_counts[DRIZZILE] + field_counts[INTELEON]
            bench_room = my_state.benchMax - len(my_state.bench)
            return 55000 if (line_in_play < my_state.benchMax and bench_room > 0) else 1000

        if card_id == ULTRA_BALL:
            need = (
                hand_counts[INTELEON] == 0 and hand_counts[DRIZZILE] == 0
                and field_counts[INTELEON] == 0
            )
            return 42000 if (need and len(my_state.hand or []) >= 3) else 15000

        if card_id == POKE_PAD:
            return 41000

        if card_id == DAWN:
            return 45000

        if card_id == HILDA:
            return 44000

        if card_id == BOSS_ORDERS:
            return 39000 if boss_orders_worth_it() else -1

        if card_id == NIGHT_STRETCHER:
            needed = discard_counts[BASIC_WATER_ENERGY] > 0 or any(
                discard_counts[pid] > 0 for pid in EVOLUTION_LINE
            )
            return 30000 if needed else -1

        if card_id == LILLIES_DETERMINATION:
            weak_hand = len(my_state.hand or []) <= 3 or (
                hand_counts[SOBBLE] + hand_counts[DRIZZILE] + hand_counts[INTELEON]
                + field_counts[SOBBLE] + field_counts[DRIZZILE] + field_counts[INTELEON] == 0
            )
            return 20000 if weak_hand else 2000

        return 100

    # -----------------------------------------------------------------
    # Score every legal option.
    # -----------------------------------------------------------------
    scores = []
    for o in select.option:
        score = 0

        if o.type == OptionType.NUMBER:
            score = o.number if o.number is not None else 0

        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                # Decline: going second means an extra opening draw, which
                # matters more for a Stage-2 deck's consistency than the
                # tempo of attacking one turn earlier.
                score = -1
            elif context == SelectContext.MORE_DEVOLVE:
                score = -1  # never volunteer to devolve further
            else:
                score = 1

        elif o.type == OptionType.NO:
            score = 0

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                if context in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
                    score = pokemon_fetch_score(card.id) if isinstance(card, Pokemon) else 100
                elif context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if isinstance(card, Pokemon):
                        score = bench_priority(card) if o.playerIndex == my_index else target_value(card)
                elif context in (
                    SelectContext.TO_BENCH, SelectContext.TO_HAND, SelectContext.LOOK,
                    SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM, SelectContext.TO_FIELD,
                    SelectContext.NOT_MOVE,
                ):
                    score = pick_score(card.id)
                elif context == SelectContext.DISCARD:
                    score = discard_score(card.id)
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(card, o.area == AreaType.ACTIVE) if isinstance(card, Pokemon) else -1
                elif context == SelectContext.EVOLVES_FROM:
                    energy_bonus = len(card.energies) * 200 if isinstance(card, Pokemon) else 0
                    score = 1000 + energy_bonus
                elif context == SelectContext.EVOLVES_TO:
                    score = evolve_to_score(card.id)
                elif context in (SelectContext.DAMAGE, SelectContext.EFFECT_TARGET):
                    # Defensive fallback in case an effect (e.g. Bring Down)
                    # surfaces an explicit target choice instead of resolving
                    # automatically: never point it at our own board.
                    if isinstance(card, Pokemon):
                        score = target_value(card) if o.playerIndex != my_index else -1
                elif context in (
                    SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
                    SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER,
                ):
                    if isinstance(card, Pokemon):
                        score = (300 - card.hp) if o.playerIndex == my_index else card.hp
                elif context == SelectContext.DEVOLVE:
                    score = -1 if o.playerIndex == my_index else 100
                else:
                    score = pick_score(card.id) if isinstance(card, Pokemon) else 50

        elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY, OptionType.TOOL_CARD):
            score = 10  # our deck never needs to distinguish among attached energies/tools

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            score = play_priority(card.id) if card is not None else 0

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is not None and card.id == BASIC_WATER_ENERGY and isinstance(pokemon, Pokemon):
                score = attach_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
            else:
                score = -1

        elif o.type == OptionType.EVOLVE:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = evolve_to_score(card.id) if card is not None else 100
            if isinstance(pokemon, Pokemon):
                score += len(pokemon.energies) * 100

        elif o.type == OptionType.ABILITY:
            score = 100  # our deck has no Pokemon Abilities; harmless default

        elif o.type == OptionType.DISCARD:
            card = get_card(obs, o.area, o.index, my_index)
            score = discard_score(card.id) if card is not None else 0

        elif o.type == OptionType.RETREAT:
            score = retreat_score()

        elif o.type == OptionType.ATTACK:
            score = attack_score(o.attackId)

        elif o.type == OptionType.SKILL:
            score = 0

        elif o.type == OptionType.SPECIAL_CONDITION:
            score = 0

        elif o.type == OptionType.END:
            score = -100  # last resort: any real action should outscore passing

        else:
            score = 0

        scores.append(score)

    # -----------------------------------------------------------------
    # Resolve: take the top select.maxCount options by score. For
    # variable-count selections (minCount < maxCount) we stop adding once
    # we've satisfied minCount and the remaining candidates score negative
    # (i.e. "optional and not worth taking"), rather than always maxing out.
    # -----------------------------------------------------------------
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    output = []
    for rank in range(select.maxCount):
        idx = ranked[rank]
        optional_and_bad = (
            select.minCount < select.maxCount
            and rank >= select.minCount
            and scores[idx] < 0
        )
        if optional_and_bad:
            continue
        output.append(idx)
    return output
