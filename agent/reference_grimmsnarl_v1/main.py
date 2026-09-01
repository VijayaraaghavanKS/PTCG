import os
from collections import defaultdict, Counter

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
                     Card, Pokemon, all_card_data, all_attack, to_observation_class)

"""
reference_grimmsnarl_v1 -- QUICK, rough reference build of the real
"Marnie's Grimmsnarl ex control" archetype (13.1% of our real ladder
opponent field per the 2026-08-12 opponent-mining pass, never previously
modeled locally). NOT tuned for competitiveness -- built only as a
reasonable-faith local sparring partner so bc_v2 (pure behavior-cloning
agent) can be tested against an opponent shape it has never seen before,
instead of only our original 3 fast-aggro references.

Deck: Marnie's Impidimp/Morgrem/Grimmsnarl ex line + Fezandipiti ex as a
zero-setup secondary attacker. Grimmsnarl ex's "Punk Up" ability fires when
it's played from hand TO EVOLVE a Pokemon (including via Rare Candy) --
search up to 5 Basic {D} Energy from deck and attach them all at once. This
makes the deck deceptively fast despite being Stage 2: turn 1 Impidimp,
turn 2 Rare Candy straight to Grimmsnarl ex triggers Punk Up (up to 5 free
energy), then Shadow Bullet (180 dmg + 30 bench snipe, only costs {D}{D})
is live immediately. Real opponent-mining data shows we beat this archetype
63.6% of the time already with our existing decks -- it is a genuinely
strong, fast real deck, not a strawman.

Structure follows this project's established staged-heuristic convention
(see ga_v1/main.py): forced/trivial handling, then a per-turn `compute_plan`
pass (attacker/attack/target selection), then a flat per-option scorer that
mostly just checks "does this option serve the plan".
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

Impidimp = 646
Morgrem = 647
Grimmsnarl_ex = 648
Fezandipiti_ex = 140
Rare_Candy = 1079
Ultra_Ball = 1121
Buddy_Poffin = 1086
Night_Stretcher = 1097
Boss_Orders = 1182
Carmine = 1192
Hilda = 1225
Dawn = 1231
Judge = 1213
Switch = 1123
Precious_Trolley = 1126
Basic_D_Energy = 7

MAIN_LINE = (Impidimp, Morgrem, Grimmsnarl_ex)
DRAW_SUPPORTERS = (Carmine, Hilda, Dawn)


def get_card(obs: Observation, area, index, player_index):
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


def prize_value(pokemon_id: int) -> int:
    data = card_table.get(pokemon_id)
    if data is None:
        return 1
    if data.megaEx:
        return 3
    if data.ex:
        return 2
    return 1


def energy_ready(pokemon: Pokemon, attack) -> tuple:
    req = Counter(attack.energies)
    have = Counter(pokemon.energies)
    colorless_needed = req.get(EnergyType.COLORLESS, 0)
    missing = 0
    leftover_have = dict(have)
    for t, c in req.items():
        if t == EnergyType.COLORLESS:
            continue
        h = leftover_have.get(t, 0)
        use = min(h, c)
        leftover_have[t] = h - use
        missing += (c - use)
    if colorless_needed:
        leftover = sum(leftover_have.values())
        use = min(leftover, colorless_needed)
        missing += colorless_needed - use
    return (missing == 0, missing == 1)


def effective_damage(attacker: Pokemon, attack, target: Pokemon) -> int:
    dmg = attack.damage or 0
    tdata = card_table.get(target.id)
    if tdata is None:
        return dmg
    adata = card_table.get(attacker.id)
    atype = adata.energyType if adata is not None else None
    if tdata.weakness is not None and tdata.weakness == atype:
        dmg *= 2
    elif tdata.resistance is not None and tdata.resistance == atype:
        dmg = max(0, dmg - 30)
    return dmg


def opponent_best_damage_estimate(op_active: Pokemon) -> int:
    if op_active is None:
        return 0
    data = card_table.get(op_active.id)
    if data is None:
        return 0
    best = 0
    have = len(op_active.energies)
    for aid in data.attacks:
        atk = attack_table.get(aid)
        if atk is None:
            continue
        req = len(atk.energies)
        if have + 1 >= req:
            best = max(best, atk.damage or 0)
    return best


class Plan:
    attacker = -1
    attack_id = -1
    target_active = True
    target_bench_index = -1
    needs_energy = False
    wins_game = False
    value = -1.0
    use_gust = False
    danger = False


plan = Plan()


def compute_plan(obs, state, select, my_index, my_state, op_state, hand_counts):
    global plan
    plan = Plan()

    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None
    if my_active is None or op_active is None:
        return

    threat = opponent_best_damage_estimate(op_active)
    if my_active.hp <= threat:
        plan.danger = True

    can_reposition = (not state.retreated)
    my_cards = [(0, my_active)] + [(i + 1, p) for i, p in enumerate(my_state.bench)]
    op_bench = list(op_state.bench)
    has_boss = hand_counts[Boss_Orders] > 0

    best_value = -1.0
    for idx, pokemon in my_cards:
        if idx != 0 and not can_reposition:
            continue
        data = card_table.get(pokemon.id)
        if data is None:
            continue
        for aid in data.attacks:
            atk = attack_table.get(aid)
            if atk is None or (atk.damage or 0) <= 0:
                continue
            ready, needs_one = energy_ready(pokemon, atk)
            if not ready and not (needs_one and not state.energyAttached and hand_counts[Basic_D_Energy] > 0):
                continue

            targets = [(True, -1, op_active)]
            if has_boss:
                for bi, bp in enumerate(op_bench):
                    targets.append((False, bi, bp))

            for is_active_target, bench_idx, target in targets:
                dmg = effective_damage(pokemon, atk, target)
                is_ko = dmg >= target.hp
                pv = prize_value(target.id)
                if is_ko:
                    value = 300.0 * pv
                    wins = is_ko and len(op_state.prize) <= pv
                else:
                    value = 40.0 * min(1.0, dmg / max(1, target.hp))
                    wins = False
                value += 15.0 * len(pokemon.energies)
                if idx == 0:
                    value += 30.0
                if not is_active_target:
                    value += 250.0 * (pv / 3.0)
                if wins:
                    value = 1e9

                if value > best_value:
                    best_value = value
                    plan.attacker = idx
                    plan.attack_id = aid
                    plan.target_active = is_active_target
                    plan.target_bench_index = bench_idx
                    plan.needs_energy = needs_one and not ready
                    plan.wins_game = wins
                    plan.use_gust = not is_active_target
                    plan.value = value


def _hand_target_score(card_id, field_counts, hand_counts):
    if card_id == Impidimp:
        if field_counts[Impidimp] + field_counts[Morgrem] + field_counts[Grimmsnarl_ex] >= 4:
            return 100.0
        return 3500.0
    if card_id == Morgrem:
        if field_counts[Impidimp] >= 1:
            return 800.0  # normal-evolve backup; Rare Candy path preferred
        return 100.0
    if card_id == Grimmsnarl_ex:
        if field_counts[Impidimp] >= 1 or field_counts[Morgrem] >= 1:
            return 5000.0  # Punk Up payoff -- highest priority once a base is down
        return 200.0
    if card_id == Fezandipiti_ex:
        if field_counts[Fezandipiti_ex] >= 1:
            return 150.0
        return 2200.0
    if card_id == Rare_Candy:
        return 4800.0  # gated further at PLAY-time by whether it can fire
    if card_id == Basic_D_Energy:
        return 1900.0
    if card_id in (Ultra_Ball, Buddy_Poffin):
        return 2000.0
    if card_id in DRAW_SUPPORTERS:
        return 2100.0
    if card_id == Boss_Orders:
        return 1950.0
    if card_id == Night_Stretcher:
        return 1500.0
    if card_id == Judge:
        return 900.0
    if card_id == Switch:
        return 1200.0
    if card_id == Precious_Trolley:
        return 1800.0
    return 500.0


def _attach_score(pokemon, active):
    if pokemon is None:
        return -1.0
    if pokemon.id not in (Impidimp, Morgrem, Grimmsnarl_ex, Fezandipiti_ex):
        return -1.0
    data = card_table.get(pokemon.id)
    if data is None:
        return -1.0
    max_req = 0
    for aid in data.attacks:
        atk = attack_table.get(aid)
        if atk is not None:
            max_req = max(max_req, len(atk.energies))
    have = len(pokemon.energies)
    if have >= max_req and max_req > 0:
        return 100.0
    score = 8000.0 + 60.0 * (have + 1)
    if active:
        score += 20.0
    return score


def _play_score(card_id, data, field_counts, hand_counts, state, my_state):
    if data is None:
        return -1.0
    if data.cardType == CardType.POKEMON:
        if card_id == Morgrem and field_counts[Impidimp] < 1:
            return -1.0
        if card_id == Grimmsnarl_ex and field_counts[Impidimp] < 1 and field_counts[Morgrem] < 1:
            return -1.0
        return 20000.0
    if data.cardType == CardType.SUPPORTER and state.supporterPlayed:
        return -1.0

    if card_id == Rare_Candy:
        # Only useful if we have an eligible Basic in play this-not-turn and Grimmsnarl ex in hand.
        if field_counts[Impidimp] >= 1 and hand_counts[Grimmsnarl_ex] >= 1:
            return 9500.0  # top priority: triggers Punk Up
        return -1.0
    if card_id == Switch:
        if plan.attacker >= 1:
            return 6000.0
        return -1.0
    if card_id == Boss_Orders:
        if plan.use_gust:
            return 6200.0
        return -1.0
    if card_id in DRAW_SUPPORTERS:
        hand_size = len(my_state.hand or [])
        need = max(0.0, 5.0 - hand_size)
        return 5000.0 + 50.0 * need
    if card_id in (Ultra_Ball, Buddy_Poffin):
        return 4800.0
    if card_id == Judge:
        return 800.0
    if card_id == Night_Stretcher:
        return 4500.0
    if card_id == Precious_Trolley:
        return 4700.0
    return 3000.0


def agent(obs_dict: dict) -> list:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    for c in ([my_state.active[0]] if my_state.active and my_state.active[0] else []) + list(my_state.bench):
        field_counts[c.id] += 1
    for c in my_state.hand:
        hand_counts[c.id] += 1

    if context == SelectContext.MAIN:
        compute_plan(obs, state, select, my_index, my_state, op_state, hand_counts)

    my_active = my_state.active[0] if my_state.active else None

    scores = []
    for o in select.option:
        score = 0.0
        if o.type == OptionType.NUMBER:
            score = float(o.number)
        elif o.type == OptionType.YES:
            score = -1.0 if context == SelectContext.IS_FIRST else 1.0
        elif o.type == OptionType.NO:
            score = 0.0
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                hp = card.hp if isinstance(card, Pokemon) else 0
                maxhp = card.maxHp if isinstance(card, Pokemon) else 1
                hp_frac = (hp / maxhp) if isinstance(card, Pokemon) and maxhp > 0 else 0.0

                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if o.playerIndex == my_index:
                        score = 100.0 + 15.0 * energy_count + 80.0 * hp_frac
                        if plan.attacker >= 1 and (plan.attacker - 1) == o.index:
                            score += 500.0
                    else:
                        score = 250.0 * (prize_value(card.id) / 3.0) + (100.0 - hp_frac * 100.0)
                        if plan.use_gust and plan.target_bench_index == o.index:
                            score += 1000.0
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    score = 40.0 if card.id == Impidimp else (30.0 if card.id == Fezandipiti_ex else 5.0)
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 25.0
                    if card.id in (Impidimp, Fezandipiti_ex):
                        score += 20.0
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.LOOK):
                    score = _hand_target_score(card.id, field_counts, hand_counts)
                elif context == SelectContext.DISCARD:
                    score = -_hand_target_score(card.id, field_counts, hand_counts) * 0.5
                elif context == SelectContext.ATTACH_FROM:
                    score = _attach_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.EVOLVES_FROM:
                    score = 100.0 + 15.0 * energy_count + hp_frac * 20.0
                elif context == SelectContext.EVOLVES_TO:
                    bonus = 5000.0 if card.id == Grimmsnarl_ex else 50.0
                    score = bonus
                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if hp > 0:
                        score = 1000.0 - hp
                else:
                    score = _hand_target_score(card.id, field_counts, hand_counts)
        elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY, OptionType.TOOL_CARD):
            score = 1.0 if o.playerIndex != my_index else 0.5
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table.get(card.id)
            score = _play_score(card.id, data, field_counts, hand_counts, state, my_state)
        elif o.type == OptionType.ATTACH:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = _attach_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
            is_planned_target = (
                (plan.attacker == 0 and o.inPlayArea == AreaType.ACTIVE) or
                (plan.attacker >= 1 and o.inPlayArea == AreaType.BENCH and o.inPlayIndex == plan.attacker - 1)
            )
            if plan.needs_energy and is_planned_target:
                score += 5000.0
        elif o.type == OptionType.EVOLVE:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            bonus = 3000.0 if (card is not None and card.id == Grimmsnarl_ex) else 0.0
            score = 9000.0 + bonus
            if pokemon is not None:
                score += 15.0 * len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            score = 26000.0
        elif o.type == OptionType.DISCARD:
            score = 0.0
        elif o.type == OptionType.RETREAT:
            hp_frac = (my_active.hp / my_active.maxHp) if (my_active is not None and my_active.maxHp > 0) else 1.0
            score = -1.0
            if plan.danger:
                score = 90.0 * (1.5 - hp_frac)
            if plan.attacker >= 1:
                score = max(score, 2000.0)
        elif o.type == OptionType.ATTACK:
            if plan.wins_game and o.attackId == plan.attack_id:
                score = 1e9
            elif o.attackId == plan.attack_id and not plan.needs_energy:
                score = 1000.0 + max(0.0, plan.value) * 0.01
            else:
                score = 200.0
        elif o.type == OptionType.END:
            score = -5.0

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    out = desc_indices[:select.maxCount]
    while len(out) < select.minCount:
        for i in desc_indices:
            if i not in out:
                out.append(i)
                break
        else:
            break
    return out
