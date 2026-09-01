import os
import json
from collections import defaultdict, Counter

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
                     Card, Pokemon, all_card_data, all_attack, to_observation_class)

"""
GA v1 - Mega Lucario ex Deck (same decklist as mega_lucario_ref, reused deliberately:
this build is a methodology prototype, not a deck-novelty experiment).

Clean-room implementation. Structure is staged like our other heuristic agents
(forced/trivial moves -> lethal/KO detection -> loss-shielding retreat logic ->
value-progress fallback scoring), but the fallback stage's numeric coefficients are
NOT hand-guessed in this file. They live in WEIGHTS, a flat vector loaded from an
external source (env var GA_WEIGHTS_FILE, else local weights.json, else the
DEFAULT_WEIGHTS baseline below). That external vector is what genetic_search.py
evolves against real self-play fitness.

The coarse "which tier of action beats which" ordering (e.g. always finish setup
actions before attacking, since attacking ends the turn) is treated as structural
game-flow knowledge and is fixed, matching the same convention our other agents use
for forced-move handling. Only the fine-grained preference *within* a tier is
GA-tunable.
"""

# ---------------------------------------------------------------------------
# Deck loading
# ---------------------------------------------------------------------------
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = [int(csv[i]) for i in range(60)]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
attack_table = {a.attackId: a for a in all_attack()}

# Decklist (identical to mega_lucario_ref/deck.csv)
Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Gravity_Mountain = 1252
Basic_Fighting_Energy = 6

COSMIC_BEAM = 980  # Solrock, needs Lunatone in play, ignores weak/resist

ATTACKER_LINE_1 = (Riolu, Mega_Lucario_ex)
ATTACKER_LINE_2 = (Makuhita, Hariyama)
ATTACKER_LINE_3 = (Lunatone, Solrock)
DRAW_SUPPORTERS = (Carmine, Lillie_Determination)
SEARCH_ITEMS = (Dusk_Ball, Fighting_Gong, Poke_Pad)

# ---------------------------------------------------------------------------
# Tunable weight vector (GA-evolved). Keys are the "genes".
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "damage_progress": 40.0,        # score per fraction of target HP dealt on a non-KO attack
    "ko_prize_value": 300.0,        # score per prize point earned on a KO'ing attack
    "energy_on_attacker": 15.0,     # score per energy already on the attacker chosen
    "active_attacker_bonus": 30.0,  # prefer attacking with the current active over switching in
    "gust_bonus": 250.0,            # value of using Boss's Orders to snipe a good bench target
    "energy_attach_readiness": 60.0,  # prefer attaching energy to whoever is closest to attack-ready
    "energy_attach_active_bonus": 20.0,  # prefer attaching to the active Pokemon
    "evolution_progress": 70.0,     # value of evolving (scaled by stage gained)
    "card_draw_value": 55.0,        # value of draw supporters, scaled by hand need
    "bench_development": 25.0,      # value of playing basics to the bench
    "search_value": 35.0,           # value of tutor/search items
    "retreat_danger_weight": 90.0,  # how strongly to retreat away from lethal danger
    "switch_target_health": 8.0,    # value per HP% when picking who becomes active
    "tool_hp_value": 20.0,          # value of attaching Hero's Cape
    "stadium_value": 10.0,          # value of playing Gravity Mountain
    "discard_synergy": 15.0,        # value of discard-for-value effects (Lunar Cycle etc.)
}

WEIGHT_KEYS = list(DEFAULT_WEIGHTS.keys())


def _load_weights() -> dict:
    w = dict(DEFAULT_WEIGHTS)
    paths = []
    env_path = os.environ.get("GA_WEIGHTS_FILE")
    if env_path:
        paths.append(env_path)
    paths.append("weights.json")
    for p in paths:
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    data = json.load(f)
                for k in WEIGHT_KEYS:
                    if k in data:
                        w[k] = float(data[k])
            except Exception:
                pass
            break
    return w


WEIGHTS = _load_weights()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def prize_value(pokemon_id: int) -> int:
    data = card_table.get(pokemon_id)
    if data is None:
        return 1
    if data.megaEx:
        return 3
    if data.ex:
        return 2
    return 1


def stage_value(card_id: int) -> int:
    data = card_table.get(card_id)
    if data is None:
        return 0
    if data.stage2:
        return 2
    if data.stage1:
        return 1
    return 0


class Plan:
    attacker = -1        # -1 = none, 0 = active, 1+ = bench index+1
    attack_id = -1
    target_active = True
    target_bench_index = -1
    needs_energy = False  # needs 1 more energy attached this turn to fire
    wins_game = False
    value = -1.0
    use_gust = False
    danger = False        # my current active is likely to be KO'd next opp turn


plan = Plan()
pre_turn = -1


def energy_ready(pokemon: Pokemon, attack) -> tuple:
    """Return (currently_ready, needs_one_more)."""
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
    dmg = attack.damage
    if attack.attackId == COSMIC_BEAM:
        return dmg  # ignores weakness/resistance per card text
    tdata = card_table.get(target.id)
    if tdata is None:
        return dmg
    if tdata.weakness is not None and tdata.weakness == EnergyType.FIGHTING:
        dmg *= 2
    elif tdata.resistance is not None and tdata.resistance == EnergyType.FIGHTING:
        dmg = max(0, dmg - 30)
    return dmg


def opponent_best_damage_estimate(op_active: Pokemon) -> int:
    """Rough estimate of the most damage the opponent's active could deal next turn."""
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
        if have + 1 >= req:  # assume they can attach one more energy next turn
            best = max(best, atk.damage)
    return best


def compute_plan(obs: Observation, state, select, my_index, my_state, op_state,
                  field_counts, hand_counts):
    global plan
    plan = Plan()

    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None
    if my_active is None or op_active is None:
        return

    # Loss-shielding signal: is my active in danger of being KO'd next opponent turn?
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
            if atk is None:
                continue
            if aid == COSMIC_BEAM and field_counts[Lunatone] == 0:
                continue
            ready, needs_one = energy_ready(pokemon, atk)
            if not ready and not (needs_one and not state.energyAttached and hand_counts[Basic_Fighting_Energy] > 0):
                continue

            # Default target: opponent's active.
            targets = [(True, -1, op_active)]
            if has_boss:
                for bi, bp in enumerate(op_bench):
                    targets.append((False, bi, bp))

            for is_active_target, bench_idx, target in targets:
                dmg = effective_damage(pokemon, atk, target)
                is_ko = dmg >= target.hp
                pv = prize_value(target.id)
                if is_ko:
                    value = WEIGHTS["ko_prize_value"] * pv
                    wins = is_ko and len(op_state.prize) <= pv
                else:
                    value = WEIGHTS["damage_progress"] * min(1.0, dmg / max(1, target.hp))
                    wins = False
                value += WEIGHTS["energy_on_attacker"] * len(pokemon.energies)
                if idx == 0:
                    value += WEIGHTS["active_attacker_bonus"]
                if not is_active_target:
                    value += WEIGHTS["gust_bonus"] * (pv / 3.0)
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


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
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

    global pre_turn
    if pre_turn != state.turn:
        pre_turn = state.turn

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)
    for c in ([my_state.active[0]] if my_state.active and my_state.active[0] else []) + list(my_state.bench):
        field_counts[c.id] += 1
    for c in my_state.hand:
        hand_counts[c.id] += 1
    for c in my_state.discard:
        discard_counts[c.id] += 1

    stadium_id = state.stadium[0].id if state.stadium else 0

    if context == SelectContext.MAIN:
        compute_plan(obs, state, select, my_index, my_state, op_state, field_counts, hand_counts)

    my_active = my_state.active[0] if my_state.active else None

    scores = []
    for o in select.option:
        score = 0.0
        if o.type == OptionType.NUMBER:
            score = float(o.number)
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = -1.0  # prefer going second: first player skips their turn-1 draw/attack in this ruleset
            else:
                score = 1.0
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
                        score = 100.0
                        score += WEIGHTS["energy_on_attacker"] * energy_count
                        score += WEIGHTS["switch_target_health"] * hp_frac * 10.0
                        if plan.attacker >= 1 and (plan.attacker - 1) == o.index:
                            score += 500.0
                    else:
                        # choosing opponent's pokemon (Boss's Orders gust target)
                        score = WEIGHTS["gust_bonus"] * (prize_value(card.id) / 3.0)
                        score += (100.0 - hp_frac * 100.0)
                        if plan.use_gust and plan.target_bench_index == o.index:
                            score += 1000.0
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Riolu:
                        score = 40.0
                    elif card.id == Makuhita:
                        score = 30.0
                    elif card.id == Solrock:
                        score = 20.0 if state.firstPlayer != my_index else 15.0
                    elif card.id == Lunatone:
                        score = 10.0
                    else:
                        score = 5.0
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = WEIGHTS["bench_development"]
                    if card.id in (Riolu, Makuhita):
                        score += WEIGHTS["evolution_progress"] * 0.3
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.LOOK):
                    score = _hand_target_score(card.id, field_counts, hand_counts)
                elif context == SelectContext.DISCARD:
                    # discarding excess/least useful copies
                    score = -_hand_target_score(card.id, field_counts, hand_counts) * 0.5
                    if card.id == Basic_Fighting_Energy:
                        score += WEIGHTS["discard_synergy"]
                elif context == SelectContext.ATTACH_FROM:
                    score = _attach_score(card, o.area == AreaType.ACTIVE)
                elif context in (SelectContext.EVOLVES_FROM,):
                    score = 100.0 + WEIGHTS["energy_on_attacker"] * energy_count + hp_frac * 20.0
                elif context in (SelectContext.EVOLVES_TO,):
                    score = 50.0 + WEIGHTS["evolution_progress"] * stage_value(card.id)
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
            score = _play_score(card.id, data, field_counts, hand_counts, discard_counts, stadium_id, state)
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is not None and card.id == Hero_Cape:
                score = 7000.0 + WEIGHTS["tool_hp_value"]
                if pokemon is not None and pokemon.id == Mega_Lucario_ex:
                    score += WEIGHTS["tool_hp_value"]
            else:
                score = _attach_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                is_planned_target = (
                    (plan.attacker == 0 and o.inPlayArea == AreaType.ACTIVE) or
                    (plan.attacker >= 1 and o.inPlayArea == AreaType.BENCH and o.inPlayIndex == plan.attacker - 1)
                )
                if plan.needs_energy and is_planned_target:
                    score += 5000.0
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            card = get_card(obs, o.area, o.index, my_index)
            score = 9000.0 + WEIGHTS["evolution_progress"] * stage_value(card.id if card else 0)
            if pokemon is not None:
                score += WEIGHTS["energy_on_attacker"] * len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None and card.id == Lunatone:
                if hand_counts[Basic_Fighting_Energy] > 0:
                    score = 25000.0 + WEIGHTS["card_draw_value"] + WEIGHTS["discard_synergy"]
                else:
                    score = -1.0
            elif card is not None and card.id == Hariyama:
                score = 28000.0 + WEIGHTS["gust_bonus"] * 0.2
            else:
                score = 26000.0
        elif o.type == OptionType.DISCARD:
            score = 0.0
        elif o.type == OptionType.RETREAT:
            if my_active is not None and my_active.maxHp > 0:
                hp_frac = my_active.hp / my_active.maxHp
            else:
                hp_frac = 1.0
            score = -1.0
            if plan.danger:
                score = WEIGHTS["retreat_danger_weight"] * (1.5 - hp_frac)
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


def _hand_target_score(card_id: int, field_counts, hand_counts) -> float:
    if card_id == Riolu:
        if field_counts[Riolu] + field_counts[Mega_Lucario_ex] >= 3:
            return 100.0
        return 3000.0 + WEIGHTS["evolution_progress"]
    if card_id == Mega_Lucario_ex:
        if field_counts[Riolu] >= 1:
            return 4000.0 + WEIGHTS["evolution_progress"] * 2
        return 500.0
    if card_id == Makuhita:
        if field_counts[Makuhita] + field_counts[Hariyama] >= 2:
            return 100.0
        return 2500.0 + WEIGHTS["evolution_progress"] * 0.5
    if card_id == Hariyama:
        if field_counts[Makuhita] >= 1:
            return 3800.0 + WEIGHTS["evolution_progress"] * 2
        return 300.0
    if card_id in (Lunatone, Solrock):
        if field_counts[card_id] >= 1:
            return 50.0
        return 1800.0
    if card_id == Basic_Fighting_Energy:
        return 2200.0 + WEIGHTS["energy_attach_readiness"] * 0.3
    if card_id in SEARCH_ITEMS:
        return 2000.0 + WEIGHTS["search_value"]
    if card_id in DRAW_SUPPORTERS:
        return 2400.0 + WEIGHTS["card_draw_value"]
    if card_id == Boss_Orders:
        return 2100.0 + WEIGHTS["gust_bonus"] * 0.3
    if card_id == Switch:
        return 1500.0
    if card_id == Hero_Cape:
        return 1400.0 + WEIGHTS["tool_hp_value"]
    if card_id == Premium_Power_Pro:
        return 1600.0
    if card_id == Gravity_Mountain:
        return 800.0 + WEIGHTS["stadium_value"]
    return 500.0


def _attach_score(pokemon, active: bool) -> float:
    if pokemon is None:
        return -1.0
    if pokemon.id not in (Makuhita, Hariyama, Lunatone, Solrock, Riolu, Mega_Lucario_ex):
        return -1.0
    data = card_table.get(pokemon.id)
    if data is None:
        return -1.0
    # find the pokemon's most expensive attack to know how much energy it wants
    max_req = 0
    for aid in data.attacks:
        atk = attack_table.get(aid)
        if atk is not None:
            max_req = max(max_req, len(atk.energies))
    have = len(pokemon.energies)
    if have >= max_req and max_req > 0:
        return 100.0  # already fully loaded, low priority
    score = 8000.0
    score += WEIGHTS["energy_attach_readiness"] * (have + 1)
    if active:
        score += WEIGHTS["energy_attach_active_bonus"]
    return score


def _play_score(card_id, data, field_counts, hand_counts, discard_counts, stadium_id, state) -> float:
    if data is None:
        return -1.0
    if data.cardType == CardType.POKEMON:
        score = 20000.0
        if card_id in (Lunatone, Solrock) and field_counts[card_id] >= 1:
            score = -1.0
        elif card_id == Riolu and field_counts[Riolu] + field_counts[Mega_Lucario_ex] >= 3:
            score = -1.0
        elif card_id == Makuhita and field_counts[Makuhita] + field_counts[Hariyama] >= 2:
            score = -1.0
        return score

    if data.cardType == CardType.SUPPORTER and state.supporterPlayed:
        return -1.0

    if card_id == Switch:
        if plan.attacker >= 1:
            return 6000.0
        return -1.0
    if card_id == Premium_Power_Pro:
        if plan.attacker >= -1 and not plan.wins_game:
            # does +30 dmg help close a gap to KO?
            return 5500.0
        return -1.0
    if card_id == Boss_Orders:
        if plan.use_gust:
            return 6200.0 + WEIGHTS["gust_bonus"]
        return -1.0
    if card_id in DRAW_SUPPORTERS:
        hand_size = len(getattr(state.players[state.yourIndex], "hand", None) or [])
        need = max(0.0, 5.0 - hand_size)
        return 5000.0 + WEIGHTS["card_draw_value"] * (1.0 + need * 0.2)
    if card_id in SEARCH_ITEMS:
        return 4800.0 + WEIGHTS["search_value"]
    if card_id == Hero_Cape:
        return -1.0  # played via ATTACH option, not PLAY, in this engine's flow typically
    if card_id == Gravity_Mountain:
        if stadium_id == Gravity_Mountain:
            return -1.0
        return 3000.0 + WEIGHTS["stadium_value"]

    return 3000.0
