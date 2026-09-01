import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Log, LogType, Observation, SelectContext, OptionType, Card, Pokemon, State, all_card_data, all_attack, to_observation_class

"""
Dragapult ex Deck
Advanced Level
This deck focuses on setting up multiple knockouts to take at least three Prize cards in a single turn with its Phantom Dive attack.
"""

# Load deck.csv in the dataset
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))
    
# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}
try:
    attack_table = {a.attackId: a for a in all_attack()}
except Exception:  # noqa: BLE001
    attack_table = {}

# dragapult_combined_final_v3_lucario: gate for the new no-trade-danger-retreat
# patch investigated this session (see AGENT_LOG.md). "L" in PATCH_ABLATE
# disables it for isolated A/B testing.
PATCH_ABLATE = os.environ.get("PATCH_ABLATE", "")
PATCH_NOTRADE_DANGER_RETREAT = "L" not in PATCH_ABLATE

# Decklist
Dreepy = 119  # ×4
Drakloak = 120  # ×4
Dragapult_ex = 121  # ×3
Fezandipiti_ex = 140  # ×1
Latias_ex = 184  # ×1
Budew = 235  # ×2
Meowth_ex = 1071  # ×1
Rare_Candy = 1079  # ×2
Unfair_Stamp = 1080  # ×1
Buddy_Buddy_Poffin = 1086  # ×4
Night_Stretcher = 1097  # ×2
Crushing_Hammer = 1120  # ×4
Ultra_Ball = 1121  # ×4
Poke_Pad = 1152  # x3
Lucky_Helmet = 1156  # ×1
Boss_Orders = 1182  # ×3
Crispin = 1198  # ×4
Brock_Scouting = 1210  # ×2
Lillie_Determination = 1227  # ×4
Team_Rocket_Watchtower = 1256  # ×2
Basic_Fire_Energy = 2  # ×4
Basic_Psychic_Energy = 5  # ×4

UNNECESSARY = -10000000

class AttackPlan:
    attack: int = 0
    counter: list[int] = []

can_switch = False
can_attack = False
can_main_attack = False
can_energy_attach = False
use_support = 0  # The Supporter card planned for use.
bench_attacker = False  # Whether there is a Benched Pokémon that is ready to attack
pre_turn_log: list[Log] = []
current_turn_log: list[Log] = []

prize: list[int] = []
card_counts: defaultdict[int, int] = defaultdict(int)
serial_set: set[int] = set()
plan_a = AttackPlan()
plan_b = AttackPlan()
plan_a_lethal = False  # True iff this turn's plan_a.attack achieves a full-KO-worthy prize count.


def _op_best_attack_damage(op_active: Pokemon | None, my_active: Pokemon | None) -> int:
    """Cheap, purely analytic (no search) estimate of the most damage the
    opponent's active Pokemon could plausibly deal to our active on their
    very next turn. Ported unchanged from dragapult_patched_v1's own
    danger estimator (same project, already validated there via ablation).
    Deliberately conservative: a forgiving +1 energy allowance (they could
    attach one more before attacking) so we don't under-estimate danger."""
    if op_active is None or my_active is None:
        return 0
    op_data = card_table.get(op_active.id)
    my_data = card_table.get(my_active.id)
    if op_data is None or my_data is None:
        return 0
    have = len(op_active.energies)
    best = 0
    for atk_id in op_data.attacks:
        atk = attack_table.get(atk_id)
        if atk is None:
            continue
        cost = len(atk.energies)
        if have + 1 < cost:
            continue
        dmg = atk.damage
        if my_data.weakness is not None and op_data.energyType == my_data.weakness:
            dmg *= 2
        elif my_data.resistance is not None and op_data.energyType == my_data.resistance:
            dmg = max(0, dmg - 30)
        best = max(best, dmg)
    return best


def _active_in_lethal_danger(my_state, op_state) -> bool:
    """Is our current active Pokemon estimated to be within lethal range of
    the opponent's own next attack? Fails safe (returns False) on any
    missing data."""
    try:
        my_active = my_state.active[0] if my_state.active else None
        op_active = op_state.active[0] if op_state.active else None
        if my_active is None or op_active is None:
            return False
        return _op_best_attack_damage(op_active, my_active) >= my_active.hp
    except Exception:
        return False


def no_damage_dex(id: int) -> bool:
    """Checks if the defending Pokémon possesses innate immunities preventing Dragapult ex from hitting it."""
    # Drednaw, Milotic ex, Sylveon, Crustle
    return id == 158 or id == 207 or id == 330 or id == 345


def no_damage_counter(pokemon: Pokemon) -> bool:
    """Checks if a target prevents placement of Phantom Dive's 6 bench damage counters (via abilities/Energy)."""
    # Poltchageist, Empoleon ex, Skeledirge, Milotic ex, Misty's Magikarp, Antique Cover Fossil
    if pokemon.id == 28 or pokemon.id == 199 or pokemon.id == 203 or pokemon.id == 207 or pokemon.id == 362 or pokemon.id == 1136:
        return True
    for card in pokemon.energyCards:
        # Mist Energy, Rock Fighting Energy
        if card.id == 11 or card.id == 20:
            return True
    return False


def prize_count(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12:  # Legacy Energy
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
                count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Heuristically evaluates the tactical worth of targeting a specific Pokémon on the opponent's field."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    
    id = pokemon.id
    # Noctowl, Fan Rotom, Archaludon ex, Meowth ex
    if id == 173 or id == 174 or id == 190 or id == 1071:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


def add_card_count(card: Card | Pokemon | None, my_index: int):
    if card == None:
        return
    if isinstance(card, Pokemon) or card.playerIndex == my_index:
        if card.serial not in serial_set:
            card_counts[card.id] -= 1
            serial_set.add(card.serial)
    if isinstance(card, Pokemon):
        for c in card.energyCards:
            add_card_count(c, my_index)
        for c in card.tools:
            add_card_count(c, my_index)
        for c in card.preEvolution:
            add_card_count(c, my_index)

def set_card_counts(obs: Observation, my_index: int):
    card_counts.clear()
    serial_set.clear()
    for id in my_deck:
        card_counts[id] += 1
    
    state = obs.current
    my_state = state.players[my_index]
    for card in my_state.hand:
        add_card_count(card, my_index)
    for card in my_state.discard:
        add_card_count(card, my_index)
    for card in my_state.bench:
        add_card_count(card, my_index)
    for card in my_state.active:
        add_card_count(card, my_index)
    for card in state.stadium:
        add_card_count(card, my_index)
    if state.looking != None:
        for card in state.looking:
            add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)

    
def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
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

def dragapult_one_attach_from_attack(pokemon: Pokemon | None, state) -> bool:
    """dragapult_wall_gust_v1: True if `pokemon` is our active Dragapult ex,
    already has exactly 1 energy, we haven't used this turn's one
    energy-attach action yet, and we're holding in hand the specific basic
    energy type Phantom Dive still needs (1 Fire + 1 Psychic) -- i.e.
    reaching a legal attack THIS turn is guaranteed via a single attach,
    not just "eventually".

    Why this exists: main_option_proc's plan (which target to hit, and
    critically whether that target is on the BENCH -- the trigger for
    scoring Boss's Orders sky-high to gust a non-immune target into active
    against a Crustle-style wall) only gets computed when can_main_attack is
    ALREADY true this decision. Early in a turn, before this turn's own
    energy attach has happened, that's not yet true even when it's about to
    become true by the end of the turn -- so the plan (and Boss's Orders'
    score) silently stays at its "nothing to do" default while an earlier
    decision this same turn commits the turn's one-Supporter slot to
    something else (observed via instrumented play: Crispin's "I still need
    energy" branch, id-gated on the exact same not-can_main_attack
    condition). By the time energy lands and can_main_attack flips true,
    the Supporter slot is already spent and Boss's Orders can't be played
    even though the plan now correctly wants it. This lets the plan be
    computed one decision earlier -- ONLY in the narrow, verifiable case
    where the attack is already guaranteed this turn -- so Boss's Orders
    can compete fairly for the slot before it's gone.
    """
    if pokemon is None or pokemon.id != Dragapult_ex or len(pokemon.energies) != 1:
        return False
    if state.energyAttached:
        return False
    have_fire = any(c.id == Basic_Fire_Energy for c in pokemon.energyCards)
    need_id = Basic_Psychic_Energy if have_fire else Basic_Fire_Energy
    return any(c.id == need_id for c in state.players[state.yourIndex].hand or [])


def main_option_proc(obs: Observation, damage: int):
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global can_switch
    global can_attack
    global can_main_attack
    global can_energy_attach

    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == 154:  # Phantom Dive
                can_main_attack = True

    global plan_a_lethal
    plan_a.attack = -1
    plan_b.attack = -1
    plan_a_lethal = False
    active = my_state.active[0] if my_state.active else None
    op_active_now = op_state.active[0] if op_state.active else None
    # dragapult_combined_final_v2: scope the early (pre-attach) plan computation to the
    # actual verified use case -- gusting a non-immune target past a Crustle-style wall.
    # Diagnosed against mega_lucario_ref: without this scope, the early plan also fires
    # vs ordinary (non-immune) opponents and routinely nominates the highest-*scored*
    # bench target (e.g. a 340 HP Mega Lucario ex sitting on the bench, valued highly by
    # pokemon_score's energy/prize weighting) even when it has no same-turn KO payoff --
    # winning the Supporter slot away from Crispin for a redirect that, per direct
    # instrumentation, converted to a same-turn KO 0% of the time. Restricting the early
    # trigger to "current active is immune" makes this byte-for-byte identical to
    # dragapult_deck_tune_v1 alone (no early computation at all) against every matchup
    # that isn't a true-immunity wall, while keeping the verified Crustle-wall benefit.
    op_is_immune = op_active_now is not None and no_damage_dex(op_active_now.id)
    attack_guaranteed_this_turn = can_main_attack or (
        op_is_immune and dragapult_one_attach_from_attack(active, state)
    )
    if not attack_guaranteed_this_turn and not (bench_attacker and can_switch):
        return
    
    cards = [op_state.active[0]]
    for pokemon in op_state.bench:
        cards.append(pokemon)
    counter_indices = []
    ci = []
    ci.append(0)
    remain_damage = 60
    while ci:
        index = ci[-1]
        hp = cards[index].hp
        if remain_damage >= hp:
            counter_indices.append(ci.copy())
            if index < len(cards) - 1:
                remain_damage -= hp
                ci.append(index + 1)
                continue
        if index == len(cards) - 1:
            ci.pop()
            if ci:
                remain_damage += cards[ci[-1]].hp
        if ci:
            ci[-1] += 1
    counter_indices.append([])

    remain_prize = len(my_state.prize)
    plan_score = 0
    for i, pokemon in enumerate(cards):
        base_prize_count = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize_count += prize_count(pokemon, True)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
        if remain_prize <= base_prize_count:
            max_score = 50000
        else:
            for indices in counter_indices:
                if i in indices:
                    continue
                prize = base_prize_count
                score = base_score
                for index in indices:
                    prize += prize_count(cards[index], False)
                    score += pokemon_score(cards[index], False)
                if remain_prize <= prize:
                    score = 50000
                else:
                    if prize >= 2:
                        if remain_prize <= 4:
                            score -= 1200
                    elif prize == 1:
                        score -= 300
                    else:
                        score += 1200
                if max_score < score:
                    max_score = score
                    ci = indices
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter
    # dragapult_combined_final_v3_lucario: True iff the finally-chosen plan_a
    # target achieves a full-KO-worthy prize count this turn (the code's own
    # 50000 sentinel, set above whenever remain_prize <= achieved prize).
    # Consumed by the new no-trade-danger-retreat check below -- attacking
    # for zero KO value while in guaranteed lethal danger is exactly the
    # pattern this patch targets; a lethal attack should never be skipped.
    plan_a_lethal = (plan_score == 50000)

def compute_option_scores(obs: Observation) -> list[int]:
    """Refactored out of the original single-function `agent()` (see git history /
    dragapult_day1/main.py for the untouched original) so bc_reranker_v1's main.py
    can call the exact same, proven per-option scoring logic directly and use the
    raw scores for BC-margin-gated reranking, without duplicating ~500 lines of
    tactical logic. Behavior is byte-for-byte identical to the original -- this is
    a pure extraction, computing and returning the `scores` list that the original
    function built just before its final sort-and-select tail.
    """
    global pre_turn_log
    global current_turn_log

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
            
    if state.turn == 0:
        prize.clear()
        pre_turn_log.clear()
        current_turn_log.clear()
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            if log.type == LogType.TURN_END:
                pre_turn_log = current_turn_log
                current_turn_log = []

    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK:
            if log.attackId == 323:  # Itchy Pollen
                no_item = True
        elif log.type == LogType.MOVE_CARD:
            if (log.playerIndex == my_index
                and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE)
                and log.toArea == AreaType.DISCARD):
                pre_ko = True

    if select.deck != None:
        set_card_counts(obs, my_index)
        for card in select.deck:
            card_counts[card.id] -= 1
        prize.clear()
        for id in card_counts:
            for _ in range(card_counts[id]):
                prize.append(id)
                
    set_card_counts(obs, my_index)
    for id in prize:
        card_counts[id] -= 1
    deck_counts = card_counts

    prize_diff = len(my_state.prize) - len(op_state.prize)
    
    global bench_attacker

    # Number of cards per card ID on the Bench and in the Active Spot
    field_counts = defaultdict(int)
    # Number of cards per card ID in hand
    hand_counts = defaultdict(int)
    # Number of cards per card ID in discard pile
    discard_counts = defaultdict(int)
    
    active_id = 0
    bench_attacker = False
    can_evolve_dreepy = False
    evolve_dreepy_count = 0
    can_evolve_drakloak = False
    damage = 200
    for card in my_state.active:
        if card == None:
            continue
        active_id = card.id
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
    for card in my_state.bench:
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
        if card.id == Dragapult_ex and len(card.energies) >= 2:
            bench_attacker = True
    main_pokemon_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    support_count = 0

    for card in my_state.discard:
        discard_counts[card.id] += 1

    def attach_score(attach_id: int, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        if card_table[attach_id].cardType == CardType.TOOL:
            # Attach tool
            score = 60000
            if active:
                score += 1000
            return score
        
        # Attach energy
        if pokemon.id == Budew:
            return -1
        elif pokemon.id == Meowth_ex or pokemon.id == Fezandipiti_ex or pokemon.id == Latias_ex:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                if bench_attacker or field_counts[Budew] >= 1:
                    return 22000
                else:
                    return 18000
            else:
                return -1
        if active and can_main_attack:
            return -1
        score = 20000
        if energy_count >= 2:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                score += 200
            else:
                return -1
        elif energy_count == 1:
            if attach_id == pokemon.energyCards[0].id:
                return -1
            if pokemon.id == Dragapult_ex:
                score += 250
            elif pokemon.id == Dreepy:
                score -= 150
            else:
                score -= 200
            if active:
                score += 200
        else:  # energy_count == 0
            if active:
                if bench_attacker:
                    score += 400
            else:
                if pokemon.id == Dragapult_ex:
                    score += 150
                elif pokemon.id == Dreepy:
                    score += 100
                else:
                    score += 50
                if bench_attacker:
                    score -= 200
        if no_more_dex and (pokemon.id == Dreepy or pokemon.id == Drakloak):
            score -= 500
        return score
    
    def hand_score(id: int, ignore_count: bool):
        score = 0
        if id == Dreepy:
            if main_pokemon_count >= 3:
                score = 1000
            else:
                score = 18000
        elif id == Drakloak:
            if can_evolve_dreepy:
                score = 20000
            else:
                score = 3000
        elif id == Dragapult_ex:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Rare_Candy] >= 1 and not no_item:
                score = 40000
            elif can_evolve_drakloak:
                if field_counts[id] == 0:
                    score = 30000
                elif field_counts[id] == 1:
                    score = 10000
                else:
                    score = 50
            else:
                if field_counts[id] >= 2:
                    score = 50
                else:
                    score = 2000
        elif id == Fezandipiti_ex:
            if pre_ko:
                score = 50000
            elif prize_diff <= -2:
                score = 5
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
        elif id == Latias_ex:
            if active_id == Fezandipiti_ex or active_id == Meowth_ex or active_id == Dreepy:
                if field_counts[Drakloak] + field_counts[Dragapult_ex] == 0:
                    score = 28000
                else:
                    score = 15000
            else:
                score = 10
        elif id == Budew:
            if field_counts[id] + field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1:
                score = UNNECESSARY
            elif state.turn >= 2:
                score = 30000
        elif id == Meowth_ex:
            if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                score = 5
            elif state.supporterPlayed:
                score = 40
            else:
                score = 35000
        elif id == Rare_Candy:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1:
                score = 40000
        elif id == Unfair_Stamp:
            if pre_ko:
                score = 80000
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            else:
                score = 80
        elif id == Buddy_Buddy_Poffin:
            count = deck_counts[Dreepy]
            if count == 0:
                score = UNNECESSARY
            else:
                if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                    count += 1
                if count >= 2:
                    score = 35000
        elif id == Night_Stretcher:
            for i in discard_counts:
                if discard_counts[i] >= 1:
                    card_type = card_table[i].cardType
                    if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                        score = max(score, hand_score(i, ignore_count))
        elif id == Crushing_Hammer:
            score = 20
        elif id == Ultra_Ball:
            if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
                score = 70
            else:
                score = 5
        elif id == Poke_Pad:
            score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
        elif id == Lucky_Helmet:
            score = 15
        elif id == Boss_Orders:
            # unchanged from dragapult_deck_tune_v1 -- the scoping fix lives at
            # main_option_proc's early-trigger site (see op_is_immune there), not here,
            # so this scoring line's behavior is byte-identical to deck_tune_v1 alone
            # whenever the early pre-attach plan path hasn't fired.
            if plan_a.attack > 0:
                score = 60000
        elif id == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                if not can_main_attack and not bench_attacker and field_counts[Dragapult_ex] >= 1:
                    score = 55000
                else:
                    score = 25000
        elif id == Brock_Scouting:
            if not ignore_count or support_count == 0:
                if state.turn == 2 and field_counts[Budew] + field_counts[Latias_ex] == 0:
                    score = 50000
                else:
                    score = 30000
        elif id == Lillie_Determination:
            if not ignore_count or support_count == 0:
                score = 45000
        elif id == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                score = 4000
        elif id == Basic_Fire_Energy or id == Basic_Psychic_Energy:
            if can_main_attack and (len(op_state.prize) <= 2
                or (bench_attacker and len(op_state.prize) <= 4)):
                score = UNNECESSARY
            else:
                max_score = -10000
                for pokemon in my_state.active:
                    if pokemon == None:
                        continue
                    max_score = max(max_score, attach_score(id, pokemon, True))
                for pokemon in my_state.bench:
                    max_score = max(max_score, attach_score(id, pokemon, False))
                score = max_score - 5000
                if can_main_attack or bench_attacker:
                    score /= 10
        
        if not ignore_count and hand_counts[id] > 0:
            if id == Drakloak and hand_counts[id] < evolve_dreepy_count:
                score -= 10
            elif id == Dreepy:
                score -= 100
            else:
                score -= 100000
        return score

    global use_support
    if context == SelectContext.MAIN:
        main_option_proc(obs, damage)
                    
        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, state.yourIndex)
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        score = hand_score(card.id, True)
                        if support_score < score:
                            support_score = score
                            use_support = card.id

    hand_scores = []
    negative_hand_count = 0
    for card in my_state.hand:
        score = hand_score(card.id, False)
        hand_scores.append(score)
        if score < 0:
            negative_hand_count += 1
        hand_counts[card.id] += 1
        if card_table[card.id].cardType == CardType.SUPPORTER and card.id != Boss_Orders:
            support_count += 1

    no_draw = (my_state.deckCount <= 8)  # Whether to restrict actions that reduce the deck

    # dragapult_combined_final_v3_lucario: no-trade-danger-retreat. Found via
    # direct instrumentation of real losses vs mega_lucario_ref (see
    # AGENT_LOG.md): in 19/107 sampled losses, our Dragapult ex attacked at
    # (near-)full HP, its Phantom Dive secured NO knockout that turn (Mega
    # Lucario ex's 340-440 HP -- boosted via Hero's Cape/evolution -- easily
    # tanks 200 damage), and the very next opposing turn one-shot it for a
    # totally free 3-prize KO (Mega Brave 270 + Premium Power Pro's +30 =
    # 300, comfortably >= Dragapult ex's 320 max HP). The existing do_switch
    # logic never even considers retreating here because its only gate is
    # "not can_main_attack" -- once Dragapult CAN attack, it always does,
    # regardless of whether that attack has any payoff this turn. This adds
    # a narrow, separate trigger: only when attacking secures no lethal
    # value at all (plan_a_lethal False) AND our active is genuinely in the
    # opponent's next-turn lethal range (analytic estimator, ported
    # unchanged from dragapult_patched_v1's own already-validated Patch D)
    # AND we have somewhere to retreat to. Deliberately does NOT touch the
    # ordinary bench_attacker/Budew retreat paths above, and never fires
    # when the attack IS lethal-valuable this turn.
    no_trade_danger_retreat = False
    if PATCH_NOTRADE_DANGER_RETREAT and can_main_attack and not plan_a_lethal:
        try:
            my_active_pkmn = my_state.active[0] if my_state.active else None
            if (my_active_pkmn is not None
                and active_id == Dragapult_ex
                and len(my_state.bench) >= 1
                and prize_count(my_active_pkmn, False) >= 2
                and _active_in_lethal_danger(my_state, op_state)):
                no_trade_danger_retreat = True
        except Exception:
            no_trade_danger_retreat = False

    do_switch = (not can_main_attack and (bench_attacker or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2))) or no_trade_danger_retreat
    effect_card_id = 0 if select.effect == None else select.effect.id
    context_card_id = 0 if select.contextCard == None else select.contextCard.id
    
    scores = []  # Score for each action
    for o in select.option:
        score = 0  # The default and baseline score is 0.
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = -1
            else:
                score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card != None:
                energy_count = 0
                hp = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                    hp = card.hp
                if (context == SelectContext.SWITCH
                    or context == SelectContext.TO_ACTIVE
                    or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    # Selection of the Pokémon to send to the Active Spot
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:
                            score += 10000
                        elif card.id == Drakloak:
                            if energy_count >= 1:
                                score += 20000
                            else:
                                score -= 10000
                        elif card.id == Dragapult_ex:
                            score += 50000
                        elif card.id == Budew:
                            if context != SelectContext.SWITCH:
                                score += 100000
                            elif not bench_attacker:
                                score += 30000
                        elif card.id == Fezandipiti_ex:
                            score -= 1000
                        elif card.id == Meowth_ex:
                            score -= 2000
                    else:
                        if plan_a.attack == o.index + 1:
                            score += 100000
                    score += energy_count * 1000
                    score += hp
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if my_index == state.firstPlayer or card.id != Dreepy:
                        score = -1
                elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id == Crispin:
                        # Reverse scoring
                        score = 100000 - hand_score(card.id, True)
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        support_count -= 1
                    score = -hand_score(card.id, False)
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE:
                                    score += 10000
                            elif 40 <= hp <= 90:
                                score += 10000 + hp * 20
                            elif hp <= 30:
                                score += -10000 + hp * 20
                            if card.id == 133 or card.id == 351:
                                score += 30000
                        else:
                            index = o.index + 1
                            if index in plan_b.counter:
                                score += 100000
                            else:
                                remain_damage = select.remainDamageCounter * 10
                                if 210 <= hp <= 200 + remain_damage:
                                    score += 30000
                                elif 20 <= hp <= 60 + remain_damage:
                                    score += 10000
                                elif hp == 10:
                                    score -= 100000
                            if no_damage_counter(card):
                                score = -1
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex:
                        score += 200
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            # Discarding energy (Retreat or Crushing Hammer)
            if o.playerIndex != state.yourIndex:
                if o.area == AreaType.BENCH:
                    score = 20
                else:
                    score = 10
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                    score += 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            if card.id == Dreepy:
                score = 51000
            elif card.id == Fezandipiti_ex:
                if card_score > 0:
                    score = 53000
                else:
                    score = -1
            elif card.id == Latias_ex:
                if active_id != Drakloak and active_id != Dragapult_ex:
                    score = 51000
                else:
                    score = -1
            elif card.id == Budew:
                if field_counts[Budew] == 0 and field_counts[Dragapult_ex] == 0:
                    score = 52000
                else:
                    score = -1
            elif card.id == Meowth_ex:
                if state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif support_count == 0:
                    score = 50000
                elif support_count == hand_counts[Boss_Orders] and not plan_a.attack <= 0:
                    score = 50000
                else:
                    score = -1
            elif card.id == Rare_Candy:
                if no_more_dex:
                    score = -1
                else:
                    score = 75000
            elif card.id == Unfair_Stamp:
                score = 15000
            elif card.id == Night_Stretcher:
                if card_score >= 18000:
                    score = 42000
                else:
                    score = -1
            elif card.id == Crushing_Hammer:
                score = 40000
            elif card.id == Boss_Orders:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
            elif card.id == Lillie_Determination:
                if card.id == use_support:
                    score = 14000
                else:
                    score = -1
            elif card.id == Team_Rocket_Watchtower:
                if stadium_id > 0 or state.turn == 1:
                    score = 80000
                else:
                    score = -1
            elif no_draw:
                score = -1
            elif card.id == Buddy_Buddy_Poffin:
                if deck_counts[Dreepy] > 0:
                    score = 46000
                else:
                    score = -1
            elif card.id == Ultra_Ball:
                if negative_hand_count >= 2:
                    score = 44000
                else:
                    score = -1
            elif card.id == Poke_Pad:
                if deck_counts[Dreepy] + deck_counts[Drakloak] > 0:
                    score = 45000
                else:
                    score = -1
            elif card.id == Crispin or card.id == Brock_Scouting:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Dreepy:
                score += 30000
            elif field_counts[Dragapult_ex] >= 2 or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                score = -1
            else:
                score += 70000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if no_draw:
                score = -1
            elif card.id == 1267:  # Lumiose City
                score = 1
            else:
                score = 40000
        elif o.type == OptionType.RETREAT:
            if do_switch:
                score = 10000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId

        scores.append(score)

    return scores


def select_from_scores(select: "SelectData", scores: list[int]) -> list[int]:
    """The original function's final sort-and-select tail, unchanged."""
    context = select.context
    output = []
    if len(scores) >= 1:
        # Select in descending order of score
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            # If the score is negative, do not select it if skipping is possible
            if (sorted_scores[i][1] >= 0
                or select.minCount > i
                or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])

    return output


def heuristic_agent(obs_dict: dict) -> list[int]:
    """Main Agent Function -- identical behavior to the original dragapult_day1
    agent(), just re-composed from the two functions above.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return my_deck

    scores = compute_option_scores(obs)
    return select_from_scores(obs.select, scores)
